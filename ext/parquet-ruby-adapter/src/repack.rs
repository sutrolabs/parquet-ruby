use std::fs::{self, File};
use std::path::PathBuf;

use arrow_schema::SchemaRef;
use magnus::value::ReprValue;
use magnus::{Error as MagnusError, RArray, RHash, Ruby, Value};
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use parquet::arrow::ArrowWriter;
use parquet::file::properties::WriterProperties;

use crate::types::ParquetRepackArgs;
use crate::utils::{parse_compression, parse_parquet_repack_args};

struct RepackedFile {
    path: String,
    num_rows: usize,
}

/// Repack Parquet files by streaming Arrow record batches into new Parquet files.
pub fn repack(ruby: &Ruby, args: &[Value]) -> Result<Value, MagnusError> {
    let repack_args = parse_parquet_repack_args(ruby, args)?;
    let files = repack_files(&repack_args)?;

    let result = RArray::with_capacity(files.len());
    for file in files {
        let hash = RHash::new();
        hash.aset("path", file.path)?;
        hash.aset("num_rows", file.num_rows)?;
        result.push(hash)?;
    }

    Ok(result.as_value())
}

fn repack_files(args: &ParquetRepackArgs) -> Result<Vec<RepackedFile>, MagnusError> {
    let schema = read_schema(&args.read_from[0])?;
    validate_input_schemas(args, schema.clone())?;

    let mut repacked_files = Vec::new();
    let mut output_index = 0usize;
    let mut current_output_rows = 0usize;
    let mut current_writer = None;

    for input_path in &args.read_from {
        let mut builder = create_reader_builder(input_path)?;

        if let Some(max_read_rows_per_chunk) = args.max_read_rows_per_chunk {
            builder = builder.with_batch_size(max_read_rows_per_chunk);
        }

        let reader = builder
            .build()
            .map_err(|e| MagnusError::new(magnus::exception::runtime_error(), e.to_string()))?;

        for batch_result in reader {
            let batch = batch_result
                .map_err(|e| MagnusError::new(magnus::exception::runtime_error(), e.to_string()))?;
            let mut offset = 0usize;

            while offset < batch.num_rows() {
                if current_writer.is_none() {
                    let (writer, path) = create_writer(args, schema.clone(), output_index)?;
                    current_writer = Some(writer);
                    repacked_files.push(RepackedFile { path, num_rows: 0 });
                }

                let rows_remaining_in_batch = batch.num_rows() - offset;
                let rows_remaining_in_output = args.rows_per_file - current_output_rows;
                let rows_to_write = rows_remaining_in_batch.min(rows_remaining_in_output);

                let batch_slice = batch.slice(offset, rows_to_write);
                current_writer
                    .as_mut()
                    .expect("writer must be present")
                    .write(&batch_slice)
                    .map_err(|e| {
                        MagnusError::new(magnus::exception::runtime_error(), e.to_string())
                    })?;

                offset += rows_to_write;
                current_output_rows += rows_to_write;
                repacked_files
                    .last_mut()
                    .expect("output file metadata must be present")
                    .num_rows += rows_to_write;

                if current_output_rows == args.rows_per_file {
                    close_writer(current_writer.take())?;
                    output_index += 1;
                    current_output_rows = 0;
                }
            }
        }
    }

    close_writer(current_writer.take())?;

    Ok(repacked_files)
}

fn validate_input_schemas(
    args: &ParquetRepackArgs,
    schema: SchemaRef,
) -> Result<(), MagnusError> {
    for input_path in args.read_from.iter().skip(1) {
        let input_schema = read_schema(input_path)?;

        if input_schema.as_ref() != schema.as_ref() {
            return Err(MagnusError::new(
                magnus::exception::runtime_error(),
                format!("Input file schema does not match first file: {}", input_path),
            ));
        }
    }

    Ok(())
}

fn read_schema(path: &str) -> Result<SchemaRef, MagnusError> {
    let builder = create_reader_builder(path)?;
    Ok(builder.schema().clone())
}

fn create_reader_builder(
    path: &str,
) -> Result<ParquetRecordBatchReaderBuilder<File>, MagnusError> {
    let file = File::open(path).map_err(|e| {
        MagnusError::new(
            magnus::exception::runtime_error(),
            format!("Failed to open input file '{}': {}", path, e),
        )
    })?;

    ParquetRecordBatchReaderBuilder::try_new(file)
        .map_err(|e| MagnusError::new(magnus::exception::runtime_error(), e.to_string()))
}

fn create_writer(
    args: &ParquetRepackArgs,
    schema: SchemaRef,
    output_index: usize,
) -> Result<(ArrowWriter<File>, String), MagnusError> {
    fs::create_dir_all(&args.output_dir).map_err(|e| {
        MagnusError::new(
            magnus::exception::runtime_error(),
            format!(
                "Failed to create output directory '{}': {}",
                args.output_dir, e
            ),
        )
    })?;

    let output_path =
        PathBuf::from(&args.output_dir).join(format!("batch-{}.parquet", output_index));
    let output_path_string = output_path.to_string_lossy().into_owned();

    let file = File::create(&output_path).map_err(|e| {
        MagnusError::new(
            magnus::exception::runtime_error(),
            format!("Failed to create output file '{}': {}", output_path_string, e),
        )
    })?;

    let compression = parse_compression(args.compression.clone())?;
    let props = WriterProperties::builder()
        .set_compression(compression)
        .build();

    let writer = ArrowWriter::try_new(file, schema, Some(props))
        .map_err(|e| MagnusError::new(magnus::exception::runtime_error(), e.to_string()))?;

    Ok((writer, output_path_string))
}

fn close_writer(writer: Option<ArrowWriter<File>>) -> Result<(), MagnusError> {
    if let Some(writer) = writer {
        writer
            .close()
            .map_err(|e| MagnusError::new(magnus::exception::runtime_error(), e.to_string()))?;
    }

    Ok(())
}
