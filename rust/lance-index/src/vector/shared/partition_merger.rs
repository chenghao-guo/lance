// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Shared helpers for IVF partition merging and metadata writing.
//!
//! The helpers here are used by both the distributed index merger
//! (`vector::distributed::index_merger`) and the classic IVF index
//! builder in the `lance` crate. They keep writer initialization and
//! IVF / index metadata writing in one place.

use std::ops::Range;
use std::sync::Arc;

use arrow_schema::{DataType, Field, Schema as ArrowSchema};
use bytes::Bytes;
use lance_core::{datatypes::Schema as LanceSchema, Error, Result, ROW_ID_FIELD};
use lance_file::reader::FileReader as V2Reader;
use lance_file::writer::{FileWriter, FileWriterOptions};
use lance_linalg::distance::DistanceType;
use prost::Message;

use crate::pb;
use crate::vector::flat::index::FlatMetadata;
use crate::vector::ivf::storage::{IvfModel, IVF_METADATA_KEY};
use crate::vector::pq::storage::{ProductQuantizationMetadata, PQ_METADATA_KEY};
use crate::vector::quantizer::QuantizerMetadata;
use crate::vector::sq::storage::{ScalarQuantizationMetadata, SQ_METADATA_KEY};
use crate::vector::storage::STORAGE_METADATA_KEY;
use crate::vector::{DISTANCE_TYPE_KEY, PQ_CODE_COLUMN, SQ_CODE_COLUMN};
use crate::{IndexMetadata as IndexMetaSchema, INDEX_METADATA_SCHEMA_KEY};

/// Supported vector index types for unified IVF metadata writing.
///
/// This mirrors the vector variants in [`crate::IndexType`] that are
/// used by IVF-based indices. Keeping this here avoids pulling the
/// full `IndexType` dependency into helpers that only need the string
/// representation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SupportedIndexType {
    IvfFlat,
    IvfPq,
    IvfSq,
    IvfHnswFlat,
    IvfHnswPq,
    IvfHnswSq,
}

impl SupportedIndexType {
    /// Get the index type string used in metadata.
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::IvfFlat => "IVF_FLAT",
            Self::IvfPq => "IVF_PQ",
            Self::IvfSq => "IVF_SQ",
            Self::IvfHnswFlat => "IVF_HNSW_FLAT",
            Self::IvfHnswPq => "IVF_HNSW_PQ",
            Self::IvfHnswSq => "IVF_HNSW_SQ",
        }
    }

    /// Map an index type string (as stored in metadata) to a
    /// [`SupportedIndexType`] if it is one of the IVF variants this
    /// helper understands.
    pub fn from_index_type_str(s: &str) -> Option<Self> {
        match s {
            "IVF_FLAT" => Some(Self::IvfFlat),
            "IVF_PQ" => Some(Self::IvfPq),
            "IVF_SQ" => Some(Self::IvfSq),
            "IVF_HNSW_FLAT" => Some(Self::IvfHnswFlat),
            "IVF_HNSW_PQ" => Some(Self::IvfHnswPq),
            "IVF_HNSW_SQ" => Some(Self::IvfHnswSq),
            _ => None,
        }
    }

    /// Detect index type from reader metadata and schema.
    ///
    /// This is primarily used by the distributed index merger when
    /// consolidating partial auxiliary files.
    pub fn detect(reader: &V2Reader, schema: &ArrowSchema) -> Result<Self> {
        let has_pq_code_col = schema.fields.iter().any(|f| f.name() == PQ_CODE_COLUMN);
        let has_sq_code_col = schema.fields.iter().any(|f| f.name() == SQ_CODE_COLUMN);

        let is_pq = reader
            .metadata()
            .file_schema
            .metadata
            .contains_key(PQ_METADATA_KEY)
            || has_pq_code_col;
        let is_sq = reader
            .metadata()
            .file_schema
            .metadata
            .contains_key(SQ_METADATA_KEY)
            || has_sq_code_col;

        // Detect HNSW-related columns
        let has_hnsw_vector_id_col = schema.fields.iter().any(|f| f.name() == "__vector_id");
        let has_hnsw_pointer_col = schema.fields.iter().any(|f| f.name() == "__pointer");
        let has_hnsw = has_hnsw_vector_id_col || has_hnsw_pointer_col;

        let index_type = match (has_hnsw, is_pq, is_sq) {
            (false, false, false) => Self::IvfFlat,
            (false, true, false) => Self::IvfPq,
            (false, false, true) => Self::IvfSq,
            (true, false, false) => Self::IvfHnswFlat,
            (true, true, false) => Self::IvfHnswPq,
            (true, false, true) => Self::IvfHnswSq,
            _ => {
                return Err(Error::NotSupported {
                    source: "Unsupported index type combination detected".into(),
                    location: snafu::location!(),
                });
            }
        };

        Ok(index_type)
    }
}

/// Initialize schema-level metadata on a writer for a given storage.
///
/// It writes the distance type and the storage metadata (as a vector payload),
/// and optionally the raw storage metadata under a storage-specific metadata
/// key (e.g. [`PQ_METADATA_KEY`] or [`SQ_METADATA_KEY`]).
fn init_writer_for_storage(
    w: &mut FileWriter,
    dt: DistanceType,
    storage_meta_json: &str,
    storage_meta_key: &str,
) -> Result<()> {
    // distance type
    w.add_schema_metadata(DISTANCE_TYPE_KEY, dt.to_string());
    // storage metadata (vector of one entry for future extensibility)
    let meta_vec_json = serde_json::to_string(&vec![storage_meta_json.to_string()])?;
    w.add_schema_metadata(STORAGE_METADATA_KEY, meta_vec_json);
    if !storage_meta_key.is_empty() {
        w.add_schema_metadata(storage_meta_key, storage_meta_json.to_string());
    }
    Ok(())
}

/// Create and initialize a unified writer for FLAT storage.
pub async fn init_writer_for_flat(
    object_store: &lance_io::object_store::ObjectStore,
    aux_out: &object_store::path::Path,
    d0: usize,
    dt: DistanceType,
) -> Result<FileWriter> {
    let arrow_schema = ArrowSchema::new(vec![
        (*ROW_ID_FIELD).clone(),
        Field::new(
            crate::vector::flat::storage::FLAT_COLUMN,
            DataType::FixedSizeList(
                Arc::new(Field::new("item", DataType::Float32, true)),
                d0 as i32,
            ),
            true,
        ),
    ]);
    let writer = object_store.create(aux_out).await?;
    let mut w = FileWriter::try_new(
        writer,
        LanceSchema::try_from(&arrow_schema)?,
        FileWriterOptions::default(),
    )?;
    let meta_json = serde_json::to_string(&FlatMetadata { dim: d0 })?;
    init_writer_for_storage(&mut w, dt, &meta_json, "")?;
    Ok(w)
}

/// Create and initialize a unified writer for PQ storage.
///
/// This always writes the codebook into the unified file and resets
/// `buffer_index` in the metadata to point at the new location.
pub async fn init_writer_for_pq(
    object_store: &lance_io::object_store::ObjectStore,
    aux_out: &object_store::path::Path,
    dt: DistanceType,
    pm: &ProductQuantizationMetadata,
) -> Result<FileWriter> {
    let num_bytes = if pm.nbits == 4 {
        pm.num_sub_vectors / 2
    } else {
        pm.num_sub_vectors
    };
    let arrow_schema = ArrowSchema::new(vec![
        (*ROW_ID_FIELD).clone(),
        Field::new(
            PQ_CODE_COLUMN,
            DataType::FixedSizeList(
                Arc::new(Field::new("item", DataType::UInt8, true)),
                num_bytes as i32,
            ),
            true,
        ),
    ]);
    let writer = object_store.create(aux_out).await?;
    let mut w = FileWriter::try_new(
        writer,
        LanceSchema::try_from(&arrow_schema)?,
        FileWriterOptions::default(),
    )?;
    let mut pm_init = pm.clone();
    let cb = pm_init.codebook.as_ref().ok_or_else(|| Error::Index {
        message: "PQ codebook missing".to_string(),
        location: snafu::location!(),
    })?;
    let codebook_tensor: pb::Tensor = pb::Tensor::try_from(cb)?;
    let buf = Bytes::from(codebook_tensor.encode_to_vec());
    let pos = w.add_global_buffer(buf).await?;
    pm_init.set_buffer_index(pos);
    let pm_json = serde_json::to_string(&pm_init)?;
    init_writer_for_storage(&mut w, dt, &pm_json, PQ_METADATA_KEY)?;
    Ok(w)
}

/// Create and initialize a unified writer for SQ storage.
pub async fn init_writer_for_sq(
    object_store: &lance_io::object_store::ObjectStore,
    aux_out: &object_store::path::Path,
    dt: DistanceType,
    sq_meta: &ScalarQuantizationMetadata,
) -> Result<FileWriter> {
    let d0 = sq_meta.dim;
    let arrow_schema = ArrowSchema::new(vec![
        (*ROW_ID_FIELD).clone(),
        Field::new(
            SQ_CODE_COLUMN,
            DataType::FixedSizeList(
                Arc::new(Field::new("item", DataType::UInt8, true)),
                d0 as i32,
            ),
            true,
        ),
    ]);
    let writer = object_store.create(aux_out).await?;
    let mut w = FileWriter::try_new(
        writer,
        LanceSchema::try_from(&arrow_schema)?,
        FileWriterOptions::default(),
    )?;
    let meta_json = serde_json::to_string(sq_meta)?;
    init_writer_for_storage(&mut w, dt, &meta_json, SQ_METADATA_KEY)?;
    Ok(w)
}

/// Write unified IVF and index metadata to the writer.
///
/// This writes the IVF model into a global buffer and stores its
/// position under [`IVF_METADATA_KEY`], and attaches a compact
/// [`IndexMetaSchema`] payload under [`INDEX_METADATA_SCHEMA_KEY`].
pub async fn write_unified_ivf_and_index_metadata(
    w: &mut FileWriter,
    ivf_model: &IvfModel,
    dt: DistanceType,
    idx_type: SupportedIndexType,
) -> Result<()> {
    let pb_ivf: pb::Ivf = (ivf_model).try_into()?;
    let pos = w
        .add_global_buffer(Bytes::from(pb_ivf.encode_to_vec()))
        .await?;
    w.add_schema_metadata(IVF_METADATA_KEY, pos.to_string());
    let idx_meta = IndexMetaSchema {
        index_type: idx_type.as_str().to_string(),
        distance_type: dt.to_string(),
    };
    w.add_schema_metadata(INDEX_METADATA_SCHEMA_KEY, serde_json::to_string(&idx_meta)?);
    Ok(())
}

/// Stream and write a range of rows from reader into writer.
///
/// The caller is responsible for ensuring that `range` corresponds to a
/// contiguous row interval for a single IVF partition.
pub async fn write_partition_rows(
    reader: &V2Reader,
    w: &mut FileWriter,
    range: Range<usize>,
) -> Result<()> {
    let mut stream = reader.read_stream(
        lance_io::ReadBatchParams::Range(range),
        u32::MAX,
        4,
        lance_encoding::decoder::FilterExpression::no_filter(),
    )?;
    use futures::StreamExt as _;
    while let Some(rb) = stream.next().await {
        let rb = rb?;
        w.write_batch(&rb).await?;
    }
    Ok(())
}
