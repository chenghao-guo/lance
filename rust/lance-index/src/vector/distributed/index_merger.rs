// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Index merging mechanisms for distributed vector index building

use crate::vector::shared::partition_merger::{
    init_writer_for_flat, init_writer_for_pq, init_writer_for_sq, write_partition_rows,
    write_unified_ivf_and_index_metadata, SupportedIndexType,
};
use arrow::datatypes::Float32Type;
use arrow_array::cast::AsArray;
use arrow_array::{Array, FixedSizeListArray};
use futures::StreamExt as _;
use lance_core::{Error, Result};
use snafu::location;
use std::sync::Arc;

/// Strict bitwise equality check for FixedSizeListArray values.
/// Returns true only if length, value_length and all underlying primitive values are equal.
fn fixed_size_list_equal(a: &FixedSizeListArray, b: &FixedSizeListArray) -> bool {
    if a.len() != b.len() || a.value_length() != b.value_length() {
        return false;
    }
    use arrow_schema::DataType;
    match (a.value_type(), b.value_type()) {
        (DataType::Float32, DataType::Float32) => {
            let va = a.values().as_primitive::<Float32Type>();
            let vb = b.values().as_primitive::<Float32Type>();
            va.values() == vb.values()
        }
        (DataType::Float64, DataType::Float64) => {
            let va = a.values().as_primitive::<arrow_array::types::Float64Type>();
            let vb = b.values().as_primitive::<arrow_array::types::Float64Type>();
            va.values() == vb.values()
        }
        (DataType::Float16, DataType::Float16) => {
            let va = a.values().as_primitive::<arrow_array::types::Float16Type>();
            let vb = b.values().as_primitive::<arrow_array::types::Float16Type>();
            va.values() == vb.values()
        }
        _ => false,
    }
}

/// Relaxed numeric equality check within tolerance to accommodate minor serialization
/// differences while still enforcing global-training invariants.
fn fixed_size_list_almost_equal(a: &FixedSizeListArray, b: &FixedSizeListArray, tol: f32) -> bool {
    if a.len() != b.len() || a.value_length() != b.value_length() {
        return false;
    }
    use arrow_schema::DataType;
    match (a.value_type(), b.value_type()) {
        (DataType::Float32, DataType::Float32) => {
            let va = a.values().as_primitive::<Float32Type>();
            let vb = b.values().as_primitive::<Float32Type>();
            let av = va.values();
            let bv = vb.values();
            if av.len() != bv.len() {
                return false;
            }
            for i in 0..av.len() {
                if (av[i] - bv[i]).abs() > tol {
                    return false;
                }
            }
            true
        }
        (DataType::Float64, DataType::Float64) => {
            let va = a.values().as_primitive::<arrow_array::types::Float64Type>();
            let vb = b.values().as_primitive::<arrow_array::types::Float64Type>();
            let av = va.values();
            let bv = vb.values();
            if av.len() != bv.len() {
                return false;
            }
            for i in 0..av.len() {
                if (av[i] - bv[i]).abs() > tol as f64 {
                    return false;
                }
            }
            true
        }
        (DataType::Float16, DataType::Float16) => {
            let va = a.values().as_primitive::<arrow_array::types::Float16Type>();
            let vb = b.values().as_primitive::<arrow_array::types::Float16Type>();
            let av = va.values();
            let bv = vb.values();
            if av.len() != bv.len() {
                return false;
            }
            for i in 0..av.len() {
                let da = av[i].to_f32();
                let db = bv[i].to_f32();
                if (da - db).abs() > tol {
                    return false;
                }
            }
            true
        }
        _ => false,
    }
}

// Merge partial vector index auxiliary files into a unified auxiliary.idx
use crate::pb;
use crate::vector::flat::index::FlatMetadata;
use crate::vector::ivf::storage::{IvfModel as IvfStorageModel, IVF_METADATA_KEY};
use crate::vector::pq::storage::{ProductQuantizationMetadata, PQ_METADATA_KEY};
use crate::vector::sq::storage::{ScalarQuantizationMetadata, SQ_METADATA_KEY};
use crate::vector::storage::STORAGE_METADATA_KEY;
use crate::vector::DISTANCE_TYPE_KEY;
use crate::IndexMetadata as IndexMetaSchema;
use crate::{INDEX_AUXILIARY_FILE_NAME, INDEX_METADATA_SCHEMA_KEY};
use arrow_schema::{DataType, Schema as ArrowSchema};
use lance_file::reader::{FileReader as V2Reader, FileReaderOptions as V2ReaderOptions};
use lance_file::writer::FileWriter as V2Writer;
use lance_io::scheduler::{ScanScheduler, SchedulerConfig};
use lance_io::utils::CachedFileSize;
use lance_linalg::distance::DistanceType;

/// Detect and return supported index type from reader and schema.
///
/// This is a lightweight wrapper around SupportedIndexType::detect to keep
/// detection logic self-contained within this module.
fn detect_supported_index_type(
    reader: &V2Reader,
    schema: &ArrowSchema,
) -> Result<SupportedIndexType> {
    SupportedIndexType::detect(reader, schema)
}

/// Parse a deterministic sort key from a `partial_*` directory name.
///
/// The returned tuple is `(min_fragment_id, dataset_version)` where:
/// - `min_fragment_id` is taken from the first integer token; if missing or parse
///   fails, `u32::MAX` is used.
/// - `dataset_version` is taken from the second integer token; if missing or
///   parse fails, `0` is used.
fn parse_partial_dir_key(pname: &str) -> (u32, u64) {
    // Strip well-known prefix but still handle unexpected names defensively.
    let name = pname.strip_prefix("partial_").unwrap_or(pname);

    let mut ints: Vec<String> = Vec::new();
    let mut current = String::new();

    for ch in name.chars() {
        if ch.is_ascii_digit() {
            current.push(ch);
        } else if !current.is_empty() {
            ints.push(current.clone());
            current.clear();
        }
    }
    if !current.is_empty() {
        ints.push(current);
    }

    let min_fragment_id = ints
        .get(0)
        .and_then(|s| s.parse::<u32>().ok())
        .unwrap_or(u32::MAX);
    let dataset_version = ints.get(1).and_then(|s| s.parse::<u64>().ok()).unwrap_or(0);

    (min_fragment_id, dataset_version)
}

/// Derive the sort key for a partial shard from its parent directory name.
fn partial_aux_sort_key(path: &object_store::path::Path) -> (u32, u64) {
    let parts: Vec<_> = path.parts().collect();
    if parts.len() < 2 {
        return (u32::MAX, 0);
    }
    let parent = parts[parts.len() - 2].as_ref();
    parse_partial_dir_key(parent)
}

/// Merge all partial_* vector index auxiliary files under `index_dir/{uuid}/partial_*/auxiliary.idx`
/// into `index_dir/{uuid}/auxiliary.idx`.
///
/// Supports IVF_FLAT, IVF_PQ, IVF_SQ, IVF_HNSW_FLAT, IVF_HNSW_PQ, IVF_HNSW_SQ storage types.
/// For PQ and SQ, this assumes all partial indices share the same quantizer/codebook
/// and distance type; it will reuse the first encountered metadata.
pub async fn merge_partial_vector_auxiliary_files(
    object_store: &lance_io::object_store::ObjectStore,
    index_dir: &object_store::path::Path,
) -> Result<()> {
    let mut aux_paths: Vec<object_store::path::Path> = Vec::new();
    let mut stream = object_store.list(Some(index_dir.clone()));
    while let Some(item) = stream.next().await {
        if let Ok(meta) = item {
            if let Some(fname) = meta.location.filename() {
                if fname == INDEX_AUXILIARY_FILE_NAME {
                    // Check parent dir name starts with partial_
                    let parts: Vec<_> = meta.location.parts().collect();
                    if parts.len() >= 2 {
                        let pname = parts[parts.len() - 2].as_ref();
                        if pname.starts_with("partial_") {
                            aux_paths.push(meta.location.clone());
                        }
                    }
                }
            }
        }
    }

    // Ensure deterministic ordering of partial_* shards before merging.
    aux_paths.sort_by_key(|p| partial_aux_sort_key(p));

    if aux_paths.is_empty() {
        // If a unified auxiliary file already exists at the root, no merge is required.
        let aux_out = index_dir.child(INDEX_AUXILIARY_FILE_NAME);
        if object_store.exists(&aux_out).await.unwrap_or(false) {
            log::warn!(
                "No partial_* auxiliary files found under index dir: {}, but unified auxiliary file already exists; skipping merge",
                index_dir
            );
            return Ok(());
        }
        // For certain index types (e.g., FLAT/HNSW-only) the merge may be a no-op in distributed setups
        // where shards were committed directly. In such cases, proceed without error to avoid blocking
        // index manifest merge. PQ/SQ variants still require merging artifacts and will be handled by
        // downstream open logic if missing.
        log::warn!(
            "No partial_* auxiliary files found under index dir: {}; proceeding without merge for index types that do not require auxiliary shards",
            index_dir
        );
        return Ok(());
    }

    // Prepare IVF model and storage metadata aggregation
    let mut distance_type: Option<DistanceType> = None;
    let mut pq_meta: Option<ProductQuantizationMetadata> = None;
    let mut sq_meta: Option<ScalarQuantizationMetadata> = None;
    let mut dim: Option<usize> = None;
    let mut detected_index_type: Option<SupportedIndexType> = None;

    // Prepare output path; we'll create writer once when we know schema
    let aux_out = index_dir.child(INDEX_AUXILIARY_FILE_NAME);

    // We'll delay creating the V2 writer until we know the vector schema (dim and quantizer type)
    let mut v2w_opt: Option<V2Writer> = None;

    // We'll also need a scheduler to open readers efficiently
    let sched = ScanScheduler::new(
        Arc::new(object_store.clone()),
        SchedulerConfig::max_bandwidth(object_store),
    );

    // Track IVF partition count consistency and accumulate lengths per partition
    let mut nlist_opt: Option<usize> = None;
    let mut accumulated_lengths: Vec<u32> = Vec::new();
    let mut first_centroids: Option<FixedSizeListArray> = None;

    // Track per-shard IVF lengths to reorder writing to partitions later
    let mut shard_infos: Vec<(object_store::path::Path, Vec<u32>)> = Vec::new();

    // Iterate over each shard auxiliary file and merge its metadata and collect lengths
    for aux in &aux_paths {
        let fh = sched.open_file(aux, &CachedFileSize::unknown()).await?;
        let reader = V2Reader::try_open(
            fh,
            None,
            Arc::default(),
            &lance_core::cache::LanceCache::no_cache(),
            V2ReaderOptions::default(),
        )
        .await?;
        let meta = reader.metadata();

        // Read distance type
        let dt = meta
            .file_schema
            .metadata
            .get(DISTANCE_TYPE_KEY)
            .ok_or_else(|| Error::Index {
                message: format!("Missing {} in shard", DISTANCE_TYPE_KEY),
                location: location!(),
            })?;
        let dt: DistanceType = DistanceType::try_from(dt.as_str())?;
        if distance_type.is_none() {
            distance_type = Some(dt);
        } else if distance_type.as_ref().map(|v| *v != dt).unwrap_or(false) {
            return Err(Error::Index {
                message: "Distance type mismatch across shards".to_string(),
                location: location!(),
            });
        }

        // Detect index type (first iteration only)
        if detected_index_type.is_none() {
            // Try to derive precise type from sibling partial index.idx metadata if available
            // Try resolve sibling index.idx path by trimming the last component of aux path
            let parent_str = {
                let s = aux.as_ref();
                if let Some((p, _)) = s.trim_end_matches('/').rsplit_once('/') {
                    p.to_string()
                } else {
                    s.to_string()
                }
            };
            let idx_path = object_store::path::Path::from(format!(
                "{}/{}",
                parent_str,
                crate::INDEX_FILE_NAME
            ));
            if object_store.exists(&idx_path).await.unwrap_or(false) {
                let fh2 = sched
                    .open_file(&idx_path, &CachedFileSize::unknown())
                    .await?;
                let idx_reader = V2Reader::try_open(
                    fh2,
                    None,
                    Arc::default(),
                    &lance_core::cache::LanceCache::no_cache(),
                    V2ReaderOptions::default(),
                )
                .await?;
                if let Some(idx_meta_json) = idx_reader
                    .metadata()
                    .file_schema
                    .metadata
                    .get(INDEX_METADATA_SCHEMA_KEY)
                {
                    let idx_meta: IndexMetaSchema = serde_json::from_str(idx_meta_json)?;
                    detected_index_type = Some(match idx_meta.index_type.as_str() {
                        "IVF_FLAT" => SupportedIndexType::IvfFlat,
                        "IVF_PQ" => SupportedIndexType::IvfPq,
                        "IVF_SQ" => SupportedIndexType::IvfSq,
                        "IVF_HNSW_FLAT" => SupportedIndexType::IvfHnswFlat,
                        "IVF_HNSW_PQ" => SupportedIndexType::IvfHnswPq,
                        "IVF_HNSW_SQ" => SupportedIndexType::IvfHnswSq,
                        other => {
                            return Err(Error::Index {
                                message: format!(
                                    "Unsupported index type in shard index.idx: {}",
                                    other
                                ),
                                location: location!(),
                            });
                        }
                    });
                }
            }
            // Fallback: infer from auxiliary schema
            if detected_index_type.is_none() {
                let schema_arrow: ArrowSchema = reader.schema().as_ref().into();
                detected_index_type = Some(detect_supported_index_type(&reader, &schema_arrow)?);
            }
        }

        // Read IVF lengths from global buffer
        let ivf_idx: u32 = reader
            .metadata()
            .file_schema
            .metadata
            .get(IVF_METADATA_KEY)
            .ok_or_else(|| Error::Index {
                message: "IVF meta missing".to_string(),
                location: location!(),
            })?
            .parse()
            .map_err(|_| Error::Index {
                message: "IVF index parse error".to_string(),
                location: location!(),
            })?;
        let bytes = reader.read_global_buffer(ivf_idx).await?;
        let pb_ivf: pb::Ivf = prost::Message::decode(bytes)?;
        let lengths = pb_ivf.lengths.clone();
        let nlist = lengths.len();

        if nlist_opt.is_none() {
            nlist_opt = Some(nlist);
            accumulated_lengths = vec![0; nlist];
            // Try load centroids tensor if present
            if let Some(tensor) = pb_ivf.centroids_tensor.as_ref() {
                let arr = FixedSizeListArray::try_from(tensor)?;
                first_centroids = Some(arr.clone());
                let d0 = arr.value_length() as usize;
                if dim.is_none() {
                    dim = Some(d0);
                }
            }
        } else if nlist_opt.as_ref().map(|v| *v != nlist).unwrap_or(false) {
            return Err(Error::Index {
                message: "IVF partition count mismatch across shards".to_string(),
                location: location!(),
            });
        }

        // Handle logic based on detected index type
        let idx_type = detected_index_type.ok_or_else(|| Error::Index {
            message: "Unable to detect index type".to_string(),
            location: location!(),
        })?;
        match idx_type {
            SupportedIndexType::IvfSq => {
                // Handle Scalar Quantization (SQ) storage for IVF_SQ
                let sq_json = if let Some(sq_json) =
                    reader.metadata().file_schema.metadata.get(SQ_METADATA_KEY)
                {
                    sq_json.clone()
                } else if let Some(storage_meta_json) = reader
                    .metadata()
                    .file_schema
                    .metadata
                    .get(STORAGE_METADATA_KEY)
                {
                    // Try to extract SQ metadata from storage metadata
                    let storage_metadata_vec: Vec<String> = serde_json::from_str(storage_meta_json)
                        .map_err(|e| Error::Index {
                            message: format!("Failed to parse storage metadata: {}", e),
                            location: location!(),
                        })?;
                    if let Some(first_meta) = storage_metadata_vec.first() {
                        // Check if this is SQ metadata by trying to parse it
                        if let Ok(_sq_meta) =
                            serde_json::from_str::<ScalarQuantizationMetadata>(first_meta)
                        {
                            first_meta.clone()
                        } else {
                            return Err(Error::Index {
                                message: "SQ metadata missing in storage metadata".to_string(),
                                location: location!(),
                            });
                        }
                    } else {
                        return Err(Error::Index {
                            message: "SQ metadata missing in storage metadata".to_string(),
                            location: location!(),
                        });
                    }
                } else {
                    return Err(Error::Index {
                        message: "SQ metadata missing".to_string(),
                        location: location!(),
                    });
                };

                let sq_meta_parsed: ScalarQuantizationMetadata = serde_json::from_str(&sq_json)
                    .map_err(|e| Error::Index {
                        message: format!("SQ metadata parse error: {}", e),
                        location: location!(),
                    })?;

                let d0 = sq_meta_parsed.dim;
                dim.get_or_insert(d0);
                if let Some(dprev) = dim {
                    if dprev != d0 {
                        return Err(Error::Index {
                            message: "Dimension mismatch across shards".to_string(),
                            location: location!(),
                        });
                    }
                }

                if sq_meta.is_none() {
                    sq_meta = Some(sq_meta_parsed.clone());
                }
                if v2w_opt.is_none() {
                    let w = init_writer_for_sq(object_store, &aux_out, dt, &sq_meta_parsed).await?;
                    v2w_opt = Some(w);
                }
            }
            SupportedIndexType::IvfPq => {
                // Handle Product Quantization (PQ) storage
                // Load PQ metadata JSON; construct ProductQuantizationMetadata
                let pm_json = if let Some(pm_json) =
                    reader.metadata().file_schema.metadata.get(PQ_METADATA_KEY)
                {
                    pm_json.clone()
                } else if let Some(storage_meta_json) = reader
                    .metadata()
                    .file_schema
                    .metadata
                    .get(STORAGE_METADATA_KEY)
                {
                    // Try to extract PQ metadata from storage metadata
                    let storage_metadata_vec: Vec<String> = serde_json::from_str(storage_meta_json)
                        .map_err(|e| Error::Index {
                            message: format!("Failed to parse storage metadata: {}", e),
                            location: location!(),
                        })?;
                    if let Some(first_meta) = storage_metadata_vec.first() {
                        // Check if this is PQ metadata by trying to parse it
                        if let Ok(_pq_meta) =
                            serde_json::from_str::<ProductQuantizationMetadata>(first_meta)
                        {
                            first_meta.clone()
                        } else {
                            return Err(Error::Index {
                                message: "PQ metadata missing in storage metadata".to_string(),
                                location: location!(),
                            });
                        }
                    } else {
                        return Err(Error::Index {
                            message: "PQ metadata missing in storage metadata".to_string(),
                            location: location!(),
                        });
                    }
                } else {
                    return Err(Error::Index {
                        message: "PQ metadata missing".to_string(),
                        location: location!(),
                    });
                };
                let mut pm: ProductQuantizationMetadata =
                    serde_json::from_str(&pm_json).map_err(|e| Error::Index {
                        message: format!("PQ metadata parse error: {}", e),
                        location: location!(),
                    })?;
                // Load codebook from global buffer if not present
                if pm.codebook.is_none() {
                    let tensor_bytes = reader
                        .read_global_buffer(pm.codebook_position as u32)
                        .await?;
                    let codebook_tensor: crate::pb::Tensor = prost::Message::decode(tensor_bytes)?;
                    pm.codebook = Some(FixedSizeListArray::try_from(&codebook_tensor)?);
                }
                let d0 = pm.dimension;
                dim.get_or_insert(d0);
                if let Some(dprev) = dim {
                    if dprev != d0 {
                        return Err(Error::Index {
                            message: "Dimension mismatch across shards".to_string(),
                            location: location!(),
                        });
                    }
                }
                if let Some(existing_pm) = pq_meta.as_ref() {
                    // Enforce structural equality
                    if existing_pm.num_sub_vectors != pm.num_sub_vectors
                        || existing_pm.nbits != pm.nbits
                        || existing_pm.dimension != pm.dimension
                    {
                        return Err(Error::Index {
                            message: format!(
                                "Distributed PQ merge: structural mismatch across shards; first(dim={}, m={}, nbits={}), current(dim={}, m={}, nbits={})",
                                existing_pm.dimension,
                                existing_pm.num_sub_vectors,
                                existing_pm.nbits,
                                pm.dimension,
                                pm.num_sub_vectors,
                                pm.nbits
                            ),
                            location: location!(),
                        });
                    }
                    // Enforce codebook equality with tolerance for minor serialization diffs
                    let existing_cb =
                        existing_pm.codebook.as_ref().ok_or_else(|| Error::Index {
                            message: "PQ codebook missing in first shard".to_string(),
                            location: location!(),
                        })?;
                    let current_cb = pm.codebook.as_ref().ok_or_else(|| Error::Index {
                        message: "PQ codebook missing in shard".to_string(),
                        location: location!(),
                    })?;
                    if !fixed_size_list_equal(existing_cb, current_cb) {
                        const TOL: f32 = 1e-5;
                        if !fixed_size_list_almost_equal(existing_cb, current_cb, TOL) {
                            return Err(Error::Index {
                                message: "PQ codebook content mismatch across shards".to_string(),
                                location: location!(),
                            });
                        } else {
                            log::warn!("PQ codebook differs within tolerance; proceeding with first shard codebook");
                        }
                    }
                }
                if pq_meta.is_none() {
                    pq_meta = Some(pm.clone());
                }
                if v2w_opt.is_none() {
                    let w = init_writer_for_pq(object_store, &aux_out, dt, &pm).await?;
                    v2w_opt = Some(w);
                }
            }
            SupportedIndexType::IvfFlat => {
                // Handle FLAT storage
                // FLAT: infer dimension from vector column using first shard's schema
                let schema: ArrowSchema = reader.schema().as_ref().into();
                let flat_field = schema
                    .fields
                    .iter()
                    .find(|f| f.name() == crate::vector::flat::storage::FLAT_COLUMN)
                    .ok_or_else(|| Error::Index {
                        message: "FLAT column missing".to_string(),
                        location: location!(),
                    })?;
                let d0 = match flat_field.data_type() {
                    DataType::FixedSizeList(_, sz) => *sz as usize,
                    _ => 0,
                };
                dim.get_or_insert(d0);
                if let Some(dprev) = dim {
                    if dprev != d0 {
                        return Err(Error::Index {
                            message: "Dimension mismatch across shards".to_string(),
                            location: location!(),
                        });
                    }
                }
                if v2w_opt.is_none() {
                    let w = init_writer_for_flat(object_store, &aux_out, d0, dt).await?;
                    v2w_opt = Some(w);
                }
            }
            SupportedIndexType::IvfHnswFlat => {
                // Treat HNSW_FLAT storage the same as FLAT: create schema with ROW_ID + flat vectors
                // Determine dimension from shard schema (flat column) or fallback to STORAGE_METADATA_KEY
                let schema_arrow: ArrowSchema = reader.schema().as_ref().into();
                // Try to find flat column and derive dim
                let d0 = if let Some(flat_field) = schema_arrow
                    .fields
                    .iter()
                    .find(|f| f.name() == crate::vector::flat::storage::FLAT_COLUMN)
                {
                    match flat_field.data_type() {
                        DataType::FixedSizeList(_, sz) => *sz as usize,
                        _ => 0,
                    }
                } else {
                    // Fallback to STORAGE_METADATA_KEY FlatMetadata
                    if let Some(storage_meta_json) = reader
                        .metadata()
                        .file_schema
                        .metadata
                        .get(STORAGE_METADATA_KEY)
                    {
                        let storage_metadata_vec: Vec<String> =
                            serde_json::from_str(storage_meta_json).map_err(|e| Error::Index {
                                message: format!("Failed to parse storage metadata: {}", e),
                                location: location!(),
                            })?;
                        if let Some(first_meta) = storage_metadata_vec.first() {
                            if let Ok(flat_meta) = serde_json::from_str::<FlatMetadata>(first_meta)
                            {
                                flat_meta.dim
                            } else {
                                return Err(Error::Index {
                                    message: "FLAT metadata missing in storage metadata"
                                        .to_string(),
                                    location: location!(),
                                });
                            }
                        } else {
                            return Err(Error::Index {
                                message: "FLAT metadata missing in storage metadata".to_string(),
                                location: location!(),
                            });
                        }
                    } else {
                        return Err(Error::Index {
                            message: "FLAT column missing and no storage metadata".to_string(),
                            location: location!(),
                        });
                    }
                };
                dim.get_or_insert(d0);
                if let Some(dprev) = dim {
                    if dprev != d0 {
                        return Err(Error::Index {
                            message: "Dimension mismatch across shards".to_string(),
                            location: location!(),
                        });
                    }
                }
                if v2w_opt.is_none() {
                    let w = init_writer_for_flat(object_store, &aux_out, d0, dt).await?;
                    v2w_opt = Some(w);
                }
            }
            SupportedIndexType::IvfHnswPq => {
                // Treat HNSW_PQ storage the same as PQ: reuse PQ metadata and schema creation
                let pm_json = if let Some(pm_json) =
                    reader.metadata().file_schema.metadata.get(PQ_METADATA_KEY)
                {
                    pm_json.clone()
                } else if let Some(storage_meta_json) = reader
                    .metadata()
                    .file_schema
                    .metadata
                    .get(STORAGE_METADATA_KEY)
                {
                    let storage_metadata_vec: Vec<String> = serde_json::from_str(storage_meta_json)
                        .map_err(|e| Error::Index {
                            message: format!("Failed to parse storage metadata: {}", e),
                            location: location!(),
                        })?;
                    if let Some(first_meta) = storage_metadata_vec.first() {
                        if let Ok(_pq_meta) =
                            serde_json::from_str::<ProductQuantizationMetadata>(first_meta)
                        {
                            first_meta.clone()
                        } else {
                            return Err(Error::Index {
                                message: "PQ metadata missing in storage metadata".to_string(),
                                location: location!(),
                            });
                        }
                    } else {
                        return Err(Error::Index {
                            message: "PQ metadata missing in storage metadata".to_string(),
                            location: location!(),
                        });
                    }
                } else {
                    return Err(Error::Index {
                        message: "PQ metadata missing".to_string(),
                        location: location!(),
                    });
                };
                let mut pm: ProductQuantizationMetadata =
                    serde_json::from_str(&pm_json).map_err(|e| Error::Index {
                        message: format!("PQ metadata parse error: {}", e),
                        location: location!(),
                    })?;
                if pm.codebook.is_none() {
                    let tensor_bytes = reader
                        .read_global_buffer(pm.codebook_position as u32)
                        .await?;
                    let codebook_tensor: crate::pb::Tensor = prost::Message::decode(tensor_bytes)?;
                    pm.codebook = Some(FixedSizeListArray::try_from(&codebook_tensor)?);
                }
                let d0 = pm.dimension;
                dim.get_or_insert(d0);
                if let Some(dprev) = dim {
                    if dprev != d0 {
                        return Err(Error::Index {
                            message: "Dimension mismatch across shards".to_string(),
                            location: location!(),
                        });
                    }
                }
                if let Some(existing_pm) = pq_meta.as_ref() {
                    // Enforce structural equality
                    if existing_pm.num_sub_vectors != pm.num_sub_vectors
                        || existing_pm.nbits != pm.nbits
                        || existing_pm.dimension != pm.dimension
                    {
                        return Err(Error::Index {
                            message: format!(
                                "Distributed PQ merge (HNSW_PQ): structural mismatch across shards; first(dim={}, m={}, nbits={}), current(dim={}, m={}, nbits={})",
                                existing_pm.dimension,
                                existing_pm.num_sub_vectors,
                                existing_pm.nbits,
                                pm.dimension,
                                pm.num_sub_vectors,
                                pm.nbits
                            ),
                            location: location!(),
                        });
                    }
                    // Enforce codebook equality with tolerance for minor serialization diffs
                    let existing_cb =
                        existing_pm.codebook.as_ref().ok_or_else(|| Error::Index {
                            message: "PQ codebook missing in first shard".to_string(),
                            location: location!(),
                        })?;
                    let current_cb = pm.codebook.as_ref().ok_or_else(|| Error::Index {
                        message: "PQ codebook missing in shard".to_string(),
                        location: location!(),
                    })?;
                    if !fixed_size_list_equal(existing_cb, current_cb) {
                        const TOL: f32 = 1e-5;
                        if !fixed_size_list_almost_equal(existing_cb, current_cb, TOL) {
                            return Err(Error::Index {
                                message: "PQ codebook content mismatch across shards".to_string(),
                                location: location!(),
                            });
                        } else {
                            log::warn!("PQ codebook differs within tolerance; proceeding with first shard codebook");
                        }
                    }
                }
                if pq_meta.is_none() {
                    pq_meta = Some(pm.clone());
                }
                if v2w_opt.is_none() {
                    let w = init_writer_for_pq(object_store, &aux_out, dt, &pm).await?;
                    v2w_opt = Some(w);
                }
            }
            SupportedIndexType::IvfHnswSq => {
                // Treat HNSW_SQ storage the same as SQ: reuse SQ metadata and schema creation
                let sq_json = if let Some(sq_json) =
                    reader.metadata().file_schema.metadata.get(SQ_METADATA_KEY)
                {
                    sq_json.clone()
                } else if let Some(storage_meta_json) = reader
                    .metadata()
                    .file_schema
                    .metadata
                    .get(STORAGE_METADATA_KEY)
                {
                    let storage_metadata_vec: Vec<String> = serde_json::from_str(storage_meta_json)
                        .map_err(|e| Error::Index {
                            message: format!("Failed to parse storage metadata: {}", e),
                            location: location!(),
                        })?;
                    if let Some(first_meta) = storage_metadata_vec.first() {
                        if let Ok(_sq_meta) =
                            serde_json::from_str::<ScalarQuantizationMetadata>(first_meta)
                        {
                            first_meta.clone()
                        } else {
                            return Err(Error::Index {
                                message: "SQ metadata missing in storage metadata".to_string(),
                                location: location!(),
                            });
                        }
                    } else {
                        return Err(Error::Index {
                            message: "SQ metadata missing in storage metadata".to_string(),
                            location: location!(),
                        });
                    }
                } else {
                    return Err(Error::Index {
                        message: "SQ metadata missing".to_string(),
                        location: location!(),
                    });
                };
                let sq_meta_parsed: ScalarQuantizationMetadata = serde_json::from_str(&sq_json)
                    .map_err(|e| Error::Index {
                        message: format!("SQ metadata parse error: {}", e),
                        location: location!(),
                    })?;
                let d0 = sq_meta_parsed.dim;
                dim.get_or_insert(d0);
                if let Some(dprev) = dim {
                    if dprev != d0 {
                        return Err(Error::Index {
                            message: "Dimension mismatch across shards".to_string(),
                            location: location!(),
                        });
                    }
                }
                if sq_meta.is_none() {
                    sq_meta = Some(sq_meta_parsed.clone());
                }
                if v2w_opt.is_none() {
                    let w = init_writer_for_sq(object_store, &aux_out, dt, &sq_meta_parsed).await?;
                    v2w_opt = Some(w);
                }
            }
        }

        // Collect per-shard lengths to write grouped by partition later
        shard_infos.push((aux.clone(), lengths.clone()));
        // Accumulate overall lengths per partition for unified IVF model
        for pid in 0..nlist {
            let part_len = lengths[pid];
            accumulated_lengths[pid] = accumulated_lengths[pid].saturating_add(part_len);
        }
    }

    // Write rows grouped by partition across all shards to ensure contiguous ranges per partition

    if v2w_opt.is_none() {
        return Err(Error::Index {
            message: "Failed to initialize unified writer".to_string(),
            location: location!(),
        });
    }
    let nlist = nlist_opt.ok_or_else(|| Error::Index {
        message: "Missing IVF partition count".to_string(),
        location: location!(),
    })?;
    for pid in 0..nlist {
        for (path, lens) in shard_infos.iter() {
            let part_len = lens[pid] as usize;
            if part_len == 0 {
                continue;
            }
            let offset: usize = lens.iter().take(pid).map(|x| *x as usize).sum();
            let fh = sched.open_file(path, &CachedFileSize::unknown()).await?;
            let reader = V2Reader::try_open(
                fh,
                None,
                Arc::default(),
                &lance_core::cache::LanceCache::no_cache(),
                V2ReaderOptions::default(),
            )
            .await?;
            if let Some(w) = v2w_opt.as_mut() {
                write_partition_rows(&reader, w, offset..offset + part_len).await?;
            }
        }
    }

    // Write unified IVF metadata into global buffer & set schema metadata
    if let Some(w) = v2w_opt.as_mut() {
        let mut ivf_model = if let Some(c) = first_centroids {
            IvfStorageModel::new(c, None)
        } else {
            IvfStorageModel::empty()
        };
        for len in accumulated_lengths.iter() {
            ivf_model.add_partition(*len);
        }
        let dt2 = distance_type.ok_or_else(|| Error::Index {
            message: "Distance type missing".to_string(),
            location: location!(),
        })?;
        let idx_type_final = detected_index_type.ok_or_else(|| Error::Index {
            message: "Unable to detect index type".to_string(),
            location: location!(),
        })?;
        write_unified_ivf_and_index_metadata(w, &ivf_model, dt2, idx_type_final).await?;
        w.finish().await?;
    } else {
        return Err(Error::Index {
            message: "Failed to initialize unified writer".to_string(),
            location: location!(),
        });
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    use arrow_array::{FixedSizeListArray, Float32Array, RecordBatch, UInt64Array, UInt8Array};
    use arrow_schema::Field;
    use bytes::Bytes;
    use futures::StreamExt;
    use lance_arrow::FixedSizeListArrayExt;
    use lance_core::ROW_ID_FIELD;
    use lance_file::writer::FileWriterOptions as V2WriterOptions;
    use lance_io::object_store::ObjectStore;
    use lance_io::scheduler::{ScanScheduler, SchedulerConfig};
    use lance_io::utils::CachedFileSize;
    use lance_linalg::distance::DistanceType;
    use object_store::path::Path;
    use prost::Message;

    async fn write_flat_partial_aux(
        store: &ObjectStore,
        aux_path: &Path,
        dim: i32,
        lengths: &[u32],
        base_row_id: u64,
        distance_type: DistanceType,
    ) -> Result<usize> {
        let arrow_schema = ArrowSchema::new(vec![
            (*ROW_ID_FIELD).clone(),
            Field::new(
                crate::vector::flat::storage::FLAT_COLUMN,
                DataType::FixedSizeList(Arc::new(Field::new("item", DataType::Float32, true)), dim),
                true,
            ),
        ]);

        let writer = store.create(aux_path).await?;
        let mut v2w = V2Writer::try_new(
            writer,
            lance_core::datatypes::Schema::try_from(&arrow_schema)?,
            V2WriterOptions::default(),
        )?;

        // Distance type metadata for this shard.
        v2w.add_schema_metadata(DISTANCE_TYPE_KEY, distance_type.to_string());

        // IVF metadata: only lengths are needed by the merger.
        let ivf_meta = pb::Ivf {
            centroids: Vec::new(),
            offsets: Vec::new(),
            lengths: lengths.to_vec(),
            centroids_tensor: None,
            loss: None,
        };
        let buf = Bytes::from(ivf_meta.encode_to_vec());
        let pos = v2w.add_global_buffer(buf).await?;
        v2w.add_schema_metadata(IVF_METADATA_KEY, pos.to_string());

        // Build row ids and vectors grouped by partition so that ranges match lengths.
        let total_rows: usize = lengths.iter().map(|v| *v as usize).sum();
        let mut row_ids = Vec::with_capacity(total_rows);
        let mut values = Vec::with_capacity(total_rows * dim as usize);

        let mut current_row_id = base_row_id;
        for (pid, len) in lengths.iter().enumerate() {
            for _ in 0..*len {
                row_ids.push(current_row_id);
                current_row_id += 1;
                for d in 0..dim {
                    // Simple deterministic payload; only layout matters for merge.
                    values.push(pid as f32 + d as f32 * 0.01);
                }
            }
        }

        let row_id_arr = UInt64Array::from(row_ids);
        let value_arr = Float32Array::from(values);
        let fsl = FixedSizeListArray::try_new_from_values(value_arr, dim).unwrap();
        let batch = RecordBatch::try_new(
            Arc::new(arrow_schema),
            vec![Arc::new(row_id_arr), Arc::new(fsl)],
        )
        .unwrap();

        v2w.write_batch(&batch).await?;
        v2w.finish().await?;
        Ok(total_rows)
    }

    #[tokio::test]
    async fn test_merge_ivf_flat_success_basic() {
        let object_store = ObjectStore::memory();
        let index_dir = Path::from("index/uuid");

        let partial0 = index_dir.child("partial_0");
        let partial1 = index_dir.child("partial_1");
        let aux0 = partial0.child(INDEX_AUXILIARY_FILE_NAME);
        let aux1 = partial1.child(INDEX_AUXILIARY_FILE_NAME);

        let lengths0 = vec![2_u32, 1_u32];
        let lengths1 = vec![1_u32, 2_u32];
        let dim = 2_i32;

        write_flat_partial_aux(&object_store, &aux0, dim, &lengths0, 0, DistanceType::L2)
            .await
            .unwrap();
        write_flat_partial_aux(&object_store, &aux1, dim, &lengths1, 100, DistanceType::L2)
            .await
            .unwrap();

        merge_partial_vector_auxiliary_files(&object_store, &index_dir)
            .await
            .unwrap();

        let aux_out = index_dir.child(INDEX_AUXILIARY_FILE_NAME);
        assert!(object_store.exists(&aux_out).await.unwrap());

        // Use ScanScheduler to obtain a FileScheduler (required by V2Reader::try_open)
        let sched = ScanScheduler::new(
            Arc::new(object_store.clone()),
            SchedulerConfig::max_bandwidth(&object_store),
        );
        let fh = sched
            .open_file(&aux_out, &CachedFileSize::unknown())
            .await
            .unwrap();
        let reader = V2Reader::try_open(
            fh,
            None,
            Arc::default(),
            &lance_core::cache::LanceCache::no_cache(),
            V2ReaderOptions::default(),
        )
        .await
        .unwrap();
        let meta = reader.metadata();

        // Validate IVF lengths aggregation.
        let ivf_idx: u32 = meta
            .file_schema
            .metadata
            .get(IVF_METADATA_KEY)
            .unwrap()
            .parse()
            .unwrap();
        let bytes = reader.read_global_buffer(ivf_idx).await.unwrap();
        let pb_ivf: pb::Ivf = prost::Message::decode(bytes).unwrap();
        let expected_lengths: Vec<u32> = lengths0
            .iter()
            .zip(lengths1.iter())
            .map(|(a, b)| *a + *b)
            .collect();
        assert_eq!(pb_ivf.lengths, expected_lengths);

        // Validate index metadata schema.
        let idx_meta_json = meta
            .file_schema
            .metadata
            .get(INDEX_METADATA_SCHEMA_KEY)
            .unwrap();
        let idx_meta: IndexMetaSchema = serde_json::from_str(idx_meta_json).unwrap();
        assert_eq!(idx_meta.index_type, "IVF_FLAT");
        assert_eq!(idx_meta.distance_type, DistanceType::L2.to_string());

        // Validate total number of rows.
        let mut total_rows = 0usize;
        let mut stream = reader
            .read_stream(
                lance_io::ReadBatchParams::RangeFull,
                u32::MAX,
                4,
                lance_encoding::decoder::FilterExpression::no_filter(),
            )
            .unwrap();
        while let Some(batch) = stream.next().await {
            total_rows += batch.unwrap().num_rows();
        }
        let expected_total: usize = expected_lengths.iter().map(|v| *v as usize).sum();
        assert_eq!(total_rows, expected_total);
    }

    #[tokio::test]
    async fn test_merge_distance_type_mismatch() {
        let object_store = ObjectStore::memory();
        let index_dir = Path::from("index/uuid");

        let partial0 = index_dir.child("partial_0");
        let partial1 = index_dir.child("partial_1");
        let aux0 = partial0.child(INDEX_AUXILIARY_FILE_NAME);
        let aux1 = partial1.child(INDEX_AUXILIARY_FILE_NAME);

        let lengths = vec![2_u32, 2_u32];
        let dim = 2_i32;

        write_flat_partial_aux(&object_store, &aux0, dim, &lengths, 0, DistanceType::L2)
            .await
            .unwrap();
        write_flat_partial_aux(
            &object_store,
            &aux1,
            dim,
            &lengths,
            100,
            DistanceType::Cosine,
        )
        .await
        .unwrap();

        let res = merge_partial_vector_auxiliary_files(&object_store, &index_dir).await;
        match res {
            Err(Error::Index { message, .. }) => {
                assert!(
                    message.contains("Distance type mismatch"),
                    "unexpected message: {}",
                    message
                );
            }
            other => panic!(
                "expected Error::Index for distance type mismatch, got {:?}",
                other
            ),
        }
    }

    #[allow(clippy::too_many_arguments)]
    async fn write_pq_partial_aux(
        store: &ObjectStore,
        aux_path: &Path,
        nbits: u32,
        num_sub_vectors: usize,
        dimension: usize,
        lengths: &[u32],
        base_row_id: u64,
        distance_type: DistanceType,
        codebook: &FixedSizeListArray,
    ) -> Result<usize> {
        let num_bytes = if nbits == 4 {
            // Two 4-bit codes per byte.
            num_sub_vectors / 2
        } else {
            num_sub_vectors
        };

        let arrow_schema = ArrowSchema::new(vec![
            (*ROW_ID_FIELD).clone(),
            Field::new(
                crate::vector::PQ_CODE_COLUMN,
                DataType::FixedSizeList(
                    Arc::new(Field::new("item", DataType::UInt8, true)),
                    num_bytes as i32,
                ),
                true,
            ),
        ]);

        let writer = store.create(aux_path).await?;
        let mut v2w = V2Writer::try_new(
            writer,
            lance_core::datatypes::Schema::try_from(&arrow_schema)?,
            V2WriterOptions::default(),
        )?;

        // Distance type metadata for this shard.
        v2w.add_schema_metadata(DISTANCE_TYPE_KEY, distance_type.to_string());

        // PQ metadata with codebook stored in a global buffer.
        let mut pq_meta = ProductQuantizationMetadata {
            codebook_position: 0,
            nbits,
            num_sub_vectors,
            dimension,
            codebook: Some(codebook.clone()),
            codebook_tensor: Vec::new(),
            transposed: true,
        };

        let codebook_tensor: pb::Tensor = pb::Tensor::try_from(codebook)?;
        let codebook_buf = Bytes::from(codebook_tensor.encode_to_vec());
        let codebook_pos = v2w.add_global_buffer(codebook_buf).await?;
        pq_meta.codebook_position = codebook_pos as usize;

        let pq_meta_json = serde_json::to_string(&pq_meta)?;
        v2w.add_schema_metadata(PQ_METADATA_KEY, pq_meta_json);

        // IVF metadata: only lengths are needed by the merger.
        let ivf_meta = pb::Ivf {
            centroids: Vec::new(),
            offsets: Vec::new(),
            lengths: lengths.to_vec(),
            centroids_tensor: None,
            loss: None,
        };
        let buf = Bytes::from(ivf_meta.encode_to_vec());
        let ivf_pos = v2w.add_global_buffer(buf).await?;
        v2w.add_schema_metadata(IVF_METADATA_KEY, ivf_pos.to_string());

        // Build row ids and PQ codes grouped by partition so that ranges match lengths.
        let total_rows: usize = lengths.iter().map(|v| *v as usize).sum();
        let mut row_ids = Vec::with_capacity(total_rows);
        let mut codes = Vec::with_capacity(total_rows * num_bytes);

        let mut current_row_id = base_row_id;
        for (pid, len) in lengths.iter().enumerate() {
            for _ in 0..*len {
                row_ids.push(current_row_id);
                current_row_id += 1;
                for b in 0..num_bytes {
                    // Simple deterministic payload; merge only cares about layout.
                    codes.push((pid + b) as u8);
                }
            }
        }

        let row_id_arr = UInt64Array::from(row_ids);
        let codes_arr = UInt8Array::from(codes);
        let codes_fsl =
            FixedSizeListArray::try_new_from_values(codes_arr, num_bytes as i32).unwrap();
        let batch = RecordBatch::try_new(
            Arc::new(arrow_schema),
            vec![Arc::new(row_id_arr), Arc::new(codes_fsl)],
        )
        .unwrap();

        v2w.write_batch(&batch).await?;
        v2w.finish().await?;
        Ok(total_rows)
    }

    #[tokio::test]
    async fn test_merge_ivf_pq_success() {
        let object_store = ObjectStore::memory();
        let index_dir = Path::from("index/uuid_pq");

        let partial0 = index_dir.child("partial_0");
        let partial1 = index_dir.child("partial_1");
        let aux0 = partial0.child(INDEX_AUXILIARY_FILE_NAME);
        let aux1 = partial1.child(INDEX_AUXILIARY_FILE_NAME);

        let lengths0 = vec![2_u32, 1_u32];
        let lengths1 = vec![1_u32, 2_u32];

        // PQ parameters.
        let nbits = 4_u32;
        let num_sub_vectors = 2_usize;
        let dimension = 8_usize;

        // Deterministic PQ codebook shared by both shards.
        let num_centroids = 1_usize << nbits;
        let num_codebook_vectors = num_centroids * num_sub_vectors;
        let total_values = num_codebook_vectors * dimension;
        let values = Float32Array::from_iter((0..total_values).map(|v| v as f32));
        let codebook = FixedSizeListArray::try_new_from_values(values, dimension as i32).unwrap();

        // Non-overlapping row id ranges across shards.
        write_pq_partial_aux(
            &object_store,
            &aux0,
            nbits,
            num_sub_vectors,
            dimension,
            &lengths0,
            0,
            DistanceType::L2,
            &codebook,
        )
        .await
        .unwrap();

        write_pq_partial_aux(
            &object_store,
            &aux1,
            nbits,
            num_sub_vectors,
            dimension,
            &lengths1,
            1_000,
            DistanceType::L2,
            &codebook,
        )
        .await
        .unwrap();

        // Merge PQ auxiliary files.
        merge_partial_vector_auxiliary_files(&object_store, &index_dir)
            .await
            .unwrap();

        // 3) Unified auxiliary file exists.
        let aux_out = index_dir.child(INDEX_AUXILIARY_FILE_NAME);
        assert!(object_store.exists(&aux_out).await.unwrap());

        // Open merged auxiliary file.
        let sched = ScanScheduler::new(
            Arc::new(object_store.clone()),
            SchedulerConfig::max_bandwidth(&object_store),
        );
        let fh = sched
            .open_file(&aux_out, &CachedFileSize::unknown())
            .await
            .unwrap();
        let reader = V2Reader::try_open(
            fh,
            None,
            Arc::default(),
            &lance_core::cache::LanceCache::no_cache(),
            V2ReaderOptions::default(),
        )
        .await
        .unwrap();
        let meta = reader.metadata();

        // 4) Unified IVF metadata lengths equal shard-wise sums.
        let ivf_idx: u32 = meta
            .file_schema
            .metadata
            .get(IVF_METADATA_KEY)
            .unwrap()
            .parse()
            .unwrap();
        let bytes = reader.read_global_buffer(ivf_idx).await.unwrap();
        let pb_ivf: pb::Ivf = prost::Message::decode(bytes).unwrap();
        let expected_lengths: Vec<u32> = lengths0
            .iter()
            .zip(lengths1.iter())
            .map(|(a, b)| *a + *b)
            .collect();
        assert_eq!(pb_ivf.lengths, expected_lengths);

        // 5) Index metadata schema reports IVF_PQ and correct distance type.
        let idx_meta_json = meta
            .file_schema
            .metadata
            .get(INDEX_METADATA_SCHEMA_KEY)
            .unwrap();
        let idx_meta: IndexMetaSchema = serde_json::from_str(idx_meta_json).unwrap();
        assert_eq!(idx_meta.index_type, "IVF_PQ");
        assert_eq!(idx_meta.distance_type, DistanceType::L2.to_string());

        // 6) PQ metadata and codebook are preserved.
        let pq_meta_json = meta.file_schema.metadata.get(PQ_METADATA_KEY).unwrap();
        let pq_meta: ProductQuantizationMetadata = serde_json::from_str(pq_meta_json).unwrap();
        assert_eq!(pq_meta.nbits, nbits);
        assert_eq!(pq_meta.num_sub_vectors, num_sub_vectors);
        assert_eq!(pq_meta.dimension, dimension);

        let codebook_pos = pq_meta.codebook_position as u32;
        let cb_bytes = reader.read_global_buffer(codebook_pos).await.unwrap();
        let cb_tensor: pb::Tensor = prost::Message::decode(cb_bytes).unwrap();
        let merged_codebook = FixedSizeListArray::try_from(&cb_tensor).unwrap();

        assert!(fixed_size_list_equal(&codebook, &merged_codebook));
    }

    #[tokio::test]
    async fn test_merge_ivf_pq_codebook_mismatch() {
        let object_store = ObjectStore::memory();
        let index_dir = Path::from("index/uuid_pq_mismatch");

        let partial0 = index_dir.child("partial_0");
        let partial1 = index_dir.child("partial_1");
        let aux0 = partial0.child(INDEX_AUXILIARY_FILE_NAME);
        let aux1 = partial1.child(INDEX_AUXILIARY_FILE_NAME);

        let lengths0 = vec![2_u32, 1_u32];
        let lengths1 = vec![1_u32, 2_u32];

        // PQ parameters.
        let nbits = 4_u32;
        let num_sub_vectors = 2_usize;
        let dimension = 8_usize;

        // Base PQ codebook for shard 0.
        let num_centroids = 1_usize << nbits;
        let num_codebook_vectors = num_centroids * num_sub_vectors;
        let total_values = num_codebook_vectors * dimension;
        let values0 = Float32Array::from_iter((0..total_values).map(|v| v as f32));
        let codebook0 = FixedSizeListArray::try_new_from_values(values0, dimension as i32).unwrap();

        // Different PQ codebook for shard 1 with values shifted beyond tolerance.
        let values1 = Float32Array::from_iter((0..total_values).map(|v| v as f32 + 1.0));
        let codebook1 = FixedSizeListArray::try_new_from_values(values1, dimension as i32).unwrap();

        // Non-overlapping row id ranges across shards.
        write_pq_partial_aux(
            &object_store,
            &aux0,
            nbits,
            num_sub_vectors,
            dimension,
            &lengths0,
            0,
            DistanceType::L2,
            &codebook0,
        )
        .await
        .unwrap();

        write_pq_partial_aux(
            &object_store,
            &aux1,
            nbits,
            num_sub_vectors,
            dimension,
            &lengths1,
            1_000,
            DistanceType::L2,
            &codebook1,
        )
        .await
        .unwrap();

        let res = merge_partial_vector_auxiliary_files(&object_store, &index_dir).await;
        match res {
            Err(Error::Index { message, .. }) => {
                assert!(
                    message.contains("PQ codebook content mismatch"),
                    "unexpected message: {}",
                    message
                );
            }
            other => panic!(
                "expected Error::Index with PQ codebook content mismatch, got {:?}",
                other
            ),
        }
    }

    #[tokio::test]
    async fn test_merge_ivf_pq_num_sub_vectors_mismatch() {
        let object_store = ObjectStore::memory();
        let index_dir = Path::from("index/uuid_pq_mismatch_m");

        let partial0 = index_dir.child("partial_0");
        let partial1 = index_dir.child("partial_1");
        let aux0 = partial0.child(INDEX_AUXILIARY_FILE_NAME);
        let aux1 = partial1.child(INDEX_AUXILIARY_FILE_NAME);

        let lengths0 = vec![2_u32, 1_u32];
        let lengths1 = vec![1_u32, 2_u32];

        // PQ parameters: same nbits and dimension, different num_sub_vectors.
        let nbits = 4_u32;
        let dimension = 8_usize;
        let num_sub_vectors0 = 4_usize;
        let num_sub_vectors1 = 2_usize;

        // Deterministic PQ codebook shared by both shards.
        let num_centroids = 1_usize << nbits;
        let num_codebook_vectors = num_centroids * num_sub_vectors0.max(num_sub_vectors1);
        let total_values = num_codebook_vectors * dimension;
        let values = Float32Array::from_iter((0..total_values).map(|v| v as f32));
        let codebook = FixedSizeListArray::try_new_from_values(values, dimension as i32).unwrap();

        // Shard 0: num_sub_vectors = 4.
        write_pq_partial_aux(
            &object_store,
            &aux0,
            nbits,
            num_sub_vectors0,
            dimension,
            &lengths0,
            0,
            DistanceType::L2,
            &codebook,
        )
        .await
        .unwrap();

        // Shard 1: num_sub_vectors = 2 (structural mismatch).
        write_pq_partial_aux(
            &object_store,
            &aux1,
            nbits,
            num_sub_vectors1,
            dimension,
            &lengths1,
            10_000,
            DistanceType::L2,
            &codebook,
        )
        .await
        .unwrap();

        let res = merge_partial_vector_auxiliary_files(&object_store, &index_dir).await;
        match res {
            Err(Error::Index { message, .. }) => {
                assert!(
                    message.contains("structural mismatch"),
                    "unexpected message: {}",
                    message
                );
            }
            other => panic!(
                "expected Error::Index for PQ num_sub_vectors mismatch, got {:?}",
                other
            ),
        }
    }
}
