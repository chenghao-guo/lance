# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors

from enum import Enum

from lance.indices.builder import IndexConfig, IndicesBuilder
from lance.indices.ivf import IvfModel
from lance.indices.pq import PqModel

__all__ = ["IndicesBuilder", "IndexConfig", "PqModel", "IvfModel", "IndexFileVersion"]

from lance.lance import indices as _indices


def get_ivf_model(dataset, index_name: str):
    inner = getattr(dataset, "_ds", dataset)
    return _indices.get_ivf_model(inner, index_name)


def get_pq_codebook(dataset, index_name: str):
    inner = getattr(dataset, "_ds", dataset)
    return _indices.get_pq_codebook(inner, index_name)


def get_partial_pq_codebooks(dataset, index_name: str):
    inner = getattr(dataset, "_ds", dataset)
    return _indices.get_partial_pq_codebooks(inner, index_name)


__all__ += ["get_ivf_model", "get_pq_codebook", "get_partial_pq_codebooks"]


class IndexFileVersion(str, Enum):
    LEGACY = "Legacy"
    V3 = "V3"


class SupportedDistributedIndices(str, Enum):
    # Scalar index types
    BTREE = "BTREE"
    INVERTED = "INVERTED"
    # Precise vector index types supported by distributed merge
    IVF_FLAT = "IVF_FLAT"
    IVF_PQ = "IVF_PQ"
    IVF_SQ = "IVF_SQ"
    IVF_HNSW_FLAT = "IVF_HNSW_FLAT"
    IVF_HNSW_PQ = "IVF_HNSW_PQ"
    IVF_HNSW_SQ = "IVF_HNSW_SQ"
    # Deprecated generic placeholder (kept for backward compatibility)
    VECTOR = "VECTOR"
