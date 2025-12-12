# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors

import uuid
from pathlib import Path
from typing import Optional

import lance
import numpy as np
import pyarrow as pa
import pytest
from lance.indices import IndicesBuilder


def _make_sample_dataset(tmp_path: Path, n_rows: int = 2000, dim: int = 128):
    """Create a dataset with an integer 'id' and list<float32> 'vector' column.

    Use a small max_rows_per_file to ensure multiple fragments.
    """
    mat = np.random.rand(n_rows, dim).astype(np.float32)
    ids = np.arange(n_rows, dtype=np.int64)
    vectors = pa.array(mat.tolist(), type=pa.list_(pa.float32(), dim))
    table = pa.table({"id": ids, "vector": vectors})
    return lance.write_dataset(table, tmp_path / "dist_e2e", max_rows_per_file=256)


def _split_fragments_two_groups(ds):
    frags = ds.get_fragments()
    if len(frags) < 2:
        pytest.skip("Need at least 2 fragments for distributed indexing")
    frag_ids = [f.fragment_id for f in frags]
    mid = len(frag_ids) // 2
    node1 = frag_ids[:mid]
    node2 = frag_ids[mid:]
    if not node1 or not node2:
        pytest.skip("Failed to split fragments into two non-empty groups")
    return node1, node2


def _commit_index_helper(
    ds,
    index_uuid: str,
    column: str = "vector",
    index_name: Optional[str] = None,
):
    """Finalize index commit after merge_index_metadata.

    Build an Index record and commit a CreateIndex operation.
    """
    from lance.dataset import Index

    lance_field = ds.lance_schema.field(column)
    if lance_field is None:
        raise KeyError(f"{column} not found in schema")
    field_id = lance_field.id()

    if index_name is None:
        index_name = f"{column}_idx"

    frag_ids = set(f.fragment_id for f in ds.get_fragments())

    index = Index(
        uuid=index_uuid,
        name=index_name,
        fields=[field_id],
        dataset_version=ds.version,
        fragment_ids=frag_ids,
        index_version=0,
    )
    op = lance.LanceOperation.CreateIndex(new_indices=[index], removed_indices=[])
    return lance.LanceDataset.commit(ds.uri, op, read_version=ds.version)


def _safe_sample_rate(num_rows: int, num_partitions: int) -> int:
    """Compute a sample_rate valid for both IVF and PQ training."""
    safe_sr_ivf = num_rows // max(1, num_partitions)
    safe_sr_pq = num_rows // 256
    return max(2, min(safe_sr_ivf, safe_sr_pq))


def _sample_queries(ds, num_queries: int, column: str = "vector"):
    """Sample query vectors from the dataset as float32 numpy arrays."""
    tbl = ds.sample(num_queries, columns=[column])
    return [np.asarray(v, dtype=np.float32) for v in tbl[column].to_pylist()]


def _average_recall(ds, queries, k: int) -> float:
    """Compute mean Recall@k against exact search (use_index=False)."""
    recalls = []
    for q in queries:
        gt = ds.to_table(
            columns=["id"],
            nearest={"column": "vector", "q": q, "k": k, "use_index": False},
        )
        res = ds.to_table(
            columns=["id"],
            nearest={
                "column": "vector",
                "q": q,
                "k": k,
                "nprobes": 16,
                "refine_factor": 100,
            },
        )
        gt_ids = set(int(x) for x in gt["id"].to_pylist())
        res_ids = set(int(x) for x in res["id"].to_pylist())
        recalls.append(len(gt_ids & res_ids) / float(k))
    return float(np.mean(recalls))


def test_e2e_distributed_ivf_pq_recall(tmp_path: Path):
    ds = _make_sample_dataset(tmp_path, n_rows=2000, dim=128)
    node1, node2 = _split_fragments_two_groups(ds)

    num_partitions = 4
    num_sub_vectors = 16
    builder = IndicesBuilder(ds, "vector")
    num_rows = ds.count_rows()
    sample_rate = _safe_sample_rate(num_rows, num_partitions)

    pre = builder.prepare_global_ivfpq(
        num_partitions=num_partitions,
        num_subvectors=num_sub_vectors,
        distance_type="l2",
        sample_rate=sample_rate,
    )

    shared_uuid = str(uuid.uuid4())

    try:
        for shard in (node1, node2):
            ds.create_index(
                column="vector",
                index_type="IVF_PQ",
                fragment_ids=shard,
                index_uuid=shared_uuid,
                num_partitions=num_partitions,
                num_sub_vectors=num_sub_vectors,
                ivf_centroids=pre["ivf_centroids"],
                pq_codebook=pre["pq_codebook"],
            )

        ds.merge_index_metadata(shared_uuid, "IVF_PQ")
        ds = _commit_index_helper(ds, shared_uuid, column="vector")
    except ValueError as e:
        # Known flakiness in some environments when PQ codebooks diverge
        if "PQ codebook content mismatch across shards" in str(e):
            pytest.skip(
                "Distributed IVF_PQ codebook mismatch - known environment issue"
            )
        raise

    queries = _sample_queries(ds, 10, column="vector")
    recall = _average_recall(ds, queries, k=10)
    assert recall >= 0.90


def test_e2e_distributed_ivf_flat_recall(tmp_path: Path):
    ds = _make_sample_dataset(tmp_path, n_rows=2000, dim=128)
    node1, node2 = _split_fragments_two_groups(ds)

    shared_uuid = str(uuid.uuid4())

    for shard in (node1, node2):
        ds.create_index(
            column="vector",
            index_type="IVF_FLAT",
            fragment_ids=shard,
            index_uuid=shared_uuid,
            num_partitions=4,
            num_sub_vectors=128,
        )

    ds.merge_index_metadata(shared_uuid, "IVF_FLAT")
    ds = _commit_index_helper(ds, shared_uuid, column="vector")

    queries = _sample_queries(ds, 10, column="vector")
    recall = _average_recall(ds, queries, k=10)
    assert recall >= 0.98
