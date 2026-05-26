"""Shared runtime utilities for the pipelines subpackage.

This module is intentionally minimal — it holds the small set of helpers
that every pipeline will need (per-stage telemetry, parquet
checkpointing for the "low" memory mode, an explicit-release helper for
the inter-stage cleanup the canonical pipeline relies on).

Functional helpers ONLY here; pipeline orchestration goes in the
per-pipeline module (e.g. :mod:`guidedLP.pipelines.canonical`).
"""

from __future__ import annotations

import gc
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import polars as pl

from guidedLP.common.edgelist import EdgeList


MemoryMode = Literal["fast", "balanced", "low"]


@dataclass
class StageStats:
    """Per-stage telemetry record returned by every pipeline."""

    name: str
    duration_s: float
    input_edges: int = 0
    output_edges: int = 0
    output_nodes: int = 0

    def __str__(self) -> str:
        return (
            f"{self.name:<42s} {self.duration_s:7.2f}s | "
            f"{self.input_edges:>13,} → {self.output_edges:>13,} edges, "
            f"{self.output_nodes:>10,} nodes"
        )


def write_edgelist_parquet(el: EdgeList, path: Path) -> dict[str, Any]:
    """Write an EdgeList's frame to parquet and return its metadata.

    The metadata dict is what :func:`read_edgelist_parquet` needs to
    reconstruct the EdgeList — kept tiny (a handful of fields) so the
    caller can hold it in memory between stages without checkpointing
    a separate sidecar.
    """
    el.df.write_parquet(path, compression="lz4")
    return {
        "directed": el.directed,
        "bipartite": el.bipartite,
        "n_nodes": el.n_nodes,
        "code_dtype": el.code_dtype,
    }


def read_edgelist_parquet(path: Path, metadata: dict[str, Any]) -> EdgeList:
    """Reconstruct an EdgeList from a parquet file + its metadata dict."""
    df = pl.read_parquet(path)
    return EdgeList(
        df=df,
        directed=metadata["directed"],
        bipartite=metadata["bipartite"],
        n_nodes=metadata["n_nodes"],
        code_dtype=metadata["code_dtype"],
    )


def maybe_free(memory_mode: MemoryMode) -> None:
    """Trigger gc.collect() iff the caller will reuse the freed memory.

    Use AFTER ``del``-ing the previous stage's intermediates. The
    rationale is the same as in ``benchmarks/projection_chart.py``'s
    earlier discussion: forcing a sweep returns mmap'd pages to the OS,
    which costs page faults on the next allocation — but in pipeline
    mode the next allocation reuses those pages immediately, so the
    page-fault cost is amortized by the freed memory not having to
    coexist with the new stage's working set.

    No-op for ``memory_mode="fast"``.
    """
    if memory_mode == "fast":
        return
    gc.collect()
