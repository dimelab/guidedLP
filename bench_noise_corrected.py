"""Benchmark for apply_backbone(method="noise_corrected") on a large EdgeList.

Run on the current branch, then `git stash`, run again, `git stash pop` —
the two outputs are directly comparable since the inputs are seeded.
"""
from __future__ import annotations

import gc
import os
import resource
import sys
import time

import numpy as np
import polars as pl

from guidedLP.network.backboning import apply_backbone
from guidedLP.network.construction import build_edgelist_from_frame


# macOS reports ru_maxrss in bytes; Linux reports in kilobytes.
_RSS_TO_MB = 1.0 / (1024 * 1024) if sys.platform == "darwin" else 1.0 / 1024


def rss_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * _RSS_TO_MB


def reset_peak_rss() -> None:
    # macOS / Linux don't expose a clean "reset peak" — best we can do is GC
    # so the next allocation cliff is observable above the new baseline.
    gc.collect()


def synth_edges(n_nodes: int, n_edges: int, seed: int = 0) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    # Power-law-ish degree by sampling with replacement weighted by node index^(-0.5).
    pweights = (np.arange(1, n_nodes + 1, dtype=np.float64) ** (-0.5))
    pweights /= pweights.sum()
    src = rng.choice(n_nodes, size=n_edges, p=pweights)
    tgt = rng.choice(n_nodes, size=n_edges, p=pweights)
    # Drop self-loops.
    keep = src != tgt
    src, tgt = src[keep], tgt[keep]
    # Log-normal weights, realistic for many weighted graphs.
    w = rng.lognormal(mean=0.0, sigma=1.0, size=src.size)
    return pl.DataFrame({
        "source": [f"n{i}" for i in src],
        "target": [f"n{i}" for i in tgt],
        "weight": w,
    })


def bench(n_nodes: int, n_edges: int, *, streaming: bool, label: str) -> None:
    print(f"=== {label} (streaming={streaming}) ===", flush=True)

    print(f"  generating ~{n_edges:,} random edges across {n_nodes:,} nodes …", flush=True)
    edges_frame = synth_edges(n_nodes, n_edges, seed=42)
    print(f"  raw edges:        {edges_frame.height:,}", flush=True)

    edge_list, mapper = build_edgelist_from_frame(
        edges_frame,
        source_col="source", target_col="target", weight_col="weight",
        directed=False, remove_duplicates=False, verbose=False,
    )
    n_in = edge_list.number_of_edges()
    print(f"  edgelist edges:   {n_in:,}", flush=True)
    print(f"  baseline RSS:     {rss_mb():.1f} MB", flush=True)

    # Drop the raw frame; the EdgeList holds what we need.
    del edges_frame
    reset_peak_rss()
    rss_before = rss_mb()

    # Old code lacks `streaming`; pass it only when supported.
    kwargs = dict(method="noise_corrected", threshold=1.0, verbose=False)
    if "streaming" in apply_backbone.__code__.co_varnames:
        kwargs["streaming"] = streaming
    elif streaming:
        print("  (streaming requested but apply_backbone has no such kwarg; running eager)", flush=True)

    t0 = time.perf_counter()
    result, _ = apply_backbone(edge_list, mapper, **kwargs)
    dt = time.perf_counter() - t0

    rss_peak = rss_mb()
    n_out = result.number_of_edges()
    print(f"  result edges:     {n_out:,}  ({100 * n_out / n_in:.1f}% kept)", flush=True)
    print(f"  wall time:        {dt:.2f}s", flush=True)
    print(f"  peak RSS:         {rss_peak:.1f} MB (Δ {rss_peak - rss_before:+.1f} MB above pre-call baseline)", flush=True)
    print(flush=True)


if __name__ == "__main__":
    n_nodes = int(os.environ.get("N_NODES", "500000"))
    n_edges = int(os.environ.get("N_EDGES", "3000000"))
    mode = os.environ.get("MODE", "eager")  # "eager", "streaming", or "both"

    if mode in ("eager", "both"):
        bench(n_nodes, n_edges, streaming=False, label="noise_corrected")
    if mode in ("streaming", "both"):
        bench(n_nodes, n_edges, streaming=True, label="noise_corrected")
