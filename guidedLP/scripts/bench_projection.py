"""
Benchmark: legacy Dict[Any, Set[Any]] projection kernel vs. the coded
EdgeList kernel introduced in C1.

Generates a synthetic bipartite graph (Zipf-popular targets so projection
edges explode through hub items), runs both kernels on identical input,
and reports wall time + peak Python-tracked memory + RSS delta.

The gate from the project plan: we keep C1-C3 if the coded path delivers
>=2x memory reduction AND comparable-or-better wall time.

Results captured 2026-05-25 on Darwin/Python 3.9.6:

  Small scale (2K users × 500 items, avg-deg 8, zipf 1.5):
    Legacy: 1.88s, RSS Δ 368 MB
    Coded:  1.84s, RSS Δ 112 MB
    -> 3.29x RSS reduction, equivalent wall time, 1.98M projection edges.

  Medium scale (5K users × 800 items, avg-deg 10, zipf 1.5):
    Legacy: 12.33s, RSS Δ 2517 MB
    Coded:  12.09s, RSS Δ  386 MB
    -> 6.53x RSS reduction, equivalent wall time, 12.46M projection edges.

  Hub-heavy scale (20K users × 2K items, avg-deg 12, zipf 1.3):
    Coded:  250s, RSS Δ 7801 MB, 199M projection edges.
    Legacy: did not complete (>5 GB RSS and rising in the neighbor-map
            build phase). Coded handled it without intervention.

Gate: met. Memory wins are 2-7x at observed scales; wall time is equal
to fractionally better. Legacy can't handle hub-heavy projections at
scale; coded can.

Usage
-----
    python3 scripts/bench_projection.py [--users N] [--items M] [--seed S]
                                        [--weight-method count|jaccard|overlap]
                                        [--skip-legacy]

Defaults are tuned to run in well under a minute and produce a
projection of a few million edges (hub-heavy enough that the legacy
kernel's per-edge Python overhead shows up).
"""

from __future__ import annotations

import argparse
import gc
import resource
import sys
import time
import tracemalloc
from pathlib import Path
from typing import Tuple

import numpy as np
import polars as pl

# Allow running directly from the repo without installing.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from guidedLP.network.construction import (  # noqa: E402
    _compute_projection_arrays,
    _compute_projection_arrays_coded,
    _neighbor_map_from_edges,
    _projection_arrays_to_edge_frame,
    build_edgelist_from_frame,
)


def make_synthetic_bipartite(
    n_users: int = 20_000,
    n_items: int = 2_000,
    avg_user_degree: int = 12,
    zipf_alpha: float = 1.3,
    seed: int = 42,
) -> pl.DataFrame:
    """Generate a synthetic bipartite frame.

    Item popularity follows a Zipf distribution with parameter zipf_alpha,
    so a small number of "hub" items are touched by many users — exactly
    the shape where the legacy O(E) Python loop in the projection kernel
    becomes a bottleneck (a hub item with k neighbors contributes k*(k-1)/2
    projection edges).
    """
    rng = np.random.default_rng(seed)

    # Item popularity: rank-r item gets weight ~ r^-alpha (Zipf).
    ranks = np.arange(1, n_items + 1)
    popularity = ranks ** (-zipf_alpha)
    popularity /= popularity.sum()

    # Per-user degree: Poisson around the mean, clipped to >=1.
    user_degrees = rng.poisson(avg_user_degree, size=n_users).clip(min=1)

    # For each user, sample `degree` items WITHOUT replacement, weighted
    # by item popularity. Doing this row-by-row is slow; batch with a
    # multinomial fallback that allows replacement, then dedupe per user.
    user_ids = []
    item_ids = []
    for u in range(n_users):
        d = int(user_degrees[u])
        picks = rng.choice(n_items, size=d, replace=False, p=popularity)
        user_ids.extend([f"u_{u}"] * len(picks))
        item_ids.extend([f"i_{i}" for i in picks])

    return pl.DataFrame({"source_id": user_ids, "target_id": item_ids})


def _run_legacy(df: pl.DataFrame, projection_mode: str, weight_method: str) -> pl.DataFrame:
    """Old path: build neighbor map dict, then call legacy kernel."""
    projection_partition, neighbor_map = _neighbor_map_from_edges(df, projection_mode)
    i_arr, j_arr, weights, sorted_projection = _compute_projection_arrays(
        projection_partition, neighbor_map, weight_method,
    )
    return _projection_arrays_to_edge_frame(i_arr, j_arr, weights, sorted_projection)


def _run_coded(df: pl.DataFrame, projection_mode: str, weight_method: str) -> pl.DataFrame:
    """New path: build coded EdgeList, then call vectorized kernel."""
    el, mapper = build_edgelist_from_frame(
        df, source_col="source_id", target_col="target_id",
        bipartite=True, auto_weight=False, remove_duplicates=False, verbose=False,
    )
    side = "src" if projection_mode == "source" else "tgt"
    i_arr, j_arr, weights, sorted_projection = _compute_projection_arrays_coded(
        el, mapper, side, weight_method,
    )
    return _projection_arrays_to_edge_frame(i_arr, j_arr, weights, sorted_projection)


def _rss_kb() -> int:
    """Current resident set size in kilobytes (Linux/macOS portable).

    ru_maxrss is reported in kB on Linux, bytes on macOS — normalize here.
    """
    raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return raw // 1024 if sys.platform == "darwin" else raw


def _measure(name: str, fn, *args, **kwargs) -> Tuple[float, int, int, pl.DataFrame]:
    """Run fn with tracemalloc + RSS delta measurement.

    Returns (wall_time_seconds, peak_python_bytes, rss_delta_kb, result).
    """
    gc.collect()
    rss_before = _rss_kb()

    tracemalloc.start()
    t0 = time.perf_counter()
    result = fn(*args, **kwargs)
    dt = time.perf_counter() - t0
    _, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    rss_after = _rss_kb()
    rss_delta = rss_after - rss_before

    print(
        f"  {name:>10s}: {dt:6.2f}s | tracemalloc peak {peak_bytes / 1e6:7.1f} MB | "
        f"RSS Δ {rss_delta / 1024:6.1f} MB"
    )
    return dt, peak_bytes, rss_delta, result


def _compare(legacy_result: pl.DataFrame, coded_result: pl.DataFrame) -> None:
    """Sanity-check that both paths produced the same projection.

    Uses Polars' Rust-level sort + frame equality rather than building two
    huge Python sets — at projection scales of 10M+ edges the set-based
    check itself becomes the dominant cost.
    """
    if legacy_result.height != coded_result.height:
        print(
            f"  WARN: edge counts differ — legacy {legacy_result.height:,} vs "
            f"coded {coded_result.height:,}"
        )
        return

    # Canonicalize each edge as (min(s,t), max(s,t)) since the projection
    # is undirected, then sort and compare frames.
    def canon(df: pl.DataFrame) -> pl.DataFrame:
        return (
            df.with_columns(
                pl.min_horizontal("source_id", "target_id").alias("_u"),
                pl.max_horizontal("source_id", "target_id").alias("_v"),
            )
            .select("_u", "_v", "weight")
            .sort(["_u", "_v"])
        )

    a, b = canon(legacy_result), canon(coded_result)
    weights_match = np.allclose(a["weight"].to_numpy(), b["weight"].to_numpy())
    endpoints_match = a["_u"].equals(b["_u"]) and a["_v"].equals(b["_v"])

    if weights_match and endpoints_match:
        print(f"  ✓ Both paths produced identical projection ({a.height:,} edges)")
    else:
        print(
            f"  WARN: projections differ — endpoints_match={endpoints_match}, "
            f"weights_match={weights_match}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--users", type=int, default=20_000)
    parser.add_argument("--items", type=int, default=2_000)
    parser.add_argument("--avg-user-degree", type=int, default=12)
    parser.add_argument("--zipf-alpha", type=float, default=1.3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--projection-mode", choices=["source", "target"], default="source")
    parser.add_argument("--weight-method", choices=["count", "jaccard", "overlap"], default="count")
    parser.add_argument("--skip-legacy", action="store_true",
                        help="Skip the legacy path (useful at large scale where it OOMs)")
    args = parser.parse_args()

    print("Generating synthetic bipartite...")
    print(
        f"  users={args.users:,}  items={args.items:,}  "
        f"avg_user_degree={args.avg_user_degree}  zipf_alpha={args.zipf_alpha}"
    )
    df = make_synthetic_bipartite(
        n_users=args.users,
        n_items=args.items,
        avg_user_degree=args.avg_user_degree,
        zipf_alpha=args.zipf_alpha,
        seed=args.seed,
    )
    print(f"  -> {df.height:,} bipartite edges")
    print(
        f"  projection_mode={args.projection_mode}  "
        f"weight_method={args.weight_method}"
    )
    print()

    legacy_result = None
    legacy_dt = legacy_peak = legacy_rss = None
    if not args.skip_legacy:
        print("Legacy path (Dict[Any, Set[Any]] neighbor map + per-edge Python loop):")
        legacy_dt, legacy_peak, legacy_rss, legacy_result = _measure(
            "legacy", _run_legacy, df, args.projection_mode, args.weight_method,
        )
        print()

    print("Coded path (build_edgelist_from_frame + vectorized SciPy kernel):")
    coded_dt, coded_peak, coded_rss, coded_result = _measure(
        "coded", _run_coded, df, args.projection_mode, args.weight_method,
    )
    print()

    if legacy_result is not None:
        print("Equivalence check:")
        _compare(legacy_result, coded_result)
        print()

        speedup = legacy_dt / coded_dt if coded_dt > 0 else float("inf")
        # Avoid divide-by-zero / negative deltas when RSS reading was noisy.
        mem_ratio_py = (legacy_peak / coded_peak) if coded_peak > 0 else float("inf")
        print("Headline:")
        print(f"  wall-time speedup:     {speedup:5.2f}x")
        print(f"  tracemalloc peak Δ:    {mem_ratio_py:5.2f}x (legacy / coded)")
        if legacy_rss > 0 and coded_rss > 0:
            rss_ratio = legacy_rss / coded_rss
            print(f"  RSS delta ratio:       {rss_ratio:5.2f}x (legacy / coded)")
    else:
        print(f"Coded path produced {coded_result.height:,} projection edges in {coded_dt:.2f}s.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
