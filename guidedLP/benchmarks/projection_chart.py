"""Per-function compute & memory projections.

Each function is calibrated against its OWN input — what was actually fed
to that function during measurement — and projected linearly. To compose
a pipeline, look up each step at the size you expect to hand it.

For ``project_bipartite`` and ``temporal_bipartite_to_unipartite`` the
output edge count is roughly a constant multiple of the input edge count
under fixed hub-heaviness — captured in the OUTPUT BLOWUP RATIO below
(matching the user's real-data measurements).

Run::

    python benchmarks/projection_chart.py

Prints the projection tables and writes a chart at
``benchmarks/projection_chart.png``.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter


# Output blowup ratios under the user's hub-heaviness. Listed for
# composing pipeline projections (input to the next function = output of
# the previous). Source: user's real run — 12.6M bipartite → 170.9M
# project_bipartite edges (jaccard) / 191.9M temporal edges.
PROJECTION_OUTPUT_RATIO = {
    "project_bipartite(jaccard)": 170.9e6 / 12.634e6,    # ≈ 13.5
    "temporal_bipartite_to_unipartite": 191.9e6 / 12.634e6,  # ≈ 15.2
}


@dataclass
class FunctionSpec:
    label: str
    input_unit: str               # what the x-axis means for this function
    n_ref: float                  # input size at the calibration measurement
    t_ref_s: float                # measured wall-clock at n_ref
    mem_bytes_per_unit: float     # peak memory per input unit
    color: str
    is_estimated: bool = False


# All calibrations are per-function input — i.e. what you'd hand to THAT
# function, not the size of the bipartite at the pipeline head.
SPECS: list[FunctionSpec] = [
    # build_edgelist_from_frame: 27.9M input rows → 14.44s. Memory dominated
    # by the input Polars frame (Utf8 IDs) + the working set during dedup
    # and ID encoding.
    FunctionSpec(
        label="build_edgelist_from_frame",
        input_unit="input rows",
        n_ref=27.9e6,
        t_ref_s=14.44,
        mem_bytes_per_unit=80,
        color="#1f77b4",
    ),

    # build_graph_from_edgelist: no direct measurement; estimated from the
    # NetworkIt addEdge throughput (~5x slower per edge than the Polars
    # path because the Python→C++ addEdge loop is serialized).
    FunctionSpec(
        label="build_graph_from_edgelist (est.)",
        input_unit="edges in the frame",
        n_ref=23.28e6,
        t_ref_s=23.0,
        mem_bytes_per_unit=40,
        color="#ff7f0e",
        is_estimated=True,
    ),

    # apply_backbone(bipartite_svn): 23.3M edges in 1.72s, Polars-vectorized.
    FunctionSpec(
        label="apply_backbone(bipartite_svn)",
        input_unit="edges in the input graph",
        n_ref=23.28e6,
        t_ref_s=1.72,
        mem_bytes_per_unit=60,
        color="#9467bd",
    ),

    # apply_backbone(noise_corrected, undirected): 170.9M edges in 4.87s.
    # The undirected branch hits the in-memory engine without paging.
    FunctionSpec(
        label="apply_backbone(noise_corrected, undirected)",
        input_unit="edges in the input graph",
        n_ref=170.9e6,
        t_ref_s=4.87,
        mem_bytes_per_unit=100,
        color="#2ca02c",
    ),

    # apply_backbone(noise_corrected, directed): 191.9M edges in 148.67s
    # (target_fraction=0.2 override path; the threshold-only path on this
    # input had previously been measured at ~269s — run-to-run variance and
    # the slightly faster sort+head step). The peak memory is dominated by
    # the chained with_columns intermediates.
    FunctionSpec(
        label="apply_backbone(noise_corrected, directed)",
        input_unit="edges in the input graph",
        n_ref=191.9e6,
        t_ref_s=148.67,
        mem_bytes_per_unit=130,
        color="#d62728",
    ),

    # project_bipartite(jaccard): 12.6M bipartite edges in 14.47s. Time
    # scales with OUTPUT edges (≈13.5× input under your hub-heaviness);
    # under fixed shape that's also linear in input. Memory dominated by
    # the projected CSR (output blowup × ~24 bytes per output edge).
    FunctionSpec(
        label="project_bipartite(jaccard)",
        input_unit="bipartite edges",
        n_ref=12.634e6,
        t_ref_s=14.47,
        mem_bytes_per_unit=24 * (170.9e6 / 12.634e6),  # 24 B/out × 13.5 out/in
        color="#8c564b",
    ),

    # temporal_bipartite_to_unipartite: 12.6M bipartite in 168.64s. Time
    # and memory both proportional to OUTPUT (which is ≈15.2× input on
    # your data, and pre-aggregation rows are another 3–5× the output).
    FunctionSpec(
        label="temporal_bipartite_to_unipartite",
        input_unit="bipartite edges",
        n_ref=12.634e6,
        t_ref_s=168.64,
        mem_bytes_per_unit=40 * (191.9e6 / 12.634e6),  # 40 B/out × 15.2 out/in
        color="#e377c2",
    ),

    # guided_label_propagation: no direct measurement; estimated from sparse
    # matvec throughput (~2 GFLOPS, 5 classes, 100 iterations).
    #   t ≈ 0.4 µs × E × n_classes × n_iter
    # For E=12.6M, that's ~2.5s; with overhead, allow 5s.
    FunctionSpec(
        label="guided_label_propagation (est., 5 cls, 100 iter)",
        input_unit="edges in the input graph",
        n_ref=12.634e6,
        t_ref_s=5.0,
        mem_bytes_per_unit=20,
        color="#17becf",
        is_estimated=True,
    ),
]


def project(spec: FunctionSpec, n_in: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Linear projection of time (s) and peak memory (GB) for an input-size array."""
    t_s = spec.t_ref_s * (n_in / spec.n_ref)
    mem_gb = spec.mem_bytes_per_unit * n_in / (1024 ** 3)
    return t_s, mem_gb


def humanize_seconds(s: float) -> str:
    if s < 1e-3: return f"{s*1e6:.0f} us"
    if s < 1:    return f"{s*1e3:.0f} ms"
    if s < 60:   return f"{s:.1f} s"
    if s < 3600: return f"{s/60:.1f} min"
    if s < 86400:return f"{s/3600:.1f} h"
    return f"{s/86400:.1f} d"


def humanize_gb(gb: float) -> str:
    if gb < 1:    return f"{gb*1024:.0f} MB"
    if gb < 1024: return f"{gb:.0f} GB"
    return f"{gb/1024:.1f} TB"


def render_chart(targets: np.ndarray) -> str:
    """Render a log-log chart showing each function's scaling on its own input.

    Returns the output path.
    """
    n_in = np.logspace(np.log10(targets[0]), np.log10(targets[-1]), 200)
    fig, (ax_t, ax_m) = plt.subplots(1, 2, figsize=(15, 7), constrained_layout=True)

    for spec in SPECS:
        t_s, mem_gb = project(spec, n_in)
        ls = "--" if spec.is_estimated else "-"
        ax_t.plot(n_in, t_s, color=spec.color, linestyle=ls, linewidth=1.7,
                  label=spec.label)
        ax_m.plot(n_in, mem_gb, color=spec.color, linestyle=ls, linewidth=1.7,
                  label=spec.label)

    for ax, refs in [
        (ax_t, [(1, "1 s"), (60, "1 min"), (3600, "1 h"), (86400, "1 day")]),
        (ax_m, [(1, "1 GB"), (16, "16 GB (laptop)"), (192, "192 GB (server)"), (1024, "1 TB")]),
    ]:
        for v, lbl in refs:
            ax.axhline(v, color="0.7", linestyle=":", linewidth=0.7)
            ax.text(targets[0] * 1.05, v * 1.15, lbl, color="0.5", fontsize=8)

    # Pink shading above 192 GB on the memory panel.
    ax_m.axhspan(192, 1e5, color="red", alpha=0.05, zorder=0)

    # Common formatting.
    def _fmt(n, _pos):
        if n < 1e3:  return f"{n:.0f}"
        if n < 1e6:  return f"{n/1e3:.0f}k"
        if n < 1e9:  return f"{n/1e6:.0f}M"
        return f"{n/1e9:.0f}B"

    for ax, ylabel, title, ylim in [
        (ax_t, "Wall-clock time (s)", "Compute time vs input size", (1e-4, 1e7)),
        (ax_m, "Peak memory (GB)", "Peak memory vs input size", (1e-3, 1e5)),
    ]:
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(targets[0], targets[-1])
        ax.set_ylim(ylim)
        ax.set_xlabel("Input to THIS function (rows or edges)")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(loc="upper left", fontsize=8, framealpha=0.92)
        ax.grid(True, which="both", alpha=0.2)
        ax.xaxis.set_major_formatter(FuncFormatter(_fmt))

    fig.suptitle(
        "guidedLP per-function scaling (each function's x-axis is its own input)",
        fontsize=12, y=1.04,
    )

    out = "/Users/jakobbk/Documents/postdoc/codespace/guidedLabelPropagation/guidedLP/benchmarks/projection_chart.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    return out


def main() -> None:
    targets = np.array([1e6, 10e6, 100e6, 1e9, 4e9])
    headers = [f"@{humanize_count(t)}" for t in targets]

    print("\nPROJECTED WALL-CLOCK TIME (per function, input = what THAT function consumes)\n")
    print(f"{'function':<52s}  {'input unit':<26s}  " + "  ".join(f"{h:>9s}" for h in headers))
    print("-" * 130)
    for spec in SPECS:
        t_s, _ = project(spec, targets)
        row = "  ".join(f"{humanize_seconds(t):>9s}" for t in t_s)
        est = " (est.)" if spec.is_estimated else ""
        label = spec.label.replace(" (est., 5 cls, 100 iter)", "")
        label = label.replace(" (est.)", "")
        print(f"{label:<52s}  {spec.input_unit:<26s}  {row}{est}")

    print("\nPROJECTED PEAK MEMORY (per function, * = exceeds 192 GB ceiling)\n")
    print(f"{'function':<52s}  {'bytes/unit':>10s}  " + "  ".join(f"{h:>9s}" for h in headers))
    print("-" * 130)
    for spec in SPECS:
        _, mem_gb = project(spec, targets)
        cells = []
        for m in mem_gb:
            cell = humanize_gb(m)
            if m > 192:
                cell = "*" + cell
            cells.append(f"{cell:>9s}")
        bpu = f"{spec.mem_bytes_per_unit:.0f}"
        label = spec.label.replace(" (est., 5 cls, 100 iter)", "")
        label = label.replace(" (est.)", "")
        print(f"{label:<52s}  {bpu:>10s}  {'  '.join(cells)}")

    print("\nOUTPUT BLOWUP RATIOS (input → output edge count) for chaining functions:")
    for fn, ratio in PROJECTION_OUTPUT_RATIO.items():
        print(f"  {fn:<46s} {ratio:5.1f}× under your real-data hub-heaviness")

    print("\nExample composition: project then noise-correct a 100M-edge bipartite (jaccard):")
    e_proj = 100e6 * PROJECTION_OUTPUT_RATIO["project_bipartite(jaccard)"]
    t_proj, m_proj = project(SPECS[5], np.array([100e6]))   # project_bipartite
    t_nc,   m_nc   = project(SPECS[3], np.array([e_proj]))  # NC undirected
    print(f"  step 1: project_bipartite(100M bipartite) → {humanize_count(e_proj)} edges, "
          f"{humanize_seconds(t_proj[0])}, {humanize_gb(m_proj[0])}")
    print(f"  step 2: apply_backbone(NC, undir.) on {humanize_count(e_proj)} edges  → "
          f"{humanize_seconds(t_nc[0])}, {humanize_gb(m_nc[0])}")
    print()

    out = render_chart(targets)
    print(f"Chart written: {out}")


def humanize_count(n: float) -> str:
    if n < 1e3:  return f"{n:.0f}"
    if n < 1e6:  return f"{n/1e3:.0f}k"
    if n < 1e9:  return f"{n/1e6:.0f}M"
    return f"{n/1e9:.0f}B"


if __name__ == "__main__":
    main()
