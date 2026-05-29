"""Undirected bipartite pipeline: raw → bipartite → backbone → projection → backbone.

Composes the four stages most co-occurrence-style analyses use:

1. :func:`build_edgelist_from_frame` — turn the raw input into a bipartite
   coded EdgeList.
2. :func:`apply_backbone(method="bipartite_svn")` — filter the bipartite
   edges to the statistically significant subset.
3. :func:`project_bipartite` — undirected shared-neighbor projection onto
   one partition. Edge weights come from the topology
   (``count`` / ``jaccard`` / ``overlap``); default is ``jaccard``.

Optional Stage 3.5 (enabled by passing ``content_seeds`` as a DataFrame):
attach caller-supplied edges (e.g. synthetic label-anchor / stat-user
edges) onto the projection via :meth:`guidedLP.common.EdgeList.attach`
before the projection backbone runs. Because the projection is
undirected, the caller emits each edge **once** — no mirroring is
required, unlike the directed canonical pipeline.

4. :func:`apply_backbone(method="noise_corrected")` — backbone the
   projection. ``apply_backbone`` reads ``edge_list.directed`` from the
   projection so the undirected ``2 · Σw`` normalization is selected
   automatically.

Three memory modes control inter-stage release AND within-call streaming:

- ``"fast"`` — no inter-stage cleanup; ``build_edgelist_from_frame``
  and both ``apply_backbone`` calls use the in-memory engine. Same
  memory profile as making the calls by hand, max throughput.
- ``"balanced"`` (default) — explicitly ``del`` previous stages and
  ``gc.collect()`` between steps; AND passes ``streaming=True`` to
  ``build_edgelist_from_frame`` and the two ``apply_backbone`` calls.
  ~30% slower than ``"fast"`` with substantially lower peak.
- ``"low"`` — additionally checkpoint each stage's EdgeList to parquet
  on disk and release the in-memory frame. Peak memory becomes the max
  *single* stage's working set rather than the sum across overlapping
  stages. Inherits ``streaming=True`` from ``"balanced"``.

The directed sibling of this pipeline is
:func:`guidedLP.pipelines.run_canonical_pipeline`, which uses
:func:`temporal_bipartite_to_unipartite` for citation-direction edges.
Use the canonical pipeline for attribution analyses (later-sharer →
earlier-sharer edges, suitable for PageRank / HITS); use this pipeline
for symmetric co-occurrence / similarity analyses where direction is
not meaningful.
"""

from __future__ import annotations

import shutil
import tempfile
import time as _time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import polars as pl

from guidedLP.common.edgelist import EdgeList
from guidedLP.common.id_mapper import IDMapper
from guidedLP.common.exceptions import ValidationError
from guidedLP.network.backboning import apply_backbone
from guidedLP.network.construction import (
    build_edgelist_from_frame,
    project_bipartite,
)
from guidedLP.pipelines._runtime import (
    MemoryMode,
    StageStats,
    maybe_free,
    read_edgelist_parquet,
    write_edgelist_parquet,
)


@dataclass
class UndirectedBipartitePipelineResult:
    """Return value of :func:`run_undirected_bipartite_pipeline`.

    Attributes
    ----------
    edgelist : EdgeList
        Final backboned projection. Always undirected and weighted
        (``edgelist.directed is False``, ``edgelist.bipartite is False``).
        If ``content_seeds`` was provided, the projection backbone ran
        on the projection union with the caller's extra edges, so any
        of those edges that survived the backbone are also present here.
    id_mapper : IDMapper
        Mapper for ``edgelist``'s codes. Covers exactly the nodes
        surviving the projection backbone — including any new nodes
        introduced by ``content_seeds``.
    stage_stats : list[StageStats]
        Per-stage telemetry in execution order.
    intermediates : dict[str, Any], optional
        Only populated when ``keep_intermediates=True``. Keys:
        ``"bipartite"``, ``"bipartite_filtered"``, ``"projection"``,
        each mapped to an ``(EdgeList, IDMapper)`` tuple. When
        ``content_seeds`` was attached, ``"projection_attached"`` (the
        union of projection + caller's edges, pre-backbone) is also
        present.
    """

    edgelist: EdgeList
    id_mapper: IDMapper
    stage_stats: List[StageStats]
    intermediates: Optional[Dict[str, Any]] = None

    @property
    def total_duration_s(self) -> float:
        return sum(s.duration_s for s in self.stage_stats)


def run_undirected_bipartite_pipeline(
    source: Union[str, Path, pl.DataFrame],
    *,
    source_col: str,
    target_col: str,
    # Projection orientation.
    projection_mode: str = "source",
    projection_weight_method: str = "jaccard",
    # Stage 1: build_edgelist.
    min_source_degree: Optional[int] = None,
    min_target_degree: Optional[int] = None,
    weight_col: Optional[str] = None,
    auto_weight: bool = False,
    bipartite_overlap: str = "drop",
    # Stage 2: bipartite backbone.
    bipartite_alpha: float = 0.01,
    bipartite_correction: str = "fdr_bh",
    bipartite_target_fraction: Optional[float] = None,
    bipartite_min_node_retention: Optional[float] = None,
    # Stage 3.5 (optional): attach caller-supplied edges.
    content_seeds: Optional[pl.DataFrame] = None,
    # Stage 4: projection backbone.
    projection_threshold: float = 1.0,
    projection_target_fraction: Optional[float] = None,
    # Cross-stage protection.
    protected_nodes: Optional[List[Any]] = None,
    # Memory & I/O.
    memory_mode: MemoryMode = "balanced",
    checkpoint_dir: Optional[Union[str, Path]] = None,
    keep_intermediates: bool = False,
    verbose: bool = True,
) -> UndirectedBipartitePipelineResult:
    """Run the undirected raw → backboned-projection pipeline in one call.

    Parameters
    ----------
    source : str | Path | pl.DataFrame
        Raw input. File paths (``.csv``, ``.parquet``) are read via
        :func:`build_edgelist_from_frame`; a DataFrame is consumed
        directly. Unlike the canonical pipeline, no timestamp column is
        required — this pipeline does not carry timestamp passthrough
        through Stage 1. A per-edge ``weight_col`` is optional (see
        below); when supplied, it feeds the bipartite_svn backbone at
        Stage 2 but is overwritten by topological weights in Stage 3.
    source_col, target_col : str
        Column names for the bipartite endpoints. The conventional
        choice is ``source_col`` = user-side and ``target_col`` =
        content-side, then ``projection_mode="source"`` to collapse
        content into user-user co-occurrence edges.
    projection_mode : {"source", "target"}, default "source"
        Which partition to project onto. Forwarded to
        :func:`project_bipartite`. ``"source"`` collapses the target
        partition (e.g. content) and produces a graph on the source
        partition (e.g. users).
    projection_weight_method : {"count", "jaccard", "overlap"}, default "jaccard"
        Shared-neighbor weight formula. ``"jaccard"`` is the default
        because it's bounded in ``[0, 1]`` (well-behaved for downstream
        GLP and noise-corrected backboning); ``"count"`` is the raw
        number of shared neighbors; ``"overlap"`` is asymmetric
        (``|A ∩ B| / min(|A|, |B|)``). Forwarded to
        :func:`project_bipartite`.
    min_source_degree, min_target_degree : int, optional
        Degree filters applied during the build step.
    weight_col : str, optional
        Name of a per-row weight column on the input. When set, Stage 1
        forwards it to :func:`build_edgelist_from_frame` with
        ``auto_weight=False`` and ``remove_duplicates=True``: duplicate
        ``(source, target)`` rows are collapsed keeping the
        **first occurrence's weight** (no summation — pre-aggregate
        upstream if you want sum semantics). The resulting weights feed
        Stage 2's weighted bipartite_svn. Stage 3
        (:func:`project_bipartite`) ignores edge weights and computes
        its own topological weight, so this only affects Stage 2.
        Mutually exclusive with ``auto_weight=True``.
    auto_weight : bool, default False
        If True, count duplicate ``(source, target)`` rows to set the
        bipartite edge weight. The resulting weights feed Stage 2's
        weighted bipartite_svn. Mutually exclusive with ``weight_col``.
    bipartite_overlap : str, default "drop"
        Overlap-resolution policy when nodes appear on both sides of
        the bipartite. ``"drop"`` removes the offending nodes; see
        :func:`build_edgelist_from_frame` for other options.
    bipartite_alpha, bipartite_correction : float, str
        ``apply_backbone(method="bipartite_svn")`` parameters.
    bipartite_target_fraction : float, optional
        If set, overrides ``bipartite_alpha`` and keeps the top fraction
        of bipartite edges by p-value.
    bipartite_min_node_retention : float, optional
        Post-filter for Stage 2: drop any bipartite node whose surviving
        edges fall below this fraction of its original incident-edge
        count. Protected nodes (see ``protected_nodes``) are exempt.
    content_seeds : pl.DataFrame, optional
        Extra edges to attach onto the projection *before* the projection
        backbone runs (Stage 3.5). Must have columns ``source_id``,
        ``target_id``, ``weight`` (Float64). The caller is fully
        responsible for the contents — IDs may reference existing nodes
        in the projection or new ones (new IDs get added to the mapper).
        Because the projection is **undirected**, each anchor edge
        should be emitted **once** — :meth:`EdgeList.attach` does not
        mirror, and the backboning / GLP downstream treats the EdgeList
        as undirected so both orientations are observationally
        equivalent. (This differs from the canonical directed pipeline,
        where mirroring is required.) Attached edges are subject to the
        noise-corrected backbone — use ``protected_nodes`` (or pick
        weights / a permissive ``projection_target_fraction``) if you
        need them to survive.
    projection_threshold, projection_target_fraction : float, optional
        ``apply_backbone(method="noise_corrected")`` parameters.
        ``projection_target_fraction`` is the recommended way to size
        the final backbone.
    protected_nodes : list, optional
        Original IDs to exempt from filtering in *both* backbone stages.
        Edges incident to a protected node are forced kept by Stage 2
        (bipartite_svn) and Stage 4 (noise_corrected). The same list is
        forwarded to both calls — IDs that aren't present in a given
        stage's mapper produce a warning and are skipped.
    memory_mode : {"fast", "balanced", "low"}, default "balanced"
        See module docstring. ``"balanced"`` and ``"low"`` pass
        ``streaming=True`` to both ``apply_backbone`` calls for ~2×
        lower within-call peak at ~30% wall-clock cost.
    checkpoint_dir : str | Path, optional
        Where to write parquet checkpoints in ``memory_mode="low"``.
        If unset and ``memory_mode="low"``, a temporary directory is
        created and cleaned up on return.
    keep_intermediates : bool, default False
        Retain references to each stage's ``(EdgeList, IDMapper)`` on
        the returned result. Disables all inter-stage release, so this
        is incompatible with ``memory_mode="low"``.
    verbose : bool, default True
        Per-stage one-line summaries via the underlying functions'
        own verbose output, plus a final TOTAL line.

    Returns
    -------
    UndirectedBipartitePipelineResult
        See dataclass docstring.

    Raises
    ------
    ValidationError
        On invalid argument combinations (e.g. ``memory_mode="low"``
        with ``keep_intermediates=True``).
    """
    if memory_mode == "low" and keep_intermediates:
        raise ValidationError(
            "memory_mode='low' is incompatible with keep_intermediates=True; "
            "the checkpoint cycle frees the in-memory frame between stages."
        )
    if memory_mode not in ("fast", "balanced", "low"):
        raise ValidationError(
            f"memory_mode must be 'fast', 'balanced', or 'low'; got {memory_mode!r}"
        )
    if weight_col is not None and auto_weight:
        raise ValidationError(
            "weight_col and auto_weight=True are mutually exclusive; "
            "use weight_col to read weights from an input column, or "
            "auto_weight to count duplicate (source, target) rows."
        )

    created_tempdir = False
    if memory_mode == "low":
        if checkpoint_dir is None:
            checkpoint_dir = Path(tempfile.mkdtemp(prefix="glp_pipeline_"))
            created_tempdir = True
        else:
            checkpoint_dir = Path(checkpoint_dir)
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
    elif checkpoint_dir is not None:
        checkpoint_dir = Path(checkpoint_dir)

    stats: List[StageStats] = []
    intermediates: Optional[Dict[str, Any]] = {} if keep_intermediates else None
    pipeline_start = _time.perf_counter()
    # Enable per-call streaming in the two apply_backbone steps when the
    # caller has opted into any memory-conscious mode. "fast" stays
    # in-memory for maximum throughput.
    stream_backbones = memory_mode != "fast"

    try:
        # Stage 1: build the bipartite EdgeList. No timestamp
        # passthrough; project_bipartite (Stage 3) overwrites edge
        # weights with its own topological score, so weights here only
        # affect the bipartite_svn backbone at Stage 2. When weight_col
        # is set, we pair it with remove_duplicates=True (keep-first on
        # (src, tgt)); the alternative weight-sum branch is incompatible
        # with the auto_weight kwarg's contract and is not exposed here.
        t0 = _time.perf_counter()
        el_bp, mapper_bp = build_edgelist_from_frame(
            source,
            source_col=source_col,
            target_col=target_col,
            weight_col=weight_col,
            bipartite=True,
            bipartite_overlap=bipartite_overlap,
            min_source_degree=min_source_degree,
            min_target_degree=min_target_degree,
            auto_weight=auto_weight,
            remove_duplicates=weight_col is not None,
            streaming=stream_backbones,
            verbose=verbose,
        )
        stats.append(StageStats(
            name="build_edgelist_from_frame",
            duration_s=_time.perf_counter() - t0,
            output_edges=el_bp.number_of_edges(),
            output_nodes=el_bp.n_nodes,
        ))
        if intermediates is not None:
            intermediates["bipartite"] = (el_bp, mapper_bp)

        # Stage 2: bipartite-side backbone.
        t0 = _time.perf_counter()
        n_in_2 = el_bp.number_of_edges()
        el_bp_kept, mapper_bp_kept = apply_backbone(
            el_bp,
            id_mapper=mapper_bp,
            method="bipartite_svn",
            alpha=bipartite_alpha,
            correction=bipartite_correction,
            target_fraction=bipartite_target_fraction,
            min_node_retention=bipartite_min_node_retention,
            streaming=stream_backbones,
            verbose=verbose,
            protected_nodes=protected_nodes,
        )
        stats.append(StageStats(
            name="apply_backbone(bipartite_svn)",
            duration_s=_time.perf_counter() - t0,
            input_edges=n_in_2,
            output_edges=el_bp_kept.number_of_edges(),
            output_nodes=el_bp_kept.n_nodes,
        ))
        if intermediates is not None:
            intermediates["bipartite_filtered"] = (el_bp_kept, mapper_bp_kept)
        else:
            del el_bp, mapper_bp
            maybe_free(memory_mode)

        # Optional disk checkpoint between stages 2 and 3.
        bp_ckpt_path: Optional[Path] = None
        bp_meta: Optional[dict] = None
        if memory_mode == "low" and intermediates is None:
            bp_ckpt_path = checkpoint_dir / "01_bipartite_kept.parquet"
            bp_meta = write_edgelist_parquet(el_bp_kept, bp_ckpt_path)
            del el_bp_kept
            maybe_free(memory_mode)
            el_bp_kept = read_edgelist_parquet(bp_ckpt_path, bp_meta)

        # Stage 3: undirected shared-neighbor projection.
        t0 = _time.perf_counter()
        n_in_3 = el_bp_kept.number_of_edges()
        el_proj, mapper_proj = project_bipartite(
            el_bp_kept,
            id_mapper=mapper_bp_kept,
            projection_mode=projection_mode,
            weight_method=projection_weight_method,
            output_format="edgelist",
            verbose=verbose,
        )
        stats.append(StageStats(
            name="project_bipartite",
            duration_s=_time.perf_counter() - t0,
            input_edges=n_in_3,
            output_edges=el_proj.number_of_edges(),
            output_nodes=el_proj.n_nodes,
        ))
        if intermediates is not None:
            intermediates["projection"] = (el_proj, mapper_proj)
        else:
            del el_bp_kept, mapper_bp_kept
            maybe_free(memory_mode)
            if bp_ckpt_path is not None:
                bp_ckpt_path.unlink(missing_ok=True)

        # Stage 3.5 (optional): attach caller-supplied edges to the
        # projection BEFORE the noise_corrected backbone runs. The
        # end-user owns the contents of ``content_seeds`` (schema:
        # source_id / target_id / weight). No filtering, deduplication,
        # or directional mirroring is applied here — and because the
        # projection is undirected, no mirroring is needed.
        if content_seeds is not None:
            t0 = _time.perf_counter()
            n_in_attach = el_proj.number_of_edges()
            el_proj, mapper_proj = el_proj.attach(content_seeds, mapper_proj)
            stats.append(StageStats(
                name="attach_content_seeds",
                duration_s=_time.perf_counter() - t0,
                input_edges=n_in_attach,
                output_edges=el_proj.number_of_edges(),
                output_nodes=el_proj.n_nodes,
            ))
            if intermediates is not None:
                intermediates["projection_attached"] = (el_proj, mapper_proj)

        # Stage 4: projection-side backbone. apply_backbone reads
        # el_proj.directed (False here) and selects the undirected
        # 2·Σw normalization automatically.
        t0 = _time.perf_counter()
        n_in_4 = el_proj.number_of_edges()
        el_final, mapper_final = apply_backbone(
            el_proj,
            id_mapper=mapper_proj,
            method="noise_corrected",
            threshold=projection_threshold,
            target_fraction=projection_target_fraction,
            streaming=stream_backbones,
            verbose=verbose,
            protected_nodes=protected_nodes,
        )
        stats.append(StageStats(
            name="apply_backbone(noise_corrected)",
            duration_s=_time.perf_counter() - t0,
            input_edges=n_in_4,
            output_edges=el_final.number_of_edges(),
            output_nodes=el_final.n_nodes,
        ))
        if intermediates is None:
            del el_proj, mapper_proj
            maybe_free(memory_mode)

        if verbose:
            total = _time.perf_counter() - pipeline_start
            print(
                f"[run_undirected_bipartite_pipeline] TOTAL {total:.2f}s | "
                f"mode={memory_mode} | "
                f"final: {el_final.number_of_edges():,} edges, "
                f"{el_final.n_nodes:,} nodes"
            )

        return UndirectedBipartitePipelineResult(
            edgelist=el_final,
            id_mapper=mapper_final,
            stage_stats=stats,
            intermediates=intermediates,
        )

    finally:
        # Clean up the auto-created tempdir even on failure paths.
        if created_tempdir and checkpoint_dir is not None and checkpoint_dir.exists():
            shutil.rmtree(checkpoint_dir, ignore_errors=True)
