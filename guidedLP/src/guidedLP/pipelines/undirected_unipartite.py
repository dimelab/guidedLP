"""Undirected unipartite pipeline: unipartite edge frame → backbone.

Composes the two-or-three stages of the already-unipartite workflow:

1. :func:`build_edgelist_from_frame(bipartite=False)` — wrap a caller-supplied
   ``(source, target, weight)`` edge frame as a coded
   :class:`~guidedLP.common.edgelist.EdgeList` + :class:`IDMapper`.

Optional Stage 1.5 (enabled by passing ``content_seeds`` as a DataFrame):
attach caller-supplied edges (e.g. synthetic label-anchor / stat-user edges)
onto the unipartite EdgeList via :meth:`~guidedLP.common.edgelist.EdgeList.attach`
before the backbone runs. The graph is **undirected** so each anchor edge
should be emitted **once** (no mirroring) — same convention as
:func:`run_undirected_bipartite_pipeline`.

2. :func:`apply_backbone(method="noise_corrected")` — backbone the unipartite
   EdgeList. ``apply_backbone`` reads ``edge_list.directed`` (``False`` here)
   so the undirected ``2 · Σw`` normalization is selected automatically.

This pipeline is **agnostic about how the unipartite edges were produced**.
The input is just a frame of pairwise edges between nodes of a single
partition with associated weights — the provenance (embedding cosine
similarity, signed network, manually scored ties, etc.) is the caller's
business. Use the bipartite sibling
(:func:`run_undirected_bipartite_pipeline`) when you have a raw
two-partition (e.g. user ↔ hashtag) input and want the bipartite SVN
backbone + shared-neighbor projection chain. Use this one when you've
already collapsed your data to a single-partition similarity / weight
graph and only need the unipartite ``noise_corrected`` pass.

Embedding-similarity entry point: see
:func:`guidedLP.preprocessing.embedding_similarity.extract_embedding_similarity_edgelist`
— call it first, then pass its output frame here.

Three memory modes control inter-stage release AND within-call streaming:

- ``"fast"`` — no inter-stage cleanup; ``build_edgelist_from_frame`` and
  ``apply_backbone`` use the in-memory engine. Same memory profile as
  making the calls by hand, max throughput.
- ``"balanced"`` (default) — explicitly ``del`` previous stages and
  ``gc.collect()`` between steps; AND passes ``streaming=True`` to
  ``build_edgelist_from_frame`` and ``apply_backbone``. ~30% slower than
  ``"fast"`` with substantially lower peak.
- ``"low"`` — additionally checkpoint the unipartite EdgeList to parquet
  on disk between Stage 1(.5) and Stage 2. Peak memory becomes the max
  *single* stage's working set rather than the sum across overlapping
  stages. Inherits ``streaming=True`` from ``"balanced"``.
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
from guidedLP.common.exceptions import ValidationError
from guidedLP.common.id_mapper import IDMapper
from guidedLP.network.backboning import apply_backbone
from guidedLP.network.construction import build_edgelist_from_frame
from guidedLP.pipelines._runtime import (
    MemoryMode,
    StageStats,
    maybe_free,
    read_edgelist_parquet,
    write_edgelist_parquet,
)


@dataclass
class UndirectedUnipartitePipelineResult:
    """Return value of :func:`run_undirected_unipartite_pipeline`.

    Attributes
    ----------
    edgelist : EdgeList
        Final backboned unipartite edge list. Always undirected
        (``edgelist.directed is False``, ``edgelist.bipartite is False``).
        If ``content_seeds`` was provided, the backbone ran on the
        union with the caller's extra edges, so any of those edges that
        survived the backbone are present here.
    id_mapper : IDMapper
        Mapper for ``edgelist``'s codes. Covers exactly the nodes
        surviving the backbone — including any new nodes introduced by
        ``content_seeds``.
    stage_stats : list[StageStats]
        Per-stage telemetry in execution order.
    intermediates : dict[str, Any], optional
        Only populated when ``keep_intermediates=True``. Keys:
        ``"unipartite"`` (the (EdgeList, IDMapper) after Stage 1) and
        — when ``content_seeds`` was attached — ``"unipartite_attached"``.
    """

    edgelist: EdgeList
    id_mapper: IDMapper
    stage_stats: List[StageStats]
    intermediates: Optional[Dict[str, Any]] = None

    @property
    def total_duration_s(self) -> float:
        return sum(s.duration_s for s in self.stage_stats)


def run_undirected_unipartite_pipeline(
    source: Union[str, Path, pl.DataFrame],
    *,
    source_col: str = "source",
    target_col: str = "target",
    weight_col: Optional[str] = "weight",
    # Stage 1: build_edgelist.
    auto_weight: bool = False,
    allow_self_loops: bool = True,
    remove_duplicates: bool = False,
    # Stage 1.5 (optional): caller-supplied extras.
    content_seeds: Optional[pl.DataFrame] = None,
    # Stage 2: unipartite backbone.
    projection_threshold: float = 1.0,
    projection_target_fraction: Optional[float] = None,
    enable_backbone: bool = True,
    # Cross-stage protection.
    protected_nodes: Optional[List[Any]] = None,
    # Memory & I/O.
    memory_mode: MemoryMode = "balanced",
    checkpoint_dir: Optional[Union[str, Path]] = None,
    keep_intermediates: bool = False,
    verbose: bool = True,
) -> UndirectedUnipartitePipelineResult:
    """Wrap an already-unipartite edge frame and backbone it in one call.

    The pipeline is **agnostic about the provenance** of the edges in
    ``source``. As long as the frame describes a unipartite, undirected,
    weighted (or to-be-auto-weighted) edge set between nodes of a single
    partition, it can be backboned here. Typical providers include
    :func:`extract_embedding_similarity_edgelist` (pairwise cosine over
    sender-aggregated embeddings), manual scoring outputs, or any other
    upstream similarity / interaction construction.

    Parameters
    ----------
    source : str | Path | pl.DataFrame
        Already-unipartite edge frame. File paths (``.csv`` / ``.parquet``)
        are read via :func:`build_edgelist_from_frame`; a DataFrame is
        consumed directly. Must contain ``source_col`` and ``target_col``;
        a ``weight_col`` is optional (see below).
    source_col, target_col : str, defaults ``"source"`` / ``"target"``
        Endpoint column names. Values pass through untouched and become
        the node IDs in the final EdgeList (via :class:`IDMapper`).
    weight_col : str or None, default ``"weight"``
        Name of a per-row weight column on the input. When set, Stage 1
        reads the weights directly. Pass ``None`` to either compute
        weights from duplicate ``(source, target)`` row counts
        (``auto_weight=True``) or treat all edges as unit weight
        (``auto_weight=False``). The ``noise_corrected`` backbone in
        Stage 2 requires non-negative weights — pre-shift / threshold
        upstream if your raw weights can dip below zero (e.g. raw
        cosine similarity).
    auto_weight : bool, default False
        If True, count duplicate ``(source, target)`` rows to set the
        edge weight. Mutually exclusive with ``weight_col`` being set.
    allow_self_loops : bool, default True
        Forwarded to :func:`build_edgelist_from_frame`. Set to False to
        drop ``source == target`` rows during Stage 1.
    remove_duplicates : bool, default False
        Forwarded to :func:`build_edgelist_from_frame`. Set to True to
        collapse duplicate ``(source, target)`` rows; with ``weight_col``
        set, the **first occurrence's weight** is kept (no summation —
        pre-aggregate upstream if you want sum semantics).
    content_seeds : pl.DataFrame, optional
        Extra edges to attach to the unipartite EdgeList *before* the
        backbone runs (Stage 1.5). Same schema as the bipartite sibling:
        ``source_id`` / ``target_id`` / ``weight`` (Float64). Because the
        EdgeList is undirected, each anchor edge should be emitted once
        (no mirroring). New IDs are added to the mapper.
    projection_threshold, projection_target_fraction : float, optional
        ``apply_backbone(method="noise_corrected")`` parameters.
        ``projection_target_fraction`` is the recommended way to size the
        final backbone — set to ``1.0`` to keep all edges (useful on small
        graphs where the default threshold filters too aggressively).
    enable_backbone : bool, default True
        Skip the Stage 2 backbone entirely when False. The returned
        ``edgelist`` is then the post-attach (or pre-attach, if no content
        seeds) EdgeList straight out of Stage 1. Useful when the input
        frame has already been sized upstream and an additional
        statistical pass would over-prune.
    protected_nodes : list, optional
        Original IDs to exempt from filtering in the Stage 2 backbone. IDs
        not present in the mapper at backbone time produce a warning and
        are skipped.
    memory_mode : {"fast", "balanced", "low"}, default "balanced"
        Inter-stage cleanup behavior. ``"balanced"`` adds ``gc.collect()``
        between stages and passes ``streaming=True`` to
        ``build_edgelist_from_frame`` and ``apply_backbone``; ``"low"``
        additionally checkpoints the unipartite EdgeList to parquet on
        disk between Stage 1(.5) and Stage 2.
    checkpoint_dir : str | Path, optional
        Where to write parquet checkpoints in ``memory_mode="low"``. If
        unset, a temporary directory is created and cleaned up on return.
    keep_intermediates : bool, default False
        Retain references to each stage's output. Incompatible with
        ``memory_mode="low"``.
    verbose : bool, default True
        Per-stage one-liners (from the underlying functions' own verbose
        output) plus a final TOTAL line.

    Returns
    -------
    UndirectedUnipartitePipelineResult
        See dataclass docstring.

    Raises
    ------
    ValidationError
        Invalid argument combinations (``memory_mode="low"`` with
        ``keep_intermediates=True``; ``weight_col`` set together with
        ``auto_weight=True``).
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
    stream_backbones = memory_mode != "fast"

    try:
        # Stage 1: wrap the (source, target, weight?) frame as a coded
        # undirected unipartite EdgeList + IDMapper.
        t0 = _time.perf_counter()
        el_uni, mapper_uni = build_edgelist_from_frame(
            source,
            source_col=source_col,
            target_col=target_col,
            weight_col=weight_col,
            directed=False,
            bipartite=False,
            auto_weight=auto_weight,
            allow_self_loops=allow_self_loops,
            remove_duplicates=remove_duplicates,
            streaming=stream_backbones,
            verbose=verbose,
        )
        stats.append(StageStats(
            name="build_edgelist_from_frame",
            duration_s=_time.perf_counter() - t0,
            output_edges=el_uni.number_of_edges(),
            output_nodes=el_uni.n_nodes,
        ))
        if intermediates is not None:
            intermediates["unipartite"] = (el_uni, mapper_uni)

        # Stage 1.5 (optional): attach caller-supplied edges. Undirected,
        # so each anchor edge is emitted once — no mirroring.
        if content_seeds is not None:
            t0 = _time.perf_counter()
            n_in_attach = el_uni.number_of_edges()
            el_uni, mapper_uni = el_uni.attach(content_seeds, mapper_uni)
            stats.append(StageStats(
                name="attach_content_seeds",
                duration_s=_time.perf_counter() - t0,
                input_edges=n_in_attach,
                output_edges=el_uni.number_of_edges(),
                output_nodes=el_uni.n_nodes,
            ))
            if intermediates is not None:
                intermediates["unipartite_attached"] = (el_uni, mapper_uni)

        # Optional disk checkpoint between Stage 1(.5) and Stage 2.
        ckpt_path: Optional[Path] = None
        ckpt_meta: Optional[dict] = None
        if memory_mode == "low" and intermediates is None:
            ckpt_path = checkpoint_dir / "01_unipartite.parquet"
            ckpt_meta = write_edgelist_parquet(el_uni, ckpt_path)
            del el_uni
            maybe_free(memory_mode)
            el_uni = read_edgelist_parquet(ckpt_path, ckpt_meta)

        # Stage 2: unipartite noise_corrected backbone. apply_backbone reads
        # el_uni.directed (False here) and uses the undirected normalization.
        if enable_backbone:
            t0 = _time.perf_counter()
            n_in_2 = el_uni.number_of_edges()
            el_final, mapper_final = apply_backbone(
                el_uni,
                id_mapper=mapper_uni,
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
                input_edges=n_in_2,
                output_edges=el_final.number_of_edges(),
                output_nodes=el_final.n_nodes,
            ))
            if intermediates is None:
                del el_uni, mapper_uni
                maybe_free(memory_mode)
                if ckpt_path is not None:
                    ckpt_path.unlink(missing_ok=True)
        else:
            # Backbone disabled — return the post-Stage-1(.5) EdgeList as-is.
            el_final, mapper_final = el_uni, mapper_uni

        if verbose:
            total = _time.perf_counter() - pipeline_start
            print(
                f"[run_undirected_unipartite_pipeline] TOTAL {total:.2f}s | "
                f"mode={memory_mode} | "
                f"final: {el_final.number_of_edges():,} edges, "
                f"{el_final.n_nodes:,} nodes"
            )

        return UndirectedUnipartitePipelineResult(
            edgelist=el_final,
            id_mapper=mapper_final,
            stage_stats=stats,
            intermediates=intermediates,
        )

    finally:
        if created_tempdir and checkpoint_dir is not None and checkpoint_dir.exists():
            shutil.rmtree(checkpoint_dir, ignore_errors=True)
