"""Canonical attribution pipeline: raw → bipartite → backbone → temporal → backbone.

Composes the four canonical stages most attribution-style analyses use:

1. :func:`build_edgelist_from_frame` — turn the raw input into a bipartite
   coded EdgeList. The ``timestamp`` column (if any) is carried as a
   passthrough so it survives downstream.
2. :func:`apply_backbone(method="bipartite_svn")` — filter the bipartite
   edges to the statistically significant subset.
3. :func:`temporal_bipartite_to_unipartite` — citation-direction
   projection onto one partition (default: the source-side).
4. :func:`apply_backbone(method="noise_corrected")` — backbone the
   projection. ``target_fraction`` is the recommended knob here on
   directed graphs (see ``docs/architecture/chunked_projection_design.md``
   for why the threshold path tends to keep ~100% on directed inputs).

Optional Stage 5 (enabled by passing ``content_seeds``):

5. :func:`make_stat_user_edges` — when labels live on the intermediate
   (content) partition rather than the projected (user) partition,
   collapse each labeled content node into a synthetic "stat user" that
   stays anchored to its label via GLP's ``(1-α)Y`` term. The synthetic
   edges are concatenated onto the projection AFTER both backbones (the
   stat users' artificial degree profile isn't calibrated for SVN /
   noise-corrected), and their IDs are returned on the result so the
   caller can hand them to ``guided_label_propagation`` as ``seed_labels``
   + ``exclude_from_output``. See README example 5 and
   docs/architecture/glp.md ("Bipartite Graphs & Stat-User Augmentation")
   for the design rationale.

Three memory modes control inter-stage release AND within-call streaming:

- ``"fast"`` — no inter-stage cleanup; ``build_edgelist_from_frame``
  and both ``apply_backbone`` calls use the in-memory engine. Same
  memory profile as making the calls by hand, max throughput.
- ``"balanced"`` (default) — explicitly ``del`` previous stages and
  ``gc.collect()`` between steps; drops the raw input frame as soon
  as it's encoded; AND passes ``streaming=True`` to
  ``build_edgelist_from_frame`` (streams the Utf8→UInt32 encode) and
  the two ``apply_backbone`` calls (streams their wide-column /
  ``poisson.sf`` work). ~30% slower than ``"fast"`` with substantially
  lower peak across all four stages.
- ``"low"`` — additionally checkpoint each stage's EdgeList to parquet
  on disk and release the in-memory frame. Peak memory becomes the max
  *single* stage's working set rather than the sum across overlapping
  stages. Inherits ``streaming=True`` from ``"balanced"``. Adds disk I/O
  time (~few seconds on typical hardware).
"""

from __future__ import annotations

import gc
import shutil
import tempfile
import time as _time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union

import polars as pl

from guidedLP.common.edgelist import EdgeList
from guidedLP.common.id_mapper import IDMapper
from guidedLP.common.exceptions import ValidationError
from guidedLP.common.seed_input import SeedInput
from guidedLP.glp.utils import make_stat_user_edges
from guidedLP.network.backboning import apply_backbone
from guidedLP.network.construction import (
    build_edgelist_from_frame,
    temporal_bipartite_to_unipartite,
)
from guidedLP.pipelines._runtime import (
    MemoryMode,
    StageStats,
    maybe_free,
    read_edgelist_parquet,
    write_edgelist_parquet,
)


# Internal column name used to carry the user's weight through
# build_edgelist_from_frame as a passthrough. Cannot collide with the
# reserved names {"src", "tgt", "weight"} that EdgeList uses for its
# coded columns.
_INTERNAL_WEIGHT_COL = "__glp_pipeline_weight"


def _load_source_for_rename(source: Union[str, Path, pl.DataFrame]) -> pl.DataFrame:
    """Materialize file paths to a Polars DataFrame so the weight rename
    can run. Pass-through for DataFrames.
    """
    if isinstance(source, pl.DataFrame):
        return source
    path = Path(source)
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pl.read_parquet(path)
    if suffix in (".csv", ".tsv", ".txt"):
        sep = "\t" if suffix == ".tsv" else ","
        return pl.read_csv(path, separator=sep)
    raise ValidationError(
        f"Unsupported source file extension: {suffix!r}. "
        f"Expected one of .parquet, .csv, .tsv, .txt — or pass a "
        f"pre-loaded Polars DataFrame."
    )


@dataclass
class CanonicalPipelineResult:
    """Return value of :func:`run_canonical_pipeline`.

    Attributes
    ----------
    edgelist : EdgeList
        Final backboned projection. If ``content_seeds`` was provided,
        this also contains the synthetic stat-user edges concatenated
        with the real user-user edges.
    id_mapper : IDMapper
        Mapper for ``edgelist``'s codes. Only covers nodes surviving the
        projection backbone — much smaller than the bipartite input's
        mapper. When stat-user augmentation ran, the synthetic stat-user
        IDs (``stat_node_ids``) are also present in the mapper.
    stage_stats : list[StageStats]
        Per-stage telemetry in execution order.
    intermediates : dict[str, Any], optional
        Only populated when ``keep_intermediates=True``. Keys:
        ``"bipartite"``, ``"bipartite_filtered"``, ``"projection"``,
        each mapped to an ``(EdgeList, IDMapper)`` tuple. When stat-user
        augmentation ran, ``"projection_backboned"`` (the pre-augmented
        EdgeList) is also present.
    stat_seeds : dict[str, str], optional
        Stat-user seed dict (``__stat__{content}`` → label) produced by
        :func:`guidedLP.glp.make_stat_user_edges`. Pass to
        :func:`guidedLP.glp.guided_label_propagation` as ``seed_labels``.
        ``None`` when ``content_seeds`` was not provided.
    stat_node_ids : set[str], optional
        Set of synthetic stat-user IDs that appear in ``edgelist``. Pass
        to ``guided_label_propagation`` as ``exclude_from_output`` to drop
        them from the final result. ``None`` when ``content_seeds`` was
        not provided.
    """

    edgelist: EdgeList
    id_mapper: IDMapper
    stage_stats: List[StageStats]
    intermediates: Optional[Dict[str, Any]] = None
    stat_seeds: Optional[Dict[str, str]] = None
    stat_node_ids: Optional[Set[str]] = None

    @property
    def total_duration_s(self) -> float:
        return sum(s.duration_s for s in self.stage_stats)


def run_canonical_pipeline(
    source: Union[str, Path, pl.DataFrame],
    *,
    source_col: str,
    target_col: str,
    timestamp_col: str,
    weight_col: Optional[str] = None,
    # Projection orientation.
    intermediate_col: Optional[str] = None,
    projected_col: Optional[str] = None,
    # Stage 1: build_edgelist.
    min_source_degree: Optional[int] = None,
    min_target_degree: Optional[int] = None,
    auto_weight: bool = False,
    bipartite_overlap: str = "drop",
    # Stage 2: bipartite backbone.
    bipartite_alpha: float = 0.01,
    bipartite_correction: str = "fdr_bh",
    bipartite_target_fraction: Optional[float] = None,
    bipartite_min_node_retention: Optional[float] = None,
    # Stage 3: temporal projection.
    add_edge_weights: bool = True,
    remove_self_loops: bool = True,
    presort_temporal: bool = True,
    # Stage 4: projection backbone.
    projection_threshold: float = 1.0,
    projection_target_fraction: Optional[float] = None,
    # Stage 5 (optional): stat-user augmentation.
    content_seeds: Optional[SeedInput] = None,
    stat_prefix: str = "__stat__",
    # Cross-stage protection.
    protected_nodes: Optional[List[Any]] = None,
    # Memory & I/O.
    memory_mode: MemoryMode = "balanced",
    checkpoint_dir: Optional[Union[str, Path]] = None,
    keep_intermediates: bool = False,
    verbose: bool = True,
) -> CanonicalPipelineResult:
    """Run the canonical raw → backboned-projection pipeline in one call.

    Parameters
    ----------
    source : str | Path | pl.DataFrame
        Raw input. File paths (``.csv``, ``.parquet``) are read via
        Polars; a DataFrame is consumed directly.
    source_col, target_col : str
        Column names for the bipartite endpoints.
    timestamp_col : str
        Column carrying the per-edge timestamp; required for the
        temporal projection step.
    weight_col : str, optional
        Per-edge weight column on the raw input. When provided it is
        carried through to the temporal-projection stage (used in the
        ``(w_i + w_j) / 2 * 1 / (1 + Δdays)`` formula) but the
        bipartite-side backbone (Stage 2) still runs on raw row counts —
        see Stage 1's implementation comment for why. Pre-aggregate the
        input to one row per ``(source, target)`` pair if you need
        weighted bipartite_svn.
    intermediate_col, projected_col : str, optional
        Which side of the bipartite to collapse vs preserve in the
        projection. Default: ``intermediate_col=target_col`` and
        ``projected_col=source_col`` (i.e. project onto the source
        partition — the user-side in typical user-content data).
    min_source_degree, min_target_degree : int, optional
        Degree filters applied during the build step.
    auto_weight : bool
        If True, count duplicate edges to set ``weight``.
    bipartite_overlap : str
        Overlap-resolution policy when nodes appear on both sides of
        the bipartite. ``"drop"`` removes the offending nodes; see
        :func:`build_edgelist_from_frame` for other options.
    bipartite_alpha, bipartite_correction : float, str
        ``apply_backbone(method="bipartite_svn")`` parameters.
    bipartite_target_fraction : float, optional
        If set, overrides ``bipartite_alpha`` and keeps the top fraction
        of bipartite edges by p-value.
    bipartite_min_node_retention : float, optional
        Post-filter for stage 2: drop any bipartite node whose surviving
        edges fall below this fraction of its original incident-edge
        count (e.g. ``0.1`` removes nodes that lost more than 90% of
        their edges to the SVN cut). Useful for eliminating generic
        high-degree nodes whose edges are mostly noise. ``None`` (default)
        keeps every node the per-edge SVN selected. Protected nodes
        (see ``protected_nodes``) are exempt.
    add_edge_weights, remove_self_loops : bool
        Forwarded to :func:`temporal_bipartite_to_unipartite`.
    presort_temporal : bool, default True
        If True (default), sort the bipartite by
        ``[intermediate_col, timestamp_col DESC]`` before the temporal
        step — needed for the citation-direction edges to be correct.
        Set False only if you know the input EdgeList is already in
        this order (e.g. because you pre-sorted the raw frame and the
        intervening steps preserved row order).
    projection_threshold, projection_target_fraction : float, optional
        ``apply_backbone(method="noise_corrected")`` parameters.
        ``projection_target_fraction`` is the recommended way to size
        the final backbone on directed projections.
    content_seeds : SeedInput, optional
        Labels living on the intermediate (content) partition — i.e.
        ``intermediate_col``. When provided, enables the optional
        Stage 5: each labeled content node is converted to a synthetic
        "stat user" via :func:`guidedLP.glp.make_stat_user_edges`, whose
        edges are concatenated with the projection. The returned
        ``stat_seeds`` / ``stat_node_ids`` should be passed to
        ``guided_label_propagation`` as ``seed_labels`` and
        ``exclude_from_output`` respectively. When ``None``, Stage 5 is
        skipped and ``stat_seeds`` / ``stat_node_ids`` on the result are
        also ``None``. Accepts any of the four ``SeedInput`` shapes (see
        :func:`guidedLP.common.normalize_seed_input`).
    stat_prefix : str, default "__stat__"
        Prefix used to generate stat-user IDs from content IDs. Pick a
        value that cannot collide with any real user ID — a
        ``ValidationError`` is raised on collision. Ignored when
        ``content_seeds`` is None.
    protected_nodes : list, optional
        Original IDs to exempt from filtering in *both* backbone stages.
        Edges incident to a protected node are forced kept by stage 2
        (bipartite_svn) and stage 4 (noise_corrected); protected nodes
        also survive each stage's ``min_node_retention`` /
        ``keep_disconnected`` post-passes. The same list is forwarded to
        both calls — IDs that aren't present in a given stage's mapper
        produce a warning and are skipped, so it's safe to pass a list
        that only some stages know about (typical use: protect specific
        ``projected_col`` nodes, which exist in both the bipartite and
        projection mappers).
    memory_mode : {"fast", "balanced", "low"}, default "balanced"
        See module docstring. In short: ``"fast"`` runs the backbones
        in-memory for max throughput; ``"balanced"`` and ``"low"``
        additionally pass ``streaming=True`` to both ``apply_backbone``
        calls for ~2× lower within-call peak at ~30% wall-clock cost.
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
        own verbose output.

    Returns
    -------
    CanonicalPipelineResult
        See dataclass docstring.

    Raises
    ------
    ValidationError
        On invalid argument combinations (e.g. ``memory_mode="low"``
        with ``keep_intermediates=True``).
    """
    # Argument resolution.
    if intermediate_col is None:
        intermediate_col = target_col
    if projected_col is None:
        projected_col = source_col

    if memory_mode == "low" and keep_intermediates:
        raise ValidationError(
            "memory_mode='low' is incompatible with keep_intermediates=True; "
            "the checkpoint cycle frees the in-memory frame between stages."
        )
    if memory_mode not in ("fast", "balanced", "low"):
        raise ValidationError(
            f"memory_mode must be 'fast', 'balanced', or 'low'; got {memory_mode!r}"
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
        # Caller passed a dir without low-memory mode — silently ignore;
        # they may be reusing a config block.
        checkpoint_dir = Path(checkpoint_dir)

    stats: List[StageStats] = []
    intermediates: Optional[Dict[str, Any]] = {} if keep_intermediates else None
    pipeline_start = _time.perf_counter()
    # Enable per-call streaming in the two apply_backbone steps when the
    # caller has opted into any memory-conscious mode. "fast" stays
    # in-memory for maximum throughput; "balanced" and "low" trade ~30%
    # wall-clock for ~2× lower peak memory inside the backbone calls.
    stream_backbones = memory_mode != "fast"

    try:
        # Stage 1: build the bipartite EdgeList with timestamp (and optional
        # user weight) carried as passthrough columns.
        #
        # build_edgelist_from_frame rejects the combination
        # ``weight_col != None + passthrough_cols`` because its weight-sum
        # branch would group_by (src, tgt) and drop per-row passthrough
        # values. Routing the user's weight through ``passthrough_cols``
        # instead preserves both per-row timestamp AND per-row weight on
        # the EdgeList, so the temporal projection downstream can still
        # use the original weight values via its own ``weight_col``
        # parameter. The user's weight column is renamed to a reserved-
        # safe internal name first, since passthrough_cols cannot include
        # the literal name "weight" (which is reserved for the EdgeList's
        # own weight column).
        #
        # Side effect: the bipartite-side backbone (Stage 2) runs on the
        # raw row count (effective weight = 1.0 per row), not the user's
        # weight. If you need a weighted bipartite_svn, pre-aggregate the
        # input to one row per (source, target) pair before calling this
        # pipeline.
        t0 = _time.perf_counter()
        passthrough = [timestamp_col]
        internal_weight: Optional[str] = None
        # Track whether WE materialized the source frame (so we know it's
        # safe to release after build_edgelist). True iff we loaded a file
        # path OR did a rename — both produce a frame WE own. False when
        # the caller passed a DataFrame and we didn't need to rename it
        # (in which case the caller still expects to use it after the call).
        source_owned_by_us = False
        # Materialize upfront when EITHER a weight rename OR stat-user
        # augmentation needs an in-memory frame; the latter calls
        # make_stat_user_edges directly on the source below.
        needs_materialized_source = (
            weight_col is not None or content_seeds is not None
        )
        if needs_materialized_source and not isinstance(source, pl.DataFrame):
            source = _load_source_for_rename(source)
            source_owned_by_us = True
        if weight_col is not None:
            if weight_col not in source.columns:
                raise ValidationError(
                    f"weight_col={weight_col!r} not found in source columns: "
                    f"{source.columns}"
                )
            if weight_col != _INTERNAL_WEIGHT_COL:
                source = source.rename({weight_col: _INTERNAL_WEIGHT_COL})
                source_owned_by_us = True
            internal_weight = _INTERNAL_WEIGHT_COL
            passthrough.append(internal_weight)

        # Stat-user augmentation (precompute): generate synthetic
        # label-as-node edges from the raw bipartite NOW, while the source
        # frame is alive. The result is a small frame in original IDs
        # (one row per real user × seed-content engagement) that we hold
        # onto until after Stage 4 then concat with the projection. See
        # make_stat_user_edges for the rationale and the
        # "Bipartite Graphs & Stat-User Augmentation" section of
        # docs/architecture/glp.md.
        stat_edges_df: Optional[pl.DataFrame] = None
        stat_seeds: Optional[Dict[str, str]] = None
        stat_node_ids: Optional[Set[str]] = None
        if content_seeds is not None:
            # The projection collapses intermediate_col into edges between
            # projected_col nodes; stat users live in the projected (user)
            # partition and anchor the intermediate (content) labels.
            stat_edges_df, stat_seeds, stat_node_ids = make_stat_user_edges(
                source,
                content_seeds,
                user_col=projected_col,
                content_col=intermediate_col,
                weight_col=internal_weight,
                stat_prefix=stat_prefix,
            )
        el_bp, mapper_bp = build_edgelist_from_frame(
            source,
            source_col=source_col,
            target_col=target_col,
            weight_col=None,
            bipartite=True,
            bipartite_overlap=bipartite_overlap,
            min_source_degree=min_source_degree,
            min_target_degree=min_target_degree,
            auto_weight=auto_weight,
            remove_duplicates=False,
            passthrough_cols=passthrough,
            streaming=stream_backbones,
            verbose=verbose,
        )
        stats.append(StageStats(
            name="build_edgelist_from_frame",
            duration_s=_time.perf_counter() - t0,
            output_edges=el_bp.number_of_edges(),
            output_nodes=el_bp.n_nodes,
        ))
        # Drop the raw input frame ASAP — it can be 5-10x larger than
        # el_bp (Utf8 IDs vs UInt32 codes). The EdgeList shares no
        # buffers with the source frame (codes are freshly encoded), so
        # dropping the source releases real memory. Skipped in
        # memory_mode="fast" to match its general no-cleanup semantics.
        # Stat-user edges (if requested) were already materialized above
        # in original IDs and do not retain a reference to source.
        if memory_mode != "fast" and source_owned_by_us:
            source = None
            gc.collect()
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

        # Optional re-sort for the temporal step's row-order contract:
        # within each intermediate group, latest-first.
        if presort_temporal:
            # Map projected/intermediate user-facing names to el's coded
            # column names (src/tgt). Mirrors the convention in
            # temporal_bipartite_to_unipartite.
            inter_coded = "src" if intermediate_col == source_col else "tgt"
            sorted_df = el_bp_kept.df.sort(
                [inter_coded, timestamp_col],
                descending=[False, True],
            )
            el_bp_kept = EdgeList(
                df=sorted_df,
                directed=el_bp_kept.directed,
                bipartite=el_bp_kept.bipartite,
                n_nodes=el_bp_kept.n_nodes,
                code_dtype=el_bp_kept.code_dtype,
            )

        # Optional disk checkpoint between stages 2 and 3.
        bp_ckpt_path: Optional[Path] = None
        bp_meta: Optional[dict] = None
        if memory_mode == "low" and intermediates is None:
            bp_ckpt_path = checkpoint_dir / "01_bipartite_kept.parquet"
            bp_meta = write_edgelist_parquet(el_bp_kept, bp_ckpt_path)
            del el_bp_kept
            maybe_free(memory_mode)
            el_bp_kept = read_edgelist_parquet(bp_ckpt_path, bp_meta)

        # Stage 3: temporal projection.
        t0 = _time.perf_counter()
        n_in_3 = el_bp_kept.number_of_edges()
        el_proj, mapper_proj = temporal_bipartite_to_unipartite(
            el_bp_kept,
            id_mapper=mapper_bp_kept,
            source_col=source_col,
            target_col=target_col,
            timestamp_col=timestamp_col,
            weight_col=internal_weight,
            intermediate_col=intermediate_col,
            projected_col=projected_col,
            add_edge_weights=add_edge_weights,
            remove_self_loops=remove_self_loops,
            output_format="edgelist",
            verbose=verbose,
        )
        stats.append(StageStats(
            name="temporal_bipartite_to_unipartite",
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

        # Stage 4: projection-side backbone.
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

        # Stage 5 (optional): stat-user augmentation. Concat the synthetic
        # label-as-node edges (precomputed above) onto the projection so a
        # downstream GLP run can propagate from content seeds via the
        # (1-α)Y anchor on each stat user. Skipped when content_seeds is
        # None.
        if content_seeds is not None and stat_edges_df is not None:
            t0 = _time.perf_counter()
            n_in_5 = el_final.number_of_edges()

            # Keep the pre-augmentation projection accessible when the
            # caller asked for intermediates.
            if intermediates is not None:
                intermediates["projection_backboned"] = (el_final, mapper_final)

            # Decode the final EdgeList to original IDs so it can be
            # concatenated with stat_edges_df (which is already in
            # original IDs).
            src_codes = el_final.df["src"].to_list()
            tgt_codes = el_final.df["tgt"].to_list()
            weights = (
                el_final.df["weight"].to_list()
                if el_final.is_weighted()
                else [1.0] * el_final.number_of_edges()
            )
            final_originals = pl.DataFrame(
                {
                    "source_id": mapper_final.get_original_batch(src_codes),
                    "target_id": mapper_final.get_original_batch(tgt_codes),
                    "weight": weights,
                }
            )

            # Restrict stat-user edges to users that survived the
            # projection backbone. Targeting a pruned user here would
            # silently reintroduce it as an orphan via the re-encode
            # below, polluting the final mapper with nodes the backbone
            # explicitly removed.
            surviving = [
                mapper_final.has_original(u)
                for u in stat_edges_df["target_id"].to_list()
            ]
            stat_edges_filtered = stat_edges_df.filter(pl.Series(surviving))
            # Stat users whose every target was pruned have no edges to
            # contribute — drop them from stat_seeds / stat_node_ids so
            # the caller doesn't try to seed orphan nodes.
            remaining_stat: Set[str] = set(
                stat_edges_filtered["source_id"].to_list()
            )
            stat_seeds = {
                k: v for k, v in stat_seeds.items() if k in remaining_stat
            }
            stat_node_ids = stat_node_ids & remaining_stat

            # Mirror the synthetic edges when the projection is directed.
            # The temporal projection emits citation-direction edges
            # (later → earlier), but stat users sit outside that temporal
            # model and we want their anchoring signal to reach real
            # users regardless of the propagation direction GLP runs in.
            if el_final.directed and stat_edges_filtered.height > 0:
                reversed_stat = stat_edges_filtered.select(
                    [
                        pl.col("target_id").alias("source_id"),
                        pl.col("source_id").alias("target_id"),
                        pl.col("weight"),
                    ]
                )
                stat_edges_filtered = pl.concat(
                    [stat_edges_filtered, reversed_stat]
                )

            # Cast both frames' ID columns to Utf8 so the concat is
            # dtype-safe even when the user's real IDs aren't strings —
            # stat-user IDs are always strings (``f"{prefix}{content}"``).
            # build_edgelist_from_frame assigns fresh codes either way,
            # so downstream callers see consistent Utf8 originals via the
            # returned id_mapper.
            final_originals = final_originals.with_columns(
                pl.col("source_id").cast(pl.Utf8),
                pl.col("target_id").cast(pl.Utf8),
            )
            stat_edges_filtered = stat_edges_filtered.with_columns(
                pl.col("source_id").cast(pl.Utf8),
                pl.col("target_id").cast(pl.Utf8),
            )
            augmented = pl.concat([final_originals, stat_edges_filtered])

            el_final, mapper_final = build_edgelist_from_frame(
                augmented,
                source_col="source_id",
                target_col="target_id",
                weight_col="weight",
                directed=el_final.directed,
                bipartite=False,
                auto_weight=False,
                remove_duplicates=False,
                streaming=stream_backbones,
                verbose=verbose,
            )
            stats.append(StageStats(
                name="stat_user_augmentation",
                duration_s=_time.perf_counter() - t0,
                input_edges=n_in_5,
                output_edges=el_final.number_of_edges(),
                output_nodes=el_final.n_nodes,
            ))

        if verbose:
            total = _time.perf_counter() - pipeline_start
            print(
                f"[run_canonical_pipeline] TOTAL {total:.2f}s | "
                f"mode={memory_mode} | "
                f"final: {el_final.number_of_edges():,} edges, "
                f"{el_final.n_nodes:,} nodes"
            )

        return CanonicalPipelineResult(
            edgelist=el_final,
            id_mapper=mapper_final,
            stage_stats=stats,
            intermediates=intermediates,
            stat_seeds=stat_seeds,
            stat_node_ids=stat_node_ids,
        )

    finally:
        # Clean up the auto-created tempdir even on failure paths.
        if created_tempdir and checkpoint_dir is not None and checkpoint_dir.exists():
            shutil.rmtree(checkpoint_dir, ignore_errors=True)
