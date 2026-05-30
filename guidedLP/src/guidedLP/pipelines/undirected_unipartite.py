"""Undirected unipartite pipeline: posts/embeddings → similarity → backbone.

Composes the two-or-three stages of the direct embedding-similarity workflow:

1. :func:`extract_embedding_similarity_edgelist` — encode posts (or read
   pre-embedded vectors), aggregate per sender, compute pairwise cosine
   similarity, and apply the hybrid top-``k`` + similarity-floor filter to
   emit a unipartite ``(source, target, weight)`` edge frame.
2. :func:`build_edgelist_from_frame(bipartite=False)` — wrap that frame as a
   coded :class:`~guidedLP.common.edgelist.EdgeList` + :class:`IDMapper`.

Optional Stage 2.5 (enabled by passing ``content_seeds`` as a DataFrame):
attach caller-supplied edges (e.g. synthetic label-anchor / stat-user edges)
onto the unipartite EdgeList via :meth:`~guidedLP.common.edgelist.EdgeList.attach`
before the backbone runs. The graph is **undirected** so each anchor edge
should be emitted **once** (no mirroring) — same convention as
:func:`run_undirected_bipartite_pipeline`.

3. :func:`apply_backbone(method="noise_corrected")` — backbone the unipartite
   EdgeList. ``apply_backbone`` reads ``edge_list.directed`` (``False`` here)
   so the undirected ``2 · Σw`` normalization is selected automatically.

This pipeline is the embedding-direct counterpart to
:func:`run_undirected_bipartite_pipeline`. Use this one when actor similarity
is best captured by the *semantic geometry* of an embedding model (sentence-
transformer vectors over posts, for example), not by co-occurrence of shared
content items. Use the bipartite sibling when actor similarity comes from
explicit shared items (hashtags, URLs, domains, keywords) — that path
preserves item-level structure through the SVN backbone, which this pipeline
cannot, since it has no bipartite stage.

The three memory modes from the bipartite sibling apply identically here.
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
from guidedLP.preprocessing.embedding_extraction import _DEFAULT_MODEL
from guidedLP.preprocessing.embedding_similarity import (
    extract_embedding_similarity_edgelist,
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
        ``"similarity_edges"`` (the pl.DataFrame straight out of stage 1),
        ``"unipartite"`` (the (EdgeList, IDMapper) after Stage 2), and
        — when ``content_seeds`` was attached — ``"unipartite_attached"``.
    """

    edgelist: EdgeList
    id_mapper: IDMapper
    stage_stats: List[StageStats]
    intermediates: Optional[Dict[str, Any]] = None

    @property
    def total_duration_s(self) -> float:
        return sum(s.duration_s for s in self.stage_stats)


def _load_source(source: Union[str, Path, pl.DataFrame]) -> pl.DataFrame:
    """Materialize ``source`` as an in-memory ``pl.DataFrame``.

    Path inputs are read with the obvious polars reader keyed on suffix
    (``.csv`` / ``.parquet``). DataFrame inputs pass through unchanged.
    Other file formats raise rather than silently misreading.
    """
    if isinstance(source, pl.DataFrame):
        return source
    p = Path(source)
    if p.suffix == ".csv":
        return pl.read_csv(p)
    if p.suffix == ".parquet":
        return pl.read_parquet(p)
    raise ValidationError(
        f"Unsupported source file extension {p.suffix!r}; "
        "expected .csv or .parquet (or pass a pl.DataFrame directly).",
        field="source",
        value=str(source),
    )


def run_undirected_unipartite_pipeline(
    source: Union[str, Path, pl.DataFrame],
    *,
    # Input columns
    sender_col: str = "sender",
    post_col: str = "post",
    embedding_col: Optional[str] = None,
    datetime_col: Optional[str] = "datetime",
    # Embedding model + caching (used only when embedding_col is None)
    model: Union[str, Any] = _DEFAULT_MODEL,
    batch_size: int = 32,
    show_progress: bool = False,
    device: Optional[str] = None,
    normalize_embeddings: bool = True,
    save_path: Optional[Union[str, Path]] = None,
    create_new: bool = False,
    aggregate_inline: bool = True,
    chunk_size: Optional[int] = None,
    aggregation: str = "mean",
    # Stage 1: hybrid k-NN + similarity floor
    metric: str = "cosine",
    k: int = 30,
    similarity_threshold: float = 0.0,
    mutual: bool = False,
    weight_transform: str = "shift",
    shift_amount: float = 2.0,
    power: float = 0.5,
    similarity_n_jobs: int = 1,
    # Stage 2.5 (optional): caller-supplied extras
    content_seeds: Optional[pl.DataFrame] = None,
    # Stage 3: unipartite backbone
    projection_threshold: float = 1.0,
    projection_target_fraction: Optional[float] = None,
    enable_backbone: bool = True,
    # Cross-stage protection
    protected_nodes: Optional[List[Any]] = None,
    # Memory & I/O
    memory_mode: MemoryMode = "balanced",
    checkpoint_dir: Optional[Union[str, Path]] = None,
    keep_intermediates: bool = False,
    verbose: bool = True,
) -> UndirectedUnipartitePipelineResult:
    """Build and backbone an embedding-similarity unipartite graph in one call.

    The pipeline replaces the bipartite + projection pair of
    :func:`run_undirected_bipartite_pipeline` with a single direct
    similarity-based construction step
    (:func:`extract_embedding_similarity_edgelist`), then runs the same
    unipartite ``noise_corrected`` backbone on the result. There is no
    bipartite stage and no bipartite backbone — the embedding step compresses
    the sender↔dimension structure into pairwise sender similarity directly,
    so the SVN test that the bipartite sibling uses has no analog here.

    Parameters
    ----------
    source : str | Path | pl.DataFrame
        Raw input post table. File paths (``.csv`` / ``.parquet``) are loaded
        with polars; a DataFrame is consumed directly. Must contain
        ``sender_col`` plus either ``post_col`` (Utf8 raw text — encoded with
        the chosen model) or ``embedding_col`` (List/Array of per-post
        embedding vectors — no model loaded).
    sender_col : str, default "sender"
        Author column. Values are passed through untouched and become the
        node IDs in the final EdgeList.
    post_col : str, default "post"
        Post-text column. Used only when ``embedding_col`` is None.
    embedding_col : str, optional
        If set, the pipeline reads per-post embeddings directly from this
        column (List/Array of numeric values, equal length per row) and the
        ``[embeddings]`` extra is *not* required.
    datetime_col : str or None, default "datetime"
        Timestamp column. Used only for null-filtering parity with
        :func:`extract_embedding_similarity_edgelist`; not propagated through
        Stage 2 onward (edge-level timestamps are ill-defined in a similarity
        graph). Pass ``None`` to drop the requirement.
    model, batch_size, show_progress, device, normalize_embeddings,
    save_path, create_new, aggregate_inline, chunk_size, aggregation
        Forwarded verbatim to
        :func:`extract_embedding_similarity_edgelist`. See its docstring for
        full semantics. The on-disk cache (``save_path``) is interchangeable
        with the cache produced by :func:`extract_embedding_features` as
        long as the corpus / ``aggregation`` / ``normalize_embeddings``
        match.
    metric : {"cosine"}, default "cosine"
        Pairwise similarity metric. Other metrics may be added upstream
        later — this pipeline simply forwards.
    k : int, default 30
        Top-``k`` cap per sender. Bounds the max degree in dense embedding
        regions; ``20``-``50`` is a typical starting range.
    similarity_threshold : float, default 0.0
        Minimum cosine similarity to keep an edge. The lever for isolating
        outlier actors whose nearest neighbors are still dissimilar in
        absolute terms. Applied to **raw** cosine similarity (before
        ``weight_transform``), so switching transforms never silently
        shifts the effective filter level.
    mutual : bool, default False
        If True, keep an edge only when *both* directions of the top-``k``
        relation hold (Ozaki et al., CoNLL 2011). Sparser and more
        outlier-resistant; may disconnect parts of the graph.
    weight_transform : {"shift", "raw", "abs", "power_shift"}, default "shift"
        How to convert raw cosine similarity into the edge weight.
        **The pipeline default differs from the underlying function**
        (which defaults to ``"raw"``): the unipartite backbone in Stage 3
        runs ``noise_corrected``, which assumes non-negative weights, so
        we default to ``"shift"`` (cosine + ``shift_amount``) here. Pass
        ``"raw"`` if you've already restricted ``similarity_threshold`` to
        a non-negative value and want raw similarities on the output edges.
    shift_amount, power : float
        Knobs for the ``"shift"`` / ``"power_shift"`` transforms.
    similarity_n_jobs : int, default 1
        Thread count for the top-``k`` selection in
        :func:`extract_embedding_similarity_edgelist`. ``-1`` uses all
        available CPU cores. Above ~10k senders this is the lever that
        matters on many-core machines; the BLAS matmul ``X @ X.T`` is
        already multi-threaded regardless.
    content_seeds : pl.DataFrame, optional
        Extra edges to attach to the unipartite EdgeList *before* the
        backbone runs (Stage 2.5). Same schema as the bipartite sibling:
        ``source_id`` / ``target_id`` / ``weight`` (Float64). Because the
        EdgeList is undirected, each anchor edge should be emitted once
        (no mirroring). New IDs are added to the mapper.
    projection_threshold, projection_target_fraction : float, optional
        ``apply_backbone(method="noise_corrected")`` parameters.
        ``projection_target_fraction`` is the recommended way to size the
        final backbone — set to ``1.0`` to keep all edges (useful on small
        graphs where the default threshold filters too aggressively).
    enable_backbone : bool, default True
        Skip the Stage 3 backbone entirely when False. The returned
        ``edgelist`` is then the post-attach (or pre-attach, if no content
        seeds) EdgeList straight out of Stage 2. Useful when you've already
        sized the graph via ``k`` + ``similarity_threshold`` and don't want
        an additional statistical pass.
    protected_nodes : list, optional
        Original IDs to exempt from filtering in the Stage 3 backbone. IDs
        not present in the mapper at backbone time produce a warning and
        are skipped.
    memory_mode : {"fast", "balanced", "low"}, default "balanced"
        Inter-stage cleanup behavior. ``"balanced"`` adds ``gc.collect()``
        between stages and passes ``streaming=True`` to ``apply_backbone``;
        ``"low"`` additionally checkpoints the unipartite EdgeList to
        parquet on disk between Stage 2 and Stage 3.
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
        Invalid argument combinations (e.g. ``memory_mode="low"`` with
        ``keep_intermediates=True``), unsupported ``source`` file format.
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
        # Stage 1: encode + aggregate + hybrid top-k/threshold filter.
        # Produces a polars DataFrame keyed [source, target, weight].
        t0 = _time.perf_counter()
        df_in = _load_source(source)
        sim_edges = extract_embedding_similarity_edgelist(
            df_in,
            sender_col=sender_col,
            post_col=post_col,
            embedding_col=embedding_col,
            datetime_col=datetime_col,
            source_col="source_id",  # match build_edgelist_from_frame's expected schema downstream
            target_col="target_id",
            weight_col="weight",
            model=model,
            batch_size=batch_size,
            show_progress=show_progress,
            device=device,
            normalize_embeddings=normalize_embeddings,
            save_path=save_path,
            create_new=create_new,
            aggregate_inline=aggregate_inline,
            chunk_size=chunk_size,
            aggregation=aggregation,
            metric=metric,
            k=k,
            similarity_threshold=similarity_threshold,
            mutual=mutual,
            weight_transform=weight_transform,
            shift_amount=shift_amount,
            power=power,
            n_jobs=similarity_n_jobs,
            verbose=verbose,
        )
        # Free the raw input frame; downstream stages don't need it.
        del df_in
        maybe_free(memory_mode)
        stats.append(StageStats(
            name="extract_embedding_similarity_edgelist",
            duration_s=_time.perf_counter() - t0,
            output_edges=sim_edges.height,
            output_nodes=0,  # not tracked here; populated after Stage 2
        ))
        if intermediates is not None:
            intermediates["similarity_edges"] = sim_edges

        # Stage 2: wrap the (source_id, target_id, weight) frame as a coded
        # undirected unipartite EdgeList + IDMapper.
        t0 = _time.perf_counter()
        n_in_2 = sim_edges.height
        el_uni, mapper_uni = build_edgelist_from_frame(
            sim_edges,
            source_col="source_id",
            target_col="target_id",
            weight_col="weight",
            bipartite=False,
            streaming=stream_backbones,
            verbose=verbose,
        )
        stats.append(StageStats(
            name="build_edgelist_from_frame",
            duration_s=_time.perf_counter() - t0,
            input_edges=n_in_2,
            output_edges=el_uni.number_of_edges(),
            output_nodes=el_uni.n_nodes,
        ))
        if intermediates is not None:
            intermediates["unipartite"] = (el_uni, mapper_uni)
        else:
            del sim_edges
            maybe_free(memory_mode)

        # Stage 2.5 (optional): attach caller-supplied edges. Undirected,
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

        # Optional disk checkpoint between Stage 2(.5) and Stage 3.
        ckpt_path: Optional[Path] = None
        ckpt_meta: Optional[dict] = None
        if memory_mode == "low" and intermediates is None:
            ckpt_path = checkpoint_dir / "01_unipartite.parquet"
            ckpt_meta = write_edgelist_parquet(el_uni, ckpt_path)
            del el_uni
            maybe_free(memory_mode)
            el_uni = read_edgelist_parquet(ckpt_path, ckpt_meta)

        # Stage 3: unipartite noise_corrected backbone. apply_backbone reads
        # el_uni.directed (False here) and uses the undirected normalization.
        if enable_backbone:
            t0 = _time.perf_counter()
            n_in_3 = el_uni.number_of_edges()
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
                input_edges=n_in_3,
                output_edges=el_final.number_of_edges(),
                output_nodes=el_final.n_nodes,
            ))
            if intermediates is None:
                del el_uni, mapper_uni
                maybe_free(memory_mode)
                if ckpt_path is not None:
                    ckpt_path.unlink(missing_ok=True)
        else:
            # Backbone disabled — return the post-Stage-2(.5) EdgeList as-is.
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
