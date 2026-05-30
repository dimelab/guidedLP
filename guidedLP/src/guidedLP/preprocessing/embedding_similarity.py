"""
Direct embedding -> unipartite actor-actor edgelist via hybrid top-k + threshold.

This is the alternative to the bipartite-projection path (``extract_embedding_features``
-> ``build_graph_from_edgelist(bipartite=True)`` -> ``project_bipartite``).
Instead of materializing an intermediate bipartite sender->dimension graph and
reconstructing pairwise similarity from co-occurrence counts (which discards
the embedding's signed geometry), we compute pairwise cosine similarity
directly on per-sender embeddings and keep an edge between two senders ``i``
and ``j`` only if **both**:

1. ``j`` is among ``i``'s top-``k`` most similar senders (or vice versa), AND
2. their similarity is at least ``similarity_threshold``.

This *hybrid k-NN + similarity-floor* construction sidesteps two well-known
failure modes of single-criterion graphs:

- A pure top-``k`` graph forces every node to have the same degree, which
  fabricates edges for esoteric/outlier actors whose top-``k`` are still
  dissimilar in absolute terms. The threshold floor isolates those actors.
- A pure :math:`\\varepsilon`-graph respects natural sparsity but a single
  global threshold over- or under-connects regions of differing density.
  The top-``k`` cap bounds the degree in dense regions.

Setting ``mutual=True`` requires both directions of the top-``k`` relation to
hold — a strictly stronger filter that yields even sparser graphs and is
specifically known to help label propagation (see Ozaki et al., 2011). The
default (``mutual=False``) symmetrizes the directed top-``k`` candidates and
keeps an edge whenever the threshold is met from either side.

References
----------
.. [1] M. Maier, M. Hein, and U. von Luxburg, "Cluster identification in
       nearest-neighbor graphs," ALT 2009. Shows that the *choice* of graph
       construction (k-NN vs. mutual k-NN vs. epsilon) materially changes
       downstream cluster recovery and convergence behavior.
.. [2] H. Ozaki, M. Shimbo, M. Komachi, and Y. Matsumoto, "Using the mutual
       k-nearest neighbor graphs for semi-supervised classification of natural
       language data," CoNLL 2011. Demonstrates that mutual k-NN graphs
       outperform plain k-NN for label propagation specifically because
       outlier nodes get correctly isolated — the directly applicable result
       for this library's GLP use case.
.. [3] T. Jebara, J. Wang, and S.-F. Chang, "Graph construction and b-matching
       for semi-supervised learning," ICML 2009. Frames graph construction as
       a sparsification problem and motivates degree-constrained alternatives
       to pure k-NN.
.. [4] W. Dong, M. Charikar, and K. Li, "Efficient k-nearest neighbor graph
       construction for generic similarity measures," WWW 2011. The NN-Descent
       algorithm; relevant when the brute-force ``O(N^2)`` similarity matrix
       becomes the bottleneck (typically ``N > ~50k``).
"""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np
import polars as pl

from guidedLP.common.exceptions import ValidationError
from guidedLP.preprocessing.embedding_extraction import (
    _DEFAULT_MODEL,
    _VALID_WEIGHT_TRANSFORMS,
    _apply_weight_transform,
    _get_aggregated_embeddings,
    _l2_normalize_rows,
    _validate_embedding_input,
    _validate_text_input,
)


_VALID_METRICS = frozenset({"cosine"})


def _topk_indices(
    sims: np.ndarray, k_eff: int, n_jobs: int
) -> np.ndarray:
    """
    Return an ``(N, k_eff)`` array of top-``k_eff`` column indices per row of
    ``sims`` (the indices into the input row's columns whose values are the
    ``k_eff`` largest, in unspecified order).

    NumPy's :func:`numpy.argpartition` releases the GIL, so we can split the
    rows of ``sims`` across a thread pool and get near-linear speedup until
    memory bandwidth saturates. BLAS already parallelizes the upstream
    ``X @ X.T`` matrix multiply; this routine is the *single-threaded*
    leftover that becomes the bottleneck on many-core machines.

    ``n_jobs`` follows scikit-learn convention: ``1`` is serial (the default,
    no thread-pool overhead), ``-1`` is "all available cores", any other
    positive integer specifies an explicit thread count. Workloads too small
    to amortize pool overhead fall back to a single-threaded call.
    """
    n = sims.shape[0]
    if n_jobs == 1:
        return np.argpartition(-sims, k_eff - 1, axis=1)[:, :k_eff]

    if n_jobs == -1:
        n_jobs = os.cpu_count() or 1

    # Below this size, the thread-pool spin-up + chunk-dispatch overhead is
    # comparable to the work itself. The threshold is conservative — pinning
    # it down precisely would need per-machine profiling.
    if n < max(512, 2 * n_jobs):
        return np.argpartition(-sims, k_eff - 1, axis=1)[:, :k_eff]

    chunk_size = (n + n_jobs - 1) // n_jobs  # ceil(n / n_jobs)
    out = np.empty((n, k_eff), dtype=np.int64)

    def _do_chunk(start: int) -> None:
        end = min(start + chunk_size, n)
        out[start:end] = np.argpartition(
            -sims[start:end], k_eff - 1, axis=1
        )[:, :k_eff]

    with ThreadPoolExecutor(max_workers=n_jobs) as ex:
        # list() forces consumption so exceptions in workers propagate.
        list(ex.map(_do_chunk, range(0, n, chunk_size)))

    return out


def _print_filter_summary(
    n_input_senders: int,
    n_output_edges: int,
    n_retained_senders: int,
    k: int,
    similarity_threshold: float,
    mutual: bool,
) -> None:
    """Single-line stats summary printed when ``verbose=True``.

    Matches the formatting used by ``build_edgelist_from_frame`` /
    ``apply_backbone``: bracketed function tag, then pipe-separated key
    figures. Reports total input sender count, output edge count, and the
    fraction of senders that ended up with at least one edge (the others got
    isolated by the threshold filter — the "outliers dropped" diagnostic).
    """
    if n_input_senders > 0:
        pct = 100.0 * n_retained_senders / n_input_senders
    else:
        pct = 0.0
    mutual_str = f" mutual={mutual}" if mutual else ""
    print(
        f"[extract_embedding_similarity_edgelist] "
        f"{n_input_senders:,} senders → "
        f"{n_retained_senders:,} retained ({pct:.1f}%), "
        f"{n_output_edges:,} edges | "
        f"k={k} threshold={similarity_threshold}{mutual_str}"
    )


def extract_embedding_similarity_edgelist(
    df: pl.DataFrame,
    *,
    # Column names (input)
    sender_col: str = "sender",
    post_col: str = "post",
    embedding_col: Optional[str] = None,
    datetime_col: Optional[str] = "datetime",
    # Column names (output)
    source_col: str = "source",
    target_col: str = "target",
    weight_col: str = "weight",
    # Embedding model (used only when embedding_col is None)
    model: Union[str, Any] = _DEFAULT_MODEL,
    batch_size: int = 32,
    show_progress: bool = False,
    device: Optional[str] = None,
    normalize_embeddings: bool = True,
    # On-disk cache for the from-scratch path
    save_path: Optional[Union[str, Path]] = None,
    create_new: bool = False,
    aggregate_inline: bool = True,
    chunk_size: Optional[int] = None,
    # Per-sender aggregation
    aggregation: str = "mean",
    # Hybrid k-NN + similarity floor
    metric: str = "cosine",
    k: int = 30,
    similarity_threshold: float = 0.0,
    mutual: bool = False,
    # Output weight transform (same machinery as extract_embedding_features)
    weight_transform: str = "raw",
    shift_amount: float = 2.0,
    power: float = 0.5,
    # Performance
    n_jobs: int = 1,
    # Logging
    verbose: bool = True,
) -> pl.DataFrame:
    """
    Build a unipartite actor-actor edgelist directly from per-actor embeddings.

    The encoding / caching / chunking / aggregation knobs (``model``,
    ``save_path``, ``aggregate_inline``, ``chunk_size``, ``aggregation``, ...)
    are identical to :func:`extract_embedding_features` so callers can swap
    between the bipartite-projection path and this direct path without
    reorganizing their pipeline. Only the *output stage* differs: instead of
    unrolling each per-sender vector into ``D`` bipartite edges, we compute
    pairwise cosine similarity, filter by a hybrid top-``k`` + threshold rule,
    and emit a unipartite ``(source, target, weight)`` edgelist.

    Edges are undirected — for each surviving pair the row appears once with
    ``source`` and ``target`` chosen so the original-ID lexicographic order is
    preserved (``source <= target``). Senders whose neighbors all fall below
    ``similarity_threshold`` get degree 0 and simply do not appear in the
    output — they remain "isolated" in the downstream graph, which is the
    intended treatment for esoteric/outlier actors.

    Parameters
    ----------
    df : pl.DataFrame
        Input post table. See :func:`extract_embedding_features` for the two
        accepted input shapes (from-scratch text encoding vs. pre-embedded).
    sender_col : str, default "sender"
        Author column in the input.
    post_col : str, default "post"
        Post-text column. Used only when ``embedding_col`` is None. Must be Utf8.
    embedding_col : str, optional
        If set, this column is read as the per-post embedding vector and the
        encoding step is skipped. Must be a List or Array column of numeric
        values, with the same length on every row.
    datetime_col : str or None, default "datetime"
        Timestamp column used for null-filtering parity with
        :func:`extract_embedding_features`. Not propagated to the output —
        edge-level timestamps are ambiguous in a similarity graph. Pass
        ``None`` to skip the column requirement entirely.
    source_col, target_col, weight_col : str
        Output column names. Defaults match
        :func:`~guidedLP.network.construction.build_graph_from_edgelist`.
    model, batch_size, show_progress, device, normalize_embeddings,
    save_path, create_new, aggregate_inline, chunk_size, aggregation
        Forwarded verbatim to the encode-and-aggregate pipeline. See
        :func:`extract_embedding_features` for full semantics — the cache file
        produced by either function is interchangeable as long as the
        ``aggregation`` / ``normalize_embeddings`` / corpus are unchanged.
        ``save_path`` is honored on **both** input paths — when
        ``embedding_col`` is set, the per-sender aggregated matrix is still
        cached (the encoding step is skipped because the per-post vectors
        already live in the input column, but the aggregation pass is
        cache-worthy in its own right).
    metric : {"cosine"}, default "cosine"
        Pairwise similarity metric. Cosine is the standard choice for
        sentence-transformer embeddings (their training objective makes
        directions, not magnitudes, semantically meaningful). Other metrics
        may be added later.
    k : int, default 30
        Top-``k`` cap per sender — only the ``k`` most-similar candidates per
        sender are considered as neighbors. Caps the maximum degree at ``k``
        (or ``2*k`` in the symmetrized non-mutual mode if both directions
        contribute distinct neighbors after threshold filtering).
    similarity_threshold : float, default 0.0
        Minimum cosine similarity required to keep an edge. For L2-normalized
        sentence-transformer embeddings, similarities live in ``[-1, 1]``;
        pairs of unrelated documents typically score near 0, so a small
        positive value (0.2-0.4) is a reasonable "must be moderately
        similar" floor. The default 0.0 means "must be at least non-negatively
        correlated" — the most permissive non-degenerate threshold.
    mutual : bool, default False
        If True, keep an edge only when *both* directions of the top-``k``
        relation hold (Ozaki et al., 2011). Sparser and more outlier-resistant
        than the default symmetrized mode at the cost of potentially
        disconnecting parts of the graph.
    weight_transform : {"raw", "shift", "abs", "power_shift"}, default "raw"
        How to convert the raw cosine similarity ``s in [similarity_threshold, 1]``
        into the output edge weight. Same machinery as
        :func:`extract_embedding_features` — the transforms were originally
        defined for L2-normalized per-dim values in ``[-1, 1]`` and apply
        identically here.

        - ``"raw"`` (default) — ``weight = s``. Negative weights are kept; the
          downstream consumer must handle them (e.g.
          :func:`~guidedLP.network.construction.build_graph_from_edgelist`
          rejects them, so pair this with a non-negative ``similarity_threshold``).
        - ``"shift"`` — ``weight = s + shift_amount``. With the default
          ``shift_amount=2.0`` and any ``similarity_threshold >= -1``, weights
          land in ``[1, 3]`` — all positive, original geometry preserved
          exactly. The standard choice when feeding into a propagation step
          that needs positive weights.
        - ``"abs"`` — ``weight = |s|``. Drops sign — anti-aligned pairs look
          identical to aligned ones, useful only for "treat similarity and
          dissimilarity symmetrically" interpretations.
        - ``"power_shift"`` — ``weight = sign(s) * |s|^power + shift_amount``.
          Use ``power < 1`` (default ``0.5``) to spread out tightly-clustered
          small similarities; ``power > 1`` compresses them. Signed geometry
          is preserved.
    shift_amount : float, default 2.0
        Constant added when ``weight_transform`` is ``"shift"`` or
        ``"power_shift"``. Ignored otherwise.
    power : float, default 0.5
        Exponent applied to ``|similarity|`` when
        ``weight_transform="power_shift"``. Must be positive.
    n_jobs : int, default 1
        Thread count for the top-``k`` selection (``argpartition`` over each
        row of the similarity matrix). ``1`` runs serially with zero overhead,
        ``-1`` uses all available CPU cores, any other positive integer
        specifies an explicit thread count. NumPy's argpartition releases the
        GIL, so threads scale near-linearly until memory bandwidth saturates.

        The upstream ``X @ X.T`` matrix multiply is *already* multi-threaded
        via BLAS regardless of this setting, so on few-core machines or small
        ``N`` it's usually not worth changing. The lever matters on many-core
        servers (16+ cores) at ``N >= ~10k``, where the single-threaded
        argpartition becomes the dominant cost. Workloads below ~512 senders
        ignore the setting and run serially.
    verbose : bool, default True
        Print a one-line summary of the filter outcome when the function
        returns: total senders in, edges out, fraction of senders retained
        (i.e. ended up with at least one above-threshold neighbor — the
        complement is the "isolated by threshold" cohort). Same formatting as
        :func:`~guidedLP.network.construction.build_graph_from_edgelist` /
        :func:`~guidedLP.network.backboning.apply_backbone`. Set ``False`` in
        tight loops or scripted runs.

    Returns
    -------
    pl.DataFrame
        Edge list with columns ``[source_col, target_col, weight_col]``, one
        row per undirected pair. Weights are raw cosine similarities in
        ``[-1, 1]`` (or ``[similarity_threshold, 1]`` after filtering). Ready
        to feed into :func:`~guidedLP.network.construction.build_graph_from_edgelist`
        with ``bipartite=False``.

    Raises
    ------
    ValidationError
        Missing columns, wrong dtype, invalid ``metric`` / ``k`` /
        ``aggregation``, or any condition that
        :func:`extract_embedding_features` would reject.
    ImportError
        From-scratch path attempted without ``sentence-transformers`` installed.

    Complexity
    ----------
    - Encoding + aggregation: identical to :func:`extract_embedding_features`.
    - Pairwise similarity: ``O(N^2 * D)`` time and ``O(N^2)`` memory for the
      similarity matrix (``N`` = number of unique senders, ``D`` = embedding
      dimension). Tractable for ``N`` up to ~50k on a workstation; larger
      corpora should use an approximate-NN backend such as FAISS (not yet
      wired in — see Reference [4]).
    - Top-``k`` selection: ``O(N^2)`` via ``argpartition``.
    - Output: at most ``N * k`` rows before symmetrization/dedup; usually much
      less after threshold filtering.

    Examples
    --------
    Pre-embedded path, three senders. Alice and Bob are aligned, Carol is
    anti-aligned with Alice. With ``k=2, similarity_threshold=0.0`` the edge
    Alice-Carol (similarity ``-1``) gets dropped despite being a top-``2``
    neighbor on both sides, leaving Carol isolated.

    >>> import polars as pl
    >>> df = pl.DataFrame({
    ...     "sender":    ["alice", "alice", "bob", "bob", "carol", "carol"],
    ...     "embedding": [[1.0, 0.0, 0.0]] * 2 + [[0.9, 0.1, 0.0]] * 2 +
    ...                  [[-1.0, 0.0, 0.0]] * 2,
    ...     "datetime":  ["2024-01-01"] * 6,
    ... })
    >>> edges = extract_embedding_similarity_edgelist(
    ...     df, embedding_col="embedding", k=2, similarity_threshold=0.0,
    ... )
    >>> set(edges["source"].to_list()) | set(edges["target"].to_list())
    {'alice', 'bob'}
    """
    # ---- Validate output-specific knobs -----------------------------------
    if metric not in _VALID_METRICS:
        raise ValidationError(
            f"metric must be one of {sorted(_VALID_METRICS)}, got {metric!r}",
            field="metric",
            value=metric,
        )

    if not isinstance(k, int) or k <= 0:
        raise ValidationError(
            "k must be a positive integer",
            field="k",
            value=k,
        )

    if not isinstance(n_jobs, int) or (n_jobs != -1 and n_jobs < 1):
        raise ValidationError(
            "n_jobs must be -1 (all cores) or a positive integer",
            field="n_jobs",
            value=n_jobs,
        )

    if weight_transform not in _VALID_WEIGHT_TRANSFORMS:
        raise ValidationError(
            f"weight_transform must be one of "
            f"{sorted(_VALID_WEIGHT_TRANSFORMS)}, got {weight_transform!r}",
            field="weight_transform",
            value=weight_transform,
        )

    if weight_transform == "power_shift" and power <= 0:
        raise ValidationError(
            "power must be > 0 for weight_transform='power_shift'",
            field="power",
            value=power,
        )

    if chunk_size is not None and chunk_size <= 0:
        raise ValidationError(
            "chunk_size must be a positive integer",
            field="chunk_size",
            value=chunk_size,
        )

    # ---- Validate inputs (delegate to extract_embedding_features's helpers)
    if embedding_col is not None:
        _validate_embedding_input(df, sender_col, embedding_col, datetime_col)
        df_work = df.drop_nulls(subset=[sender_col, embedding_col])
    else:
        _validate_text_input(df, sender_col, post_col, datetime_col)
        df_work = df.drop_nulls(subset=[sender_col, post_col])

    # chunk_size + save_path requirement only applies to the legacy per-post
    # cache (aggregate_inline=False) — same constraint as extract_embedding_features.
    if (
        chunk_size is not None
        and embedding_col is None
        and save_path is None
        and not aggregate_inline
    ):
        raise ValidationError(
            "chunk_size requires save_path on the from-scratch path when "
            "aggregate_inline=False: chunked encoding streams each chunk to a "
            "memory-mapped .npy file, so a path to write it is mandatory. "
            "With aggregate_inline=True (default), chunks are aggregated into "
            "a small per-sender accumulator and disk caching is optional.",
            field="chunk_size",
        )

    # ---- Empty / degenerate input short-circuits --------------------------
    empty_schema = {
        source_col: df.schema[sender_col],
        target_col: df.schema[sender_col],
        weight_col: pl.Float64,
    }
    if df_work.height == 0:
        if verbose:
            _print_filter_summary(0, 0, 0, k, similarity_threshold, mutual)
        return pl.DataFrame(schema=empty_schema)

    # ---- Acquire (and aggregate) embeddings -------------------------------
    unique_senders, aggregated = _get_aggregated_embeddings(
        df_work,
        sender_col=sender_col,
        post_col=post_col,
        embedding_col=embedding_col,
        model=model,
        batch_size=batch_size,
        show_progress=show_progress,
        device=device,
        normalize_embeddings=normalize_embeddings,
        save_path=Path(save_path) if save_path is not None else None,
        create_new=create_new,
        aggregate_inline=aggregate_inline,
        chunk_size=chunk_size,
        aggregation=aggregation,
    )

    n_senders = aggregated.shape[0]
    if n_senders < 2:
        # One sender, no pairs to compare.
        if verbose:
            _print_filter_summary(
                n_senders, 0, 0, k, similarity_threshold, mutual
            )
        return pl.DataFrame(schema=empty_schema)

    # ---- Pairwise similarity ----------------------------------------------
    # Cosine similarity treats direction, not magnitude, as meaningful. Mean
    # aggregation of unit vectors yields non-unit results in general, so we
    # always re-normalize the aggregated rows before the inner-product
    # computation regardless of the upstream `normalize_embeddings` flag.
    #
    # We hold the (N, N) similarity matrix in float32 — the precision drop
    # (~1e-7 vs ~1e-15) is irrelevant for similarity-based top-k filtering,
    # but the byte savings double the maximum tractable N on a fixed memory
    # budget and modern SIMD (AVX-512 / SVE / NEON) executes the matmul
    # faster on fp32 than fp64. Output weights are cast back to float64
    # below so downstream consumers see the same dtype as before.
    sims = _l2_normalize_rows(aggregated).astype(np.float32, copy=False)
    sims = sims @ sims.T  # (N, N) float32, entries in [-1, 1]
    np.fill_diagonal(sims, -np.inf)  # exclude self-loops from top-k selection

    # ---- Top-k selection via argpartition ---------------------------------
    # argpartition is O(N) per row vs argsort's O(N log N). We negate so the
    # *largest* similarities end up in the leading positions. With n_jobs > 1
    # the work is split across a thread pool — see `_topk_indices`.
    k_eff = min(k, n_senders - 1)
    top_idx = _topk_indices(sims, k_eff, n_jobs)

    # Build directed candidate edges (i -> j where j is in i's top-k_eff).
    row_idx = np.repeat(np.arange(n_senders), k_eff)
    col_idx = top_idx.reshape(-1)
    top_sim = sims[row_idx, col_idx]

    # ---- Apply similarity floor -------------------------------------------
    keep = top_sim >= similarity_threshold
    if not keep.any():
        if verbose:
            _print_filter_summary(
                n_senders, 0, 0, k, similarity_threshold, mutual
            )
        return pl.DataFrame(schema=empty_schema)

    src_int = row_idx[keep]
    tgt_int = col_idx[keep]
    # Promote back to float64 for the output — internal float32 was a memory/
    # compute optimization, not an API change. The threshold above was applied
    # to the *raw* cosine similarity; the weight_transform below is applied
    # after, so the threshold semantics stay "minimum cosine similarity"
    # regardless of how the output weights are encoded.
    sim_kept = top_sim[keep].astype(np.float64, copy=False)
    sim_kept = _apply_weight_transform(
        sim_kept, weight_transform, shift_amount, power
    )

    # ---- Canonicalize to undirected (src < tgt) ---------------------------
    swap = src_int > tgt_int
    s_canon = np.where(swap, tgt_int, src_int)
    t_canon = np.where(swap, src_int, tgt_int)

    # ---- Symmetrize / mutual filter via group-by --------------------------
    # After canonicalization, each undirected pair (a, b) appears once if only
    # one of (a -> b), (b -> a) survived the threshold, and twice if both did.
    # The mutual flag picks between these regimes.
    pair_df = pl.DataFrame(
        {
            "_src": s_canon,
            "_tgt": t_canon,
            "_w": sim_kept,
        }
    )

    if mutual:
        pair_df = (
            pair_df.group_by(["_src", "_tgt"])
            .agg(
                [
                    pl.len().alias("_count"),
                    pl.col("_w").first().alias("_w"),
                ]
            )
            .filter(pl.col("_count") == 2)
            .drop("_count")
        )
    else:
        pair_df = pair_df.group_by(["_src", "_tgt"]).agg(
            pl.col("_w").first().alias("_w")
        )

    if pair_df.height == 0:
        if verbose:
            _print_filter_summary(
                n_senders, 0, 0, k, similarity_threshold, mutual
            )
        return pl.DataFrame(schema=empty_schema)

    # Count senders with at least one surviving edge. Done on the still-coded
    # pair_df (cheaper than the original-ID join below) — the mapping from
    # codes to original sender IDs is 1:1, so the unique count is the same.
    n_retained_senders = (
        pl.concat([pair_df["_src"], pair_df["_tgt"]]).n_unique()
    )
    n_output_edges = pair_df.height

    # ---- Map internal indices back to original sender IDs -----------------
    sender_lookup = pl.DataFrame(
        {"_idx": np.arange(n_senders, dtype=np.int64), "_sender": unique_senders}
    )
    pair_df = (
        pair_df.with_columns(pl.col("_src").cast(pl.Int64))
        .join(sender_lookup, left_on="_src", right_on="_idx")
        .rename({"_sender": source_col})
        .drop("_src")
        .with_columns(pl.col("_tgt").cast(pl.Int64))
        .join(sender_lookup, left_on="_tgt", right_on="_idx")
        .rename({"_sender": target_col})
        .drop("_tgt")
        .rename({"_w": weight_col})
        .select([source_col, target_col, weight_col])
    )

    if verbose:
        _print_filter_summary(
            n_senders,
            n_output_edges,
            n_retained_senders,
            k,
            similarity_threshold,
            mutual,
        )

    return pair_df
