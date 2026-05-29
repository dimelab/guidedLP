"""
Embedding-based content extraction from a ``[sender, post, datetime]`` post table.

Each post is mapped to a fixed-dimensional embedding vector (encoded on-the-fly
with a sentence-transformers model, or supplied via ``embedding_col``), then
mean-aggregated per sender. The per-sender vector is melted into a long-form
bipartite edge list mapping each sender to each dimension ("feature") of the
embedding space, with the aggregated value as the edge weight — ready to drop
into :func:`guidedLP.network.construction.build_graph_from_edgelist`.

The default pipeline preserves the similarity geometry of the embedding space:

1. L2-normalize each per-post embedding, so every component lies in ``[-1, 1]``.
2. Mean-aggregate across each sender's posts (componentwise mean of bounded
   values stays in ``[-1, 1]``).
3. Shift weights by ``+2`` so the final edge weights lie in ``[1, 3]`` — all
   positive, but the relative ordering and magnitudes of the embedding
   components are preserved exactly. Two senders that are componentwise
   aligned end up with high weights on the same features; two senders that
   are anti-aligned (one ``+1``, the other ``-1`` on the same dim) end up
   with a weight gap of ``2`` on that feature, correctly registering the
   dissimilarity in any downstream bipartite projection.

Set ``weight_transform="abs"`` if you want pure magnitude (loses sign
information — anti-aligned senders look the same as aligned ones), or
``weight_transform="raw"`` to keep signed values (downstream code must
handle negatives).

The from-scratch encoding path requires the optional ``[embeddings]`` extra:
``pip install 'guidedLP[embeddings]'`` (sentence-transformers + torch). The
pre-embedded path (passing ``embedding_col=``) has no extra dependency.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Tuple, Union

import numpy as np
import polars as pl

from guidedLP.common.exceptions import ValidationError


# Multilingual MiniLM: 384-dim, ~470MB, handles 50+ languages — the sensible
# default for the mixed-language corpora we typically see in this project.
_DEFAULT_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

_VALID_AGGREGATIONS = frozenset({"mean", "sum", "max"})
_VALID_WEIGHT_TRANSFORMS = frozenset({"shift", "abs", "raw"})

_EMBEDDINGS_INSTALL_HINT = (
    "Embedding from raw text requires the 'sentence-transformers' package. "
    "Install with: pip install 'guidedLP[embeddings]'"
)


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------


def _validate_text_input(
    df: pl.DataFrame,
    sender_col: str,
    post_col: str,
    datetime_col: Optional[str],
) -> None:
    """Check columns + post dtype for the from-scratch (raw text) path."""
    if not isinstance(df, pl.DataFrame):
        raise ValidationError(
            f"Expected polars.DataFrame, got {type(df).__name__}",
            expected="polars.DataFrame",
        )

    required = [sender_col, post_col]
    if datetime_col is not None:
        required.append(datetime_col)
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValidationError(
            f"Input DataFrame is missing required column(s) {missing}. "
            f"Available columns: {list(df.columns)}",
            details={"missing": missing, "available": list(df.columns)},
        )

    post_dtype = df.schema[post_col]
    if post_dtype != pl.Utf8:
        raise ValidationError(
            f"Post column '{post_col}' must be a string column, got {post_dtype}",
            field=post_col,
            expected="Utf8/String",
        )


def _validate_embedding_input(
    df: pl.DataFrame,
    sender_col: str,
    embedding_col: str,
    datetime_col: Optional[str],
) -> None:
    """Check columns + embedding dtype for the pre-embedded path."""
    if not isinstance(df, pl.DataFrame):
        raise ValidationError(
            f"Expected polars.DataFrame, got {type(df).__name__}",
            expected="polars.DataFrame",
        )

    required = [sender_col, embedding_col]
    if datetime_col is not None:
        required.append(datetime_col)
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValidationError(
            f"Input DataFrame is missing required column(s) {missing}. "
            f"Available columns: {list(df.columns)}",
            details={"missing": missing, "available": list(df.columns)},
        )

    dtype = df.schema[embedding_col]
    if not isinstance(dtype, (pl.List, pl.Array)):
        raise ValidationError(
            f"Embedding column '{embedding_col}' must be a List or Array of "
            f"numeric values, got {dtype}",
            field=embedding_col,
            expected="List[Float] / Array[Float]",
        )


# ---------------------------------------------------------------------------
# Embedding-acquisition helpers
# ---------------------------------------------------------------------------


def _encode_posts(
    posts: list,
    model: Union[str, Any],
    batch_size: int,
    show_progress: bool,
    device: Optional[str],
) -> np.ndarray:
    """
    Encode ``posts`` (list[str]) into an ``(N, D)`` float64 ndarray of raw
    (un-normalized) model outputs. L2 normalization is intentionally deferred
    to the main function so the on-disk cache stays normalize-agnostic — a
    caller can flip ``normalize_embeddings`` between runs without having to
    invalidate the cache.

    ``model`` may be either a model identifier string (in which case
    ``SentenceTransformer(model, device=device)`` is constructed — this branch
    requires the ``[embeddings]`` extra to be installed) or an already-loaded
    encoder object exposing an ``.encode`` method (in which case ``device`` is
    ignored — set it on the model itself before calling, and ``sentence-
    transformers`` does not need to be importable).
    """
    if isinstance(model, str):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:  # pragma: no cover
            raise ImportError(_EMBEDDINGS_INSTALL_HINT) from e
        encoder = SentenceTransformer(model, device=device)
    else:
        encoder = model

    raw = encoder.encode(
        posts,
        batch_size=batch_size,
        show_progress_bar=show_progress,
        convert_to_numpy=True,
        normalize_embeddings=False,
    )
    return np.asarray(raw, dtype=np.float64)


def _embeddings_from_column(
    df: pl.DataFrame,
    embedding_col: str,
) -> np.ndarray:
    """Materialize the embedding list/array column as a raw ``(N, D)`` ndarray.

    Normalization is applied in the main function, identically to the
    from-scratch path — see ``_encode_posts``.
    """
    rows = df.get_column(embedding_col).to_list()
    try:
        return np.vstack(rows).astype(np.float64)
    except (ValueError, TypeError) as e:
        raise ValidationError(
            f"All entries in '{embedding_col}' must be numeric vectors of the "
            "same length",
            field=embedding_col,
        ) from e


def _l2_normalize_rows(arr: np.ndarray) -> np.ndarray:
    """Return a copy of ``arr`` with each row scaled to unit L2 norm.

    All-zero rows are left as-is (the alternative — division by zero — would
    propagate NaNs through the aggregation).
    """
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms = np.where(norms == 0.0, 1.0, norms)
    return arr / norms


# ---------------------------------------------------------------------------
# On-disk cache for the from-scratch encoding path
# ---------------------------------------------------------------------------


def _cache_path(save_path: Union[str, Path]) -> Path:
    """Canonicalize a user-provided cache path to always end in ``.npy``.

    Lets callers pass ``"out/embeddings"`` or ``"out/embeddings.npy"``
    interchangeably; both end up at the same on-disk file.
    """
    p = Path(save_path)
    return p if p.suffix == ".npy" else p.with_name(p.name + ".npy")


def _save_embeddings_cache(path: Path, arr: np.ndarray) -> None:
    """Write ``arr`` to ``path`` (creating parent dirs if needed)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, arr)


def _load_embeddings_cache(path: Path, expected_n: int) -> np.ndarray:
    """Load a previously-saved embeddings matrix, validating its shape.

    The cache is row-aligned with the input post table (after null filtering).
    The only sanity check we do here is the row count — anything stronger
    (hashes, column comparison) would couple the cache to specific input
    columns. If the count doesn't match, the caller almost certainly fed in
    a different corpus and needs to regenerate via ``create_new=True``.
    """
    arr = np.load(path)
    if arr.ndim != 2:
        raise ValidationError(
            f"Cached embeddings at '{path}' must be 2-D, got shape {arr.shape}. "
            "Pass create_new=True to regenerate.",
            field="save_path",
            value=str(path),
        )
    if arr.shape[0] != expected_n:
        raise ValidationError(
            f"Cached embeddings at '{path}' have {arr.shape[0]} rows but the "
            f"input DataFrame has {expected_n} after null filtering. The cache "
            "is row-aligned with the input, so this almost certainly means a "
            "different corpus was passed. Pass create_new=True to regenerate.",
            field="save_path",
            value=str(path),
            details={
                "cached_rows": int(arr.shape[0]),
                "input_rows": int(expected_n),
            },
        )
    return arr.astype(np.float64, copy=False)


# ---------------------------------------------------------------------------
# Per-sender aggregation
# ---------------------------------------------------------------------------


def _aggregate_per_sender(
    senders: np.ndarray,
    embeddings: np.ndarray,
    aggregation: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Group rows of ``embeddings`` by ``senders`` and aggregate componentwise.

    Returns ``(unique_senders, aggregated)`` where ``aggregated`` has shape
    ``(n_unique, D)``. Done in numpy (``np.add.at`` / ``np.maximum.at``) rather
    than polars because materializing D columns just to group-mean them is
    wasteful when ``D`` is large (typical SBERT: 384 or 768).
    """
    unique, inverse = np.unique(senders, return_inverse=True)
    n_unique, dim = len(unique), embeddings.shape[1]

    if aggregation == "mean":
        counts = np.bincount(inverse, minlength=n_unique).astype(np.float64)
        summed = np.zeros((n_unique, dim), dtype=np.float64)
        np.add.at(summed, inverse, embeddings)
        return unique, summed / counts[:, None]

    if aggregation == "sum":
        summed = np.zeros((n_unique, dim), dtype=np.float64)
        np.add.at(summed, inverse, embeddings)
        return unique, summed

    if aggregation == "max":
        out = np.full((n_unique, dim), -np.inf, dtype=np.float64)
        np.maximum.at(out, inverse, embeddings)
        return unique, out

    raise ValidationError(
        f"aggregation must be one of {sorted(_VALID_AGGREGATIONS)}, "
        f"got {aggregation!r}",
        field="aggregation",
        value=aggregation,
    )


def _apply_weight_transform(
    weights: np.ndarray,
    transform: str,
    shift_amount: float,
) -> np.ndarray:
    """Apply the configured sign/shift transformation to the aggregated weights."""
    if transform == "shift":
        return weights + shift_amount
    if transform == "abs":
        return np.abs(weights)
    if transform == "raw":
        return weights
    raise ValidationError(
        f"weight_transform must be one of {sorted(_VALID_WEIGHT_TRANSFORMS)}, "
        f"got {transform!r}",
        field="weight_transform",
        value=transform,
    )


# ---------------------------------------------------------------------------
# Public extractor
# ---------------------------------------------------------------------------


def extract_embedding_features(
    df: pl.DataFrame,
    *,
    # Column names
    sender_col: str = "sender",
    post_col: str = "post",
    embedding_col: Optional[str] = None,
    datetime_col: Optional[str] = "datetime",
    feature_col: str = "feature",
    weight_col: str = "weight",
    first_seen_col: str = "first_seen",
    # Embedding model (used only when embedding_col is None)
    model: Union[str, Any] = _DEFAULT_MODEL,
    batch_size: int = 32,
    show_progress: bool = False,
    device: Optional[str] = None,
    normalize_embeddings: bool = True,
    # On-disk cache for the from-scratch path
    save_path: Optional[Union[str, Path]] = None,
    create_new: bool = False,
    # Aggregation per sender
    aggregation: str = "mean",
    # Weight transformation
    weight_transform: str = "shift",
    shift_amount: float = 2.0,
    # Sparsification
    top_k: Optional[int] = None,
    min_weight: Optional[float] = None,
    # Feature naming
    feature_prefix: str = "dim_",
) -> pl.DataFrame:
    """
    Embed posts and emit a sender→embedding-feature bipartite edge list.

    Each post is mapped to a fixed-dimensional embedding vector, those vectors
    are aggregated per sender (mean by default), and the resulting per-sender
    vector is unrolled into one edge per dimension. The output is a long-form
    DataFrame ready for :func:`guidedLP.network.construction.build_graph_from_edgelist`.

    Two paths into the function:

    - **From scratch.** Leave ``embedding_col=None`` and pass posts in
      ``post_col``. A sentence-transformers model is loaded (multilingual
      MiniLM by default) and the posts are encoded in batches. Requires
      ``pip install 'guidedLP[embeddings]'``.
    - **Pre-embedded.** Pass ``embedding_col=`` naming a column whose values
      are equal-length numeric lists/arrays (one embedding vector per post).
      No model is loaded; no optional dependency required.

    Parameters
    ----------
    df : pl.DataFrame
        Input post table.
    sender_col : str, default "sender"
        Author column. Values are passed through untouched.
    post_col : str, default "post"
        Post-text column. Used only when ``embedding_col`` is None. Must be Utf8.
    embedding_col : str, optional
        If set, this column is read as the per-post embedding vector and the
        encoding step is skipped. Must be a List or Array column of numeric
        values, with the same length on every row.
    datetime_col : str or None, default "datetime"
        Timestamp column to carry through as ``first_seen_col`` (the per-sender
        earliest timestamp). Pass ``None`` to omit it entirely.
    feature_col, weight_col, first_seen_col : str
        Output column names.
    model : str or SentenceTransformer, default multilingual MiniLM
        Either a sentence-transformers model identifier (loaded on first call)
        or an already-loaded ``SentenceTransformer`` instance. Reusing a loaded
        model across calls avoids the per-call startup cost.
    batch_size : int, default 32
        Per-batch size handed to ``encoder.encode``. Bigger batches are faster
        on GPU but use more memory.
    show_progress : bool, default False
        Forwarded to sentence-transformers' ``show_progress_bar``.
    device : str, optional
        Torch device string (e.g. ``"cuda"``, ``"cpu"``, ``"mps"``). Only
        consulted when ``model`` is passed as a string. ``None`` lets
        sentence-transformers auto-select.
    normalize_embeddings : bool, default True
        L2-normalize every per-post vector before aggregation. Applied
        identically in both paths (and identically whether the embeddings came
        from the model or from disk cache), so the on-disk cache stays
        normalize-agnostic — flipping this flag between runs does **not**
        require ``create_new=True``. Setting this False with
        ``weight_transform="shift"`` (default) is a configuration error in
        spirit — the shift amount assumes bounded ``[-1, 1]`` components,
        which is only true for L2-normalized vectors.
    save_path : str or pathlib.Path, optional
        If set, the per-post raw embedding matrix is cached on disk at this
        path (as ``.npy``; the ``.npy`` suffix is appended if missing). On
        subsequent calls with ``create_new=False`` (default) and the file
        present, encoding is skipped and the cached matrix is loaded — a
        large win when iterating on aggregation / sparsification / transform
        choices, since model inference is by far the most expensive step.
        Only meaningful for the from-scratch path: passing ``save_path``
        together with ``embedding_col`` raises ``ValidationError``. The cache
        is row-aligned with the post table (after null filtering), so feeding
        in a different corpus must be paired with ``create_new=True``.
    create_new : bool, default False
        Force re-encoding even if ``save_path`` points at an existing cache,
        and overwrite the file with the freshly-encoded matrix. Has no effect
        when ``save_path`` is ``None``.
    aggregation : {"mean", "sum", "max"}, default "mean"
        How to collapse a sender's per-post vectors into a single vector.
        ``"mean"`` is the safest with the default shift (the mean of
        ``[-1, 1]`` components is also in ``[-1, 1]``); ``"sum"`` and ``"max"``
        can land outside the assumed range and may interact poorly with the
        default shift amount.
    weight_transform : {"shift", "abs", "raw"}, default "shift"
        How to convert the aggregated (potentially negative) values into edge
        weights:

        - ``"shift"`` — ``weight = value + shift_amount``. With the defaults
          (L2-normalized + ``shift_amount=2.0``), output weights lie in
          ``[1, 3]`` and preserve the embedding geometry exactly.
        - ``"abs"`` — ``weight = |value|``. Drops sign — anti-aligned senders
          look identical to aligned ones.
        - ``"raw"`` — ``weight = value``. Keeps negatives; downstream code
          must handle them.
    shift_amount : float, default 2.0
        Constant added when ``weight_transform="shift"``. Ignored otherwise.
    top_k : int, optional
        If set, keep only each sender's top-``k`` features by ``|weight|``.
        Default ``None`` means keep all dimensions.
    min_weight : float, optional
        If set, drop edges whose ``|weight|`` falls below this threshold.
    feature_prefix : str, default "dim_"
        Prefix used to name the feature/target nodes: ``dim_0``, ``dim_1``, …

    Returns
    -------
    pl.DataFrame
        Long-form edge list with columns
        ``[sender_col, feature_col, weight_col(, first_seen_col)]``, one row
        per (sender, dimension) pair after any sparsification.

    Raises
    ------
    ValidationError
        Missing columns, wrong dtype, invalid mode arguments, or ragged
        per-row embedding lengths in the pre-embedded path.
    ImportError
        From-scratch path attempted without ``sentence-transformers`` installed.

    Complexity
    ----------
    - From-scratch path: dominated by model inference, O(N_posts) at the
      model's per-post cost.
    - Aggregation: O(N_posts · D) for the numpy reduce.
    - Output: ``N_senders × D`` rows before sparsification.

    Examples
    --------
    Pre-embedded path (no model load):

    >>> import polars as pl
    >>> df = pl.DataFrame({
    ...     "sender":    ["alice", "alice", "bob"],
    ...     "embedding": [[1.0, 0.0, -1.0], [0.5, 0.5, -0.5], [-1.0, 1.0, 0.0]],
    ...     "datetime":  ["2024-01-01", "2024-01-02", "2024-01-03"],
    ... })
    >>> out = extract_embedding_features(
    ...     df, embedding_col="embedding", normalize_embeddings=False
    ... )
    >>> sorted(out.columns)
    ['feature', 'first_seen', 'sender', 'weight']

    From-scratch path (requires ``[embeddings]`` extra):

    >>> # df has columns sender, post, datetime
    >>> # edges = extract_embedding_features(df)               # doctest: +SKIP
    >>> # graph, mapper = build_graph_from_edgelist(            # doctest: +SKIP
    >>> #     edges, source_col="sender", target_col="feature",
    >>> #     weight_col="weight", bipartite=True,
    >>> # )
    """
    # ---- Validate + pick the embeddings source ----------------------------
    if embedding_col is not None:
        if save_path is not None:
            raise ValidationError(
                "save_path is only meaningful for the from-scratch encoding "
                "path; when embedding_col is provided no encoding step runs, "
                "so there is nothing to cache.",
                field="save_path",
            )
        _validate_embedding_input(df, sender_col, embedding_col, datetime_col)
        df_work = df.drop_nulls(subset=[sender_col, embedding_col])
    else:
        _validate_text_input(df, sender_col, post_col, datetime_col)
        df_work = df.drop_nulls(subset=[sender_col, post_col])

    if top_k is not None and top_k <= 0:
        raise ValidationError(
            "top_k must be a positive integer",
            field="top_k",
            value=top_k,
        )

    # ---- Empty input: short-circuit before loading a model ----------------
    if df_work.height == 0:
        empty_schema: dict = {
            sender_col: df.schema[sender_col],
            feature_col: pl.Utf8,
            weight_col: pl.Float64,
        }
        if datetime_col is not None:
            empty_schema[first_seen_col] = df.schema[datetime_col]
        return pl.DataFrame(schema=empty_schema)

    # ---- Acquire embeddings (raw, pre-normalization) ----------------------
    if embedding_col is not None:
        embeddings = _embeddings_from_column(df_work, embedding_col)
    else:
        cache_p = _cache_path(save_path) if save_path is not None else None
        if cache_p is not None and not create_new and cache_p.exists():
            embeddings = _load_embeddings_cache(cache_p, df_work.height)
        else:
            posts = df_work.get_column(post_col).to_list()
            embeddings = _encode_posts(
                posts,
                model=model,
                batch_size=batch_size,
                show_progress=show_progress,
                device=device,
            )
            if cache_p is not None:
                _save_embeddings_cache(cache_p, embeddings)

    if normalize_embeddings:
        embeddings = _l2_normalize_rows(embeddings)

    # ---- Aggregate per sender ---------------------------------------------
    senders = df_work.get_column(sender_col).to_numpy()
    unique_senders, aggregated = _aggregate_per_sender(
        senders, embeddings, aggregation
    )
    aggregated = _apply_weight_transform(
        aggregated, weight_transform, shift_amount
    )

    # ---- Build long-form directly (avoid melt cost on wide schemas) -------
    n, dim = aggregated.shape
    feature_names = np.array(
        [f"{feature_prefix}{i}" for i in range(dim)], dtype=object
    )

    long = pl.DataFrame(
        {
            sender_col: np.repeat(unique_senders, dim),
            feature_col: np.tile(feature_names, n),
            weight_col: aggregated.reshape(-1),
        }
    )

    # ---- Carry first-seen timestamp through --------------------------------
    if datetime_col is not None:
        first_seen = df_work.group_by(sender_col, maintain_order=False).agg(
            pl.col(datetime_col).min().alias(first_seen_col)
        )
        long = long.join(first_seen, on=sender_col, how="left")

    # ---- Sparsification ---------------------------------------------------
    if min_weight is not None:
        long = long.filter(pl.col(weight_col).abs() >= float(min_weight))

    if top_k is not None:
        long = (
            long.with_columns(pl.col(weight_col).abs().alias("_glp_abs_w"))
            .sort([sender_col, "_glp_abs_w"], descending=[False, True])
            .group_by(sender_col, maintain_order=True)
            .head(top_k)
            .drop("_glp_abs_w")
        )

    return long
