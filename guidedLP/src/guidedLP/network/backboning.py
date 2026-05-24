"""
Network backboning module for the Guided Label Propagation library.

This module implements network backbone extraction methods that identify the
most significant edges in weighted (or unweighted) networks. The single public
entry point is :func:`apply_backbone`, which dispatches on a ``method`` keyword
to one of:

- ``"disparity"`` — Serrano et al. (2009) disparity filter, fully vectorized.
- ``"noise_corrected"`` — Coscia & Neffke (2017) Bayesian noise-corrected
  backbone, vectorized in Polars.
- ``"bipartite_svn"`` — Tumminello et al. (2011) Statistically Validated
  Network filter for bipartite graphs, with Benjamini-Hochberg FDR correction
  and an optional node-retention post-filter.
- ``"weight"`` — simple weight threshold (with median fallback or explicit
  threshold / ``target_edges`` cap).
- ``"degree"`` — keep highest-degree nodes (median fallback or ``target_nodes``).
"""

from typing import Tuple, Optional, Dict, Any, Union, List
import time as _time
import warnings

import numpy as np
import networkit as nk
import polars as pl
from scipy.stats import poisson

from guidedLP.common.id_mapper import IDMapper
from guidedLP.common.exceptions import (
    ComputationError,
    ConfigurationError,
    ValidationError,
)
from guidedLP.common.logging_config import get_logger, log_function_entry, LoggingTimer

logger = get_logger(__name__)

# Canonical list of supported methods.
AVAILABLE_BACKBONE_METHODS = [
    "disparity",
    "noise_corrected",
    "bipartite_svn",
    "weight",
    "degree",
]
# Legacy alias retained for tests/old call sites.
AVAILABLE_METHODS = AVAILABLE_BACKBONE_METHODS


def apply_backbone(
    graph: nk.Graph,
    id_mapper: IDMapper,
    method: str = "disparity",
    *,
    alpha: float = 0.05,
    threshold: float = 1.0,
    target_nodes: Optional[int] = None,
    target_edges: Optional[int] = None,
    weight_threshold: Optional[float] = None,
    keep_disconnected: bool = False,
    correction: str = "fdr_bh",
    min_node_retention: Optional[float] = None,
    return_filtered_edges: bool = False,
    verbose: bool = True,
) -> Union[Tuple[nk.Graph, IDMapper], Tuple[nk.Graph, IDMapper, pl.DataFrame]]:
    """
    Extract a network backbone by filtering edges with one of several methods.

    Parameters
    ----------
    graph : nk.Graph
        NetworkIt graph to filter. Weighted graphs are required for
        ``method="disparity"`` and ``method="noise_corrected"``.
    id_mapper : IDMapper
        Bidirectional mapping between original and internal node IDs.
    method : str, default "disparity"
        Backbone extraction method. One of
        ``{"disparity", "noise_corrected", "bipartite_svn", "weight", "degree"}``.

        - ``"disparity"``: Serrano et al. disparity filter
          ``α_ij = (1 − p_ij)^(k−1)``. Edges with ``α < alpha`` are kept.
        - ``"noise_corrected"``: Coscia & Neffke (2017) Bayesian noise-corrected
          backbone. Compares observed edge weight to the configuration-model
          expectation and keeps edges whose lift exceeds ``threshold`` posterior
          standard deviations.
        - ``"bipartite_svn"``: Tumminello et al. (2011) SVN filter for
          bipartite graphs. Per-edge p-values under a Poisson configuration
          null with multiple-testing correction (``correction``) and an
          optional node-retention post-filter (``min_node_retention``).
        - ``"weight"``: keep edges with weight ≥ threshold. The threshold is
          (in priority order) ``weight_threshold`` if given, else the value
          at rank ``target_edges`` if given, else the median weight.
        - ``"degree"``: keep nodes with degree ≥ threshold. The threshold is
          (in priority order) the value at rank ``target_nodes`` if given,
          else the median degree.
    alpha : float, default 0.05
        Significance level for ``method="disparity"`` (disparity threshold)
        and ``method="bipartite_svn"`` (per-edge p-value cutoff). Must be in
        ``(0, 1)``.
    threshold : float, default 1.0
        Standard-deviation multiplier for ``method="noise_corrected"``. An
        edge is kept iff ``score − threshold · sdev_cij > 0`` — i.e. the
        observed weight exceeds the null expectation by at least ``threshold``
        posterior standard deviations. Must be ``> 0``.
    target_nodes : int, optional
        For ``method="degree"``: keep approximately this many top-degree nodes.
        Conflicts with ``target_edges``.
    target_edges : int, optional
        For ``method="weight"``: keep approximately this many highest-weight
        edges. For ``method="disparity"``: cap the kept set to this many
        edges (lowest α first). Conflicts with ``target_nodes``.
    weight_threshold : float, optional
        For ``method="weight"``: explicit weight cutoff. Takes precedence over
        ``target_edges`` if both are set.
    keep_disconnected : bool, default False
        If True, keep isolated nodes that resulted from edge filtering. If
        False (default), drop them and renumber internal IDs.
    correction : str, default "fdr_bh"
        Multiple-testing correction for ``method="bipartite_svn"``. One of:

        - ``"fdr_bh"`` (recommended): Benjamini-Hochberg false-discovery rate.
          Scales gracefully to millions of edges.
        - ``"bonferroni"``: per-edge cutoff ``alpha / |E|``. Extremely
          conservative on large graphs.
        - ``"none"``: use ``alpha`` directly. Most permissive.
    min_node_retention : float, optional
        Only used by ``method="bipartite_svn"``. When set to a value in
        ``(0, 1]``, applied as a *post-filter* after per-edge SVN: any node
        whose retention ratio ``surviving_edges / original_edges`` is below
        this threshold is removed entirely (along with its remaining edges).
        Use this to eliminate generic high-degree nodes whose edges are
        mostly noise.
    return_filtered_edges : bool, default False
        If True, return a third element: a Polars DataFrame with per-edge
        filtering details. Schema depends on ``method``:

        - all methods: ``source_id``, ``target_id``, ``weight``, ``kept``
        - ``disparity``: also ``p_value``, ``alpha_score``
        - ``noise_corrected``: also ``score``, ``sdev_cij``
        - ``bipartite_svn``: also ``p_value``
        - ``weight`` / ``degree``: ``p_value`` and ``alpha_score`` are NaN
    verbose : bool, default True
        Print a one-line summary at the end (timing + node/edge retention).

    Returns
    -------
    Tuple[nk.Graph, IDMapper]
        The backbone graph and the updated ID mapper.
    Tuple[nk.Graph, IDMapper, pl.DataFrame]
        If ``return_filtered_edges=True``, additionally a per-edge results
        DataFrame.

    Raises
    ------
    ValidationError
        If ``method`` is unknown, parameters are out of range, or required
        graph properties are missing (e.g. weighted graph for disparity).
    ConfigurationError
        If ``method="degree"`` is used without ``target_nodes`` and no
        sensible default applies, or required parameters are missing.
    ComputationError
        If extraction fails or produces an empty graph.

    Notes
    -----
    Time complexity:
        - ``disparity``: O(E) (fully vectorized in NumPy).
        - ``noise_corrected``: O(E) (vectorized in Polars).
        - ``bipartite_svn``: O(E + E log E) with BH correction.
        - ``weight``: O(E) (O(E log E) when sorting for ``target_edges``).
        - ``degree``: O(V + E).

    Examples
    --------
    >>> backbone, mapper = apply_backbone(g, m, method="disparity", alpha=0.05)
    >>> backbone, mapper = apply_backbone(g, m, method="noise_corrected", threshold=1.0)
    >>> backbone, mapper = apply_backbone(g, m, method="bipartite_svn", alpha=0.01)
    >>> backbone, mapper, edges = apply_backbone(
    ...     g, m, method="disparity", alpha=0.05, return_filtered_edges=True
    ... )

    References
    ----------
    Serrano, M. Á., Boguñá, M., & Vespignani, A. (2009). Extracting the
    multiscale backbone of complex weighted networks. PNAS, 106(16), 6483–6488.

    Coscia, M., & Neffke, F. M. H. (2017). Network backboning with noisy data.
    ICDE 2017.

    Tumminello, M., Miccichè, S., Lillo, F., Piilo, J., & Mantegna, R. N.
    (2011). Statistically validated networks in bipartite complex systems.
    PLoS ONE, 6(3), e17994.
    """
    log_function_entry(
        "apply_backbone",
        graph_nodes=graph.numberOfNodes(),
        graph_edges=graph.numberOfEdges(),
        method=method,
        alpha=alpha,
        target_nodes=target_nodes,
        target_edges=target_edges,
    )

    _t_start = _time.perf_counter()

    _validate_backbone_parameters(
        method=method,
        alpha=alpha,
        threshold=threshold,
        target_nodes=target_nodes,
        target_edges=target_edges,
        weight_threshold=weight_threshold,
        correction=correction,
        min_node_retention=min_node_retention,
        graph=graph,
    )

    # Empty-graph short-circuit.
    if graph.numberOfNodes() == 0 or graph.numberOfEdges() == 0:
        warnings.warn("Empty graph provided. Returning empty graph.")
        empty_graph = nk.Graph(
            graph.numberOfNodes(),
            weighted=graph.isWeighted(),
            directed=graph.isDirected(),
        )
        empty_mapper = id_mapper if graph.numberOfNodes() > 0 else IDMapper()
        _print_backbone_summary(verbose, _t_start, graph, empty_graph, method)
        if return_filtered_edges:
            return empty_graph, empty_mapper, _empty_edges_df(method)
        return empty_graph, empty_mapper

    with LoggingTimer("apply_backbone", {
        "method": method,
        "nodes": graph.numberOfNodes(),
        "edges": graph.numberOfEdges(),
    }):
        try:
            if method == "disparity":
                result = _apply_disparity_filter(
                    graph, id_mapper, alpha, target_edges,
                    keep_disconnected, return_filtered_edges,
                )
            elif method == "noise_corrected":
                result = _apply_noise_corrected_filter(
                    graph, id_mapper, threshold,
                    keep_disconnected, return_filtered_edges,
                )
            elif method == "bipartite_svn":
                result = _apply_bipartite_svn_filter(
                    graph, id_mapper, alpha, correction,
                    keep_disconnected, min_node_retention,
                    return_filtered_edges,
                )
            elif method == "weight":
                result = _apply_weight_threshold(
                    graph, id_mapper, target_edges, weight_threshold,
                    keep_disconnected, return_filtered_edges,
                )
            elif method == "degree":
                result = _apply_degree_threshold(
                    graph, id_mapper, target_nodes,
                    keep_disconnected, return_filtered_edges,
                )
            else:
                # Defensive: _validate_backbone_parameters already guards this.
                raise ValidationError(f"Unsupported backbone method: {method}")

            if return_filtered_edges:
                backbone_graph, _, _ = result
            else:
                backbone_graph, _ = result

            logger.info(
                "Backbone extraction completed: %d → %d edges, %d → %d nodes",
                graph.numberOfEdges(), backbone_graph.numberOfEdges(),
                graph.numberOfNodes(), backbone_graph.numberOfNodes(),
            )

            _print_backbone_summary(verbose, _t_start, graph, backbone_graph, method)
            return result

        except (ConfigurationError, ComputationError, ValidationError):
            raise
        except Exception as e:
            raise ComputationError(
                f"Backbone extraction failed: {e}",
                context={
                    "operation": "apply_backbone",
                    "method": method,
                    "error_type": "computation",
                },
            ) from e


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def _validate_backbone_parameters(
    *,
    method: str,
    alpha: float,
    threshold: float,
    target_nodes: Optional[int],
    target_edges: Optional[int],
    weight_threshold: Optional[float],
    correction: str,
    min_node_retention: Optional[float],
    graph: nk.Graph,
) -> None:
    """Validate the cross-method parameter surface of :func:`apply_backbone`."""
    if method not in AVAILABLE_BACKBONE_METHODS:
        raise ValidationError(
            f"Invalid backbone method: {method}. "
            f"Available methods: {AVAILABLE_BACKBONE_METHODS}"
        )

    if target_nodes is not None and target_edges is not None:
        raise ValidationError("Cannot specify both target_nodes and target_edges")

    if target_nodes is not None and target_nodes <= 0:
        raise ValidationError(f"target_nodes must be positive, got {target_nodes}")
    if target_edges is not None and target_edges <= 0:
        raise ValidationError(f"target_edges must be positive, got {target_edges}")
    if weight_threshold is not None and weight_threshold <= 0:
        raise ValidationError(
            f"weight_threshold must be positive, got {weight_threshold}"
        )

    if method in ("disparity", "bipartite_svn"):
        if not (0.0 < alpha < 1.0):
            raise ValidationError(
                f"alpha must be between 0 and 1, got {alpha}"
            )

    if method == "noise_corrected":
        if threshold <= 0:
            raise ValidationError(
                f"threshold must be > 0 for noise_corrected, got {threshold}"
            )

    if method == "disparity" and not graph.isWeighted():
        raise ValidationError("Disparity filter requires a weighted graph")

    if method == "bipartite_svn":
        if correction not in ("fdr_bh", "bonferroni", "none"):
            raise ValidationError(
                f"Invalid correction value: {correction!r}. "
                f"Expected 'fdr_bh', 'bonferroni', or 'none'."
            )
        if (
            min_node_retention is not None
            and not (0.0 < min_node_retention <= 1.0)
        ):
            raise ValidationError(
                f"min_node_retention must be in (0.0, 1.0] or None, "
                f"got {min_node_retention!r}."
            )


# ---------------------------------------------------------------------------
# Disparity filter (Serrano et al. 2009)
# ---------------------------------------------------------------------------

def _apply_disparity_filter(
    graph: nk.Graph,
    id_mapper: IDMapper,
    alpha: float,
    target_edges: Optional[int],
    keep_disconnected: bool,
    return_filtered_edges: bool,
) -> Union[Tuple[nk.Graph, IDMapper], Tuple[nk.Graph, IDMapper, pl.DataFrame]]:
    """Apply Serrano et al.'s disparity filter, fully vectorized over edges."""
    logger.debug("Applying disparity filter with alpha=%.3f", alpha)

    n_nodes = graph.numberOfNodes()
    directed = graph.isDirected()

    sources_l: list = []
    targets_l: list = []
    weights_l: list = []
    for u, v in graph.iterEdges():
        sources_l.append(u)
        targets_l.append(v)
        weights_l.append(graph.weight(u, v))

    if not sources_l:
        empty = nk.Graph(n_nodes, weighted=True, directed=directed)
        if return_filtered_edges:
            return empty, id_mapper, _empty_edges_df("disparity")
        return empty, id_mapper

    sources = np.asarray(sources_l, dtype=np.int64)
    targets = np.asarray(targets_l, dtype=np.int64)
    weights = np.asarray(weights_l, dtype=np.float64)

    # Strengths and degree counts in O(E) each.
    if directed:
        strength = np.bincount(sources, weights=weights, minlength=n_nodes)
        degree_counts = (
            np.bincount(sources, minlength=n_nodes)
            + np.bincount(targets, minlength=n_nodes)
        )
    else:
        strength = (
            np.bincount(sources, weights=weights, minlength=n_nodes)
            + np.bincount(targets, weights=weights, minlength=n_nodes)
        )
        degree_counts = (
            np.bincount(sources, minlength=n_nodes)
            + np.bincount(targets, minlength=n_nodes)
        )

    s_u = strength[sources]
    s_v = strength[targets]
    k_u = degree_counts[sources].astype(np.float64)
    k_v = degree_counts[targets].astype(np.float64)

    s_u_safe = np.where(s_u > 0, s_u, 1.0)
    p_uv = weights / s_u_safe
    alpha_uv = _disparity_alpha(1.0 - p_uv, k_u - 1.0)

    if directed:
        final_alpha = alpha_uv
        p_value_col = p_uv
    else:
        s_v_safe = np.where(s_v > 0, s_v, 1.0)
        p_vu = weights / s_v_safe
        alpha_vu = _disparity_alpha(1.0 - p_vu, k_v - 1.0)
        final_alpha = np.minimum(alpha_uv, alpha_vu)
        p_value_col = np.minimum(p_uv, p_vu)

    keep_mask = (final_alpha < alpha) | (final_alpha >= 1.0)

    invalid_mask = (s_u == 0) | (s_v == 0)
    if np.any(invalid_mask):
        final_alpha = np.where(invalid_mask, 1.0, final_alpha)
        p_value_col = np.where(invalid_mask, 1.0, p_value_col)
        keep_mask = keep_mask & ~invalid_mask

    # Optional target_edges cap: keep the lowest-α edges (Serrano et al.'s
    # natural ordering of statistical significance).
    if target_edges is not None and int(keep_mask.sum()) > target_edges:
        sorted_idx = np.argsort(final_alpha)
        new_mask = np.zeros_like(keep_mask)
        new_mask[sorted_idx[:target_edges]] = True
        keep_mask = new_mask & ~invalid_mask

    kept_sources = sources[keep_mask]
    kept_targets = targets[keep_mask]
    kept_weights = weights[keep_mask]

    backbone_graph, updated_mapper = _assemble_backbone(
        n_nodes, directed, weighted=True,
        kept_sources=kept_sources, kept_targets=kept_targets,
        kept_weights=kept_weights,
        keep_disconnected=keep_disconnected, id_mapper=id_mapper,
    )

    if return_filtered_edges:
        edge_df = pl.DataFrame({
            "source_id": id_mapper.get_original_batch(sources.tolist()),
            "target_id": id_mapper.get_original_batch(targets.tolist()),
            "weight": weights,
            "p_value": p_value_col,
            "alpha_score": final_alpha,
            "kept": keep_mask,
        })
        return backbone_graph, updated_mapper, edge_df

    return backbone_graph, updated_mapper


def _disparity_alpha(base: np.ndarray, exponent: np.ndarray) -> np.ndarray:
    """Vectorized ``(1 − p)^(k − 1)`` matching :func:`_safe_power` edge cases.

    Elementwise, in precedence order:
    - exponent <= 0 -> 1.0 (leaf-node sentinel where k <= 1)
    - base <= 0     -> 0.0 (p = 1, the edge carries all of u's strength)
    - base >= 1     -> 1.0 (p = 0, zero-weight edge)
    - otherwise     -> exp(exponent * log(base)), clamped against underflow
    """
    result = np.ones_like(base, dtype=np.float64)
    formula_applies = exponent > 0
    compute_mask = formula_applies & (base > 0) & (base < 1)
    if np.any(compute_mask):
        log_result = exponent[compute_mask] * np.log(base[compute_mask])
        np.maximum(log_result, -700.0, out=log_result)
        result[compute_mask] = np.exp(log_result)
    result[formula_applies & (base <= 0)] = 0.0
    return result


def _safe_power(base: float, exponent: float) -> float:
    """Numerically stable power calculation for the disparity filter.

    Retained as a scalar helper for tests that exercise the original edge
    cases (very small base, large exponent, invalid inputs).
    """
    if base <= 0:
        return 0.0
    if base >= 1:
        return 1.0
    if exponent <= 0:
        return 1.0
    try:
        log_result = exponent * np.log(base)
        if log_result < -700:
            return 0.0
        return np.exp(log_result)
    except (OverflowError, ValueError):
        return 0.0


# ---------------------------------------------------------------------------
# Noise-corrected backbone (Coscia & Neffke 2017)
# ---------------------------------------------------------------------------

def _apply_noise_corrected_filter(
    graph: nk.Graph,
    id_mapper: IDMapper,
    threshold: float,
    keep_disconnected: bool,
    return_filtered_edges: bool,
) -> Union[Tuple[nk.Graph, IDMapper], Tuple[nk.Graph, IDMapper, pl.DataFrame]]:
    """Apply the Coscia & Neffke (2017) noise-corrected backbone in Polars."""
    logger.debug("Applying noise-corrected backbone with threshold=%.3f", threshold)

    n_nodes = graph.numberOfNodes()
    directed = graph.isDirected()

    sources_l: list = []
    targets_l: list = []
    weights_l: list = []
    for u, v in graph.iterEdges():
        sources_l.append(u)
        targets_l.append(v)
        weights_l.append(graph.weight(u, v))

    if not sources_l:
        empty = nk.Graph(n_nodes, weighted=True, directed=directed)
        if return_filtered_edges:
            return empty, id_mapper, _empty_edges_df("noise_corrected")
        return empty, id_mapper

    sources = np.asarray(sources_l, dtype=np.int64)
    targets = np.asarray(targets_l, dtype=np.int64)
    weights = np.asarray(weights_l, dtype=np.float64)

    # Symmetrize for undirected graphs so group_by sums give full node strengths.
    if directed:
        df = pl.DataFrame({"o": sources, "e": targets, "w": weights})
    else:
        df = pl.DataFrame({
            "o": np.concatenate([sources, targets]),
            "e": np.concatenate([targets, sources]),
            "w": np.concatenate([weights, weights]),
        })

    n_total = float(df["w"].sum())
    if n_total <= 0:
        raise ComputationError(
            "Noise-corrected backbone requires positive total edge weight",
            context={"operation": "noise_corrected"},
        )

    src_sum = df.group_by("o").agg(pl.col("w").sum().alias("o_sum"))
    trg_sum = df.group_by("e").agg(pl.col("w").sum().alias("e_sum"))
    df = df.join(src_sum, on="o", how="left").join(trg_sum, on="e", how="left")

    df = df.with_columns([
        ((pl.col("o_sum") * pl.col("e_sum")) / (n_total * n_total))
            .alias("mean_prior_probability"),
        (n_total / (pl.col("o_sum") * pl.col("e_sum"))).alias("kappa"),
    ])
    df = df.with_columns(
        ((pl.col("kappa") * pl.col("w") - 1) / (pl.col("kappa") * pl.col("w") + 1))
            .alias("score")
    )
    df = df.with_columns(
        (
            (pl.col("o_sum") * pl.col("e_sum")
             * (n_total - pl.col("o_sum")) * (n_total - pl.col("e_sum")))
            / ((n_total ** 4) * (n_total - 1))
        ).alias("var_prior_probability")
    )
    df = df.with_columns([
        (
            (pl.col("mean_prior_probability") ** 2 / pl.col("var_prior_probability"))
            * (1 - pl.col("mean_prior_probability"))
            - pl.col("mean_prior_probability")
        ).alias("alpha_prior"),
        (
            (pl.col("mean_prior_probability") / pl.col("var_prior_probability"))
            * (1 - pl.col("mean_prior_probability") ** 2)
            - (1 - pl.col("mean_prior_probability"))
        ).alias("beta_prior"),
    ])
    df = df.with_columns([
        (pl.col("alpha_prior") + pl.col("w")).alias("alpha_post"),
        (n_total - pl.col("w") + pl.col("beta_prior")).alias("beta_post"),
    ])
    df = df.with_columns(
        (pl.col("alpha_post") / (pl.col("alpha_post") + pl.col("beta_post")))
            .alias("expected_pij")
    )
    df = df.with_columns(
        (pl.col("expected_pij") * (1 - pl.col("expected_pij")) * n_total)
            .alias("variance_nij")
    )
    df = df.with_columns(
        (
            (1.0 / (pl.col("o_sum") * pl.col("e_sum")))
            - (n_total * (pl.col("o_sum") + pl.col("e_sum"))
               / ((pl.col("o_sum") * pl.col("e_sum")) ** 2))
        ).alias("d")
    )
    df = df.with_columns(
        (
            pl.col("variance_nij")
            * ((2 * (pl.col("kappa") + pl.col("w") * pl.col("d")))
               / ((pl.col("kappa") * pl.col("w") + 1) ** 2)) ** 2
        ).alias("variance_cij")
    )
    df = df.with_columns(
        pl.col("variance_cij").clip(lower_bound=0.0).sqrt().alias("sdev_cij")
    )

    if not directed:
        df = df.filter(pl.col("o") <= pl.col("e"))

    df = df.with_columns(
        ((pl.col("score") - threshold * pl.col("sdev_cij")) > 0).alias("kept")
    )

    kept_df = df.filter(pl.col("kept"))
    kept_sources = kept_df["o"].to_numpy()
    kept_targets = kept_df["e"].to_numpy()
    kept_weights = kept_df["w"].to_numpy()

    backbone_graph, updated_mapper = _assemble_backbone(
        n_nodes, directed, weighted=True,
        kept_sources=kept_sources, kept_targets=kept_targets,
        kept_weights=kept_weights,
        keep_disconnected=keep_disconnected, id_mapper=id_mapper,
    )

    if return_filtered_edges:
        internal_sources = df["o"].to_list()
        internal_targets = df["e"].to_list()
        edge_df = pl.DataFrame({
            "source_id": id_mapper.get_original_batch(internal_sources),
            "target_id": id_mapper.get_original_batch(internal_targets),
            "weight": df["w"],
            "score": df["score"],
            "sdev_cij": df["sdev_cij"],
            "kept": df["kept"],
        })
        return backbone_graph, updated_mapper, edge_df

    return backbone_graph, updated_mapper


# ---------------------------------------------------------------------------
# Bipartite SVN filter (Tumminello et al. 2011)
# ---------------------------------------------------------------------------

def _apply_bipartite_svn_filter(
    graph: nk.Graph,
    id_mapper: IDMapper,
    alpha: float,
    correction: str,
    keep_disconnected: bool,
    min_node_retention: Optional[float],
    return_filtered_edges: bool,
) -> Union[Tuple[nk.Graph, IDMapper], Tuple[nk.Graph, IDMapper, pl.DataFrame]]:
    """Tumminello et al. (2011) Statistically Validated Network filter.

    Per-edge p-value under a Poisson configuration null with optional
    multiple-testing correction and node-retention post-filter.
    """
    n_nodes = graph.numberOfNodes()
    if graph.numberOfEdges() == 0:
        if return_filtered_edges:
            return graph, id_mapper, _empty_edges_df("bipartite_svn")
        return graph, id_mapper

    edges_list = list(graph.iterEdges())
    n_edges = len(edges_list)
    u_arr = np.empty(n_edges, dtype=np.int64)
    v_arr = np.empty(n_edges, dtype=np.int64)
    w_arr = np.empty(n_edges, dtype=np.float64)
    for i, (u, v) in enumerate(edges_list):
        u_arr[i] = u
        v_arr[i] = v
        w_arr[i] = graph.weight(u, v) if graph.isWeighted() else 1.0

    is_weighted_effective = graph.isWeighted() and not np.allclose(w_arr, 1.0)

    strengths = np.zeros(n_nodes, dtype=np.float64)
    np.add.at(strengths, u_arr, w_arr)
    np.add.at(strengths, v_arr, w_arr)

    W_total = w_arr.sum()
    if W_total <= 0:
        logger.warning("Bipartite SVN filter received zero total weight; returning empty graph.")
        empty = _build_empty_like(graph)
        if return_filtered_edges:
            return empty, IDMapper(), _empty_edges_df("bipartite_svn")
        return empty, IDMapper()

    # Bipartite configuration-model null: μ = s_u · s_v / W_total (the
    # partition-wise strength sums are both W_total for a bipartite graph,
    # so the standard "2·W_total" unipartite normalisation does not apply).
    mu = strengths[u_arr] * strengths[v_arr] / W_total

    if is_weighted_effective:
        p_values = poisson.sf(np.ceil(w_arr) - 1, mu)
    else:
        p_values = -np.expm1(-mu)

    if correction == "bonferroni":
        keep_mask = p_values <= (alpha / n_edges)
    elif correction == "fdr_bh":
        keep_mask = _benjamini_hochberg_mask(p_values, alpha)
    elif correction == "none":
        keep_mask = p_values <= alpha
    else:
        # Defensive: validated upstream.
        raise ValidationError(
            f"Invalid correction: {correction!r}. "
            f"Expected 'fdr_bh', 'bonferroni', or 'none'."
        )

    # Optional node-level retention filter.
    node_dropped_mask: Optional[np.ndarray] = None
    if min_node_retention is not None:
        original_degree = np.zeros(n_nodes, dtype=np.int64)
        np.add.at(original_degree, u_arr, 1)
        np.add.at(original_degree, v_arr, 1)

        kept_u = u_arr[keep_mask]
        kept_v = v_arr[keep_mask]
        surviving_degree = np.zeros(n_nodes, dtype=np.int64)
        np.add.at(surviving_degree, kept_u, 1)
        np.add.at(surviving_degree, kept_v, 1)

        with np.errstate(divide="ignore", invalid="ignore"):
            retention = np.where(
                original_degree > 0,
                surviving_degree / original_degree,
                1.0,
            )
        node_dropped_mask = retention < min_node_retention

        kept_node_mask = ~node_dropped_mask
        keep_mask = keep_mask & kept_node_mask[u_arr] & kept_node_mask[v_arr]

        logger.info(
            "Bipartite SVN filter: weighted=%s, alpha=%.3g, correction=%s, "
            "min_node_retention=%.2f — dropped %d nodes by retention; "
            "kept %d/%d edges (%.1f%%)",
            is_weighted_effective, alpha, correction, min_node_retention,
            int(node_dropped_mask.sum()),
            int(keep_mask.sum()), n_edges,
            100.0 * int(keep_mask.sum()) / n_edges if n_edges else 0.0,
        )
    else:
        logger.info(
            "Bipartite SVN filter: weighted=%s, alpha=%.3g, correction=%s — "
            "kept %d/%d edges (%.1f%%)",
            is_weighted_effective, alpha, correction,
            int(keep_mask.sum()), n_edges,
            100.0 * int(keep_mask.sum()) / n_edges if n_edges else 0.0,
        )

    # Build the backbone graph; bipartite_svn historically uses the
    # removeNode approach (preserves internal IDs and bipartite partition info).
    backbone_graph = nk.Graph(
        n_nodes, directed=graph.isDirected(), weighted=graph.isWeighted()
    )
    for u, v, w in zip(u_arr[keep_mask], v_arr[keep_mask], w_arr[keep_mask]):
        if graph.isWeighted():
            backbone_graph.addEdge(int(u), int(v), float(w))
        else:
            backbone_graph.addEdge(int(u), int(v))

    if node_dropped_mask is not None:
        for node in np.where(node_dropped_mask)[0]:
            if backbone_graph.hasNode(int(node)):
                backbone_graph.removeNode(int(node))

    if not keep_disconnected:
        nodes_to_remove = [
            node for node in range(backbone_graph.numberOfNodes())
            if backbone_graph.hasNode(node) and backbone_graph.degree(node) == 0
        ]
        for node in nodes_to_remove:
            backbone_graph.removeNode(node)

    updated_mapper = IDMapper()
    surviving_originals: List[Any] = []
    for internal_id in range(graph.numberOfNodes()):
        if backbone_graph.hasNode(internal_id):
            try:
                original_id = id_mapper.get_original(internal_id)
            except KeyError:
                continue
            updated_mapper.add_mapping(original_id, internal_id)
            surviving_originals.append(original_id)

    if id_mapper.has_bipartite_partitions():
        survivors = set(surviving_originals)
        new_src = id_mapper.source_partition_originals & survivors
        new_tgt = id_mapper.target_partition_originals & survivors
        if new_src.isdisjoint(new_tgt):
            updated_mapper.set_bipartite_partitions(new_src, new_tgt)
        else:
            updated_mapper.source_partition_originals = new_src
            updated_mapper.target_partition_originals = new_tgt

    if return_filtered_edges:
        edge_df = pl.DataFrame({
            "source_id": id_mapper.get_original_batch(u_arr.tolist()),
            "target_id": id_mapper.get_original_batch(v_arr.tolist()),
            "weight": w_arr,
            "p_value": p_values,
            "kept": keep_mask,
        })
        return backbone_graph, updated_mapper, edge_df

    return backbone_graph, updated_mapper


def _benjamini_hochberg_mask(p_values: np.ndarray, alpha: float) -> np.ndarray:
    """Boolean mask: which p-values are rejected under Benjamini-Hochberg FDR."""
    n = p_values.shape[0]
    if n == 0:
        return np.zeros(0, dtype=bool)

    sorted_idx = np.argsort(p_values)
    sorted_p = p_values[sorted_idx]
    thresholds = np.arange(1, n + 1, dtype=np.float64) / n * alpha
    below = sorted_p <= thresholds

    if not below.any():
        return np.zeros(n, dtype=bool)

    last_reject_in_sorted = int(np.where(below)[0].max())
    reject_mask = np.zeros(n, dtype=bool)
    reject_mask[sorted_idx[: last_reject_in_sorted + 1]] = True
    return reject_mask


# ---------------------------------------------------------------------------
# Weight threshold
# ---------------------------------------------------------------------------

def _apply_weight_threshold(
    graph: nk.Graph,
    id_mapper: IDMapper,
    target_edges: Optional[int],
    weight_threshold: Optional[float],
    keep_disconnected: bool,
    return_filtered_edges: bool,
) -> Union[Tuple[nk.Graph, IDMapper], Tuple[nk.Graph, IDMapper, pl.DataFrame]]:
    """Keep edges with weight above a threshold.

    Threshold selection (priority order):
      1. explicit ``weight_threshold`` if given,
      2. value at rank ``target_edges`` if given,
      3. median of all weights (fallback).
    """
    if not graph.isWeighted():
        logger.warning("Weight threshold applied to unweighted graph (all weights = 1.0)")

    n_nodes = graph.numberOfNodes()
    directed = graph.isDirected()

    sources_l: list = []
    targets_l: list = []
    weights_l: list = []
    for u, v in graph.iterEdges():
        sources_l.append(u)
        targets_l.append(v)
        weights_l.append(graph.weight(u, v) if graph.isWeighted() else 1.0)

    if not sources_l:
        empty = nk.Graph(n_nodes, weighted=True, directed=directed)
        if return_filtered_edges:
            return empty, id_mapper, _empty_edges_df("weight")
        return empty, id_mapper

    sources = np.asarray(sources_l, dtype=np.int64)
    targets = np.asarray(targets_l, dtype=np.int64)
    weights = np.asarray(weights_l, dtype=np.float64)

    if weight_threshold is not None:
        threshold = float(weight_threshold)
        keep_mask = weights >= threshold
        logger.debug("Weight threshold: %s (explicit)", threshold)
    elif target_edges is not None:
        # Take exactly target_edges highest-weight edges (ties broken by
        # original order). Avoids the threshold-with-ties overshoot.
        if target_edges >= len(weights):
            keep_mask = np.ones(len(weights), dtype=bool)
        else:
            sorted_idx = np.argsort(-weights, kind="stable")
            keep_mask = np.zeros(len(weights), dtype=bool)
            keep_mask[sorted_idx[:target_edges]] = True
        logger.debug("Weight threshold: top %d edges", target_edges)
    else:
        threshold = float(np.median(weights))
        keep_mask = weights >= threshold
        logger.debug("Weight threshold: %s (median fallback)", threshold)

    kept_sources = sources[keep_mask]
    kept_targets = targets[keep_mask]
    kept_weights = weights[keep_mask]

    backbone_graph, updated_mapper = _assemble_backbone(
        n_nodes, directed, weighted=True,
        kept_sources=kept_sources, kept_targets=kept_targets,
        kept_weights=kept_weights,
        keep_disconnected=keep_disconnected, id_mapper=id_mapper,
    )

    if return_filtered_edges:
        edge_df = pl.DataFrame({
            "source_id": id_mapper.get_original_batch(sources.tolist()),
            "target_id": id_mapper.get_original_batch(targets.tolist()),
            "weight": weights,
            "p_value": np.full(len(weights), np.nan),
            "alpha_score": np.full(len(weights), np.nan),
            "kept": keep_mask,
        })
        return backbone_graph, updated_mapper, edge_df

    return backbone_graph, updated_mapper


# ---------------------------------------------------------------------------
# Degree threshold
# ---------------------------------------------------------------------------

def _apply_degree_threshold(
    graph: nk.Graph,
    id_mapper: IDMapper,
    target_nodes: Optional[int],
    keep_disconnected: bool,
    return_filtered_edges: bool,
) -> Union[Tuple[nk.Graph, IDMapper], Tuple[nk.Graph, IDMapper, pl.DataFrame]]:
    """Keep nodes with degree above a threshold.

    Threshold selection (priority order):
      1. value at rank ``target_nodes`` if given,
      2. median degree (fallback).
    """
    n_nodes = graph.numberOfNodes()
    directed = graph.isDirected()
    weighted = graph.isWeighted()

    degrees = np.array(
        [
            graph.degree(u) if not directed else graph.degreeIn(u) + graph.degreeOut(u)
            for u in range(n_nodes)
        ],
        dtype=np.int64,
    )

    if target_nodes is not None:
        # Take exactly target_nodes highest-degree nodes (ties broken by node id).
        if target_nodes >= n_nodes:
            keep_node_mask = np.ones(n_nodes, dtype=bool)
        else:
            sorted_idx = np.argsort(-degrees, kind="stable")
            keep_node_mask = np.zeros(n_nodes, dtype=bool)
            keep_node_mask[sorted_idx[:target_nodes]] = True
        logger.debug("Degree threshold: top %d nodes", target_nodes)
    else:
        threshold = int(np.median(degrees))
        keep_node_mask = degrees >= threshold
        logger.debug("Degree threshold: %d (median fallback)", threshold)

    sources_l: list = []
    targets_l: list = []
    weights_l: list = []
    edge_kept_l: list = []
    for u, v in graph.iterEdges():
        sources_l.append(u)
        targets_l.append(v)
        weights_l.append(graph.weight(u, v) if weighted else 1.0)
        edge_kept_l.append(bool(keep_node_mask[u] and keep_node_mask[v]))

    sources = np.asarray(sources_l, dtype=np.int64)
    targets = np.asarray(targets_l, dtype=np.int64)
    weights = np.asarray(weights_l, dtype=np.float64)
    edge_kept = np.asarray(edge_kept_l, dtype=bool)

    kept_sources = sources[edge_kept]
    kept_targets = targets[edge_kept]
    kept_weights = weights[edge_kept]

    # For "degree", we keep nodes regardless of whether they ended up
    # disconnected after edge filtering (the node-level filter is the point).
    # `keep_disconnected` here controls whether *all* kept nodes survive
    # (True) or only those that still have at least one edge (False).
    if keep_disconnected:
        node_survives = keep_node_mask.copy()
    else:
        # Recompute survival based on kept edges.
        node_survives = np.zeros(n_nodes, dtype=bool)
        if kept_sources.size > 0:
            node_survives[kept_sources] = True
            node_survives[kept_targets] = True

    surviving_node_ids = np.where(node_survives)[0]
    node_mapping = {int(old): new for new, old in enumerate(surviving_node_ids)}

    backbone_graph = nk.Graph(
        len(surviving_node_ids), weighted=weighted, directed=directed
    )
    for u, v, w in zip(kept_sources, kept_targets, kept_weights):
        if weighted:
            backbone_graph.addEdge(node_mapping[int(u)], node_mapping[int(v)], float(w))
        else:
            backbone_graph.addEdge(node_mapping[int(u)], node_mapping[int(v)])

    updated_mapper = IDMapper()
    for old_internal, new_internal in node_mapping.items():
        try:
            updated_mapper.add_mapping(id_mapper.get_original(old_internal), new_internal)
        except KeyError:
            pass

    if return_filtered_edges:
        edge_df = pl.DataFrame({
            "source_id": id_mapper.get_original_batch(sources.tolist()),
            "target_id": id_mapper.get_original_batch(targets.tolist()),
            "weight": weights,
            "p_value": np.full(len(weights), np.nan),
            "alpha_score": np.full(len(weights), np.nan),
            "kept": edge_kept,
        })
        return backbone_graph, updated_mapper, edge_df

    return backbone_graph, updated_mapper


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _assemble_backbone(
    n_nodes: int,
    directed: bool,
    weighted: bool,
    kept_sources: np.ndarray,
    kept_targets: np.ndarray,
    kept_weights: np.ndarray,
    keep_disconnected: bool,
    id_mapper: IDMapper,
) -> Tuple[nk.Graph, IDMapper]:
    """Construct the backbone graph and updated id mapper.

    If ``keep_disconnected`` is True, the backbone graph has the same node
    set as the original. Otherwise, isolated nodes are dropped and the
    surviving nodes are renumbered to a consecutive 0..N-1 internal-ID
    space (the new id_mapper reflects this renumbering).
    """
    if keep_disconnected:
        backbone_graph = nk.Graph(n_nodes, weighted=weighted, directed=directed)
        for u, v, w in zip(kept_sources, kept_targets, kept_weights):
            if weighted:
                backbone_graph.addEdge(int(u), int(v), float(w))
            else:
                backbone_graph.addEdge(int(u), int(v))
        return backbone_graph, id_mapper

    if kept_sources.size == 0:
        return nk.Graph(0, weighted=weighted, directed=directed), IDMapper()

    connected = sorted({int(x) for x in kept_sources} | {int(x) for x in kept_targets})
    node_mapping = {old: new for new, old in enumerate(connected)}

    backbone_graph = nk.Graph(len(connected), weighted=weighted, directed=directed)
    for u, v, w in zip(kept_sources, kept_targets, kept_weights):
        if weighted:
            backbone_graph.addEdge(node_mapping[int(u)], node_mapping[int(v)], float(w))
        else:
            backbone_graph.addEdge(node_mapping[int(u)], node_mapping[int(v)])

    updated_mapper = IDMapper()
    for old_internal, new_internal in node_mapping.items():
        updated_mapper.add_mapping(id_mapper.get_original(old_internal), new_internal)

    return backbone_graph, updated_mapper


def _build_empty_like(graph: nk.Graph) -> nk.Graph:
    """Create an empty graph with the same directed/weighted flags."""
    return nk.Graph(0, weighted=graph.isWeighted(), directed=graph.isDirected())


def _empty_edges_df(method: str) -> pl.DataFrame:
    """Return an empty edge-results DataFrame with the correct schema."""
    base = {"source_id": [], "target_id": [], "weight": [], "kept": []}
    if method == "disparity" or method in ("weight", "degree"):
        base["p_value"] = []
        base["alpha_score"] = []
    elif method == "noise_corrected":
        base["score"] = []
        base["sdev_cij"] = []
    elif method == "bipartite_svn":
        base["p_value"] = []
    return pl.DataFrame(base)


def _print_backbone_summary(
    verbose: bool,
    t_start: float,
    original_graph: nk.Graph,
    backbone_graph: nk.Graph,
    method: str,
) -> None:
    """One-line summary printed at the bottom of apply_backbone.

    Mirrors :func:`guidedLP.network.construction._print_backbone_summary`
    (duplicated to avoid the circular import).
    """
    if not verbose:
        return
    dt = _time.perf_counter() - t_start
    n0, e0 = original_graph.numberOfNodes(), original_graph.numberOfEdges()
    n1, e1 = backbone_graph.numberOfNodes(), backbone_graph.numberOfEdges()
    node_pct = (100 * n1 / n0) if n0 else 0.0
    edge_pct = (100 * e1 / e0) if e0 else 0.0
    print(
        f"[apply_backbone] {dt:.2f}s | method={method} | "
        f"{n0:,} → {n1:,} nodes ({node_pct:.1f}% kept) | "
        f"{e0:,} → {e1:,} edges ({edge_pct:.1f}% kept)"
    )


# ---------------------------------------------------------------------------
# Summary statistics
# ---------------------------------------------------------------------------

def get_backbone_statistics(
    original_graph: nk.Graph,
    backbone_graph: nk.Graph,
    filtered_edges: Optional[pl.DataFrame] = None,
) -> Dict[str, Any]:
    """
    Compute summary statistics comparing original and backbone graphs.

    Parameters
    ----------
    original_graph : nk.Graph
        Graph before backbone extraction.
    backbone_graph : nk.Graph
        Graph after backbone extraction.
    filtered_edges : pl.DataFrame, optional
        Per-edge filtering details returned by ``apply_backbone`` when
        ``return_filtered_edges=True``. When provided, adds weight and (for
        the disparity filter) alpha-score statistics under
        ``weight_statistics`` and ``alpha_statistics`` keys.

    Returns
    -------
    Dict[str, Any]
        Always contains: ``original_nodes``, ``original_edges``,
        ``backbone_nodes``, ``backbone_edges``, ``node_retention``,
        ``edge_retention``, ``compression_ratio``, ``original_density``,
        ``backbone_density``, ``density_ratio``,
        ``node_retention_rate``, ``edge_retention_rate``.
    """
    n0 = original_graph.numberOfNodes()
    e0 = original_graph.numberOfEdges()
    n1 = backbone_graph.numberOfNodes()
    e1 = backbone_graph.numberOfEdges()

    node_retention = n1 / max(n0, 1)
    edge_retention = e1 / max(e0, 1)

    stats: Dict[str, Any] = {
        "original_nodes": n0,
        "original_edges": e0,
        "backbone_nodes": n1,
        "backbone_edges": e1,
        "node_retention": node_retention,
        "edge_retention": edge_retention,
        # Aliases used by some callers / tests:
        "node_retention_rate": node_retention,
        "edge_retention_rate": edge_retention,
        "compression_ratio": (n0 + e0) / max(n1 + e1, 1),
    }

    original_density = e0 / max(n0 * (n0 - 1) / 2, 1)
    backbone_density = e1 / max(n1 * (n1 - 1) / 2, 1)
    stats["original_density"] = original_density
    stats["backbone_density"] = backbone_density
    stats["density_ratio"] = backbone_density / max(original_density, 1e-10)

    if filtered_edges is not None:
        kept = filtered_edges.filter(pl.col("kept"))
        if kept.height > 0:
            stats["weight_statistics"] = {
                "kept_weight_mean": float(kept["weight"].mean()),
                "kept_weight_std": float(kept["weight"].std()),
                "kept_weight_min": float(kept["weight"].min()),
                "kept_weight_max": float(kept["weight"].max()),
            }
            if "alpha_score" in kept.columns:
                alpha_scores = kept["alpha_score"].drop_nulls()
                if len(alpha_scores) > 0:
                    stats["alpha_statistics"] = {
                        "alpha_mean": float(alpha_scores.mean()),
                        "alpha_std": float(alpha_scores.std()),
                        "alpha_min": float(alpha_scores.min()),
                        "alpha_max": float(alpha_scores.max()),
                    }

    return stats


# Backwards-compatible alias for the older `get_backbone_summary` name.
get_backbone_summary = get_backbone_statistics
