"""
Network backboning module for the Guided Label Propagation library.

This module implements various network backbone extraction methods to identify
the most significant edges in weighted networks. The primary method is the
disparity filter by Serrano et al. (2009).

Mathematical Background:
The disparity filter identifies statistically significant edges by comparing
edge weights against a uniform null model. For each node, edge weights are
normalized and compared against the expected distribution under random allocation.
"""

from typing import Tuple, Optional, Dict, Any, Union
import warnings

import numpy as np
import networkit as nk
import polars as pl

from guidedLP.common.id_mapper import IDMapper
from guidedLP.common.exceptions import (
    ComputationError,
    ConfigurationError,
    ValidationError,
    validate_parameter,
    require_positive
)
from guidedLP.common.logging_config import get_logger, log_function_entry, LoggingTimer

logger = get_logger(__name__)

# Available backboning methods
AVAILABLE_METHODS = ["disparity", "noise_corrected", "weight_threshold", "degree_threshold"]


def apply_backbone(
    graph: nk.Graph,
    id_mapper: IDMapper,
    method: str = "disparity",
    alpha: float = 0.05,
    threshold: float = 1.0,
    target_nodes: Optional[int] = None,
    target_edges: Optional[int] = None,
    weight_threshold: Optional[float] = None,
    keep_disconnected: bool = False,
    return_filtered_edges: bool = False
) -> Union[Tuple[nk.Graph, IDMapper], Tuple[nk.Graph, IDMapper, pl.DataFrame]]:
    """
    Extract network backbone by filtering statistically insignificant edges.

    This function implements multiple backbone extraction methods with the
    disparity filter as the primary approach. The disparity filter identifies
    edges that are statistically significant given each node's degree.

    Parameters
    ----------
    graph : nk.Graph
        NetworkIt weighted graph for backbone extraction
    id_mapper : IDMapper
        Bidirectional mapping between original and internal node IDs
    method : str, default "disparity"
        Backbone extraction method:
        - "disparity": Serrano et al. disparity filter
        - "noise_corrected": Coscia & Neffke noise-corrected backbone (recommended
          for bipartite projections and heterogeneous edge weight distributions)
        - "weight_threshold": Simple weight-based filtering
        - "degree_threshold": Filter nodes by degree threshold
    alpha : float, default 0.05
        Significance level for disparity filter (p-value threshold).
        Lower values = more stringent filtering = fewer edges retained.
    threshold : float, default 1.0
        Standard-deviation threshold for noise_corrected backbone. An edge is
        kept iff ``score - threshold * sdev_cij > 0`` — i.e. the observed
        weight exceeds the configuration-model expectation by at least
        ``threshold`` standard deviations. Higher values = fewer edges retained.
    target_nodes : int, optional
        Approximate target number of nodes to retain (for degree_threshold)
        Conflicts with target_edges
    target_edges : int, optional
        Approximate target number of edges to retain (for weight_threshold)
        Conflicts with target_nodes
    weight_threshold : float, optional
        Manual weight threshold (overrides target_edges for weight_threshold method)
    keep_disconnected : bool, default False
        Whether to keep isolated nodes after edge filtering
    return_filtered_edges : bool, default False
        If True, return DataFrame with edge filtering results
        
    Returns
    -------
    backbone_graph : nk.Graph
        Filtered NetworkIt graph with backbone edges
    updated_mapper : IDMapper
        ID mapper with nodes remaining in backbone
    filtered_edges : pl.DataFrame, optional
        DataFrame with edge filtering details (if return_filtered_edges=True).
        Columns are method-dependent:
        - all methods: source_id, target_id, weight, kept
        - disparity: also p_value, alpha_score
        - noise_corrected: also score, sdev_cij
        - weight_threshold, degree_threshold: p_value and alpha_score are NaN
        
    Raises
    ------
    ConfigurationError
        If invalid method specified or conflicting parameters
    ComputationError
        If backbone extraction fails or results in empty graph
    ValidationError
        If parameters are invalid (alpha not in (0,1), negative thresholds)
        
    Examples
    --------
    >>> # Basic disparity filter
    >>> backbone_graph, backbone_mapper = apply_backbone(
    ...     graph, id_mapper, method="disparity", alpha=0.01
    ... )
    
    >>> # Weight threshold keeping ~1000 edges
    >>> backbone_graph, backbone_mapper = apply_backbone(
    ...     graph, id_mapper, method="weight_threshold", target_edges=1000
    ... )
    
    >>> # Get detailed edge filtering results
    >>> backbone_graph, backbone_mapper, edge_details = apply_backbone(
    ...     graph, id_mapper, method="disparity", alpha=0.05, 
    ...     return_filtered_edges=True
    ... )
    >>> print(f"Kept {edge_details.filter(pl.col('kept')).height} edges")
        
    Notes
    -----
    Mathematical Formulation (Disparity Filter):

    For each node i with degree k:
    1. Normalize edge weights: p_ij = w_ij / Σw_ik
    2. Calculate disparity: α_ij = (1 - p_ij)^(k-1)
    3. Keep edges where α_ij < alpha

    Mathematical Formulation (Noise-Corrected Backbone):

    For each edge (i, j) with weight w_ij, given node strengths s_i, s_j and
    total network weight n.. = Σw_ij:
    1. Expected weight under the configuration null model: ⟨w_ij⟩ = s_i s_j / n..
    2. Lift score: score_ij = (κ w_ij − 1) / (κ w_ij + 1), where κ = n.. / (s_i s_j)
    3. Posterior variance from a Beta–Binomial Bayesian update gives sdev_cij
    4. Keep edges where score_ij − threshold · sdev_cij > 0

    Time Complexity:
    - Disparity filter: O(E) where E = number of edges
    - Noise-corrected: O(E) — fully vectorized in Polars
    - Weight threshold: O(E log E) for sorting
    - Degree threshold: O(V + E) where V = number of nodes
    
    Space Complexity: O(V + E) for sparse matrix operations
    
    Performance Optimization:
    - Disparity scoring is fully vectorized over edges with NumPy (no Python
      per-edge loop, no sparse matrix construction)
    - Noise-corrected scoring is vectorized in Polars
    - Numerical edge cases (zero weights, isolated nodes) handled elementwise
    
    The disparity filter assumes edge weights follow a uniform null model.
    For networks where this assumption is violated, consider alternative
    backbone methods or preprocessing steps.
    """
    log_function_entry("apply_backbone", 
                      n_nodes=graph.numberOfNodes(),
                      n_edges=graph.numberOfEdges(),
                      method=method,
                      alpha=alpha)
    
    # Validate parameters
    _validate_backbone_parameters(
        method, alpha, threshold, target_nodes, target_edges, weight_threshold
    )
    
    # Handle empty graph
    if graph.numberOfNodes() == 0 or graph.numberOfEdges() == 0:
        warnings.warn("Empty graph provided. Returning empty graph.")
        empty_graph = nk.Graph(0, weighted=graph.isWeighted(), directed=graph.isDirected())
        empty_mapper = IDMapper()
        if return_filtered_edges:
            empty_edges = pl.DataFrame({
                'source_id': [], 'target_id': [], 'weight': [], 
                'p_value': [], 'alpha_score': [], 'kept': []
            })
            return empty_graph, empty_mapper, empty_edges
        return empty_graph, empty_mapper
    
    with LoggingTimer("apply_backbone", {
        "method": method, "nodes": graph.numberOfNodes(), "edges": graph.numberOfEdges()
    }):
        try:
            if method == "disparity":
                result = _apply_disparity_filter(
                    graph, id_mapper, alpha, keep_disconnected, return_filtered_edges
                )
            elif method == "noise_corrected":
                result = _apply_noise_corrected_filter(
                    graph, id_mapper, threshold, keep_disconnected, return_filtered_edges
                )
            elif method == "weight_threshold":
                result = _apply_weight_threshold(
                    graph, id_mapper, target_edges, weight_threshold, 
                    keep_disconnected, return_filtered_edges
                )
            elif method == "degree_threshold":
                result = _apply_degree_threshold(
                    graph, id_mapper, target_nodes, keep_disconnected, return_filtered_edges
                )
            else:
                raise ConfigurationError(f"Unknown backbone method: {method}")
            
            # Log results
            if return_filtered_edges:
                backbone_graph, updated_mapper, edge_results = result
                n_kept = edge_results.filter(pl.col('kept')).height
                logger.info("Backbone extraction completed: %d → %d edges (%.1f%% retained)",
                           graph.numberOfEdges(), n_kept, 100 * n_kept / graph.numberOfEdges())
            else:
                backbone_graph, updated_mapper = result
                logger.info("Backbone extraction completed: %d → %d edges",
                           graph.numberOfEdges(), backbone_graph.numberOfEdges())
            
            return result
            
        except Exception as e:
            if isinstance(e, (ConfigurationError, ComputationError, ValidationError)):
                raise
            else:
                raise ComputationError(
                    f"Backbone extraction failed: {str(e)}",
                    operation=f"apply_backbone_{method}",
                    error_type="computation",
                    resource_info={
                        "nodes": graph.numberOfNodes(), 
                        "edges": graph.numberOfEdges()
                    },
                    cause=e
                )


def _validate_backbone_parameters(
    method: str,
    alpha: float,
    threshold: float,
    target_nodes: Optional[int],
    target_edges: Optional[int],
    weight_threshold: Optional[float]
) -> None:
    """Validate backbone extraction parameters."""

    # Validate method
    if method not in AVAILABLE_METHODS:
        raise ValidationError(
            f"Invalid backbone method: {method}. Available methods: {AVAILABLE_METHODS}"
        )

    # Validate alpha for disparity filter
    if method == "disparity":
        if not 0 < alpha < 1:
            raise ValidationError(f"Alpha must be in (0, 1), got {alpha}")

    # Validate threshold for noise-corrected backbone (sigma multiplier; must be > 0)
    if method == "noise_corrected":
        if threshold <= 0:
            raise ValidationError(
                f"threshold must be > 0 for noise_corrected, got {threshold}"
            )

    # Validate conflicting parameters
    if target_nodes is not None and target_edges is not None:
        raise ValidationError("Cannot specify both target_nodes and target_edges")

    # Validate positive targets
    if target_nodes is not None:
        require_positive(target_nodes, "target_nodes")
    if target_edges is not None:
        require_positive(target_edges, "target_edges")
    if weight_threshold is not None:
        require_positive(weight_threshold, "weight_threshold")


def _apply_disparity_filter(
    graph: nk.Graph,
    id_mapper: IDMapper,
    alpha: float,
    keep_disconnected: bool,
    return_filtered_edges: bool
) -> Union[Tuple[nk.Graph, IDMapper], Tuple[nk.Graph, IDMapper, pl.DataFrame]]:
    """
    Apply Serrano et al.'s disparity filter, fully vectorized over edges.

    Strengths and degree counts are computed with ``np.bincount`` (O(E), one
    pass each); the per-edge disparity scores α_ij = (1 − p_ij)^(k − 1) are
    evaluated as a single ``np.power``-equivalent over the edge arrays.

    Behavior preserved against the original implementation:
    - α = 1.0 sentinel when an endpoint has k ≤ 1 (leaf nodes)
    - undirected edges keep ``min(α_uv, α_vu)`` and ``min(p_uv, p_vu)``
    - edges incident on a zero-strength node are forcibly dropped
    - keep iff ``α < alpha`` or ``α >= 1.0`` (sentinel)
    """
    logger.debug("Applying disparity filter with alpha=%.3f", alpha)

    n_nodes = graph.numberOfNodes()
    directed = graph.isDirected()

    # Single C++ pass for edge extraction. iterEdges yields each edge once
    # (NetworkIt orders u <= v internally for undirected).
    sources_l: list = []
    targets_l: list = []
    weights_l: list = []
    for u, v in graph.iterEdges():
        sources_l.append(u)
        targets_l.append(v)
        weights_l.append(graph.weight(u, v))

    if not sources_l:
        empty_graph = nk.Graph(n_nodes, weighted=True, directed=directed)
        if return_filtered_edges:
            empty_edges = pl.DataFrame({
                'source_id': [], 'target_id': [], 'weight': [],
                'p_value': [], 'alpha_score': [], 'kept': []
            })
            return empty_graph, id_mapper, empty_edges
        return empty_graph, id_mapper

    sources = np.asarray(sources_l, dtype=np.int64)
    targets = np.asarray(targets_l, dtype=np.int64)
    weights = np.asarray(weights_l, dtype=np.float64)

    # Strengths and degree counts in O(E) each, no sparse matrix needed.
    if directed:
        # Match the original: normalization uses out-strength
        # (adj_matrix.sum(axis=1)) and degree count is degreeOut + degreeIn.
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

    # Avoid division-by-zero warnings for invalid edges; the invalid_mask
    # below overrides their final values.
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

    # Keep condition: α below threshold OR sentinel (k ≤ 1 ⇒ α = 1.0).
    keep_mask = (final_alpha < alpha) | (final_alpha >= 1.0)

    # Defensive guard for edges incident on a zero-strength node. Matches
    # the original isolated-node branch: alpha = 1.0, p_value = 1.0, kept = False.
    invalid_mask = (s_u == 0) | (s_v == 0)
    if np.any(invalid_mask):
        final_alpha = np.where(invalid_mask, 1.0, final_alpha)
        p_value_col = np.where(invalid_mask, 1.0, p_value_col)
        keep_mask = keep_mask & ~invalid_mask

    kept_sources = sources[keep_mask]
    kept_targets = targets[keep_mask]
    kept_weights = weights[keep_mask]

    if keep_disconnected:
        backbone_graph = nk.Graph(n_nodes, weighted=True, directed=directed)
        for u, v, w in zip(kept_sources, kept_targets, kept_weights):
            backbone_graph.addEdge(int(u), int(v), float(w))
        updated_mapper = id_mapper
    elif kept_sources.size == 0:
        backbone_graph = nk.Graph(0, weighted=True, directed=directed)
        updated_mapper = IDMapper()
    else:
        connected_nodes = sorted({int(x) for x in kept_sources} | {int(x) for x in kept_targets})
        node_mapping = {old_id: new_id for new_id, old_id in enumerate(connected_nodes)}

        backbone_graph = nk.Graph(len(connected_nodes), weighted=True, directed=directed)
        for u, v, w in zip(kept_sources, kept_targets, kept_weights):
            backbone_graph.addEdge(node_mapping[int(u)], node_mapping[int(v)], float(w))

        updated_mapper = IDMapper()
        for old_internal, new_internal in node_mapping.items():
            updated_mapper.add_mapping(id_mapper.get_original(old_internal), new_internal)

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
    """Vectorized (1 − p)^(k − 1) matching :func:`_safe_power`'s edge cases.

    Elementwise, in precedence order:
    - exponent <= 0 -> 1.0 (leaf-node sentinel where k <= 1; takes priority
      because the original guards ``k_u > 1`` *before* calling _safe_power)
    - base <= 0     -> 0.0 (p = 1, i.e. the edge carries all of u's strength)
    - base >= 1     -> 1.0 (p = 0, zero-weight edge)
    - otherwise     -> exp(exponent * log(base)), clamped against underflow
    """
    result = np.ones_like(base, dtype=np.float64)
    formula_applies = exponent > 0  # gate matching the original `if k_u > 1` branch
    compute_mask = formula_applies & (base > 0) & (base < 1)
    if np.any(compute_mask):
        log_result = exponent[compute_mask] * np.log(base[compute_mask])
        # Avoid exp underflow for very small bases / large exponents.
        np.maximum(log_result, -700.0, out=log_result)
        result[compute_mask] = np.exp(log_result)
    # base <= 0 collapses to 0 only where the formula applies; where exponent
    # <= 0 the sentinel 1.0 stands regardless of base.
    result[formula_applies & (base <= 0)] = 0.0
    return result


def _safe_power(base: float, exponent: float) -> float:
    """
    Numerically stable power calculation for disparity filter.
    
    Handles edge cases:
    - Very small base values (near 0)
    - Large exponents that could cause underflow
    - Invalid inputs (negative base, etc.)
    """
    if base <= 0:
        return 0.0
    if base >= 1:
        return 1.0
    if exponent <= 0:
        return 1.0
    
    # Use logarithmic computation for numerical stability
    try:
        log_result = exponent * np.log(base)
        if log_result < -700:  # Prevent underflow (exp(-700) ≈ 0)
            return 0.0
        return np.exp(log_result)
    except (OverflowError, ValueError):
        return 0.0


def _apply_noise_corrected_filter(
    graph: nk.Graph,
    id_mapper: IDMapper,
    threshold: float,
    keep_disconnected: bool,
    return_filtered_edges: bool,
) -> Union[Tuple[nk.Graph, IDMapper], Tuple[nk.Graph, IDMapper, pl.DataFrame]]:
    """
    Apply the Coscia & Neffke (2017) noise-corrected backbone, vectorized in Polars.

    Compares each observed edge weight against the configuration-model
    expectation E[w_ij] = s_i * s_j / n.. and computes a Bayesian posterior
    on the lift score with a Beta–Binomial prior. Edges are kept where
    ``score - threshold * sdev_cij > 0``.
    """
    logger.debug("Applying noise-corrected backbone with threshold=%.3f", threshold)

    n_nodes = graph.numberOfNodes()
    directed = graph.isDirected()

    # Extract edges once. iterEdges() yields each edge once (NetworkIt orders
    # u <= v for undirected internally).
    sources_l: list = []
    targets_l: list = []
    weights_l: list = []
    for u, v in graph.iterEdges():
        sources_l.append(u)
        targets_l.append(v)
        weights_l.append(graph.weight(u, v))

    if not sources_l:
        empty_graph = nk.Graph(n_nodes, weighted=True, directed=directed)
        if return_filtered_edges:
            empty_edges = pl.DataFrame({
                "source_id": [], "target_id": [], "weight": [],
                "score": [], "sdev_cij": [], "kept": [],
            })
            return empty_graph, id_mapper, empty_edges
        return empty_graph, id_mapper

    sources = np.asarray(sources_l, dtype=np.int64)
    targets = np.asarray(targets_l, dtype=np.int64)
    weights = np.asarray(weights_l, dtype=np.float64)

    # Build a directed-style frame: for undirected graphs each edge appears
    # in both directions so that group_by("o").sum gives the full node
    # strength s_i = Σ_j w_ij. The downstream `o <= e` filter restores
    # one row per undirected edge.
    if directed:
        df = pl.DataFrame({"o": sources, "e": targets, "w": weights})
    else:
        df = pl.DataFrame({
            "o": np.concatenate([sources, targets]),
            "e": np.concatenate([targets, sources]),
            "w": np.concatenate([weights, weights]),
        })

    # Total weight n.. is the sum over the (possibly symmetrized) frame, so
    # for undirected graphs n.. = 2m (matches the configuration model where
    # E[w_ij] = s_i * s_j / 2m).
    n_total = float(df["w"].sum())

    if n_total <= 0:
        raise ComputationError(
            "Noise-corrected backbone requires positive total edge weight",
            operation="noise_corrected",
        )

    src_sum = df.group_by("o").agg(pl.col("w").sum().alias("o_sum"))
    trg_sum = df.group_by("e").agg(pl.col("w").sum().alias("e_sum"))
    df = df.join(src_sum, on="o", how="left").join(trg_sum, on="e", how="left")

    # Bayesian noise-corrected score (Coscia & Neffke 2017). See the
    # Mathematical Formulation section in apply_backbone's docstring.
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

    # Collapse the symmetrized frame back to one row per undirected edge.
    if not directed:
        df = df.filter(pl.col("o") <= pl.col("e"))

    df = df.with_columns(
        ((pl.col("score") - threshold * pl.col("sdev_cij")) > 0).alias("kept")
    )

    # Extract kept edges into a graph
    kept_df = df.filter(pl.col("kept"))
    kept_sources = kept_df["o"].to_numpy()
    kept_targets = kept_df["e"].to_numpy()
    kept_weights = kept_df["w"].to_numpy()

    if keep_disconnected or len(kept_sources) == 0:
        backbone_graph = nk.Graph(n_nodes, weighted=True, directed=directed)
        for u, v, w in zip(kept_sources, kept_targets, kept_weights):
            backbone_graph.addEdge(int(u), int(v), float(w))
        updated_mapper = id_mapper if len(kept_sources) > 0 or keep_disconnected else IDMapper()
        if len(kept_sources) == 0 and not keep_disconnected:
            backbone_graph = nk.Graph(0, weighted=True, directed=directed)
    else:
        connected_nodes = set(int(x) for x in kept_sources) | set(int(x) for x in kept_targets)
        node_mapping = {old_id: new_id for new_id, old_id in enumerate(sorted(connected_nodes))}

        backbone_graph = nk.Graph(len(connected_nodes), weighted=True, directed=directed)
        for u, v, w in zip(kept_sources, kept_targets, kept_weights):
            backbone_graph.addEdge(node_mapping[int(u)], node_mapping[int(v)], float(w))

        updated_mapper = IDMapper()
        for old_internal, new_internal in node_mapping.items():
            updated_mapper.add_mapping(id_mapper.get_original(old_internal), new_internal)

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


def _apply_weight_threshold(
    graph: nk.Graph,
    id_mapper: IDMapper,
    target_edges: Optional[int],
    weight_threshold: Optional[float],
    keep_disconnected: bool,
    return_filtered_edges: bool
) -> Union[Tuple[nk.Graph, IDMapper], Tuple[nk.Graph, IDMapper, pl.DataFrame]]:
    """Apply simple weight-based threshold filtering."""
    logger.debug("Applying weight threshold filter")
    
    # Extract edge weights
    edge_weights = []
    edges_data = []
    
    for u in graph.iterNodes():
        for v in graph.iterNeighbors(u):
            if not graph.isDirected() and u > v:
                continue
            weight = graph.weight(u, v)
            edge_weights.append(weight)
            edges_data.append((u, v, weight))
    
    if not edge_weights:
        # No edges
        empty_graph = nk.Graph(graph.numberOfNodes(), weighted=True, directed=graph.isDirected())
        if return_filtered_edges:
            empty_edges = pl.DataFrame({
                'source_id': [], 'target_id': [], 'weight': [], 
                'p_value': [], 'alpha_score': [], 'kept': []
            })
            return empty_graph, id_mapper, empty_edges
        return empty_graph, id_mapper
    
    # Determine threshold
    if weight_threshold is not None:
        threshold = weight_threshold
    elif target_edges is not None:
        # Calculate threshold to keep approximately target_edges
        sorted_weights = sorted(edge_weights, reverse=True)
        threshold_idx = min(target_edges - 1, len(sorted_weights) - 1)
        threshold = sorted_weights[threshold_idx]
    else:
        raise ConfigurationError("Must specify either weight_threshold or target_edges")
    
    # Filter edges
    kept_edges = []
    edge_results = []
    
    for u, v, w in edges_data:
        keep_edge = w >= threshold
        
        edge_results.append({
            'source_id': id_mapper.get_original(u),
            'target_id': id_mapper.get_original(v),
            'weight': w,
            'p_value': np.nan,  # Not applicable for weight threshold
            'alpha_score': np.nan,  # Not applicable
            'kept': bool(keep_edge)
        })
        
        if keep_edge:
            kept_edges.append((u, v, w))
    
    # Create backbone graph (similar to disparity filter)
    backbone_graph = nk.Graph(graph.numberOfNodes(), weighted=True, directed=graph.isDirected())
    
    for u, v, w in kept_edges:
        backbone_graph.addEdge(u, v, w)
    
    # Handle disconnected nodes (same logic as disparity filter)
    if not keep_disconnected:
        connected_nodes = set()
        for u, v, _ in kept_edges:
            connected_nodes.add(u)
            connected_nodes.add(v)
        
        if connected_nodes:
            node_mapping = {old_id: new_id for new_id, old_id in enumerate(sorted(connected_nodes))}
            
            new_graph = nk.Graph(len(connected_nodes), weighted=True, directed=graph.isDirected())
            for u, v, w in kept_edges:
                new_u, new_v = node_mapping[u], node_mapping[v]
                new_graph.addEdge(new_u, new_v, w)
            
            new_mapper = IDMapper()
            for old_internal, new_internal in node_mapping.items():
                original_id = id_mapper.get_original(old_internal)
                new_mapper.add_mapping(original_id, new_internal)
            
            backbone_graph = new_graph
            updated_mapper = new_mapper
        else:
            backbone_graph = nk.Graph(0, weighted=True, directed=graph.isDirected())
            updated_mapper = IDMapper()
    else:
        updated_mapper = id_mapper
    
    if return_filtered_edges:
        edge_df = pl.DataFrame(edge_results)
        return backbone_graph, updated_mapper, edge_df
    
    return backbone_graph, updated_mapper


def _apply_degree_threshold(
    graph: nk.Graph,
    id_mapper: IDMapper,
    target_nodes: Optional[int],
    keep_disconnected: bool,
    return_filtered_edges: bool
) -> Union[Tuple[nk.Graph, IDMapper], Tuple[nk.Graph, IDMapper, pl.DataFrame]]:
    """Apply degree-based node filtering."""
    logger.debug("Applying degree threshold filter")
    
    # Calculate node degrees
    node_degrees = []
    for u in graph.iterNodes():
        degree = graph.degree(u) if not graph.isDirected() else (graph.degreeIn(u) + graph.degreeOut(u))
        node_degrees.append((u, degree))
    
    if target_nodes is not None:
        # Keep top target_nodes by degree
        sorted_nodes = sorted(node_degrees, key=lambda x: x[1], reverse=True)
        nodes_to_keep = set(u for u, _ in sorted_nodes[:target_nodes])
    else:
        raise ConfigurationError("Must specify target_nodes for degree_threshold method")
    
    # Extract edges for kept nodes
    kept_edges = []
    edge_results = []
    
    for u in graph.iterNodes():
        for v in graph.iterNeighbors(u):
            if not graph.isDirected() and u > v:
                continue
            
            weight = graph.weight(u, v)
            keep_edge = u in nodes_to_keep and v in nodes_to_keep
            
            edge_results.append({
                'source_id': id_mapper.get_original(u),
                'target_id': id_mapper.get_original(v),
                'weight': weight,
                'p_value': np.nan,
                'alpha_score': np.nan,
                'kept': bool(keep_edge)
            })
            
            if keep_edge:
                kept_edges.append((u, v, weight))
    
    # Create subgraph with kept nodes
    if nodes_to_keep:
        node_mapping = {old_id: new_id for new_id, old_id in enumerate(sorted(nodes_to_keep))}
        
        new_graph = nk.Graph(len(nodes_to_keep), weighted=True, directed=graph.isDirected())
        for u, v, w in kept_edges:
            new_u, new_v = node_mapping[u], node_mapping[v]
            new_graph.addEdge(new_u, new_v, w)
        
        new_mapper = IDMapper()
        for old_internal, new_internal in node_mapping.items():
            original_id = id_mapper.get_original(old_internal)
            new_mapper.add_mapping(original_id, new_internal)
        
        backbone_graph = new_graph
        updated_mapper = new_mapper
    else:
        backbone_graph = nk.Graph(0, weighted=True, directed=graph.isDirected())
        updated_mapper = IDMapper()
    
    if return_filtered_edges:
        edge_df = pl.DataFrame(edge_results)
        return backbone_graph, updated_mapper, edge_df
    
    return backbone_graph, updated_mapper


def get_backbone_summary(
    original_graph: nk.Graph,
    backbone_graph: nk.Graph,
    filtered_edges: Optional[pl.DataFrame] = None
) -> Dict[str, Any]:
    """
    Generate summary statistics for backbone extraction results.
    
    Parameters
    ----------
    original_graph : nk.Graph
        Original graph before backbone extraction
    backbone_graph : nk.Graph
        Resulting backbone graph
    filtered_edges : pl.DataFrame, optional
        DataFrame from apply_backbone with return_filtered_edges=True
        
    Returns
    -------
    Dict[str, Any]
        Summary statistics including retention rates, weight distributions, etc.
        
    Examples
    --------
    >>> backbone_graph, mapper, edge_details = apply_backbone(
    ...     graph, id_mapper, return_filtered_edges=True
    ... )
    >>> summary = get_backbone_summary(graph, backbone_graph, edge_details)
    >>> print(f"Edge retention: {summary['edge_retention_rate']:.1%}")
    """
    summary = {
        'original_nodes': original_graph.numberOfNodes(),
        'original_edges': original_graph.numberOfEdges(),
        'backbone_nodes': backbone_graph.numberOfNodes(), 
        'backbone_edges': backbone_graph.numberOfEdges(),
        'node_retention_rate': backbone_graph.numberOfNodes() / max(original_graph.numberOfNodes(), 1),
        'edge_retention_rate': backbone_graph.numberOfEdges() / max(original_graph.numberOfEdges(), 1)
    }
    
    if filtered_edges is not None:
        kept_edges = filtered_edges.filter(pl.col('kept'))
        
        if kept_edges.height > 0:
            summary['weight_statistics'] = {
                'kept_weight_mean': float(kept_edges['weight'].mean()),
                'kept_weight_std': float(kept_edges['weight'].std()),
                'kept_weight_min': float(kept_edges['weight'].min()),
                'kept_weight_max': float(kept_edges['weight'].max())
            }
            
            # Alpha score statistics (for disparity filter)
            alpha_scores = kept_edges['alpha_score'].drop_nulls()
            if len(alpha_scores) > 0:
                summary['alpha_statistics'] = {
                    'alpha_mean': float(alpha_scores.mean()),
                    'alpha_std': float(alpha_scores.std()),
                    'alpha_min': float(alpha_scores.min()),
                    'alpha_max': float(alpha_scores.max())
                }
    
    return summary