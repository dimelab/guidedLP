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
import scipy.sparse as sp
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
AVAILABLE_METHODS = ["disparity", "weight_threshold", "degree_threshold"]


def apply_backbone(
    graph: nk.Graph,
    id_mapper: IDMapper,
    method: str = "disparity",
    alpha: float = 0.05,
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
        - "disparity": Serrano et al. disparity filter (recommended)
        - "weight_threshold": Simple weight-based filtering
        - "degree_threshold": Filter nodes by degree threshold
    alpha : float, default 0.05
        Significance level for disparity filter (p-value threshold)
        Lower values = more stringent filtering = fewer edges retained
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
        DataFrame with edge filtering details (if return_filtered_edges=True)
        Contains columns: source_id, target_id, weight, p_value, alpha_score, kept
        
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
    
    Time Complexity:
    - Disparity filter: O(E) where E = number of edges
    - Weight threshold: O(E log E) for sorting
    - Degree threshold: O(V + E) where V = number of nodes
    
    Space Complexity: O(V + E) for sparse matrix operations
    
    Performance Optimization:
    - Uses scipy.sparse matrices for memory efficiency with large graphs
    - Vectorized operations for numerical stability
    - Handles numerical edge cases (zero weights, isolated nodes)
    
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
    _validate_backbone_parameters(method, alpha, target_nodes, target_edges, weight_threshold)
    
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
    Apply disparity filter using efficient sparse matrix operations.
    
    Mathematical Implementation:
    1. Extract adjacency matrix as scipy.sparse.csr_matrix for efficiency
    2. Calculate node degree sums using sparse matrix operations  
    3. Normalize edge weights: p_ij = w_ij / degree_sum_i
    4. Calculate disparity scores: α_ij = (1 - p_ij)^(degree_i - 1)
    5. Keep edges where α_ij < alpha
    
    Numerical Stability:
    - Handle zero-weight edges and isolated nodes
    - Use log-space calculations for very small probabilities
    - Clip extreme values to prevent overflow/underflow
    """
    logger.debug("Applying disparity filter with alpha=%.3f", alpha)
    
    # Convert NetworkIt graph to sparse matrix for efficient computation
    n_nodes = graph.numberOfNodes()
    
    # Extract edge data
    edges_data = []
    for u in graph.iterNodes():
        for v in graph.iterNeighbors(u):
            if not graph.isDirected() and u > v:
                continue  # Avoid duplicate edges in undirected graphs
            weight = graph.weight(u, v)
            edges_data.append((u, v, weight))
    
    if not edges_data:
        # No edges to filter
        empty_graph = nk.Graph(n_nodes, weighted=True, directed=graph.isDirected())
        if return_filtered_edges:
            empty_edges = pl.DataFrame({
                'source_id': [], 'target_id': [], 'weight': [], 
                'p_value': [], 'alpha_score': [], 'kept': []
            })
            return empty_graph, id_mapper, empty_edges
        return empty_graph, id_mapper
    
    # Convert to arrays for vectorized operations
    edges_array = np.array(edges_data)
    sources = edges_array[:, 0].astype(int)
    targets = edges_array[:, 1].astype(int)  
    weights = edges_array[:, 2].astype(float)
    
    # Build sparse adjacency matrix
    if graph.isDirected():
        adj_matrix = sp.coo_matrix((weights, (sources, targets)), shape=(n_nodes, n_nodes))
    else:
        # For undirected graphs, add symmetric entries
        all_sources = np.concatenate([sources, targets])
        all_targets = np.concatenate([targets, sources])
        all_weights = np.concatenate([weights, weights])
        adj_matrix = sp.coo_matrix((all_weights, (all_sources, all_targets)), shape=(n_nodes, n_nodes))
    
    adj_matrix = adj_matrix.tocsr()  # Convert to CSR for efficient row operations
    
    # Calculate degree sums (strength) for each node
    degree_sums = np.array(adj_matrix.sum(axis=1)).flatten()
    
    # Identify nodes with non-zero degree
    valid_nodes = degree_sums > 0
    
    # Calculate node degrees (number of connections)
    if graph.isDirected():
        degrees = np.array([graph.degreeOut(u) + graph.degreeIn(u) for u in range(n_nodes)])
    else:
        degrees = np.array([graph.degree(u) for u in range(n_nodes)])
    
    # Calculate disparity scores for each edge
    edge_results = []
    kept_edges = []
    
    for i, (u, v, w) in enumerate(edges_data):
        u, v = int(u), int(v)
        
        # Skip edges from/to isolated nodes
        if degree_sums[u] == 0 or degree_sums[v] == 0:
            edge_results.append({
                'source_id': id_mapper.get_original(u),
                'target_id': id_mapper.get_original(v), 
                'weight': w,
                'p_value': 1.0,
                'alpha_score': 1.0,
                'kept': False
            })
            continue
        
        # Calculate normalized weights (probabilities)
        p_uv = w / degree_sums[u]
        if not graph.isDirected():
            p_vu = w / degree_sums[v]
        
        # Calculate disparity scores using numerical stable computation
        # α = (1 - p)^(k-1)
        k_u = degrees[u] 
        alpha_uv = _safe_power(1 - p_uv, k_u - 1) if k_u > 1 else 1.0
        
        if not graph.isDirected():
            k_v = degrees[v]
            alpha_vu = _safe_power(1 - p_vu, k_v - 1) if k_v > 1 else 1.0
            # For undirected graphs, edge is significant if it's significant from either direction
            final_alpha = min(alpha_uv, alpha_vu)
        else:
            final_alpha = alpha_uv
        
        # Edge is kept if disparity score is below threshold
        keep_edge = final_alpha < alpha
        
        edge_results.append({
            'source_id': id_mapper.get_original(u),
            'target_id': id_mapper.get_original(v),
            'weight': w,
            'p_value': min(p_uv, p_vu) if not graph.isDirected() else p_uv,
            'alpha_score': final_alpha,
            'kept': keep_edge
        })
        
        if keep_edge:
            kept_edges.append((u, v, w))
    
    # Create backbone graph
    backbone_graph = nk.Graph(n_nodes, weighted=True, directed=graph.isDirected())
    
    for u, v, w in kept_edges:
        backbone_graph.addEdge(u, v, w)
    
    # Handle disconnected nodes
    if not keep_disconnected:
        # Remove isolated nodes and update mapper
        connected_nodes = set()
        for u, v, _ in kept_edges:
            connected_nodes.add(u)
            connected_nodes.add(v)
        
        if connected_nodes:
            # Create subgraph with only connected nodes
            node_mapping = {old_id: new_id for new_id, old_id in enumerate(sorted(connected_nodes))}
            
            new_graph = nk.Graph(len(connected_nodes), weighted=True, directed=graph.isDirected())
            for u, v, w in kept_edges:
                new_u, new_v = node_mapping[u], node_mapping[v]
                new_graph.addEdge(new_u, new_v, w)
            
            # Update ID mapper
            new_mapper = IDMapper()
            for old_internal, new_internal in node_mapping.items():
                original_id = id_mapper.get_original(old_internal)
                new_mapper.add_mapping(original_id, new_internal)
            
            backbone_graph = new_graph
            updated_mapper = new_mapper
        else:
            # No edges kept
            backbone_graph = nk.Graph(0, weighted=True, directed=graph.isDirected())
            updated_mapper = IDMapper()
    else:
        updated_mapper = id_mapper
    
    if return_filtered_edges:
        edge_df = pl.DataFrame(edge_results)
        return backbone_graph, updated_mapper, edge_df
    
    return backbone_graph, updated_mapper


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
            'kept': keep_edge
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
                'kept': keep_edge
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
            if alpha_scores.height > 0:
                summary['alpha_statistics'] = {
                    'alpha_mean': float(alpha_scores.mean()),
                    'alpha_std': float(alpha_scores.std()),
                    'alpha_min': float(alpha_scores.min()),
                    'alpha_max': float(alpha_scores.max())
                }
    
    return summary