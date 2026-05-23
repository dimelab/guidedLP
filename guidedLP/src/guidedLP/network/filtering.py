"""
Network filtering and backboning module for the Guided Label Propagation library.

This module provides functionality for filtering networks based on various criteria
and extracting network backbones using statistical filtering methods. Supports
efficient sparse matrix operations for large graphs.
"""

from typing import List, Dict, Any, Optional, Tuple, Union, Set
import warnings
from collections import defaultdict

import polars as pl
import networkit as nk
import numpy as np
import scipy.sparse as sp
from scipy.stats import norm

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

# Available filter types
SUPPORTED_FILTER_TYPES = [
    "min_degree", "max_degree", "min_weight", "giant_component_only",
    "nodes", "exclude_nodes", "centrality"
]

# Available backbone methods
AVAILABLE_BACKBONE_METHODS = ["disparity", "weight", "degree"]


def filter_graph(
    graph: nk.Graph,
    id_mapper: IDMapper,
    filters: Dict[str, Any],
    combine: str = "and"
) -> Tuple[nk.Graph, IDMapper]:
    """
    Apply various filters to a graph based on specified criteria.
    
    This function provides a flexible framework for filtering networks using
    multiple criteria such as degree bounds, component selection, and centrality
    thresholds. Filters can be combined using AND or OR logic.
    
    Parameters
    ----------
    graph : nk.Graph
        NetworkIt graph object to filter
    id_mapper : IDMapper
        Bidirectional mapping between original and internal node IDs
    filters : Dict[str, Any]
        Dictionary specifying filter criteria. Supported filters:
        - "min_degree": int - Minimum degree threshold
        - "max_degree": int - Maximum degree threshold  
        - "min_weight": float - Minimum edge weight threshold
        - "giant_component_only": bool - Keep only largest connected component
        - "nodes": List[str] - Keep only these nodes (original IDs)
        - "exclude_nodes": List[str] - Remove these nodes (original IDs)
        - "centrality": Dict - Filter by centrality metrics
          {"metric": str, "min_value": float}
    combine : str, default "and"
        How to combine multiple filters: "and" or "or"
    
    Returns
    -------
    Tuple[nk.Graph, IDMapper]
        filtered_graph : nk.Graph
            Filtered NetworkIt graph
        updated_mapper : IDMapper
            Updated ID mapper containing only remaining nodes
    
    Examples
    --------
    Basic degree filtering:
    
    >>> filters = {"min_degree": 3, "max_degree": 50}
    >>> filtered_g, new_mapper = filter_graph(graph, mapper, filters)
    
    Multiple criteria with OR logic:
    
    >>> filters = {
    ...     "min_degree": 10,
    ...     "nodes": ["important_node1", "important_node2"]
    ... }
    >>> filtered_g, new_mapper = filter_graph(graph, mapper, filters, combine="or")
    
    Giant component extraction:
    
    >>> filters = {"giant_component_only": True}
    >>> filtered_g, new_mapper = filter_graph(graph, mapper, filters)
    
    Centrality-based filtering:
    
    >>> filters = {
    ...     "centrality": {"metric": "betweenness", "min_value": 0.01}
    ... }
    >>> filtered_g, new_mapper = filter_graph(graph, mapper, filters)
    
    Raises
    ------
    ValidationError
        If filter specifications are invalid or conflicting
    ComputationError
        If filtering results in an empty graph or other computation errors
    
    Notes
    -----
    Time Complexity:
        O(N + E) for most filters, O(N²) for component detection
    
    Space Complexity:
        O(N) for node masks, O(E) for edge operations
    
    The function creates boolean masks for each filter criterion and combines
    them according to the specified logic. For efficiency, component detection
    is only performed when necessary.
    """
    log_function_entry(
        "filter_graph",
        graph_nodes=graph.numberOfNodes(),
        graph_edges=graph.numberOfEdges(),
        filters=list(filters.keys()),
        combine=combine
    )
    
    # Validate parameters
    _validate_filter_parameters(filters, combine)
    
    # Handle empty graph
    if graph.numberOfNodes() == 0:
        logger.warning("Empty graph provided. Returning empty graph.")
        return graph, id_mapper
    
    with LoggingTimer("filter_graph", {"filters": list(filters.keys()), "nodes": graph.numberOfNodes()}):
        try:
            # Create masks for each filter
            node_masks = []
            edge_masks = []
            
            for filter_type, filter_value in filters.items():
                if filter_type in ["min_degree", "max_degree"]:
                    mask = _apply_degree_filter(graph, filter_type, filter_value)
                    node_masks.append(mask)
                    
                elif filter_type == "min_weight":
                    mask = _apply_weight_filter(graph, filter_value)
                    edge_masks.append(mask)
                    
                elif filter_type == "giant_component_only" and filter_value:
                    mask = _apply_component_filter(graph)
                    node_masks.append(mask)
                    
                elif filter_type == "nodes":
                    mask = _apply_node_inclusion_filter(graph, id_mapper, filter_value)
                    node_masks.append(mask)
                    
                elif filter_type == "exclude_nodes":
                    mask = _apply_node_exclusion_filter(graph, id_mapper, filter_value)
                    node_masks.append(mask)
                    
                elif filter_type == "centrality":
                    mask = _apply_centrality_filter(graph, id_mapper, filter_value)
                    node_masks.append(mask)
            
            # Combine masks
            final_node_mask = _combine_masks(node_masks, combine) if node_masks else None
            final_edge_mask = _combine_masks(edge_masks, combine) if edge_masks else None
            
            # Apply filters to create new graph
            filtered_graph, updated_mapper = _apply_masks_to_graph(
                graph, id_mapper, final_node_mask, final_edge_mask
            )
            
            # Validate result
            if filtered_graph.numberOfNodes() == 0:
                raise ComputationError(
                    "All nodes were filtered out. Consider relaxing filter criteria.",
                    context={"operation": "filter_graph", "filters": filters}
                )
            
            logger.info(
                f"Graph filtering completed: {graph.numberOfNodes()} → {filtered_graph.numberOfNodes()} nodes, "
                f"{graph.numberOfEdges()} → {filtered_graph.numberOfEdges()} edges"
            )
            
            return filtered_graph, updated_mapper
            
        except Exception as e:
            raise ComputationError(
                f"Graph filtering failed: {str(e)}",
                context={
                    "operation": "filter_graph",
                    "filters": filters,
                    "error_type": "computation"
                }
            ) from e


def apply_backbone(
    graph: nk.Graph,
    id_mapper: IDMapper,
    method: str = "disparity",
    target_nodes: Optional[int] = None,
    target_edges: Optional[int] = None,
    alpha: float = 0.05,
    keep_disconnected: bool = False
) -> Tuple[nk.Graph, IDMapper]:
    """
    Extract network backbone by filtering edges using statistical methods.
    
    This function implements several backbone extraction methods to identify
    the most significant edges in a weighted network. The disparity filter
    follows Serrano et al.'s method for preserving statistically significant
    edge weights.
    
    Parameters
    ----------
    graph : nk.Graph
        NetworkIt graph object (must be weighted for disparity filter)
    id_mapper : IDMapper
        Bidirectional mapping between original and internal node IDs
    method : str, default "disparity"
        Backbone extraction method:
        - "disparity": Disparity filter (Serrano et al.) for weighted graphs
        - "weight": Simple weight threshold filtering
        - "degree": Node degree threshold filtering
    target_nodes : int, optional
        Target number of nodes to keep (conflicts with target_edges)
    target_edges : int, optional
        Target number of edges to keep
    alpha : float, default 0.05
        Significance level for disparity filter (typical range: 0.01-0.1)
    keep_disconnected : bool, default False
        Whether to keep isolated nodes after edge filtering
    
    Returns
    -------
    Tuple[nk.Graph, IDMapper]
        backbone_graph : nk.Graph
            Filtered NetworkIt graph with backbone edges
        updated_mapper : IDMapper
            Updated ID mapper for remaining nodes
    
    Examples
    --------
    Disparity filter for weighted networks:
    
    >>> backbone, new_mapper = apply_backbone(
    ...     weighted_graph, mapper, method="disparity", alpha=0.05
    ... )
    
    Weight threshold with target edge count:
    
    >>> backbone, new_mapper = apply_backbone(
    ...     graph, mapper, method="weight", target_edges=1000
    ... )
    
    Degree threshold with target node count:
    
    >>> backbone, new_mapper = apply_backbone(
    ...     graph, mapper, method="degree", target_nodes=500
    ... )
    
    Raises
    ------
    ValidationError
        If method is invalid or conflicting parameters are specified
    ComputationError
        If backbone extraction fails or results in empty graph
    
    Notes
    -----
    Time Complexity:
        - Disparity: O(E log E) for sorting operations
        - Weight: O(E) for threshold filtering
        - Degree: O(N + E) for node filtering
    
    Space Complexity:
        O(E) for edge operations, O(N) for sparse matrices
    
    The disparity filter uses the formula:
    α_ij = (1 - p_ij)^(k-1) where p_ij = w_ij / Σw_ik
    
    Edges are kept if α_ij < alpha (statistically significant).
    
    References
    ----------
    .. [1] Serrano, M. Ángeles, Marián Boguñá, and Alessandro Vespignani.
           "Extracting the multiscale backbone of complex weighted networks."
           Proceedings of the national academy of sciences 106.16 (2009): 6483-6488.
    """
    log_function_entry(
        "apply_backbone",
        graph_nodes=graph.numberOfNodes(),
        graph_edges=graph.numberOfEdges(),
        method=method,
        target_nodes=target_nodes,
        target_edges=target_edges,
        alpha=alpha
    )
    
    # Validate parameters
    _validate_backbone_parameters(method, target_nodes, target_edges, alpha, graph)
    
    # Handle empty graph
    if graph.numberOfNodes() == 0:
        logger.warning("Empty graph provided. Returning empty graph.")
        return graph, id_mapper
    
    with LoggingTimer("apply_backbone", {"method": method, "nodes": graph.numberOfNodes(), "edges": graph.numberOfEdges()}):
        try:
            if method == "disparity":
                backbone_graph, updated_mapper = _apply_disparity_filter(
                    graph, id_mapper, alpha, target_edges, keep_disconnected
                )
            elif method == "weight":
                backbone_graph, updated_mapper = _apply_weight_threshold(
                    graph, id_mapper, target_edges, keep_disconnected
                )
            elif method == "degree":
                backbone_graph, updated_mapper = _apply_degree_threshold(
                    graph, id_mapper, target_nodes, keep_disconnected
                )
            else:
                raise ValidationError(f"Unsupported backbone method: {method}")
            
            # Validate result
            if backbone_graph.numberOfEdges() == 0:
                raise ComputationError(
                    "All edges were filtered out. Consider relaxing parameters.",
                    context={"operation": "apply_backbone", "method": method}
                )
            
            logger.info(
                f"Backbone extraction completed: {graph.numberOfEdges()} → {backbone_graph.numberOfEdges()} edges, "
                f"{graph.numberOfNodes()} → {backbone_graph.numberOfNodes()} nodes"
            )
            
            return backbone_graph, updated_mapper
            
        except Exception as e:
            raise ComputationError(
                f"Backbone extraction failed: {str(e)}",
                context={
                    "operation": "apply_backbone",
                    "method": method,
                    "error_type": "computation"
                }
            ) from e


# Helper functions for validation

def _validate_filter_parameters(filters: Dict[str, Any], combine: str) -> None:
    """Validate filter graph parameters."""
    if not filters:
        raise ValidationError("At least one filter must be specified")
    
    if combine not in ["and", "or"]:
        raise ValidationError("combine parameter must be 'and' or 'or'")
    
    for filter_type in filters.keys():
        if filter_type not in SUPPORTED_FILTER_TYPES:
            raise ValidationError(
                f"Unsupported filter type: {filter_type}. "
                f"Supported types: {SUPPORTED_FILTER_TYPES}"
            )
    
    # Check for conflicting degree filters
    if "min_degree" in filters and "max_degree" in filters:
        if filters["min_degree"] > filters["max_degree"]:
            raise ValidationError("min_degree cannot be greater than max_degree")
    
    # Validate centrality filter format
    if "centrality" in filters:
        centrality_filter = filters["centrality"]
        if not isinstance(centrality_filter, dict):
            raise ValidationError("centrality filter must be a dictionary")
        if "metric" not in centrality_filter or "min_value" not in centrality_filter:
            raise ValidationError("centrality filter must have 'metric' and 'min_value' keys")


def _validate_backbone_parameters(
    method: str,
    target_nodes: Optional[int],
    target_edges: Optional[int],
    alpha: float,
    graph: nk.Graph
) -> None:
    """Validate backbone parameters."""
    if method not in AVAILABLE_BACKBONE_METHODS:
        raise ValidationError(
            f"Invalid backbone method: {method}. "
            f"Available methods: {AVAILABLE_BACKBONE_METHODS}"
        )
    
    if target_nodes is not None and target_edges is not None:
        raise ValidationError("Cannot specify both target_nodes and target_edges")
    
    if target_nodes is not None and target_nodes <= 0:
        raise ValidationError("target_nodes must be positive")
    
    if target_edges is not None and target_edges <= 0:
        raise ValidationError("target_edges must be positive")
    
    if not (0.0 < alpha < 1.0):
        raise ValidationError("alpha must be between 0 and 1")
    
    if method == "disparity" and not graph.isWeighted():
        raise ValidationError("Disparity filter requires a weighted graph")


# Helper functions for filtering

def _apply_degree_filter(graph: nk.Graph, filter_type: str, threshold: int) -> np.ndarray:
    """Apply degree-based node filter."""
    n_nodes = graph.numberOfNodes()
    degrees = np.array([graph.degree(u) for u in range(n_nodes)])
    
    if filter_type == "min_degree":
        return degrees >= threshold
    elif filter_type == "max_degree":
        return degrees <= threshold
    else:
        raise ValueError(f"Unknown degree filter type: {filter_type}")


def _apply_weight_filter(graph: nk.Graph, min_weight: float) -> np.ndarray:
    """Apply weight-based edge filter."""
    if not graph.isWeighted():
        logger.warning("Weight filter applied to unweighted graph. All edges have weight 1.0")
        return np.ones(graph.numberOfEdges(), dtype=bool)
    
    weights = []
    for u, v in graph.iterEdges():
        weights.append(graph.weight(u, v))
    
    return np.array(weights) >= min_weight


def _apply_component_filter(graph: nk.Graph) -> np.ndarray:
    """Apply giant component filter."""
    if graph.isDirected():
        # For directed graphs, use weakly connected components
        cc = nk.components.WeaklyConnectedComponents(graph)
    else:
        cc = nk.components.ConnectedComponents(graph)
    
    cc.run()
    component_sizes = cc.getComponentSizes()
    
    if not component_sizes:
        return np.zeros(graph.numberOfNodes(), dtype=bool)
    
    largest_component_id = max(component_sizes, key=component_sizes.get)
    
    mask = np.zeros(graph.numberOfNodes(), dtype=bool)
    for node in range(graph.numberOfNodes()):
        if cc.componentOfNode(node) == largest_component_id:
            mask[node] = True
    
    return mask


def _apply_node_inclusion_filter(
    graph: nk.Graph, 
    id_mapper: IDMapper, 
    node_list: List[str]
) -> np.ndarray:
    """Apply node inclusion filter."""
    mask = np.zeros(graph.numberOfNodes(), dtype=bool)
    
    for original_id in node_list:
        try:
            internal_id = id_mapper.get_internal(original_id)
            if internal_id < graph.numberOfNodes():
                mask[internal_id] = True
        except KeyError:
            logger.warning(f"Node {original_id} not found in graph")
    
    return mask


def _apply_node_exclusion_filter(
    graph: nk.Graph,
    id_mapper: IDMapper,
    node_list: List[str]
) -> np.ndarray:
    """Apply node exclusion filter."""
    mask = np.ones(graph.numberOfNodes(), dtype=bool)
    
    for original_id in node_list:
        try:
            internal_id = id_mapper.get_internal(original_id)
            if internal_id < graph.numberOfNodes():
                mask[internal_id] = False
        except KeyError:
            logger.warning(f"Node {original_id} not found in graph")
    
    return mask


def _apply_centrality_filter(
    graph: nk.Graph,
    id_mapper: IDMapper,
    centrality_config: Dict[str, Any]
) -> np.ndarray:
    """Apply centrality-based filter."""
    from .analysis import extract_centrality
    
    metric = centrality_config["metric"]
    min_value = centrality_config["min_value"]
    
    # Calculate centrality
    centrality_df = extract_centrality(
        graph, id_mapper, metrics=[metric], normalized=True, n_jobs=1
    )
    
    # Create mask based on threshold
    centrality_col = f"{metric}_centrality"
    centrality_values = centrality_df[centrality_col].to_list()
    
    mask = np.array(centrality_values) >= min_value
    return mask


def _combine_masks(masks: List[np.ndarray], combine: str) -> np.ndarray:
    """Combine multiple boolean masks using AND or OR logic."""
    if not masks:
        return None
    
    if len(masks) == 1:
        return masks[0]
    
    combined = masks[0].copy()
    for mask in masks[1:]:
        if combine == "and":
            combined = combined & mask
        elif combine == "or":
            combined = combined | mask
        else:
            raise ValueError(f"Invalid combine operation: {combine}")
    
    return combined


def _apply_masks_to_graph(
    graph: nk.Graph,
    id_mapper: IDMapper,
    node_mask: Optional[np.ndarray],
    edge_mask: Optional[np.ndarray]
) -> Tuple[nk.Graph, IDMapper]:
    """Apply node and edge masks to create filtered graph."""
    # Start with all nodes if no node mask
    if node_mask is None:
        kept_nodes = set(range(graph.numberOfNodes()))
    else:
        kept_nodes = set(np.where(node_mask)[0])
    
    # Apply edge mask by removing filtered edges
    if edge_mask is not None:
        # Get edges that should be removed
        edges_to_remove = []
        edge_idx = 0
        for u, v in graph.iterEdges():
            if not edge_mask[edge_idx]:
                edges_to_remove.append((u, v))
            edge_idx += 1
        
        # Remove edges (this may isolate some nodes)
        filtered_graph = nk.Graph(graph)
        for u, v in edges_to_remove:
            if filtered_graph.hasEdge(u, v):
                filtered_graph.removeEdge(u, v)
    else:
        filtered_graph = nk.Graph(graph)
    
    # Remove nodes not in mask
    nodes_to_remove = []
    for node in range(graph.numberOfNodes()):
        if node not in kept_nodes:
            nodes_to_remove.append(node)
    
    for node in nodes_to_remove:
        if filtered_graph.hasNode(node):
            filtered_graph.removeNode(node)
    
    # Create updated ID mapper
    updated_mapper = IDMapper()
    for internal_id in range(graph.numberOfNodes()):
        if internal_id in kept_nodes and filtered_graph.hasNode(internal_id):
            try:
                original_id = id_mapper.get_original(internal_id)
                updated_mapper.add_mapping(original_id, internal_id)
            except KeyError:
                pass
    
    return filtered_graph, updated_mapper


# Backbone extraction methods

def _apply_disparity_filter(
    graph: nk.Graph,
    id_mapper: IDMapper,
    alpha: float,
    target_edges: Optional[int],
    keep_disconnected: bool
) -> Tuple[nk.Graph, IDMapper]:
    """Apply disparity filter for backbone extraction."""
    logger.debug(f"Applying disparity filter with alpha={alpha}")
    
    # Collect edge data
    edges_data = []
    for u, v in graph.iterEdges():
        weight = graph.weight(u, v)
        edges_data.append((u, v, weight))
    
    if not edges_data:
        return graph, id_mapper
    
    # Convert to numpy arrays for efficiency
    sources = np.array([e[0] for e in edges_data])
    targets = np.array([e[1] for e in edges_data])
    weights = np.array([e[2] for e in edges_data])
    
    # Calculate degree sums for each node
    n_nodes = graph.numberOfNodes()
    degree_sums = np.zeros(n_nodes)
    degrees = np.zeros(n_nodes, dtype=int)
    
    for i, (u, v, w) in enumerate(edges_data):
        degree_sums[u] += w
        degrees[u] += 1
        if not graph.isDirected():
            degree_sums[v] += w
            degrees[v] += 1
    
    # Calculate normalized weights (p_ij)
    normalized_weights = np.zeros_like(weights)
    for i, (u, v, w) in enumerate(edges_data):
        if degree_sums[u] > 0:
            normalized_weights[i] = w / degree_sums[u]
        else:
            normalized_weights[i] = 0.0
    
    # Calculate disparity scores (α_ij)
    alpha_scores = np.ones_like(weights)
    for i, (u, v, w) in enumerate(edges_data):
        k = degrees[u]
        p_ij = normalized_weights[i]
        
        if k > 1 and 0 < p_ij < 1:
            # Use numerically stable computation
            try:
                log_alpha = (k - 1) * np.log(1 - p_ij)
                if log_alpha > -700:  # Prevent underflow
                    alpha_scores[i] = np.exp(log_alpha)
                else:
                    alpha_scores[i] = 0.0
            except (ValueError, OverflowError):
                alpha_scores[i] = 0.0
        elif k == 1:
            alpha_scores[i] = 1.0
        else:
            alpha_scores[i] = 0.0
    
    # Filter edges based on significance
    significant_edges = alpha_scores < alpha
    
    # Apply target_edges constraint if specified
    if target_edges is not None and np.sum(significant_edges) > target_edges:
        # Keep edges with lowest alpha scores
        sorted_indices = np.argsort(alpha_scores)
        significant_edges = np.zeros_like(significant_edges)
        significant_edges[sorted_indices[:target_edges]] = True
    
    # Create backbone graph
    backbone_graph = nk.Graph(
        directed=graph.isDirected(),
        weighted=graph.isWeighted()
    )
    
    # Add nodes
    for node in range(n_nodes):
        backbone_graph.addNode()
    
    # Add significant edges
    edges_kept = 0
    for i, (u, v, w) in enumerate(edges_data):
        if significant_edges[i]:
            backbone_graph.addEdge(u, v, w)
            edges_kept += 1
    
    logger.debug(f"Disparity filter kept {edges_kept}/{len(edges_data)} edges")
    
    # Remove disconnected nodes if requested
    if not keep_disconnected:
        nodes_to_remove = []
        for node in range(backbone_graph.numberOfNodes()):
            if backbone_graph.degree(node) == 0:
                nodes_to_remove.append(node)
        
        for node in nodes_to_remove:
            backbone_graph.removeNode(node)
    
    # Update ID mapper
    updated_mapper = IDMapper()
    for internal_id in range(n_nodes):
        if backbone_graph.hasNode(internal_id):
            try:
                original_id = id_mapper.get_original(internal_id)
                updated_mapper.add_mapping(original_id, internal_id)
            except KeyError:
                pass
    
    return backbone_graph, updated_mapper


def _apply_weight_threshold(
    graph: nk.Graph,
    id_mapper: IDMapper,
    target_edges: Optional[int],
    keep_disconnected: bool
) -> Tuple[nk.Graph, IDMapper]:
    """Apply weight threshold for backbone extraction."""
    if not graph.isWeighted():
        logger.warning("Weight threshold applied to unweighted graph")
        return graph, id_mapper
    
    # Collect edge weights
    weights = []
    edges = []
    for u, v in graph.iterEdges():
        weight = graph.weight(u, v)
        weights.append(weight)
        edges.append((u, v, weight))
    
    weights = np.array(weights)
    
    # Determine threshold
    if target_edges is not None:
        if target_edges >= len(weights):
            threshold = 0.0
        else:
            sorted_weights = np.sort(weights)[::-1]  # Descending order
            threshold = sorted_weights[target_edges - 1]
    else:
        threshold = np.median(weights)
    
    logger.debug(f"Weight threshold: {threshold}")
    
    # Create backbone graph
    backbone_graph = nk.Graph(
        directed=graph.isDirected(),
        weighted=graph.isWeighted()
    )
    
    # Add nodes
    for node in range(graph.numberOfNodes()):
        backbone_graph.addNode()
    
    # Add edges above threshold
    edges_kept = 0
    for u, v, w in edges:
        if w >= threshold:
            backbone_graph.addEdge(u, v, w)
            edges_kept += 1
    
    logger.debug(f"Weight filter kept {edges_kept}/{len(edges)} edges")
    
    # Remove disconnected nodes if requested
    if not keep_disconnected:
        nodes_to_remove = []
        for node in range(backbone_graph.numberOfNodes()):
            if backbone_graph.degree(node) == 0:
                nodes_to_remove.append(node)
        
        for node in nodes_to_remove:
            backbone_graph.removeNode(node)
    
    # Update ID mapper
    updated_mapper = IDMapper()
    for internal_id in range(graph.numberOfNodes()):
        if backbone_graph.hasNode(internal_id):
            try:
                original_id = id_mapper.get_original(internal_id)
                updated_mapper.add_mapping(original_id, internal_id)
            except KeyError:
                pass
    
    return backbone_graph, updated_mapper


def _apply_degree_threshold(
    graph: nk.Graph,
    id_mapper: IDMapper,
    target_nodes: Optional[int],
    keep_disconnected: bool
) -> Tuple[nk.Graph, IDMapper]:
    """Apply degree threshold for backbone extraction."""
    # Calculate degrees
    degrees = np.array([graph.degree(u) for u in range(graph.numberOfNodes())])
    
    # Determine threshold
    if target_nodes is not None:
        if target_nodes >= len(degrees):
            threshold = 0
        else:
            sorted_degrees = np.sort(degrees)[::-1]  # Descending order
            threshold = sorted_degrees[target_nodes - 1]
    else:
        threshold = int(np.median(degrees))
    
    logger.debug(f"Degree threshold: {threshold}")
    
    # Create node mask
    keep_nodes = degrees >= threshold
    
    # Apply filter
    filtered_graph, updated_mapper = _apply_masks_to_graph(
        graph, id_mapper, keep_nodes, None
    )
    
    nodes_kept = np.sum(keep_nodes)
    logger.debug(f"Degree filter kept {nodes_kept}/{len(degrees)} nodes")
    
    return filtered_graph, updated_mapper


def get_backbone_statistics(
    original_graph: nk.Graph,
    backbone_graph: nk.Graph
) -> Dict[str, Any]:
    """
    Calculate statistics comparing original graph to backbone.
    
    Parameters
    ----------
    original_graph : nk.Graph
        Original graph before backbone extraction
    backbone_graph : nk.Graph
        Backbone graph after filtering
        
    Returns
    -------
    Dict[str, Any]
        Statistics including compression ratios and structural measures
    """
    stats = {
        "original_nodes": original_graph.numberOfNodes(),
        "original_edges": original_graph.numberOfEdges(),
        "backbone_nodes": backbone_graph.numberOfNodes(),
        "backbone_edges": backbone_graph.numberOfEdges(),
        "node_retention": backbone_graph.numberOfNodes() / max(original_graph.numberOfNodes(), 1),
        "edge_retention": backbone_graph.numberOfEdges() / max(original_graph.numberOfEdges(), 1),
        "compression_ratio": (
            (original_graph.numberOfNodes() + original_graph.numberOfEdges()) /
            max(backbone_graph.numberOfNodes() + backbone_graph.numberOfEdges(), 1)
        )
    }
    
    # Calculate density change
    original_density = (
        original_graph.numberOfEdges() / 
        max(original_graph.numberOfNodes() * (original_graph.numberOfNodes() - 1) / 2, 1)
    )
    backbone_density = (
        backbone_graph.numberOfEdges() / 
        max(backbone_graph.numberOfNodes() * (backbone_graph.numberOfNodes() - 1) / 2, 1)
    )
    
    stats["original_density"] = original_density
    stats["backbone_density"] = backbone_density
    stats["density_ratio"] = backbone_density / max(original_density, 1e-10)
    
    return stats