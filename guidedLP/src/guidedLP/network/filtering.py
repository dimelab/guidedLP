"""
Network filtering module for the Guided Label Propagation library.

This module provides :func:`filter_graph`, which applies one or more filters
(degree bounds, weight bounds, component selection, node inclusion/exclusion,
centrality thresholds) to a NetworkIt graph. Backbone-extraction methods
live in :mod:`guidedLP.network.backboning`.
"""

from typing import List, Dict, Any, Optional, Tuple

import polars as pl
import networkit as nk
import numpy as np

from guidedLP.common.id_mapper import IDMapper
from guidedLP.common.exceptions import (
    ComputationError,
    ValidationError,
)
from guidedLP.common.logging_config import get_logger, log_function_entry, LoggingTimer

logger = get_logger(__name__)

# Available filter types
SUPPORTED_FILTER_TYPES = [
    "min_degree", "max_degree", "min_weight", "giant_component_only",
    "nodes", "exclude_nodes", "centrality"
]


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
