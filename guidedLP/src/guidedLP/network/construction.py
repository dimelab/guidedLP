"""
Network construction module for the Guided Label Propagation library.

This module provides functionality for constructing NetworkIt graphs from edge lists
while preserving original node IDs and supporting various graph types including
directed, undirected, weighted, unweighted, and bipartite graphs.
"""

from typing import Union, Tuple, Optional, List, Dict, Any, Set
import warnings
from pathlib import Path

import polars as pl
import networkit as nk
import numpy as np

from guidedLP.common.id_mapper import IDMapper
from guidedLP.common.exceptions import (
    GraphConstructionError,
    ValidationError,
    DataFormatError,
    ConfigurationError,
    validate_parameter
)
from guidedLP.common.validators import validate_edgelist_dataframe
from guidedLP.common.logging_config import get_logger, log_function_entry, LoggingTimer

logger = get_logger(__name__)


def build_graph_from_edgelist(
    edgelist: Union[str, pl.DataFrame],
    source_col: str = "source",
    target_col: str = "target",
    weight_col: Optional[str] = None,
    directed: bool = False,
    bipartite: bool = False,
    auto_weight: bool = True,
    allow_self_loops: bool = True,
    remove_duplicates: bool = False
) -> Tuple[nk.Graph, IDMapper]:
    """
    Construct a NetworkIt graph from an edge list with ID preservation.
    
    This function creates a NetworkIt graph from edge list data while preserving
    original node identifiers through an ID mapping system. It supports various
    graph types and handles weight calculation from duplicate edges.
    
    Parameters
    ----------
    edgelist : Union[str, pl.DataFrame]
        Path to CSV file or Polars DataFrame containing edge data
    source_col : str, default "source"
        Name of the source node column in the edge list
    target_col : str, default "target"  
        Name of the target node column in the edge list
    weight_col : str, optional
        Name of the edge weight column. If None and auto_weight is True,
        weights will be calculated from duplicate edge counts
    directed : bool, default False
        If True, create a directed graph; otherwise undirected
    bipartite : bool, default False
        If True, validate that source and target nodes form distinct sets
        (bipartite graph structure)
    auto_weight : bool, default True
        If True and weight_col is None, calculate edge weights by counting
        duplicate (source, target) pairs
    allow_self_loops : bool, default True
        If True, allow edges from a node to itself; otherwise remove them
    remove_duplicates : bool, default False
        If True, remove duplicate edges before processing (keeps first occurrence)
        
    Returns
    -------
    graph : nk.Graph
        Constructed NetworkIt graph object
    id_mapper : IDMapper
        Bidirectional mapping between original IDs and NetworkIt internal IDs
        
    Raises
    ------
    ValidationError
        If edge list data is invalid or missing required columns
    GraphConstructionError
        If graph construction fails due to NetworkIt issues
    DataFormatError
        If input file cannot be read or parsed
        
    Examples
    --------
    >>> import polars as pl
    >>> edges = pl.DataFrame({
    ...     "source": ["A", "B", "C", "A"],
    ...     "target": ["B", "C", "A", "B"]
    ... })
    >>> graph, id_mapper = build_graph_from_edgelist(edges)
    >>> graph.numberOfNodes()
    3
    >>> graph.numberOfEdges() 
    4
    
    >>> # Directed graph with weights
    >>> weighted_edges = pl.DataFrame({
    ...     "from": ["A", "B"], 
    ...     "to": ["B", "A"],
    ...     "weight": [1.5, 2.0]
    ... })
    >>> graph, mapper = build_graph_from_edgelist(
    ...     weighted_edges, 
    ...     source_col="from", 
    ...     target_col="to",
    ...     weight_col="weight", 
    ...     directed=True
    ... )
    
    Notes
    -----
    Time Complexity: O(E) where E is the number of edges
    Space Complexity: O(V + E) where V is the number of unique nodes
    
    The function performs the following steps:
    1. Load edge list data using Polars (lazy evaluation for CSV files)
    2. Validate column existence and data quality
    3. Calculate weights from duplicates if auto_weight is enabled
    4. Extract unique node IDs and create ID mapping
    5. Construct NetworkIt graph with appropriate configuration
    6. Add edges with weights to the graph
    
    For bipartite graphs, the function validates that source and target nodes
    form completely distinct sets with no overlap.
    """
    log_function_entry("build_graph_from_edgelist", 
                      edgelist=type(edgelist).__name__, 
                      directed=directed, bipartite=bipartite)
    
    with LoggingTimer("build_graph_from_edgelist"):
        try:
            # Step 1: Load edge list data
            df = _load_edge_list(edgelist)
            
            # Step 2: Validate edge list structure and data
            _validate_edge_list(df, source_col, target_col, weight_col)
            
            # Step 3: Handle empty edge list
            if df.is_empty():
                warnings.warn("Empty edge list provided. Creating empty graph.")
                empty_graph = nk.Graph(0, weighted=(weight_col is not None), directed=directed)
                empty_mapper = IDMapper()
                return empty_graph, empty_mapper
            
            # Step 4: Process edges (weights, duplicates, self-loops)
            processed_df = _process_edges(
                df, source_col, target_col, weight_col, 
                auto_weight, allow_self_loops, remove_duplicates
            )
            
            # Step 5: Create ID mapping
            id_mapper = _create_id_mapping(processed_df, source_col, target_col)
            
            # Step 6: Validate bipartite structure if requested
            if bipartite:
                _validate_bipartite_structure(processed_df, source_col, target_col, id_mapper)
            
            # Step 7: Construct NetworkIt graph
            graph = _construct_graph(processed_df, id_mapper, source_col, target_col, 
                                   weight_col, directed, bipartite)
            
            logger.info("Graph construction completed: %d nodes, %d edges, directed=%s, weighted=%s",
                       graph.numberOfNodes(), graph.numberOfEdges(), directed, 
                       weight_col is not None)
            
            return graph, id_mapper
            
        except Exception as e:
            if isinstance(e, (ValidationError, GraphConstructionError, DataFormatError)):
                raise
            else:
                raise GraphConstructionError(
                    f"Unexpected error during graph construction: {str(e)}",
                    operation="build_graph_from_edgelist",
                    cause=e
                )


def _load_edge_list(edgelist: Union[str, pl.DataFrame]) -> pl.DataFrame:
    """
    Load edge list from file path or return DataFrame as-is.
    
    Parameters
    ----------
    edgelist : Union[str, pl.DataFrame]
        File path to CSV or DataFrame
        
    Returns
    -------
    pl.DataFrame
        Loaded edge list data
        
    Raises
    ------
    DataFormatError
        If file cannot be read or parsed
    """
    if isinstance(edgelist, str):
        try:
            # Convert to Path for better handling
            file_path = Path(edgelist)
            
            if not file_path.exists():
                raise DataFormatError(
                    f"Edge list file not found: {edgelist}",
                    format_type="CSV",
                    file_path=edgelist
                )
            
            # Use Polars lazy loading for efficiency
            logger.debug("Loading edge list from file: %s", edgelist)
            df = pl.read_csv(file_path)
            
        except pl.exceptions.ComputeError as e:
            raise DataFormatError(
                f"Failed to parse CSV file: {str(e)}",
                format_type="CSV",
                file_path=edgelist,
                cause=e
            )
        except Exception as e:
            raise DataFormatError(
                f"Error reading file: {str(e)}",
                format_type="CSV", 
                file_path=edgelist,
                cause=e
            )
    
    elif isinstance(edgelist, pl.DataFrame):
        df = edgelist
    else:
        raise DataFormatError(
            f"Invalid edgelist type: {type(edgelist)}. Expected str or pl.DataFrame",
            format_type="DataFrame"
        )
    
    return df


def _validate_edge_list(
    df: pl.DataFrame,
    source_col: str,
    target_col: str,
    weight_col: Optional[str]
) -> None:
    """
    Validate edge list DataFrame structure and data quality.
    
    Parameters
    ----------
    df : pl.DataFrame
        Edge list DataFrame
    source_col : str
        Source column name
    target_col : str
        Target column name  
    weight_col : str, optional
        Weight column name
        
    Raises
    ------
    ValidationError
        If validation fails
    """
    logger.debug("Validating edge list with %d rows", len(df))
    
    # Use the existing validator from common module
    validate_edgelist_dataframe(
        df,
        source_col=source_col,
        target_col=target_col,
        weight_col=weight_col,
        allow_self_loops=True,  # Will be handled separately
        allow_duplicates=True   # Will be handled separately
    )


def _process_edges(
    df: pl.DataFrame,
    source_col: str,
    target_col: str,
    weight_col: Optional[str],
    auto_weight: bool,
    allow_self_loops: bool,
    remove_duplicates: bool
) -> pl.DataFrame:
    """
    Process edges: calculate weights, handle duplicates, filter self-loops.
    
    Parameters
    ----------
    df : pl.DataFrame
        Input edge list
    source_col : str
        Source column name
    target_col : str
        Target column name
    weight_col : str, optional
        Weight column name
    auto_weight : bool
        Whether to calculate weights from duplicates
    allow_self_loops : bool
        Whether to allow self-loops
    remove_duplicates : bool
        Whether to remove duplicate edges
        
    Returns
    -------
    pl.DataFrame
        Processed edge list with weights
    """
    logger.debug("Processing edges: auto_weight=%s, allow_self_loops=%s, remove_duplicates=%s",
                auto_weight, allow_self_loops, remove_duplicates)
    
    processed_df = df.clone()
    
    # Remove self-loops if not allowed
    if not allow_self_loops:
        initial_count = len(processed_df)
        processed_df = processed_df.filter(pl.col(source_col) != pl.col(target_col))
        removed_count = initial_count - len(processed_df)
        if removed_count > 0:
            logger.info("Removed %d self-loop edges", removed_count)
    
    # Handle weights and duplicates
    if weight_col is None and auto_weight:
        # Calculate weights from duplicate edge counts
        logger.debug("Calculating automatic weights from duplicate edges")
        processed_df = processed_df.group_by([source_col, target_col]).agg(
            pl.len().alias("weight")
        ).with_columns(pl.col("weight").cast(pl.Float64))
        weight_col = "weight"
        
    elif weight_col is not None and not remove_duplicates:
        # Sum weights for duplicate edges
        logger.debug("Aggregating weights for duplicate edges")
        processed_df = processed_df.group_by([source_col, target_col]).agg(
            pl.col(weight_col).sum().alias(weight_col)
        )
        
    elif remove_duplicates:
        # Remove duplicate edges (keep first occurrence)
        initial_count = len(processed_df)
        processed_df = processed_df.unique(subset=[source_col, target_col], keep="first")
        removed_count = initial_count - len(processed_df)
        if removed_count > 0:
            logger.info("Removed %d duplicate edges", removed_count)
    
    return processed_df


def _create_id_mapping(
    df: pl.DataFrame,
    source_col: str,
    target_col: str
) -> IDMapper:
    """
    Create bidirectional ID mapping between original and NetworkIt IDs.
    
    Parameters
    ----------
    df : pl.DataFrame
        Edge list DataFrame
    source_col : str
        Source column name
    target_col : str
        Target column name
        
    Returns
    -------
    IDMapper
        Configured ID mapper
        
    Raises
    ------
    GraphConstructionError
        If ID mapping creation fails
    """
    try:
        # Extract all unique node IDs from both source and target columns
        logger.debug("Creating ID mapping from edge list")
        
        source_ids = df[source_col].unique().to_list()
        target_ids = df[target_col].unique().to_list()
        
        # Union of source and target IDs
        all_ids = list(set(source_ids + target_ids))
        all_ids.sort(key=str)  # Sort for deterministic mapping
        
        # Create ID mapper
        id_mapper = IDMapper()
        for internal_id, original_id in enumerate(all_ids):
            id_mapper.add_mapping(original_id, internal_id)
        
        logger.debug("Created ID mapping for %d unique nodes", len(all_ids))
        return id_mapper
        
    except Exception as e:
        raise GraphConstructionError(
            f"Failed to create ID mapping: {str(e)}",
            operation="create_id_mapping",
            cause=e
        )


def _validate_bipartite_structure(
    df: pl.DataFrame,
    source_col: str,
    target_col: str,
    id_mapper: IDMapper
) -> None:
    """
    Validate that the graph has bipartite structure (disjoint source/target sets).
    
    Parameters
    ----------
    df : pl.DataFrame
        Edge list DataFrame
    source_col : str
        Source column name
    target_col : str
        Target column name
    id_mapper : IDMapper
        ID mapper containing all nodes
        
    Raises
    ------
    GraphConstructionError
        If graph is not bipartite
    """
    logger.debug("Validating bipartite graph structure")
    
    source_ids = set(df[source_col].unique().to_list())
    target_ids = set(df[target_col].unique().to_list())
    
    # Check for overlap between source and target sets
    overlap = source_ids.intersection(target_ids)
    
    if overlap:
        raise GraphConstructionError(
            f"Graph is not bipartite: {len(overlap)} nodes appear in both source and target",
            graph_type="bipartite",
            details={
                "overlapping_nodes": list(overlap)[:10],  # Show first 10
                "total_overlap": len(overlap),
                "source_partition_size": len(source_ids),
                "target_partition_size": len(target_ids)
            }
        )
    
    logger.debug("Bipartite structure validated: %d source nodes, %d target nodes", 
                len(source_ids), len(target_ids))


def _construct_graph(
    df: pl.DataFrame,
    id_mapper: IDMapper,
    source_col: str,
    target_col: str,
    weight_col: Optional[str],
    directed: bool,
    bipartite: bool
) -> nk.Graph:
    """
    Construct NetworkIt graph from processed edge list.
    
    Parameters
    ----------
    df : pl.DataFrame
        Processed edge list with weights
    id_mapper : IDMapper
        ID mapping between original and internal IDs
    source_col : str
        Source column name
    target_col : str
        Target column name
    weight_col : str, optional
        Weight column name
    directed : bool
        Whether graph is directed
    bipartite : bool
        Whether graph is bipartite
        
    Returns
    -------
    nk.Graph
        Constructed NetworkIt graph
        
    Raises
    ------
    GraphConstructionError
        If graph construction fails
    """
    try:
        # Create NetworkIt graph
        n_nodes = id_mapper.size()
        weighted = weight_col is not None
        
        logger.debug("Creating NetworkIt graph: %d nodes, weighted=%s, directed=%s", 
                    n_nodes, weighted, directed)
        
        graph = nk.Graph(n_nodes, weighted=weighted, directed=directed)
        
        # Add edges to graph
        for row in df.iter_rows(named=True):
            source_id = id_mapper.get_internal(row[source_col])
            target_id = id_mapper.get_internal(row[target_col])
            
            if weighted:
                weight = float(row[weight_col])
                graph.addEdge(source_id, target_id, weight)
            else:
                graph.addEdge(source_id, target_id)
        
        # Set bipartite attribute if requested (NetworkIt doesn't have built-in bipartite support)
        if bipartite:
            # Store bipartite information in graph attributes (if NetworkIt supports it)
            # This is mainly for documentation purposes
            pass
        
        logger.debug("Graph construction completed: %d edges added", graph.numberOfEdges())
        return graph
        
    except Exception as e:
        raise GraphConstructionError(
            f"Failed to construct NetworkIt graph: {str(e)}",
            operation="construct_graph",
            node_count=id_mapper.size() if id_mapper else None,
            edge_count=len(df) if df is not None else None,
            cause=e
        )


def get_graph_info(graph: nk.Graph, id_mapper: IDMapper) -> Dict[str, Any]:
    """
    Get comprehensive information about a constructed graph.
    
    Parameters
    ----------
    graph : nk.Graph
        NetworkIt graph
    id_mapper : IDMapper
        ID mapper for the graph
        
    Returns
    -------
    Dict[str, Any]
        Dictionary containing graph statistics and properties
        
    Examples
    --------
    >>> graph, mapper = build_graph_from_edgelist(edges)
    >>> info = get_graph_info(graph, mapper)
    >>> print(f"Nodes: {info['num_nodes']}, Edges: {info['num_edges']}")
    """
    return {
        "num_nodes": graph.numberOfNodes(),
        "num_edges": graph.numberOfEdges(),
        "directed": graph.isDirected(),
        "weighted": graph.isWeighted(),
        "has_self_loops": graph.numberOfSelfLoops() > 0,
        "num_self_loops": graph.numberOfSelfLoops(),
        "density": graph.density() if graph.numberOfNodes() > 1 else 0.0,
        "node_id_mapping_size": id_mapper.size(),
        "is_connected": nk.components.ConnectedComponents(graph).run().numberOfComponents() == 1
    }


def validate_graph_construction(
    graph: nk.Graph,
    id_mapper: IDMapper,
    expected_nodes: Optional[int] = None,
    expected_edges: Optional[int] = None
) -> None:
    """
    Validate that graph construction was successful.
    
    Parameters
    ----------
    graph : nk.Graph
        Constructed graph to validate
    id_mapper : IDMapper
        ID mapper to validate
    expected_nodes : int, optional
        Expected number of nodes
    expected_edges : int, optional
        Expected number of edges
        
    Raises
    ------
    GraphConstructionError
        If validation fails
    """
    # Basic consistency checks
    if graph.numberOfNodes() != id_mapper.size():
        raise GraphConstructionError(
            "Inconsistent node count between graph and ID mapper",
            details={
                "graph_nodes": graph.numberOfNodes(),
                "mapper_size": id_mapper.size()
            }
        )
    
    # Check expected counts if provided
    if expected_nodes is not None and graph.numberOfNodes() != expected_nodes:
        raise GraphConstructionError(
            f"Unexpected node count: expected {expected_nodes}, got {graph.numberOfNodes()}"
        )
    
    if expected_edges is not None and graph.numberOfEdges() != expected_edges:
        raise GraphConstructionError(
            f"Unexpected edge count: expected {expected_edges}, got {graph.numberOfEdges()}"
        )
    
    logger.debug("Graph construction validation passed")


def project_bipartite(
    graph: nk.Graph,
    id_mapper: IDMapper,
    projection_mode: str = "source",
    weight_method: str = "count"
) -> Tuple[nk.Graph, IDMapper]:
    """
    Project bipartite graph to unipartite by connecting nodes with shared neighbors.
    
    This function projects a bipartite graph onto one of its partitions, creating
    a unipartite graph where nodes are connected if they share neighbors in the
    other partition. Edge weights are calculated using various similarity measures.
    
    Parameters
    ----------
    graph : nk.Graph
        Bipartite NetworkIt graph to project
    id_mapper : IDMapper
        Original ID mapper containing all bipartite graph nodes
    projection_mode : str, default "source"
        Which partition to project onto:
        - "source": Project onto the source partition (nodes that appear as sources)
        - "target": Project onto the target partition (nodes that appear as targets)
    weight_method : str, default "count"
        Method for calculating projection weights:
        - "count": Number of shared neighbors
        - "jaccard": Jaccard similarity of neighbor sets
        - "overlap": Overlap coefficient (min of neighbor set sizes)
        
    Returns
    -------
    projected_graph : nk.Graph
        Unipartite NetworkIt graph containing only projected nodes
    new_id_mapper : IDMapper
        Updated ID mapper containing only nodes in the projected graph
        
    Raises
    ------
    GraphConstructionError
        If the input graph is not bipartite or projection fails
    ConfigurationError
        If invalid projection_mode or weight_method is specified
        
    Examples
    --------
    >>> # Create bipartite graph (users -> items)
    >>> edges = pl.DataFrame({
    ...     "user": ["u1", "u1", "u2", "u2", "u3"],
    ...     "item": ["i1", "i2", "i1", "i3", "i2"]
    ... })
    >>> graph, mapper = build_graph_from_edgelist(
    ...     edges, source_col="user", target_col="item", bipartite=True
    ... )
    >>> 
    >>> # Project onto users (connect users who like same items)
    >>> user_graph, user_mapper = project_bipartite(graph, mapper, "source", "count")
    >>> 
    >>> # Project onto items (connect items liked by same users)  
    >>> item_graph, item_mapper = project_bipartite(graph, mapper, "target", "jaccard")
    
    Notes
    -----
    Time Complexity: O(N² × D) worst case, where N is the size of the projection
    partition and D is the average degree in the other partition.
    
    Space Complexity: O(N²) in the worst case for a fully connected projection.
    
    The function identifies bipartite partitions by analyzing edge patterns rather
    than requiring explicit partition information. This makes it robust to various
    bipartite graph construction approaches.
    
    Weight Methods:
    - **Count**: Simple count of shared neighbors. Fast and intuitive.
    - **Jaccard**: |A ∩ B| / |A ∪ B|. Normalized similarity measure.
    - **Overlap**: |A ∩ B| / min(|A|, |B|). Asymmetric similarity measure.
    """
    log_function_entry("project_bipartite", 
                      projection_mode=projection_mode, weight_method=weight_method)
    
    # Validate parameters
    validate_parameter(projection_mode, ["source", "target"], "projection_mode", "project_bipartite")
    validate_parameter(weight_method, ["count", "jaccard", "overlap"], "weight_method", "project_bipartite")
    
    with LoggingTimer("project_bipartite"):
        try:
            # Step 1: Identify bipartite partitions
            source_partition, target_partition = _identify_bipartite_partitions(graph, id_mapper)
            
            # Step 2: Select projection partition and other partition
            if projection_mode == "source":
                projection_partition = source_partition
                other_partition = target_partition
            else:  # projection_mode == "target"
                projection_partition = target_partition
                other_partition = source_partition
            
            logger.info("Projecting bipartite graph: %d nodes in projection partition, %d in other partition",
                       len(projection_partition), len(other_partition))
            
            # Step 3: Build neighbor mapping for projection partition nodes
            neighbor_map = _build_neighbor_mapping(graph, id_mapper, projection_partition, other_partition)
            
            # Step 4: Create projected graph and new ID mapper
            projected_graph, new_id_mapper = _create_projected_graph(
                projection_partition, neighbor_map, weight_method, graph.isWeighted()
            )
            
            logger.info("Bipartite projection completed: %d nodes, %d edges",
                       projected_graph.numberOfNodes(), projected_graph.numberOfEdges())
            
            return projected_graph, new_id_mapper
            
        except Exception as e:
            if isinstance(e, (GraphConstructionError, ValidationError)):
                raise
            else:
                raise GraphConstructionError(
                    f"Bipartite projection failed: {str(e)}",
                    operation="project_bipartite",
                    cause=e
                )


def _identify_bipartite_partitions(graph: nk.Graph, id_mapper: IDMapper) -> Tuple[List[Any], List[Any]]:
    """
    Identify the two partitions of a bipartite graph.
    
    Parameters
    ----------
    graph : nk.Graph
        NetworkIt graph (should be bipartite)
    id_mapper : IDMapper
        ID mapper for the graph
        
    Returns
    -------
    source_partition : List[Any]
        List of original IDs in the source partition
    target_partition : List[Any]
        List of original IDs in the target partition
        
    Raises
    ------
    GraphConstructionError
        If graph is not bipartite
    """
    logger.debug("Identifying bipartite partitions")
    
    if graph.numberOfNodes() == 0:
        return [], []
    
    # Use NetworkIt's bipartiteness check if available, otherwise implement our own
    try:
        # Try to detect bipartiteness by attempting 2-coloring
        source_partition = []
        target_partition = []
        
        # Color nodes using BFS to detect bipartiteness
        colors = {}  # node -> color (0 or 1)
        queue = []
        
        # Start with first node
        first_node = 0
        colors[first_node] = 0
        queue.append(first_node)
        source_partition.append(id_mapper.get_original(first_node))
        
        while queue:
            current = queue.pop(0)
            current_color = colors[current]
            
            # Check all neighbors
            for neighbor in graph.iterNeighborsOf(current):
                if neighbor in colors:
                    # Check if coloring is consistent
                    if colors[neighbor] == current_color:
                        raise GraphConstructionError(
                            "Graph is not bipartite: found odd cycle",
                            operation="identify_bipartite_partitions",
                            details={"conflicting_nodes": [current, neighbor]}
                        )
                else:
                    # Color neighbor with opposite color
                    neighbor_color = 1 - current_color
                    colors[neighbor] = neighbor_color
                    queue.append(neighbor)
                    
                    if neighbor_color == 0:
                        source_partition.append(id_mapper.get_original(neighbor))
                    else:
                        target_partition.append(id_mapper.get_original(neighbor))
        
        # Handle disconnected components
        for node in range(graph.numberOfNodes()):
            if node not in colors:
                # Start new component
                colors[node] = 0
                queue.append(node)
                source_partition.append(id_mapper.get_original(node))
                
                while queue:
                    current = queue.pop(0)
                    current_color = colors[current]
                    
                    for neighbor in graph.iterNeighborsOf(current):
                        if neighbor in colors:
                            if colors[neighbor] == current_color:
                                raise GraphConstructionError(
                                    "Graph is not bipartite: found odd cycle in disconnected component",
                                    operation="identify_bipartite_partitions"
                                )
                        else:
                            neighbor_color = 1 - current_color
                            colors[neighbor] = neighbor_color
                            queue.append(neighbor)
                            
                            if neighbor_color == 0:
                                source_partition.append(id_mapper.get_original(neighbor))
                            else:
                                target_partition.append(id_mapper.get_original(neighbor))
        
        logger.debug("Identified partitions: %d source nodes, %d target nodes",
                    len(source_partition), len(target_partition))
        
        return source_partition, target_partition
        
    except Exception as e:
        if isinstance(e, GraphConstructionError):
            raise
        else:
            raise GraphConstructionError(
                f"Failed to identify bipartite partitions: {str(e)}",
                operation="identify_bipartite_partitions",
                cause=e
            )


def _build_neighbor_mapping(
    graph: nk.Graph,
    id_mapper: IDMapper,
    projection_partition: List[Any],
    other_partition: List[Any]
) -> Dict[Any, Set[Any]]:
    """
    Build mapping from projection partition nodes to their neighbors in other partition.
    
    Parameters
    ----------
    graph : nk.Graph
        NetworkIt graph
    id_mapper : IDMapper
        ID mapper
    projection_partition : List[Any]
        Nodes to project (original IDs)
    other_partition : List[Any]
        Other partition nodes (original IDs)
        
    Returns
    -------
    Dict[Any, Set[Any]]
        Mapping from projection node to set of neighbor nodes in other partition
    """
    logger.debug("Building neighbor mapping for %d projection nodes", len(projection_partition))
    
    neighbor_map = {}
    other_partition_set = set(other_partition)
    
    for proj_node_orig in projection_partition:
        proj_node_internal = id_mapper.get_internal(proj_node_orig)
        neighbors = set()
        
        # Find neighbors in the other partition
        for neighbor_internal in graph.iterNeighborsOf(proj_node_internal):
            neighbor_orig = id_mapper.get_original(neighbor_internal)
            if neighbor_orig in other_partition_set:
                neighbors.add(neighbor_orig)
        
        neighbor_map[proj_node_orig] = neighbors
    
    return neighbor_map


def _create_projected_graph(
    projection_partition: List[Any],
    neighbor_map: Dict[Any, Set[Any]],
    weight_method: str,
    preserve_weights: bool
) -> Tuple[nk.Graph, IDMapper]:
    """
    Create the projected unipartite graph.
    
    Parameters
    ----------
    projection_partition : List[Any]
        Nodes in projection partition (original IDs)
    neighbor_map : Dict[Any, Set[Any]]
        Mapping from projection nodes to their neighbors
    weight_method : str
        Weight calculation method
    preserve_weights : bool
        Whether to create weighted graph
        
    Returns
    -------
    projected_graph : nk.Graph
        Projected unipartite graph
    new_id_mapper : IDMapper
        ID mapper for projected graph
    """
    logger.debug("Creating projected graph with %d nodes using %s weights",
                len(projection_partition), weight_method)
    
    # Create new ID mapper for projected nodes only
    new_id_mapper = IDMapper()
    sorted_projection = sorted(projection_partition, key=str)  # For deterministic mapping
    
    for internal_id, original_id in enumerate(sorted_projection):
        new_id_mapper.add_mapping(original_id, internal_id)
    
    # Create projected graph
    n_nodes = len(projection_partition)
    projected_graph = nk.Graph(n_nodes, weighted=True, directed=False)  # Always weighted for projections
    
    # Add edges between nodes that share neighbors
    for i, node1 in enumerate(sorted_projection):
        for j, node2 in enumerate(sorted_projection):
            if i >= j:  # Avoid duplicate edges and self-loops
                continue
            
            neighbors1 = neighbor_map[node1]
            neighbors2 = neighbor_map[node2]
            shared_neighbors = neighbors1.intersection(neighbors2)
            
            if len(shared_neighbors) > 0:
                # Calculate edge weight based on method
                weight = _calculate_projection_weight(
                    neighbors1, neighbors2, shared_neighbors, weight_method
                )
                
                if weight > 0:
                    internal1 = new_id_mapper.get_internal(node1)
                    internal2 = new_id_mapper.get_internal(node2)
                    projected_graph.addEdge(internal1, internal2, weight)
    
    return projected_graph, new_id_mapper


def _calculate_projection_weight(
    neighbors1: Set[Any],
    neighbors2: Set[Any],
    shared_neighbors: Set[Any],
    weight_method: str
) -> float:
    """
    Calculate projection edge weight using specified method.
    
    Parameters
    ----------
    neighbors1 : Set[Any]
        Neighbor set of first node
    neighbors2 : Set[Any]
        Neighbor set of second node
    shared_neighbors : Set[Any]
        Intersection of neighbor sets
    weight_method : str
        Weight calculation method
        
    Returns
    -------
    float
        Calculated weight
    """
    if weight_method == "count":
        return float(len(shared_neighbors))
    
    elif weight_method == "jaccard":
        union_size = len(neighbors1.union(neighbors2))
        if union_size == 0:
            return 0.0
        return len(shared_neighbors) / union_size
    
    elif weight_method == "overlap":
        min_size = min(len(neighbors1), len(neighbors2))
        if min_size == 0:
            return 0.0
        return len(shared_neighbors) / min_size
    
    else:
        # This shouldn't happen due to parameter validation, but be safe
        raise ValueError(f"Unknown weight method: {weight_method}")


def get_bipartite_info(graph: nk.Graph, id_mapper: IDMapper) -> Dict[str, Any]:
    """
    Get information about a bipartite graph structure.
    
    Parameters
    ----------
    graph : nk.Graph
        Graph to analyze (should be bipartite)
    id_mapper : IDMapper
        ID mapper for the graph
        
    Returns
    -------
    Dict[str, Any]
        Dictionary containing bipartite graph information
    """
    try:
        source_partition, target_partition = _identify_bipartite_partitions(graph, id_mapper)
        
        return {
            "is_bipartite": True,
            "source_partition_size": len(source_partition),
            "target_partition_size": len(target_partition),
            "source_nodes": source_partition,
            "target_nodes": target_partition,
            "total_nodes": len(source_partition) + len(target_partition),
            "total_edges": graph.numberOfEdges()
        }
    except GraphConstructionError:
        return {
            "is_bipartite": False,
            "error": "Graph is not bipartite"
        }


def temporal_bipartite_to_unipartite(
    edgelist: Union[str, pl.DataFrame],
    source_col: str = "source",
    target_col: str = "target", 
    timestamp_col: str = "timestamp",
    weight_col: Optional[str] = None,
    intermediate_col: str = "target",
    projected_col: str = "source",
    remove_self_loops: bool = True,
    add_edge_weights: bool = True
) -> Tuple[nk.Graph, IDMapper]:
    """
    Convert temporal bipartite edgelist to unipartite graph using temporal precedence.
    
    This function implements the temporal bipartite-to-unipartite conversion logic
    where nodes sharing the same intermediate node are connected based on temporal
    ordering. Uses ascending timestamp sort combined with upper triangular matrix
    indexing to preserve temporal causality (earlier → later flow).
    
    The algorithm:
    1. Groups edges by intermediate node (disappearing column)
    2. Within each group, sorts by timestamp in ascending order
    3. Creates directed edges using upper triangular matrix indices
    4. Results in proper temporal flow: earlier events → later events
    
    Parameters
    ----------
    edgelist : Union[str, pl.DataFrame]
        Edge list data containing temporal bipartite relationships.
        Can be file path (CSV/Parquet) or Polars DataFrame.
    source_col : str, default "source"
        Name of source node column in edgelist
    target_col : str, default "target"
        Name of target node column in edgelist
    timestamp_col : str, default "timestamp"
        Name of timestamp column for temporal ordering
    weight_col : Optional[str], default None
        Name of weight column. If None, edges get unit weight.
    intermediate_col : str, default "target"
        Column representing intermediate nodes (will disappear in projection).
        These nodes group the temporal relationships.
    projected_col : str, default "source"
        Column representing nodes to preserve in unipartite projection.
        These nodes will be connected based on shared intermediates.
    remove_self_loops : bool, default True
        Whether to remove self-loops in the resulting unipartite graph
    add_edge_weights : bool, default True
        Whether to calculate edge weights based on temporal relationships
        
    Returns
    -------
    Tuple[nk.Graph, IDMapper]
        NetworkIt directed graph and ID mapper for the projected unipartite network
        
    Raises
    ------
    DataFormatError
        If required columns are missing or data format is invalid
    ValidationError
        If timestamp column contains invalid data or insufficient temporal variation
    GraphConstructionError
        If projection fails due to data structure issues
        
    Examples
    --------
    Convert user-item temporal interactions to user-user influence network:
    
    >>> import polars as pl
    >>> from src.network.construction import temporal_bipartite_to_unipartite
    >>> 
    >>> # Sample temporal bipartite data: users interacting with items over time
    >>> data = pl.DataFrame({
    ...     "user": ["Alice", "Bob", "Charlie", "Alice", "Bob"],
    ...     "item": ["item1", "item1", "item1", "item2", "item2"], 
    ...     "timestamp": ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"]
    ... })
    >>>
    >>> # Convert to user-user influence network
    >>> graph, mapper = temporal_bipartite_to_unipartite(
    ...     data, 
    ...     source_col="user",
    ...     target_col="item",
    ...     timestamp_col="timestamp",
    ...     intermediate_col="item",  # Items disappear
    ...     projected_col="user"      # Users remain, get connected
    ... )
    
    Notes
    -----
    The temporal logic ensures proper causality: if users A and B both interact
    with item X, and A interacts first, then A → B edge is created (A influences B).
    
    This is particularly useful for:
    - User-item → User-user influence networks
    - Author-paper → Author-author citation networks  
    - Social media user-content → User-user information flow
    """
    
    with LoggingTimer("temporal_bipartite_to_unipartite"):
        log_function_entry("temporal_bipartite_to_unipartite", 
                          intermediate_col=intermediate_col, projected_col=projected_col)
        
        # Load and validate input data
        if isinstance(edgelist, str):
            edgelist = _load_edgelist_file(edgelist)
        
        # Validate required columns
        required_cols = [source_col, target_col, timestamp_col]
        missing_cols = [col for col in required_cols if col not in edgelist.columns]
        if missing_cols:
            raise DataFormatError(f"Missing required columns: {missing_cols}")
        
        if intermediate_col not in [source_col, target_col]:
            raise ConfigurationError(
                f"intermediate_col '{intermediate_col}' must be either '{source_col}' or '{target_col}'"
            )
        if projected_col not in [source_col, target_col]:
            raise ConfigurationError(
                f"projected_col '{projected_col}' must be either '{source_col}' or '{target_col}'"
            )
        if intermediate_col == projected_col:
            raise ConfigurationError("intermediate_col and projected_col cannot be the same")
            
        logger.info(f"Processing temporal bipartite conversion: {len(edgelist)} edges")
        logger.info(f"Intermediate column: {intermediate_col}, Projected column: {projected_col}")
        
        try:
            # Convert timestamp to datetime if needed
            if edgelist[timestamp_col].dtype != pl.Datetime:
                try:
                    edgelist = edgelist.with_columns(
                        pl.col(timestamp_col).str.to_datetime().alias(timestamp_col)
                    )
                except Exception as e:
                    raise ValidationError(f"Cannot parse timestamp column '{timestamp_col}': {e}")
            
            # Sort by intermediate node, then by timestamp ascending (critical for temporal logic)
            edgelist_sorted = edgelist.sort([intermediate_col, timestamp_col], descending=[False, False])
            
            logger.info(f"Sorted edgelist by {intermediate_col}, then {timestamp_col} (ascending)")
            
            # Group by intermediate node and create temporal edges
            projection_edges = []
            edge_weights = []
            
            for intermediate_node, group in edgelist_sorted.group_by(intermediate_col):
                # Get projected nodes in temporal order (ascending timestamps)
                projected_nodes = group[projected_col].to_list()
                timestamps = group[timestamp_col].to_list()
                weights = group[weight_col].to_list() if weight_col else [1.0] * len(projected_nodes)
                
                # Skip groups with only one node (no edges possible)
                if len(projected_nodes) <= 1:
                    continue
                
                # Remove duplicates while preserving order
                unique_nodes = []
                unique_timestamps = []
                unique_weights = []
                seen = set()
                for i, node in enumerate(projected_nodes):
                    if node not in seen:
                        unique_nodes.append(node)
                        unique_timestamps.append(timestamps[i])
                        unique_weights.append(weights[i])
                        seen.add(node)
                
                # Skip if only one unique node after deduplication
                if len(unique_nodes) <= 1:
                    continue
                    
                # Create temporal edges using upper triangular indices
                # This creates: earlier timestamp → later timestamp (proper temporal causality)
                n_nodes = len(unique_nodes)
                upper_tri_indices = np.triu_indices(n_nodes, k=1)
                
                for i, j in zip(upper_tri_indices[0], upper_tri_indices[1]):
                    source_node = unique_nodes[i]  # Earlier timestamp (due to ascending sort)
                    target_node = unique_nodes[j]  # Later timestamp
                    
                    # Calculate edge weight based on temporal relationship
                    if add_edge_weights:
                        # Weight inversely related to temporal distance
                        time_diff = (unique_timestamps[i] - unique_timestamps[j]).total_seconds()
                        temporal_weight = 1.0 / (1.0 + abs(time_diff) / 86400.0)  # Decay by days
                        edge_weight = (unique_weights[i] + unique_weights[j]) / 2.0 * temporal_weight
                    else:
                        edge_weight = 1.0
                    
                    projection_edges.append((source_node, target_node))
                    edge_weights.append(edge_weight)
                    
                logger.debug(f"Created {len(upper_tri_indices[0])} temporal edges for intermediate '{intermediate_node}'")
            
            if not projection_edges:
                logger.warning("No temporal edges created - all intermediate nodes had ≤1 unique projected node")
                # Create empty graph
                empty_graph = nk.Graph(0, weighted=True, directed=True)
                empty_mapper = IDMapper()
                return empty_graph, empty_mapper
            
            # Create unipartite edgelist DataFrame
            unipartite_df = pl.DataFrame({
                "source": [edge[0] for edge in projection_edges],
                "target": [edge[1] for edge in projection_edges], 
                "weight": edge_weights
            })
            
            # Remove self-loops if requested
            if remove_self_loops:
                original_count = len(unipartite_df)
                unipartite_df = unipartite_df.filter(pl.col("source") != pl.col("target"))
                removed_count = original_count - len(unipartite_df)
                if removed_count > 0:
                    logger.info(f"Removed {removed_count} self-loops")
            
            logger.info(f"Created temporal unipartite projection: {len(unipartite_df)} edges")
            
            # Build final directed graph
            graph, id_mapper = build_graph_from_edgelist(
                unipartite_df,
                source_col="source",
                target_col="target", 
                weight_col="weight",
                directed=True,
                allow_self_loops=not remove_self_loops
            )
            
            logger.info(f"Final temporal unipartite graph: {graph.numberOfNodes()} nodes, {graph.numberOfEdges()} edges")
            
            return graph, id_mapper
            
        except Exception as e:
            raise GraphConstructionError(
                f"Temporal bipartite-to-unipartite conversion failed: {str(e)}",
                operation="temporal_bipartite_to_unipartite",
                cause=e
            )