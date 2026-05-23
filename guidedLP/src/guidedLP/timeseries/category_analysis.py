"""
Category Analysis for Temporal Networks

This module provides functionality to analyze connections between different node categories
over time, enabling the study of segregation, integration, and cross-category interactions
in temporal networks.
"""

from typing import List, Tuple, Optional, Union
from datetime import datetime
import warnings

import polars as pl
import networkit as nk

from guidedLP.common.id_mapper import IDMapper


def analyze_cross_category_connections(
    temporal_graphs: List[Tuple[datetime, nk.Graph, IDMapper]],
    metadata: pl.DataFrame,
    category_column: str,
    edge_weight: str = "count"
) -> pl.DataFrame:
    """
    Track connections between node categories over time.
    
    This function analyzes how different categories of nodes connect to each other
    across temporal slices, providing insights into segregation vs. integration
    patterns, polarization, and cross-category interactions.
    
    Parameters
    ----------
    temporal_graphs : List[Tuple[datetime, nk.Graph, IDMapper]]
        List of (date, graph, id_mapper) tuples from create_temporal_slices()
    metadata : pl.DataFrame
        DataFrame containing node metadata with at least 'node_id' and category columns
    category_column : str
        Name of the column in metadata containing category labels
    edge_weight : str, default "count"
        How to aggregate edge weights between categories:
        - "count": Count number of edges
        - "sum": Sum edge weights (requires weighted graphs)
        - "mean": Average edge weights
        
    Returns
    -------
    pl.DataFrame
        DataFrame with columns:
        - date: Time slice date
        - category_a: First category
        - category_b: Second category  
        - connection_strength: Aggregated edge weight between categories
        - edge_count: Number of edges between categories
        
    Examples
    --------
    >>> metadata = pl.DataFrame({
    ...     "node_id": ["A", "B", "C", "D"],
    ...     "department": ["sales", "engineering", "sales", "marketing"]
    ... })
    >>> result = analyze_cross_category_connections(
    ...     temporal_graphs, metadata, "department"
    ... )
    >>> result.head()
    ┌────────────┬─────────────┬─────────────┬────────────────────┬────────────┐
    │ date       ┆ category_a  ┆ category_b  ┆ connection_strength ┆ edge_count │
    ├────────────┼─────────────┼─────────────┼────────────────────┼────────────┤
    │ 2024-01-01 ┆ sales       ┆ sales       ┆ 15                 ┆ 15         │
    │ 2024-01-01 ┆ sales       ┆ engineering ┆ 8                  ┆ 8          │
    │ 2024-01-01 ┆ engineering ┆ marketing   ┆ 3                  ┆ 3          │
    └────────────┴─────────────┴─────────────┴────────────────────┴────────────┘
        
    Notes
    -----
    - Categories are ordered alphabetically to avoid duplicate pairs (A->B and B->A)
    - Self-connections (same category) are included
    - Missing categories in metadata are treated as "unknown"
    - Empty graphs produce no output rows for that time slice
    """
    if not temporal_graphs:
        raise ValueError("temporal_graphs cannot be empty")
        
    if metadata.is_empty():
        raise ValueError("metadata cannot be empty")
        
    # Validate required columns
    required_cols = {"node_id", category_column}
    missing_cols = required_cols - set(metadata.columns)
    if missing_cols:
        raise ValueError(f"metadata missing required columns: {missing_cols}")
        
    # Validate edge_weight parameter
    valid_weights = {"count", "sum", "mean"}
    if edge_weight not in valid_weights:
        raise ValueError(f"edge_weight must be one of {valid_weights}, got {edge_weight}")
    
    # Create category lookup dictionary
    category_lookup = dict(zip(
        metadata["node_id"].to_list(),
        metadata[category_column].to_list()
    ))
    
    all_results = []
    
    for date, graph, id_mapper in temporal_graphs:
        if graph.numberOfNodes() == 0:
            # Skip empty graphs
            continue
            
        # Get edge information
        edges = []
        weights = []
        
        if graph.isWeighted():
            for edge in graph.iterEdges():
                u_internal, v_internal = edge[0], edge[1]
                weight = graph.weight(u_internal, v_internal)
                
                # Convert to external IDs
                u_external = id_mapper.get_original(u_internal)
                v_external = id_mapper.get_original(v_internal)
                
                edges.append((u_external, v_external))
                weights.append(weight)
        else:
            for edge in graph.iterEdges():
                u_internal, v_internal = edge[0], edge[1]
                weight = 1.0
                
                # Convert to external IDs
                u_external = id_mapper.get_original(u_internal)
                v_external = id_mapper.get_original(v_internal)
                
                edges.append((u_external, v_external))
                weights.append(weight)
        
        if not edges:
            # No edges in this time slice
            continue
            
        # Group edges by category pairs
        category_connections = {}
        
        for (u_external, v_external), weight in zip(edges, weights):
            # Get categories (default to "unknown" if not found)
            cat_u = category_lookup.get(u_external, "unknown")
            cat_v = category_lookup.get(v_external, "unknown")
            
            # Order categories alphabetically to avoid duplicates
            cat_pair = tuple(sorted([cat_u, cat_v]))
            
            if cat_pair not in category_connections:
                category_connections[cat_pair] = {
                    "weights": [],
                    "count": 0
                }
                
            category_connections[cat_pair]["weights"].append(weight)
            category_connections[cat_pair]["count"] += 1
        
        # Calculate aggregated connection strengths
        for (cat_a, cat_b), connection_data in category_connections.items():
            weights_list = connection_data["weights"]
            edge_count = connection_data["count"]
            
            if edge_weight == "count":
                connection_strength = edge_count
            elif edge_weight == "sum":
                connection_strength = sum(weights_list)
            elif edge_weight == "mean":
                connection_strength = sum(weights_list) / len(weights_list)
            
            all_results.append({
                "date": date,
                "category_a": cat_a,
                "category_b": cat_b,
                "connection_strength": connection_strength,
                "edge_count": edge_count
            })
    
    if not all_results:
        # No valid connections found
        warnings.warn("No connections found between categories across all time slices")
        return pl.DataFrame({
            "date": [],
            "category_a": [],
            "category_b": [],
            "connection_strength": [],
            "edge_count": []
        })
    
    # Convert to DataFrame and ensure proper types
    result_df = pl.DataFrame(all_results)
    
    # Sort by date, then by categories for consistent output
    result_df = result_df.sort(["date", "category_a", "category_b"])
    
    return result_df


def calculate_category_segregation_index(
    category_connections: pl.DataFrame,
    normalize: bool = True
) -> pl.DataFrame:
    """
    Calculate segregation index between categories over time.
    
    The segregation index measures the tendency for nodes to connect within
    their own category vs. across categories. Higher values indicate more
    segregation (within-group connections), lower values indicate more
    integration (between-group connections).
    
    Parameters
    ----------
    category_connections : pl.DataFrame
        Output from analyze_cross_category_connections()
    normalize : bool, default True
        Whether to normalize index to [0, 1] range
        
    Returns
    -------
    pl.DataFrame
        DataFrame with columns:
        - date: Time slice date
        - segregation_index: Segregation measure
        - within_category_strength: Total within-category connections
        - between_category_strength: Total between-category connections
        
    Notes
    -----
    Segregation index = within_category_connections / total_connections
    
    Values close to 1.0 indicate high segregation (mostly within-group connections)
    Values close to 0.0 indicate high integration (mostly between-group connections)
    """
    if category_connections.is_empty():
        return pl.DataFrame({
            "date": [],
            "segregation_index": [],
            "within_category_strength": [],
            "between_category_strength": []
        })
    
    # Identify within-category vs between-category connections
    segregation_data = category_connections.with_columns([
        (pl.col("category_a") == pl.col("category_b")).alias("is_within_category")
    ])
    
    # Group by date and calculate segregation metrics
    result = segregation_data.group_by("date").agg([
        pl.when(pl.col("is_within_category"))
        .then(pl.col("connection_strength"))
        .otherwise(0)
        .sum()
        .alias("within_category_strength"),
        
        pl.when(~pl.col("is_within_category"))
        .then(pl.col("connection_strength"))
        .otherwise(0)
        .sum()
        .alias("between_category_strength"),
        
        pl.col("connection_strength").sum().alias("total_strength")
    ])
    
    # Calculate segregation index
    result = result.with_columns([
        (pl.col("within_category_strength") / pl.col("total_strength")).alias("segregation_index")
    ])
    
    # Handle edge case where total_strength is 0
    result = result.with_columns([
        pl.when(pl.col("total_strength") == 0)
        .then(0.0)
        .otherwise(pl.col("segregation_index"))
        .alias("segregation_index")
    ])
    
    # Select final columns and sort by date
    result = result.select([
        "date",
        "segregation_index", 
        "within_category_strength",
        "between_category_strength"
    ]).sort("date")
    
    return result


def analyze_category_centrality_by_time(
    temporal_graphs: List[Tuple[datetime, nk.Graph, IDMapper]],
    metadata: pl.DataFrame,
    category_column: str,
    centrality_metric: str = "degree"
) -> pl.DataFrame:
    """
    Calculate centrality metrics for each category over time.
    
    This function computes centrality measures for nodes in each category,
    allowing analysis of which categories are most central/influential
    at different time points.
    
    Parameters
    ----------
    temporal_graphs : List[Tuple[datetime, nk.Graph, IDMapper]]
        List of (date, graph, id_mapper) tuples from create_temporal_slices()
    metadata : pl.DataFrame
        DataFrame containing node metadata with category information
    category_column : str
        Name of the column in metadata containing category labels
    centrality_metric : str, default "degree"
        Centrality metric to calculate ("degree", "betweenness", "closeness")
        
    Returns
    -------
    pl.DataFrame
        DataFrame with columns:
        - date: Time slice date
        - category: Category name
        - mean_centrality: Average centrality for nodes in category
        - median_centrality: Median centrality for nodes in category
        - max_centrality: Maximum centrality for nodes in category
        - node_count: Number of nodes in category present in graph
        
    Notes
    -----
    Only nodes present in each time slice are included in calculations.
    Categories with no nodes in a time slice are excluded from that slice.
    """
    from guidedLP.network.analysis import extract_centrality
    
    if not temporal_graphs:
        raise ValueError("temporal_graphs cannot be empty")
        
    if metadata.is_empty():
        raise ValueError("metadata cannot be empty")
        
    # Validate required columns
    required_cols = {"node_id", category_column}
    missing_cols = required_cols - set(metadata.columns)
    if missing_cols:
        raise ValueError(f"metadata missing required columns: {missing_cols}")
    
    # Create category lookup
    category_lookup = dict(zip(
        metadata["node_id"].to_list(),
        metadata[category_column].to_list()
    ))
    
    all_results = []
    
    for date, graph, id_mapper in temporal_graphs:
        if graph.numberOfNodes() == 0:
            continue
            
        # Calculate centrality for all nodes
        try:
            centrality_df = extract_centrality(
                graph, 
                id_mapper, 
                metrics=[centrality_metric]
            )
        except Exception as e:
            warnings.warn(f"Failed to calculate centrality for {date}: {e}")
            continue
        
        if centrality_df.is_empty():
            continue
            
        # Add category information
        centrality_with_categories = centrality_df.with_columns([
            pl.col("node_id").map_elements(
                lambda x: category_lookup.get(x, "unknown"),
                return_dtype=pl.String
            ).alias("category")
        ])
        
        # Group by category and calculate statistics
        category_stats = centrality_with_categories.group_by("category").agg([
            pl.col(centrality_metric).mean().alias("mean_centrality"),
            pl.col(centrality_metric).median().alias("median_centrality"),
            pl.col(centrality_metric).max().alias("max_centrality"),
            pl.col(centrality_metric).count().alias("node_count")
        ])
        
        # Add date column
        category_stats = category_stats.with_columns([
            pl.lit(date).alias("date")
        ])
        
        all_results.append(category_stats)
    
    if not all_results:
        warnings.warn("No valid centrality data found across all time slices")
        return pl.DataFrame({
            "date": [],
            "category": [],
            "mean_centrality": [],
            "median_centrality": [],
            "max_centrality": [],
            "node_count": []
        })
    
    # Combine all results
    result_df = pl.concat(all_results)
    
    # Sort by date, then by category
    result_df = result_df.sort(["date", "category"])
    
    # Select columns in desired order
    result_df = result_df.select([
        "date",
        "category", 
        "mean_centrality",
        "median_centrality",
        "max_centrality",
        "node_count"
    ])
    
    return result_df