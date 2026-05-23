"""
Network analysis module for the Guided Label Propagation library.

This module provides functionality for analyzing network properties, with a focus
on centrality measures that quantify node importance in different ways. All
centrality calculations leverage NetworkIt's high-performance implementations.
"""

from typing import List, Dict, Any, Optional, Union
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

import polars as pl
import networkit as nk
import numpy as np

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

# Available centrality metrics and their implementations
AVAILABLE_METRICS = [
    "degree", "betweenness", "closeness", "eigenvector", "pagerank", "katz"
]


def extract_centrality(
    graph: nk.Graph,
    id_mapper: IDMapper,
    metrics: List[str] = ["degree", "betweenness", "closeness", "eigenvector"],
    normalized: bool = True,
    n_jobs: int = -1
) -> pl.DataFrame:
    """
    Calculate centrality metrics for all nodes in a graph.
    
    This function computes various centrality measures using NetworkIt's
    optimized implementations. Centrality metrics quantify the importance
    or influence of nodes in a network from different perspectives.
    
    Parameters
    ----------
    graph : nk.Graph
        NetworkIt graph for which to calculate centrality metrics
    id_mapper : IDMapper
        Bidirectional mapping between original and internal node IDs
    metrics : List[str], default ["degree", "betweenness", "closeness", "eigenvector"]
        List of centrality metrics to calculate. Available options:
        - "degree": Node degree (number of connections)
        - "betweenness": Fraction of shortest paths passing through node
        - "closeness": Inverse of average distance to all other nodes
        - "eigenvector": Importance based on importance of neighbors
        - "pagerank": Google's PageRank algorithm (random walk probability)
        - "katz": Katz centrality (weighted sum of walks of all lengths)
    normalized : bool, default True
        Whether to normalize centrality scores to the range [0, 1]. 
        Normalization makes metrics comparable across different graphs.
    n_jobs : int, default -1
        Number of parallel jobs for computation. 
        - -1: Use all available CPU cores
        - 1: Sequential computation
        - >1: Use specified number of cores
        
    Returns
    -------
    pl.DataFrame
        DataFrame with centrality metrics, containing columns:
        - "node_id": Original node identifiers
        - "{metric}_centrality": One column per requested metric (e.g., "degree_centrality")
        
    Raises
    ------
    ConfigurationError
        If invalid metric names are specified or n_jobs is invalid
    ComputationError
        If centrality calculation fails due to graph properties or computational issues
        
    Examples
    --------
    >>> import polars as pl
    >>> edges = pl.DataFrame({
    ...     "source": ["A", "B", "C", "A"],
    ...     "target": ["B", "C", "A", "C"]
    ... })
    >>> graph, mapper = build_graph_from_edgelist(edges)
    >>> centrality_df = extract_centrality(graph, mapper, ["degree", "betweenness"])
    >>> print(centrality_df)
    ┌─────────┬──────────────────┬─────────────────────────┐
    │ node_id ┆ degree_centrality ┆ betweenness_centrality │
    │ ---     ┆ ---              ┆ ---                    │
    │ str     ┆ f64              ┆ f64                    │
    ├─────────┼──────────────────┼─────────────────────────┤
    │ A       ┆ 1.0              ┆ 0.5                    │
    │ B       ┆ 0.67             ┆ 0.0                    │
    │ C       ┆ 1.0              ┆ 0.5                    │
    └─────────┴──────────────────┴─────────────────────────┘
    
    >>> # Calculate only PageRank with custom parameters
    >>> pagerank_df = extract_centrality(graph, mapper, ["pagerank"], normalized=False)
    
    Notes
    -----
    Time Complexity: Varies by metric
    - Degree: O(V + E)
    - Betweenness: O(V * E) for unweighted, O(V * E + V² * log V) for weighted
    - Closeness: O(V * E) for unweighted, O(V * E + V² * log V) for weighted  
    - Eigenvector: O(V * E * k) where k is number of iterations
    - PageRank: O(V * E * k) where k is number of iterations
    - Katz: O(V * E * k) where k is number of iterations
    
    Space Complexity: O(V) for result storage
    
    Centrality Interpretations:
    - **Degree**: Simple connectivity measure; higher = more connections
    - **Betweenness**: Bridge/broker measure; higher = more shortest paths pass through
    - **Closeness**: Efficiency measure; higher = shorter average distance to others
    - **Eigenvector**: Prestige measure; higher = connected to other important nodes
    - **PageRank**: Authority measure; higher = more "vote" weight from neighbors
    - **Katz**: Similar to eigenvector but handles disconnected graphs better
    
    Parallel Processing:
    For large graphs (>10,000 nodes), parallel processing can provide significant
    speedup, especially for computationally expensive metrics like betweenness
    and closeness centrality.
    """
    log_function_entry("extract_centrality", 
                      n_nodes=graph.numberOfNodes(), 
                      metrics=metrics, 
                      normalized=normalized,
                      n_jobs=n_jobs)
    
    # Validate parameters
    _validate_centrality_parameters(metrics, n_jobs)
    
    # Handle empty graph
    if graph.numberOfNodes() == 0:
        warnings.warn("Empty graph provided. Returning empty DataFrame.")
        columns = ["node_id"] + [f"{metric}_centrality" for metric in metrics]
        return pl.DataFrame({col: [] for col in columns})
    
    with LoggingTimer("extract_centrality", {"metrics": metrics, "nodes": graph.numberOfNodes()}):
        try:
            # Calculate centralities
            if n_jobs == 1 or len(metrics) == 1:
                # Sequential computation
                centrality_data = _calculate_centralities_sequential(
                    graph, id_mapper, metrics, normalized
                )
            else:
                # Parallel computation
                centrality_data = _calculate_centralities_parallel(
                    graph, id_mapper, metrics, normalized, n_jobs
                )
            
            # Create DataFrame
            result_df = _create_centrality_dataframe(centrality_data, id_mapper)
            
            logger.info("Centrality calculation completed: %d nodes, %d metrics",
                       len(result_df), len(metrics))
            
            return result_df
            
        except Exception as e:
            if isinstance(e, (ConfigurationError, ComputationError)):
                raise
            else:
                raise ComputationError(
                    f"Centrality calculation failed: {str(e)}",
                    operation="extract_centrality",
                    error_type="computation",
                    resource_info={"nodes": graph.numberOfNodes(), "edges": graph.numberOfEdges()},
                    cause=e
                )


def _validate_centrality_parameters(metrics: List[str], n_jobs: int) -> None:
    """
    Validate parameters for centrality calculation.
    
    Parameters
    ----------
    metrics : List[str]
        List of centrality metrics to validate
    n_jobs : int
        Number of parallel jobs to validate
        
    Raises
    ------
    ConfigurationError
        If parameters are invalid
    """
    # Validate metrics
    if not metrics:
        raise ValidationError("At least one centrality metric must be specified")
    
    invalid_metrics = [m for m in metrics if m not in AVAILABLE_METRICS]
    if invalid_metrics:
        raise ValidationError(
            f"Invalid centrality metrics: {invalid_metrics}. "
            f"Available metrics: {AVAILABLE_METRICS}"
        )
    
    # Validate n_jobs
    if n_jobs == 0:
        raise ValidationError("n_jobs cannot be 0")
    
    max_cores = multiprocessing.cpu_count()
    if n_jobs == -1:
        # Use all cores - this is fine
        pass
    elif n_jobs > max_cores:
        warnings.warn(
            f"Requested {n_jobs} cores but only {max_cores} available. "
            f"Using {max_cores} cores."
        )


def _calculate_centralities_sequential(
    graph: nk.Graph,
    id_mapper: IDMapper,
    metrics: List[str],
    normalized: bool
) -> Dict[str, np.ndarray]:
    """
    Calculate centralities sequentially (single-threaded).
    
    Parameters
    ----------
    graph : nk.Graph
        NetworkIt graph
    id_mapper : IDMapper
        ID mapper
    metrics : List[str]
        Centrality metrics to calculate
    normalized : bool
        Whether to normalize results
        
    Returns
    -------
    Dict[str, np.ndarray]
        Dictionary mapping metric names to centrality arrays
    """
    logger.debug("Calculating centralities sequentially")
    
    centrality_data = {}
    
    for metric in metrics:
        logger.debug("Calculating %s centrality", metric)
        
        try:
            with LoggingTimer(f"{metric}_centrality", {"nodes": graph.numberOfNodes()}):
                values = _calculate_single_centrality(graph, metric, normalized)
                centrality_data[metric] = values
                
        except Exception as e:
            raise ComputationError(
                f"Failed to calculate {metric} centrality: {str(e)}",
                operation=f"calculate_{metric}",
                error_type="numerical" if "convergence" in str(e).lower() else "computation",
                cause=e
            )
    
    return centrality_data


def _calculate_centralities_parallel(
    graph: nk.Graph,
    id_mapper: IDMapper,
    metrics: List[str],
    normalized: bool,
    n_jobs: int
) -> Dict[str, np.ndarray]:
    """
    Calculate centralities in parallel using multiprocessing.
    
    Parameters
    ----------
    graph : nk.Graph
        NetworkIt graph
    id_mapper : IDMapper
        ID mapper
    metrics : List[str]
        Centrality metrics to calculate
    normalized : bool
        Whether to normalize results
    n_jobs : int
        Number of parallel jobs
        
    Returns
    -------
    Dict[str, np.ndarray]
        Dictionary mapping metric names to centrality arrays
    """
    logger.debug("Calculating centralities in parallel with %d jobs", n_jobs)
    
    # Determine actual number of workers
    max_workers = min(len(metrics), multiprocessing.cpu_count() if n_jobs == -1 else n_jobs)
    
    centrality_data = {}
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit tasks
        future_to_metric = {}
        for metric in metrics:
            future = executor.submit(_calculate_single_centrality_worker, 
                                   graph, metric, normalized)
            future_to_metric[future] = metric
        
        # Collect results
        for future in as_completed(future_to_metric):
            metric = future_to_metric[future]
            try:
                values = future.result()
                centrality_data[metric] = values
                logger.debug("Completed %s centrality calculation", metric)
            except Exception as e:
                raise ComputationError(
                    f"Failed to calculate {metric} centrality in parallel: {str(e)}",
                    operation=f"calculate_{metric}_parallel",
                    error_type="computation",
                    cause=e
                )
    
    return centrality_data


def _calculate_single_centrality_worker(
    graph: nk.Graph, 
    metric: str, 
    normalized: bool
) -> np.ndarray:
    """
    Worker function for parallel centrality calculation.
    
    This function is designed to be called by multiprocessing and should
    not depend on any module-level state.
    
    Parameters
    ----------
    graph : nk.Graph
        NetworkIt graph
    metric : str
        Centrality metric name
    normalized : bool
        Whether to normalize results
        
    Returns
    -------
    np.ndarray
        Centrality values for all nodes
    """
    return _calculate_single_centrality(graph, metric, normalized)


def _calculate_single_centrality(
    graph: nk.Graph, 
    metric: str, 
    normalized: bool
) -> np.ndarray:
    """
    Calculate a single centrality metric using NetworkIt.
    
    Parameters
    ----------
    graph : nk.Graph
        NetworkIt graph
    metric : str
        Centrality metric name
    normalized : bool
        Whether to normalize results
        
    Returns
    -------
    np.ndarray
        Centrality values for all nodes (indexed by internal node ID)
        
    Raises
    ------
    ComputationError
        If centrality calculation fails
    """
    try:
        if metric == "degree":
            # Degree centrality
            if graph.isDirected():
                # For directed graphs, we'll use total degree (in + out)
                in_degrees = np.array([graph.degreeIn(v) for v in graph.iterNodes()])
                out_degrees = np.array([graph.degreeOut(v) for v in graph.iterNodes()])
                values = in_degrees + out_degrees
            else:
                values = np.array([graph.degree(v) for v in graph.iterNodes()])
            
            if normalized and graph.numberOfNodes() > 1:
                max_possible = graph.numberOfNodes() - 1
                if graph.isDirected():
                    max_possible *= 2  # In directed graphs, max degree is 2*(n-1)
                values = values / max_possible
                
        elif metric == "betweenness":
            # Betweenness centrality
            bc = nk.centrality.Betweenness(graph, normalized=normalized)
            bc.run()
            values = np.array(bc.scores())
            
        elif metric == "closeness":
            # Closeness centrality
            # Use harmonic closeness for disconnected graphs
            cc = nk.centrality.HarmonicCloseness(graph, normalized=normalized)
            cc.run()
            values = np.array(cc.scores())
            
        elif metric == "eigenvector":
            # Eigenvector centrality
            if graph.numberOfNodes() == 1:
                # Single node case
                values = np.array([1.0])
            else:
                try:
                    ec = nk.centrality.EigenvectorCentrality(graph)
                    ec.run()
                    values = np.array(ec.scores())
                    
                    if normalized:
                        # NetworkIt eigenvector centrality may not be normalized
                        max_val = np.max(values) if len(values) > 0 else 1.0
                        if max_val > 0:
                            values = values / max_val
                except:
                    # Fallback for disconnected or problematic graphs
                    logger.warning("Eigenvector centrality failed, using degree as fallback")
                    values = _calculate_single_centrality(graph, "degree", normalized)
                    
        elif metric == "pagerank":
            # PageRank centrality
            pr = nk.centrality.PageRank(graph, 0.85)
            pr.run()
            values = np.array(pr.scores())
            
            if normalized:
                # PageRank is typically already normalized, but ensure [0,1] range
                min_val, max_val = np.min(values), np.max(values)
                if max_val > min_val:
                    values = (values - min_val) / (max_val - min_val)
                    
        elif metric == "katz":
            # Katz centrality
            try:
                kc = nk.centrality.KatzCentrality(graph, alpha=0.1)
                kc.run()
                values = np.array(kc.scores())
                
                if normalized:
                    max_val = np.max(values) if len(values) > 0 else 1.0
                    if max_val > 0:
                        values = values / max_val
            except:
                # Fallback for problematic graphs
                logger.warning("Katz centrality failed, using eigenvector as fallback")
                values = _calculate_single_centrality(graph, "eigenvector", normalized)
                
        else:
            # This should never happen due to parameter validation
            raise ValueError(f"Unknown centrality metric: {metric}")
        
        # Ensure values are finite and non-negative
        values = np.nan_to_num(values, nan=0.0, posinf=1.0, neginf=0.0)
        values = np.clip(values, 0.0, None)  # Ensure non-negative
        
        return values
        
    except Exception as e:
        if isinstance(e, ComputationError):
            raise
        else:
            raise ComputationError(
                f"Failed to calculate {metric} centrality: {str(e)}",
                operation=f"calculate_{metric}",
                error_type="numerical",
                cause=e
            )


def _create_centrality_dataframe(
    centrality_data: Dict[str, np.ndarray],
    id_mapper: IDMapper
) -> pl.DataFrame:
    """
    Create Polars DataFrame from centrality calculation results.
    
    Parameters
    ----------
    centrality_data : Dict[str, np.ndarray]
        Dictionary mapping metric names to centrality arrays
    id_mapper : IDMapper
        ID mapper for converting internal to original IDs
        
    Returns
    -------
    pl.DataFrame
        DataFrame with node_id and centrality columns
    """
    if not centrality_data:
        return pl.DataFrame({"node_id": []})
    
    # Get original node IDs
    n_nodes = id_mapper.size()
    original_ids = []
    
    for internal_id in range(n_nodes):
        try:
            original_id = id_mapper.get_original(internal_id)
            original_ids.append(original_id)
        except KeyError:
            # This should not happen if id_mapper is consistent
            logger.warning("Missing mapping for internal ID %d", internal_id)
            original_ids.append(f"missing_{internal_id}")
    
    # Build DataFrame data
    df_data = {"node_id": original_ids}
    
    for metric, values in centrality_data.items():
        column_name = f"{metric}_centrality"
        
        # Ensure values array has correct length
        if len(values) != n_nodes:
            logger.warning(
                "Centrality array length (%d) doesn't match number of nodes (%d) for metric %s",
                len(values), n_nodes, metric
            )
            # Pad or truncate as needed
            if len(values) < n_nodes:
                values = np.pad(values, (0, n_nodes - len(values)), constant_values=0.0)
            else:
                values = values[:n_nodes]
        
        df_data[column_name] = values.tolist()
    
    # Create and sort DataFrame
    df = pl.DataFrame(df_data)
    df = df.sort("node_id")
    
    return df


def get_centrality_summary(centrality_df: pl.DataFrame) -> Dict[str, Any]:
    """
    Get summary statistics for centrality measures.
    
    Parameters
    ----------
    centrality_df : pl.DataFrame
        DataFrame returned by extract_centrality()
        
    Returns
    -------
    Dict[str, Any]
        Summary statistics for each centrality metric
        
    Examples
    --------
    >>> centrality_df = extract_centrality(graph, mapper, ["degree", "betweenness"])
    >>> summary = get_centrality_summary(centrality_df)
    >>> print(summary["degree_centrality"]["mean"])
    0.666...
    """
    summary = {}
    
    # Get centrality columns (exclude node_id)
    centrality_cols = [col for col in centrality_df.columns if col.endswith("_centrality")]
    
    for col in centrality_cols:
        values = centrality_df[col]
        
        summary[col] = {
            "count": len(values),
            "mean": float(values.mean()),
            "std": float(values.std()),
            "min": float(values.min()),
            "max": float(values.max()),
            "median": float(values.median()),
            "q25": float(values.quantile(0.25)),
            "q75": float(values.quantile(0.75))
        }
    
    return summary


def identify_central_nodes(
    centrality_df: pl.DataFrame,
    metric: str = "betweenness_centrality",
    top_k: int = 10,
    threshold: Optional[float] = None
) -> List[str]:
    """
    Identify the most central nodes based on a specific centrality metric.
    
    Parameters
    ----------
    centrality_df : pl.DataFrame
        DataFrame returned by extract_centrality()
    metric : str, default "betweenness_centrality"
        Centrality metric to use for ranking
    top_k : int, default 10
        Number of top nodes to return
    threshold : float, optional
        Minimum centrality value threshold. If specified, only nodes
        with centrality >= threshold are returned.
        
    Returns
    -------
    List[str]
        List of node IDs sorted by centrality (descending)
        
    Examples
    --------
    >>> centrality_df = extract_centrality(graph, mapper)
    >>> top_nodes = identify_central_nodes(centrality_df, "degree_centrality", top_k=5)
    >>> print(top_nodes)
    """
    if metric not in centrality_df.columns:
        available_metrics = [col for col in centrality_df.columns if col.endswith("_centrality")]
        raise ValueError(
            f"Metric '{metric}' not found in DataFrame. "
            f"Available metrics: {available_metrics}"
        )
    
    # Sort by centrality metric in descending order
    result = centrality_df.sort(metric, descending=True)
    
    # Apply threshold if specified
    if threshold is not None:
        result = result.filter(pl.col(metric) >= threshold)
    
    # Take top k nodes
    result = result.head(top_k)
    
    return result["node_id"].to_list()


def compare_centrality_metrics(
    centrality_df: pl.DataFrame,
    metric1: str,
    metric2: str
) -> Dict[str, float]:
    """
    Compare two centrality metrics by calculating correlation and rank correlation.
    
    Parameters
    ----------
    centrality_df : pl.DataFrame
        DataFrame with centrality metrics
    metric1 : str
        First centrality metric column name
    metric2 : str  
        Second centrality metric column name
        
    Returns
    -------
    Dict[str, float]
        Dictionary with correlation statistics
        
    Examples
    --------
    >>> centrality_df = extract_centrality(graph, mapper, ["degree", "betweenness"])
    >>> correlation = compare_centrality_metrics(
    ...     centrality_df, "degree_centrality", "betweenness_centrality"
    ... )
    >>> print(f"Pearson correlation: {correlation['pearson']:.3f}")
    """
    if metric1 not in centrality_df.columns:
        raise ValueError(f"Metric '{metric1}' not found in DataFrame")
    if metric2 not in centrality_df.columns:
        raise ValueError(f"Metric '{metric2}' not found in DataFrame")
    
    values1 = centrality_df[metric1].to_numpy()
    values2 = centrality_df[metric2].to_numpy()
    
    # Pearson correlation
    pearson_corr = np.corrcoef(values1, values2)[0, 1]
    
    # Spearman rank correlation
    from scipy.stats import spearmanr
    spearman_corr, _ = spearmanr(values1, values2)
    
    return {
        "pearson": float(pearson_corr) if not np.isnan(pearson_corr) else 0.0,
        "spearman": float(spearman_corr) if not np.isnan(spearman_corr) else 0.0,
        "n_nodes": len(values1)
    }