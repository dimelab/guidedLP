"""
Community detection module for the Guided Label Propagation library.

This module provides functionality for detecting community structure in networks
using various algorithms, with a focus on the Louvain method. Supports multiple
iterations for consensus calculation and provides quality metrics for evaluation.
"""

from typing import List, Dict, Any, Optional, Union, Tuple
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
from collections import Counter, defaultdict

import polars as pl
import networkit as nk
import numpy as np
from sklearn.metrics import normalized_mutual_info_score

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

# Available community detection algorithms
AVAILABLE_ALGORITHMS = ["louvain"]


def detect_communities(
    graph: nk.Graph,
    id_mapper: IDMapper,
    algorithm: str = "louvain",
    iterations: int = 1,
    resolution: float = 1.0,
    min_similarity: Optional[float] = None,
    random_seed: Optional[int] = None,
    n_jobs: int = -1
) -> pl.DataFrame:
    """
    Detect communities using specified algorithm with quality metrics.
    
    This function runs community detection algorithms (currently Louvain) multiple
    times to assess stability and consensus. Provides comprehensive quality metrics
    and supports parallel execution for multiple iterations.
    
    Parameters
    ----------
    graph : nk.Graph
        NetworkIt graph object
    id_mapper : IDMapper
        Bidirectional mapping between original and internal node IDs
    algorithm : str, default "louvain"
        Community detection algorithm to use. Currently supports:
        - "louvain": Louvain method for modularity optimization
    iterations : int, default 1
        Number of runs with different random initializations.
        Multiple iterations enable consensus and stability calculation.
    resolution : float, default 1.0
        Resolution parameter for community detection. Higher values
        tend to produce more, smaller communities.
    min_similarity : float, optional
        If specified, only partitions with normalized mutual information
        >= min_similarity are included in consensus calculation.
    random_seed : int, optional
        Random seed for reproducibility. Each iteration uses seed + iteration.
    n_jobs : int, default -1
        Number of parallel jobs for multiple iterations.
        -1 uses all available cores, 1 forces sequential execution.
    
    Returns
    -------
    pl.DataFrame
        DataFrame with community assignments and quality metrics.
        Columns include:
        - node_id: Original node ID
        - community_iter_{i}: Community assignment for iteration i (0-indexed)
        - community_consensus: Consensus community (most frequent across iterations)
        - stability: Fraction of iterations with consensus assignment [0, 1]
        - modularity_iter_{i}: Modularity score for iteration i
        - num_communities_iter_{i}: Number of communities for iteration i
        
        For single iteration, consensus and stability columns are identical
        to the single iteration result.
    
    Examples
    --------
    Single iteration community detection:
    
    >>> result = detect_communities(graph, mapper, iterations=1)
    >>> print(result.select(["node_id", "community_iter_0", "modularity_iter_0"]))
    
    Multiple iterations with consensus:
    
    >>> result = detect_communities(graph, mapper, iterations=10, random_seed=42)
    >>> stable_nodes = result.filter(pl.col("stability") > 0.8)
    >>> print(f"Found {len(stable_nodes)} stable community assignments")
    
    Quality filtering:
    
    >>> result = detect_communities(
    ...     graph, mapper, iterations=10, min_similarity=0.7
    ... )
    >>> avg_modularity = result.select(pl.col("modularity_iter_0")).mean()
    
    Raises
    ------
    ValidationError
        If algorithm is not supported, iterations < 1, or parameters are invalid
    ComputationError
        If community detection fails or produces invalid results
    
    Notes
    -----
    Time Complexity:
        O(E × I) where E is number of edges and I is number of iterations.
        The Louvain algorithm typically runs in O(E) time per iteration.
    
    Space Complexity:
        O(N × I) for storing community assignments across iterations.
    
    The consensus calculation uses majority voting across iterations. Stability
    scores indicate how consistent each node's community assignment is across
    different random initializations, providing insight into the robustness
    of the community structure.
    
    References
    ----------
    .. [1] Blondel, V. D., et al. "Fast unfolding of communities in large networks."
           Journal of Statistical Mechanics (2008).
    .. [2] Fortunato, S. "Community detection in graphs." 
           Physics Reports 486.3-5 (2010): 75-174.
    """
    log_function_entry(
        "detect_communities",
        graph_nodes=graph.numberOfNodes(),
        graph_edges=graph.numberOfEdges(),
        algorithm=algorithm,
        iterations=iterations,
        resolution=resolution,
        random_seed=random_seed,
        n_jobs=n_jobs
    )
    
    # Validate parameters
    _validate_community_parameters(algorithm, iterations, resolution, min_similarity, n_jobs)
    
    # Handle empty graph
    if graph.numberOfNodes() == 0:
        warnings.warn("Empty graph provided. Returning empty DataFrame.")
        columns = ["node_id", "community_consensus", "stability"]
        if iterations == 1:
            columns.extend(["community_iter_0", "modularity_iter_0", "num_communities_iter_0"])
        else:
            for i in range(iterations):
                columns.extend([f"community_iter_{i}", f"modularity_iter_{i}", f"num_communities_iter_{i}"])
        return pl.DataFrame({col: [] for col in columns})
    
    # Handle graph with no edges (isolated nodes)
    if graph.numberOfEdges() == 0:
        logger.info("Graph has no edges. Each node forms its own community.")
        return _handle_no_edges_graph(graph, id_mapper, iterations)
    
    with LoggingTimer("detect_communities", {"algorithm": algorithm, "iterations": iterations, "nodes": graph.numberOfNodes()}):
        try:
            # Run community detection
            if iterations == 1 or n_jobs == 1:
                # Sequential execution
                partitions_data = _run_community_detection_sequential(
                    graph, algorithm, iterations, resolution, random_seed
                )
            else:
                # Parallel execution for multiple iterations
                partitions_data = _run_community_detection_parallel(
                    graph, algorithm, iterations, resolution, random_seed, n_jobs
                )
            
            # Filter partitions by similarity if requested
            if min_similarity is not None and iterations > 1:
                partitions_data = _filter_partitions_by_similarity(
                    partitions_data, min_similarity
                )
                if not partitions_data:
                    warnings.warn(
                        f"No partitions met minimum similarity threshold {min_similarity}. "
                        "Returning results from first iteration only."
                    )
                    # Fall back to first iteration
                    partitions_data = _run_community_detection_sequential(
                        graph, algorithm, 1, resolution, random_seed
                    )
            
            # Calculate consensus and stability
            if len(partitions_data) > 1:
                consensus_data = _calculate_consensus_and_stability(partitions_data)
            else:
                # Single iteration case
                partition_data = partitions_data[0]
                consensus_data = {
                    "partitions": partition_data["partition"],
                    "consensus": partition_data["partition"],
                    "stability": [1.0] * len(partition_data["partition"])
                }
            
            # Map internal IDs to original IDs and create result DataFrame
            result_df = _create_result_dataframe(
                consensus_data, partitions_data, id_mapper, graph.numberOfNodes()
            )
            
            logger.info(
                f"Community detection completed: {len(partitions_data)} iterations, "
                f"{result_df.height} nodes, "
                f"avg_modularity={np.mean([p['modularity'] for p in partitions_data]):.3f}"
            )
            
            return result_df
            
        except Exception as e:
            raise ComputationError(
                f"Community detection failed: {str(e)}",
                context={
                    "operation": "detect_communities",
                    "algorithm": algorithm,
                    "iterations": iterations,
                    "error_type": "computation"
                }
            ) from e


def _validate_community_parameters(
    algorithm: str,
    iterations: int,
    resolution: float,
    min_similarity: Optional[float],
    n_jobs: int
) -> None:
    """
    Validate community detection parameters.
    
    Parameters
    ----------
    algorithm : str
        Community detection algorithm name
    iterations : int
        Number of iterations to run
    resolution : float
        Resolution parameter
    min_similarity : float, optional
        Minimum similarity threshold
    n_jobs : int
        Number of parallel jobs
        
    Raises
    ------
    ValidationError
        If any parameter is invalid
    """
    # Validate algorithm
    if algorithm not in AVAILABLE_ALGORITHMS:
        raise ValidationError(
            f"Invalid algorithm '{algorithm}'. "
            f"Available algorithms: {AVAILABLE_ALGORITHMS}"
        )
    
    # Validate iterations
    if iterations < 1:
        raise ValidationError("iterations must be >= 1")
    
    # Validate resolution
    if resolution <= 0:
        raise ValidationError("resolution must be > 0")
    
    # Validate min_similarity
    if min_similarity is not None:
        if not (0.0 <= min_similarity <= 1.0):
            raise ValidationError("min_similarity must be between 0.0 and 1.0")
        if iterations == 1:
            warnings.warn(
                "min_similarity specified with iterations=1, will be ignored"
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
            f"n_jobs ({n_jobs}) exceeds available cores ({max_cores}). "
            f"Using {max_cores} cores."
        )


def _run_community_detection_sequential(
    graph: nk.Graph,
    algorithm: str,
    iterations: int,
    resolution: float,
    random_seed: Optional[int]
) -> List[Dict[str, Any]]:
    """
    Run community detection sequentially for multiple iterations.
    
    Parameters
    ----------
    graph : nk.Graph
        NetworkIt graph
    algorithm : str
        Algorithm name
    iterations : int
        Number of iterations
    resolution : float
        Resolution parameter
    random_seed : int, optional
        Base random seed
        
    Returns
    -------
    List[Dict[str, Any]]
        List of partition results with metadata
    """
    logger.debug(f"Running {algorithm} sequentially for {iterations} iterations")
    
    partitions_data = []
    
    for i in range(iterations):
        # Set seed for reproducibility
        current_seed = None if random_seed is None else random_seed + i
        
        try:
            partition_data = _run_single_community_detection(
                graph, algorithm, resolution, current_seed, iteration=i
            )
            partitions_data.append(partition_data)
            
        except Exception as e:
            logger.warning(f"Iteration {i} failed: {str(e)}")
            # Continue with other iterations
            continue
    
    if not partitions_data:
        raise ComputationError(
            "All community detection iterations failed",
            context={"operation": "sequential_community_detection"}
        )
    
    return partitions_data


def _run_community_detection_parallel(
    graph: nk.Graph,
    algorithm: str,
    iterations: int,
    resolution: float,
    random_seed: Optional[int],
    n_jobs: int
) -> List[Dict[str, Any]]:
    """
    Run community detection in parallel for multiple iterations.
    
    Parameters
    ----------
    graph : nk.Graph
        NetworkIt graph
    algorithm : str
        Algorithm name
    iterations : int
        Number of iterations
    resolution : float
        Resolution parameter
    random_seed : int, optional
        Base random seed
    n_jobs : int
        Number of parallel jobs
        
    Returns
    -------
    List[Dict[str, Any]]
        List of partition results with metadata
    """
    logger.debug(f"Running {algorithm} in parallel for {iterations} iterations with {n_jobs} jobs")
    
    # Determine actual number of workers
    max_cores = multiprocessing.cpu_count()
    actual_workers = min(max_cores, iterations) if n_jobs == -1 else min(n_jobs, max_cores, iterations)
    
    partitions_data = []
    
    with ProcessPoolExecutor(max_workers=actual_workers) as executor:
        # Submit all tasks
        future_to_iteration = {}
        for i in range(iterations):
            current_seed = None if random_seed is None else random_seed + i
            future = executor.submit(
                _run_single_community_detection,
                graph, algorithm, resolution, current_seed, i
            )
            future_to_iteration[future] = i
        
        # Collect results
        for future in as_completed(future_to_iteration):
            iteration = future_to_iteration[future]
            try:
                partition_data = future.result()
                partitions_data.append(partition_data)
            except Exception as e:
                logger.warning(f"Parallel iteration {iteration} failed: {str(e)}")
                continue
    
    if not partitions_data:
        raise ComputationError(
            "All parallel community detection iterations failed",
            context={"operation": "parallel_community_detection"}
        )
    
    # Sort by iteration for consistent ordering
    partitions_data.sort(key=lambda x: x["iteration"])
    
    return partitions_data


def _run_single_community_detection(
    graph: nk.Graph,
    algorithm: str,
    resolution: float,
    random_seed: Optional[int],
    iteration: int = 0
) -> Dict[str, Any]:
    """
    Run a single community detection iteration.
    
    Parameters
    ----------
    graph : nk.Graph
        NetworkIt graph
    algorithm : str
        Algorithm name ("louvain")
    resolution : float
        Resolution parameter
    random_seed : int, optional
        Random seed for this iteration
    iteration : int, default 0
        Iteration number for tracking
        
    Returns
    -------
    Dict[str, Any]
        Dictionary containing partition and quality metrics
    """
    try:
        if algorithm == "louvain":
            # Set random seed if provided
            if random_seed is not None:
                nk.setSeed(random_seed, useThreadId=False)
            
            # Run Louvain algorithm
            louvain = nk.community.PLM(graph, refine=True, gamma=resolution)
            louvain.run()
            
            # Get partition
            partition = louvain.getPartition()
            partition_vector = [partition.subsetOf(node) for node in range(graph.numberOfNodes())]
            
            # Calculate quality metrics
            modularity = nk.community.Modularity().getQuality(partition, graph)
            num_communities = partition.numberOfSubsets()
            
            # Calculate coverage (fraction of edges within communities)
            coverage = _calculate_coverage(graph, partition_vector)
            
            return {
                "iteration": iteration,
                "partition": partition_vector,
                "modularity": modularity,
                "num_communities": num_communities,
                "coverage": coverage,
                "algorithm": algorithm,
                "resolution": resolution
            }
        
        else:
            raise ValueError(f"Unsupported algorithm: {algorithm}")
            
    except Exception as e:
        raise ComputationError(
            f"Failed to run {algorithm} algorithm: {str(e)}",
            context={
                "operation": f"run_{algorithm}",
                "iteration": iteration,
                "error_type": "algorithm_failure"
            }
        ) from e


def _calculate_coverage(graph: nk.Graph, partition: List[int]) -> float:
    """
    Calculate coverage: fraction of edges within communities.
    
    Parameters
    ----------
    graph : nk.Graph
        NetworkIt graph
    partition : List[int]
        Community assignment for each node
        
    Returns
    -------
    float
        Coverage value [0, 1]
    """
    if graph.numberOfEdges() == 0:
        return 1.0
    
    intra_community_edges = 0
    total_edges = 0
    
    for u, v in graph.iterEdges():
        total_edges += 1
        if partition[u] == partition[v]:
            intra_community_edges += 1
    
    return intra_community_edges / total_edges if total_edges > 0 else 1.0


def _filter_partitions_by_similarity(
    partitions_data: List[Dict[str, Any]],
    min_similarity: float
) -> List[Dict[str, Any]]:
    """
    Filter partitions based on pairwise similarity threshold.
    
    Parameters
    ----------
    partitions_data : List[Dict[str, Any]]
        List of partition results
    min_similarity : float
        Minimum normalized mutual information threshold
        
    Returns
    -------
    List[Dict[str, Any]]
        Filtered partitions that meet similarity criteria
    """
    if len(partitions_data) <= 1:
        return partitions_data
    
    logger.debug(f"Filtering partitions with min_similarity={min_similarity}")
    
    # Calculate pairwise similarities
    similarities = []
    partitions = [p["partition"] for p in partitions_data]
    
    for i in range(len(partitions)):
        for j in range(i + 1, len(partitions)):
            nmi = normalized_mutual_info_score(partitions[i], partitions[j])
            similarities.append(nmi)
    
    avg_similarity = np.mean(similarities)
    logger.debug(f"Average pairwise NMI: {avg_similarity:.3f}")
    
    if avg_similarity < min_similarity:
        logger.warning(
            f"Average partition similarity ({avg_similarity:.3f}) below threshold ({min_similarity})"
        )
        return []
    
    # Keep partitions that have high similarity with others
    keep_indices = set()
    for i in range(len(partitions)):
        similarities_with_i = []
        for j in range(len(partitions)):
            if i != j:
                nmi = normalized_mutual_info_score(partitions[i], partitions[j])
                similarities_with_i.append(nmi)
        
        if np.mean(similarities_with_i) >= min_similarity:
            keep_indices.add(i)
    
    filtered_partitions = [partitions_data[i] for i in sorted(keep_indices)]
    
    logger.debug(f"Kept {len(filtered_partitions)}/{len(partitions_data)} partitions after similarity filtering")
    
    return filtered_partitions


def _calculate_consensus_and_stability(
    partitions_data: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Calculate consensus partition and stability scores.
    
    Parameters
    ----------
    partitions_data : List[Dict[str, Any]]
        List of partition results from multiple iterations
        
    Returns
    -------
    Dict[str, Any]
        Dictionary with consensus data:
        - partitions: List of all partitions
        - consensus: Consensus partition
        - stability: Per-node stability scores
    """
    if not partitions_data:
        raise ValueError("No partitions provided for consensus calculation")
    
    partitions = [p["partition"] for p in partitions_data]
    n_nodes = len(partitions[0])
    n_iterations = len(partitions)
    
    logger.debug(f"Calculating consensus from {n_iterations} partitions for {n_nodes} nodes")
    
    # Calculate consensus using majority voting
    consensus_partition = []
    stability_scores = []
    
    for node in range(n_nodes):
        # Get community assignments for this node across all iterations
        assignments = [partition[node] for partition in partitions]
        
        # Find most common assignment (consensus)
        assignment_counts = Counter(assignments)
        consensus_assignment = assignment_counts.most_common(1)[0][0]
        
        # Calculate stability as fraction of iterations with consensus assignment
        stability = assignment_counts[consensus_assignment] / n_iterations
        
        consensus_partition.append(consensus_assignment)
        stability_scores.append(stability)
    
    # Relabel consensus communities to be contiguous starting from 0
    consensus_partition = _relabel_communities(consensus_partition)
    
    avg_stability = np.mean(stability_scores)
    logger.debug(f"Consensus calculated with average stability: {avg_stability:.3f}")
    
    return {
        "partitions": partitions,
        "consensus": consensus_partition,
        "stability": stability_scores
    }


def _handle_no_edges_graph(
    graph: nk.Graph,
    id_mapper: IDMapper,
    iterations: int
) -> pl.DataFrame:
    """
    Handle graphs with no edges (isolated nodes).
    
    In graphs with no edges, each node forms its own community.
    This is a degenerate case but needs to be handled gracefully.
    
    Parameters
    ----------
    graph : nk.Graph
        Graph with no edges
    id_mapper : IDMapper
        ID mapper for original node IDs
    iterations : int
        Number of iterations requested
        
    Returns
    -------
    pl.DataFrame
        Community detection results with each node in its own community
    """
    n_nodes = graph.numberOfNodes()
    
    # Each node is its own community (0, 1, 2, ...)
    partition = list(range(n_nodes))
    
    # Map internal node IDs to original node IDs
    original_ids = []
    for internal_id in range(n_nodes):
        try:
            original_id = id_mapper.get_original(internal_id)
            original_ids.append(original_id)
        except KeyError:
            logger.warning(f"Internal ID {internal_id} not found in mapper")
            original_ids.append(f"unknown_{internal_id}")
    
    # Build result dictionary
    result_data = {
        "node_id": original_ids,
        "community_consensus": partition,
        "stability": [1.0] * n_nodes  # Perfect stability for degenerate case
    }
    
    # Add iteration results (all identical since deterministic)
    for i in range(iterations):
        result_data[f"community_iter_{i}"] = partition.copy()
        result_data[f"modularity_iter_{i}"] = [0.0] * n_nodes  # No edges = zero modularity
        result_data[f"num_communities_iter_{i}"] = [n_nodes] * n_nodes  # Each node is own community
    
    # Create DataFrame and sort by node_id
    result_df = pl.DataFrame(result_data)
    result_df = result_df.sort("node_id")
    
    return result_df


def _relabel_communities(partition: List[int]) -> List[int]:
    """
    Relabel community IDs to be contiguous starting from 0.
    
    Parameters
    ----------
    partition : List[int]
        Original partition with potentially non-contiguous community IDs
        
    Returns
    -------
    List[int]
        Relabeled partition with contiguous community IDs
    """
    unique_communities = sorted(set(partition))
    community_map = {old_id: new_id for new_id, old_id in enumerate(unique_communities)}
    return [community_map[community_id] for community_id in partition]


def _create_result_dataframe(
    consensus_data: Dict[str, Any],
    partitions_data: List[Dict[str, Any]],
    id_mapper: IDMapper,
    n_nodes: int
) -> pl.DataFrame:
    """
    Create final result DataFrame with original node IDs.
    
    Parameters
    ----------
    consensus_data : Dict[str, Any]
        Consensus and stability data
    partitions_data : List[Dict[str, Any]]
        All partition results
    id_mapper : IDMapper
        ID mapper for original node IDs
    n_nodes : int
        Number of nodes in graph
        
    Returns
    -------
    pl.DataFrame
        Final result DataFrame with all community data
    """
    # Map internal node IDs to original node IDs
    original_ids = []
    for internal_id in range(n_nodes):
        try:
            original_id = id_mapper.get_original(internal_id)
            original_ids.append(original_id)
        except KeyError:
            # This shouldn't happen, but handle gracefully
            logger.warning(f"Internal ID {internal_id} not found in mapper")
            original_ids.append(f"unknown_{internal_id}")
    
    # Build result dictionary
    result_data = {
        "node_id": original_ids,
        "community_consensus": consensus_data["consensus"],
        "stability": consensus_data["stability"]
    }
    
    # Add individual iteration results
    for i, partition_data in enumerate(partitions_data):
        result_data[f"community_iter_{i}"] = partition_data["partition"]
        result_data[f"modularity_iter_{i}"] = [partition_data["modularity"]] * n_nodes
        result_data[f"num_communities_iter_{i}"] = [partition_data["num_communities"]] * n_nodes
    
    # Create DataFrame and sort by node_id
    result_df = pl.DataFrame(result_data)
    result_df = result_df.sort("node_id")
    
    return result_df


def get_community_summary(
    communities_df: pl.DataFrame,
    iteration: Optional[int] = None
) -> Dict[str, Any]:
    """
    Get summary statistics for community detection results.
    
    Parameters
    ----------
    communities_df : pl.DataFrame
        DataFrame returned by detect_communities()
    iteration : int, optional
        Specific iteration to summarize. If None, uses consensus.
        
    Returns
    -------
    Dict[str, Any]
        Summary statistics including:
        - num_communities: Number of communities
        - modularity: Modularity score (if available)
        - avg_stability: Average stability across nodes
        - community_sizes: List of community sizes
        - size_distribution: Statistics about community size distribution
    """
    if iteration is None:
        # Use consensus results
        community_col = "community_consensus"
        modularity = None
        if "modularity_iter_0" in communities_df.columns:
            modularity = communities_df["modularity_iter_0"][0]
    else:
        # Use specific iteration
        community_col = f"community_iter_{iteration}"
        modularity_col = f"modularity_iter_{iteration}"
        
        if community_col not in communities_df.columns:
            raise ValueError(f"Iteration {iteration} not found in results")
        
        modularity = communities_df[modularity_col][0] if modularity_col in communities_df.columns else None
    
    # Calculate community statistics
    community_assignments = communities_df[community_col].to_list()
    community_counts = Counter(community_assignments)
    
    community_sizes = list(community_counts.values())
    num_communities = len(community_sizes)
    
    # Size distribution statistics
    size_stats = {
        "min": min(community_sizes) if community_sizes else 0,
        "max": max(community_sizes) if community_sizes else 0,
        "mean": np.mean(community_sizes) if community_sizes else 0,
        "median": np.median(community_sizes) if community_sizes else 0,
        "std": np.std(community_sizes) if community_sizes else 0
    }
    
    # Average stability
    avg_stability = communities_df["stability"].mean() if "stability" in communities_df.columns else None
    
    return {
        "num_communities": num_communities,
        "modularity": modularity,
        "avg_stability": avg_stability,
        "community_sizes": sorted(community_sizes, reverse=True),
        "size_distribution": size_stats,
        "total_nodes": len(community_assignments)
    }


def identify_stable_communities(
    communities_df: pl.DataFrame,
    min_stability: float = 0.8,
    min_size: int = 3
) -> pl.DataFrame:
    """
    Identify communities with high stability and minimum size.
    
    Parameters
    ----------
    communities_df : pl.DataFrame
        DataFrame returned by detect_communities()
    min_stability : float, default 0.8
        Minimum stability threshold for nodes
    min_size : int, default 3
        Minimum community size
        
    Returns
    -------
    pl.DataFrame
        Filtered DataFrame with only stable communities
    """
    # Filter by stability
    stable_nodes = communities_df.filter(pl.col("stability") >= min_stability)
    
    if stable_nodes.height == 0:
        return stable_nodes
    
    # Count community sizes
    community_sizes = (
        stable_nodes
        .group_by("community_consensus")
        .agg(pl.count().alias("size"))
    )
    
    # Filter by minimum size
    large_communities = community_sizes.filter(pl.col("size") >= min_size)
    
    if large_communities.height == 0:
        return pl.DataFrame()
    
    # Keep only nodes in large, stable communities
    large_community_ids = large_communities["community_consensus"].to_list()
    
    result = stable_nodes.filter(
        pl.col("community_consensus").is_in(large_community_ids)
    )
    
    return result