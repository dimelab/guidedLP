"""
Guided Label Propagation implementation.

This module implements the core guided label propagation algorithm for semi-supervised
network analysis. It uses matrix-based calculations with sparse operations for efficiency
and supports both directed and undirected graphs.

Mathematical Foundation:
- Initialize label matrix Y where Y[i,j] = 1 if node i has label j (seed), 0 otherwise
- Create transition matrix P = D^-1 A (row-normalized adjacency matrix)
- Iteratively update: F^(t+1) = α P F^(t) + (1-α) Y
- Continue until convergence: max|F^(t+1) - F^(t)| < threshold

For directed graphs, run propagation twice:
- Out-degree: Use A as-is (influence propagation)
- In-degree: Use A^T (receptivity propagation)
"""

from typing import Dict, List, Tuple, Union, Optional, Any
import warnings
import numpy as np
import scipy.sparse as sp
import polars as pl
import networkit as nk

from guidedLP.common.id_mapper import IDMapper
from guidedLP.common.exceptions import (
    ValidationError,
    ConfigurationError,
    ConvergenceError,
    ComputationError,
    validate_parameter,
    require_positive,
    check_convergence
)
from guidedLP.common.logging_config import get_logger, LoggingTimer

logger = get_logger(__name__)


def guided_label_propagation(
    graph: nk.Graph,
    id_mapper: IDMapper,
    seed_labels: Dict[Any, str],
    labels: List[str],
    alpha: float = 0.85,
    max_iterations: int = 100,
    convergence_threshold: float = 1e-6,
    normalize: bool = True,
    directional: bool = True,
    n_jobs: int = 1,
    enable_noise_category: bool = False,
    noise_ratio: float = 0.1,
    confidence_threshold: float = 0.0
) -> Union[pl.DataFrame, Tuple[pl.DataFrame, pl.DataFrame]]:
    """
    Propagate labels from seed nodes through network using guided label propagation.
    
    This function implements the guided label propagation algorithm using efficient
    sparse matrix operations. For directed graphs, it can compute both out-degree
    (influence) and in-degree (receptivity) propagation.
    
    Mathematical Algorithm:
    1. Initialize label matrix Y (n × k) where n=nodes, k=labels
    2. Create transition matrix P = D^-1 A (row-normalized adjacency)
    3. Iteratively update: F^(t+1) = α P F^(t) + (1-α) Y
    4. Check convergence: max|F^(t+1) - F^(t)| < threshold
    5. Return final probabilities with original node IDs
    
    Time Complexity: O(k × i × E) where k=labels, i=iterations, E=edges
    Space Complexity: O(n × k) for label probability matrix
    
    Parameters
    ----------
    graph : nk.Graph
        NetworkIt graph (directed or undirected, must be weighted)
    id_mapper : IDMapper
        Bidirectional mapping between original and internal node IDs
    seed_labels : Dict[Any, str]
        Mapping from original node IDs to their known labels
        Example: {"user_123": "left", "user_456": "right"}
    labels : List[str]
        Complete list of possible labels (must include all values in seed_labels)
    alpha : float, default 0.85
        Propagation coefficient (0-1). Higher values emphasize neighbor influence,
        lower values preserve seed labels more strongly
    max_iterations : int, default 100
        Maximum number of propagation iterations
    convergence_threshold : float, default 1e-6
        Stop when maximum probability change between iterations < threshold
    normalize : bool, default True
        Normalize final probabilities to sum to 1.0 per node
    directional : bool, default True
        For directed graphs, compute both out-degree and in-degree propagation
    n_jobs : int, default 1
        Number of parallel jobs (reserved for future multi-label parallelization)
    enable_noise_category : bool, default True
        Automatically add a "noise" category for nodes with weak label associations.
        Helps identify outlier nodes and improves classification confidence.
    noise_ratio : float, default 0.1
        Fraction of non-seed nodes to randomly assign as noise seeds (0.0-1.0).
        Only used when enable_noise_category=True.
    confidence_threshold : float, default 0.0
        Minimum probability threshold for classification. Nodes with max probability
        below this threshold are classified as "uncertain" (0.0-1.0).
    
    Returns
    -------
    Union[pl.DataFrame, Tuple[pl.DataFrame, pl.DataFrame]]
        If directional=False OR graph is undirected:
            Single DataFrame with columns:
            - node_id: Original node ID
            - {label}_prob: Probability for each label (float)
            - dominant_label: Label with highest probability (str)
            - confidence: Maximum probability value (float)
            - is_seed: Whether node was in seed set (bool)
        
        If directional=True AND graph is directed:
            Tuple of (out_degree_df, in_degree_df) with same schema
    
    Raises
    ------
    ValidationError
        If seed_labels contains labels not in `labels` list, or if seeds
        reference nodes not in the graph
    ConfigurationError
        If alpha not in [0,1] or other parameter validation fails
    ConvergenceError
        If algorithm fails to converge within max_iterations
    ComputationError
        If matrix operations fail (e.g., due to numerical issues)
    
    Examples
    --------
    >>> # Basic usage
    >>> seeds = {"node1": "A", "node2": "B", "node3": "A"}
    >>> result = guided_label_propagation(graph, mapper, seeds, ["A", "B"])
    
    >>> # Directed graph with both propagation directions
    >>> out_result, in_result = guided_label_propagation(
    ...     directed_graph, mapper, seeds, ["A", "B"], directional=True
    ... )
    
    >>> # Lower alpha for stronger seed influence
    >>> result = guided_label_propagation(
    ...     graph, mapper, seeds, ["A", "B"], alpha=0.5
    ... )
    
    Notes
    -----
    - Seeds must be present in the graph (original IDs must map to internal IDs)
    - For disconnected components without seeds, nodes get uniform probability
    - Zero-degree nodes (isolates) retain their initial state
    - For directed graphs, out-degree measures influence, in-degree measures receptivity
    - Algorithm uses sparse matrix operations for memory efficiency on large graphs
    """
    
    logger.info(f"Starting guided label propagation with {len(seed_labels)} seeds, "
               f"{len(labels)} labels, alpha={alpha}")
    
    # Validate inputs
    _validate_inputs(graph, id_mapper, seed_labels, labels, alpha, 
                    max_iterations, convergence_threshold, enable_noise_category,
                    noise_ratio, confidence_threshold)
    
    # Process labels and seeds with noise category support
    processed_labels, processed_seed_labels = _process_noise_category(
        graph, id_mapper, seed_labels, labels, enable_noise_category, noise_ratio
    )
    
    # Check if graph is directed
    is_directed = graph.isDirected()
    logger.info(f"Graph type: {'directed' if is_directed else 'undirected'}, "
               f"nodes={graph.numberOfNodes()}, edges={graph.numberOfEdges()}")
    
    # For undirected graphs or directional=False, run single propagation
    if not is_directed or not directional:
        result = _run_single_propagation(
            graph, id_mapper, processed_seed_labels, processed_labels, alpha,
            max_iterations, convergence_threshold, normalize,
            direction="undirected" if not is_directed else "out_degree"
        )
        
        # Apply confidence thresholding if enabled
        if confidence_threshold > 0.0:
            result = _apply_confidence_threshold(result, confidence_threshold)
        
        logger.info("Completed single propagation")
        return result
    
    # For directed graphs with directional=True, run both directions
    else:
        logger.info("Running directional propagation for directed graph")
        
        # Out-degree propagation (influence)
        out_result = _run_single_propagation(
            graph, id_mapper, processed_seed_labels, processed_labels, alpha,
            max_iterations, convergence_threshold, normalize,
            direction="out_degree"
        )
        
        # In-degree propagation (receptivity) - use transposed adjacency
        in_result = _run_single_propagation(
            graph, id_mapper, processed_seed_labels, processed_labels, alpha,
            max_iterations, convergence_threshold, normalize,
            direction="in_degree"
        )
        
        # Apply confidence thresholding if enabled
        if confidence_threshold > 0.0:
            out_result = _apply_confidence_threshold(out_result, confidence_threshold)
            in_result = _apply_confidence_threshold(in_result, confidence_threshold)
        
        logger.info("Completed directional propagation")
        return out_result, in_result


def _validate_inputs(
    graph: nk.Graph,
    id_mapper: IDMapper,
    seed_labels: Dict[Any, str],
    labels: List[str],
    alpha: float,
    max_iterations: int,
    convergence_threshold: float,
    enable_noise_category: bool,
    noise_ratio: float,
    confidence_threshold: float
) -> None:
    """Validate all input parameters."""
    
    # Validate graph
    if graph.numberOfNodes() == 0:
        raise ValidationError("Graph has no nodes")
    
    if graph.numberOfEdges() == 0:
        warnings.warn("Graph has no edges - propagation will only affect seed nodes")
    
    # Validate alpha
    if not 0 <= alpha <= 1:
        raise ConfigurationError(
            f"Alpha must be between 0 and 1, got {alpha}",
            parameter="alpha",
            value=alpha
        )
    
    # Validate iteration parameters
    require_positive(max_iterations, "max_iterations")
    require_positive(convergence_threshold, "convergence_threshold")
    
    # Validate labels
    if not labels:
        raise ValidationError("Labels list cannot be empty")
    
    if len(set(labels)) != len(labels):
        raise ValidationError("Labels list contains duplicates")
    
    # Warn about single label scenarios
    if len(labels) == 1 and not enable_noise_category:
        warnings.warn(
            "GLP with single label provides limited discriminative power. "
            "Consider enabling noise category or adding additional labels.",
            UserWarning
        )
    
    # Validate noise category parameters
    if enable_noise_category:
        if not 0.0 <= noise_ratio <= 1.0:
            raise ConfigurationError(
                f"Noise ratio must be between 0.0 and 1.0, got {noise_ratio}",
                parameter="noise_ratio",
                value=noise_ratio
            )
    
    # Validate confidence threshold
    if not 0.0 <= confidence_threshold <= 1.0:
        raise ConfigurationError(
            f"Confidence threshold must be between 0.0 and 1.0, got {confidence_threshold}",
            parameter="confidence_threshold",
            value=confidence_threshold
        )
    
    # Validate seed_labels
    if not seed_labels:
        raise ValidationError("Seed labels dictionary cannot be empty")
    
    # Check that all seed labels are in the labels list
    seed_label_values = set(seed_labels.values())
    unknown_labels = seed_label_values - set(labels)
    if unknown_labels:
        raise ValidationError(
            f"Seed labels contain unknown labels: {unknown_labels}",
            details={"unknown_labels": list(unknown_labels), "valid_labels": labels}
        )
    
    # Check that all seed nodes exist in the graph (via ID mapper)
    missing_seeds = []
    for seed_id in seed_labels.keys():
        if not id_mapper.has_original(seed_id):
            missing_seeds.append(seed_id)
    
    if missing_seeds:
        raise ValidationError(
            f"Seed nodes not found in graph: {missing_seeds[:5]}{'...' if len(missing_seeds) > 5 else ''}",
            details={"missing_seeds_count": len(missing_seeds), "missing_seeds_sample": missing_seeds[:10]}
        )


def _run_single_propagation(
    graph: nk.Graph,
    id_mapper: IDMapper,
    seed_labels: Dict[Any, str],
    labels: List[str],
    alpha: float,
    max_iterations: int,
    convergence_threshold: float,
    normalize: bool,
    direction: str
) -> pl.DataFrame:
    """Run a single propagation (either undirected, out-degree, or in-degree)."""
    
    with LoggingTimer(f"Running {direction} propagation"):
        
        # Setup matrices and initial conditions
        n_nodes = graph.numberOfNodes()
        n_labels = len(labels)
        
        logger.debug(f"Matrix dimensions: {n_nodes} nodes × {n_labels} labels")
        
        # Create initial label matrix Y (n × k)
        Y = _initialize_label_matrix(graph, id_mapper, seed_labels, labels)
        
        # Create transition matrix P
        P = _create_transition_matrix(graph, direction)
        
        # Initialize propagation matrix
        F = Y.copy()
        
        # Iterative propagation
        converged_iteration = _iterative_propagation(
            F, P, Y, alpha, max_iterations, convergence_threshold
        )
        
        # Post-process results
        result_df = _create_results_dataframe(
            F, Y, labels, id_mapper, normalize, converged_iteration, direction
        )
        
        logger.info(f"Propagation converged after {converged_iteration} iterations")
        return result_df


def _initialize_label_matrix(
    graph: nk.Graph,
    id_mapper: IDMapper,
    seed_labels: Dict[Any, str],
    labels: List[str]
) -> np.ndarray:
    """
    Create initial label matrix Y from seed labels.
    
    This function creates the initial n×k label matrix Y where:
    - Y[i,j] = 1.0 if node i is a seed with label j
    - Y[i,j] = 0.0 otherwise
    
    Parameters
    ----------
    graph : nk.Graph
        NetworkIt graph to get node count
    id_mapper : IDMapper
        Mapping between original and internal node IDs
    seed_labels : Dict[Any, str]
        Mapping from original node IDs to their labels
    labels : List[str]
        Complete list of possible labels
    
    Returns
    -------
    np.ndarray
        Initial label matrix Y of shape (n_nodes, n_labels)
    
    Examples
    --------
    >>> seeds = {"node_1": "A", "node_3": "B"}
    >>> labels = ["A", "B"]
    >>> Y = _initialize_label_matrix(graph, mapper, seeds, labels)
    >>> Y.shape
    (n_nodes, 2)
    """
    n_nodes = graph.numberOfNodes()
    n_labels = len(labels)
    label_to_idx = {label: i for i, label in enumerate(labels)}
    
    Y = np.zeros((n_nodes, n_labels), dtype=np.float64)
    
    # Set seed nodes
    for original_id, label in seed_labels.items():
        internal_id = id_mapper.get_internal(original_id)
        label_idx = label_to_idx[label]
        Y[internal_id, label_idx] = 1.0
    
    logger.debug(f"Initialized {len(seed_labels)} seed nodes in label matrix")
    return Y


def _create_transition_matrix(graph: nk.Graph, direction: str) -> sp.csr_matrix:
    """Create row-normalized transition matrix P = D^-1 A."""
    
    with LoggingTimer("Creating transition matrix"):
        
        n_nodes = graph.numberOfNodes()
        
        # Build adjacency matrix in COO format for efficiency
        row_indices = []
        col_indices = []
        edge_weights = []
        
        for u, v in graph.iterEdges():
            weight = graph.weight(u, v)
            
            if direction == "in_degree":
                # For in-degree: transpose the adjacency matrix (v -> u)
                row_indices.append(v)
                col_indices.append(u)
                edge_weights.append(weight)
                
                # Add reverse edge for undirected graphs
                if not graph.isDirected():
                    row_indices.append(u)
                    col_indices.append(v)
                    edge_weights.append(weight)
                    
            else:  # out_degree or undirected
                # Standard adjacency matrix (u -> v)
                row_indices.append(u)
                col_indices.append(v)
                edge_weights.append(weight)
                
                # Add reverse edge for undirected graphs
                if not graph.isDirected():
                    row_indices.append(v)
                    col_indices.append(u)
                    edge_weights.append(weight)
        
        # Create sparse adjacency matrix
        adj_matrix = sp.coo_matrix(
            (edge_weights, (row_indices, col_indices)),
            shape=(n_nodes, n_nodes),
            dtype=np.float64
        ).tocsr()
        
        # Create row-normalized transition matrix P = D^-1 A
        # Calculate row sums (degrees)
        row_sums = np.array(adj_matrix.sum(axis=1)).flatten()
        
        # Handle zero-degree nodes
        zero_degree_mask = (row_sums == 0)
        if zero_degree_mask.any():
            logger.debug(f"Found {zero_degree_mask.sum()} zero-degree nodes")
            # For zero-degree nodes, use zero rows (no propagation)
            row_sums[zero_degree_mask] = 1.0  # Avoid division by zero
        
        # Create diagonal matrix for normalization
        row_sums_inv = 1.0 / row_sums
        row_sums_inv[zero_degree_mask] = 0.0  # Reset zero-degree nodes
        
        # Normalize: P = D^-1 A
        P = sp.diags(row_sums_inv).dot(adj_matrix)
        
        logger.debug(f"Transition matrix shape: {P.shape}, nnz: {P.nnz}")
        return P


def _propagate_iteration(
    F: np.ndarray,
    P: sp.csr_matrix,
    Y: np.ndarray,
    alpha: float
) -> np.ndarray:
    """
    Perform a single propagation iteration step.
    
    This function implements the core propagation update:
    F^(t+1) = α P F^(t) + (1-α) Y
    
    Parameters
    ----------
    F : np.ndarray
        Current label probability matrix (n_nodes × n_labels)
    P : sp.csr_matrix
        Row-normalized transition matrix (n_nodes × n_nodes)
    Y : np.ndarray
        Initial seed label matrix (n_nodes × n_labels)
    alpha : float
        Propagation coefficient (0-1)
    
    Returns
    -------
    np.ndarray
        Updated label probability matrix F^(t+1)
    
    Raises
    ------
    ComputationError
        If matrix operations fail due to numerical issues
    
    Examples
    --------
    >>> F_new = _propagate_iteration(F_old, P, Y, alpha=0.85)
    >>> F_new.shape == F_old.shape
    True
    
    Notes
    -----
    This function performs the mathematical core of label propagation:
    - P @ F: propagates current probabilities through network connections
    - α controls balance between neighbor influence vs. seed retention
    - (1-α) Y ensures seeds maintain their original labels
    """
    try:
        # Matrix multiplication: P @ F (propagate through network)
        propagated = P.dot(F)
        
        # Update: F = α * propagated + (1-α) * Y
        F_new = alpha * propagated + (1 - alpha) * Y
        
        return F_new
        
    except Exception as e:
        raise ComputationError(
            f"Matrix operation failed during propagation iteration",
            operation="matrix_multiplication",
            error_type="numerical",
            cause=e
        )


def _check_convergence(
    F_new: np.ndarray,
    F_old: np.ndarray,
    threshold: float
) -> Tuple[bool, float]:
    """
    Check if propagation has converged.
    
    Convergence is determined by checking if the maximum absolute change
    in any probability value is below the specified threshold.
    
    Parameters
    ----------
    F_new : np.ndarray
        New label probability matrix
    F_old : np.ndarray
        Previous label probability matrix
    threshold : float
        Convergence threshold
    
    Returns
    -------
    Tuple[bool, float]
        (has_converged, max_change) where:
        - has_converged: True if max_change < threshold
        - max_change: Maximum absolute change between F_new and F_old
    
    Examples
    --------
    >>> converged, change = _check_convergence(F_new, F_old, 1e-6)
    >>> if converged:
    ...     print(f"Converged with change {change}")
    
    Notes
    -----
    Uses the L∞ norm (maximum absolute difference) as the convergence criterion.
    This is appropriate for probability matrices where we want all values to
    stabilize within the threshold.
    """
    max_change = np.max(np.abs(F_new - F_old))
    has_converged = max_change < threshold
    
    return has_converged, max_change


def _create_results_dataframe(
    F: np.ndarray,
    Y: np.ndarray,
    labels: List[str],
    id_mapper: IDMapper,
    normalize: bool,
    converged_iteration: int,
    direction: str
) -> pl.DataFrame:
    """
    Convert propagation results to Polars DataFrame with original node IDs.
    
    This function post-processes the final probability matrix and creates
    a structured DataFrame with all necessary output columns.
    
    Parameters
    ----------
    F : np.ndarray
        Final label probability matrix (n_nodes × n_labels)
    Y : np.ndarray
        Initial seed label matrix (for seed identification)
    labels : List[str]
        List of label names
    id_mapper : IDMapper
        Mapping between original and internal node IDs
    normalize : bool
        Whether to normalize probabilities to sum to 1.0
    converged_iteration : int
        Number of iterations until convergence
    direction : str
        Propagation direction ("undirected", "out_degree", "in_degree")
    
    Returns
    -------
    pl.DataFrame
        Structured results with columns:
        - node_id: Original node ID
        - {label}_prob: Probability for each label
        - dominant_label: Label with highest probability
        - confidence: Maximum probability value
        - is_seed: Whether node was in seed set
    
    Examples
    --------
    >>> df = _create_results_dataframe(F, Y, ["A", "B"], mapper, True, 15, "undirected")
    >>> print(df.columns)
    ['node_id', 'A_prob', 'B_prob', 'dominant_label', 'confidence', 'is_seed']
    
    Notes
    -----
    - Handles probability normalization if requested
    - Identifies seed nodes from initial matrix Y
    - Maps internal node IDs back to original IDs
    - Calculates dominant label and confidence scores
    """
    with LoggingTimer("Post-processing results"):
        
        n_nodes, n_labels = F.shape
        
        # Normalize probabilities if requested
        if normalize:
            row_sums = F.sum(axis=1, keepdims=True)
            # Handle rows with zero probabilities
            zero_mask = (row_sums.flatten() == 0)
            if zero_mask.any():
                # For nodes with no probabilities, set uniform distribution
                uniform_prob = 1.0 / n_labels
                F[zero_mask, :] = uniform_prob
                row_sums = F.sum(axis=1, keepdims=True)
            
            F = F / row_sums
        
        # Calculate dominant label and confidence
        dominant_indices = np.argmax(F, axis=1)
        dominant_labels = [labels[idx] for idx in dominant_indices]
        confidence_scores = np.max(F, axis=1)
        
        # Determine which nodes were seeds
        seed_mask = (Y.sum(axis=1) > 0)
        
        # Create result data
        result_data = {}
        
        # Original node IDs
        original_ids = id_mapper.get_original_batch(list(range(n_nodes)))
        result_data["node_id"] = original_ids
        
        # Probability columns for each label
        for i, label in enumerate(labels):
            result_data[f"{label}_prob"] = F[:, i].tolist()
        
        # Additional columns
        result_data["dominant_label"] = dominant_labels
        result_data["confidence"] = confidence_scores.tolist()
        result_data["is_seed"] = seed_mask.tolist()
        
        # Create DataFrame
        df = pl.DataFrame(result_data)
        
        logger.info(f"Created result DataFrame with {len(df)} nodes")
        logger.debug(f"Label distribution: {df['dominant_label'].value_counts().to_dict()}")
        
        return df


def _iterative_propagation(
    F: np.ndarray,
    P: sp.csr_matrix,
    Y: np.ndarray,
    alpha: float,
    max_iterations: int,
    convergence_threshold: float
) -> int:
    """
    Perform iterative label propagation until convergence.
    
    This function runs the core iterative propagation loop using the helper
    functions for individual operations. It continues until convergence or
    maximum iterations are reached.
    
    Parameters
    ----------
    F : np.ndarray
        Initial label probability matrix (modified in-place)
    P : sp.csr_matrix
        Row-normalized transition matrix
    Y : np.ndarray
        Initial seed label matrix
    alpha : float
        Propagation coefficient
    max_iterations : int
        Maximum number of iterations
    convergence_threshold : float
        Convergence threshold for stopping
    
    Returns
    -------
    int
        Number of iterations until convergence
    
    Raises
    ------
    ConvergenceError
        If algorithm fails to converge within max_iterations
    """
    with LoggingTimer("Iterative propagation"):
        
        for iteration in range(max_iterations):
            F_prev = F.copy()
            
            # Perform single propagation iteration
            F_new = _propagate_iteration(F, P, Y, alpha)
            F[:] = F_new  # Update in-place
            
            # Check convergence
            has_converged, max_change = _check_convergence(F, F_prev, convergence_threshold)
            
            if iteration % 10 == 0 or has_converged:
                logger.debug(f"Iteration {iteration}: max_change = {max_change:.2e}")
            
            if has_converged:
                return iteration + 1
        
        # If we reach here, algorithm didn't converge
        final_change = max_change  # From last iteration
        
        # Check if result might still be usable
        if final_change <= 10 * convergence_threshold:
            warnings.warn(
                f"Propagation did not fully converge (final_change={final_change:.2e}, "
                f"threshold={convergence_threshold:.2e}) but result may be usable"
            )
            return max_iterations
        
        raise ConvergenceError(
            "Label propagation failed to converge",
            algorithm="guided_label_propagation",
            iterations=max_iterations,
            max_iterations=max_iterations,
            final_change=final_change,
            threshold=convergence_threshold
        )



def get_propagation_info(
    graph: nk.Graph,
    seed_labels: Dict[Any, str],
    labels: List[str]
) -> Dict[str, Any]:
    """
    Get information about a potential propagation run without executing it.
    
    This utility function provides estimates and diagnostics for a guided label
    propagation run, helping users understand the computational requirements
    and potential issues before running the full algorithm.
    
    Parameters
    ----------
    graph : nk.Graph
        NetworkIt graph to analyze
    seed_labels : Dict[Any, str]
        Mapping from node IDs to their labels
    labels : List[str]
        List of all possible labels
    
    Returns
    -------
    Dict[str, Any]
        Information dictionary containing:
        - graph_stats: Basic graph statistics
        - seed_stats: Seed set analysis
        - memory_estimate: Estimated memory requirements
        - computational_estimate: Expected computational complexity
        - potential_issues: List of potential problems
    
    Examples
    --------
    >>> info = get_propagation_info(graph, seeds, ["A", "B"])
    >>> print(f"Estimated memory: {info['memory_estimate']['total_mb']:.1f} MB")
    >>> if info['potential_issues']:
    ...     print("Warnings:", info['potential_issues'])
    """
    
    n_nodes = graph.numberOfNodes()
    n_edges = graph.numberOfEdges()
    n_labels = len(labels)
    n_seeds = len(seed_labels)
    
    # Basic graph statistics
    graph_stats = {
        "nodes": n_nodes,
        "edges": n_edges,
        "labels": n_labels,
        "seeds": n_seeds,
        "is_directed": graph.isDirected(),
        "density": n_edges / (n_nodes * (n_nodes - 1) / 2) if n_nodes > 1 else 0,
        "seed_ratio": n_seeds / n_nodes if n_nodes > 0 else 0
    }
    
    # Seed set analysis
    seed_label_counts = {}
    for label in labels:
        seed_label_counts[label] = sum(1 for l in seed_labels.values() if l == label)
    
    seed_stats = {
        "seeds_per_label": seed_label_counts,
        "min_seeds_per_label": min(seed_label_counts.values()) if seed_label_counts else 0,
        "max_seeds_per_label": max(seed_label_counts.values()) if seed_label_counts else 0,
        "label_balance": min(seed_label_counts.values()) / max(seed_label_counts.values()) 
                       if seed_label_counts and max(seed_label_counts.values()) > 0 else 0
    }
    
    # Memory estimates (in MB)
    float64_size = 8  # bytes
    label_matrix_mb = (n_nodes * n_labels * float64_size) / (1024 * 1024)
    adjacency_matrix_mb = (n_edges * 3 * float64_size) / (1024 * 1024)  # COO format
    
    memory_estimate = {
        "label_matrix_mb": label_matrix_mb,
        "adjacency_matrix_mb": adjacency_matrix_mb,
        "total_mb": label_matrix_mb * 2 + adjacency_matrix_mb,  # F, Y, and P matrices
        "large_network": label_matrix_mb > 100  # Flag for networks > 100MB
    }
    
    # Computational estimates
    estimated_iterations = min(50, max(10, int(np.log10(n_nodes) * 10)))  # Rough heuristic
    computational_estimate = {
        "estimated_iterations": estimated_iterations,
        "ops_per_iteration": n_labels * n_edges,
        "total_ops_estimate": estimated_iterations * n_labels * n_edges,
        "expected_runtime_class": "fast" if n_edges < 10000 else "medium" if n_edges < 100000 else "slow"
    }
    
    # Potential issues
    potential_issues = []
    
    if n_seeds == 0:
        potential_issues.append("No seed nodes provided")
    
    if seed_stats["min_seeds_per_label"] == 0:
        missing_labels = [l for l, count in seed_label_counts.items() if count == 0]
        potential_issues.append(f"No seeds for labels: {missing_labels}")
    
    if seed_stats["label_balance"] < 0.1:
        potential_issues.append("Highly imbalanced seed set - consider balancing")
    
    if n_edges == 0:
        potential_issues.append("Graph has no edges - propagation will be limited")
    
    if memory_estimate["total_mb"] > 1000:
        potential_issues.append("High memory usage expected (>1GB)")
    
    if graph_stats["density"] < 0.001 and n_nodes > 1000:
        potential_issues.append("Very sparse graph - may have convergence issues")
    
    return {
        "graph_stats": graph_stats,
        "seed_stats": seed_stats,
        "memory_estimate": memory_estimate,
        "computational_estimate": computational_estimate,
        "potential_issues": potential_issues
    }


def _process_noise_category(
    graph: nk.Graph,
    id_mapper: IDMapper,
    seed_labels: Dict[Any, str],
    labels: List[str],
    enable_noise_category: bool,
    noise_ratio: float
) -> Tuple[List[str], Dict[Any, str]]:
    """
    Process labels and seed labels to include noise category if enabled.
    
    This function implements the noise category functionality from reference 
    implementations, automatically adding a "noise" category and generating
    noise seeds to improve classification robustness.
    
    Parameters
    ----------
    graph : nk.Graph
        NetworkIt graph
    id_mapper : IDMapper
        ID mapper for the graph
    seed_labels : Dict[Any, str]
        Original seed labels mapping
    labels : List[str]
        Original labels list
    enable_noise_category : bool
        Whether to add noise category
    noise_ratio : float
        Fraction of non-seed nodes to use as noise seeds
    
    Returns
    -------
    processed_labels : List[str]
        Labels list with noise category added if enabled
    processed_seed_labels : Dict[Any, str]
        Seed labels with noise seeds added if enabled
    
    Notes
    -----
    Noise seeds are randomly sampled from non-seed nodes to provide
    the algorithm with examples of nodes that don't belong to any
    specific category. This improves classification confidence and
    helps identify outlier nodes.
    """
    import random
    
    processed_labels = labels.copy()
    processed_seed_labels = seed_labels.copy()
    
    if not enable_noise_category:
        return processed_labels, processed_seed_labels
    
    # Add noise category if not already present
    if "noise" not in processed_labels:
        processed_labels.append("noise")
        logger.info("Added 'noise' category to labels")
    
    # Generate noise seeds if noise category was added
    if "noise" not in seed_labels.values():
        noise_seeds = _generate_noise_seeds(
            graph, id_mapper, seed_labels, noise_ratio
        )
        processed_seed_labels.update(noise_seeds)
        logger.info(f"Generated {len(noise_seeds)} noise seeds")
    
    return processed_labels, processed_seed_labels


def _generate_noise_seeds(
    graph: nk.Graph,
    id_mapper: IDMapper,
    seed_labels: Dict[Any, str],
    noise_ratio: float
) -> Dict[Any, str]:
    """
    Generate noise seeds from non-seed nodes.
    
    Randomly samples nodes that are not already in the seed set
    and assigns them the "noise" label. This provides the algorithm
    with examples of nodes that don't strongly belong to any category.
    
    Parameters
    ----------
    graph : nk.Graph
        NetworkIt graph
    id_mapper : IDMapper
        ID mapper for the graph
    seed_labels : Dict[Any, str]
        Existing seed labels
    noise_ratio : float
        Fraction of non-seed nodes to use as noise seeds
    
    Returns
    -------
    Dict[Any, str]
        Mapping of selected nodes to "noise" label
    """
    import random
    
    # Get all nodes that are not already seeds
    all_nodes = set(range(graph.numberOfNodes()))
    seed_node_internals = {id_mapper.get_internal(seed_id) for seed_id in seed_labels.keys()}
    non_seed_nodes = all_nodes - seed_node_internals
    
    if not non_seed_nodes:
        logger.warning("No non-seed nodes available for noise seed generation")
        return {}
    
    # Calculate number of noise seeds
    n_existing_seeds = len(seed_labels)
    n_noise_seeds = max(1, int(noise_ratio * n_existing_seeds))
    n_noise_seeds = min(n_noise_seeds, len(non_seed_nodes))
    
    # Randomly sample noise seeds
    random.seed(42)  # For reproducibility
    selected_internal_ids = random.sample(list(non_seed_nodes), n_noise_seeds)
    
    # Convert back to original IDs
    noise_seeds = {}
    for internal_id in selected_internal_ids:
        original_id = id_mapper.get_original(internal_id)
        noise_seeds[original_id] = "noise"
    
    logger.debug(f"Selected {len(noise_seeds)} noise seeds from {len(non_seed_nodes)} candidates")
    return noise_seeds


def _apply_confidence_threshold(
    result_df: pl.DataFrame,
    confidence_threshold: float
) -> pl.DataFrame:
    """
    Apply confidence thresholding to classification results.
    
    Nodes with maximum probability below the threshold are reclassified
    as "uncertain", providing a mechanism to identify low-confidence
    predictions that should not be trusted.
    
    Parameters
    ----------
    result_df : pl.DataFrame
        Results DataFrame from GLP
    confidence_threshold : float
        Minimum confidence threshold (0.0-1.0)
    
    Returns
    -------
    pl.DataFrame
        Results with uncertain classifications applied
    """
    if confidence_threshold <= 0.0:
        return result_df
    
    # Identify low-confidence nodes
    low_confidence_mask = result_df["confidence"] < confidence_threshold
    n_uncertain = low_confidence_mask.sum()
    
    if n_uncertain > 0:
        # Update dominant label for low-confidence nodes
        result_df = result_df.with_columns([
            pl.when(pl.col("confidence") < confidence_threshold)
            .then(pl.lit("uncertain"))
            .otherwise(pl.col("dominant_label"))
            .alias("dominant_label")
        ])
        
        logger.info(f"Classified {n_uncertain} nodes as 'uncertain' due to low confidence")
    
    return result_df