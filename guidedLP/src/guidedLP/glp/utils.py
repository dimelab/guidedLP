"""
Utility functions for Guided Label Propagation.

This module provides helper functions for preprocessing data, parameter tuning,
and other utilities to support the GLP workflow. It includes functions for
creating balanced seed sets and suggesting optimal alpha values based on
network properties.
"""

from typing import Dict, List, Any, Optional
import warnings
import random
import numpy as np
import networkit as nk

from guidedLP.common.exceptions import (
    ValidationError,
    ConfigurationError,
    ComputationError
)
from guidedLP.common.logging_config import get_logger, LoggingTimer

logger = get_logger(__name__)


def create_balanced_seed_set(
    candidate_seeds: Dict[Any, str],
    labels: List[str],
    n_per_label: Optional[int] = None,
    method: str = "undersample",
    random_seed: Optional[int] = None
) -> Dict[Any, str]:
    """
    Create balanced seed set from potentially imbalanced candidate data.
    
    This function addresses class imbalance in seed node selection by creating
    a balanced set where each label has the same number of seed nodes. This
    prevents bias toward majority classes during label propagation.
    
    Parameters
    ----------
    candidate_seeds : Dict[Any, str]
        All available labeled nodes (node_id -> label)
    labels : List[str]
        All possible labels to balance
    n_per_label : Optional[int], default None
        Target number of seeds per label. If None, uses the size of the
        smallest class (natural undersampling).
    method : str, default "undersample"
        Balancing method:
        - "undersample": Randomly sample n_per_label from majority classes
        - "oversample": Allow duplicates in minority classes to reach n_per_label
    random_seed : Optional[int], default None
        Random seed for reproducible sampling
        
    Returns
    -------
    Dict[Any, str]
        Balanced seed set with equal representation per label
        
    Raises
    ------
    ValidationError
        If candidate_seeds is empty, labels not found in seeds, or insufficient seeds
    ConfigurationError
        If method is invalid or n_per_label is invalid
        
    Examples
    --------
    >>> # Undersample to balance classes
    >>> candidates = {"n1": "A", "n2": "A", "n3": "A", "n4": "B", "n5": "B"}
    >>> balanced = create_balanced_seed_set(candidates, ["A", "B"])
    >>> # Result: 2 seeds per class (undersampling A)
    
    >>> # Oversample to reach target size
    >>> balanced = create_balanced_seed_set(
    ...     candidates, ["A", "B"], n_per_label=3, method="oversample"
    ... )
    >>> # Result: 3 seeds per class (oversampling B with replacement)
    
    >>> # Reproducible balancing
    >>> balanced = create_balanced_seed_set(
    ...     candidates, ["A", "B"], random_seed=42
    ... )
    
    Notes
    -----
    - Undersampling reduces total number of seeds but ensures balance
    - Oversampling maintains or increases total seeds with possible duplicates
    - For very small seed sets, consider using all available seeds instead
    - Balanced seeds often improve propagation performance on imbalanced networks
    """
    
    logger.info(f"Creating balanced seed set from {len(candidate_seeds)} candidates, "
               f"method={method}, n_per_label={n_per_label}")
    
    # Validate inputs
    _validate_balance_inputs(candidate_seeds, labels, n_per_label, method)
    
    # Set random seed if provided
    if random_seed is not None:
        random.seed(random_seed)
        np.random.seed(random_seed)
    
    with LoggingTimer("Creating balanced seed set"):
        
        # Group seeds by label
        seeds_by_label = _group_seeds_by_label(candidate_seeds, labels)
        
        # Determine target size per label
        target_size = _determine_target_size(seeds_by_label, n_per_label, method)
        
        logger.info(f"Target size per label: {target_size}")
        
        # Apply balancing method
        if method == "undersample":
            balanced_seeds = _undersample_seeds(seeds_by_label, target_size)
        elif method == "oversample":
            balanced_seeds = _oversample_seeds(seeds_by_label, target_size)
        else:
            raise ConfigurationError(
                f"Unknown balancing method: {method}",
                parameter="method",
                value=method
            )
        
        logger.info(f"Created balanced seed set with {len(balanced_seeds)} total seeds")
        
        # Log final distribution
        final_counts = {}
        for node_id, label in balanced_seeds.items():
            final_counts[label] = final_counts.get(label, 0) + 1
        logger.debug(f"Final label distribution: {final_counts}")
        
        return balanced_seeds


def suggest_alpha_value(
    graph: nk.Graph,
    seed_count: int,
    method: str = "network_structure"
) -> float:
    """
    Suggest optimal alpha value based on network properties.
    
    This function provides data-driven recommendations for the alpha parameter
    in guided label propagation based on network structure and seed density.
    Alpha controls the balance between neighbor influence and seed retention.
    
    Parameters
    ----------
    graph : nk.Graph
        NetworkIt graph to analyze
    seed_count : int
        Number of seed nodes that will be used
    method : str, default "network_structure"
        Method for alpha suggestion:
        - "network_structure": Based on clustering coefficient and density
        - "seed_ratio": Based on ratio of seeds to total nodes
        
    Returns
    -------
    float
        Suggested alpha value between 0.0 and 1.0
        
    Raises
    ------
    ValidationError
        If graph is empty or seed_count is invalid
    ConfigurationError
        If method is unknown
        
    Examples
    --------
    >>> # Alpha based on network clustering
    >>> alpha = suggest_alpha_value(graph, seed_count=50, method="network_structure")
    >>> print(f"Suggested alpha: {alpha:.3f}")
    
    >>> # Alpha based on seed density
    >>> alpha = suggest_alpha_value(graph, seed_count=20, method="seed_ratio")
    
    >>> # Use suggestion in GLP
    >>> alpha = suggest_alpha_value(graph, len(seeds))
    >>> results = guided_label_propagation(graph, mapper, seeds, labels, alpha=alpha)
    
    Notes
    -----
    Mathematical Formulations:
    
    **Network Structure Method:**
    - High clustering → lower alpha (favor local propagation)
    - Low clustering → higher alpha (broader propagation needed)
    - Formula: alpha = 0.5 + 0.4 * (1 - clustering_coefficient)
    
    **Seed Ratio Method:**
    - Many seeds → lower alpha (seeds provide strong signal)
    - Few seeds → higher alpha (need more propagation)
    - Formula: alpha = 0.95 - 0.5 * (seed_count / total_nodes)
    
    Suggestions are heuristic starting points and may need fine-tuning
    based on specific network properties and propagation goals.
    """
    
    logger.info(f"Suggesting alpha value for graph with {graph.numberOfNodes()} nodes, "
               f"{seed_count} seeds, method={method}")
    
    # Validate inputs
    _validate_alpha_inputs(graph, seed_count, method)
    
    with LoggingTimer(f"Computing alpha suggestion using {method} method"):
        
        if method == "network_structure":
            alpha = _alpha_from_network_structure(graph)
        elif method == "seed_ratio":
            alpha = _alpha_from_seed_ratio(graph, seed_count)
        else:
            raise ConfigurationError(
                f"Unknown alpha suggestion method: {method}",
                parameter="method",
                value=method
            )
        
        # Ensure alpha is in valid range
        alpha = max(0.1, min(0.99, alpha))
        
        logger.info(f"Suggested alpha: {alpha:.3f}")
        
        return alpha


def _validate_balance_inputs(
    candidate_seeds: Dict[Any, str],
    labels: List[str],
    n_per_label: Optional[int],
    method: str
) -> None:
    """Validate inputs for seed balancing."""
    
    if not candidate_seeds:
        raise ValidationError("candidate_seeds cannot be empty")
    
    if not labels:
        raise ValidationError("labels list cannot be empty")
    
    if method not in ["undersample", "oversample"]:
        raise ConfigurationError(
            f"method must be 'undersample' or 'oversample', got '{method}'",
            parameter="method",
            value=method
        )
    
    if n_per_label is not None and n_per_label <= 0:
        raise ConfigurationError(
            f"n_per_label must be positive, got {n_per_label}",
            parameter="n_per_label",
            value=n_per_label
        )
    
    # Check that all labels appear in candidate seeds
    seed_labels = set(candidate_seeds.values())
    missing_labels = set(labels) - seed_labels
    if missing_labels:
        raise ValidationError(
            f"Labels not found in candidate seeds: {missing_labels}",
            details={"missing_labels": list(missing_labels), "available_labels": list(seed_labels)}
        )


def _validate_alpha_inputs(graph: nk.Graph, seed_count: int, method: str) -> None:
    """Validate inputs for alpha suggestion."""
    
    if graph.numberOfNodes() == 0:
        raise ValidationError("graph cannot be empty")
    
    if seed_count <= 0:
        raise ValidationError(f"seed_count must be positive, got {seed_count}")
    
    if seed_count > graph.numberOfNodes():
        raise ValidationError(
            f"seed_count ({seed_count}) cannot exceed graph size ({graph.numberOfNodes()})"
        )
    
    if method not in ["network_structure", "seed_ratio"]:
        raise ConfigurationError(
            f"method must be 'network_structure' or 'seed_ratio', got '{method}'",
            parameter="method",
            value=method
        )


def _group_seeds_by_label(
    candidate_seeds: Dict[Any, str], 
    labels: List[str]
) -> Dict[str, List[Any]]:
    """Group seed nodes by their labels."""
    
    seeds_by_label = {label: [] for label in labels}
    
    for node_id, label in candidate_seeds.items():
        if label in seeds_by_label:
            seeds_by_label[label].append(node_id)
    
    # Log distribution
    counts = {label: len(nodes) for label, nodes in seeds_by_label.items()}
    logger.debug(f"Original label distribution: {counts}")
    
    return seeds_by_label


def _determine_target_size(
    seeds_by_label: Dict[str, List[Any]], 
    n_per_label: Optional[int], 
    method: str
) -> int:
    """Determine target number of seeds per label."""
    
    label_counts = {label: len(nodes) for label, nodes in seeds_by_label.items()}
    
    if n_per_label is not None:
        target_size = n_per_label
        
        # Validate that target is achievable
        if method == "undersample":
            min_count = min(label_counts.values())
            if target_size > min_count:
                raise ValidationError(
                    f"Cannot undersample to {target_size} per label: "
                    f"minimum available is {min_count}"
                )
        
    else:
        # Use size of smallest class (natural undersampling)
        target_size = min(label_counts.values())
        
        if target_size == 0:
            raise ValidationError("At least one label has no seeds available")
    
    return target_size


def _undersample_seeds(
    seeds_by_label: Dict[str, List[Any]], 
    target_size: int
) -> Dict[Any, str]:
    """Undersample majority classes to achieve balance."""
    
    balanced_seeds = {}
    
    for label, nodes in seeds_by_label.items():
        if len(nodes) >= target_size:
            # Randomly sample target_size nodes
            sampled_nodes = random.sample(nodes, target_size)
        else:
            # Use all available nodes (shouldn't happen with proper validation)
            sampled_nodes = nodes
            warnings.warn(f"Label '{label}' has fewer than {target_size} seeds")
        
        # Add to balanced set
        for node_id in sampled_nodes:
            balanced_seeds[node_id] = label
    
    return balanced_seeds


def _oversample_seeds(
    seeds_by_label: Dict[str, List[Any]], 
    target_size: int
) -> Dict[Any, str]:
    """Oversample minority classes to achieve balance."""
    
    balanced_seeds = {}
    
    for label, nodes in seeds_by_label.items():
        if len(nodes) >= target_size:
            # Randomly sample target_size nodes without replacement
            sampled_nodes = random.sample(nodes, target_size)
        else:
            # Sample with replacement to reach target size
            sampled_nodes = random.choices(nodes, k=target_size)
        
        # Add to balanced set, handling duplicates by creating unique keys
        for i, node_id in enumerate(sampled_nodes):
            # For oversampling, create unique keys for potential duplicates
            unique_key = f"{node_id}_oversample_{i}" if sampled_nodes.count(node_id) > 1 and i > sampled_nodes.index(node_id) else node_id
            balanced_seeds[unique_key] = label
    
    return balanced_seeds


def _alpha_from_network_structure(graph: nk.Graph) -> float:
    """Suggest alpha based on network clustering and structure."""
    
    try:
        # Calculate global clustering coefficient
        if graph.numberOfEdges() == 0:
            # No edges - use default
            clustering_coeff = 0.0
        else:
            # Use NetworkIt's clustering coefficient
            clustering = nk.centrality.LocalClusteringCoefficient(graph)
            clustering.run()
            
            # Get global clustering coefficient (average)
            clustering_scores = clustering.scores()
            if clustering_scores:
                clustering_coeff = np.mean(clustering_scores)
            else:
                clustering_coeff = 0.0
        
        # Formula: alpha = 0.5 + 0.4 * (1 - clustering_coefficient)
        # High clustering → lower alpha (0.5)
        # Low clustering → higher alpha (0.9)
        alpha = 0.5 + 0.4 * (1 - clustering_coeff)
        
        logger.debug(f"Network clustering coefficient: {clustering_coeff:.3f}, "
                    f"suggested alpha: {alpha:.3f}")
        
        return alpha
        
    except Exception as e:
        logger.warning(f"Failed to compute network structure alpha: {e}, using default")
        return 0.85  # Default fallback


def _alpha_from_seed_ratio(graph: nk.Graph, seed_count: int) -> float:
    """Suggest alpha based on seed density in network."""
    
    total_nodes = graph.numberOfNodes()
    seed_ratio = seed_count / total_nodes
    
    # Formula: alpha = 0.95 - 0.5 * (seed_count / total_nodes)
    # Many seeds → lower alpha (favor seed retention)
    # Few seeds → higher alpha (need more propagation)
    alpha = 0.95 - 0.5 * seed_ratio
    
    logger.debug(f"Seed ratio: {seed_ratio:.3f} ({seed_count}/{total_nodes}), "
                f"suggested alpha: {alpha:.3f}")
    
    return alpha


def get_seed_statistics(seed_labels: Dict[Any, str], labels: List[str]) -> Dict[str, Any]:
    """
    Analyze seed set statistics and balance.
    
    This utility function provides diagnostic information about a seed set,
    including distribution balance, coverage, and recommendations.
    
    Parameters
    ----------
    seed_labels : Dict[Any, str]
        Seed nodes and their labels
    labels : List[str]
        All possible labels
        
    Returns
    -------
    Dict[str, Any]
        Statistics about the seed set including:
        - label_counts: Number of seeds per label
        - total_seeds: Total number of seed nodes
        - balance_ratio: Ratio of min to max class size
        - is_balanced: Whether distribution is reasonably balanced
        - recommendations: Suggested improvements
        
    Examples
    --------
    >>> stats = get_seed_statistics(seed_labels, ["A", "B", "C"])
    >>> print(f"Balance ratio: {stats['balance_ratio']:.2f}")
    >>> if not stats['is_balanced']:
    ...     print("Recommendations:", stats['recommendations'])
    """
    
    if not seed_labels:
        return {
            "label_counts": {},
            "total_seeds": 0,
            "balance_ratio": 0.0,
            "is_balanced": False,
            "recommendations": ["Add seed nodes - current set is empty"]
        }
    
    # Count seeds per label
    label_counts = {label: 0 for label in labels}
    for label in seed_labels.values():
        if label in label_counts:
            label_counts[label] += 1
    
    total_seeds = len(seed_labels)
    
    # Only consider labels that have seeds for balance ratio calculation
    non_zero_counts = [count for count in label_counts.values() if count > 0]
    max_count = max(non_zero_counts) if non_zero_counts else 0
    min_count = min(non_zero_counts) if non_zero_counts else 0
    
    # Calculate balance ratio
    balance_ratio = min_count / max_count if max_count > 0 else 0.0
    
    # Determine if balanced (ratio > 0.5 is reasonably balanced)
    is_balanced = balance_ratio >= 0.5
    
    # Generate recommendations
    recommendations = []
    if balance_ratio < 0.3:
        recommendations.append("Consider using create_balanced_seed_set() to balance classes")
    
    # Check for missing labels (labels with 0 counts)
    missing_labels = [label for label, count in label_counts.items() if count == 0]
    if missing_labels:
        recommendations.append(f"Add seeds for missing labels: {missing_labels}")
    
    if total_seeds < len(labels) * 2:
        recommendations.append("Consider adding more seeds per label for better propagation")
    
    if not recommendations:
        recommendations.append("Seed set appears well-balanced")
    
    return {
        "label_counts": label_counts,
        "total_seeds": total_seeds,
        "balance_ratio": balance_ratio,
        "is_balanced": is_balanced,
        "recommendations": recommendations
    }