"""
Utility functions for Guided Label Propagation.

This module provides helper functions for preprocessing data, parameter tuning,
and other utilities to support the GLP workflow. It includes functions for
creating balanced seed sets and suggesting optimal alpha values based on
network properties.
"""

from typing import Any, Dict, List, Optional, Tuple
import warnings
import random
import numpy as np
import polars as pl
import networkit as nk

try:
    import pandas as _pd  # type: ignore[import-not-found]
    _HAS_PANDAS = True
except ImportError:  # pragma: no cover
    _pd = None  # type: ignore[assignment]
    _HAS_PANDAS = False

from guidedLP.common.id_mapper import IDMapper
from guidedLP.common.exceptions import (
    ValidationError,
    ConfigurationError,
    ComputationError
)
from guidedLP.common.logging_config import get_logger, LoggingTimer
from guidedLP.common.seed_input import SeedInput, normalize_seed_input

logger = get_logger(__name__)


def create_balanced_seed_set(
    candidate_seeds: SeedInput,
    labels: List[str],
    n_per_label: Optional[int] = None,
    method: str = "undersample",
    random_seed: Optional[int] = None,
    seed_node_col: str = "node_id",
    seed_label_col: str = "label",
) -> Dict[Any, str]:
    """
    Create balanced seed set from potentially imbalanced candidate data.

    This function addresses class imbalance in seed node selection by creating
    a balanced set where each label has the same number of seed nodes. This
    prevents bias toward majority classes during label propagation.

    Parameters
    ----------
    candidate_seeds : SeedInput
        All available labeled nodes in any of four supported shapes (see
        :func:`guidedLP.common.normalize_seed_input`).
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
    seed_node_col : str, default "node_id"
        Column name for node IDs when ``candidate_seeds`` is a DataFrame.
    seed_label_col : str, default "label"
        Column name for labels when ``candidate_seeds`` is a DataFrame.
        
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
    candidate_seeds = normalize_seed_input(
        candidate_seeds, seed_node_col, seed_label_col
    )

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


def get_seed_statistics(
    seed_labels: SeedInput,
    labels: List[str],
    seed_node_col: str = "node_id",
    seed_label_col: str = "label",
) -> Dict[str, Any]:
    """
    Analyze seed set statistics and balance.

    This utility function provides diagnostic information about a seed set,
    including distribution balance, coverage, and recommendations.

    Parameters
    ----------
    seed_labels : SeedInput
        Seed nodes and their labels in any of four supported shapes (see
        :func:`guidedLP.common.normalize_seed_input`).
    labels : List[str]
        All possible labels
    seed_node_col : str, default "node_id"
        Column name for node IDs when ``seed_labels`` is a DataFrame.
    seed_label_col : str, default "label"
        Column name for labels when ``seed_labels`` is a DataFrame.

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
    seed_labels = normalize_seed_input(seed_labels, seed_node_col, seed_label_col)

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


def check_seed_coverage(
    id_mapper: IDMapper,
    seeds: SeedInput,
    test_seeds: Optional[SeedInput] = None,
    seed_node_col: str = "node_id",
    seed_label_col: str = "label",
    missing_sample_size: int = 10,
) -> Dict[str, Any]:
    """
    Diagnose how many seed nodes are actually present in a graph's IDMapper.

    Useful after preprocessing steps that can drop nodes — backboning, bipartite
    projection, giant-component filtering — to verify the surviving seed set is
    still adequate before running propagation. Returns a structured report
    rather than printing; callers can pretty-print as they see fit.

    Parameters
    ----------
    id_mapper : IDMapper
        Mapper for the graph you intend to propagate over.
    seeds : SeedInput
        Training seeds in any of the four supported shapes (see
        :func:`guidedLP.common.normalize_seed_input`).
    test_seeds : Optional[SeedInput], default None
        If provided, the returned report also covers the test set and the
        train/test overlap.
    seed_node_col : str, default "node_id"
        Column name for node IDs when an input is a DataFrame.
    seed_label_col : str, default "label"
        Column name for labels when an input is a DataFrame.
    missing_sample_size : int, default 10
        Cap on how many missing node IDs to include in each ``missing_sample``
        list. Set to 0 to disable sampling.

    Notes
    -----
    Entries with **null labels** (``None`` in dict values, ``null`` in polars,
    ``NaN`` / ``None`` in pandas) are silently dropped from the input before
    counting. The number of dropped rows per side is reported in the
    ``skipped_null_labels`` field. This makes the function safe to call on a
    DataFrame produced by a left-join with a sparse labels table — unlabeled
    rows just don't count toward train/test totals.

    Null *node IDs*, by contrast, are still treated as a data error by
    :func:`normalize_seed_input` (which raises).

    Returns
    -------
    Dict[str, Any]
        Always contains a ``"train"`` subdict. When ``test_seeds`` is supplied,
        also contains ``"test"`` and ``"overlap"`` subdicts.

        ``train`` / ``test`` subdict::

            {
                "total":               int,      # seeds remaining after null-label drop
                "present":             int,      # how many are in the mapper
                "missing":             int,      # how many are not
                "coverage":            float,    # present / total, or 0.0 if total == 0
                "by_label": {
                    label: {
                        "total":    int,
                        "present":  int,
                        "missing":  int,
                        "coverage": float,
                    },
                    ...
                },
                "missing_sample":      List[Any],   # up to missing_sample_size IDs
                "skipped_null_labels": int,         # rows dropped due to null labels
            }

        ``overlap`` subdict::

            {
                "count":       int,                       # nodes in both train and test
                "conflicting": int,                       # of which have different labels
                "sample":      List[Tuple[node, train_label, test_label]],
            }

    Examples
    --------
    >>> # After backboning, check how many seeds survived
    >>> report = check_seed_coverage(backbone_mapper, seeds)
    >>> if report["train"]["coverage"] < 0.8:
    ...     print(f"Only {report['train']['present']}/{report['train']['total']} "
    ...           f"seeds survived. Missing sample: {report['train']['missing_sample']}")

    >>> # Verify both train and test before validation
    >>> report = check_seed_coverage(mapper, train, test_seeds=test)
    >>> if report["overlap"]["conflicting"] > 0:
    ...     print(f"Warning: {report['overlap']['conflicting']} nodes have "
    ...           f"different train/test labels")
    """
    seeds_filtered, train_skipped = _drop_null_label_seeds(
        seeds, seed_label_col
    )
    train_normalized = normalize_seed_input(
        seeds_filtered, seed_node_col, seed_label_col
    )
    train_report = _coverage_report(
        train_normalized, id_mapper, missing_sample_size
    )
    train_report["skipped_null_labels"] = train_skipped

    result: Dict[str, Any] = {"train": train_report}

    if test_seeds is None:
        return result

    test_filtered, test_skipped = _drop_null_label_seeds(
        test_seeds, seed_label_col
    )
    test_normalized = normalize_seed_input(
        test_filtered, seed_node_col, seed_label_col
    )
    test_report = _coverage_report(
        test_normalized, id_mapper, missing_sample_size
    )
    test_report["skipped_null_labels"] = test_skipped
    result["test"] = test_report

    # Overlap analysis between train and test (independent of mapper membership).
    overlap_nodes = set(train_normalized) & set(test_normalized)
    conflicts = [
        (node, train_normalized[node], test_normalized[node])
        for node in overlap_nodes
        if train_normalized[node] != test_normalized[node]
    ]
    overlap_sample_cap = missing_sample_size if missing_sample_size > 0 else 0
    overlap_sample = (
        conflicts[:overlap_sample_cap] if conflicts else list(overlap_nodes)[:overlap_sample_cap]
    )
    result["overlap"] = {
        "count": len(overlap_nodes),
        "conflicting": len(conflicts),
        "sample": overlap_sample,
    }

    return result


def _drop_null_label_seeds(
    seeds: Any, label_col: str
) -> Tuple[Any, int]:
    """Filter out seed entries with null labels.

    Operates on any :data:`SeedInput` shape. For the inverse dict format
    ``{label: [nodes]}`` a ``None`` *key* (i.e. a None label) drops the whole
    list of nodes attached to it. For all other shapes a None / null / NaN
    value in the label position drops the row.

    Returns
    -------
    filtered : same type as the input (or the input unchanged for unknown types)
    skipped : int
        Number of entries dropped. Always 0 when the function can't infer the
        structure — let :func:`normalize_seed_input` raise the real error
        downstream.
    """
    # Dict shapes
    if isinstance(seeds, dict):
        if not seeds:
            return seeds, 0
        first_value = next(iter(seeds.values()))
        if isinstance(first_value, (list, tuple, set)):
            # Inverse format {label: [nodes]} — drop whole groups where the
            # label key is None.
            filtered_dict = {k: v for k, v in seeds.items() if k is not None}
            skipped_keys = [k for k in seeds if k is None]
            n_dropped = sum(len(seeds[k]) for k in skipped_keys)
            return filtered_dict, n_dropped
        # Canonical {node: label} — drop entries where label is None.
        filtered_dict = {k: v for k, v in seeds.items() if v is not None}
        return filtered_dict, len(seeds) - len(filtered_dict)

    # Polars DataFrame
    if isinstance(seeds, pl.DataFrame):
        if label_col not in seeds.columns:
            # Let normalize_seed_input raise the helpful "missing column" error.
            return seeds, 0
        before = len(seeds)
        filtered = seeds.filter(pl.col(label_col).is_not_null())
        return filtered, before - len(filtered)

    # Pandas DataFrame (only if pandas is importable)
    if _HAS_PANDAS and isinstance(seeds, _pd.DataFrame):
        if label_col not in seeds.columns:
            return seeds, 0
        before = len(seeds)
        filtered = seeds.dropna(subset=[label_col])
        return filtered, before - len(filtered)

    # Unknown type — pass through; normalize_seed_input will raise.
    return seeds, 0


def _coverage_report(
    seed_dict: Dict[Any, str],
    id_mapper: IDMapper,
    missing_sample_size: int,
) -> Dict[str, Any]:
    """Build the per-side coverage subdict for a normalized seed dict."""
    total = len(seed_dict)
    present_nodes = [n for n in seed_dict if id_mapper.has_original(n)]
    missing_nodes = [n for n in seed_dict if not id_mapper.has_original(n)]
    present_count = len(present_nodes)
    missing_count = len(missing_nodes)
    coverage = present_count / total if total > 0 else 0.0

    # Per-label breakdown.
    by_label: Dict[str, Dict[str, Any]] = {}
    for node, label in seed_dict.items():
        bucket = by_label.setdefault(
            label,
            {"total": 0, "present": 0, "missing": 0, "coverage": 0.0},
        )
        bucket["total"] += 1
        if id_mapper.has_original(node):
            bucket["present"] += 1
        else:
            bucket["missing"] += 1
    for bucket in by_label.values():
        bucket["coverage"] = (
            bucket["present"] / bucket["total"] if bucket["total"] > 0 else 0.0
        )

    cap = missing_sample_size if missing_sample_size > 0 else 0
    return {
        "total": total,
        "present": present_count,
        "missing": missing_count,
        "coverage": coverage,
        "by_label": by_label,
        "missing_sample": missing_nodes[:cap],
    }