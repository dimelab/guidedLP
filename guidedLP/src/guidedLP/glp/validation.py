"""
Validation functionality for Guided Label Propagation.

This module provides validation tools for assessing GLP performance using
train/test splits, external validation sets, and cross-validation approaches.
It implements comprehensive evaluation metrics and diagnostic tools.
"""

from typing import Dict, List, Any, Optional, Tuple
import warnings
import numpy as np
import polars as pl
import networkit as nk
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold
from sklearn.metrics import (
    accuracy_score, 
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report
)
import concurrent.futures

from .propagation import guided_label_propagation
from guidedLP.common.id_mapper import IDMapper
from guidedLP.common.exceptions import (
    ValidationError,
    ConfigurationError,
    ComputationError
)
from guidedLP.common.logging_config import get_logger, LoggingTimer

logger = get_logger(__name__)


def train_test_split_validation(
    graph: nk.Graph,
    id_mapper: IDMapper,
    seed_labels: Dict[Any, str],
    labels: List[str],
    test_size: float = 0.2,
    stratify: bool = True,
    random_seed: Optional[int] = None,
    **glp_kwargs
) -> Dict[str, Any]:
    """
    Validate GLP performance using train/test split of seed nodes.
    
    This function splits the seed nodes into training and test sets, runs
    guided label propagation on the training seeds only, and evaluates
    the predictions on the held-out test seeds. This provides an assessment
    of how well GLP can generalize beyond the training data.
    
    Mathematical Evaluation:
    - Accuracy: (TP + TN) / (TP + TN + FP + FN)
    - Precision: TP / (TP + FP) per label
    - Recall: TP / (TP + FN) per label  
    - F1-Score: 2 * (Precision * Recall) / (Precision + Recall) per label
    - Confusion Matrix: Actual vs Predicted label counts
    
    Parameters
    ----------
    graph : nk.Graph
        NetworkIt graph for propagation
    id_mapper : IDMapper
        Mapping between original and internal node IDs
    seed_labels : Dict[Any, str]
        Complete seed node labels (will be split for validation)
    labels : List[str]
        All possible labels
    test_size : float, default 0.2
        Fraction of seeds to hold out for testing (0.0-1.0)
    stratify : bool, default True
        Whether to maintain label proportions in train/test splits
    random_seed : Optional[int], default None
        Random seed for reproducible splits
    **glp_kwargs
        Additional arguments passed to guided_label_propagation()
        
    Returns
    -------
    Dict[str, Any]
        Comprehensive validation results containing:
        - accuracy: Overall accuracy score (float)
        - precision: Per-label precision scores (Dict[str, float])
        - recall: Per-label recall scores (Dict[str, float]) 
        - f1_score: Per-label F1 scores (Dict[str, float])
        - macro_precision: Macro-averaged precision (float)
        - macro_recall: Macro-averaged recall (float)
        - macro_f1: Macro-averaged F1 score (float)
        - confusion_matrix: Confusion matrix (np.ndarray)
        - test_predictions: Predictions on test set (pl.DataFrame)
        - train_size: Number of training seeds (int)
        - test_size: Number of test seeds (int)
        - convergence_iterations: Iterations until convergence (int)
        - classification_report: Detailed sklearn report (str)
        
    Raises
    ------
    ValidationError
        If seed_labels is empty, test_size invalid, or insufficient seeds per label
    ConfigurationError
        If test_size not in [0,1] or other parameter validation fails
    ComputationError
        If propagation fails during validation
        
    Examples
    --------
    >>> # Basic validation with 20% test split
    >>> results = train_test_split_validation(
    ...     graph, mapper, seeds, ["A", "B"], test_size=0.2
    ... )
    >>> print(f"Accuracy: {results['accuracy']:.3f}")
    >>> print(f"F1 scores: {results['f1_score']}")
    
    >>> # Stratified split with custom GLP parameters
    >>> results = train_test_split_validation(
    ...     graph, mapper, seeds, ["A", "B"], 
    ...     stratify=True, alpha=0.9, max_iterations=200
    ... )
    
    >>> # Reproducible validation
    >>> results = train_test_split_validation(
    ...     graph, mapper, seeds, ["A", "B"],
    ...     random_seed=42, test_size=0.3
    ... )
    
    Notes
    -----
    - Requires sufficient seeds per label for meaningful splits
    - Stratified splitting maintains label balance but requires multiple seeds per label
    - Test predictions use dominant_label from GLP output
    - Metrics calculated using sklearn.metrics for consistency
    - For very small seed sets, consider using cross_validate() instead
    """
    
    logger.info(f"Starting train/test split validation with test_size={test_size}, "
               f"stratify={stratify}, total_seeds={len(seed_labels)}")
    
    # Validate inputs
    _validate_split_inputs(seed_labels, labels, test_size, stratify)
    
    with LoggingTimer("Train/test split validation"):
        
        # Step 1: Split seeds into train/test
        train_seeds, test_seeds, train_labels_list, test_labels_list = _split_seed_data(
            seed_labels, labels, test_size, stratify, random_seed
        )
        
        logger.info(f"Split: {len(train_seeds)} training seeds, {len(test_seeds)} test seeds")
        
        # Step 2: Run GLP on training seeds only
        try:
            predictions = guided_label_propagation(
                graph, id_mapper, train_seeds, labels, **glp_kwargs
            )
        except Exception as e:
            raise ComputationError(
                "Failed to run guided label propagation during validation",
                operation="train_test_split_validation",
                error_type="propagation_failure",
                cause=e
            )
        
        # Step 3: Extract predictions for test nodes
        test_predictions = _extract_test_predictions(predictions, test_seeds)
        
        # Step 4: Calculate comprehensive metrics
        metrics = _calculate_validation_metrics(
            test_labels_list, 
            test_predictions["dominant_label"].to_list(),
            labels
        )
        
        # Step 5: Build comprehensive results
        results = {
            "accuracy": metrics["accuracy"],
            "precision": metrics["precision"],
            "recall": metrics["recall"], 
            "f1_score": metrics["f1_score"],
            "macro_precision": metrics["macro_precision"],
            "macro_recall": metrics["macro_recall"],
            "macro_f1": metrics["macro_f1"],
            "confusion_matrix": metrics["confusion_matrix"],
            "test_predictions": test_predictions,
            "train_size": len(train_seeds),
            "test_size": len(test_seeds),
            "classification_report": metrics["classification_report"],
            # Note: convergence_iterations not directly available from current GLP API
            # Could be added in future if GLP returns convergence info
        }
        
        logger.info(f"Validation complete. Accuracy: {results['accuracy']:.3f}, "
                   f"Macro F1: {results['macro_f1']:.3f}")
        
        return results


def _validate_split_inputs(
    seed_labels: Dict[Any, str], 
    labels: List[str], 
    test_size: float, 
    stratify: bool
) -> None:
    """Validate inputs for train/test split validation."""
    
    if not seed_labels:
        raise ValidationError("seed_labels cannot be empty for validation")
    
    if not 0.0 < test_size < 1.0:
        raise ConfigurationError(
            f"test_size must be between 0 and 1, got {test_size}",
            parameter="test_size",
            value=test_size
        )
    
    if len(seed_labels) < 2:
        raise ValidationError(
            f"Need at least 2 seeds for train/test split, got {len(seed_labels)}"
        )
    
    # Check label distribution for stratification
    if stratify:
        label_counts = {}
        for label in seed_labels.values():
            label_counts[label] = label_counts.get(label, 0) + 1
        
        min_count = min(label_counts.values())
        if min_count < 2:
            warnings.warn(
                f"Stratified split may fail: some labels have < 2 seeds. "
                f"Label counts: {label_counts}. Consider setting stratify=False."
            )


def _split_seed_data(
    seed_labels: Dict[Any, str],
    labels: List[str], 
    test_size: float,
    stratify: bool,
    random_seed: Optional[int]
) -> Tuple[Dict[Any, str], Dict[Any, str], List[str], List[str]]:
    """Split seed data into train/test sets using sklearn."""
    
    # Convert to parallel lists for sklearn
    seed_ids = list(seed_labels.keys())
    seed_labels_list = list(seed_labels.values())
    
    try:
        # Perform train/test split
        if stratify:
            train_ids, test_ids, train_labels_list, test_labels_list = train_test_split(
                seed_ids, 
                seed_labels_list,
                test_size=test_size,
                stratify=seed_labels_list,
                random_state=random_seed
            )
        else:
            train_ids, test_ids, train_labels_list, test_labels_list = train_test_split(
                seed_ids,
                seed_labels_list, 
                test_size=test_size,
                random_state=random_seed
            )
            
    except ValueError as e:
        # Handle stratification errors
        if "least populated class" in str(e):
            raise ValidationError(
                f"Cannot perform stratified split: {e}. "
                f"Try setting stratify=False or providing more seeds per label."
            )
        else:
            raise ValidationError(f"Train/test split failed: {e}")
    
    # Convert back to dictionaries
    train_seeds = dict(zip(train_ids, train_labels_list))
    test_seeds = dict(zip(test_ids, test_labels_list))
    
    logger.debug(f"Train labels distribution: {_count_labels(train_labels_list)}")
    logger.debug(f"Test labels distribution: {_count_labels(test_labels_list)}")
    
    return train_seeds, test_seeds, train_labels_list, test_labels_list


def _extract_test_predictions(
    predictions: pl.DataFrame, 
    test_seeds: Dict[Any, str]
) -> pl.DataFrame:
    """Extract predictions for test seed nodes."""
    
    test_node_ids = list(test_seeds.keys())
    
    # Filter predictions to test nodes only
    test_predictions = predictions.filter(
        pl.col("node_id").is_in(test_node_ids)
    )
    
    if len(test_predictions) != len(test_seeds):
        missing_nodes = set(test_node_ids) - set(test_predictions["node_id"].to_list())
        raise ComputationError(
            f"Missing predictions for {len(missing_nodes)} test nodes",
            operation="extract_test_predictions",
            details={"missing_count": len(missing_nodes), "missing_sample": list(missing_nodes)[:5]}
        )
    
    return test_predictions


def _calculate_validation_metrics(
    true_labels: List[str],
    predicted_labels: List[str], 
    all_labels: List[str]
) -> Dict[str, Any]:
    """Calculate comprehensive validation metrics using sklearn."""
    
    # Overall accuracy
    accuracy = accuracy_score(true_labels, predicted_labels)
    
    # Per-label metrics
    precision_scores, recall_scores, f1_scores, support = precision_recall_fscore_support(
        true_labels, predicted_labels, labels=all_labels, average=None, zero_division=0
    )
    
    # Macro averages
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        true_labels, predicted_labels, average='macro', zero_division=0
    )
    
    # Confusion matrix
    conf_matrix = confusion_matrix(true_labels, predicted_labels, labels=all_labels)
    
    # Classification report for detailed diagnostics
    class_report = classification_report(true_labels, predicted_labels, labels=all_labels)
    
    # Convert to dictionaries for easy access
    precision_dict = dict(zip(all_labels, precision_scores))
    recall_dict = dict(zip(all_labels, recall_scores))
    f1_dict = dict(zip(all_labels, f1_scores))
    
    return {
        "accuracy": float(accuracy),
        "precision": precision_dict,
        "recall": recall_dict,
        "f1_score": f1_dict,
        "macro_precision": float(macro_precision),
        "macro_recall": float(macro_recall), 
        "macro_f1": float(macro_f1),
        "confusion_matrix": conf_matrix,
        "classification_report": class_report
    }


def _count_labels(labels: List[str]) -> Dict[str, int]:
    """Count occurrences of each label."""
    counts = {}
    for label in labels:
        counts[label] = counts.get(label, 0) + 1
    return counts


def external_validation(
    predictions: pl.DataFrame,
    validation_labels: Dict[Any, str],
    labels: List[str]
) -> Dict[str, Any]:
    """
    Validate GLP results against independent labeled dataset.
    
    This function compares GLP predictions against an external validation
    set (e.g., expert-coded data, ground truth from external source) to
    assess generalization beyond the seed set.
    
    Parameters
    ----------
    predictions : pl.DataFrame
        Output from guided_label_propagation()
    validation_labels : Dict[Any, str]
        Independent labeled nodes (node_id -> label)
    labels : List[str]
        All possible labels
        
    Returns
    -------
    Dict[str, Any]
        Same metrics dictionary as train_test_split_validation()
        
    Examples
    --------
    >>> # Validate against expert-coded sample
    >>> expert_labels = {"node_1": "A", "node_5": "B", ...}
    >>> results = external_validation(glp_predictions, expert_labels, ["A", "B"])
    >>> print(f"External validation accuracy: {results['accuracy']:.3f}")
    
    Notes
    -----
    - Validation nodes must be present in predictions DataFrame
    - Uses dominant_label from predictions for comparison
    - Useful for assessing real-world performance
    """
    
    logger.info(f"Starting external validation with {len(validation_labels)} validation nodes")
    
    if not validation_labels:
        raise ValidationError("validation_labels cannot be empty")
    
    with LoggingTimer("External validation"):
        
        # Extract predictions for validation nodes
        validation_predictions = _extract_test_predictions(predictions, validation_labels)
        
        # Get true and predicted labels
        true_labels = [validation_labels[node_id] for node_id in validation_predictions["node_id"]]
        predicted_labels = validation_predictions["dominant_label"].to_list()
        
        # Calculate metrics
        metrics = _calculate_validation_metrics(true_labels, predicted_labels, labels)
        
        # Build results (similar structure to train_test_split_validation)
        results = {
            "accuracy": metrics["accuracy"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1_score": metrics["f1_score"],
            "macro_precision": metrics["macro_precision"],
            "macro_recall": metrics["macro_recall"],
            "macro_f1": metrics["macro_f1"],
            "confusion_matrix": metrics["confusion_matrix"],
            "test_predictions": validation_predictions,
            "validation_size": len(validation_labels),
            "classification_report": metrics["classification_report"]
        }
        
        logger.info(f"External validation complete. Accuracy: {results['accuracy']:.3f}")
        
        return results


def get_validation_summary(validation_results: Dict[str, Any]) -> str:
    """
    Generate a human-readable summary of validation results.
    
    Parameters
    ----------
    validation_results : Dict[str, Any]
        Results from train_test_split_validation() or external_validation()
        
    Returns
    -------
    str
        Formatted summary string
        
    Examples
    --------
    >>> results = train_test_split_validation(graph, mapper, seeds, labels)
    >>> summary = get_validation_summary(results)
    >>> print(summary)
    """
    
    summary_lines = [
        "=== GLP Validation Summary ===",
        f"Overall Accuracy: {validation_results['accuracy']:.3f}",
        f"Macro F1-Score: {validation_results['macro_f1']:.3f}",
        f"Test Size: {validation_results.get('test_size', validation_results.get('validation_size', 'N/A'))}",
        "",
        "Per-Label Performance:"
    ]
    
    # Add per-label metrics
    for label in validation_results['precision'].keys():
        precision = validation_results['precision'][label]
        recall = validation_results['recall'][label] 
        f1 = validation_results['f1_score'][label]
        summary_lines.append(f"  {label}: P={precision:.3f}, R={recall:.3f}, F1={f1:.3f}")
    
    summary_lines.extend([
        "",
        "Confusion Matrix:",
        str(validation_results['confusion_matrix'])
    ])
    
    return "\n".join(summary_lines)


def cross_validate(
    graph: nk.Graph,
    id_mapper: IDMapper,
    seed_labels: Dict[Any, str],
    labels: List[str],
    k_folds: int = 5,
    stratify: bool = True,
    random_seed: Optional[int] = None,
    n_jobs: int = 1,
    **glp_kwargs
) -> Dict[str, Any]:
    """
    Perform K-fold cross-validation for robust GLP performance estimation.
    
    This function splits the seed nodes into k folds and performs cross-validation
    by using each fold as a test set while training on the remaining folds.
    This provides a more robust estimate of GLP performance than a single
    train/test split.
    
    Mathematical Evaluation:
    - For each fold k: Train on (k-1)/k seeds, test on 1/k seeds
    - Aggregate metrics: Mean ± Standard Deviation across folds
    - Confusion matrices: Element-wise sum across folds
    - Per-label metrics: Averaged across folds
    
    Parameters
    ----------
    graph : nk.Graph
        NetworkIt graph for propagation
    id_mapper : IDMapper
        Mapping between original and internal node IDs
    seed_labels : Dict[Any, str]
        Complete seed node labels (will be split into k folds)
    labels : List[str]
        All possible labels
    k_folds : int, default 5
        Number of cross-validation folds
    stratify : bool, default True
        Whether to maintain label proportions across folds
    random_seed : Optional[int], default None
        Random seed for reproducible fold generation
    n_jobs : int, default 1
        Number of parallel jobs for fold processing (1 = sequential)
    **glp_kwargs
        Additional arguments passed to guided_label_propagation()
        
    Returns
    -------
    Dict[str, Any]
        Cross-validation results containing:
        - mean_accuracy: Mean accuracy across folds (float)
        - std_accuracy: Standard deviation of accuracy (float)
        - fold_accuracies: Accuracy for each fold (List[float])
        - mean_precision: Mean precision per label (Dict[str, float])
        - mean_recall: Mean recall per label (Dict[str, float])
        - mean_f1: Mean F1 score per label (Dict[str, float])
        - std_precision: Std precision per label (Dict[str, float])
        - std_recall: Std recall per label (Dict[str, float])
        - std_f1: Std F1 score per label (Dict[str, float])
        - fold_results: Full results for each fold (List[Dict])
        - aggregate_confusion_matrix: Sum of all confusion matrices (np.ndarray)
        - mean_macro_f1: Mean macro F1 across folds (float)
        - std_macro_f1: Standard deviation of macro F1 (float)
        
    Raises
    ------
    ValidationError
        If insufficient seeds for k-fold splitting or invalid k_folds
    ConfigurationError
        If k_folds < 2 or other parameter validation fails
    ComputationError
        If propagation fails during any fold
        
    Examples
    --------
    >>> # Basic 5-fold cross-validation
    >>> results = cross_validate(
    ...     graph, mapper, seeds, ["A", "B"], k_folds=5
    ... )
    >>> print(f"Mean accuracy: {results['mean_accuracy']:.3f} ± {results['std_accuracy']:.3f}")
    
    >>> # 10-fold with custom GLP parameters
    >>> results = cross_validate(
    ...     graph, mapper, seeds, ["A", "B"], 
    ...     k_folds=10, alpha=0.9, max_iterations=200
    ... )
    
    >>> # Reproducible cross-validation
    >>> results = cross_validate(
    ...     graph, mapper, seeds, ["A", "B"],
    ...     k_folds=5, random_seed=42
    ... )
    
    Notes
    -----
    - Requires sufficient seeds per label for meaningful k-fold splits
    - Stratified folding maintains label balance but requires multiple seeds per label
    - Each fold uses (k-1)/k seeds for training, 1/k seeds for testing
    - Results provide confidence intervals for performance estimates
    - Parallel processing (n_jobs > 1) can speed up computation for large k_folds
    """
    
    logger.info(f"Starting {k_folds}-fold cross-validation with {len(seed_labels)} seeds, "
               f"stratify={stratify}, n_jobs={n_jobs}")
    
    # Validate inputs
    _validate_cv_inputs(seed_labels, labels, k_folds, stratify)
    
    with LoggingTimer(f"{k_folds}-fold cross-validation"):
        
        # Generate fold splits
        fold_splits = _generate_cv_folds(
            seed_labels, labels, k_folds, stratify, random_seed
        )
        
        logger.info(f"Generated {len(fold_splits)} folds")
        
        # Process folds (sequentially or in parallel)
        if n_jobs == 1:
            fold_results = _process_folds_sequential(
                fold_splits, graph, id_mapper, labels, glp_kwargs
            )
        else:
            fold_results = _process_folds_parallel(
                fold_splits, graph, id_mapper, labels, glp_kwargs, n_jobs
            )
        
        # Aggregate results across folds
        aggregated_results = _aggregate_cv_results(fold_results, labels, k_folds)
        
        logger.info(f"Cross-validation complete. Mean accuracy: {aggregated_results['mean_accuracy']:.3f} "
                   f"± {aggregated_results['std_accuracy']:.3f}")
        
        return aggregated_results


def _validate_cv_inputs(
    seed_labels: Dict[Any, str], 
    labels: List[str], 
    k_folds: int, 
    stratify: bool
) -> None:
    """Validate inputs for cross-validation."""
    
    if not seed_labels:
        raise ValidationError("seed_labels cannot be empty for cross-validation")
    
    if k_folds < 2:
        raise ConfigurationError(
            f"k_folds must be at least 2, got {k_folds}",
            parameter="k_folds",
            value=k_folds
        )
    
    if len(seed_labels) < k_folds:
        raise ValidationError(
            f"Need at least {k_folds} seeds for {k_folds}-fold CV, got {len(seed_labels)}"
        )
    
    # Check label distribution for stratification
    if stratify:
        label_counts = _count_labels(list(seed_labels.values()))
        min_count = min(label_counts.values())
        
        if min_count < k_folds:
            warnings.warn(
                f"Stratified {k_folds}-fold CV may fail: some labels have < {k_folds} seeds. "
                f"Label counts: {label_counts}. Consider setting stratify=False or reducing k_folds."
            )


def _generate_cv_folds(
    seed_labels: Dict[Any, str],
    labels: List[str],
    k_folds: int,
    stratify: bool,
    random_seed: Optional[int]
) -> List[Tuple[Dict[Any, str], Dict[Any, str]]]:
    """Generate k-fold splits for cross-validation."""
    
    # Convert to parallel lists for sklearn
    seed_ids = list(seed_labels.keys())
    seed_labels_list = list(seed_labels.values())
    
    try:
        # Create fold splitter
        if stratify:
            kfold = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=random_seed)
            splits = kfold.split(seed_ids, seed_labels_list)
        else:
            kfold = KFold(n_splits=k_folds, shuffle=True, random_state=random_seed)
            splits = kfold.split(seed_ids)
            
    except ValueError as e:
        # Handle stratification errors
        if "least populated class" in str(e):
            raise ValidationError(
                f"Cannot perform stratified {k_folds}-fold CV: {e}. "
                f"Try setting stratify=False or reducing k_folds."
            )
        else:
            raise ValidationError(f"K-fold split failed: {e}")
    
    # Convert splits to train/test dictionaries
    fold_splits = []
    for fold_idx, (train_indices, test_indices) in enumerate(splits):
        train_ids = [seed_ids[i] for i in train_indices]
        test_ids = [seed_ids[i] for i in test_indices]
        train_labels_list = [seed_labels_list[i] for i in train_indices]
        test_labels_list = [seed_labels_list[i] for i in test_indices]
        
        train_seeds = dict(zip(train_ids, train_labels_list))
        test_seeds = dict(zip(test_ids, test_labels_list))
        
        fold_splits.append((train_seeds, test_seeds))
        
        logger.debug(f"Fold {fold_idx}: {len(train_seeds)} train, {len(test_seeds)} test")
    
    return fold_splits


def _process_folds_sequential(
    fold_splits: List[Tuple[Dict[Any, str], Dict[Any, str]]],
    graph: nk.Graph,
    id_mapper: IDMapper,
    labels: List[str],
    glp_kwargs: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """Process folds sequentially."""
    
    fold_results = []
    
    for fold_idx, (train_seeds, test_seeds) in enumerate(fold_splits):
        logger.debug(f"Processing fold {fold_idx + 1}/{len(fold_splits)}")
        
        try:
            fold_result = _process_single_fold(
                train_seeds, test_seeds, graph, id_mapper, labels, glp_kwargs
            )
            fold_result["fold_index"] = fold_idx
            fold_results.append(fold_result)
            
        except Exception as e:
            raise ComputationError(
                f"Failed to process fold {fold_idx}",
                operation="cross_validate",
                error_type="fold_processing_failure",
                details={"fold_index": fold_idx},
                cause=e
            )
    
    return fold_results


def _process_folds_parallel(
    fold_splits: List[Tuple[Dict[Any, str], Dict[Any, str]]],
    graph: nk.Graph,
    id_mapper: IDMapper,
    labels: List[str],
    glp_kwargs: Dict[str, Any],
    n_jobs: int
) -> List[Dict[str, Any]]:
    """Process folds in parallel."""
    
    logger.info(f"Processing {len(fold_splits)} folds with {n_jobs} parallel jobs")
    
    def process_fold_wrapper(fold_data):
        fold_idx, (train_seeds, test_seeds) = fold_data
        try:
            result = _process_single_fold(
                train_seeds, test_seeds, graph, id_mapper, labels, glp_kwargs
            )
            result["fold_index"] = fold_idx
            return result
        except Exception as e:
            raise ComputationError(
                f"Failed to process fold {fold_idx}",
                operation="cross_validate_parallel",
                error_type="fold_processing_failure",
                details={"fold_index": fold_idx},
                cause=e
            )
    
    # Prepare data for parallel processing
    fold_data = list(enumerate(fold_splits))
    
    # Process in parallel
    max_workers = min(n_jobs, len(fold_splits)) if n_jobs > 0 else len(fold_splits)
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        fold_results = list(executor.map(process_fold_wrapper, fold_data))
    
    # Sort by fold index to maintain order
    fold_results.sort(key=lambda x: x["fold_index"])
    
    return fold_results


def _process_single_fold(
    train_seeds: Dict[Any, str],
    test_seeds: Dict[Any, str],
    graph: nk.Graph,
    id_mapper: IDMapper,
    labels: List[str],
    glp_kwargs: Dict[str, Any]
) -> Dict[str, Any]:
    """Process a single fold of cross-validation."""
    
    # Run GLP on training seeds
    predictions = guided_label_propagation(
        graph, id_mapper, train_seeds, labels, **glp_kwargs
    )
    
    # Extract predictions for test nodes
    test_predictions = _extract_test_predictions(predictions, test_seeds)
    
    # Calculate metrics for this fold
    true_labels = [test_seeds[node_id] for node_id in test_predictions["node_id"]]
    predicted_labels = test_predictions["dominant_label"].to_list()
    
    metrics = _calculate_validation_metrics(true_labels, predicted_labels, labels)
    
    return {
        "accuracy": metrics["accuracy"],
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "f1_score": metrics["f1_score"],
        "macro_precision": metrics["macro_precision"],
        "macro_recall": metrics["macro_recall"],
        "macro_f1": metrics["macro_f1"],
        "confusion_matrix": metrics["confusion_matrix"],
        "train_size": len(train_seeds),
        "test_size": len(test_seeds)
    }


def _aggregate_cv_results(
    fold_results: List[Dict[str, Any]],
    labels: List[str],
    k_folds: int
) -> Dict[str, Any]:
    """Aggregate results across all folds."""
    
    # Extract metrics from all folds
    accuracies = [result["accuracy"] for result in fold_results]
    macro_f1s = [result["macro_f1"] for result in fold_results]
    
    # Per-label metrics
    precision_by_label = {label: [] for label in labels}
    recall_by_label = {label: [] for label in labels}
    f1_by_label = {label: [] for label in labels}
    
    confusion_matrices = []
    
    for result in fold_results:
        for label in labels:
            precision_by_label[label].append(result["precision"][label])
            recall_by_label[label].append(result["recall"][label])
            f1_by_label[label].append(result["f1_score"][label])
        
        confusion_matrices.append(result["confusion_matrix"])
    
    # Calculate means and standard deviations
    mean_accuracy = np.mean(accuracies)
    std_accuracy = np.std(accuracies, ddof=1) if len(accuracies) > 1 else 0.0
    
    mean_macro_f1 = np.mean(macro_f1s)
    std_macro_f1 = np.std(macro_f1s, ddof=1) if len(macro_f1s) > 1 else 0.0
    
    # Per-label aggregations
    mean_precision = {}
    std_precision = {}
    mean_recall = {}
    std_recall = {}
    mean_f1 = {}
    std_f1 = {}
    
    for label in labels:
        mean_precision[label] = np.mean(precision_by_label[label])
        std_precision[label] = np.std(precision_by_label[label], ddof=1) if len(precision_by_label[label]) > 1 else 0.0
        
        mean_recall[label] = np.mean(recall_by_label[label])
        std_recall[label] = np.std(recall_by_label[label], ddof=1) if len(recall_by_label[label]) > 1 else 0.0
        
        mean_f1[label] = np.mean(f1_by_label[label])
        std_f1[label] = np.std(f1_by_label[label], ddof=1) if len(f1_by_label[label]) > 1 else 0.0
    
    # Aggregate confusion matrices
    aggregate_confusion_matrix = np.sum(confusion_matrices, axis=0)
    
    return {
        "mean_accuracy": float(mean_accuracy),
        "std_accuracy": float(std_accuracy),
        "fold_accuracies": accuracies,
        "mean_precision": mean_precision,
        "std_precision": std_precision,
        "mean_recall": mean_recall,
        "std_recall": std_recall,
        "mean_f1": mean_f1,
        "std_f1": std_f1,
        "mean_macro_f1": float(mean_macro_f1),
        "std_macro_f1": float(std_macro_f1),
        "fold_results": fold_results,
        "aggregate_confusion_matrix": aggregate_confusion_matrix,
        "k_folds": k_folds
    }