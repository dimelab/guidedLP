"""
Evaluation functions for Guided Label Propagation results.

This module provides tools for analyzing and comparing GLP propagation results:
- Label distribution analysis across the network
- Confidence assessment and uncertainty identification  
- Directional propagation comparison (in-degree vs out-degree)
- Quality metrics for propagation outcomes
"""

from typing import Dict, List, Any, Tuple
import numpy as np
import polars as pl

from guidedLP.common.exceptions import ValidationError
from guidedLP.common.logging_config import get_logger, LoggingTimer

logger = get_logger(__name__)


def analyze_label_distribution(
    predictions: pl.DataFrame,
    labels: List[str]
) -> Dict[str, Any]:
    """
    Analyze distribution of label probabilities across network.
    
    This function provides comprehensive statistics about how labels are
    distributed in the network after propagation, including confidence
    measures and uncertainty identification.
    
    Parameters
    ----------
    predictions : pl.DataFrame
        Output from guided_label_propagation() containing:
        - node_id: Original node identifiers
        - {label}_prob: Probability columns for each label
        - dominant_label: Label with highest probability
        - confidence: Maximum probability value
        - is_seed: Boolean indicating seed status
    labels : List[str]
        List of all possible labels
        
    Returns
    -------
    Dict[str, Any]
        Comprehensive distribution analysis containing:
        - label_counts: Dict[str, int] - Number of nodes per dominant label
        - mean_confidence: float - Average confidence across all nodes
        - confidence_by_label: Dict[str, float] - Mean confidence per label
        - probability_distributions: Dict[str, np.ndarray] - Probability histograms
        - high_confidence_nodes: pl.DataFrame - Nodes with confidence > 0.8
        - uncertain_nodes: pl.DataFrame - Nodes with confidence < 0.5
        - seed_vs_nonseed_confidence: Dict[str, float] - Confidence by seed status
        - label_entropy: float - Overall entropy of label distribution
        
    Raises
    ------
    ValidationError
        If predictions DataFrame is invalid or missing required columns
        
    Examples
    --------
    >>> # Analyze propagation results
    >>> analysis = analyze_label_distribution(predictions, ["left", "right"])
    >>> print(f"Mean confidence: {analysis['mean_confidence']:.3f}")
    >>> print(f"High confidence nodes: {len(analysis['high_confidence_nodes'])}")
    
    >>> # Check label balance
    >>> counts = analysis['label_counts']
    >>> print(f"Label distribution: {counts}")
    
    >>> # Identify uncertain classifications
    >>> uncertain = analysis['uncertain_nodes']
    >>> print(f"Uncertain nodes: {len(uncertain)} ({len(uncertain)/len(predictions)*100:.1f}%)")
    
    Notes
    -----
    The analysis helps assess propagation quality:
    - High mean confidence suggests clear label boundaries
    - Many uncertain nodes may indicate complex network structure
    - Imbalanced label_counts could suggest propagation bias
    - High entropy indicates diverse, uncertain classifications
    """
    
    logger.info(f"Analyzing label distribution for {len(predictions)} nodes, {len(labels)} labels")
    
    # Validate inputs
    _validate_predictions_dataframe(predictions, labels)
    
    with LoggingTimer("Analyzing label distribution"):
        
        # Basic label counts
        label_counts = _calculate_label_counts(predictions)
        
        # Confidence statistics
        mean_confidence = float(predictions["confidence"].mean())
        confidence_by_label = _calculate_confidence_by_label(predictions, labels)
        seed_vs_nonseed_confidence = _calculate_seed_confidence_comparison(predictions)
        
        # Probability distributions (histograms)
        probability_distributions = _calculate_probability_distributions(predictions, labels)
        
        # High/low confidence node identification
        high_confidence_nodes = predictions.filter(pl.col("confidence") > 0.8)
        uncertain_nodes = predictions.filter(pl.col("confidence") < 0.5)
        
        # Label entropy calculation
        label_entropy = _calculate_label_entropy(predictions, labels)
        
        logger.info(f"Distribution analysis complete: {len(label_counts)} labels, "
                   f"mean confidence {mean_confidence:.3f}, "
                   f"{len(high_confidence_nodes)} high confidence nodes, "
                   f"{len(uncertain_nodes)} uncertain nodes")
        
        return {
            "label_counts": label_counts,
            "mean_confidence": mean_confidence,
            "confidence_by_label": confidence_by_label,
            "probability_distributions": probability_distributions,
            "high_confidence_nodes": high_confidence_nodes,
            "uncertain_nodes": uncertain_nodes,
            "seed_vs_nonseed_confidence": seed_vs_nonseed_confidence,
            "label_entropy": label_entropy
        }


def compare_directional_results(
    out_degree_predictions: pl.DataFrame,
    in_degree_predictions: pl.DataFrame,
    labels: List[str]
) -> Dict[str, Any]:
    """
    Compare in-degree vs. out-degree propagation results.
    
    This function analyzes differences between propagation based on incoming
    vs. outgoing edges, helping identify directional asymmetries and nodes
    with interesting influence patterns.
    
    Parameters
    ----------
    out_degree_predictions : pl.DataFrame
        Predictions from out-degree based propagation
    in_degree_predictions : pl.DataFrame
        Predictions from in-degree based propagation
    labels : List[str]
        List of all possible labels
        
    Returns
    -------
    Dict[str, Any]
        Directional comparison analysis containing:
        - agreement_rate: float - Fraction of nodes with same dominant label
        - correlation_by_label: Dict[str, float] - Probability correlation per label
        - divergent_nodes: pl.DataFrame - Nodes with different dominant labels
        - direction_bias: Dict[str, str] - Which direction favors each label
        - confidence_comparison: Dict[str, float] - Mean confidence by direction
        - probability_differences: pl.DataFrame - Per-node probability differences
        
    Raises
    ------
    ValidationError
        If DataFrames have different nodes or missing columns
        
    Examples
    --------
    >>> # Compare directional propagation results
    >>> comparison = compare_directional_results(out_preds, in_preds, ["A", "B"])
    >>> print(f"Agreement rate: {comparison['agreement_rate']:.3f}")
    
    >>> # Find nodes with directional differences
    >>> divergent = comparison['divergent_nodes']
    >>> print(f"Nodes with different classifications: {len(divergent)}")
    
    >>> # Check directional bias per label
    >>> bias = comparison['direction_bias']
    >>> for label, direction in bias.items():
    ...     print(f"Label '{label}' favored by {direction}-degree propagation")
    
    >>> # Analyze probability correlations
    >>> correlations = comparison['correlation_by_label']
    >>> for label, corr in correlations.items():
    ...     print(f"Label '{label}' correlation: {corr:.3f}")
    
    Notes
    -----
    Directional Analysis Insights:
    - Low agreement_rate suggests strong directional asymmetries
    - High correlation but different dominant labels indicates subtle differences
    - Divergent nodes often represent influential or bridge positions
    - Direction bias reveals which labels spread along different edge directions
    """
    
    logger.info(f"Comparing directional results for {len(out_degree_predictions)} nodes")
    
    # Validate inputs
    _validate_directional_inputs(out_degree_predictions, in_degree_predictions, labels)
    
    with LoggingTimer("Comparing directional propagation results"):
        
        # Merge DataFrames for comparison
        merged_df = _merge_directional_dataframes(out_degree_predictions, in_degree_predictions, labels)
        
        # Calculate agreement rate
        agreement_rate = _calculate_agreement_rate(merged_df)
        
        # Calculate probability correlations per label
        correlation_by_label = _calculate_label_correlations(merged_df, labels)
        
        # Identify divergent nodes
        divergent_nodes = _identify_divergent_nodes(merged_df, labels)
        
        # Determine direction bias per label
        direction_bias = _calculate_direction_bias(merged_df, labels)
        
        # Compare confidence between directions
        confidence_comparison = _compare_directional_confidence(merged_df)
        
        # Calculate probability differences
        probability_differences = _calculate_probability_differences(merged_df, labels)
        
        logger.info(f"Directional comparison complete: {agreement_rate:.3f} agreement rate, "
                   f"{len(divergent_nodes)} divergent nodes, "
                   f"direction biases: {direction_bias}")
        
        return {
            "agreement_rate": agreement_rate,
            "correlation_by_label": correlation_by_label,
            "divergent_nodes": divergent_nodes,
            "direction_bias": direction_bias,
            "confidence_comparison": confidence_comparison,
            "probability_differences": probability_differences
        }


def _validate_predictions_dataframe(predictions: pl.DataFrame, labels: List[str]) -> None:
    """Validate predictions DataFrame has required columns."""
    
    if predictions.is_empty():
        raise ValidationError("predictions DataFrame cannot be empty")
    
    required_columns = ["node_id", "dominant_label", "confidence", "is_seed"]
    missing_columns = [col for col in required_columns if col not in predictions.columns]
    if missing_columns:
        raise ValidationError(f"Missing required columns: {missing_columns}")
    
    # Check for probability columns
    prob_columns = [f"{label}_prob" for label in labels]
    missing_prob_columns = [col for col in prob_columns if col not in predictions.columns]
    if missing_prob_columns:
        raise ValidationError(f"Missing probability columns: {missing_prob_columns}")


def _validate_directional_inputs(
    out_df: pl.DataFrame, 
    in_df: pl.DataFrame, 
    labels: List[str]
) -> None:
    """Validate directional DataFrames for comparison."""
    
    # Validate both DataFrames individually
    _validate_predictions_dataframe(out_df, labels)
    _validate_predictions_dataframe(in_df, labels)
    
    # Check they have same nodes
    out_nodes = set(out_df["node_id"].to_list())
    in_nodes = set(in_df["node_id"].to_list())
    
    if out_nodes != in_nodes:
        raise ValidationError(
            f"DataFrames must have same nodes. "
            f"Out-degree: {len(out_nodes)}, In-degree: {len(in_nodes)}, "
            f"Difference: {len(out_nodes.symmetric_difference(in_nodes))}"
        )


def _calculate_label_counts(predictions: pl.DataFrame) -> Dict[str, int]:
    """Calculate count of nodes per dominant label."""
    
    counts = predictions.group_by("dominant_label").count()
    return {row["dominant_label"]: row["count"] for row in counts.iter_rows(named=True)}


def _calculate_confidence_by_label(predictions: pl.DataFrame, labels: List[str]) -> Dict[str, float]:
    """Calculate mean confidence for each label."""
    
    confidence_by_label = {}
    
    for label in labels:
        label_nodes = predictions.filter(pl.col("dominant_label") == label)
        if len(label_nodes) > 0:
            confidence_by_label[label] = float(label_nodes["confidence"].mean())
        else:
            confidence_by_label[label] = 0.0
    
    return confidence_by_label


def _calculate_seed_confidence_comparison(predictions: pl.DataFrame) -> Dict[str, float]:
    """Compare confidence between seed and non-seed nodes."""
    
    seed_confidence = float(predictions.filter(pl.col("is_seed") == True)["confidence"].mean())
    nonseed_confidence = float(predictions.filter(pl.col("is_seed") == False)["confidence"].mean())
    
    return {
        "seed_confidence": seed_confidence,
        "nonseed_confidence": nonseed_confidence,
        "confidence_difference": seed_confidence - nonseed_confidence
    }


def _calculate_probability_distributions(
    predictions: pl.DataFrame, 
    labels: List[str]
) -> Dict[str, np.ndarray]:
    """Calculate probability distribution histograms for each label."""
    
    probability_distributions = {}
    
    for label in labels:
        prob_column = f"{label}_prob"
        probabilities = predictions[prob_column].to_numpy()
        
        # Create histogram with 20 bins from 0 to 1
        hist, _ = np.histogram(probabilities, bins=20, range=(0, 1))
        probability_distributions[label] = hist
    
    return probability_distributions


def _calculate_label_entropy(predictions: pl.DataFrame, labels: List[str]) -> float:
    """Calculate entropy of label distribution."""
    
    # Get probability matrix (n_nodes x n_labels)
    prob_columns = [f"{label}_prob" for label in labels]
    prob_matrix = predictions.select(prob_columns).to_numpy()
    
    # Calculate entropy for each node, then average
    # Entropy = -sum(p * log(p)) for each node
    # Add small epsilon to avoid log(0)
    epsilon = 1e-10
    prob_matrix = np.clip(prob_matrix, epsilon, 1.0)
    
    node_entropies = -np.sum(prob_matrix * np.log(prob_matrix), axis=1)
    mean_entropy = np.mean(node_entropies)
    
    return float(mean_entropy)


def _merge_directional_dataframes(
    out_df: pl.DataFrame, 
    in_df: pl.DataFrame, 
    labels: List[str]
) -> pl.DataFrame:
    """Merge out-degree and in-degree DataFrames for comparison."""
    
    # Select relevant columns and add suffixes
    out_cols = ["node_id", "dominant_label", "confidence"] + [f"{label}_prob" for label in labels]
    in_cols = ["node_id", "dominant_label", "confidence"] + [f"{label}_prob" for label in labels]
    
    out_selected = out_df.select(out_cols).rename({
        "dominant_label": "out_dominant_label",
        "confidence": "out_confidence",
        **{f"{label}_prob": f"out_{label}_prob" for label in labels}
    })
    
    in_selected = in_df.select(in_cols).rename({
        "dominant_label": "in_dominant_label", 
        "confidence": "in_confidence",
        **{f"{label}_prob": f"in_{label}_prob" for label in labels}
    })
    
    # Merge on node_id
    merged = out_selected.join(in_selected, on="node_id", how="inner")
    
    return merged


def _calculate_agreement_rate(merged_df: pl.DataFrame) -> float:
    """Calculate fraction of nodes with same dominant label in both directions."""
    
    agreements = merged_df.filter(
        pl.col("out_dominant_label") == pl.col("in_dominant_label")
    )
    
    return len(agreements) / len(merged_df)


def _calculate_label_correlations(merged_df: pl.DataFrame, labels: List[str]) -> Dict[str, float]:
    """Calculate probability correlations for each label between directions."""
    
    correlations = {}
    
    for label in labels:
        out_col = f"out_{label}_prob"
        in_col = f"in_{label}_prob"
        
        # Calculate Pearson correlation
        out_probs = merged_df[out_col].to_numpy()
        in_probs = merged_df[in_col].to_numpy()
        
        correlation = np.corrcoef(out_probs, in_probs)[0, 1]
        
        # Handle NaN case (when one direction has constant probabilities)
        if np.isnan(correlation):
            correlation = 0.0
            
        correlations[label] = float(correlation)
    
    return correlations


def _identify_divergent_nodes(merged_df: pl.DataFrame, labels: List[str]) -> pl.DataFrame:
    """Identify nodes with different dominant labels between directions."""
    
    divergent = merged_df.filter(
        pl.col("out_dominant_label") != pl.col("in_dominant_label")
    )
    
    # Add probability difference columns for analysis
    prob_diff_exprs = []
    for label in labels:
        prob_diff_exprs.append(
            (pl.col(f"out_{label}_prob") - pl.col(f"in_{label}_prob")).alias(f"{label}_prob_diff")
        )
    
    if len(divergent) > 0:
        divergent = divergent.with_columns(prob_diff_exprs)
    
    return divergent


def _calculate_direction_bias(merged_df: pl.DataFrame, labels: List[str]) -> Dict[str, str]:
    """Determine which direction favors each label."""
    
    direction_bias = {}
    
    for label in labels:
        out_col = f"out_{label}_prob"
        in_col = f"in_{label}_prob"
        
        out_mean = merged_df[out_col].mean()
        in_mean = merged_df[in_col].mean()
        
        if out_mean > in_mean:
            direction_bias[label] = "out"
        elif in_mean > out_mean:
            direction_bias[label] = "in"
        else:
            direction_bias[label] = "equal"
    
    return direction_bias


def _compare_directional_confidence(merged_df: pl.DataFrame) -> Dict[str, float]:
    """Compare confidence levels between directions."""
    
    out_confidence = float(merged_df["out_confidence"].mean())
    in_confidence = float(merged_df["in_confidence"].mean())
    
    return {
        "out_degree_confidence": out_confidence,
        "in_degree_confidence": in_confidence,
        "confidence_difference": out_confidence - in_confidence
    }


def _calculate_probability_differences(merged_df: pl.DataFrame, labels: List[str]) -> pl.DataFrame:
    """Calculate per-node probability differences between directions."""
    
    # Add probability difference columns
    prob_diff_exprs = []
    for label in labels:
        prob_diff_exprs.append(
            (pl.col(f"out_{label}_prob") - pl.col(f"in_{label}_prob")).alias(f"{label}_prob_diff")
        )
    
    # First create a DataFrame with the differences
    result_with_diffs = merged_df.with_columns(prob_diff_exprs)
    
    # Add absolute maximum difference
    abs_diff_exprs = [pl.col(f"{label}_prob_diff").abs() for label in labels]
    max_abs_diff = pl.max_horizontal(abs_diff_exprs).alias("max_abs_diff")
    
    # Select node_id and differences
    result = result_with_diffs.select(
        ["node_id"] + [f"{label}_prob_diff" for label in labels] + [max_abs_diff]
    )
    
    return result