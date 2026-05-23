"""
Tests for evaluation module functions.

This module tests the analysis and comparison functionality for GLP results:
- Label distribution analysis
- Directional propagation comparison
- Confidence assessment
- Error handling and edge cases
"""

import pytest
import numpy as np
import polars as pl
from unittest.mock import patch

from src.glp.evaluation import (
    analyze_label_distribution,
    compare_directional_results,
    _validate_predictions_dataframe,
    _validate_directional_inputs,
    _calculate_label_counts,
    _calculate_confidence_by_label,
    _calculate_seed_confidence_comparison,
    _calculate_probability_distributions,
    _calculate_label_entropy,
    _merge_directional_dataframes,
    _calculate_agreement_rate,
    _calculate_label_correlations,
    _identify_divergent_nodes,
    _calculate_direction_bias,
    _compare_directional_confidence,
    _calculate_probability_differences
)

from src.common.exceptions import ValidationError


class TestAnalyzeLabelDistribution:
    """Test label distribution analysis functionality."""
    
    def test_basic_distribution_analysis(self):
        """Test basic label distribution analysis."""
        # Create sample predictions DataFrame
        predictions = pl.DataFrame({
            "node_id": ["n1", "n2", "n3", "n4", "n5"],
            "A_prob": [0.8, 0.2, 0.7, 0.1, 0.6],
            "B_prob": [0.2, 0.8, 0.3, 0.9, 0.4],
            "dominant_label": ["A", "B", "A", "B", "A"],
            "confidence": [0.8, 0.8, 0.7, 0.9, 0.6],
            "is_seed": [True, True, False, False, False]
        })
        labels = ["A", "B"]
        
        result = analyze_label_distribution(predictions, labels)
        
        # Check basic structure
        expected_keys = [
            "label_counts", "mean_confidence", "confidence_by_label",
            "probability_distributions", "high_confidence_nodes", 
            "uncertain_nodes", "seed_vs_nonseed_confidence", "label_entropy"
        ]
        assert all(key in result for key in expected_keys)
        
        # Check label counts
        assert result["label_counts"] == {"A": 3, "B": 2}
        
        # Check mean confidence
        assert result["mean_confidence"] == pytest.approx(0.76, abs=0.01)
        
        # Check confidence by label
        assert result["confidence_by_label"]["A"] == pytest.approx(0.7, abs=0.01)
        assert result["confidence_by_label"]["B"] == pytest.approx(0.85, abs=0.01)
        
        # Check high confidence nodes (confidence > 0.8)
        assert len(result["high_confidence_nodes"]) == 1  # Only n4 with 0.9
        
        # Check uncertain nodes (confidence < 0.5)
        assert len(result["uncertain_nodes"]) == 0
        
        # Check probability distributions
        assert "A" in result["probability_distributions"]
        assert "B" in result["probability_distributions"]
        assert len(result["probability_distributions"]["A"]) == 20  # 20 bins
    
    def test_seed_vs_nonseed_confidence(self):
        """Test seed vs non-seed confidence comparison."""
        predictions = pl.DataFrame({
            "node_id": ["n1", "n2", "n3", "n4"],
            "A_prob": [0.9, 0.8, 0.6, 0.5],
            "B_prob": [0.1, 0.2, 0.4, 0.5],
            "dominant_label": ["A", "A", "A", "A"],
            "confidence": [0.9, 0.8, 0.6, 0.5],
            "is_seed": [True, True, False, False]
        })
        labels = ["A", "B"]
        
        result = analyze_label_distribution(predictions, labels)
        
        seed_conf = result["seed_vs_nonseed_confidence"]
        assert seed_conf["seed_confidence"] == pytest.approx(0.85, abs=0.01)
        assert seed_conf["nonseed_confidence"] == pytest.approx(0.55, abs=0.01)
        assert seed_conf["confidence_difference"] == pytest.approx(0.3, abs=0.01)
    
    def test_label_entropy_calculation(self):
        """Test label entropy calculation."""
        # Create predictions with known entropy
        predictions = pl.DataFrame({
            "node_id": ["n1", "n2"],
            "A_prob": [1.0, 0.5],  # First node certain, second uncertain
            "B_prob": [0.0, 0.5],
            "dominant_label": ["A", "A"],
            "confidence": [1.0, 0.5],
            "is_seed": [True, False]
        })
        labels = ["A", "B"]
        
        result = analyze_label_distribution(predictions, labels)
        
        # Entropy should be > 0 due to uncertain node
        assert result["label_entropy"] > 0
        assert result["label_entropy"] < np.log(2)  # Max entropy for 2 labels
    
    def test_uncertain_nodes_identification(self):
        """Test identification of uncertain nodes."""
        predictions = pl.DataFrame({
            "node_id": ["n1", "n2", "n3"],
            "A_prob": [0.9, 0.4, 0.6],
            "B_prob": [0.1, 0.6, 0.4],
            "dominant_label": ["A", "B", "A"],
            "confidence": [0.9, 0.6, 0.6],
            "is_seed": [True, False, False]
        })
        labels = ["A", "B"]
        
        result = analyze_label_distribution(predictions, labels)
        
        # Only high confidence nodes (> 0.8)
        assert len(result["high_confidence_nodes"]) == 1
        assert result["high_confidence_nodes"]["node_id"].to_list() == ["n1"]
        
        # No uncertain nodes (< 0.5)
        assert len(result["uncertain_nodes"]) == 0
    
    def test_missing_label_handling(self):
        """Test handling when some labels have no nodes."""
        predictions = pl.DataFrame({
            "node_id": ["n1", "n2"],
            "A_prob": [0.8, 0.7],
            "B_prob": [0.2, 0.3],
            "dominant_label": ["A", "A"],  # Only label A
            "confidence": [0.8, 0.7],
            "is_seed": [True, False]
        })
        labels = ["A", "B"]
        
        result = analyze_label_distribution(predictions, labels)
        
        # Check label counts
        assert result["label_counts"] == {"A": 2}
        assert "B" not in result["label_counts"]  # No nodes with dominant label B
        
        # Check confidence by label
        assert result["confidence_by_label"]["A"] > 0
        assert result["confidence_by_label"]["B"] == 0.0  # No nodes with this label


class TestCompareDirectionalResults:
    """Test directional results comparison functionality."""
    
    def test_basic_directional_comparison(self):
        """Test basic directional comparison."""
        # Create sample out-degree predictions
        out_preds = pl.DataFrame({
            "node_id": ["n1", "n2", "n3"],
            "A_prob": [0.8, 0.2, 0.6],
            "B_prob": [0.2, 0.8, 0.4],
            "dominant_label": ["A", "B", "A"],
            "confidence": [0.8, 0.8, 0.6],
            "is_seed": [True, True, False]
        })
        
        # Create sample in-degree predictions (slightly different)
        in_preds = pl.DataFrame({
            "node_id": ["n1", "n2", "n3"],
            "A_prob": [0.7, 0.3, 0.5],
            "B_prob": [0.3, 0.7, 0.5],
            "dominant_label": ["A", "B", "A"],  # Same dominant labels
            "confidence": [0.7, 0.7, 0.5],
            "is_seed": [True, True, False]
        })
        
        labels = ["A", "B"]
        
        result = compare_directional_results(out_preds, in_preds, labels)
        
        # Check basic structure
        expected_keys = [
            "agreement_rate", "correlation_by_label", "divergent_nodes",
            "direction_bias", "confidence_comparison", "probability_differences"
        ]
        assert all(key in result for key in expected_keys)
        
        # Check agreement rate (all nodes have same dominant label)
        assert result["agreement_rate"] == 1.0
        
        # Check correlations (should be positive)
        assert "A" in result["correlation_by_label"]
        assert "B" in result["correlation_by_label"]
        
        # Check no divergent nodes
        assert len(result["divergent_nodes"]) == 0
        
        # Check direction bias
        assert "A" in result["direction_bias"]
        assert "B" in result["direction_bias"]
    
    def test_divergent_nodes_identification(self):
        """Test identification of divergent nodes."""
        out_preds = pl.DataFrame({
            "node_id": ["n1", "n2", "n3"],
            "A_prob": [0.8, 0.3, 0.6],
            "B_prob": [0.2, 0.7, 0.4],
            "dominant_label": ["A", "B", "A"],
            "confidence": [0.8, 0.7, 0.6],
            "is_seed": [True, False, False]
        })
        
        in_preds = pl.DataFrame({
            "node_id": ["n1", "n2", "n3"],
            "A_prob": [0.7, 0.6, 0.4],
            "B_prob": [0.3, 0.4, 0.6],
            "dominant_label": ["A", "A", "B"],  # n2 and n3 differ
            "confidence": [0.7, 0.6, 0.6],
            "is_seed": [True, False, False]
        })
        
        labels = ["A", "B"]
        
        result = compare_directional_results(out_preds, in_preds, labels)
        
        # Check agreement rate
        assert result["agreement_rate"] == pytest.approx(1/3, abs=0.01)  # Only n1 agrees
        
        # Check divergent nodes
        assert len(result["divergent_nodes"]) == 2  # n2 and n3
        divergent_ids = result["divergent_nodes"]["node_id"].to_list()
        assert "n2" in divergent_ids
        assert "n3" in divergent_ids
        
        # Check probability differences exist
        assert "A_prob_diff" in result["divergent_nodes"].columns
        assert "B_prob_diff" in result["divergent_nodes"].columns
    
    def test_direction_bias_calculation(self):
        """Test direction bias calculation."""
        # Create data where A is favored by out-degree, B by in-degree
        out_preds = pl.DataFrame({
            "node_id": ["n1", "n2"],
            "A_prob": [0.9, 0.8],  # High A probabilities
            "B_prob": [0.1, 0.2],
            "dominant_label": ["A", "A"],
            "confidence": [0.9, 0.8],
            "is_seed": [True, False]
        })
        
        in_preds = pl.DataFrame({
            "node_id": ["n1", "n2"],
            "A_prob": [0.2, 0.3],  # Low A probabilities
            "B_prob": [0.8, 0.7],  # High B probabilities
            "dominant_label": ["B", "B"],
            "confidence": [0.8, 0.7],
            "is_seed": [True, False]
        })
        
        labels = ["A", "B"]
        
        result = compare_directional_results(out_preds, in_preds, labels)
        
        # Check direction bias
        assert result["direction_bias"]["A"] == "out"  # A favored by out-degree
        assert result["direction_bias"]["B"] == "in"   # B favored by in-degree
    
    def test_confidence_comparison(self):
        """Test confidence comparison between directions."""
        out_preds = pl.DataFrame({
            "node_id": ["n1", "n2"],
            "A_prob": [0.9, 0.8],
            "B_prob": [0.1, 0.2],
            "dominant_label": ["A", "A"],
            "confidence": [0.9, 0.8],  # High confidence
            "is_seed": [True, False]
        })
        
        in_preds = pl.DataFrame({
            "node_id": ["n1", "n2"],
            "A_prob": [0.6, 0.7],
            "B_prob": [0.4, 0.3],
            "dominant_label": ["A", "A"],
            "confidence": [0.6, 0.7],  # Lower confidence
            "is_seed": [True, False]
        })
        
        labels = ["A", "B"]
        
        result = compare_directional_results(out_preds, in_preds, labels)
        
        conf_comp = result["confidence_comparison"]
        assert conf_comp["out_degree_confidence"] == pytest.approx(0.85, abs=0.01)
        assert conf_comp["in_degree_confidence"] == pytest.approx(0.65, abs=0.01)
        assert conf_comp["confidence_difference"] == pytest.approx(0.2, abs=0.01)
    
    def test_probability_differences_calculation(self):
        """Test probability differences calculation."""
        out_preds = pl.DataFrame({
            "node_id": ["n1", "n2"],
            "A_prob": [0.8, 0.6],
            "B_prob": [0.2, 0.4],
            "dominant_label": ["A", "A"],
            "confidence": [0.8, 0.6],
            "is_seed": [True, False]
        })
        
        in_preds = pl.DataFrame({
            "node_id": ["n1", "n2"],
            "A_prob": [0.6, 0.4],
            "B_prob": [0.4, 0.6],
            "dominant_label": ["A", "B"],
            "confidence": [0.6, 0.6],
            "is_seed": [True, False]
        })
        
        labels = ["A", "B"]
        
        result = compare_directional_results(out_preds, in_preds, labels)
        
        prob_diffs = result["probability_differences"]
        
        # Check structure
        expected_cols = ["node_id", "A_prob_diff", "B_prob_diff", "max_abs_diff"]
        assert all(col in prob_diffs.columns for col in expected_cols)
        
        # Check calculated differences
        diffs_data = prob_diffs.to_dicts()
        
        # Node n1: A_diff = 0.8 - 0.6 = 0.2, B_diff = 0.2 - 0.4 = -0.2
        n1_row = next(row for row in diffs_data if row["node_id"] == "n1")
        assert n1_row["A_prob_diff"] == pytest.approx(0.2, abs=0.01)
        assert n1_row["B_prob_diff"] == pytest.approx(-0.2, abs=0.01)
        assert n1_row["max_abs_diff"] == pytest.approx(0.2, abs=0.01)


class TestInputValidation:
    """Test input validation for evaluation functions."""
    
    def test_validate_predictions_dataframe_missing_columns(self):
        """Test validation with missing required columns."""
        # Missing confidence column
        invalid_df = pl.DataFrame({
            "node_id": ["n1", "n2"],
            "A_prob": [0.8, 0.2],
            "B_prob": [0.2, 0.8],
            "dominant_label": ["A", "B"],
            "is_seed": [True, False]
        })
        labels = ["A", "B"]
        
        with pytest.raises(ValidationError, match="Missing required columns"):
            _validate_predictions_dataframe(invalid_df, labels)
    
    def test_validate_predictions_dataframe_missing_prob_columns(self):
        """Test validation with missing probability columns."""
        invalid_df = pl.DataFrame({
            "node_id": ["n1", "n2"],
            "A_prob": [0.8, 0.2],
            # Missing B_prob
            "dominant_label": ["A", "A"],
            "confidence": [0.8, 0.2],
            "is_seed": [True, False]
        })
        labels = ["A", "B"]
        
        with pytest.raises(ValidationError, match="Missing probability columns"):
            _validate_predictions_dataframe(invalid_df, labels)
    
    def test_validate_predictions_dataframe_empty(self):
        """Test validation with empty DataFrame."""
        empty_df = pl.DataFrame()
        labels = ["A", "B"]
        
        with pytest.raises(ValidationError, match="predictions DataFrame cannot be empty"):
            _validate_predictions_dataframe(empty_df, labels)
    
    def test_validate_directional_inputs_different_nodes(self):
        """Test validation when directional DataFrames have different nodes."""
        out_df = pl.DataFrame({
            "node_id": ["n1", "n2"],
            "A_prob": [0.8, 0.2],
            "B_prob": [0.2, 0.8],
            "dominant_label": ["A", "B"],
            "confidence": [0.8, 0.8],
            "is_seed": [True, False]
        })
        
        in_df = pl.DataFrame({
            "node_id": ["n1", "n3"],  # Different node (n3 instead of n2)
            "A_prob": [0.7, 0.3],
            "B_prob": [0.3, 0.7],
            "dominant_label": ["A", "B"],
            "confidence": [0.7, 0.7],
            "is_seed": [True, False]
        })
        
        labels = ["A", "B"]
        
        with pytest.raises(ValidationError, match="DataFrames must have same nodes"):
            _validate_directional_inputs(out_df, in_df, labels)


class TestHelperFunctions:
    """Test individual helper functions."""
    
    def test_calculate_label_counts(self):
        """Test label counting function."""
        df = pl.DataFrame({
            "dominant_label": ["A", "A", "B", "A", "B"],
            "other_col": [1, 2, 3, 4, 5]
        })
        
        counts = _calculate_label_counts(df)
        assert counts == {"A": 3, "B": 2}
    
    def test_calculate_confidence_by_label(self):
        """Test confidence calculation by label."""
        df = pl.DataFrame({
            "dominant_label": ["A", "A", "B", "B"],
            "confidence": [0.8, 0.6, 0.9, 0.7]
        })
        labels = ["A", "B", "C"]  # Include label with no nodes
        
        conf_by_label = _calculate_confidence_by_label(df, labels)
        
        assert conf_by_label["A"] == pytest.approx(0.7, abs=0.01)  # (0.8 + 0.6) / 2
        assert conf_by_label["B"] == pytest.approx(0.8, abs=0.01)  # (0.9 + 0.7) / 2
        assert conf_by_label["C"] == 0.0  # No nodes with this label
    
    def test_calculate_probability_distributions(self):
        """Test probability distribution calculation."""
        df = pl.DataFrame({
            "A_prob": [0.1, 0.5, 0.9, 0.3, 0.7],
            "B_prob": [0.9, 0.5, 0.1, 0.7, 0.3]
        })
        labels = ["A", "B"]
        
        distributions = _calculate_probability_distributions(df, labels)
        
        assert "A" in distributions
        assert "B" in distributions
        assert len(distributions["A"]) == 20  # 20 bins
        assert len(distributions["B"]) == 20
        assert isinstance(distributions["A"], np.ndarray)
    
    def test_merge_directional_dataframes(self):
        """Test merging of directional DataFrames."""
        out_df = pl.DataFrame({
            "node_id": ["n1", "n2"],
            "A_prob": [0.8, 0.2],
            "B_prob": [0.2, 0.8],
            "dominant_label": ["A", "B"],
            "confidence": [0.8, 0.8],
            "other_col": [1, 2]
        })
        
        in_df = pl.DataFrame({
            "node_id": ["n1", "n2"],
            "A_prob": [0.7, 0.3],
            "B_prob": [0.3, 0.7],
            "dominant_label": ["A", "B"],
            "confidence": [0.7, 0.7],
            "other_col": [10, 20]
        })
        
        labels = ["A", "B"]
        
        merged = _merge_directional_dataframes(out_df, in_df, labels)
        
        # Check structure
        expected_cols = [
            "node_id", "out_dominant_label", "out_confidence", "out_A_prob", "out_B_prob",
            "in_dominant_label", "in_confidence", "in_A_prob", "in_B_prob"
        ]
        assert all(col in merged.columns for col in expected_cols)
        assert len(merged) == 2
    
    def test_calculate_agreement_rate(self):
        """Test agreement rate calculation."""
        merged_df = pl.DataFrame({
            "out_dominant_label": ["A", "B", "A"],
            "in_dominant_label": ["A", "A", "A"]  # First and third agree
        })
        
        rate = _calculate_agreement_rate(merged_df)
        assert rate == pytest.approx(2/3, abs=0.01)
    
    def test_calculate_label_correlations(self):
        """Test label correlation calculation."""
        merged_df = pl.DataFrame({
            "out_A_prob": [0.8, 0.6, 0.4],
            "in_A_prob": [0.7, 0.5, 0.3],  # Positive correlation
            "out_B_prob": [0.2, 0.4, 0.6],
            "in_B_prob": [0.6, 0.4, 0.2]   # Negative correlation
        })
        labels = ["A", "B"]
        
        correlations = _calculate_label_correlations(merged_df, labels)
        
        assert correlations["A"] > 0.9  # Strong positive correlation
        assert correlations["B"] < -0.9  # Strong negative correlation
    
    def test_calculate_label_correlations_constant_values(self):
        """Test correlation calculation with constant values (NaN case)."""
        merged_df = pl.DataFrame({
            "out_A_prob": [0.5, 0.5, 0.5],  # Constant values
            "in_A_prob": [0.7, 0.5, 0.3],
            "out_B_prob": [0.2, 0.4, 0.6],
            "in_B_prob": [0.2, 0.4, 0.6]   # Perfect correlation
        })
        labels = ["A", "B"]
        
        correlations = _calculate_label_correlations(merged_df, labels)
        
        assert correlations["A"] == 0.0  # NaN converted to 0
        assert correlations["B"] == pytest.approx(1.0, abs=0.01)  # Perfect correlation


class TestAnalyzeLabelDistributionInputValidation:
    """Test input validation for analyze_label_distribution."""
    
    def test_empty_predictions(self):
        """Test with empty predictions DataFrame."""
        empty_df = pl.DataFrame()
        labels = ["A", "B"]
        
        with pytest.raises(ValidationError):
            analyze_label_distribution(empty_df, labels)
    
    def test_missing_columns(self):
        """Test with missing required columns."""
        incomplete_df = pl.DataFrame({
            "node_id": ["n1"],
            "A_prob": [0.8]
            # Missing other required columns
        })
        labels = ["A"]
        
        with pytest.raises(ValidationError):
            analyze_label_distribution(incomplete_df, labels)


class TestCompareDirectionalResultsInputValidation:
    """Test input validation for compare_directional_results."""
    
    def test_empty_dataframes(self):
        """Test with empty DataFrames."""
        empty_df = pl.DataFrame()
        labels = ["A", "B"]
        
        with pytest.raises(ValidationError):
            compare_directional_results(empty_df, empty_df, labels)
    
    def test_mismatched_nodes(self):
        """Test with DataFrames having different sets of nodes."""
        df1 = pl.DataFrame({
            "node_id": ["n1", "n2"],
            "A_prob": [0.8, 0.2],
            "B_prob": [0.2, 0.8],
            "dominant_label": ["A", "B"],
            "confidence": [0.8, 0.8],
            "is_seed": [True, False]
        })
        
        df2 = pl.DataFrame({
            "node_id": ["n1", "n3"],  # Different second node
            "A_prob": [0.7, 0.3],
            "B_prob": [0.3, 0.7],
            "dominant_label": ["A", "B"],
            "confidence": [0.7, 0.7],
            "is_seed": [True, False]
        })
        
        labels = ["A", "B"]
        
        with pytest.raises(ValidationError, match="DataFrames must have same nodes"):
            compare_directional_results(df1, df2, labels)