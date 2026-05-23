"""
Tests for GLP validation functionality.

This module provides comprehensive testing for the validation functions,
including train/test split validation, external validation, and metric
calculations with various configurations and edge cases.
"""

import pytest
import numpy as np
import polars as pl
import networkit as nk
from typing import Dict, List, Any

from src.glp.validation import (
    train_test_split_validation,
    external_validation,
    cross_validate,
    get_validation_summary,
    _validate_split_inputs,
    _validate_cv_inputs,
    _split_seed_data,
    _generate_cv_folds,
    _extract_test_predictions,
    _calculate_validation_metrics,
    _aggregate_cv_results,
    _count_labels
)
from src.common.id_mapper import IDMapper
from src.common.exceptions import (
    ValidationError,
    ConfigurationError,
    ComputationError
)


class TestTrainTestSplitValidation:
    """Test train/test split validation functionality."""
    
    def setup_method(self):
        """Set up test graph and data."""
        # Create a simple connected graph
        self.graph = nk.Graph(8, weighted=True, directed=False)
        # Create two clusters
        # Cluster 1: nodes 0,1,2,3
        for i in range(3):
            self.graph.addEdge(i, i+1, 1.0)
        self.graph.addEdge(0, 3, 1.0)  # Close cluster 1
        
        # Cluster 2: nodes 4,5,6,7  
        for i in range(4, 7):
            self.graph.addEdge(i, i+1, 1.0)
        self.graph.addEdge(4, 7, 1.0)  # Close cluster 2
        
        # Connect clusters
        self.graph.addEdge(3, 4, 0.5)
        
        # Set up ID mapper
        self.id_mapper = IDMapper()
        for i in range(8):
            self.id_mapper.add_mapping(f"node_{i}", i)
        
        # Create balanced seed labels
        self.seed_labels = {
            "node_0": "A", "node_1": "A", "node_2": "A",  # 3 A's in cluster 1
            "node_4": "B", "node_5": "B", "node_6": "B"   # 3 B's in cluster 2
        }
        self.labels = ["A", "B"]
    
    def test_basic_validation(self):
        """Test basic train/test split validation."""
        results = train_test_split_validation(
            self.graph, self.id_mapper, self.seed_labels, self.labels,
            test_size=0.3, random_seed=42
        )
        
        # Check result structure
        expected_keys = {
            "accuracy", "precision", "recall", "f1_score",
            "macro_precision", "macro_recall", "macro_f1",
            "confusion_matrix", "test_predictions", 
            "train_size", "test_size", "classification_report"
        }
        assert set(results.keys()) >= expected_keys
        
        # Check data types and ranges
        assert isinstance(results["accuracy"], float)
        assert 0.0 <= results["accuracy"] <= 1.0
        
        assert isinstance(results["precision"], dict)
        assert set(results["precision"].keys()) == {"A", "B"}
        
        assert isinstance(results["confusion_matrix"], np.ndarray)
        assert results["confusion_matrix"].shape == (2, 2)
        
        assert isinstance(results["test_predictions"], pl.DataFrame)
        
        # Check split sizes
        assert results["train_size"] + results["test_size"] == len(self.seed_labels)
        assert results["test_size"] > 0
        assert results["train_size"] > 0
    
    def test_stratified_vs_non_stratified(self):
        """Test stratified vs non-stratified splitting."""
        # Stratified split
        results_stratified = train_test_split_validation(
            self.graph, self.id_mapper, self.seed_labels, self.labels,
            test_size=0.33, stratify=True, random_seed=42
        )
        
        # Non-stratified split
        results_non_stratified = train_test_split_validation(
            self.graph, self.id_mapper, self.seed_labels, self.labels,
            test_size=0.33, stratify=False, random_seed=42
        )
        
        # Both should work
        assert isinstance(results_stratified["accuracy"], float)
        assert isinstance(results_non_stratified["accuracy"], float)
        
        # Results may differ due to different splitting strategies
        assert results_stratified["test_size"] == results_non_stratified["test_size"]
    
    def test_different_test_sizes(self):
        """Test validation with different test_size values."""
        test_sizes = [0.17, 0.33, 0.5]  # Use sizes that work well with 6 seeds
        
        for test_size in test_sizes:
            results = train_test_split_validation(
                self.graph, self.id_mapper, self.seed_labels, self.labels,
                test_size=test_size, random_seed=42, stratify=False  # Disable stratification for flexibility
            )
            
            # Check that test size is approximately correct
            expected_test_size = round(len(self.seed_labels) * test_size)
            actual_test_size = results["test_size"]
            
            # Allow for rounding differences
            assert abs(actual_test_size - expected_test_size) <= 1
            
            # Ensure we have training data
            assert results["train_size"] >= 1
            assert results["test_size"] >= 1
    
    def test_perfect_classification_synthetic(self):
        """Test with synthetic data that should achieve perfect classification."""
        # Create isolated clusters
        perfect_graph = nk.Graph(6, weighted=True, directed=False)
        
        # Cluster A: nodes 0,1,2
        perfect_graph.addEdge(0, 1, 1.0)
        perfect_graph.addEdge(1, 2, 1.0)
        perfect_graph.addEdge(0, 2, 1.0)
        
        # Cluster B: nodes 3,4,5  
        perfect_graph.addEdge(3, 4, 1.0)
        perfect_graph.addEdge(4, 5, 1.0)
        perfect_graph.addEdge(3, 5, 1.0)
        
        # No connection between clusters
        
        perfect_mapper = IDMapper()
        for i in range(6):
            perfect_mapper.add_mapping(f"node_{i}", i)
        
        perfect_seeds = {
            "node_0": "A", "node_1": "A",  # Seeds in cluster A
            "node_3": "B", "node_4": "B"   # Seeds in cluster B
        }
        
        results = train_test_split_validation(
            perfect_graph, perfect_mapper, perfect_seeds, ["A", "B"],
            test_size=0.5, random_seed=42,
            alpha=0.8, max_iterations=50
        )
        
        # Should achieve high accuracy on this well-separated data
        assert results["accuracy"] >= 0.8  # Allow some tolerance
        assert results["macro_f1"] >= 0.7
    
    def test_glp_kwargs_passing(self):
        """Test that GLP kwargs are properly passed through."""
        # Test with different alpha value
        results_low_alpha = train_test_split_validation(
            self.graph, self.id_mapper, self.seed_labels, self.labels,
            test_size=0.3, random_seed=42, alpha=0.1
        )
        
        results_high_alpha = train_test_split_validation(
            self.graph, self.id_mapper, self.seed_labels, self.labels,
            test_size=0.3, random_seed=42, alpha=0.9
        )
        
        # Both should work (results may differ)
        assert isinstance(results_low_alpha["accuracy"], float)
        assert isinstance(results_high_alpha["accuracy"], float)
    
    def test_reproducibility(self):
        """Test that results are reproducible with same random seed."""
        results1 = train_test_split_validation(
            self.graph, self.id_mapper, self.seed_labels, self.labels,
            test_size=0.3, random_seed=123
        )
        
        results2 = train_test_split_validation(
            self.graph, self.id_mapper, self.seed_labels, self.labels,
            test_size=0.3, random_seed=123
        )
        
        # Results should be identical
        assert results1["accuracy"] == results2["accuracy"]
        assert results1["train_size"] == results2["train_size"]
        assert results1["test_size"] == results2["test_size"]
        assert np.array_equal(results1["confusion_matrix"], results2["confusion_matrix"])


class TestInputValidation:
    """Test input validation for train/test split."""
    
    def setup_method(self):
        """Set up minimal test data."""
        self.graph = nk.Graph(3, weighted=True)
        self.graph.addEdge(0, 1, 1.0)
        
        self.id_mapper = IDMapper()
        for i in range(3):
            self.id_mapper.add_mapping(f"node_{i}", i)
    
    def test_empty_seed_labels(self):
        """Test validation with empty seed labels."""
        with pytest.raises(ValidationError, match="seed_labels cannot be empty"):
            train_test_split_validation(
                self.graph, self.id_mapper, {}, ["A", "B"]
            )
    
    def test_invalid_test_size(self):
        """Test validation with invalid test_size values."""
        seed_labels = {"node_0": "A", "node_1": "B"}
        
        # Test size too small
        with pytest.raises(ConfigurationError, match="test_size must be between 0 and 1"):
            train_test_split_validation(
                self.graph, self.id_mapper, seed_labels, ["A", "B"],
                test_size=0.0
            )
        
        # Test size too large
        with pytest.raises(ConfigurationError, match="test_size must be between 0 and 1"):
            train_test_split_validation(
                self.graph, self.id_mapper, seed_labels, ["A", "B"],
                test_size=1.0
            )
        
        # Test size way too large
        with pytest.raises(ConfigurationError, match="test_size must be between 0 and 1"):
            train_test_split_validation(
                self.graph, self.id_mapper, seed_labels, ["A", "B"],
                test_size=1.5
            )
    
    def test_insufficient_seeds(self):
        """Test validation with insufficient seeds."""
        # Only one seed
        with pytest.raises(ValidationError, match="Need at least 2 seeds"):
            train_test_split_validation(
                self.graph, self.id_mapper, {"node_0": "A"}, ["A", "B"]
            )
    
    def test_stratification_warnings(self):
        """Test warnings for problematic stratification."""
        # Unbalanced labels that might cause stratification issues
        unbalanced_seeds = {
            "node_0": "A", "node_1": "A", "node_2": "A",  # 3 A's
            "node_3": "B"  # 1 B - will cause stratification warning
        }
        
        graph = nk.Graph(4, weighted=True)
        graph.addEdge(0, 1, 1.0)
        
        mapper = IDMapper()
        for i in range(4):
            mapper.add_mapping(f"node_{i}", i)
        
        # Should warn about stratification but still work
        with pytest.warns(UserWarning, match="Stratified split may fail"):
            results = train_test_split_validation(
                graph, mapper, unbalanced_seeds, ["A", "B"],
                stratify=True, test_size=0.25
            )
            # Should still return results
            assert isinstance(results["accuracy"], float)


class TestExternalValidation:
    """Test external validation functionality."""
    
    def setup_method(self):
        """Set up test data."""
        # Create mock GLP predictions
        self.predictions = pl.DataFrame({
            "node_id": ["node_1", "node_2", "node_3", "node_4", "node_5"],
            "A_prob": [0.9, 0.7, 0.2, 0.1, 0.3],
            "B_prob": [0.1, 0.3, 0.8, 0.9, 0.7],
            "dominant_label": ["A", "A", "B", "B", "B"],
            "confidence": [0.9, 0.7, 0.8, 0.9, 0.7],
            "is_seed": [False, False, False, False, False]
        })
        
        self.labels = ["A", "B"]
    
    def test_perfect_external_validation(self):
        """Test external validation with perfect agreement."""
        # Validation labels that match predictions exactly
        validation_labels = {
            "node_1": "A", "node_2": "A", "node_3": "B", 
            "node_4": "B", "node_5": "B"
        }
        
        results = external_validation(
            self.predictions, validation_labels, self.labels
        )
        
        # Should achieve perfect accuracy
        assert results["accuracy"] == 1.0
        assert results["macro_f1"] == 1.0
        assert all(p == 1.0 for p in results["precision"].values())
        assert all(r == 1.0 for r in results["recall"].values())
    
    def test_partial_external_validation(self):
        """Test external validation with partial agreement."""
        # Validation labels that partially disagree with predictions
        validation_labels = {
            "node_1": "A",  # Agrees (predicted A)
            "node_2": "B",  # Disagrees (predicted A, actual B)
            "node_3": "B",  # Agrees (predicted B)
            "node_4": "A",  # Disagrees (predicted B, actual A)
        }
        
        results = external_validation(
            self.predictions, validation_labels, self.labels
        )
        
        # Should have 50% accuracy (2 out of 4 correct)
        assert results["accuracy"] == 0.5
        assert 0.0 <= results["macro_f1"] <= 1.0
        assert results["validation_size"] == 4
    
    def test_external_validation_subset(self):
        """Test external validation with subset of predictions."""
        # Validate only some nodes
        validation_labels = {
            "node_1": "A",
            "node_3": "B"
        }
        
        results = external_validation(
            self.predictions, validation_labels, self.labels
        )
        
        assert results["validation_size"] == 2
        assert len(results["test_predictions"]) == 2
        assert results["accuracy"] == 1.0  # Both match
    
    def test_external_validation_missing_nodes(self):
        """Test external validation with missing prediction nodes."""
        # Include node not in predictions
        validation_labels = {
            "node_1": "A",
            "node_999": "B"  # Not in predictions
        }
        
        with pytest.raises(ComputationError, match="Missing predictions"):
            external_validation(
                self.predictions, validation_labels, self.labels
            )
    
    def test_empty_validation_labels(self):
        """Test external validation with empty validation set."""
        with pytest.raises(ValidationError, match="validation_labels cannot be empty"):
            external_validation(self.predictions, {}, self.labels)


class TestHelperFunctions:
    """Test validation helper functions."""
    
    def test_validate_split_inputs(self):
        """Test input validation helper."""
        seed_labels = {"node_1": "A", "node_2": "B"}
        labels = ["A", "B"]
        
        # Valid inputs should not raise
        _validate_split_inputs(seed_labels, labels, 0.2, True)
        _validate_split_inputs(seed_labels, labels, 0.8, False)
        
        # Invalid inputs should raise
        with pytest.raises(ValidationError):
            _validate_split_inputs({}, labels, 0.2, True)  # Empty seeds
        
        with pytest.raises(ConfigurationError):
            _validate_split_inputs(seed_labels, labels, 1.5, True)  # Invalid test_size
        
        with pytest.raises(ValidationError):
            _validate_split_inputs({"node_1": "A"}, labels, 0.2, True)  # Too few seeds
    
    def test_split_seed_data(self):
        """Test seed data splitting."""
        seed_labels = {
            "node_1": "A", "node_2": "A", "node_3": "A",
            "node_4": "B", "node_5": "B", "node_6": "B"
        }
        labels = ["A", "B"]
        
        # Test stratified split
        train_seeds, test_seeds, train_labels, test_labels = _split_seed_data(
            seed_labels, labels, 0.33, stratify=True, random_seed=42
        )
        
        assert len(train_seeds) + len(test_seeds) == len(seed_labels)
        assert len(train_labels) == len(train_seeds)
        assert len(test_labels) == len(test_seeds)
        
        # Check that split maintains some balance (stratified)
        train_counts = _count_labels(train_labels)
        test_counts = _count_labels(test_labels)
        
        assert "A" in train_counts and "B" in train_counts  # Both labels in training
        
        # Test non-stratified split
        train_seeds2, test_seeds2, _, _ = _split_seed_data(
            seed_labels, labels, 0.33, stratify=False, random_seed=42
        )
        
        assert len(train_seeds2) + len(test_seeds2) == len(seed_labels)
    
    def test_calculate_validation_metrics(self):
        """Test metric calculation."""
        true_labels = ["A", "A", "B", "B", "A"]
        predicted_labels = ["A", "B", "B", "B", "A"]
        labels = ["A", "B"]
        
        metrics = _calculate_validation_metrics(true_labels, predicted_labels, labels)
        
        # Check structure
        assert "accuracy" in metrics
        assert "precision" in metrics
        assert "recall" in metrics
        assert "f1_score" in metrics
        assert "confusion_matrix" in metrics
        
        # Check accuracy calculation (4 out of 5 correct)
        assert metrics["accuracy"] == 0.8
        
        # Check confusion matrix shape
        assert metrics["confusion_matrix"].shape == (2, 2)
        
        # Check that precision/recall are per-label
        assert set(metrics["precision"].keys()) == {"A", "B"}
        assert set(metrics["recall"].keys()) == {"A", "B"}
    
    def test_count_labels(self):
        """Test label counting utility."""
        labels = ["A", "A", "B", "A", "C", "B"]
        counts = _count_labels(labels)
        
        expected = {"A": 3, "B": 2, "C": 1}
        assert counts == expected
        
        # Test empty list
        assert _count_labels([]) == {}
        
        # Test single label
        assert _count_labels(["A"]) == {"A": 1}


class TestValidationSummary:
    """Test validation summary generation."""
    
    def test_get_validation_summary(self):
        """Test validation summary formatting."""
        # Mock validation results
        validation_results = {
            "accuracy": 0.85,
            "macro_f1": 0.82,
            "test_size": 20,
            "precision": {"A": 0.9, "B": 0.8},
            "recall": {"A": 0.85, "B": 0.9},
            "f1_score": {"A": 0.875, "B": 0.85},
            "confusion_matrix": np.array([[10, 2], [1, 7]])
        }
        
        summary = get_validation_summary(validation_results)
        
        # Check that summary contains key information
        assert "Accuracy: 0.850" in summary
        assert "Macro F1-Score: 0.820" in summary
        assert "Test Size: 20" in summary
        assert "A: P=0.900" in summary
        assert "B: P=0.800" in summary
        assert "Confusion Matrix" in summary
        
        # Should be a multi-line string
        assert "\n" in summary
        assert summary.startswith("=== GLP Validation Summary ===")


class TestIntegrationScenarios:
    """Test realistic integration scenarios."""
    
    def setup_method(self):
        """Set up realistic test scenario."""
        # Create a larger graph with clear community structure
        self.graph = nk.Graph(20, weighted=True, directed=False)
        
        # Community 1: nodes 0-9
        for i in range(9):
            self.graph.addEdge(i, i+1, 1.0)
        self.graph.addEdge(0, 9, 1.0)  # Close the ring
        
        # Add internal connections
        self.graph.addEdge(0, 5, 0.8)
        self.graph.addEdge(2, 7, 0.8)
        
        # Community 2: nodes 10-19
        for i in range(10, 19):
            self.graph.addEdge(i, i+1, 1.0)
        self.graph.addEdge(10, 19, 1.0)  # Close the ring
        
        # Add internal connections
        self.graph.addEdge(10, 15, 0.8)
        self.graph.addEdge(12, 17, 0.8)
        
        # Weak inter-community links
        self.graph.addEdge(4, 14, 0.3)
        self.graph.addEdge(6, 16, 0.3)
        
        # Set up ID mapper
        self.id_mapper = IDMapper()
        for i in range(20):
            self.id_mapper.add_mapping(f"node_{i}", i)
    
    def test_realistic_community_validation(self):
        """Test validation on realistic community structure."""
        # Create seeds based on community structure
        community_seeds = {
            # Community 1 - label "community_1"
            "node_0": "community_1", "node_2": "community_1", 
            "node_4": "community_1", "node_6": "community_1",
            "node_8": "community_1",
            
            # Community 2 - label "community_2"  
            "node_10": "community_2", "node_12": "community_2",
            "node_14": "community_2", "node_16": "community_2", 
            "node_18": "community_2"
        }
        
        labels = ["community_1", "community_2"]
        
        results = train_test_split_validation(
            self.graph, self.id_mapper, community_seeds, labels,
            test_size=0.3, random_seed=42, alpha=0.8, max_iterations=100
        )
        
        # Should achieve reasonable accuracy on this structured data
        assert results["accuracy"] >= 0.6  # Allow for some uncertainty
        assert results["train_size"] >= 6  # Should have training data
        assert results["test_size"] >= 2   # Should have test data
        
        # Check that we have predictions for test nodes
        assert len(results["test_predictions"]) == results["test_size"]
        
        # Summary should be readable
        summary = get_validation_summary(results)
        assert len(summary) > 100  # Should be substantial


class TestCrossValidation:
    """Test K-fold cross-validation functionality."""
    
    def setup_method(self):
        """Set up test data for cross-validation."""
        # Create a graph with community structure for cross-validation
        self.graph = nk.Graph(12, weighted=True, directed=False)
        
        # Community 1: nodes 0-5
        for i in range(5):
            self.graph.addEdge(i, i+1, 1.0)
        self.graph.addEdge(0, 5, 1.0)  # Close the ring
        
        # Community 2: nodes 6-11
        for i in range(6, 11):
            self.graph.addEdge(i, i+1, 1.0)
        self.graph.addEdge(6, 11, 1.0)  # Close the ring
        
        # Connect communities weakly
        self.graph.addEdge(2, 8, 0.3)
        
        # Set up ID mapper
        self.id_mapper = IDMapper()
        for i in range(12):
            self.id_mapper.add_mapping(f"node_{i}", i)
        
        # Create balanced seed labels (5 per community)
        self.seed_labels = {
            # Community 1 - label "A"
            "node_0": "A", "node_1": "A", "node_2": "A", 
            "node_3": "A", "node_4": "A",
            
            # Community 2 - label "B"
            "node_6": "B", "node_7": "B", "node_8": "B",
            "node_9": "B", "node_10": "B"
        }
        self.labels = ["A", "B"]
    
    def test_basic_cross_validation(self):
        """Test basic K-fold cross-validation."""
        results = cross_validate(
            self.graph, self.id_mapper, self.seed_labels, self.labels,
            k_folds=5, random_seed=42
        )
        
        # Check result structure
        expected_keys = {
            "mean_accuracy", "std_accuracy", "fold_accuracies",
            "mean_precision", "std_precision", "mean_recall", "std_recall",
            "mean_f1", "std_f1", "mean_macro_f1", "std_macro_f1",
            "fold_results", "aggregate_confusion_matrix", "k_folds"
        }
        assert set(results.keys()) >= expected_keys
        
        # Check data types and ranges
        assert isinstance(results["mean_accuracy"], float)
        assert 0.0 <= results["mean_accuracy"] <= 1.0
        assert isinstance(results["std_accuracy"], float)
        assert results["std_accuracy"] >= 0.0
        
        assert isinstance(results["fold_accuracies"], list)
        assert len(results["fold_accuracies"]) == 5
        
        assert isinstance(results["mean_precision"], dict)
        assert set(results["mean_precision"].keys()) == {"A", "B"}
        
        assert isinstance(results["aggregate_confusion_matrix"], np.ndarray)
        assert results["aggregate_confusion_matrix"].shape == (2, 2)
        
        assert results["k_folds"] == 5
        assert len(results["fold_results"]) == 5
    
    def test_different_k_folds(self):
        """Test cross-validation with different k values."""
        k_values = [3, 5]  # Use smaller k values for the 10 seeds we have
        
        for k in k_values:
            results = cross_validate(
                self.graph, self.id_mapper, self.seed_labels, self.labels,
                k_folds=k, random_seed=42
            )
            
            assert results["k_folds"] == k
            assert len(results["fold_results"]) == k
            assert len(results["fold_accuracies"]) == k
            
            # Each fold should have had at least 1 test sample
            for fold_result in results["fold_results"]:
                assert fold_result["test_size"] >= 1
                assert fold_result["train_size"] >= 1
    
    def test_stratified_vs_non_stratified_cv(self):
        """Test stratified vs non-stratified cross-validation."""
        # Stratified CV
        results_stratified = cross_validate(
            self.graph, self.id_mapper, self.seed_labels, self.labels,
            k_folds=5, stratify=True, random_seed=42
        )
        
        # Non-stratified CV
        results_non_stratified = cross_validate(
            self.graph, self.id_mapper, self.seed_labels, self.labels,
            k_folds=5, stratify=False, random_seed=42
        )
        
        # Both should work
        assert isinstance(results_stratified["mean_accuracy"], float)
        assert isinstance(results_non_stratified["mean_accuracy"], float)
        
        # Should have same number of folds
        assert results_stratified["k_folds"] == results_non_stratified["k_folds"]
    
    def test_cross_validation_reproducibility(self):
        """Test that cross-validation is reproducible with same random seed."""
        results1 = cross_validate(
            self.graph, self.id_mapper, self.seed_labels, self.labels,
            k_folds=3, random_seed=123
        )
        
        results2 = cross_validate(
            self.graph, self.id_mapper, self.seed_labels, self.labels,
            k_folds=3, random_seed=123
        )
        
        # Results should be identical
        assert results1["mean_accuracy"] == results2["mean_accuracy"]
        assert results1["fold_accuracies"] == results2["fold_accuracies"]
        assert np.array_equal(results1["aggregate_confusion_matrix"], 
                             results2["aggregate_confusion_matrix"])
    
    def test_cv_with_glp_kwargs(self):
        """Test cross-validation with custom GLP parameters."""
        results = cross_validate(
            self.graph, self.id_mapper, self.seed_labels, self.labels,
            k_folds=3, random_seed=42,
            alpha=0.7, max_iterations=100  # Custom GLP parameters
        )
        
        # Should work with custom parameters
        assert isinstance(results["mean_accuracy"], float)
        assert results["k_folds"] == 3
    
    def test_cv_aggregation_statistics(self):
        """Test that cross-validation statistics are calculated correctly."""
        results = cross_validate(
            self.graph, self.id_mapper, self.seed_labels, self.labels,
            k_folds=3, random_seed=42
        )
        
        # Check that mean is actually the mean of fold results
        calculated_mean = np.mean(results["fold_accuracies"])
        assert abs(results["mean_accuracy"] - calculated_mean) < 1e-10
        
        # Check that standard deviation is calculated correctly
        if len(results["fold_accuracies"]) > 1:
            calculated_std = np.std(results["fold_accuracies"], ddof=1)
            assert abs(results["std_accuracy"] - calculated_std) < 1e-10
        
        # Check confusion matrix aggregation
        manual_sum = np.zeros_like(results["aggregate_confusion_matrix"])
        for fold_result in results["fold_results"]:
            manual_sum += fold_result["confusion_matrix"]
        
        assert np.array_equal(results["aggregate_confusion_matrix"], manual_sum)


class TestCrossValidationInputValidation:
    """Test input validation for cross-validation."""
    
    def setup_method(self):
        """Set up minimal test data."""
        self.graph = nk.Graph(6, weighted=True)
        for i in range(5):
            self.graph.addEdge(i, i+1, 1.0)
        
        self.id_mapper = IDMapper()
        for i in range(6):
            self.id_mapper.add_mapping(f"node_{i}", i)
    
    def test_cv_empty_seed_labels(self):
        """Test cross-validation with empty seed labels."""
        with pytest.raises(ValidationError, match="seed_labels cannot be empty"):
            cross_validate(self.graph, self.id_mapper, {}, ["A", "B"])
    
    def test_cv_invalid_k_folds(self):
        """Test cross-validation with invalid k_folds values."""
        seed_labels = {"node_0": "A", "node_1": "B", "node_2": "A"}
        
        # k_folds too small
        with pytest.raises(ConfigurationError, match="k_folds must be at least 2"):
            cross_validate(self.graph, self.id_mapper, seed_labels, ["A", "B"], k_folds=1)
        
        # k_folds = 0
        with pytest.raises(ConfigurationError, match="k_folds must be at least 2"):
            cross_validate(self.graph, self.id_mapper, seed_labels, ["A", "B"], k_folds=0)
    
    def test_cv_insufficient_seeds(self):
        """Test cross-validation with insufficient seeds for k folds."""
        seed_labels = {"node_0": "A", "node_1": "B"}  # Only 2 seeds
        
        # Need at least k_folds seeds
        with pytest.raises(ValidationError, match="Need at least 5 seeds for 5-fold CV"):
            cross_validate(self.graph, self.id_mapper, seed_labels, ["A", "B"], k_folds=5)
    
    def test_cv_stratification_warnings(self):
        """Test warnings for problematic stratification in CV."""
        # Create unbalanced seeds that will cause stratification issues
        unbalanced_seeds = {
            "node_0": "A", "node_1": "A", "node_2": "A",  # 3 A's
            "node_3": "B", "node_4": "B"  # 2 B's - insufficient for 5-fold stratified
        }
        
        # Should warn about stratification but still work
        with pytest.warns(UserWarning, match="Stratified.*-fold CV may fail"):
            results = cross_validate(
                self.graph, self.id_mapper, unbalanced_seeds, ["A", "B"],
                k_folds=5, stratify=True
            )
            # Should still return results
            assert isinstance(results["mean_accuracy"], float)


class TestCrossValidationHelperFunctions:
    """Test cross-validation helper functions."""
    
    def test_validate_cv_inputs(self):
        """Test cross-validation input validation helper."""
        seed_labels = {f"node_{i}": "A" if i < 3 else "B" for i in range(6)}
        labels = ["A", "B"]
        
        # Valid inputs should not raise
        _validate_cv_inputs(seed_labels, labels, 3, True)
        _validate_cv_inputs(seed_labels, labels, 5, False)
        
        # Invalid inputs should raise
        with pytest.raises(ValidationError):
            _validate_cv_inputs({}, labels, 3, True)  # Empty seeds
        
        with pytest.raises(ConfigurationError):
            _validate_cv_inputs(seed_labels, labels, 1, True)  # Invalid k_folds
        
        with pytest.raises(ValidationError):
            _validate_cv_inputs({"node_0": "A"}, labels, 3, True)  # Too few seeds
    
    def test_generate_cv_folds(self):
        """Test cross-validation fold generation."""
        seed_labels = {
            "node_0": "A", "node_1": "A", "node_2": "A",
            "node_3": "B", "node_4": "B", "node_5": "B"
        }
        labels = ["A", "B"]
        
        # Test stratified folding
        fold_splits = _generate_cv_folds(
            seed_labels, labels, 3, stratify=True, random_seed=42
        )
        
        assert len(fold_splits) == 3
        
        # Check that each split has train and test sets
        total_seeds = len(seed_labels)
        for train_seeds, test_seeds in fold_splits:
            assert len(train_seeds) + len(test_seeds) == total_seeds
            assert len(test_seeds) >= 1
            assert len(train_seeds) >= 1
            
            # No overlap between train and test
            assert set(train_seeds.keys()).isdisjoint(set(test_seeds.keys()))
        
        # Test non-stratified folding
        fold_splits_unstrat = _generate_cv_folds(
            seed_labels, labels, 3, stratify=False, random_seed=42
        )
        
        assert len(fold_splits_unstrat) == 3
    
    def test_aggregate_cv_results(self):
        """Test cross-validation results aggregation."""
        # Create mock fold results
        fold_results = [
            {
                "accuracy": 0.8,
                "precision": {"A": 0.9, "B": 0.7},
                "recall": {"A": 0.8, "B": 0.8},
                "f1_score": {"A": 0.85, "B": 0.75},
                "macro_f1": 0.8,
                "confusion_matrix": np.array([[4, 1], [1, 4]])
            },
            {
                "accuracy": 0.6,
                "precision": {"A": 0.7, "B": 0.5},
                "recall": {"A": 0.6, "B": 0.6},
                "f1_score": {"A": 0.65, "B": 0.55},
                "macro_f1": 0.6,
                "confusion_matrix": np.array([[3, 2], [2, 3]])
            }
        ]
        
        labels = ["A", "B"]
        k_folds = 2
        
        aggregated = _aggregate_cv_results(fold_results, labels, k_folds)
        
        # Check accuracy aggregation
        expected_mean_accuracy = (0.8 + 0.6) / 2
        assert abs(aggregated["mean_accuracy"] - expected_mean_accuracy) < 1e-10
        
        # Check per-label precision aggregation
        expected_mean_precision_A = (0.9 + 0.7) / 2
        assert abs(aggregated["mean_precision"]["A"] - expected_mean_precision_A) < 1e-10
        
        # Check confusion matrix aggregation
        expected_aggregate_cm = np.array([[7, 3], [3, 7]])
        assert np.array_equal(aggregated["aggregate_confusion_matrix"], expected_aggregate_cm)
        
        # Check standard deviation calculation
        expected_std_accuracy = np.std([0.8, 0.6], ddof=1)
        assert abs(aggregated["std_accuracy"] - expected_std_accuracy) < 1e-10


class TestAdditionalExternalValidationTests:
    """Additional tests for external validation to ensure comprehensive coverage."""
    
    def test_external_validation_detailed_metrics(self):
        """Test external validation with detailed metric checking."""
        # Create mock predictions with known accuracy
        predictions = pl.DataFrame({
            "node_id": ["node_1", "node_2", "node_3", "node_4"],
            "A_prob": [0.9, 0.8, 0.2, 0.1],
            "B_prob": [0.1, 0.2, 0.8, 0.9],
            "dominant_label": ["A", "A", "B", "B"],
            "confidence": [0.9, 0.8, 0.8, 0.9],
            "is_seed": [False, False, False, False]
        })
        
        # Create validation labels with known accuracy (75% - 3 out of 4 correct)
        validation_labels = {
            "node_1": "A",  # Correct
            "node_2": "B",  # Incorrect (predicted A, actual B)
            "node_3": "B",  # Correct
            "node_4": "B"   # Correct
        }
        
        labels = ["A", "B"]
        
        results = external_validation(predictions, validation_labels, labels)
        
        # Should have 75% accuracy (3 out of 4 correct)
        assert results["accuracy"] == 0.75
        assert results["validation_size"] == 4
        
        # Check confusion matrix
        # True: [A, B, B, B], Predicted: [A, A, B, B]
        # Rows = true labels, Cols = predicted labels
        # A row: 1 predicted A (correct), 0 predicted B
        # B row: 1 predicted A (incorrect), 2 predicted B (correct)
        expected_cm = np.array([[1, 0], [1, 2]])
        assert np.array_equal(results["confusion_matrix"], expected_cm)


if __name__ == "__main__":
    pytest.main([__file__])