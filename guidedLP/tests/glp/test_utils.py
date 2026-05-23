"""
Tests for utils module functions.

This module tests the utility functions for GLP:
- Balanced seed set creation
- Alpha value suggestion
- Seed statistics analysis
- Error handling and edge cases
"""

import pytest
import random
import numpy as np
import networkit as nk
from unittest.mock import patch, MagicMock

from src.glp.utils import (
    create_balanced_seed_set,
    suggest_alpha_value,
    get_seed_statistics,
    _validate_balance_inputs,
    _validate_alpha_inputs,
    _group_seeds_by_label,
    _determine_target_size,
    _undersample_seeds,
    _oversample_seeds,
    _alpha_from_network_structure,
    _alpha_from_seed_ratio
)

from src.common.exceptions import ValidationError, ConfigurationError


class TestCreateBalancedSeedSet:
    """Test balanced seed set creation functionality."""
    
    def test_basic_undersampling(self):
        """Test basic undersampling to balance classes."""
        candidate_seeds = {
            "n1": "A", "n2": "A", "n3": "A", "n4": "A",  # 4 A's
            "n5": "B", "n6": "B"  # 2 B's
        }
        labels = ["A", "B"]
        
        balanced = create_balanced_seed_set(
            candidate_seeds, labels, method="undersample", random_seed=42
        )
        
        # Should have 2 of each label (size of smallest class)
        label_counts = {}
        for label in balanced.values():
            label_counts[label] = label_counts.get(label, 0) + 1
        
        assert label_counts["A"] == 2
        assert label_counts["B"] == 2
        assert len(balanced) == 4
        
        # All original B's should be included
        b_nodes = [node for node, label in balanced.items() if label == "B"]
        assert "n5" in b_nodes
        assert "n6" in b_nodes
    
    def test_basic_oversampling(self):
        """Test basic oversampling to reach target size."""
        candidate_seeds = {
            "n1": "A", "n2": "A", "n3": "A", "n4": "A",  # 4 A's
            "n5": "B", "n6": "B"  # 2 B's
        }
        labels = ["A", "B"]
        
        balanced = create_balanced_seed_set(
            candidate_seeds, labels, n_per_label=3, method="oversample", random_seed=42
        )
        
        # Should have 3 of each label
        label_counts = {}
        for label in balanced.values():
            label_counts[label] = label_counts.get(label, 0) + 1
        
        assert label_counts["A"] == 3
        assert label_counts["B"] == 3
        assert len(balanced) == 6
    
    def test_specified_target_size_undersample(self):
        """Test undersampling with specified target size."""
        candidate_seeds = {
            "n1": "A", "n2": "A", "n3": "A", "n4": "A",
            "n5": "B", "n6": "B", "n7": "B"
        }
        labels = ["A", "B"]
        
        balanced = create_balanced_seed_set(
            candidate_seeds, labels, n_per_label=2, method="undersample", random_seed=42
        )
        
        # Should have exactly 2 of each label
        label_counts = {}
        for label in balanced.values():
            label_counts[label] = label_counts.get(label, 0) + 1
        
        assert label_counts["A"] == 2
        assert label_counts["B"] == 2
        assert len(balanced) == 4
    
    def test_reproducible_sampling(self):
        """Test that random seed produces reproducible results."""
        candidate_seeds = {
            "n1": "A", "n2": "A", "n3": "A", "n4": "A",
            "n5": "B", "n6": "B"
        }
        labels = ["A", "B"]
        
        balanced1 = create_balanced_seed_set(
            candidate_seeds, labels, random_seed=42
        )
        balanced2 = create_balanced_seed_set(
            candidate_seeds, labels, random_seed=42
        )
        
        assert balanced1 == balanced2
    
    def test_three_label_balancing(self):
        """Test balancing with three labels."""
        candidate_seeds = {
            "n1": "A", "n2": "A", "n3": "A", "n4": "A", "n5": "A",  # 5 A's
            "n6": "B", "n7": "B", "n8": "B",  # 3 B's
            "n9": "C", "n10": "C"  # 2 C's
        }
        labels = ["A", "B", "C"]
        
        balanced = create_balanced_seed_set(
            candidate_seeds, labels, method="undersample", random_seed=42
        )
        
        # Should have 2 of each label (size of smallest class C)
        label_counts = {}
        for label in balanced.values():
            label_counts[label] = label_counts.get(label, 0) + 1
        
        assert label_counts["A"] == 2
        assert label_counts["B"] == 2
        assert label_counts["C"] == 2
        assert len(balanced) == 6
    
    def test_oversampling_with_duplicates(self):
        """Test that oversampling can handle duplicates correctly."""
        candidate_seeds = {
            "n1": "A", "n2": "A",
            "n3": "B"  # Only 1 B
        }
        labels = ["A", "B"]
        
        balanced = create_balanced_seed_set(
            candidate_seeds, labels, n_per_label=2, method="oversample", random_seed=42
        )
        
        # Should have 2 of each label
        label_counts = {}
        for label in balanced.values():
            label_counts[label] = label_counts.get(label, 0) + 1
        
        assert label_counts["A"] == 2
        assert label_counts["B"] == 2
        
        # The single B node might be duplicated (but dict will have same key)
        assert "n3" in balanced
        assert balanced["n3"] == "B"


class TestSuggestAlphaValue:
    """Test alpha value suggestion functionality."""
    
    def test_network_structure_method(self):
        """Test alpha suggestion based on network structure."""
        # Create a simple graph
        graph = nk.Graph(4, directed=False)
        graph.addEdge(0, 1)
        graph.addEdge(1, 2)
        graph.addEdge(2, 3)
        
        alpha = suggest_alpha_value(graph, seed_count=2, method="network_structure")
        
        # Should return a value between 0.1 and 0.99
        assert 0.1 <= alpha <= 0.99
        assert isinstance(alpha, float)
    
    def test_seed_ratio_method(self):
        """Test alpha suggestion based on seed ratio."""
        # Create a simple graph
        graph = nk.Graph(10, directed=False)
        for i in range(9):
            graph.addEdge(i, i+1)
        
        # Few seeds relative to network size
        alpha_few = suggest_alpha_value(graph, seed_count=2, method="seed_ratio")
        
        # Many seeds relative to network size
        alpha_many = suggest_alpha_value(graph, seed_count=8, method="seed_ratio")
        
        # Few seeds should suggest higher alpha (more propagation needed)
        # Many seeds should suggest lower alpha (seeds provide strong signal)
        assert alpha_few > alpha_many
        assert 0.1 <= alpha_few <= 0.99
        assert 0.1 <= alpha_many <= 0.99
    
    def test_alpha_clamping(self):
        """Test that alpha values are clamped to valid range."""
        # Create graph that might produce extreme values
        graph = nk.Graph(2, directed=False)
        graph.addEdge(0, 1)
        
        # Test with extreme seed ratios
        alpha_extreme = suggest_alpha_value(graph, seed_count=2, method="seed_ratio")
        
        assert 0.1 <= alpha_extreme <= 0.99
    
    def test_empty_graph_handling(self):
        """Test alpha suggestion for graph with no edges."""
        # Create graph with nodes but no edges
        graph = nk.Graph(5, directed=False)
        
        alpha = suggest_alpha_value(graph, seed_count=2, method="network_structure")
        
        # Should still return valid alpha
        assert 0.1 <= alpha <= 0.99
    
    @patch('src.glp.utils.nk.centrality.LocalClusteringCoefficient')
    def test_network_structure_exception_handling(self, mock_clustering):
        """Test exception handling in network structure method."""
        # Mock clustering coefficient to raise exception
        mock_clustering.side_effect = Exception("Network error")
        
        graph = nk.Graph(4, directed=False)
        graph.addEdge(0, 1)
        
        alpha = suggest_alpha_value(graph, seed_count=2, method="network_structure")
        
        # Should return default fallback value
        assert alpha == 0.85
    
    def test_directed_graph(self):
        """Test alpha suggestion with directed graph."""
        graph = nk.Graph(4, directed=True)
        graph.addEdge(0, 1)
        graph.addEdge(1, 2)
        graph.addEdge(2, 3)
        
        alpha = suggest_alpha_value(graph, seed_count=2, method="network_structure")
        
        assert 0.1 <= alpha <= 0.99


class TestGetSeedStatistics:
    """Test seed statistics analysis functionality."""
    
    def test_basic_statistics(self):
        """Test basic seed statistics calculation."""
        seed_labels = {
            "n1": "A", "n2": "A", "n3": "A",
            "n4": "B", "n5": "B"
        }
        labels = ["A", "B", "C"]
        
        stats = get_seed_statistics(seed_labels, labels)
        
        # Check basic structure
        expected_keys = ["label_counts", "total_seeds", "balance_ratio", "is_balanced", "recommendations"]
        assert all(key in stats for key in expected_keys)
        
        # Check counts
        assert stats["label_counts"]["A"] == 3
        assert stats["label_counts"]["B"] == 2
        assert stats["label_counts"]["C"] == 0
        assert stats["total_seeds"] == 5
        
        # Check balance ratio (min/max = 2/3)
        assert stats["balance_ratio"] == pytest.approx(2/3, abs=0.01)
        
        # Should be reasonably balanced (ratio > 0.5)
        assert stats["is_balanced"] == True
    
    def test_imbalanced_seeds(self):
        """Test statistics for imbalanced seed set."""
        seed_labels = {
            "n1": "A", "n2": "A", "n3": "A", "n4": "A", "n5": "A",  # 5 A's
            "n6": "B"  # 1 B
        }
        labels = ["A", "B"]
        
        stats = get_seed_statistics(seed_labels, labels)
        
        # Check balance ratio (1/5 = 0.2)
        assert stats["balance_ratio"] == pytest.approx(0.2, abs=0.01)
        
        # Should not be balanced
        assert stats["is_balanced"] == False
        
        # Should recommend balancing
        recommendations = stats["recommendations"]
        assert any("create_balanced_seed_set" in rec for rec in recommendations)
    
    def test_missing_labels(self):
        """Test statistics when some labels have no seeds."""
        seed_labels = {
            "n1": "A", "n2": "A",
            "n3": "B"
        }
        labels = ["A", "B", "C"]  # C has no seeds
        
        stats = get_seed_statistics(seed_labels, labels)
        
        # Check counts
        assert stats["label_counts"]["A"] == 2
        assert stats["label_counts"]["B"] == 1
        assert stats["label_counts"]["C"] == 0
        
        # Balance ratio should be 0.5 (1/2, ignoring 0 counts)
        assert stats["balance_ratio"] == pytest.approx(0.5, abs=0.01)
        assert stats["is_balanced"] == True  # 0.5 >= 0.5 threshold
        
        # Should recommend adding missing labels
        recommendations = stats["recommendations"]
        assert any("missing labels" in rec for rec in recommendations)
        assert any("C" in rec for rec in recommendations)
    
    def test_empty_seed_set(self):
        """Test statistics for empty seed set."""
        seed_labels = {}
        labels = ["A", "B"]
        
        stats = get_seed_statistics(seed_labels, labels)
        
        assert stats["total_seeds"] == 0
        assert stats["balance_ratio"] == 0.0
        assert stats["is_balanced"] == False
        
        recommendations = stats["recommendations"]
        assert any("empty" in rec for rec in recommendations)
    
    def test_well_balanced_seeds(self):
        """Test statistics for well-balanced seed set."""
        seed_labels = {
            "n1": "A", "n2": "A", "n3": "A",
            "n4": "B", "n5": "B", "n6": "B",
            "n7": "C", "n8": "C", "n9": "C"
        }
        labels = ["A", "B", "C"]
        
        stats = get_seed_statistics(seed_labels, labels)
        
        # Perfect balance
        assert stats["balance_ratio"] == 1.0
        assert stats["is_balanced"] == True
        
        # Should indicate good balance
        recommendations = stats["recommendations"]
        assert any("well-balanced" in rec for rec in recommendations)
    
    def test_few_seeds_per_label(self):
        """Test recommendation for few seeds per label."""
        seed_labels = {
            "n1": "A",
            "n2": "B"
        }
        labels = ["A", "B"]
        
        stats = get_seed_statistics(seed_labels, labels)
        
        # Should recommend more seeds
        recommendations = stats["recommendations"]
        assert any("more seeds" in rec for rec in recommendations)


class TestInputValidation:
    """Test input validation for utils functions."""
    
    def test_validate_balance_inputs_empty_candidates(self):
        """Test validation with empty candidate seeds."""
        with pytest.raises(ValidationError, match="candidate_seeds cannot be empty"):
            _validate_balance_inputs({}, ["A", "B"], None, "undersample")
    
    def test_validate_balance_inputs_empty_labels(self):
        """Test validation with empty labels list."""
        with pytest.raises(ValidationError, match="labels list cannot be empty"):
            _validate_balance_inputs({"n1": "A"}, [], None, "undersample")
    
    def test_validate_balance_inputs_invalid_method(self):
        """Test validation with invalid method."""
        with pytest.raises(ConfigurationError, match="method must be"):
            _validate_balance_inputs({"n1": "A"}, ["A"], None, "invalid_method")
    
    def test_validate_balance_inputs_invalid_n_per_label(self):
        """Test validation with invalid n_per_label."""
        with pytest.raises(ConfigurationError, match="n_per_label must be positive"):
            _validate_balance_inputs({"n1": "A"}, ["A"], -1, "undersample")
    
    def test_validate_balance_inputs_missing_labels(self):
        """Test validation when labels not found in seeds."""
        candidate_seeds = {"n1": "A", "n2": "B"}
        labels = ["A", "B", "C"]  # C not in seeds
        
        with pytest.raises(ValidationError, match="Labels not found in candidate seeds"):
            _validate_balance_inputs(candidate_seeds, labels, None, "undersample")
    
    def test_validate_alpha_inputs_empty_graph(self):
        """Test alpha validation with empty graph."""
        empty_graph = nk.Graph(0)
        
        with pytest.raises(ValidationError, match="graph cannot be empty"):
            _validate_alpha_inputs(empty_graph, 1, "network_structure")
    
    def test_validate_alpha_inputs_invalid_seed_count(self):
        """Test alpha validation with invalid seed count."""
        graph = nk.Graph(5)
        
        with pytest.raises(ValidationError, match="seed_count must be positive"):
            _validate_alpha_inputs(graph, 0, "network_structure")
    
    def test_validate_alpha_inputs_too_many_seeds(self):
        """Test alpha validation with too many seeds."""
        graph = nk.Graph(3)
        
        with pytest.raises(ValidationError, match="seed_count .* cannot exceed graph size"):
            _validate_alpha_inputs(graph, 5, "network_structure")
    
    def test_validate_alpha_inputs_invalid_method(self):
        """Test alpha validation with invalid method."""
        graph = nk.Graph(5)
        
        with pytest.raises(ConfigurationError, match="method must be"):
            _validate_alpha_inputs(graph, 2, "invalid_method")


class TestHelperFunctions:
    """Test individual helper functions."""
    
    def test_group_seeds_by_label(self):
        """Test seed grouping by label."""
        candidate_seeds = {
            "n1": "A", "n2": "A", "n3": "B", "n4": "A", "n5": "B"
        }
        labels = ["A", "B", "C"]
        
        grouped = _group_seeds_by_label(candidate_seeds, labels)
        
        assert set(grouped["A"]) == {"n1", "n2", "n4"}
        assert set(grouped["B"]) == {"n3", "n5"}
        assert grouped["C"] == []
    
    def test_determine_target_size_natural_undersampling(self):
        """Test target size determination for natural undersampling."""
        seeds_by_label = {
            "A": ["n1", "n2", "n3"],  # 3 seeds
            "B": ["n4", "n5"]         # 2 seeds
        }
        
        target_size = _determine_target_size(seeds_by_label, None, "undersample")
        assert target_size == 2  # Size of smallest class
    
    def test_determine_target_size_specified(self):
        """Test target size determination with specified value."""
        seeds_by_label = {
            "A": ["n1", "n2", "n3"],
            "B": ["n4", "n5"]
        }
        
        target_size = _determine_target_size(seeds_by_label, 2, "undersample")
        assert target_size == 2
    
    def test_determine_target_size_undersample_validation(self):
        """Test target size validation for undersampling."""
        seeds_by_label = {
            "A": ["n1", "n2", "n3"],
            "B": ["n4", "n5"]  # Only 2 seeds
        }
        
        # Requesting 3 per label but B only has 2
        with pytest.raises(ValidationError, match="Cannot undersample to 3"):
            _determine_target_size(seeds_by_label, 3, "undersample")
    
    def test_determine_target_size_zero_seeds(self):
        """Test target size with label having zero seeds."""
        seeds_by_label = {
            "A": ["n1", "n2"],
            "B": []  # No seeds
        }
        
        with pytest.raises(ValidationError, match="At least one label has no seeds"):
            _determine_target_size(seeds_by_label, None, "undersample")
    
    def test_undersample_seeds(self):
        """Test undersampling implementation."""
        seeds_by_label = {
            "A": ["n1", "n2", "n3", "n4"],  # 4 seeds
            "B": ["n5", "n6"]               # 2 seeds
        }
        target_size = 2
        
        # Set random seed for reproducibility
        random.seed(42)
        balanced = _undersample_seeds(seeds_by_label, target_size)
        
        # Count by label
        label_counts = {}
        for label in balanced.values():
            label_counts[label] = label_counts.get(label, 0) + 1
        
        assert label_counts["A"] == 2
        assert label_counts["B"] == 2
        assert len(balanced) == 4
    
    def test_oversample_seeds(self):
        """Test oversampling implementation."""
        seeds_by_label = {
            "A": ["n1", "n2"],  # 2 seeds
            "B": ["n3"]         # 1 seed
        }
        target_size = 2
        
        # Set random seed for reproducibility
        random.seed(42)
        balanced = _oversample_seeds(seeds_by_label, target_size)
        
        # Count by label
        label_counts = {}
        for label in balanced.values():
            label_counts[label] = label_counts.get(label, 0) + 1
        
        assert label_counts["A"] == 2
        assert label_counts["B"] == 2
        
        # B should be oversampled (n3 might appear multiple times in concept,
        # but dict will only store one entry per key)
        assert "n3" in balanced
        assert balanced["n3"] == "B"
    
    def test_alpha_from_network_structure_no_edges(self):
        """Test alpha calculation for graph with no edges."""
        graph = nk.Graph(5, directed=False)  # No edges added
        
        alpha = _alpha_from_network_structure(graph)
        
        # Should return reasonable alpha for zero clustering
        assert 0.1 <= alpha <= 0.99
    
    def test_alpha_from_network_structure_high_clustering(self):
        """Test alpha calculation for highly clustered graph."""
        # Create a triangle (high clustering)
        graph = nk.Graph(3, directed=False)
        graph.addEdge(0, 1)
        graph.addEdge(1, 2)
        graph.addEdge(2, 0)
        
        alpha = _alpha_from_network_structure(graph)
        
        # High clustering should lead to lower alpha
        assert 0.1 <= alpha <= 0.99
    
    def test_alpha_from_seed_ratio_extremes(self):
        """Test alpha calculation for extreme seed ratios."""
        graph = nk.Graph(10, directed=False)
        
        # Very few seeds
        alpha_few = _alpha_from_seed_ratio(graph, 1)
        
        # Many seeds
        alpha_many = _alpha_from_seed_ratio(graph, 9)
        
        # Few seeds should suggest higher alpha
        assert alpha_few > alpha_many
        assert 0.1 <= alpha_few <= 0.99
        assert 0.1 <= alpha_many <= 0.99


class TestCreateBalancedSeedSetInputValidation:
    """Test input validation for create_balanced_seed_set."""
    
    def test_empty_candidate_seeds(self):
        """Test with empty candidate seeds."""
        with pytest.raises(ValidationError):
            create_balanced_seed_set({}, ["A", "B"])
    
    def test_empty_labels(self):
        """Test with empty labels list."""
        with pytest.raises(ValidationError):
            create_balanced_seed_set({"n1": "A"}, [])
    
    def test_invalid_method(self):
        """Test with invalid balancing method."""
        with pytest.raises(ConfigurationError):
            create_balanced_seed_set({"n1": "A"}, ["A"], method="invalid")
    
    def test_negative_n_per_label(self):
        """Test with negative n_per_label."""
        with pytest.raises(ConfigurationError):
            create_balanced_seed_set({"n1": "A"}, ["A"], n_per_label=-1)


class TestSuggestAlphaValueInputValidation:
    """Test input validation for suggest_alpha_value."""
    
    def test_empty_graph(self):
        """Test with empty graph."""
        empty_graph = nk.Graph(0)
        with pytest.raises(ValidationError):
            suggest_alpha_value(empty_graph, 1)
    
    def test_zero_seed_count(self):
        """Test with zero seed count."""
        graph = nk.Graph(5)
        with pytest.raises(ValidationError):
            suggest_alpha_value(graph, 0)
    
    def test_too_many_seeds(self):
        """Test with more seeds than nodes."""
        graph = nk.Graph(3)
        with pytest.raises(ValidationError):
            suggest_alpha_value(graph, 5)
    
    def test_invalid_method(self):
        """Test with invalid method."""
        graph = nk.Graph(5)
        with pytest.raises(ConfigurationError):
            suggest_alpha_value(graph, 2, method="invalid")