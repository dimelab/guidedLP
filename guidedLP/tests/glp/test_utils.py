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

from guidedLP.glp.utils import (
    create_balanced_seed_set,
    suggest_alpha_value,
    get_seed_statistics,
    check_seed_coverage,
    make_stat_user_edges,
    _validate_balance_inputs,
    _validate_alpha_inputs,
    _group_seeds_by_label,
    _determine_target_size,
    _undersample_seeds,
    _oversample_seeds,
    _alpha_from_network_structure,
    _alpha_from_seed_ratio
)

from guidedLP.common.id_mapper import IDMapper
from guidedLP.common.exceptions import ValidationError, ConfigurationError

import polars as pl


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
    
    @patch('guidedLP.glp.utils.nk.centrality.LocalClusteringCoefficient')
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


class TestCheckSeedCoverage:
    """Test the check_seed_coverage diagnostic."""

    def _make_mapper(self, originals):
        m = IDMapper()
        for i, orig in enumerate(originals):
            m.add_mapping(orig, i)
        return m

    def test_train_only_all_present(self):
        mapper = self._make_mapper(["u1", "u2", "u3", "u4"])
        seeds = {"u1": "A", "u2": "B", "u3": "A"}
        report = check_seed_coverage(mapper, seeds)
        assert "train" in report
        assert "test" not in report
        assert "overlap" not in report

        train = report["train"]
        assert train["total"] == 3
        assert train["present"] == 3
        assert train["missing"] == 0
        assert train["coverage"] == 1.0
        assert train["missing_sample"] == []
        assert train["by_label"]["A"] == {
            "total": 2, "present": 2, "missing": 0, "coverage": 1.0,
        }
        assert train["by_label"]["B"] == {
            "total": 1, "present": 1, "missing": 0, "coverage": 1.0,
        }

    def test_train_some_missing(self):
        mapper = self._make_mapper(["u1", "u3"])
        seeds = {"u1": "A", "u2": "A", "u3": "B", "u4": "B"}
        report = check_seed_coverage(mapper, seeds)
        train = report["train"]
        assert train["total"] == 4
        assert train["present"] == 2
        assert train["missing"] == 2
        assert train["coverage"] == 0.5
        assert set(train["missing_sample"]) == {"u2", "u4"}
        assert train["by_label"]["A"]["coverage"] == 0.5
        assert train["by_label"]["B"]["coverage"] == 0.5

    def test_empty_seeds_does_not_divide_by_zero(self):
        mapper = self._make_mapper(["u1", "u2"])
        report = check_seed_coverage(mapper, {})
        assert report["train"] == {
            "total": 0, "present": 0, "missing": 0, "coverage": 0.0,
            "by_label": {}, "missing_sample": [], "skipped_null_labels": 0,
        }

    def test_test_set_separately_reported(self):
        mapper = self._make_mapper(["u1", "u2", "u3", "u4"])
        train = {"u1": "A", "u2": "B"}
        test = {"u3": "A", "u4": "B", "u5": "A"}  # u5 missing
        report = check_seed_coverage(mapper, train, test_seeds=test)
        assert report["train"]["coverage"] == 1.0
        assert report["test"]["total"] == 3
        assert report["test"]["present"] == 2
        assert report["test"]["missing"] == 1
        assert report["test"]["missing_sample"] == ["u5"]

    def test_overlap_detection_no_conflict(self):
        mapper = self._make_mapper(["u1", "u2", "u3"])
        train = {"u1": "A", "u2": "B", "u3": "A"}
        test = {"u2": "B", "u3": "A"}  # both overlap with same labels
        report = check_seed_coverage(mapper, train, test_seeds=test)
        assert report["overlap"]["count"] == 2
        assert report["overlap"]["conflicting"] == 0

    def test_overlap_detection_with_conflict(self):
        mapper = self._make_mapper(["u1", "u2", "u3"])
        train = {"u1": "A", "u2": "A"}
        test = {"u1": "B", "u2": "A"}  # u1 has conflicting label
        report = check_seed_coverage(mapper, train, test_seeds=test)
        assert report["overlap"]["count"] == 2
        assert report["overlap"]["conflicting"] == 1
        # Sample should include the conflicting node with both labels
        sample_nodes = [t[0] for t in report["overlap"]["sample"]]
        assert "u1" in sample_nodes

    def test_accepts_dataframe_input(self):
        mapper = self._make_mapper(["u1", "u2", "u3"])
        train_df = pl.DataFrame({"node_id": ["u1", "u2"], "label": ["A", "B"]})
        test_df = pl.DataFrame({"node_id": ["u3", "missing"], "label": ["A", "B"]})
        report = check_seed_coverage(mapper, train_df, test_seeds=test_df)
        assert report["train"]["coverage"] == 1.0
        assert report["test"]["missing"] == 1

    def test_accepts_inverse_dict(self):
        mapper = self._make_mapper(["u1", "u2"])
        seeds_inverse = {"A": ["u1"], "B": ["u2", "u3"]}  # u3 missing
        report = check_seed_coverage(mapper, seeds_inverse)
        assert report["train"]["total"] == 3
        assert report["train"]["missing"] == 1

    def test_missing_sample_size_cap(self):
        mapper = self._make_mapper(["u1"])
        seeds = {f"missing_{i}": "A" for i in range(20)}
        report = check_seed_coverage(mapper, seeds, missing_sample_size=5)
        assert len(report["train"]["missing_sample"]) == 5

    def test_missing_sample_size_zero_disables_sampling(self):
        mapper = self._make_mapper(["u1"])
        seeds = {f"missing_{i}": "A" for i in range(5)}
        report = check_seed_coverage(mapper, seeds, missing_sample_size=0)
        assert report["train"]["missing_sample"] == []

    def test_null_labels_in_dict_are_skipped(self):
        mapper = self._make_mapper(["u1", "u2", "u3"])
        # u3 has no known label yet → None should be silently dropped.
        seeds = {"u1": "A", "u2": "B", "u3": None}
        report = check_seed_coverage(mapper, seeds)
        assert report["train"]["total"] == 2
        assert report["train"]["present"] == 2
        assert report["train"]["skipped_null_labels"] == 1
        assert set(report["train"]["by_label"]) == {"A", "B"}

    def test_null_labels_in_polars(self):
        mapper = self._make_mapper(["u1", "u2", "u3", "u4"])
        df = pl.DataFrame({
            "node_id": ["u1", "u2", "u3", "u4"],
            "label":   ["A",  None, "B", None],
        })
        report = check_seed_coverage(mapper, df)
        assert report["train"]["total"] == 2
        assert report["train"]["skipped_null_labels"] == 2
        assert report["train"]["coverage"] == 1.0

    def test_null_labels_in_pandas(self):
        pd = pytest.importorskip("pandas")
        mapper = self._make_mapper(["u1", "u2", "u3"])
        df = pd.DataFrame({
            "node_id": ["u1", "u2", "u3"],
            "label":   ["A", None, "B"],
        })
        report = check_seed_coverage(mapper, df)
        assert report["train"]["total"] == 2
        assert report["train"]["skipped_null_labels"] == 1

    def test_null_labels_counted_per_side(self):
        mapper = self._make_mapper(["u1", "u2", "u3", "u4"])
        train = {"u1": "A", "u2": None, "u3": "B"}                    # 1 null
        test  = {"u4": "A", "u5": None, "u6": None, "u7": "B"}        # 2 nulls
        report = check_seed_coverage(mapper, train, test_seeds=test)
        assert report["train"]["skipped_null_labels"] == 1
        assert report["test"]["skipped_null_labels"] == 2

    def test_null_label_in_inverse_dict_drops_whole_group(self):
        """An inverse-dict entry with key=None should drop its whole node list."""
        mapper = self._make_mapper(["u1", "u2", "u3"])
        seeds = {"A": ["u1"], None: ["u2", "u3"]}  # the None-keyed group → all dropped
        report = check_seed_coverage(mapper, seeds)
        assert report["train"]["total"] == 1
        assert report["train"]["skipped_null_labels"] == 2


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


class TestMakeStatUserEdges:
    """Tests for the label-as-node augmentation function."""

    def _engagement(self):
        return pl.DataFrame({
            "user": ["u1", "u1", "u2", "u3"],
            "content": ["c1", "c2", "c1", "c2"],
            "weight": [1.0, 2.0, 1.0, 3.0],
        })

    def test_basic_happy_path(self):
        """Generates one edge per (user, seed_content) pair with engagement weights."""
        engagement = self._engagement()
        edges, seeds, ids = make_stat_user_edges(
            engagement, {"c1": "left", "c2": "right"}
        )

        # Schema matches project_bipartite output
        assert edges.columns == ["source_id", "target_id", "weight"]
        assert len(edges) == 4

        # Stat IDs returned
        assert ids == {"__stat__c1", "__stat__c2"}

        # Seed dict has labels for both stat users
        assert seeds == {"__stat__c1": "left", "__stat__c2": "right"}

        # Edge weights match input engagement (no dedup needed here)
        rows = {
            (r["source_id"], r["target_id"]): r["weight"]
            for r in edges.iter_rows(named=True)
        }
        assert rows[("__stat__c1", "u1")] == 1.0
        assert rows[("__stat__c1", "u2")] == 1.0
        assert rows[("__stat__c2", "u1")] == 2.0
        assert rows[("__stat__c2", "u3")] == 3.0

    def test_duplicates_are_summed(self):
        """Duplicate (user, content) rows aggregate by summing weights."""
        engagement = pl.DataFrame({
            "user": ["u1", "u1", "u1", "u2"],
            "content": ["c1", "c1", "c1", "c1"],
            "weight": [1.0, 2.0, 3.0, 5.0],
        })
        edges, _, _ = make_stat_user_edges(engagement, {"c1": "A"})

        rows = {r["target_id"]: r["weight"] for r in edges.iter_rows(named=True)}
        assert rows["u1"] == 6.0  # 1 + 2 + 3
        assert rows["u2"] == 5.0

    def test_unweighted_input(self):
        """weight_col=None falls back to unit weights, still aggregates."""
        engagement = pl.DataFrame({
            "user": ["u1", "u1", "u2"],
            "content": ["c1", "c1", "c1"],
        })
        edges, _, _ = make_stat_user_edges(
            engagement, {"c1": "A"}, weight_col=None
        )

        rows = {r["target_id"]: r["weight"] for r in edges.iter_rows(named=True)}
        assert rows["u1"] == 2.0  # 1 + 1 (the two dup rows)
        assert rows["u2"] == 1.0

    def test_unmatched_seed_warns_and_is_dropped(self):
        """Seeds with no engagements warn but don't break the call."""
        engagement = pl.DataFrame({
            "user": ["u1", "u2"],
            "content": ["c1", "c1"],
            "weight": [1.0, 1.0],
        })
        with pytest.warns(UserWarning, match="have no engagements"):
            edges, seeds, ids = make_stat_user_edges(
                engagement, {"c1": "A", "c_missing": "B"}
            )

        # Only c1 contributes
        assert seeds == {"__stat__c1": "A"}
        assert ids == {"__stat__c1"}
        assert len(edges) == 2

    def test_all_seeds_unmatched_raises(self):
        """If no seed has any engagement, that's a hard error, not a warning."""
        engagement = pl.DataFrame({
            "user": ["u1"],
            "content": ["c_real"],
            "weight": [1.0],
        })
        import warnings as _warnings
        with _warnings.catch_warnings():
            _warnings.simplefilter("ignore")
            with pytest.raises(ValidationError, match="None of the seed nodes"):
                make_stat_user_edges(engagement, {"c_missing": "A"})

    def test_id_collision_detected(self):
        """Stat IDs colliding with real user IDs raise immediately."""
        engagement = pl.DataFrame({
            "user": ["u1", "__stat__c1"],  # second user collides
            "content": ["c1", "c1"],
            "weight": [1.0, 1.0],
        })
        with pytest.raises(ValidationError, match="collide"):
            make_stat_user_edges(engagement, {"c1": "A"})

    def test_custom_prefix(self):
        """stat_prefix is configurable."""
        engagement = self._engagement()
        edges, seeds, ids = make_stat_user_edges(
            engagement,
            {"c1": "A"},
            stat_prefix="LBL_",
        )
        assert ids == {"LBL_c1"}
        assert seeds == {"LBL_c1": "A"}
        sources = set(edges["source_id"].to_list())
        assert sources == {"LBL_c1"}

    def test_custom_column_names(self):
        """user_col / content_col / weight_col can all be overridden."""
        engagement = pl.DataFrame({
            "uid": ["u1", "u2"],
            "post": ["p1", "p1"],
            "score": [4.0, 5.0],
        })
        edges, seeds, _ = make_stat_user_edges(
            engagement,
            {"p1": "A"},
            user_col="uid",
            content_col="post",
            weight_col="score",
        )
        rows = {r["target_id"]: r["weight"] for r in edges.iter_rows(named=True)}
        assert rows == {"u1": 4.0, "u2": 5.0}
        assert seeds == {"__stat__p1": "A"}

    def test_empty_seeds_raises(self):
        """Empty seed input is a configuration error."""
        engagement = self._engagement()
        with pytest.raises(ValidationError, match="empty"):
            make_stat_user_edges(engagement, {})

    def test_missing_column_raises(self):
        """Misnamed user_col / content_col raise with a helpful message."""
        engagement = self._engagement()
        with pytest.raises(ValidationError, match="missing required"):
            make_stat_user_edges(
                engagement, {"c1": "A"}, user_col="nonexistent"
            )

    def test_missing_weight_column_raises(self):
        """Misnamed weight_col raises (vs. silently treating as unweighted)."""
        engagement = self._engagement()
        with pytest.raises(ValidationError, match="weight_col"):
            make_stat_user_edges(
                engagement, {"c1": "A"}, weight_col="not_there"
            )

    def test_label_keyed_seed_input(self):
        """Seed input shape {label: [nodes]} works (via normalize_seed_input)."""
        engagement = self._engagement()
        edges, seeds, _ = make_stat_user_edges(
            engagement, {"left": ["c1"], "right": ["c2"]}
        )
        assert seeds == {"__stat__c1": "left", "__stat__c2": "right"}
        assert len(edges) == 4

    def test_dataframe_seed_input(self):
        """Seed input as a polars DataFrame works."""
        engagement = self._engagement()
        seed_df = pl.DataFrame({
            "node_id": ["c1", "c2"],
            "label": ["A", "B"],
        })
        edges, seeds, _ = make_stat_user_edges(engagement, seed_df)
        assert seeds == {"__stat__c1": "A", "__stat__c2": "B"}
        assert len(edges) == 4

    def test_invalid_engagement_type_raises(self):
        """Non-DataFrame engagement input raises a clear error."""
        with pytest.raises(ValidationError, match="DataFrame"):
            make_stat_user_edges([("u1", "c1", 1.0)], {"c1": "A"})

    def test_integer_content_ids(self):
        """Integer content IDs are cast to string in the resulting stat IDs."""
        engagement = pl.DataFrame({
            "user": ["u1", "u2"],
            "content": [42, 42],
            "weight": [1.0, 1.0],
        })
        edges, seeds, ids = make_stat_user_edges(engagement, {42: "A"})
        assert ids == {"__stat__42"}
        assert seeds == {"__stat__42": "A"}
        # The synthesized source_id column is Utf8.
        assert edges["source_id"].dtype == pl.Utf8


class TestExcludeFromOutput:
    """Tests for the exclude_from_output parameter on GLP and ensemble."""

    def _augmented_path_graph(self):
        """
        Build a 5-node graph: u1 - u2 - u3 - u4 plus a stat-user attached to
        {u1, u2}. Returns (graph, mapper).
        """
        g = nk.Graph(5, weighted=True, directed=False)
        g.addEdge(0, 1, 1.0)
        g.addEdge(1, 2, 1.0)
        g.addEdge(2, 3, 1.0)
        g.addEdge(4, 0, 1.0)
        g.addEdge(4, 1, 1.0)
        mapper = IDMapper.from_originals(
            ["u1", "u2", "u3", "u4", "__stat__c1"]
        )
        return g, mapper

    def test_glp_drops_excluded_nodes(self):
        """guided_label_propagation hides excluded nodes from output."""
        from guidedLP.glp import guided_label_propagation

        graph, mapper = self._augmented_path_graph()
        result = guided_label_propagation(
            graph, mapper,
            {"__stat__c1": "A"},
            ["A"],
            enable_noise_category=True,
            exclude_from_output={"__stat__c1"},
        )
        assert "__stat__c1" not in result["node_id"].to_list()
        assert set(result["node_id"].to_list()) == {"u1", "u2", "u3", "u4"}

    def test_glp_no_exclude_is_no_op(self):
        """None / empty set leaves output unchanged."""
        from guidedLP.glp import guided_label_propagation

        graph, mapper = self._augmented_path_graph()
        r_none = guided_label_propagation(
            graph, mapper, {"__stat__c1": "A"}, ["A"],
            enable_noise_category=True, exclude_from_output=None,
        )
        r_empty = guided_label_propagation(
            graph, mapper, {"__stat__c1": "A"}, ["A"],
            enable_noise_category=True, exclude_from_output=set(),
        )
        assert len(r_none) == 5
        assert len(r_empty) == 5

    def test_glp_propagation_unaffected_by_exclude(self):
        """
        Excluded nodes still participate in propagation; non-excluded rows
        have identical probabilities with vs. without the filter.
        """
        from guidedLP.glp import guided_label_propagation

        graph, mapper = self._augmented_path_graph()
        # Disable noise (which uses RNG) for a strict equality comparison.
        r_full = guided_label_propagation(
            graph, mapper, {"__stat__c1": "A"}, ["A"], enable_noise_category=False,
        )
        r_filtered = guided_label_propagation(
            graph, mapper, {"__stat__c1": "A"}, ["A"], enable_noise_category=False,
            exclude_from_output={"__stat__c1"},
        )

        full_users = r_full.filter(pl.col("node_id") != "__stat__c1").sort("node_id")
        filtered_users = r_filtered.sort("node_id")
        assert full_users["A_prob"].to_list() == filtered_users["A_prob"].to_list()

    def test_glp_unknown_id_silently_ignored(self):
        """exclude_from_output containing unknown IDs is fine — just no-op for them."""
        from guidedLP.glp import guided_label_propagation

        graph, mapper = self._augmented_path_graph()
        result = guided_label_propagation(
            graph, mapper, {"__stat__c1": "A"}, ["A"],
            enable_noise_category=False,
            exclude_from_output={"__stat__c1", "never_existed"},
        )
        # Only the real stat user is dropped; missing id is silently ignored.
        assert len(result) == 4

    def test_glp_directional_filters_both_passes(self):
        """For directed graphs, exclude_from_output applies to both DFs."""
        from guidedLP.glp import guided_label_propagation

        g = nk.Graph(5, weighted=True, directed=True)
        g.addEdge(0, 1, 1.0); g.addEdge(1, 2, 1.0); g.addEdge(2, 3, 1.0)
        g.addEdge(4, 0, 1.0); g.addEdge(4, 1, 1.0)
        mapper = IDMapper.from_originals(
            ["u1", "u2", "u3", "u4", "__stat__c1"]
        )

        out_df, in_df = guided_label_propagation(
            g, mapper, {"__stat__c1": "A"}, ["A"],
            directional=True, enable_noise_category=False,
            exclude_from_output={"__stat__c1"},
        )
        assert "__stat__c1" not in out_df["node_id"].to_list()
        assert "__stat__c1" not in in_df["node_id"].to_list()

    def test_glp_combined_with_confidence_threshold(self):
        """exclude_from_output is applied AFTER confidence threshold."""
        from guidedLP.glp import guided_label_propagation

        graph, mapper = self._augmented_path_graph()
        result = guided_label_propagation(
            graph, mapper, {"__stat__c1": "A"}, ["A"],
            enable_noise_category=True, noise_ratio=0.5,
            confidence_threshold=0.9,
            exclude_from_output={"__stat__c1"},
        )
        # Stat user filtered; remaining rows may include "uncertain" labels.
        assert "__stat__c1" not in result["node_id"].to_list()
        assert len(result) == 4

    def test_ensemble_drops_excluded_nodes(self):
        """ensemble_label_propagation honors exclude_from_output via glp_kwargs."""
        from guidedLP.glp import ensemble_label_propagation

        graph, mapper = self._augmented_path_graph()
        result = ensemble_label_propagation(
            graph, mapper, {"__stat__c1": "A"}, ["A"],
            n_epochs=3, enable_noise_category=True, noise_ratio=0.5,
            exclude_from_output={"__stat__c1"},
        )
        assert "__stat__c1" not in result["node_id"].to_list()
        assert len(result) == 4

    def test_ensemble_with_variance_and_exclude(self):
        """exclude_from_output works alongside return_variance=True."""
        from guidedLP.glp import ensemble_label_propagation

        graph, mapper = self._augmented_path_graph()
        result = ensemble_label_propagation(
            graph, mapper, {"__stat__c1": "A"}, ["A"],
            n_epochs=3, enable_noise_category=True, noise_ratio=0.5,
            return_variance=True,
            exclude_from_output={"__stat__c1"},
        )
        assert "__stat__c1" not in result["node_id"].to_list()
        # std columns present and aligned (4 surviving rows)
        assert "A_prob_std" in result.columns
        assert len(result) == 4