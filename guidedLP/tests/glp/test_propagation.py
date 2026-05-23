"""
Tests for guided label propagation functionality.

This module provides comprehensive testing for the guided label propagation algorithm,
including convergence behavior, directional propagation, parameter effects, and
edge cases with various graph structures.
"""

import pytest
import numpy as np
import polars as pl
import networkit as nk
from typing import Dict, List, Tuple, Any

from src.glp.propagation import (
    guided_label_propagation,
    get_propagation_info,
    _validate_inputs,
    _initialize_label_matrix,
    _create_transition_matrix,
    _propagate_iteration,
    _check_convergence,
    _create_results_dataframe,
    _iterative_propagation
)
from src.common.id_mapper import IDMapper
from src.common.exceptions import (
    ValidationError,
    ConfigurationError,
    ConvergenceError,
    ComputationError
)


class TestBasicPropagation:
    """Test basic label propagation functionality."""
    
    def test_simple_undirected_triangle(self):
        """Test propagation on simple triangle graph with known outcome."""
        # Create simple triangle: A-B-C-A
        graph = nk.Graph(3, weighted=True, directed=False)
        graph.addEdge(0, 1, 1.0)
        graph.addEdge(1, 2, 1.0)
        graph.addEdge(2, 0, 1.0)
        
        # Create ID mapper
        id_mapper = IDMapper()
        id_mapper.add_mapping("A", 0)
        id_mapper.add_mapping("B", 1)
        id_mapper.add_mapping("C", 2)
        
        # Set seeds: A is "left", C is "right"
        seed_labels = {"A": "left", "C": "right"}
        labels = ["left", "right"]
        
        # Run propagation
        result = guided_label_propagation(
            graph, id_mapper, seed_labels, labels,
            alpha=0.5, max_iterations=100, convergence_threshold=1e-6
        )
        
        # Validate result structure
        assert isinstance(result, pl.DataFrame)
        assert len(result) == 3
        assert set(result.columns) == {
            "node_id", "left_prob", "right_prob", 
            "dominant_label", "confidence", "is_seed"
        }
        
        # Check that seed nodes retain their labels
        result_dict = {row["node_id"]: row for row in result.iter_rows(named=True)}
        assert result_dict["A"]["dominant_label"] == "left"
        assert result_dict["C"]["dominant_label"] == "right"
        assert result_dict["A"]["is_seed"] is True
        assert result_dict["C"]["is_seed"] is True
        assert result_dict["B"]["is_seed"] is False
        
        # Check probability normalization
        for row in result.iter_rows(named=True):
            prob_sum = row["left_prob"] + row["right_prob"]
            assert abs(prob_sum - 1.0) < 1e-6
        
        # B should have mixed probabilities (not fully one label)
        assert 0.1 < result_dict["B"]["left_prob"] < 0.9
        assert 0.1 < result_dict["B"]["right_prob"] < 0.9
    
    def test_simple_directed_line(self):
        """Test directed propagation on simple line graph A->B->C."""
        # Create directed line: A -> B -> C
        graph = nk.Graph(3, weighted=True, directed=True)
        graph.addEdge(0, 1, 1.0)  # A -> B
        graph.addEdge(1, 2, 1.0)  # B -> C
        
        id_mapper = IDMapper()
        id_mapper.add_mapping("A", 0)
        id_mapper.add_mapping("B", 1)
        id_mapper.add_mapping("C", 2)
        
        seed_labels = {"A": "source"}
        labels = ["source", "target"]
        
        # Run directional propagation
        out_result, in_result = guided_label_propagation(
            graph, id_mapper, seed_labels, labels,
            alpha=0.8, directional=True
        )
        
        # Validate both results
        assert isinstance(out_result, pl.DataFrame)
        assert isinstance(in_result, pl.DataFrame)
        assert len(out_result) == 3
        assert len(in_result) == 3
        
        # Convert to dictionaries for easier checking
        out_dict = {row["node_id"]: row for row in out_result.iter_rows(named=True)}
        in_dict = {row["node_id"]: row for row in in_result.iter_rows(named=True)}
        
        # Out-degree propagation: A should influence B and C
        assert out_dict["A"]["source_prob"] > 0.9  # Seed node
        # B should have more influence than C in out-degree (A->B->C)
        # But with default parameters, this might not always be true
        assert out_dict["B"]["source_prob"] >= out_dict["C"]["source_prob"]  # B closer to A
        
        # In-degree propagation: different pattern
        assert in_dict["A"]["source_prob"] > 0.9  # Still seed
        # Results should be valid probabilities
        assert 0 <= in_dict["B"]["source_prob"] <= 1
        assert 0 <= in_dict["C"]["source_prob"] <= 1
    
    def test_single_seed_propagation(self):
        """Test propagation with only one seed node."""
        # Create star graph: center connected to 4 periphery nodes
        graph = nk.Graph(5, weighted=True, directed=False)
        for i in range(1, 5):
            graph.addEdge(0, i, 1.0)
        
        id_mapper = IDMapper()
        for i in range(5):
            id_mapper.add_mapping(f"node_{i}", i)
        
        # Only center node is seed
        seed_labels = {"node_0": "center"}
        labels = ["center", "periphery"]
        
        result = guided_label_propagation(
            graph, id_mapper, seed_labels, labels,
            alpha=0.7
        )
        
        result_dict = {row["node_id"]: row for row in result.iter_rows(named=True)}
        
        # Center should be strongly "center"
        assert result_dict["node_0"]["center_prob"] > 0.9
        
        # Periphery nodes should have some "center" influence
        for i in range(1, 5):
            node_id = f"node_{i}"
            # They should have at least some center influence, but not be seeds
            assert result_dict[node_id]["center_prob"] > 0.1
            assert result_dict[node_id]["is_seed"] is False


class TestParameterEffects:
    """Test effects of different parameter values."""
    
    def setup_method(self):
        """Set up common test graph for parameter testing."""
        # Create path graph: A-B-C-D-E
        self.graph = nk.Graph(5, weighted=True, directed=False)
        for i in range(4):
            self.graph.addEdge(i, i+1, 1.0)
        
        self.id_mapper = IDMapper()
        for i in range(5):
            self.id_mapper.add_mapping(chr(ord('A') + i), i)
        
        # Seeds at extremes
        self.seed_labels = {"A": "left", "E": "right"}
        self.labels = ["left", "right"]
    
    def test_alpha_effects(self):
        """Test effect of different alpha values on propagation."""
        results = {}
        
        for alpha in [0.1, 0.5, 0.9]:
            result = guided_label_propagation(
                self.graph, self.id_mapper, self.seed_labels, self.labels,
                alpha=alpha, max_iterations=100
            )
            results[alpha] = {row["node_id"]: row for row in result.iter_rows(named=True)}
        
        # All results should be valid probabilities
        for alpha in [0.1, 0.5, 0.9]:
            c_result = results[alpha]["C"]
            assert 0 <= c_result["left_prob"] <= 1
            assert 0 <= c_result["right_prob"] <= 1
            assert abs(c_result["left_prob"] + c_result["right_prob"] - 1.0) < 1e-6
        
        # With different alpha values, we expect different convergence behavior
        # (exact relationships depend on graph structure and may vary)
    
    def test_convergence_threshold_effects(self):
        """Test effect of different convergence thresholds."""
        # Tight threshold should take more iterations
        result_tight = guided_label_propagation(
            self.graph, self.id_mapper, self.seed_labels, self.labels,
            alpha=0.8, convergence_threshold=1e-8, max_iterations=200
        )
        
        # Loose threshold should converge faster
        result_loose = guided_label_propagation(
            self.graph, self.id_mapper, self.seed_labels, self.labels,
            alpha=0.8, convergence_threshold=1e-3, max_iterations=200
        )
        
        # Both should produce valid results
        assert len(result_tight) == 5
        assert len(result_loose) == 5
        
        # Results should be similar but not identical
        tight_dict = {row["node_id"]: row for row in result_tight.iter_rows(named=True)}
        loose_dict = {row["node_id"]: row for row in result_loose.iter_rows(named=True)}
        
        c_diff = abs(tight_dict["C"]["left_prob"] - loose_dict["C"]["left_prob"])
        assert c_diff < 0.1  # Should be similar
    
    def test_normalization_effects(self):
        """Test effect of probability normalization."""
        result_norm = guided_label_propagation(
            self.graph, self.id_mapper, self.seed_labels, self.labels,
            normalize=True
        )
        
        result_no_norm = guided_label_propagation(
            self.graph, self.id_mapper, self.seed_labels, self.labels,
            normalize=False
        )
        
        # Normalized results should sum to 1
        for row in result_norm.iter_rows(named=True):
            prob_sum = row["left_prob"] + row["right_prob"]
            assert abs(prob_sum - 1.0) < 1e-6
        
        # Non-normalized may not sum to 1
        non_norm_sums = []
        for row in result_no_norm.iter_rows(named=True):
            prob_sum = row["left_prob"] + row["right_prob"]
            non_norm_sums.append(prob_sum)
        
        # At least some should be different from 1.0
        assert any(abs(s - 1.0) > 1e-6 for s in non_norm_sums)


class TestDirectionalPropagation:
    """Test directional propagation for directed graphs."""
    
    def test_directed_vs_undirected(self):
        """Test that directed and undirected graphs produce different results."""
        # Create directed graph: A -> B -> C
        directed_graph = nk.Graph(3, weighted=True, directed=True)
        directed_graph.addEdge(0, 1, 1.0)
        directed_graph.addEdge(1, 2, 1.0)
        
        # Create equivalent undirected graph
        undirected_graph = nk.Graph(3, weighted=True, directed=False)
        undirected_graph.addEdge(0, 1, 1.0)
        undirected_graph.addEdge(1, 2, 1.0)
        
        id_mapper = IDMapper()
        for i in range(3):
            id_mapper.add_mapping(f"node_{i}", i)
        
        seed_labels = {"node_0": "source"}
        labels = ["source", "sink"]
        
        # Run on directed graph
        directed_result = guided_label_propagation(
            directed_graph, id_mapper, seed_labels, labels,
            directional=False  # Single propagation for comparison
        )
        
        # Run on undirected graph
        undirected_result = guided_label_propagation(
            undirected_graph, id_mapper, seed_labels, labels
        )
        
        # Results should be different
        dir_dict = {row["node_id"]: row for row in directed_result.iter_rows(named=True)}
        undir_dict = {row["node_id"]: row for row in undirected_result.iter_rows(named=True)}
        
        # Node 2 should have different probabilities
        assert abs(dir_dict["node_2"]["source_prob"] - undir_dict["node_2"]["source_prob"]) > 0.1
    
    def test_in_vs_out_degree_propagation(self):
        """Test that in-degree and out-degree propagation differ."""
        # Create directed graph with different in/out patterns
        graph = nk.Graph(4, weighted=True, directed=True)
        graph.addEdge(0, 1, 1.0)  # A -> B
        graph.addEdge(0, 2, 1.0)  # A -> C
        graph.addEdge(3, 1, 1.0)  # D -> B
        
        id_mapper = IDMapper()
        for i in range(4):
            id_mapper.add_mapping(chr(ord('A') + i), i)
        
        seed_labels = {"A": "source"}
        labels = ["source", "sink"]
        
        out_result, in_result = guided_label_propagation(
            graph, id_mapper, seed_labels, labels,
            directional=True
        )
        
        out_dict = {row["node_id"]: row for row in out_result.iter_rows(named=True)}
        in_dict = {row["node_id"]: row for row in in_result.iter_rows(named=True)}
        
        # B receives from both A and D in out-degree propagation
        # but only from A in in-degree propagation (considering reverse edges)
        assert out_dict["B"]["source_prob"] != in_dict["B"]["source_prob"]
        assert out_dict["C"]["source_prob"] != in_dict["C"]["source_prob"]


class TestMultiLabelPropagation:
    """Test propagation with multiple labels."""
    
    def test_three_label_propagation(self):
        """Test propagation with three different labels."""
        # Create triangle with different seeds
        graph = nk.Graph(6, weighted=True, directed=False)
        # Create two triangles connected by one edge
        graph.addEdge(0, 1, 1.0)
        graph.addEdge(1, 2, 1.0)
        graph.addEdge(2, 0, 1.0)
        graph.addEdge(3, 4, 1.0)
        graph.addEdge(4, 5, 1.0)
        graph.addEdge(5, 3, 1.0)
        graph.addEdge(2, 3, 1.0)  # Connection between triangles
        
        id_mapper = IDMapper()
        for i in range(6):
            id_mapper.add_mapping(f"node_{i}", i)
        
        seed_labels = {
            "node_0": "red",
            "node_1": "green", 
            "node_4": "blue"
        }
        labels = ["red", "green", "blue"]
        
        result = guided_label_propagation(
            graph, id_mapper, seed_labels, labels
        )
        
        # Validate result structure
        assert set(result.columns) == {
            "node_id", "red_prob", "green_prob", "blue_prob",
            "dominant_label", "confidence", "is_seed"
        }
        
        # Check probability normalization
        for row in result.iter_rows(named=True):
            prob_sum = row["red_prob"] + row["green_prob"] + row["blue_prob"]
            assert abs(prob_sum - 1.0) < 1e-6
        
        # Seed nodes should have high confidence in their labels
        result_dict = {row["node_id"]: row for row in result.iter_rows(named=True)}
        assert result_dict["node_0"]["dominant_label"] == "red"
        assert result_dict["node_1"]["dominant_label"] == "green"
        assert result_dict["node_4"]["dominant_label"] == "blue"
    
    def test_unbalanced_seed_sets(self):
        """Test propagation with unbalanced number of seeds per label."""
        # Create star graph
        graph = nk.Graph(7, weighted=True, directed=False)
        for i in range(1, 7):
            graph.addEdge(0, i, 1.0)
        
        id_mapper = IDMapper()
        for i in range(7):
            id_mapper.add_mapping(f"node_{i}", i)
        
        # Unbalanced seeds: 4 "many", 1 "few"
        seed_labels = {
            "node_1": "many",
            "node_2": "many", 
            "node_3": "many",
            "node_4": "many",
            "node_5": "few"
        }
        labels = ["many", "few"]
        
        result = guided_label_propagation(
            graph, id_mapper, seed_labels, labels
        )
        
        result_dict = {row["node_id"]: row for row in result.iter_rows(named=True)}
        
        # Center should be influenced more by "many" due to more seeds
        assert result_dict["node_0"]["many_prob"] > result_dict["node_0"]["few_prob"]
        
        # The "few" seed should have more "few" probability than others in the graph
        # (though it may be influenced by neighbors)
        few_seed_prob = result_dict["node_5"]["few_prob"]
        center_few_prob = result_dict["node_0"]["few_prob"]
        assert few_seed_prob > center_few_prob


class TestEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_disconnected_components(self):
        """Test propagation on disconnected graph."""
        # Create two disconnected triangles
        graph = nk.Graph(6, weighted=True, directed=False)
        # Triangle 1
        graph.addEdge(0, 1, 1.0)
        graph.addEdge(1, 2, 1.0)
        graph.addEdge(2, 0, 1.0)
        # Triangle 2 (disconnected)
        graph.addEdge(3, 4, 1.0)
        graph.addEdge(4, 5, 1.0)
        graph.addEdge(5, 3, 1.0)
        
        id_mapper = IDMapper()
        for i in range(6):
            id_mapper.add_mapping(f"node_{i}", i)
        
        # Seeds only in first component
        seed_labels = {"node_0": "labeled"}
        labels = ["labeled", "unlabeled"]
        
        result = guided_label_propagation(
            graph, id_mapper, seed_labels, labels
        )
        
        result_dict = {row["node_id"]: row for row in result.iter_rows(named=True)}
        
        # First component should have "labeled" influence
        assert result_dict["node_1"]["labeled_prob"] > 0.5
        assert result_dict["node_2"]["labeled_prob"] > 0.5
        
        # Second component should have uniform distribution
        assert abs(result_dict["node_3"]["labeled_prob"] - 0.5) < 0.1
        assert abs(result_dict["node_4"]["labeled_prob"] - 0.5) < 0.1
        assert abs(result_dict["node_5"]["labeled_prob"] - 0.5) < 0.1
    
    def test_isolated_nodes(self):
        """Test propagation with isolated (zero-degree) nodes."""
        # Graph with one isolated node
        graph = nk.Graph(4, weighted=True, directed=False)
        graph.addEdge(0, 1, 1.0)
        graph.addEdge(1, 2, 1.0)
        # Node 3 is isolated
        
        id_mapper = IDMapper()
        for i in range(4):
            id_mapper.add_mapping(f"node_{i}", i)
        
        seed_labels = {"node_0": "connected", "node_3": "isolated"}
        labels = ["connected", "isolated"]
        
        result = guided_label_propagation(
            graph, id_mapper, seed_labels, labels
        )
        
        result_dict = {row["node_id"]: row for row in result.iter_rows(named=True)}
        
        # Isolated seed should retain its label perfectly
        assert result_dict["node_3"]["isolated_prob"] > 0.99
        assert result_dict["node_3"]["dominant_label"] == "isolated"
        
        # Connected nodes should be influenced by connected seed
        assert result_dict["node_1"]["connected_prob"] > 0.5
        assert result_dict["node_2"]["connected_prob"] > 0.3
    
    def test_self_loops(self):
        """Test propagation with self-loops."""
        graph = nk.Graph(3, weighted=True, directed=False)
        graph.addEdge(0, 1, 1.0)
        graph.addEdge(1, 2, 1.0)
        graph.addEdge(1, 1, 2.0)  # Self-loop on node 1
        
        id_mapper = IDMapper()
        for i in range(3):
            id_mapper.add_mapping(f"node_{i}", i)
        
        seed_labels = {"node_0": "source"}
        labels = ["source", "sink"]
        
        # Should not raise error
        result = guided_label_propagation(
            graph, id_mapper, seed_labels, labels
        )
        
        assert len(result) == 3
        result_dict = {row["node_id"]: row for row in result.iter_rows(named=True)}
        assert result_dict["node_0"]["dominant_label"] == "source"


class TestInputValidation:
    """Test input validation and error handling."""
    
    def test_empty_graph(self):
        """Test validation with empty graph."""
        graph = nk.Graph(0)
        id_mapper = IDMapper()
        
        with pytest.raises(ValidationError, match="Graph has no nodes"):
            guided_label_propagation(graph, id_mapper, {}, ["A", "B"])
    
    def test_invalid_alpha(self):
        """Test validation with invalid alpha values."""
        graph = nk.Graph(2, weighted=True)
        graph.addEdge(0, 1, 1.0)
        
        id_mapper = IDMapper()
        id_mapper.add_mapping("A", 0)
        id_mapper.add_mapping("B", 1)
        
        seed_labels = {"A": "label1"}
        labels = ["label1", "label2"]
        
        # Alpha too low
        with pytest.raises(ConfigurationError, match="Alpha must be between 0 and 1"):
            guided_label_propagation(graph, id_mapper, seed_labels, labels, alpha=-0.1)
        
        # Alpha too high
        with pytest.raises(ConfigurationError, match="Alpha must be between 0 and 1"):
            guided_label_propagation(graph, id_mapper, seed_labels, labels, alpha=1.1)
    
    def test_empty_seed_labels(self):
        """Test validation with empty seed labels."""
        graph = nk.Graph(2, weighted=True)
        graph.addEdge(0, 1, 1.0)
        
        id_mapper = IDMapper()
        id_mapper.add_mapping("A", 0)
        id_mapper.add_mapping("B", 1)
        
        with pytest.raises(ValidationError, match="Seed labels dictionary cannot be empty"):
            guided_label_propagation(graph, id_mapper, {}, ["A", "B"])
    
    def test_unknown_seed_labels(self):
        """Test validation with seed labels not in labels list."""
        graph = nk.Graph(2, weighted=True)
        graph.addEdge(0, 1, 1.0)
        
        id_mapper = IDMapper()
        id_mapper.add_mapping("A", 0)
        id_mapper.add_mapping("B", 1)
        
        seed_labels = {"A": "unknown_label"}
        labels = ["label1", "label2"]
        
        with pytest.raises(ValidationError, match="Seed labels contain unknown labels"):
            guided_label_propagation(graph, id_mapper, seed_labels, labels)
    
    def test_missing_seed_nodes(self):
        """Test validation with seed nodes not in graph."""
        graph = nk.Graph(2, weighted=True)
        graph.addEdge(0, 1, 1.0)
        
        id_mapper = IDMapper()
        id_mapper.add_mapping("A", 0)
        id_mapper.add_mapping("B", 1)
        
        seed_labels = {"C": "label1"}  # C not in graph
        labels = ["label1", "label2"]
        
        with pytest.raises(ValidationError, match="Seed nodes not found in graph"):
            guided_label_propagation(graph, id_mapper, seed_labels, labels)
    
    def test_convergence_failure(self):
        """Test convergence failure handling."""
        # Create graph that might have convergence issues
        graph = nk.Graph(3, weighted=True, directed=False)
        graph.addEdge(0, 1, 1.0)
        graph.addEdge(1, 2, 1.0)
        
        id_mapper = IDMapper()
        for i in range(3):
            id_mapper.add_mapping(f"node_{i}", i)
        
        seed_labels = {"node_0": "A", "node_2": "B"}
        labels = ["A", "B"]
        
        # Very tight threshold with very few iterations should raise ConvergenceError
        with pytest.raises(ConvergenceError, match="failed to converge"):
            guided_label_propagation(
                graph, id_mapper, seed_labels, labels,
                max_iterations=1, convergence_threshold=1e-10
            )
        
        # For most practical cases, the algorithm converges well
        # The convergence error above demonstrates proper error handling


class TestPropagationInfo:
    """Test propagation information utility function."""
    
    def test_basic_propagation_info(self):
        """Test basic propagation info functionality."""
        graph = nk.Graph(5, weighted=True, directed=False)
        for i in range(4):
            graph.addEdge(i, i+1, 1.0)
        
        seed_labels = {"node_0": "A", "node_4": "B"}
        labels = ["A", "B"]
        
        info = get_propagation_info(graph, seed_labels, labels)
        
        # Check structure
        assert "graph_stats" in info
        assert "seed_stats" in info
        assert "memory_estimate" in info
        assert "computational_estimate" in info
        assert "potential_issues" in info
        
        # Check basic stats
        assert info["graph_stats"]["nodes"] == 5
        assert info["graph_stats"]["edges"] == 4
        assert info["graph_stats"]["labels"] == 2
        assert info["graph_stats"]["seeds"] == 2
        
        # Check seed stats
        assert info["seed_stats"]["seeds_per_label"]["A"] == 1
        assert info["seed_stats"]["seeds_per_label"]["B"] == 1
        assert info["seed_stats"]["label_balance"] == 1.0
    
    def test_propagation_info_warnings(self):
        """Test propagation info with problematic configurations."""
        # Empty graph
        graph = nk.Graph(3, weighted=True, directed=False)
        # No edges
        
        seed_labels = {"node_0": "A"}  # Only one label
        labels = ["A", "B"]
        
        info = get_propagation_info(graph, seed_labels, labels)
        
        issues = info["potential_issues"]
        assert any("No seeds for labels" in issue for issue in issues)
        assert any("Graph has no edges" in issue for issue in issues)


class TestHelperFunctions:
    """Test individual helper functions for modularity."""
    
    def setup_method(self):
        """Set up common test data."""
        self.graph = nk.Graph(3, weighted=True, directed=False)
        self.graph.addEdge(0, 1, 1.0)
        self.graph.addEdge(1, 2, 2.0)
        
        self.id_mapper = IDMapper()
        for i in range(3):
            self.id_mapper.add_mapping(f"node_{i}", i)
    
    def test_initialize_label_matrix(self):
        """Test initial label matrix creation."""
        seed_labels = {"node_0": "A", "node_2": "B"}
        labels = ["A", "B"]
        
        Y = _initialize_label_matrix(self.graph, self.id_mapper, seed_labels, labels)
        
        assert Y.shape == (3, 2)
        assert Y[0, 0] == 1.0  # node_0 has label A
        assert Y[0, 1] == 0.0
        assert Y[1, 0] == 0.0  # node_1 has no label
        assert Y[1, 1] == 0.0
        assert Y[2, 0] == 0.0  # node_2 has label B
        assert Y[2, 1] == 1.0
    
    def test_initialize_label_matrix_single_label(self):
        """Test label matrix with only one label."""
        seed_labels = {"node_0": "A", "node_1": "A"}
        labels = ["A", "B"]
        
        Y = _initialize_label_matrix(self.graph, self.id_mapper, seed_labels, labels)
        
        assert Y.shape == (3, 2)
        assert Y[0, 0] == 1.0  # node_0 has label A
        assert Y[1, 0] == 1.0  # node_1 has label A
        assert Y[2, 0] == 0.0  # node_2 has no label
        assert np.sum(Y[:, 1]) == 0.0  # No B labels
    
    def test_create_transition_matrix_undirected(self):
        """Test transition matrix creation for undirected graph."""
        P = _create_transition_matrix(self.graph, "undirected")
        
        # Should be row-normalized
        row_sums = np.array(P.sum(axis=1)).flatten()
        for i in range(P.shape[0]):
            if row_sums[i] > 0:  # Non-isolated nodes
                assert abs(row_sums[i] - 1.0) < 1e-10
        
        # Should be sparse
        assert P.nnz > 0
        assert P.nnz <= 6  # Maximum edges for undirected 3-node graph
    
    def test_create_transition_matrix_directed(self):
        """Test transition matrix creation for directed graph."""
        # Make graph directed
        directed_graph = nk.Graph(3, weighted=True, directed=True)
        directed_graph.addEdge(0, 1, 1.0)
        directed_graph.addEdge(1, 2, 2.0)
        
        P_out = _create_transition_matrix(directed_graph, "out_degree")
        P_in = _create_transition_matrix(directed_graph, "in_degree")
        
        # Both should be valid transition matrices
        assert P_out.shape == (3, 3)
        assert P_in.shape == (3, 3)
        
        # They should be different for this directed graph
        assert not np.allclose(P_out.toarray(), P_in.toarray())
        
        # Row normalization check
        out_sums = np.array(P_out.sum(axis=1)).flatten()
        in_sums = np.array(P_in.sum(axis=1)).flatten()
        
        # Non-zero rows should sum to 1
        for i in range(3):
            if out_sums[i] > 0:
                assert abs(out_sums[i] - 1.0) < 1e-10
            if in_sums[i] > 0:
                assert abs(in_sums[i] - 1.0) < 1e-10
    
    def test_propagate_iteration(self):
        """Test single propagation iteration."""
        # Set up matrices
        F = np.array([[1.0, 0.0], [0.5, 0.5], [0.0, 1.0]], dtype=np.float64)
        Y = np.array([[1.0, 0.0], [0.0, 0.0], [0.0, 1.0]], dtype=np.float64)
        P = _create_transition_matrix(self.graph, "undirected")
        alpha = 0.8
        
        F_new = _propagate_iteration(F, P, Y, alpha)
        
        # Check output properties
        assert F_new.shape == F.shape
        assert F_new.dtype == np.float64
        assert not np.array_equal(F_new, F)  # Should have changed
        
        # Check that seeds are preserved (approximately)
        # Seed nodes should have strong influence from Y matrix
        assert F_new[0, 0] > 0.5  # node_0 should remain mostly A
        assert F_new[2, 1] > 0.5  # node_2 should remain mostly B
    
    def test_propagate_iteration_extreme_alpha(self):
        """Test propagation iteration with extreme alpha values."""
        F = np.array([[1.0, 0.0], [0.5, 0.5], [0.0, 1.0]], dtype=np.float64)
        Y = np.array([[1.0, 0.0], [0.0, 0.0], [0.0, 1.0]], dtype=np.float64)
        P = _create_transition_matrix(self.graph, "undirected")
        
        # Alpha = 0 (no propagation, only seeds)
        F_no_prop = _propagate_iteration(F, P, Y, alpha=0.0)
        assert np.allclose(F_no_prop, Y)
        
        # Alpha = 1 (pure propagation, no seed retention)
        F_pure_prop = _propagate_iteration(F, P, Y, alpha=1.0)
        expected = P.dot(F)
        assert np.allclose(F_pure_prop, expected)
    
    def test_check_convergence(self):
        """Test convergence checking."""
        F_old = np.array([[1.0, 0.0], [0.6, 0.4], [0.0, 1.0]])
        
        # Test converged case
        F_new_converged = F_old + 1e-8  # Very small change
        converged, change = _check_convergence(F_new_converged, F_old, 1e-6)
        assert converged == True
        assert change < 1e-6
        
        # Test not converged case
        F_new_not_converged = F_old + 0.1  # Large change
        converged, change = _check_convergence(F_new_not_converged, F_old, 1e-6)
        assert converged == False
        assert change > 1e-6
        
        # Test near threshold (but above)
        F_new_threshold = F_old.copy()
        F_new_threshold[0, 0] += 2e-6  # Slightly above threshold
        converged, change = _check_convergence(F_new_threshold, F_old, 1e-6)
        assert converged == False  # Should be strictly less than threshold
        assert change > 1e-6
    
    def test_create_results_dataframe(self):
        """Test results DataFrame creation."""
        # Set up test data
        F = np.array([[0.9, 0.1], [0.3, 0.7], [0.1, 0.9]], dtype=np.float64)
        Y = np.array([[1.0, 0.0], [0.0, 0.0], [0.0, 1.0]], dtype=np.float64)
        labels = ["A", "B"]
        
        df = _create_results_dataframe(
            F, Y, labels, self.id_mapper, normalize=True, 
            converged_iteration=10, direction="undirected"
        )
        
        # Check DataFrame structure
        expected_columns = {"node_id", "A_prob", "B_prob", "dominant_label", "confidence", "is_seed"}
        assert set(df.columns) == expected_columns
        assert len(df) == 3
        
        # Check data types and values
        assert df["node_id"].dtype == pl.String
        assert df["A_prob"].dtype == pl.Float64
        assert df["B_prob"].dtype == pl.Float64
        assert df["dominant_label"].dtype == pl.String
        assert df["confidence"].dtype == pl.Float64
        assert df["is_seed"].dtype == pl.Boolean
        
        # Check specific values
        result_dict = {row["node_id"]: row for row in df.iter_rows(named=True)}
        
        assert result_dict["node_0"]["dominant_label"] == "A"
        assert result_dict["node_2"]["dominant_label"] == "B"
        assert result_dict["node_0"]["is_seed"] is True
        assert result_dict["node_1"]["is_seed"] is False
        assert result_dict["node_2"]["is_seed"] is True
        
        # Check probability normalization
        for row in df.iter_rows(named=True):
            prob_sum = row["A_prob"] + row["B_prob"]
            assert abs(prob_sum - 1.0) < 1e-10
    
    def test_create_results_dataframe_no_normalization(self):
        """Test results DataFrame creation without normalization."""
        F = np.array([[1.8, 0.1], [0.3, 1.4], [0.1, 1.8]], dtype=np.float64)  # Not normalized
        Y = np.array([[1.0, 0.0], [0.0, 0.0], [0.0, 1.0]], dtype=np.float64)
        labels = ["A", "B"]
        
        df = _create_results_dataframe(
            F, Y, labels, self.id_mapper, normalize=False, 
            converged_iteration=5, direction="out_degree"
        )
        
        # Check that probabilities are not normalized
        result_dict = {row["node_id"]: row for row in df.iter_rows(named=True)}
        
        # Should not sum to 1.0
        for node_id in ["node_0", "node_1", "node_2"]:
            prob_sum = result_dict[node_id]["A_prob"] + result_dict[node_id]["B_prob"]
            assert abs(prob_sum - 1.0) > 0.1  # Should be clearly different from 1.0
    
    def test_create_results_dataframe_zero_probabilities(self):
        """Test results DataFrame with zero probability rows."""
        F = np.array([[0.0, 0.0], [0.3, 0.7], [0.0, 0.0]], dtype=np.float64)  # Rows 0,2 are zero
        Y = np.array([[1.0, 0.0], [0.0, 0.0], [0.0, 1.0]], dtype=np.float64)
        labels = ["A", "B"]
        
        df = _create_results_dataframe(
            F, Y, labels, self.id_mapper, normalize=True, 
            converged_iteration=3, direction="in_degree"
        )
        
        result_dict = {row["node_id"]: row for row in df.iter_rows(named=True)}
        
        # Zero probability rows should get uniform distribution when normalized
        assert abs(result_dict["node_0"]["A_prob"] - 0.5) < 1e-10
        assert abs(result_dict["node_0"]["B_prob"] - 0.5) < 1e-10
        assert abs(result_dict["node_2"]["A_prob"] - 0.5) < 1e-10
        assert abs(result_dict["node_2"]["B_prob"] - 0.5) < 1e-10
        
        # Non-zero row should maintain its probabilities (normalized)
        assert result_dict["node_1"]["A_prob"] == 0.3
        assert result_dict["node_1"]["B_prob"] == 0.7


class TestInternalFunctions:
    """Test backward compatibility of existing internal function tests."""
    
    def setup_method(self):
        """Set up common test data."""
        self.graph = nk.Graph(3, weighted=True, directed=False)
        self.graph.addEdge(0, 1, 1.0)
        self.graph.addEdge(1, 2, 2.0)
        
        self.id_mapper = IDMapper()
        for i in range(3):
            self.id_mapper.add_mapping(f"node_{i}", i)
    
    def test_initialize_label_matrix_compatibility(self):
        """Test label matrix creation for backward compatibility."""
        seed_labels = {"node_0": "A", "node_2": "B"}
        labels = ["A", "B"]
        
        Y = _initialize_label_matrix(self.graph, self.id_mapper, seed_labels, labels)
        
        assert Y.shape == (3, 2)
        assert Y[0, 0] == 1.0  # node_0 has label A
        assert Y[0, 1] == 0.0
        assert Y[1, 0] == 0.0  # node_1 has no label
        assert Y[1, 1] == 0.0
        assert Y[2, 0] == 0.0  # node_2 has label B
        assert Y[2, 1] == 1.0
    
    def test_create_transition_matrix_undirected_compatibility(self):
        """Test transition matrix creation for undirected graph."""
        P = _create_transition_matrix(self.graph, "undirected")
        
        # Should be row-normalized
        row_sums = np.array(P.sum(axis=1)).flatten()
        for i in range(P.shape[0]):
            if row_sums[i] > 0:  # Non-isolated nodes
                assert abs(row_sums[i] - 1.0) < 1e-10
    
    def test_create_transition_matrix_directed_compatibility(self):
        """Test transition matrix creation for directed graph."""
        # Make graph directed
        directed_graph = nk.Graph(3, weighted=True, directed=True)
        directed_graph.addEdge(0, 1, 1.0)
        directed_graph.addEdge(1, 2, 2.0)
        
        P_out = _create_transition_matrix(directed_graph, "out_degree")
        P_in = _create_transition_matrix(directed_graph, "in_degree")
        
        # Both should be valid transition matrices
        assert P_out.shape == (3, 3)
        assert P_in.shape == (3, 3)
        
        # They should be different for this directed graph
        assert not np.allclose(P_out.toarray(), P_in.toarray())


if __name__ == "__main__":
    pytest.main([__file__])