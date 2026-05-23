"""
Tests for network analysis module.

This module tests centrality calculations, network metrics extraction,
and analytical functions for graph analysis.
"""

import pytest
import polars as pl
import networkit as nk
import numpy as np
from typing import Dict, Any, List

from src.network.analysis import (
    extract_centrality,
    get_centrality_summary,
    identify_central_nodes
)
from src.common.id_mapper import IDMapper
from src.common.exceptions import ValidationError, ComputationError


class TestExtractCentrality:
    """Test centrality calculation functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        # Create simple test graph
        self.simple_graph = nk.Graph(4)
        self.simple_graph.addEdge(0, 1)
        self.simple_graph.addEdge(1, 2)
        self.simple_graph.addEdge(2, 3)
        
        # Simple linear chain: A-B-C-D
        self.simple_mapper = IDMapper()
        self.simple_mapper.add_mapping("A", 0)
        self.simple_mapper.add_mapping("B", 1)
        self.simple_mapper.add_mapping("C", 2)
        self.simple_mapper.add_mapping("D", 3)

        # Create star graph (node 0 connected to all others)
        self.star_graph = nk.Graph(5)
        for i in range(1, 5):
            self.star_graph.addEdge(0, i)
        
        self.star_mapper = IDMapper()
        for i in range(5):
            self.star_mapper.add_mapping(f"node_{i}", i)

        # Create complete graph
        self.complete_graph = nk.Graph(4)
        for i in range(4):
            for j in range(i + 1, 4):
                self.complete_graph.addEdge(i, j)
        
        self.complete_mapper = IDMapper()
        for i in range(4):
            self.complete_mapper.add_mapping(f"n{i}", i)

    def test_single_metric_degree(self):
        """Test degree centrality calculation."""
        result = extract_centrality(
            self.simple_graph,
            self.simple_mapper,
            metrics=["degree"],
            normalized=False
        )
        
        assert isinstance(result, pl.DataFrame)
        assert result.shape == (4, 2)  # 4 nodes, 2 columns (node_id + degree)
        assert "node_id" in result.columns
        assert "degree_centrality" in result.columns
        
        # Check degree values for linear chain
        degrees = result.sort("node_id")["degree_centrality"].to_list()
        assert degrees == [1, 2, 2, 1]  # A and D have degree 1, B and C have degree 2

    def test_single_metric_betweenness(self):
        """Test betweenness centrality calculation."""
        result = extract_centrality(
            self.simple_graph,
            self.simple_mapper,
            metrics=["betweenness"],
            normalized=False
        )
        
        assert "betweenness_centrality" in result.columns
        
        # In linear chain, middle nodes have higher betweenness
        betweenness = dict(zip(
            result["node_id"].to_list(),
            result["betweenness_centrality"].to_list()
        ))
        assert betweenness["A"] == 0.0  # End nodes have no betweenness
        assert betweenness["D"] == 0.0
        assert betweenness["B"] > 0  # Middle nodes have positive betweenness
        assert betweenness["C"] > 0

    def test_multiple_metrics(self):
        """Test calculation of multiple centrality metrics."""
        result = extract_centrality(
            self.simple_graph,
            self.simple_mapper,
            metrics=["degree", "betweenness", "closeness"],
            normalized=True
        )
        
        assert result.shape == (4, 4)  # 4 nodes, 4 columns
        assert "degree_centrality" in result.columns
        assert "betweenness_centrality" in result.columns
        assert "closeness_centrality" in result.columns
        
        # Check that all values are between 0 and 1 (normalized)
        for col in ["degree_centrality", "betweenness_centrality", "closeness_centrality"]:
            values = result[col].to_list()
            assert all(0 <= v <= 1 for v in values)

    def test_star_graph_centrality(self):
        """Test centrality on star graph with known values."""
        result = extract_centrality(
            self.star_graph,
            self.star_mapper,
            metrics=["degree", "betweenness", "closeness"],
            normalized=False
        )
        
        # In star graph, center node (node_0) should have highest centrality
        center_row = result.filter(pl.col("node_id") == "node_0")
        leaf_rows = result.filter(pl.col("node_id") != "node_0")
        
        center_degree = center_row["degree_centrality"].item()
        leaf_degrees = leaf_rows["degree_centrality"].to_list()
        
        assert center_degree == 4  # Center connected to 4 leaves
        assert all(deg == 1 for deg in leaf_degrees)  # Leaves have degree 1
        
        # Center should have highest betweenness (all paths go through it)
        center_betweenness = center_row["betweenness_centrality"].item()
        leaf_betweenness = leaf_rows["betweenness_centrality"].to_list()
        
        assert center_betweenness > 0
        assert all(bet == 0 for bet in leaf_betweenness)  # Leaves have no betweenness

    def test_complete_graph_centrality(self):
        """Test centrality on complete graph (all nodes equal)."""
        result = extract_centrality(
            self.complete_graph,
            self.complete_mapper,
            metrics=["degree", "betweenness", "closeness"],
            normalized=False
        )
        
        # In complete graph, all nodes should have equal centrality
        degrees = result["degree_centrality"].to_list()
        assert all(deg == 3 for deg in degrees)  # Each node connected to 3 others
        
        betweenness = result["betweenness_centrality"].to_list()
        assert all(bet == 0 for bet in betweenness)  # No shortest paths in complete graph
        
        closeness = result["closeness_centrality"].to_list()
        # All nodes should have same closeness (direct connection to all)
        assert len(set(closeness)) == 1  # All values identical

    def test_normalized_vs_unnormalized(self):
        """Test difference between normalized and unnormalized centrality."""
        norm_result = extract_centrality(
            self.star_graph,
            self.star_mapper,
            metrics=["degree", "closeness"],
            normalized=True
        )
        
        unnorm_result = extract_centrality(
            self.star_graph,
            self.star_mapper,
            metrics=["degree", "closeness"],
            normalized=False
        )
        
        # Normalized values should be in [0, 1]
        norm_degrees = norm_result["degree_centrality"].to_list()
        assert all(0 <= deg <= 1 for deg in norm_degrees)
        
        # Unnormalized should have raw counts
        unnorm_degrees = unnorm_result["degree_centrality"].to_list()
        assert max(unnorm_degrees) == 4  # Center node degree
        assert min(unnorm_degrees) == 1  # Leaf node degree

    def test_eigenvector_centrality(self):
        """Test eigenvector centrality calculation."""
        result = extract_centrality(
            self.star_graph,
            self.star_mapper,
            metrics=["eigenvector"],
            normalized=True
        )
        
        assert "eigenvector_centrality" in result.columns
        
        # Center node should have highest eigenvector centrality
        center_row = result.filter(pl.col("node_id") == "node_0")
        center_eigen = center_row["eigenvector_centrality"].item()
        
        leaf_rows = result.filter(pl.col("node_id") != "node_0")
        leaf_eigen = leaf_rows["eigenvector_centrality"].to_list()
        
        assert center_eigen > max(leaf_eigen)

    def test_pagerank_centrality(self):
        """Test PageRank centrality calculation."""
        result = extract_centrality(
            self.simple_graph,
            self.simple_mapper,
            metrics=["pagerank"],
            normalized=False
        )
        
        assert "pagerank_centrality" in result.columns
        
        # PageRank values should sum to approximately 1
        pagerank_sum = sum(result["pagerank_centrality"].to_list())
        assert abs(pagerank_sum - 1.0) < 0.01

    def test_katz_centrality(self):
        """Test Katz centrality calculation."""
        result = extract_centrality(
            self.simple_graph,
            self.simple_mapper,
            metrics=["katz"],
            normalized=True
        )
        
        assert "katz_centrality" in result.columns
        
        # All values should be positive and normalized
        katz_values = result["katz_centrality"].to_list()
        assert all(val > 0 for val in katz_values)
        assert all(0 <= val <= 1 for val in katz_values)

    def test_all_metrics(self):
        """Test calculation of all available metrics."""
        result = extract_centrality(
            self.simple_graph,
            self.simple_mapper,
            metrics=["degree", "betweenness", "closeness", 
                    "eigenvector", "pagerank", "katz"],
            normalized=True
        )
        
        expected_columns = [
            "node_id", "degree_centrality", "betweenness_centrality",
            "closeness_centrality", "eigenvector_centrality",
            "pagerank_centrality", "katz_centrality"
        ]
        
        assert set(result.columns) == set(expected_columns)
        assert result.shape == (4, 7)

    def test_parallel_processing(self):
        """Test parallel processing functionality."""
        # Test with n_jobs=2
        result_parallel = extract_centrality(
            self.simple_graph,
            self.simple_mapper,
            metrics=["degree", "betweenness"],
            normalized=True,
            n_jobs=2
        )
        
        # Test sequential (n_jobs=1)
        result_sequential = extract_centrality(
            self.simple_graph,
            self.simple_mapper,
            metrics=["degree", "betweenness"],
            normalized=True,
            n_jobs=1
        )
        
        # Results should be identical
        assert result_parallel.equals(result_sequential)

    def test_invalid_metric_names(self):
        """Test error handling for invalid metric names."""
        with pytest.raises(ValidationError) as exc_info:
            extract_centrality(
                self.simple_graph,
                self.simple_mapper,
                metrics=["invalid_metric"]
            )
        
        assert "invalid_metric" in str(exc_info.value)
        assert "Available metrics" in str(exc_info.value)

    def test_mixed_valid_invalid_metrics(self):
        """Test error handling for mix of valid and invalid metrics."""
        with pytest.raises(ValidationError):
            extract_centrality(
                self.simple_graph,
                self.simple_mapper,
                metrics=["degree", "invalid_metric", "betweenness"]
            )

    def test_empty_metrics_list(self):
        """Test error handling for empty metrics list."""
        with pytest.raises(ValidationError) as exc_info:
            extract_centrality(
                self.simple_graph,
                self.simple_mapper,
                metrics=[]
            )
        
        assert "at least one" in str(exc_info.value).lower()

    def test_empty_graph(self):
        """Test handling of empty graph."""
        empty_graph = nk.Graph(0)
        empty_mapper = IDMapper()
        
        result = extract_centrality(
            empty_graph,
            empty_mapper,
            metrics=["degree"]
        )
        
        assert result.shape == (0, 2)
        assert list(result.columns) == ["node_id", "degree_centrality"]

    def test_single_node_graph(self):
        """Test handling of single node graph."""
        single_graph = nk.Graph(1)
        single_mapper = IDMapper()
        single_mapper.add_mapping("only_node", 0)
        
        result = extract_centrality(
            single_graph,
            single_mapper,
            metrics=["degree", "betweenness", "closeness"]
        )
        
        assert result.shape == (1, 4)
        assert result["node_id"].item() == "only_node"
        assert result["degree_centrality"].item() == 0
        assert result["betweenness_centrality"].item() == 0

    def test_disconnected_graph(self):
        """Test handling of disconnected graph."""
        disconnected = nk.Graph(4)
        disconnected.addEdge(0, 1)  # Component 1: 0-1
        disconnected.addEdge(2, 3)  # Component 2: 2-3
        
        mapper = IDMapper()
        for i in range(4):
            mapper.add_mapping(f"node_{i}", i)
        
        result = extract_centrality(
            disconnected,
            mapper,
            metrics=["degree", "betweenness", "closeness"],
            normalized=False
        )
        
        assert result.shape == (4, 4)
        
        # All nodes should have degree 1
        degrees = result["degree_centrality"].to_list()
        assert all(deg == 1 for deg in degrees)
        
        # No node should have betweenness (no paths between components)
        betweenness = result["betweenness_centrality"].to_list()
        assert all(bet == 0 for bet in betweenness)

    def test_directed_graph(self):
        """Test centrality on directed graph."""
        directed_graph = nk.Graph(3, directed=True)
        directed_graph.addEdge(0, 1)
        directed_graph.addEdge(1, 2)
        
        mapper = IDMapper()
        for i in range(3):
            mapper.add_mapping(f"node_{i}", i)
        
        result = extract_centrality(
            directed_graph,
            mapper,
            metrics=["degree"],
            normalized=False
        )
        
        # Should handle directed graphs correctly
        assert result.shape == (3, 2)
        degrees = result.sort("node_id")["degree_centrality"].to_list()
        # For directed graphs, degree counts both in and out edges
        # Node 0: out-degree 1, Node 1: in-degree 1 + out-degree 1, Node 2: in-degree 1
        assert degrees[1] >= degrees[0]  # Middle node should have higher or equal degree

    def test_result_sorting(self):
        """Test that results are sorted by node_id."""
        result = extract_centrality(
            self.simple_graph,
            self.simple_mapper,
            metrics=["degree"]
        )
        
        node_ids = result["node_id"].to_list()
        assert node_ids == sorted(node_ids)

    def test_id_mapping_consistency(self):
        """Test that ID mapping is handled correctly."""
        result = extract_centrality(
            self.simple_graph,
            self.simple_mapper,
            metrics=["degree"]
        )
        
        # Should return original IDs, not internal IDs
        node_ids = result["node_id"].to_list()
        assert set(node_ids) == {"A", "B", "C", "D"}
        assert all(isinstance(nid, str) for nid in node_ids)


class TestCentralitySummary:
    """Test centrality summary functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.graph = nk.Graph(5)
        for i in range(1, 5):
            self.graph.addEdge(0, i)  # Star graph
        
        self.mapper = IDMapper()
        for i in range(5):
            self.mapper.add_mapping(f"node_{i}", i)

    def test_centrality_summary(self):
        """Test centrality summary statistics."""
        centrality_df = extract_centrality(
            self.graph,
            self.mapper,
            metrics=["degree", "betweenness"]
        )
        
        summary = get_centrality_summary(centrality_df)
        
        assert isinstance(summary, Dict)
        assert "degree_centrality" in summary
        assert "betweenness_centrality" in summary
        
        degree_stats = summary["degree_centrality"]
        assert "mean" in degree_stats
        assert "std" in degree_stats
        assert "min" in degree_stats
        assert "max" in degree_stats
        assert "median" in degree_stats

    def test_identify_central_nodes(self):
        """Test identification of central nodes."""
        centrality_df = extract_centrality(
            self.graph,
            self.mapper,
            metrics=["degree", "betweenness"]
        )
        
        central_nodes = identify_central_nodes(
            centrality_df,
            metric="degree_centrality",
            top_k=2
        )
        
        assert isinstance(central_nodes, List)
        assert len(central_nodes) == 2
        assert "node_0" in central_nodes  # Center node should be most central


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_invalid_n_jobs(self):
        """Test invalid n_jobs parameter."""
        graph = nk.Graph(2)
        graph.addEdge(0, 1)
        mapper = IDMapper()
        mapper.add_mapping("A", 0)
        mapper.add_mapping("B", 1)
        
        with pytest.raises(ValidationError) as exc_info:
            extract_centrality(
                graph,
                mapper,
                metrics=["degree"],
                n_jobs=0
            )
        
        assert "n_jobs" in str(exc_info.value)

    def test_computation_error_handling(self):
        """Test handling of computation errors."""
        # This test would require mocking NetworkIt functions to raise errors
        # For now, we test that the structure is in place
        graph = nk.Graph(2)
        mapper = IDMapper()
        
        # Should handle empty mapper gracefully
        result = extract_centrality(graph, mapper, metrics=["degree"])
        assert result.shape == (0, 2)

    def test_large_graph_performance(self):
        """Test performance on moderately large graph."""
        # Create a larger graph for performance testing
        n_nodes = 100
        large_graph = nk.Graph(n_nodes)
        
        # Create random edges
        np.random.seed(42)
        for _ in range(200):
            i, j = np.random.choice(n_nodes, 2, replace=False)
            if not large_graph.hasEdge(i, j):
                large_graph.addEdge(i, j)
        
        mapper = IDMapper()
        for i in range(n_nodes):
            mapper.add_mapping(f"node_{i:03d}", i)
        
        # Should complete without errors
        result = extract_centrality(
            large_graph,
            mapper,
            metrics=["degree", "betweenness"],
            n_jobs=2
        )
        
        assert result.shape == (n_nodes, 3)
        assert len(result["node_id"].unique()) == n_nodes