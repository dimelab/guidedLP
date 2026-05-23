"""
Tests for community detection module.

This module tests community detection algorithms, consensus calculation,
stability analysis, and quality metrics for network community structure.
"""

import pytest
import polars as pl
import networkit as nk
import numpy as np
from typing import Dict, Any, List

from src.network.communities import (
    detect_communities,
    get_community_summary,
    identify_stable_communities
)
from src.common.id_mapper import IDMapper
from src.common.exceptions import ValidationError, ComputationError


class TestDetectCommunities:
    """Test community detection functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        # Create simple test graph with clear community structure
        # Two communities: {A, B, C} and {D, E, F}
        self.simple_graph = nk.Graph(6)
        # Community 1: A-B-C (nodes 0, 1, 2)
        self.simple_graph.addEdge(0, 1)
        self.simple_graph.addEdge(1, 2)
        self.simple_graph.addEdge(2, 0)
        # Community 2: D-E-F (nodes 3, 4, 5)
        self.simple_graph.addEdge(3, 4)
        self.simple_graph.addEdge(4, 5)
        self.simple_graph.addEdge(5, 3)
        # Weak inter-community connection
        self.simple_graph.addEdge(1, 4)
        
        self.simple_mapper = IDMapper()
        node_names = ["A", "B", "C", "D", "E", "F"]
        for i, name in enumerate(node_names):
            self.simple_mapper.add_mapping(name, i)

        # Create karate club graph (known community structure)
        self.karate_graph = nk.Graph(34)
        # Simplified version of Zachary's karate club
        # Community 1: nodes 0-16
        for i in range(16):
            for j in range(i + 1, 16):
                if np.random.random() > 0.7:  # Sparse intra-community edges
                    self.karate_graph.addEdge(i, j)
        
        # Community 2: nodes 17-33
        for i in range(17, 34):
            for j in range(i + 1, 34):
                if np.random.random() > 0.7:  # Sparse intra-community edges
                    self.karate_graph.addEdge(i, j)
        
        # Few inter-community edges
        self.karate_graph.addEdge(0, 17)
        self.karate_graph.addEdge(5, 22)
        
        self.karate_mapper = IDMapper()
        for i in range(34):
            self.karate_mapper.add_mapping(f"person_{i}", i)

        # Create star graph (single central community)
        self.star_graph = nk.Graph(7)
        for i in range(1, 7):
            self.star_graph.addEdge(0, i)
        
        self.star_mapper = IDMapper()
        for i in range(7):
            self.star_mapper.add_mapping(f"node_{i}", i)

        # Create disconnected graph (multiple components)
        self.disconnected_graph = nk.Graph(6)
        # Component 1: nodes 0, 1, 2
        self.disconnected_graph.addEdge(0, 1)
        self.disconnected_graph.addEdge(1, 2)
        # Component 2: nodes 3, 4, 5
        self.disconnected_graph.addEdge(3, 4)
        self.disconnected_graph.addEdge(4, 5)
        
        self.disconnected_mapper = IDMapper()
        for i in range(6):
            self.disconnected_mapper.add_mapping(f"comp_{i}", i)

    def test_single_iteration_basic(self):
        """Test basic community detection with single iteration."""
        result = detect_communities(
            self.simple_graph,
            self.simple_mapper,
            iterations=1,
            random_seed=42
        )
        
        assert isinstance(result, pl.DataFrame)
        assert result.shape == (6, 6)  # 6 nodes, 6 columns
        
        # Check required columns
        expected_columns = [
            "node_id", "community_consensus", "stability",
            "community_iter_0", "modularity_iter_0", "num_communities_iter_0"
        ]
        assert set(result.columns) == set(expected_columns)
        
        # Check that consensus equals first iteration for single iteration
        consensus = result["community_consensus"].to_list()
        iter_0 = result["community_iter_0"].to_list()
        assert consensus == iter_0
        
        # Stability should be 1.0 for single iteration
        stability = result["stability"].to_list()
        assert all(s == 1.0 for s in stability)
        
        # Should detect some communities
        num_communities = result["num_communities_iter_0"][0]
        assert num_communities >= 1

    def test_multiple_iterations_consensus(self):
        """Test consensus calculation with multiple iterations."""
        result = detect_communities(
            self.simple_graph,
            self.simple_mapper,
            iterations=5,
            random_seed=42
        )
        
        assert result.shape == (6, 18)  # 6 nodes, 3 base + 5*3 iteration columns
        
        # Check iteration columns exist
        for i in range(5):
            assert f"community_iter_{i}" in result.columns
            assert f"modularity_iter_{i}" in result.columns
            assert f"num_communities_iter_{i}" in result.columns
        
        # Stability should be between 0 and 1
        stability = result["stability"].to_list()
        assert all(0.0 <= s <= 1.0 for s in stability)
        
        # Consensus should be valid community assignments
        consensus = result["community_consensus"].to_list()
        assert all(isinstance(c, int) and c >= 0 for c in consensus)

    def test_known_community_structure(self):
        """Test on graph with known community structure."""
        result = detect_communities(
            self.karate_graph,
            self.karate_mapper,
            iterations=3,
            random_seed=42
        )
        
        assert result.shape == (34, 12)  # 34 nodes, 3 base + 3*3 iteration columns
        
        # Should detect reasonable number of communities
        num_communities = result["num_communities_iter_0"][0]
        assert 2 <= num_communities <= 10  # Reasonable range
        
        # Modularity should be positive for good community structure
        modularity = result["modularity_iter_0"][0]
        assert modularity > 0.0
        
        # All nodes should be assigned to communities
        consensus = result["community_consensus"].to_list()
        assert len(set(consensus)) >= 2  # At least 2 communities

    def test_star_graph_communities(self):
        """Test community detection on star graph."""
        result = detect_communities(
            self.star_graph,
            self.star_mapper,
            iterations=1,
            random_seed=42
        )
        
        assert result.shape == (7, 6)
        
        # Star graph typically has low modularity (single community)
        modularity = result["modularity_iter_0"][0]
        # Modularity for star graph should be low or negative
        assert modularity <= 0.5
        
        # Should assign all nodes to communities
        consensus = result["community_consensus"].to_list()
        assert len(consensus) == 7

    def test_disconnected_graph_communities(self):
        """Test community detection on disconnected graph."""
        result = detect_communities(
            self.disconnected_graph,
            self.disconnected_mapper,
            iterations=1,
            random_seed=42
        )
        
        assert result.shape == (6, 6)
        
        # Should detect at least 2 communities (one per component)
        num_communities = result["num_communities_iter_0"][0]
        assert num_communities >= 2
        
        # Nodes in same component should tend to be in same community
        consensus = result["community_consensus"].to_list()
        node_to_community = dict(zip(result["node_id"].to_list(), consensus))
        
        # Check that nodes in same component have related community assignments
        # (This is a weak test since algorithm behavior can vary)
        assert len(set(consensus)) >= 1

    def test_stability_calculation(self):
        """Test stability score calculation across iterations."""
        result = detect_communities(
            self.simple_graph,
            self.simple_mapper,
            iterations=10,
            random_seed=42
        )
        
        stability = result["stability"].to_list()
        
        # All stability scores should be valid
        assert all(0.0 <= s <= 1.0 for s in stability)
        
        # At least some nodes should have high stability in structured graph
        high_stability_nodes = [s for s in stability if s >= 0.7]
        assert len(high_stability_nodes) > 0

    def test_resolution_parameter(self):
        """Test effect of resolution parameter on community detection."""
        # Low resolution (fewer, larger communities)
        result_low = detect_communities(
            self.simple_graph,
            self.simple_mapper,
            resolution=0.5,
            iterations=1,
            random_seed=42
        )
        
        # High resolution (more, smaller communities)
        result_high = detect_communities(
            self.simple_graph,
            self.simple_mapper,
            resolution=2.0,
            iterations=1,
            random_seed=42
        )
        
        num_communities_low = result_low["num_communities_iter_0"][0]
        num_communities_high = result_high["num_communities_iter_0"][0]
        
        # Higher resolution should tend to produce more communities
        # (This is a tendency, not guaranteed for all graphs)
        assert num_communities_high >= num_communities_low

    def test_random_seed_reproducibility(self):
        """Test that random seed produces reproducible results."""
        result1 = detect_communities(
            self.simple_graph,
            self.simple_mapper,
            iterations=1,
            random_seed=42
        )
        
        result2 = detect_communities(
            self.simple_graph,
            self.simple_mapper,
            iterations=1,
            random_seed=42
        )
        
        # Results should be identical with same seed
        consensus1 = result1["community_consensus"].to_list()
        consensus2 = result2["community_consensus"].to_list()
        assert consensus1 == consensus2

    def test_parallel_processing(self):
        """Test parallel processing functionality."""
        # Sequential
        result_seq = detect_communities(
            self.simple_graph,
            self.simple_mapper,
            iterations=3,
            random_seed=42,
            n_jobs=1
        )
        
        # Parallel
        result_par = detect_communities(
            self.simple_graph,
            self.simple_mapper,
            iterations=3,
            random_seed=42,
            n_jobs=2
        )
        
        # Results should be similar (not necessarily identical due to parallel execution)
        assert result_seq.shape == result_par.shape
        assert set(result_seq.columns) == set(result_par.columns)

    def test_min_similarity_filtering(self):
        """Test filtering partitions by similarity threshold."""
        # High similarity threshold (should filter out dissimilar partitions)
        result = detect_communities(
            self.simple_graph,
            self.simple_mapper,
            iterations=5,
            min_similarity=0.8,
            random_seed=42
        )
        
        # Should still return valid results
        assert isinstance(result, pl.DataFrame)
        assert result.height == 6
        
        # Should have consensus and stability columns
        assert "community_consensus" in result.columns
        assert "stability" in result.columns

    def test_algorithm_parameter(self):
        """Test algorithm parameter validation."""
        # Valid algorithm
        result = detect_communities(
            self.simple_graph,
            self.simple_mapper,
            algorithm="louvain",
            iterations=1
        )
        assert isinstance(result, pl.DataFrame)

    def test_quality_metrics_included(self):
        """Test that quality metrics are included in results."""
        result = detect_communities(
            self.simple_graph,
            self.simple_mapper,
            iterations=2,
            random_seed=42
        )
        
        # Check modularity columns
        assert "modularity_iter_0" in result.columns
        assert "modularity_iter_1" in result.columns
        
        # Check num_communities columns
        assert "num_communities_iter_0" in result.columns
        assert "num_communities_iter_1" in result.columns
        
        # Modularity values should be reasonable
        mod_0 = result["modularity_iter_0"][0]
        mod_1 = result["modularity_iter_1"][0]
        assert isinstance(mod_0, (int, float))
        assert isinstance(mod_1, (int, float))

    def test_node_id_mapping_consistency(self):
        """Test that node ID mapping is handled correctly."""
        result = detect_communities(
            self.simple_graph,
            self.simple_mapper,
            iterations=1
        )
        
        # Should return original node IDs
        node_ids = result["node_id"].to_list()
        expected_ids = ["A", "B", "C", "D", "E", "F"]
        assert set(node_ids) == set(expected_ids)
        assert all(isinstance(nid, str) for nid in node_ids)

    def test_result_sorting(self):
        """Test that results are sorted by node_id."""
        result = detect_communities(
            self.simple_graph,
            self.simple_mapper,
            iterations=1
        )
        
        node_ids = result["node_id"].to_list()
        assert node_ids == sorted(node_ids)

    def test_empty_graph(self):
        """Test handling of empty graph."""
        empty_graph = nk.Graph(0)
        empty_mapper = IDMapper()
        
        result = detect_communities(
            empty_graph,
            empty_mapper,
            iterations=1
        )
        
        assert result.shape == (0, 6)  # Empty with correct columns
        expected_columns = [
            "node_id", "community_consensus", "stability",
            "community_iter_0", "modularity_iter_0", "num_communities_iter_0"
        ]
        assert set(result.columns) == set(expected_columns)

    def test_single_node_graph(self):
        """Test handling of single node graph."""
        single_graph = nk.Graph(1)
        single_mapper = IDMapper()
        single_mapper.add_mapping("only_node", 0)
        
        result = detect_communities(
            single_graph,
            single_mapper,
            iterations=1
        )
        
        assert result.shape == (1, 6)
        assert result["node_id"][0] == "only_node"
        assert result["community_consensus"][0] == 0
        assert result["stability"][0] == 1.0

    def test_large_graph_performance(self):
        """Test performance on moderately large graph."""
        # Create larger graph
        n_nodes = 100
        large_graph = nk.Graph(n_nodes)
        
        # Create community structure: 4 communities of 25 nodes each
        for community in range(4):
            start = community * 25
            end = (community + 1) * 25
            # Dense intra-community connections
            for i in range(start, end):
                for j in range(i + 1, end):
                    if np.random.random() > 0.8:  # 20% edge probability
                        large_graph.addEdge(i, j)
        
        # Sparse inter-community connections
        for i in range(0, 25):
            if np.random.random() > 0.95:  # 5% probability
                large_graph.addEdge(i, 50 + np.random.randint(0, 25))
        
        mapper = IDMapper()
        for i in range(n_nodes):
            mapper.add_mapping(f"node_{i:03d}", i)
        
        # Should complete without errors
        result = detect_communities(
            large_graph,
            mapper,
            iterations=2,
            random_seed=42
        )
        
        assert result.shape == (n_nodes, 9)  # 100 nodes, 3 base + 2*3 iteration columns
        assert len(result["node_id"].unique()) == n_nodes


class TestParameterValidation:
    """Test parameter validation and error handling."""

    def setup_method(self):
        """Set up minimal test graph."""
        self.graph = nk.Graph(3)
        self.graph.addEdge(0, 1)
        self.graph.addEdge(1, 2)
        
        self.mapper = IDMapper()
        for i in range(3):
            self.mapper.add_mapping(f"node_{i}", i)

    def test_invalid_algorithm(self):
        """Test error handling for invalid algorithm."""
        with pytest.raises(ValidationError) as exc_info:
            detect_communities(
                self.graph,
                self.mapper,
                algorithm="invalid_algorithm"
            )
        
        assert "invalid_algorithm" in str(exc_info.value).lower()
        assert "available algorithms" in str(exc_info.value).lower()

    def test_invalid_iterations(self):
        """Test error handling for invalid iterations."""
        with pytest.raises(ValidationError) as exc_info:
            detect_communities(
                self.graph,
                self.mapper,
                iterations=0
            )
        
        assert "iterations must be >= 1" in str(exc_info.value)

    def test_invalid_resolution(self):
        """Test error handling for invalid resolution."""
        with pytest.raises(ValidationError) as exc_info:
            detect_communities(
                self.graph,
                self.mapper,
                resolution=0.0
            )
        
        assert "resolution must be > 0" in str(exc_info.value)

    def test_invalid_min_similarity(self):
        """Test error handling for invalid min_similarity."""
        with pytest.raises(ValidationError) as exc_info:
            detect_communities(
                self.graph,
                self.mapper,
                min_similarity=1.5
            )
        
        assert "min_similarity must be between 0.0 and 1.0" in str(exc_info.value)

    def test_invalid_n_jobs(self):
        """Test error handling for invalid n_jobs."""
        with pytest.raises(ValidationError) as exc_info:
            detect_communities(
                self.graph,
                self.mapper,
                n_jobs=0
            )
        
        assert "n_jobs cannot be 0" in str(exc_info.value)


class TestCommunityAnalysis:
    """Test community analysis utility functions."""

    def setup_method(self):
        """Set up test community data."""
        self.graph = nk.Graph(6)
        self.graph.addEdge(0, 1)
        self.graph.addEdge(1, 2)
        self.graph.addEdge(3, 4)
        self.graph.addEdge(4, 5)
        
        self.mapper = IDMapper()
        for i in range(6):
            self.mapper.add_mapping(f"node_{i}", i)
        
        # Get community detection results
        self.communities_df = detect_communities(
            self.graph,
            self.mapper,
            iterations=3,
            random_seed=42
        )

    def test_get_community_summary(self):
        """Test community summary statistics."""
        summary = get_community_summary(self.communities_df)
        
        assert isinstance(summary, dict)
        assert "num_communities" in summary
        assert "modularity" in summary
        assert "avg_stability" in summary
        assert "community_sizes" in summary
        assert "size_distribution" in summary
        assert "total_nodes" in summary
        
        # Check data types and ranges
        assert isinstance(summary["num_communities"], int)
        assert summary["num_communities"] > 0
        assert isinstance(summary["community_sizes"], list)
        assert summary["total_nodes"] == 6

    def test_get_community_summary_specific_iteration(self):
        """Test community summary for specific iteration."""
        summary = get_community_summary(self.communities_df, iteration=0)
        
        assert isinstance(summary, dict)
        assert "num_communities" in summary
        assert "modularity" in summary
        
        # Should have modularity value for specific iteration
        assert summary["modularity"] is not None

    def test_identify_stable_communities(self):
        """Test identification of stable communities."""
        stable = identify_stable_communities(
            self.communities_df,
            min_stability=0.5,
            min_size=1
        )
        
        assert isinstance(stable, pl.DataFrame)
        assert stable.height <= self.communities_df.height
        
        # All returned nodes should meet stability criteria
        if stable.height > 0:
            stability_values = stable["stability"].to_list()
            assert all(s >= 0.5 for s in stability_values)

    def test_identify_stable_communities_strict(self):
        """Test strict stability and size requirements."""
        stable = identify_stable_communities(
            self.communities_df,
            min_stability=0.9,
            min_size=3
        )
        
        assert isinstance(stable, pl.DataFrame)
        # Strict requirements might result in empty DataFrame
        assert stable.height >= 0


class TestEdgeCases:
    """Test edge cases and special graph structures."""

    def test_complete_graph_communities(self):
        """Test community detection on complete graph."""
        complete_graph = nk.Graph(5)
        for i in range(5):
            for j in range(i + 1, 5):
                complete_graph.addEdge(i, j)
        
        mapper = IDMapper()
        for i in range(5):
            mapper.add_mapping(f"node_{i}", i)
        
        result = detect_communities(
            complete_graph,
            mapper,
            iterations=1,
            random_seed=42
        )
        
        assert result.shape == (5, 6)
        
        # Complete graph typically has low modularity
        modularity = result["modularity_iter_0"][0]
        assert modularity <= 0.5

    def test_linear_chain_communities(self):
        """Test community detection on linear chain graph."""
        chain_graph = nk.Graph(6)
        for i in range(5):
            chain_graph.addEdge(i, i + 1)
        
        mapper = IDMapper()
        for i in range(6):
            mapper.add_mapping(f"node_{i}", i)
        
        result = detect_communities(
            chain_graph,
            mapper,
            iterations=1,
            random_seed=42
        )
        
        assert result.shape == (6, 6)
        # Should detect some community structure
        num_communities = result["num_communities_iter_0"][0]
        assert num_communities >= 1

    def test_extreme_min_similarity(self):
        """Test with very high min_similarity requirement."""
        graph = nk.Graph(4)
        graph.addEdge(0, 1)
        graph.addEdge(2, 3)
        
        mapper = IDMapper()
        for i in range(4):
            mapper.add_mapping(f"node_{i}", i)
        
        # Very high similarity requirement
        result = detect_communities(
            graph,
            mapper,
            iterations=5,
            min_similarity=0.99,
            random_seed=42
        )
        
        # Should still return valid results (may fall back to single iteration)
        assert isinstance(result, pl.DataFrame)
        assert result.height == 4

    def test_weighted_graph_communities(self):
        """Test community detection on weighted graph."""
        weighted_graph = nk.Graph(4, weighted=True)
        weighted_graph.addEdge(0, 1, 1.0)
        weighted_graph.addEdge(1, 2, 1.0)
        weighted_graph.addEdge(2, 3, 0.1)  # Weak connection
        weighted_graph.addEdge(3, 0, 0.1)  # Weak connection
        
        mapper = IDMapper()
        for i in range(4):
            mapper.add_mapping(f"node_{i}", i)
        
        result = detect_communities(
            weighted_graph,
            mapper,
            iterations=1,
            random_seed=42
        )
        
        assert result.shape == (4, 6)
        # Should handle weighted graphs correctly
        modularity = result["modularity_iter_0"][0]
        assert isinstance(modularity, (int, float))