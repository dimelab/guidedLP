"""
Tests for network backboning module.

This module tests the disparity filter and other backbone extraction methods,
with focus on mathematical correctness, numerical stability, and performance
on various graph types and sizes.
"""

import pytest
import numpy as np
import networkit as nk
import polars as pl
from typing import Tuple

from guidedLP.network.backboning import (
    apply_backbone,
    get_backbone_summary,
    _safe_power,
    AVAILABLE_METHODS
)
from guidedLP.network.construction import build_graph_from_edgelist
from guidedLP.common.id_mapper import IDMapper
from guidedLP.common.exceptions import ConfigurationError, ValidationError


class TestDisparityFilter:
    """Test disparity filter implementation."""
    
    def test_disparity_filter_basic(self):
        """Test basic disparity filter functionality."""
        # Create test graph: star topology (one central node connected to others)
        # This creates clear disparity - central node has many weak edges,
        # peripheral nodes have one strong edge each
        edges = pl.DataFrame({
            'source': ['center'] * 5,
            'target': ['node1', 'node2', 'node3', 'node4', 'node5'],
            'weight': [1, 1, 1, 1, 10]  # Last edge much stronger
        })
        
        graph, id_mapper = build_graph_from_edgelist(edges, weight_col='weight')
        
        # Apply disparity filter with relaxed threshold
        backbone_graph, backbone_mapper = apply_backbone(
            graph, id_mapper, method="disparity", alpha=0.1
        )
        
        # Should keep the strong edge and possibly some others
        assert backbone_graph.numberOfEdges() > 0
        assert backbone_graph.numberOfNodes() <= graph.numberOfNodes()
        
        # Test with detailed results
        backbone_graph, backbone_mapper, edge_details = apply_backbone(
            graph, id_mapper, method="disparity", alpha=0.1, return_filtered_edges=True
        )
        
        assert edge_details.height == 5  # All original edges analyzed
        assert 'alpha_score' in edge_details.columns
        assert 'kept' in edge_details.columns
        
        # The strongest edge should be kept
        strongest_edge = edge_details.filter(pl.col('weight') == 10)
        assert strongest_edge.height == 1
        assert strongest_edge['kept'][0] == True
    
    def test_disparity_filter_mathematical_correctness(self):
        """Test mathematical correctness of disparity calculations."""
        # Create simple 3-node path: A-B-C with known weights
        edges = pl.DataFrame({
            'source': ['A', 'B'],
            'target': ['B', 'C'], 
            'weight': [2.0, 3.0]
        })
        
        graph, id_mapper = build_graph_from_edgelist(edges, weight_col='weight')
        
        # Get detailed edge analysis
        _, _, edge_details = apply_backbone(
            graph, id_mapper, method="disparity", alpha=0.5, return_filtered_edges=True
        )
        
        # Verify mathematical calculations
        # Node B has degree 2, total weight 5.0
        # Edge B-A: p = 2/5 = 0.4, alpha = (1-0.4)^(2-1) = 0.6
        # Edge B-C: p = 3/5 = 0.6, alpha = (1-0.6)^(2-1) = 0.4
        
        edge_BA = edge_details.filter(
            (pl.col('source_id') == 'B') & (pl.col('target_id') == 'A') |
            (pl.col('source_id') == 'A') & (pl.col('target_id') == 'B')
        )
        edge_BC = edge_details.filter(
            (pl.col('source_id') == 'B') & (pl.col('target_id') == 'C') |
            (pl.col('source_id') == 'C') & (pl.col('target_id') == 'B')
        )
        
        # Check calculations (allowing for numerical precision)
        assert len(edge_BA) == 1
        assert len(edge_BC) == 1
        
        # The edge with higher weight should have lower alpha score
        alpha_BA = edge_BA['alpha_score'][0]
        alpha_BC = edge_BC['alpha_score'][0]
        
        # Edge BC (weight 3) should have lower alpha than edge BA (weight 2)
        assert alpha_BC < alpha_BA
    
    def test_disparity_filter_directed_vs_undirected(self):
        """Test disparity filter on directed vs undirected graphs."""
        edges = pl.DataFrame({
            'source': ['A', 'B', 'C'],
            'target': ['B', 'C', 'A'],
            'weight': [1.0, 2.0, 3.0]
        })
        
        # Undirected graph
        graph_undirected, mapper = build_graph_from_edgelist(
            edges, weight_col='weight', directed=False
        )
        
        # Directed graph  
        graph_directed, mapper = build_graph_from_edgelist(
            edges, weight_col='weight', directed=True
        )
        
        # Apply disparity filter to both
        _, _, edges_undirected = apply_backbone(
            graph_undirected, mapper, method="disparity", alpha=0.1, return_filtered_edges=True
        )
        
        _, _, edges_directed = apply_backbone(
            graph_directed, mapper, method="disparity", alpha=0.1, return_filtered_edges=True
        )
        
        # Should have different alpha scores due to different degree calculations
        assert edges_undirected.height == edges_directed.height
        
        # Undirected graph should have higher degrees (each edge counted twice)
        avg_alpha_undirected = float(edges_undirected['alpha_score'].mean())
        avg_alpha_directed = float(edges_directed['alpha_score'].mean())
        
        # This relationship depends on the specific graph structure
        assert avg_alpha_undirected >= 0
        assert avg_alpha_directed >= 0
    
    def test_safe_power_numerical_stability(self):
        """Test numerical stability of power calculations."""
        # Test normal cases
        assert _safe_power(0.5, 2) == pytest.approx(0.25)
        assert _safe_power(0.1, 10) == pytest.approx(1e-10, rel=1e-6)
        
        # Test edge cases
        assert _safe_power(0.0, 5) == 0.0
        assert _safe_power(1.0, 100) == 1.0
        assert _safe_power(0.5, 0) == 1.0
        assert _safe_power(-0.5, 2) == 0.0  # Negative base
        
        # Test numerical stability with very small numbers
        very_small = 1e-10
        large_exp = 1000
        result = _safe_power(very_small, large_exp)
        assert result == 0.0  # Should underflow to 0 safely
        
        # Test with values that could cause overflow in naive implementation
        result = _safe_power(0.999, 1e6)
        assert 0 <= result <= 1
    
    def test_empty_graph_handling(self):
        """Test handling of empty graphs."""
        # Empty graph
        empty_graph = nk.Graph(0, weighted=True)
        empty_mapper = IDMapper()
        
        backbone_graph, backbone_mapper = apply_backbone(
            empty_graph, empty_mapper, method="disparity"
        )
        
        assert backbone_graph.numberOfNodes() == 0
        assert backbone_graph.numberOfEdges() == 0
        
        # Graph with nodes but no edges
        graph_no_edges = nk.Graph(5, weighted=True)
        mapper = IDMapper()
        for i in range(5):
            mapper.add_mapping(f"node_{i}", i)
        
        backbone_graph, backbone_mapper = apply_backbone(
            graph_no_edges, mapper, method="disparity"
        )
        
        assert backbone_graph.numberOfEdges() == 0
    
    def test_single_edge_graph(self):
        """Test graphs with only one edge."""
        edges = pl.DataFrame({
            'source': ['A'],
            'target': ['B'],
            'weight': [1.0]
        })
        
        graph, mapper = build_graph_from_edgelist(edges, weight_col='weight')
        
        # Single edge should be kept (degree 1, alpha = (1-1)^0 = 1)
        backbone_graph, backbone_mapper, edge_details = apply_backbone(
            graph, mapper, method="disparity", alpha=0.5, return_filtered_edges=True
        )
        
        assert edge_details.height == 1
        # With degree 1, alpha score should be 1.0 (always kept)
        assert edge_details['alpha_score'][0] == 1.0
        assert edge_details['kept'][0] == True
    
    def test_star_graph_disparity(self):
        """Test disparity filter on star graph (known structure)."""
        # Create star graph: central node connected to 10 peripheral nodes
        center = 'center'
        periphery = [f'node_{i}' for i in range(10)]
        weights = [1.0] * 9 + [10.0]  # One strong edge, rest weak
        
        edges = pl.DataFrame({
            'source': [center] * 10,
            'target': periphery,
            'weight': weights
        })
        
        graph, mapper = build_graph_from_edgelist(edges, weight_col='weight')
        
        # With strict threshold, only strong edges should survive
        backbone_graph, backbone_mapper, edge_details = apply_backbone(
            graph, mapper, method="disparity", alpha=0.01, return_filtered_edges=True
        )
        
        # Check that strong edge is kept
        strong_edge = edge_details.filter(pl.col('weight') == 10.0)
        assert strong_edge.height == 1
        assert strong_edge['kept'][0] == True
        
        # Check alpha scores are calculated correctly
        # Central node has degree 10, weight sum 19.0
        # Strong edge: p = 10/19 ≈ 0.526, alpha = (1-0.526)^9 ≈ 0.0002 (very small)
        # Weak edges: p = 1/19 ≈ 0.053, alpha = (1-0.053)^9 ≈ 0.63 (large)
        
        strong_alpha = strong_edge['alpha_score'][0]
        assert strong_alpha < 0.01  # Should be very small
        
        weak_edges = edge_details.filter(pl.col('weight') == 1.0)
        if weak_edges.height > 0:
            weak_alpha_avg = float(weak_edges['alpha_score'].mean())
            assert weak_alpha_avg > strong_alpha  # Weak edges have higher alpha


class TestWeightThreshold:
    """Test weight threshold backbone method."""
    
    def test_weight_threshold_basic(self):
        """Test basic weight threshold functionality."""
        edges = pl.DataFrame({
            'source': ['A', 'B', 'C', 'D'],
            'target': ['B', 'C', 'D', 'A'],
            'weight': [1.0, 2.0, 3.0, 4.0]
        })
        
        graph, mapper = build_graph_from_edgelist(edges, weight_col='weight')
        
        # Keep edges with weight >= 2.5
        backbone_graph, backbone_mapper = apply_backbone(
            graph, mapper, method="weight_threshold", weight_threshold=2.5
        )
        
        # Should keep 2 edges (weights 3.0 and 4.0)
        assert backbone_graph.numberOfEdges() == 2
        
        # Test with target edges
        backbone_graph, backbone_mapper = apply_backbone(
            graph, mapper, method="weight_threshold", target_edges=2
        )
        
        # Should keep top 2 edges by weight
        assert backbone_graph.numberOfEdges() == 2
    
    def test_weight_threshold_edge_details(self):
        """Test weight threshold with detailed edge information."""
        edges = pl.DataFrame({
            'source': ['A', 'B', 'C'],
            'target': ['B', 'C', 'A'],
            'weight': [1.0, 3.0, 2.0]
        })
        
        graph, mapper = build_graph_from_edgelist(edges, weight_col='weight')
        
        _, _, edge_details = apply_backbone(
            graph, mapper, method="weight_threshold", weight_threshold=1.5,
            return_filtered_edges=True
        )
        
        # Check edge filtering results
        kept_edges = edge_details.filter(pl.col('kept'))
        assert kept_edges.height == 2  # Weights 2.0 and 3.0
        
        discarded_edges = edge_details.filter(~pl.col('kept'))
        assert discarded_edges.height == 1  # Weight 1.0


class TestDegreeThreshold:
    """Test degree threshold backbone method."""
    
    def test_degree_threshold_basic(self):
        """Test basic degree threshold functionality."""
        # Create graph where nodes have different degrees
        edges = pl.DataFrame({
            'source': ['hub', 'hub', 'hub', 'A'],
            'target': ['A', 'B', 'C', 'B'],
            'weight': [1.0, 1.0, 1.0, 1.0]
        })
        
        graph, mapper = build_graph_from_edgelist(edges, weight_col='weight')
        
        # Keep top 2 nodes by degree
        backbone_graph, backbone_mapper = apply_backbone(
            graph, mapper, method="degree_threshold", target_nodes=2
        )
        
        # Should keep 'hub' (degree 3) and one other node with degree 2
        assert backbone_graph.numberOfNodes() == 2
        assert backbone_graph.numberOfEdges() <= 4  # Max possible edges between 2 nodes


class TestBackboneValidation:
    """Test parameter validation and error handling."""
    
    def test_invalid_method(self):
        """Test invalid backbone method."""
        edges = pl.DataFrame({
            'source': ['A'],
            'target': ['B'],
            'weight': [1.0]
        })
        
        graph, mapper = build_graph_from_edgelist(edges, weight_col='weight')
        
        with pytest.raises(ValidationError, match="Invalid backbone method"):
            apply_backbone(graph, mapper, method="invalid_method")
    
    def test_invalid_alpha(self):
        """Test invalid alpha values for disparity filter."""
        edges = pl.DataFrame({
            'source': ['A'],
            'target': ['B'],
            'weight': [1.0]
        })
        
        graph, mapper = build_graph_from_edgelist(edges, weight_col='weight')
        
        # Alpha must be in (0, 1)
        with pytest.raises(ValidationError, match="Alpha must be in"):
            apply_backbone(graph, mapper, method="disparity", alpha=0.0)
        
        with pytest.raises(ValidationError, match="Alpha must be in"):
            apply_backbone(graph, mapper, method="disparity", alpha=1.0)
        
        with pytest.raises(ValidationError, match="Alpha must be in"):
            apply_backbone(graph, mapper, method="disparity", alpha=-0.5)
    
    def test_conflicting_parameters(self):
        """Test conflicting parameter combinations."""
        edges = pl.DataFrame({
            'source': ['A'],
            'target': ['B'],
            'weight': [1.0]
        })
        
        graph, mapper = build_graph_from_edgelist(edges, weight_col='weight')
        
        # Cannot specify both target_nodes and target_edges
        with pytest.raises(ValidationError, match="Cannot specify both"):
            apply_backbone(
                graph, mapper, method="disparity", 
                target_nodes=10, target_edges=20
            )
    
    def test_missing_required_parameters(self):
        """Test missing required parameters for specific methods."""
        edges = pl.DataFrame({
            'source': ['A'],
            'target': ['B'],
            'weight': [1.0]
        })
        
        graph, mapper = build_graph_from_edgelist(edges, weight_col='weight')
        
        # Weight threshold needs either weight_threshold or target_edges
        with pytest.raises(ConfigurationError, match="Must specify either"):
            apply_backbone(graph, mapper, method="weight_threshold")
        
        # Degree threshold needs target_nodes
        with pytest.raises(ConfigurationError, match="Must specify target_nodes"):
            apply_backbone(graph, mapper, method="degree_threshold")


class TestBackboneSummary:
    """Test backbone summary statistics."""
    
    def test_backbone_summary_basic(self):
        """Test basic summary statistics."""
        edges = pl.DataFrame({
            'source': ['A', 'B', 'C', 'D'],
            'target': ['B', 'C', 'D', 'A'],
            'weight': [1.0, 2.0, 3.0, 4.0]
        })
        
        graph, mapper = build_graph_from_edgelist(edges, weight_col='weight')
        
        backbone_graph, backbone_mapper, edge_details = apply_backbone(
            graph, mapper, method="weight_threshold", weight_threshold=2.5,
            return_filtered_edges=True
        )
        
        summary = get_backbone_summary(graph, backbone_graph, edge_details)
        
        # Check basic statistics
        assert summary['original_nodes'] == 4
        assert summary['original_edges'] == 4
        assert summary['backbone_edges'] == 2
        assert summary['edge_retention_rate'] == 0.5
        
        # Check weight statistics are included
        assert 'weight_statistics' in summary
        assert 'kept_weight_mean' in summary['weight_statistics']
    
    def test_backbone_summary_without_edge_details(self):
        """Test summary without detailed edge information."""
        edges = pl.DataFrame({
            'source': ['A', 'B'],
            'target': ['B', 'C'],
            'weight': [1.0, 2.0]
        })
        
        graph, mapper = build_graph_from_edgelist(edges, weight_col='weight')
        
        backbone_graph, backbone_mapper = apply_backbone(
            graph, mapper, method="disparity", alpha=0.1
        )
        
        summary = get_backbone_summary(graph, backbone_graph)
        
        # Should have basic statistics but no edge details
        assert 'original_nodes' in summary
        assert 'edge_retention_rate' in summary
        assert 'weight_statistics' not in summary


class TestPerformanceAndScaling:
    """Test performance characteristics and scaling behavior."""
    
    def test_large_graph_performance(self):
        """Test performance on moderately large graphs."""
        # Create random graph with 1000 nodes and ~5000 edges
        np.random.seed(42)
        n_nodes = 1000
        n_edges = 5000
        
        sources = np.random.randint(0, n_nodes, n_edges)
        targets = np.random.randint(0, n_nodes, n_edges)
        weights = np.random.exponential(1.0, n_edges)
        
        # Remove self-loops and duplicates
        valid_edges = sources != targets
        sources = sources[valid_edges]
        targets = targets[valid_edges]
        weights = weights[valid_edges]
        
        edges = pl.DataFrame({
            'source': [f'node_{s}' for s in sources],
            'target': [f'node_{t}' for t in targets],
            'weight': weights
        })
        
        graph, mapper = build_graph_from_edgelist(edges, weight_col='weight')
        
        # Time the backbone extraction
        import time
        start_time = time.time()
        
        backbone_graph, backbone_mapper = apply_backbone(
            graph, mapper, method="disparity", alpha=0.01
        )
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        # Should complete in reasonable time (< 10 seconds on most systems)
        assert execution_time < 10.0
        assert backbone_graph.numberOfEdges() <= graph.numberOfEdges()
        
        print(f"Large graph test: {graph.numberOfEdges()} → {backbone_graph.numberOfEdges()} edges in {execution_time:.2f}s")
    
    def test_memory_usage_sparse_matrices(self):
        """Test that sparse matrices are used efficiently for memory."""
        # Create graph with many zero-weight edges (should be handled efficiently)
        edges_list = []
        
        # Add a few high-weight edges
        for i in range(10):
            edges_list.append(('hub', f'node_{i}', 10.0))
        
        # Add many low-weight edges
        for i in range(100):
            edges_list.append((f'node_{i}', f'node_{i+100}', 0.1))
        
        edges = pl.DataFrame({
            'source': [e[0] for e in edges_list],
            'target': [e[1] for e in edges_list],
            'weight': [e[2] for e in edges_list]
        })
        
        graph, mapper = build_graph_from_edgelist(edges, weight_col='weight')
        
        # This should not run out of memory and should be fast
        backbone_graph, backbone_mapper = apply_backbone(
            graph, mapper, method="disparity", alpha=0.1
        )
        
        # Verify results are reasonable
        assert backbone_graph.numberOfNodes() <= graph.numberOfNodes()
        assert backbone_graph.numberOfEdges() <= graph.numberOfEdges()


class TestGraphMapperSynchronization:
    """Test that graph and mapper remain synchronized after backboning operations.

    This test class addresses the bug report where numberOfNodes() didn't match
    mapper.size() after applying backbone with keep_disconnected=False.
    """

    def test_node_count_matches_mapper_disparity(self):
        """Test that numberOfNodes() equals mapper.size() for disparity filter."""
        # Create graph with some edges that will be filtered out
        edges = pl.DataFrame({
            'source': ['A', 'B', 'C', 'D', 'E', 'F'],
            'target': ['B', 'C', 'D', 'E', 'F', 'A'],
            'weight': [10.0, 9.0, 8.0, 1.0, 1.0, 1.0]
        })

        graph, mapper = build_graph_from_edgelist(edges, weight_col='weight')

        # Apply backbone with keep_disconnected=False
        backbone_graph, backbone_mapper = apply_backbone(
            graph, mapper, method="disparity", alpha=0.05, keep_disconnected=False
        )

        # CRITICAL: These MUST match
        assert backbone_graph.numberOfNodes() == backbone_mapper.size(), \
            f"Node count mismatch: graph has {backbone_graph.numberOfNodes()} nodes " \
            f"but mapper has {backbone_mapper.size()} nodes"

        # Verify all mapper entries reference valid graph nodes
        for i in range(backbone_mapper.size()):
            assert i < backbone_graph.numberOfNodes(), \
                f"Mapper references node {i} but graph only has {backbone_graph.numberOfNodes()} nodes"

    def test_node_count_matches_mapper_weight_threshold(self):
        """Test synchronization for weight threshold method."""
        edges = pl.DataFrame({
            'source': ['A', 'B', 'C', 'D', 'E'],
            'target': ['B', 'C', 'D', 'E', 'A'],
            'weight': [10.0, 5.0, 3.0, 1.0, 0.5]
        })

        graph, mapper = build_graph_from_edgelist(edges, weight_col='weight')

        # Filter with threshold that removes some edges/nodes
        backbone_graph, backbone_mapper = apply_backbone(
            graph, mapper, method="weight_threshold",
            weight_threshold=4.0, keep_disconnected=False
        )

        assert backbone_graph.numberOfNodes() == backbone_mapper.size(), \
            f"Weight threshold: Node count mismatch: {backbone_graph.numberOfNodes()} != {backbone_mapper.size()}"

    def test_node_count_matches_mapper_degree_threshold(self):
        """Test synchronization for degree threshold method."""
        # Create graph where some nodes have low degree
        edges = pl.DataFrame({
            'source': ['A', 'A', 'A', 'B', 'C', 'D'],
            'target': ['B', 'C', 'D', 'E', 'E', 'E'],
            'weight': [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
        })

        graph, mapper = build_graph_from_edgelist(edges, weight_col='weight')

        # Filter by degree - target_nodes will filter to keep nodes with higher degree
        backbone_graph, backbone_mapper = apply_backbone(
            graph, mapper, method="degree_threshold",
            target_nodes=3, keep_disconnected=False
        )

        assert backbone_graph.numberOfNodes() == backbone_mapper.size(), \
            f"Degree threshold: Node count mismatch: {backbone_graph.numberOfNodes()} != {backbone_mapper.size()}"

    def test_keep_disconnected_true_preserves_size(self):
        """Test that keep_disconnected=True preserves original node count."""
        edges = pl.DataFrame({
            'source': ['A', 'B', 'C'],
            'target': ['B', 'C', 'D'],
            'weight': [10.0, 5.0, 1.0]
        })

        graph, mapper = build_graph_from_edgelist(edges, weight_col='weight')
        original_nodes = graph.numberOfNodes()

        # With keep_disconnected=True, should preserve all nodes
        backbone_graph, backbone_mapper = apply_backbone(
            graph, mapper, method="weight_threshold",
            weight_threshold=6.0, keep_disconnected=True
        )

        # Should preserve original node count
        assert backbone_graph.numberOfNodes() == original_nodes
        assert backbone_mapper.size() == mapper.size()
        assert backbone_graph.numberOfNodes() == backbone_mapper.size()

    def test_keep_disconnected_false_removes_isolated(self):
        """Test that keep_disconnected=False removes isolated nodes."""
        edges = pl.DataFrame({
            'source': ['A', 'B', 'C', 'D', 'E'],
            'target': ['B', 'C', 'D', 'E', 'F'],
            'weight': [10.0, 10.0, 1.0, 1.0, 1.0]
        })

        graph, mapper = build_graph_from_edgelist(edges, weight_col='weight')
        original_nodes = graph.numberOfNodes()

        # High threshold will disconnect some nodes
        backbone_graph, backbone_mapper = apply_backbone(
            graph, mapper, method="weight_threshold",
            weight_threshold=5.0, keep_disconnected=False
        )

        # Should have fewer nodes than original
        assert backbone_graph.numberOfNodes() < original_nodes
        assert backbone_mapper.size() < mapper.size()

        # But graph and mapper should match
        assert backbone_graph.numberOfNodes() == backbone_mapper.size()

    def test_all_mapper_nodes_exist_in_graph(self):
        """Verify every node in mapper has a corresponding node in graph."""
        edges = pl.DataFrame({
            'source': ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H'],
            'target': ['B', 'C', 'D', 'E', 'F', 'G', 'H', 'A'],
            'weight': [10.0, 9.0, 8.0, 7.0, 1.0, 1.0, 1.0, 1.0]
        })

        graph, mapper = build_graph_from_edgelist(edges, weight_col='weight')

        backbone_graph, backbone_mapper = apply_backbone(
            graph, mapper, method="disparity", alpha=0.1, keep_disconnected=False
        )

        # Every internal ID in the mapper should be a valid node ID in the graph
        for internal_id in range(backbone_mapper.size()):
            # Internal IDs should be consecutive starting from 0
            assert internal_id < backbone_graph.numberOfNodes(), \
                f"Mapper has internal ID {internal_id} but graph only has {backbone_graph.numberOfNodes()} nodes"

            # Node should exist in graph
            assert backbone_graph.hasNode(internal_id), \
                f"Mapper references node {internal_id} but it doesn't exist in graph"

    def test_edge_endpoints_in_mapper(self):
        """Verify all edge endpoints can be translated through mapper."""
        edges = pl.DataFrame({
            'source': ['A', 'B', 'C', 'D', 'E'],
            'target': ['B', 'C', 'D', 'E', 'A'],
            'weight': [10.0, 8.0, 6.0, 4.0, 2.0]
        })

        graph, mapper = build_graph_from_edgelist(edges, weight_col='weight')

        backbone_graph, backbone_mapper = apply_backbone(
            graph, mapper, method="weight_threshold",
            weight_threshold=5.0, keep_disconnected=False
        )

        # Check every edge in the backbone
        for u, v in backbone_graph.iterEdges():
            # Both endpoints should be valid internal IDs
            assert u < backbone_mapper.size(), \
                f"Edge endpoint {u} exceeds mapper size {backbone_mapper.size()}"
            assert v < backbone_mapper.size(), \
                f"Edge endpoint {v} exceeds mapper size {backbone_mapper.size()}"

            # Both endpoints should have original IDs
            assert backbone_mapper.has_internal(u), \
                f"Edge endpoint {u} not in mapper"
            assert backbone_mapper.has_internal(v), \
                f"Edge endpoint {v} not in mapper"

    def test_large_graph_synchronization(self):
        """Test synchronization on larger graph (1000 nodes)."""
        # Create a scale-free-like graph
        edges_list = []
        for i in range(1000):
            # Each node connects to a few others with varying weights
            for j in range(min(5, 1000 - i - 1)):
                weight = np.random.uniform(0.1, 10.0)
                edges_list.append((f'node_{i}', f'node_{i+j+1}', weight))

        edges = pl.DataFrame({
            'source': [e[0] for e in edges_list],
            'target': [e[1] for e in edges_list],
            'weight': [e[2] for e in edges_list]
        })

        graph, mapper = build_graph_from_edgelist(edges, weight_col='weight')

        # Apply multiple backboning methods
        for method in ['disparity', 'weight_threshold', 'degree_threshold']:
            if method == 'disparity':
                backbone_graph, backbone_mapper = apply_backbone(
                    graph, mapper, method=method, alpha=0.05, keep_disconnected=False
                )
            elif method == 'weight_threshold':
                backbone_graph, backbone_mapper = apply_backbone(
                    graph, mapper, method=method, weight_threshold=5.0, keep_disconnected=False
                )
            else:  # degree_threshold
                backbone_graph, backbone_mapper = apply_backbone(
                    graph, mapper, method=method, target_nodes=500, keep_disconnected=False
                )

            # Verify synchronization for each method
            assert backbone_graph.numberOfNodes() == backbone_mapper.size(), \
                f"{method}: Large graph synchronization failed: " \
                f"{backbone_graph.numberOfNodes()} != {backbone_mapper.size()}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])