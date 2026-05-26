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
    get_backbone_statistics,
    _safe_power,
    AVAILABLE_METHODS,
)
from guidedLP.network.construction import build_graph_from_edgelist
from guidedLP.common.id_mapper import IDMapper
from guidedLP.common.exceptions import ValidationError, ComputationError


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
            graph, mapper, method="weight", weight_threshold=2.5
        )
        
        # Should keep 2 edges (weights 3.0 and 4.0)
        assert backbone_graph.numberOfEdges() == 2
        
        # Test with target edges
        backbone_graph, backbone_mapper = apply_backbone(
            graph, mapper, method="weight", target_edges=2
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
            graph, mapper, method="weight", weight_threshold=1.5,
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
            graph, mapper, method="degree", target_nodes=2
        )
        
        # Should keep 'hub' (degree 3) and one other node with degree 2
        assert backbone_graph.numberOfNodes() == 2
        assert backbone_graph.numberOfEdges() <= 4  # Max possible edges between 2 nodes


class TestNoiseCorrected:
    """Test the Coscia & Neffke noise-corrected backbone."""

    def test_noise_corrected_basic(self):
        """The highest-lift edge (largest deviation from configuration-model
        expectation) should be ranked highest by score, even when another edge
        has greater raw weight."""
        # Strengths under this edgelist: A=12 (1+1+10), B=2, C=2, D=10.
        # The configuration-model normalization uses n.. = Σ s_i = 2m = 26.
        # Expected vs. observed:
        #   A-B: E=12*2/26≈0.92, obs=1  → kappa*w=13/12, score=1/25
        #   A-C: same as A-B    → score=1/25
        #   A-D: E=12*10/26≈4.6, obs=10 → kappa*w=13/6,  score=7/19
        #   B-C: E=2*2/26≈0.15,  obs=1  → kappa*w=13/2,  score=11/15  (highest lift)
        edges = pl.DataFrame({
            'source': ['A', 'A', 'A', 'B'],
            'target': ['B', 'C', 'D', 'C'],
            'weight': [1.0, 1.0, 10.0, 1.0],
        })
        graph, mapper = build_graph_from_edgelist(edges, weight_col='weight')

        backbone_graph, _, edge_details = apply_backbone(
            graph, mapper, method="noise_corrected", threshold=0.5,
            return_filtered_edges=True,
        )

        assert edge_details.height == 4
        assert {'score', 'sdev_cij', 'kept'}.issubset(set(edge_details.columns))

        bc = edge_details.filter(
            ((pl.col('source_id') == 'B') & (pl.col('target_id') == 'C'))
            | ((pl.col('source_id') == 'C') & (pl.col('target_id') == 'B'))
        )
        ad = edge_details.filter(
            ((pl.col('source_id') == 'A') & (pl.col('target_id') == 'D'))
            | ((pl.col('source_id') == 'D') & (pl.col('target_id') == 'A'))
        )
        assert bc.height == 1 and ad.height == 1

        # Score ranking reflects lift, not raw weight.
        assert float(bc['score'][0]) == pytest.approx(11.0 / 15.0, rel=1e-6)
        assert float(ad['score'][0]) == pytest.approx(7.0 / 19.0, rel=1e-6)
        assert float(bc['score'][0]) == float(edge_details['score'].max())

        # Returned graph must agree with the `kept` column.
        assert backbone_graph.numberOfEdges() == int(edge_details['kept'].sum())

    def test_threshold_monotonic(self):
        """Raising the threshold can only keep fewer (or equal) edges."""
        np.random.seed(7)
        n_edges = 200
        srcs = np.random.randint(0, 30, n_edges)
        tgts = np.random.randint(0, 30, n_edges)
        valid = srcs != tgts
        srcs, tgts = srcs[valid], tgts[valid]
        weights = np.random.exponential(2.0, srcs.size)

        edges = pl.DataFrame({
            'source': [f'n{s}' for s in srcs],
            'target': [f'n{t}' for t in tgts],
            'weight': weights,
        })
        graph, mapper = build_graph_from_edgelist(edges, weight_col='weight')

        kept_counts = []
        for thr in [0.5, 1.0, 2.0, 4.0]:
            bb, _ = apply_backbone(
                graph, mapper, method="noise_corrected",
                threshold=thr, keep_disconnected=True,
            )
            kept_counts.append(bb.numberOfEdges())

        # Strictly non-increasing in threshold.
        assert kept_counts == sorted(kept_counts, reverse=True)

    def test_directed_vs_undirected(self):
        """Noise-corrected runs on both graph types and produces sensible output."""
        edges = pl.DataFrame({
            'source': ['A', 'B', 'C', 'A'],
            'target': ['B', 'C', 'A', 'C'],
            'weight': [1.0, 5.0, 2.0, 3.0],
        })

        for directed in (False, True):
            graph, mapper = build_graph_from_edgelist(
                edges, weight_col='weight', directed=directed
            )
            _, _, edge_details = apply_backbone(
                graph, mapper, method="noise_corrected",
                threshold=1.0, return_filtered_edges=True,
            )
            # One row per edge in the original graph.
            assert edge_details.height == graph.numberOfEdges()
            # Scores live in [-1, 1] by construction.
            assert float(edge_details['score'].min()) >= -1.0 - 1e-9
            assert float(edge_details['score'].max()) <= 1.0 + 1e-9
            # sdev is non-negative.
            assert float(edge_details['sdev_cij'].min()) >= -1e-12

    def test_empty_graph(self):
        """Empty graphs return an empty backbone without error."""
        empty_graph = nk.Graph(0, weighted=True)
        empty_mapper = IDMapper()
        bb, _ = apply_backbone(empty_graph, empty_mapper, method="noise_corrected")
        assert bb.numberOfNodes() == 0
        assert bb.numberOfEdges() == 0

    def test_invalid_threshold(self):
        """threshold must be strictly positive."""
        edges = pl.DataFrame({
            'source': ['A'], 'target': ['B'], 'weight': [1.0],
        })
        graph, mapper = build_graph_from_edgelist(edges, weight_col='weight')

        with pytest.raises(ValidationError, match="threshold must be > 0"):
            apply_backbone(graph, mapper, method="noise_corrected", threshold=0.0)
        with pytest.raises(ValidationError, match="threshold must be > 0"):
            apply_backbone(graph, mapper, method="noise_corrected", threshold=-1.0)

    def test_noise_corrected_in_available_methods(self):
        """The method name must be advertised in AVAILABLE_METHODS."""
        assert "noise_corrected" in AVAILABLE_METHODS


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
        with pytest.raises(ValidationError, match="alpha must be"):
            apply_backbone(graph, mapper, method="disparity", alpha=0.0)
        
        with pytest.raises(ValidationError, match="alpha must be"):
            apply_backbone(graph, mapper, method="disparity", alpha=1.0)
        
        with pytest.raises(ValidationError, match="alpha must be"):
            apply_backbone(graph, mapper, method="disparity", alpha=-0.5)
    
    def test_conflicting_parameters(self):
        """Test conflicting parameter combinations."""
        edges = pl.DataFrame({
            'source': ['A'],
            'target': ['B'],
            'weight': [1.0]
        })
        
        graph, mapper = build_graph_from_edgelist(edges, weight_col='weight')
        
        # Cannot specify more than one of target_nodes / target_edges / target_fraction
        with pytest.raises(ValidationError, match="Specify at most one"):
            apply_backbone(
                graph, mapper, method="disparity",
                target_nodes=10, target_edges=20
            )
    
    def test_weight_and_degree_fallback_to_median(self):
        """Weight and degree methods fall back to median when no explicit
        threshold or target is given (no error is raised)."""
        edges = pl.DataFrame({
            'source': ['A', 'B', 'C'],
            'target': ['B', 'C', 'A'],
            'weight': [1.0, 2.0, 3.0],
        })

        graph, mapper = build_graph_from_edgelist(edges, weight_col='weight')

        # method="weight" with no explicit threshold/target_edges → median.
        bb_weight, _ = apply_backbone(graph, mapper, method="weight")
        assert bb_weight.numberOfEdges() <= graph.numberOfEdges()

        # method="degree" with no target_nodes → median degree.
        bb_degree, _ = apply_backbone(graph, mapper, method="degree")
        assert bb_degree.numberOfNodes() <= graph.numberOfNodes()


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
            graph, mapper, method="weight", weight_threshold=2.5,
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
            graph, mapper, method="weight",
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
            graph, mapper, method="degree",
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
            graph, mapper, method="weight",
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
            graph, mapper, method="weight",
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
            graph, mapper, method="weight",
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
        for method in ['disparity', 'noise_corrected', 'weight', 'degree']:
            if method == 'disparity':
                backbone_graph, backbone_mapper = apply_backbone(
                    graph, mapper, method=method, alpha=0.05, keep_disconnected=False
                )
            elif method == 'noise_corrected':
                backbone_graph, backbone_mapper = apply_backbone(
                    graph, mapper, method=method, threshold=1.0, keep_disconnected=False
                )
            elif method == 'weight':
                backbone_graph, backbone_mapper = apply_backbone(
                    graph, mapper, method=method, weight_threshold=5.0, keep_disconnected=False
                )
            else:  # degree
                backbone_graph, backbone_mapper = apply_backbone(
                    graph, mapper, method=method, target_nodes=500, keep_disconnected=False
                )

            # Verify synchronization for each method
            assert backbone_graph.numberOfNodes() == backbone_mapper.size(), \
                f"{method}: Large graph synchronization failed: " \
                f"{backbone_graph.numberOfNodes()} != {backbone_mapper.size()}"


# ---------------------------------------------------------------------------
# Tests ported from the old test_filtering.py — coverage that's unique to the
# consolidated apply_backbone (median-fallback, target_edges cap on disparity,
# disparity validation messages, zero/extreme weights).
# ---------------------------------------------------------------------------

class TestApplyBackbonePorted:
    """Backbone coverage ported from test_filtering.py."""

    def setup_method(self):
        self.weighted_graph = nk.Graph(5, weighted=True)
        self.weighted_graph.addEdge(0, 1, 10.0)
        self.weighted_graph.addEdge(0, 2, 5.0)
        self.weighted_graph.addEdge(0, 3, 1.0)
        self.weighted_graph.addEdge(0, 4, 0.5)
        self.weighted_graph.addEdge(1, 2, 3.0)

        self.weighted_mapper = IDMapper()
        for i in range(5):
            self.weighted_mapper.add_mapping(f"node_{i}", i)

        self.unweighted_graph = nk.Graph(6)
        for i in range(1, 6):
            self.unweighted_graph.addEdge(0, i)
        self.unweighted_graph.addEdge(1, 2)
        self.unweighted_graph.addEdge(3, 4)

        self.unweighted_mapper = IDMapper()
        for i in range(6):
            self.unweighted_mapper.add_mapping(f"unw_{i}", i)

    def test_disparity_filter_target_edges(self):
        """target_edges caps the disparity-filtered edge count."""
        target = 3
        backbone, _ = apply_backbone(
            self.weighted_graph, self.weighted_mapper,
            method="disparity", alpha=0.1, target_edges=target,
        )
        assert backbone.numberOfEdges() <= target

    def test_weight_method_median_fallback(self):
        """method='weight' with no target_edges/weight_threshold uses the median."""
        backbone, _ = apply_backbone(
            self.weighted_graph, self.weighted_mapper, method="weight",
        )
        assert backbone.numberOfEdges() <= self.weighted_graph.numberOfEdges()

    def test_weight_method_unweighted_graph(self):
        """method='weight' tolerates unweighted input (all weights = 1.0)."""
        backbone, _ = apply_backbone(
            self.unweighted_graph, self.unweighted_mapper, method="weight",
        )
        assert backbone.numberOfNodes() == self.unweighted_graph.numberOfNodes()

    def test_degree_method_median_fallback(self):
        """method='degree' with no target_nodes uses the median degree."""
        backbone, _ = apply_backbone(
            self.unweighted_graph, self.unweighted_mapper, method="degree",
        )
        assert backbone.numberOfNodes() <= self.unweighted_graph.numberOfNodes()

    def test_get_backbone_statistics_full_keys(self):
        """get_backbone_statistics exposes density and compression metrics."""
        backbone, _ = apply_backbone(
            self.weighted_graph, self.weighted_mapper,
            method="disparity", alpha=0.1,
        )
        stats = get_backbone_statistics(self.weighted_graph, backbone)
        for key in (
            "original_nodes", "original_edges",
            "backbone_nodes", "backbone_edges",
            "node_retention", "edge_retention",
            "compression_ratio",
            "original_density", "backbone_density", "density_ratio",
        ):
            assert key in stats
        assert 0 <= stats["node_retention"] <= 1
        assert 0 <= stats["edge_retention"] <= 1
        assert stats["compression_ratio"] >= 1

    def test_all_edges_filtered_returns_empty(self):
        """Extreme alpha that filters everything returns an empty backbone
        (rather than raising) — the user can inspect numberOfEdges()."""
        backbone, _ = apply_backbone(
            self.weighted_graph, self.weighted_mapper,
            method="disparity", alpha=0.000001,
        )
        assert backbone.numberOfEdges() == 0


class TestBackboneParameterValidationPorted:
    """Validation tests ported from test_filtering.py."""

    def setup_method(self):
        self.graph = nk.Graph(3, weighted=True)
        self.graph.addEdge(0, 1, 1.0)
        self.graph.addEdge(1, 2, 2.0)
        self.mapper = IDMapper()
        for i in range(3):
            self.mapper.add_mapping(f"node_{i}", i)

    def test_backbone_negative_targets(self):
        with pytest.raises(ValidationError, match="target_nodes must be positive"):
            apply_backbone(self.graph, self.mapper, target_nodes=-1)

    def test_disparity_requires_weighted_graph(self):
        unweighted = nk.Graph(3)
        unweighted.addEdge(0, 1)
        unweighted.addEdge(1, 2)
        with pytest.raises(ValidationError, match="Disparity filter requires a weighted graph"):
            apply_backbone(unweighted, self.mapper, method="disparity")


class TestBackboneEdgeCasesPorted:
    """Edge cases ported from test_filtering.py."""

    def test_star_graph_disparity(self):
        star_graph = nk.Graph(5, weighted=True)
        star_graph.addEdge(0, 1, 10.0)
        star_graph.addEdge(0, 2, 5.0)
        star_graph.addEdge(0, 3, 1.0)
        star_graph.addEdge(0, 4, 0.1)
        mapper = IDMapper()
        for i in range(5):
            mapper.add_mapping(f"star_{i}", i)

        backbone, _ = apply_backbone(
            star_graph, mapper, method="disparity", alpha=0.1,
        )
        assert backbone.numberOfEdges() < star_graph.numberOfEdges()

    def test_disparity_on_500_node_random_graph(self):
        n_nodes = 500
        large_graph = nk.Graph(n_nodes, weighted=True)
        np.random.seed(42)
        for _ in range(1000):
            u, v = np.random.choice(n_nodes, 2, replace=False)
            weight = np.random.exponential(2.0)
            if not large_graph.hasEdge(u, v):
                large_graph.addEdge(u, v, weight)
        mapper = IDMapper()
        for i in range(n_nodes):
            mapper.add_mapping(f"large_{i}", i)

        backbone, _ = apply_backbone(
            large_graph, mapper, method="disparity", alpha=0.05,
        )
        assert backbone.numberOfNodes() <= n_nodes
        assert backbone.numberOfEdges() <= large_graph.numberOfEdges()

    def test_all_equal_weights_disparity(self):
        equal_graph = nk.Graph(4, weighted=True)
        for u, v in [(0, 1), (0, 2), (0, 3), (1, 2)]:
            equal_graph.addEdge(u, v, 1.0)
        mapper = IDMapper()
        for i in range(4):
            mapper.add_mapping(f"eq_{i}", i)

        try:
            backbone, _ = apply_backbone(
                equal_graph, mapper, method="disparity", alpha=0.5,
            )
            assert backbone.numberOfNodes() <= 4
            assert backbone.numberOfEdges() <= 4
        except ComputationError:
            # All edges filtered out is also valid for the equal-weight case.
            pass

    def test_zero_weight_edges(self):
        zero_graph = nk.Graph(3, weighted=True)
        zero_graph.addEdge(0, 1, 0.0)
        zero_graph.addEdge(1, 2, 1.0)
        mapper = IDMapper()
        for i in range(3):
            mapper.add_mapping(f"zero_{i}", i)

        backbone, _ = apply_backbone(
            zero_graph, mapper, method="weight", target_edges=1,
        )
        assert backbone.numberOfEdges() == 1

    def test_numerical_stability_large_weights(self):
        large_weight_graph = nk.Graph(3, weighted=True)
        large_weight_graph.addEdge(0, 1, 1e6)
        large_weight_graph.addEdge(0, 2, 1e-6)
        large_weight_graph.addEdge(1, 2, 1.0)
        mapper = IDMapper()
        for i in range(3):
            mapper.add_mapping(f"big_{i}", i)

        backbone, _ = apply_backbone(
            large_weight_graph, mapper, method="disparity", alpha=0.1,
        )
        assert backbone.numberOfNodes() >= 1
        assert backbone.numberOfEdges() >= 0


class TestApplyBackboneProtectedNodes:
    """``protected_nodes`` exemption across methods + input paths."""

    def _star_graph(self):
        """5-leaf star with one strong arm; the rest are weak."""
        edges = pl.DataFrame({
            "source": ["center"] * 5,
            "target": ["l1", "l2", "l3", "l4", "l5"],
            "weight": [1.0, 1.0, 1.0, 1.0, 10.0],
        })
        return build_graph_from_edgelist(edges, weight_col="weight")

    def test_protected_node_keeps_weak_edge_disparity(self):
        """Weak-edge endpoint marked protected → its edge is forced kept."""
        g, m = self._star_graph()
        # Without protection a tight disparity drops most weak edges; with
        # l1 protected, the (center, l1) edge must survive.
        backbone, _, edges = apply_backbone(
            g, m, method="disparity", alpha=0.01,
            return_filtered_edges=True,
            protected_nodes=["l1"],
        )
        kept_l1 = edges.filter(
            ((pl.col("source_id") == "l1") | (pl.col("target_id") == "l1"))
            & pl.col("kept")
        )
        assert kept_l1.height == 1
        assert backbone.numberOfEdges() >= 1

    def test_protected_node_weight_method(self):
        """``method='weight'`` with a low target — weak edge survives via protection."""
        edges = pl.DataFrame({
            "source": ["a", "a", "a"],
            "target": ["b", "c", "d"],
            "weight": [1.0, 5.0, 10.0],
        })
        g, m = build_graph_from_edgelist(edges, weight_col="weight")
        # target_edges=1 would normally keep only the (a, d, 10) edge.
        backbone, mapper = apply_backbone(
            g, m, method="weight", target_edges=1,
            protected_nodes=["b"],
        )
        # Now b's edge (a-b) must also be kept → 2 edges total.
        assert backbone.numberOfEdges() == 2

    def test_protected_node_degree_method(self):
        """``method='degree'`` — protected low-degree node survives and its edge holds."""
        # Hub-and-spoke + an outlier pair: hub has high degree, spokes & pair are low.
        edges = pl.DataFrame({
            "source": ["hub", "hub", "hub", "hub", "x"],
            "target": ["s1", "s2", "s3", "s4", "y"],
            "weight": [1.0] * 5,
        })
        g, m = build_graph_from_edgelist(edges, weight_col="weight")
        # target_nodes=2 keeps hub + one of its spokes (or rather: top-2 by degree).
        # hub has degree 4, all others have degree 1 — so kept = {hub, one tied spoke}.
        # Protect 'x' (degree 1) → x survives, and edge x-y is forced kept,
        # so y also survives.
        backbone, mapper = apply_backbone(
            g, m, method="degree", target_nodes=2,
            protected_nodes=["x"],
        )
        # x must be in the surviving set.
        survivors = {mapper.get_original(i) for i in range(backbone.numberOfNodes())
                     if backbone.hasNode(i)}
        assert "x" in survivors
        # Edge x-y kept → y also present.
        assert "y" in survivors

    def test_protection_overrides_keep_disconnected(self):
        """``keep_disconnected=False`` cannot remove a protected node."""
        # Build a graph where protected node 'iso' has no incident edges that
        # would survive any reasonable score-based filter — but we'll fake it
        # by giving it a weight-1 edge in a graph dominated by weight-100 edges.
        edges = pl.DataFrame({
            "source": ["a", "b", "iso"],
            "target": ["b", "c", "side"],
            "weight": [100.0, 100.0, 1.0],
        })
        g, m = build_graph_from_edgelist(edges, weight_col="weight")
        # Weight threshold cuts the iso-side edge; without protection iso is
        # isolated and dropped. With protection iso (and its edge) survive.
        backbone, mapper = apply_backbone(
            g, m, method="weight", weight_threshold=50.0,
            protected_nodes=["iso"], keep_disconnected=False,
        )
        survivors = {mapper.get_original(i) for i in range(backbone.numberOfNodes())
                     if backbone.hasNode(i)}
        assert "iso" in survivors

    def test_protection_overrides_min_node_retention(self):
        """``min_node_retention`` does not remove protected nodes (bipartite_svn)."""
        # A bipartite-ish graph; we run bipartite_svn with an aggressive
        # retention threshold and check protected survives.
        rng = np.random.default_rng(0)
        users = [f"u{i}" for i in range(8)]
        items = [f"i{i}" for i in range(8)]
        srcs, tgts, wgts = [], [], []
        for u in users:
            for it in rng.choice(items, size=3, replace=False):
                srcs.append(u); tgts.append(it); wgts.append(float(rng.integers(1, 5)))
        edges = pl.DataFrame({"source": srcs, "target": tgts, "weight": wgts})
        g, m = build_graph_from_edgelist(edges, weight_col="weight")

        backbone, mapper = apply_backbone(
            g, m, method="bipartite_svn", alpha=0.5, correction="fdr_bh",
            min_node_retention=0.95,  # very strict; would normally drop most users
            protected_nodes=["u0"], keep_disconnected=True,
        )
        survivors = {mapper.get_original(i) for i in range(backbone.numberOfNodes())
                     if backbone.hasNode(i)}
        assert "u0" in survivors

    def test_protected_node_frame_input(self):
        """Frame-input path forwards ``protected_nodes`` to the kept mask."""
        edges = pl.DataFrame({
            "source_id": ["a", "a", "a"],
            "target_id": ["b", "c", "d"],
            "weight": [1.0, 5.0, 10.0],
        })
        out = apply_backbone(
            edges, None, method="weight", target_edges=1,
            protected_nodes=["b"], verbose=False,
        )
        # Lean output: two rows now (a-d via score, a-b via protection).
        assert out.height == 2

    def test_missing_protected_id_warns(self, caplog):
        """Unknown protected IDs warn but don't error."""
        import logging
        g, m = self._star_graph()
        with caplog.at_level(logging.WARNING):
            apply_backbone(
                g, m, method="disparity", alpha=0.5,
                protected_nodes=["l1", "does-not-exist"],
                verbose=False,
            )
        assert any("protected nodes not present in graph" in r.message
                   for r in caplog.records)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])