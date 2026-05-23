"""
Tests for network construction functionality.

This module provides comprehensive testing for the network construction module,
including graph building from edge lists, ID mapping, weight calculation,
and various graph types (directed/undirected, weighted/unweighted, bipartite).
"""

import pytest
import polars as pl
import networkit as nk
import tempfile
import os
from pathlib import Path
from typing import List, Dict, Any

from src.network.construction import (
    build_graph_from_edgelist,
    project_bipartite,
    get_graph_info,
    get_bipartite_info,
    validate_graph_construction,
    _load_edge_list,
    _validate_edge_list,
    _process_edges,
    _create_id_mapping,
    _validate_bipartite_structure,
    _construct_graph,
    _identify_bipartite_partitions,
    _calculate_projection_weight
)
from src.common.id_mapper import IDMapper
from src.common.exceptions import (
    GraphConstructionError,
    ValidationError,
    DataFormatError,
    ConfigurationError
)


class TestBasicGraphConstruction:
    """Test basic graph construction functionality."""
    
    def test_simple_undirected_graph(self):
        """Test construction of simple undirected graph."""
        edges = pl.DataFrame({
            "source": ["A", "B", "C"],
            "target": ["B", "C", "A"]
        })
        
        graph, id_mapper = build_graph_from_edgelist(edges)
        
        assert graph.numberOfNodes() == 3
        assert graph.numberOfEdges() == 3
        assert not graph.isDirected()
        assert not graph.isWeighted()
        assert id_mapper.size() == 3
        
        # Verify ID mapping works
        assert id_mapper.has_original("A")
        assert id_mapper.has_original("B") 
        assert id_mapper.has_original("C")
    
    def test_simple_directed_graph(self):
        """Test construction of simple directed graph."""
        edges = pl.DataFrame({
            "source": ["A", "B", "C"],
            "target": ["B", "C", "A"]
        })
        
        graph, id_mapper = build_graph_from_edgelist(edges, directed=True)
        
        assert graph.numberOfNodes() == 3
        assert graph.numberOfEdges() == 3
        assert graph.isDirected()
        assert not graph.isWeighted()
        assert id_mapper.size() == 3
    
    def test_weighted_graph(self):
        """Test construction of weighted graph."""
        edges = pl.DataFrame({
            "source": ["A", "B", "C"],
            "target": ["B", "C", "A"],
            "weight": [1.5, 2.0, 0.5]
        })
        
        graph, id_mapper = build_graph_from_edgelist(edges, weight_col="weight")
        
        assert graph.numberOfNodes() == 3
        assert graph.numberOfEdges() == 3
        assert graph.isWeighted()
        
        # Test edge weights
        a_id = id_mapper.get_internal("A")
        b_id = id_mapper.get_internal("B")
        assert graph.weight(a_id, b_id) == 1.5
    
    def test_custom_column_names(self):
        """Test graph construction with custom column names."""
        edges = pl.DataFrame({
            "from_node": ["user1", "user2", "user3"],
            "to_node": ["user2", "user3", "user1"],
            "edge_weight": [1.0, 2.0, 3.0]
        })
        
        graph, id_mapper = build_graph_from_edgelist(
            edges,
            source_col="from_node",
            target_col="to_node", 
            weight_col="edge_weight"
        )
        
        assert graph.numberOfNodes() == 3
        assert graph.numberOfEdges() == 3
        assert graph.isWeighted()
        assert id_mapper.has_original("user1")
    
    def test_mixed_node_types(self):
        """Test graph construction with mixed node ID types."""
        edges = pl.DataFrame({
            "source": ["A", 1, "C", 2],
            "target": [1, "C", 2, "A"]
        })
        
        graph, id_mapper = build_graph_from_edgelist(edges)
        
        assert graph.numberOfNodes() == 4
        assert graph.numberOfEdges() == 4
        
        # Test that both string and numeric IDs work
        assert id_mapper.has_original("A")
        assert id_mapper.has_original(1)
        assert id_mapper.has_original("C")
        assert id_mapper.has_original(2)


class TestAutoWeightCalculation:
    """Test automatic weight calculation from duplicate edges."""
    
    def test_auto_weight_from_duplicates(self):
        """Test automatic weight calculation from duplicate edges."""
        edges = pl.DataFrame({
            "source": ["A", "A", "B", "B", "B"],
            "target": ["B", "B", "C", "C", "C"]
        })
        
        graph, id_mapper = build_graph_from_edgelist(edges, auto_weight=True)
        
        assert graph.numberOfNodes() == 3
        assert graph.numberOfEdges() == 2  # A->B and B->C
        assert graph.isWeighted()
        
        # Check calculated weights
        a_id = id_mapper.get_internal("A")
        b_id = id_mapper.get_internal("B")
        c_id = id_mapper.get_internal("C")
        
        assert graph.weight(a_id, b_id) == 2.0  # 2 A->B edges
        assert graph.weight(b_id, c_id) == 3.0  # 3 B->C edges
    
    def test_auto_weight_disabled(self):
        """Test that auto weight is disabled when weight column provided."""
        edges = pl.DataFrame({
            "source": ["A", "A", "B"],
            "target": ["B", "B", "C"],
            "weight": [1.0, 2.0, 3.0]
        })
        
        graph, id_mapper = build_graph_from_edgelist(
            edges, 
            weight_col="weight", 
            auto_weight=True
        )
        
        assert graph.numberOfNodes() == 3
        assert graph.numberOfEdges() == 2  # Duplicates merged by summing weights
        assert graph.isWeighted()
        
        # Weight should be sum: 1.0 + 2.0 = 3.0
        a_id = id_mapper.get_internal("A")
        b_id = id_mapper.get_internal("B")
        assert graph.weight(a_id, b_id) == 3.0
    
    def test_no_auto_weight(self):
        """Test graph construction without auto weight calculation."""
        edges = pl.DataFrame({
            "source": ["A", "A", "B"],
            "target": ["B", "B", "C"]
        })
        
        graph, id_mapper = build_graph_from_edgelist(edges, auto_weight=False)
        
        assert graph.numberOfNodes() == 3
        assert graph.numberOfEdges() == 3  # All edges kept
        assert not graph.isWeighted()


class TestBipartiteGraphs:
    """Test bipartite graph construction and validation."""
    
    def test_valid_bipartite_graph(self):
        """Test construction of valid bipartite graph."""
        edges = pl.DataFrame({
            "source": ["A", "B", "C"],  # Source partition
            "target": ["X", "Y", "Z"]   # Target partition
        })
        
        graph, id_mapper = build_graph_from_edgelist(edges, bipartite=True)
        
        assert graph.numberOfNodes() == 6
        assert graph.numberOfEdges() == 3
        assert id_mapper.size() == 6
    
    def test_invalid_bipartite_graph(self):
        """Test that invalid bipartite structure raises error."""
        edges = pl.DataFrame({
            "source": ["A", "B", "A"],  # A appears in both
            "target": ["B", "C", "X"]   # B appears in both
        })
        
        with pytest.raises(GraphConstructionError, match="not bipartite"):
            build_graph_from_edgelist(edges, bipartite=True)
    
    def test_bipartite_with_weights(self):
        """Test bipartite graph with weights."""
        edges = pl.DataFrame({
            "source": ["user1", "user2"],
            "target": ["item1", "item2"],
            "weight": [3.5, 4.0]
        })
        
        graph, id_mapper = build_graph_from_edgelist(
            edges, 
            weight_col="weight", 
            bipartite=True
        )
        
        assert graph.numberOfNodes() == 4
        assert graph.numberOfEdges() == 2
        assert graph.isWeighted()


class TestEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_empty_edgelist(self):
        """Test handling of empty edge list."""
        empty_edges = pl.DataFrame({
            "source": [],
            "target": []
        })
        
        with pytest.warns(UserWarning, match="Empty edge list"):
            graph, id_mapper = build_graph_from_edgelist(empty_edges)
        
        assert graph.numberOfNodes() == 0
        assert graph.numberOfEdges() == 0
        assert id_mapper.size() == 0
    
    def test_self_loops_allowed(self):
        """Test that self-loops are allowed by default."""
        edges = pl.DataFrame({
            "source": ["A", "A", "B"],
            "target": ["A", "B", "B"]  # A->A and B->B are self-loops
        })
        
        graph, id_mapper = build_graph_from_edgelist(edges)
        
        assert graph.numberOfNodes() == 2
        assert graph.numberOfEdges() == 3
        assert graph.numberOfSelfLoops() == 2
    
    def test_self_loops_disabled(self):
        """Test disabling self-loops."""
        edges = pl.DataFrame({
            "source": ["A", "A", "B"],
            "target": ["A", "B", "B"]
        })
        
        graph, id_mapper = build_graph_from_edgelist(edges, allow_self_loops=False)
        
        assert graph.numberOfNodes() == 2
        assert graph.numberOfEdges() == 1  # Only A->B remains
        assert graph.numberOfSelfLoops() == 0
    
    def test_duplicate_removal(self):
        """Test explicit duplicate edge removal."""
        edges = pl.DataFrame({
            "source": ["A", "A", "B", "B"],
            "target": ["B", "B", "C", "C"],
            "weight": [1.0, 2.0, 3.0, 4.0]
        })
        
        graph, id_mapper = build_graph_from_edgelist(
            edges, 
            weight_col="weight",
            remove_duplicates=True
        )
        
        assert graph.numberOfNodes() == 3
        assert graph.numberOfEdges() == 2
        
        # Should keep first occurrence weights
        a_id = id_mapper.get_internal("A")
        b_id = id_mapper.get_internal("B")
        c_id = id_mapper.get_internal("C")
        
        assert graph.weight(a_id, b_id) == 1.0  # First A->B weight
        assert graph.weight(b_id, c_id) == 3.0  # First B->C weight
    
    def test_missing_columns(self):
        """Test error handling for missing columns."""
        edges = pl.DataFrame({
            "source": ["A", "B"],
            # Missing target column
        })
        
        with pytest.raises(ValidationError, match="Missing required columns"):
            build_graph_from_edgelist(edges)
    
    def test_null_values_in_nodes(self):
        """Test error handling for null node values."""
        edges = pl.DataFrame({
            "source": ["A", None, "C"],
            "target": ["B", "C", "A"]
        })
        
        with pytest.raises(ValidationError, match="null values"):
            build_graph_from_edgelist(edges)


class TestFileLoading:
    """Test loading edge lists from CSV files."""
    
    def test_load_from_csv_file(self):
        """Test loading edge list from CSV file."""
        edges_data = {
            "source": ["A", "B", "C"],
            "target": ["B", "C", "A"],
            "weight": [1.0, 2.0, 3.0]
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            # Write CSV header and data
            f.write("source,target,weight\n")
            for i in range(len(edges_data["source"])):
                f.write(f"{edges_data['source'][i]},{edges_data['target'][i]},{edges_data['weight'][i]}\n")
            temp_file = f.name
        
        try:
            graph, id_mapper = build_graph_from_edgelist(
                temp_file,
                weight_col="weight"
            )
            
            assert graph.numberOfNodes() == 3
            assert graph.numberOfEdges() == 3
            assert graph.isWeighted()
            
        finally:
            os.unlink(temp_file)
    
    def test_nonexistent_file(self):
        """Test error handling for nonexistent files."""
        with pytest.raises(DataFormatError, match="file not found"):
            build_graph_from_edgelist("/nonexistent/file.csv")
    
    def test_invalid_csv_format(self):
        """Test error handling for invalid CSV format."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write("invalid,csv,format,\n")
            f.write("A,B,C,D,E,F,G\n")  # Too many columns
            temp_file = f.name
        
        try:
            with pytest.raises(DataFormatError):
                build_graph_from_edgelist(temp_file)
        finally:
            os.unlink(temp_file)
    
    def test_invalid_input_type(self):
        """Test error handling for invalid input types."""
        with pytest.raises(DataFormatError, match="Invalid edgelist type"):
            build_graph_from_edgelist(["invalid", "list", "type"])


class TestHelperFunctions:
    """Test individual helper functions."""
    
    def test_load_edge_list_dataframe(self):
        """Test _load_edge_list with DataFrame input."""
        edges = pl.DataFrame({
            "source": ["A", "B"],
            "target": ["B", "C"]
        })
        
        result = _load_edge_list(edges)
        assert result.equals(edges)
    
    def test_create_id_mapping(self):
        """Test _create_id_mapping function."""
        edges = pl.DataFrame({
            "source": ["A", "B", "C"],
            "target": ["B", "C", "A"]
        })
        
        id_mapper = _create_id_mapping(edges, "source", "target")
        
        assert id_mapper.size() == 3
        assert id_mapper.has_original("A")
        assert id_mapper.has_original("B")
        assert id_mapper.has_original("C")
        
        # Check that mapping is deterministic (sorted)
        internal_ids = []
        for original in ["A", "B", "C"]:
            internal_ids.append(id_mapper.get_internal(original))
        assert internal_ids == sorted(internal_ids)
    
    def test_validate_bipartite_structure_valid(self):
        """Test _validate_bipartite_structure with valid bipartite graph."""
        edges = pl.DataFrame({
            "source": ["A", "B"],
            "target": ["X", "Y"]
        })
        id_mapper = _create_id_mapping(edges, "source", "target")
        
        # Should not raise exception
        _validate_bipartite_structure(edges, "source", "target", id_mapper)
    
    def test_validate_bipartite_structure_invalid(self):
        """Test _validate_bipartite_structure with invalid bipartite graph."""
        edges = pl.DataFrame({
            "source": ["A", "B"],
            "target": ["B", "C"]  # B appears in both
        })
        id_mapper = _create_id_mapping(edges, "source", "target")
        
        with pytest.raises(GraphConstructionError, match="not bipartite"):
            _validate_bipartite_structure(edges, "source", "target", id_mapper)
    
    def test_process_edges_auto_weight(self):
        """Test _process_edges with auto weight calculation."""
        edges = pl.DataFrame({
            "source": ["A", "A", "B"],
            "target": ["B", "B", "C"]
        })
        
        processed = _process_edges(
            edges, "source", "target", None, 
            auto_weight=True, allow_self_loops=True, remove_duplicates=False
        )
        
        assert "weight" in processed.columns
        assert len(processed) == 2  # Duplicates merged
        
        # Check weights
        weights = processed.filter(
            (pl.col("source") == "A") & (pl.col("target") == "B")
        )["weight"].to_list()
        assert weights[0] == 2.0


class TestGraphInfo:
    """Test graph information and validation functions."""
    
    def test_get_graph_info(self):
        """Test get_graph_info function."""
        edges = pl.DataFrame({
            "source": ["A", "B", "C", "A"],
            "target": ["B", "C", "A", "A"],  # Includes self-loop
            "weight": [1.0, 2.0, 3.0, 0.5]
        })
        
        graph, id_mapper = build_graph_from_edgelist(edges, weight_col="weight")
        info = get_graph_info(graph, id_mapper)
        
        assert info["num_nodes"] == 3
        assert info["num_edges"] == 4
        assert not info["directed"]
        assert info["weighted"]
        assert info["has_self_loops"]
        assert info["num_self_loops"] == 1
        assert info["node_id_mapping_size"] == 3
        assert "density" in info
        assert "is_connected" in info
    
    def test_validate_graph_construction_success(self):
        """Test successful graph construction validation."""
        edges = pl.DataFrame({
            "source": ["A", "B"],
            "target": ["B", "C"]
        })
        
        graph, id_mapper = build_graph_from_edgelist(edges)
        
        # Should not raise exception
        validate_graph_construction(graph, id_mapper, expected_nodes=3, expected_edges=2)
    
    def test_validate_graph_construction_failure(self):
        """Test graph construction validation failure."""
        edges = pl.DataFrame({
            "source": ["A", "B"],
            "target": ["B", "C"]
        })
        
        graph, id_mapper = build_graph_from_edgelist(edges)
        
        # Should raise exception for wrong expected counts
        with pytest.raises(GraphConstructionError, match="Unexpected node count"):
            validate_graph_construction(graph, id_mapper, expected_nodes=5)
        
        with pytest.raises(GraphConstructionError, match="Unexpected edge count"):
            validate_graph_construction(graph, id_mapper, expected_edges=10)


class TestPerformance:
    """Test performance characteristics and scalability."""
    
    def test_large_graph_construction(self):
        """Test construction of moderately large graph."""
        # Create edge list with 1000 nodes and 5000 edges
        n_nodes = 1000
        n_edges = 5000
        
        # Generate random edges
        import random
        random.seed(42)  # For reproducibility
        
        sources = [f"node_{random.randint(0, n_nodes-1)}" for _ in range(n_edges)]
        targets = [f"node_{random.randint(0, n_nodes-1)}" for _ in range(n_edges)]
        
        edges = pl.DataFrame({
            "source": sources,
            "target": targets
        })
        
        graph, id_mapper = build_graph_from_edgelist(edges)
        
        # Basic checks
        assert graph.numberOfEdges() <= n_edges  # May be fewer due to duplicates/self-loops
        assert id_mapper.size() <= n_nodes * 2  # At most n_nodes unique IDs
        
        # Performance check - should complete reasonably quickly
        # If this test takes too long, there may be a performance issue
        info = get_graph_info(graph, id_mapper)
        assert info["num_nodes"] > 0
        assert info["num_edges"] > 0


class TestIntegrationScenarios:
    """Test realistic integration scenarios."""
    
    def test_social_network_scenario(self):
        """Test scenario similar to social network analysis."""
        # Users following each other
        edges = pl.DataFrame({
            "follower": ["alice", "bob", "charlie", "alice", "bob"],
            "followed": ["bob", "charlie", "alice", "charlie", "alice"],
            "timestamp": ["2023-01-01", "2023-01-02", "2023-01-03", "2023-01-04", "2023-01-05"]
        })
        
        # Build directed graph
        graph, id_mapper = build_graph_from_edgelist(
            edges,
            source_col="follower",
            target_col="followed", 
            directed=True
        )
        
        assert graph.numberOfNodes() == 3
        assert graph.isDirected()
        
        # Test that all users are mapped
        for user in ["alice", "bob", "charlie"]:
            assert id_mapper.has_original(user)
    
    def test_bipartite_recommendation_scenario(self):
        """Test scenario for user-item recommendation system."""
        # Users rating items
        edges = pl.DataFrame({
            "user": ["u1", "u1", "u2", "u2", "u3"],
            "item": ["movie_a", "movie_b", "movie_a", "movie_c", "movie_b"],
            "rating": [5.0, 4.0, 3.0, 5.0, 2.0]
        })
        
        graph, id_mapper = build_graph_from_edgelist(
            edges,
            source_col="user",
            target_col="item",
            weight_col="rating",
            bipartite=True
        )
        
        assert graph.numberOfNodes() == 6  # 3 users + 3 items
        assert graph.isWeighted()
        assert graph.numberOfEdges() == 5
        
        # Verify bipartite structure
        users = ["u1", "u2", "u3"]
        items = ["movie_a", "movie_b", "movie_c"]
        
        for user in users:
            assert id_mapper.has_original(user)
        for item in items:
            assert id_mapper.has_original(item)
    
    def test_collaboration_network_scenario(self):
        """Test scenario for collaboration/co-authorship network."""
        # Authors collaborating on papers (auto-weight by number of collaborations)
        edges = pl.DataFrame({
            "author1": ["Smith", "Smith", "Jones", "Brown", "Smith"],
            "author2": ["Jones", "Brown", "Brown", "Davis", "Jones"]  # Smith-Jones appears twice
        })
        
        graph, id_mapper = build_graph_from_edgelist(
            edges,
            source_col="author1",
            target_col="author2",
            auto_weight=True  # Count collaboration frequency
        )
        
        assert graph.numberOfNodes() == 4
        assert graph.isWeighted()
        
        # Smith-Jones should have weight 2 (two collaborations)
        smith_id = id_mapper.get_internal("Smith")
        jones_id = id_mapper.get_internal("Jones")
        assert graph.weight(smith_id, jones_id) == 2.0


class TestBipartiteProjection:
    """Test bipartite graph projection functionality."""
    
    def test_source_projection_count_weights(self):
        """Test projection onto source partition with count weights."""
        # Create bipartite graph: users -> items
        edges = pl.DataFrame({
            "user": ["u1", "u1", "u2", "u2", "u3", "u3"],
            "item": ["i1", "i2", "i1", "i3", "i2", "i3"]
        })
        
        graph, id_mapper = build_graph_from_edgelist(
            edges,
            source_col="user", 
            target_col="item",
            bipartite=True
        )
        
        # Project onto users (connect users who share items)
        user_graph, user_mapper = project_bipartite(
            graph, id_mapper, "source", "count"
        )
        
        assert user_graph.numberOfNodes() == 3  # u1, u2, u3
        assert user_graph.isWeighted()
        assert not user_graph.isDirected()
        
        # u1 and u2 share item i1 (1 shared item)
        u1_id = user_mapper.get_internal("u1")
        u2_id = user_mapper.get_internal("u2")
        assert user_graph.weight(u1_id, u2_id) == 1.0
        
        # u2 and u3 share item i3 (1 shared item)
        u3_id = user_mapper.get_internal("u3")
        assert user_graph.weight(u2_id, u3_id) == 1.0
        
        # u1 and u3 share item i2 (1 shared item)
        assert user_graph.weight(u1_id, u3_id) == 1.0
    
    def test_target_projection_count_weights(self):
        """Test projection onto target partition with count weights."""
        edges = pl.DataFrame({
            "user": ["u1", "u1", "u2", "u2", "u3"],
            "item": ["i1", "i2", "i1", "i2", "i3"]
        })
        
        graph, id_mapper = build_graph_from_edgelist(
            edges,
            source_col="user",
            target_col="item", 
            bipartite=True
        )
        
        # Project onto items (connect items shared by same users)
        item_graph, item_mapper = project_bipartite(
            graph, id_mapper, "target", "count"
        )
        
        assert item_graph.numberOfNodes() == 3  # i1, i2, i3
        assert item_graph.isWeighted()
        
        # i1 and i2 are both liked by u1 and u2 (2 shared users)
        i1_id = item_mapper.get_internal("i1")
        i2_id = item_mapper.get_internal("i2")
        assert item_graph.weight(i1_id, i2_id) == 2.0
        
        # i2 and i3 are only shared by u3 (but u3 doesn't like i1)
        # Actually, only u3 likes i3, so no connection to others
        i3_id = item_mapper.get_internal("i3")
        assert not item_graph.hasEdge(i2_id, i3_id)  # No shared users
    
    def test_jaccard_weight_method(self):
        """Test Jaccard similarity weight calculation."""
        edges = pl.DataFrame({
            "user": ["u1", "u1", "u1", "u2", "u2", "u3"],
            "item": ["i1", "i2", "i3", "i1", "i2", "i1"]
        })
        
        graph, id_mapper = build_graph_from_edgelist(
            edges,
            source_col="user",
            target_col="item",
            bipartite=True
        )
        
        user_graph, user_mapper = project_bipartite(
            graph, id_mapper, "source", "jaccard"
        )
        
        # u1 likes {i1, i2, i3}, u2 likes {i1, i2}
        # Jaccard(u1, u2) = |{i1, i2}| / |{i1, i2, i3}| = 2/3
        u1_id = user_mapper.get_internal("u1")
        u2_id = user_mapper.get_internal("u2")
        expected_jaccard = 2.0 / 3.0
        assert abs(user_graph.weight(u1_id, u2_id) - expected_jaccard) < 1e-10
        
        # u1 likes {i1, i2, i3}, u3 likes {i1}
        # Jaccard(u1, u3) = |{i1}| / |{i1, i2, i3}| = 1/3
        u3_id = user_mapper.get_internal("u3")
        expected_jaccard_u1_u3 = 1.0 / 3.0
        assert abs(user_graph.weight(u1_id, u3_id) - expected_jaccard_u1_u3) < 1e-10
    
    def test_overlap_weight_method(self):
        """Test overlap coefficient weight calculation."""
        edges = pl.DataFrame({
            "user": ["u1", "u1", "u1", "u2", "u2"],
            "item": ["i1", "i2", "i3", "i1", "i2"]
        })
        
        graph, id_mapper = build_graph_from_edgelist(
            edges,
            source_col="user",
            target_col="item",
            bipartite=True
        )
        
        user_graph, user_mapper = project_bipartite(
            graph, id_mapper, "source", "overlap"
        )
        
        # u1 likes {i1, i2, i3}, u2 likes {i1, i2}
        # Overlap(u1, u2) = |{i1, i2}| / min(3, 2) = 2/2 = 1.0
        u1_id = user_mapper.get_internal("u1")
        u2_id = user_mapper.get_internal("u2")
        assert user_graph.weight(u1_id, u2_id) == 1.0
    
    def test_no_shared_neighbors(self):
        """Test projection when nodes have no shared neighbors."""
        edges = pl.DataFrame({
            "user": ["u1", "u2", "u3"],
            "item": ["i1", "i2", "i3"]  # No shared items
        })
        
        graph, id_mapper = build_graph_from_edgelist(
            edges,
            source_col="user",
            target_col="item",
            bipartite=True
        )
        
        user_graph, user_mapper = project_bipartite(
            graph, id_mapper, "source", "count"
        )
        
        assert user_graph.numberOfNodes() == 3
        assert user_graph.numberOfEdges() == 0  # No connections
    
    def test_projection_with_disconnected_nodes(self):
        """Test projection with some isolated nodes."""
        edges = pl.DataFrame({
            "user": ["u1", "u1", "u2"],  # u3 has no connections
            "item": ["i1", "i2", "i1"]
        })
        
        # Add u3 manually to ensure it's in the bipartite graph
        edges = pl.concat([
            edges,
            pl.DataFrame({"user": ["u3"], "item": ["i3"]})
        ])
        
        graph, id_mapper = build_graph_from_edgelist(
            edges,
            source_col="user",
            target_col="item", 
            bipartite=True
        )
        
        user_graph, user_mapper = project_bipartite(
            graph, id_mapper, "source", "count"
        )
        
        assert user_graph.numberOfNodes() == 3
        
        # u1 and u2 share i1
        u1_id = user_mapper.get_internal("u1")
        u2_id = user_mapper.get_internal("u2")
        assert user_graph.hasEdge(u1_id, u2_id)
        
        # u3 is isolated
        u3_id = user_mapper.get_internal("u3")
        assert user_graph.degree(u3_id) == 0
    
    def test_projection_invalid_mode(self):
        """Test error handling for invalid projection mode."""
        edges = pl.DataFrame({
            "user": ["u1", "u2"],
            "item": ["i1", "i2"]
        })
        
        graph, id_mapper = build_graph_from_edgelist(
            edges,
            source_col="user",
            target_col="item",
            bipartite=True
        )
        
        with pytest.raises(ConfigurationError, match="Invalid value for parameter 'projection_mode'"):
            project_bipartite(graph, id_mapper, "invalid_mode", "count")
    
    def test_projection_invalid_weight_method(self):
        """Test error handling for invalid weight method."""
        edges = pl.DataFrame({
            "user": ["u1", "u2"],
            "item": ["i1", "i2"]
        })
        
        graph, id_mapper = build_graph_from_edgelist(
            edges,
            source_col="user",
            target_col="item",
            bipartite=True
        )
        
        with pytest.raises(ConfigurationError, match="Invalid value for parameter 'weight_method'"):
            project_bipartite(graph, id_mapper, "source", "invalid_method")
    
    def test_projection_non_bipartite_graph(self):
        """Test error when trying to project non-bipartite graph."""
        # Create regular graph (not bipartite)
        edges = pl.DataFrame({
            "source": ["A", "B", "C", "A"],
            "target": ["B", "C", "A", "C"]  # A appears in both source and target
        })
        
        graph, id_mapper = build_graph_from_edgelist(edges)
        
        with pytest.raises(GraphConstructionError, match="not bipartite"):
            project_bipartite(graph, id_mapper, "source", "count")
    
    def test_projection_preserves_original_ids(self):
        """Test that projection preserves original node IDs correctly."""
        edges = pl.DataFrame({
            "author": ["smith", "jones", "brown"],
            "paper": ["paper1", "paper1", "paper2"]
        })
        
        graph, id_mapper = build_graph_from_edgelist(
            edges,
            source_col="author",
            target_col="paper",
            bipartite=True
        )
        
        author_graph, author_mapper = project_bipartite(
            graph, id_mapper, "source", "count"
        )
        
        # Check that all original author IDs are preserved
        assert author_mapper.has_original("smith")
        assert author_mapper.has_original("jones") 
        assert author_mapper.has_original("brown")
        
        # Check that paper IDs are NOT in the projected mapper
        assert not author_mapper.has_original("paper1")
        assert not author_mapper.has_original("paper2")
    
    def test_empty_bipartite_graph_projection(self):
        """Test projection of empty bipartite graph."""
        edges = pl.DataFrame({
            "user": [],
            "item": []
        })
        
        with pytest.warns(UserWarning, match="Empty edge list"):
            graph, id_mapper = build_graph_from_edgelist(
                edges,
                source_col="user",
                target_col="item",
                bipartite=True
            )
        
        projected_graph, projected_mapper = project_bipartite(
            graph, id_mapper, "source", "count"
        )
        
        assert projected_graph.numberOfNodes() == 0
        assert projected_graph.numberOfEdges() == 0
        assert projected_mapper.size() == 0


class TestProjectionHelperFunctions:
    """Test helper functions for bipartite projection."""
    
    def test_identify_bipartite_partitions_valid(self):
        """Test identification of valid bipartite partitions."""
        edges = pl.DataFrame({
            "user": ["u1", "u2"],
            "item": ["i1", "i2"]
        })
        
        graph, id_mapper = build_graph_from_edgelist(
            edges,
            source_col="user",
            target_col="item",
            bipartite=True
        )
        
        source_partition, target_partition = _identify_bipartite_partitions(graph, id_mapper)
        
        # Check that partitions are disjoint and complete
        all_nodes = set(source_partition + target_partition)
        expected_nodes = {"u1", "u2", "i1", "i2"}
        assert all_nodes == expected_nodes
        
        # Check that partitions are disjoint
        assert len(set(source_partition).intersection(set(target_partition))) == 0
    
    def test_identify_bipartite_partitions_invalid(self):
        """Test detection of non-bipartite graph."""
        # Create triangle (odd cycle)
        edges = pl.DataFrame({
            "source": ["A", "B", "C"],
            "target": ["B", "C", "A"]
        })
        
        graph, id_mapper = build_graph_from_edgelist(edges)
        
        with pytest.raises(GraphConstructionError, match="not bipartite"):
            _identify_bipartite_partitions(graph, id_mapper)
    
    def test_calculate_projection_weight_count(self):
        """Test count weight calculation."""
        neighbors1 = {"a", "b", "c"}
        neighbors2 = {"b", "c", "d"}
        shared = {"b", "c"}
        
        weight = _calculate_projection_weight(neighbors1, neighbors2, shared, "count")
        assert weight == 2.0
    
    def test_calculate_projection_weight_jaccard(self):
        """Test Jaccard weight calculation."""
        neighbors1 = {"a", "b", "c"}
        neighbors2 = {"b", "c", "d"} 
        shared = {"b", "c"}
        
        # Jaccard = |intersection| / |union| = 2 / 4 = 0.5
        weight = _calculate_projection_weight(neighbors1, neighbors2, shared, "jaccard")
        assert weight == 0.5
    
    def test_calculate_projection_weight_overlap(self):
        """Test overlap coefficient calculation."""
        neighbors1 = {"a", "b", "c"}  # Size 3
        neighbors2 = {"b", "c"}       # Size 2
        shared = {"b", "c"}           # Size 2
        
        # Overlap = |intersection| / min(|A|, |B|) = 2 / min(3, 2) = 2/2 = 1.0
        weight = _calculate_projection_weight(neighbors1, neighbors2, shared, "overlap")
        assert weight == 1.0
    
    def test_calculate_projection_weight_edge_cases(self):
        """Test weight calculation edge cases."""
        # Empty intersection
        neighbors1 = {"a", "b"}
        neighbors2 = {"c", "d"}
        shared = set()
        
        assert _calculate_projection_weight(neighbors1, neighbors2, shared, "count") == 0.0
        assert _calculate_projection_weight(neighbors1, neighbors2, shared, "jaccard") == 0.0
        assert _calculate_projection_weight(neighbors1, neighbors2, shared, "overlap") == 0.0
        
        # Empty neighbor sets
        empty1 = set()
        empty2 = set()
        empty_shared = set()
        
        assert _calculate_projection_weight(empty1, empty2, empty_shared, "jaccard") == 0.0
        assert _calculate_projection_weight(empty1, empty2, empty_shared, "overlap") == 0.0


class TestBipartiteInfo:
    """Test bipartite graph information functions."""
    
    def test_get_bipartite_info_valid(self):
        """Test getting info from valid bipartite graph."""
        edges = pl.DataFrame({
            "user": ["u1", "u2", "u3"],
            "item": ["i1", "i2", "i1"]
        })
        
        graph, id_mapper = build_graph_from_edgelist(
            edges,
            source_col="user",
            target_col="item",
            bipartite=True
        )
        
        info = get_bipartite_info(graph, id_mapper)
        
        assert info["is_bipartite"] is True
        assert info["source_partition_size"] + info["target_partition_size"] == 5
        assert info["total_nodes"] == 5
        assert info["total_edges"] == 3
        assert "source_nodes" in info
        assert "target_nodes" in info
    
    def test_get_bipartite_info_invalid(self):
        """Test getting info from non-bipartite graph."""
        edges = pl.DataFrame({
            "source": ["A", "B", "C"],
            "target": ["B", "C", "A"]  # Creates triangle
        })
        
        graph, id_mapper = build_graph_from_edgelist(edges)
        
        info = get_bipartite_info(graph, id_mapper)
        
        assert info["is_bipartite"] is False
        assert "error" in info


class TestProjectionIntegrationScenarios:
    """Test realistic bipartite projection scenarios."""
    
    def test_movie_recommendation_projection(self):
        """Test user-movie bipartite projection."""
        # Users rating movies
        edges = pl.DataFrame({
            "user": ["alice", "alice", "bob", "bob", "charlie"],
            "movie": ["titanic", "avatar", "titanic", "inception", "avatar"]
        })
        
        graph, id_mapper = build_graph_from_edgelist(
            edges,
            source_col="user",
            target_col="movie",
            bipartite=True
        )
        
        # Project onto users (find similar users)
        user_graph, user_mapper = project_bipartite(
            graph, id_mapper, "source", "jaccard"
        )
        
        assert user_graph.numberOfNodes() == 3
        
        # Alice and Bob both watched Titanic
        alice_id = user_mapper.get_internal("alice")
        bob_id = user_mapper.get_internal("bob")
        assert user_graph.hasEdge(alice_id, bob_id)
        
        # Alice and Charlie both watched Avatar  
        charlie_id = user_mapper.get_internal("charlie")
        assert user_graph.hasEdge(alice_id, charlie_id)
        
        # Project onto movies (find similar movies)
        movie_graph, movie_mapper = project_bipartite(
            graph, id_mapper, "target", "overlap"
        )
        
        assert movie_graph.numberOfNodes() == 4
        
        # Check that movies watched by same users are connected
        titanic_id = movie_mapper.get_internal("titanic")
        avatar_id = movie_mapper.get_internal("avatar")
        # Only Alice watched both Titanic and Avatar
        if movie_graph.hasEdge(titanic_id, avatar_id):
            assert movie_graph.weight(titanic_id, avatar_id) > 0
    
    def test_scientific_collaboration_projection(self):
        """Test author-paper collaboration projection."""
        edges = pl.DataFrame({
            "author": ["smith", "jones", "brown", "smith", "davis"],
            "paper": ["paper1", "paper1", "paper1", "paper2", "paper2"]
        })
        
        graph, id_mapper = build_graph_from_edgelist(
            edges,
            source_col="author",
            target_col="paper",
            bipartite=True
        )
        
        # Project onto authors (collaboration network)
        collab_graph, collab_mapper = project_bipartite(
            graph, id_mapper, "source", "count"
        )
        
        assert collab_graph.numberOfNodes() == 4
        
        # Smith, Jones, and Brown collaborated on paper1 
        smith_id = collab_mapper.get_internal("smith")
        jones_id = collab_mapper.get_internal("jones") 
        brown_id = collab_mapper.get_internal("brown")
        
        assert collab_graph.hasEdge(smith_id, jones_id)
        assert collab_graph.hasEdge(smith_id, brown_id)
        assert collab_graph.hasEdge(jones_id, brown_id)
        
        # Smith and Davis collaborated on paper2
        davis_id = collab_mapper.get_internal("davis")
        assert collab_graph.hasEdge(smith_id, davis_id)


if __name__ == "__main__":
    # Run tests if script is executed directly
    pytest.main([__file__])