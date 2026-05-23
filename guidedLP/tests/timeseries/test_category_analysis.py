"""
Tests for category analysis functionality in temporal networks.
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock

import polars as pl
import networkit as nk

from src.timeseries.category_analysis import (
    analyze_cross_category_connections,
    calculate_category_segregation_index,
    analyze_category_centrality_by_time
)
from src.common.id_mapper import IDMapper


class TestAnalyzeCrossCategoryConnections:
    """Test the analyze_cross_category_connections function."""
    
    @pytest.fixture
    def sample_metadata(self):
        """Create sample metadata with node categories."""
        return pl.DataFrame({
            "node_id": ["A", "B", "C", "D", "E", "F"],
            "department": ["sales", "sales", "engineering", "engineering", "marketing", "marketing"],
            "level": ["junior", "senior", "junior", "senior", "junior", "senior"]
        })
    
    @pytest.fixture
    def sample_temporal_graphs(self):
        """Create sample temporal graphs for testing."""
        graphs = []
        
        # Day 1: Sales-heavy connections
        graph1 = nk.Graph(6, weighted=False, directed=False)
        mapper1 = IDMapper()
        
        # Map external to internal IDs
        nodes = ["A", "B", "C", "D", "E", "F"]
        id_map = {}
        for i, node in enumerate(nodes):
            mapper1.add_mapping(node, i)
            id_map[node] = i
        
        # Add edges: mostly within sales and engineering
        graph1.addEdge(id_map["A"], id_map["B"])  # sales-sales
        graph1.addEdge(id_map["A"], id_map["C"])  # sales-engineering
        graph1.addEdge(id_map["C"], id_map["D"])  # engineering-engineering
        graph1.addEdge(id_map["E"], id_map["F"])  # marketing-marketing
        
        graphs.append((datetime(2024, 1, 1), graph1, mapper1))
        
        # Day 2: More cross-department connections
        graph2 = nk.Graph(6, weighted=False, directed=False)
        mapper2 = IDMapper()
        
        id_map2 = {}
        for i, node in enumerate(nodes):
            mapper2.add_mapping(node, i)
            id_map2[node] = i
        
        # Add edges: more cross-department
        graph2.addEdge(id_map2["A"], id_map2["E"])  # sales-marketing
        graph2.addEdge(id_map2["B"], id_map2["D"])  # sales-engineering
        graph2.addEdge(id_map2["C"], id_map2["F"])  # engineering-marketing
        graph2.addEdge(id_map2["D"], id_map2["E"])  # engineering-marketing
        
        graphs.append((datetime(2024, 1, 2), graph2, mapper2))
        
        return graphs
    
    def test_basic_category_analysis(self, sample_temporal_graphs, sample_metadata):
        """Test basic category connection analysis."""
        result = analyze_cross_category_connections(
            sample_temporal_graphs,
            sample_metadata,
            "department"
        )
        
        assert not result.is_empty()
        assert set(result.columns) == {"date", "category_a", "category_b", "connection_strength", "edge_count"}
        
        # Check that we have data for both dates
        dates = result["date"].unique().to_list()
        assert len(dates) == 2
        assert datetime(2024, 1, 1) in dates
        assert datetime(2024, 1, 2) in dates
        
        # Check that connection strength equals edge count for unweighted graphs
        assert (result["connection_strength"] == result["edge_count"]).all()
    
    def test_category_pair_ordering(self, sample_temporal_graphs, sample_metadata):
        """Test that category pairs are ordered alphabetically to avoid duplicates."""
        result = analyze_cross_category_connections(
            sample_temporal_graphs,
            sample_metadata,
            "department"
        )
        
        # Check that all category pairs are ordered alphabetically
        for row in result.iter_rows(named=True):
            cat_a, cat_b = row["category_a"], row["category_b"]
            assert cat_a <= cat_b, f"Categories not ordered: {cat_a} > {cat_b}"
    
    def test_weighted_graph_analysis(self, sample_metadata):
        """Test analysis with weighted graphs."""
        # Create weighted graph
        graph = nk.Graph(4, weighted=True, directed=False)
        mapper = IDMapper()
        
        nodes = ["A", "B", "C", "D"]
        id_map = {}
        for i, node in enumerate(nodes):
            mapper.add_mapping(node, i)
            id_map[node] = i
        
        # Add weighted edges
        graph.addEdge(id_map["A"], id_map["B"], 2.5)  # sales-sales
        graph.addEdge(id_map["A"], id_map["C"], 1.0)  # sales-engineering
        graph.addEdge(id_map["C"], id_map["D"], 3.0)  # engineering-engineering
        
        temporal_graphs = [(datetime(2024, 1, 1), graph, mapper)]
        
        # Test sum aggregation
        result_sum = analyze_cross_category_connections(
            temporal_graphs, sample_metadata, "department", edge_weight="sum"
        )
        
        # Test mean aggregation
        result_mean = analyze_cross_category_connections(
            temporal_graphs, sample_metadata, "department", edge_weight="mean"
        )
        
        # Find engineering-engineering connection
        eng_eng = result_sum.filter(
            (pl.col("category_a") == "engineering") & 
            (pl.col("category_b") == "engineering")
        )
        assert len(eng_eng) == 1
        assert eng_eng["connection_strength"][0] == 3.0
        assert eng_eng["edge_count"][0] == 1
        
        # Check mean calculation
        eng_eng_mean = result_mean.filter(
            (pl.col("category_a") == "engineering") & 
            (pl.col("category_b") == "engineering")
        )
        assert eng_eng_mean["connection_strength"][0] == 3.0  # single edge, so mean = sum
    
    def test_missing_category_handling(self, sample_temporal_graphs):
        """Test handling of nodes not in metadata."""
        # Metadata missing some nodes
        incomplete_metadata = pl.DataFrame({
            "node_id": ["A", "B", "C"],
            "department": ["sales", "sales", "engineering"]
        })
        
        result = analyze_cross_category_connections(
            sample_temporal_graphs,
            incomplete_metadata,
            "department"
        )
        
        # Should have "unknown" category for missing nodes
        categories = set()
        for row in result.iter_rows(named=True):
            categories.add(row["category_a"])
            categories.add(row["category_b"])
        
        assert "unknown" in categories
    
    def test_empty_graph_handling(self, sample_metadata):
        """Test handling of empty graphs."""
        # Create empty graph
        empty_graph = nk.Graph(0, weighted=False, directed=False)
        empty_mapper = IDMapper()
        
        temporal_graphs = [(datetime(2024, 1, 1), empty_graph, empty_mapper)]
        
        result = analyze_cross_category_connections(
            temporal_graphs, sample_metadata, "department"
        )
        
        # Should return empty DataFrame with correct schema
        assert result.is_empty()
        assert set(result.columns) == {"date", "category_a", "category_b", "connection_strength", "edge_count"}
    
    def test_input_validation(self, sample_temporal_graphs, sample_metadata):
        """Test input validation."""
        # Empty temporal graphs
        with pytest.raises(ValueError, match="temporal_graphs cannot be empty"):
            analyze_cross_category_connections([], sample_metadata, "department")
        
        # Empty metadata
        empty_metadata = pl.DataFrame()
        with pytest.raises(ValueError, match="metadata cannot be empty"):
            analyze_cross_category_connections(sample_temporal_graphs, empty_metadata, "department")
        
        # Missing required columns
        bad_metadata = pl.DataFrame({"name": ["A", "B"], "dept": ["sales", "engineering"]})
        with pytest.raises(ValueError, match="metadata missing required columns"):
            analyze_cross_category_connections(sample_temporal_graphs, bad_metadata, "department")
        
        # Invalid edge weight
        with pytest.raises(ValueError, match="edge_weight must be one of"):
            analyze_cross_category_connections(
                sample_temporal_graphs, sample_metadata, "department", edge_weight="invalid"
            )
    
    def test_different_category_columns(self, sample_temporal_graphs, sample_metadata):
        """Test analysis with different category columns."""
        # Use 'level' instead of 'department'
        result = analyze_cross_category_connections(
            sample_temporal_graphs,
            sample_metadata,
            "level"
        )
        
        assert not result.is_empty()
        
        # Should have junior/senior categories
        categories = set()
        for row in result.iter_rows(named=True):
            categories.add(row["category_a"])
            categories.add(row["category_b"])
        
        assert "junior" in categories
        assert "senior" in categories


class TestCalculateCategorySegregationIndex:
    """Test the calculate_category_segregation_index function."""
    
    @pytest.fixture
    def sample_connections(self):
        """Create sample category connections data."""
        return pl.DataFrame({
            "date": [datetime(2024, 1, 1)] * 4 + [datetime(2024, 1, 2)] * 3,
            "category_a": ["sales", "sales", "engineering", "marketing", 
                          "sales", "engineering", "marketing"],
            "category_b": ["sales", "engineering", "engineering", "marketing",
                          "sales", "engineering", "marketing"], 
            "connection_strength": [10, 5, 8, 3, 15, 12, 6],
            "edge_count": [10, 5, 8, 3, 15, 12, 6]
        })
    
    def test_basic_segregation_calculation(self, sample_connections):
        """Test basic segregation index calculation."""
        result = calculate_category_segregation_index(sample_connections)
        
        assert not result.is_empty()
        assert set(result.columns) == {
            "date", "segregation_index", "within_category_strength", "between_category_strength"
        }
        
        # Check first date calculation
        day1 = result.filter(pl.col("date") == datetime(2024, 1, 1))
        assert len(day1) == 1
        
        # Within-category: sales-sales (10) + engineering-engineering (8) + marketing-marketing (3) = 21
        # Between-category: sales-engineering (5) = 5
        # Total: 26
        # Segregation index: 21/26 ≈ 0.808
        
        assert abs(day1["within_category_strength"][0] - 21) < 1e-6
        assert abs(day1["between_category_strength"][0] - 5) < 1e-6
        assert abs(day1["segregation_index"][0] - 21/26) < 1e-6
    
    def test_empty_connections(self):
        """Test with empty connections DataFrame."""
        empty_df = pl.DataFrame({
            "date": [],
            "category_a": [],
            "category_b": [],
            "connection_strength": [],
            "edge_count": []
        })
        
        result = calculate_category_segregation_index(empty_df)
        assert result.is_empty()
        assert set(result.columns) == {
            "date", "segregation_index", "within_category_strength", "between_category_strength"
        }
    
    def test_only_within_category_connections(self):
        """Test with only within-category connections."""
        within_only = pl.DataFrame({
            "date": [datetime(2024, 1, 1)] * 2,
            "category_a": ["sales", "engineering"],
            "category_b": ["sales", "engineering"],
            "connection_strength": [10, 5],
            "edge_count": [10, 5]
        })
        
        result = calculate_category_segregation_index(within_only)
        
        assert len(result) == 1
        assert result["segregation_index"][0] == 1.0  # Complete segregation
        assert result["within_category_strength"][0] == 15
        assert result["between_category_strength"][0] == 0
    
    def test_only_between_category_connections(self):
        """Test with only between-category connections."""
        between_only = pl.DataFrame({
            "date": [datetime(2024, 1, 1)] * 2,
            "category_a": ["sales", "engineering"],
            "category_b": ["engineering", "marketing"],
            "connection_strength": [10, 5],
            "edge_count": [10, 5]
        })
        
        result = calculate_category_segregation_index(between_only)
        
        assert len(result) == 1
        assert result["segregation_index"][0] == 0.0  # Complete integration
        assert result["within_category_strength"][0] == 0
        assert result["between_category_strength"][0] == 15


class TestAnalyzeCategoryCentralityByTime:
    """Test the analyze_category_centrality_by_time function."""
    
    @pytest.fixture
    def sample_metadata(self):
        """Create sample metadata."""
        return pl.DataFrame({
            "node_id": ["A", "B", "C", "D"],
            "department": ["sales", "sales", "engineering", "engineering"]
        })
    
    @pytest.fixture
    def mock_extract_centrality(self):
        """Mock the extract_centrality function."""
        with patch('src.network.analysis.extract_centrality') as mock:
            # Return sample centrality data
            mock.return_value = pl.DataFrame({
                "node_id": ["A", "B", "C", "D"],
                "degree": [3, 5, 2, 4]
            })
            yield mock
    
    def test_basic_centrality_analysis(self, sample_metadata, mock_extract_centrality):
        """Test basic category centrality analysis."""
        # Create sample temporal graphs
        graph = nk.Graph(4, weighted=False, directed=False)
        mapper = IDMapper()
        
        # Add nodes to mapper
        nodes = ["A", "B", "C", "D"]
        for i, node_id in enumerate(nodes):
            mapper.add_mapping(node_id, i)
        
        temporal_graphs = [(datetime(2024, 1, 1), graph, mapper)]
        
        result = analyze_category_centrality_by_time(
            temporal_graphs,
            sample_metadata,
            "department",
            "degree"
        )
        
        assert not result.is_empty()
        assert set(result.columns) == {
            "date", "category", "mean_centrality", "median_centrality", 
            "max_centrality", "node_count"
        }
        
        # Check that both departments are present
        departments = result["category"].unique().to_list()
        assert "sales" in departments
        assert "engineering" in departments
        
        # Verify calculations
        sales_data = result.filter(pl.col("category") == "sales")
        assert len(sales_data) == 1
        assert sales_data["mean_centrality"][0] == 4.0  # (3+5)/2
        assert sales_data["node_count"][0] == 2
        assert sales_data["max_centrality"][0] == 5.0
    
    def test_empty_graph_handling(self, sample_metadata, mock_extract_centrality):
        """Test handling of empty graphs."""
        empty_graph = nk.Graph(0, weighted=False, directed=False)
        empty_mapper = IDMapper()
        
        temporal_graphs = [(datetime(2024, 1, 1), empty_graph, empty_mapper)]
        
        result = analyze_category_centrality_by_time(
            temporal_graphs,
            sample_metadata,
            "department"
        )
        
        # Should return empty DataFrame
        assert result.is_empty()
        assert set(result.columns) == {
            "date", "category", "mean_centrality", "median_centrality", 
            "max_centrality", "node_count"
        }
    
    def test_centrality_calculation_failure(self, sample_metadata):
        """Test handling when centrality calculation fails."""
        with patch('src.network.analysis.extract_centrality') as mock:
            mock.side_effect = Exception("Centrality calculation failed")
            
            graph = nk.Graph(4, weighted=False, directed=False)
            mapper = IDMapper()
            temporal_graphs = [(datetime(2024, 1, 1), graph, mapper)]
            
            with pytest.warns(UserWarning, match="Failed to calculate centrality"):
                result = analyze_category_centrality_by_time(
                    temporal_graphs,
                    sample_metadata,
                    "department"
                )
                
                assert result.is_empty()
    
    def test_input_validation(self, sample_metadata):
        """Test input validation."""
        # Empty temporal graphs
        with pytest.raises(ValueError, match="temporal_graphs cannot be empty"):
            analyze_category_centrality_by_time([], sample_metadata, "department")
        
        # Empty metadata
        empty_metadata = pl.DataFrame()
        graph = nk.Graph(1, weighted=False, directed=False)
        mapper = IDMapper()
        temporal_graphs = [(datetime(2024, 1, 1), graph, mapper)]
        
        with pytest.raises(ValueError, match="metadata cannot be empty"):
            analyze_category_centrality_by_time(temporal_graphs, empty_metadata, "department")
    
    def test_unknown_category_handling(self, mock_extract_centrality):
        """Test handling of nodes with unknown categories."""
        # Metadata missing some nodes
        incomplete_metadata = pl.DataFrame({
            "node_id": ["A", "B"],
            "department": ["sales", "sales"]
        })
        
        # Mock returns data for all nodes including C, D not in metadata
        mock_extract_centrality.return_value = pl.DataFrame({
            "node_id": ["A", "B", "C", "D"],
            "degree": [3, 5, 2, 4]
        })
        
        graph = nk.Graph(4, weighted=False, directed=False)
        mapper = IDMapper()
        temporal_graphs = [(datetime(2024, 1, 1), graph, mapper)]
        
        result = analyze_category_centrality_by_time(
            temporal_graphs,
            incomplete_metadata,
            "department"
        )
        
        # Should have both "sales" and "unknown" categories
        categories = result["category"].unique().to_list()
        assert "sales" in categories
        assert "unknown" in categories
        
        # Check unknown category stats
        unknown_data = result.filter(pl.col("category") == "unknown")
        assert len(unknown_data) == 1
        assert unknown_data["mean_centrality"][0] == 3.0  # (2+4)/2
        assert unknown_data["node_count"][0] == 2


class TestIntegration:
    """Integration tests for category analysis functions."""
    
    def test_end_to_end_category_analysis(self):
        """Test complete category analysis workflow."""
        # Create comprehensive test data
        metadata = pl.DataFrame({
            "node_id": ["A", "B", "C", "D", "E", "F"],
            "department": ["sales", "sales", "engineering", "engineering", "marketing", "marketing"],
            "level": ["junior", "senior", "junior", "senior", "junior", "senior"]
        })
        
        # Create temporal graphs
        graphs = []
        for day in range(3):
            graph = nk.Graph(6, weighted=False, directed=False)
            mapper = IDMapper()
            
            # Create node mappings
            nodes = ["A", "B", "C", "D", "E", "F"]
            for i, node in enumerate(nodes):
                mapper.add_mapping(node, i)
            
            # Add different edge patterns each day
            if day == 0:
                # Day 0: High segregation
                graph.addEdge(0, 1)  # A-B: sales-sales
                graph.addEdge(2, 3)  # C-D: eng-eng
                graph.addEdge(4, 5)  # E-F: marketing-marketing
            elif day == 1:
                # Day 1: Mixed connections
                graph.addEdge(0, 2)  # A-C: sales-eng
                graph.addEdge(1, 4)  # B-E: sales-marketing
                graph.addEdge(3, 5)  # D-F: eng-marketing
            else:
                # Day 2: High integration
                graph.addEdge(0, 4)  # A-E: sales-marketing
                graph.addEdge(1, 3)  # B-D: sales-eng
                graph.addEdge(2, 5)  # C-F: eng-marketing
            
            graphs.append((datetime(2024, 1, day + 1), graph, mapper))
        
        # Test cross-category connections
        connections = analyze_cross_category_connections(
            graphs, metadata, "department"
        )
        
        assert not connections.is_empty()
        assert len(connections["date"].unique()) == 3
        
        # Test segregation index
        segregation = calculate_category_segregation_index(connections)
        
        assert not segregation.is_empty()
        assert len(segregation) == 3
        
        # Day 0 should have high segregation (all within-category)
        day0_seg = segregation.filter(pl.col("date") == datetime(2024, 1, 1))
        assert day0_seg["segregation_index"][0] == 1.0
        
        # Day 2 should have low segregation (all between-category)
        day2_seg = segregation.filter(pl.col("date") == datetime(2024, 1, 3))
        assert day2_seg["segregation_index"][0] == 0.0