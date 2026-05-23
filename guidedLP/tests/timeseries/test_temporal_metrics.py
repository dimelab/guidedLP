"""
Tests for temporal metrics module.

This module tests the temporal network metrics functionality including:
- Metric extraction across time slices
- Statistical aggregation over time
- Missing value handling
- Temporal alignment and consistency
"""

import pytest
from datetime import datetime
from unittest.mock import patch, MagicMock
import warnings

import polars as pl
import networkit as nk
import numpy as np

from src.timeseries.temporal_metrics import (
    extract_temporal_metrics,
    calculate_temporal_statistics,
    _validate_temporal_metrics_inputs,
    _validate_statistics_inputs,
    _create_temporal_grid,
    _create_statistic_expression,
    _calculate_trend_slope
)

from src.common.exceptions import ValidationError, ConfigurationError
from src.common.id_mapper import IDMapper


class TestExtractTemporalMetrics:
    """Test temporal metrics extraction functionality."""
    
    def create_sample_temporal_graphs(self):
        """Create sample temporal graphs for testing."""
        temporal_graphs = []
        
        # Time slice 1: Simple triangle
        mapper1 = IDMapper()
        mapper1.add_mapping("A", 0)
        mapper1.add_mapping("B", 1)
        mapper1.add_mapping("C", 2)
        
        graph1 = nk.Graph(3)
        graph1.addEdge(0, 1)  # A-B
        graph1.addEdge(1, 2)  # B-C
        graph1.addEdge(2, 0)  # C-A
        
        temporal_graphs.append((datetime(2024, 1, 1), graph1, mapper1))
        
        # Time slice 2: Add node D and more connections
        mapper2 = IDMapper()
        mapper2.add_mapping("A", 0)
        mapper2.add_mapping("B", 1)
        mapper2.add_mapping("C", 2)
        mapper2.add_mapping("D", 3)
        
        graph2 = nk.Graph(4)
        graph2.addEdge(0, 1)  # A-B
        graph2.addEdge(1, 2)  # B-C
        graph2.addEdge(2, 0)  # C-A
        graph2.addEdge(3, 0)  # D-A
        graph2.addEdge(3, 1)  # D-B
        
        temporal_graphs.append((datetime(2024, 1, 2), graph2, mapper2))
        
        # Time slice 3: Remove node C
        mapper3 = IDMapper()
        mapper3.add_mapping("A", 0)
        mapper3.add_mapping("B", 1)
        mapper3.add_mapping("D", 2)
        
        graph3 = nk.Graph(3)
        graph3.addEdge(0, 1)  # A-B
        graph3.addEdge(2, 0)  # D-A
        graph3.addEdge(2, 1)  # D-B
        
        temporal_graphs.append((datetime(2024, 1, 3), graph3, mapper3))
        
        return temporal_graphs
    
    def test_basic_metric_extraction(self):
        """Test basic temporal metric extraction."""
        temporal_graphs = self.create_sample_temporal_graphs()
        
        # Extract degree centrality
        metrics_df = extract_temporal_metrics(
            temporal_graphs,
            metrics=["degree"],
            n_jobs=1
        )
        
        # Check basic structure
        expected_columns = ["node_id", "date", "degree"]
        assert all(col in metrics_df.columns for col in expected_columns)
        
        # Should have all nodes across all time slices
        unique_nodes = set(metrics_df["node_id"].unique())
        expected_nodes = {"A", "B", "C", "D"}
        assert unique_nodes == expected_nodes
        
        # Should have 3 time slices
        unique_dates = metrics_df["date"].unique().sort()
        assert len(unique_dates) == 3
        
        # Each node should appear in each time slice
        assert len(metrics_df) == 4 * 3  # 4 nodes × 3 time slices
        
        # Check degree values for specific cases
        # Node A in slice 1 should have degree 2 (connected to B and C)
        slice1_a = metrics_df.filter(
            (pl.col("node_id") == "A") & 
            (pl.col("date") == datetime(2024, 1, 1))
        )
        assert len(slice1_a) == 1
        assert slice1_a["degree"].item() > 0  # Should have positive degree
        
        # Node D in slice 1 should have degree 0 (doesn't exist in graph)
        slice1_d = metrics_df.filter(
            (pl.col("node_id") == "D") & 
            (pl.col("date") == datetime(2024, 1, 1))
        )
        assert len(slice1_d) == 1
        assert slice1_d["degree"].item() == 0.0  # Missing node filled with 0
    
    def test_multiple_metrics_extraction(self):
        """Test extraction of multiple metrics simultaneously."""
        temporal_graphs = self.create_sample_temporal_graphs()
        
        # Extract multiple metrics
        metrics_df = extract_temporal_metrics(
            temporal_graphs,
            metrics=["degree", "betweenness"],
            n_jobs=1
        )
        
        # Check columns
        expected_columns = ["node_id", "date", "degree", "betweenness"]
        assert all(col in metrics_df.columns for col in expected_columns)
        
        # Check that all metric values are numeric
        assert metrics_df["degree"].dtype == pl.Float64
        assert metrics_df["betweenness"].dtype == pl.Float64
        
        # Check no null values (should be filled with 0)
        assert metrics_df["degree"].null_count() == 0
        assert metrics_df["betweenness"].null_count() == 0
    
    def test_empty_graph_handling(self):
        """Test handling of empty graphs in temporal sequence."""
        # Create temporal graphs with an empty graph
        temporal_graphs = []
        
        # Empty graph
        mapper1 = IDMapper()
        graph1 = nk.Graph(0)
        temporal_graphs.append((datetime(2024, 1, 1), graph1, mapper1))
        
        # Non-empty graph
        mapper2 = IDMapper()
        mapper2.add_mapping("A", 0)
        mapper2.add_mapping("B", 1)
        graph2 = nk.Graph(2)
        graph2.addEdge(0, 1)
        temporal_graphs.append((datetime(2024, 1, 2), graph2, mapper2))
        
        metrics_df = extract_temporal_metrics(
            temporal_graphs,
            metrics=["degree"]
        )
        
        # Should handle empty graphs gracefully
        assert not metrics_df.is_empty()
        
        # All nodes should appear in all time slices
        unique_nodes = set(metrics_df["node_id"].unique())
        assert "A" in unique_nodes
        assert "B" in unique_nodes
        
        # Nodes should have degree 0 in slice 1 (empty graph)
        slice1_data = metrics_df.filter(pl.col("date") == datetime(2024, 1, 1))
        assert all(slice1_data["degree"] == 0.0)
    
    def test_missing_nodes_filled_with_zeros(self):
        """Test that missing nodes are filled with zero values."""
        temporal_graphs = self.create_sample_temporal_graphs()
        
        metrics_df = extract_temporal_metrics(
            temporal_graphs,
            metrics=["degree"]
        )
        
        # Node C should have degree 0 in slice 3 (not present in graph)
        slice3_c = metrics_df.filter(
            (pl.col("node_id") == "C") & 
            (pl.col("date") == datetime(2024, 1, 3))
        )
        assert len(slice3_c) == 1
        assert slice3_c["degree"].item() == 0.0
        
        # Node D should have degree 0 in slice 1 (not present in graph)
        slice1_d = metrics_df.filter(
            (pl.col("node_id") == "D") & 
            (pl.col("date") == datetime(2024, 1, 1))
        )
        assert len(slice1_d) == 1
        assert slice1_d["degree"].item() == 0.0
    
    def test_temporal_ordering(self):
        """Test that results are properly ordered by date and node_id."""
        temporal_graphs = self.create_sample_temporal_graphs()
        
        metrics_df = extract_temporal_metrics(
            temporal_graphs,
            metrics=["degree"]
        )
        
        # Check that dates are in order
        dates = metrics_df["date"].unique().sort()
        assert dates[0] < dates[1] < dates[2]
        
        # Check that within each date, nodes are sorted
        for date in dates:
            slice_data = metrics_df.filter(pl.col("date") == date)
            nodes = slice_data["node_id"].to_list()
            assert nodes == sorted(nodes)


class TestCalculateTemporalStatistics:
    """Test temporal statistics calculation functionality."""
    
    def create_sample_temporal_metrics(self):
        """Create sample temporal metrics DataFrame for testing."""
        data = {
            "node_id": ["A", "A", "A", "B", "B", "B", "C", "C", "C"],
            "date": [
                datetime(2024, 1, 1), datetime(2024, 1, 2), datetime(2024, 1, 3),
                datetime(2024, 1, 1), datetime(2024, 1, 2), datetime(2024, 1, 3),
                datetime(2024, 1, 1), datetime(2024, 1, 2), datetime(2024, 1, 3)
            ],
            "degree": [2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 0.5, 0.5, 0.5],  # A increasing, B increasing, C stable
            "betweenness": [0.1, 0.2, 0.3, 0.0, 0.1, 0.2, 0.0, 0.0, 0.0]  # A increasing, B increasing, C zero
        }
        return pl.DataFrame(data)
    
    def test_basic_statistics_calculation(self):
        """Test basic statistical calculations."""
        temporal_metrics = self.create_sample_temporal_metrics()
        
        stats_df = calculate_temporal_statistics(
            temporal_metrics,
            statistics=["mean", "std"]
        )
        
        # Check structure
        expected_columns = ["node_id", "degree_mean", "degree_std", "betweenness_mean", "betweenness_std"]
        assert all(col in stats_df.columns for col in expected_columns)
        
        # Check node count
        assert len(stats_df) == 3  # A, B, C
        
        # Check specific calculations
        node_a = stats_df.filter(pl.col("node_id") == "A").to_dicts()[0]
        assert node_a["degree_mean"] == pytest.approx(3.0, abs=0.01)  # (2+3+4)/3
        assert node_a["degree_std"] == pytest.approx(1.0, abs=0.01)   # std([2,3,4])
        
        node_c = stats_df.filter(pl.col("node_id") == "C").to_dicts()[0]
        assert node_c["degree_mean"] == pytest.approx(0.5, abs=0.01)  # (0.5+0.5+0.5)/3
        assert node_c["degree_std"] == pytest.approx(0.0, abs=0.01)   # std([0.5,0.5,0.5]) = 0
    
    def test_trend_calculation(self):
        """Test linear trend calculation."""
        temporal_metrics = self.create_sample_temporal_metrics()
        
        stats_df = calculate_temporal_statistics(
            temporal_metrics,
            statistics=["trend"]
        )
        
        # Check structure
        expected_columns = ["node_id", "degree_trend", "betweenness_trend"]
        assert all(col in stats_df.columns for col in expected_columns)
        
        # Node A has increasing degree (2, 3, 4) - positive trend
        node_a = stats_df.filter(pl.col("node_id") == "A").to_dicts()[0]
        assert node_a["degree_trend"] > 0  # Positive trend
        
        # Node C has stable degree (0.5, 0.5, 0.5) - zero trend
        node_c = stats_df.filter(pl.col("node_id") == "C").to_dicts()[0]
        assert abs(node_c["degree_trend"]) < 0.01  # Near zero trend
    
    def test_volatility_calculation(self):
        """Test coefficient of variation calculation."""
        temporal_metrics = self.create_sample_temporal_metrics()
        
        stats_df = calculate_temporal_statistics(
            temporal_metrics,
            statistics=["volatility"]
        )
        
        # Check structure
        expected_columns = ["node_id", "degree_volatility", "betweenness_volatility"]
        assert all(col in stats_df.columns for col in expected_columns)
        
        # Node C has zero volatility (constant values)
        node_c = stats_df.filter(pl.col("node_id") == "C").to_dicts()[0]
        assert node_c["degree_volatility"] == pytest.approx(0.0, abs=0.01)
        
        # Node A should have some volatility (non-constant values)
        node_a = stats_df.filter(pl.col("node_id") == "A").to_dicts()[0]
        assert node_a["degree_volatility"] > 0
    
    def test_growth_calculation(self):
        """Test percentage growth calculation."""
        temporal_metrics = self.create_sample_temporal_metrics()
        
        stats_df = calculate_temporal_statistics(
            temporal_metrics,
            statistics=["growth"]
        )
        
        # Check structure
        expected_columns = ["node_id", "degree_growth", "betweenness_growth"]
        assert all(col in stats_df.columns for col in expected_columns)
        
        # Node A: degree grows from 2.0 to 4.0 = 100% growth
        node_a = stats_df.filter(pl.col("node_id") == "A").to_dicts()[0]
        assert node_a["degree_growth"] == pytest.approx(100.0, abs=0.01)  # (4-2)/2 * 100
        
        # Node C: degree stable at 0.5 = 0% growth
        node_c = stats_df.filter(pl.col("node_id") == "C").to_dicts()[0]
        assert node_c["degree_growth"] == pytest.approx(0.0, abs=0.01)  # (0.5-0.5)/0.5 * 100
    
    def test_min_max_range_calculation(self):
        """Test min, max, and range calculations."""
        temporal_metrics = self.create_sample_temporal_metrics()
        
        stats_df = calculate_temporal_statistics(
            temporal_metrics,
            statistics=["min", "max", "range"]
        )
        
        # Check structure
        expected_columns = ["node_id", "degree_min", "degree_max", "degree_range", 
                           "betweenness_min", "betweenness_max", "betweenness_range"]
        assert all(col in stats_df.columns for col in expected_columns)
        
        # Node A: degree values [2, 3, 4]
        node_a = stats_df.filter(pl.col("node_id") == "A").to_dicts()[0]
        assert node_a["degree_min"] == 2.0
        assert node_a["degree_max"] == 4.0
        assert node_a["degree_range"] == 2.0  # 4 - 2
    
    def test_all_statistics_combined(self):
        """Test calculation of all statistics together."""
        temporal_metrics = self.create_sample_temporal_metrics()
        
        stats_df = calculate_temporal_statistics(
            temporal_metrics,
            statistics=["mean", "std", "trend", "volatility", "growth", "min", "max", "range"]
        )
        
        # Should have many columns for each metric
        degree_columns = [col for col in stats_df.columns if col.startswith("degree_")]
        betweenness_columns = [col for col in stats_df.columns if col.startswith("betweenness_")]
        
        assert len(degree_columns) == 8  # 8 statistics
        assert len(betweenness_columns) == 8  # 8 statistics
        
        # All values should be numeric (no nulls)
        for col in degree_columns + betweenness_columns:
            assert stats_df[col].null_count() == 0
    
    def test_zero_growth_handling(self):
        """Test handling of zero initial values in growth calculation."""
        # Create data with zero initial values
        data = {
            "node_id": ["A", "A", "A"],
            "date": [datetime(2024, 1, 1), datetime(2024, 1, 2), datetime(2024, 1, 3)],
            "degree": [0.0, 1.0, 2.0]  # Starts from zero
        }
        temporal_metrics = pl.DataFrame(data)
        
        stats_df = calculate_temporal_statistics(
            temporal_metrics,
            statistics=["growth"]
        )
        
        # Growth from zero should handle gracefully (infinity or special value)
        node_a = stats_df.filter(pl.col("node_id") == "A").to_dicts()[0]
        growth_value = node_a["degree_growth"]
        
        # Should be a large positive number or infinity (depending on implementation)
        assert growth_value == float('inf') or growth_value > 1000


class TestInputValidation:
    """Test input validation for temporal metrics functions."""
    
    def test_empty_temporal_graphs(self):
        """Test validation with empty temporal graphs."""
        with pytest.raises(ValidationError, match="temporal_graphs cannot be empty"):
            extract_temporal_metrics([], metrics=["degree"])
    
    def test_empty_metrics_list(self):
        """Test validation with empty metrics list."""
        # Create minimal temporal graph
        mapper = IDMapper()
        mapper.add_mapping("A", 0)
        graph = nk.Graph(1)
        temporal_graphs = [(datetime(2024, 1, 1), graph, mapper)]
        
        with pytest.raises(ValidationError, match="metrics list cannot be empty"):
            extract_temporal_metrics(temporal_graphs, metrics=[])
    
    def test_invalid_metric_names(self):
        """Test validation with invalid metric names."""
        # Create minimal temporal graph
        mapper = IDMapper()
        mapper.add_mapping("A", 0)
        graph = nk.Graph(1)
        temporal_graphs = [(datetime(2024, 1, 1), graph, mapper)]
        
        with pytest.raises(ConfigurationError, match="Invalid metrics"):
            extract_temporal_metrics(temporal_graphs, metrics=["invalid_metric"])
    
    def test_invalid_temporal_graph_structure(self):
        """Test validation with malformed temporal graphs."""
        # Wrong tuple structure
        with pytest.raises(ValidationError, match="must be tuple of"):
            extract_temporal_metrics([("not", "a", "proper", "tuple")], metrics=["degree"])
        
        # Wrong types in tuple
        with pytest.raises(ValidationError, match="must be datetime"):
            extract_temporal_metrics([("not_datetime", nk.Graph(0), IDMapper())], metrics=["degree"])
    
    def test_empty_temporal_metrics_dataframe(self):
        """Test statistics calculation with empty DataFrame."""
        empty_df = pl.DataFrame()
        
        with pytest.raises(ValidationError, match="temporal_metrics DataFrame cannot be empty"):
            calculate_temporal_statistics(empty_df, statistics=["mean"])
    
    def test_missing_required_columns(self):
        """Test statistics calculation with missing required columns."""
        # DataFrame missing 'date' column
        incomplete_df = pl.DataFrame({
            "node_id": ["A", "B"],
            "degree": [1.0, 2.0]
        })
        
        with pytest.raises(ValidationError, match="Missing required columns"):
            calculate_temporal_statistics(incomplete_df, statistics=["mean"])
    
    def test_invalid_statistic_names(self):
        """Test validation with invalid statistic names."""
        temporal_metrics = pl.DataFrame({
            "node_id": ["A", "B"],
            "date": [datetime(2024, 1, 1), datetime(2024, 1, 1)],
            "degree": [1.0, 2.0]
        })
        
        with pytest.raises(ConfigurationError, match="Invalid statistics"):
            calculate_temporal_statistics(temporal_metrics, statistics=["invalid_stat"])


class TestHelperFunctions:
    """Test individual helper functions."""
    
    def test_create_temporal_grid(self):
        """Test temporal grid creation."""
        nodes = pl.Series("node_id", ["A", "B", "C"])
        dates = pl.Series("date", [datetime(2024, 1, 1), datetime(2024, 1, 2)])
        metrics = ["degree", "betweenness"]
        
        grid = _create_temporal_grid(nodes, dates, metrics)
        
        # Should have 3 nodes × 2 dates = 6 rows
        assert len(grid) == 6
        
        # Should have correct columns
        expected_columns = ["node_id", "date", "degree", "betweenness"]
        assert all(col in grid.columns for col in expected_columns)
        
        # All combinations should be present
        combinations = set()
        for row in grid.iter_rows(named=True):
            combinations.add((row["node_id"], row["date"]))
        
        expected_combinations = {
            ("A", datetime(2024, 1, 1)), ("A", datetime(2024, 1, 2)),
            ("B", datetime(2024, 1, 1)), ("B", datetime(2024, 1, 2)),
            ("C", datetime(2024, 1, 1)), ("C", datetime(2024, 1, 2))
        }
        assert combinations == expected_combinations
    
    def test_create_statistic_expression_basic(self):
        """Test basic statistic expression creation."""
        # Test mean expression
        mean_expr = _create_statistic_expression("degree", "mean")
        assert mean_expr is not None
        
        # Test with sample data
        test_df = pl.DataFrame({"degree": [1.0, 2.0, 3.0]})
        result = test_df.select(mean_expr)
        assert result["degree_mean"].item() == 2.0
    
    def test_create_statistic_expression_volatility(self):
        """Test volatility (coefficient of variation) expression."""
        volatility_expr = _create_statistic_expression("degree", "volatility")
        assert volatility_expr is not None
        
        # Test with sample data
        test_df = pl.DataFrame({"degree": [1.0, 2.0, 3.0]})  # mean=2, std=1, volatility=0.5
        result = test_df.select(volatility_expr)
        assert result["degree_volatility"].item() == pytest.approx(0.5, abs=0.01)
        
        # Test with zero mean (should return 0)
        zero_df = pl.DataFrame({"degree": [0.0, 0.0, 0.0]})
        result = zero_df.select(volatility_expr)
        assert result["degree_volatility"].item() == 0.0
    
    def test_calculate_trend_slope(self):
        """Test trend slope calculation."""
        # Increasing values
        increasing = pl.Series([1.0, 2.0, 3.0, 4.0])
        slope = _calculate_trend_slope(increasing)
        assert slope > 0  # Positive slope
        
        # Decreasing values
        decreasing = pl.Series([4.0, 3.0, 2.0, 1.0])
        slope = _calculate_trend_slope(decreasing)
        assert slope < 0  # Negative slope
        
        # Constant values
        constant = pl.Series([2.0, 2.0, 2.0, 2.0])
        slope = _calculate_trend_slope(constant)
        assert abs(slope) < 0.01  # Near zero slope
        
        # Single value (should return 0)
        single = pl.Series([1.0])
        slope = _calculate_trend_slope(single)
        assert slope == 0.0
        
        # Empty series (should return 0)
        empty = pl.Series([], dtype=pl.Float64)
        slope = _calculate_trend_slope(empty)
        assert slope == 0.0


class TestIntegrationWithMockedMetrics:
    """Test integration with mocked centrality calculations."""
    
    @patch('src.timeseries.temporal_metrics.extract_centrality')
    def test_extract_temporal_metrics_with_mock(self, mock_extract_centrality):
        """Test temporal metrics extraction with mocked centrality function."""
        # Setup mock return values
        mock_extract_centrality.side_effect = [
            # Slice 1 results
            pl.DataFrame({
                "node_id": ["A", "B"],
                "degree_centrality": [0.5, 0.3]
            }),
            # Slice 2 results
            pl.DataFrame({
                "node_id": ["A", "B", "C"],
                "degree_centrality": [0.6, 0.4, 0.2]
            })
        ]
        
        # Create sample temporal graphs
        temporal_graphs = []
        for i, date in enumerate([datetime(2024, 1, 1), datetime(2024, 1, 2)]):
            mapper = IDMapper()
            graph = nk.Graph(i + 2)  # Different sizes
            temporal_graphs.append((date, graph, mapper))
        
        # Extract metrics
        result = extract_temporal_metrics(
            temporal_graphs,
            metrics=["degree"],
            n_jobs=1
        )
        
        # Check that extract_centrality was called for each slice
        assert mock_extract_centrality.call_count == 2
        
        # Check result structure
        assert not result.is_empty()
        assert "node_id" in result.columns
        assert "date" in result.columns
        assert "degree" in result.columns
        
        # Check that all nodes appear in all slices
        unique_nodes = set(result["node_id"].unique())
        assert "A" in unique_nodes
        assert "B" in unique_nodes  
        assert "C" in unique_nodes
        
        # Check that missing values are filled with 0
        slice1_c = result.filter(
            (pl.col("node_id") == "C") & 
            (pl.col("date") == datetime(2024, 1, 1))
        )
        assert len(slice1_c) == 1
        assert slice1_c["degree"].item() == 0.0  # C wasn't in slice 1


class TestEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_single_time_slice(self):
        """Test with only one time slice."""
        mapper = IDMapper()
        mapper.add_mapping("A", 0)
        mapper.add_mapping("B", 1)
        graph = nk.Graph(2)
        graph.addEdge(0, 1)
        
        temporal_graphs = [(datetime(2024, 1, 1), graph, mapper)]
        
        metrics_df = extract_temporal_metrics(
            temporal_graphs,
            metrics=["degree"]
        )
        
        # Should work with single slice
        assert not metrics_df.is_empty()
        assert len(metrics_df) == 2  # 2 nodes
        assert all(metrics_df["date"] == datetime(2024, 1, 1))
    
    def test_statistics_with_single_time_point(self):
        """Test statistics calculation with single time point."""
        temporal_metrics = pl.DataFrame({
            "node_id": ["A", "B"],
            "date": [datetime(2024, 1, 1), datetime(2024, 1, 1)],
            "degree": [2.0, 1.0]
        })
        
        stats_df = calculate_temporal_statistics(
            temporal_metrics,
            statistics=["mean", "std", "trend"]
        )
        
        # Should handle single time point gracefully
        assert not stats_df.is_empty()
        
        # Standard deviation should be 0 for single point
        assert all(stats_df["degree_std"] == 0.0)
        
        # Trend should be 0 for single point
        assert all(stats_df["degree_trend"] == 0.0)
    
    def test_all_zero_metrics(self):
        """Test handling of all-zero metric values."""
        temporal_metrics = pl.DataFrame({
            "node_id": ["A", "A", "A"],
            "date": [datetime(2024, 1, 1), datetime(2024, 1, 2), datetime(2024, 1, 3)],
            "degree": [0.0, 0.0, 0.0]
        })
        
        stats_df = calculate_temporal_statistics(
            temporal_metrics,
            statistics=["mean", "std", "volatility", "growth"]
        )
        
        # Should handle all zeros gracefully
        node_a = stats_df.filter(pl.col("node_id") == "A").to_dicts()[0]
        
        assert node_a["degree_mean"] == 0.0
        assert node_a["degree_std"] == 0.0
        assert node_a["degree_volatility"] == 0.0  # 0/0 handled as 0
        assert node_a["degree_growth"] == 0.0  # 0% growth