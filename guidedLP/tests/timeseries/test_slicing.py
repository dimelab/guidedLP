"""
Tests for temporal slicing module.

This module tests the temporal network slicing functionality including:
- Different time intervals (daily, weekly, monthly, yearly)
- Rolling window support
- Cumulative graph construction
- Date filtering and edge cases
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock
import warnings

import polars as pl
import networkit as nk

from src.timeseries.slicing import (
    create_temporal_slices,
    align_node_ids_across_slices,
    _validate_temporal_inputs,
    _load_and_prepare_data,
    _filter_by_date_range,
    _generate_date_slices,
    _create_standard_slices,
    _create_cumulative_slices,
    _create_rolling_window_slices,
    _get_slice_start,
    _get_window_start
)

from src.common.exceptions import ValidationError, DataFormatError, ConfigurationError
from src.common.id_mapper import IDMapper


class TestCreateTemporalSlices:
    """Test main temporal slicing functionality."""
    
    def create_sample_data(self) -> pl.DataFrame:
        """Create sample temporal edge data for testing."""
        return pl.DataFrame({
            "source": ["A", "B", "C", "A", "B", "C", "D", "E"],
            "target": ["B", "C", "D", "E", "A", "E", "A", "B"],
            "timestamp": [
                "2024-01-01 10:00:00",
                "2024-01-01 14:00:00", 
                "2024-01-02 09:00:00",
                "2024-01-02 15:00:00",
                "2024-01-03 11:00:00",
                "2024-01-03 16:00:00",
                "2024-01-04 12:00:00",
                "2024-01-05 13:00:00"
            ]
        })
    
    def test_basic_daily_slicing(self):
        """Test basic daily temporal slicing."""
        data = self.create_sample_data()
        
        slices = create_temporal_slices(
            data,
            slice_interval="daily",
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 1, 3, 23, 59, 59)
        )
        
        # Should have 3 daily slices
        assert len(slices) == 3
        
        # Check slice dates
        expected_dates = [
            datetime(2024, 1, 1, 23, 59, 59, 999999),
            datetime(2024, 1, 2, 23, 59, 59, 999999),
            datetime(2024, 1, 3, 23, 59, 59, 999999)
        ]
        
        for i, (date, graph, mapper) in enumerate(slices):
            assert date == expected_dates[i]
            assert isinstance(graph, nk.Graph)
            assert isinstance(mapper, IDMapper)
            
        # First slice should have 2 edges (A-B, B-C)
        assert slices[0][1].numberOfEdges() == 2
        
        # Second slice should have 2 edges (C-D, A-E)
        assert slices[1][1].numberOfEdges() == 2
        
        # Third slice should have 2 edges (B-A, C-E)
        assert slices[2][1].numberOfEdges() == 2
    
    def test_weekly_slicing(self):
        """Test weekly temporal slicing."""
        # Create data spanning multiple weeks
        data = pl.DataFrame({
            "source": ["A", "B", "C", "D"],
            "target": ["B", "C", "D", "A"],
            "timestamp": [
                "2024-01-01 10:00:00",  # Monday
                "2024-01-03 10:00:00",  # Wednesday  
                "2024-01-08 10:00:00",  # Next Monday
                "2024-01-10 10:00:00"   # Wednesday
            ]
        })
        
        slices = create_temporal_slices(
            data,
            slice_interval="weekly",
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 1, 14)
        )
        
        # Should have 2 weekly slices
        assert len(slices) == 2
        
        # First week should have 2 edges
        assert slices[0][1].numberOfEdges() == 2
        
        # Second week should have 2 edges
        assert slices[1][1].numberOfEdges() == 2
    
    def test_monthly_slicing(self):
        """Test monthly temporal slicing."""
        data = pl.DataFrame({
            "source": ["A", "B", "C", "D"],
            "target": ["B", "C", "D", "A"],
            "timestamp": [
                "2024-01-15 10:00:00",
                "2024-01-20 10:00:00",
                "2024-02-10 10:00:00",
                "2024-02-25 10:00:00"
            ]
        })
        
        slices = create_temporal_slices(
            data,
            slice_interval="monthly",
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 2, 28)
        )
        
        # Should have 2 monthly slices
        assert len(slices) == 2
        
        # Check slice dates (end of months)
        assert slices[0][0].month == 1
        assert slices[0][0].day == 31
        assert slices[1][0].month == 2
        assert slices[1][0].day == 29  # 2024 is leap year
        
        # Each month should have 2 edges
        assert slices[0][1].numberOfEdges() == 2
        assert slices[1][1].numberOfEdges() == 2
    
    def test_yearly_slicing(self):
        """Test yearly temporal slicing."""
        data = pl.DataFrame({
            "source": ["A", "B", "C", "D"],
            "target": ["B", "C", "D", "A"],
            "timestamp": [
                "2023-06-15 10:00:00",
                "2023-12-20 10:00:00",
                "2024-03-10 10:00:00",
                "2024-09-25 10:00:00"
            ]
        })
        
        slices = create_temporal_slices(
            data,
            slice_interval="yearly",
            start_date=datetime(2023, 1, 1),
            end_date=datetime(2024, 12, 31)
        )
        
        # Should have 2 yearly slices
        assert len(slices) == 2
        
        # Check slice dates (end of years)
        assert slices[0][0].year == 2023
        assert slices[0][0].month == 12
        assert slices[0][0].day == 31
        assert slices[1][0].year == 2024
        
        # Each year should have 2 edges
        assert slices[0][1].numberOfEdges() == 2
        assert slices[1][1].numberOfEdges() == 2
    
    def test_rolling_window_slicing(self):
        """Test rolling window temporal slicing."""
        data = self.create_sample_data()
        
        # 2-day rolling window
        slices = create_temporal_slices(
            data,
            slice_interval="daily",
            rolling_window=2,
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 1, 4, 23, 59, 59)
        )
        
        # Should have 4 daily slices
        assert len(slices) == 4
        
        # First slice (day 1): includes day 1 only -> 2 edges
        assert slices[0][1].numberOfEdges() == 2
        
        # Second slice (day 2): includes days 1-2 -> 4 edges
        assert slices[1][1].numberOfEdges() == 4
        
        # Third slice (day 3): includes days 2-3 -> 4 edges
        assert slices[2][1].numberOfEdges() == 4
        
        # Fourth slice (day 4): includes days 3-4 -> 3 edges
        assert slices[3][1].numberOfEdges() == 3
    
    def test_cumulative_slicing(self):
        """Test cumulative temporal slicing."""
        data = self.create_sample_data()
        
        slices = create_temporal_slices(
            data,
            slice_interval="daily",
            cumulative=True,
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 1, 4, 23, 59, 59)
        )
        
        # Should have 4 daily slices
        assert len(slices) == 4
        
        # Cumulative: each slice should have more edges than previous
        assert slices[0][1].numberOfEdges() == 2  # Day 1: 2 edges
        assert slices[1][1].numberOfEdges() == 4  # Days 1-2: 4 edges
        assert slices[2][1].numberOfEdges() == 6  # Days 1-3: 6 edges
        assert slices[3][1].numberOfEdges() == 7  # Days 1-4: 7 edges
    
    def test_date_filtering(self):
        """Test start_date and end_date filtering."""
        data = self.create_sample_data()
        
        # Filter to middle dates only
        slices = create_temporal_slices(
            data,
            slice_interval="daily",
            start_date=datetime(2024, 1, 2),
            end_date=datetime(2024, 1, 3, 23, 59, 59)
        )
        
        # Should have 2 slices (days 2-3)
        assert len(slices) == 2
        
        # Check dates
        assert slices[0][0].day == 2
        assert slices[1][0].day == 3
    
    def test_empty_slices(self):
        """Test handling of empty time slices."""
        # Create data with gaps
        data = pl.DataFrame({
            "source": ["A", "B"],
            "target": ["B", "A"],
            "timestamp": [
                "2024-01-01 10:00:00",
                "2024-01-03 10:00:00"  # Skip day 2
            ]
        })
        
        slices = create_temporal_slices(
            data,
            slice_interval="daily",
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 1, 3, 23, 59, 59)
        )
        
        # Should have 3 slices
        assert len(slices) == 3
        
        # Day 2 should be empty
        assert slices[1][1].numberOfEdges() == 0
        assert slices[1][1].numberOfNodes() == 0
    
    def test_graph_kwargs_passthrough(self):
        """Test that graph construction kwargs are passed through."""
        data = self.create_sample_data()
        
        slices = create_temporal_slices(
            data,
            slice_interval="daily",
            directed=True,  # Pass graph construction parameter
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 1, 1, 23, 59, 59)
        )
        
        # Graph should be directed
        assert slices[0][1].isDirected() == True


class TestAlignNodeIdsAcrossSlices:
    """Test node ID alignment across temporal slices."""
    
    def test_basic_alignment(self):
        """Test basic node ID alignment."""
        # Create sample temporal graphs with different mappers
        mapper1 = IDMapper()
        mapper1.add_mapping("A", 0)
        mapper1.add_mapping("B", 1)
        id_a1 = mapper1.get_internal("A")
        id_b1 = mapper1.get_internal("B")
        
        graph1 = nk.Graph(2)
        graph1.addEdge(id_a1, id_b1)
        
        mapper2 = IDMapper()
        mapper2.add_mapping("B", 0)  # B gets different internal ID
        mapper2.add_mapping("C", 1)
        id_b2 = mapper2.get_internal("B")
        id_c2 = mapper2.get_internal("C")
        
        graph2 = nk.Graph(2)
        graph2.addEdge(id_b2, id_c2)
        
        temporal_graphs = [
            (datetime(2024, 1, 1), graph1, mapper1),
            (datetime(2024, 1, 2), graph2, mapper2)
        ]
        
        aligned_graphs, global_mapper = align_node_ids_across_slices(temporal_graphs)
        
        # Should have 2 aligned graphs
        assert len(aligned_graphs) == 2
        
        # Global mapper should have all 3 nodes
        assert global_mapper.size() == 3
        # Check that all nodes are in global mapper
        assert global_mapper.has_original("A")
        assert global_mapper.has_original("B")
        assert global_mapper.has_original("C")
        
        # Node B should have same internal ID across graphs
        b_id = global_mapper.get_internal("B")
        
        # Check that graphs have correct structure
        date1, graph1_aligned = aligned_graphs[0]
        date2, graph2_aligned = aligned_graphs[1]
        
        assert date1 == datetime(2024, 1, 1)
        assert date2 == datetime(2024, 1, 2)
        
        # Both graphs should have same number of nodes (global size)
        assert graph1_aligned.numberOfNodes() == 3
        assert graph2_aligned.numberOfNodes() == 3
    
    def test_empty_temporal_graphs(self):
        """Test alignment with empty input."""
        aligned_graphs, global_mapper = align_node_ids_across_slices([])
        
        assert len(aligned_graphs) == 0
        assert global_mapper.size() == 0
    
    def test_weighted_graph_alignment(self):
        """Test alignment preserves edge weights."""
        mapper1 = IDMapper()
        mapper1.add_mapping("A", 0)
        mapper1.add_mapping("B", 1)
        id_a = mapper1.get_internal("A")
        id_b = mapper1.get_internal("B")
        
        graph1 = nk.Graph(2, weighted=True)
        graph1.addEdge(id_a, id_b, 2.5)
        
        temporal_graphs = [
            (datetime(2024, 1, 1), graph1, mapper1)
        ]
        
        aligned_graphs, global_mapper = align_node_ids_across_slices(temporal_graphs)
        
        date, aligned_graph = aligned_graphs[0]
        
        # Should preserve weights
        assert aligned_graph.isWeighted() == True
        
        # Find edge and check weight
        new_a = global_mapper.get_internal("A")
        new_b = global_mapper.get_internal("B")
        assert aligned_graph.weight(new_a, new_b) == 2.5


class TestInputValidation:
    """Test input validation for temporal slicing."""
    
    def test_invalid_slice_interval(self):
        """Test validation of slice_interval parameter."""
        with pytest.raises(ConfigurationError, match="slice_interval must be one of"):
            _validate_temporal_inputs("invalid", None, False, None, None)
    
    def test_invalid_rolling_window(self):
        """Test validation of rolling_window parameter."""
        with pytest.raises(ConfigurationError, match="rolling_window must be positive"):
            _validate_temporal_inputs("daily", 0, False, None, None)
        
        with pytest.raises(ConfigurationError, match="rolling_window must be positive"):
            _validate_temporal_inputs("daily", -1, False, None, None)
    
    def test_cumulative_with_rolling_window(self):
        """Test that cumulative and rolling_window cannot be used together."""
        with pytest.raises(ConfigurationError, match="Cannot use both cumulative=True and rolling_window"):
            _validate_temporal_inputs("daily", 3, True, None, None)
    
    def test_invalid_date_range(self):
        """Test validation of date range."""
        start = datetime(2024, 1, 2)
        end = datetime(2024, 1, 1)  # End before start
        
        with pytest.raises(ValidationError, match="start_date must be before end_date"):
            _validate_temporal_inputs("daily", None, False, start, end)
    
    def test_missing_timestamp_column(self):
        """Test handling of missing timestamp column."""
        data = pl.DataFrame({
            "source": ["A"],
            "target": ["B"]
            # No timestamp column
        })
        
        with pytest.raises(ValidationError, match="Timestamp column 'timestamp' not found"):
            _load_and_prepare_data(data, "timestamp")
    
    def test_invalid_timestamp_format(self):
        """Test handling of invalid timestamp format."""
        data = pl.DataFrame({
            "source": ["A"],
            "target": ["B"],
            "timestamp": ["invalid_date"]
        })
        
        with pytest.raises(DataFormatError, match="Failed to parse timestamp column"):
            _load_and_prepare_data(data, "timestamp")
    
    def test_null_timestamps(self):
        """Test handling of null timestamps."""
        data = pl.DataFrame({
            "source": ["A", "B"],
            "target": ["B", "C"],
            "timestamp": ["2024-01-01 10:00:00", None]
        })
        
        with warnings.catch_warnings(record=True) as w:
            result = _load_and_prepare_data(data, "timestamp")
            
            # Should warn about null timestamps
            assert len(w) == 1
            assert "null timestamps" in str(w[0].message)
        
        # Should have only 1 row after filtering
        assert len(result) == 1
    
    def test_no_valid_data_after_filtering(self):
        """Test handling when no data remains after date filtering."""
        data = pl.DataFrame({
            "source": ["A"],
            "target": ["B"],
            "timestamp": ["2024-01-01 10:00:00"]
        })
        
        # Parse timestamp first like the actual function does
        data = data.with_columns(
            pl.col("timestamp").str.to_datetime().alias("timestamp")
        )
        
        with pytest.raises(ValidationError, match="No data in specified date range"):
            _filter_by_date_range(
                data, 
                "timestamp", 
                datetime(2024, 2, 1),  # Start after data
                datetime(2024, 2, 28)
            )


class TestHelperFunctions:
    """Test individual helper functions."""
    
    def test_generate_date_slices_daily(self):
        """Test daily date slice generation."""
        start = datetime(2024, 1, 1)
        end = datetime(2024, 1, 3)
        
        slices = _generate_date_slices(start, end, "daily")
        
        assert len(slices) == 3
        assert slices[0] == datetime(2024, 1, 1, 23, 59, 59, 999999)
        assert slices[1] == datetime(2024, 1, 2, 23, 59, 59, 999999)
        assert slices[2] == datetime(2024, 1, 3, 23, 59, 59, 999999)
    
    def test_generate_date_slices_weekly(self):
        """Test weekly date slice generation."""
        # Start on a Wednesday (2024-01-03)
        start = datetime(2024, 1, 3)
        end = datetime(2024, 1, 21)
        
        slices = _generate_date_slices(start, end, "weekly")
        
        # Should start from first Monday on or after start date
        # First slice should end on Sunday (2024-01-07)
        assert len(slices) == 3
        assert slices[0].weekday() == 6  # Sunday
    
    def test_generate_date_slices_monthly(self):
        """Test monthly date slice generation."""
        start = datetime(2024, 1, 15)
        end = datetime(2024, 3, 10)
        
        slices = _generate_date_slices(start, end, "monthly")
        
        assert len(slices) == 3
        # Should end on last day of each month
        assert slices[0].day == 31  # Jan 31
        assert slices[1].day == 29  # Feb 29 (leap year)
        assert slices[2].day == 31  # Mar 31
    
    def test_generate_date_slices_yearly(self):
        """Test yearly date slice generation."""
        start = datetime(2023, 6, 15)
        end = datetime(2025, 3, 10)
        
        slices = _generate_date_slices(start, end, "yearly")
        
        assert len(slices) == 3
        # Should end on Dec 31 of each year
        assert slices[0] == datetime(2023, 12, 31, 23, 59, 59, 999999)
        assert slices[1] == datetime(2024, 12, 31, 23, 59, 59, 999999)
        assert slices[2] == datetime(2025, 12, 31, 23, 59, 59, 999999)
    
    def test_get_slice_start(self):
        """Test slice start calculation."""
        # Test daily
        end = datetime(2024, 1, 15, 23, 59, 59, 999999)
        start = _get_slice_start(end, "daily")
        assert start == datetime(2024, 1, 15, 0, 0, 0, 0)
        
        # Test weekly (end is Sunday)
        end = datetime(2024, 1, 14, 23, 59, 59, 999999)  # Sunday
        start = _get_slice_start(end, "weekly")
        assert start == datetime(2024, 1, 8, 0, 0, 0, 0)  # Monday
        
        # Test monthly
        end = datetime(2024, 1, 31, 23, 59, 59, 999999)
        start = _get_slice_start(end, "monthly")
        assert start == datetime(2024, 1, 1, 0, 0, 0, 0)
        
        # Test yearly
        end = datetime(2024, 12, 31, 23, 59, 59, 999999)
        start = _get_slice_start(end, "yearly")
        assert start == datetime(2024, 1, 1, 0, 0, 0, 0)
    
    def test_get_window_start(self):
        """Test rolling window start calculation."""
        end = datetime(2024, 1, 10)
        
        # Test daily rolling window
        start = _get_window_start(end, "daily", 3)
        assert start == datetime(2024, 1, 8)  # 3 days back
        
        # Test weekly rolling window (2-week window ending Jan 10)
        start = _get_window_start(end, "weekly", 2)
        # Should go back 1 week (2-1) from Jan 10, but start from beginning of that week
        # Jan 10 is Wednesday, 1 week back is Jan 3 (also Wed), start of that week is Monday Dec 31
        assert start == datetime(2024, 1, 1, 0, 0)  # Monday of that week
        
        # Test monthly rolling window
        start = _get_window_start(end, "monthly", 3)
        assert start == datetime(2023, 11, 1)  # 2 months back
        
        # Test yearly rolling window
        start = _get_window_start(end, "yearly", 2)
        assert start == datetime(2023, 1, 1)  # 1 year back


class TestCreateTemporalSlicesInputValidation:
    """Test input validation for create_temporal_slices."""
    
    def test_nonexistent_csv_file(self):
        """Test handling of nonexistent CSV file."""
        with pytest.raises(DataFormatError, match="Failed to read CSV file"):
            create_temporal_slices("nonexistent_file.csv")
    
    def test_all_null_timestamps_after_parsing(self):
        """Test when all timestamps are null after parsing."""
        data = pl.DataFrame({
            "source": ["A"],
            "target": ["B"],
            "timestamp": [None]
        })
        
        with pytest.raises(ValidationError, match="No valid data after timestamp parsing"):
            create_temporal_slices(data)


class TestCreateStandardSlices:
    """Test standard slice creation."""
    
    def test_standard_slices_with_mock_build_graph(self):
        """Test standard slice creation with mocked graph building."""
        data = pl.DataFrame({
            "source": ["A", "B"],
            "target": ["B", "C"],
            "timestamp": [
                datetime(2024, 1, 1, 10, 0),
                datetime(2024, 1, 2, 10, 0)
            ]
        })
        
        date_slices = [
            datetime(2024, 1, 1, 23, 59, 59, 999999),
            datetime(2024, 1, 2, 23, 59, 59, 999999)
        ]
        
        with patch('src.timeseries.slicing.build_graph_from_edgelist') as mock_build:
            mock_graph = MagicMock(spec=nk.Graph)
            mock_mapper = MagicMock(spec=IDMapper)
            mock_build.return_value = (mock_graph, mock_mapper)
            
            result = _create_standard_slices(data, date_slices, "timestamp", "daily")
            
            # Should have 2 slices
            assert len(result) == 2
            
            # Should have called build_graph twice
            assert mock_build.call_count == 2


class TestCreateCumulativeSlices:
    """Test cumulative slice creation."""
    
    def test_cumulative_slices_with_mock_build_graph(self):
        """Test cumulative slice creation with mocked graph building."""
        data = pl.DataFrame({
            "source": ["A", "B", "C"],
            "target": ["B", "C", "D"],
            "timestamp": [
                datetime(2024, 1, 1, 10, 0),
                datetime(2024, 1, 2, 10, 0),
                datetime(2024, 1, 3, 10, 0)
            ]
        })
        
        date_slices = [
            datetime(2024, 1, 1, 23, 59, 59, 999999),
            datetime(2024, 1, 2, 23, 59, 59, 999999),
            datetime(2024, 1, 3, 23, 59, 59, 999999)
        ]
        
        with patch('src.timeseries.slicing.build_graph_from_edgelist') as mock_build:
            mock_graph = MagicMock(spec=nk.Graph)
            mock_mapper = MagicMock(spec=IDMapper)
            mock_build.return_value = (mock_graph, mock_mapper)
            
            result = _create_cumulative_slices(data, date_slices, "timestamp")
            
            # Should have 3 slices
            assert len(result) == 3
            
            # Should have called build_graph three times
            assert mock_build.call_count == 3
            
            # Check that cumulative data was passed (1st call gets 1 edge, 2nd gets 2, 3rd gets 3)
            calls = mock_build.call_args_list
            assert len(calls[0][0][0]) == 1  # First slice: 1 edge
            assert len(calls[1][0][0]) == 2  # Second slice: 2 edges (cumulative)
            assert len(calls[2][0][0]) == 3  # Third slice: 3 edges (cumulative)


class TestDateRangeHandling:
    """Test various date range scenarios."""
    
    def test_automatic_date_range_detection(self):
        """Test that date range is automatically detected from data."""
        data = pl.DataFrame({
            "source": ["A", "B"],
            "target": ["B", "C"],
            "timestamp": [
                "2024-01-05 10:00:00",
                "2024-01-10 10:00:00"
            ]
        })
        
        slices = create_temporal_slices(data, slice_interval="daily")
        
        # Should include all days from 5th to 10th
        assert len(slices) == 6
        assert slices[0][0].day == 5
        assert slices[-1][0].day == 10
    
    def test_partial_date_range_specification(self):
        """Test specifying only start_date or only end_date."""
        data = pl.DataFrame({
            "source": ["A", "B", "C"],
            "target": ["B", "C", "D"],
            "timestamp": [
                "2024-01-01 10:00:00",
                "2024-01-05 10:00:00",
                "2024-01-10 10:00:00"
            ]
        })
        
        # Only specify start_date
        slices = create_temporal_slices(
            data, 
            slice_interval="daily",
            start_date=datetime(2024, 1, 3)
        )
        
        # Should start from Jan 3rd and go to last data point (Jan 10th)
        assert slices[0][0].day == 3
        assert slices[-1][0].day == 10
        
        # Only specify end_date
        slices = create_temporal_slices(
            data,
            slice_interval="daily", 
            end_date=datetime(2024, 1, 7)
        )
        
        # Should start from first data point (Jan 1st) and end at Jan 6th (data only goes to Jan 10 but we cap at Jan 7)
        assert slices[0][0].day == 1
        assert slices[-1][0].day == 6  # Last slice with data in range