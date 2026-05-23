"""
Temporal slicing module for time-series network analysis.

This module provides functionality for creating time-sliced networks from edge lists
with timestamps, supporting various temporal granularities, rolling windows, and 
cumulative graph construction for temporal analysis workflows.
"""

from typing import Union, List, Tuple, Optional, Dict, Any
from datetime import datetime, timedelta
import warnings

import polars as pl
import networkit as nk

from guidedLP.network.construction import build_graph_from_edgelist
from guidedLP.common.id_mapper import IDMapper
from guidedLP.common.exceptions import (
    ValidationError,
    DataFormatError,
    ConfigurationError
)
from guidedLP.common.logging_config import get_logger, LoggingTimer

logger = get_logger(__name__)


def create_temporal_slices(
    edgelist: Union[str, pl.DataFrame],
    timestamp_col: str = "timestamp",
    slice_interval: str = "daily",
    rolling_window: Optional[int] = None,
    cumulative: bool = False,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    **graph_kwargs
) -> List[Tuple[datetime, nk.Graph, IDMapper]]:
    """
    Create time-sliced networks from edge list with timestamps.
    
    This function takes an edge list with timestamp information and creates
    a series of temporal network slices based on the specified time intervals.
    Supports various aggregation methods including rolling windows and cumulative
    edge accumulation.
    
    Parameters
    ----------
    edgelist : Union[str, pl.DataFrame]
        Path to CSV file or Polars DataFrame containing edge data with timestamps
    timestamp_col : str, default "timestamp"
        Name of the column containing timestamp information
    slice_interval : str, default "daily"
        Temporal granularity for slicing:
        - "daily": One graph per day
        - "weekly": One graph per week (Monday-Sunday)
        - "monthly": One graph per month
        - "yearly": One graph per year
    rolling_window : Optional[int], default None
        Number of intervals to include in rolling window.
        If specified, each slice includes data from current + (window-1) previous intervals.
        Example: rolling_window=7 with slice_interval="daily" creates 7-day rolling windows.
    cumulative : bool, default False
        If True, each slice includes all edges from start_date up to current slice date.
        Cannot be used with rolling_window.
    start_date : Optional[datetime], default None
        Start date for analysis. If None, uses earliest timestamp in data.
    end_date : Optional[datetime], default None
        End date for analysis. If None, uses latest timestamp in data.
    **graph_kwargs
        Additional arguments passed to build_graph_from_edgelist()
        
    Returns
    -------
    List[Tuple[datetime, nk.Graph, IDMapper]]
        List of tuples containing (slice_date, graph, id_mapper) for each time slice.
        slice_date represents the end date of the temporal slice.
        
    Raises
    ------
    ValidationError
        If input parameters are invalid or data format is incorrect
    ConfigurationError
        If incompatible options are specified (e.g., cumulative + rolling_window)
    DataFormatError
        If timestamp column cannot be parsed or data is malformed
        
    Examples
    --------
    >>> # Create daily slices from edge list
    >>> slices = create_temporal_slices("edges.csv", slice_interval="daily")
    >>> print(f"Created {len(slices)} daily network slices")
    
    >>> # Create weekly rolling window slices
    >>> slices = create_temporal_slices(
    ...     edgelist_df, 
    ...     slice_interval="weekly",
    ...     rolling_window=4  # 4-week rolling windows
    ... )
    
    >>> # Create cumulative monthly slices
    >>> slices = create_temporal_slices(
    ...     "temporal_edges.csv",
    ...     slice_interval="monthly", 
    ...     cumulative=True,
    ...     start_date=datetime(2024, 1, 1)
    ... )
    
    >>> # Access slice data
    >>> for date, graph, mapper in slices[:3]:
    ...     print(f"Date: {date}, Nodes: {graph.numberOfNodes()}, Edges: {graph.numberOfEdges()}")
    
    Notes
    -----
    Temporal Logic:
    - **Non-cumulative, no rolling**: Each slice contains only edges from that time period
    - **Cumulative**: Each slice contains all edges from start_date up to slice_date
    - **Rolling window**: Each slice contains edges from (slice_date - window) to slice_date
    
    Date Handling:
    - Daily: Each slice represents one calendar day (00:00:00 to 23:59:59)
    - Weekly: Monday (00:00:00) to Sunday (23:59:59)
    - Monthly: First day of month (00:00:00) to last day (23:59:59)
    - Yearly: January 1st (00:00:00) to December 31st (23:59:59)
    
    Empty Slices:
    - If a time slice contains no edges, an empty graph is created with a warning
    - ID mappers maintain consistency across all slices for node tracking
    """
    
    logger.info(f"Creating temporal slices with interval={slice_interval}, "
               f"rolling_window={rolling_window}, cumulative={cumulative}")
    
    # Validate inputs
    _validate_temporal_inputs(slice_interval, rolling_window, cumulative, start_date, end_date)
    
    with LoggingTimer("Creating temporal slices"):
        
        # Load and prepare data
        df = _load_and_prepare_data(edgelist, timestamp_col)
        
        # Filter by date range if specified
        if start_date or end_date:
            df = _filter_by_date_range(df, timestamp_col, start_date, end_date)
        
        # Determine actual date range from data
        data_start = df[timestamp_col].min()
        data_end = df[timestamp_col].max()
        
        if start_date is None:
            start_date = data_start
        if end_date is None:
            end_date = data_end
        
        logger.info(f"Date range: {start_date} to {end_date}")
        
        # Generate date slices
        date_slices = _generate_date_slices(start_date, end_date, slice_interval)
        
        logger.info(f"Generated {len(date_slices)} date slices")
        
        # Create graphs for each slice
        if cumulative:
            temporal_graphs = _create_cumulative_slices(df, date_slices, timestamp_col, **graph_kwargs)
        elif rolling_window:
            temporal_graphs = _create_rolling_window_slices(
                df, date_slices, timestamp_col, slice_interval, rolling_window, **graph_kwargs
            )
        else:
            temporal_graphs = _create_standard_slices(df, date_slices, timestamp_col, slice_interval, **graph_kwargs)
        
        logger.info(f"Created {len(temporal_graphs)} temporal graphs")
        
        return temporal_graphs


def align_node_ids_across_slices(
    temporal_graphs: List[Tuple[datetime, nk.Graph, IDMapper]]
) -> Tuple[List[Tuple[datetime, nk.Graph]], IDMapper]:
    """
    Create consistent node ID mapping across all time slices.
    
    This function ensures that the same node has the same internal ID across
    all temporal slices, which is critical for tracking nodes over time and
    performing temporal analysis.
    
    Parameters
    ----------
    temporal_graphs : List[Tuple[datetime, nk.Graph, IDMapper]]
        List of temporal graphs with individual ID mappers
        
    Returns
    -------
    Tuple[List[Tuple[datetime, nk.Graph]], IDMapper]
        - List of (date, graph) tuples with aligned node IDs
        - Single global IDMapper for all slices
        
    Examples
    --------
    >>> # Align node IDs across temporal slices
    >>> aligned_graphs, global_mapper = align_node_ids_across_slices(temporal_graphs)
    >>> 
    >>> # Now same nodes have consistent IDs across time
    >>> for date, graph in aligned_graphs:
    ...     original_id = "user123"
    ...     internal_id = global_mapper.get_internal(original_id)
    ...     # internal_id is same across all time slices if node exists
    """
    
    logger.info(f"Aligning node IDs across {len(temporal_graphs)} temporal slices")
    
    if not temporal_graphs:
        return [], IDMapper()
    
    with LoggingTimer("Aligning node IDs across slices"):
        
        # Collect all unique node IDs across all slices
        all_original_ids = set()
        for date, graph, mapper in temporal_graphs:
            all_original_ids.update(mapper.original_to_internal.keys())
        
        # Create global mapper with all nodes
        global_mapper = IDMapper()
        for i, original_id in enumerate(sorted(all_original_ids)):  # Sort for reproducibility
            global_mapper.add_mapping(original_id, i)
        
        logger.info(f"Global mapper created with {len(all_original_ids)} unique nodes")
        
        # Reconstruct graphs with aligned IDs
        aligned_graphs = []
        for date, old_graph, old_mapper in temporal_graphs:
            
            # Create new graph with same properties
            new_graph = nk.Graph(
                n=global_mapper.size(),
                weighted=old_graph.isWeighted(),
                directed=old_graph.isDirected()
            )
            
            # Map edges to new ID space
            for u, v in old_graph.iterEdges():
                # Get original IDs
                orig_u = old_mapper.get_original(u)
                orig_v = old_mapper.get_original(v)
                
                # Get new internal IDs
                new_u = global_mapper.get_internal(orig_u)
                new_v = global_mapper.get_internal(orig_v)
                
                # Add edge with weight if applicable
                if old_graph.isWeighted():
                    weight = old_graph.weight(u, v)
                    new_graph.addEdge(new_u, new_v, weight)
                else:
                    new_graph.addEdge(new_u, new_v)
            
            aligned_graphs.append((date, new_graph))
        
        logger.info(f"Aligned {len(aligned_graphs)} graphs with consistent node IDs")
        
        return aligned_graphs, global_mapper


def _validate_temporal_inputs(
    slice_interval: str,
    rolling_window: Optional[int],
    cumulative: bool,
    start_date: Optional[datetime],
    end_date: Optional[datetime]
) -> None:
    """Validate temporal slicing parameters."""
    
    valid_intervals = ["daily", "weekly", "monthly", "yearly"]
    if slice_interval not in valid_intervals:
        raise ConfigurationError(
            f"slice_interval must be one of {valid_intervals}, got '{slice_interval}'"
        )
    
    if rolling_window is not None and rolling_window <= 0:
        raise ConfigurationError(f"rolling_window must be positive, got {rolling_window}")
    
    if cumulative and rolling_window is not None:
        raise ConfigurationError("Cannot use both cumulative=True and rolling_window together")
    
    if start_date and end_date and start_date >= end_date:
        raise ValidationError("start_date must be before end_date")


def _load_and_prepare_data(edgelist: Union[str, pl.DataFrame], timestamp_col: str) -> pl.DataFrame:
    """Load edge list data and prepare timestamp column."""
    
    # Load data
    if isinstance(edgelist, str):
        try:
            df = pl.read_csv(edgelist)
        except Exception as e:
            raise DataFormatError(f"Failed to read CSV file '{edgelist}': {e}")
    else:
        df = edgelist.clone()
    
    # Validate timestamp column exists
    if timestamp_col not in df.columns:
        raise ValidationError(f"Timestamp column '{timestamp_col}' not found in data")
    
    # Parse timestamp column to datetime
    try:
        df = df.with_columns(
            pl.col(timestamp_col).str.to_datetime().alias(timestamp_col)
        )
    except Exception:
        # Try parsing as different formats
        try:
            df = df.with_columns(
                pl.col(timestamp_col).cast(pl.Datetime).alias(timestamp_col)
            )
        except Exception as e:
            raise DataFormatError(f"Failed to parse timestamp column '{timestamp_col}': {e}")
    
    # Check for null timestamps
    null_count = df[timestamp_col].null_count()
    if null_count > 0:
        warnings.warn(f"Found {null_count} null timestamps, these rows will be excluded")
        df = df.filter(pl.col(timestamp_col).is_not_null())
    
    if len(df) == 0:
        raise ValidationError("No valid data after timestamp parsing")
    
    logger.debug(f"Loaded {len(df)} edges with timestamps from {df[timestamp_col].min()} to {df[timestamp_col].max()}")
    
    return df


def _filter_by_date_range(
    df: pl.DataFrame, 
    timestamp_col: str, 
    start_date: Optional[datetime], 
    end_date: Optional[datetime]
) -> pl.DataFrame:
    """Filter DataFrame by date range."""
    
    if start_date:
        df = df.filter(pl.col(timestamp_col) >= pl.lit(start_date))
    
    if end_date:
        df = df.filter(pl.col(timestamp_col) <= pl.lit(end_date))
    
    if len(df) == 0:
        raise ValidationError("No data in specified date range")
    
    return df


def _generate_date_slices(start_date: datetime, end_date: datetime, slice_interval: str) -> List[datetime]:
    """Generate list of slice dates based on interval."""
    
    slices = []
    current_date = start_date
    
    if slice_interval == "daily":
        while current_date <= end_date:
            slices.append(current_date.replace(hour=23, minute=59, second=59, microsecond=999999))
            current_date += timedelta(days=1)
    
    elif slice_interval == "weekly":
        # Start from Monday of the week containing start_date
        days_since_monday = current_date.weekday()
        monday_of_week = current_date - timedelta(days=days_since_monday)
        current_date = monday_of_week
        
        while current_date <= end_date:
            # End of week (Sunday)
            week_end = current_date + timedelta(days=6)
            week_end = week_end.replace(hour=23, minute=59, second=59, microsecond=999999)
            if week_end.date() <= end_date.date():  # Only include if week end is within range
                slices.append(week_end)
            current_date += timedelta(days=7)
    
    elif slice_interval == "monthly":
        current_date = current_date.replace(day=1)  # Start from first of month
        
        while current_date <= end_date:
            # Last day of month
            if current_date.month == 12:
                next_month = current_date.replace(year=current_date.year + 1, month=1)
            else:
                next_month = current_date.replace(month=current_date.month + 1)
            
            month_end = (next_month - timedelta(days=1)).replace(hour=23, minute=59, second=59, microsecond=999999)
            slices.append(month_end)
            current_date = next_month
    
    elif slice_interval == "yearly":
        current_date = current_date.replace(month=1, day=1)  # Start from Jan 1
        
        while current_date <= end_date:
            year_end = current_date.replace(month=12, day=31, hour=23, minute=59, second=59, microsecond=999999)
            slices.append(year_end)
            current_date = current_date.replace(year=current_date.year + 1)
    
    return slices


def _create_standard_slices(
    df: pl.DataFrame, 
    date_slices: List[datetime], 
    timestamp_col: str,
    slice_interval: str,
    **graph_kwargs
) -> List[Tuple[datetime, nk.Graph, IDMapper]]:
    """Create standard (non-cumulative, non-rolling) temporal slices."""
    
    temporal_graphs = []
    
    for i, slice_end in enumerate(date_slices):
        # Determine slice start
        slice_start = _get_slice_start(slice_end, slice_interval)
        
        # Filter edges for this slice
        slice_edges = df.filter(
            (pl.col(timestamp_col) >= pl.lit(slice_start)) & 
            (pl.col(timestamp_col) <= pl.lit(slice_end))
        )
        
        # Create graph for this slice
        if len(slice_edges) == 0:
            logger.warning(f"No edges found for slice ending {slice_end}")
            # Create empty graph
            graph = nk.Graph()
            mapper = IDMapper()
        else:
            graph, mapper = build_graph_from_edgelist(slice_edges, **graph_kwargs)
        
        temporal_graphs.append((slice_end, graph, mapper))
        
        logger.debug(f"Slice {slice_end}: {len(slice_edges)} edges -> {graph.numberOfNodes()} nodes, {graph.numberOfEdges()} edges")
    
    return temporal_graphs


def _create_cumulative_slices(
    df: pl.DataFrame,
    date_slices: List[datetime],
    timestamp_col: str,
    **graph_kwargs
) -> List[Tuple[datetime, nk.Graph, IDMapper]]:
    """Create cumulative temporal slices."""
    
    temporal_graphs = []
    
    for slice_end in date_slices:
        # Include all edges up to this date
        cumulative_edges = df.filter(pl.col(timestamp_col) <= pl.lit(slice_end))
        
        # Create graph for cumulative edges
        if len(cumulative_edges) == 0:
            logger.warning(f"No edges found for cumulative slice ending {slice_end}")
            graph = nk.Graph()
            mapper = IDMapper()
        else:
            graph, mapper = build_graph_from_edgelist(cumulative_edges, **graph_kwargs)
        
        temporal_graphs.append((slice_end, graph, mapper))
        
        logger.debug(f"Cumulative slice {slice_end}: {len(cumulative_edges)} edges -> {graph.numberOfNodes()} nodes, {graph.numberOfEdges()} edges")
    
    return temporal_graphs


def _create_rolling_window_slices(
    df: pl.DataFrame,
    date_slices: List[datetime],
    timestamp_col: str,
    slice_interval: str,
    rolling_window: int,
    **graph_kwargs
) -> List[Tuple[datetime, nk.Graph, IDMapper]]:
    """Create rolling window temporal slices."""
    
    temporal_graphs = []
    
    for slice_end in date_slices:
        # Calculate window start
        window_start = _get_window_start(slice_end, slice_interval, rolling_window)
        
        # Filter edges for this window
        window_edges = df.filter(
            (pl.col(timestamp_col) >= pl.lit(window_start)) & 
            (pl.col(timestamp_col) <= pl.lit(slice_end))
        )
        
        # Create graph for this window
        if len(window_edges) == 0:
            logger.warning(f"No edges found for rolling window ending {slice_end}")
            graph = nk.Graph()
            mapper = IDMapper()
        else:
            graph, mapper = build_graph_from_edgelist(window_edges, **graph_kwargs)
        
        temporal_graphs.append((slice_end, graph, mapper))
        
        logger.debug(f"Rolling window {slice_end}: {len(window_edges)} edges -> {graph.numberOfNodes()} nodes, {graph.numberOfEdges()} edges")
    
    return temporal_graphs


def _get_slice_start(slice_end: datetime, slice_interval: str) -> datetime:
    """Get start date for a slice given its end date and interval."""
    
    if slice_interval == "daily":
        return slice_end.replace(hour=0, minute=0, second=0, microsecond=0)
    
    elif slice_interval == "weekly":
        # slice_end is Sunday, go back to Monday of this week
        days_since_monday = 6  # Sunday is 6 days after Monday
        monday = slice_end - timedelta(days=days_since_monday)
        return monday.replace(hour=0, minute=0, second=0, microsecond=0)
    
    elif slice_interval == "monthly":
        return slice_end.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    
    elif slice_interval == "yearly":
        return slice_end.replace(month=1, day=1, hour=0, minute=0, second=0, microsecond=0)


def _get_window_start(slice_end: datetime, slice_interval: str, rolling_window: int) -> datetime:
    """Get start date for rolling window."""
    
    if slice_interval == "daily":
        # Get start of the interval that is rolling_window periods back
        target_date = slice_end - timedelta(days=rolling_window - 1)
        return target_date.replace(hour=0, minute=0, second=0, microsecond=0)
    
    elif slice_interval == "weekly":
        # Get start of the week that is rolling_window weeks back
        target_date = slice_end - timedelta(weeks=rolling_window - 1)
        days_since_monday = target_date.weekday()
        monday = target_date - timedelta(days=days_since_monday)
        return monday.replace(hour=0, minute=0, second=0, microsecond=0)
    
    elif slice_interval == "monthly":
        # Go back rolling_window months
        year = slice_end.year
        month = slice_end.month - (rolling_window - 1)
        
        while month <= 0:
            month += 12
            year -= 1
        
        return datetime(year, month, 1)
    
    elif slice_interval == "yearly":
        return datetime(slice_end.year - (rolling_window - 1), 1, 1)