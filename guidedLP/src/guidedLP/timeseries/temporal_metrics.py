"""
Temporal metrics extraction and analysis for time-series networks.

This module provides functionality for calculating network metrics across
temporal slices and performing statistical analysis of metric evolution
over time, including trend analysis and temporal aggregation.
"""

from typing import List, Tuple, Dict, Any, Optional, Union
from datetime import datetime
import warnings

import polars as pl
import networkit as nk
import numpy as np
from scipy import stats

from guidedLP.network.analysis import extract_centrality, AVAILABLE_METRICS
from guidedLP.common.id_mapper import IDMapper
from guidedLP.common.exceptions import (
    ValidationError,
    ConfigurationError,
    ComputationError
)
from guidedLP.common.logging_config import get_logger, LoggingTimer

logger = get_logger(__name__)


def extract_temporal_metrics(
    temporal_graphs: List[Tuple[datetime, nk.Graph, IDMapper]],
    metrics: List[str] = ["degree", "betweenness"],
    n_jobs: int = -1
) -> pl.DataFrame:
    """
    Calculate network metrics for all nodes across all time slices.
    
    This function computes centrality metrics for each temporal slice and
    combines them into a single DataFrame for temporal analysis. The resulting
    data structure enables tracking how node importance evolves over time.
    
    Parameters
    ----------
    temporal_graphs : List[Tuple[datetime, nk.Graph, IDMapper]]
        List of temporal network slices from create_temporal_slices()
        Each tuple contains (slice_date, graph, id_mapper)
    metrics : List[str], default ["degree", "betweenness"]
        List of centrality metrics to calculate. Available options:
        - "degree": Node degree (number of connections)
        - "betweenness": Fraction of shortest paths passing through node
        - "closeness": Inverse of average distance to all other nodes
        - "eigenvector": Importance based on importance of neighbors
        - "pagerank": Google's PageRank algorithm 
        - "katz": Katz centrality
    n_jobs : int, default -1
        Number of parallel jobs for computation
        
    Returns
    -------
    pl.DataFrame
        DataFrame with temporal metrics containing columns:
        - "node_id": Original node identifiers
        - "date": Date of the time slice
        - "{metric}": One column per requested metric (e.g., "degree", "betweenness")
        Missing values are filled with 0.0 for nodes not present in a slice
        
    Raises
    ------
    ValidationError
        If temporal_graphs is empty or contains invalid data
    ConfigurationError
        If invalid metric names are specified
        
    Examples
    --------
    >>> # Extract degree and betweenness across temporal slices
    >>> metrics_df = extract_temporal_metrics(
    ...     temporal_graphs,
    ...     metrics=["degree", "betweenness"]
    ... )
    >>> print(metrics_df)
    ┌─────────┬────────────┬────────┬─────────────┐
    │ node_id ┆ date       ┆ degree ┆ betweenness │
    │ ---     ┆ ---        ┆ ---    ┆ ---         │
    │ str     ┆ datetime   ┆ f64    ┆ f64         │
    ╞═════════╪════════════╪════════╪═════════════╡
    │ A       ┆ 2024-01-01 ┆ 2.0    ┆ 0.5         │
    │ B       ┆ 2024-01-01 ┆ 1.0    ┆ 0.0         │
    
    >>> # Track centrality evolution over time
    >>> for node in metrics_df["node_id"].unique():
    ...     node_data = metrics_df.filter(pl.col("node_id") == node)
    ...     print(f"Node {node}: degree trend = {node_data['degree'].to_list()}")
    
    Notes
    -----
    Temporal Alignment:
    - All nodes that appear in any time slice are included in every slice
    - Missing values (nodes not present in a slice) are filled with 0.0
    - This ensures consistent temporal tracking of all nodes
    
    Performance:
    - Metrics are calculated in parallel across time slices when n_jobs > 1
    - Memory usage scales with: num_slices × num_unique_nodes × num_metrics
    """
    
    logger.info(f"Extracting temporal metrics from {len(temporal_graphs)} time slices, "
               f"metrics={metrics}")
    
    # Validate inputs
    _validate_temporal_metrics_inputs(temporal_graphs, metrics)
    
    with LoggingTimer("Extracting temporal metrics"):
        
        # Extract metrics for each time slice
        slice_metrics = []
        for i, (slice_date, graph, mapper) in enumerate(temporal_graphs):
            logger.debug(f"Processing slice {i+1}/{len(temporal_graphs)}: {slice_date}")
            
            if graph.numberOfNodes() == 0:
                logger.warning(f"Empty graph for slice {slice_date}, creating empty metrics")
                # Create empty DataFrame with correct schema
                empty_metrics = pl.DataFrame({
                    "node_id": pl.Series([], dtype=pl.Utf8),
                    **{metric: pl.Series([], dtype=pl.Float64) for metric in metrics}
                })
            else:
                # Extract centrality metrics for this slice
                centrality_df = extract_centrality(
                    graph, mapper, metrics=metrics, normalized=True, n_jobs=n_jobs
                )
                
                # Rename columns to remove "_centrality" suffix for cleaner output
                rename_map = {}
                for metric in metrics:
                    old_col = f"{metric}_centrality"
                    if old_col in centrality_df.columns:
                        rename_map[old_col] = metric
                
                empty_metrics = centrality_df.rename(rename_map)
            
            # Add date column
            empty_metrics = empty_metrics.with_columns(
                pl.lit(slice_date).alias("date")
            )
            
            slice_metrics.append(empty_metrics)
        
        if not slice_metrics:
            logger.warning("No metric data extracted, returning empty DataFrame")
            return pl.DataFrame()
        
        # Combine all slices
        logger.debug("Combining metrics from all time slices")
        combined_metrics = pl.concat(slice_metrics, how="vertical")
        
        # Get all unique nodes across all time slices
        all_nodes = combined_metrics["node_id"].unique().sort()
        all_dates = combined_metrics["date"].unique().sort()
        
        logger.info(f"Found {len(all_nodes)} unique nodes across {len(all_dates)} time slices")
        
        # Create complete temporal grid (all nodes × all dates)
        complete_grid = _create_temporal_grid(all_nodes, all_dates, metrics)
        
        # Join with actual data to fill in metrics
        result = complete_grid.join(
            combined_metrics,
            on=["node_id", "date"],
            how="left"
        )
        
        # Fill missing values with 0.0 for all metric columns
        fill_expressions = []
        for metric in metrics:
            # Use suffix "_right" if column was duplicated during join
            metric_col = f"{metric}_right" if f"{metric}_right" in result.columns else metric
            fill_expressions.append(
                pl.col(metric_col).fill_null(0.0).alias(metric)
            )
        
        # Select final columns and fill nulls
        final_columns = ["node_id", "date"] + fill_expressions
        result = result.select(final_columns)
        
        # Sort by date, then node_id for consistent output
        result = result.sort(["date", "node_id"])
        
        logger.info(f"Extracted temporal metrics: {len(result)} rows, "
                   f"{len(metrics)} metrics, {len(all_dates)} time slices")
        
        return result


def calculate_temporal_statistics(
    temporal_metrics: pl.DataFrame,
    statistics: List[str] = ["mean", "std", "trend"]
) -> pl.DataFrame:
    """
    Aggregate statistics across time slices for temporal analysis.
    
    This function computes summary statistics for each node's metric values
    across time, enabling identification of stable vs. volatile nodes,
    trending patterns, and overall temporal behavior.
    
    Parameters
    ----------
    temporal_metrics : pl.DataFrame
        Output from extract_temporal_metrics() containing:
        - "node_id": Node identifiers
        - "date": Time slice dates
        - {metric columns}: Metric values over time
    statistics : List[str], default ["mean", "std", "trend"]
        Which statistics to calculate:
        - "mean": Average value across time
        - "std": Standard deviation (absolute volatility)
        - "trend": Linear trend slope (direction of change)
        - "volatility": Coefficient of variation (relative volatility)
        - "growth": Percentage change from first to last value
        - "min": Minimum value across time
        - "max": Maximum value across time
        - "range": Difference between max and min
        
    Returns
    -------
    pl.DataFrame
        DataFrame with per-node temporal statistics containing:
        - "node_id": Original node identifiers
        - "{metric}_{statistic}": One column per metric-statistic combination
        
    Raises
    ------
    ValidationError
        If temporal_metrics DataFrame is invalid or empty
    ConfigurationError
        If invalid statistic names are specified
        
    Examples
    --------
    >>> # Calculate temporal statistics
    >>> stats_df = calculate_temporal_statistics(
    ...     temporal_metrics,
    ...     statistics=["mean", "std", "trend"]
    ... )
    >>> print(stats_df)
    ┌─────────┬──────────────┬─────────────┬───────────────┐
    │ node_id ┆ degree_mean  ┆ degree_std  ┆ degree_trend  │
    │ ---     ┆ ---          ┆ ---         ┆ ---           │
    │ str     ┆ f64          ┆ f64         ┆ f64           │
    ╞═════════╪══════════════╪═════════════╪═══════════════╡
    │ A       ┆ 2.5          ┆ 0.5         ┆ 0.1           │
    │ B       ┆ 1.8          ┆ 1.2         ┆ -0.05         │
    
    >>> # Identify nodes with strongest growth trends
    >>> trending_nodes = stats_df.filter(
    ...     pl.col("degree_trend") > 0.1
    ... ).sort("degree_trend", descending=True)
    
    >>> # Find most volatile nodes
    >>> volatile_nodes = stats_df.filter(
    ...     pl.col("degree_volatility") > 0.5
    ... ).sort("degree_volatility", descending=True)
    
    Notes
    -----
    Statistical Interpretations:
    - **Mean**: Average importance/centrality over time
    - **Std**: Absolute volatility in metric values
    - **Trend**: Positive = increasing, negative = decreasing importance
    - **Volatility**: Coefficient of variation (std/mean) for relative volatility
    - **Growth**: (last_value - first_value) / first_value × 100
    
    Missing Data Handling:
    - Nodes with all zero values have volatility = 0 (no variation)
    - Trend calculation requires at least 2 time points
    - Growth calculation handles zero initial values gracefully
    """
    
    logger.info(f"Calculating temporal statistics: {statistics}")
    
    # Validate inputs
    _validate_statistics_inputs(temporal_metrics, statistics)
    
    with LoggingTimer("Calculating temporal statistics"):
        
        # Get metric columns (exclude node_id and date)
        exclude_cols = {"node_id", "date"}
        metric_columns = [col for col in temporal_metrics.columns if col not in exclude_cols]
        
        if not metric_columns:
            raise ValidationError("No metric columns found in temporal_metrics DataFrame")
        
        logger.debug(f"Processing {len(metric_columns)} metrics: {metric_columns}")
        
        # Calculate statistics for each metric
        stat_expressions = []
        
        for metric in metric_columns:
            for stat in statistics:
                stat_expr = _create_statistic_expression(metric, stat)
                if stat_expr is not None:
                    stat_expressions.append(stat_expr)
        
        if not stat_expressions:
            raise ConfigurationError("No valid statistics could be calculated")
        
        # Group by node and calculate statistics
        result = temporal_metrics.group_by("node_id").agg(stat_expressions)
        
        # Sort by node_id for consistent output
        result = result.sort("node_id")
        
        logger.info(f"Calculated temporal statistics for {len(result)} nodes, "
                   f"{len(stat_expressions)} statistics")
        
        return result


def _validate_temporal_metrics_inputs(
    temporal_graphs: List[Tuple[datetime, nk.Graph, IDMapper]],
    metrics: List[str]
) -> None:
    """Validate inputs for extract_temporal_metrics."""
    
    if not temporal_graphs:
        raise ValidationError("temporal_graphs cannot be empty")
    
    if not metrics:
        raise ValidationError("metrics list cannot be empty")
    
    # Check for valid metric names
    invalid_metrics = [m for m in metrics if m not in AVAILABLE_METRICS]
    if invalid_metrics:
        raise ConfigurationError(
            f"Invalid metrics: {invalid_metrics}. "
            f"Available metrics: {AVAILABLE_METRICS}"
        )
    
    # Basic validation of temporal_graphs structure
    for i, item in enumerate(temporal_graphs):
        if not isinstance(item, tuple) or len(item) != 3:
            raise ValidationError(
                f"temporal_graphs[{i}] must be tuple of (datetime, Graph, IDMapper)"
            )
        
        date, graph, mapper = item
        if not isinstance(date, datetime):
            raise ValidationError(f"temporal_graphs[{i}][0] must be datetime")
        if not isinstance(graph, nk.Graph):
            raise ValidationError(f"temporal_graphs[{i}][1] must be NetworkIt Graph")
        if not isinstance(mapper, IDMapper):
            raise ValidationError(f"temporal_graphs[{i}][2] must be IDMapper")


def _validate_statistics_inputs(
    temporal_metrics: pl.DataFrame,
    statistics: List[str]
) -> None:
    """Validate inputs for calculate_temporal_statistics."""
    
    if temporal_metrics.is_empty():
        raise ValidationError("temporal_metrics DataFrame cannot be empty")
    
    if not statistics:
        raise ValidationError("statistics list cannot be empty")
    
    # Check required columns
    required_cols = {"node_id", "date"}
    missing_cols = required_cols - set(temporal_metrics.columns)
    if missing_cols:
        raise ValidationError(f"Missing required columns: {missing_cols}")
    
    # Check for valid statistic names
    valid_statistics = {
        "mean", "std", "trend", "volatility", "growth", 
        "min", "max", "range"
    }
    invalid_stats = [s for s in statistics if s not in valid_statistics]
    if invalid_stats:
        raise ConfigurationError(
            f"Invalid statistics: {invalid_stats}. "
            f"Available statistics: {sorted(valid_statistics)}"
        )


def _create_temporal_grid(
    all_nodes: pl.Series,
    all_dates: pl.Series,
    metrics: List[str]
) -> pl.DataFrame:
    """Create complete temporal grid (all nodes × all dates)."""
    
    # Create Cartesian product of nodes and dates
    nodes_df = pl.DataFrame({"node_id": all_nodes})
    dates_df = pl.DataFrame({"date": all_dates})
    
    # Cross join to get all combinations
    grid = nodes_df.join(dates_df, how="cross")
    
    # Add placeholder columns for metrics (will be filled with actual values)
    for metric in metrics:
        grid = grid.with_columns(pl.lit(None, dtype=pl.Float64).alias(metric))
    
    return grid


def _create_statistic_expression(metric: str, statistic: str) -> Optional[pl.Expr]:
    """Create Polars expression for calculating a specific statistic."""
    
    metric_col = pl.col(metric)
    
    if statistic == "mean":
        return metric_col.mean().alias(f"{metric}_mean")
    
    elif statistic == "std":
        return metric_col.std().fill_null(0.0).alias(f"{metric}_std")
    
    elif statistic == "min":
        return metric_col.min().alias(f"{metric}_min")
    
    elif statistic == "max":
        return metric_col.max().alias(f"{metric}_max")
    
    elif statistic == "range":
        return (metric_col.max() - metric_col.min()).alias(f"{metric}_range")
    
    elif statistic == "volatility":
        # Coefficient of variation: std / mean
        # Handle division by zero by returning 0
        mean_expr = metric_col.mean()
        std_expr = metric_col.std()
        return pl.when(mean_expr != 0.0).then(std_expr / mean_expr).otherwise(0.0).alias(f"{metric}_volatility")
    
    elif statistic == "growth":
        # Percentage change from first to last value
        first_val = metric_col.first()
        last_val = metric_col.last()
        return pl.when(first_val != 0.0).then(
            ((last_val - first_val) / first_val * 100.0)
        ).otherwise(
            pl.when(last_val != 0.0).then(float('inf')).otherwise(0.0)
        ).alias(f"{metric}_growth")
    
    elif statistic == "trend":
        # Linear trend slope using least squares regression
        # This requires more complex calculation that we'll handle separately
        return _create_trend_expression(metric)
    
    else:
        logger.warning(f"Unknown statistic: {statistic}")
        return None


def _create_trend_expression(metric: str) -> pl.Expr:
    """Create expression for linear trend calculation."""
    
    # For trend calculation, we need to use a custom function
    # that calculates linear regression slope
    return pl.col(metric).map_elements(
        lambda series: _calculate_trend_slope(series),
        return_dtype=pl.Float64
    ).alias(f"{metric}_trend")


def _calculate_trend_slope(values: pl.Series) -> float:
    """Calculate linear trend slope for a series of values."""
    
    try:
        # Convert to numpy for calculations
        y = values.to_numpy()
        
        # Filter out NaN values
        valid_mask = ~np.isnan(y)
        if valid_mask.sum() < 2:
            return 0.0  # Need at least 2 points for trend
        
        y_valid = y[valid_mask]
        x_valid = np.arange(len(y_valid))
        
        # Calculate linear regression slope
        if len(x_valid) < 2:
            return 0.0
        
        slope, _, _, _, _ = stats.linregress(x_valid, y_valid)
        return float(slope) if not np.isnan(slope) else 0.0
        
    except Exception as e:
        logger.warning(f"Failed to calculate trend slope: {e}")
        return 0.0