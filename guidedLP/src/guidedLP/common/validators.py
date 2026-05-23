"""
Input validation utilities for the Guided Label Propagation library.

This module provides validation functions for various input data formats used
throughout the GLP library, ensuring data quality and consistency before
processing begins.
"""

from typing import Dict, List, Any, Union, Optional, Set
import polars as pl
from datetime import datetime
import warnings

from .exceptions import ValidationError


def validate_edgelist_dataframe(
    df: pl.DataFrame,
    source_col: str = "source",
    target_col: str = "target",
    weight_col: Optional[str] = None,
    timestamp_col: Optional[str] = None,
    allow_self_loops: bool = True,
    allow_duplicates: bool = True
) -> None:
    """
    Validate an edge list DataFrame for network construction.
    
    Checks that the DataFrame has required columns, proper data types,
    no null values in critical columns, and valid data ranges.
    
    Parameters
    ----------
    df : pl.DataFrame
        Edge list DataFrame to validate
    source_col : str, default "source"
        Name of the source node column
    target_col : str, default "target"
        Name of the target node column
    weight_col : str, optional
        Name of the edge weight column (if present)
    timestamp_col : str, optional
        Name of the timestamp column (if present)
    allow_self_loops : bool, default True
        Whether to allow edges from a node to itself
    allow_duplicates : bool, default True
        Whether to allow duplicate edges (same source-target pairs)
        
    Raises
    ------
    ValidationError
        If the DataFrame fails any validation checks
        
    Examples
    --------
    >>> import polars as pl
    >>> df = pl.DataFrame({
    ...     "source": ["A", "B", "C"],
    ...     "target": ["B", "C", "A"],
    ...     "weight": [1.0, 2.0, 1.5]
    ... })
    >>> validate_edgelist_dataframe(df, weight_col="weight")  # Should pass
    
    >>> invalid_df = pl.DataFrame({
    ...     "source": ["A", None, "C"],
    ...     "target": ["B", "C", "A"]
    ... })
    >>> validate_edgelist_dataframe(invalid_df)  # doctest: +SKIP
    ValidationError: Validation error in field 'source': Column contains null values
    
    Notes
    -----
    - Source and target columns must contain hashable, non-null values
    - Weight column (if specified) must be numeric and non-negative
    - Timestamp column (if specified) must be datetime or valid datetime strings
    - This function validates structure but not semantic correctness
    """
    if df.is_empty():
        raise ValidationError("DataFrame is empty", field="dataframe")
    
    # Check required columns exist
    required_cols = [source_col, target_col]
    optional_cols = [col for col in [weight_col, timestamp_col] if col is not None]
    all_required_cols = required_cols + optional_cols
    
    missing_cols = [col for col in all_required_cols if col not in df.columns]
    if missing_cols:
        raise ValidationError(
            f"Missing required columns: {missing_cols}",
            field="columns",
            details={"available_columns": df.columns, "missing": missing_cols}
        )
    
    # Check for null values in required columns
    for col in required_cols:
        null_count = df[col].null_count()
        if null_count > 0:
            raise ValidationError(
                f"Column contains {null_count} null values",
                field=col,
                details={"null_count": null_count, "total_rows": len(df)}
            )
    
    # Validate source and target columns contain hashable types
    for col in [source_col, target_col]:
        # Check if column contains any unhashable types (lists, dicts, etc.)
        try:
            # Try to convert to a set to test hashability
            unique_values = df[col].unique().to_list()
            test_set = set(unique_values)
        except TypeError as e:
            raise ValidationError(
                f"Column contains unhashable values that cannot be used as node IDs",
                field=col,
                details={"error": str(e)}
            )
    
    # Validate weight column if present
    if weight_col is not None:
        weight_series = df[weight_col]
        
        # Check data type is numeric
        if not weight_series.dtype.is_numeric():
            raise ValidationError(
                f"Weight column must be numeric, got {weight_series.dtype}",
                field=weight_col,
                details={"dtype": str(weight_series.dtype)}
            )
        
        # Check for null values in weights (warn but don't fail)
        null_count = weight_series.null_count()
        if null_count > 0:
            warnings.warn(
                f"Weight column contains {null_count} null values. "
                "These will be treated as zero weight or may cause errors in analysis."
            )
        
        # Check for negative weights
        non_null_weights = weight_series.drop_nulls()
        if len(non_null_weights) > 0:
            min_weight = non_null_weights.min()
            if min_weight < 0:
                negative_count = (non_null_weights < 0).sum()
                raise ValidationError(
                    f"Weight column contains {negative_count} negative values. "
                    f"Minimum weight: {min_weight}",
                    field=weight_col,
                    details={"min_weight": min_weight, "negative_count": negative_count}
                )
    
    # Validate timestamp column if present
    if timestamp_col is not None:
        validate_timestamps(df[timestamp_col], column_name=timestamp_col)
    
    # Check for self-loops if not allowed
    if not allow_self_loops:
        self_loop_mask = df[source_col] == df[target_col]
        self_loop_count = self_loop_mask.sum()
        if self_loop_count > 0:
            raise ValidationError(
                f"Found {self_loop_count} self-loops (edges from node to itself)",
                field="edges",
                details={"self_loop_count": self_loop_count, "allow_self_loops": False}
            )
    
    # Check for duplicate edges if not allowed
    if not allow_duplicates:
        edge_cols = [source_col, target_col]
        duplicate_mask = df.select(edge_cols).is_duplicated()
        duplicate_count = duplicate_mask.sum()
        if duplicate_count > 0:
            raise ValidationError(
                f"Found {duplicate_count} duplicate edges (same source-target pairs)",
                field="edges",
                details={"duplicate_count": duplicate_count, "allow_duplicates": False}
            )
    
    # Additional checks for data quality
    total_rows = len(df)
    unique_edges = df.select([source_col, target_col]).n_unique()
    
    # Warn if many duplicates (might indicate data quality issues)
    if allow_duplicates and unique_edges < total_rows * 0.5:
        duplicate_ratio = 1 - (unique_edges / total_rows)
        warnings.warn(
            f"High proportion of duplicate edges ({duplicate_ratio:.1%}). "
            "This might indicate data quality issues or the need for edge weight aggregation."
        )


def validate_timestamps(
    timestamps: pl.Series,
    column_name: str = "timestamp",
    allow_nulls: bool = True,
    min_date: Optional[datetime] = None,
    max_date: Optional[datetime] = None
) -> None:
    """
    Validate timestamp data for time-series network analysis.
    
    Checks that timestamps are in valid datetime format, within reasonable
    date ranges, and contain sufficient temporal variation for analysis.
    
    Parameters
    ----------
    timestamps : pl.Series
        Series containing timestamp data
    column_name : str, default "timestamp"
        Name of the timestamp column (for error messages)
    allow_nulls : bool, default True
        Whether to allow null/missing timestamps
    min_date : datetime, optional
        Minimum allowed date (inclusive)
    max_date : datetime, optional
        Maximum allowed date (inclusive)
        
    Raises
    ------
    ValidationError
        If timestamps fail validation checks
        
    Examples
    --------
    >>> import polars as pl
    >>> from datetime import datetime
    >>> timestamps = pl.Series(["2023-01-01", "2023-01-02", "2023-01-03"])
    >>> validate_timestamps(timestamps)  # Should pass
    
    >>> invalid_timestamps = pl.Series(["2023-01-01", "invalid_date", "2023-01-03"])
    >>> validate_timestamps(invalid_timestamps)  # doctest: +SKIP
    ValidationError: Validation error in field 'timestamp': Failed to parse timestamps
    
    Notes
    -----
    - Accepts datetime objects, ISO format strings, or other parseable formats
    - Validates temporal ordering and range reasonableness
    - Warns about potential issues like single timestamps or outliers
    """
    if timestamps.is_empty():
        raise ValidationError("Timestamp series is empty", field=column_name)
    
    # Check for null values
    null_count = timestamps.null_count()
    if null_count > 0 and not allow_nulls:
        raise ValidationError(
            f"Column contains {null_count} null timestamps",
            field=column_name,
            details={"null_count": null_count, "allow_nulls": False}
        )
    
    # Try to parse timestamps if they're not already datetime type
    if not timestamps.dtype.is_temporal():
        try:
            # Attempt to parse as datetime
            parsed_timestamps = timestamps.str.to_datetime()
        except Exception as e:
            # Try alternative parsing methods
            try:
                parsed_timestamps = pl.Series(timestamps.name).cast(pl.Datetime)
            except Exception:
                raise ValidationError(
                    f"Failed to parse timestamps. Expected datetime format or parseable strings",
                    field=column_name,
                    details={"error": str(e), "sample_values": timestamps.head(5).to_list()}
                )
    else:
        parsed_timestamps = timestamps
    
    # Work with non-null timestamps for further validation
    non_null_timestamps = parsed_timestamps.drop_nulls()
    
    if len(non_null_timestamps) == 0:
        if not allow_nulls:
            raise ValidationError("All timestamps are null", field=column_name)
        return  # Nothing more to validate
    
    # Check date range if specified
    if min_date is not None or max_date is not None:
        min_ts = non_null_timestamps.min()
        max_ts = non_null_timestamps.max()
        
        if min_date is not None and min_ts < min_date:
            raise ValidationError(
                f"Timestamps contain dates before minimum allowed date {min_date}",
                field=column_name,
                details={"min_found": min_ts, "min_allowed": min_date}
            )
        
        if max_date is not None and max_ts > max_date:
            raise ValidationError(
                f"Timestamps contain dates after maximum allowed date {max_date}",
                field=column_name,
                details={"max_found": max_ts, "max_allowed": max_date}
            )
    
    # Check for reasonable date range (warn about potential issues)
    min_ts = non_null_timestamps.min()
    max_ts = non_null_timestamps.max()
    
    # Warn if all timestamps are identical
    if min_ts == max_ts:
        warnings.warn(
            f"All timestamps are identical ({min_ts}). "
            "Time-series analysis may not be meaningful."
        )
    
    # Warn if date range is suspiciously large (> 50 years)
    date_range = max_ts - min_ts
    if date_range.total_seconds() > 50 * 365 * 24 * 3600:  # 50 years in seconds
        warnings.warn(
            f"Timestamp range is very large ({date_range}). "
            "Please verify this is correct for your analysis."
        )
    
    # Check for unreasonable future dates (more than 1 year from now)
    now = datetime.now()
    if max_ts > now:
        future_delta = max_ts - now
        if future_delta.total_seconds() > 365 * 24 * 3600:  # 1 year
            warnings.warn(
                f"Timestamps contain dates far in the future (max: {max_ts}). "
                "Please verify this is correct."
            )


def validate_seed_labels(
    seed_labels: Dict[Any, str],
    labels: List[str],
    min_seeds_per_label: int = 1,
    check_balance: bool = True,
    max_imbalance_ratio: float = 10.0
) -> None:
    """
    Validate seed labels for Guided Label Propagation.
    
    Checks that seed labels are properly formatted, contain valid labels,
    have sufficient representation per class, and are reasonably balanced.
    
    Parameters
    ----------
    seed_labels : Dict[Any, str]
        Dictionary mapping node IDs to their labels
    labels : List[str]
        List of all valid labels in the label space
    min_seeds_per_label : int, default 1
        Minimum number of seed nodes required per label
    check_balance : bool, default True
        Whether to check for label balance and warn about imbalances
    max_imbalance_ratio : float, default 10.0
        Maximum ratio between most and least frequent labels before warning
        
    Raises
    ------
    ValidationError
        If seed labels fail validation checks
        
    Examples
    --------
    >>> seed_labels = {"user1": "left", "user2": "right", "user3": "left"}
    >>> labels = ["left", "right"]
    >>> validate_seed_labels(seed_labels, labels)  # Should pass
    
    >>> invalid_seeds = {"user1": "invalid_label", "user2": "right"}
    >>> validate_seed_labels(invalid_seeds, labels)  # doctest: +SKIP
    ValidationError: Validation error in field 'seed_labels': Invalid labels found
    
    Notes
    -----
    - All seed labels must be present in the labels list
    - Recommends balanced representation across labels for better performance
    - Node IDs must be hashable (same as edge list requirements)
    """
    if not seed_labels:
        raise ValidationError("Seed labels dictionary is empty", field="seed_labels")
    
    if not labels:
        raise ValidationError("Labels list is empty", field="labels")
    
    # Convert labels to set for efficient lookup
    valid_labels = set(labels)
    
    # Check that all seed labels are valid
    seed_label_values = list(seed_labels.values())
    invalid_labels = [label for label in seed_label_values if label not in valid_labels]
    
    if invalid_labels:
        unique_invalid = list(set(invalid_labels))
        raise ValidationError(
            f"Invalid labels found: {unique_invalid}",
            field="seed_labels",
            details={
                "invalid_labels": unique_invalid,
                "valid_labels": labels,
                "count_invalid": len(invalid_labels)
            }
        )
    
    # Check that all node IDs are hashable
    try:
        node_ids = list(seed_labels.keys())
        test_set = set(node_ids)
    except TypeError as e:
        raise ValidationError(
            "Seed node IDs must be hashable",
            field="seed_labels",
            details={"error": str(e)}
        )
    
    # Count seeds per label
    label_counts = {}
    for label in labels:
        label_counts[label] = seed_label_values.count(label)
    
    # Check minimum seeds per label
    insufficient_labels = [
        label for label, count in label_counts.items() 
        if count < min_seeds_per_label
    ]
    
    if insufficient_labels:
        raise ValidationError(
            f"Insufficient seeds for labels: {insufficient_labels}",
            field="seed_labels",
            details={
                "insufficient_labels": insufficient_labels,
                "label_counts": label_counts,
                "min_required": min_seeds_per_label
            }
        )
    
    # Check for missing labels (labels with no seeds)
    missing_labels = [label for label, count in label_counts.items() if count == 0]
    if missing_labels:
        raise ValidationError(
            f"No seeds found for labels: {missing_labels}",
            field="seed_labels",
            details={
                "missing_labels": missing_labels,
                "label_counts": label_counts
            }
        )
    
    # Check label balance if requested
    if check_balance and len(labels) > 1:
        counts = [count for count in label_counts.values() if count > 0]
        min_count = min(counts)
        max_count = max(counts)
        
        if min_count > 0:  # Avoid division by zero
            imbalance_ratio = max_count / min_count
            
            if imbalance_ratio > max_imbalance_ratio:
                warnings.warn(
                    f"Highly imbalanced seed labels detected. "
                    f"Ratio of most to least frequent: {imbalance_ratio:.1f}x. "
                    f"Label counts: {label_counts}. "
                    f"Consider using create_balanced_seed_set() for better performance."
                )
            elif imbalance_ratio > 3.0:  # Moderate imbalance
                warnings.warn(
                    f"Moderately imbalanced seed labels. "
                    f"Ratio: {imbalance_ratio:.1f}x. Label counts: {label_counts}"
                )
    
    # Additional quality checks
    total_seeds = len(seed_labels)
    unique_nodes = len(set(seed_labels.keys()))
    
    # Check for duplicate node IDs (shouldn't happen with dict, but just in case)
    if total_seeds != unique_nodes:
        raise ValidationError(
            "Duplicate node IDs found in seed labels",
            field="seed_labels",
            details={"total_seeds": total_seeds, "unique_nodes": unique_nodes}
        )
    
    # Warn if very few seeds
    if total_seeds < len(labels) * 5:
        warnings.warn(
            f"Very few seed nodes ({total_seeds}) for {len(labels)} labels. "
            "Consider adding more seeds for better propagation quality."
        )
    
    # Warn about single-seed labels (high risk)
    single_seed_labels = [label for label, count in label_counts.items() if count == 1]
    if single_seed_labels:
        warnings.warn(
            f"Labels with only one seed node: {single_seed_labels}. "
            "This may lead to unstable propagation results."
        )


def validate_metadata_dataframe(
    df: pl.DataFrame,
    id_col: str = "node_id",
    required_cols: Optional[List[str]] = None,
    allow_missing_ids: bool = False
) -> None:
    """
    Validate metadata DataFrame for joining with network data.
    
    Checks that metadata has proper structure for joining with network
    analysis results, including proper ID column and no critical missing data.
    
    Parameters
    ----------
    df : pl.DataFrame
        Metadata DataFrame to validate
    id_col : str, default "node_id"
        Name of the node ID column that will be used for joining
    required_cols : List[str], optional
        List of columns that must be present and non-null
    allow_missing_ids : bool, default False
        Whether to allow null values in the ID column
        
    Raises
    ------
    ValidationError
        If metadata DataFrame fails validation checks
        
    Examples
    --------
    >>> import polars as pl
    >>> metadata = pl.DataFrame({
    ...     "node_id": ["A", "B", "C"],
    ...     "category": ["type1", "type2", "type1"],
    ...     "score": [0.8, 0.6, 0.9]
    ... })
    >>> validate_metadata_dataframe(metadata, required_cols=["category"])  # Should pass
    
    Notes
    -----
    - ID column must contain hashable, preferably unique values
    - Duplicate IDs are allowed but will generate warnings
    - Metadata will be joined with network results on the ID column
    """
    if df.is_empty():
        raise ValidationError("Metadata DataFrame is empty", field="metadata")
    
    # Check ID column exists
    if id_col not in df.columns:
        raise ValidationError(
            f"ID column '{id_col}' not found",
            field="metadata",
            details={"available_columns": df.columns}
        )
    
    # Check for null values in ID column
    id_null_count = df[id_col].null_count()
    if id_null_count > 0 and not allow_missing_ids:
        raise ValidationError(
            f"ID column contains {id_null_count} null values",
            field=id_col,
            details={"null_count": id_null_count}
        )
    
    # Check ID column contains hashable values
    try:
        unique_ids = df[id_col].drop_nulls().unique().to_list()
        test_set = set(unique_ids)
    except TypeError as e:
        raise ValidationError(
            "ID column contains unhashable values",
            field=id_col,
            details={"error": str(e)}
        )
    
    # Check for duplicate IDs (warn but don't fail)
    total_non_null_ids = df[id_col].drop_nulls().len()
    unique_id_count = len(unique_ids)
    
    if unique_id_count < total_non_null_ids:
        duplicate_count = total_non_null_ids - unique_id_count
        warnings.warn(
            f"Metadata contains {duplicate_count} duplicate IDs. "
            "Only the first occurrence will be used in joins."
        )
    
    # Check required columns if specified
    if required_cols:
        missing_required = [col for col in required_cols if col not in df.columns]
        if missing_required:
            raise ValidationError(
                f"Missing required metadata columns: {missing_required}",
                field="metadata",
                details={"missing_columns": missing_required, "available": df.columns}
            )
        
        # Check for null values in required columns
        for col in required_cols:
            null_count = df[col].null_count()
            if null_count > 0:
                warnings.warn(
                    f"Required column '{col}' contains {null_count} null values"
                )