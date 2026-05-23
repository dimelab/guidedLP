"""
Tests for input validation functions.

This module provides comprehensive testing for all validation functions
in the validators module, covering both valid inputs (should pass) and
invalid inputs (should raise ValidationError).
"""

import pytest
import polars as pl
from datetime import datetime, timedelta
import warnings
from typing import Dict, Any

from src.common.exceptions import ValidationError
from src.common.validators import (
    validate_edgelist_dataframe,
    validate_timestamps,
    validate_seed_labels,
    validate_metadata_dataframe
)


class TestValidationError:
    """Test the ValidationError exception class."""
    
    def test_basic_error(self):
        """Test basic error creation and message."""
        error = ValidationError("Test error message")
        assert str(error) == "Test error message"
        assert error.field is None
        assert error.details == {}
    
    def test_error_with_field(self):
        """Test error with field specification."""
        error = ValidationError("Column is invalid", field="source")
        assert "Validation error in field 'source': Column is invalid" in str(error)
        assert error.field == "source"
    
    def test_error_with_details(self):
        """Test error with additional details."""
        details = {"count": 5, "max_allowed": 10}
        error = ValidationError("Too many items", details=details)
        assert "Too many items" in str(error)
        assert "count=5" in str(error)
        assert "max_allowed=10" in str(error)
        assert error.details == details
    
    def test_error_with_field_and_details(self):
        """Test error with both field and details."""
        error = ValidationError(
            "Invalid values", 
            field="weight", 
            details={"invalid_count": 3}
        )
        assert "Validation error in field 'weight'" in str(error)
        assert "invalid_count=3" in str(error)


class TestValidateEdgelistDataframe:
    """Test edge list DataFrame validation."""
    
    def test_valid_basic_edgelist(self):
        """Test validation of valid basic edge list."""
        df = pl.DataFrame({
            "source": ["A", "B", "C"],
            "target": ["B", "C", "A"]
        })
        
        # Should not raise any exception
        validate_edgelist_dataframe(df)
    
    def test_valid_edgelist_with_weights(self):
        """Test validation of edge list with weights."""
        df = pl.DataFrame({
            "source": ["A", "B", "C"],
            "target": ["B", "C", "A"],
            "weight": [1.0, 2.5, 0.8]
        })
        
        validate_edgelist_dataframe(df, weight_col="weight")
    
    def test_valid_edgelist_with_timestamps(self):
        """Test validation of edge list with timestamps."""
        df = pl.DataFrame({
            "source": ["A", "B", "C"],
            "target": ["B", "C", "A"],
            "timestamp": ["2023-01-01", "2023-01-02", "2023-01-03"]
        })
        
        validate_edgelist_dataframe(df, timestamp_col="timestamp")
    
    def test_valid_edgelist_custom_columns(self):
        """Test validation with custom column names."""
        df = pl.DataFrame({
            "from_node": ["A", "B", "C"],
            "to_node": ["B", "C", "A"],
            "edge_weight": [1.0, 2.0, 3.0]
        })
        
        validate_edgelist_dataframe(
            df, 
            source_col="from_node", 
            target_col="to_node",
            weight_col="edge_weight"
        )
    
    def test_empty_dataframe(self):
        """Test validation fails for empty DataFrame."""
        df = pl.DataFrame()
        
        with pytest.raises(ValidationError, match="DataFrame is empty"):
            validate_edgelist_dataframe(df)
    
    def test_missing_required_columns(self):
        """Test validation fails for missing required columns."""
        df = pl.DataFrame({
            "source": ["A", "B"],
            # Missing target column
        })
        
        with pytest.raises(ValidationError, match="Missing required columns"):
            validate_edgelist_dataframe(df)
    
    def test_missing_optional_columns(self):
        """Test validation fails when specified optional column is missing."""
        df = pl.DataFrame({
            "source": ["A", "B"],
            "target": ["B", "A"]
        })
        
        with pytest.raises(ValidationError, match="Missing required columns"):
            validate_edgelist_dataframe(df, weight_col="nonexistent_weight")
    
    def test_null_values_in_source(self):
        """Test validation fails for null values in source column."""
        df = pl.DataFrame({
            "source": ["A", None, "C"],
            "target": ["B", "C", "A"]
        })
        
        with pytest.raises(ValidationError, match="Column contains.*null values"):
            validate_edgelist_dataframe(df)
    
    def test_null_values_in_target(self):
        """Test validation fails for null values in target column."""
        df = pl.DataFrame({
            "source": ["A", "B", "C"],
            "target": ["B", None, "A"]
        })
        
        with pytest.raises(ValidationError, match="Column contains.*null values"):
            validate_edgelist_dataframe(df)
    
    def test_unhashable_source_values(self):
        """Test validation fails for unhashable source values."""
        # Lists are not hashable
        df = pl.DataFrame({
            "source": [["A"], ["B"], ["C"]],
            "target": ["B", "C", "A"]
        })
        
        with pytest.raises(ValidationError, match="unhashable values"):
            validate_edgelist_dataframe(df)
    
    def test_invalid_weight_type(self):
        """Test validation fails for non-numeric weight column."""
        df = pl.DataFrame({
            "source": ["A", "B", "C"],
            "target": ["B", "C", "A"],
            "weight": ["high", "low", "medium"]  # String weights
        })
        
        with pytest.raises(ValidationError, match="Weight column must be numeric"):
            validate_edgelist_dataframe(df, weight_col="weight")
    
    def test_negative_weights(self):
        """Test validation fails for negative weights."""
        df = pl.DataFrame({
            "source": ["A", "B", "C"],
            "target": ["B", "C", "A"],
            "weight": [1.0, -2.0, 3.0]  # Negative weight
        })
        
        with pytest.raises(ValidationError, match="negative values"):
            validate_edgelist_dataframe(df, weight_col="weight")
    
    def test_null_weights_warning(self):
        """Test warning for null values in weight column."""
        df = pl.DataFrame({
            "source": ["A", "B", "C"],
            "target": ["B", "C", "A"],
            "weight": [1.0, None, 3.0]
        })
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            validate_edgelist_dataframe(df, weight_col="weight")
            assert len(w) == 1
            assert "null values" in str(w[0].message)
    
    def test_self_loops_allowed(self):
        """Test self-loops are allowed by default."""
        df = pl.DataFrame({
            "source": ["A", "A", "B"],  # A -> A is self-loop
            "target": ["A", "B", "B"]   # B -> B is self-loop
        })
        
        # Should not raise exception (self-loops allowed by default)
        validate_edgelist_dataframe(df)
    
    def test_self_loops_not_allowed(self):
        """Test validation fails for self-loops when not allowed."""
        df = pl.DataFrame({
            "source": ["A", "A", "B"],
            "target": ["A", "B", "B"]  # Two self-loops
        })
        
        with pytest.raises(ValidationError, match="self-loops"):
            validate_edgelist_dataframe(df, allow_self_loops=False)
    
    def test_duplicates_allowed(self):
        """Test duplicate edges are allowed by default."""
        df = pl.DataFrame({
            "source": ["A", "A", "B"],  # A -> B appears twice
            "target": ["B", "B", "C"]
        })
        
        # Should not raise exception (duplicates allowed by default)
        validate_edgelist_dataframe(df)
    
    def test_duplicates_not_allowed(self):
        """Test validation fails for duplicates when not allowed."""
        df = pl.DataFrame({
            "source": ["A", "A", "B"],
            "target": ["B", "B", "C"]  # A -> B appears twice
        })
        
        with pytest.raises(ValidationError, match="duplicate edges"):
            validate_edgelist_dataframe(df, allow_duplicates=False)
    
    def test_high_duplicate_warning(self):
        """Test warning for high proportion of duplicates."""
        # Create mostly duplicate edges
        df = pl.DataFrame({
            "source": ["A"] * 8 + ["B", "C"],  # 8 A->B, 1 B->C, 1 C->A
            "target": ["B"] * 8 + ["C", "A"]
        })
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            validate_edgelist_dataframe(df)
            assert len(w) == 1
            assert "duplicate edges" in str(w[0].message)
    
    def test_mixed_id_types(self):
        """Test validation with mixed ID types (string and numeric)."""
        df = pl.DataFrame({
            "source": ["A", 1, "C"],  # Mixed string and int
            "target": [2, "B", "A"]
        })
        
        # Should pass - mixed hashable types are OK
        validate_edgelist_dataframe(df)


class TestValidateTimestamps:
    """Test timestamp validation."""
    
    def test_valid_datetime_strings(self):
        """Test validation of valid datetime strings."""
        timestamps = pl.Series(["2023-01-01", "2023-01-02", "2023-01-03"])
        
        # Should not raise exception
        validate_timestamps(timestamps)
    
    def test_valid_datetime_objects(self):
        """Test validation of datetime objects."""
        dates = [
            datetime(2023, 1, 1),
            datetime(2023, 1, 2),
            datetime(2023, 1, 3)
        ]
        timestamps = pl.Series(dates)
        
        validate_timestamps(timestamps)
    
    def test_empty_timestamps(self):
        """Test validation fails for empty timestamp series."""
        timestamps = pl.Series([])
        
        with pytest.raises(ValidationError, match="Timestamp series is empty"):
            validate_timestamps(timestamps)
    
    def test_all_null_timestamps_not_allowed(self):
        """Test validation fails when all timestamps are null and not allowed."""
        timestamps = pl.Series([None, None, None])
        
        with pytest.raises(ValidationError, match="All timestamps are null"):
            validate_timestamps(timestamps, allow_nulls=False)
    
    def test_some_null_timestamps_not_allowed(self):
        """Test validation fails for some null timestamps when not allowed."""
        timestamps = pl.Series(["2023-01-01", None, "2023-01-03"])
        
        with pytest.raises(ValidationError, match="null timestamps"):
            validate_timestamps(timestamps, allow_nulls=False)
    
    def test_null_timestamps_allowed(self):
        """Test validation passes for null timestamps when allowed."""
        timestamps = pl.Series(["2023-01-01", None, "2023-01-03"])
        
        # Should not raise exception
        validate_timestamps(timestamps, allow_nulls=True)
    
    def test_invalid_timestamp_format(self):
        """Test validation fails for invalid timestamp format."""
        timestamps = pl.Series(["2023-01-01", "invalid_date", "2023-01-03"])
        
        with pytest.raises(ValidationError, match="Failed to parse timestamps"):
            validate_timestamps(timestamps)
    
    def test_min_date_validation(self):
        """Test validation with minimum date constraint."""
        timestamps = pl.Series(["2020-01-01", "2023-01-02", "2023-01-03"])
        min_date = datetime(2022, 1, 1)
        
        with pytest.raises(ValidationError, match="before minimum allowed date"):
            validate_timestamps(timestamps, min_date=min_date)
    
    def test_max_date_validation(self):
        """Test validation with maximum date constraint."""
        timestamps = pl.Series(["2023-01-01", "2023-01-02", "2025-01-03"])
        max_date = datetime(2024, 1, 1)
        
        with pytest.raises(ValidationError, match="after maximum allowed date"):
            validate_timestamps(timestamps, max_date=max_date)
    
    def test_identical_timestamps_warning(self):
        """Test warning for identical timestamps."""
        timestamps = pl.Series(["2023-01-01", "2023-01-01", "2023-01-01"])
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            validate_timestamps(timestamps)
            assert len(w) == 1
            assert "identical" in str(w[0].message)
    
    def test_large_date_range_warning(self):
        """Test warning for very large date range."""
        timestamps = pl.Series(["1970-01-01", "2040-01-01"])  # 70 year range
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            validate_timestamps(timestamps)
            assert len(w) == 1
            assert "very large" in str(w[0].message)
    
    def test_future_dates_warning(self):
        """Test warning for dates far in the future."""
        future_date = datetime.now() + timedelta(days=400)  # > 1 year from now
        timestamps = pl.Series(["2023-01-01", future_date.isoformat()])
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            validate_timestamps(timestamps)
            assert len(w) == 1
            assert "far in the future" in str(w[0].message)


class TestValidateSeedLabels:
    """Test seed labels validation."""
    
    def test_valid_seed_labels(self):
        """Test validation of valid seed labels."""
        seed_labels = {
            "user1": "left",
            "user2": "right", 
            "user3": "left",
            "user4": "right"
        }
        labels = ["left", "right"]
        
        # Should not raise exception
        validate_seed_labels(seed_labels, labels)
    
    def test_empty_seed_labels(self):
        """Test validation fails for empty seed labels."""
        seed_labels = {}
        labels = ["left", "right"]
        
        with pytest.raises(ValidationError, match="Seed labels dictionary is empty"):
            validate_seed_labels(seed_labels, labels)
    
    def test_empty_labels_list(self):
        """Test validation fails for empty labels list."""
        seed_labels = {"user1": "left"}
        labels = []
        
        with pytest.raises(ValidationError, match="Labels list is empty"):
            validate_seed_labels(seed_labels, labels)
    
    def test_invalid_labels(self):
        """Test validation fails for invalid labels in seeds."""
        seed_labels = {
            "user1": "left",
            "user2": "invalid_label",  # Not in labels list
            "user3": "right"
        }
        labels = ["left", "right"]
        
        with pytest.raises(ValidationError, match="Invalid labels found"):
            validate_seed_labels(seed_labels, labels)
    
    def test_unhashable_node_ids(self):
        """Test validation fails for unhashable node IDs."""
        seed_labels = {
            ["user1"]: "left",  # List is not hashable
            "user2": "right"
        }
        labels = ["left", "right"]
        
        with pytest.raises(ValidationError, match="must be hashable"):
            validate_seed_labels(seed_labels, labels)
    
    def test_insufficient_seeds_per_label(self):
        """Test validation fails when not enough seeds per label."""
        seed_labels = {
            "user1": "left",
            "user2": "left",  # 2 left seeds
            "user3": "right"  # 1 right seed
        }
        labels = ["left", "right"]
        
        with pytest.raises(ValidationError, match="Insufficient seeds"):
            validate_seed_labels(seed_labels, labels, min_seeds_per_label=2)
    
    def test_missing_labels(self):
        """Test validation fails for labels with no seeds."""
        seed_labels = {
            "user1": "left",
            "user2": "left"
            # No "right" seeds
        }
        labels = ["left", "right"]
        
        with pytest.raises(ValidationError, match="No seeds found for labels"):
            validate_seed_labels(seed_labels, labels)
    
    def test_high_imbalance_warning(self):
        """Test warning for highly imbalanced seed labels."""
        seed_labels = {}
        
        # 20 left seeds, 1 right seed (20:1 ratio)
        for i in range(20):
            seed_labels[f"left_user_{i}"] = "left"
        seed_labels["right_user"] = "right"
        
        labels = ["left", "right"]
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            validate_seed_labels(seed_labels, labels)
            assert len(w) == 1
            assert "imbalanced" in str(w[0].message)
    
    def test_moderate_imbalance_warning(self):
        """Test warning for moderate imbalance."""
        seed_labels = {
            "user1": "left", "user2": "left", "user3": "left",
            "user4": "left", "user5": "left",  # 5 left seeds
            "user6": "right"  # 1 right seed (5:1 ratio)
        }
        labels = ["left", "right"]
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            validate_seed_labels(seed_labels, labels)
            assert len(w) == 1
            assert "Moderately imbalanced" in str(w[0].message)
    
    def test_few_seeds_warning(self):
        """Test warning for very few total seeds."""
        seed_labels = {
            "user1": "left",
            "user2": "right"  # Only 2 seeds for 2 labels
        }
        labels = ["left", "right"]
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            validate_seed_labels(seed_labels, labels)
            assert len(w) == 1
            assert "Very few seed nodes" in str(w[0].message)
    
    def test_single_seed_labels_warning(self):
        """Test warning for labels with only one seed."""
        seed_labels = {
            "user1": "left",  # Only 1 left seed
            "user2": "right", "user3": "right"  # 2 right seeds
        }
        labels = ["left", "right"]
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            validate_seed_labels(seed_labels, labels)
            assert len(w) == 1
            assert "only one seed node" in str(w[0].message)
    
    def test_disable_balance_check(self):
        """Test disabling balance checking."""
        seed_labels = {}
        
        # Highly imbalanced
        for i in range(20):
            seed_labels[f"left_user_{i}"] = "left"
        seed_labels["right_user"] = "right"
        
        labels = ["left", "right"]
        
        # Should not produce imbalance warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            validate_seed_labels(seed_labels, labels, check_balance=False)
            # May still have other warnings, but not imbalance warning
            imbalance_warnings = [warning for warning in w 
                                if "imbalanced" in str(warning.message)]
            assert len(imbalance_warnings) == 0
    
    def test_multiple_labels(self):
        """Test validation with multiple labels."""
        seed_labels = {
            "user1": "politics", "user2": "politics",
            "user3": "sports", "user4": "sports", 
            "user5": "tech", "user6": "tech"
        }
        labels = ["politics", "sports", "tech"]
        
        # Should not raise exception
        validate_seed_labels(seed_labels, labels)
    
    def test_numeric_and_string_labels(self):
        """Test validation with mixed label types."""
        seed_labels = {
            "user1": "category_a",
            "user2": "category_b",
            "user3": "category_a"
        }
        labels = ["category_a", "category_b"]
        
        validate_seed_labels(seed_labels, labels)


class TestValidateMetadataDataframe:
    """Test metadata DataFrame validation."""
    
    def test_valid_metadata(self):
        """Test validation of valid metadata DataFrame."""
        df = pl.DataFrame({
            "node_id": ["A", "B", "C"],
            "category": ["type1", "type2", "type1"],
            "score": [0.8, 0.6, 0.9]
        })
        
        # Should not raise exception
        validate_metadata_dataframe(df)
    
    def test_empty_metadata(self):
        """Test validation fails for empty metadata."""
        df = pl.DataFrame()
        
        with pytest.raises(ValidationError, match="Metadata DataFrame is empty"):
            validate_metadata_dataframe(df)
    
    def test_missing_id_column(self):
        """Test validation fails for missing ID column."""
        df = pl.DataFrame({
            "category": ["type1", "type2"],
            "score": [0.8, 0.6]
            # Missing node_id column
        })
        
        with pytest.raises(ValidationError, match="ID column.*not found"):
            validate_metadata_dataframe(df)
    
    def test_custom_id_column(self):
        """Test validation with custom ID column name."""
        df = pl.DataFrame({
            "user_id": ["A", "B", "C"],
            "category": ["type1", "type2", "type1"]
        })
        
        # Should not raise exception
        validate_metadata_dataframe(df, id_col="user_id")
    
    def test_null_ids_not_allowed(self):
        """Test validation fails for null IDs when not allowed."""
        df = pl.DataFrame({
            "node_id": ["A", None, "C"],
            "category": ["type1", "type2", "type1"]
        })
        
        with pytest.raises(ValidationError, match="null values"):
            validate_metadata_dataframe(df, allow_missing_ids=False)
    
    def test_null_ids_allowed(self):
        """Test validation passes for null IDs when allowed."""
        df = pl.DataFrame({
            "node_id": ["A", None, "C"],
            "category": ["type1", "type2", "type1"]
        })
        
        # Should not raise exception
        validate_metadata_dataframe(df, allow_missing_ids=True)
    
    def test_unhashable_ids(self):
        """Test validation fails for unhashable ID values."""
        df = pl.DataFrame({
            "node_id": [["A"], ["B"], ["C"]],  # Lists are not hashable
            "category": ["type1", "type2", "type1"]
        })
        
        with pytest.raises(ValidationError, match="unhashable values"):
            validate_metadata_dataframe(df)
    
    def test_duplicate_ids_warning(self):
        """Test warning for duplicate IDs."""
        df = pl.DataFrame({
            "node_id": ["A", "B", "A"],  # Duplicate A
            "category": ["type1", "type2", "type3"]
        })
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            validate_metadata_dataframe(df)
            assert len(w) == 1
            assert "duplicate IDs" in str(w[0].message)
    
    def test_missing_required_columns(self):
        """Test validation fails for missing required columns."""
        df = pl.DataFrame({
            "node_id": ["A", "B", "C"],
            "category": ["type1", "type2", "type1"]
            # Missing "score" column
        })
        
        with pytest.raises(ValidationError, match="Missing required metadata columns"):
            validate_metadata_dataframe(df, required_cols=["category", "score"])
    
    def test_null_values_in_required_columns_warning(self):
        """Test warning for null values in required columns."""
        df = pl.DataFrame({
            "node_id": ["A", "B", "C"],
            "category": ["type1", None, "type1"],  # Null in required column
            "score": [0.8, 0.6, 0.9]
        })
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            validate_metadata_dataframe(df, required_cols=["category"])
            assert len(w) == 1
            assert "null values" in str(w[0].message)
    
    def test_valid_required_columns(self):
        """Test validation passes with valid required columns."""
        df = pl.DataFrame({
            "node_id": ["A", "B", "C"],
            "category": ["type1", "type2", "type1"],
            "score": [0.8, 0.6, 0.9],
            "optional": [None, "value", None]  # Optional column with nulls
        })
        
        # Should not raise exception
        validate_metadata_dataframe(df, required_cols=["category", "score"])


class TestIntegrationScenarios:
    """Test realistic integration scenarios with multiple validators."""
    
    def test_complete_network_data_validation(self):
        """Test validation of complete network dataset."""
        # Valid edge list
        edges = pl.DataFrame({
            "source": ["A", "B", "C", "A"],
            "target": ["B", "C", "A", "C"],
            "weight": [1.0, 2.0, 1.5, 0.8],
            "timestamp": ["2023-01-01", "2023-01-02", "2023-01-03", "2023-01-04"]
        })
        
        # Valid metadata
        metadata = pl.DataFrame({
            "node_id": ["A", "B", "C"],
            "type": ["user", "user", "business"],
            "active": [True, True, False]
        })
        
        # Valid seed labels
        seed_labels = {"A": "positive", "B": "negative"}
        labels = ["positive", "negative"]
        
        # All should pass
        validate_edgelist_dataframe(edges, weight_col="weight", timestamp_col="timestamp")
        validate_metadata_dataframe(metadata, required_cols=["type"])
        validate_seed_labels(seed_labels, labels)
    
    def test_invalid_network_data_chain(self):
        """Test how validation errors propagate in realistic scenario."""
        # Invalid edge list (null source)
        edges = pl.DataFrame({
            "source": ["A", None, "C"],
            "target": ["B", "C", "A"],
            "weight": [1.0, -2.0, 1.5]  # Also negative weight
        })
        
        # First validation should fail on null source
        with pytest.raises(ValidationError, match="null values"):
            validate_edgelist_dataframe(edges, weight_col="weight")
    
    def test_warning_accumulation(self):
        """Test multiple warnings from different validators."""
        # Edge list with many duplicates
        edges = pl.DataFrame({
            "source": ["A"] * 8 + ["B", "C"],
            "target": ["B"] * 8 + ["C", "A"],
            "timestamp": ["2023-01-01"] * 10  # All identical timestamps
        })
        
        # Metadata with duplicate IDs
        metadata = pl.DataFrame({
            "node_id": ["A", "B", "A"],
            "category": ["type1", "type2", "type3"]
        })
        
        # Imbalanced seeds
        seed_labels = {}
        for i in range(10):
            seed_labels[f"user_{i}"] = "majority"
        seed_labels["minority_user"] = "minority"
        labels = ["majority", "minority"]
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            
            validate_edgelist_dataframe(edges, timestamp_col="timestamp")
            validate_metadata_dataframe(metadata)
            validate_seed_labels(seed_labels, labels)
            
            # Should have multiple warnings
            assert len(w) >= 3  # At least one from each validator


if __name__ == "__main__":
    # Run tests if script is executed directly
    pytest.main([__file__])