"""
Tests for the IDMapper class.

This module provides comprehensive testing for the IDMapper class including:
- Basic functionality tests
- Edge cases and error conditions  
- Batch operations
- Serialization/deserialization
- Type checking and validation
"""

import pytest
from typing import Any, List
import uuid

from src.common.id_mapper import IDMapper


class TestIDMapperBasic:
    """Test basic IDMapper functionality."""
    
    def test_empty_mapper(self):
        """Test empty mapper initialization and properties."""
        mapper = IDMapper()
        
        assert mapper.size() == 0
        assert len(mapper) == 0
        assert mapper.is_empty()
        assert str(mapper) == "IDMapper(empty)"
        assert repr(mapper) == "IDMapper(size=0)"
    
    def test_add_single_mapping(self):
        """Test adding a single mapping."""
        mapper = IDMapper()
        mapper.add_mapping("user_123", 0)
        
        assert mapper.size() == 1
        assert not mapper.is_empty()
        assert mapper.get_internal("user_123") == 0
        assert mapper.get_original(0) == "user_123"
        assert mapper.has_original("user_123")
        assert mapper.has_internal(0)
        assert "user_123" in mapper
        assert 0 in mapper
    
    def test_add_multiple_mappings(self):
        """Test adding multiple mappings."""
        mapper = IDMapper()
        
        # Add several mappings
        mappings = [
            ("user_123", 0),
            ("user_456", 1), 
            ("user_789", 2),
            (12345, 3),  # Integer original ID
            (uuid.uuid4(), 4),  # UUID original ID
        ]
        
        for original, internal in mappings:
            mapper.add_mapping(original, internal)
        
        assert mapper.size() == len(mappings)
        
        # Verify all mappings
        for original, internal in mappings:
            assert mapper.get_internal(original) == internal
            assert mapper.get_original(internal) == original
    
    def test_mapping_consistency(self):
        """Test bidirectional mapping consistency."""
        mapper = IDMapper()
        
        test_data = [
            ("user_a", 0),
            ("user_b", 1),
            (42, 2),
            ("user_c", 3),
        ]
        
        # Add all mappings
        for original, internal in test_data:
            mapper.add_mapping(original, internal)
        
        # Verify round-trip consistency
        for original, internal in test_data:
            assert mapper.get_original(mapper.get_internal(original)) == original
            assert mapper.get_internal(mapper.get_original(internal)) == internal


class TestIDMapperErrors:
    """Test error conditions and edge cases."""
    
    def test_get_internal_not_found(self):
        """Test KeyError when getting non-existent original ID."""
        mapper = IDMapper()
        mapper.add_mapping("user_123", 0)
        
        with pytest.raises(KeyError, match="Original ID 'unknown_user' not found"):
            mapper.get_internal("unknown_user")
    
    def test_get_original_not_found(self):
        """Test KeyError when getting non-existent internal ID."""
        mapper = IDMapper()
        mapper.add_mapping("user_123", 0)
        
        with pytest.raises(KeyError, match="Internal ID 99 not found"):
            mapper.get_original(99)
    
    def test_get_original_invalid_type(self):
        """Test TypeError when internal ID is not integer."""
        mapper = IDMapper()
        
        with pytest.raises(TypeError, match="Internal ID must be integer"):
            mapper.get_original("not_an_int")
        
        with pytest.raises(TypeError, match="Internal ID must be integer"):
            mapper.get_original(3.14)
    
    def test_add_mapping_duplicate_original(self):
        """Test ValueError when adding duplicate original ID."""
        mapper = IDMapper()
        mapper.add_mapping("user_123", 0)
        
        with pytest.raises(ValueError, match="Original ID 'user_123' already mapped"):
            mapper.add_mapping("user_123", 1)
    
    def test_add_mapping_duplicate_internal(self):
        """Test ValueError when adding duplicate internal ID."""
        mapper = IDMapper()
        mapper.add_mapping("user_123", 0)
        
        with pytest.raises(ValueError, match="Internal ID 0 already mapped"):
            mapper.add_mapping("user_456", 0)
    
    def test_add_mapping_negative_internal(self):
        """Test ValueError when adding negative internal ID."""
        mapper = IDMapper()
        
        with pytest.raises(ValueError, match="Internal ID must be non-negative"):
            mapper.add_mapping("user_123", -1)
    
    def test_add_mapping_invalid_internal_type(self):
        """Test TypeError when internal ID is not integer."""
        mapper = IDMapper()
        
        with pytest.raises(TypeError, match="Internal ID must be integer"):
            mapper.add_mapping("user_123", "not_int")
        
        with pytest.raises(TypeError, match="Internal ID must be integer"):
            mapper.add_mapping("user_123", 3.14)
    
    def test_add_mapping_unhashable_original(self):
        """Test TypeError when original ID is not hashable."""
        mapper = IDMapper()
        
        # Lists are not hashable
        with pytest.raises(TypeError, match="Original ID must be hashable"):
            mapper.add_mapping(["not", "hashable"], 0)
        
        # Dictionaries are not hashable
        with pytest.raises(TypeError, match="Original ID must be hashable"):
            mapper.add_mapping({"not": "hashable"}, 0)


class TestIDMapperBatchOperations:
    """Test batch operations for efficiency."""
    
    def test_get_internal_batch_success(self):
        """Test successful batch internal ID retrieval."""
        mapper = IDMapper()
        
        # Add test mappings
        test_data = [("user_a", 0), ("user_b", 1), ("user_c", 2)]
        for original, internal in test_data:
            mapper.add_mapping(original, internal)
        
        # Test batch operation
        original_ids = ["user_a", "user_c", "user_b"]
        expected_internal = [0, 2, 1]
        
        result = mapper.get_internal_batch(original_ids)
        assert result == expected_internal
    
    def test_get_original_batch_success(self):
        """Test successful batch original ID retrieval."""
        mapper = IDMapper()
        
        # Add test mappings
        test_data = [("user_a", 0), ("user_b", 1), ("user_c", 2)]
        for original, internal in test_data:
            mapper.add_mapping(original, internal)
        
        # Test batch operation
        internal_ids = [2, 0, 1]
        expected_original = ["user_c", "user_a", "user_b"]
        
        result = mapper.get_original_batch(internal_ids)
        assert result == expected_original
    
    def test_get_internal_batch_empty_list(self):
        """Test batch operation with empty list."""
        mapper = IDMapper()
        mapper.add_mapping("user_123", 0)
        
        result = mapper.get_internal_batch([])
        assert result == []
    
    def test_get_original_batch_empty_list(self):
        """Test batch operation with empty list."""
        mapper = IDMapper()
        mapper.add_mapping("user_123", 0)
        
        result = mapper.get_original_batch([])
        assert result == []
    
    def test_get_internal_batch_not_found(self):
        """Test batch operation with non-existent original ID."""
        mapper = IDMapper()
        mapper.add_mapping("user_123", 0)
        
        with pytest.raises(KeyError, match="Original ID 'unknown' not found"):
            mapper.get_internal_batch(["user_123", "unknown"])
    
    def test_get_original_batch_not_found(self):
        """Test batch operation with non-existent internal ID."""
        mapper = IDMapper()
        mapper.add_mapping("user_123", 0)
        
        with pytest.raises(KeyError, match="Internal ID 99 not found"):
            mapper.get_original_batch([0, 99])
    
    def test_get_internal_batch_invalid_type(self):
        """Test batch operation with invalid input type."""
        mapper = IDMapper()
        
        with pytest.raises(TypeError, match="original_ids must be a list"):
            mapper.get_internal_batch("not_a_list")
    
    def test_get_original_batch_invalid_type(self):
        """Test batch operation with invalid input type."""
        mapper = IDMapper()
        
        with pytest.raises(TypeError, match="internal_ids must be a list"):
            mapper.get_original_batch("not_a_list")
    
    def test_get_original_batch_invalid_element_type(self):
        """Test batch operation with invalid element types."""
        mapper = IDMapper()
        mapper.add_mapping("user_123", 0)
        
        with pytest.raises(TypeError, match="All internal IDs must be integers"):
            mapper.get_original_batch([0, "not_int"])


class TestIDMapperSerialization:
    """Test serialization and deserialization."""
    
    def test_to_dict_empty(self):
        """Test serialization of empty mapper."""
        mapper = IDMapper()
        result = mapper.to_dict()
        
        expected = {
            'original_to_internal': {},
            'internal_to_original': {}
        }
        assert result == expected
    
    def test_to_dict_with_data(self):
        """Test serialization with data."""
        mapper = IDMapper()
        mapper.add_mapping("user_123", 0)
        mapper.add_mapping("user_456", 1)
        mapper.add_mapping(789, 2)
        
        result = mapper.to_dict()
        
        expected = {
            'original_to_internal': {"user_123": 0, "user_456": 1, 789: 2},
            'internal_to_original': {"0": "user_123", "1": "user_456", "2": 789}
        }
        assert result == expected
    
    def test_from_dict_empty(self):
        """Test deserialization of empty mapper."""
        mapping_dict = {
            'original_to_internal': {},
            'internal_to_original': {}
        }
        
        mapper = IDMapper.from_dict(mapping_dict)
        assert mapper.is_empty()
        assert mapper.size() == 0
    
    def test_from_dict_with_data(self):
        """Test deserialization with data."""
        mapping_dict = {
            'original_to_internal': {"user_123": 0, "user_456": 1, 789: 2},
            'internal_to_original': {"0": "user_123", "1": "user_456", "2": 789}
        }
        
        mapper = IDMapper.from_dict(mapping_dict)
        
        assert mapper.size() == 3
        assert mapper.get_internal("user_123") == 0
        assert mapper.get_internal("user_456") == 1
        assert mapper.get_internal(789) == 2
        assert mapper.get_original(0) == "user_123"
        assert mapper.get_original(1) == "user_456"
        assert mapper.get_original(2) == 789
    
    def test_from_dict_missing_keys(self):
        """Test deserialization with missing required keys."""
        # Missing 'internal_to_original'
        with pytest.raises(KeyError, match="Missing required key"):
            IDMapper.from_dict({'original_to_internal': {}})
        
        # Missing 'original_to_internal'
        with pytest.raises(KeyError, match="Missing required key"):
            IDMapper.from_dict({'internal_to_original': {}})
    
    def test_from_dict_invalid_internal_ids(self):
        """Test deserialization with invalid internal ID format."""
        mapping_dict = {
            'original_to_internal': {"user_123": 0},
            'internal_to_original': {"not_int": "user_123"}
        }
        
        with pytest.raises(ValueError, match="Invalid internal ID"):
            IDMapper.from_dict(mapping_dict)
    
    def test_from_dict_inconsistent_sizes(self):
        """Test deserialization with inconsistent mapping sizes."""
        mapping_dict = {
            'original_to_internal': {"user_123": 0, "user_456": 1},
            'internal_to_original': {"0": "user_123"}  # Missing user_456
        }
        
        with pytest.raises(ValueError, match="Inconsistent mapping sizes"):
            IDMapper.from_dict(mapping_dict)
    
    def test_from_dict_inconsistent_mappings(self):
        """Test deserialization with inconsistent bidirectional mappings."""
        # Original points to internal ID that doesn't exist in reverse mapping
        mapping_dict = {
            'original_to_internal': {"user_123": 0},
            'internal_to_original': {"1": "user_123"}  # Wrong internal ID
        }
        
        with pytest.raises(ValueError, match="Inconsistent mapping.*not in reverse mapping"):
            IDMapper.from_dict(mapping_dict)
        
        # Internal points to original that doesn't match
        mapping_dict = {
            'original_to_internal': {"user_123": 0},
            'internal_to_original': {"0": "different_user"}
        }
        
        with pytest.raises(ValueError, match="Inconsistent mapping.*different_user"):
            IDMapper.from_dict(mapping_dict)
    
    def test_round_trip_serialization(self):
        """Test that serialization followed by deserialization preserves data."""
        # Create original mapper
        original_mapper = IDMapper()
        test_data = [
            ("user_a", 0),
            ("user_b", 1),
            (12345, 2),
            ("user_c", 3),
        ]
        
        for original, internal in test_data:
            original_mapper.add_mapping(original, internal)
        
        # Serialize and deserialize
        mapping_dict = original_mapper.to_dict()
        restored_mapper = IDMapper.from_dict(mapping_dict)
        
        # Verify all data is preserved
        assert restored_mapper.size() == original_mapper.size()
        
        for original, internal in test_data:
            assert restored_mapper.get_internal(original) == internal
            assert restored_mapper.get_original(internal) == original


class TestIDMapperUtilities:
    """Test utility methods and special cases."""
    
    def test_clear(self):
        """Test clearing all mappings."""
        mapper = IDMapper()
        mapper.add_mapping("user_123", 0)
        mapper.add_mapping("user_456", 1)
        
        assert mapper.size() == 2
        
        mapper.clear()
        
        assert mapper.size() == 0
        assert mapper.is_empty()
        assert not mapper.has_original("user_123")
        assert not mapper.has_internal(0)
    
    def test_contains_operator(self):
        """Test __contains__ operator (in keyword)."""
        mapper = IDMapper()
        mapper.add_mapping("user_123", 0)
        mapper.add_mapping(456, 1)
        
        # Test original IDs
        assert "user_123" in mapper
        assert 456 in mapper
        assert "unknown" not in mapper
        
        # Test internal IDs
        assert 0 in mapper
        assert 1 in mapper
        assert 99 not in mapper
    
    def test_string_representations(self):
        """Test __str__ and __repr__ methods."""
        # Empty mapper
        mapper = IDMapper()
        assert str(mapper) == "IDMapper(empty)"
        assert repr(mapper) == "IDMapper(size=0)"
        
        # Mapper with few items
        mapper.add_mapping("user_a", 0)
        mapper.add_mapping("user_b", 1)
        
        str_repr = str(mapper)
        assert "user_a" in str_repr
        assert "user_b" in str_repr
        assert repr(mapper) == "IDMapper(size=2)"
        
        # Mapper with many items (should truncate)
        for i in range(2, 10):
            mapper.add_mapping(f"user_{i}", i)
        
        str_repr = str(mapper)
        assert "..." in str_repr  # Should show truncation
        assert repr(mapper) == "IDMapper(size=10)"
    
    def test_different_original_id_types(self):
        """Test various types of original IDs."""
        mapper = IDMapper()
        
        # Test different hashable types
        test_cases = [
            ("string_id", 0),
            (12345, 1),
            (3.14159, 2),
            ((1, 2, 3), 3),  # Tuple
            (frozenset([1, 2, 3]), 4),  # Frozenset
            (uuid.uuid4(), 5),  # UUID
            (True, 6),  # Boolean
            (None, 7),  # None
        ]
        
        for original_id, internal_id in test_cases:
            mapper.add_mapping(original_id, internal_id)
            assert mapper.get_internal(original_id) == internal_id
            assert mapper.get_original(internal_id) == original_id
        
        assert mapper.size() == len(test_cases)
    
    def test_large_dataset_performance(self):
        """Test performance with larger dataset."""
        mapper = IDMapper()
        
        # Add many mappings
        n_mappings = 1000
        for i in range(n_mappings):
            mapper.add_mapping(f"user_{i}", i)
        
        assert mapper.size() == n_mappings
        
        # Test batch operations
        original_ids = [f"user_{i}" for i in range(0, n_mappings, 10)]
        internal_ids = list(range(0, n_mappings, 10))
        
        result_internal = mapper.get_internal_batch(original_ids)
        assert result_internal == internal_ids
        
        result_original = mapper.get_original_batch(internal_ids)
        assert result_original == original_ids
    
    def test_edge_case_zero_internal_id(self):
        """Test that internal ID 0 is handled correctly."""
        mapper = IDMapper()
        mapper.add_mapping("zero_user", 0)
        
        assert mapper.get_internal("zero_user") == 0
        assert mapper.get_original(0) == "zero_user"
        assert 0 in mapper
        assert mapper.has_internal(0)
    
    def test_non_consecutive_internal_ids(self):
        """Test that non-consecutive internal IDs work correctly."""
        mapper = IDMapper()
        
        # Add non-consecutive internal IDs
        test_cases = [
            ("user_a", 0),
            ("user_b", 5),
            ("user_c", 100),
            ("user_d", 2),
        ]
        
        for original, internal in test_cases:
            mapper.add_mapping(original, internal)
        
        # Verify all mappings work
        for original, internal in test_cases:
            assert mapper.get_internal(original) == internal
            assert mapper.get_original(internal) == original
        
        assert mapper.size() == len(test_cases)


class TestIDMapperTypeHints:
    """Test type checking and hint validation."""
    
    def test_type_annotations(self):
        """Test that type annotations are correctly specified."""
        # This test mainly ensures the class has proper type hints
        # In a real project, you might use mypy for static type checking
        
        mapper = IDMapper()
        
        # Test that methods return expected types
        mapper.add_mapping("test", 0)
        
        internal_result = mapper.get_internal("test")
        assert isinstance(internal_result, int)
        
        original_result = mapper.get_original(0)
        assert original_result == "test"
        
        size_result = mapper.size()
        assert isinstance(size_result, int)
        
        batch_result = mapper.get_internal_batch(["test"])
        assert isinstance(batch_result, list)
        assert all(isinstance(x, int) for x in batch_result)
    
    def test_generic_type_handling(self):
        """Test that the class handles various generic types correctly."""
        mapper = IDMapper()
        
        # Test with different original ID types
        original_ids = [
            "string",
            123,
            3.14,
            (1, 2),
            uuid.uuid4(),
        ]
        
        for i, original_id in enumerate(original_ids):
            mapper.add_mapping(original_id, i)
            
            # Verify type preservation
            retrieved = mapper.get_original(i)
            assert type(retrieved) == type(original_id)
            assert retrieved == original_id


if __name__ == "__main__":
    # Run tests if script is executed directly
    pytest.main([__file__])