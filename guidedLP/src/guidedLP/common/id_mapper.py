"""
ID mapping utilities for the Guided Label Propagation library.

This module provides bidirectional mapping between original node IDs (arbitrary types)
and internal NetworkIt integer IDs (0, 1, 2, ...). This is critical because NetworkIt
requires consecutive integer node IDs, but input data uses arbitrary identifiers.
"""

from typing import Any, Dict, List, Union, Optional, TypeVar
import warnings

# Type variable for original ID types
OriginalIDType = TypeVar('OriginalIDType')


class IDMapper:
    """
    Bidirectional mapping between original and internal node IDs.
    
    NetworkIt graphs require consecutive integer node IDs starting from 0,
    but input data typically uses arbitrary identifiers (strings, UUIDs, etc.).
    This class provides efficient bidirectional mapping to handle this conversion.
    
    Attributes
    ----------
    original_to_internal : Dict[Any, int]
        Maps original IDs to NetworkIt internal IDs (0, 1, 2, ...)
    internal_to_original : Dict[int, Any]
        Maps NetworkIt internal IDs to original IDs
    
    Examples
    --------
    >>> mapper = IDMapper()
    >>> mapper.add_mapping("user_123", 0)
    >>> mapper.add_mapping("user_456", 1)
    >>> internal = mapper.get_internal("user_123")  # Returns 0
    >>> original = mapper.get_original(0)           # Returns "user_123"
    
    Notes
    -----
    - Original IDs can be any hashable type (str, int, UUID, tuple, etc.)
    - Internal IDs are always consecutive integers starting from 0
    - Mappings are bidirectional and must be consistent
    - Thread-safe for read operations, not thread-safe for modifications
    """
    
    def __init__(self) -> None:
        """Initialize empty ID mapper."""
        self.original_to_internal: Dict[Any, int] = {}
        self.internal_to_original: Dict[int, Any] = {}
    
    def get_internal(self, original_id: Any) -> int:
        """
        Get internal NetworkIt ID for a given original ID.
        
        Parameters
        ----------
        original_id : Any
            Original node identifier from input data
            
        Returns
        -------
        int
            Corresponding NetworkIt internal ID (0, 1, 2, ...)
            
        Raises
        ------
        KeyError
            If original_id is not found in the mapping
            
        Examples
        --------
        >>> mapper = IDMapper()
        >>> mapper.add_mapping("user_123", 0)
        >>> mapper.get_internal("user_123")
        0
        >>> mapper.get_internal("unknown_user")  # doctest: +SKIP
        Traceback (most recent call last):
        KeyError: 'unknown_user'
        """
        try:
            return self.original_to_internal[original_id]
        except KeyError:
            raise KeyError(f"Original ID '{original_id}' not found in mapping")
    
    def get_original(self, internal_id: int) -> Any:
        """
        Get original ID for a given internal NetworkIt ID.
        
        Parameters
        ----------
        internal_id : int
            NetworkIt internal node ID
            
        Returns
        -------
        Any
            Corresponding original node identifier
            
        Raises
        ------
        KeyError
            If internal_id is not found in the mapping
        TypeError
            If internal_id is not an integer
            
        Examples
        --------
        >>> mapper = IDMapper()
        >>> mapper.add_mapping("user_123", 0)
        >>> mapper.get_original(0)
        'user_123'
        >>> mapper.get_original(99)  # doctest: +SKIP
        Traceback (most recent call last):
        KeyError: 'Internal ID 99 not found in mapping'
        """
        if not isinstance(internal_id, int):
            raise TypeError(f"Internal ID must be integer, got {type(internal_id)}")
        
        try:
            return self.internal_to_original[internal_id]
        except KeyError:
            raise KeyError(f"Internal ID {internal_id} not found in mapping")
    
    def get_internal_batch(self, original_ids: List[Any]) -> List[int]:
        """
        Get internal IDs for a batch of original IDs (efficient batch operation).
        
        Parameters
        ----------
        original_ids : List[Any]
            List of original node identifiers
            
        Returns
        -------
        List[int]
            List of corresponding NetworkIt internal IDs
            
        Raises
        ------
        KeyError
            If any original_id is not found in the mapping
        TypeError
            If original_ids is not a list
            
        Examples
        --------
        >>> mapper = IDMapper()
        >>> mapper.add_mapping("user_123", 0)
        >>> mapper.add_mapping("user_456", 1)
        >>> mapper.get_internal_batch(["user_123", "user_456"])
        [0, 1]
        
        Notes
        -----
        This method is more efficient than calling get_internal() in a loop
        for large batches, as it minimizes function call overhead.
        """
        if not isinstance(original_ids, list):
            raise TypeError(f"original_ids must be a list, got {type(original_ids)}")
        
        result = []
        for original_id in original_ids:
            try:
                result.append(self.original_to_internal[original_id])
            except KeyError:
                raise KeyError(f"Original ID '{original_id}' not found in mapping")
        
        return result
    
    def get_original_batch(self, internal_ids: List[int]) -> List[Any]:
        """
        Get original IDs for a batch of internal IDs (efficient batch operation).
        
        Parameters
        ----------
        internal_ids : List[int]
            List of NetworkIt internal node IDs
            
        Returns
        -------
        List[Any]
            List of corresponding original node identifiers
            
        Raises
        ------
        KeyError
            If any internal_id is not found in the mapping
        TypeError
            If internal_ids is not a list or contains non-integers
            
        Examples
        --------
        >>> mapper = IDMapper()
        >>> mapper.add_mapping("user_123", 0)
        >>> mapper.add_mapping("user_456", 1)
        >>> mapper.get_original_batch([0, 1])
        ['user_123', 'user_456']
        
        Notes
        -----
        This method is more efficient than calling get_original() in a loop
        for large batches, as it minimizes function call overhead.
        """
        if not isinstance(internal_ids, list):
            raise TypeError(f"internal_ids must be a list, got {type(internal_ids)}")
        
        result = []
        for internal_id in internal_ids:
            if not isinstance(internal_id, int):
                raise TypeError(f"All internal IDs must be integers, got {type(internal_id)}")
            
            try:
                result.append(self.internal_to_original[internal_id])
            except KeyError:
                raise KeyError(f"Internal ID {internal_id} not found in mapping")
        
        return result
    
    def add_mapping(self, original_id: Any, internal_id: int) -> None:
        """
        Add a new ID mapping pair.
        
        Parameters
        ----------
        original_id : Any
            Original node identifier (must be hashable)
        internal_id : int
            NetworkIt internal ID (must be non-negative integer)
            
        Raises
        ------
        ValueError
            If original_id or internal_id already exists in mapping
            If internal_id is negative
        TypeError
            If internal_id is not an integer
            If original_id is not hashable
            
        Examples
        --------
        >>> mapper = IDMapper()
        >>> mapper.add_mapping("user_123", 0)
        >>> mapper.add_mapping("user_456", 1)
        >>> mapper.size()
        2
        
        Notes
        -----
        Both original_id and internal_id must be unique. This ensures
        bidirectional mapping consistency. Internal IDs should typically
        be consecutive starting from 0 for NetworkIt compatibility.
        """
        # Type validation
        if not isinstance(internal_id, int):
            raise TypeError(f"Internal ID must be integer, got {type(internal_id)}")
        
        if internal_id < 0:
            raise ValueError(f"Internal ID must be non-negative, got {internal_id}")
        
        # Check if original_id is hashable
        try:
            hash(original_id)
        except TypeError:
            raise TypeError(f"Original ID must be hashable, got {type(original_id)}")
        
        # Check for existing mappings
        if original_id in self.original_to_internal:
            existing_internal = self.original_to_internal[original_id]
            raise ValueError(
                f"Original ID '{original_id}' already mapped to internal ID {existing_internal}"
            )
        
        if internal_id in self.internal_to_original:
            existing_original = self.internal_to_original[internal_id]
            raise ValueError(
                f"Internal ID {internal_id} already mapped to original ID '{existing_original}'"
            )
        
        # Add bidirectional mapping
        self.original_to_internal[original_id] = internal_id
        self.internal_to_original[internal_id] = original_id
    
    def size(self) -> int:
        """
        Get the number of mapped node IDs.
        
        Returns
        -------
        int
            Number of nodes in the mapping
            
        Examples
        --------
        >>> mapper = IDMapper()
        >>> mapper.size()
        0
        >>> mapper.add_mapping("user_123", 0)
        >>> mapper.size()
        1
        
        Notes
        -----
        Both dictionaries should always have the same size due to
        bidirectional mapping constraints.
        """
        return len(self.original_to_internal)
    
    def to_dict(self) -> Dict[str, Dict]:
        """
        Export mapping as dictionary for serialization.
        
        Returns
        -------
        Dict[str, Dict]
            Dictionary with 'original_to_internal' and 'internal_to_original' keys
            
        Examples
        --------
        >>> mapper = IDMapper()
        >>> mapper.add_mapping("user_123", 0)
        >>> result = mapper.to_dict()
        >>> result['original_to_internal']['user_123']
        0
        >>> result['internal_to_original'][0]
        'user_123'
        
        Notes
        -----
        The returned dictionary can be serialized to JSON or other formats
        for persistence. Use from_dict() to reconstruct the IDMapper.
        """
        return {
            'original_to_internal': dict(self.original_to_internal),
            'internal_to_original': {str(k): v for k, v in self.internal_to_original.items()}
        }
    
    @classmethod
    def from_dict(cls, mapping: Dict[str, Dict]) -> 'IDMapper':
        """
        Create IDMapper from dictionary (deserialization).
        
        Parameters
        ----------
        mapping : Dict[str, Dict]
            Dictionary with 'original_to_internal' and 'internal_to_original' keys
            
        Returns
        -------
        IDMapper
            Reconstructed IDMapper instance
            
        Raises
        ------
        ValueError
            If mapping dictionary has invalid structure or inconsistent data
        KeyError
            If required keys are missing from mapping dictionary
            
        Examples
        --------
        >>> mapping_dict = {
        ...     'original_to_internal': {'user_123': 0, 'user_456': 1},
        ...     'internal_to_original': {'0': 'user_123', '1': 'user_456'}
        ... }
        >>> mapper = IDMapper.from_dict(mapping_dict)
        >>> mapper.get_internal('user_123')
        0
        
        Notes
        -----
        This method validates that the bidirectional mapping is consistent
        and rebuilds both internal dictionaries.
        """
        try:
            original_to_internal = mapping['original_to_internal']
            internal_to_original_str = mapping['internal_to_original']
        except KeyError as e:
            raise KeyError(f"Missing required key in mapping dictionary: {e}")
        
        # Convert string keys back to integers for internal_to_original
        try:
            internal_to_original = {int(k): v for k, v in internal_to_original_str.items()}
        except (ValueError, TypeError) as e:
            raise ValueError(f"Invalid internal ID in mapping: {e}")
        
        # Create new mapper and validate consistency
        mapper = cls()
        
        # Validate mapping consistency
        if len(original_to_internal) != len(internal_to_original):
            raise ValueError(
                f"Inconsistent mapping sizes: {len(original_to_internal)} vs {len(internal_to_original)}"
            )
        
        # Add mappings with validation
        for original_id, internal_id in original_to_internal.items():
            # Verify bidirectional consistency
            if internal_id not in internal_to_original:
                raise ValueError(
                    f"Inconsistent mapping: original '{original_id}' -> {internal_id}, "
                    f"but {internal_id} not in reverse mapping"
                )
            
            if internal_to_original[internal_id] != original_id:
                raise ValueError(
                    f"Inconsistent mapping: original '{original_id}' -> {internal_id}, "
                    f"but {internal_id} -> '{internal_to_original[internal_id]}'"
                )
            
            mapper.original_to_internal[original_id] = internal_id
            mapper.internal_to_original[internal_id] = original_id
        
        return mapper
    
    def is_empty(self) -> bool:
        """
        Check if the mapper is empty.
        
        Returns
        -------
        bool
            True if no mappings exist, False otherwise
            
        Examples
        --------
        >>> mapper = IDMapper()
        >>> mapper.is_empty()
        True
        >>> mapper.add_mapping("user_123", 0)
        >>> mapper.is_empty()
        False
        """
        return len(self.original_to_internal) == 0
    
    def has_original(self, original_id: Any) -> bool:
        """
        Check if an original ID exists in the mapping.
        
        Parameters
        ----------
        original_id : Any
            Original node identifier to check
            
        Returns
        -------
        bool
            True if original_id exists in mapping, False otherwise
            
        Examples
        --------
        >>> mapper = IDMapper()
        >>> mapper.add_mapping("user_123", 0)
        >>> mapper.has_original("user_123")
        True
        >>> mapper.has_original("unknown_user")
        False
        """
        return original_id in self.original_to_internal
    
    def has_internal(self, internal_id: int) -> bool:
        """
        Check if an internal ID exists in the mapping.
        
        Parameters
        ----------
        internal_id : int
            Internal node ID to check
            
        Returns
        -------
        bool
            True if internal_id exists in mapping, False otherwise
            
        Examples
        --------
        >>> mapper = IDMapper()
        >>> mapper.add_mapping("user_123", 0)
        >>> mapper.has_internal(0)
        True
        >>> mapper.has_internal(99)
        False
        """
        return internal_id in self.internal_to_original
    
    def clear(self) -> None:
        """
        Remove all mappings from the mapper.
        
        Examples
        --------
        >>> mapper = IDMapper()
        >>> mapper.add_mapping("user_123", 0)
        >>> mapper.size()
        1
        >>> mapper.clear()
        >>> mapper.size()
        0
        """
        self.original_to_internal.clear()
        self.internal_to_original.clear()
    
    def __len__(self) -> int:
        """Return the number of mappings (same as size())."""
        return self.size()
    
    def __contains__(self, item: Any) -> bool:
        """
        Check if an ID (original or internal) exists in the mapping.
        
        Parameters
        ----------
        item : Any
            ID to check (original or internal)
            
        Returns
        -------
        bool
            True if item exists as either original or internal ID
            
        Examples
        --------
        >>> mapper = IDMapper()
        >>> mapper.add_mapping("user_123", 0)
        >>> "user_123" in mapper
        True
        >>> 0 in mapper
        True
        >>> "unknown" in mapper
        False
        """
        if isinstance(item, int):
            return self.has_internal(item)
        else:
            return self.has_original(item)
    
    def __repr__(self) -> str:
        """String representation of the mapper."""
        return f"IDMapper(size={self.size()})"
    
    def __str__(self) -> str:
        """Human-readable string representation."""
        if self.is_empty():
            return "IDMapper(empty)"
        
        # Show first few mappings as examples
        items = list(self.original_to_internal.items())[:3]
        examples = [f"'{orig}' -> {internal}" for orig, internal in items]
        
        if self.size() > 3:
            examples.append(f"... ({self.size() - 3} more)")
        
        return f"IDMapper({', '.join(examples)})"