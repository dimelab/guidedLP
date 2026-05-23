"""
Custom exception hierarchy for the Guided Label Propagation library.

This module defines a comprehensive exception hierarchy that provides clear,
contextual error reporting for different failure modes in network analysis
and label propagation operations.

The hierarchy follows best practices for exception design:
- Clear inheritance structure with meaningful base classes
- Rich error messages with context and debugging information
- Optional details for programmatic error handling
- Consistent error reporting patterns across the library
"""

from typing import Dict, Any, Optional, List, Union
import traceback


class NetworkAnalysisError(Exception):
    """
    Base exception for all network analysis errors.
    
    This is the root exception class for the GLP library. All other custom
    exceptions inherit from this class, allowing users to catch all library-
    specific errors with a single except clause.
    
    Parameters
    ----------
    message : str
        Human-readable error message describing what went wrong
    details : Dict[str, Any], optional
        Additional structured information about the error for debugging
        or programmatic handling
    cause : Exception, optional
        The underlying exception that caused this error (for exception chaining)
    context : Dict[str, Any], optional
        Additional context about the operation that failed
        
    Attributes
    ----------
    message : str
        The error message
    details : Dict[str, Any]
        Additional error details
    cause : Exception, optional
        The underlying cause
    context : Dict[str, Any]
        Operation context
        
    Examples
    --------
    >>> raise NetworkAnalysisError("Graph construction failed")
    >>> raise NetworkAnalysisError(
    ...     "Invalid network size", 
    ...     details={"nodes": 0, "edges": 10}
    ... )
    
    Notes
    -----
    This base class provides a consistent interface for all library exceptions
    and includes common functionality for error reporting and debugging.
    """
    
    def __init__(
        self,
        message: str,
        details: Optional[Dict[str, Any]] = None,
        cause: Optional[Exception] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> None:
        self.message = message
        self.details = details or {}
        self.cause = cause
        self.context = context or {}
        
        # Build comprehensive error message
        full_message = message
        
        if self.details:
            detail_parts = []
            for key, value in self.details.items():
                if isinstance(value, (list, dict)) and len(str(value)) > 100:
                    # Truncate long collections
                    detail_parts.append(f"{key}=<{type(value).__name__} with {len(value)} items>")
                else:
                    detail_parts.append(f"{key}={value}")
            
            if detail_parts:
                full_message += f" (Details: {', '.join(detail_parts)})"
        
        if self.context:
            context_parts = [f"{k}={v}" for k, v in self.context.items()]
            if context_parts:
                full_message += f" (Context: {', '.join(context_parts)})"
        
        super().__init__(full_message)
        
        # Chain exceptions if cause is provided
        if cause is not None:
            self.__cause__ = cause
    
    def add_context(self, **kwargs: Any) -> 'NetworkAnalysisError':
        """
        Add additional context to the exception.
        
        Parameters
        ----------
        **kwargs
            Key-value pairs to add to the context
            
        Returns
        -------
        NetworkAnalysisError
            Self, for method chaining
            
        Examples
        --------
        >>> error = NetworkAnalysisError("Failed")
        >>> error.add_context(operation="build_graph", step="validation")
        """
        self.context.update(kwargs)
        return self
    
    def get_debug_info(self) -> Dict[str, Any]:
        """
        Get comprehensive debugging information.
        
        Returns
        -------
        Dict[str, Any]
            Dictionary containing all available error information
        """
        return {
            "exception_type": self.__class__.__name__,
            "message": self.message,
            "details": self.details,
            "context": self.context,
            "cause": str(self.cause) if self.cause else None,
            "traceback": traceback.format_exc() if hasattr(self, '__traceback__') else None
        }


class ValidationError(NetworkAnalysisError):
    """
    Exception raised for input validation errors.
    
    This exception is raised when input data does not meet the requirements
    for processing by the GLP library. It provides clear error messages
    to help users identify and fix data issues.
    
    Parameters
    ----------
    message : str
        Descriptive error message explaining the validation failure
    field : str, optional
        Name of the field or column that failed validation
    value : Any, optional
        The invalid value that caused the error
    expected : str, optional
        Description of what was expected
    details : Dict[str, Any], optional
        Additional details about the validation failure
        
    Examples
    --------
    >>> raise ValidationError("Source column contains null values", field="source")
    >>> raise ValidationError(
    ...     "Invalid timestamp format", 
    ...     field="timestamp",
    ...     value="invalid_date",
    ...     expected="ISO datetime format"
    ... )
    
    Notes
    -----
    ValidationError is the most commonly encountered exception, as it handles
    all data quality issues before processing begins. It provides rich context
    to help users correct their input data.
    """
    
    def __init__(
        self,
        message: str,
        field: Optional[str] = None,
        value: Optional[Any] = None,
        expected: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> None:
        self.field = field
        self.value = value
        self.expected = expected
        
        # Build enhanced details
        enhanced_details = details or {}
        if field is not None:
            enhanced_details["field"] = field
        if value is not None:
            enhanced_details["invalid_value"] = value
        if expected is not None:
            enhanced_details["expected"] = expected
        
        # Enhance message with field information
        if field:
            enhanced_message = f"Validation error in field '{field}': {message}"
        else:
            enhanced_message = f"Validation error: {message}"
        
        super().__init__(enhanced_message, details=enhanced_details, **kwargs)


class GraphConstructionError(NetworkAnalysisError):
    """
    Exception raised during graph construction and manipulation.
    
    This exception covers errors that occur when building NetworkIt graphs
    from input data, including issues with node/edge creation, ID mapping,
    and graph property validation.
    
    Parameters
    ----------
    message : str
        Description of the graph construction error
    graph_type : str, optional
        Type of graph being constructed (e.g., "directed", "bipartite")
    node_count : int, optional
        Number of nodes in the graph when error occurred
    edge_count : int, optional
        Number of edges processed when error occurred
    operation : str, optional
        Specific operation that failed (e.g., "add_edges", "create_nodes")
        
    Examples
    --------
    >>> raise GraphConstructionError(
    ...     "Failed to add edges to graph",
    ...     operation="add_edges",
    ...     edge_count=1500
    ... )
    >>> raise GraphConstructionError(
    ...     "Invalid bipartite graph structure",
    ...     graph_type="bipartite",
    ...     details={"source_partition_size": 100, "target_partition_size": 0}
    ... )
    
    Notes
    -----
    GraphConstructionError helps diagnose issues with NetworkIt graph creation,
    which is a critical step in the analysis pipeline. It provides context
    about graph properties and the specific operation that failed.
    """
    
    def __init__(
        self,
        message: str,
        graph_type: Optional[str] = None,
        node_count: Optional[int] = None,
        edge_count: Optional[int] = None,
        operation: Optional[str] = None,
        **kwargs
    ) -> None:
        self.graph_type = graph_type
        self.node_count = node_count
        self.edge_count = edge_count
        self.operation = operation
        
        # Build context information
        context = {}
        if graph_type:
            context["graph_type"] = graph_type
        if node_count is not None:
            context["node_count"] = node_count
        if edge_count is not None:
            context["edge_count"] = edge_count
        if operation:
            context["operation"] = operation
        
        super().__init__(message, context=context, **kwargs)


class ConvergenceError(NetworkAnalysisError):
    """
    Exception raised when iterative algorithms fail to converge.
    
    This exception is used for algorithms that use iterative methods,
    particularly the Guided Label Propagation algorithm, when they
    fail to reach convergence within specified limits.
    
    Parameters
    ----------
    message : str
        Description of the convergence failure
    algorithm : str, optional
        Name of the algorithm that failed to converge
    iterations : int, optional
        Number of iterations completed before giving up
    max_iterations : int, optional
        Maximum iterations allowed
    final_change : float, optional
        Final change value when convergence was checked
    threshold : float, optional
        Convergence threshold that was not met
    
    Examples
    --------
    >>> raise ConvergenceError(
    ...     "Label propagation did not converge",
    ...     algorithm="GLP",
    ...     iterations=1000,
    ...     max_iterations=1000,
    ...     final_change=0.01,
    ...     threshold=1e-6
    ... )
    
    Notes
    -----
    ConvergenceError provides diagnostic information to help users understand
    why an algorithm didn't converge and potentially adjust parameters.
    In some cases, partial results may still be usable despite non-convergence.
    """
    
    def __init__(
        self,
        message: str,
        algorithm: Optional[str] = None,
        iterations: Optional[int] = None,
        max_iterations: Optional[int] = None,
        final_change: Optional[float] = None,
        threshold: Optional[float] = None,
        **kwargs
    ) -> None:
        self.algorithm = algorithm
        self.iterations = iterations
        self.max_iterations = max_iterations
        self.final_change = final_change
        self.threshold = threshold
        
        # Build details for convergence diagnostics
        details = kwargs.get('details', {})
        if algorithm:
            details["algorithm"] = algorithm
        if iterations is not None:
            details["iterations_completed"] = iterations
        if max_iterations is not None:
            details["max_iterations"] = max_iterations
        if final_change is not None:
            details["final_change"] = final_change
        if threshold is not None:
            details["convergence_threshold"] = threshold
        
        # Add convergence ratio if both values available
        if final_change is not None and threshold is not None and threshold > 0:
            details["convergence_ratio"] = final_change / threshold
        
        kwargs["details"] = details
        super().__init__(message, **kwargs)
    
    def is_partial_result_usable(self) -> bool:
        """
        Determine if partial results might still be usable despite non-convergence.
        
        Returns
        -------
        bool
            True if results might be partially usable
            
        Notes
        -----
        This heuristic suggests that if the algorithm made significant progress
        (final change is within 10x of threshold), results might still be meaningful.
        """
        if self.final_change is None or self.threshold is None:
            return False
        
        return self.final_change <= (10 * self.threshold)


class ConfigurationError(NetworkAnalysisError):
    """
    Exception raised for invalid configuration or parameter values.
    
    This exception handles errors related to invalid parameter combinations,
    unsupported configurations, or conflicting settings that prevent
    proper execution of analysis functions.
    
    Parameters
    ----------
    message : str
        Description of the configuration error
    parameter : str, optional
        Name of the problematic parameter
    value : Any, optional
        The invalid parameter value
    valid_options : List[Any], optional
        List of valid options for the parameter
    function : str, optional
        Name of the function where the error occurred
        
    Examples
    --------
    >>> raise ConfigurationError(
    ...     "Invalid propagation method",
    ...     parameter="method",
    ...     value="invalid_method",
    ...     valid_options=["in_degree", "out_degree", "undirected"]
    ... )
    >>> raise ConfigurationError(
    ...     "Conflicting parameters: cannot specify both target_nodes and target_edges",
    ...     function="apply_backbone"
    ... )
    
    Notes
    -----
    ConfigurationError helps users identify parameter issues early, before
    expensive computations begin. It provides guidance on valid options
    and parameter combinations.
    """
    
    def __init__(
        self,
        message: str,
        parameter: Optional[str] = None,
        value: Optional[Any] = None,
        valid_options: Optional[List[Any]] = None,
        function: Optional[str] = None,
        **kwargs
    ) -> None:
        self.parameter = parameter
        self.value = value
        self.valid_options = valid_options
        self.function = function
        
        # Build enhanced details
        details = kwargs.get('details', {})
        if parameter:
            details["parameter"] = parameter
        if value is not None:
            details["invalid_value"] = value
        if valid_options:
            details["valid_options"] = valid_options
        if function:
            details["function"] = function
        
        # Enhance message with parameter guidance
        enhanced_message = message
        if parameter and valid_options:
            enhanced_message += f". Valid options for '{parameter}': {valid_options}"
        
        kwargs["details"] = details
        super().__init__(enhanced_message, **kwargs)


class ComputationError(NetworkAnalysisError):
    """
    Exception raised when computational operations fail.
    
    This exception covers numerical errors, memory issues, and other
    computational failures that can occur during analysis operations.
    
    Parameters
    ----------
    message : str
        Description of the computational error
    operation : str, optional
        The computational operation that failed
    error_type : str, optional
        Type of computational error (e.g., "numerical", "memory", "overflow")
    resource_info : Dict[str, Any], optional
        Information about computational resources when error occurred
        
    Examples
    --------
    >>> raise ComputationError(
    ...     "Matrix operation failed due to singular matrix",
    ...     operation="matrix_inversion",
    ...     error_type="numerical"
    ... )
    >>> raise ComputationError(
    ...     "Insufficient memory for large graph",
    ...     operation="centrality_calculation",
    ...     error_type="memory",
    ...     resource_info={"nodes": 1000000, "estimated_memory_gb": 32}
    ... )
    
    Notes
    -----
    ComputationError helps distinguish between data issues (ValidationError)
    and computational/resource issues that may require different solutions.
    """
    
    def __init__(
        self,
        message: str,
        operation: Optional[str] = None,
        error_type: Optional[str] = None,
        resource_info: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> None:
        self.operation = operation
        self.error_type = error_type
        self.resource_info = resource_info or {}
        
        # Build context
        context = {}
        if operation:
            context["operation"] = operation
        if error_type:
            context["error_type"] = error_type
        
        # Add resource info to details
        details = kwargs.get('details', {})
        details.update(self.resource_info)
        
        kwargs["details"] = details
        kwargs["context"] = context
        
        super().__init__(message, **kwargs)


class DataFormatError(ValidationError):
    """
    Exception raised for data format and structure errors.
    
    This is a specialized ValidationError for issues specifically related
    to data format, file structure, or schema problems.
    
    Parameters
    ----------
    message : str
        Description of the format error
    format_type : str, optional
        Expected format (e.g., "CSV", "DataFrame", "GEXF")
    file_path : str, optional
        Path to the problematic file
    line_number : int, optional
        Line number where error occurred (for file parsing)
        
    Examples
    --------
    >>> raise DataFormatError(
    ...     "Invalid CSV header",
    ...     format_type="CSV",
    ...     file_path="/path/to/edges.csv",
    ...     line_number=1
    ... )
    
    Notes
    -----
    DataFormatError provides specific context for file and format issues,
    which are common when working with external data sources.
    """
    
    def __init__(
        self,
        message: str,
        format_type: Optional[str] = None,
        file_path: Optional[str] = None,
        line_number: Optional[int] = None,
        **kwargs
    ) -> None:
        details = kwargs.get('details', {})
        
        if format_type:
            details["format_type"] = format_type
        if file_path:
            details["file_path"] = file_path
        if line_number is not None:
            details["line_number"] = line_number
        
        kwargs["details"] = details
        super().__init__(message, **kwargs)


# Convenience functions for common error patterns

def validate_parameter(
    value: Any,
    valid_options: List[Any],
    parameter_name: str,
    function_name: Optional[str] = None
) -> None:
    """
    Validate that a parameter value is in the list of valid options.
    
    Parameters
    ----------
    value : Any
        The parameter value to validate
    valid_options : List[Any]
        List of valid options
    parameter_name : str
        Name of the parameter
    function_name : str, optional
        Name of the function being called
        
    Raises
    ------
    ConfigurationError
        If value is not in valid_options
    """
    if value not in valid_options:
        raise ConfigurationError(
            f"Invalid value for parameter '{parameter_name}': {value}",
            parameter=parameter_name,
            value=value,
            valid_options=valid_options,
            function=function_name
        )


def require_positive(
    value: Union[int, float],
    parameter_name: str,
    allow_zero: bool = False
) -> None:
    """
    Validate that a numeric parameter is positive.
    
    Parameters
    ----------
    value : Union[int, float]
        The numeric value to validate
    parameter_name : str
        Name of the parameter
    allow_zero : bool, default False
        Whether to allow zero values
        
    Raises
    ------
    ConfigurationError
        If value is not positive (or non-negative if allow_zero=True)
    """
    if allow_zero and value < 0:
        raise ConfigurationError(
            f"Parameter '{parameter_name}' must be non-negative, got {value}",
            parameter=parameter_name,
            value=value
        )
    elif not allow_zero and value <= 0:
        raise ConfigurationError(
            f"Parameter '{parameter_name}' must be positive, got {value}",
            parameter=parameter_name,
            value=value
        )


def check_convergence(
    change: float,
    threshold: float,
    iteration: int,
    max_iterations: int,
    algorithm: str = "iterative algorithm"
) -> None:
    """
    Check convergence and raise ConvergenceError if not converged.
    
    Parameters
    ----------
    change : float
        Current change magnitude
    threshold : float
        Convergence threshold
    iteration : int
        Current iteration number
    max_iterations : int
        Maximum allowed iterations
    algorithm : str, default "iterative algorithm"
        Name of the algorithm
        
    Raises
    ------
    ConvergenceError
        If iteration >= max_iterations and change > threshold
    """
    if iteration >= max_iterations and change > threshold:
        raise ConvergenceError(
            f"{algorithm} failed to converge within {max_iterations} iterations",
            algorithm=algorithm,
            iterations=iteration,
            max_iterations=max_iterations,
            final_change=change,
            threshold=threshold
        )