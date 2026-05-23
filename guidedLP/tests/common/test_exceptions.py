"""
Tests for the custom exception hierarchy.

This module provides comprehensive testing for all custom exceptions
in the GLP library, ensuring proper inheritance, message formatting,
and contextual information handling.
"""

import pytest
from typing import Dict, Any

from src.common.exceptions import (
    NetworkAnalysisError,
    ValidationError,
    GraphConstructionError,
    ConvergenceError,
    ConfigurationError,
    ComputationError,
    DataFormatError,
    validate_parameter,
    require_positive,
    check_convergence
)


class TestNetworkAnalysisError:
    """Test the base NetworkAnalysisError class."""
    
    def test_basic_error(self):
        """Test basic error creation."""
        error = NetworkAnalysisError("Test error")
        assert str(error) == "Test error"
        assert error.message == "Test error"
        assert error.details == {}
        assert error.context == {}
        assert error.cause is None
    
    def test_error_with_details(self):
        """Test error with details."""
        details = {"nodes": 100, "edges": 500}
        error = NetworkAnalysisError("Graph error", details=details)
        
        assert "Graph error" in str(error)
        assert "nodes=100" in str(error)
        assert "edges=500" in str(error)
        assert error.details == details
    
    def test_error_with_context(self):
        """Test error with context."""
        context = {"operation": "build_graph", "step": "validation"}
        error = NetworkAnalysisError("Failed", context=context)
        
        assert "Failed" in str(error)
        assert "operation=build_graph" in str(error)
        assert "step=validation" in str(error)
        assert error.context == context
    
    def test_error_with_cause(self):
        """Test error chaining with cause."""
        original_error = ValueError("Original problem")
        error = NetworkAnalysisError("Wrapper error", cause=original_error)
        
        assert error.cause == original_error
        assert error.__cause__ == original_error
    
    def test_add_context(self):
        """Test adding context to existing error."""
        error = NetworkAnalysisError("Test")
        result = error.add_context(operation="test", phase="setup")
        
        assert result is error  # Returns self for chaining
        assert error.context["operation"] == "test"
        assert error.context["phase"] == "setup"
    
    def test_get_debug_info(self):
        """Test comprehensive debug information."""
        error = NetworkAnalysisError(
            "Test error",
            details={"count": 5},
            context={"op": "test"}
        )
        
        debug_info = error.get_debug_info()
        
        assert debug_info["exception_type"] == "NetworkAnalysisError"
        assert debug_info["message"] == "Test error"
        assert debug_info["details"] == {"count": 5}
        assert debug_info["context"] == {"op": "test"}
    
    def test_large_details_truncation(self):
        """Test truncation of large details."""
        large_list = list(range(1000))
        error = NetworkAnalysisError("Test", details={"large_data": large_list})
        
        error_str = str(error)
        assert "large_data=<list with 1000 items>" in error_str
        # Should not contain the full list
        assert "999" not in error_str


class TestValidationError:
    """Test ValidationError (inherits from NetworkAnalysisError)."""
    
    def test_basic_validation_error(self):
        """Test basic validation error."""
        error = ValidationError("Invalid input")
        assert "Validation error: Invalid input" in str(error)
    
    def test_validation_error_with_field(self):
        """Test validation error with field."""
        error = ValidationError("Null values found", field="source")
        assert "Validation error in field 'source': Null values found" in str(error)
        assert error.field == "source"
    
    def test_validation_error_with_value_and_expected(self):
        """Test validation error with value and expected."""
        error = ValidationError(
            "Invalid format",
            field="timestamp",
            value="invalid_date",
            expected="ISO datetime format"
        )
        
        assert error.field == "timestamp"
        assert error.value == "invalid_date"
        assert error.expected == "ISO datetime format"
        assert "invalid_value=invalid_date" in str(error)
        assert "expected=ISO datetime format" in str(error)
    
    def test_validation_error_inheritance(self):
        """Test that ValidationError properly inherits from NetworkAnalysisError."""
        error = ValidationError("Test")
        assert isinstance(error, NetworkAnalysisError)
        assert isinstance(error, ValidationError)


class TestGraphConstructionError:
    """Test GraphConstructionError."""
    
    def test_basic_graph_error(self):
        """Test basic graph construction error."""
        error = GraphConstructionError("Failed to build graph")
        assert "Failed to build graph" in str(error)
    
    def test_graph_error_with_context(self):
        """Test graph error with context information."""
        error = GraphConstructionError(
            "Edge addition failed",
            graph_type="directed",
            node_count=100,
            edge_count=1500,
            operation="add_edges"
        )
        
        assert error.graph_type == "directed"
        assert error.node_count == 100
        assert error.edge_count == 1500
        assert error.operation == "add_edges"
        
        error_str = str(error)
        assert "graph_type=directed" in error_str
        assert "node_count=100" in error_str
        assert "edge_count=1500" in error_str
        assert "operation=add_edges" in error_str
    
    def test_graph_error_inheritance(self):
        """Test inheritance."""
        error = GraphConstructionError("Test")
        assert isinstance(error, NetworkAnalysisError)


class TestConvergenceError:
    """Test ConvergenceError."""
    
    def test_basic_convergence_error(self):
        """Test basic convergence error."""
        error = ConvergenceError("Algorithm did not converge")
        assert "Algorithm did not converge" in str(error)
    
    def test_convergence_error_with_diagnostics(self):
        """Test convergence error with diagnostic information."""
        error = ConvergenceError(
            "GLP failed to converge",
            algorithm="GLP",
            iterations=1000,
            max_iterations=1000,
            final_change=0.01,
            threshold=1e-6
        )
        
        assert error.algorithm == "GLP"
        assert error.iterations == 1000
        assert error.max_iterations == 1000
        assert error.final_change == 0.01
        assert error.threshold == 1e-6
        
        error_str = str(error)
        assert "algorithm=GLP" in error_str
        assert "iterations_completed=1000" in error_str
        assert "final_change=0.01" in error_str
        assert "convergence_threshold=1e-06" in error_str
        assert "convergence_ratio=" in error_str  # Should calculate ratio
    
    def test_is_partial_result_usable_true(self):
        """Test when partial results are usable."""
        error = ConvergenceError(
            "Not converged",
            final_change=1e-5,
            threshold=1e-6
        )
        
        # final_change is within 10x of threshold
        assert error.is_partial_result_usable() is True
    
    def test_is_partial_result_usable_false(self):
        """Test when partial results are not usable."""
        error = ConvergenceError(
            "Not converged",
            final_change=1e-3,
            threshold=1e-6
        )
        
        # final_change is > 10x threshold
        assert error.is_partial_result_usable() is False
    
    def test_is_partial_result_usable_missing_values(self):
        """Test when values are missing for usability check."""
        error = ConvergenceError("Not converged")
        assert error.is_partial_result_usable() is False


class TestConfigurationError:
    """Test ConfigurationError."""
    
    def test_basic_config_error(self):
        """Test basic configuration error."""
        error = ConfigurationError("Invalid parameter combination")
        assert "Invalid parameter combination" in str(error)
    
    def test_config_error_with_parameter_info(self):
        """Test configuration error with parameter details."""
        error = ConfigurationError(
            "Invalid method",
            parameter="method",
            value="invalid_method",
            valid_options=["option1", "option2", "option3"],
            function="test_function"
        )
        
        assert error.parameter == "method"
        assert error.value == "invalid_method"
        assert error.valid_options == ["option1", "option2", "option3"]
        assert error.function == "test_function"
        
        error_str = str(error)
        assert "Valid options for 'method'" in error_str
        assert "option1" in error_str
        assert "parameter=method" in error_str
        assert "function=test_function" in error_str


class TestComputationError:
    """Test ComputationError."""
    
    def test_basic_computation_error(self):
        """Test basic computation error."""
        error = ComputationError("Matrix operation failed")
        assert "Matrix operation failed" in str(error)
    
    def test_computation_error_with_details(self):
        """Test computation error with operation details."""
        error = ComputationError(
            "Insufficient memory",
            operation="centrality_calculation",
            error_type="memory",
            resource_info={"nodes": 1000000, "estimated_memory_gb": 32}
        )
        
        assert error.operation == "centrality_calculation"
        assert error.error_type == "memory"
        assert error.resource_info["nodes"] == 1000000
        
        error_str = str(error)
        assert "operation=centrality_calculation" in error_str
        assert "error_type=memory" in error_str
        assert "nodes=1000000" in error_str


class TestDataFormatError:
    """Test DataFormatError (inherits from ValidationError)."""
    
    def test_basic_format_error(self):
        """Test basic data format error."""
        error = DataFormatError("Invalid CSV structure")
        assert "Validation error: Invalid CSV structure" in str(error)
    
    def test_format_error_with_file_info(self):
        """Test format error with file information."""
        error = DataFormatError(
            "Invalid header",
            format_type="CSV",
            file_path="/path/to/file.csv",
            line_number=1
        )
        
        error_str = str(error)
        assert "format_type=CSV" in error_str
        assert "file_path=/path/to/file.csv" in error_str
        assert "line_number=1" in error_str
    
    def test_format_error_inheritance(self):
        """Test inheritance chain."""
        error = DataFormatError("Test")
        assert isinstance(error, ValidationError)
        assert isinstance(error, NetworkAnalysisError)


class TestConvenienceFunctions:
    """Test convenience functions for common error patterns."""
    
    def test_validate_parameter_valid(self):
        """Test parameter validation with valid input."""
        # Should not raise exception
        validate_parameter("option1", ["option1", "option2"], "test_param")
    
    def test_validate_parameter_invalid(self):
        """Test parameter validation with invalid input."""
        with pytest.raises(ConfigurationError) as exc_info:
            validate_parameter("invalid", ["option1", "option2"], "test_param", "test_function")
        
        error = exc_info.value
        assert error.parameter == "test_param"
        assert error.value == "invalid"
        assert error.valid_options == ["option1", "option2"]
        assert error.function == "test_function"
    
    def test_require_positive_valid(self):
        """Test positive number validation with valid input."""
        # Should not raise exceptions
        require_positive(5, "test_param")
        require_positive(1.5, "test_param")
        require_positive(0, "test_param", allow_zero=True)
    
    def test_require_positive_invalid(self):
        """Test positive number validation with invalid input."""
        with pytest.raises(ConfigurationError) as exc_info:
            require_positive(-5, "test_param")
        
        error = exc_info.value
        assert "must be positive" in str(error)
        assert error.parameter == "test_param"
        assert error.value == -5
    
    def test_require_positive_zero_not_allowed(self):
        """Test zero not allowed by default."""
        with pytest.raises(ConfigurationError):
            require_positive(0, "test_param")
    
    def test_require_positive_zero_allowed(self):
        """Test zero allowed when explicitly enabled."""
        # Should not raise exception
        require_positive(0, "test_param", allow_zero=True)
    
    def test_require_positive_negative_with_zero_allowed(self):
        """Test negative still fails even with allow_zero=True."""
        with pytest.raises(ConfigurationError) as exc_info:
            require_positive(-1, "test_param", allow_zero=True)
        
        error = exc_info.value
        assert "must be non-negative" in str(error)
    
    def test_check_convergence_converged(self):
        """Test convergence check when converged."""
        # Should not raise exception (change < threshold)
        check_convergence(1e-7, 1e-6, 50, 100, "test_algorithm")
    
    def test_check_convergence_not_converged_within_iterations(self):
        """Test convergence check when not converged but within iteration limit."""
        # Should not raise exception (iteration < max_iterations)
        check_convergence(1e-3, 1e-6, 50, 100, "test_algorithm")
    
    def test_check_convergence_failed(self):
        """Test convergence check when failed to converge."""
        with pytest.raises(ConvergenceError) as exc_info:
            check_convergence(1e-3, 1e-6, 100, 100, "test_algorithm")
        
        error = exc_info.value
        assert error.algorithm == "test_algorithm"
        assert error.iterations == 100
        assert error.max_iterations == 100
        assert error.final_change == 1e-3
        assert error.threshold == 1e-6


class TestExceptionChaining:
    """Test exception chaining and context preservation."""
    
    def test_exception_chaining(self):
        """Test that exceptions can be properly chained."""
        try:
            # Simulate nested error scenario
            try:
                raise ValueError("Original error")
            except ValueError as e:
                raise GraphConstructionError("Graph build failed", cause=e)
        except GraphConstructionError as graph_error:
            assert isinstance(graph_error.__cause__, ValueError)
            assert "Original error" in str(graph_error.__cause__)
    
    def test_context_accumulation(self):
        """Test accumulating context through error chain."""
        error = NetworkAnalysisError("Base error")
        error.add_context(module="network", phase="construction")
        error.add_context(operation="add_edges", batch=1)
        
        assert error.context["module"] == "network"
        assert error.context["phase"] == "construction"
        assert error.context["operation"] == "add_edges"
        assert error.context["batch"] == 1
    
    def test_debug_info_completeness(self):
        """Test that debug info includes all relevant information."""
        original_error = ValueError("Root cause")
        
        error = GraphConstructionError(
            "Graph failed",
            graph_type="directed",
            node_count=100,
            cause=original_error
        )
        error.add_context(operation="test")
        
        debug_info = error.get_debug_info()
        
        assert debug_info["exception_type"] == "GraphConstructionError"
        assert debug_info["message"] == "Graph failed"
        assert debug_info["context"]["operation"] == "test"
        assert debug_info["cause"] == "Root cause"
        assert "graph_type=directed" in str(error)


if __name__ == "__main__":
    # Run tests if script is executed directly
    pytest.main([__file__])