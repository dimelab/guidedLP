"""
Tests for logging configuration.

This module provides comprehensive testing for the logging configuration
functionality, including environment variable handling, file logging,
JSON formatting, and performance logging features.
"""

import pytest
import logging
import tempfile
import os
import json
import time
from pathlib import Path
from unittest.mock import patch, MagicMock
from io import StringIO

from src.common.logging_config import (
    setup_logging,
    get_logger,
    configure_external_library_logging,
    log_function_entry,
    log_performance_metric,
    LoggingTimer,
    JSONFormatter,
    PerformanceFilter,
    ENV_LOG_LEVEL,
    ENV_LOG_FILE,
    ENV_LOG_DIR,
    ENV_LOG_CONSOLE,
    ENV_LOG_JSON,
    ENV_LOG_PERFORMANCE
)


class TestSetupLogging:
    """Test the main setup_logging function."""
    
    def setup_method(self):
        """Clear any existing handlers before each test."""
        logger = logging.getLogger("glp")
        logger.handlers.clear()
        logger.setLevel(logging.NOTSET)
    
    def test_basic_setup(self):
        """Test basic logging setup with defaults."""
        logger = setup_logging()
        
        assert logger.name == "glp"
        assert logger.level == logging.INFO
        assert len(logger.handlers) == 1  # Console handler only
        assert isinstance(logger.handlers[0], logging.StreamHandler)
        assert not logger.propagate
    
    def test_custom_level(self):
        """Test setting custom log level."""
        logger = setup_logging(level="DEBUG")
        assert logger.level == logging.DEBUG
        
        logger = setup_logging(level="WARNING", force_setup=True)
        assert logger.level == logging.WARNING
    
    def test_invalid_level(self):
        """Test that invalid log level raises ValueError."""
        with pytest.raises(ValueError, match="Invalid logging level"):
            setup_logging(level="INVALID_LEVEL")
    
    def test_file_logging(self):
        """Test file logging setup."""
        with tempfile.TemporaryDirectory() as temp_dir:
            log_file = os.path.join(temp_dir, "test.log")
            
            logger = setup_logging(log_file=log_file, console=False)
            
            # Should have file handler only
            assert len(logger.handlers) == 1
            assert isinstance(logger.handlers[0], logging.handlers.RotatingFileHandler)
            
            # Test that file is created and logging works
            logger.info("Test message")
            
            assert os.path.exists(log_file)
            with open(log_file, 'r') as f:
                content = f.read()
                assert "Test message" in content
    
    def test_both_console_and_file(self):
        """Test setup with both console and file handlers."""
        with tempfile.TemporaryDirectory() as temp_dir:
            log_file = os.path.join(temp_dir, "test.log")
            
            logger = setup_logging(log_file=log_file, console=True)
            
            # Should have both handlers
            assert len(logger.handlers) == 2
            handler_types = [type(h) for h in logger.handlers]
            assert logging.StreamHandler in handler_types
            assert logging.handlers.RotatingFileHandler in handler_types
    
    def test_console_disabled(self):
        """Test disabling console logging."""
        logger = setup_logging(console=False)
        
        # Should have no handlers (no file specified, console disabled)
        assert len(logger.handlers) == 0
    
    def test_log_directory_creation(self):
        """Test that log directory is created if it doesn't exist."""
        with tempfile.TemporaryDirectory() as temp_dir:
            log_dir = os.path.join(temp_dir, "new_dir", "logs")
            log_file = os.path.join(log_dir, "test.log")
            
            logger = setup_logging(log_file=log_file, console=False)
            
            # Directory should be created
            assert os.path.exists(log_dir)
            assert os.path.isdir(log_dir)
            
            # Logging should work
            logger.info("Test message")
            assert os.path.exists(log_file)
    
    def test_force_setup_reconfiguration(self):
        """Test force reconfiguration of existing logger."""
        # Initial setup
        logger1 = setup_logging(level="INFO", console=True)
        initial_handlers = len(logger1.handlers)
        
        # Second setup without force should not change anything
        logger2 = setup_logging(level="DEBUG", console=False)
        assert logger2 is logger1  # Same logger instance
        assert len(logger2.handlers) == initial_handlers
        assert logger2.level == logging.INFO  # Unchanged
        
        # Force setup should reconfigure
        logger3 = setup_logging(level="DEBUG", console=False, force_setup=True)
        assert logger3 is logger1  # Same logger instance
        assert logger3.level == logging.DEBUG  # Changed
        # Handlers may be different (console disabled)
    
    def test_json_formatting(self):
        """Test JSON formatting option."""
        with tempfile.TemporaryDirectory() as temp_dir:
            log_file = os.path.join(temp_dir, "test.log")
            
            logger = setup_logging(
                log_file=log_file,
                console=False,
                json_format=True
            )
            
            logger.info("Test message", extra={"custom_field": "custom_value"})
            
            # Read and parse JSON log
            with open(log_file, 'r') as f:
                log_line = f.read().strip()
                log_data = json.loads(log_line)
            
            assert log_data["level"] == "INFO"
            assert log_data["message"] == "Test message"
            assert log_data["logger"] == "glp"
            assert "timestamp" in log_data
            assert log_data["custom_field"] == "custom_value"
    
    def test_performance_logging(self):
        """Test performance logging setup."""
        with tempfile.TemporaryDirectory() as temp_dir:
            log_file = os.path.join(temp_dir, "test.log")
            
            logger = setup_logging(
                log_file=log_file,
                console=False,
                performance_logging=True
            )
            
            # Performance logger should be created
            perf_logger = logging.getLogger("glp.performance")
            assert len(perf_logger.handlers) > 0
            
            # Performance log file should be created
            perf_log_file = os.path.join(temp_dir, "performance.log")
            assert os.path.exists(perf_log_file)


class TestEnvironmentVariables:
    """Test environment variable configuration."""
    
    def setup_method(self):
        """Clear logger before each test."""
        logger = logging.getLogger("glp")
        logger.handlers.clear()
        logger.setLevel(logging.NOTSET)
    
    def test_log_level_from_env(self):
        """Test setting log level via environment variable."""
        with patch.dict(os.environ, {ENV_LOG_LEVEL: "DEBUG"}):
            logger = setup_logging()
            assert logger.level == logging.DEBUG
    
    def test_log_file_from_env(self):
        """Test setting log file via environment variable."""
        with tempfile.TemporaryDirectory() as temp_dir:
            log_file = os.path.join(temp_dir, "env_test.log")
            
            with patch.dict(os.environ, {ENV_LOG_FILE: log_file}):
                logger = setup_logging(console=False)
                
                logger.info("Environment test")
                assert os.path.exists(log_file)
    
    def test_log_dir_from_env(self):
        """Test setting log directory via environment variable."""
        with tempfile.TemporaryDirectory() as temp_dir:
            log_dir = os.path.join(temp_dir, "env_logs")
            
            with patch.dict(os.environ, {ENV_LOG_DIR: log_dir}):
                logger = setup_logging(console=False)
                
                # Should create glp.log in the specified directory
                expected_file = os.path.join(log_dir, "glp.log")
                logger.info("Directory test")
                assert os.path.exists(expected_file)
    
    def test_boolean_env_vars(self):
        """Test parsing boolean environment variables."""
        # Test console disabled
        with patch.dict(os.environ, {ENV_LOG_CONSOLE: "false"}):
            logger = setup_logging()
            # Should have no handlers (no file, console disabled)
            assert len(logger.handlers) == 0
        
        # Test JSON enabled
        with tempfile.TemporaryDirectory() as temp_dir:
            log_file = os.path.join(temp_dir, "json_test.log")
            
            with patch.dict(os.environ, {ENV_LOG_JSON: "true"}):
                logger = setup_logging(log_file=log_file, console=False, force_setup=True)
                
                # Check that JSON formatter is used
                handler = logger.handlers[0]
                assert isinstance(handler.formatter, JSONFormatter)
    
    def test_env_var_boolean_parsing(self):
        """Test various boolean value formats in environment variables."""
        true_values = ["true", "True", "TRUE", "yes", "Yes", "1", "on", "ON"]
        false_values = ["false", "False", "FALSE", "no", "No", "0", "off", "OFF"]
        
        for true_val in true_values:
            with patch.dict(os.environ, {ENV_LOG_PERFORMANCE: true_val}):
                logger = setup_logging(force_setup=True)
                perf_logger = logging.getLogger("glp.performance")
                # Should have performance filter
                filters = [f for f in perf_logger.filters if isinstance(f, PerformanceFilter)]
                assert len(filters) > 0
        
        for false_val in false_values:
            with patch.dict(os.environ, {ENV_LOG_PERFORMANCE: false_val}):
                logger = setup_logging(force_setup=True)
                perf_logger = logging.getLogger("glp.performance")
                # Should not have performance filter
                filters = [f for f in perf_logger.filters if isinstance(f, PerformanceFilter)]
                assert len(filters) == 0


class TestJSONFormatter:
    """Test the JSON formatter."""
    
    def test_basic_json_formatting(self):
        """Test basic JSON formatting."""
        formatter = JSONFormatter()
        
        # Create a log record
        record = logging.LogRecord(
            name="test_logger",
            level=logging.INFO,
            pathname="/path/to/file.py",
            lineno=42,
            msg="Test message",
            args=(),
            exc_info=None
        )
        record.funcName = "test_function"
        record.module = "test_module"
        
        formatted = formatter.format(record)
        data = json.loads(formatted)
        
        assert data["level"] == "INFO"
        assert data["message"] == "Test message"
        assert data["logger"] == "test_logger"
        assert data["function"] == "test_function"
        assert data["module"] == "test_module"
        assert data["line"] == 42
        assert "timestamp" in data
    
    def test_json_with_extra_fields(self):
        """Test JSON formatting with extra fields."""
        formatter = JSONFormatter()
        
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="/path/to/file.py",
            lineno=1,
            msg="Test",
            args=(),
            exc_info=None
        )
        record.custom_field = "custom_value"
        record.operation = "test_op"
        
        formatted = formatter.format(record)
        data = json.loads(formatted)
        
        assert data["custom_field"] == "custom_value"
        assert data["operation"] == "test_op"
    
    def test_json_with_exception(self):
        """Test JSON formatting with exception information."""
        formatter = JSONFormatter()
        
        try:
            raise ValueError("Test exception")
        except ValueError:
            record = logging.LogRecord(
                name="test",
                level=logging.ERROR,
                pathname="/path/to/file.py",
                lineno=1,
                msg="Error occurred",
                args=(),
                exc_info=True
            )
        
        formatted = formatter.format(record)
        data = json.loads(formatted)
        
        assert data["level"] == "ERROR"
        assert data["message"] == "Error occurred"
        assert "exception" in data
        assert "ValueError: Test exception" in data["exception"]


class TestPerformanceFilter:
    """Test the performance filter."""
    
    def test_performance_keywords_detected(self):
        """Test that performance-related messages are detected."""
        filter_obj = PerformanceFilter()
        
        # Messages that should pass the filter
        performance_messages = [
            "Performance: operation took 1.5s",
            "Timing analysis completed",
            "Memory usage: 512MB",
            "CPU optimization applied",
            "Benchmark results available",
            "Profiling data collected"
        ]
        
        for msg in performance_messages:
            record = logging.LogRecord(
                name="test",
                level=logging.INFO,
                pathname="/path",
                lineno=1,
                msg=msg,
                args=(),
                exc_info=None
            )
            assert filter_obj.filter(record) is True
    
    def test_non_performance_messages_filtered(self):
        """Test that non-performance messages are filtered out."""
        filter_obj = PerformanceFilter()
        
        # Messages that should not pass the filter
        regular_messages = [
            "Starting analysis",
            "Processing node data",
            "Graph construction complete",
            "Validation successful"
        ]
        
        for msg in regular_messages:
            record = logging.LogRecord(
                name="test",
                level=logging.INFO,
                pathname="/path",
                lineno=1,
                msg=msg,
                args=(),
                exc_info=None
            )
            assert filter_obj.filter(record) is False


class TestUtilityFunctions:
    """Test utility functions."""
    
    def test_get_logger(self):
        """Test get_logger function."""
        logger = get_logger("test.module")
        assert logger.name == "test.module"
        assert isinstance(logger, logging.Logger)
        
        # Same name should return same logger
        logger2 = get_logger("test.module")
        assert logger is logger2
    
    def test_configure_external_library_logging(self):
        """Test external library logging configuration."""
        # Test with defaults
        configure_external_library_logging()
        
        networkit_logger = logging.getLogger("networkit")
        assert networkit_logger.level == logging.WARNING
        
        # Test with custom configuration
        custom_config = {"test_library": "ERROR"}
        configure_external_library_logging(custom_config)
        
        test_logger = logging.getLogger("test_library")
        assert test_logger.level == logging.ERROR
    
    def test_log_function_entry(self):
        """Test function entry logging."""
        # Setup logger to capture debug messages
        logger = setup_logging(level="DEBUG", console=False, force_setup=True)
        
        with patch.object(logger, 'debug') as mock_debug:
            log_function_entry("test_function", param1="value1", param2=42)
            
            mock_debug.assert_called_once()
            call_args = mock_debug.call_args[0]
            assert "Entering test_function" in call_args[0]
            assert "param1=value1" in call_args[1]
            assert "param2=42" in call_args[1]
    
    def test_log_performance_metric(self):
        """Test performance metric logging."""
        # Setup performance logging
        setup_logging(performance_logging=True, console=False, force_setup=True)
        
        perf_logger = logging.getLogger("glp.performance")
        
        with patch.object(perf_logger, 'info') as mock_info:
            log_performance_metric(
                "test_operation", 
                1.234, 
                {"nodes": 1000, "edges": 5000}
            )
            
            mock_info.assert_called_once()
            call_args = mock_info.call_args
            
            # Check message
            message = call_args[0][0]
            assert "Performance: test_operation completed in 1.234s" in message
            assert "nodes=1000" in message
            assert "edges=5000" in message
            
            # Check extra fields
            extra = call_args[1]["extra"]
            assert extra["operation"] == "test_operation"
            assert extra["duration"] == 1.234
            assert extra["nodes"] == 1000
            assert extra["edges"] == 5000


class TestLoggingTimer:
    """Test the LoggingTimer context manager."""
    
    def test_timer_basic_usage(self):
        """Test basic timer usage."""
        setup_logging(performance_logging=True, console=False, force_setup=True)
        
        with patch('src.common.logging_config.log_performance_metric') as mock_log:
            with LoggingTimer("test_operation"):
                time.sleep(0.01)  # Small delay
            
            mock_log.assert_called_once()
            args = mock_log.call_args[0]
            assert args[0] == "test_operation"
            assert args[1] > 0  # Duration should be positive
            assert args[2] == {}  # No details
    
    def test_timer_with_details(self):
        """Test timer with additional details."""
        setup_logging(performance_logging=True, console=False, force_setup=True)
        
        details = {"nodes": 1000, "algorithm": "test"}
        
        with patch('src.common.logging_config.log_performance_metric') as mock_log:
            with LoggingTimer("detailed_operation", details):
                time.sleep(0.01)
            
            mock_log.assert_called_once()
            args = mock_log.call_args[0]
            assert args[0] == "detailed_operation"
            assert args[1] > 0
            assert args[2] == details
    
    def test_timer_with_exception(self):
        """Test that timer still logs even if exception occurs."""
        setup_logging(performance_logging=True, console=False, force_setup=True)
        
        with patch('src.common.logging_config.log_performance_metric') as mock_log:
            with pytest.raises(ValueError):
                with LoggingTimer("error_operation"):
                    raise ValueError("Test error")
            
            # Should still log the timing
            mock_log.assert_called_once()
            args = mock_log.call_args[0]
            assert args[0] == "error_operation"


class TestIntegrationScenarios:
    """Test realistic integration scenarios."""
    
    def test_complete_logging_setup(self):
        """Test a complete logging setup scenario."""
        with tempfile.TemporaryDirectory() as temp_dir:
            log_file = os.path.join(temp_dir, "integration_test.log")
            
            # Setup comprehensive logging
            logger = setup_logging(
                level="DEBUG",
                log_file=log_file,
                console=True,
                json_format=False,
                performance_logging=True,
                force_setup=True
            )
            
            # Configure external libraries
            configure_external_library_logging()
            
            # Test different types of logging
            app_logger = get_logger("glp.network")
            app_logger.info("Starting network analysis")
            app_logger.debug("Processing node %s", "node_123")
            
            # Test performance logging
            with LoggingTimer("network_construction", {"nodes": 1000}):
                time.sleep(0.01)
            
            # Test function entry logging
            log_function_entry("build_graph", source_col="source", target_col="target")
            
            # Verify log file was created and contains expected content
            assert os.path.exists(log_file)
            
            with open(log_file, 'r') as f:
                content = f.read()
                assert "Starting network analysis" in content
                assert "Processing node node_123" in content
                assert "Performance: network_construction" in content
                assert "Entering build_graph" in content


if __name__ == "__main__":
    # Run tests if script is executed directly
    pytest.main([__file__])