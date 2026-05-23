"""
Logging configuration for the Guided Label Propagation library.

This module provides centralized logging configuration with support for:
- Console and file output handlers
- Environment variable configuration
- Structured formatting with timestamps and context
- Performance-aware logging for large-scale operations
- Integration with the custom exception hierarchy
"""

import logging
import logging.handlers
import os
import sys
from pathlib import Path
from typing import Optional, Dict, Any, Union
from datetime import datetime
import json


# Default configuration constants
DEFAULT_LOG_LEVEL = "INFO"
DEFAULT_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s.%(funcName)s:%(lineno)d | %(message)s"
DEFAULT_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
DEFAULT_LOG_DIR = "logs"
DEFAULT_MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
DEFAULT_BACKUP_COUNT = 5

# Environment variable names
ENV_LOG_LEVEL = "GLP_LOG_LEVEL"
ENV_LOG_FILE = "GLP_LOG_FILE"
ENV_LOG_DIR = "GLP_LOG_DIR"
ENV_LOG_FORMAT = "GLP_LOG_FORMAT"
ENV_LOG_CONSOLE = "GLP_LOG_CONSOLE"
ENV_LOG_JSON = "GLP_LOG_JSON"
ENV_LOG_PERFORMANCE = "GLP_LOG_PERFORMANCE"


class PerformanceFilter(logging.Filter):
    """
    Filter for performance-related log messages.
    
    This filter can be used to separate performance metrics and timing
    information from general application logs, useful for benchmarking
    and optimization work.
    """
    
    def filter(self, record: logging.LogRecord) -> bool:
        """
        Filter records based on performance-related criteria.
        
        Parameters
        ----------
        record : logging.LogRecord
            The log record to filter
            
        Returns
        -------
        bool
            True if record should be logged, False otherwise
        """
        # Look for performance-related keywords in the message
        performance_keywords = [
            "performance", "timing", "duration", "elapsed", "benchmark",
            "memory", "cpu", "optimization", "profiling", "metrics"
        ]
        
        message = record.getMessage().lower()
        return any(keyword in message for keyword in performance_keywords)


class JSONFormatter(logging.Formatter):
    """
    JSON formatter for structured logging.
    
    This formatter outputs log records as JSON objects, making them
    suitable for log aggregation systems and automated analysis.
    """
    
    def format(self, record: logging.LogRecord) -> str:
        """
        Format the log record as JSON.
        
        Parameters
        ----------
        record : logging.LogRecord
            The log record to format
            
        Returns
        -------
        str
            JSON-formatted log message
        """
        log_obj = {
            "timestamp": datetime.fromtimestamp(record.created).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
            "message": record.getMessage(),
        }
        
        # Add exception information if present
        if record.exc_info:
            log_obj["exception"] = self.formatException(record.exc_info)
        
        # Add any extra fields from the log record
        for key, value in record.__dict__.items():
            if key not in {
                "name", "msg", "args", "levelname", "levelno", "pathname",
                "filename", "module", "lineno", "funcName", "created",
                "msecs", "relativeCreated", "thread", "threadName",
                "processName", "process", "getMessage", "exc_info", "exc_text",
                "stack_info", "message"
            }:
                log_obj[key] = value
        
        return json.dumps(log_obj)


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance with the given name.
    
    This function provides a consistent way to create loggers throughout
    the application. It ensures that all loggers use the configured
    formatting and handlers.
    
    Parameters
    ----------
    name : str
        The name for the logger (typically __name__)
        
    Returns
    -------
    logging.Logger
        Configured logger instance
        
    Examples
    --------
    >>> logger = get_logger(__name__)
    >>> logger.info("Starting analysis")
    >>> logger.debug("Processing node %s", node_id)
    
    Notes
    -----
    The logger will inherit configuration from the root GLP logger
    set up by setup_logging(). If setup_logging() hasn't been called,
    the logger will use Python's default configuration.
    """
    return logging.getLogger(name)


def setup_logging(
    level: Optional[str] = None,
    log_file: Optional[str] = None,
    log_dir: Optional[str] = None,
    console: Optional[bool] = None,
    json_format: Optional[bool] = None,
    performance_logging: Optional[bool] = None,
    format_string: Optional[str] = None,
    date_format: Optional[str] = None,
    max_file_size: Optional[int] = None,
    backup_count: Optional[int] = None,
    force_setup: bool = False
) -> logging.Logger:
    """
    Set up logging configuration for the GLP library.
    
    This function configures the root logger for the GLP library with
    appropriate handlers, formatters, and filters. It supports both
    console and file output, with optional JSON formatting for structured
    logging.
    
    Parameters
    ----------
    level : str, optional
        Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        If None, uses environment variable GLP_LOG_LEVEL or defaults to INFO.
    log_file : str, optional
        Path to log file. If None, uses environment variable GLP_LOG_FILE.
        If not specified and log_dir is provided, uses 'glp.log' in log_dir.
    log_dir : str, optional
        Directory for log files. If None, uses environment variable GLP_LOG_DIR
        or defaults to 'logs' directory.
    console : bool, optional
        Whether to enable console logging. If None, uses environment variable
        GLP_LOG_CONSOLE or defaults to True.
    json_format : bool, optional
        Whether to use JSON formatting. If None, uses environment variable
        GLP_LOG_JSON or defaults to False.
    performance_logging : bool, optional
        Whether to enable performance logging filter. If None, uses environment
        variable GLP_LOG_PERFORMANCE or defaults to False.
    format_string : str, optional
        Custom format string for log messages. If None, uses environment
        variable GLP_LOG_FORMAT or defaults to standard format.
    date_format : str, optional
        Date format for timestamps. Defaults to ISO-like format.
    max_file_size : int, optional
        Maximum size for log files before rotation (bytes). Defaults to 10MB.
    backup_count : int, optional
        Number of backup files to keep. Defaults to 5.
    force_setup : bool, default False
        Whether to force reconfiguration if logging is already set up.
        
    Returns
    -------
    logging.Logger
        The configured root logger for the GLP library
        
    Raises
    ------
    ValueError
        If invalid logging level is specified
    OSError
        If log directory cannot be created
        
    Examples
    --------
    >>> # Basic setup with defaults
    >>> logger = setup_logging()
    
    >>> # Setup with file logging
    >>> logger = setup_logging(level="DEBUG", log_file="analysis.log")
    
    >>> # Setup with JSON formatting for production
    >>> logger = setup_logging(
    ...     level="INFO",
    ...     log_dir="/var/log/glp",
    ...     json_format=True,
    ...     console=False
    ... )
    
    Environment Variables
    --------------------
    GLP_LOG_LEVEL : str
        Default logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    GLP_LOG_FILE : str
        Path to log file
    GLP_LOG_DIR : str
        Directory for log files
    GLP_LOG_FORMAT : str
        Custom format string for log messages
    GLP_LOG_CONSOLE : str
        Enable console logging (true/false, yes/no, 1/0)
    GLP_LOG_JSON : str
        Enable JSON formatting (true/false, yes/no, 1/0)
    GLP_LOG_PERFORMANCE : str
        Enable performance logging filter (true/false, yes/no, 1/0)
        
    Notes
    -----
    This function should be called once at the start of the application.
    Subsequent calls will not reconfigure logging unless force_setup=True.
    
    The function creates a logger hierarchy with "glp" as the root logger
    name, allowing fine-grained control over different components.
    """
    # Get root logger for GLP
    root_logger = logging.getLogger("glp")
    
    # Check if already configured (unless forced)
    if not force_setup and root_logger.handlers:
        return root_logger
    
    # Clear existing handlers if forcing setup
    if force_setup:
        root_logger.handlers.clear()
    
    # Resolve configuration from parameters and environment variables
    config = _resolve_logging_config(
        level=level,
        log_file=log_file,
        log_dir=log_dir,
        console=console,
        json_format=json_format,
        performance_logging=performance_logging,
        format_string=format_string,
        date_format=date_format,
        max_file_size=max_file_size,
        backup_count=backup_count
    )
    
    # Set logging level
    try:
        log_level = getattr(logging, config["level"].upper())
        root_logger.setLevel(log_level)
    except AttributeError:
        raise ValueError(f"Invalid logging level: {config['level']}")
    
    # Create formatter
    if config["json_format"]:
        formatter = JSONFormatter()
    else:
        formatter = logging.Formatter(
            fmt=config["format_string"],
            datefmt=config["date_format"]
        )
    
    # Add console handler if requested
    if config["console"]:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)
    
    # Add file handler if log file is specified
    if config["log_file"]:
        # Create log directory if it doesn't exist
        log_path = Path(config["log_file"])
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Use rotating file handler to manage file sizes
        file_handler = logging.handlers.RotatingFileHandler(
            filename=config["log_file"],
            maxBytes=config["max_file_size"],
            backupCount=config["backup_count"],
            encoding="utf-8"
        )
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
    
    # Add performance filter if requested
    if config["performance_logging"]:
        perf_filter = PerformanceFilter()
        
        # Create separate performance logger
        perf_logger = logging.getLogger("glp.performance")
        perf_logger.addFilter(perf_filter)
        
        # Optionally create separate performance log file
        if config["log_file"]:
            perf_log_file = str(log_path.parent / "performance.log")
            perf_handler = logging.handlers.RotatingFileHandler(
                filename=perf_log_file,
                maxBytes=config["max_file_size"],
                backupCount=config["backup_count"],
                encoding="utf-8"
            )
            perf_handler.setFormatter(formatter)
            perf_handler.addFilter(perf_filter)
            perf_logger.addHandler(perf_handler)
    
    # Prevent propagation to the root logger to avoid duplicate messages
    root_logger.propagate = False
    
    # Log successful configuration
    root_logger.info(
        "Logging configured: level=%s, console=%s, file=%s, json=%s",
        config["level"], config["console"], 
        config["log_file"] or "None", config["json_format"]
    )
    
    return root_logger


def _resolve_logging_config(**kwargs) -> Dict[str, Any]:
    """
    Resolve logging configuration from parameters and environment variables.
    
    Parameters take precedence over environment variables, which take
    precedence over defaults.
    
    Parameters
    ----------
    **kwargs
        Configuration parameters from setup_logging()
        
    Returns
    -------
    Dict[str, Any]
        Resolved configuration dictionary
    """
    def _get_bool_env(env_var: str, default: bool) -> bool:
        """Parse boolean from environment variable."""
        value = os.getenv(env_var, "").lower()
        if value in ("true", "yes", "1", "on"):
            return True
        elif value in ("false", "no", "0", "off"):
            return False
        else:
            return default
    
    # Resolve each configuration option
    level = kwargs.get("level") or os.getenv(ENV_LOG_LEVEL, DEFAULT_LOG_LEVEL)
    
    log_dir = kwargs.get("log_dir") or os.getenv(ENV_LOG_DIR, DEFAULT_LOG_DIR)
    log_file = kwargs.get("log_file") or os.getenv(ENV_LOG_FILE)
    
    # If no specific log file but log_dir is specified, use default name
    if not log_file and log_dir:
        log_file = os.path.join(log_dir, "glp.log")
    
    console = kwargs.get("console")
    if console is None:
        console = _get_bool_env(ENV_LOG_CONSOLE, True)
    
    json_format = kwargs.get("json_format")
    if json_format is None:
        json_format = _get_bool_env(ENV_LOG_JSON, False)
    
    performance_logging = kwargs.get("performance_logging")
    if performance_logging is None:
        performance_logging = _get_bool_env(ENV_LOG_PERFORMANCE, False)
    
    format_string = (
        kwargs.get("format_string") or 
        os.getenv(ENV_LOG_FORMAT, DEFAULT_LOG_FORMAT)
    )
    
    date_format = kwargs.get("date_format") or DEFAULT_DATE_FORMAT
    max_file_size = kwargs.get("max_file_size") or DEFAULT_MAX_FILE_SIZE
    backup_count = kwargs.get("backup_count") or DEFAULT_BACKUP_COUNT
    
    return {
        "level": level,
        "log_file": log_file,
        "console": console,
        "json_format": json_format,
        "performance_logging": performance_logging,
        "format_string": format_string,
        "date_format": date_format,
        "max_file_size": max_file_size,
        "backup_count": backup_count,
    }


def configure_external_library_logging(
    libraries: Optional[Dict[str, str]] = None,
    suppress_verbose: bool = True
) -> None:
    """
    Configure logging for external libraries used by GLP.
    
    This function helps manage the verbosity of external libraries
    like NetworkIt, Polars, and others that may produce excessive
    log output during normal operation.
    
    Parameters
    ----------
    libraries : Dict[str, str], optional
        Dictionary mapping library names to desired log levels.
        If None, uses sensible defaults for common libraries.
    suppress_verbose : bool, default True
        Whether to suppress verbose output from known chatty libraries
        
    Examples
    --------
    >>> # Use defaults
    >>> configure_external_library_logging()
    
    >>> # Custom configuration
    >>> configure_external_library_logging({
    ...     "networkit": "WARNING",
    ...     "urllib3": "ERROR"
    ... })
    """
    # Default library configurations
    default_config = {
        "networkit": "WARNING",    # NetworkIt can be verbose
        "urllib3": "WARNING",      # HTTP library used by various packages
        "requests": "WARNING",     # HTTP requests
        "matplotlib": "WARNING",   # Plotting library
        "numba": "WARNING",        # JIT compiler warnings
        "joblib": "WARNING",       # Parallel processing
    }
    
    # Additional libraries to suppress if requested
    if suppress_verbose:
        default_config.update({
            "PIL": "WARNING",          # Image processing
            "asyncio": "WARNING",      # Async operations
            "concurrent.futures": "WARNING",  # Threading
        })
    
    # Use provided config or defaults
    config = libraries or default_config
    
    # Configure each library
    for library_name, level in config.items():
        try:
            library_logger = logging.getLogger(library_name)
            library_level = getattr(logging, level.upper())
            library_logger.setLevel(library_level)
        except AttributeError:
            # Invalid level, skip this library
            continue


def log_function_entry(func_name: str, **kwargs) -> None:
    """
    Log function entry with parameters (for debugging).
    
    Parameters
    ----------
    func_name : str
        Name of the function being entered
    **kwargs
        Function parameters to log
        
    Examples
    --------
    >>> def my_function(param1, param2=None):
    ...     log_function_entry("my_function", param1=param1, param2=param2)
    ...     # function implementation
    """
    logger = get_logger("glp.debug")
    if logger.isEnabledFor(logging.DEBUG):
        param_str = ", ".join([f"{k}={v}" for k, v in kwargs.items()])
        logger.debug("Entering %s(%s)", func_name, param_str)


def log_performance_metric(
    operation: str,
    duration: float,
    details: Optional[Dict[str, Any]] = None
) -> None:
    """
    Log performance metrics for operations.
    
    Parameters
    ----------
    operation : str
        Name of the operation that was timed
    duration : float
        Duration in seconds
    details : Dict[str, Any], optional
        Additional details about the operation (node count, etc.)
        
    Examples
    --------
    >>> import time
    >>> start = time.time()
    >>> # ... do work ...
    >>> duration = time.time() - start
    >>> log_performance_metric("graph_construction", duration, 
    ...                       {"nodes": 1000, "edges": 5000})
    """
    logger = get_logger("glp.performance")
    
    message = f"Performance: {operation} completed in {duration:.3f}s"
    
    if details:
        detail_str = ", ".join([f"{k}={v}" for k, v in details.items()])
        message += f" ({detail_str})"
    
    logger.info(message, extra={"operation": operation, "duration": duration, **(details or {})})


# Context manager for timing operations
class LoggingTimer:
    """
    Context manager for timing operations with automatic logging.
    
    Examples
    --------
    >>> with LoggingTimer("expensive_operation", {"nodes": 1000}):
    ...     # expensive operation here
    ...     pass
    """
    
    def __init__(self, operation: str, details: Optional[Dict[str, Any]] = None):
        self.operation = operation
        self.details = details or {}
        self.start_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.start_time is not None:
            duration = time.time() - self.start_time
            log_performance_metric(self.operation, duration, self.details)


# Import time for LoggingTimer
import time