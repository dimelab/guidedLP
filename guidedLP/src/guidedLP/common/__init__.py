"""
Common utilities for the Guided Label Propagation library.

This module provides shared functionality used across all other modules:
- ID mapping between original and NetworkIt integer IDs
- Input validation for edge lists and parameters
- Export functions for various output formats
- Graph building utilities
- Custom exception hierarchy
"""

# Exception hierarchy - available for import throughout the library
from .exceptions import (
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

# Core utilities
from .id_mapper import IDMapper
from .validators import (
    validate_edgelist_dataframe,
    validate_timestamps,
    validate_seed_labels,
    validate_metadata_dataframe
)

# Logging configuration
from .logging_config import (
    setup_logging,
    get_logger,
    configure_external_library_logging,
    log_function_entry,
    log_performance_metric,
    LoggingTimer,
    JSONFormatter,
    PerformanceFilter
)

# Imports will be added as modules are implemented
# from .exporters import export_gexf, export_csv
# from .graph_builder import GraphBuilder