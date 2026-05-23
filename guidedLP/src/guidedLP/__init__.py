"""
Guided Label Propagation (GLP) - Large-scale network analysis library.

This package provides efficient network analysis capabilities with a focus on
Guided Label Propagation, a semi-supervised community detection technique for
computational social science research.

Modules:
    common: Shared utilities for ID mapping, validation, and export
    network: Graph construction and network analysis tools  
    glp: Guided Label Propagation implementation
    timeseries: Temporal network analysis capabilities
"""

__version__ = "0.1.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"

# Expose main functionality at package level
# Users can import as: from guidedLP import guided_label_propagation, build_graph_from_edgelist

# Core classes and functions
try:
    from common.id_mapper import IDMapper
    from common.exceptions import (
        GraphConstructionError, 
        ValidationError, 
        DataFormatError,
        ConfigurationError
    )
    __all__ = ["IDMapper", "GraphConstructionError", "ValidationError", "DataFormatError", "ConfigurationError"]
except ImportError:
    # Fallback if relative imports don't work
    __all__ = []

# Note: Main functions like guided_label_propagation and build_graph_from_edgelist
# should be imported directly from their modules:
# from glp.propagation import guided_label_propagation
# from network.construction import build_graph_from_edgelist, temporal_bipartite_to_unipartite