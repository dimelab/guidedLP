"""
Guided Label Propagation (GLP) module.

This module implements the core GLP algorithm for semi-supervised community detection:
- Label probability calculation from seed nodes
- Directional propagation (in-degree and out-degree based)
- Matrix-based efficient propagation using sparse matrices
- Train/test split evaluation framework
- External validation set testing
- Utility functions for seed balancing and parameter tuning
- Result analysis and directional comparison tools
"""

# Core propagation functions
from .propagation import (
    guided_label_propagation,
    get_propagation_info
)

# Validation functions
from .validation import (
    train_test_split_validation,
    external_validation,
    cross_validate,
    get_validation_summary
)

# Utility functions
from .utils import (
    create_balanced_seed_set,
    suggest_alpha_value,
    get_seed_statistics
)

# Evaluation functions
from .evaluation import (
    analyze_label_distribution,
    compare_directional_results
)