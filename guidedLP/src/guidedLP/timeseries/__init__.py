"""
Time-series network analysis module.

This module provides temporal network analysis capabilities:
- Time-sliced network construction (daily, weekly, monthly, yearly)
- Rolling window analysis
- Cumulative vs. non-cumulative graph construction
- Node ID alignment across temporal slices
- Temporal metric tracking and aggregation
- Cross-category connection analysis over time
"""

# Temporal slicing functions
from .slicing import (
    create_temporal_slices,
    align_node_ids_across_slices
)

# Temporal metrics functions
from .temporal_metrics import (
    extract_temporal_metrics,
    calculate_temporal_statistics
)

# Category analysis functions
from .category_analysis import (
    analyze_cross_category_connections,
    calculate_category_segregation_index,
    analyze_category_centrality_by_time
)

# Future modules will be added as they are implemented  
# from .aggregation import TemporalAggregator