# API Reference

This directory contains comprehensive API documentation for all modules in the Guided Label Propagation library.

## Module Overview

### Core Modules

- **[network/](network.md)** - Graph construction, analysis, and manipulation
- **[glp/](glp.md)** - Guided Label Propagation algorithms and validation
- **[timeseries/](timeseries.md)** - Temporal network analysis and slicing
- **[common/](common.md)** - Shared utilities and helper functions

## Quick Navigation

### Graph Construction
- [`build_graph_from_edgelist()`](network.md#build_graph_from_edgelist) - Create graphs from edge lists
- [`project_bipartite()`](network.md#project_bipartite) - Bipartite graph projections

### Label Propagation
- [`guided_label_propagation()`](glp.md#guided_label_propagation) - Core GLP algorithm
- [`train_test_split_validation()`](glp.md#train_test_split_validation) - Model validation

### Temporal Analysis
- [`create_temporal_slices()`](timeseries.md#create_temporal_slices) - Time-sliced networks
- [`calculate_temporal_metrics()`](timeseries.md#calculate_temporal_metrics) - Evolution tracking

### Utilities
- [`IDMapper`](common.md#idmapper) - ID mapping between formats
- [`export_results()`](common.md#export_results) - Data export utilities

## Code Examples

See individual module documentation for comprehensive examples. Quick start:

```python
import polars as pl
from src.network.construction import build_graph_from_edgelist
from src.glp.propagation import guided_label_propagation

# Load data
edges = pl.read_csv("network.csv")
graph, mapper = build_graph_from_edgelist(edges, "source", "target")

# Run GLP
seeds = {"community_a": ["node1", "node2"], "community_b": ["node3"]}
results = guided_label_propagation(graph, mapper, seeds)
```

## Documentation Conventions

- **Parameters**: All parameters documented with types and descriptions
- **Returns**: Return values explained with types and structure  
- **Raises**: All possible exceptions listed with conditions
- **Examples**: Practical usage examples for each function
- **Notes**: Performance characteristics and implementation details