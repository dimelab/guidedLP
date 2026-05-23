# Guided Label Propagation - Complete API Reference

## Table of Contents

- [Overview](#overview)
- [Quick Start](#quick-start)  
- [Core Modules](#core-modules)
- [Function Index](#function-index)
- [Examples by Use Case](#examples-by-use-case)
- [Performance Guide](#performance-guide)

## Overview

The Guided Label Propagation library provides a comprehensive toolkit for semi-supervised network analysis. The API is organized into four main modules:

| Module | Purpose | Key Functions |
|--------|---------|---------------|
| **[network](network.md)** | Graph construction and analysis | `build_graph_from_edgelist()`, `project_bipartite()`, `extract_centrality()` |
| **[glp](glp.md)** | Label propagation algorithms | `guided_label_propagation()`, `train_test_split_validation()` |
| **[timeseries](timeseries.md)** | Temporal network analysis | `create_temporal_slices()`, `calculate_temporal_metrics()` |
| **[common](common.md)** | Utilities and shared functions | `IDMapper`, validation, I/O functions |

## Quick Start

```python
# Complete workflow in 5 steps
import polars as pl
from src.network.construction import build_graph_from_edgelist
from src.glp.propagation import guided_label_propagation

# 1. Load data
edges = pl.read_csv("network.csv")
seeds = {"node1": "community_a", "node2": "community_b"}

# 2. Build graph  
graph, mapper = build_graph_from_edgelist(edges, "source", "target")

# 3. Run GLP
results = guided_label_propagation(graph, mapper, seeds, ["community_a", "community_b"])

# 4. Analyze results
for node_id, probabilities in results.items():
    predicted = max(probabilities, key=probabilities.get)
    confidence = max(probabilities.values())
    print(f"{node_id}: {predicted} ({confidence:.3f})")
```

## Core Modules

### 🏗️ Network Construction and Analysis

**Primary Module**: [`src.network`](network.md)

- **Graph Building**: Convert edge lists to NetworkIt graphs with ID preservation
- **Graph Analysis**: Calculate centrality measures and network properties  
- **Bipartite Operations**: Project bipartite graphs to unipartite networks
- **Community Detection**: Find communities using various algorithms

**Key Functions**:
```python
# Graph construction
graph, mapper = build_graph_from_edgelist(edges, "source", "target", "weight")

# Centrality analysis
centrality = extract_centrality(graph, mapper, ["degree", "betweenness", "pagerank"])

# Bipartite projection
user_graph, user_mapper = project_bipartite(bipartite_graph, mapper, "source")
```

### 🎯 Guided Label Propagation

**Primary Module**: [`src.glp`](glp.md)

- **Core Algorithm**: Semi-supervised node classification via label propagation
- **Validation**: Train/test splits, cross-validation, performance metrics
- **Evaluation**: Confidence scoring, prediction analysis, diagnostic tools

**Key Functions**:
```python
# Core GLP algorithm
results = guided_label_propagation(graph, mapper, seed_labels, labels, alpha=0.85)

# Performance validation
accuracy, metrics = train_test_split_validation(graph, mapper, known_labels, labels)

# Cross-validation
cv_results = cross_validate(graph, mapper, known_labels, labels, cv_folds=5)
```

### ⏰ Temporal Network Analysis

**Primary Module**: [`src.timeseries`](timeseries.md)

- **Time Slicing**: Create temporal network snapshots from timestamped data
- **Evolution Tracking**: Monitor how network properties change over time
- **Dynamic Communities**: Track community stability and evolution

**Key Functions**:
```python
# Create time slices
daily_slices = create_temporal_slices(temporal_data, slice_duration="1d")

# Track metrics over time
temporal_metrics = calculate_temporal_metrics(daily_graphs, ["degree", "betweenness"])

# Network evolution
evolution = track_network_evolution(daily_graphs)
```

### 🔧 Utilities and Common Functions  

**Primary Module**: [`src.common`](common.md)

- **ID Mapping**: Bidirectional conversion between original and internal IDs
- **Validation**: Data quality checks and parameter validation
- **I/O Operations**: Export results, load data, configuration management

**Key Functions**:
```python
# ID mapping
mapper = IDMapper()
mapper.add_mapping("user_alice", 0)

# Data validation
validate_edgelist_dataframe(edges, "source", "target", "weight")

# Export results
export_results(glp_results, "classifications.csv")
```

## Function Index

### Graph Operations
- [`build_graph_from_edgelist()`](network.md#build_graph_from_edgelist) - Create NetworkIt graphs from edge lists
- [`project_bipartite()`](network.md#project_bipartite) - Project bipartite graphs to unipartite
- [`get_graph_info()`](network.md#get_graph_info) - Get comprehensive graph statistics
- [`validate_graph_construction()`](network.md#validate_graph_construction) - Validate graph construction

### Centrality and Analysis
- [`extract_centrality()`](network.md#extract_centrality) - Calculate multiple centrality measures
- [`detect_communities()`](network.md#detect_communities) - Community detection algorithms
- [`export_graph()`](network.md#export_graph) - Export graphs to various formats

### Label Propagation
- [`guided_label_propagation()`](glp.md#guided_label_propagation) - Core GLP algorithm
- [`train_test_split_validation()`](glp.md#train_test_split_validation) - Validate with train/test split
- [`cross_validate()`](glp.md#cross_validate) - K-fold cross-validation
- [`calculate_prediction_confidence()`](glp.md#calculate_prediction_confidence) - Assess prediction confidence

### Temporal Analysis
- [`create_temporal_slices()`](timeseries.md#create_temporal_slices) - Time-slice networks
- [`create_temporal_graphs()`](timeseries.md#create_temporal_graphs) - Create graph snapshots
- [`calculate_temporal_metrics()`](timeseries.md#calculate_temporal_metrics) - Track metrics over time
- [`track_network_evolution()`](timeseries.md#track_network_evolution) - Monitor network changes

### Utilities
- [`IDMapper`](common.md#idmapper) - ID mapping between formats
- [`validate_edgelist_dataframe()`](common.md#validate_edgelist_dataframe) - Validate edge list data
- [`export_results()`](common.md#export_results) - Export analysis results

## Examples by Use Case

### 🗳️ Political Affiliation Analysis

Classify unknown users' political leanings based on known partisan accounts:

```python
# Load social network data
edges = pl.read_csv("twitter_network.csv")
graph, mapper = build_graph_from_edgelist(edges, "follower", "following")

# Define known political accounts
political_seeds = {
    "@realdonaldtrump": "conservative",
    "@aoc": "progressive", 
    "@berniesanders": "progressive",
    "@tedcruz": "conservative"
}

# Run validation first
accuracy, metrics = train_test_split_validation(
    graph, mapper, political_seeds, ["conservative", "progressive"],
    test_size=0.3, random_seed=42
)
print(f"Validation accuracy: {accuracy:.3f}")

# Classify all users
political_affiliations = guided_label_propagation(
    graph, mapper, political_seeds, ["conservative", "progressive"],
    alpha=0.85, max_iterations=100
)

# Analyze high-confidence predictions
confidence = calculate_prediction_confidence(political_affiliations)
high_conf = {k: v for k, v in confidence.items() if v > 0.8}
print(f"High-confidence classifications: {len(high_conf)}")
```

### 🛒 Brand Affinity Detection

Identify brand preferences in consumer networks:

```python
# User-product interaction network
interactions = pl.read_csv("user_product_ratings.csv")

# Build bipartite graph
bipartite_graph, mapper = build_graph_from_edgelist(
    interactions, "user_id", "product_id", "rating", bipartite=True
)

# Project to user similarity network
user_graph, user_mapper = project_bipartite(
    bipartite_graph, mapper, projection_mode="source", weight_method="jaccard"
)

# Define brand seed users (verified accounts, brand ambassadors)
brand_seeds = {
    "verified_apple_fan": "apple",
    "samsung_ambassador": "samsung",
    "google_employee": "google"
}

# Detect brand affinity
brand_affinity = guided_label_propagation(
    user_graph, user_mapper, brand_seeds, ["apple", "samsung", "google"],
    alpha=0.9  # Higher alpha for stronger brand loyalty propagation
)
```

### 📊 Temporal Community Evolution

Track how communities evolve in dynamic networks:

```python
# Load temporal interaction data
temporal_data = pl.read_csv("temporal_collaborations.csv")

# Create daily network snapshots
daily_graphs = create_temporal_graphs(
    temporal_data, timestamp_col="interaction_date", slice_duration="1d"
)

# Track centrality of key individuals over time
key_people = ["leader_1", "leader_2", "influencer_3"]
temporal_metrics = calculate_temporal_metrics(
    daily_graphs, 
    metrics=["degree", "betweenness", "pagerank"],
    node_subset=key_people
)

# Detect community evolution
temporal_communities = detect_temporal_communities(daily_graphs, track_stability=True)

# Analyze network growth
evolution = track_network_evolution(daily_graphs)
print(f"Network grew from {evolution['nodes'].min()} to {evolution['nodes'].max()} nodes")
```

### 🎓 Academic Collaboration Analysis

Map research communities in citation networks:

```python
# Load co-authorship network
collaborations = pl.read_csv("academic_collaborations.csv")
graph, mapper = build_graph_from_edgelist(
    collaborations, "author_a", "author_b", "num_papers"
)

# Define department seed nodes
department_seeds = {
    "prof_cs_1": "computer_science",
    "prof_bio_1": "biology",
    "prof_phys_1": "physics"
}

# Classify researchers by field
research_fields = guided_label_propagation(
    graph, mapper, department_seeds, 
    ["computer_science", "biology", "physics"],
    alpha=0.7  # Lower alpha to limit propagation distance
)

# Identify interdisciplinary researchers (low confidence predictions)
confidence = calculate_prediction_confidence(research_fields)
interdisciplinary = {
    node: conf for node, conf in confidence.items() 
    if conf < 0.6 and node not in department_seeds
}

print(f"Found {len(interdisciplinary)} interdisciplinary researchers")
```

## Performance Guide

### 📈 Scalability Guidelines

| Network Size | Recommended Approach | Memory Usage | Time Complexity |
|--------------|---------------------|--------------|-----------------|
| < 1K nodes | Standard workflow | < 100MB | Seconds |
| 1K - 10K nodes | Optimize parameters | 100MB - 1GB | Minutes |
| 10K - 100K nodes | Use chunking/parallel | 1GB - 10GB | Hours |
| > 100K nodes | Distributed processing | > 10GB | Days |

### ⚡ Optimization Strategies

**For Large Networks**:
```python
# Use optimized parameters
large_results = guided_label_propagation(
    large_graph, mapper, seeds, labels,
    alpha=0.9,           # Higher alpha for faster convergence
    max_iterations=50,   # Fewer iterations
    convergence_threshold=1e-4,  # Less strict convergence
    normalize=False      # Skip normalization for speed
)
```

**Memory-Efficient Temporal Analysis**:
```python
# Process temporal data in batches
batch_size = 100  # Process 100 time slices at once
temporal_slices = create_temporal_slices(large_temporal_data, "1d")

for i in range(0, len(temporal_slices), batch_size):
    batch_slices = dict(list(temporal_slices.items())[i:i+batch_size])
    batch_graphs = create_temporal_graphs(batch_slices)
    batch_metrics = calculate_temporal_metrics(batch_graphs)
    batch_metrics.write_csv(f"metrics_batch_{i}.csv")
```

**Parallel Processing**:
```python
# Use parallel centrality calculation
centrality = extract_centrality(
    graph, mapper, 
    metrics=["degree", "betweenness", "closeness"],
    n_jobs=-1  # Use all CPU cores
)
```

## Error Handling

The library provides comprehensive error handling with specific exception types:

```python
from src.common.exceptions import (
    ValidationError, GraphConstructionError, ConvergenceError
)

try:
    graph, mapper = build_graph_from_edgelist(edges)
    results = guided_label_propagation(graph, mapper, seeds, labels)
except ValidationError as e:
    print(f"Data validation failed: {e}")
except GraphConstructionError as e:
    print(f"Could not build graph: {e}")
except ConvergenceError as e:
    print(f"Algorithm did not converge: {e}")
```

## Getting Help

- **Examples**: See individual module documentation for detailed examples
- **Performance Issues**: Check the [Performance Guide](#performance-guide) 
- **API Questions**: Refer to function docstrings for parameter details
- **Bug Reports**: Use the GitHub issues tracker
- **Feature Requests**: Use GitHub discussions

## Version Information

This documentation covers version 0.1.0 of the Guided Label Propagation library.

**API Stability**: 
- ✅ Core functions (`build_graph_from_edgelist`, `guided_label_propagation`) are stable
- ⚠️ Advanced features and utilities may change in future versions
- 📋 Deprecated functions will be marked with warnings before removal