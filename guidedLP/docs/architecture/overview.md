# System Architecture Overview

## Project Purpose
This system provides large-scale network analysis capabilities for social science research, with a focus on computational efficiency and a novel semi-supervised community detection technique called Guided Label Propagation (GLP).

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────┐
│                     Input Data Layer                     │
│  (CSV/Parquet edge lists, metadata, timestamps)          │
└────────────────┬────────────────────────────────────────┘
                 │
┌────────────────▼────────────────────────────────────────┐
│              Data Processing Layer                       │
│  (Polars DataFrames, validation, ID mapping)            │
└────────────┬───────────┬────────────┬───────────────────┘
             │           │            │
     ┌───────▼──┐   ┌───▼─────┐   ┌─▼──────────────┐
     │ Network  │   │   GLP   │   │  Time-Series   │
     │ Module   │   │ Module  │   │    Module      │
     └─────┬────┘   └───┬─────┘   └──┬─────────────┘
           │            │             │
           └────────────┼─────────────┘
                        │
          ┌─────────────▼────────────────┐
          │    Export & Analysis Layer   │
          │  (GEXF, CSV, metrics, viz)   │
          └──────────────────────────────┘
```

## Core Modules

### 1. Network Construction & Analysis Module
**Purpose**: Build and analyze large-scale networks from edge lists

**Key Capabilities**:
- Graph construction (unipartite, bipartite)
- Bipartite projection
- Network backboning
- Centrality metrics extraction
- Community detection (Louvain-based)
- Filtering and sampling
- Graph export

**Dependencies**: NetworkIt, Polars, NumPy

### 2. Guided Label Propagation (GLP) Module
**Purpose**: Semi-supervised community detection based on seed nodes

**Key Capabilities**:
- Label probability calculation from seed nodes
- Directional propagation (in-degree, out-degree)
- Matrix-based efficient propagation
- Train/test split evaluation
- External validation set testing

**Dependencies**: Network Module, NumPy, SciPy (for sparse matrices)

**Innovation**: Unlike unsupervised community detection (which finds arbitrary clusters), GLP identifies affinity toward predefined categories of interest (e.g., political left/right, brand affinity, topic relevance).

### 3. Time-Series Network Module
**Purpose**: Analyze network evolution over time

**Key Capabilities**:
- Time-sliced network construction (daily, weekly, monthly, yearly)
- Rolling window analysis
- Cumulative vs. non-cumulative graphs
- Temporal metric tracking
- Cross-category connection analysis

**Dependencies**: Network Module, Polars (for datetime handling)

## Module Independence & Interoperability

### Independence Principle
Each module can be used standalone:
- **Network Module**: Can be used without GLP or time-series
- **GLP Module**: Requires network module but not time-series
- **Time-Series Module**: Requires network module but not GLP

### Shared Components
```
src/
├── common/
│   ├── graph_builder.py      # Shared graph construction
│   ├── id_mapper.py           # Original ID ↔ NetworkIt ID mapping
│   ├── validators.py          # Input validation
│   └── exporters.py           # Common export functions
├── network/
│   ├── construction.py
│   ├── analysis.py
│   ├── backboning.py
│   └── communities.py
├── glp/
│   ├── propagation.py
│   ├── validation.py
│   └── evaluation.py
└── timeseries/
    ├── slicing.py
    ├── aggregation.py
    └── temporal_metrics.py
```

## Data Flow Patterns

### Pattern 1: Basic Network Analysis
```
Edge List CSV → Load & Validate → Build Graph → Analyze → Export Results
```

### Pattern 2: Guided Label Propagation
```
Edge List → Build Graph → Define Seeds → Propagate Labels → 
→ Calculate Probabilities → Validate → Export Affiliations
```

### Pattern 3: Time-Series Analysis
```
Edge List + Timestamps → Time Slice → Build Graphs per Slice → 
→ Extract Metrics → Aggregate Temporal Trends → Export
```

## ID Mapping Strategy

**Critical Requirement**: NetworkIt uses integer node IDs (0, 1, 2, ...), but input data uses arbitrary IDs (usernames, UUIDs, etc.).

**Solution**:
```python
class IDMapper:
    original_to_internal: dict  # "user123" → 0
    internal_to_original: dict  # 0 → "user123"
```

**Usage**:
- Input processing: Map original IDs → internal IDs
- Graph operations: Use internal IDs (NetworkIt compatible)
- Output/Export: Map back to original IDs
- Metadata joins: Use original IDs

## Performance Strategy

### Computational Bottlenecks
1. **Graph construction**: Large edge lists → NetworkIt graph
2. **Matrix operations**: Label propagation calculations
3. **Community detection**: Iterative optimization algorithms
4. **Time-series**: Multiple graph constructions

### Optimization Approaches
1. **Use NetworkIt's C++ backend** for graph operations
2. **Vectorize with NumPy** for numerical computations
3. **Parallel processing** for independent operations (time slices, iterations)
4. **Lazy evaluation with Polars** for data preprocessing
5. **Sparse matrices** for GLP calculations on large networks

## Extension Points

### Future Enhancements
- **Frontend Interface**: Web-based visualization and interaction
- **Streaming**: Real-time network updates
- **Additional Algorithms**: More community detection methods
- **GPU Acceleration**: For matrix operations in GLP
- **Distributed Computing**: For extremely large networks

### API Design Considerations
Current backend should be designed with REST API in mind:
- Clear function signatures (inputs/outputs)
- JSON-serializable return types (where appropriate)
- Stateless operations (no hidden global state)
- Error handling with descriptive messages

## Testing Strategy

### Unit Tests
- Individual functions in each module
- Edge cases and error conditions
- ID mapping correctness

### Integration Tests
- Full pipeline workflows
- Module interactions
- End-to-end scenarios

### Performance Tests
- Benchmark on standard network sizes
- Compare against baseline implementations
- Monitor memory usage and CPU utilization

### Validation Tests
- GLP accuracy on labeled test data
- Community detection quality metrics (modularity, etc.)
- Time-series consistency checks
