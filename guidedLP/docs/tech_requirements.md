# Technical Requirements

## Performance Requirements

### Scalability Targets
- Must efficiently handle networks with **10,000+ nodes** and **100,000+ edges**
- Should gracefully scale to millions of edges
- All operations should be benchmarked for computational complexity

### Optimization Priorities
1. **Speed** (Primary): Minimize computation time for all operations
2. **Scalability**: Performance should scale efficiently with network size
3. **Memory** (Secondary): Provide memory-efficient alternatives when requested

### Multi-Processing
- Implement parallel processing for operations that benefit from it
- Use Python's `multiprocessing` module for CPU-bound tasks
- Document which operations are parallelized and expected speedup

## Required Python Frameworks

### Core Dependencies
1. **NetworkIt** (Primary network library)
   - Use for all graph construction and analysis
   - Leverage built-in algorithms (Louvain, centrality metrics, etc.)
   - Exploit C++ backend for performance

2. **Polars** (DataFrame operations)
   - All tabular data manipulation must use Polars, NOT Pandas
   - Leverage lazy evaluation where applicable
   - Use Polars for reading CSV files

3. **NumPy** (Numerical operations)
   - Matrix operations for GLP calculations
   - Numerical computations and array operations
   - Interface between Polars and NetworkIt

### Additional Libraries
- **SciPy**: For sparse matrix operations (if needed for GLP)
- **pytest**: Unit testing framework
- **typing**: Full type hint support

## Graph Specifications

### Supported Graph Types
✅ **Supported:**
- Unipartite graphs
- Bipartite graphs (with projection to unipartite)
- Directed graphs
- Undirected graphs
- Weighted edges

❌ **NOT Supported:**
- Multi-edge graphs (multiple edges between same nodes)
- Multi-layer graphs
- Hypergraphs

### Node ID Requirements
- **CRITICAL**: Original node IDs from input data must be preserved
- Use ID mapping between original IDs and NetworkIt's internal node IDs
- All exports must include original IDs
- Metadata joins should use original IDs as keys

### Edge Weight Handling
- Support custom edge weights from input data (third column in CSV)
- If no weights provided: calculate from duplicate source-target pairs
- Weight calculation: `weight = count(source, target)`

## Input Data Format

### Primary Input: Edge List
**Required columns:**
- `source`: Source node ID (any hashable type)
- `target`: Target node ID (any hashable type)

**Optional columns:**
- `weight`: Edge weight (numeric)
- `timestamp`: DateTime for time-series analysis (datetime format)

**Format support:**
- CSV files (primary)
- Polars DataFrame (programmatic input)

### Metadata Input
- Separate file/DataFrame with node attributes
- Must include ID column matching original edge list IDs
- Used for filtering, categorization, and exports

## Code Architecture Guidelines

### Design Approach
- **Flexibility**: Use OOP, functional, or hybrid approach based on efficiency
- **Modularity**: Three independent modules (network, GLP, time-series)
- **Reusability**: Common utilities should be shared across modules

### Module Independence
Each module should function independently:
1. **Network Module**: Graph construction and basic analysis
2. **GLP Module**: Guided label propagation (depends on network module)
3. **Time-Series Module**: Temporal network analysis (depends on network module)

### Interface Design
- **Current Phase**: Backend API (Python functions/classes)
- **Future**: Design interfaces with potential REST API in mind
- **Testing**: Jupyter notebooks or `test.py` scripts for now

## Performance Benchmarking

### Required Metrics
Document for each major function:
- **Time Complexity**: Big O notation
- **Space Complexity**: Memory usage pattern
- **Benchmark Results**: Performance on standard network sizes (1K, 10K, 100K, 1M nodes)

### Test Networks
Maintain standard test networks of various sizes for benchmarking:
- Small: 1,000 nodes, ~5,000 edges
- Medium: 10,000 nodes, ~50,000 edges  
- Large: 100,000 nodes, ~500,000 edges
- Extra Large: 1,000,000 nodes, ~5,000,000 edges

## Export Formats

### Graph Exports
- **GEXF** (primary): For Gephi visualization
- **GraphML**: Alternative XML format
- **CSV/Parquet**: Node and edge lists with metadata

### Data Exports
- **Parquet** (preferred): Efficient columnar storage
- **CSV**: For human readability and external tools
- **JSON**: For configuration and metadata

## Version Control
- Use Git for version control
- Semantic versioning for releases
- Tag performance benchmarks with Git commits
