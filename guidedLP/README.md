# Guided Label Propagation (GLP)

Large-scale network analysis with semi-supervised community detection for computational social science research.

## Overview

This project provides efficient network analysis capabilities with a focus on **Guided Label Propagation (GLP)**, a novel semi-supervised community detection technique. Unlike traditional unsupervised methods that find arbitrary clusters, GLP identifies how unknown nodes in a network relate to predefined categories of interest (e.g., political affiliation, brand preference, topic relevance).

## Key Features

### 🚀 High-Performance Network Analysis
- **Large-scale optimization**: Designed for networks with 10,000+ nodes
- **NetworkIt backend**: Leverages C++ performance for graph operations  
- **Sparse matrix operations**: Memory-efficient computations using SciPy
- **Parallel processing**: Multi-threaded operations where beneficial

### 🎯 Guided Label Propagation (GLP)
- **Semi-supervised approach**: Uses seed nodes to guide community detection
- **Directional propagation**: Supports both in-degree and out-degree based propagation
- **Probability estimation**: Calculates affinity scores for unknown nodes
- **Validation framework**: Built-in train/test split and external validation

### 📊 Comprehensive Network Toolkit
- **Graph construction**: Unipartite and bipartite networks from edge lists
- **Temporal bipartite projection**: Convert temporal bipartite to directed unipartite with causality preservation
- **Network backboning**: Statistical significance filtering
- **Centrality measures**: Degree, betweenness, closeness, eigenvector centrality
- **Community detection**: Louvain algorithm integration
- **Temporal analysis**: Time-sliced network evolution

### 🔄 Flexible Data Pipeline
- **Polars integration**: Fast DataFrame operations for large datasets
- **Multiple formats**: Support for CSV, Parquet input/output
- **ID preservation**: Maintains original node identifiers throughout analysis
- **Export options**: GEXF, GraphML, CSV outputs

## Installation

### Prerequisites

- Python 3.9 or higher
- Git (for development installation)

### Installation

Install the package directly with pip:

```bash
# Clone the repository
git clone https://github.com/alterpublics/guided-label-propagation.git
cd guided-label-propagation/guidedLP

# Install the package in development mode
pip install -e .

# Or install dependencies separately if needed
pip install -r requirements.txt
```

### Development Setup

For development work:

```bash
# Install development dependencies  
pip install pytest pytest-cov ruff black mypy
```

### Verify Installation

After installation, you can verify everything works correctly:

```bash
python test_installation.py
```

This will test all key functionality and confirm your installation is working properly.

# Or install with all optional dependencies
pip install -e ".[dev,docs,viz]"
```

### Optional Dependencies

```bash
# For visualization capabilities
pip install "guided-label-propagation[viz]"

# For development and testing
pip install "guided-label-propagation[dev]"

# For documentation building
pip install "guided-label-propagation[docs]"
```

### Verify Installation

```bash
python -c "import guided_lp; print('Installation successful!')"
```

## Quick Start

### Basic Example

```python
import polars as pl
from guidedLP.network.construction import build_graph_from_edgelist
from guidedLP.glp.propagation import guided_label_propagation

# Load edge list data
edges = pl.read_csv("network_data.csv")

# Build network
graph, id_mapper = build_graph_from_edgelist(
    edges, 
    source_col="user_a", 
    target_col="user_b",
    weight_col="weight"  # optional
)

# Define seed nodes for each community
seed_nodes = {
    "progressive": ["user123", "user456", "user789"],
    "conservative": ["user321", "user654", "user987"]
}

# Run Guided Label Propagation
results = guided_label_propagation(
    graph=graph,
    seeds=seed_nodes,
    id_mapper=id_mapper,
    max_iterations=100,
    threshold=0.01
)

# Export results
export_results(results, "political_affiliation_scores.csv")
print(f"Classified {len(results)} nodes with community probabilities")
```

### Using Test Fixtures

Try the library with sample data:

```python
import polars as pl
import json
from guidedLP.network.construction import build_graph_from_edgelist
from guidedLP.glp.propagation import guided_label_propagation

# Load sample datasets
edges = pl.read_csv("tests/fixtures/sample_edgelist.csv")
with open("tests/fixtures/sample_seeds.json", "r") as f:
    seeds = json.load(f)

# Convert seeds to proper format
seed_nodes = {}
for node, community in seeds.items():
    if community not in seed_nodes:
        seed_nodes[community] = []
    seed_nodes[community].append(node)

# Build graph and run GLP
graph, id_mapper = build_graph_from_edgelist(edges, "source", "target", "weight")
results = guided_label_propagation(graph, seed_nodes, id_mapper)

print(f"Sample analysis complete: {len(results)} nodes classified")
```

## Architecture

The system is organized into three main modules:

```
src/
├── common/          # Shared utilities (ID mapping, validation, export)
├── network/         # Graph construction and analysis
├── glp/            # Guided Label Propagation implementation  
└── timeseries/     # Temporal network analysis
```

### Module Independence
- **Network module**: Standalone graph analysis capabilities
- **GLP module**: Requires network module, adds semi-supervised detection
- **Time-series module**: Temporal analysis, can work with or without GLP

## Performance Characteristics

- **Graph construction**: O(E + V) using NetworkIt
- **Label propagation**: O(I × E) where I is iterations, E is edges
- **Memory usage**: Sparse matrices for networks with >50% zero entries
- **Parallel support**: Multi-threaded centrality calculations and time-slicing

## Examples

### 1. Political Affiliation Analysis

```python
# Analyze political leaning in social networks
from guidedLP.glp.validation import train_test_split_validation
from guidedLP.network.construction import build_graph_from_edgelist

# Load political Twitter network
political_edges = pl.read_csv("political_network.csv")
graph, id_mapper = build_graph_from_edgelist(
    political_edges, "follower", "following"
)

# Define known political accounts as seeds
political_seeds = {
    "progressive": ["@aoc", "@berniesanders", "@ewarren"],
    "conservative": ["@realdonaldtrump", "@tedcruz", "@marcorubio"]
}

# Run validation to test accuracy
accuracy, metrics = train_test_split_validation(
    graph=graph,
    seeds=political_seeds,
    id_mapper=id_mapper,
    test_size=0.2
)

print(f"Political classification accuracy: {accuracy:.3f}")
```

### 2. Temporal Network Analysis

```python
# Track community evolution over time
from guidedLP.timeseries.slicing import create_temporal_slices
from guidedLP.timeseries.temporal_metrics import extract_temporal_metrics

# Load temporal network data
temporal_data = pl.read_csv("tests/fixtures/sample_temporal.csv")

# Create time slices
time_slices = create_time_slices(
    temporal_data,
    time_col="timestamp",
    slice_duration="1d"  # daily slices
)

# Analyze each time slice
for date, slice_edges in time_slices.items():
    graph, id_mapper = build_graph_from_edgelist(
        slice_edges, "source", "target", "weight"
    )
    
    results = guided_label_propagation(graph, seeds, id_mapper)
    print(f"{date}: {len(results)} nodes classified")
```

### 3. Academic Collaboration Networks

```python
# Map research communities in citation networks
academic_edges = pl.read_csv("collaboration_network.csv")
graph, id_mapper = build_graph_from_edgelist(
    academic_edges, "author_a", "author_b", "num_collaborations"
)

# Use known department affiliations as seeds
department_seeds = {
    "computer_science": ["researcher1", "researcher2"],
    "biology": ["researcher3", "researcher4"],
    "physics": ["researcher5", "researcher6"]
}

results = guided_label_propagation(graph, department_seeds, id_mapper)

# Analyze interdisciplinary collaboration
for node_id, probabilities in results.items():
    if max(probabilities.values()) < 0.7:  # Low confidence
        print(f"{node_id}: Likely interdisciplinary researcher")
```

### 4. Temporal Bipartite-to-Unipartite Conversion

```python
# Convert user-item interactions to user influence networks  
import polars as pl
from guidedLP.network.construction import temporal_bipartite_to_unipartite

# Load temporal bipartite data (users interacting with items over time)
data = pl.DataFrame({
    "user": ["Alice", "Bob", "Charlie", "Alice", "Bob"],
    "item": ["item1", "item1", "item1", "item2", "item2"], 
    "timestamp": ["2024-01-01 09:00", "2024-01-01 11:00", "2024-01-01 13:00",
                  "2024-01-02 10:00", "2024-01-02 15:00"]
})

# Convert to directed user-user influence network  
influence_graph, user_mapper = temporal_bipartite_to_unipartite(
    data,
    source_col="user",
    target_col="item",
    timestamp_col="timestamp",
    intermediate_col="item",    # Items disappear
    projected_col="user",       # Users get connected
    add_edge_weights=True       # Include temporal decay
)

print(f"Created {influence_graph.numberOfNodes()} user influence network")
print(f"Temporal relationships: {influence_graph.numberOfEdges()} edges")

# Expected edges: Alice → Bob → Charlie (temporal precedence preserved)
```

## Use Cases

- **Political Affiliation Analysis**: Identify political leaning of unknown users based on known partisan seed accounts
- **Brand Affinity Detection**: Determine brand preferences in social networks using verified brand accounts as seeds  
- **Research Community Mapping**: Map academic collaboration networks and identify research area affiliations
- **Temporal Network Evolution**: Track how community structures evolve over time in dynamic networks
- **Temporal Influence Networks**: Convert user-item interactions to user-user influence networks with proper temporal causality
- **Content Recommendation**: Classify users for targeted content delivery
- **Fraud Detection**: Identify suspicious accounts based on known fraudulent patterns

## Documentation

### API Reference
- [Network Construction](docs/api/network.md) - Graph building and analysis
- [Guided Label Propagation](docs/api/glp.md) - Core GLP algorithms
- [Validation](docs/api/validation.md) - Model validation and metrics
- [Temporal Analysis](docs/api/timeseries.md) - Time-series network analysis
- [Utilities](docs/api/common.md) - Common utilities and I/O

### Guides
- [Getting Started](docs/getting_started.md) - Detailed tutorial
- [Architecture Overview](docs/architecture/overview.md) - System design
- [Performance Guidelines](docs/performance.md) - Optimization tips
- [Data Formats](docs/data_formats.md) - Input/output specifications

### Examples
- [Basic GLP Analysis](examples/example_glp_analysis.py)
- [Network Analysis](examples/example_network_analysis.py) 
- [Temporal Networks](examples/example_timeseries.py)
- [Complete Workflows](examples/)

### Online Documentation
- **Full Documentation**: [https://guided-label-propagation.readthedocs.io](https://guided-label-propagation.readthedocs.io)
- **API Reference**: [https://guided-label-propagation.readthedocs.io/api/](https://guided-label-propagation.readthedocs.io/api/)

## System Requirements

### Core Dependencies
- **Python**: 3.9 or higher
- **NetworkIt**: 11.0+ (C++ graph library for performance)
- **Polars**: 0.20.0+ (Fast DataFrame operations)
- **NumPy**: 1.24.0+ (Numerical computing)
- **SciPy**: 1.10.0+ (Sparse matrices and scientific computing)

### Platform Support
- **Linux**: Full support (recommended for large-scale analysis)
- **macOS**: Full support 
- **Windows**: Supported (may require Visual C++ redistributable)

### Performance Notes
- Minimum 8GB RAM recommended for networks with >10,000 nodes
- SSD storage recommended for large temporal datasets
- Multi-core CPU beneficial for parallel operations

## Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

### Quick Start for Contributors

1. **Fork and Clone**
   ```bash
   git clone https://github.com/yourusername/guided-label-propagation.git
   cd guided-label-propagation
   ```

2. **Set Up Development Environment**
   ```bash
   # Install with all development dependencies
   pip install -e ".[dev,docs,viz]"
   
   # Install pre-commit hooks
   pre-commit install
   ```

3. **Run Tests**
   ```bash
   # Run all tests
   pytest
   
   # Run with coverage
   pytest --cov=src --cov-report=html
   
   # Run specific test modules
   pytest tests/glp/test_propagation.py
   ```

4. **Code Quality**
   ```bash
   # Format code
   black src/ tests/
   
   # Lint code
   ruff check src/ tests/
   
   # Type checking
   mypy src/
   ```

5. **Development Workflow**
   - Create feature branch: `git checkout -b feature/amazing-feature`
   - Make changes following code style (ruff + black)
   - Add tests for new functionality
   - Update documentation if needed
   - Commit changes: `git commit -m 'Add amazing feature'`
   - Push branch: `git push origin feature/amazing-feature`
   - Open a Pull Request

### Code Style

- **Formatting**: Black (line length: 88)
- **Linting**: Ruff with strict settings
- **Type Hints**: Required for all public functions
- **Documentation**: Google-style docstrings
- **Testing**: Pytest with >90% coverage target

### Testing Guidelines

- Write tests for all new features
- Use the fixtures in `tests/fixtures/` for consistent test data
- Include integration tests for complete workflows
- Test edge cases and error conditions

## License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

### Summary
- ✅ Commercial use allowed
- ✅ Modification allowed
- ✅ Distribution allowed
- ✅ Private use allowed
- ❌ No warranty provided
- ❌ No liability accepted

## Citation

If you use this software in your research, please cite:

```bibtex
@software{guided_label_propagation,
  title={Guided Label Propagation: Semi-supervised Community Detection for Large-Scale Networks},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/guided-label-propagation}
}
```

## Support and Community

### Getting Help
- **Documentation**: [https://guided-label-propagation.readthedocs.io](https://guided-label-propagation.readthedocs.io)
- **GitHub Issues**: [Report bugs and request features](https://github.com/yourusername/guided-label-propagation/issues)
- **GitHub Discussions**: [Ask questions and share ideas](https://github.com/yourusername/guided-label-propagation/discussions)
- **Examples**: See the [examples/](examples/) directory

### Stay Updated
- **Releases**: [GitHub Releases](https://github.com/yourusername/guided-label-propagation/releases)
- **Changelog**: [CHANGELOG.md](CHANGELOG.md)

### Contact
- **Maintainers**: [GitHub Team](https://github.com/yourusername/guided-label-propagation/graphs/contributors)
- **Email**: your.email@example.com

---

**Made with ❤️ for the computational social science community**