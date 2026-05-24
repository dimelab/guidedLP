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
    weight_col="weight",  # optional
)

# Seed nodes can be supplied in any of four shapes — pick whichever is
# convenient. They are normalized internally to {node_id: label}.
#
# (1) node_id -> label                              (canonical dict)
# (2) label -> [node_ids]                           (label-keyed dict)
# (3) polars.DataFrame  with `node_id` + `label`    columns
# (4) pandas.DataFrame  with `node_id` + `label`    columns
seeds = {"user123": "progressive", "user321": "conservative"}

# Run Guided Label Propagation
results = guided_label_propagation(
    graph=graph,
    id_mapper=id_mapper,
    seed_labels=seeds,
    labels=["progressive", "conservative"],
    alpha=0.85,
    max_iterations=100,
    convergence_threshold=1e-6,
)

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
    seeds = json.load(f)  # already {node_id: label}

# Build graph and run GLP
graph, id_mapper = build_graph_from_edgelist(
    edges, source_col="source", target_col="target", weight_col="weight"
)
results = guided_label_propagation(
    graph=graph,
    id_mapper=id_mapper,
    seed_labels=seeds,
    labels=sorted(set(seeds.values())),
)

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

### 1. Network Backboning Before Propagation

For dense weighted networks, the **disparity filter** (Serrano et al., 2009) extracts the statistically significant subset of edges before label propagation. The propagation step then runs on a cleaner network where weak/incidental connections aren't diluting the signal.

**When backboning helps:**
- Edge weights span many orders of magnitude — a few strong ties, many incidental ones (e.g. one-off retweets vs. sustained interaction).
- The network is dense enough that random walks mix categories quickly and confidence collapses.
- You need to cut compute time on a very large network — fewer edges, fewer matrix–vector products per GLP iteration.

**When to skip it:**
- Network is unweighted, or already sparse.
- Edge weight variance is small (all edges carry roughly equal information).
- Weak edges *are* the signal you want to study.
- You have plenty of seeds per category — propagation already has enough anchors to overwhelm noise.

```python
import polars as pl
from guidedLP.network.construction import build_graph_from_edgelist
from guidedLP.network.filtering import apply_backbone
from guidedLP.glp.propagation import guided_label_propagation

edges = pl.read_csv("interactions.csv")
graph, id_mapper = build_graph_from_edgelist(
    edges, source_col="user_a", target_col="user_b", weight_col="weight",
)
print(f"Full graph:   {graph.numberOfNodes():>6} nodes, {graph.numberOfEdges():>7} edges")

# Disparity filter: keep edges whose normalized weight is statistically
# significant against a null model of uniform weight distribution at each
# node. Lower alpha = stricter backbone.
backbone, backbone_mapper = apply_backbone(
    graph, id_mapper,
    method="disparity",
    alpha=0.05,                # typical range 0.01–0.1
    keep_disconnected=False,   # drop nodes that lost all their edges
)
print(f"Backbone:     {backbone.numberOfNodes():>6} nodes, {backbone.numberOfEdges():>7} edges")

# IMPORTANT: apply_backbone returns a NEW id_mapper. Always pass it (not the
# original) to downstream functions. Some seed nodes may also be dropped if
# all their edges fell below significance — use check_seed_coverage to verify
# what survived (per label) before propagation.
from guidedLP.glp.utils import check_seed_coverage

seeds = {"@aoc": "left", "@berniesanders": "left",
         "@realdonaldtrump": "right", "@tedcruz": "right"}
report = check_seed_coverage(backbone_mapper, seeds)
if report["train"]["coverage"] < 1.0:
    print(f"WARNING: {report['train']['missing']} seed(s) dropped by backboning: "
          f"{report['train']['missing_sample']}")
    for label, stats in report["train"]["by_label"].items():
        print(f"  {label}: {stats['present']}/{stats['total']} survived")

# Keep only seeds that are still in the backbone for the propagation call.
seeds_in_backbone = {k: v for k, v in seeds.items() if backbone_mapper.has_original(k)}

results = guided_label_propagation(
    graph=backbone,
    id_mapper=backbone_mapper,
    seed_labels=seeds_in_backbone,
    labels=["left", "right"],
)
```

Other `method` values: `"weight"` (simple weight threshold — use with `target_edges=N`) and `"degree"` (keep top-N nodes by degree — use with `target_nodes=N`).

### 2. Bipartite to Unipartite Projection

Many computational social science datasets are naturally **bipartite** — users connected to hashtags, authors to papers, accounts to URLs they share. GLP works on a unipartite graph where seeds and unknown nodes live in the same partition, so the typical preprocessing step is to project: *connect two users if they share content, mediated by the items between them*.

**When to project:**
- Your edge list joins two distinct node types (user → hashtag, author → paper).
- Your seeds live in one of the two partitions and you want propagation within that partition.
- "Similarity" in your domain is defined by shared connections to the other partition.

**Choose `weight_method` based on what "similarity" should mean:**
- `"count"` — raw number of shared neighbors. Simple, but a single power-user who touches everything looks similar to everyone.
- `"jaccard"` — |A ∩ B| / |A ∪ B|. Symmetric and normalized. Good default when node degrees vary widely.
- `"overlap"` — |A ∩ B| / min(|A|, |B|). Asymmetric; favors "the small set is contained in the big one".

```python
import polars as pl
from guidedLP.network.construction import build_graph_from_edgelist, project_bipartite
from guidedLP.glp.propagation import guided_label_propagation

# Edge list joining users to hashtags they used
edges = pl.read_csv("user_hashtag_uses.csv")  # columns: user, hashtag, count
bipartite, full_mapper = build_graph_from_edgelist(
    edges,
    source_col="user",
    target_col="hashtag",
    weight_col="count",
    bipartite=True,             # validate that source/target are disjoint sets
)
print(f"Bipartite:  {bipartite.numberOfNodes()} nodes "
      f"({sum(1 for _ in edges['user'].unique())} users + "
      f"{sum(1 for _ in edges['hashtag'].unique())} hashtags)")

# Project onto USERS: two users get connected if they share hashtags.
# jaccard avoids high-degree users dominating the similarity.
user_graph, user_mapper = project_bipartite(
    bipartite, full_mapper,
    projection_mode="source",   # users appear in the source column
    weight_method="jaccard",
)
print(f"User-user:  {user_graph.numberOfNodes()} nodes, {user_graph.numberOfEdges()} edges")

# user_mapper now contains ONLY users (the projected partition). Seeds must
# refer to users — hashtag IDs from the original bipartite mapper won't be
# found and would raise a validation error.
seeds = {"@alice": "progressive", "@bob": "conservative"}
results = guided_label_propagation(
    graph=user_graph, id_mapper=user_mapper,
    seed_labels=seeds, labels=["progressive", "conservative"],
)
```

**Other directions:**
- Set `projection_mode="target"` to project onto the *hashtag* partition instead — useful if your seeds are labelled hashtags and you want to classify unknown ones.
- For temporal bipartite data where edge order matters (A shared the item *before* B → A may have influenced B), see Example #6 below — `temporal_bipartite_to_unipartite` produces a *directed* unipartite graph that preserves this causality.

**Watch out for:**
- Projections can blow up edge count: O(N² × D) worst case. On dense bipartite graphs, consider backboning either before projecting (sparsifies the bipartite layer) or after (sparsifies the projection itself).
- Seeds must be in the projection partition; verify with `check_seed_coverage(user_mapper, seeds)` before propagation.

### 3. Political Affiliation Analysis

```python
# Analyze political leaning in social networks
from guidedLP.glp.validation import train_test_split_validation
from guidedLP.network.construction import build_graph_from_edgelist

# Load political Twitter network
political_edges = pl.read_csv("political_network.csv")
graph, id_mapper = build_graph_from_edgelist(
    political_edges, "follower", "following"
)

# Define known political accounts as seeds.
# Any of the four supported shapes works here — label-keyed dict is convenient
# when you collected seeds in lists per category.
political_seeds = {
    "progressive": ["@aoc", "@berniesanders", "@ewarren"],
    "conservative": ["@realdonaldtrump", "@tedcruz", "@marcorubio"],
}

# Run validation to test accuracy
metrics = train_test_split_validation(
    graph=graph,
    id_mapper=id_mapper,
    seed_labels=political_seeds,
    labels=["progressive", "conservative"],
    test_size=0.2,
)

print(f"Political classification accuracy: {metrics['accuracy']:.3f}")
```

### 4. Temporal Network Analysis

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
        slice_edges, source_col="source", target_col="target", weight_col="weight"
    )

    results = guided_label_propagation(
        graph=graph, id_mapper=id_mapper, seed_labels=seeds, labels=list(set(seeds.values()))
    )
    print(f"{date}: {len(results)} nodes classified")
```

### 5. Academic Collaboration Networks

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

results = guided_label_propagation(
    graph=graph,
    id_mapper=id_mapper,
    seed_labels=department_seeds,  # label-keyed dict works directly
    labels=["computer_science", "biology", "physics"],
)

# Analyze interdisciplinary collaboration: rows where the dominant label
# wins by a small margin are candidate cross-disciplinary researchers.
low_confidence = results.filter(results["confidence"] < 0.7)
for row in low_confidence.iter_rows(named=True):
    print(f"{row['node_id']}: Likely interdisciplinary researcher")
```

### 6. Temporal Bipartite-to-Unipartite Conversion

When edge order matters — A shared an item *before* B did, so B may have been attributing to A — use `temporal_bipartite_to_unipartite`. It produces a **directed** unipartite graph using **citation convention**: edges point from the *later* sharer to the *earlier* one. Under this convention the earliest sharer accumulates the most incoming edges, and PageRank / HITS-Authority naturally surface them as the influential sources.

**Important: this function does not sort.** It trusts the input row order. You must pre-sort the edgelist by intermediate column ascending, then by timestamp **descending** (latest first) within each item group:

```python
import polars as pl
from guidedLP.network.construction import temporal_bipartite_to_unipartite

data = pl.DataFrame({
    "user":      ["Alice", "Bob",   "Charlie", "Alice", "Bob"],
    "item":      ["item1", "item1", "item1",   "item2", "item2"],
    "timestamp": ["2024-01-01 09:00", "2024-01-01 11:00", "2024-01-01 13:00",
                  "2024-01-02 10:00", "2024-01-02 15:00"],
}).with_columns(pl.col("timestamp").str.to_datetime())

# REQUIRED pre-sort. The function will use this row order as-is.
data = data.sort(["item", "timestamp"], descending=[False, True])

attribution_graph, user_mapper = temporal_bipartite_to_unipartite(
    data,
    source_col="user", target_col="item", timestamp_col="timestamp",
    intermediate_col="item",   # Items disappear in the projection
    projected_col="user",      # Users remain, get connected
    add_edge_weights=True,     # Weight decays with temporal gap between sharers
)

print(f"Graph: {attribution_graph.numberOfNodes()} nodes, "
      f"{attribution_graph.numberOfEdges()} edges (directed)")
# Citation-direction edges produced:
#   item1: Charlie → Bob,  Charlie → Alice,  Bob → Alice
#   item2: Bob → Alice                                       (collapses with item1's Bob→Alice)
# Alice (earliest sharer of both items) ends up with the most incoming edges.

# This makes PageRank / HITS-Authority surface Alice as the influential source.
```

If your data isn't in the required order, the function silently produces wrong-direction edges — no validation is performed (the assumption is that callers working with large datasets pre-sort once and reuse). When in doubt, re-sort right before the call.

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