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
- **`EdgeList` container**: Polars-backed coded edge store paired with an `IDMapper` — peer of `nk.Graph`. Stores `src` / `tgt` as `UInt32` codes (~5–10× smaller than Utf8) and powers a vectorized SciPy projection kernel that delivers ~3–7× peak-RSS reduction on hub-heavy bipartite projections versus the legacy `Dict[Any, Set[Any]]` path.

### 🎯 Guided Label Propagation (GLP)
- **Semi-supervised approach**: Uses seed nodes to guide community detection
- **Directional propagation**: Supports both in-degree and out-degree based propagation
- **Probability estimation**: Calculates affinity scores for unknown nodes
- **Validation framework**: Built-in train/test split and external validation

### 📊 Comprehensive Network Toolkit
- **Graph construction**: Unipartite and bipartite networks from edge lists
- **Temporal bipartite projection**: Convert temporal bipartite to directed unipartite with causality preservation
- **Network backboning**: Statistical significance filtering
- **Seed-centered filtering**: Prune to a neighborhood around your seeds via k-hop, Personalized PageRank, or Local Tightness Expansion
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

Both **bipartite** and **unipartite** graphs can be backboned, but the right statistical test is different for each because the null hypothesis you're testing against differs.

| Graph type | Recommended method | What it filters |
|---|---|---|
| **Bipartite** (users ↔ items, authors ↔ papers, …) | `method="bipartite_svn"` | Edges to "generic" high-degree items get dropped because their expected weight under a configuration-model null is high. No degree threshold to tune. |
| **Unipartite weighted** (user ↔ user, etc.) | `method="disparity"` | Within each node's edges, keeps the statistically over-weighted ties and drops the diluted ones (Serrano et al. 2009). Useful when weights span many orders of magnitude. |

**When backboning helps in general:**
- Many edges carry incidental / noise mass that dilutes propagation.
- The network is dense enough that random walks mix categories quickly and confidence collapses.
- You need to cut compute time on a very large network — fewer edges, fewer matrix–vector products per GLP iteration.

**When to skip:** the network is already sparse, edge weights are roughly uniform, weak edges *are* the signal you want to study, or you have plenty of seeds per category.

#### Bipartite case — `bipartite_svn` (Tumminello et al. 2011)

Filter generic items *before* projecting, so the user-user projection is cheaper and cleaner.

```python
import polars as pl
from guidedLP.network.construction import build_graph_from_edgelist, project_bipartite
from guidedLP.network.backboning import apply_backbone
from guidedLP.glp.propagation import guided_label_propagation
from guidedLP.glp.utils import check_seed_coverage

# user-hashtag bipartite from share logs
edges = pl.read_csv("user_hashtag_uses.csv")  # columns: user, hashtag, count
bipartite, full_mapper = build_graph_from_edgelist(
    edges,
    source_col="user", target_col="hashtag", weight_col="count",
    bipartite=True,
)
print(f"Bipartite:   {bipartite.numberOfNodes():>6} nodes, {bipartite.numberOfEdges():>8} edges")

# Statistically Validated Network filter on the bipartite layer.
# For each edge (u, item), test whether the observed weight is more than
# expected under a configuration-model null preserving node strengths.
# - alpha       = per-edge significance cutoff (0.01–0.05 typical)
# - correction  = multiple-testing correction:
#                 "fdr_bh" (default): Benjamini-Hochberg FDR — scales to
#                                     millions of edges; recommended.
#                 "bonferroni":       very conservative — at |E| ≥ ~10⁵
#                                     even α = 0.99 can filter everything.
#                 "none":             use α directly; most permissive.
# - min_node_retention = optional post-filter that drops whole nodes whose
#                        share of surviving edges falls below the threshold.
#                        Use this to *eliminate* generic items rather than
#                        just trim their fringes. Captures the intuition
#                        "if most of a node's edges were noise, the node is
#                        noise." Typical values: 0.5 (lost >half ⇒ drop),
#                        0.3 (lost >70% ⇒ drop). Leave as None for no
#                        node-level filtering.
backbone, backbone_mapper = apply_backbone(
    bipartite, full_mapper,
    method="bipartite_svn",
    alpha=0.05,
    correction="fdr_bh",
    min_node_retention=0.5,   # leave None for per-edge SVN only
    keep_disconnected=False,
)
print(f"Backbone:    {backbone.numberOfNodes():>6} nodes, {backbone.numberOfEdges():>8} edges")

# Now the project step is much cheaper.
user_graph, user_mapper = project_bipartite(
    backbone, backbone_mapper, projection_mode="source", weight_method="jaccard",
)

# Always check seed survival — backboning drops nodes whose edges weren't
# significant. check_seed_coverage handles all SeedInput shapes and gives a
# per-label breakdown.
seeds = {"@aoc": "left", "@berniesanders": "left",
         "@realdonaldtrump": "right", "@tedcruz": "right"}
report = check_seed_coverage(user_mapper, seeds)
print(f"Seeds surviving: {report['train']['present']}/{report['train']['total']}")
seeds = {k: v for k, v in seeds.items() if user_mapper.has_original(k)}

results = guided_label_propagation(
    graph=user_graph, id_mapper=user_mapper,
    seed_labels=seeds, labels=["left", "right"],
)
```

The `bipartite_svn` filter exposes three knobs — `alpha`, `correction`, and `min_node_retention`:

| Flag | Default | What it does |
|---|---|---|
| `alpha` | `0.05` | Significance cutoff for per-edge p-values. Tighter (e.g. `0.01`) keeps fewer edges. |
| `correction` | `"fdr_bh"` | Multiple-testing correction across the per-edge tests. Benjamini-Hochberg FDR is the right default. `"bonferroni"` is only sensible for small graphs (thousands of edges) — on millions it filters everything. `"none"` disables correction. |
| `min_node_retention` | `None` | Optional post-filter. After per-edge SVN, each node's `surviving_edges / original_edges` ratio is computed; nodes below the threshold are removed entirely along with their remaining edges. Use this to eliminate generic items that survived per-edge SVN with a fraction of their edges. Typical values: `0.5` (drop nodes that lost more than half), `0.3` (more aggressive). |

The per-edge test naturally drops edges to generic items (their expected weight under the null is high), but a generic item that's connected to thousands of users typically still keeps hundreds of edges and stays in the graph. The `min_node_retention` post-filter catches those: a generic item whose 1000-edge connectivity got pruned to 100 edges has retention `0.1` and gets removed at `min_node_retention=0.5`.

#### Unipartite case — `disparity` (Serrano et al. 2009)

When you already have a unipartite weighted graph (e.g. retweet network, collaboration network), the disparity filter is the right tool. It evaluates each edge from each endpoint's perspective: is this edge's weight a meaningful share of the node's total, or is it just noise?

```python
graph, mapper = build_graph_from_edgelist(
    pl.read_csv("retweets.csv"),
    source_col="user_a", target_col="user_b", weight_col="weight",
)
backbone, backbone_mapper = apply_backbone(
    graph, mapper,
    method="disparity",
    alpha=0.05,                # typical range 0.01–0.1; lower → stricter
    keep_disconnected=False,
)
```

The disparity filter needs **weighted** input with meaningful weight variance. On uniformly-weighted graphs it has very little to discriminate on and barely filters anything.

Other `method` values: `"weight"` (threshold by edge weight — use with `target_edges=N`) and `"degree"` (keep top-N nodes by degree — use with `target_nodes=N`).

### 2. Seed-Centered Filtering Before Propagation

`filter_by_seed_proximity` prunes a graph to a neighborhood around your seed set. It complements backboning: backboning sparsifies *globally* by statistical significance, while seed-proximity filtering trims *locally* by relevance to the labelled nodes.

**When this helps:**
- Your seeds cover one part of a much larger graph and irrelevant subgraphs are diluting propagation.
- You want to run GLP on a manageable region of a huge network without losing the structure around your seeds.
- You're doing exploratory analysis and want a graph small enough to inspect visually.

**When to skip:** seeds are dispersed across the whole network, or you specifically want the long-tail propagation behavior of GLP across the full graph.

Three methods are supported, each returning `(graph, id_mapper)` so they can be chained:

| Method | What it does | When to pick |
|---|---|---|
| `"khop"` (default) | BFS up to `hops` levels from the seed set. | Predictable size, fast. Watch out for hub nodes — one celebrity seed can pull in millions of nodes at 2 hops. |
| `"ppr"` | Personalized PageRank from seeds; keep `top_n` and/or above `min_ppr`. | Theoretically aligned with GLP itself (same `αPF + (1−α)Y` propagation kernel). The most natural pre-filter when you're about to run GLP. |
| `"lte"` | NetworkIt's `LocalTightnessExpansion` grown from the seed set. | Adaptive — stops at community boundaries automatically. No size knob to tune, so result size depends on graph structure. |

```python
from guidedLP.network.filtering import filter_by_seed_proximity

seeds = ["@aoc", "@berniesanders", "@realdonaldtrump", "@tedcruz"]

# Two-hop BFS neighborhood
g_sub, m_sub = filter_by_seed_proximity(
    graph, mapper, seeds, method="khop", hops=2,
)

# Personalized PageRank keeping the top 5000 nodes by PPR mass.
# Set min_ppr=<threshold> instead (or alongside) to filter by mass.
g_sub, m_sub = filter_by_seed_proximity(
    graph, mapper, seeds,
    method="ppr",
    top_n=5000,
    ppr_alpha=0.85,   # higher → mass spreads further from seeds
)

# Local Tightness Expansion — adaptive, no size knob
g_sub, m_sub = filter_by_seed_proximity(
    graph, mapper, seeds, method="lte",
)
```

**Chaining methods.** Each call returns a fresh graph with contiguous internal IDs `0..K-1`, so methods can be stacked:

```python
# Step 1: bound size with a 3-hop frontier (prevents hub explosion).
g_hop, m_hop = filter_by_seed_proximity(
    graph, mapper, seeds, method="khop", hops=3,
)
# Step 2: within that frontier, keep only the tightly-connected core.
g_core, m_core = filter_by_seed_proximity(
    g_hop, m_hop, seeds, method="lte",
)
# g_core is ready to feed into guided_label_propagation.
```

**Seed input.** Accepts a list/tuple/set of original node IDs or a polars DataFrame (column name configurable via `seed_column`, default `"node_id"`). Labels aren't required — only the set of seeds.

**Directed graphs.** Use `direction={"out", "in", "both"}` (default `"both"`) to control which edges to follow for `"khop"` and `"ppr"`. `"lte"` always operates on an undirected view but preserves the original direction in the returned graph.

**Tip.** `include_seeds=True` (the default) guarantees seeds survive the filter, but run `check_seed_coverage(m_sub, seeds)` afterwards if seeds are mapped via a label dict — it's a cheap sanity check before launching GLP.

### 3. Bipartite to Unipartite Projection

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
# Count unique users / hashtags as actually present in the constructed
# graph (null-row drop, degree filters, and bipartite overlap policy can
# all change what survives, so the raw input columns aren't a reliable
# proxy). build_graph_from_edgelist(bipartite=True) records the surviving
# source and target node sets on the IDMapper:
n_users = len(full_mapper.source_partition_originals)
n_hashtags = len(full_mapper.target_partition_originals)
print(f"Bipartite:  {bipartite.numberOfNodes()} nodes "
      f"({n_users} users + {n_hashtags} hashtags)")

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

**For hub-heavy bipartites at scale, use the `EdgeList` path.** When a few popular intermediate items (a viral hashtag, a popular URL) are touched by thousands of users, the projection edge count can explode by orders of magnitude through a single hub. The legacy graph-input path builds a `Dict[Any, Set[Any]]` neighbor map and a per-edge Python accumulator loop that becomes the memory bottleneck; the coded `EdgeList` path skips both and runs entirely on `UInt32` codes through a vectorized SciPy kernel. On synthetic Zipf-popular bipartites we measure ~3–7× peak-RSS reduction; at 240K bipartite → 199M projection edges the legacy path OOMs while the coded path completes cleanly.

```python
from guidedLP.network.construction import build_edgelist_from_frame, project_bipartite

# build_edgelist_from_frame takes the same kwargs as build_graph_from_edgelist
# plus an optional code_dtype (UInt32 default; pass UInt64 above ~4.29B nodes).
bipartite_el, full_mapper = build_edgelist_from_frame(
    edges, source_col="user", target_col="hashtag", weight_col="count",
    bipartite=True,
)
# Default output matches input: EdgeList in → (EdgeList, IDMapper) out.
user_el, user_mapper = project_bipartite(
    bipartite_el, full_mapper, projection_mode="source", weight_method="jaccard",
)
# Force a graph or frame if downstream needs one:
#   output_format="graph"     → (nk.Graph, IDMapper)
#   output_format="dataframe" → pl.DataFrame  (source_id, target_id, weight)
```

### 4. Evaluating GLP Quality with Held-Out Seeds

`train_test_split_validation` is the standard "is my model good?" check for GLP. It holds out a fraction of your labelled seeds, trains GLP on the rest, and scores predictions on the held-out portion. The returned dict carries accuracy, per-label precision / recall / F1, a confusion matrix, and the sklearn classification report — everything you need to write a results section.

```python
import polars as pl
from guidedLP.network.construction import build_graph_from_edgelist
from guidedLP.glp.validation import train_test_split_validation

political_edges = pl.read_csv("political_network.csv")
graph, id_mapper = build_graph_from_edgelist(
    political_edges, source_col="follower", target_col="following",
)

# Seeds in any SeedInput shape. Label-keyed dict is convenient when you
# collected accounts in lists per category.
political_seeds = {
    "progressive":  ["@aoc", "@berniesanders", "@ewarren",
                     "@progressive_1", "@progressive_2", "@progressive_3"],
    "conservative": ["@realdonaldtrump", "@tedcruz", "@marcorubio",
                     "@conservative_1", "@conservative_2", "@conservative_3"],
}

# Stratified 20% holdout (default). Any guided_label_propagation kwarg can
# be passed through — alpha, directional, weight_transform, etc.
results = train_test_split_validation(
    graph=graph, id_mapper=id_mapper,
    seed_labels=political_seeds, labels=["progressive", "conservative"],
    test_size=0.2,
    stratify=True,            # preserve label proportions in the split
    random_seed=42,           # reproducible
    alpha=0.85,               # **glp_kwargs — threaded into GLP
    directional=False,
)

# The result dict has everything you need to summarize quality.
print(f"Train size: {results['train_size']}  Test size: {results['test_size']}")
print(f"Accuracy:   {results['accuracy']:.3f}")
print(f"Macro-F1:   {results['macro_f1']:.3f}")
for label in ["progressive", "conservative"]:
    p = results["precision"][label]
    r = results["recall"][label]
    f = results["f1_score"][label]
    print(f"  {label:>13}  P={p:.2f}  R={r:.2f}  F1={f:.2f}")

print(results["classification_report"])   # sklearn-formatted per-label table
print(results["confusion_matrix"])        # rows=true, cols=predicted (np.ndarray)

# Drill into individual errors. `test_predictions` is the raw GLP frame
# restricted to the held-out seeds; cross-reference against the input
# seed dict for the ground truth.
test_df = results["test_predictions"]
flat_seeds = {k: v for label, ids in political_seeds.items() for k, v in [(i, label) for i in ids]}
errors = test_df.with_columns(
    pl.col("node_id").replace_strict(flat_seeds).alias("true_label")
).filter(pl.col("dominant_label") != pl.col("true_label"))
print(f"Misclassified: {errors.height} of {test_df.height}")
```

**Custom test set instead of a random split.** When you have a separately-curated ground-truth set (e.g. hand-labelled accounts you specifically *don't* want in training), pass it via `test_seeds`. Overlapping IDs are pulled out of training; on label conflicts the test set's label wins (with a warning). `test_size`, `stratify`, and `random_seed` are ignored in this mode.

```python
known_holdout = {
    "@verified_left_1":  "progressive",  "@verified_left_2":  "progressive",
    "@verified_right_1": "conservative", "@verified_right_2": "conservative",
}
results = train_test_split_validation(
    graph=graph, id_mapper=id_mapper,
    seed_labels=political_seeds,        # train on these
    test_seeds=known_holdout,           # evaluate on these (curated ground truth)
    labels=["progressive", "conservative"],
)
```

**Small seed sets?** A single held-out split is noisy when you only have a handful of seeds per label. `cross_validate` runs K-fold over the seeds and returns mean ± std for each metric — same kwargs as `train_test_split_validation`, more stable estimates:

```python
from guidedLP.glp.validation import cross_validate

cv = cross_validate(
    graph=graph, id_mapper=id_mapper,
    seed_labels=political_seeds, labels=["progressive", "conservative"],
    k_folds=5, stratify=True, random_seed=42,
)
print(f"5-fold CV: accuracy={cv['mean_accuracy']:.3f} ± {cv['std_accuracy']:.3f}, "
      f"macro-F1={cv['mean_macro_f1']:.3f} ± {cv['std_macro_f1']:.3f}")
```

**Validate with ensembling instead of single-run GLP.** Both `train_test_split_validation` and `cross_validate` accept a `propagator` kwarg. Default is `guided_label_propagation`; pass `ensemble_label_propagation` to score each fold with a bagged, noise-resampled ensemble. Any propagator-specific kwargs (`n_epochs`, `base_seed`, `enable_noise_category`, …) ride along through `**glp_kwargs`. The propagator must return a single DataFrame, so pass `directional=False` if you'd otherwise get a tuple.

```python
from guidedLP.glp.propagation import ensemble_label_propagation
from guidedLP.glp.validation import train_test_split_validation, cross_validate

# Single split, ensemble-scored
results = train_test_split_validation(
    graph=graph, id_mapper=id_mapper,
    seed_labels=political_seeds, labels=["progressive", "conservative"],
    test_size=0.2, random_seed=42,
    propagator=ensemble_label_propagation,
    # ensemble-specific kwargs flow through:
    n_epochs=20, enable_noise_category=True, noise_ratio=0.3,
    base_seed=42,
    # GLP kwargs still work too:
    alpha=0.85, directional=False,
)

# K-fold CV with the same switch
cv = cross_validate(
    graph=graph, id_mapper=id_mapper,
    seed_labels=political_seeds, labels=["progressive", "conservative"],
    k_folds=5, random_seed=42,
    propagator=ensemble_label_propagation,
    n_epochs=20, enable_noise_category=True, directional=False,
)
```

When to prefer ensemble validation: you're planning to deploy with `ensemble_label_propagation` and want fold-level metrics that reflect the same model. When to skip: validation is just a sanity check before a final run — single-run GLP is cheaper and gives comparable rankings of hyperparameter choices.

### 5. Temporal Network Analysis

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

### 6. Academic Collaboration Networks

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

### 7. Temporal Bipartite-to-Unipartite Conversion

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

### 8. Refining GLP Output

Three controls layered on top of `guided_label_propagation`, each addressing a different failure mode of the default pipeline. Use them individually or compose them.

#### 8a. Edge-weight compression

When a small fraction of edges have weights orders of magnitude larger than the rest (viral retweets, hub co-occurrences, runaway co-citations), those edges dominate the propagation regardless of `alpha`. The `weight_transform` parameter applies a per-edge callable when the transition matrix is built, so you can dampen weight outliers without rebuilding the graph or pre-normalizing in your data pipeline.

```python
from guidedLP.glp import (
    guided_label_propagation,
    tanh_transform, log1p_transform, winsorize_transform,
)

# Pick a transform that matches the shape of your weight distribution:
# - log1p_transform()     : gentle, unbounded. Good first choice when weights
#                           span several orders of magnitude.
# - winsorize_transform(c): hard cap at threshold c. Linear up to c, then flat.
#                           Use when you know which weight value is "too high."
# - tanh_transform()      : S-curve saturation; mimics the historical stlp
#                           transform. Most aggressive.

results = guided_label_propagation(
    graph, id_mapper, seeds, labels,
    weight_transform=log1p_transform(),
)

# Any positive-output callable works too — the helpers are just conveniences:
import math
results = guided_label_propagation(
    graph, id_mapper, seeds, labels,
    weight_transform=lambda w: math.log10(w + 1.0),
)
```

**When to use:** edge weights span multiple orders of magnitude and a few hub edges visibly skew the results (e.g. one viral retweet that's 1000× the median weight).

**When to skip:** weights are already roughly uniform, or you've pre-normalized upstream.

**Caveat:** transforms must satisfy `f(w) > 0` for `w > 0`. A transform that maps positive weights to zero will be interpreted as graph isolates by the propagation engine.

#### 8b. Audience-composition pass

The directional GLP (`directional=True`) returns a `(forward, backward)` tuple where the backward pass is a full propagation on `Aᵀ` from the original seeds — answering *"is node n upstream of a seed?"*. A semantically different question — *"what's the label profile of the nodes pointing at n?"* — requires a one-hop aggregation of the converged forward result along incoming edges. `audience_composition_pass` does exactly that.

```python
from guidedLP.glp import (
    guided_label_propagation,
    audience_composition_pass,
)

# Step 1: forward pass. Use directional=False to get a single DataFrame
# (audience_composition_pass consumes a single result, not the tuple).
fwd = guided_label_propagation(
    graph, id_mapper, seeds, labels, directional=False,
)

# Step 2: audience pass — single sparse matmul, no iteration.
audience = audience_composition_pass(
    graph, id_mapper, fwd, labels,
    # If the forward pass used a weight_transform, pass the same one here:
    # weight_transform=log1p_transform(),
)

# audience["left_prob"] reads as:
# "in-degree-weighted average of forward-pass left_prob across n's in-neighbors"
# i.e. "what fraction of accounts pointing at n were forward-labeled left?"
```

**When to use:**
- Audience studies — "who is this account/user reaching?"
- Reception analysis — "what kind of community cites this paper?"
- Recommender-system diagnostics — "which items are consumed by users with profile X?"

**When to skip:** undirected graphs (the pass requires a directed graph and raises otherwise), or when you specifically want the upstream-reachability question that the `Aᵀ` pass already answers.

**Note:** `is_seed` in the audience output is carried through from `forward_result` and identifies forward-pass seeds — the audience pass has no seeds of its own. Nodes with zero in-degree fall back to a uniform distribution (no in-neighbors to aggregate from).

#### 8c. Stochastic ensembling

When `enable_noise_category=True`, the noise seeds are sampled randomly from non-seed nodes. A single GLP run is sensitive to which nodes happen to be chosen. `ensemble_label_propagation` runs GLP `n_epochs` times with different noise samples and averages the result — bagging that recovers the variance-reduction behavior of the historical `stlp` implementation's `epochs` loop.

```python
from guidedLP.glp import ensemble_label_propagation

result = ensemble_label_propagation(
    graph, id_mapper, seeds, labels,
    n_epochs=20,
    base_seed=42,                  # per-epoch seed = base_seed + epoch_index
    enable_noise_category=True,    # required — otherwise ensembling is a no-op
    noise_ratio=0.3,
    return_variance=True,          # adds {label}_prob_std columns
)

# Same return shape as guided_label_propagation:
# - single DataFrame for undirected / directional=False
# - (out_df, in_df) tuple for directed + directional=True
#
# Probability columns are averages across epochs. With return_variance=True,
# each {label}_prob is paired with a {label}_prob_std column for confidence
# intervals (sample std with Bessel's correction).
```

**When to use:** you're already running GLP with `enable_noise_category=True` and want robust probabilities + per-label confidence intervals.

**When to skip:** noise is disabled (the function warns and short-circuits to a single GLP run, because ensembling deterministic runs reduces to repeating the same answer), or single-run probabilities are precise enough for your purpose.

**Caveats:**
- `is_seed` in the ensemble output reflects only user-supplied seeds, not per-epoch noise samples (which vary across epochs).
- `dominant_label` and `confidence` are recomputed from averaged probabilities, not voted across epochs — averaging gives the variance reduction; voting would lose it. A node can have a different dominant label in the ensemble than in any single epoch; this is expected.
- Epochs run serially. The natural parallelization is `concurrent.futures.ProcessPoolExecutor` over epochs — each epoch is independent and reads `graph` read-only — but isn't yet wired into the function.

#### Composing the three

The three controls are independent and combine cleanly. A typical "robust" pipeline:

```python
from guidedLP.glp import (
    ensemble_label_propagation,
    audience_composition_pass,
    log1p_transform,
)

# Ensemble with weight compression + noise resampling
forward_ensemble = ensemble_label_propagation(
    graph, id_mapper, seeds, labels,
    n_epochs=20,
    directional=False,                 # single DataFrame for audience pass
    weight_transform=log1p_transform(),
    enable_noise_category=True,
    noise_ratio=0.3,
)

# Audience pass on the averaged forward result, with the same transform
audience = audience_composition_pass(
    graph, id_mapper, forward_ensemble, labels + ["noise"],
    weight_transform=log1p_transform(),
)
```

### 9. End-to-end pipeline wrapper

For the canonical attribution workflow — raw input → bipartite EdgeList → bipartite-side backbone → temporal projection → projection-side backbone — `guidedLP.pipelines.run_canonical_pipeline` composes all four stages in a single call with explicit memory management between steps. Compared to calling the four functions by hand, the wrapper releases intermediates between stages so they don't co-exist in RAM, and optionally checkpoints to disk for memory-constrained runs.

**Minimal copy-paste version** — the typical call without the explanatory comments:

```python
from guidedLP.pipelines import run_canonical_pipeline

result = run_canonical_pipeline(
    source="shares.parquet",
    source_col="user",
    target_col="item",
    timestamp_col="timestamp",
    weight_col="weight",
    intermediate_col="item",
    projected_col="user",
    min_source_degree=25,
    min_target_degree=None,
    auto_weight=False,
    bipartite_overlap="drop",
    bipartite_alpha=0.01,
    bipartite_correction="fdr_bh",
    bipartite_target_fraction=None,
    add_edge_weights=True,
    remove_self_loops=True,
    presort_temporal=True,
    projection_threshold=1.0,
    projection_target_fraction=0.2,
    memory_mode="balanced",
    checkpoint_dir=None,
    keep_intermediates=False,
    verbose=True,
)

backbone = result.edgelist
mapper = result.id_mapper
print(f"total: {result.total_duration_s:.1f}s | "
      f"{backbone.number_of_edges():,} edges, {backbone.n_nodes:,} nodes")
```

When `verbose=True`, the pipeline prints a per-stage summary plus a final TOTAL line including memory mode and final edge/node counts:

```
[build_edgelist_from_frame] 15.29s | 27,893,278 input rows → 1,877,850 nodes, 23,280,338 edges (UInt32)
[apply_backbone] 1.92s | method=bipartite_svn | EdgeList: 23,280,338 → 12,634,460 edges kept (54.3%)
[temporal_bipartite_to_unipartite] 169.45s | intermediate=item, projected=user, output=edgelist | 12,634,460 input rows → 191,864,084 projection edges
[apply_backbone] 148.67s | method=noise_corrected | EdgeList: 191,864,084 → 38,372,817 edges kept (20.0%)
[run_canonical_pipeline] TOTAL 335.50s | mode=balanced | final: 38,372,817 edges, 104,680 nodes
```

`result.total_duration_s` programmatically returns the sum of per-stage durations; the printed TOTAL also captures inter-stage cleanup (`gc.collect`, sorting) so the two numbers differ slightly.

---

**Reference version** — the same call with **every** parameter shown explicitly, grouped by stage. Defaults are inlined; this is the reference for what each knob does.

```python
from guidedLP.pipelines import run_canonical_pipeline

result = run_canonical_pipeline(
    # ---- Input ---------------------------------------------------------
    # Raw input: a file path (CSV / Parquet) OR a Polars DataFrame.
    source="shares.parquet",

    # Required column names on the input frame.
    source_col="user",                        # one side of the bipartite
    target_col="item",                        # other side of the bipartite
    timestamp_col="timestamp",                # required for the temporal step
    weight_col="weight",                      # optional; None → unit weights

    # ---- Projection orientation ----------------------------------------
    # Which side becomes the intermediate (collapses) vs the projected
    # side (preserved as nodes in the output). Default: collapse the
    # target side, project onto the source side.
    intermediate_col="item",                  # default: target_col
    projected_col="user",                     # default: source_col

    # ---- Stage 1: build_edgelist_from_frame ----------------------------
    # Degree filters applied at build time (None = no filter).
    min_source_degree=25,                     # default: None
    min_target_degree=None,                   # default: None
    # Polars-only convenience: aggregate duplicate (src, tgt) rows into
    # a count weight. Leave False if your input already has explicit weights.
    auto_weight=False,                        # default: False
    # Bipartite overlap policy when a node appears on BOTH sides:
    #   "drop"  : remove the offending node entirely (default)
    #   "side_<source|target>" : keep it on the named side only
    #   "error" : raise on overlap
    bipartite_overlap="drop",                 # default: "drop"

    # ---- Stage 2: apply_backbone(method="bipartite_svn") --------------
    # Per-edge significance level (Poisson configuration null).
    bipartite_alpha=0.01,                     # default: 0.01
    # Multiple-testing correction:
    #   "fdr_bh"     : Benjamini-Hochberg FDR (default; scales to millions of edges)
    #   "bonferroni" : per-edge cutoff alpha / |E| (very conservative)
    #   "none"       : use alpha directly (most permissive)
    bipartite_correction="fdr_bh",            # default: "fdr_bh"
    # Override alpha/correction with a top-K-by-p-value filter. Set this
    # if you want a specific kept-edge count regardless of significance.
    bipartite_target_fraction=None,           # default: None

    # ---- Stage 3: temporal_bipartite_to_unipartite --------------------
    # Edge-weight formula on the projected graph:
    #   True  : (w_later + w_earlier) / 2 * 1 / (1 + Δdays)
    #   False : unit weights
    add_edge_weights=True,                    # default: True
    # Drop self-loops in the projection (rare in practice — only fires
    # when the same projected node co-occurs under different intermediates).
    remove_self_loops=True,                   # default: True
    # Re-sort the bipartite by [intermediate, timestamp DESC] before the
    # temporal kernel. Required for correct citation-direction edges
    # unless you've pre-sorted upstream.
    presort_temporal=True,                    # default: True

    # ---- Stage 4: apply_backbone(method="noise_corrected") ------------
    # Standard-deviation multiplier for the significance margin
    # (kept iff score - threshold * sdev_cij > 0). Ignored when
    # projection_target_fraction is set.
    projection_threshold=1.0,                 # default: 1.0
    # Override threshold and keep the top fraction by significance margin.
    # Recommended on directed projections (where the threshold filter
    # tends to keep ~100% of edges).
    projection_target_fraction=0.2,           # default: None

    # ---- Memory & I/O --------------------------------------------------
    # "fast"     : no inter-stage cleanup (same as calling by hand)
    # "balanced" : del previous + gc.collect() between stages (default)
    # "low"      : additionally checkpoint each EdgeList to parquet and
    #              release the in-memory frame between stages
    memory_mode="balanced",                   # default: "balanced"
    # Where to write parquet checkpoints in "low" mode. None → create
    # a tempdir that's cleaned up on return.
    checkpoint_dir=None,                      # default: None
    # Retain references to each stage's (EdgeList, IDMapper) on the
    # returned result. Incompatible with memory_mode="low".
    keep_intermediates=False,                 # default: False
    # Per-stage one-line summaries from the underlying functions.
    verbose=True,                             # default: True
)

# ---- Outputs -----------------------------------------------------------
backbone = result.edgelist                    # final backboned projection
mapper   = result.id_mapper                   # paired IDMapper

# Per-stage telemetry (StageStats: name, duration_s, input_edges,
# output_edges, output_nodes).
for stage in result.stage_stats:
    print(stage)
# build_edgelist_from_frame                      15.29s |             0 →    23,280,338 edges,  1,877,850 nodes
# apply_backbone(bipartite_svn)                   1.92s |    23,280,338 →    12,634,460 edges,  1,877,828 nodes
# temporal_bipartite_to_unipartite              169.45s |    12,634,460 →   191,864,084 edges,    104,680 nodes
# apply_backbone(noise_corrected)               148.67s |   191,864,084 →    38,372,817 edges,    104,680 nodes

print(f"total: {result.total_duration_s:.1f}s | "
      f"final: {backbone.number_of_edges():,} edges, {backbone.n_nodes:,} nodes")

# With keep_intermediates=True, result.intermediates holds:
#   "bipartite"          : (EdgeList, IDMapper) after build
#   "bipartite_filtered" : (EdgeList, IDMapper) after bipartite_svn
#   "projection"         : (EdgeList, IDMapper) after temporal projection
```

**Memory modes** in practice — the three modes produce **byte-identical output**; they only differ in peak memory and wall-clock:

| Mode | Inter-stage cleanup | Within-call `streaming` (build + backbones) | Disk I/O | When to use |
| --- | --- | --- | --- | --- |
| `"fast"` | none | off | none | plenty of RAM, want max speed |
| `"balanced"` (default) | `del` + `gc.collect()` between stages; raw input released after build | on | none | the 80% case — moderately lower peak, ~30% slower stages |
| `"low"` | additionally checkpoint EdgeList to parquet between stages | on | a few seconds of parquet write+read | memory-constrained; targets the case where pipeline peak would otherwise be the sum of two overlapping stages |

**When to use this vs. direct calls**:
- *Use the wrapper* when you're running the canonical four-stage pipeline as-is and don't need to inspect intermediates between every step.
- *Skip the wrapper* when your workflow deviates (extra steps in between, different projection method, additional joins/filters between stages) — the lower-level functions remain the public API and compose freely.

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