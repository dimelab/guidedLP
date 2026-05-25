# Guided Label Propagation (GLP)

Large-scale network analysis with semi-supervised community detection for computational social science research.

## Overview

This project provides efficient network analysis capabilities with a focus on **Guided Label Propagation (GLP)**, a novel semi-supervised community detection technique. Unlike traditional unsupervised methods that find arbitrary clusters, GLP identifies how unknown nodes in a network relate to predefined categories of interest (e.g., political affiliation, brand preference, topic relevance).


## Installation

### Prerequisites

- Python 3.9 or higher
- Git (for development installation)

### Install

```bash
# Clone the repository
git clone https://github.com/alterpublics/guided-label-propagation.git
cd guided-label-propagation/guidedLP

# Install in development mode
pip install -e .

# Or install with optional extras (dev tooling, docs, visualization)
pip install -e ".[dev]"
pip install -e ".[docs]"
pip install -e ".[viz]"
pip install -e ".[dev,docs,viz]"   # all three
```

### Verify Installation

```bash
# Run the post-install smoke test from the repo root
python test_installation.py

# Or just check the package imports
python -c "import guidedLP; print('Installation successful!')"
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
# convenient. They all get normalized internally to {node_id: label}.

# (1) node_id -> label  (canonical)
seeds = {"user123": "progressive", "user321": "conservative"}

# (2) label -> [node_ids]
# seeds = {
#     "progressive": ["user123", "user456", "user789"],
#     "conservative": ["user321", "user654", "user987"],
# }

# (3) polars.DataFrame with `node_id` and `label` columns
# seeds = pl.read_csv("seeds.csv")

# (4) pandas.DataFrame with `node_id` and `label` columns
# import pandas as pd
# seeds = pd.read_csv("seeds.csv")

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

## Graph, DataFrame, or EdgeList: pick your shape

`apply_backbone`, `filter_graph`, and `project_bipartite` all accept **either** a NetworkIt graph, a Polars edge frame (columns `source_id`, `target_id`, `weight`), or a coded **EdgeList**. The output type defaults to matching the input but can be forced with `output_format="graph"`, `output_format="dataframe"`, or `output_format="edgelist"`.

Supported combos:

| Input → Output | When to use |
|---|---|
| `graph → graph` (default for graph in) | Existing workflows — nothing changes. |
| `graph → dataframe` / `graph → edgelist` | Inspect or chain a result without rebuilding. |
| `frame → frame`  (default for frame in) | Stay in dataframe land — chain filtering, backboning, projection without ever building a graph. Faster when you don't need NetworkIt's traversals between steps. |
| `edgelist → edgelist` (default for edgelist in) | Hub-heavy bipartite projection at scale. The coded path is ~3–7× more memory-efficient than the dataframe path on hub-heavy data, and handles projections (~200M edges) that OOM the graph path. |
| `frame → graph` | Not supported — call `build_graph_from_edgelist()` on the returned frame instead. |

`EdgeList` is a small wrapper around a Polars DataFrame whose `src` / `tgt` columns hold `UInt32` codes (NetworkIt-style internal IDs) plus an optional `weight`. It pairs with an `IDMapper` for translation back to original IDs. Use it when the raw frame is too large to keep around as strings, or when you're about to do a bipartite projection big enough that the legacy `Dict[Any, Set[Any]]` neighbor map matters.

Utilities to bridge the three worlds:

- `build_graph_from_edgelist(df, …)` — frame → graph (composes through `build_edgelist_from_frame` + `edgelist_to_graph` internally)
- `build_edgelist_from_frame(df, …)` — frame → EdgeList (same pre-processing as the graph builder, but skips the NetworkIt build)
- `edgelist_to_graph(el, mapper)` — EdgeList → graph (cheap; codes are already internal IDs)
- `graph_to_edgelist(graph, mapper)` — graph → EdgeList (cheap; sibling of `graph_to_edges`)
- `graph_to_edges(graph, mapper)` — graph → frame (originals, ready for the dataframe path)

Functions that genuinely *need* graph traversal (`filter_by_seed_proximity`, the `"giant_component_only"` / `"centrality"` filters in `filter_graph`, all GLP propagation and centrality algorithms) only accept graph input; they'll raise a clear error if handed a frame or EdgeList, naming the conversion utility to call.

## Examples

### 1. Backboning Before Propagation

Both **bipartite** and **unipartite** graphs can be backboned, but the statistical test should match the graph type:

- **Bipartite** (users ↔ items) — use `method="bipartite_svn"` (Tumminello et al. 2011). For each edge, tests whether the observed weight exceeds the expectation under a configuration-model null preserving node strengths. Filters out edges to generic high-degree items naturally, with no degree threshold to tune.
- **Unipartite weighted** (user ↔ user, etc.) — use `method="disparity"` (Serrano et al. 2009). For each edge, tests whether its weight is statistically over-represented relative to its endpoints' other ties. Requires meaningful weight variance.

Apply backboning when the graph has many incidental edges that dilute propagation, or when you need to cut compute time on a very large network. Skip it for already-sparse graphs and when weak edges *are* the signal.

```python
from guidedLP.network.construction import build_graph_from_edgelist, project_bipartite
from guidedLP.network.backboning import apply_backbone
from guidedLP.glp.propagation import guided_label_propagation
from guidedLP.glp.utils import check_seed_coverage

# ── BIPARTITE: user ↔ hashtag, then project onto users ────────────────────
bipartite, full_mapper = build_graph_from_edgelist(
    edges, source_col="user", target_col="hashtag", weight_col="count",
    bipartite=True,
)
# Filter generic hashtags out of the bipartite layer BEFORE projecting,
# so the projection step gets cheaper too.
backbone, backbone_mapper = apply_backbone(
    bipartite, full_mapper,
    method="bipartite_svn",
    alpha=0.05,
    correction="fdr_bh",   # Benjamini-Hochberg FDR (default); scales to |E| in the millions.
                           # "bonferroni" is too conservative at scale; "none" disables correction.
)
user_graph, user_mapper = project_bipartite(
    backbone, backbone_mapper, projection_mode="source", weight_method="jaccard",
)

# ── UNIPARTITE: weighted retweet network ──────────────────────────────────
# Disparity filter expects meaningful weight variance.
graph, id_mapper = build_graph_from_edgelist(edges, "user_a", "user_b", "weight")
backbone, backbone_mapper = apply_backbone(
    graph, id_mapper, method="disparity", alpha=0.05,
)

# ── Common follow-up: check seed survival, then propagate ─────────────────
seeds = {"@aoc": "left", "@realdonaldtrump": "right"}
report = check_seed_coverage(backbone_mapper, seeds)
print(f"Seeds surviving backbone: {report['train']['present']}/{report['train']['total']}")
seeds = {k: v for k, v in seeds.items() if backbone_mapper.has_original(k)}

results = guided_label_propagation(
    graph=backbone, id_mapper=backbone_mapper,
    seed_labels=seeds, labels=["left", "right"],
)
```

Other methods: `method="weight"` (threshold + `target_edges=N`) or `method="degree"` (keep top-N by degree + `target_nodes=N`) — simpler but require manual tuning.

### 2. Seed-Centered Filtering Before Propagation

`filter_by_seed_proximity` prunes a graph to a neighborhood around your seed set — the local-relevance counterpart to backboning. Useful when seeds cover one part of a much larger graph and you want to run GLP on a manageable region without losing local structure.

Three methods, all returning `(graph, id_mapper)` so they can be chained:

| Method | What it does | Pick when |
|---|---|---|
| `"khop"` (default) | BFS up to `hops` levels from the seed set. | You want a predictable, bounded neighborhood. Watch hub explosion at hops≥2. |
| `"ppr"` | Personalized PageRank from seeds; keep `top_n` and/or above `min_ppr`. | You're about to run GLP — same `αPF + (1−α)Y` kernel, so the filter is internally consistent with the propagation. |
| `"lte"` | NetworkIt's `LocalTightnessExpansion` from the seed set. | Seeds sit inside a tight community and you want the filter to find its boundary. |

```python
from guidedLP.network.filtering import filter_by_seed_proximity

seeds = ["@aoc", "@berniesanders", "@realdonaldtrump", "@tedcruz"]

# k-hop
g_sub, m_sub = filter_by_seed_proximity(graph, mapper, seeds, method="khop", hops=2)

# PPR with a top-N cap
g_sub, m_sub = filter_by_seed_proximity(graph, mapper, seeds, method="ppr", top_n=5000)

# Chain: cap hub explosion with k-hop, then trim to the tight core with LTE
g_hop, m_hop  = filter_by_seed_proximity(graph, mapper, seeds, method="khop", hops=3)
g_core, m_core = filter_by_seed_proximity(g_hop, m_hop, seeds, method="lte")
```

Seeds accept a list/tuple/set or a polars DataFrame (column name via `seed_column`, default `"node_id"`). For directed graphs, `direction={"out", "in", "both"}` controls edge following for `"khop"` and `"ppr"`. The returned graph always has contiguous internal IDs, so it's ready to feed straight into GLP.

### 3. Bipartite to Unipartite Projection

Many computational social science datasets are naturally **bipartite** — users connected to hashtags, authors to papers, accounts to URLs. GLP runs on a unipartite graph, so the usual preprocessing is to project onto the partition where your seeds live: two users get connected if they share items in the other partition.

```python
from guidedLP.network.construction import build_graph_from_edgelist, project_bipartite

# Bipartite edge list: user → hashtag uses
bipartite, full_mapper = build_graph_from_edgelist(
    edges, source_col="user", target_col="hashtag", weight_col="count",
    bipartite=True,
)

# Project onto users. jaccard normalizes so a power user doesn't look
# similar to everyone; use "count" for raw shared-neighbor count, or
# "overlap" (|A∩B|/min(|A|,|B|)) when one set is contained in the other.
user_graph, user_mapper = project_bipartite(
    bipartite, full_mapper, projection_mode="source", weight_method="jaccard",
)

# Seeds must refer to nodes in the projected partition (here: users).
# The new mapper contains only users — verify with check_seed_coverage(user_mapper, seeds).
```

**For hub-heavy bipartites at scale** — datasets where a few popular intermediate items (a viral hashtag, a popular URL) are touched by thousands of users — switch to the `EdgeList` path. The projection edge count can explode by orders of magnitude through a single hub, and the coded path delivers ~3–7× peak-RSS reduction. On a 20K-user × 2K-item Zipf-popular bipartite (~240K input edges → 199M projection edges), the legacy graph path runs out of memory; the EdgeList path completes cleanly.

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

For temporal bipartite data where edge order matters (A shared an item before B → B may attribute back to A), use `temporal_bipartite_to_unipartite` instead. It produces a *directed* unipartite graph using **citation convention**: edges point from later sharer → earlier sharer, so PageRank / HITS-Authority naturally surface the original sources. The function trusts your row order — pre-sort the edgelist by intermediate column ascending, then timestamp descending.

### 4. Frame-Native Pipeline (No Graph Until the End)

For real-world social-media style data — bipartite edges (users ↔ hashtags / URLs / domains) that you need to filter, backbone, project, and re-backbone before feeding to GLP — keep everything on a Polars frame and only materialise a NetworkIt graph at the very last step. Each frame-to-graph round trip costs an `iterEdges` walk in one direction and an `addEdge` loop in the other; on a million-edge network those loops dominate every per-stage runtime. Skipping them between stages is the whole point of the dual-input API.

The pipeline below mirrors a common social-CSS preprocessing chain — **prune low-activity users → bipartite SVN → project to user-user → unipartite disparity → GLP**. Every stage operates on a frame; the only graph built is the final user-user graph that GLP consumes.

```python
import polars as pl
from guidedLP.network.filtering import filter_graph
from guidedLP.network.backboning import apply_backbone
from guidedLP.network.construction import build_graph_from_edgelist, project_bipartite
from guidedLP.glp.propagation import guided_label_propagation
from guidedLP.glp.utils import check_seed_coverage

# Load raw bipartite edges. Whatever column names you already have, just
# normalise to source_id / target_id / weight (the schema all frame-mode
# functions in this library expect).
edges = pl.read_csv("user_hashtag_counts.csv").rename({
    "user": "source_id",
    "hashtag": "target_id",
    "count": "weight",
})

# ── Step 1: min_source_degree filter ──────────────────────────────────────
# Drop users with fewer than 5 distinct hashtags. The matching
# *_target_degree filters exist too, for trimming low-activity hashtags.
edges = filter_graph(edges, filters={"min_source_degree": 5})

# ── Step 2: bipartite backbone ────────────────────────────────────────────
# Remove edges to generic high-frequency hashtags. Returns a lean frame
# (source_id / target_id / weight, kept rows only) — score columns are
# dropped to keep the chain memory-light. Pass include_scores=True if you
# want the diagnostic columns.
edges = apply_backbone(
    edges,
    method="bipartite_svn", alpha=0.05, correction="fdr_bh",
    directed=False,
)

# ── Step 3: project bipartite → unipartite ────────────────────────────────
# Connect users who share hashtags. Jaccard normalises so a power user
# doesn't end up artificially similar to everyone.
user_edges = project_bipartite(
    edges,
    projection_mode="source", weight_method="jaccard",
)

# ── Step 4: unipartite backbone ───────────────────────────────────────────
# A second backbone on the projection is good practice — the projection
# step inflates edge counts (every shared hashtag becomes a user-user edge)
# and disparity prunes the resulting noise.
user_edges = apply_backbone(
    user_edges,
    method="disparity", alpha=0.05, directed=False,
)

# ── Step 5: build the graph (only now!) and run GLP ───────────────────────
user_graph, user_mapper = build_graph_from_edgelist(
    user_edges,
    source_col="source_id", target_col="target_id", weight_col="weight",
)

seeds = {"@aoc": "left", "@berniesanders": "left",
         "@realdonaldtrump": "right", "@tedcruz": "right"}

# Some seeds may not survive the filtering chain; drop missing ones first.
report = check_seed_coverage(user_mapper, seeds)
print(f"Seeds surviving the pipeline: "
      f"{report['train']['present']}/{report['train']['total']}")
seeds = {k: v for k, v in seeds.items() if user_mapper.has_original(k)}

results = guided_label_propagation(
    graph=user_graph, id_mapper=user_mapper,
    seed_labels=seeds, labels=["left", "right"],
)
```

**Mixing shapes within a pipeline.** If you already have a graph but want to peek at an intermediate result as a frame, pass `output_format="dataframe"`. The default return is a lean frame (only `source_id` / `target_id` / `weight`, kept rows only); add `include_scores=True` when you want the diagnostic columns to inspect:

```python
# Lean — ready to chain into the next pipeline stage.
edges_df = apply_backbone(
    graph, id_mapper, method="disparity", alpha=0.05,
    output_format="dataframe",
)

# Full diagnostic frame — all edges with score columns and the `kept` boolean.
diag_df = apply_backbone(
    graph, id_mapper, method="disparity", alpha=0.05,
    output_format="dataframe", include_scores=True,
)
strongest = diag_df.filter(pl.col("kept")).sort("alpha_score").head(20)
```

And to go the other way — from an existing graph into the dataframe pipeline — use `graph_to_edges`:

```python
from guidedLP.network.construction import graph_to_edges

edges = graph_to_edges(graph, id_mapper)   # source_id, target_id, weight (originals)
edges = apply_backbone(edges, method="noise_corrected", threshold=1.0, directed=False)
```

### 5. Political Affiliation Analysis

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

### 6. Temporal Network Analysis

```python
# Track community evolution over time
from guidedLP.timeseries.slicing import create_temporal_slices
from guidedLP.timeseries.temporal_metrics import extract_temporal_metrics

# Load temporal network data
temporal_data = pl.read_csv("tests/fixtures/sample_temporal.csv")

# Create time slices (daily). slice_interval also accepts "weekly"/"monthly"/"yearly".
time_slices = create_temporal_slices(
    temporal_data,
    timestamp_col="timestamp",
    slice_interval="daily",
)

seeds = {"@aoc": "left", "@realdonaldtrump": "right"}

# Analyze each time slice
for slice_date, slice_graph, slice_mapper in time_slices:
    results = guided_label_propagation(
        graph=slice_graph,
        id_mapper=slice_mapper,
        seed_labels=seeds,
        labels=["left", "right"],
    )
    print(f"{slice_date}: {len(results)} nodes classified")
```



## System Requirements

### Core Dependencies
- **Python**: 3.9 or higher
- **NetworkIt**: 11.0+ (C++ graph library for performance)
- **Polars**: 0.20.0+ (Fast DataFrame operations)
- **NumPy**: 1.24.0+ (Numerical computing)
- **SciPy**: 1.10.0+ (Sparse matrices and scientific computing)

### Performance Notes
- Minimum 8GB RAM recommended for networks with >10,000 nodes
- SSD storage recommended for large temporal datasets
- Multi-core CPU beneficial for parallel operations


## License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

