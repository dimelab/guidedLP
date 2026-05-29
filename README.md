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

## Preprocessing: Raw Posts → Edge Lists

When the input is a `[sender, post, datetime]` post table (e.g. a social-media corpus) rather than a pre-built edge list, the `preprocessing` module turns that text column into the bipartite edge-list shape `build_graph_from_edgelist` expects. The four extractors share the same `[sender, content(, datetime)]` output schema, so they all drop straight into graph construction:

- **`extract_urls`** — one row per URL in `post`
- **`extract_domains`** — one row per URL, reduced to its host
- **`extract_keywords`** — words / tokens per sender, with optional NLP preprocessing and an opt-in RAKE keyphrase mode
- **`extract_embedding_features`** — semantic embeddings per post, mean-pooled per sender, exploded into one edge per embedding dimension

The first three run as a single vectorized pass over the Polars column (Rust regex engine, no Python row loops). All four preserve original sender IDs end-to-end and emit a long-form edge list. Posts with no matches (or no embedding) contribute no rows. `datetime_col=None` drops the timestamp column if you don't need it.

```python
import polars as pl
from guidedLP.preprocessing import extract_urls, extract_domains, extract_keywords

posts = pl.DataFrame({
    "sender": ["alice", "alice", "bob", "carol", "dave", "eve"],
    "post": [
        "Big read: https://nytimes.com/article?utm_source=twitter&fbclid=abc",
        "Same story: https://nytimes.com/article?utm_source=facebook",
        "no links here — just opinions on climate policy",
        "Watching https://youtu.be/abc123?si=tracker",
        "Same vid: https://www.youtube.com/watch?v=abc123&t=10",
        "HTTPS://EXAMPLE.com/News and (https://en.wikipedia.org/wiki/Foo).",
    ],
    "datetime": ["2024-01-01", "2024-01-02", "2024-01-03",
                 "2024-01-04", "2024-01-05", "2024-01-06"],
})
```

### URL and domain extraction with normalization

By default both URL extractors **normalize** each match before returning it. The normalization step lowercases scheme + host (preserving path case, which is case-sensitive in HTTP), strips well-known tracking parameters (`utm_*`, `fbclid`, `gclid`, `mc_cid`, `igshid`, etc.), collapses `youtu.be/<id>` and `youtube.com/watch?...&v=<id>` to a canonical `https://youtube.com/watch?v=<id>`, cleans up HTML-escaped `amp;`, and strips trailing sentence punctuation. This is the right default for graph construction: the same article shared with two different `utm_source=` codes would otherwise become two distinct URL nodes.

```python
urls = extract_urls(posts)
# Columns: [sender, datetime, url] — one row per URL.
# Alice's two nytimes URLs collapse to "https://nytimes.com/article".
# Carol's and Dave's YouTube links collapse to "https://youtube.com/watch?v=abc123".
# Eve's "HTTPS://EXAMPLE.com/News" becomes "https://example.com/News" (path case preserved).

domains = extract_domains(posts)
# Columns: [sender, datetime, domain] — host only, lowercased, leading "www." stripped.
# Both YouTube variants now share domain="youtube.com".
```

To keep the raw extracted URLs, pass `normalize=False`. To keep a leading `www.`, pass `strip_www=False` to `extract_domains`. The tracking-parameter list and per-step rules are documented in the `_normalize_url_expr` helper in `guidedLP/preprocessing/text_extraction.py`; the list is intentionally curated to exclude ambiguous single-letter and very generic names (`r`, `type`, `ref`, `src`, `feature`, …) that could be meaningful elsewhere.

### Keyword extraction with optional NLP preprocessing

`extract_keywords` defaults to *every word, aggregated per sender* — collapsing the raw N_posts × words_per_post mention count down to N_senders × vocab_size by emitting `[sender, keyword, mentions, first_seen]`. This is the memory-bounded form most downstream graph constructions want; for 10M posts × ~80 unique words per post, the difference between aggregated and long output is 2–4 orders of magnitude.

```python
# Default: all words, aggregated, lowercase, Unicode-aware tokenization.
words = extract_keywords(posts)
# Columns: [sender, keyword, mentions, first_seen]

# Filter to a fixed vocabulary — useful when you have a topic dictionary.
topics = extract_keywords(posts, keywords=["climate", "policy", "vaccine"])

# Add NLP preprocessing — requires `pip install "guidedLP[nlp]"`.
# `True` triggers auto-detection from a random post sample; pass an explicit
# ISO 639-1 code ("en", "da", …) to skip detection.
processed = extract_keywords(
    posts,
    stop_words=True,      # True | "en" | iterable of words | False (default)
    stem=True,            # True | "en" | False — Snowball stemmer
    lemmatize=False,      # True | "en" | False — simplemma lemmatizer
    min_word_length=3,    # drop one- and two-character tokens
)

# One row per mention (no aggregation) — useful for temporal analysis but the
# row count is total mentions, so use on small corpora or via `output="lazy"`.
mentions = extract_keywords(posts, output="long")

# Lazy mode returns a LazyFrame so you can stream straight to disk:
extract_keywords(posts, output="lazy").sink_parquet("words.parquet")
```

When NLP preprocessing is on, the user-supplied `keywords=` list is also stemmed/lemmatized before comparison, so `keywords=["climate"]` with `stem=True` still matches the stemmed form (`"climat"`) of words like *climate*, *climates*, *climatic* in the corpus.

### Semantic embedding features

`extract_embedding_features` is the "what *kind of content* does this sender post, semantically?" counterpart to the literal-text extractors. Each post is mapped to a fixed-dimensional embedding vector, those vectors are mean-pooled per sender, and the resulting per-sender vector is exploded into one edge per dimension (`dim_0`, `dim_1`, …). The output schema is `[sender, feature, weight(, first_seen)]` — the same shape the other extractors produce, just with continuous-valued weights instead of mention counts. Two senders who post about the same topics end up with high weights on the same `dim_*` features, so a bipartite projection of senders → dimensions yields a semantic similarity graph.

```python
from guidedLP.preprocessing import extract_embedding_features

# Path 1: pre-embedded posts (no optional dependency).
# `embedding_col` must be a List/Array column of equal-length numeric vectors.
posts_with_vecs = pl.DataFrame({
    "sender":    ["alice", "alice", "bob"],
    "embedding": [[0.1, 0.8, -0.2], [0.2, 0.7, -0.1], [-0.5, 0.1, 0.9]],
    "datetime":  ["2024-01-01", "2024-01-02", "2024-01-03"],
})
edges = extract_embedding_features(posts_with_vecs, embedding_col="embedding")
# Columns: [sender, feature, weight, first_seen]
# Three senders × three dims = up to 9 rows (alice's two posts mean-pooled to one vector).

# Path 2: encode posts from scratch with sentence-transformers.
# Requires `pip install "guidedLP[embeddings]"` — pulls in torch.
# Default model is multilingual MiniLM (384-dim, handles 50+ languages).
edges = extract_embedding_features(
    posts,
    model="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",  # default
    batch_size=64,
    device="cuda",                    # "cuda" / "cpu" / "mps" — None = auto
    save_path="cache/posts.npy",      # cache the encoded matrix on disk
    create_new=False,                 # reuse cache on subsequent runs
    aggregation="mean",               # "mean" / "sum" / "max"
    top_k=64,                         # keep each sender's top-64 features by |weight|
)
```

By default the function L2-normalizes every per-post vector before pooling and then shifts the aggregated values by `+2` so output weights lie in `[1, 3]` — positive, but the relative ordering and magnitudes of the original components are preserved exactly, so two senders that are component-wise *anti*-aligned end up with a weight gap that a downstream Jaccard / weighted projection registers correctly. Pass `weight_transform="abs"` to use pure magnitude (drops sign) or `weight_transform="raw"` to keep signed weights (downstream code must handle negatives). `top_k` and `min_weight` sparsify each sender's output before emitting edges, useful when the embedding is high-dimensional (384 dims × 1M senders = 384M edges if you keep everything).

The on-disk cache (`save_path`) is the big speedup when iterating: model inference dominates the from-scratch path, so caching the encoded matrix lets you re-run with different `aggregation` / `weight_transform` / `top_k` choices in seconds instead of minutes. The cache is normalize-agnostic — flipping `normalize_embeddings` doesn't require `create_new=True`.

```python
# Hand-off: senders ↔ embedding dimensions as a weighted bipartite.
graph, mapper = build_graph_from_edgelist(
    edges,
    source_col="sender", target_col="feature", weight_col="weight",
    bipartite=True,
)
user_graph, user_mapper = project_bipartite(
    graph, mapper, projection_mode="source", weight_method="jaccard",
)
# user_graph now connects senders whose semantic profiles overlap — feed to GLP.
```

### Hand-off to graph construction

The output of any extractor drops straight into `build_graph_from_edgelist` as a bipartite edge list — senders on one side, content (URL / domain / keyword) on the other. Project to either partition to run GLP downstream, or feed the bipartite directly through stat-user augmentation (Example 5) when labels live on the content side.

```python
from guidedLP.network import build_graph_from_edgelist, project_bipartite

# sender → domain bipartite edges
edges = extract_domains(posts)

bipartite, full_mapper = build_graph_from_edgelist(
    edges,
    source_col="sender", target_col="domain",
    bipartite=True,
)

# Project onto the user partition: two senders are linked if they shared
# a domain in common. Jaccard normalizes so power users don't dominate.
user_graph, user_mapper = project_bipartite(
    bipartite, full_mapper, projection_mode="source", weight_method="jaccard",
)
# user_graph is now ready for guided_label_propagation(...). See Examples 1–5
# for backboning, filtering, and projection variants that compose with this.
```

For real-world corpora, the typical chain is: `extract_domains` → `filter_graph(min_source_degree=…)` to drop low-activity senders → `apply_backbone(method="bipartite_svn")` to drop generic high-frequency domains → `project_bipartite` → unipartite backbone → GLP. Example 4 walks through that pipeline in its frame-native form (no graph until the end).

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

### 5. Labels on the Other Partition: Stat-User Augmentation

Examples 3 and 4 covered the standard bipartite case — seeds live on the same partition you want to propagate over (seeds are users, project to users, run GLP). But labels often live on the *other* partition. A curated list of left/right-leaning news outlets (content side), and you want to score *users* by what they engage with. Two options:

- **Run GLP on the bipartite directly with content seeds**, then keep the user rows. Simple, but pays the period-2 random-walk cost (user-to-user "distance" takes two iterations per hop).
- **Stat-user augmentation** (this example). Construct synthetic "stat user" nodes that live in the user partition and stay anchored to a label throughout propagation via the `(1−α)Y` term. GLP then operates cleanly on the user-user projection.

You own the anchor edges — schema is `source_id` (the synthetic anchor ID), `target_id` (the real user), `weight` (Float64). Build the frame however suits your data; `make_stat_user_edges` is a convenience that aggregates engagement-strength weights from a bipartite frame + a `{content: label}` dict. You also own the seed labels you hand to GLP, and the set of synthetic IDs you ask GLP to drop from the output via `exclude_from_output`.

```python
import polars as pl
from guidedLP.network.construction import build_graph_from_edgelist, project_bipartite
from guidedLP.network.backboning import apply_backbone
from guidedLP.glp import make_stat_user_edges, guided_label_propagation

# Bipartite engagement: columns user / outlet / weight
engagement = pl.read_csv("user_outlet_engagement.csv")

# Labels live on the *outlet* (content) side
outlet_seeds = {
    "nytimes.com": "left", "theguardian.com": "left",
    "wsj.com": "right", "foxnews.com": "right",
}

# ── Step 1: project to user-user ─────────────────────────────────────────
# project_bipartite expects source_id / target_id on frame input.
user_edges = project_bipartite(
    engagement.rename({"user": "source_id", "outlet": "target_id"}),
    projection_mode="source", weight_method="jaccard",
    output_format="dataframe",
)

# ── Step 2: backbone the user-user graph (BEFORE augmenting) ─────────────
# Filtering / backboning runs on real edges only; augmentation comes after,
# because stat users have an artificial degree profile (degree 1 on the
# bipartite, audience-size on the projection) that backboning methods
# aren't calibrated for.
user_edges = apply_backbone(
    user_edges, method="disparity", alpha=0.05, directed=False,
)

# ── Step 3: build the anchor edges ───────────────────────────────────────
# Convenience path — make_stat_user_edges aggregates per-(user, content)
# engagement weights into the `source_id / target_id / weight` schema and
# also returns a seeds dict and the synthetic node IDs.
stat_edges, stat_seeds, stat_ids = make_stat_user_edges(
    engagement, outlet_seeds,
    user_col="user", content_col="outlet", weight_col="weight",
)
# Or hand-build the same shape — you control the weights, edge set, and
# direction. The shape below is what the rest of the example consumes:
#
#   stat_edges = pl.DataFrame({
#       "source_id": ["__stat__nytimes.com", ...],
#       "target_id": ["u123", ...],
#       "weight":    [1.0, ...],            # any Float64 scale you want
#   })
#   stat_seeds = {"__stat__nytimes.com": "left", ...}
#   stat_ids   = {"__stat__nytimes.com", ...}

# ── Step 4: concat, build graph, propagate, drop stat users from output ──
augmented = pl.concat([user_edges, stat_edges])
graph, mapper = build_graph_from_edgelist(
    augmented, source_col="source_id", target_col="target_id", weight_col="weight",
)
result = guided_label_propagation(
    graph, mapper, stat_seeds, labels=["left", "right"],
    exclude_from_output=stat_ids,   # drops __stat__... rows from result
)
# `result` has one row per real user with left_prob / right_prob.
```

**Attaching anchor edges to an existing EdgeList.** When you already have an `EdgeList` (e.g. the projection from an earlier step), `EdgeList.attach(extra_df, mapper)` concatenates and re-encodes in one call — no manual decode / concat / re-encode:

```python
projected_el, projected_mapper = project_bipartite(
    bipartite_el, full_mapper, projection_mode="source", weight_method="jaccard",
)
augmented_el, augmented_mapper = projected_el.attach(stat_edges, projected_mapper)
graph, mapper = edgelist_to_graph(augmented_el, augmented_mapper)
```

**Doing the whole thing in the canonical pipeline.** The temporal-attribution pipeline (`run_canonical_pipeline`) accepts a `content_seeds` DataFrame in the same schema and attaches it between the projection and the projection backbone (Stage 3.5). Because the attach runs *before* the backbone, you control whether the synthetic edges survive — pass `protected_nodes=list(stat_ids)` or pick weights deliberately:

```python
from guidedLP.pipelines import run_canonical_pipeline

result = run_canonical_pipeline(
    engagement, source_col="user", target_col="outlet", timestamp_col="ts",
    weight_col="weight",
    content_seeds=stat_edges,           # same source_id/target_id/weight frame
    protected_nodes=list(stat_ids),     # keep anchors past noise_corrected
)
graph, mapper = edgelist_to_graph(result.edgelist, result.id_mapper)
labels = guided_label_propagation(
    graph, mapper, stat_seeds, ["left", "right"],
    exclude_from_output=stat_ids,
)
```

See `docs/architecture/glp.md` ("Bipartite Graphs & Stat-User Augmentation") for the design rationale (why ordering matters, what the synthetic edges actually compute) and `bipartite_glp_notes.md` at the repo root for related literature (Co-HITS, BiRank, personalized PageRank with concentrated teleportation).

### 6. End-to-End Pipeline Wrappers

For the canonical attribution workflow — raw input → bipartite EdgeList → bipartite-side backbone → temporal projection → projection-side backbone — `guidedLP.pipelines.run_canonical_pipeline` composes all four stages in a single call with explicit memory management between steps. Compared to calling the four functions by hand, the wrapper releases intermediates between stages so they don't co-exist in RAM, and optionally checkpoints to disk for memory-constrained runs.

```python
from guidedLP.pipelines import run_canonical_pipeline

result = run_canonical_pipeline(
    source="shares.parquet",
    source_col="user", target_col="item", timestamp_col="timestamp",
    weight_col="weight",
    intermediate_col="item", projected_col="user",
    min_source_degree=25,
    bipartite_alpha=0.01, bipartite_correction="fdr_bh",
    add_edge_weights=True, time_decay=False, presort_temporal=True,
    projection_target_fraction=0.2,
    memory_mode="balanced",          # "fast" / "balanced" / "low"
    verbose=True,
)

backbone = result.edgelist
mapper = result.id_mapper
print(f"total: {result.total_duration_s:.1f}s | "
      f"{backbone.number_of_edges():,} edges, {backbone.n_nodes:,} nodes")
```

With `verbose=True` the pipeline prints a per-stage one-liner plus a final TOTAL with memory mode and edge/node counts:

```
[build_edgelist_from_frame] 15.29s | 27,893,278 input rows → 1,877,850 nodes, 23,280,338 edges (UInt32)
[apply_backbone] 1.92s | method=bipartite_svn | EdgeList: 23,280,338 → 12,634,460 edges kept (54.3%)
[temporal_bipartite_to_unipartite] 169.45s | 12,634,460 input rows → 191,864,084 projection edges
[apply_backbone] 148.67s | method=noise_corrected | EdgeList: 191,864,084 → 38,372,817 edges kept (20.0%)
[run_canonical_pipeline] TOTAL 335.50s | mode=balanced | final: 38,372,817 edges, 104,680 nodes
```

`result.stage_stats` exposes the same data programmatically as `StageStats(name, duration_s, input_edges, output_edges, output_nodes)`. With `keep_intermediates=True`, `result.intermediates` holds `(EdgeList, IDMapper)` snapshots after build, after `bipartite_svn`, and after the temporal projection.

**Memory modes** — all three produce byte-identical output; they only differ in peak memory and wall-clock:

| Mode | Inter-stage cleanup | Disk I/O | When to use |
|---|---|---|---|
| `"fast"` | none | none | plenty of RAM, want max speed |
| `"balanced"` (default) | `del` + `gc.collect()` between stages | none | the 80% case — moderately lower peak, ~30% slower stages |
| `"low"` | additionally checkpoints each EdgeList to parquet between stages | a few seconds | memory-constrained runs |

Use the wrapper when running the canonical four-stage pipeline as-is. Skip it when your workflow deviates — extra steps, different projection method, additional joins/filters between stages — and call the lower-level functions directly. The wrapper also accepts a `content_seeds` DataFrame to attach stat-user anchor edges between projection and projection backbone (see Example 5). When the directed citation projection is the wrong shape for your analysis, see the undirected variant below.

#### Undirected variant — `run_undirected_bipartite_pipeline`

When you want a symmetric co-occurrence graph (e.g. Jaccard similarity between users who share content) rather than directed citation attribution, use the undirected sibling. Same four-stage shape — bipartite EdgeList → `bipartite_svn` backbone → projection → `noise_corrected` backbone — but stage 3 swaps `temporal_bipartite_to_unipartite` for `project_bipartite`, so the output is undirected and weighted by a topological similarity (`"jaccard"`, `"count"`, or `"overlap"`).

```python
from guidedLP.pipelines import run_undirected_bipartite_pipeline

result = run_undirected_bipartite_pipeline(
    source="shares.parquet",
    source_col="user", target_col="item",
    projection_mode="source",                  # collapse items, keep users
    projection_weight_method="jaccard",        # "count" | "jaccard" | "overlap"
    min_source_degree=25,
    bipartite_alpha=0.01, bipartite_correction="fdr_bh",
    projection_target_fraction=0.2,
    memory_mode="balanced",
    verbose=True,
)

backbone = result.edgelist                     # undirected, weighted EdgeList
mapper = result.id_mapper
print(f"{backbone.number_of_edges():,} edges, {backbone.n_nodes:,} nodes")
```

Key differences vs `run_canonical_pipeline`:

- **No temporal inputs.** No `timestamp_col`, `weight_col`, `time_decay`, or `presort_temporal` — `project_bipartite` computes edge weights from shared-neighbor topology, ignoring per-edge weights and timestamps.
- **Default weight is Jaccard.** `projection_weight_method="jaccard"` is bounded in `[0, 1]`, which tends to behave better under `noise_corrected` backboning and downstream GLP than the raw `"count"` weights.
- **Output is undirected** → run downstream GLP with `directional=False` (a single `predictions_df`, not the `(out_df, in_df)` tuple the canonical pipeline produces).
- **`content_seeds` emits each anchor edge once.** The undirected projection treats both orientations as equivalent, so the `forward + reverse` mirroring used in Example 5 isn't needed here — supply `("__lbl_left", "u1", 1.0)` and you're done.

Memory modes (`"fast"` / `"balanced"` / `"low"`), `result.stage_stats`, `keep_intermediates`, `protected_nodes`, and the optional `content_seeds` stage all work identically to the canonical wrapper above — the only knobs that change are the stage-3 ones (`projection_mode`, `projection_weight_method`). Use this pipeline for symmetric similarity / community-style analyses; use `run_canonical_pipeline` when later-sharer → earlier-sharer attribution direction matters (PageRank / HITS-flavored analyses).

### 7. Evaluating GLP Quality with Held-Out Seeds

`train_test_split_validation` is the standard "is my model good?" check for GLP. It holds out a fraction of your labelled seeds, trains GLP on the rest, and scores predictions on the held-out portion. The returned dict carries accuracy, per-label precision / recall / F1, a confusion matrix, and the sklearn classification report — everything needed for a results section.

```python
from guidedLP.network.construction import build_graph_from_edgelist
from guidedLP.glp.validation import train_test_split_validation

graph, id_mapper = build_graph_from_edgelist(
    pl.read_csv("political_network.csv"),
    source_col="follower", target_col="following",
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
    random_seed=42,
    alpha=0.85,               # **glp_kwargs — threaded into GLP
    directional=False,
)

print(f"Accuracy: {results['accuracy']:.3f}  Macro-F1: {results['macro_f1']:.3f}")
print(results["classification_report"])   # sklearn-formatted per-label table
print(results["confusion_matrix"])        # rows=true, cols=predicted (np.ndarray)
```

`results["test_predictions"]` is the raw GLP frame restricted to the held-out seeds — cross-reference against the input seed dict to drill into individual errors.

**Custom test set instead of a random split.** When you have a separately-curated ground-truth set (e.g. hand-labelled accounts you specifically *don't* want in training), pass it via `test_seeds`. Overlapping IDs are pulled out of training; on label conflicts the test set's label wins (with a warning). `test_size`, `stratify`, and `random_seed` are ignored in this mode.

```python
known_holdout = {"@verified_left_1": "progressive", "@verified_right_1": "conservative"}
results = train_test_split_validation(
    graph=graph, id_mapper=id_mapper,
    seed_labels=political_seeds,        # train on these
    test_seeds=known_holdout,           # evaluate on these (curated ground truth)
    labels=["progressive", "conservative"],
)
```

**Small seed sets?** A single held-out split is noisy with only a handful of seeds per label. `cross_validate` runs K-fold over the seeds and returns mean ± std for each metric — same kwargs, more stable estimates:

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

**Validate with ensembling instead of single-run GLP.** Both validators accept a `propagator` kwarg. Default is `guided_label_propagation`; pass `ensemble_label_propagation` to score each fold with a bagged, noise-resampled ensemble. Propagator-specific kwargs (`n_epochs`, `base_seed`, `enable_noise_category`, …) ride along through `**glp_kwargs`. For directional GLP, pick which pass to score with `directional_pass="out"` or `directional_pass="in"` — the validator raises if you don't pick (no implicit choice).

### 8. Refining GLP Output

Three controls layered on top of `guided_label_propagation`, each addressing a different failure mode of the default pipeline. Use them individually or compose them.

**Edge-weight compression.** When a small fraction of edges have weights orders of magnitude larger than the rest (viral retweets, hub co-occurrences, runaway co-citations), those edges dominate the propagation regardless of `alpha`. The `weight_transform` parameter applies a per-edge callable when the transition matrix is built, so you can dampen weight outliers without rebuilding the graph or pre-normalizing upstream.

```python
from guidedLP.glp import (
    guided_label_propagation,
    tanh_transform, log1p_transform, winsorize_transform,
)

# Pick a transform that matches the shape of your weight distribution:
# - log1p_transform()     : gentle, unbounded. Good first choice when weights span orders of magnitude.
# - winsorize_transform(c): hard cap at threshold c. Use when you know which weight value is "too high."
# - tanh_transform()      : S-curve saturation; mimics the historical stlp transform. Most aggressive.

results = guided_label_propagation(
    graph, id_mapper, seeds, labels,
    weight_transform=log1p_transform(),
)
```

Any positive-output callable works — `weight_transform=lambda w: math.log10(w + 1.0)` is fine. The only constraint is `f(w) > 0` for `w > 0` (a transform mapping positive weights to zero looks like graph isolates to the propagation engine).

**Audience-composition pass.** Directional GLP (`directional=True`) returns a `(forward, backward)` tuple where the backward pass is a full propagation on `Aᵀ` from the original seeds — answering *"is node n upstream of a seed?"*. A semantically different question — *"what's the label profile of the nodes pointing at n?"* — requires a one-hop aggregation of the converged forward result along incoming edges. `audience_composition_pass` does exactly that:

```python
from guidedLP.glp import audience_composition_pass

# Forward pass first (directional=False — audience pass consumes a single result, not the tuple).
fwd = guided_label_propagation(graph, id_mapper, seeds, labels, directional=False)

# Audience pass — single sparse matmul, no iteration.
audience = audience_composition_pass(graph, id_mapper, fwd, labels)
# audience["left_prob"] reads as the in-degree-weighted average of forward-pass
# left_prob across n's in-neighbors — "what fraction of accounts pointing at n
# were forward-labeled left?"
```

Use for audience studies, reception analysis, or recommender-system diagnostics. Skip on undirected graphs (the pass raises) or when you specifically want the upstream-reachability question the `Aᵀ` pass already answers. If the forward pass used a `weight_transform`, pass the same one to `audience_composition_pass`. Nodes with zero in-degree fall back to a uniform distribution.

**Stochastic ensembling.** When `enable_noise_category=True`, the noise seeds are sampled randomly from non-seed nodes — a single GLP run is sensitive to which nodes happen to be chosen. `ensemble_label_propagation` runs GLP `n_epochs` times with different noise samples and averages the result:

```python
from guidedLP.glp import ensemble_label_propagation

result = ensemble_label_propagation(
    graph, id_mapper, seeds, labels,
    n_epochs=20,
    base_seed=42,                  # per-epoch seed = base_seed + epoch_index
    enable_noise_category=True,    # required — otherwise ensembling is a no-op
    noise_ratio=0.3,
    return_variance=True,          # adds {label}_prob_std columns for confidence intervals
)
```

Same return shape as `guided_label_propagation` (single DataFrame for undirected / `directional=False`; `(out_df, in_df)` tuple for directed + `directional=True`). Probability columns are averaged across epochs; with `return_variance=True` each `{label}_prob` is paired with `{label}_prob_std` (sample std with Bessel's correction). `dominant_label` and `confidence` are recomputed from averaged probabilities, not voted across epochs — averaging gives the variance reduction; voting would lose it. A node can have a different dominant label in the ensemble than in any single epoch; this is expected. `is_seed` reflects only user-supplied seeds, not per-epoch noise samples.

**Composing the three.** The controls are independent and combine cleanly:

```python
forward_ensemble = ensemble_label_propagation(
    graph, id_mapper, seeds, labels,
    n_epochs=20, directional=False,
    weight_transform=log1p_transform(),
    enable_noise_category=True, noise_ratio=0.3,
)
audience = audience_composition_pass(
    graph, id_mapper, forward_ensemble, labels + ["noise"],
    weight_transform=log1p_transform(),
)
```

### 9. Temporal Network Analysis

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

