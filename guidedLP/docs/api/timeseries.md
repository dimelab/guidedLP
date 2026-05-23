# Timeseries Module API Reference

The timeseries module provides functionality for temporal network analysis, including time-slicing, temporal metrics calculation, and dynamic community tracking.

## Module: `src.timeseries.slicing`

### create_temporal_slices()

```python
def create_temporal_slices(
    edgelist: Union[str, pl.DataFrame],
    timestamp_col: str = "timestamp",
    source_col: str = "source",
    target_col: str = "target",
    weight_col: Optional[str] = None,
    slice_duration: str = "1d",
    slice_method: str = "sliding",
    overlap: Optional[str] = None,
    start_time: Optional[Union[str, datetime]] = None,
    end_time: Optional[Union[str, datetime]] = None
) -> Dict[datetime, pl.DataFrame]
```

Create time-sliced edge lists from temporal network data.

**Parameters:**
- `edgelist`: Path to CSV file or DataFrame with timestamped edges
- `timestamp_col`: Name of the timestamp column
- `source_col`: Name of the source node column
- `target_col`: Name of the target node column  
- `weight_col`: Name of the edge weight column (optional)
- `slice_duration`: Duration of each time slice ("1h", "1d", "1w", "1M")
- `slice_method`: Slicing method ("sliding", "tumbling", "cumulative")
- `overlap`: Overlap duration for sliding windows (e.g., "12h")
- `start_time`: Analysis start time (ISO format or datetime)
- `end_time`: Analysis end time (ISO format or datetime)

**Returns:**
- Dictionary mapping slice timestamps to edge DataFrames

**Slice Methods:**
- **Sliding**: Overlapping time windows that slide by a fixed interval
- **Tumbling**: Non-overlapping adjacent time windows
- **Cumulative**: Growing windows that include all edges from start to slice time

**Examples:**

```python
import polars as pl
from src.timeseries.slicing import create_temporal_slices
from datetime import datetime

# Load temporal network data
temporal_edges = pl.read_csv("temporal_network.csv")
print(temporal_edges.head())
# timestamp           source  target  weight
# 2024-01-01 10:30:00  A       B       1.5
# 2024-01-01 14:20:00  B       C       2.0
# 2024-01-02 09:15:00  A       C       1.0

# Daily tumbling windows
daily_slices = create_temporal_slices(
    temporal_edges,
    timestamp_col="timestamp",
    slice_duration="1d",
    slice_method="tumbling"
)

print(f"Created {len(daily_slices)} daily slices")
for date, edges in daily_slices.items():
    print(f"  {date.date()}: {len(edges)} edges")

# Hourly sliding windows with 30-minute overlap
hourly_slices = create_temporal_slices(
    temporal_edges,
    slice_duration="1h",
    slice_method="sliding",
    overlap="30m"
)

# Weekly cumulative growth
cumulative_slices = create_temporal_slices(
    temporal_edges,
    slice_duration="1w", 
    slice_method="cumulative",
    start_time="2024-01-01",
    end_time="2024-02-01"
)
```

### create_temporal_graphs()

```python
def create_temporal_graphs(
    edgelist: Union[str, pl.DataFrame],
    timestamp_col: str = "timestamp",
    slice_duration: str = "1d",
    graph_params: Optional[Dict[str, Any]] = None
) -> Dict[datetime, Tuple[nk.Graph, IDMapper]]
```

Create time-sliced NetworkIt graphs from temporal data.

**Examples:**

```python
from src.timeseries.slicing import create_temporal_graphs

# Create daily network snapshots
daily_graphs = create_temporal_graphs(
    temporal_edges,
    timestamp_col="timestamp",
    slice_duration="1d",
    graph_params={
        "directed": True,
        "auto_weight": True,
        "allow_self_loops": False
    }
)

# Analyze each time slice
for date, (graph, mapper) in daily_graphs.items():
    density = graph.density()
    components = nk.components.ConnectedComponents(graph).run().numberOfComponents()
    print(f"{date.date()}: {graph.numberOfNodes()} nodes, density={density:.3f}, components={components}")
```

## Module: `src.timeseries.temporal_metrics`

### calculate_temporal_metrics()

```python
def calculate_temporal_metrics(
    temporal_graphs: Dict[datetime, Tuple[nk.Graph, IDMapper]],
    metrics: List[str] = ["degree", "betweenness", "closeness"],
    node_subset: Optional[List[Any]] = None,
    normalize: bool = True
) -> pl.DataFrame
```

Calculate network metrics across time slices for temporal analysis.

**Parameters:**
- `temporal_graphs`: Dictionary of time-sliced graphs from `create_temporal_graphs()`
- `metrics`: List of centrality measures to calculate over time
- `node_subset`: Specific nodes to analyze (None for all nodes)
- `normalize`: Whether to normalize centrality scores

**Returns:**
- DataFrame with columns: node_id, timestamp, metric1, metric2, ...

**Examples:**

```python
from src.timeseries.temporal_metrics import calculate_temporal_metrics

# Calculate metrics over time
temporal_metrics = calculate_temporal_metrics(
    daily_graphs,
    metrics=["degree", "betweenness", "closeness"],
    normalize=True
)

print(temporal_metrics.head())
# node_id  timestamp           degree  betweenness  closeness
# A        2024-01-01 00:00:00  0.67    0.33        0.75
# A        2024-01-02 00:00:00  0.50    0.25        0.60
# B        2024-01-01 00:00:00  1.00    0.67        1.00

# Focus on specific influential nodes
key_nodes = ["influencer_1", "influencer_2", "celebrity_user"]
key_metrics = calculate_temporal_metrics(
    daily_graphs,
    metrics=["degree", "pagerank"],
    node_subset=key_nodes
)

# Analyze temporal patterns
import matplotlib.pyplot as plt

for node in key_nodes:
    node_data = key_metrics.filter(pl.col("node_id") == node)
    plt.plot(node_data["timestamp"], node_data["degree"], label=node)

plt.title("Degree Centrality Over Time")
plt.xlabel("Date")
plt.ylabel("Degree Centrality")
plt.legend()
plt.show()
```

### track_network_evolution()

```python
def track_network_evolution(
    temporal_graphs: Dict[datetime, Tuple[nk.Graph, IDMapper]],
    track_metrics: List[str] = ["nodes", "edges", "density", "components"]
) -> pl.DataFrame
```

Track global network properties over time.

**Examples:**

```python
from src.timeseries.temporal_metrics import track_network_evolution

# Track network growth and evolution
evolution_df = track_network_evolution(
    daily_graphs,
    track_metrics=["nodes", "edges", "density", "components", "clustering"]
)

print(evolution_df)
# timestamp           nodes  edges  density  components  clustering
# 2024-01-01 00:00:00   50     75    0.065      3         0.42
# 2024-01-02 00:00:00   52     82    0.062      2         0.38
# 2024-01-03 00:00:00   55     95    0.064      1         0.41

# Plot network growth
import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 2, figsize=(12, 8))

axes[0,0].plot(evolution_df["timestamp"], evolution_df["nodes"])
axes[0,0].set_title("Network Size")
axes[0,0].set_ylabel("Number of Nodes")

axes[0,1].plot(evolution_df["timestamp"], evolution_df["density"])
axes[0,1].set_title("Network Density")
axes[0,1].set_ylabel("Density")

axes[1,0].plot(evolution_df["timestamp"], evolution_df["components"])
axes[1,0].set_title("Connectivity")
axes[1,0].set_ylabel("Components")

axes[1,1].plot(evolution_df["timestamp"], evolution_df["clustering"])
axes[1,1].set_title("Clustering")
axes[1,1].set_ylabel("Clustering Coefficient")

plt.tight_layout()
plt.show()
```

## Module: `src.timeseries.category_analysis`

### analyze_cross_category_connections()

```python
def analyze_cross_category_connections(
    temporal_graphs: Dict[datetime, Tuple[nk.Graph, IDMapper]],
    node_categories: Dict[Any, str],
    normalize: bool = True
) -> pl.DataFrame
```

Analyze connections between different node categories over time.

**Parameters:**
- `temporal_graphs`: Time-sliced graphs
- `node_categories`: Dictionary mapping node IDs to category labels
- `normalize`: Whether to normalize by category sizes

**Examples:**

```python
from src.timeseries.category_analysis import analyze_cross_category_connections

# Define node categories (e.g., departments in organization)
node_categories = {
    "emp_001": "engineering", "emp_002": "engineering", 
    "emp_003": "marketing", "emp_004": "marketing",
    "emp_005": "sales", "emp_006": "hr"
}

# Analyze inter-departmental collaboration over time
category_connections = analyze_cross_category_connections(
    daily_graphs,
    node_categories,
    normalize=True
)

print(category_connections.head())
# timestamp           category_a    category_b    connections  strength
# 2024-01-01 00:00:00  engineering  marketing     8           0.23
# 2024-01-01 00:00:00  engineering  sales         5           0.15
# 2024-01-01 00:00:00  marketing    hr            2           0.08

# Visualize collaboration patterns
import seaborn as sns

# Pivot for heatmap
pivot_data = category_connections.pivot(
    index="category_a", 
    columns="category_b", 
    values="strength"
).fill_null(0)

sns.heatmap(pivot_data, annot=True, cmap="Blues")
plt.title("Inter-Department Collaboration Strength")
plt.show()
```

### detect_temporal_communities()

```python
def detect_temporal_communities(
    temporal_graphs: Dict[datetime, Tuple[nk.Graph, IDMapper]],
    method: str = "louvain",
    track_stability: bool = True
) -> Dict[datetime, Dict[Any, int]]
```

Detect communities in each time slice and track their evolution.

**Examples:**

```python
from src.timeseries.category_analysis import detect_temporal_communities

# Detect communities over time
temporal_communities = detect_temporal_communities(
    daily_graphs,
    method="louvain",
    track_stability=True
)

# Analyze community stability
community_df = []
for timestamp, communities in temporal_communities.items():
    for node, community_id in communities.items():
        community_df.append({
            "timestamp": timestamp,
            "node_id": node,
            "community": community_id
        })

community_df = pl.DataFrame(community_df)

# Track how often nodes change communities
node_stability = (
    community_df
    .group_by("node_id")
    .agg([
        pl.col("community").n_unique().alias("num_communities"),
        pl.col("timestamp").count().alias("num_timepoints")
    ])
    .with_columns(
        (pl.col("num_communities") / pl.col("num_timepoints")).alias("stability")
    )
)

# Find most stable nodes
stable_nodes = node_stability.filter(pl.col("stability") == 1.0)
print(f"Found {len(stable_nodes)} nodes that never changed communities")
```

## Advanced Temporal Analysis

### Rolling Window Analysis

```python
# Analyze network with rolling 3-day windows
from src.timeseries.slicing import create_temporal_slices
from src.timeseries.temporal_metrics import calculate_temporal_metrics

# Create overlapping 3-day windows
rolling_slices = create_temporal_slices(
    temporal_edges,
    slice_duration="3d",
    slice_method="sliding",
    overlap="1d"  # Move window by 1 day each time
)

# Convert to graphs
rolling_graphs = {}
for timestamp, edges in rolling_slices.items():
    graph, mapper = build_graph_from_edgelist(edges, "source", "target", "weight")
    rolling_graphs[timestamp] = (graph, mapper)

# Calculate rolling metrics
rolling_metrics = calculate_temporal_metrics(
    rolling_graphs,
    metrics=["degree", "betweenness"]
)

# Smooth temporal patterns
smoothed_metrics = (
    rolling_metrics
    .sort("timestamp")
    .group_by("node_id")
    .map_groups(lambda group: group.with_columns([
        pl.col("degree").rolling_mean(window_size=3).alias("degree_smooth"),
        pl.col("betweenness").rolling_mean(window_size=3).alias("betweenness_smooth")
    ]))
)
```

### Event Detection

```python
# Detect significant changes in network structure
def detect_network_events(evolution_df: pl.DataFrame, threshold: float = 2.0):
    """Detect significant changes in network metrics."""
    
    # Calculate z-scores for each metric
    events = evolution_df.with_columns([
        ((pl.col("nodes") - pl.col("nodes").mean()) / pl.col("nodes").std()).alias("nodes_zscore"),
        ((pl.col("edges") - pl.col("edges").mean()) / pl.col("edges").std()).alias("edges_zscore"),
        ((pl.col("density") - pl.col("density").mean()) / pl.col("density").std()).alias("density_zscore")
    ])
    
    # Find time points with significant changes
    significant_events = events.filter(
        (pl.col("nodes_zscore").abs() > threshold) |
        (pl.col("edges_zscore").abs() > threshold) |
        (pl.col("density_zscore").abs() > threshold)
    )
    
    return significant_events

# Apply event detection
evolution_df = track_network_evolution(daily_graphs)
events = detect_network_events(evolution_df, threshold=2.0)

print("Detected network events:")
for row in events.iter_rows(named=True):
    print(f"  {row['timestamp']}: nodes_z={row['nodes_zscore']:.2f}, "
          f"edges_z={row['edges_zscore']:.2f}, density_z={row['density_zscore']:.2f}")
```

## Performance Considerations

### Large Temporal Networks

```python
# Efficient processing for large temporal datasets
from src.timeseries.slicing import create_temporal_slices

# Use longer time slices to reduce memory usage
weekly_slices = create_temporal_slices(
    large_temporal_data,
    slice_duration="1w",  # Weekly instead of daily
    slice_method="tumbling"  # Non-overlapping to save memory
)

# Process in batches
batch_size = 10
slice_dates = list(weekly_slices.keys())

for i in range(0, len(slice_dates), batch_size):
    batch_dates = slice_dates[i:i+batch_size]
    batch_graphs = {}
    
    for date in batch_dates:
        edges = weekly_slices[date]
        graph, mapper = build_graph_from_edgelist(edges, "source", "target")
        batch_graphs[date] = (graph, mapper)
    
    # Process batch
    batch_metrics = calculate_temporal_metrics(batch_graphs)
    # Save or accumulate results
    batch_metrics.write_csv(f"metrics_batch_{i//batch_size}.csv")
```

## Common Temporal Analysis Patterns

### Social Media Trend Analysis

```python
# Complete workflow for analyzing social media trends
import polars as pl
from src.network.construction import build_graph_from_edgelist
from src.timeseries.slicing import create_temporal_slices
from src.timeseries.temporal_metrics import calculate_temporal_metrics
from src.glp.propagation import guided_label_propagation

# 1. Load temporal interaction data
interactions = pl.read_csv("social_media_interactions.csv")

# 2. Create daily network slices
daily_slices = create_temporal_slices(
    interactions, 
    timestamp_col="created_at",
    source_col="user_id",
    target_col="mentioned_user_id",
    slice_duration="1d"
)

# 3. Convert to graphs and calculate metrics
daily_graphs = {}
for date, edges in daily_slices.items():
    if len(edges) > 0:  # Skip empty days
        graph, mapper = build_graph_from_edgelist(edges, "user_id", "mentioned_user_id")
        daily_graphs[date] = (graph, mapper)

temporal_metrics = calculate_temporal_metrics(daily_graphs, ["degree", "pagerank"])

# 4. Track influential users over time
influencers = (
    temporal_metrics
    .filter(pl.col("pagerank") > 0.01)  # High PageRank threshold
    .group_by("node_id")
    .count()
    .filter(pl.col("count") >= 7)  # Consistently influential
    .get_column("node_id")
    .to_list()
)

print(f"Found {len(influencers)} consistently influential users")

# 5. Run GLP analysis on peak activity day
peak_day = max(daily_graphs.keys(), key=lambda d: daily_graphs[d][0].numberOfEdges())
peak_graph, peak_mapper = daily_graphs[peak_day]

# Define seed labels (verified accounts, etc.)
seed_labels = {"verified_user_1": "brand_a", "verified_user_2": "brand_b"}
labels = ["brand_a", "brand_b"]

brand_affinity = guided_label_propagation(
    peak_graph, peak_mapper, seed_labels, labels
)

print(f"Analyzed brand affinity for {len(brand_affinity)} users on {peak_day.date()}")
```