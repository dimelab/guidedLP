# Time-Series Network Analysis Function Specifications

## Module: `src/timeseries/slicing.py`

### Function: `create_temporal_slices()`

**Purpose**: Create time-sliced networks from edge list with timestamps

**Signature**:
```python
def create_temporal_slices(
    edgelist: Union[str, pl.DataFrame],
    timestamp_col: str = "timestamp",
    slice_interval: str = "daily",
    rolling_window: Optional[int] = None,
    cumulative: bool = False,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    **graph_kwargs
) -> List[Tuple[datetime, nk.Graph, IDMapper]]:
```

**Parameters**:
- `edgelist`: Path to CSV or Polars DataFrame with timestamps
- `timestamp_col`: Name of timestamp column (must be datetime or parseable)
- `slice_interval`: Temporal granularity
  - "daily": One graph per day
  - "weekly": One graph per week
  - "monthly": One graph per month
  - "yearly": One graph per year
- `rolling_window`: Number of intervals to include in rolling average (None = no rolling)
  - Example: `rolling_window=7` with `slice_interval="daily"` = 7-day rolling window
- `cumulative`: If True, include all edges up to current time slice
- `start_date`: Start date for analysis (None = earliest in data)
- `end_date`: End date for analysis (None = latest in data)
- `**graph_kwargs`: Additional arguments for `build_graph_from_edgelist()`

**Returns**:
- List of tuples: `(slice_date, graph, id_mapper)` for each time slice

**Logic**:

1. **Load and validate data**:
   - Load edgelist with Polars
   - Parse timestamp column to datetime
   - Filter by start_date and end_date if provided

2. **Determine time slices**:
   - Group timestamps by `slice_interval` (day, week, month, year)
   - Create date range for all slices

3. **Build graphs for each slice**:
   
   **Non-cumulative, no rolling**:
   ```python
   for slice_date in date_range:
       edges_in_slice = filter_edges_by_date(edgelist, slice_date, interval)
       graph, mapper = build_graph_from_edgelist(edges_in_slice, **graph_kwargs)
       yield (slice_date, graph, mapper)
   ```
   
   **Cumulative**:
   ```python
   accumulated_edges = pl.DataFrame()
   for slice_date in date_range:
       edges_up_to_date = filter_edges_up_to(edgelist, slice_date)
       graph, mapper = build_graph_from_edgelist(edges_up_to_date, **graph_kwargs)
       yield (slice_date, graph, mapper)
   ```
   
   **Rolling window**:
   ```python
   for slice_date in date_range:
       window_start = slice_date - (rolling_window * interval)
       edges_in_window = filter_edges_between(edgelist, window_start, slice_date)
       graph, mapper = build_graph_from_edgelist(edges_in_window, **graph_kwargs)
       yield (slice_date, graph, mapper)
   ```

4. **ID Mapping consistency**:
   - Option 1: Separate mapper per slice (nodes can appear/disappear)
   - Option 2: Global mapper across all slices (recommended for consistent tracking)

**Edge Cases**:
- Empty slice (no edges): Return empty graph with warning
- Missing dates: Create empty slices or skip (user option)
- Overlapping rolling windows: Handle edge inclusion correctly

**Performance**: O(T × E) where T = number of slices, E = edges per slice

---

### Function: `align_node_ids_across_slices()`

**Purpose**: Create consistent node ID mapping across all time slices

**Signature**:
```python
def align_node_ids_across_slices(
    temporal_graphs: List[Tuple[datetime, nk.Graph, IDMapper]]
) -> Tuple[List[Tuple[datetime, nk.Graph]], IDMapper]:
```

**Purpose**: 
- Ensure same node has same internal ID across all slices
- Critical for tracking nodes over time
- Reconstructs graphs with aligned IDs

**Returns**:
- List of (date, graph) with aligned IDs
- Single global IDMapper for all slices

---

## Module: `src/timeseries/temporal_metrics.py`

### Function: `extract_temporal_metrics()`

**Purpose**: Calculate metrics for all nodes across all time slices

**Signature**:
```python
def extract_temporal_metrics(
    temporal_graphs: List[Tuple[datetime, nk.Graph, IDMapper]],
    metrics: List[str] = ["degree", "betweenness"],
    n_jobs: int = -1
) -> pl.DataFrame:
```

**Returns**:
DataFrame with schema:
```
node_id: Original node ID
date: Date of time slice
{metric}: Value for each metric
```

**Logic**:
1. For each time slice:
   - Extract specified metrics using `extract_centrality()`
   - Add date column
2. Concatenate all slices into single DataFrame
3. Fill missing values (nodes not present in slice) with 0 or NaN

**Use Cases**:
- Track how centrality evolves over time
- Identify emerging or declining nodes
- Detect temporal patterns

---

### Function: `calculate_temporal_statistics()`

**Purpose**: Aggregate statistics across time slices

**Signature**:
```python
def calculate_temporal_statistics(
    temporal_metrics: pl.DataFrame,
    statistics: List[str] = ["mean", "std", "trend"]
) -> pl.DataFrame:
```

**Parameters**:
- `temporal_metrics`: Output from `extract_temporal_metrics()`
- `statistics`: Which statistics to calculate
  - "mean": Average across time
  - "std": Standard deviation
  - "trend": Linear trend (slope)
  - "volatility": Coefficient of variation
  - "growth": Percentage change from first to last

**Returns**:
DataFrame with per-node temporal statistics

---

## Module: `src/timeseries/category_analysis.py`

### Function: `analyze_cross_category_connections()`

**Purpose**: Track connections between node categories over time

**Signature**:
```python
def analyze_cross_category_connections(
    temporal_graphs: List[Tuple[datetime, nk.Graph, IDMapper]],
    metadata: pl.DataFrame,
    category_column: str,
    edge_weight: str = "count"
) -> pl.DataFrame:
```

**Parameters**:
- `temporal_graphs`: Time-sliced graphs
- `metadata`: DataFrame with node categories (must include node_id column)
- `category_column`: Column in metadata with category labels
- `edge_weight`: How to aggregate edges ("count", "sum", "mean")

**Returns**:
DataFrame with schema:
```
date: Time slice date
category_a: First category
category_b: Second category
connection_strength: Aggregated edge weight between categories
edge_count: Number of edges between categories
```

**Logic**:
1. Join metadata to get node categories
2. For each time slice:
   - For each edge (u, v):
     - Get categories: cat_u, cat_v
     - Accumulate edge weight for (cat_u, cat_v) pair
3. Create time series of inter-category connections

**Use Cases**:
- Track polarization (within-group vs. between-group connections)
- Identify when different communities interact
- Measure integration vs. segregation over time

**Example**:
```
date       | category_a | category_b | connection_strength
2024-01-01 | politics   | sports     | 245
2024-01-01 | politics   | politics   | 1823
2024-01-02 | politics   | sports     | 198
...
```

---

## Module: `src/timeseries/aggregation.py`

### Function: `aggregate_temporal_networks()`

**Purpose**: Aggregate multiple time slices into summary statistics

**Signature**:
```python
def aggregate_temporal_networks(
    temporal_graphs: List[Tuple[datetime, nk.Graph, IDMapper]],
    aggregation_level: str = "monthly",
    metrics: List[str] = ["nodes", "edges", "density", "components"]
) -> pl.DataFrame:
```

**Parameters**:
- `temporal_graphs`: Time-sliced graphs
- `aggregation_level`: Aggregate finer slices to this level
  - "daily" → "weekly"
  - "daily" → "monthly"  
  - "weekly" → "monthly"
- `metrics`: Network-level metrics to calculate

**Returns**:
DataFrame with network-level statistics per aggregation period

**Network-level metrics**:
- `nodes`: Number of nodes
- `edges`: Number of edges
- `density`: Edge density
- `components`: Number of connected components
- `diameter`: Network diameter (largest component)
- `avg_clustering`: Average clustering coefficient

**Use Case**: 
- Summarize daily data into monthly trends
- Create high-level temporal overview
- Identify major structural changes

---

## Module: `src/timeseries/visualization.py`

### Function: `export_temporal_network_for_viz()`

**Purpose**: Export time-sliced networks for dynamic visualization (Gephi, D3.js)

**Signature**:
```python
def export_temporal_network_for_viz(
    temporal_graphs: List[Tuple[datetime, nk.Graph, IDMapper]],
    output_dir: str,
    format: str = "gexf",
    metadata: Optional[pl.DataFrame] = None
) -> None:
```

**Logic**:
1. Create output directory structure:
   ```
   output_dir/
   ├── slice_2024-01-01.gexf
   ├── slice_2024-01-02.gexf
   ├── ...
   └── metadata.csv
   ```
2. Export each time slice as separate graph file
3. Include consistent node IDs across files
4. Export metadata once (shared across slices)

**Use Case**:
- Create animated network visualizations
- Import into Gephi for temporal analysis
- Build web-based interactive timelines

---

## Utility Functions

### Function: `detect_temporal_communities()`

**Purpose**: Track community evolution over time

**Signature**:
```python
def detect_temporal_communities(
    temporal_graphs: List[Tuple[datetime, nk.Graph, IDMapper]],
    **community_kwargs
) -> pl.DataFrame:
```

**Returns**:
DataFrame with community assignments across time:
```
node_id | date_1_community | date_2_community | ... | stability_score
```

**Use Case**:
- Track how communities form and dissolve
- Identify stable vs. unstable group membership
- Detect community merges and splits

---

### Function: `calculate_temporal_motifs()`

**Purpose**: Identify recurring temporal patterns (motifs)

**Signature**:
```python
def calculate_temporal_motifs(
    temporal_graphs: List[Tuple[datetime, nk.Graph, IDMapper]],
    motif_size: int = 3,
    time_window: int = 1
) -> Dict[str, int]:
```

**Returns**:
- Dictionary of motif patterns and their frequencies

**Use Case**:
- Identify recurring temporal interaction patterns
- Detect coordination or synchronized behavior
- Network forensics and anomaly detection
