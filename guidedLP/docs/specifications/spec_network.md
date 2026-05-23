# Network Module Function Specifications

## Module: `src/network/construction.py`

### Function: `build_graph_from_edgelist()`

**Purpose**: Construct a NetworkIt graph from an edge list with ID preservation

**Signature**:
```python
def build_graph_from_edgelist(
    edgelist: Union[str, pl.DataFrame],
    source_col: str = "source",
    target_col: str = "target", 
    weight_col: Optional[str] = None,
    directed: bool = False,
    bipartite: bool = False,
    auto_weight: bool = True
) -> Tuple[nk.Graph, IDMapper]:
```

**Parameters**:
- `edgelist`: Path to CSV file or Polars DataFrame with edges
- `source_col`: Name of source node column
- `target_col`: Name of target node column
- `weight_col`: Name of weight column (None if unweighted)
- `directed`: If True, create directed graph
- `bipartite`: If True, create bipartite graph (validates source and target are distinct sets)
- `auto_weight`: If True and weight_col is None, calculate weights from duplicate edges

**Returns**:
- `graph`: NetworkIt graph object
- `id_mapper`: IDMapper object for original ↔ internal ID mapping

**Logic**:
1. Load edge list using Polars (lazy if CSV file)
2. Validate columns exist
3. If `auto_weight=True` and `weight_col=None`:
   - Group by (source, target) and count occurrences
   - Use counts as weights
4. Extract unique node IDs (source ∪ target)
5. Create ID mapping: original IDs → 0, 1, 2, ... (NetworkIt node IDs)
6. Create NetworkIt graph with appropriate settings
7. Add edges with weights
8. Return graph and mapper

**Edge Cases**:
- Empty edge list: Return empty graph with warning
- Self-loops: Include by default, provide option to remove
- Duplicate edges with weights: Sum weights
- Invalid node IDs: Raise ValueError

**Performance**: O(E) where E = number of edges

---

### Function: `project_bipartite()`

**Purpose**: Project bipartite graph to unipartite by connecting nodes with shared neighbors

**Signature**:
```python
def project_bipartite(
    graph: nk.Graph,
    id_mapper: IDMapper,
    projection_mode: str = "source",
    weight_method: str = "count"
) -> Tuple[nk.Graph, IDMapper]:
```

**Parameters**:
- `graph`: Bipartite NetworkIt graph
- `id_mapper`: Original ID mapper
- `projection_mode`: Which partition to project onto ("source" or "target")
- `weight_method`: How to calculate projection weights
  - "count": Number of shared neighbors
  - "jaccard": Jaccard similarity of neighbor sets
  - "overlap": Overlap coefficient

**Returns**:
- `projected_graph`: Unipartite NetworkIt graph
- `new_id_mapper`: Updated ID mapper for projected nodes

**Logic**:
1. Identify two partitions of bipartite graph
2. Select projection partition based on `projection_mode`
3. For each pair of nodes in projection partition:
   - Find shared neighbors in other partition
   - Calculate edge weight using `weight_method`
   - Add edge if weight > 0
4. Create new graph with projected edges
5. Update ID mapper to only include projected nodes

**Edge Cases**:
- Non-bipartite graph: Raise ValueError
- No shared neighbors: Results in disconnected nodes

**Performance**: O(N² × D) worst case, where N = partition size, D = avg degree

---

## Module: `src/network/analysis.py`

### Function: `extract_centrality()`

**Purpose**: Calculate centrality metrics for all nodes

**Signature**:
```python
def extract_centrality(
    graph: nk.Graph,
    id_mapper: IDMapper,
    metrics: List[str] = ["degree", "betweenness", "closeness", "eigenvector"],
    normalized: bool = True,
    n_jobs: int = -1
) -> pl.DataFrame:
```

**Parameters**:
- `graph`: NetworkIt graph
- `id_mapper`: ID mapper for original IDs
- `metrics`: List of centrality metrics to calculate
  - Available: "degree", "betweenness", "closeness", "eigenvector", "pagerank", "katz"
- `normalized`: Normalize centrality scores to [0, 1]
- `n_jobs`: Number of parallel jobs (-1 = all cores)

**Returns**:
- DataFrame with columns: `node_id` (original), `{metric}_centrality` (one per metric)

**Logic**:
1. For each metric in `metrics`:
   - Use NetworkIt's built-in centrality algorithms
   - Run computation (parallelize if possible)
   - Extract centrality values for all nodes
2. Map internal IDs → original IDs
3. Combine all metrics into single DataFrame
4. Sort by node_id

**Available Metrics**:
- `degree`: In/out/total degree
- `betweenness`: Fraction of shortest paths through node
- `closeness`: Inverse of average distance to all other nodes
- `eigenvector`: Importance based on neighbor importance
- `pagerank`: Google's PageRank algorithm
- `katz`: Katz centrality (weighted paths)

**Edge Cases**:
- Disconnected components: Closeness may be undefined; use harmonic mean
- Invalid metric name: Raise ValueError with available options

**Performance**: Varies by metric (O(N) to O(N³))

---

## Module: `src/network/backboning.py`

### Function: `apply_backbone()`

**Purpose**: Extract network backbone by filtering edges

**Signature**:
```python
def apply_backbone(
    graph: nk.Graph,
    id_mapper: IDMapper,
    method: str = "disparity",
    target_nodes: Optional[int] = None,
    target_edges: Optional[int] = None,
    alpha: float = 0.05,
    keep_disconnected: bool = False
) -> Tuple[nk.Graph, IDMapper]:
```

**Parameters**:
- `graph`: NetworkIt graph
- `id_mapper`: Original ID mapper
- `method`: Backboning technique
  - "disparity": Disparity filter (Serrano et al.)
  - "weight": Simple weight threshold
  - "degree": Keep nodes by degree threshold
- `target_nodes`: Keep approximately this many nodes (conflicts with target_edges)
- `target_edges`: Keep approximately this many edges
- `alpha`: Significance level for disparity filter
- `keep_disconnected`: Keep isolated nodes after filtering

**Returns**:
- `backbone_graph`: Filtered NetworkIt graph
- `updated_mapper`: ID mapper with remaining nodes

**Logic**:

**Disparity Filter**:
1. For each node, normalize edge weights: `p_ij = w_ij / Σw_ik`
2. Calculate disparity: `α_ij = (1 - p_ij)^(k-1)` where k = degree
3. Keep edges where `α_ij < alpha` (statistically significant)

**Weight Threshold**:
1. If `target_edges` specified: Calculate threshold to keep ~N edges
2. Remove edges with weight < threshold

**Degree Threshold**:
1. If `target_nodes` specified: Calculate degree threshold
2. Remove nodes with degree < threshold
3. Remove edges connected to removed nodes

**Edge Cases**:
- Both target_nodes and target_edges specified: Raise ValueError
- Threshold results in disconnected graph: Warn user
- All edges removed: Raise ValueError

**Performance**: O(E) for weight threshold, O(E log E) for disparity

---

## Module: `src/network/communities.py`

### Function: `detect_communities()`

**Purpose**: Detect communities using Louvain algorithm with quality metrics

**Signature**:
```python
def detect_communities(
    graph: nk.Graph,
    id_mapper: IDMapper,
    algorithm: str = "louvain",
    iterations: int = 1,
    resolution: float = 1.0,
    min_similarity: Optional[float] = None,
    random_seed: Optional[int] = None,
    n_jobs: int = -1
) -> pl.DataFrame:
```

**Parameters**:
- `graph`: NetworkIt graph
- `id_mapper`: Original ID mapper
- `algorithm`: Community detection algorithm ("louvain" for now)
- `iterations`: Number of runs with different random initializations
- `resolution`: Resolution parameter (higher = more communities)
- `min_similarity`: If specified, only return partitions with similarity > threshold
- `random_seed`: Random seed for reproducibility
- `n_jobs`: Parallel jobs for multiple iterations

**Returns**:
- DataFrame with columns:
  - `node_id`: Original node ID
  - `community_iter_{i}`: Community assignment for iteration i
  - `community_consensus`: Consensus community (most common across iterations)
  - `stability`: Fraction of iterations with consensus assignment

**Logic**:
1. Run Louvain algorithm `iterations` times
2. For each iteration:
   - Set random seed (seed + iteration) for reproducibility
   - Run community detection
   - Store partition
3. If `iterations > 1`:
   - Calculate partition similarity (Normalized Mutual Information)
   - If `min_similarity` specified, filter dissimilar runs
   - Calculate consensus partition using majority voting
   - Calculate per-node stability scores
4. Map internal IDs → original IDs
5. Return DataFrame with all partitions

**Quality Metrics** (included in return):
- Modularity: Q score for each partition
- Coverage: Fraction of edges within communities
- Number of communities: Count per partition

**Edge Cases**:
- Single iteration: No consensus calculation needed
- All partitions identical: Perfect stability = 1.0
- No converging partitions: Warn user, return all results

**Performance**: O(E × I) where I = iterations

---

## Module: `src/network/filtering.py`

### Function: `filter_graph()`

**Purpose**: Apply various filters to graph

**Signature**:
```python
def filter_graph(
    graph: nk.Graph,
    id_mapper: IDMapper,
    filters: Dict[str, Any],
    combine: str = "and"
) -> Tuple[nk.Graph, IDMapper]:
```

**Parameters**:
- `graph`: NetworkIt graph
- `id_mapper`: Original ID mapper
- `filters`: Dictionary of filter specifications
- `combine`: How to combine filters ("and" or "or")

**Filters Dictionary Format**:
```python
filters = {
    "min_degree": 5,                    # Minimum degree
    "max_degree": 100,                  # Maximum degree
    "min_weight": 2.0,                  # Minimum edge weight
    "giant_component_only": True,       # Keep only largest component
    "nodes": ["node1", "node2", ...],   # Keep only these nodes (original IDs)
    "exclude_nodes": ["node3", ...],    # Remove these nodes
    "centrality": {                      # Filter by centrality
        "metric": "betweenness",
        "min_value": 0.01
    }
}
```

**Returns**:
- `filtered_graph`: Filtered NetworkIt graph
- `updated_mapper`: ID mapper with remaining nodes

**Logic**:
1. Parse filters dictionary
2. For each filter:
   - Calculate which nodes/edges to keep
   - Create boolean mask
3. Combine masks using `combine` logic
4. Create subgraph with filtered nodes/edges
5. Update ID mapper

**Edge Cases**:
- No nodes pass filters: Raise ValueError
- Conflicting filters (min_degree > max_degree): Raise ValueError
- Giant component creates multiple components: Take largest

**Performance**: O(N + E) for most filters, O(N²) for component detection

---

## Module: `src/network/export.py`

### Function: `export_graph()`

**Purpose**: Export graph with metadata in various formats

**Signature**:
```python
def export_graph(
    graph: nk.Graph,
    id_mapper: IDMapper,
    output_path: str,
    format: str = "gexf",
    metadata: Optional[pl.DataFrame] = None,
    include_metrics: Optional[List[str]] = None
) -> None:
```

**Parameters**:
- `graph`: NetworkIt graph
- `id_mapper`: Original ID mapper
- `output_path`: Path for output file
- `format`: Export format ("gexf", "graphml", "edgelist", "parquet")
- `metadata`: DataFrame with node metadata (must have node_id column matching original IDs)
- `include_metrics`: List of metrics to calculate and include (uses `extract_centrality`)

**Logic**:
1. If `include_metrics` specified:
   - Calculate metrics using `extract_centrality`
   - Merge with metadata (if provided)
2. If metadata provided:
   - Join with node IDs (original)
   - Validate all nodes have metadata (warn if missing)
3. Convert NetworkIt graph to export format:
   - **GEXF/GraphML**: Include node attributes from metadata
   - **Edgelist**: CSV with source, target, weight, + node attributes
   - **Parquet**: Efficient binary format with edges and nodes tables
4. Map internal IDs → original IDs in export
5. Write to file

**GEXF Structure**:
```xml
<nodes>
  <node id="original_id" label="original_id">
    <attvalues>
      <attvalue for="metadata_col" value="..."/>
      ...
    </attvalues>
  </node>
</nodes>
```

**Edge Cases**:
- Metadata missing for some nodes: Warn but continue (use NULL)
- Unsupported format: Raise ValueError
- Overwrite existing file: Warn and ask for confirmation (or force flag)

**Performance**: O(N + E) for most formats

---

## Common Utilities: `src/common/id_mapper.py`

### Class: `IDMapper`

**Purpose**: Bidirectional mapping between original and internal node IDs

**Attributes**:
```python
class IDMapper:
    original_to_internal: Dict[Any, int]  # Original ID → NetworkIt ID
    internal_to_original: Dict[int, Any]  # NetworkIt ID → Original ID
```

**Methods**:

#### `get_internal(original_id) -> int`
Returns internal ID for given original ID. Raises KeyError if not found.

#### `get_original(internal_id) -> Any`
Returns original ID for given internal ID. Raises KeyError if not found.

#### `get_internal_batch(original_ids) -> List[int]`
Batch version for efficiency. Returns list of internal IDs.

#### `get_original_batch(internal_ids) -> List[Any]`
Batch version for efficiency. Returns list of original IDs.

#### `add_mapping(original_id, internal_id) -> None`
Add new ID pair to mapping.

#### `size() -> int`
Returns number of mapped nodes.

#### `to_dict() -> Dict`
Export mapping as dictionary (for serialization).

#### `from_dict(mapping: Dict) -> IDMapper`
Create IDMapper from dictionary (deserialization).

**Usage Example**:
```python
mapper = IDMapper()
mapper.add_mapping("user_123", 0)
mapper.add_mapping("user_456", 1)

internal = mapper.get_internal("user_123")  # → 0
original = mapper.get_original(0)           # → "user_123"
```
