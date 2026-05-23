# Network Module API Reference

The network module provides functionality for constructing, analyzing, and manipulating NetworkIt graphs from various data sources.

## Module: `src.network.construction`

### build_graph_from_edgelist()

```python
def build_graph_from_edgelist(
    edgelist: Union[str, pl.DataFrame],
    source_col: str = "source",
    target_col: str = "target", 
    weight_col: Optional[str] = None,
    directed: bool = False,
    bipartite: bool = False,
    auto_weight: bool = True,
    allow_self_loops: bool = True,
    remove_duplicates: bool = False
) -> Tuple[nk.Graph, IDMapper]
```

Construct a NetworkIt graph from an edge list with ID preservation.

**Parameters:**
- `edgelist`: Path to CSV file or Polars DataFrame containing edge data
- `source_col`: Name of the source node column (default: "source")
- `target_col`: Name of the target node column (default: "target")
- `weight_col`: Name of edge weight column (optional)
- `directed`: Create directed graph (default: False)
- `bipartite`: Validate bipartite structure (default: False)
- `auto_weight`: Calculate weights from duplicate edges (default: True)
- `allow_self_loops`: Allow self-loops (default: True) 
- `remove_duplicates`: Remove duplicate edges (default: False)

**Returns:**
- `graph`: Constructed NetworkIt graph object
- `id_mapper`: Bidirectional mapping between original and internal IDs

**Examples:**

```python
# Basic undirected graph
import polars as pl
from src.network.construction import build_graph_from_edgelist

edges = pl.DataFrame({
    "source": ["A", "B", "C", "A"],
    "target": ["B", "C", "A", "B"]
})

graph, id_mapper = build_graph_from_edgelist(edges)
print(f"Nodes: {graph.numberOfNodes()}, Edges: {graph.numberOfEdges()}")

# Directed weighted graph
weighted_edges = pl.DataFrame({
    "from": ["user1", "user2", "user1"], 
    "to": ["user2", "user3", "user3"],
    "weight": [1.5, 2.0, 0.8]
})

graph, mapper = build_graph_from_edgelist(
    weighted_edges,
    source_col="from",
    target_col="to", 
    weight_col="weight",
    directed=True
)

# Bipartite graph (users -> items)
bipartite_edges = pl.DataFrame({
    "user": ["u1", "u1", "u2", "u3"],
    "item": ["i1", "i2", "i1", "i2"]
})

graph, mapper = build_graph_from_edgelist(
    bipartite_edges,
    source_col="user",
    target_col="item",
    bipartite=True
)
```

### project_bipartite()

```python
def project_bipartite(
    graph: nk.Graph,
    id_mapper: IDMapper,
    projection_mode: str = "source",
    weight_method: str = "count"
) -> Tuple[nk.Graph, IDMapper]
```

Project bipartite graph to unipartite by connecting nodes with shared neighbors.

**Parameters:**
- `graph`: Bipartite NetworkIt graph to project
- `id_mapper`: Original ID mapper containing all bipartite graph nodes
- `projection_mode`: Which partition to project onto ("source" or "target")
- `weight_method`: Weight calculation method ("count", "jaccard", "overlap")

**Returns:**
- `projected_graph`: Unipartite graph containing only projected nodes
- `new_id_mapper`: Updated ID mapper for projected graph

**Examples:**

```python
# Create user-item bipartite graph
edges = pl.DataFrame({
    "user": ["u1", "u1", "u2", "u2", "u3"],
    "item": ["i1", "i2", "i1", "i3", "i2"]
})

graph, mapper = build_graph_from_edgelist(
    edges, source_col="user", target_col="item", bipartite=True
)

# Project onto users (connect users who like same items)
user_graph, user_mapper = project_bipartite(
    graph, mapper, projection_mode="source", weight_method="count"
)

# Project onto items (connect items liked by same users)
item_graph, item_mapper = project_bipartite(
    graph, mapper, projection_mode="target", weight_method="jaccard"
)

print(f"User projection: {user_graph.numberOfNodes()} users, {user_graph.numberOfEdges()} connections")
print(f"Item projection: {item_graph.numberOfNodes()} items, {item_graph.numberOfEdges()} connections")
```

### temporal_bipartite_to_unipartite()

```python
def temporal_bipartite_to_unipartite(
    edgelist: Union[str, pl.DataFrame],
    source_col: str = "source",
    target_col: str = "target", 
    timestamp_col: str = "timestamp",
    weight_col: Optional[str] = None,
    intermediate_col: str = "target",
    projected_col: str = "source",
    remove_self_loops: bool = True,
    add_edge_weights: bool = True
) -> Tuple[nk.Graph, IDMapper]
```

🕒 **Convert temporal bipartite edgelist to unipartite graph using temporal precedence.**

This function transforms temporal bipartite networks (like user-item interactions over time) into directed unipartite networks that preserve temporal causality. Uses ascending timestamp sort combined with upper triangular matrix indexing to ensure proper temporal flow: **earlier events → later events**.

**Parameters:**
- `edgelist`: Path to CSV file or Polars DataFrame with temporal bipartite data
- `source_col`: Name of source node column (default: "source")
- `target_col`: Name of target node column (default: "target")
- `timestamp_col`: Name of timestamp column for temporal ordering
- `weight_col`: Name of weight column (optional)
- `intermediate_col`: Column representing intermediate nodes that disappear in projection (default: "target")
- `projected_col`: Column representing nodes to preserve and connect (default: "source")
- `remove_self_loops`: Remove self-connections in result (default: True)
- `add_edge_weights`: Calculate temporal decay weights (default: True)

**Returns:**
- `graph`: Directed NetworkIt graph with temporal influence relationships
- `id_mapper`: Bidirectional mapping for projected nodes

**Algorithm:**
1. Groups edges by intermediate node (disappearing column)
2. Within each group, sorts by timestamp in ascending order
3. Creates directed edges using upper triangular matrix indices
4. Results in proper temporal flow: earlier events → later events

**Examples:**

```python
# Convert user-item temporal interactions to user influence network
import polars as pl
from src.network.construction import temporal_bipartite_to_unipartite

# Sample temporal bipartite data
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

print(f"Influence network: {influence_graph.numberOfNodes()} users")
print(f"Temporal relationships: {influence_graph.numberOfEdges()} edges")

# Expected edges: Alice → Bob → Charlie (temporal precedence)
```

**Use Cases:**
- **Social Media**: User-content → User-user information flow networks
- **E-commerce**: User-item → User-user recommendation networks  
- **Academia**: Author-paper → Author-author citation influence
- **Communications**: User-topic → User-user discussion networks

**Temporal Logic:**
If users A and B both interact with the same item, and A interacts first, then edge A → B is created (A influences B). This preserves temporal causality and models information/influence flow.

### get_graph_info()

```python
def get_graph_info(graph: nk.Graph, id_mapper: IDMapper) -> Dict[str, Any]
```

Get comprehensive information about a constructed graph.

**Examples:**

```python
graph, mapper = build_graph_from_edgelist(edges)
info = get_graph_info(graph, mapper)

print(f"Graph Info:")
print(f"  Nodes: {info['num_nodes']}")
print(f"  Edges: {info['num_edges']}")
print(f"  Directed: {info['directed']}")
print(f"  Weighted: {info['weighted']}")
print(f"  Density: {info['density']:.4f}")
print(f"  Connected: {info['is_connected']}")
```

## Module: `src.network.analysis`

### calculate_centrality_measures()

```python
def calculate_centrality_measures(
    graph: nk.Graph,
    id_mapper: IDMapper,
    measures: List[str] = ["degree", "betweenness", "closeness"],
    normalized: bool = True
) -> pl.DataFrame
```

Calculate multiple centrality measures for all nodes in the graph.

**Parameters:**
- `graph`: NetworkIt graph to analyze
- `id_mapper`: ID mapper for converting between original and internal IDs
- `measures`: List of centrality measures to calculate
- `normalized`: Whether to normalize centrality scores

**Available Measures:**
- `"degree"`: Degree centrality
- `"betweenness"`: Betweenness centrality  
- `"closeness"`: Closeness centrality
- `"eigenvector"`: Eigenvector centrality
- `"pagerank"`: PageRank centrality

**Examples:**

```python
from src.network.analysis import calculate_centrality_measures

# Calculate all centrality measures
centrality_df = calculate_centrality_measures(
    graph, mapper, 
    measures=["degree", "betweenness", "closeness", "pagerank"]
)

print(centrality_df.head())
# node_id  degree  betweenness  closeness  pagerank
# A        0.667   0.333        0.75       0.4
# B        1.000   0.667        1.00       0.4  
# C        0.333   0.000        0.60       0.2

# Calculate specific measures only
degree_df = calculate_centrality_measures(
    graph, mapper, measures=["degree"]
)
```

## Module: `src.network.communities`

### detect_communities()

```python
def detect_communities(
    graph: nk.Graph,
    id_mapper: IDMapper,
    algorithm: str = "louvain",
    resolution: float = 1.0
) -> Dict[Any, int]
```

Detect communities in the graph using various algorithms.

**Examples:**

```python
from src.network.communities import detect_communities

# Louvain community detection
communities = detect_communities(graph, mapper, algorithm="louvain")

# Convert to DataFrame for analysis
community_df = pl.DataFrame({
    "node_id": list(communities.keys()),
    "community": list(communities.values())
})

# Count community sizes
community_sizes = community_df.group_by("community").count()
print(community_sizes)
```

## Module: `src.network.export`

### export_graph()

```python
def export_graph(
    graph: nk.Graph,
    id_mapper: IDMapper,
    filepath: str,
    format: str = "gexf"
) -> None
```

Export graph to various formats for visualization and analysis.

**Examples:**

```python
from src.network.export import export_graph

# Export to GEXF for Gephi
export_graph(graph, mapper, "network.gexf", format="gexf")

# Export to GraphML
export_graph(graph, mapper, "network.graphml", format="graphml")

# Export edge list
export_graph(graph, mapper, "edges.csv", format="edgelist")
```

## Performance Notes

- **Graph Construction**: O(E + V) time complexity using NetworkIt
- **Bipartite Projection**: O(N² × D) worst case where N is projection size, D is average degree
- **Centrality Calculations**: Parallelized where possible using NetworkIt's C++ backend
- **Memory Usage**: Sparse matrices used for graphs with >50% zero entries

## Common Patterns

### Loading and Basic Analysis

```python
# Standard workflow
import polars as pl
from src.network.construction import build_graph_from_edgelist
from src.network.analysis import calculate_centrality_measures

# 1. Load data
edges = pl.read_csv("network_data.csv")

# 2. Build graph
graph, mapper = build_graph_from_edgelist(
    edges, "source", "target", weight_col="weight"
)

# 3. Analyze
centrality = calculate_centrality_measures(graph, mapper)
info = get_graph_info(graph, mapper)

print(f"Network with {info['num_nodes']} nodes and {info['num_edges']} edges")
```

### Bipartite Analysis

```python
# User-item analysis workflow
user_item_edges = pl.read_csv("user_ratings.csv")

# Build bipartite graph
bipartite_graph, mapper = build_graph_from_edgelist(
    user_item_edges, "user_id", "item_id", "rating", bipartite=True
)

# Project to user similarity network
user_graph, user_mapper = project_bipartite(
    bipartite_graph, mapper, "source", "jaccard"
)

# Analyze user communities
user_centrality = calculate_centrality_measures(user_graph, user_mapper)
```