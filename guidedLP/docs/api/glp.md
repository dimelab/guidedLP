# Guided Label Propagation API Reference

The GLP module implements the core guided label propagation algorithm for semi-supervised community detection and node classification.

## 🆕 New Features (v0.2.0)

- **🎯 Automatic Noise Category**: Automatically adds "noise" label for nodes with weak associations
- **📊 Confidence Thresholding**: Identifies uncertain predictions below confidence thresholds  
- **🎲 Smart Noise Seed Generation**: Automatically samples non-seed nodes as noise examples
- **⚠️ Single Label Validation**: Warns about limitations of single-label scenarios
- **📈 Improved Robustness**: Better handling of outlier nodes and classification uncertainty

## Module: `src.glp.propagation`

### guided_label_propagation()

```python
def guided_label_propagation(
    graph: nk.Graph,
    id_mapper: IDMapper,
    seed_labels: Dict[Any, str],
    labels: List[str],
    alpha: float = 0.85,
    max_iterations: int = 100,
    convergence_threshold: float = 1e-6,
    normalize: bool = True,
    directional: bool = True,
    n_jobs: int = 1,
    enable_noise_category: bool = False,
    noise_ratio: float = 0.1,
    confidence_threshold: float = 0.0
) -> Union[pl.DataFrame, Tuple[pl.DataFrame, pl.DataFrame]]
```

Run guided label propagation algorithm for semi-supervised node classification.

**Parameters:**
- `graph`: NetworkIt graph to analyze
- `id_mapper`: ID mapper for converting between original and internal IDs
- `seed_labels`: Dictionary mapping seed node IDs to their known labels
- `labels`: List of all possible labels in the classification problem
- `alpha`: Propagation weight (0.0 = only seeds, 1.0 = only neighbors)
- `max_iterations`: Maximum number of iterations before forcing convergence
- `convergence_threshold`: Convergence threshold for label probability changes
- `normalize`: Whether to normalize final probabilities to sum to 1.0
- `directional`: For directed graphs, run both in/out-degree propagation
- `n_jobs`: Number of parallel jobs (reserved for future use)
- `enable_noise_category`: **NEW** - Automatically add "noise" category for uncertain nodes
- `noise_ratio`: **NEW** - Fraction of non-seed nodes to use as noise seeds (0.0-1.0)
- `confidence_threshold`: **NEW** - Minimum confidence for classification (0.0-1.0)

**Returns:**
- If `directional=False` or undirected graph: Single DataFrame with classification results
- If `directional=True` and directed graph: Tuple of (out_degree_results, in_degree_results)
- DataFrame columns: node_id, {label}_prob, dominant_label, confidence, is_seed

**Mathematical Foundation:**

The algorithm implements iterative matrix propagation:
1. Initialize label matrix Y where Y[i,j] = 1 if node i has label j (seed), 0 otherwise
2. Create transition matrix P = D^-1 A (row-normalized adjacency matrix)  
3. Iteratively update: F^(t+1) = α P F^(t) + (1-α) Y
4. Continue until convergence: max|F^(t+1) - F^(t)| < threshold

**Examples:**

```python
import polars as pl
from src.network.construction import build_graph_from_edgelist
from src.glp.propagation import guided_label_propagation

# Load social network data
edges = pl.read_csv("social_network.csv")
graph, mapper = build_graph_from_edgelist(edges, "user_a", "user_b")

# Define known political affiliations
seed_labels = {
    "user123": "progressive",
    "user456": "progressive", 
    "user789": "conservative",
    "user321": "conservative"
}

labels = ["progressive", "conservative"]

# Run GLP with probability output
probabilities = guided_label_propagation(
    graph=graph,
    id_mapper=mapper,
    seed_labels=seed_labels,
    labels=labels,
    alpha=0.85,
    return_probabilities=True
)

# Analyze results
for node_id, probs in probabilities.items():
    if node_id not in seed_labels:  # Skip seed nodes
        confidence = max(probs.values())
        predicted = max(probs, key=probs.get)
        print(f"{node_id}: {predicted} (confidence: {confidence:.3f})")

# Get hard classifications
classifications = guided_label_propagation(
    graph=graph,
    id_mapper=mapper, 
    seed_labels=seed_labels,
    labels=labels,
    return_probabilities=False
)
```

### Advanced Usage Examples

```python
# Brand affinity detection
brand_seeds = {
    "verified_apple": "apple_fans",
    "verified_samsung": "samsung_fans",
    "apple_store": "apple_fans",
    "samsung_official": "samsung_fans"
}

brand_probs = guided_label_propagation(
    graph, mapper, brand_seeds, ["apple_fans", "samsung_fans"],
    alpha=0.9,  # Higher alpha for stronger propagation
    max_iterations=150
)

# Academic collaboration analysis
research_seeds = {
    "cs_prof_1": "computer_science",
    "bio_prof_1": "biology", 
    "phys_prof_1": "physics"
}

research_fields = guided_label_propagation(
    collaboration_graph, mapper, research_seeds,
    ["computer_science", "biology", "physics"],
    alpha=0.7,  # Lower alpha to limit propagation distance
    directional=True  # Account for citation direction
)
```

## Module: `src.glp.validation`

### train_test_split_validation()

```python
def train_test_split_validation(
    graph: nk.Graph,
    id_mapper: IDMapper,
    known_labels: Dict[Any, str],
    labels: List[str],
    test_size: float = 0.2,
    random_state: Optional[int] = None,
    stratify: bool = True,
    **glp_params
) -> Tuple[float, Dict[str, float]]
```

Validate GLP performance using train-test split on known labels.

**Parameters:**
- `graph`: NetworkIt graph to analyze
- `id_mapper`: ID mapper for the graph
- `known_labels`: All known node labels for validation
- `labels`: List of possible labels
- `test_size`: Fraction of known labels to hold out for testing
- `random_state`: Random seed for reproducible splits
- `stratify`: Whether to maintain label proportions in train/test split
- `**glp_params`: Additional parameters passed to `guided_label_propagation()`

**Returns:**
- `accuracy`: Overall prediction accuracy on test set
- `metrics`: Dictionary containing precision, recall, F1-score per label

**Examples:**

```python
from src.glp.validation import train_test_split_validation

# Load labeled data
labeled_nodes = {
    "user1": "tech", "user2": "sports", "user3": "tech",
    "user4": "arts", "user5": "sports", "user6": "arts",
    # ... more labeled nodes
}

# Validate with 80/20 split
accuracy, metrics = train_test_split_validation(
    graph=graph,
    id_mapper=mapper,
    known_labels=labeled_nodes,
    labels=["tech", "sports", "arts"],
    test_size=0.2,
    random_state=42,
    alpha=0.85,  # GLP parameter
    max_iterations=100
)

print(f"Overall Accuracy: {accuracy:.3f}")
print("\\nPer-label metrics:")
for label in ["tech", "sports", "arts"]:
    prec = metrics[f"{label}_precision"]
    rec = metrics[f"{label}_recall"] 
    f1 = metrics[f"{label}_f1"]
    print(f"  {label}: P={prec:.3f}, R={rec:.3f}, F1={f1:.3f}")
```

### cross_validate()

```python
def cross_validate(
    graph: nk.Graph,
    id_mapper: IDMapper,
    known_labels: Dict[Any, str],
    labels: List[str],
    cv_folds: int = 5,
    random_state: Optional[int] = None,
    **glp_params
) -> Dict[str, List[float]]
```

Perform k-fold cross-validation for robust performance estimation.

**Examples:**

```python
from src.glp.validation import cross_validate

# 5-fold cross-validation
cv_results = cross_validate(
    graph=graph,
    id_mapper=mapper,
    known_labels=labeled_nodes,
    labels=["tech", "sports", "arts"],
    cv_folds=5,
    random_state=42,
    alpha=0.85,
    max_iterations=100
)

# Analyze cross-validation results
import numpy as np

accuracy_scores = cv_results["accuracy"]
print(f"CV Accuracy: {np.mean(accuracy_scores):.3f} (+/- {np.std(accuracy_scores)*2:.3f})")

for label in ["tech", "sports", "arts"]:
    f1_scores = cv_results[f"{label}_f1"]
    print(f"{label} F1: {np.mean(f1_scores):.3f} (+/- {np.std(f1_scores)*2:.3f})")
```

## Module: `src.glp.evaluation`

### calculate_prediction_confidence()

```python
def calculate_prediction_confidence(
    probabilities: Dict[Any, Dict[str, float]],
    confidence_method: str = "max_prob"
) -> Dict[Any, float]
```

Calculate confidence scores for GLP predictions.

**Parameters:**
- `probabilities`: Node probability dictionaries from GLP
- `confidence_method`: Method for calculating confidence ("max_prob", "entropy", "margin")

**Examples:**

```python
from src.glp.evaluation import calculate_prediction_confidence

# Run GLP to get probabilities
probabilities = guided_label_propagation(
    graph, mapper, seed_labels, labels, return_probabilities=True
)

# Calculate confidence using different methods
max_conf = calculate_prediction_confidence(probabilities, "max_prob")
entropy_conf = calculate_prediction_confidence(probabilities, "entropy") 
margin_conf = calculate_prediction_confidence(probabilities, "margin")

# Identify low-confidence predictions
low_confidence = {
    node: conf for node, conf in max_conf.items() 
    if conf < 0.6 and node not in seed_labels
}

print(f"Found {len(low_confidence)} low-confidence predictions")
for node, conf in sorted(low_confidence.items(), key=lambda x: x[1]):
    print(f"  {node}: {conf:.3f}")
```

### analyze_propagation_paths()

```python
def analyze_propagation_paths(
    graph: nk.Graph,
    id_mapper: IDMapper,
    seed_labels: Dict[Any, str],
    target_nodes: List[Any],
    max_hops: int = 3
) -> Dict[Any, List[Dict[str, Any]]]
```

Analyze how labels propagate from seeds to target nodes.

**Examples:**

```python
from src.glp.evaluation import analyze_propagation_paths

# Analyze propagation paths to specific nodes
target_nodes = ["user_unknown_1", "user_unknown_2"]

paths = analyze_propagation_paths(
    graph=graph,
    id_mapper=mapper,
    seed_labels=seed_labels,
    target_nodes=target_nodes,
    max_hops=3
)

for target, path_list in paths.items():
    print(f"\\nPropagation paths to {target}:")
    for i, path in enumerate(path_list[:3]):  # Show top 3 paths
        print(f"  Path {i+1}: {' -> '.join(path['nodes'])} (strength: {path['strength']:.3f})")
```

## Module: `src.glp.utils`

### prepare_seed_labels()

```python
def prepare_seed_labels(
    seed_data: Union[Dict[Any, str], pl.DataFrame],
    node_col: str = "node_id",
    label_col: str = "label"
) -> Tuple[Dict[Any, str], List[str]]
```

Prepare seed labels and extract unique labels from various input formats.

**Examples:**

```python
from src.glp.utils import prepare_seed_labels

# From DataFrame
seed_df = pl.DataFrame({
    "user_id": ["u1", "u2", "u3", "u4"],
    "political_affiliation": ["progressive", "progressive", "conservative", "conservative"]
})

seed_labels, labels = prepare_seed_labels(
    seed_df, node_col="user_id", label_col="political_affiliation"
)

# From dictionary (no processing needed)
seed_dict = {"u1": "progressive", "u2": "conservative"}
seed_labels, labels = prepare_seed_labels(seed_dict)
```

## Performance Optimization

### Large Networks

```python
# For large networks (>10K nodes), use optimized parameters
large_network_results = guided_label_propagation(
    graph=large_graph,
    id_mapper=large_mapper,
    seed_labels=seeds,
    labels=labels,
    alpha=0.9,  # Higher alpha for faster convergence
    max_iterations=50,  # Fewer iterations 
    convergence_threshold=1e-4,  # Less strict convergence
    normalize=False  # Skip normalization for speed
)
```

### Memory Optimization

```python
# For memory-constrained environments
memory_efficient_results = guided_label_propagation(
    graph=graph,
    id_mapper=mapper,
    seed_labels=seeds,
    labels=labels,
    return_probabilities=False,  # Return only classifications
    normalize=False  # Reduce computation
)
```

## Common Patterns

### Political Affiliation Pipeline

```python
# Complete political analysis workflow
import polars as pl
from src.network.construction import build_graph_from_edgelist
from src.glp.propagation import guided_label_propagation
from src.glp.validation import train_test_split_validation

# 1. Load network and seed data
edges = pl.read_csv("social_network.csv")
seeds_df = pl.read_csv("political_seeds.csv")

# 2. Build graph
graph, mapper = build_graph_from_edgelist(edges, "follower", "following")

# 3. Prepare seeds
seed_labels, labels = prepare_seed_labels(seeds_df, "user_id", "affiliation")

# 4. Validate approach
accuracy, metrics = train_test_split_validation(
    graph, mapper, seed_labels, labels, test_size=0.3
)
print(f"Validation accuracy: {accuracy:.3f}")

# 5. Run on full network
results = guided_label_propagation(graph, mapper, seed_labels, labels)

# 6. Analyze confidence
confidence = calculate_prediction_confidence(results)
high_confidence = {k: v for k, v in confidence.items() if v > 0.8}
print(f"High-confidence predictions: {len(high_confidence)}")
```