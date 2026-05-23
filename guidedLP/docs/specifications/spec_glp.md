# Guided Label Propagation Function Specifications

## Module: `src/glp/propagation.py`

### Function: `guided_label_propagation()`

**Purpose**: Propagate labels from seed nodes through network to calculate affinity probabilities

**Signature**:
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
    n_jobs: int = 1
) -> Union[pl.DataFrame, Tuple[pl.DataFrame, pl.DataFrame]]:
```

**Parameters**:
- `graph`: NetworkIt graph (can be directed or undirected)
- `id_mapper`: Original ID mapper
- `seed_labels`: Dict mapping original node IDs to their labels
  - Example: `{"user1": "left", "user2": "right", "user3": "left"}`
- `labels`: List of all possible labels (must include all values in seed_labels)
- `alpha`: Propagation coefficient (0-1, typically 0.85)
  - Higher = more influence from neighbors
  - Lower = seeds retain influence more strongly
- `max_iterations`: Stop after this many iterations even if not converged
- `convergence_threshold`: Stop when max probability change < threshold
- `normalize`: Normalize probabilities to sum to 1.0 per node
- `directional`: If graph is directed, run both in- and out-degree propagation
- `n_jobs`: Number of parallel jobs (for multi-label parallelization)

**Returns**:
- If `directional=False` OR graph is undirected: Single DataFrame
- If `directional=True` AND graph is directed: Tuple of (out_degree_df, in_degree_df)

**DataFrame Schema**:
```
node_id: Original node ID (Any)
{label}_prob: Probability for each label (float, one column per label)
dominant_label: Label with highest probability (str)
confidence: Maximum probability value (float)
is_seed: Boolean indicating if node was in seed set (bool)
```

**Logic**:

1. **Setup**:
   - Validate all seed labels are in `labels` list
   - Map original IDs to internal IDs
   - Create label index mapping (label string → column index)
   - Get adjacency matrix from NetworkIt (as scipy sparse matrix)

2. **Initialize Label Matrix Y** (n × k):
   - n = number of nodes
   - k = number of labels
   - For seed nodes: Y[i, j] = 1.0 if node i has label j, else 0
   - For non-seed nodes: Y[i, j] = 0.0 for all j

3. **Create Transition Matrix P**:
   - Get degree matrix D (diagonal)
   - Compute P = D⁻¹ A (row-normalized adjacency)
   - Handle zero-degree nodes (isolates): P row = zeros

4. **Iterative Propagation**:
   ```python
   F = Y.copy()  # Initialize with seed labels
   for iteration in range(max_iterations):
       F_prev = F.copy()
       
       # Propagation step
       F = alpha * P @ F + (1 - alpha) * Y
       
       # Check convergence
       max_change = np.max(np.abs(F - F_prev))
       if max_change < convergence_threshold:
           break
   ```

5. **Post-processing**:
   - If `normalize=True`: Normalize each row to sum to 1.0
   - Calculate `dominant_label` (argmax across labels)
   - Calculate `confidence` (max probability)
   - Map internal IDs → original IDs

6. **Directional Analysis** (if applicable):
   - Repeat steps 3-5 using:
     - Out-degree: Use A as-is
     - In-degree: Use A^T (transpose)
   - Return both results

**Edge Cases**:
- **Disconnected seeds**: Nodes unreachable from seeds get uniform distribution
- **Single seed**: Still works but high uncertainty for distant nodes  
- **Zero-degree nodes**: Retain initial state (0 or seed label)
- **Non-converged**: Return results after max_iterations with warning

**Performance**: 
- Time: O(k × i × E) where k=labels, i=iterations, E=edges
- Space: O(n × k) for label probability matrix

---

### Function: `run_multi_label_parallel()`

**Purpose**: Parallelize label propagation across multiple labels

**Signature**:
```python
def run_multi_label_parallel(
    graph: nk.Graph,
    id_mapper: IDMapper,
    seed_labels: Dict[Any, str],
    labels: List[str],
    alpha: float = 0.85,
    max_iterations: int = 100,
    convergence_threshold: float = 1e-6,
    n_jobs: int = -1
) -> pl.DataFrame:
```

**Logic**:
1. Split labels into separate single-label problems
2. For each label:
   - Create binary seed set (has label vs. doesn't have label)
   - Run binary propagation in parallel process
3. Combine results into multi-label probability matrix
4. Normalize probabilities across labels

**Use Case**: When k (number of labels) is large (5+), parallelizing by label is efficient

---

## Module: `src/glp/validation.py`

### Function: `train_test_split_validation()`

**Purpose**: Validate GLP using train/test split of seed nodes

**Signature**:
```python
def train_test_split_validation(
    graph: nk.Graph,
    id_mapper: IDMapper,
    seed_labels: Dict[Any, str],
    labels: List[str],
    test_size: float = 0.2,
    stratify: bool = True,
    random_seed: Optional[int] = None,
    **glp_kwargs
) -> Dict[str, Any]:
```

**Parameters**:
- `graph`, `id_mapper`, `seed_labels`, `labels`: Same as GLP function
- `test_size`: Fraction of seeds to hold out for testing (0-1)
- `stratify`: Maintain label proportions in train/test splits
- `random_seed`: For reproducible splits
- `**glp_kwargs`: Additional arguments passed to `guided_label_propagation()`

**Returns**:
Dictionary with validation metrics:
```python
{
    "accuracy": float,                     # Overall accuracy
    "precision": Dict[str, float],         # Per-label precision
    "recall": Dict[str, float],            # Per-label recall
    "f1_score": Dict[str, float],          # Per-label F1
    "confusion_matrix": np.ndarray,        # Confusion matrix
    "test_predictions": pl.DataFrame,      # Predictions on test set
    "train_size": int,                     # Number of training seeds
    "test_size": int,                      # Number of test seeds
    "convergence_iterations": int          # Iterations until convergence
}
```

**Logic**:
1. **Split seeds**:
   ```python
   from sklearn.model_selection import train_test_split
   train_seeds, test_seeds = train_test_split(
       seed_labels, 
       test_size=test_size, 
       stratify=labels if stratify else None,
       random_state=random_seed
   )
   ```

2. **Run GLP on training seeds only**:
   ```python
   predictions = guided_label_propagation(
       graph, id_mapper, train_seeds, labels, **glp_kwargs
   )
   ```

3. **Extract predictions for test nodes**:
   - Filter predictions to test seed nodes
   - Compare predicted `dominant_label` to actual labels

4. **Calculate metrics**:
   - Accuracy: fraction correct
   - Precision, Recall, F1: per-label and macro-averaged
   - Confusion matrix: predicted vs. actual

5. **Return comprehensive results**

**Use Cases**:
- Assess GLP performance before applying to full network
- Optimize hyperparameters (alpha, max_iterations)
- Compare different seed selection strategies

---

### Function: `external_validation()`

**Purpose**: Validate GLP results against independent labeled dataset

**Signature**:
```python
def external_validation(
    predictions: pl.DataFrame,
    validation_labels: Dict[Any, str],
    labels: List[str]
) -> Dict[str, Any]:
```

**Parameters**:
- `predictions`: Output from `guided_label_propagation()`
- `validation_labels`: Independent labeled nodes (Dict: node_id → label)
- `labels`: List of possible labels

**Returns**:
Same metrics dictionary as `train_test_split_validation()`

**Logic**:
1. Filter `predictions` to nodes in `validation_labels`
2. Compare predicted `dominant_label` to validation labels
3. Calculate accuracy, precision, recall, F1, confusion matrix
4. Return metrics

**Use Case**:
- Validate against expert-coded sample
- Compare to ground truth from external source
- Assess generalization beyond seed set

---

### Function: `cross_validate()`

**Purpose**: K-fold cross-validation for robust performance estimation

**Signature**:
```python
def cross_validate(
    graph: nk.Graph,
    id_mapper: IDMapper,
    seed_labels: Dict[Any, str],
    labels: List[str],
    k_folds: int = 5,
    stratify: bool = True,
    random_seed: Optional[int] = None,
    n_jobs: int = 1,
    **glp_kwargs
) -> Dict[str, Any]:
```

**Parameters**:
- Same as `train_test_split_validation()` but with `k_folds` instead of `test_size`
- `n_jobs`: Parallelize across folds

**Returns**:
```python
{
    "mean_accuracy": float,
    "std_accuracy": float,
    "fold_accuracies": List[float],
    "mean_f1": Dict[str, float],      # Per-label, averaged across folds
    "fold_results": List[Dict],       # Full results for each fold
    "aggregate_confusion_matrix": np.ndarray
}
```

**Logic**:
1. Split seeds into k folds (stratified if requested)
2. For each fold:
   - Use fold as test set, others as training
   - Run GLP and calculate metrics
   - Store results
3. Aggregate metrics across folds:
   - Mean and standard deviation
   - Per-label averages
   - Sum confusion matrices

---

## Module: `src/glp/evaluation.py`

### Function: `analyze_label_distribution()`

**Purpose**: Analyze distribution of label probabilities across network

**Signature**:
```python
def analyze_label_distribution(
    predictions: pl.DataFrame,
    labels: List[str]
) -> Dict[str, Any]:
```

**Returns**:
```python
{
    "label_counts": Dict[str, int],           # Dominant label counts
    "mean_confidence": float,                  # Average confidence score
    "confidence_by_label": Dict[str, float],  # Mean confidence per label
    "probability_distributions": Dict[str, np.ndarray],  # Histogram per label
    "high_confidence_nodes": pl.DataFrame,    # Nodes with confidence > 0.8
    "uncertain_nodes": pl.DataFrame,          # Nodes with confidence < 0.5
}
```

**Use Case**: 
- Understand label distribution in network
- Identify highly confident vs. uncertain classifications
- Quality check on propagation results

---

### Function: `compare_directional_results()`

**Purpose**: Compare in-degree vs. out-degree propagation results

**Signature**:
```python
def compare_directional_results(
    out_degree_predictions: pl.DataFrame,
    in_degree_predictions: pl.DataFrame,
    labels: List[str]
) -> Dict[str, Any]:
```

**Returns**:
```python
{
    "agreement_rate": float,                   # % nodes with same dominant label
    "correlation_by_label": Dict[str, float],  # Probability correlation per label
    "divergent_nodes": pl.DataFrame,           # Nodes with different dominant labels
    "direction_bias": Dict[str, str],          # Which direction favors each label
}
```

**Use Case**:
- Identify nodes that influence vs. are influenced differently
- Understand directional asymmetries in network
- Find nodes with interesting directional patterns

---

## Module: `src/glp/utils.py`

### Function: `create_balanced_seed_set()`

**Purpose**: Create balanced seed set from potentially imbalanced data

**Signature**:
```python
def create_balanced_seed_set(
    candidate_seeds: Dict[Any, str],
    labels: List[str],
    n_per_label: Optional[int] = None,
    method: str = "undersample",
    random_seed: Optional[int] = None
) -> Dict[Any, str]:
```

**Parameters**:
- `candidate_seeds`: All available labeled nodes
- `labels`: All possible labels
- `n_per_label`: Target number per label (None = use smallest class size)
- `method`: 
  - "undersample": Sample from majority classes
  - "oversample": Allow duplicates in minority classes
- `random_seed`: For reproducibility

**Returns**:
- Balanced seed set dictionary

**Logic**:
1. Count seeds per label
2. Determine target size per label
3. Apply sampling method:
   - **Undersample**: Randomly sample `n_per_label` from each label
   - **Oversample**: Sample with replacement from minority classes
4. Return balanced seed dictionary

**Use Case**: 
- Prevent bias toward majority class in GLP
- Ensure fair representation of all labels
- Preprocess seed set before propagation

---

### Function: `suggest_alpha_value()`

**Purpose**: Suggest optimal alpha based on network properties

**Signature**:
```python
def suggest_alpha_value(
    graph: nk.Graph,
    seed_count: int,
    method: str = "network_structure"
) -> float:
```

**Parameters**:
- `graph`: NetworkIt graph
- `seed_count`: Number of seed nodes
- `method`: 
  - "network_structure": Based on clustering coefficient, density
  - "seed_ratio": Based on seed to total node ratio

**Returns**:
- Suggested alpha value (float between 0 and 1)

**Logic**:

**Network Structure Method**:
- High clustering → lower alpha (more local propagation)
- Low clustering → higher alpha (broader propagation)
- Formula: `alpha = 0.5 + 0.4 * (1 - clustering_coefficient)`

**Seed Ratio Method**:
- Many seeds relative to network → lower alpha (seeds have more influence)
- Few seeds → higher alpha (need more propagation)
- Formula: `alpha = 0.95 - 0.5 * (seed_count / total_nodes)`

**Default**: Return 0.85 if uncertain

**Use Case**:
- Help users choose alpha without manual tuning
- Provide starting point for hyperparameter optimization
