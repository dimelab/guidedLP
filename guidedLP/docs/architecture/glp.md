# Guided Label Propagation Architecture

## Conceptual Overview

### The Problem
In large social networks (10,000+ nodes), most entities are unknown. Traditional community detection identifies clusters based purely on network structure, but these clusters are initially meaningless - researchers must manually interpret what each cluster represents.

### The Solution: Guided Label Propagation (GLP)
Instead of discovering arbitrary clusters, GLP leverages a **small set of labeled seed nodes** to discover how the entire network relates to predefined categories of interest.

**Example Use Case**:
- Research Question: "How politically polarized is this Twitter network?"
- Known Seeds: 50 left-wing accounts + 50 right-wing accounts
- GLP Output: Probability scores for all 10,000 accounts (affinity to left vs. right)

### Key Innovation
GLP is **semi-supervised**: it requires minimal labeled data but can classify the entire network based on network structure and connection patterns to those seeds.

## Methodological Foundation

### Comparison to Standard Label Propagation

**Standard Label Propagation (Unsupervised)**:
1. Detect communities using structural algorithms (Louvain, Leiden, etc.)
2. Label nodes based on which community they belong to
3. Communities are arbitrary - require post-hoc interpretation
4. Suitable when you want to discover natural divisions

**Guided Label Propagation (Semi-Supervised)**:
1. Start with known labeled nodes (seed set)
2. Propagate those labels through network connections
3. Calculate probability of unlabeled nodes belonging to each category
4. Categories are predefined and meaningful from the start
5. Suitable when you have specific categories of interest

### Theoretical Basis
GLP is inspired by:
- **Label propagation algorithms** in machine learning
- **Random walk with restart** for ranking nodes
- **PageRank-style** probability diffusion
- **Homophily principle**: connected nodes tend to be similar

## Algorithm Design

### Core Algorithm Flow

```
1. Input:
   - Graph G (directed or undirected, weighted)
   - Seed nodes S = {s₁, s₂, ..., sₙ}
   - Labels L = {l₁, l₂, ..., lₖ}
   - Seed-to-label mapping: seed_labels[sᵢ] = lⱼ

2. Initialize:
   - For each node v in G:
     - If v ∈ S: P(v, label) = 1 if v has that label, 0 otherwise
     - If v ∉ S: P(v, label) = 0 for all labels

3. Propagate:
   - Iteratively update probabilities based on neighbors
   - P'(v, label) = Σ(weight(u,v) × P(u, label)) / Σ(all weights to v)
   - Continue until convergence or max iterations

4. Output:
   - For each node v: probability vector [P(v, l₁), P(v, l₂), ..., P(v, lₖ)]
```

### Matrix Formulation (for efficiency)

Given:
- **A**: Adjacency matrix (weighted, possibly directed)
- **D**: Degree matrix (diagonal)
- **Y**: Initial label matrix (n × k), where n=nodes, k=labels
  - Y[i,j] = 1 if node i is seed with label j, 0 otherwise

Transition matrix:
- **P = D⁻¹ A** (row-normalized adjacency matrix)

Iterative propagation:
- **F⁽ᵗ⁺¹⁾ = α P F⁽ᵗ⁾ + (1-α) Y**
  - α: propagation coefficient (typically 0.85)
  - (1-α) ensures seeds retain their labels
  
Converges to steady-state label probabilities.

## Directional Analysis

### Motivation
In directed networks (e.g., Twitter follows, retweets), direction matters:
- **Out-degree propagation**: "Who does this node influence?"
- **In-degree propagation**: "Who influences this node?"

### Implementation

**Out-Degree Propagation**:
- Use forward edges: A[i,j] = weight if edge i→j exists
- Interprets: "Node j is influenced by node i's label"
- Question: "What labels does this node spread?"

**In-Degree Propagation**:
- Use reverse edges: A'[i,j] = A[j,i]  
- Interprets: "Node i is influenced by node j's label"
- Question: "What labels does this node receive?"

**Both Analyses Should Run**:
- Undirected graphs: results are identical
- Directed graphs: provides complementary insights
  - Example: Account follows mostly left-wing (in-degree) but is retweeted by right-wing (out-degree)

## Input Specifications

### Required Inputs

**1. Constructed Graph**
- NetworkIt graph object (from network module)
- Can be unipartite or bipartite (projected)
- Can be directed or undirected
- Must have weighted edges

**2. Seed Nodes**
- Format: Dict or DataFrame mapping original node IDs to labels
  ```python
  seeds = {
      "user_123": "left_wing",
      "user_456": "left_wing",
      "user_789": "right_wing",
      ...
  }
  ```
- Minimum size: Technically 1 per label, but recommend 10+ per label
- Quality > Quantity: High-confidence seeds are critical

**3. Label Set**
- List of possible labels: `["left_wing", "right_wing"]`
- Binary (2 labels) or multi-class (3+ labels) supported
- Labels must be strings (for interpretability)

### Optional Parameters

- **alpha** (float, default=0.85): Propagation coefficient
  - Higher = more propagation from neighbors
  - Lower = seeds retain influence more strongly
  
- **max_iterations** (int, default=100): Maximum propagation iterations
  
- **convergence_threshold** (float, default=1e-6): Stop when changes < threshold
  
- **normalize** (bool, default=True): Normalize probabilities to sum to 1

## Output Specifications

### Primary Output: Node Label Probabilities

**Format**: Polars DataFrame

| node_id (original) | label_1_prob | label_2_prob | ... | dominant_label | confidence |
|-------------------|-------------|-------------|-----|----------------|-----------|
| user_123 | 0.92 | 0.08 | ... | label_1 | 0.92 |
| user_456 | 0.15 | 0.85 | ... | label_2 | 0.85 |
| ... | ... | ... | ... | ... | ... |

**Columns**:
- `node_id`: Original node ID from input data
- `{label}_prob`: Probability for each label (one column per label)
- `dominant_label`: Label with highest probability
- `confidence`: Maximum probability (max of all label probs)

### Secondary Outputs

**Convergence Metrics**:
- Number of iterations until convergence
- Final convergence error
- Labels changed per iteration (for analysis)

**Directional Results** (for directed graphs):
- Separate DataFrames for out-degree and in-degree propagation
- `_out` and `_in` suffixes on output files

## Validation & Evaluation

### Train/Test Split Validation

**Purpose**: Assess how well GLP can predict held-out seeds

**Method**:
1. Split seed nodes: 80% train, 20% test
2. Run GLP using only training seeds
3. Evaluate predictions on test seeds
4. Metrics: Accuracy, precision, recall, F1 per label

**Implementation**:
```python
def validate_glp(graph, all_seeds, labels, test_size=0.2):
    train_seeds, test_seeds = train_test_split(all_seeds, test_size)
    predictions = run_glp(graph, train_seeds, labels)
    
    # Compare predictions to actual labels for test_seeds
    metrics = calculate_metrics(predictions, test_seeds)
    return metrics
```

### External Validation Set

**Purpose**: Validate against completely independent labeled data

**Method**:
1. Run GLP using original seed set
2. Obtain external validation set (manually labeled or from other source)
3. Compare GLP predictions to validation labels
4. Report accuracy and confusion matrix

**Use Case**: 
- Original seeds: Known political accounts
- Validation set: Accounts labeled by expert coders
- Check: Does GLP agree with expert judgment?

### Cross-Validation

**Purpose**: Robust performance estimation

**Method**:
- K-fold cross-validation on seed set
- For each fold: train on K-1 folds, test on remaining fold
- Average performance across folds

## Performance Optimization

### Computational Complexity

**Naive Implementation**: O(k × i × n × d)
- k: number of labels
- i: iterations until convergence  
- n: number of nodes
- d: average degree

**Optimized (Matrix)**: O(k × i × e)
- e: number of edges
- Uses sparse matrix operations

### Memory Requirements

**Sparse Matrix Storage**:
- Store only non-zero entries in adjacency matrix
- Critical for large networks (10,000+ nodes)
- Use scipy.sparse matrices

**Lazy Evaluation**:
- Don't materialize full probability matrix until output
- Stream results for very large networks

### Parallelization Opportunities

**Multi-Label Parallel**:
- Each label can propagate independently
- Run k label propagations in parallel
- Combine results at the end

**Multi-Network Parallel**:
- When running on multiple graphs (e.g., time slices)
- Process each graph in parallel

## Edge Cases & Error Handling

### Disconnected Components
- **Issue**: Seeds only in one component
- **Solution**: Other components get uniform probability distribution
- **Warning**: Alert user if seeds don't cover all components

### Unbalanced Seed Sets
- **Issue**: 100 left-wing seeds, 5 right-wing seeds
- **Solution**: Normalize initial probabilities OR weight by seed set size
- **Recommendation**: Aim for balanced seeds when possible

### No Path to Seeds
- **Issue**: Node has no path (direct or indirect) to any seed
- **Solution**: Assign uniform probability across all labels
- **Detection**: Check for unreachable nodes before propagation

### Graph Types
- **Bipartite**: Project to unipartite first (seeds must be in one partition)
- **Undirected**: Run once (in-degree = out-degree)
- **Directed**: Run both directions, provide both results

## Integration with Network Module

### Required Graph Properties
```python
# Graph must provide:
graph.number_of_nodes()
graph.number_of_edges()
graph.get_adjacency_matrix()  # Returns scipy sparse matrix
graph.get_original_id(internal_id)  # ID mapping
graph.get_internal_id(original_id)
```

### Workflow Integration
```python
# 1. Build graph (network module)
from network import build_graph
graph = build_graph(edgelist_path)

# 2. Run GLP (glp module)
from glp import guided_label_propagation
seeds = {"user1": "label_a", "user2": "label_b", ...}
results = guided_label_propagation(graph, seeds, ["label_a", "label_b"])

# 3. Export results
results.write_parquet("glp_results.parquet")
```

## Reference Implementation

### Inspiration Source
The existing `stlp` function in `net_utils.py` provides a reference implementation. Key elements to preserve:
- Matrix-based calculation approach
- Convergence checking
- Label probability normalization

### Improvements Over Reference
- **Explicit ID mapping**: Separate internal vs. original IDs
- **Directional support**: Both in and out-degree
- **Validation built-in**: Train/test split and external validation
- **Performance metrics**: Track propagation statistics
- **Better error handling**: Graceful handling of edge cases
