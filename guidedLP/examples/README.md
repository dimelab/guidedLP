# Guided Label Propagation Examples

This directory contains comprehensive examples demonstrating the key features and workflows of the Guided Label Propagation library.

## Examples Overview

### 1. `example_network_analysis.py` - Basic Network Analysis
**What it demonstrates:**
- Loading network data from edge lists
- Building graphs with ID mapping
- Calculating centrality metrics (degree, betweenness, closeness, eigenvector)
- Community detection using Louvain algorithm
- Network structure analysis (clustering, density)
- Data export to various formats

**Key learning points:**
- Basic network construction workflow
- Understanding centrality measures
- Network-level statistics
- Identifying influential nodes

### 2. `example_glp_analysis.py` - Guided Label Propagation
**What it demonstrates:**
- Semi-supervised community detection
- Label propagation with seed nodes
- Validation techniques (train/test split, cross-validation)
- Confidence analysis and prediction quality
- Comparison with ground truth
- Network visualization with community colors

**Key learning points:**
- How to set up and run GLP
- Interpreting propagation results
- Validating predictions
- Optimizing alpha parameter
- Understanding prediction confidence

### 3. `example_timeseries.py` - Temporal Network Analysis
**What it demonstrates:**
- Creating temporal network slices from timestamped data
- Tracking centrality metrics over time
- Calculating temporal statistics and trends
- Cross-category connection analysis
- Segregation index calculation
- Temporal pattern visualization

**Key learning points:**
- Temporal network construction
- Evolution of network structure
- Identifying trends and patterns
- Category-based temporal analysis
- Understanding network dynamics

## Sample Data

The `data/` directory contains realistic sample datasets:

- **`social_network.csv`**: Friendship network with interaction weights
- **`user_metadata.csv`**: User attributes (age, interests, location, communities)
- **`community_seeds.csv`**: Known community labels for some users
- **`temporal_communications.csv`**: Communication events over time

## Running the Examples

### Prerequisites
```bash
# Basic requirements
pip install polars numpy

# For visualizations (optional)
pip install matplotlib networkx
```

### Running Individual Examples
```bash
# Basic network analysis
python example_network_analysis.py

# Guided label propagation
python example_glp_analysis.py

# Temporal analysis
python example_timeseries.py
```

### Expected Output

Each example will:
1. Print detailed progress and analysis results to the console
2. Create an `output/` directory with exported files
3. Generate visualizations (if matplotlib is available)

## Output Files

After running the examples, you'll find:

### Network Analysis Output
- `social_network.gexf`: Graph file for visualization tools
- `centrality_analysis.csv`: Detailed centrality metrics
- `network_summary.txt`: Summary statistics

### GLP Analysis Output
- `glp_community_results.csv`: Community predictions with confidence
- `alpha_comparison.csv`: Results for different alpha values
- `community_visualization.png`: Network plot with community colors

### Temporal Analysis Output
- `temporal_centrality_metrics.csv`: Centrality metrics over time
- `temporal_statistics.csv`: Trend analysis and statistics
- `category_connections.csv`: Cross-category interaction patterns
- `network_growth.csv`: Network evolution metrics
- `temporal_analysis.png`: Multi-panel temporal visualization

## Understanding the Results

### Centrality Metrics
- **Degree**: Number of direct connections
- **Betweenness**: How often a node lies on shortest paths between others
- **Closeness**: How close a node is to all other nodes
- **Eigenvector**: Connections to well-connected nodes

### GLP Results
- **Dominant Label**: Predicted community for each node
- **Confidence**: Probability of the dominant label (0-1)
- **Seed Status**: Whether the node was used as a training example

### Temporal Statistics
- **Mean**: Average centrality across time
- **Trend**: Linear trend (positive = increasing influence)
- **Volatility**: Coefficient of variation (consistency)
- **Growth**: Percentage change from start to end

## Customization

You can easily adapt these examples for your own data:

1. **Replace the data files** in `data/` with your own CSV files
2. **Modify column names** in the code to match your data structure
3. **Adjust parameters** like alpha values, time intervals, or metrics
4. **Add your own analysis steps** following the established patterns

## Advanced Usage

### Custom Metrics
You can add custom centrality metrics by modifying the `metrics` list in the examples.

### Different Time Intervals
For temporal analysis, you can use:
- `"daily"`: One slice per day
- `"weekly"`: One slice per week
- `"monthly"`: One slice per month
- `"yearly"`: One slice per year

### Rolling Windows
Add `rolling_window=N` to temporal slicing for moving averages.

### Cumulative Networks
Set `cumulative=True` to include all edges up to each time point.

## Troubleshooting

### Common Issues

1. **Import errors**: Make sure you're running from the correct directory and the src path is available
2. **Visualization issues**: Install matplotlib and networkx for full visualization features
3. **Memory issues**: For large networks, consider using smaller time windows or reducing metrics
4. **Convergence warnings**: Normal for small/sparse networks; adjust alpha or threshold parameters

### Performance Tips

- Use `n_jobs=1` for reproducible results, higher values for speed
- For large temporal datasets, process subsets or use coarser time intervals
- Enable early stopping in GLP for faster convergence

## Further Reading

For more details on the algorithms and methods used:
- See the main library documentation
- Check the test files for additional usage patterns
- Review the academic papers cited in the main README