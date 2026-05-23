# Common Utilities API Reference

The common module provides shared utilities, ID mapping, validation, logging, and I/O functions used across the library.

## Module: `src.common.id_mapper`

### IDMapper Class

```python
class IDMapper:
    """
    Bidirectional mapping between original and internal node IDs.
    
    NetworkIt graphs require consecutive integer node IDs starting from 0,
    but input data typically uses arbitrary identifiers (strings, UUIDs, etc.).
    This class provides efficient bidirectional mapping to handle this conversion.
    """
```

#### Constructor

```python
def __init__(self)
```

Create an empty ID mapper.

**Examples:**

```python
from src.common.id_mapper import IDMapper

# Create empty mapper
mapper = IDMapper()

# Add mappings manually
mapper.add_mapping("user_alice", 0)
mapper.add_mapping("user_bob", 1)
mapper.add_mapping("user_charlie", 2)

print(f"Mapper size: {mapper.size()}")  # 3
```

#### add_mapping()

```python
def add_mapping(self, original_id: Any, internal_id: int) -> None
```

Add a bidirectional mapping between original and internal IDs.

**Parameters:**
- `original_id`: Original identifier (can be any hashable type)
- `internal_id`: NetworkIt internal ID (non-negative integer)

**Examples:**

```python
# Map string IDs to integers
mapper.add_mapping("alice@example.com", 0)
mapper.add_mapping("bob@example.com", 1)

# Map numeric IDs
mapper.add_mapping(12345, 2)
mapper.add_mapping(67890, 3)

# Map complex IDs
mapper.add_mapping(("user", "department_A", 1), 4)
```

#### get_internal()

```python
def get_internal(self, original_id: Any) -> int
```

Get the internal ID for an original ID.

**Examples:**

```python
internal_id = mapper.get_internal("alice@example.com")
print(f"Alice's internal ID: {internal_id}")  # 0

# Handle missing IDs
try:
    missing_id = mapper.get_internal("nonexistent@example.com")
except KeyError:
    print("ID not found in mapper")
```

#### get_original()

```python
def get_original(self, internal_id: int) -> Any
```

Get the original ID for an internal ID.

**Examples:**

```python
original_id = mapper.get_original(0)
print(f"Internal ID 0 maps to: {original_id}")  # alice@example.com

# Batch conversion
internal_ids = [0, 1, 2]
original_ids = [mapper.get_original(id) for id in internal_ids]
print(f"Original IDs: {original_ids}")
```

#### has_mapping()

```python
def has_mapping(self, original_id: Any) -> bool
```

Check if an original ID has a mapping.

#### get_all_mappings()

```python
def get_all_mappings(self) -> Dict[Any, int]
```

Get all original-to-internal mappings as a dictionary.

**Examples:**

```python
# Export all mappings
all_mappings = mapper.get_all_mappings()
print(f"Total mappings: {len(all_mappings)}")

# Save mappings to file
import json
with open("id_mappings.json", "w") as f:
    # Convert keys to strings for JSON serialization
    json_mappings = {str(k): v for k, v in all_mappings.items()}
    json.dump(json_mappings, f, indent=2)
```

#### Advanced Usage

```python
# Create mapper from existing data
def create_mapper_from_nodes(node_list: List[Any]) -> IDMapper:
    """Create ID mapper from a list of node IDs."""
    mapper = IDMapper()
    sorted_nodes = sorted(set(node_list), key=str)  # Deterministic ordering
    
    for internal_id, original_id in enumerate(sorted_nodes):
        mapper.add_mapping(original_id, internal_id)
    
    return mapper

# Example usage
node_ids = ["user_charlie", "user_alice", "user_bob", "user_alice"]  # With duplicates
mapper = create_mapper_from_nodes(node_ids)

print(f"Unique nodes: {mapper.size()}")  # 3
print("Mappings:")
for orig, internal in mapper.get_all_mappings().items():
    print(f"  {orig} -> {internal}")
```

## Module: `src.common.validators`

### validate_edgelist_dataframe()

```python
def validate_edgelist_dataframe(
    df: pl.DataFrame,
    source_col: str = "source",
    target_col: str = "target",
    weight_col: Optional[str] = None,
    allow_self_loops: bool = True,
    allow_duplicates: bool = True,
    min_weight: Optional[float] = None,
    max_weight: Optional[float] = None
) -> None
```

Validate edge list DataFrame structure and data quality.

**Parameters:**
- `df`: Edge list DataFrame to validate
- `source_col`: Source column name
- `target_col`: Target column name
- `weight_col`: Weight column name (optional)
- `allow_self_loops`: Whether self-loops are allowed
- `allow_duplicates`: Whether duplicate edges are allowed
- `min_weight`: Minimum allowed edge weight
- `max_weight`: Maximum allowed edge weight

**Examples:**

```python
import polars as pl
from src.common.validators import validate_edgelist_dataframe

# Valid edge list
edges = pl.DataFrame({
    "source": ["A", "B", "C"],
    "target": ["B", "C", "A"], 
    "weight": [1.0, 2.0, 1.5]
})

# Validate with defaults
validate_edgelist_dataframe(edges, "source", "target", "weight")
print("Edge list is valid!")

# Strict validation
try:
    validate_edgelist_dataframe(
        edges,
        source_col="source",
        target_col="target", 
        weight_col="weight",
        allow_self_loops=False,
        allow_duplicates=False,
        min_weight=0.0,
        max_weight=10.0
    )
except ValidationError as e:
    print(f"Validation failed: {e}")
```

### validate_seed_labels()

```python
def validate_seed_labels(
    seed_labels: Dict[Any, str],
    node_ids: Optional[Set[Any]] = None,
    required_labels: Optional[List[str]] = None
) -> None
```

Validate seed labels dictionary for GLP analysis.

**Examples:**

```python
from src.common.validators import validate_seed_labels

# Valid seed labels
seeds = {
    "user123": "progressive",
    "user456": "conservative", 
    "user789": "progressive"
}

# Basic validation
validate_seed_labels(seeds)

# Validate against known node set
all_nodes = {"user123", "user456", "user789", "user999"}
validate_seed_labels(seeds, node_ids=all_nodes)

# Ensure all required labels are present
required = ["progressive", "conservative", "moderate"]
try:
    validate_seed_labels(seeds, required_labels=required)
except ValidationError as e:
    print(f"Missing label: {e}")
```

## Module: `src.common.exceptions`

### Custom Exception Classes

The library defines several custom exception types for better error handling:

```python
# Base exception class
class GuidedLPError(Exception):
    """Base exception for all Guided LP errors."""

# Specific exception types
class ValidationError(GuidedLPError):
    """Raised when input validation fails."""

class GraphConstructionError(GuidedLPError):
    """Raised when graph construction fails."""

class DataFormatError(GuidedLPError):
    """Raised when input data format is invalid."""

class ConvergenceError(GuidedLPError):
    """Raised when algorithms fail to converge."""

class ComputationError(GuidedLPError):
    """Raised when numerical computations fail."""
```

**Examples:**

```python
from src.common.exceptions import ValidationError, GraphConstructionError

try:
    # Some operation that might fail
    graph, mapper = build_graph_from_edgelist(invalid_data)
except ValidationError as e:
    print(f"Data validation failed: {e}")
except GraphConstructionError as e:
    print(f"Could not build graph: {e}")
except Exception as e:
    print(f"Unexpected error: {e}")
```

### Validation Helper Functions

```python
def validate_parameter(
    value: Any,
    allowed_values: List[Any],
    param_name: str,
    function_name: str
) -> None
```

Validate that a parameter has an allowed value.

```python
def require_positive(
    value: Union[int, float],
    param_name: str,
    function_name: str
) -> None
```

Require that a numeric parameter is positive.

**Examples:**

```python
from src.common.exceptions import validate_parameter, require_positive

# Validate parameter choices
validate_parameter("louvain", ["louvain", "leiden"], "algorithm", "detect_communities")

# Validate positive numbers
require_positive(0.85, "alpha", "guided_label_propagation")
require_positive(100, "max_iterations", "guided_label_propagation")
```

## Module: `src.common.logging_config`

### get_logger()

```python
def get_logger(name: str) -> logging.Logger
```

Get a configured logger for a module.

**Examples:**

```python
from src.common.logging_config import get_logger

# Get logger for current module
logger = get_logger(__name__)

# Use in functions
def my_function():
    logger.info("Starting computation")
    logger.debug("Processing data with %d records", len(data))
    logger.warning("Found %d missing values", missing_count)
    logger.error("Computation failed: %s", error_message)
```

### LoggingTimer

```python
class LoggingTimer:
    """Context manager for timing operations with automatic logging."""
```

**Examples:**

```python
from src.common.logging_config import LoggingTimer

# Time operations automatically
with LoggingTimer("graph_construction"):
    graph, mapper = build_graph_from_edgelist(large_dataset)
    # Logs: "graph_construction completed in 2.34 seconds"

# Nested timers
with LoggingTimer("full_analysis"):
    with LoggingTimer("data_loading"):
        data = pl.read_csv("large_file.csv")
    
    with LoggingTimer("graph_building"):
        graph, mapper = build_graph_from_edgelist(data)
    
    with LoggingTimer("glp_computation"):
        results = guided_label_propagation(graph, mapper, seeds, labels)
```

## I/O and Export Utilities

### Export Functions

```python
def export_results(
    results: Dict[Any, Any],
    filepath: str,
    format: str = "csv",
    include_metadata: bool = True
) -> None
```

Export analysis results to various formats.

**Examples:**

```python
# Export GLP results
glp_results = guided_label_propagation(graph, mapper, seeds, labels)

# Export as CSV
export_results(glp_results, "political_affiliations.csv", format="csv")

# Export as JSON with metadata
export_results(
    glp_results, 
    "affiliations.json", 
    format="json",
    include_metadata=True
)

# Export centrality measures
centrality = calculate_centrality_measures(graph, mapper)
centrality.write_csv("centrality_scores.csv")
```

### Data Loading Utilities

```python
def load_network_data(
    filepath: str,
    format: str = "csv",
    **kwargs
) -> pl.DataFrame
```

Load network data from various formats with validation.

**Examples:**

```python
# Load CSV with automatic validation
edges = load_network_data("network.csv", format="csv")

# Load with custom parameters
edges = load_network_data(
    "network.tsv",
    format="csv",
    separator="\t",
    null_values=["NA", "NULL"]
)

# Load Parquet for large datasets
edges = load_network_data("large_network.parquet", format="parquet")
```

## Configuration and Settings

### Configuration Management

```python
from src.common.config import get_config, set_config

# Get current configuration
config = get_config()
print(f"Default alpha: {config['glp']['default_alpha']}")
print(f"Max iterations: {config['glp']['max_iterations']}")

# Update configuration
set_config("glp.default_alpha", 0.9)
set_config("logging.level", "DEBUG")

# Use configuration in functions
def guided_label_propagation_with_defaults(graph, mapper, seeds, labels, **kwargs):
    config = get_config()
    
    # Use config defaults if not specified
    alpha = kwargs.get("alpha", config["glp"]["default_alpha"])
    max_iter = kwargs.get("max_iterations", config["glp"]["max_iterations"])
    
    return guided_label_propagation(
        graph, mapper, seeds, labels, alpha=alpha, max_iterations=max_iter
    )
```

## Common Utility Patterns

### Robust Data Pipeline

```python
import polars as pl
from src.common.validators import validate_edgelist_dataframe
from src.common.exceptions import ValidationError, DataFormatError
from src.common.logging_config import get_logger, LoggingTimer

logger = get_logger(__name__)

def robust_graph_pipeline(filepath: str, **graph_params):
    """Robust pipeline with validation and error handling."""
    
    with LoggingTimer("full_pipeline"):
        try:
            # 1. Load data with validation
            logger.info("Loading edge list from %s", filepath)
            edges = pl.read_csv(filepath)
            
            # 2. Validate data structure
            validate_edgelist_dataframe(
                edges, 
                source_col=graph_params.get("source_col", "source"),
                target_col=graph_params.get("target_col", "target"),
                weight_col=graph_params.get("weight_col"),
                allow_self_loops=False,
                allow_duplicates=True
            )
            
            # 3. Build graph
            logger.info("Building graph with %d edges", len(edges))
            graph, mapper = build_graph_from_edgelist(edges, **graph_params)
            
            # 4. Log results
            info = get_graph_info(graph, mapper)
            logger.info("Graph built successfully: %s", info)
            
            return graph, mapper, info
            
        except ValidationError as e:
            logger.error("Data validation failed: %s", e)
            raise
        except DataFormatError as e:
            logger.error("Data format error: %s", e)
            raise
        except Exception as e:
            logger.error("Unexpected error in pipeline: %s", e)
            raise

# Usage
try:
    graph, mapper, info = robust_graph_pipeline(
        "social_network.csv",
        source_col="user_a",
        target_col="user_b", 
        directed=True
    )
    print(f"Successfully processed network: {info}")
except Exception as e:
    print(f"Pipeline failed: {e}")
```

### Memory-Efficient Processing

```python
def process_large_dataset_in_chunks(
    filepath: str,
    chunk_size: int = 10000,
    **processing_params
):
    """Process large datasets in memory-efficient chunks."""
    
    logger = get_logger(__name__)
    
    # Read in lazy mode for memory efficiency
    lazy_df = pl.scan_csv(filepath)
    total_rows = lazy_df.select(pl.count()).collect().item()
    
    logger.info("Processing %d rows in chunks of %d", total_rows, chunk_size)
    
    results = []
    for i in range(0, total_rows, chunk_size):
        chunk = lazy_df.slice(i, chunk_size).collect()
        
        # Process chunk
        chunk_result = process_chunk(chunk, **processing_params)
        results.append(chunk_result)
        
        logger.debug("Processed chunk %d-%d", i, min(i + chunk_size, total_rows))
    
    # Combine results
    return combine_results(results)
```