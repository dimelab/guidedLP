"""
Network construction module for the Guided Label Propagation library.

This module provides functionality for constructing NetworkIt graphs from edge lists
while preserving original node IDs and supporting various graph types including
directed, undirected, weighted, unweighted, and bipartite graphs.
"""

from typing import Union, Tuple, Optional, List, Dict, Any, Set
import warnings
from pathlib import Path

import polars as pl
import networkit as nk
import numpy as np
import scipy.sparse as sp

from guidedLP.common.id_mapper import IDMapper
from guidedLP.common.exceptions import (
    GraphConstructionError,
    ValidationError,
    DataFormatError,
    ConfigurationError,
    validate_parameter
)
from guidedLP.common.validators import validate_edgelist_dataframe
from guidedLP.common.logging_config import get_logger, log_function_entry, LoggingTimer

logger = get_logger(__name__)


def _print_build_summary(
    verbose: bool, t_start: float, input_rows: int, graph: nk.Graph
) -> None:
    """One-line summary printed at the bottom of build_graph_from_edgelist."""
    if not verbose:
        return
    import time as _time
    dt = _time.perf_counter() - t_start
    print(
        f"[build_graph_from_edgelist] {dt:.2f}s | "
        f"{input_rows:,} input rows → "
        f"{graph.numberOfNodes():,} nodes, {graph.numberOfEdges():,} edges"
    )


def _print_projection_summary(
    verbose: bool,
    t_start: float,
    bipartite_graph: nk.Graph,
    projected_graph: nk.Graph,
    projection_mode: str,
    weight_method: str,
) -> None:
    """One-line summary printed at the bottom of project_bipartite."""
    if not verbose:
        return
    import time as _time
    dt = _time.perf_counter() - t_start
    print(
        f"[project_bipartite] {dt:.2f}s | mode={projection_mode}, "
        f"weight={weight_method} | "
        f"bipartite: {bipartite_graph.numberOfNodes():,} nodes, "
        f"{bipartite_graph.numberOfEdges():,} edges → "
        f"projected: {projected_graph.numberOfNodes():,} nodes, "
        f"{projected_graph.numberOfEdges():,} edges"
    )


def _print_backbone_summary(
    verbose: bool,
    t_start: float,
    original_graph: nk.Graph,
    backbone_graph: nk.Graph,
    method: str,
) -> None:
    """One-line summary printed at the bottom of apply_backbone."""
    if not verbose:
        return
    import time as _time
    dt = _time.perf_counter() - t_start
    n0, e0 = original_graph.numberOfNodes(), original_graph.numberOfEdges()
    n1, e1 = backbone_graph.numberOfNodes(), backbone_graph.numberOfEdges()
    node_pct = (100 * n1 / n0) if n0 else 0.0
    edge_pct = (100 * e1 / e0) if e0 else 0.0
    print(
        f"[apply_backbone] {dt:.2f}s | method={method} | "
        f"{n0:,} → {n1:,} nodes ({node_pct:.1f}% kept) | "
        f"{e0:,} → {e1:,} edges ({edge_pct:.1f}% kept)"
    )


def build_graph_from_edgelist(
    edgelist: Union[str, pl.DataFrame],
    source_col: str = "source",
    target_col: str = "target",
    weight_col: Optional[str] = None,
    directed: bool = False,
    bipartite: bool = False,
    auto_weight: bool = True,
    allow_self_loops: bool = True,
    remove_duplicates: bool = False,
    bipartite_overlap: str = "raise",
    chunk_size: Optional[int] = None,
    min_source_degree: Optional[int] = None,
    min_target_degree: Optional[int] = None,
    verbose: bool = True,
) -> Tuple[nk.Graph, IDMapper]:
    """
    Construct a NetworkIt graph from an edge list with ID preservation.
    
    This function creates a NetworkIt graph from edge list data while preserving
    original node identifiers through an ID mapping system. It supports various
    graph types and handles weight calculation from duplicate edges.
    
    Parameters
    ----------
    edgelist : Union[str, pl.DataFrame]
        Path to CSV file or Polars DataFrame containing edge data
    source_col : str, default "source"
        Name of the source node column in the edge list
    target_col : str, default "target"  
        Name of the target node column in the edge list
    weight_col : str, optional
        Name of the edge weight column. If None and auto_weight is True,
        weights will be calculated from duplicate edge counts
    directed : bool, default False
        If True, create a directed graph; otherwise undirected
    bipartite : bool, default False
        If True, validate that source and target nodes form distinct sets
        (bipartite graph structure)
    auto_weight : bool, default True
        If True and weight_col is None, calculate edge weights by counting
        duplicate (source, target) pairs
    allow_self_loops : bool, default True
        If True, allow edges from a node to itself; otherwise remove them
    remove_duplicates : bool, default False
        If True, remove duplicate edges before processing (keeps first occurrence)
    chunk_size : Optional[int], default None
        Optional batch size for the edge-emission loop. When ``None`` (the
        default), source/target/weight columns are materialized as numpy
        arrays in one shot — fastest, slightly higher peak memory. When set
        (e.g. ``1_000_000``), edges are processed in batches of that size,
        keeping peak memory close to the input DataFrame size. Use this when
        the graph is large enough that the input edgelist already occupies a
        sizeable fraction of available RAM.
    min_source_degree : Optional[int], default None
        If set, drop every node whose count of edges in the source column is
        less than this threshold (i.e. nodes that appear as source on fewer
        than ``min_source_degree`` distinct edges, after auto-weight dedup).
        The edges incident to dropped sources are also removed. Most useful
        for bipartite preprocessing — e.g. "only keep users who posted at
        least 3 hashtags".
    min_target_degree : Optional[int], default None
        Same as ``min_source_degree`` but applied to the target column.
        Both kwargs use INDEPENDENT thresholds counted on the deduplicated
        edgelist: filtering is one-pass, not iterative. If you need
        k-core-style iterative filtering, call this function repeatedly or
        post-process with ``network.filtering.filter_graph``.
    verbose : bool, default True
        If True, print a brief summary (input row count, final node/edge
        counts, wall-clock time) when the function returns. Set to False to
        silence this output (e.g. in batch pipelines).
    bipartite_overlap : str, default "raise"
        How to handle nodes that appear in both source and target columns when
        ``bipartite=True``. Only consulted when ``bipartite`` is True.

        - ``"raise"``: default. Raise ``GraphConstructionError`` if any node is
          in both columns (strictly bipartite input required).
        - ``"drop"``: remove all edges where source or target is an overlap
          node, emit a ``UserWarning`` with the count. The resulting graph is
          strictly bipartite — safe for ``project_bipartite`` etc.
        - ``"warn"``: keep all edges, emit a ``UserWarning``. Both partitions
          on the returned :class:`IDMapper` will contain the overlap nodes.
          Downstream projection results may be ambiguous on those nodes.

    Returns
    -------
    graph : nk.Graph
        Constructed NetworkIt graph object
    id_mapper : IDMapper
        Bidirectional mapping between original IDs and NetworkIt internal IDs
        
    Raises
    ------
    ValidationError
        If edge list data is invalid or missing required columns
    GraphConstructionError
        If graph construction fails due to NetworkIt issues
    DataFormatError
        If input file cannot be read or parsed
        
    Examples
    --------
    >>> import polars as pl
    >>> edges = pl.DataFrame({
    ...     "source": ["A", "B", "C", "A"],
    ...     "target": ["B", "C", "A", "B"]
    ... })
    >>> graph, id_mapper = build_graph_from_edgelist(edges)
    >>> graph.numberOfNodes()
    3
    >>> graph.numberOfEdges() 
    4
    
    >>> # Directed graph with weights
    >>> weighted_edges = pl.DataFrame({
    ...     "from": ["A", "B"], 
    ...     "to": ["B", "A"],
    ...     "weight": [1.5, 2.0]
    ... })
    >>> graph, mapper = build_graph_from_edgelist(
    ...     weighted_edges, 
    ...     source_col="from", 
    ...     target_col="to",
    ...     weight_col="weight", 
    ...     directed=True
    ... )
    
    Notes
    -----
    Time Complexity: O(E) where E is the number of edges
    Space Complexity: O(V + E) where V is the number of unique nodes
    
    The function performs the following steps:
    1. Load edge list data using Polars (lazy evaluation for CSV files)
    2. Validate column existence and data quality
    3. Calculate weights from duplicates if auto_weight is enabled
    4. Extract unique node IDs and create ID mapping
    5. Construct NetworkIt graph with appropriate configuration
    6. Add edges with weights to the graph
    
    For bipartite graphs, the function validates that source and target nodes
    form completely distinct sets with no overlap.

    Rows with null values in the source, target, or weight (when ``weight_col``
    is set) columns are silently dropped before validation, with a
    ``UserWarning`` reporting the count. This makes the function tolerant of
    the small fraction of null cells that real-world edgelists routinely
    accumulate from joins and missing log data.
    """
    log_function_entry("build_graph_from_edgelist",
                      edgelist=type(edgelist).__name__,
                      directed=directed, bipartite=bipartite)

    import time as _time
    _t_start = _time.perf_counter()
    _input_rows: Optional[int] = None  # filled in once we've loaded the df


    with LoggingTimer("build_graph_from_edgelist"):
        try:
            # Step 1: Load edge list data
            df = _load_edge_list(edgelist)
            _input_rows = len(df)

            # Step 2: Handle empty edge list BEFORE running schema validation,
            # which would otherwise raise on the empty DataFrame and prevent
            # the warn-and-return path below from firing. An empty edgelist
            # is a non-fatal case — the function returns an empty graph.
            if df.is_empty():
                warnings.warn("Empty edge list provided. Creating empty graph.")
                empty_graph = nk.Graph(0, weighted=(weight_col is not None), directed=directed)
                empty_mapper = IDMapper()
                _print_build_summary(verbose, _t_start, _input_rows or 0, empty_graph)
                return empty_graph, empty_mapper

            # Step 2b: Confirm the requested columns exist BEFORE attempting
            # any per-column operations (drop_nulls would otherwise raise a
            # raw ColumnNotFoundError that gets wrapped as a generic
            # GraphConstructionError, hiding the real problem).
            required_cols = [source_col, target_col]
            if weight_col is not None:
                required_cols.append(weight_col)
            missing_cols = [c for c in required_cols if c not in df.columns]
            if missing_cols:
                raise ValidationError(
                    f"Missing required columns: {missing_cols}. "
                    f"Available columns: {df.columns}"
                )

            # Step 2c: Drop rows with null source/target (and null weight if a
            # weight column was specified). Treated as a non-fatal data-quality
            # issue rather than a hard error — large real-world edgelists
            # routinely contain a handful of null cells from joins or missing
            # log data, and dropping them is the universally expected
            # behaviour. Mirrors how check_seed_coverage handles null labels.
            null_check_cols = [source_col, target_col]
            if weight_col is not None:
                null_check_cols.append(weight_col)
            rows_before = len(df)
            df = df.drop_nulls(subset=null_check_cols)
            n_dropped = rows_before - len(df)
            if n_dropped > 0:
                warnings.warn(
                    f"Dropped {n_dropped} edge(s) with null value(s) in "
                    f"{null_check_cols} (from {rows_before} input rows)."
                )
            if df.is_empty():
                warnings.warn(
                    "All edges had null source/target/weight values; "
                    "creating empty graph."
                )
                empty_graph = nk.Graph(0, weighted=(weight_col is not None), directed=directed)
                empty_mapper = IDMapper()
                _print_build_summary(verbose, _t_start, _input_rows or 0, empty_graph)
                return empty_graph, empty_mapper

            # Step 3: Validate edge list structure and data
            _validate_edge_list(df, source_col, target_col, weight_col)
            
            # Step 4: Process edges (weights, duplicates, self-loops).
            # When auto_weight is on and the user didn't pass weight_col,
            # _process_edges injects a 'weight' column from duplicate counts.
            # Treat the graph as weighted only when those counts actually
            # carry information (i.e. there was at least one duplicate); for
            # a fully-unique edgelist all counts are 1, and producing a
            # weighted graph would be a surprising default.
            processed_df = _process_edges(
                df, source_col, target_col, weight_col,
                auto_weight, allow_self_loops, remove_duplicates
            )
            if (
                weight_col is None
                and auto_weight
                and "weight" in processed_df.columns
                and processed_df["weight"].max() > 1
            ):
                weight_col = "weight"
            
            # Step 5: Handle bipartite source/target overlap BEFORE building
            # the id mapper, so the "drop" policy can remove offending edges
            # cleanly. The "raise" path delegates to the existing strict
            # validator; "warn" keeps everything and "drop" filters the
            # processed DataFrame.
            bipartite_overlap_used = False
            if bipartite:
                processed_df, bipartite_overlap_used = _apply_bipartite_overlap_policy(
                    processed_df, source_col, target_col, bipartite_overlap
                )

            # Step 5b: Drop low-degree source/target nodes (and their edges).
            # Counted on the post-dedup edgelist so "degree" matches the
            # graph-theoretic meaning. Independent thresholds — not iterative.
            if min_source_degree is not None or min_target_degree is not None:
                processed_df = _apply_min_degree_filter(
                    processed_df,
                    source_col,
                    target_col,
                    min_source_degree,
                    min_target_degree,
                )
                if processed_df.is_empty():
                    warnings.warn(
                        "All edges removed by min_source_degree / "
                        "min_target_degree filtering; creating empty graph."
                    )
                    empty_graph = nk.Graph(
                        0, weighted=(weight_col is not None), directed=directed,
                    )
                    empty_mapper = IDMapper()
                    _print_build_summary(verbose, _t_start, _input_rows or 0, empty_graph)
                    return empty_graph, empty_mapper


            # Step 6: Create ID mapping (over the possibly-filtered df).
            # _create_id_mapping returns the source/target unique lists too,
            # so we can reuse them for bipartite partition recording without
            # a redundant .unique() pass over the (potentially huge) df.
            id_mapper, source_unique, target_unique = _create_id_mapping(
                processed_df, source_col, target_col,
            )

            # Step 7: Record partition info on the id mapper so
            # project_bipartite and similar functions don't have to fall back
            # to BFS coloring (which produces arbitrary labels).
            if bipartite:
                source_set = set(source_unique)
                target_set = set(target_unique)
                if bipartite_overlap_used:
                    # "warn" mode: partitions intentionally overlap. Set the
                    # attributes directly because set_bipartite_partitions()
                    # rejects overlap by design.
                    id_mapper.source_partition_originals = source_set
                    id_mapper.target_partition_originals = target_set
                else:
                    id_mapper.set_bipartite_partitions(source_set, target_set)

            # Step 8: Construct NetworkIt graph
            graph = _construct_graph(
                processed_df, id_mapper, source_col, target_col,
                weight_col, directed, bipartite,
                chunk_size=chunk_size,
            )
            
            logger.info("Graph construction completed: %d nodes, %d edges, directed=%s, weighted=%s",
                       graph.numberOfNodes(), graph.numberOfEdges(), directed,
                       weight_col is not None)

            _print_build_summary(verbose, _t_start, _input_rows or 0, graph)
            return graph, id_mapper
            
        except Exception as e:
            if isinstance(e, (ValidationError, GraphConstructionError, DataFormatError, ConfigurationError)):
                raise
            else:
                raise GraphConstructionError(
                    f"Unexpected error during graph construction: {str(e)}",
                    operation="build_graph_from_edgelist",
                    cause=e
                )


def _load_edge_list(edgelist: Union[str, pl.DataFrame]) -> pl.DataFrame:
    """
    Load edge list from file path or return DataFrame as-is.
    
    Parameters
    ----------
    edgelist : Union[str, pl.DataFrame]
        File path to CSV or DataFrame
        
    Returns
    -------
    pl.DataFrame
        Loaded edge list data
        
    Raises
    ------
    DataFormatError
        If file cannot be read or parsed
    """
    if isinstance(edgelist, str):
        try:
            # Convert to Path for better handling
            file_path = Path(edgelist)
            
            if not file_path.exists():
                raise DataFormatError(
                    f"Edge list file not found: {edgelist}",
                    format_type="CSV",
                    file_path=edgelist
                )
            
            # Use Polars lazy loading for efficiency
            logger.debug("Loading edge list from file: %s", edgelist)
            df = pl.read_csv(file_path)
            
        except pl.exceptions.ComputeError as e:
            raise DataFormatError(
                f"Failed to parse CSV file: {str(e)}",
                format_type="CSV",
                file_path=edgelist,
                cause=e
            )
        except Exception as e:
            raise DataFormatError(
                f"Error reading file: {str(e)}",
                format_type="CSV", 
                file_path=edgelist,
                cause=e
            )
    
    elif isinstance(edgelist, pl.DataFrame):
        df = edgelist
    else:
        raise DataFormatError(
            f"Invalid edgelist type: {type(edgelist)}. Expected str or pl.DataFrame",
            format_type="DataFrame"
        )
    
    return df


def _validate_edge_list(
    df: pl.DataFrame,
    source_col: str,
    target_col: str,
    weight_col: Optional[str]
) -> None:
    """
    Validate edge list DataFrame structure and data quality.
    
    Parameters
    ----------
    df : pl.DataFrame
        Edge list DataFrame
    source_col : str
        Source column name
    target_col : str
        Target column name  
    weight_col : str, optional
        Weight column name
        
    Raises
    ------
    ValidationError
        If validation fails
    """
    logger.debug("Validating edge list with %d rows", len(df))
    
    # Use the existing validator from common module
    validate_edgelist_dataframe(
        df,
        source_col=source_col,
        target_col=target_col,
        weight_col=weight_col,
        allow_self_loops=True,  # Will be handled separately
        allow_duplicates=True   # Will be handled separately
    )


def _process_edges(
    df: pl.DataFrame,
    source_col: str,
    target_col: str,
    weight_col: Optional[str],
    auto_weight: bool,
    allow_self_loops: bool,
    remove_duplicates: bool
) -> pl.DataFrame:
    """
    Process edges: calculate weights, handle duplicates, filter self-loops.
    
    Parameters
    ----------
    df : pl.DataFrame
        Input edge list
    source_col : str
        Source column name
    target_col : str
        Target column name
    weight_col : str, optional
        Weight column name
    auto_weight : bool
        Whether to calculate weights from duplicates
    allow_self_loops : bool
        Whether to allow self-loops
    remove_duplicates : bool
        Whether to remove duplicate edges
        
    Returns
    -------
    pl.DataFrame
        Processed edge list with weights
    """
    logger.debug("Processing edges: auto_weight=%s, allow_self_loops=%s, remove_duplicates=%s",
                auto_weight, allow_self_loops, remove_duplicates)

    # Polars operations are functional — filter/group_by return new
    # DataFrames without mutating the input. The previous df.clone() doubled
    # peak memory for no benefit on multi-million-row edgelists.
    processed_df = df
    
    # Remove self-loops if not allowed
    if not allow_self_loops:
        initial_count = len(processed_df)
        processed_df = processed_df.filter(pl.col(source_col) != pl.col(target_col))
        removed_count = initial_count - len(processed_df)
        if removed_count > 0:
            logger.info("Removed %d self-loop edges", removed_count)
    
    # Handle weights and duplicates
    if weight_col is None and auto_weight:
        # Calculate weights from duplicate edge counts
        logger.debug("Calculating automatic weights from duplicate edges")
        processed_df = processed_df.group_by([source_col, target_col]).agg(
            pl.len().alias("weight")
        ).with_columns(pl.col("weight").cast(pl.Float64))
        weight_col = "weight"
        
    elif weight_col is not None and not remove_duplicates:
        # Sum weights for duplicate edges
        logger.debug("Aggregating weights for duplicate edges")
        processed_df = processed_df.group_by([source_col, target_col]).agg(
            pl.col(weight_col).sum().alias(weight_col)
        )
        
    elif remove_duplicates:
        # Remove duplicate edges (keep first occurrence)
        initial_count = len(processed_df)
        processed_df = processed_df.unique(subset=[source_col, target_col], keep="first")
        removed_count = initial_count - len(processed_df)
        if removed_count > 0:
            logger.info("Removed %d duplicate edges", removed_count)
    
    return processed_df


def _create_id_mapping(
    df: pl.DataFrame,
    source_col: str,
    target_col: str
) -> Tuple[IDMapper, List[Any], List[Any]]:
    """
    Create bidirectional ID mapping between original and NetworkIt IDs.

    Returns the mapper alongside the source/target unique-value lists so
    callers (notably bipartite-partition recording) can avoid running
    ``.unique()`` a second time on the same large DataFrame.

    Parameters
    ----------
    df : pl.DataFrame
        Edge list DataFrame
    source_col : str
        Source column name
    target_col : str
        Target column name

    Returns
    -------
    id_mapper : IDMapper
        Configured ID mapper. Internal IDs assigned in sorted-by-str order
        of original IDs (deterministic across runs).
    source_unique : List[Any]
        Unique original IDs from the source column.
    target_unique : List[Any]
        Unique original IDs from the target column.

    Raises
    ------
    GraphConstructionError
        If ID mapping creation fails
    """
    try:
        logger.debug("Creating ID mapping from edge list")

        # Two C-level unique() passes — fast, and we hand the results back
        # to the caller so they don't get repeated for bipartite recording.
        source_ids = df[source_col].unique().to_list()
        target_ids = df[target_col].unique().to_list()

        # Union of source and target IDs, sorted deterministically by str.
        all_ids = sorted(set(source_ids) | set(target_ids), key=str)

        # Bulk-build the mapper. Skips the per-element type/uniqueness checks
        # that add_mapping() performs — they're redundant since `set(...)`
        # already deduped and only hashable values made it into the set.
        id_mapper = IDMapper.from_originals(all_ids)

        logger.debug("Created ID mapping for %d unique nodes", len(all_ids))
        return id_mapper, source_ids, target_ids

    except Exception as e:
        raise GraphConstructionError(
            f"Failed to create ID mapping: {str(e)}",
            operation="create_id_mapping",
            cause=e
        )


def _apply_min_degree_filter(
    df: pl.DataFrame,
    source_col: str,
    target_col: str,
    min_source_degree: Optional[int],
    min_target_degree: Optional[int],
) -> pl.DataFrame:
    """
    Drop rows whose source or target node is below the per-column degree
    threshold. Counts are taken on the input DataFrame as-is (caller must
    have already deduplicated edges if "degree" should mean distinct
    neighbors).

    Both thresholds are applied independently, using the original counts —
    not iteratively. So when both kwargs are set, the function:

      1. Counts each source value's occurrences in the source column.
      2. Counts each target value's occurrences in the target column.
      3. Drops every edge whose source count or target count is below its
         respective threshold.

    Some surviving nodes may end up with lower realized degree than the
    threshold after filtering (because the OTHER threshold removed some
    of their neighbors). That's intentional — one-pass behaviour is
    predictable; iterative filtering can be done by calling the function
    in a loop.
    """
    rows_before = len(df)
    sources_before: Optional[int] = None
    targets_before: Optional[int] = None
    sources_kept: Optional[int] = None
    targets_kept: Optional[int] = None

    keep_mask: Optional[pl.Expr] = None

    if min_source_degree is not None and min_source_degree > 1:
        src_counts = (
            df.group_by(source_col)
            .agg(pl.len().alias("_glp_src_degree"))
        )
        sources_before = len(src_counts)
        valid_sources = src_counts.filter(
            pl.col("_glp_src_degree") >= min_source_degree
        )[source_col].to_list()
        sources_kept = len(valid_sources)
        condition = pl.col(source_col).is_in(valid_sources)
        keep_mask = condition if keep_mask is None else keep_mask & condition

    if min_target_degree is not None and min_target_degree > 1:
        tgt_counts = (
            df.group_by(target_col)
            .agg(pl.len().alias("_glp_tgt_degree"))
        )
        targets_before = len(tgt_counts)
        valid_targets = tgt_counts.filter(
            pl.col("_glp_tgt_degree") >= min_target_degree
        )[target_col].to_list()
        targets_kept = len(valid_targets)
        condition = pl.col(target_col).is_in(valid_targets)
        keep_mask = condition if keep_mask is None else keep_mask & condition

    # If no filter was actually applied (both thresholds None or <= 1), return
    # the input unchanged — no warning, no work.
    if keep_mask is None:
        return df

    df = df.filter(keep_mask)
    rows_after = len(df)
    edges_dropped = rows_before - rows_after

    # One warning summarising the filter effect, only if it actually removed
    # something.
    if edges_dropped > 0 or (sources_before is not None and sources_kept != sources_before) \
       or (targets_before is not None and targets_kept != targets_before):
        parts = []
        if sources_before is not None:
            parts.append(
                f"sources: kept {sources_kept}/{sources_before} "
                f"(min_source_degree={min_source_degree})"
            )
        if targets_before is not None:
            parts.append(
                f"targets: kept {targets_kept}/{targets_before} "
                f"(min_target_degree={min_target_degree})"
            )
        warnings.warn(
            "Degree filter dropped "
            f"{edges_dropped} of {rows_before} edges; "
            + "; ".join(parts)
        )

    return df


def _apply_bipartite_overlap_policy(
    df: pl.DataFrame,
    source_col: str,
    target_col: str,
    policy: str,
) -> Tuple[pl.DataFrame, bool]:
    """
    Apply the configured bipartite-overlap policy.

    Parameters
    ----------
    df : pl.DataFrame
        Edge list (after _process_edges).
    source_col, target_col : str
        Column names.
    policy : {"raise", "drop", "warn"}
        - "raise": fail with GraphConstructionError on any overlap.
        - "drop": filter out edges where source or target is in the overlap
          set; emit a UserWarning with counts.
        - "warn": emit a UserWarning, leave the DataFrame untouched.

    Returns
    -------
    df : pl.DataFrame
        Possibly filtered edge list.
    overlap_kept : bool
        True iff overlap was kept (i.e. policy was "warn" and overlap was
        non-empty). Used by the caller to know whether the IDMapper
        partitions will intentionally overlap.
    """
    valid_policies = ("raise", "drop", "warn")
    if policy not in valid_policies:
        raise ConfigurationError(
            f"Invalid bipartite_overlap value: {policy!r}. "
            f"Expected one of {valid_policies}."
        )

    source_set = set(df[source_col].unique().to_list())
    target_set = set(df[target_col].unique().to_list())
    overlap = source_set & target_set

    if not overlap:
        return df, False

    sample = list(overlap)[:10]

    if policy == "raise":
        raise GraphConstructionError(
            f"Graph is not bipartite: {len(overlap)} nodes appear in both "
            f"source and target",
            graph_type="bipartite",
            details={
                "overlapping_nodes": sample,
                "total_overlap": len(overlap),
                "source_partition_size": len(source_set),
                "target_partition_size": len(target_set),
            },
        )

    if policy == "drop":
        rows_before = len(df)
        overlap_list = list(overlap)
        df = df.filter(
            ~pl.col(source_col).is_in(overlap_list)
            & ~pl.col(target_col).is_in(overlap_list)
        )
        n_dropped = rows_before - len(df)
        warnings.warn(
            f"bipartite_overlap='drop': removed {len(overlap)} overlap node(s) "
            f"and {n_dropped} edge(s) (sample: {sample}). Graph is now "
            f"strictly bipartite."
        )
        return df, False

    # policy == "warn"
    warnings.warn(
        f"bipartite_overlap='warn': {len(overlap)} node(s) appear in both "
        f"source and target columns (sample: {sample}). Graph is not "
        f"strictly bipartite — downstream projection results on these "
        f"nodes may be ambiguous."
    )
    return df, True


def _validate_bipartite_structure(
    df: pl.DataFrame,
    source_col: str,
    target_col: str,
    id_mapper: IDMapper
) -> None:
    """
    Validate that the graph has bipartite structure (disjoint source/target sets).
    
    Parameters
    ----------
    df : pl.DataFrame
        Edge list DataFrame
    source_col : str
        Source column name
    target_col : str
        Target column name
    id_mapper : IDMapper
        ID mapper containing all nodes
        
    Raises
    ------
    GraphConstructionError
        If graph is not bipartite
    """
    logger.debug("Validating bipartite graph structure")
    
    source_ids = set(df[source_col].unique().to_list())
    target_ids = set(df[target_col].unique().to_list())
    
    # Check for overlap between source and target sets
    overlap = source_ids.intersection(target_ids)
    
    if overlap:
        raise GraphConstructionError(
            f"Graph is not bipartite: {len(overlap)} nodes appear in both source and target",
            graph_type="bipartite",
            details={
                "overlapping_nodes": list(overlap)[:10],  # Show first 10
                "total_overlap": len(overlap),
                "source_partition_size": len(source_ids),
                "target_partition_size": len(target_ids)
            }
        )
    
    logger.debug("Bipartite structure validated: %d source nodes, %d target nodes", 
                len(source_ids), len(target_ids))


def _construct_graph(
    df: pl.DataFrame,
    id_mapper: IDMapper,
    source_col: str,
    target_col: str,
    weight_col: Optional[str],
    directed: bool,
    bipartite: bool,
    chunk_size: Optional[int] = None,
) -> nk.Graph:
    """
    Construct NetworkIt graph from processed edge list.

    The implementation maps original IDs to internal IDs using a single
    polars expression (C-level lookup against the IDMapper's dict), then
    iterates a tight ``zip`` over numpy int64 arrays to call ``addEdge``.
    The previous ``for row in df.iter_rows(named=True)`` path created a
    Python dict per row and did two dict lookups *through* the IDMapper's
    Python wrapper per edge — roughly an order of magnitude more overhead
    per edge.

    Parameters
    ----------
    df : pl.DataFrame
        Processed edge list with weights
    id_mapper : IDMapper
        ID mapping between original and internal IDs
    source_col : str
        Source column name
    target_col : str
        Target column name
    weight_col : str, optional
        Weight column name
    directed : bool
        Whether graph is directed
    bipartite : bool
        Whether graph is bipartite (informational; NetworkIt has no
        bipartite-aware storage so this is recorded on the IDMapper, not
        the graph)
    chunk_size : Optional[int], default None
        If set, process edges in batches of this size. Slightly slower
        overall but keeps peak memory close to the input DataFrame size by
        not materializing all source/target/weight columns simultaneously.
        Leave as ``None`` for fastest construction when the graph fits in
        RAM comfortably.

    Returns
    -------
    nk.Graph
        Constructed NetworkIt graph

    Raises
    ------
    GraphConstructionError
        If graph construction fails
    """
    try:
        n_nodes = id_mapper.size()
        weighted = weight_col is not None
        n_edges = len(df)

        logger.debug(
            "Creating NetworkIt graph: %d nodes, %d edges, weighted=%s, directed=%s, chunk_size=%s",
            n_nodes, n_edges, weighted, directed, chunk_size,
        )

        graph = nk.Graph(n_nodes, weighted=weighted, directed=directed)

        if n_edges == 0:
            return graph

        # Vectorized original-id → internal-id mapping. polars'
        # `replace_strict` does the lookup in Rust against the dict we pass
        # in — equivalent to a Python-level `[mapping[x] for x in col]` but
        # without the per-element Python overhead.
        mapping = id_mapper.original_to_internal

        # Use chunk_size if specified to keep peak memory bounded.
        if chunk_size is None or chunk_size >= n_edges:
            _add_edges_chunk(graph, df, source_col, target_col, weight_col, mapping, weighted)
        else:
            for start in range(0, n_edges, chunk_size):
                end = min(start + chunk_size, n_edges)
                chunk = df.slice(start, end - start)
                _add_edges_chunk(graph, chunk, source_col, target_col, weight_col, mapping, weighted)
                # Let polars/numpy intermediates GC between chunks.
                del chunk

        logger.debug("Graph construction completed: %d edges added", graph.numberOfEdges())
        return graph

    except Exception as e:
        raise GraphConstructionError(
            f"Failed to construct NetworkIt graph: {str(e)}",
            operation="construct_graph",
            node_count=id_mapper.size() if id_mapper else None,
            edge_count=len(df) if df is not None else None,
            cause=e
        )


def _add_edges_chunk(
    graph: nk.Graph,
    chunk: pl.DataFrame,
    source_col: str,
    target_col: str,
    weight_col: Optional[str],
    mapping: Dict[Any, int],
    weighted: bool,
) -> None:
    """Translate a chunk of edges to internal IDs and emit ``addEdge`` calls.

    The id translation happens in polars (Rust) via ``replace_strict``;
    everything below the zip is a tight Python loop wrapping the C
    addEdge call.
    """
    src_arr = (
        chunk[source_col]
        .replace_strict(
            old=list(mapping.keys()),
            new=list(mapping.values()),
            return_dtype=pl.Int64,
        )
        .to_numpy()
    )
    tgt_arr = (
        chunk[target_col]
        .replace_strict(
            old=list(mapping.keys()),
            new=list(mapping.values()),
            return_dtype=pl.Int64,
        )
        .to_numpy()
    )

    add_edge = graph.addEdge  # local binding avoids attribute lookup per call

    if weighted:
        w_arr = chunk[weight_col].cast(pl.Float64).to_numpy()
        for u, v, w in zip(src_arr, tgt_arr, w_arr):
            add_edge(int(u), int(v), float(w))
    else:
        for u, v in zip(src_arr, tgt_arr):
            add_edge(int(u), int(v))


def get_graph_info(graph: nk.Graph, id_mapper: IDMapper) -> Dict[str, Any]:
    """
    Get comprehensive information about a constructed graph.
    
    Parameters
    ----------
    graph : nk.Graph
        NetworkIt graph
    id_mapper : IDMapper
        ID mapper for the graph
        
    Returns
    -------
    Dict[str, Any]
        Dictionary containing graph statistics and properties
        
    Examples
    --------
    >>> graph, mapper = build_graph_from_edgelist(edges)
    >>> info = get_graph_info(graph, mapper)
    >>> print(f"Nodes: {info['num_nodes']}, Edges: {info['num_edges']}")
    """
    return {
        "num_nodes": graph.numberOfNodes(),
        "num_edges": graph.numberOfEdges(),
        "directed": graph.isDirected(),
        "weighted": graph.isWeighted(),
        "has_self_loops": graph.numberOfSelfLoops() > 0,
        "num_self_loops": graph.numberOfSelfLoops(),
        "density": nk.graphtools.density(graph) if graph.numberOfNodes() > 1 else 0.0,
        "node_id_mapping_size": id_mapper.size(),
        "is_connected": nk.components.ConnectedComponents(graph).run().numberOfComponents() == 1
    }


def validate_graph_construction(
    graph: nk.Graph,
    id_mapper: IDMapper,
    expected_nodes: Optional[int] = None,
    expected_edges: Optional[int] = None
) -> None:
    """
    Validate that graph construction was successful.
    
    Parameters
    ----------
    graph : nk.Graph
        Constructed graph to validate
    id_mapper : IDMapper
        ID mapper to validate
    expected_nodes : int, optional
        Expected number of nodes
    expected_edges : int, optional
        Expected number of edges
        
    Raises
    ------
    GraphConstructionError
        If validation fails
    """
    # Basic consistency checks
    if graph.numberOfNodes() != id_mapper.size():
        raise GraphConstructionError(
            "Inconsistent node count between graph and ID mapper",
            details={
                "graph_nodes": graph.numberOfNodes(),
                "mapper_size": id_mapper.size()
            }
        )
    
    # Check expected counts if provided
    if expected_nodes is not None and graph.numberOfNodes() != expected_nodes:
        raise GraphConstructionError(
            f"Unexpected node count: expected {expected_nodes}, got {graph.numberOfNodes()}"
        )
    
    if expected_edges is not None and graph.numberOfEdges() != expected_edges:
        raise GraphConstructionError(
            f"Unexpected edge count: expected {expected_edges}, got {graph.numberOfEdges()}"
        )
    
    logger.debug("Graph construction validation passed")


def project_bipartite(
    graph: nk.Graph,
    id_mapper: IDMapper,
    projection_mode: str = "source",
    weight_method: str = "count",
    verbose: bool = True,
) -> Tuple[nk.Graph, IDMapper]:
    """
    Project bipartite graph to unipartite by connecting nodes with shared neighbors.
    
    This function projects a bipartite graph onto one of its partitions, creating
    a unipartite graph where nodes are connected if they share neighbors in the
    other partition. Edge weights are calculated using various similarity measures.
    
    Parameters
    ----------
    graph : nk.Graph
        Bipartite NetworkIt graph to project
    id_mapper : IDMapper
        Original ID mapper containing all bipartite graph nodes
    projection_mode : str, default "source"
        Which partition to project onto:
        - "source": Project onto the source partition (nodes that appear as sources)
        - "target": Project onto the target partition (nodes that appear as targets)
    weight_method : str, default "count"
        Method for calculating projection weights:
        - "count": Number of shared neighbors
        - "jaccard": Jaccard similarity of neighbor sets
        - "overlap": Overlap coefficient (min of neighbor set sizes)
        
    Returns
    -------
    projected_graph : nk.Graph
        Unipartite NetworkIt graph containing only projected nodes
    new_id_mapper : IDMapper
        Updated ID mapper containing only nodes in the projected graph
        
    Raises
    ------
    GraphConstructionError
        If the input graph is not bipartite or projection fails
    ConfigurationError
        If invalid projection_mode or weight_method is specified
        
    Examples
    --------
    >>> # Create bipartite graph (users -> items)
    >>> edges = pl.DataFrame({
    ...     "user": ["u1", "u1", "u2", "u2", "u3"],
    ...     "item": ["i1", "i2", "i1", "i3", "i2"]
    ... })
    >>> graph, mapper = build_graph_from_edgelist(
    ...     edges, source_col="user", target_col="item", bipartite=True
    ... )
    >>> 
    >>> # Project onto users (connect users who like same items)
    >>> user_graph, user_mapper = project_bipartite(graph, mapper, "source", "count")
    >>> 
    >>> # Project onto items (connect items liked by same users)  
    >>> item_graph, item_mapper = project_bipartite(graph, mapper, "target", "jaccard")
    
    Notes
    -----
    Time Complexity: O(N² × D) worst case, where N is the size of the projection
    partition and D is the average degree in the other partition.
    
    Space Complexity: O(N²) in the worst case for a fully connected projection.
    
    The function identifies bipartite partitions by analyzing edge patterns rather
    than requiring explicit partition information. This makes it robust to various
    bipartite graph construction approaches.
    
    Weight Methods:
    - **Count**: Simple count of shared neighbors. Fast and intuitive.
    - **Jaccard**: |A ∩ B| / |A ∪ B|. Normalized similarity measure.
    - **Overlap**: |A ∩ B| / min(|A|, |B|). Asymmetric similarity measure.
    """
    log_function_entry("project_bipartite",
                      projection_mode=projection_mode, weight_method=weight_method)

    # Validate parameters
    validate_parameter(projection_mode, ["source", "target"], "projection_mode", "project_bipartite")
    validate_parameter(weight_method, ["count", "jaccard", "overlap"], "weight_method", "project_bipartite")

    import time as _time
    _t_start = _time.perf_counter()

    with LoggingTimer("project_bipartite"):
        try:
            # Step 1: Identify bipartite partitions
            source_partition, target_partition = _identify_bipartite_partitions(graph, id_mapper)
            
            # Step 2: Select projection partition and other partition
            if projection_mode == "source":
                projection_partition = source_partition
                other_partition = target_partition
            else:  # projection_mode == "target"
                projection_partition = target_partition
                other_partition = source_partition
            
            logger.info("Projecting bipartite graph: %d nodes in projection partition, %d in other partition",
                       len(projection_partition), len(other_partition))
            
            # Step 3: Build neighbor mapping for projection partition nodes
            neighbor_map = _build_neighbor_mapping(graph, id_mapper, projection_partition, other_partition)
            
            # Step 4: Create projected graph and new ID mapper
            projected_graph, new_id_mapper = _create_projected_graph(
                projection_partition, neighbor_map, weight_method, graph.isWeighted()
            )
            
            logger.info("Bipartite projection completed: %d nodes, %d edges",
                       projected_graph.numberOfNodes(), projected_graph.numberOfEdges())

            _print_projection_summary(
                verbose, _t_start, graph, projected_graph,
                projection_mode, weight_method,
            )
            return projected_graph, new_id_mapper
            
        except Exception as e:
            if isinstance(e, (GraphConstructionError, ValidationError)):
                raise
            else:
                raise GraphConstructionError(
                    f"Bipartite projection failed: {str(e)}",
                    operation="project_bipartite",
                    cause=e
                )


def _identify_bipartite_partitions(graph: nk.Graph, id_mapper: IDMapper) -> Tuple[List[Any], List[Any]]:
    """
    Identify the two partitions of a bipartite graph.
    
    Parameters
    ----------
    graph : nk.Graph
        NetworkIt graph (should be bipartite)
    id_mapper : IDMapper
        ID mapper for the graph
        
    Returns
    -------
    source_partition : List[Any]
        List of original IDs in the source partition
    target_partition : List[Any]
        List of original IDs in the target partition
        
    Raises
    ------
    GraphConstructionError
        If graph is not bipartite
    """
    logger.debug("Identifying bipartite partitions")

    if graph.numberOfNodes() == 0:
        return [], []

    # Fast path: if build_graph_from_edgelist recorded the original source/
    # target column membership on the mapper, use that directly. This keeps
    # the projection aligned with the user's mental model of which side is
    # which. Falls back to BFS coloring only when that info isn't available
    # (e.g. graph wasn't built with bipartite=True).
    if id_mapper.has_bipartite_partitions():
        logger.debug(
            "Using bipartite partitions recorded on IDMapper "
            "(%d source nodes, %d target nodes)",
            len(id_mapper.source_partition_originals),
            len(id_mapper.target_partition_originals),
        )
        return (
            list(id_mapper.source_partition_originals),
            list(id_mapper.target_partition_originals),
        )

    # Use NetworkIt's bipartiteness check if available, otherwise implement our own
    try:
        # Try to detect bipartiteness by attempting 2-coloring.
        # NOTE: the "source"/"target" labels below are arbitrary — they reflect
        # BFS color 0/1, not the original source/target columns of the
        # edgelist. Callers wanting consistent semantics should construct
        # the graph with build_graph_from_edgelist(bipartite=True).
        source_partition = []
        target_partition = []
        
        # Color nodes using BFS to detect bipartiteness
        colors = {}  # node -> color (0 or 1)
        queue = []
        
        # Start with first node
        first_node = 0
        colors[first_node] = 0
        queue.append(first_node)
        source_partition.append(id_mapper.get_original(first_node))
        
        while queue:
            current = queue.pop(0)
            current_color = colors[current]
            
            # Check all neighbors
            for neighbor in graph.iterNeighbors(current):
                if neighbor in colors:
                    # Check if coloring is consistent
                    if colors[neighbor] == current_color:
                        raise GraphConstructionError(
                            "Graph is not bipartite: found odd cycle",
                            operation="identify_bipartite_partitions",
                            details={"conflicting_nodes": [current, neighbor]}
                        )
                else:
                    # Color neighbor with opposite color
                    neighbor_color = 1 - current_color
                    colors[neighbor] = neighbor_color
                    queue.append(neighbor)
                    
                    if neighbor_color == 0:
                        source_partition.append(id_mapper.get_original(neighbor))
                    else:
                        target_partition.append(id_mapper.get_original(neighbor))
        
        # Handle disconnected components
        for node in range(graph.numberOfNodes()):
            if node not in colors:
                # Start new component
                colors[node] = 0
                queue.append(node)
                source_partition.append(id_mapper.get_original(node))
                
                while queue:
                    current = queue.pop(0)
                    current_color = colors[current]
                    
                    for neighbor in graph.iterNeighbors(current):
                        if neighbor in colors:
                            if colors[neighbor] == current_color:
                                raise GraphConstructionError(
                                    "Graph is not bipartite: found odd cycle in disconnected component",
                                    operation="identify_bipartite_partitions"
                                )
                        else:
                            neighbor_color = 1 - current_color
                            colors[neighbor] = neighbor_color
                            queue.append(neighbor)
                            
                            if neighbor_color == 0:
                                source_partition.append(id_mapper.get_original(neighbor))
                            else:
                                target_partition.append(id_mapper.get_original(neighbor))
        
        logger.debug("Identified partitions: %d source nodes, %d target nodes",
                    len(source_partition), len(target_partition))
        
        return source_partition, target_partition
        
    except Exception as e:
        if isinstance(e, GraphConstructionError):
            raise
        else:
            raise GraphConstructionError(
                f"Failed to identify bipartite partitions: {str(e)}",
                operation="identify_bipartite_partitions",
                cause=e
            )


def _build_neighbor_mapping(
    graph: nk.Graph,
    id_mapper: IDMapper,
    projection_partition: List[Any],
    other_partition: List[Any]
) -> Dict[Any, Set[Any]]:
    """
    Build mapping from projection partition nodes to their neighbors in other partition.
    
    Parameters
    ----------
    graph : nk.Graph
        NetworkIt graph
    id_mapper : IDMapper
        ID mapper
    projection_partition : List[Any]
        Nodes to project (original IDs)
    other_partition : List[Any]
        Other partition nodes (original IDs)
        
    Returns
    -------
    Dict[Any, Set[Any]]
        Mapping from projection node to set of neighbor nodes in other partition
    """
    logger.debug("Building neighbor mapping for %d projection nodes", len(projection_partition))
    
    neighbor_map = {}
    other_partition_set = set(other_partition)
    
    for proj_node_orig in projection_partition:
        proj_node_internal = id_mapper.get_internal(proj_node_orig)
        neighbors = set()
        
        # Find neighbors in the other partition
        for neighbor_internal in graph.iterNeighbors(proj_node_internal):
            neighbor_orig = id_mapper.get_original(neighbor_internal)
            if neighbor_orig in other_partition_set:
                neighbors.add(neighbor_orig)
        
        neighbor_map[proj_node_orig] = neighbors
    
    return neighbor_map


def _create_projected_graph(
    projection_partition: List[Any],
    neighbor_map: Dict[Any, Set[Any]],
    weight_method: str,
    preserve_weights: bool
) -> Tuple[nk.Graph, IDMapper]:
    """
    Create the projected unipartite graph using a sparse-matrix co-occurrence.

    Equivalent to the naive ``for i,j: |N(i) ∩ N(j)|`` double loop but uses
    ``scipy.sparse`` so the heavy lifting runs in C/BLAS:

      A   = bipartite incidence matrix (projection × intermediate, 0/1)
      AAT = A @ A.T  →  AAT[i,j] = |N(i) ∩ N(j)|       (the "count" weight)
      deg = A.sum(axis=1)                              (per-node bipartite degree)
      jaccard = count / (deg_i + deg_j − count)
      overlap = count / min(deg_i, deg_j)

    Output is bit-identical to the legacy set-intersection loop for ``count``
    (integer arithmetic both sides) and matches to within float-precision
    tolerance for ``jaccard`` and ``overlap``.

    Parameters
    ----------
    projection_partition : List[Any]
        Nodes in projection partition (original IDs)
    neighbor_map : Dict[Any, Set[Any]]
        Mapping from projection nodes to their bipartite-neighbor sets
    weight_method : str
        One of "count", "jaccard", "overlap"
    preserve_weights : bool
        Unused under this implementation; the projection is always weighted.
        Kept for API compatibility with the legacy signature.

    Returns
    -------
    projected_graph : nk.Graph
        Projected unipartite graph (undirected, weighted).
    new_id_mapper : IDMapper
        ID mapper for projected graph. Internal IDs assigned in sorted-by-str
        order of the original IDs — matches the legacy implementation exactly.
    """
    logger.debug(
        "Creating projected graph with %d nodes using %s weights (sparse path)",
        len(projection_partition), weight_method,
    )

    # Step 1: Build the new ID mapper with the same deterministic ordering
    # the legacy code used (sorted by str). This keeps the returned graph's
    # internal IDs bit-identical between implementations.
    sorted_projection = sorted(projection_partition, key=str)
    new_id_mapper = IDMapper()
    for internal_id, original_id in enumerate(sorted_projection):
        new_id_mapper.add_mapping(original_id, internal_id)

    n = len(sorted_projection)
    projected_graph = nk.Graph(n, weighted=True, directed=False)

    if n == 0:
        return projected_graph, new_id_mapper

    # Step 2: Build the bipartite incidence matrix A (n × n_intermediate).
    # Collect every intermediate-node original ID seen in neighbor_map,
    # then map them to dense column indices.
    intermediate_to_col: Dict[Any, int] = {}
    rows: List[int] = []
    cols: List[int] = []
    for i, original_id in enumerate(sorted_projection):
        neighbors = neighbor_map.get(original_id, ())
        for inter_orig in neighbors:
            col = intermediate_to_col.get(inter_orig)
            if col is None:
                col = len(intermediate_to_col)
                intermediate_to_col[inter_orig] = col
            rows.append(i)
            cols.append(col)

    n_intermediates = len(intermediate_to_col)

    # No edges in the bipartite layer ⇒ no projection edges either.
    if not rows or n_intermediates == 0:
        return projected_graph, new_id_mapper

    A = sp.csr_matrix(
        (np.ones(len(rows), dtype=np.int64), (np.asarray(rows), np.asarray(cols))),
        shape=(n, n_intermediates),
    )

    # Step 3: Shared-neighbor counts via A @ A.T. Only keep the strict upper
    # triangle — the graph is undirected and we skip self-loops (diagonal).
    AAT = (A @ A.T).tocsr()
    AAT_upper = sp.triu(AAT, k=1).tocoo()

    if AAT_upper.nnz == 0:
        return projected_graph, new_id_mapper

    i_arr = AAT_upper.row.astype(np.int64, copy=False)
    j_arr = AAT_upper.col.astype(np.int64, copy=False)
    shared = AAT_upper.data.astype(np.int64, copy=False)

    # Step 4: Compute the requested weight, then add edges with weight > 0.
    if weight_method == "count":
        # Integer counts cast to float so they round-trip through NetworkIt
        # the same way the legacy ``float(len(...))`` did.
        weights = shared.astype(np.float64)
    elif weight_method == "jaccard":
        degrees = np.asarray(A.sum(axis=1)).flatten().astype(np.int64)
        union = degrees[i_arr] + degrees[j_arr] - shared
        # union == 0 only when both nodes have zero neighbors AND zero shared,
        # i.e. the entry wouldn't be in AAT_upper anyway; guard for safety.
        with np.errstate(divide="ignore", invalid="ignore"):
            weights = np.where(union > 0, shared / union, 0.0)
    elif weight_method == "overlap":
        degrees = np.asarray(A.sum(axis=1)).flatten().astype(np.int64)
        min_deg = np.minimum(degrees[i_arr], degrees[j_arr])
        with np.errstate(divide="ignore", invalid="ignore"):
            weights = np.where(min_deg > 0, shared / min_deg, 0.0)
    else:
        raise ValueError(f"Unknown weight method: {weight_method}")

    # Step 5: Emit edges. Match the legacy behaviour of skipping weight==0
    # entries (which can occur for jaccard/overlap on degenerate inputs).
    nonzero = weights > 0
    for u, v, w in zip(i_arr[nonzero], j_arr[nonzero], weights[nonzero]):
        projected_graph.addEdge(int(u), int(v), float(w))

    return projected_graph, new_id_mapper


def _calculate_projection_weight(
    neighbors1: Set[Any],
    neighbors2: Set[Any],
    shared_neighbors: Set[Any],
    weight_method: str
) -> float:
    """
    Calculate projection edge weight using specified method.
    
    Parameters
    ----------
    neighbors1 : Set[Any]
        Neighbor set of first node
    neighbors2 : Set[Any]
        Neighbor set of second node
    shared_neighbors : Set[Any]
        Intersection of neighbor sets
    weight_method : str
        Weight calculation method
        
    Returns
    -------
    float
        Calculated weight
    """
    if weight_method == "count":
        return float(len(shared_neighbors))
    
    elif weight_method == "jaccard":
        union_size = len(neighbors1.union(neighbors2))
        if union_size == 0:
            return 0.0
        return len(shared_neighbors) / union_size
    
    elif weight_method == "overlap":
        min_size = min(len(neighbors1), len(neighbors2))
        if min_size == 0:
            return 0.0
        return len(shared_neighbors) / min_size
    
    else:
        # This shouldn't happen due to parameter validation, but be safe
        raise ValueError(f"Unknown weight method: {weight_method}")


def get_bipartite_info(graph: nk.Graph, id_mapper: IDMapper) -> Dict[str, Any]:
    """
    Get information about a bipartite graph structure.
    
    Parameters
    ----------
    graph : nk.Graph
        Graph to analyze (should be bipartite)
    id_mapper : IDMapper
        ID mapper for the graph
        
    Returns
    -------
    Dict[str, Any]
        Dictionary containing bipartite graph information
    """
    try:
        source_partition, target_partition = _identify_bipartite_partitions(graph, id_mapper)
        
        return {
            "is_bipartite": True,
            "source_partition_size": len(source_partition),
            "target_partition_size": len(target_partition),
            "source_nodes": source_partition,
            "target_nodes": target_partition,
            "total_nodes": len(source_partition) + len(target_partition),
            "total_edges": graph.numberOfEdges()
        }
    except GraphConstructionError:
        return {
            "is_bipartite": False,
            "error": "Graph is not bipartite"
        }


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
) -> Tuple[nk.Graph, IDMapper]:
    """
    Convert temporal bipartite edgelist to unipartite graph using citation-convention edges.

    Two nodes that share the same intermediate node are connected by a directed
    edge that points **from the later sharer to the earlier one** (citation /
    attribution convention). Under this convention the earliest sharer of an
    item accumulates incoming edges from everyone who shared it later, which is
    how PageRank, HITS-Authority, and similar centrality metrics surface
    influential sources.

    .. note::
        **Input order is trusted, not validated.** This function does NOT sort
        the edgelist. The caller is responsible for sorting it so that, within
        each ``intermediate_col`` group, rows are ordered by ``timestamp_col``
        **DESCENDING** (latest first). If your data is not in this order the
        edge directions will be wrong without any error being raised.

        Pre-sort with::

            edgelist = edgelist.sort([intermediate_col, timestamp_col],
                                     descending=[False, True])

    The algorithm:

    1. Groups edges by intermediate node (the disappearing column).
    2. Within each group, takes nodes in the order they appear (caller's
       descending-timestamp order).
    3. For every pair where ``i < j`` (later, earlier), emits a directed edge
       ``unique_nodes[i] → unique_nodes[j]``.

    Parameters
    ----------
    edgelist : Union[str, pl.DataFrame]
        Edge list data containing temporal bipartite relationships.
        Can be file path (CSV/Parquet) or Polars DataFrame.
    source_col : str, default "source"
        Name of source node column in edgelist
    target_col : str, default "target"
        Name of target node column in edgelist
    timestamp_col : str, default "timestamp"
        Name of timestamp column for temporal ordering
    weight_col : Optional[str], default None
        Name of weight column. If None, edges get unit weight.
    intermediate_col : str, default "target"
        Column representing intermediate nodes (will disappear in projection).
        These nodes group the temporal relationships.
    projected_col : str, default "source"
        Column representing nodes to preserve in unipartite projection.
        These nodes will be connected based on shared intermediates.
    remove_self_loops : bool, default True
        Whether to remove self-loops in the resulting unipartite graph
    add_edge_weights : bool, default True
        Whether to calculate edge weights based on temporal relationships
        
    Returns
    -------
    Tuple[nk.Graph, IDMapper]
        NetworkIt directed graph and ID mapper for the projected unipartite network
        
    Raises
    ------
    DataFormatError
        If required columns are missing or data format is invalid
    ValidationError
        If timestamp column contains invalid data or insufficient temporal variation
    GraphConstructionError
        If projection fails due to data structure issues
        
    Examples
    --------
    Convert user-item temporal interactions to a user-user attribution network:

    >>> import polars as pl
    >>> from guidedLP.network.construction import temporal_bipartite_to_unipartite
    >>>
    >>> # Raw temporal data — users interacting with items over time.
    >>> data = pl.DataFrame({
    ...     "user": ["Alice", "Bob", "Charlie", "Alice", "Bob"],
    ...     "item": ["item1", "item1", "item1", "item2", "item2"],
    ...     "timestamp": pl.Series(
    ...         ["2024-01-01", "2024-01-02", "2024-01-03",
    ...          "2024-01-04", "2024-01-05"]
    ...     ).str.to_datetime(),
    ... })
    >>>
    >>> # REQUIRED: pre-sort by intermediate_col, then timestamp DESCENDING.
    >>> data = data.sort(["item", "timestamp"], descending=[False, True])
    >>>
    >>> graph, mapper = temporal_bipartite_to_unipartite(
    ...     data,
    ...     source_col="user", target_col="item", timestamp_col="timestamp",
    ...     intermediate_col="item",  # Items disappear
    ...     projected_col="user",     # Users remain, get connected
    ... )
    >>> # Resulting edges (citation direction, later → earlier):
    >>> #   Charlie → Bob,  Charlie → Alice,  Bob → Alice          (from item1)
    >>> #   Bob     → Alice                                         (from item2)
    >>> # Alice (earliest sharer of both items) has the most incoming edges.

    Notes
    -----
    Use citation convention when:

    - You want to run PageRank / HITS / eigenvector centrality on the result
      and have "influential source" surface as high score.
    - You're modelling attribution: "who did this late sharer learn from?"
    - You want GLP's out-degree pass to answer "who did this node attribute
      to?" and the in-degree pass to answer "who attributed to this node?".

    Typical applications:

    - User-item interactions → user-user attribution networks
    - Author-paper records → author-author citation-like graphs
    - Social-media share logs → information-source attribution
    """
    
    with LoggingTimer("temporal_bipartite_to_unipartite"):
        log_function_entry("temporal_bipartite_to_unipartite", 
                          intermediate_col=intermediate_col, projected_col=projected_col)
        
        # Load and validate input data
        if isinstance(edgelist, str):
            edgelist = _load_edgelist_file(edgelist)
        
        # Validate required columns
        required_cols = [source_col, target_col, timestamp_col]
        missing_cols = [col for col in required_cols if col not in edgelist.columns]
        if missing_cols:
            raise DataFormatError(f"Missing required columns: {missing_cols}")
        
        if intermediate_col not in [source_col, target_col]:
            raise ConfigurationError(
                f"intermediate_col '{intermediate_col}' must be either '{source_col}' or '{target_col}'"
            )
        if projected_col not in [source_col, target_col]:
            raise ConfigurationError(
                f"projected_col '{projected_col}' must be either '{source_col}' or '{target_col}'"
            )
        if intermediate_col == projected_col:
            raise ConfigurationError("intermediate_col and projected_col cannot be the same")
            
        logger.info(f"Processing temporal bipartite conversion: {len(edgelist)} edges")
        logger.info(f"Intermediate column: {intermediate_col}, Projected column: {projected_col}")
        
        try:
            # Convert timestamp to datetime if needed (still needed for the
            # temporal weight calculation below — does not change row order).
            if edgelist[timestamp_col].dtype != pl.Datetime:
                try:
                    edgelist = edgelist.with_columns(
                        pl.col(timestamp_col).str.to_datetime().alias(timestamp_col)
                    )
                except Exception as e:
                    raise ValidationError(f"Cannot parse timestamp column '{timestamp_col}': {e}")

            # NOTE: this function trusts the caller's row order. The input
            # edgelist MUST be pre-sorted by intermediate_col (any order), then
            # by timestamp_col DESCENDING (latest first) within each
            # intermediate-node group. We do not re-sort because callers
            # working with very large datasets often pre-sort once and reuse.
            logger.info(
                f"Assuming pre-sorted input: by {intermediate_col} (grouped), "
                f"then {timestamp_col} DESCENDING within each group"
            )

            # Group by intermediate node and create temporal edges. preserve order
            # so the user's descending-timestamp ordering inside each group is kept.
            projection_edges = []
            edge_weights = []

            for intermediate_node, group in edgelist.group_by(intermediate_col, maintain_order=True):
                # Projected nodes in the order the user provided — descending by
                # timestamp within this group (latest sharer first, earliest last).
                projected_nodes = group[projected_col].to_list()
                timestamps = group[timestamp_col].to_list()
                weights = group[weight_col].to_list() if weight_col else [1.0] * len(projected_nodes)
                
                # Skip groups with only one node (no edges possible)
                if len(projected_nodes) <= 1:
                    continue
                
                # Remove duplicates while preserving order
                unique_nodes = []
                unique_timestamps = []
                unique_weights = []
                seen = set()
                for i, node in enumerate(projected_nodes):
                    if node not in seen:
                        unique_nodes.append(node)
                        unique_timestamps.append(timestamps[i])
                        unique_weights.append(weights[i])
                        seen.add(node)
                
                # Skip if only one unique node after deduplication
                if len(unique_nodes) <= 1:
                    continue
                    
                # Create temporal edges using upper triangular indices.
                # With input sorted DESCENDING by timestamp within this group:
                #   unique_nodes[0]   = latest sharer
                #   unique_nodes[n-1] = earliest sharer
                # np.triu_indices(n, k=1) gives pairs (i, j) with i < j, so:
                #   unique_nodes[i] = later sharer  → source
                #   unique_nodes[j] = earlier sharer → target
                # This yields citation-convention edges (later → earlier): the
                # original sharer accumulates incoming edges from everyone who
                # shared the same item later, which is what PageRank / HITS
                # Authority expect to read as "influential source".
                n_nodes = len(unique_nodes)
                upper_tri_indices = np.triu_indices(n_nodes, k=1)

                for i, j in zip(upper_tri_indices[0], upper_tri_indices[1]):
                    source_node = unique_nodes[i]  # Later sharer
                    target_node = unique_nodes[j]  # Earlier sharer (the cited source)
                    
                    # Calculate edge weight based on temporal relationship
                    if add_edge_weights:
                        # Weight inversely related to temporal distance
                        time_diff = (unique_timestamps[i] - unique_timestamps[j]).total_seconds()
                        temporal_weight = 1.0 / (1.0 + abs(time_diff) / 86400.0)  # Decay by days
                        edge_weight = (unique_weights[i] + unique_weights[j]) / 2.0 * temporal_weight
                    else:
                        edge_weight = 1.0
                    
                    projection_edges.append((source_node, target_node))
                    edge_weights.append(edge_weight)
                    
                logger.debug(f"Created {len(upper_tri_indices[0])} temporal edges for intermediate '{intermediate_node}'")
            
            if not projection_edges:
                logger.warning("No temporal edges created - all intermediate nodes had ≤1 unique projected node")
                # Create empty graph
                empty_graph = nk.Graph(0, weighted=True, directed=True)
                empty_mapper = IDMapper()
                return empty_graph, empty_mapper
            
            # Create unipartite edgelist DataFrame
            unipartite_df = pl.DataFrame({
                "source": [edge[0] for edge in projection_edges],
                "target": [edge[1] for edge in projection_edges], 
                "weight": edge_weights
            })
            
            # Remove self-loops if requested
            if remove_self_loops:
                original_count = len(unipartite_df)
                unipartite_df = unipartite_df.filter(pl.col("source") != pl.col("target"))
                removed_count = original_count - len(unipartite_df)
                if removed_count > 0:
                    logger.info(f"Removed {removed_count} self-loops")
            
            logger.info(f"Created temporal unipartite projection: {len(unipartite_df)} edges")
            
            # Build final directed graph
            graph, id_mapper = build_graph_from_edgelist(
                unipartite_df,
                source_col="source",
                target_col="target", 
                weight_col="weight",
                directed=True,
                allow_self_loops=not remove_self_loops
            )
            
            logger.info(f"Final temporal unipartite graph: {graph.numberOfNodes()} nodes, {graph.numberOfEdges()} edges")
            
            return graph, id_mapper
            
        except Exception as e:
            raise GraphConstructionError(
                f"Temporal bipartite-to-unipartite conversion failed: {str(e)}",
                operation="temporal_bipartite_to_unipartite",
                cause=e
            )