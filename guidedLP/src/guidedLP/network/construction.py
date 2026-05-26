"""
Network construction module for the Guided Label Propagation library.

This module provides functionality for constructing NetworkIt graphs from edge lists
while preserving original node IDs and supporting various graph types including
directed, undirected, weighted, unweighted, and bipartite graphs.
"""

from typing import Union, Tuple, Optional, List, Dict, Any, Set, Sequence
import warnings
from pathlib import Path

import polars as pl
import networkit as nk
import numpy as np
import scipy.sparse as sp

from guidedLP.common.id_mapper import IDMapper
from guidedLP.common.edgelist import EdgeList
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


def _print_edgelist_build_summary(
    verbose: bool, t_start: float, input_rows: int, edge_list: "EdgeList"
) -> None:
    """One-line summary printed at the bottom of build_edgelist_from_frame."""
    if not verbose:
        return
    import time as _time
    dt = _time.perf_counter() - t_start
    print(
        f"[build_edgelist_from_frame] {dt:.2f}s | "
        f"{input_rows:,} input rows → "
        f"{edge_list.number_of_nodes():,} nodes, "
        f"{edge_list.number_of_edges():,} edges "
        f"({edge_list.code_dtype})"
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


def _print_temporal_projection_summary(
    verbose: bool,
    t_start: float,
    input_rows: int,
    output_edges: int,
    output_format: str,
    intermediate_col: str,
    projected_col: str,
) -> None:
    """One-line summary printed at the bottom of temporal_bipartite_to_unipartite."""
    if not verbose:
        return
    import time as _time
    dt = _time.perf_counter() - t_start
    print(
        f"[temporal_bipartite_to_unipartite] {dt:.2f}s | "
        f"intermediate={intermediate_col}, projected={projected_col}, "
        f"output={output_format} | "
        f"{input_rows:,} input rows → {output_edges:,} projection edges"
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

    with LoggingTimer("build_graph_from_edgelist"):
        try:
            # Pre-load so we can include the input row count in the summary.
            # build_edgelist_from_frame accepts either str or DataFrame, so
            # passing the already-loaded frame avoids a double-load.
            df = _load_edge_list(edgelist)
            input_rows = len(df)

            # Compose through build_edgelist_from_frame to share all the
            # pre-encoding logic (null drop, validation, weight aggregation,
            # bipartite overlap handling, min-degree filter, ID mapping).
            # verbose=False suppresses the inner summary so only the
            # graph-level one prints.
            edge_list, id_mapper = build_edgelist_from_frame(
                df,
                source_col=source_col,
                target_col=target_col,
                weight_col=weight_col,
                directed=directed,
                bipartite=bipartite,
                auto_weight=auto_weight,
                allow_self_loops=allow_self_loops,
                remove_duplicates=remove_duplicates,
                bipartite_overlap=bipartite_overlap,
                min_source_degree=min_source_degree,
                min_target_degree=min_target_degree,
                verbose=False,
            )

            graph, _ = edgelist_to_graph(
                edge_list, id_mapper, chunk_size=chunk_size,
            )

            logger.info(
                "Graph construction completed: %d nodes, %d edges, directed=%s, weighted=%s",
                graph.numberOfNodes(), graph.numberOfEdges(), directed,
                graph.isWeighted(),
            )

            _print_build_summary(verbose, _t_start, input_rows, graph)
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


def build_edgelist_from_frame(
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
    min_source_degree: Optional[int] = None,
    min_target_degree: Optional[int] = None,
    code_dtype: Any = pl.UInt32,
    passthrough_cols: Optional[Sequence[str]] = None,
    verbose: bool = True,
) -> Tuple[EdgeList, IDMapper]:
    """
    Build a coded :class:`EdgeList` + paired :class:`IDMapper` from raw input.

    Peer of :func:`build_graph_from_edgelist`: it runs the same loading,
    validation, null-dropping, weight/duplicate processing, bipartite
    overlap handling, and min-degree filtering — but instead of constructing
    a NetworkIt graph at the end, it encodes the source/target columns into
    fixed-width integer codes (``code_dtype``) and wraps the resulting frame
    in an :class:`EdgeList`. Use this when you want a memory-efficient edge
    container without paying for graph construction up front, or when an
    operation (e.g. bipartite projection) is faster on coded edges than on
    a NetworkIt graph.

    Parameters
    ----------
    edgelist, source_col, target_col, weight_col, directed, bipartite, \
    auto_weight, allow_self_loops, remove_duplicates, bipartite_overlap, \
    min_source_degree, min_target_degree, verbose :
        See :func:`build_graph_from_edgelist` — semantics are identical.
    code_dtype : pl.DataType, default ``pl.UInt32``
        Integer width for the encoded ``src``/``tgt`` columns. ``pl.UInt32``
        supports up to ~4.29B unique nodes; pass ``pl.UInt64`` for larger
        graphs. The chosen dtype is recorded on the returned EdgeList.
    passthrough_cols : Sequence[str], optional
        Extra columns from the input edgelist to carry through onto the
        resulting EdgeList's ``df`` alongside ``src``/``tgt``/``weight``.
        Used by temporal pipelines (e.g.
        :func:`temporal_bipartite_to_unipartite`) that need to keep a
        ``timestamp`` column on the coded EdgeList. Passthrough is only
        compatible with non-aggregating modes — i.e. ``auto_weight=False``
        AND ``remove_duplicates=False`` — because per-row passthrough
        values cannot survive a ``group_by`` aggregation. Conflicting
        combinations raise :class:`ValidationError`. The columns are
        emitted with their original dtypes; ``src``/``tgt``/``weight``
        names are reserved and cannot appear in ``passthrough_cols``.

    Returns
    -------
    edge_list : EdgeList
        Coded edge container. ``edge_list.df`` has columns ``src``, ``tgt``
        (both ``code_dtype``) and optional ``weight`` (``Float64``).
    id_mapper : IDMapper
        Bidirectional mapping between original IDs and the codes used in
        ``edge_list.df``. Code value ``k`` corresponds to original ID
        ``id_mapper.get_original(k)``.

    Raises
    ------
    ValidationError
        If edge list data is invalid, columns are missing, or the node
        count exceeds the capacity of ``code_dtype``.
    GraphConstructionError
        If the underlying encoding step fails.
    DataFormatError
        If a file path is provided and the file cannot be read or parsed.

    Examples
    --------
    >>> import polars as pl
    >>> edges = pl.DataFrame({
    ...     "source": ["A", "B", "C", "A"],
    ...     "target": ["B", "C", "A", "B"],
    ... })
    >>> el, mapper = build_edgelist_from_frame(edges)
    >>> el.number_of_nodes()
    3
    >>> el.code_dtype
    UInt32

    Notes
    -----
    Memory: ``UInt32`` codes occupy 4 bytes/value vs. ~12–40 bytes for typical
    Utf8 node IDs, so a frame with two coded columns is roughly 6–20× smaller
    than the original Utf8 edgelist. This is the win that makes large bipartite
    projections tractable: the intermediate result of a hub-heavy projection
    (which can produce hundreds of millions of edges) stays small.

    Time Complexity: O(E) with one extra O(V) pass to build the IDMapper.
    """
    log_function_entry(
        "build_edgelist_from_frame",
        edgelist=type(edgelist).__name__,
        directed=directed, bipartite=bipartite, code_dtype=str(code_dtype),
    )

    # passthrough_cols validation runs up front (before loading anything) so
    # misuse is surfaced immediately. Aggregation modes drop per-row info, so
    # they're incompatible with passthrough — see kwarg docs.
    if passthrough_cols is not None:
        passthrough_cols = list(passthrough_cols)
        reserved = {"src", "tgt", "weight"}
        conflict = [c for c in passthrough_cols if c in reserved]
        if conflict:
            raise ValidationError(
                f"passthrough_cols cannot include reserved names "
                f"{sorted(reserved)}: got {conflict!r}"
            )
        if len(set(passthrough_cols)) != len(passthrough_cols):
            raise ValidationError(
                f"passthrough_cols contains duplicates: {passthrough_cols!r}"
            )
        # Auto-weight and remove_duplicates both run group_by/unique that
        # collapse multiple input rows into one — per-row passthrough values
        # cannot survive that. weight_col + remove_duplicates=False also does
        # a sum aggregation, blocked for the same reason.
        if auto_weight and weight_col is None:
            raise ValidationError(
                "passthrough_cols requires auto_weight=False (the auto-weight "
                "branch aggregates rows via group_by, which drops per-row "
                "passthrough values)."
            )
        if weight_col is not None and not remove_duplicates:
            raise ValidationError(
                "passthrough_cols requires either weight_col=None or "
                "remove_duplicates=True (the weight-sum branch aggregates "
                "rows via group_by, which drops per-row passthrough values)."
            )
        if remove_duplicates:
            raise ValidationError(
                "passthrough_cols is incompatible with remove_duplicates=True "
                "(deduplicating rows drops per-row passthrough values)."
            )

    import time as _time
    _t_start = _time.perf_counter()
    _input_rows: Optional[int] = None

    with LoggingTimer("build_edgelist_from_frame"):
        try:
            # Step 1: Load.
            df = _load_edge_list(edgelist)
            _input_rows = len(df)

            # Step 2: Empty-input warn-and-return.
            if df.is_empty():
                warnings.warn("Empty edge list provided. Creating empty EdgeList.")
                empty_el = _empty_edgelist(
                    directed=directed, bipartite=bipartite,
                    weighted=(weight_col is not None), code_dtype=code_dtype,
                )
                empty_mapper = IDMapper()
                _print_edgelist_build_summary(verbose, _t_start, _input_rows or 0, empty_el)
                return empty_el, empty_mapper

            # Step 2b: Required-column check before any per-column ops.
            required_cols = [source_col, target_col]
            if weight_col is not None:
                required_cols.append(weight_col)
            if passthrough_cols:
                required_cols.extend(passthrough_cols)
            missing_cols = [c for c in required_cols if c not in df.columns]
            if missing_cols:
                raise ValidationError(
                    f"Missing required columns: {missing_cols}. "
                    f"Available columns: {df.columns}"
                )

            # Step 2c: Drop rows with nulls in src/tgt (and weight if specified).
            # Passthrough columns are NOT null-checked — they're side data and
            # callers may legitimately have nulls there.
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
                    "creating empty EdgeList."
                )
                empty_el = _empty_edgelist(
                    directed=directed, bipartite=bipartite,
                    weighted=(weight_col is not None), code_dtype=code_dtype,
                )
                empty_mapper = IDMapper()
                _print_edgelist_build_summary(verbose, _t_start, _input_rows or 0, empty_el)
                return empty_el, empty_mapper

            # Step 3: Schema/data validation.
            _validate_edge_list(df, source_col, target_col, weight_col)

            # Step 4: Process edges (auto-weight, duplicates, self-loops).
            processed_df = _process_edges(
                df, source_col, target_col, weight_col,
                auto_weight, allow_self_loops, remove_duplicates,
            )
            if (
                weight_col is None
                and auto_weight
                and "weight" in processed_df.columns
                and processed_df["weight"].max() > 1
            ):
                weight_col = "weight"

            # Step 5: Bipartite overlap policy.
            bipartite_overlap_used = False
            if bipartite:
                processed_df, bipartite_overlap_used = _apply_bipartite_overlap_policy(
                    processed_df, source_col, target_col, bipartite_overlap,
                )

            # Step 5b: Min-degree filter.
            if min_source_degree is not None or min_target_degree is not None:
                processed_df = _apply_min_degree_filter(
                    processed_df, source_col, target_col,
                    min_source_degree, min_target_degree,
                )
                if processed_df.is_empty():
                    warnings.warn(
                        "All edges removed by min_source_degree / "
                        "min_target_degree filtering; creating empty EdgeList."
                    )
                    empty_el = _empty_edgelist(
                        directed=directed, bipartite=bipartite,
                        weighted=(weight_col is not None), code_dtype=code_dtype,
                    )
                    empty_mapper = IDMapper()
                    _print_edgelist_build_summary(verbose, _t_start, _input_rows or 0, empty_el)
                    return empty_el, empty_mapper

            # Step 6: ID mapping over the (possibly-filtered) edge list.
            id_mapper, source_unique, target_unique = _create_id_mapping(
                processed_df, source_col, target_col,
            )

            # Step 7: Record bipartite partitions on the mapper.
            # source_unique/target_unique are Polars Series — go through
            # .to_list() once so the resulting set() builds from a typed
            # list rather than iterating the Series in Python.
            if bipartite:
                source_set = set(source_unique.to_list())
                target_set = set(target_unique.to_list())
                if bipartite_overlap_used:
                    id_mapper.source_partition_originals = source_set
                    id_mapper.target_partition_originals = target_set
                else:
                    id_mapper.set_bipartite_partitions(source_set, target_set)

            # Step 8: Encode the edge frame to integer codes and wrap in EdgeList.
            coded_df = _encode_to_codes(
                processed_df, source_col, target_col,
                id_mapper, code_dtype, weight_col,
                passthrough_cols=passthrough_cols,
            )
            edge_list = EdgeList(
                df=coded_df,
                directed=directed,
                bipartite=bipartite,
                n_nodes=id_mapper.size(),
                code_dtype=code_dtype,
            )

            logger.info(
                "EdgeList construction completed: %d nodes, %d edges, "
                "directed=%s, weighted=%s, code_dtype=%s",
                edge_list.number_of_nodes(), edge_list.number_of_edges(),
                directed, weight_col is not None, code_dtype,
            )

            _print_edgelist_build_summary(verbose, _t_start, _input_rows or 0, edge_list)
            return edge_list, id_mapper

        except Exception as e:
            if isinstance(e, (ValidationError, GraphConstructionError, DataFormatError, ConfigurationError)):
                raise
            raise GraphConstructionError(
                f"Unexpected error during EdgeList construction: {str(e)}",
                operation="build_edgelist_from_frame",
                cause=e,
            )


def _empty_edgelist(
    *, directed: bool, bipartite: bool, weighted: bool, code_dtype: Any,
) -> EdgeList:
    """Build an EdgeList with zero edges and zero nodes."""
    cols = [
        pl.Series("src", [], dtype=code_dtype),
        pl.Series("tgt", [], dtype=code_dtype),
    ]
    if weighted:
        cols.append(pl.Series("weight", [], dtype=pl.Float64))
    return EdgeList(
        df=pl.DataFrame(cols),
        directed=directed,
        bipartite=bipartite,
        n_nodes=0,
        code_dtype=code_dtype,
    )


def _encode_to_codes(
    df: pl.DataFrame,
    source_col: str,
    target_col: str,
    id_mapper: IDMapper,
    code_dtype: Any,
    weight_col: Optional[str],
    passthrough_cols: Optional[Sequence[str]] = None,
) -> pl.DataFrame:
    """Translate ``source_col``/``target_col`` values from original IDs to
    integer codes using ``id_mapper``.

    Uses Polars ``replace_strict`` (Rust-level dict lookup) for vectorized
    mapping — the same primitive ``_add_edges_chunk`` already relies on for
    the NetworkIt build path.

    ``passthrough_cols`` (if provided) names additional columns that ride
    through onto the output frame with their original dtype, alongside the
    encoded ``src``/``tgt``/``weight``. Used by temporal pipelines that
    need a ``timestamp`` column on the coded EdgeList.
    """
    mapping = id_mapper.original_to_internal
    # Build the (old → new) mapping as Polars Series once. Passing Python
    # lists straight to replace_strict makes Polars rebuild them as Series
    # via the slow new_from_any_values path on every call (twice here —
    # once for src, once for tgt). Constructing the Series ourselves with
    # an explicit dtype avoids that round-trip entirely.
    old = pl.Series("_glp_old", list(mapping.keys()))
    new = pl.Series("_glp_new", list(mapping.values()), dtype=pl.Int64)

    select_exprs = [
        pl.col(source_col)
        .replace_strict(old=old, new=new, return_dtype=pl.Int64)
        .cast(code_dtype)
        .alias("src"),
        pl.col(target_col)
        .replace_strict(old=old, new=new, return_dtype=pl.Int64)
        .cast(code_dtype)
        .alias("tgt"),
    ]
    if weight_col is not None and weight_col in df.columns:
        select_exprs.append(pl.col(weight_col).cast(pl.Float64).alias("weight"))
    if passthrough_cols:
        for col in passthrough_cols:
            select_exprs.append(pl.col(col))

    return df.select(select_exprs)


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
    
    # Handle weights and duplicates. maintain_order=True keeps the input row
    # order so downstream EdgeList consumers (e.g. temporal_bipartite_to_unipartite,
    # which relies on a caller-provided pre-sort) don't have their ordering
    # silently shuffled by Polars' default hash-based group_by/unique.
    if weight_col is None and auto_weight:
        # Calculate weights from duplicate edge counts
        logger.debug("Calculating automatic weights from duplicate edges")
        processed_df = processed_df.group_by(
            [source_col, target_col], maintain_order=True,
        ).agg(
            pl.len().alias("weight")
        ).with_columns(pl.col("weight").cast(pl.Float64))
        weight_col = "weight"

    elif weight_col is not None and not remove_duplicates:
        # Sum weights for duplicate edges
        logger.debug("Aggregating weights for duplicate edges")
        processed_df = processed_df.group_by(
            [source_col, target_col], maintain_order=True,
        ).agg(
            pl.col(weight_col).sum().alias(weight_col)
        )

    elif remove_duplicates:
        # Remove duplicate edges (keep first occurrence)
        initial_count = len(processed_df)
        processed_df = processed_df.unique(
            subset=[source_col, target_col], keep="first", maintain_order=True,
        )
        removed_count = initial_count - len(processed_df)
        if removed_count > 0:
            logger.info("Removed %d duplicate edges", removed_count)
    
    return processed_df


def _create_id_mapping(
    df: pl.DataFrame,
    source_col: str,
    target_col: str
) -> Tuple[IDMapper, pl.Series, pl.Series]:
    """
    Create bidirectional ID mapping between original and NetworkIt IDs.

    Returns the mapper alongside the source/target unique-value Series so
    callers (notably bipartite-partition recording) can avoid running
    ``.unique()`` a second time on the same large DataFrame. The Series
    are kept in Polars-land — the caller materializes them into Python
    only when actually needed (the bipartite partitions need ``set(...)``,
    everything else does not).

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
        Configured ID mapper. Internal IDs assigned in natural sort order
        of original IDs (Polars Series.sort()). For string IDs this is
        lexicographic; for numeric IDs it is numeric. Deterministic across
        runs.
    source_unique : pl.Series
        Unique original IDs from the source column.
    target_unique : pl.Series
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
        source_unique = df[source_col].unique()
        target_unique = df[target_col].unique()

        # Union + sort in Polars/Rust. The previous implementation built two
        # Python sets and unioned them, then ran sorted(..., key=str) over
        # millions of Python objects — both expensive at scale.
        all_ids_series = (
            pl.concat([source_unique, target_unique])
            .unique()
            .sort()
        )

        # Single .to_list() — needed because IDMapper's backing store is a
        # Python dict and that's the only Python-level conversion we pay for.
        all_ids = all_ids_series.to_list()

        # Bulk-build the mapper. Skips the per-element type/uniqueness checks
        # that add_mapping() performs — they're redundant since the upstream
        # .unique() already deduped and only hashable dtypes made it past
        # validation.
        id_mapper = IDMapper.from_originals(all_ids)

        logger.debug("Created ID mapping for %d unique nodes", len(all_ids))
        return id_mapper, source_unique, target_unique

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

    # Compute overlap entirely in Polars: a Rust hash-join between the two
    # unique columns is far cheaper than materializing both into Python sets
    # and intersecting them (which costs ~3 conversions of millions of items
    # plus a Python-level set operation).
    src_unique = df.select(pl.col(source_col).unique().alias("_glp_id"))
    tgt_unique = df.select(pl.col(target_col).unique().alias("_glp_id"))
    overlap_df = src_unique.join(tgt_unique, on="_glp_id", how="inner")
    n_overlap = overlap_df.height

    if n_overlap == 0:
        return df, False

    # Only the small sample needs to leave Polars-land; full set never does.
    sample = overlap_df.head(10)["_glp_id"].to_list()

    if policy == "raise":
        raise GraphConstructionError(
            f"Graph is not bipartite: {n_overlap} nodes appear in both "
            f"source and target",
            graph_type="bipartite",
            details={
                "overlapping_nodes": sample,
                "total_overlap": n_overlap,
                "source_partition_size": src_unique.height,
                "target_partition_size": tgt_unique.height,
            },
        )

    if policy == "drop":
        rows_before = len(df)
        df = (
            df.join(overlap_df, left_on=source_col, right_on="_glp_id", how="anti")
              .join(overlap_df, left_on=target_col, right_on="_glp_id", how="anti")
        )
        n_dropped = rows_before - len(df)
        warnings.warn(
            f"bipartite_overlap='drop': removed {n_overlap} overlap node(s) "
            f"and {n_dropped} edge(s) (sample: {sample}). Graph is now "
            f"strictly bipartite."
        )
        return df, False

    # policy == "warn"
    warnings.warn(
        f"bipartite_overlap='warn': {n_overlap} node(s) appear in both "
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


def _extract_edge_arrays(
    graph: nk.Graph,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Walk ``graph.iterEdges()`` once, returning (sources, targets, weights).

    All three arrays are sized ``E`` and indexed by internal node IDs. Weights
    are 1.0 for unweighted graphs. Used as the single low-level edge-extraction
    primitive across the package — every other helper that needs to leave the
    graph object for numpy/Polars math should route through here.
    """
    weighted = graph.isWeighted()
    sources_l: List[int] = []
    targets_l: List[int] = []
    weights_l: List[float] = []
    for u, v in graph.iterEdges():
        sources_l.append(u)
        targets_l.append(v)
        weights_l.append(graph.weight(u, v) if weighted else 1.0)

    sources = np.asarray(sources_l, dtype=np.int64)
    targets = np.asarray(targets_l, dtype=np.int64)
    weights = np.asarray(weights_l, dtype=np.float64)
    return sources, targets, weights


def _build_induced_subgraph(
    graph: nk.Graph,
    id_mapper: IDMapper,
    keep_internal: Set[int],
) -> Tuple[nk.Graph, IDMapper]:
    """Build a new graph with contiguous internal IDs ``0..K-1``.

    Unlike approaches that use ``removeNode`` and leave holes in the internal
    ID space, this rebuilds the graph from scratch so downstream matrix
    operations can rely on ``range(numberOfNodes())``. Used by filtering,
    sampling, and influence-based reductions to materialize an induced
    subgraph plus a fresh mapper whose internals are dense.

    Parameters
    ----------
    graph : nk.Graph
        The source graph.
    id_mapper : IDMapper
        Original-ID mapper for ``graph``.
    keep_internal : Set[int]
        Internal node IDs (in ``graph``) to retain. Order of iteration is
        normalized internally via ``sorted()`` so the resulting internal IDs
        are deterministic for a given input set.

    Returns
    -------
    Tuple[nk.Graph, IDMapper]
        The induced subgraph and a fresh mapper. The new mapper preserves the
        original IDs of the kept nodes; only the internal IDs are renumbered.
    """
    sorted_kept = sorted(keep_internal)
    old_to_new = {old: new for new, old in enumerate(sorted_kept)}

    new_graph = nk.Graph(
        n=len(sorted_kept),
        weighted=graph.isWeighted(),
        directed=graph.isDirected(),
    )

    new_mapper = IDMapper()
    for old_id in sorted_kept:
        try:
            original = id_mapper.get_original(old_id)
        except KeyError:
            continue
        new_mapper.add_mapping(original, old_to_new[old_id])

    is_weighted = graph.isWeighted()
    keep_set = keep_internal if isinstance(keep_internal, set) else set(keep_internal)
    for u, v in graph.iterEdges():
        if u in keep_set and v in keep_set:
            new_u = old_to_new[u]
            new_v = old_to_new[v]
            if is_weighted:
                new_graph.addEdge(new_u, new_v, graph.weight(u, v))
            else:
                new_graph.addEdge(new_u, new_v)

    return new_graph, new_mapper


def graph_to_edges(graph: nk.Graph, id_mapper: IDMapper) -> pl.DataFrame:
    """Extract a NetworkIt graph's edges into a Polars frame keyed by original IDs.

    Parameters
    ----------
    graph : nk.Graph
        NetworkIt graph to extract from.
    id_mapper : IDMapper
        Bidirectional mapping between original and internal node IDs. Used to
        translate internal IDs back to the user-facing originals.

    Returns
    -------
    pl.DataFrame
        Columns: ``source_id``, ``target_id``, ``weight`` (Float64). ``weight``
        is always present and equals 1.0 for unweighted graphs. For undirected
        graphs each edge appears once, with the orientation NetworkIt returns
        from ``iterEdges()``.

    Notes
    -----
    Time complexity: O(E). The Python-level loop dominates; on large graphs
    this is the single biggest non-vectorised cost of the dataframe pipeline.
    """
    sources, targets, weights = _extract_edge_arrays(graph)
    if sources.size == 0:
        return pl.DataFrame(
            schema={
                "source_id": pl.Object,
                "target_id": pl.Object,
                "weight": pl.Float64,
            }
        )

    return pl.DataFrame(
        {
            "source_id": id_mapper.get_original_batch(sources.tolist()),
            "target_id": id_mapper.get_original_batch(targets.tolist()),
            "weight": weights,
        }
    )


def edgelist_to_graph(
    edge_list: EdgeList,
    id_mapper: IDMapper,
    chunk_size: Optional[int] = None,
) -> Tuple[nk.Graph, IDMapper]:
    """
    Materialize a NetworkIt :class:`nk.Graph` from a coded :class:`EdgeList`.

    No new ID mapping is created — the codes in ``edge_list.df`` *are* the
    NetworkIt internal IDs, so the passed-in mapper is returned unchanged.
    This is the inexpensive complement to :func:`build_graph_from_edgelist`:
    when you already have a coded EdgeList (e.g. from
    :func:`build_edgelist_from_frame` or :func:`graph_to_edgelist`), this
    skips the unique-pass and dict-build steps entirely.

    Parameters
    ----------
    edge_list : EdgeList
        Coded edge container. ``edge_list.df["src"]`` and ``["tgt"]`` must
        contain values in ``0..edge_list.n_nodes-1``.
    id_mapper : IDMapper
        The mapper paired with ``edge_list`` at construction time. Returned
        unchanged.
    chunk_size : Optional[int], default None
        If set, materialize the src/tgt numpy arrays in batches of this
        many edges instead of all at once. Slightly slower overall but
        keeps the peak numpy-array overhead bounded — useful when the
        EdgeList has tens of millions of edges.

    Returns
    -------
    Tuple[nk.Graph, IDMapper]
        The materialized graph (with ``edge_list.n_nodes`` nodes) and the
        unchanged mapper.

    Raises
    ------
    GraphConstructionError
        If NetworkIt rejects an edge (e.g. code out of range).

    Notes
    -----
    Time Complexity: O(E) — one ``addEdge`` call per row, no per-edge dict
    lookup. Roughly the same speed as the inner loop of
    :func:`build_graph_from_edgelist`, but without the upfront
    ``replace_strict`` mapping pass.
    """
    try:
        weighted = edge_list.is_weighted()
        n_edges = edge_list.number_of_edges()
        graph = nk.Graph(
            edge_list.n_nodes, weighted=weighted, directed=edge_list.directed,
        )

        if n_edges == 0:
            return graph, id_mapper

        if chunk_size is None or chunk_size >= n_edges:
            _add_coded_edges(graph, edge_list.df, weighted)
        else:
            for start in range(0, n_edges, chunk_size):
                end = min(start + chunk_size, n_edges)
                chunk = edge_list.df.slice(start, end - start)
                _add_coded_edges(graph, chunk, weighted)
                del chunk

        return graph, id_mapper

    except Exception as e:
        raise GraphConstructionError(
            f"Failed to materialize NetworkIt graph from EdgeList: {str(e)}",
            operation="edgelist_to_graph",
            node_count=edge_list.n_nodes,
            edge_count=edge_list.number_of_edges(),
            cause=e,
        )


def _add_coded_edges(graph: nk.Graph, coded_df: pl.DataFrame, weighted: bool) -> None:
    """Inner ``addEdge`` loop for a coded edge frame.

    Codes are already NetworkIt internal IDs, so no per-edge dict lookup
    is needed — just materialize the columns as numpy arrays and iterate.
    """
    src_arr = coded_df["src"].to_numpy()
    tgt_arr = coded_df["tgt"].to_numpy()
    add_edge = graph.addEdge

    if weighted:
        w_arr = coded_df["weight"].to_numpy()
        for u, v, w in zip(src_arr, tgt_arr, w_arr):
            add_edge(int(u), int(v), float(w))
    else:
        for u, v in zip(src_arr, tgt_arr):
            add_edge(int(u), int(v))


def graph_to_edgelist(
    graph: nk.Graph,
    id_mapper: IDMapper,
    code_dtype: Any = pl.UInt32,
) -> EdgeList:
    """
    Extract a NetworkIt :class:`nk.Graph`'s edges into a coded
    :class:`EdgeList`.

    Sibling of :func:`graph_to_edges`, which emits a frame in **original**
    IDs. This one emits internal-ID codes — useful when you want a
    memory-efficient edge container without paying the original-ID
    translation cost.

    Parameters
    ----------
    graph : nk.Graph
        Source graph.
    id_mapper : IDMapper
        The graph's paired mapper. Used to detect bipartite status (via
        ``has_bipartite_partitions``); not used for translation since codes
        are already internal IDs.
    code_dtype : pl.DataType, default ``pl.UInt32``
        Integer width for the encoded columns.

    Returns
    -------
    EdgeList
        Coded EdgeList with ``directed`` and ``bipartite`` flags taken from
        the graph and mapper respectively.

    Notes
    -----
    Time Complexity: O(E) — single walk of ``graph.iterEdges()``.
    """
    sources, targets, weights = _extract_edge_arrays(graph)

    cols = [
        pl.Series("src", sources, dtype=code_dtype),
        pl.Series("tgt", targets, dtype=code_dtype),
    ]
    if graph.isWeighted():
        cols.append(pl.Series("weight", weights, dtype=pl.Float64))

    return EdgeList(
        df=pl.DataFrame(cols),
        directed=graph.isDirected(),
        bipartite=id_mapper.has_bipartite_partitions(),
        n_nodes=graph.numberOfNodes(),
        code_dtype=code_dtype,
    )


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
    edges: Union[nk.Graph, pl.DataFrame, EdgeList],
    id_mapper: Optional[IDMapper] = None,
    projection_mode: str = "source",
    weight_method: str = "count",
    verbose: bool = True,
    *,
    output_format: Optional[str] = None,
) -> Union[Tuple[nk.Graph, IDMapper], Tuple[EdgeList, IDMapper], pl.DataFrame]:
    """
    Project a bipartite network to unipartite by connecting nodes with shared neighbors.

    Accepts a NetworkIt graph (with an ``id_mapper``), a Polars edge frame
    (with columns ``source_id``, ``target_id``), or a coded :class:`EdgeList`
    (also with an ``id_mapper``). The output format defaults to matching
    the input but can be forced via ``output_format``. ``frame → graph``
    is intentionally not supported — call :func:`build_graph_from_edgelist`
    on the returned frame if you need a graph.

    Parameters
    ----------
    edges : nk.Graph, pl.DataFrame, or EdgeList
        - ``nk.Graph``: bipartite NetworkIt graph (requires ``id_mapper``).
        - ``pl.DataFrame``: bipartite edge frame with columns ``source_id``,
          ``target_id`` (and optionally ``weight``, ignored — the projection
          always produces its own computed weights).
        - ``EdgeList``: coded bipartite EdgeList where ``src`` is the source
          partition and ``tgt`` is the target partition (requires
          ``id_mapper`` paired at construction).
    id_mapper : IDMapper, optional
        Required when ``edges`` is a ``Graph`` or ``EdgeList``; ignored when
        it's a frame (frames already use original IDs).
    projection_mode : str, default "source"
        Which partition to project onto:
        - "source": Project onto the source partition.
        - "target": Project onto the target partition.
    weight_method : str, default "count"
        Method for calculating projection weights:
        - "count": Number of shared neighbors.
        - "jaccard": |A ∩ B| / |A ∪ B|.
        - "overlap": |A ∩ B| / min(|A|, |B|).
    verbose : bool, default True
        Print a one-line summary at the end.
    output_format : str, optional
        ``"graph"``, ``"edgelist"``, ``"dataframe"``, or ``None`` (default —
        match input). ``output_format="graph"`` with a frame input is not
        supported (call :func:`build_graph_from_edgelist` on the returned
        frame instead).

    Returns
    -------
    Tuple[nk.Graph, IDMapper]
        Graph-output path; the projected unipartite graph and a new ID mapper.
    Tuple[EdgeList, IDMapper]
        EdgeList-output path; the projected edges as a coded EdgeList and a
        new ID mapper whose internal IDs match the EdgeList's codes.
    pl.DataFrame
        DataFrame-output path; the projected edges with columns
        ``source_id``, ``target_id``, ``weight``.

    Raises
    ------
    GraphConstructionError
        If the input graph is not bipartite or projection fails.
    ValidationError
        If invalid ``projection_mode``, ``weight_method``, or ``output_format``
        is specified, or if a frame input lacks required columns, or an
        EdgeList input lacks ``id_mapper``.

    Examples
    --------
    Graph input (legacy form):

    >>> edges = pl.DataFrame({
    ...     "user": ["u1", "u1", "u2", "u2", "u3"],
    ...     "item": ["i1", "i2", "i1", "i3", "i2"]
    ... })
    >>> graph, mapper = build_graph_from_edgelist(
    ...     edges, source_col="user", target_col="item", bipartite=True
    ... )
    >>> user_graph, user_mapper = project_bipartite(graph, mapper, "source", "count")

    Frame input — partitions implicit in the columns:

    >>> bipartite_edges = pl.DataFrame({
    ...     "source_id": ["u1", "u1", "u2", "u2", "u3"],
    ...     "target_id": ["i1", "i2", "i1", "i3", "i2"],
    ...     "weight": [1.0] * 5,
    ... })
    >>> projected_df = project_bipartite(bipartite_edges, projection_mode="source")

    EdgeList input — coded edges, lowest peak memory:

    >>> el, mapper = build_edgelist_from_frame(
    ...     bipartite_edges, source_col="source_id", target_col="target_id",
    ...     bipartite=True,
    ... )
    >>> projected_el, projected_mapper = project_bipartite(
    ...     el, mapper, projection_mode="source",
    ... )

    Notes
    -----
    Time Complexity: O(E × log E) for the Polars dedup + O(N_proj² × D)
    worst case for the SciPy sparse multiply (where N_proj is the
    projection-partition size and D is the average degree in the other
    partition).

    Space Complexity: O(E + N_proj²) — the bipartite incidence matrix plus
    the projected adjacency. The EdgeList/frame paths hold the bipartite
    side as UInt32 codes, so a hub-heavy projection that explodes the
    projection-edge count stays manageable.

    The graph-input path identifies bipartite partitions either from the
    IDMapper's recorded partitions (when the graph was built with
    ``bipartite=True``) or via BFS 2-coloring. The frame and EdgeList
    paths treat the ``source_id``/``src`` column as the source partition
    and ``target_id``/``tgt`` as the target partition.

    Weight Methods:
    - **Count**: Simple count of shared neighbors. Fast and intuitive.
    - **Jaccard**: |A ∩ B| / |A ∪ B|. Normalized similarity measure.
    - **Overlap**: |A ∩ B| / min(|A|, |B|). Asymmetric similarity measure.
    """
    # Validate parameters (do this before any dispatch so messages are consistent).
    validate_parameter(projection_mode, ["source", "target"], "projection_mode", "project_bipartite")
    validate_parameter(weight_method, ["count", "jaccard", "overlap"], "weight_method", "project_bipartite")
    if output_format not in (None, "graph", "edgelist", "dataframe"):
        raise ValidationError(
            f"output_format must be 'graph', 'edgelist', 'dataframe', or None; "
            f"got {output_format!r}"
        )

    import time as _time
    _t_start = _time.perf_counter()

    # EdgeList-input branch (vectorized coded path).
    if isinstance(edges, EdgeList):
        if id_mapper is None:
            raise ValidationError(
                "`id_mapper` is required when `edges` is an EdgeList"
            )
        return _project_bipartite_edgelist_path(
            edges, id_mapper, projection_mode, weight_method,
            output_format, verbose, _t_start,
        )

    # Frame-input branch.
    if isinstance(edges, pl.DataFrame):
        if output_format == "graph":
            raise ValidationError(
                "output_format='graph' with a DataFrame input is not supported. "
                "Call build_graph_from_edgelist() on the returned frame instead."
            )
        return _project_bipartite_frame_path(
            edges, projection_mode, weight_method, verbose, _t_start,
            output_format=output_format,
        )

    if not isinstance(edges, nk.Graph):
        raise ValidationError(
            f"`edges` must be a NetworkIt graph, Polars DataFrame, or EdgeList; "
            f"got {type(edges).__name__}"
        )
    if id_mapper is None:
        raise ValidationError("`id_mapper` is required when `edges` is a NetworkIt graph")

    graph = edges  # readability
    want_df_only = output_format == "dataframe"
    want_el_only = output_format == "edgelist"

    log_function_entry("project_bipartite",
                      projection_mode=projection_mode, weight_method=weight_method)

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

            # Step 4: Short-circuit when the caller wants a frame back — skip
            # the NetworkIt graph build (and its O(E) addEdge loop) entirely.
            if want_df_only:
                edge_df = _compute_projection_edges(
                    projection_partition, neighbor_map, weight_method
                )
                if verbose:
                    dt = _time.perf_counter() - _t_start
                    print(
                        f"[project_bipartite] {dt:.2f}s | mode={projection_mode}, "
                        f"weight={weight_method} | "
                        f"bipartite: {graph.numberOfNodes():,} nodes, "
                        f"{graph.numberOfEdges():,} edges → "
                        f"projected frame: {edge_df.height:,} edges"
                    )
                return edge_df

            # Step 4b: Short-circuit for EdgeList output — compute arrays via
            # the legacy kernel (we already have a neighbor map) and wrap.
            if want_el_only:
                i_arr, j_arr, weights, sorted_projection = _compute_projection_arrays(
                    projection_partition, neighbor_map, weight_method,
                )
                projected_el, new_id_mapper = _projection_arrays_to_edgelist(
                    i_arr, j_arr, weights, sorted_projection,
                )
                if verbose:
                    dt = _time.perf_counter() - _t_start
                    print(
                        f"[project_bipartite] {dt:.2f}s | mode={projection_mode}, "
                        f"weight={weight_method} | "
                        f"bipartite: {graph.numberOfNodes():,} nodes, "
                        f"{graph.numberOfEdges():,} edges → "
                        f"projected EdgeList: {projected_el.number_of_edges():,} edges"
                    )
                return projected_el, new_id_mapper

            # Step 5: Create projected graph and new ID mapper
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


def _neighbor_map_from_edges(
    df: pl.DataFrame,
    projection_mode: str,
) -> Tuple[List[Any], Dict[Any, Set[Any]]]:
    """Frame analog of :func:`_build_neighbor_mapping`.

    Treats the ``source_id`` column as the source partition and the
    ``target_id`` column as the target partition (the natural convention for
    frames produced by :func:`graph_to_edges` and friends — no BFS partition
    detection needed since partitions are implicit in the columns).

    Parameters
    ----------
    df : pl.DataFrame
        Bipartite edge frame with columns ``source_id``, ``target_id``.
    projection_mode : str
        Either ``"source"`` (project onto unique ``source_id`` values) or
        ``"target"`` (project onto unique ``target_id`` values).

    Returns
    -------
    Tuple[List[Any], Dict[Any, Set[Any]]]
        ``(projection_partition, neighbor_map)`` — the unique projection-side
        node IDs and a mapping from each to its bipartite-neighbor set.
    """
    if projection_mode == "source":
        proj_col, other_col = "source_id", "target_id"
    else:
        proj_col, other_col = "target_id", "source_id"

    # Dedupe (proj, other) pairs to mirror the set() semantics of the graph path.
    pairs = df.select(pl.col(proj_col), pl.col(other_col)).unique()
    grouped = pairs.group_by(proj_col).agg(pl.col(other_col).alias("neighbors"))

    projection_partition = grouped[proj_col].to_list()
    neighbor_map: Dict[Any, Set[Any]] = {
        row[proj_col]: set(row["neighbors"])
        for row in grouped.iter_rows(named=True)
    }
    return projection_partition, neighbor_map


def _project_bipartite_frame_path(
    df: pl.DataFrame,
    projection_mode: str,
    weight_method: str,
    verbose: bool,
    t_start: float,
    output_format: Optional[str] = None,
) -> Union[pl.DataFrame, Tuple[EdgeList, IDMapper]]:
    """Frame-input branch of :func:`project_bipartite`.

    Builds a coded :class:`EdgeList` from the input frame (which encodes
    the bipartite edges as ``UInt32`` codes paired with an IDMapper),
    dispatches to the vectorized
    :func:`_compute_projection_arrays_coded` kernel, then translates the
    projection arrays into the requested output format.

    Parameters
    ----------
    output_format : str, optional
        ``None`` or ``"dataframe"`` (default behaviour — return an
        original-IDs ``source_id``/``target_id``/``weight`` frame),
        ``"edgelist"`` (return ``(EdgeList, IDMapper)`` for the projected
        unipartite graph). ``"graph"`` is rejected upstream.
    """
    required = {"source_id", "target_id"}
    missing = required - set(df.columns)
    if missing:
        raise ValidationError(
            f"Edge frame is missing required columns: {sorted(missing)}. "
            f"Expected {sorted(required)}."
        )

    # Build a coded EdgeList from the frame, then run the vectorized
    # kernel on its UInt32 src/tgt columns. The replace_strict in
    # build_edgelist_from_frame plus the np.unique inside the kernel
    # replace the legacy Dict[Any, Set[Any]] neighbor-map step.
    edge_list, mapper = build_edgelist_from_frame(
        df,
        source_col="source_id",
        target_col="target_id",
        bipartite=True,
        auto_weight=False,        # projection treats each (src, tgt) once
        remove_duplicates=False,  # _compute_projection_arrays_coded dedupes
        verbose=False,
    )

    projection_side = "src" if projection_mode == "source" else "tgt"
    i_arr, j_arr, weights, sorted_projection = _compute_projection_arrays_coded(
        edge_list, mapper, projection_side, weight_method,
    )

    logger.info(
        "Projecting bipartite frame: %d nodes in projection partition "
        "(mode=%s, weight=%s)",
        len(sorted_projection), projection_mode, weight_method,
    )

    if output_format == "edgelist":
        projected_el, projected_mapper = _projection_arrays_to_edgelist(
            i_arr, j_arr, weights, sorted_projection,
        )
        if verbose:
            import time as _time
            dt = _time.perf_counter() - t_start
            print(
                f"[project_bipartite] {dt:.2f}s | mode={projection_mode}, "
                f"weight={weight_method} | frame: {df.height:,} bipartite edges → "
                f"projected EdgeList: {projected_el.number_of_edges():,} edges"
            )
        return projected_el, projected_mapper

    # Default: original-IDs DataFrame.
    edge_df = _projection_arrays_to_edge_frame(
        i_arr, j_arr, weights, sorted_projection,
    )

    if verbose:
        import time as _time
        dt = _time.perf_counter() - t_start
        print(
            f"[project_bipartite] {dt:.2f}s | mode={projection_mode}, "
            f"weight={weight_method} | frame: {df.height:,} bipartite edges → "
            f"{edge_df.height:,} projected edges"
        )

    return edge_df


def _project_bipartite_edgelist_path(
    edge_list: EdgeList,
    id_mapper: IDMapper,
    projection_mode: str,
    weight_method: str,
    output_format: Optional[str],
    verbose: bool,
    t_start: float,
) -> Union[Tuple[EdgeList, IDMapper], Tuple[nk.Graph, IDMapper], pl.DataFrame]:
    """EdgeList-input branch of :func:`project_bipartite`.

    Dispatches to the vectorized
    :func:`_compute_projection_arrays_coded` kernel directly — no
    re-encoding, no neighbor-map construction. The output format defaults
    to a coded :class:`EdgeList` (match input) but can be forced to
    ``"graph"`` or ``"dataframe"`` via ``output_format``.
    """
    projection_side = "src" if projection_mode == "source" else "tgt"
    i_arr, j_arr, weights, sorted_projection = _compute_projection_arrays_coded(
        edge_list, id_mapper, projection_side, weight_method,
    )

    logger.info(
        "Projecting bipartite EdgeList: %d nodes in projection partition "
        "(mode=%s, weight=%s)",
        len(sorted_projection), projection_mode, weight_method,
    )

    n_in_edges = edge_list.number_of_edges()

    if output_format == "dataframe":
        edge_df = _projection_arrays_to_edge_frame(
            i_arr, j_arr, weights, sorted_projection,
        )
        if verbose:
            import time as _time
            dt = _time.perf_counter() - t_start
            print(
                f"[project_bipartite] {dt:.2f}s | mode={projection_mode}, "
                f"weight={weight_method} | EdgeList: {n_in_edges:,} bipartite edges → "
                f"{edge_df.height:,} projected edges"
            )
        return edge_df

    if output_format == "graph":
        projected_graph, new_id_mapper = _projection_arrays_to_graph(
            i_arr, j_arr, weights, sorted_projection,
        )
        if verbose:
            import time as _time
            dt = _time.perf_counter() - t_start
            print(
                f"[project_bipartite] {dt:.2f}s | mode={projection_mode}, "
                f"weight={weight_method} | EdgeList: {n_in_edges:,} bipartite edges → "
                f"projected graph: {projected_graph.numberOfNodes():,} nodes, "
                f"{projected_graph.numberOfEdges():,} edges"
            )
        return projected_graph, new_id_mapper

    # Default ("edgelist" or None → match input).
    projected_el, new_id_mapper = _projection_arrays_to_edgelist(
        i_arr, j_arr, weights, sorted_projection,
        code_dtype=edge_list.code_dtype,
    )
    if verbose:
        import time as _time
        dt = _time.perf_counter() - t_start
        print(
            f"[project_bipartite] {dt:.2f}s | mode={projection_mode}, "
            f"weight={weight_method} | EdgeList: {n_in_edges:,} bipartite edges → "
            f"projected EdgeList: {projected_el.number_of_edges():,} edges"
        )
    return projected_el, new_id_mapper


def _compute_projection_arrays(
    projection_partition: List[Any],
    neighbor_map: Dict[Any, Set[Any]],
    weight_method: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[Any]]:
    """Sparse-matrix kernel shared by the graph-output and frame-output paths.

    Computes the bipartite co-occurrence projection
    ``AAT[i,j] = |N(i) ∩ N(j)|`` and applies the requested weight method,
    returning the projection edges as integer-index numpy arrays plus the
    deterministic projection-node ordering.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray, List[Any]]
        - ``i_arr``, ``j_arr``: integer indices into ``sorted_projection``,
          strict upper triangle (no self-loops, no duplicates).
        - ``weights``: float64 edge weights, all > 0.
        - ``sorted_projection``: the projection partition sorted by ``str``
          (deterministic; matches the legacy IDMapper internal-ID ordering).
    """
    sorted_projection = sorted(projection_partition, key=str)
    n = len(sorted_projection)

    empty_arrays = (
        np.zeros(0, dtype=np.int64),
        np.zeros(0, dtype=np.int64),
        np.zeros(0, dtype=np.float64),
        sorted_projection,
    )
    if n == 0:
        return empty_arrays

    # Build the bipartite incidence matrix A (n × n_intermediate).
    intermediate_to_col: Dict[Any, int] = {}
    rows: List[int] = []
    cols: List[int] = []
    for i, original_id in enumerate(sorted_projection):
        for inter_orig in neighbor_map.get(original_id, ()):
            col = intermediate_to_col.get(inter_orig)
            if col is None:
                col = len(intermediate_to_col)
                intermediate_to_col[inter_orig] = col
            rows.append(i)
            cols.append(col)

    n_intermediates = len(intermediate_to_col)
    if not rows or n_intermediates == 0:
        return empty_arrays

    A = sp.csr_matrix(
        (np.ones(len(rows), dtype=np.int64), (np.asarray(rows), np.asarray(cols))),
        shape=(n, n_intermediates),
    )

    # Shared-neighbor counts; keep strict upper triangle only (undirected, no self-loops).
    AAT = (A @ A.T).tocsr()
    AAT_upper = sp.triu(AAT, k=1).tocoo()

    if AAT_upper.nnz == 0:
        return empty_arrays

    i_arr = AAT_upper.row.astype(np.int64, copy=False)
    j_arr = AAT_upper.col.astype(np.int64, copy=False)
    shared = AAT_upper.data.astype(np.int64, copy=False)

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

    # Drop weight==0 entries (can occur for jaccard/overlap on degenerate inputs).
    nonzero = weights > 0
    return (
        i_arr[nonzero],
        j_arr[nonzero],
        weights[nonzero],
        sorted_projection,
    )


def _compute_projection_arrays_coded(
    edge_list: EdgeList,
    id_mapper: IDMapper,
    projection_side: str,
    weight_method: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[Any]]:
    """Vectorized projection kernel that consumes a coded :class:`EdgeList`.

    Computes the same bipartite co-occurrence projection as
    :func:`_compute_projection_arrays` (``AAT[i,j] = |N(i) ∩ N(j)|`` then
    weight-method normalization), but skips the
    ``Dict[Any, Set[Any]]`` neighbor-map construction entirely. The
    EdgeList's ``src``/``tgt`` columns already hold compact integer codes,
    so we can hand them straight to ``np.unique(return_inverse=True)`` to
    build a dense bipartite incidence and let SciPy do the rest.

    Parameters
    ----------
    edge_list : EdgeList
        Coded bipartite edge container. The convention is that ``src``
        codes belong to the source partition and ``tgt`` codes to the
        target partition (mirrors the frame-input path of
        :func:`project_bipartite`).
    id_mapper : IDMapper
        Used only to translate projection-side codes back to their
        original IDs for the returned ``sorted_projection``.
    projection_side : str
        ``"src"`` (project onto unique src codes, target codes are
        intermediates) or ``"tgt"`` (the other direction).
    weight_method : str
        ``"count"``, ``"jaccard"``, or ``"overlap"``.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray, List[Any]]
        Same shape as :func:`_compute_projection_arrays`:
        ``i_arr``, ``j_arr`` are integer indices into ``sorted_projection``
        (strict upper triangle, no self-loops, no duplicates); ``weights``
        are float64 and > 0; ``sorted_projection`` lists the projection
        partition's original IDs in code-ascending (equivalently
        ``str``-ascending — IDMapper preserves that order) sequence.

    Notes
    -----
    Ordering matches :func:`_compute_projection_arrays` because IDMapper
    assigns internal IDs in ``sorted(..., key=str)`` order of originals,
    so ``np.unique`` on codes returns them in the same lexicographic
    order as ``sorted(partition, key=str)`` would.

    Memory: peak roughly proportional to the bipartite EdgeList (~4 bytes
    per coded edge) plus the projection's CSR adjacency (sparse). The
    Python ``Dict[Any, Set[Any]]`` step in the legacy kernel is gone, so
    huge hub-heavy projections (where a single popular intermediate node
    fans out to millions of projection-edge pairs) stay tractable.

    Time Complexity: O(E) for Polars dedup + O(E_proj) for the SciPy
    sparse multiply, where E_proj is the number of projection edges.
    """
    if projection_side == "src":
        proj_col, inter_col = "src", "tgt"
    elif projection_side == "tgt":
        proj_col, inter_col = "tgt", "src"
    else:
        raise ValueError(
            f"projection_side must be 'src' or 'tgt', got {projection_side!r}"
        )

    empty_arrays = (
        np.zeros(0, dtype=np.int64),
        np.zeros(0, dtype=np.int64),
        np.zeros(0, dtype=np.float64),
        [],
    )

    if edge_list.number_of_edges() == 0:
        return empty_arrays

    # Dedupe (proj, inter) pairs to mirror the set() semantics of
    # _neighbor_map_from_edges. Without this, a duplicate edge would
    # contribute 2 to A[i,k] and inflate AAT shared-neighbor counts.
    pairs = edge_list.df.select(proj_col, inter_col).unique()

    proj_codes = pairs[proj_col].to_numpy()
    inter_codes = pairs[inter_col].to_numpy()

    # Densify both sides to contiguous 0..n-1 ranges so the incidence
    # matrix is sized (n_proj × n_inter), not (n_total × n_total).
    # np.unique returns sorted ascending values; return_inverse gives
    # the matching dense indices for each input element.
    proj_unique, proj_dense = np.unique(proj_codes, return_inverse=True)
    inter_unique, inter_dense = np.unique(inter_codes, return_inverse=True)

    n_proj = len(proj_unique)
    n_inter = len(inter_unique)

    sorted_projection = [
        id_mapper.get_original(int(c)) for c in proj_unique.tolist()
    ]

    if n_proj == 0 or n_inter == 0:
        # No edges in the partition: return empty arrays with the
        # projection node list populated for downstream introspection.
        return (
            np.zeros(0, dtype=np.int64),
            np.zeros(0, dtype=np.int64),
            np.zeros(0, dtype=np.float64),
            sorted_projection,
        )

    # int32 incidence values — binary (0/1) entries, with headroom for
    # A @ A.T to accumulate up to ~2.1B shared neighbors per cell before
    # overflow. int8 would overflow at 127.
    A = sp.csr_matrix(
        (np.ones(len(proj_dense), dtype=np.int32),
         (proj_dense, inter_dense)),
        shape=(n_proj, n_inter),
    )

    AAT_upper = sp.triu((A @ A.T).tocsr(), k=1).tocoo()

    if AAT_upper.nnz == 0:
        return (
            np.zeros(0, dtype=np.int64),
            np.zeros(0, dtype=np.int64),
            np.zeros(0, dtype=np.float64),
            sorted_projection,
        )

    i_arr = AAT_upper.row.astype(np.int64, copy=False)
    j_arr = AAT_upper.col.astype(np.int64, copy=False)
    shared = AAT_upper.data.astype(np.int64, copy=False)

    if weight_method == "count":
        weights = shared.astype(np.float64)
    elif weight_method == "jaccard":
        degrees = np.asarray(A.sum(axis=1)).flatten().astype(np.int64)
        union = degrees[i_arr] + degrees[j_arr] - shared
        with np.errstate(divide="ignore", invalid="ignore"):
            weights = np.where(union > 0, shared / union, 0.0)
    elif weight_method == "overlap":
        degrees = np.asarray(A.sum(axis=1)).flatten().astype(np.int64)
        min_deg = np.minimum(degrees[i_arr], degrees[j_arr])
        with np.errstate(divide="ignore", invalid="ignore"):
            weights = np.where(min_deg > 0, shared / min_deg, 0.0)
    else:
        raise ValueError(f"Unknown weight method: {weight_method}")

    nonzero = weights > 0
    return (
        i_arr[nonzero],
        j_arr[nonzero],
        weights[nonzero],
        sorted_projection,
    )


def _projection_arrays_to_edge_frame(
    i_arr: np.ndarray,
    j_arr: np.ndarray,
    weights: np.ndarray,
    sorted_projection: List[Any],
) -> pl.DataFrame:
    """Translate projection arrays into a (source_id, target_id, weight) frame.

    Centralizes the small array→frame conversion used by both the legacy
    :func:`_compute_projection_edges` wrapper and the new coded path
    (frame-input branch of :func:`project_bipartite`).
    """
    if i_arr.size == 0:
        return pl.DataFrame(
            schema={
                "source_id": pl.Object,
                "target_id": pl.Object,
                "weight": pl.Float64,
            }
        )
    src_originals = [sorted_projection[i] for i in i_arr.tolist()]
    tgt_originals = [sorted_projection[j] for j in j_arr.tolist()]
    return pl.DataFrame({
        "source_id": src_originals,
        "target_id": tgt_originals,
        "weight": weights,
    })


def _projection_arrays_to_graph(
    i_arr: np.ndarray,
    j_arr: np.ndarray,
    weights: np.ndarray,
    sorted_projection: List[Any],
) -> Tuple[nk.Graph, IDMapper]:
    """Materialize a projected unipartite :class:`nk.Graph` from projection arrays.

    Internal IDs in the returned graph match the densified row indices
    used by :func:`_compute_projection_arrays` /
    :func:`_compute_projection_arrays_coded` — i.e. ``sorted_projection[k]``
    is the original ID for internal node ``k``. The returned graph is
    always weighted and undirected.
    """
    n = len(sorted_projection)
    new_id_mapper = IDMapper.from_originals(sorted_projection)
    projected_graph = nk.Graph(n, weighted=True, directed=False)

    add_edge = projected_graph.addEdge
    for u, v, w in zip(i_arr, j_arr, weights):
        add_edge(int(u), int(v), float(w))

    return projected_graph, new_id_mapper


def _projection_arrays_to_edgelist(
    i_arr: np.ndarray,
    j_arr: np.ndarray,
    weights: np.ndarray,
    sorted_projection: List[Any],
    code_dtype: Any = pl.UInt32,
) -> Tuple[EdgeList, IDMapper]:
    """Wrap projection arrays into a coded :class:`EdgeList` + new IDMapper.

    Codes in the returned EdgeList are densified row indices (the same
    indices ``i_arr``/``j_arr`` use); the new IDMapper translates code
    ``k`` back to ``sorted_projection[k]``. The result is always
    weighted and undirected, with ``bipartite=False`` (it's a projected
    unipartite graph).
    """
    n = len(sorted_projection)
    new_id_mapper = IDMapper.from_originals(sorted_projection)

    df = pl.DataFrame({
        "src": pl.Series(i_arr.astype(np.int64), dtype=code_dtype),
        "tgt": pl.Series(j_arr.astype(np.int64), dtype=code_dtype),
        "weight": pl.Series(weights, dtype=pl.Float64),
    })

    edge_list = EdgeList(
        df=df,
        directed=False,
        bipartite=False,
        n_nodes=n,
        code_dtype=code_dtype,
    )
    return edge_list, new_id_mapper


def _compute_projection_edges(
    projection_partition: List[Any],
    neighbor_map: Dict[Any, Set[Any]],
    weight_method: str,
) -> pl.DataFrame:
    """Project to an edge frame keyed by original IDs.

    Thin wrapper around :func:`_compute_projection_arrays` that translates
    internal indices back to the projection-partition's original IDs. Used
    by both the frame-input path of :func:`project_bipartite` and the
    ``output_format="dataframe"`` short-circuit on the graph-input path.

    Returns
    -------
    pl.DataFrame
        Columns: ``source_id``, ``target_id``, ``weight``.
    """
    i_arr, j_arr, weights, sorted_projection = _compute_projection_arrays(
        projection_partition, neighbor_map, weight_method
    )
    return _projection_arrays_to_edge_frame(i_arr, j_arr, weights, sorted_projection)


def _create_projected_graph(
    projection_partition: List[Any],
    neighbor_map: Dict[Any, Set[Any]],
    weight_method: str,
    preserve_weights: bool,
) -> Tuple[nk.Graph, IDMapper]:
    """Build the projected unipartite NetworkIt graph + IDMapper.

    Calls :func:`_compute_projection_arrays` for the sparse-matrix kernel and
    wraps the result into a NetworkIt graph keyed by the deterministic
    sorted-by-``str`` ordering of the projection partition.

    Parameters
    ----------
    preserve_weights : bool
        Unused under this implementation; the projection is always weighted.
        Kept for API compatibility with the legacy signature.

    Returns
    -------
    projected_graph : nk.Graph
        Projected unipartite graph (undirected, weighted).
    new_id_mapper : IDMapper
        ID mapper for the projected graph. Internal IDs assigned in
        sorted-by-``str`` order of the original IDs.
    """
    logger.debug(
        "Creating projected graph with %d nodes using %s weights (sparse path)",
        len(projection_partition), weight_method,
    )

    i_arr, j_arr, weights, sorted_projection = _compute_projection_arrays(
        projection_partition, neighbor_map, weight_method
    )
    return _projection_arrays_to_graph(i_arr, j_arr, weights, sorted_projection)


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
    edgelist: Union[str, pl.DataFrame, EdgeList],
    source_col: str = "source",
    target_col: str = "target",
    timestamp_col: Optional[str] = "timestamp",
    weight_col: Optional[str] = None,
    intermediate_col: str = "target",
    projected_col: str = "source",
    remove_self_loops: bool = True,
    add_edge_weights: bool = True,
    verbose: bool = True,
    *,
    id_mapper: Optional[IDMapper] = None,
    output_format: Optional[str] = None,
) -> Union[Tuple[nk.Graph, IDMapper], Tuple[EdgeList, IDMapper], pl.DataFrame]:
    """
    Convert temporal bipartite edgelist to unipartite graph using citation-convention edges.

    Two nodes that share the same intermediate node are connected by a directed
    edge that points **from the later sharer to the earlier one** (citation /
    attribution convention). Under this convention the earliest sharer of an
    item accumulates incoming edges from everyone who shared it later, which is
    how PageRank, HITS-Authority, and similar centrality metrics surface
    influential sources.

    Accepts a file path (CSV/Parquet), a Polars DataFrame, or a coded
    :class:`EdgeList` (with its paired :class:`IDMapper`). The output
    format defaults to a NetworkIt :class:`nk.Graph` paired with an
    :class:`IDMapper` (the historical return type) but can be forced to a
    coded :class:`EdgeList` or a plain frame via ``output_format``.

    For ``EdgeList`` input the timestamp must live on ``edgelist.df``
    under ``timestamp_col`` — produce one via
    :func:`build_edgelist_from_frame` with
    ``passthrough_cols=[timestamp_col], auto_weight=False, remove_duplicates=False``.
    Because EdgeList's coded columns are named ``src``/``tgt``, the
    function maps ``intermediate_col``/``projected_col`` from the
    user-provided ``source_col``/``target_col`` to the appropriate
    ``src``/``tgt`` column internally.

    .. note::
        **Input order is trusted, not validated.** This function does NOT sort
        the edgelist. The caller is responsible for sorting it so that, within
        each ``intermediate_col`` group, rows are ordered **latest first**
        (i.e. the later sharer of each intermediate node appears earlier in
        the frame). If your data is not in this order the edge directions
        will be wrong without any error being raised.

        Pre-sort with::

            edgelist = edgelist.sort([intermediate_col, timestamp_col],
                                     descending=[False, True])

        Passing ``timestamp_col=None`` is allowed when the caller has
        already arranged rows in the required latest-first order by some
        other key (e.g. a monotonic sequence number) — in that case the
        function trusts row order entirely and the temporal-decay factor
        drops out of the edge-weight formula (see ``add_edge_weights``).

    The algorithm:

    1. Dedupes each ``intermediate_col`` group on ``projected_col`` while
       preserving the caller's descending-timestamp row order (keeps the
       latest occurrence of each projected node within the group).
    2. Self-joins the deduped frame on ``intermediate_col`` and keeps the
       upper-triangular pairs ``(i, j)`` with ``i < j`` — i.e. the later
       sharer paired with each earlier sharer.
    3. Emits a directed edge ``source = later sharer → target = earlier
       sharer`` for each pair, with the temporal-decay weight described
       under ``add_edge_weights``.

    Parameters
    ----------
    edgelist : Union[str, pl.DataFrame, EdgeList]
        Edge list data containing temporal bipartite relationships.
        Can be a file path (CSV/Parquet), a Polars DataFrame, or a coded
        :class:`EdgeList` (requires ``id_mapper``, and the EdgeList's
        ``df`` must include ``timestamp_col``).
    source_col : str, default "source"
        Name of source node column in edgelist. For EdgeList input, the
        column is conceptually ``src`` (this kwarg only matters for the
        ``intermediate_col`` / ``projected_col`` cross-check).
    target_col : str, default "target"
        Name of target node column in edgelist. For EdgeList input, the
        column is conceptually ``tgt``.
    timestamp_col : Optional[str], default "timestamp"
        Name of timestamp column for the temporal-decay factor in the
        edge-weight formula. For EdgeList input, the same column name
        must exist on ``edgelist.df`` (typically because the EdgeList
        was built with
        ``build_edgelist_from_frame(..., passthrough_cols=[timestamp_col])``).
        Pass ``None`` when the caller has pre-sorted rows by some
        non-timestamp key — edge direction still follows row order, but
        the ``Δdays`` term drops out of the weight (see
        ``add_edge_weights``).
    weight_col : Optional[str], default None
        Name of weight column. If None, edges get unit weight.
    intermediate_col : str, default "target"
        Column representing intermediate nodes (will disappear in projection).
        Must equal ``source_col`` or ``target_col``.
    projected_col : str, default "source"
        Column representing nodes to preserve in unipartite projection.
        Must equal ``source_col`` or ``target_col``.
    remove_self_loops : bool, default True
        Whether to remove self-loops in the resulting unipartite graph
    add_edge_weights : bool, default True
        Whether to calculate edge weights based on endpoint weights and
        (when available) temporal decay. The formula degrades by which
        inputs are present:

        - ``timestamp_col`` given → ``(w_later + w_earlier) / 2 * 1 / (1 + Δdays)``
        - ``timestamp_col=None`` → ``(w_later + w_earlier) / 2`` (no decay term)
        - ``add_edge_weights=False`` → every edge gets unit weight.
    verbose : bool, default True
        Print a one-line summary at the end.
    id_mapper : IDMapper, optional
        Required when ``edgelist`` is an :class:`EdgeList`; ignored when
        it's a path or a DataFrame. Used to translate the EdgeList's
        ``src``/``tgt`` codes back to original IDs at the I/O boundary.
    output_format : str, optional
        ``"graph"``, ``"edgelist"``, ``"dataframe"``, or ``None`` (default —
        equivalent to ``"graph"``, preserving the legacy return type).

    Returns
    -------
    Tuple[nk.Graph, IDMapper]
        Graph-output path (default): NetworkIt directed graph and ID
        mapper for the projected unipartite network.
    Tuple[EdgeList, IDMapper]
        EdgeList-output path: the projected edges as a coded
        :class:`EdgeList` (directed, weighted) and a new ID mapper whose
        internal IDs match the EdgeList's codes.
    pl.DataFrame
        DataFrame-output path: the projected edges with columns
        ``source_id``, ``target_id``, ``weight``.

    Raises
    ------
    DataFormatError
        If required columns are missing or data format is invalid
    ValidationError
        If timestamp column contains invalid data, ``edgelist`` is neither
        a path nor a DataFrame, or ``output_format`` is invalid.
    ConfigurationError
        If ``intermediate_col`` / ``projected_col`` don't match the
        configured source/target columns.
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

    Frame output — skip the NetworkIt build entirely:

    >>> edge_df = temporal_bipartite_to_unipartite(
    ...     data,
    ...     source_col="user", target_col="item", timestamp_col="timestamp",
    ...     intermediate_col="item", projected_col="user",
    ...     output_format="dataframe",
    ... )

    EdgeList output — coded edges, lowest peak memory:

    >>> projected_el, projected_mapper = temporal_bipartite_to_unipartite(
    ...     data,
    ...     source_col="user", target_col="item", timestamp_col="timestamp",
    ...     intermediate_col="item", projected_col="user",
    ...     output_format="edgelist",
    ... )

    EdgeList input — feed a coded, timestamp-carrying EdgeList through
    the same projection without re-encoding the original IDs:

    >>> from guidedLP.network.construction import build_edgelist_from_frame
    >>> el, mapper = build_edgelist_from_frame(
    ...     data, source_col="user", target_col="item", bipartite=True,
    ...     auto_weight=False, remove_duplicates=False,
    ...     passthrough_cols=["timestamp"],
    ... )
    >>> projected_el, projected_mapper = temporal_bipartite_to_unipartite(
    ...     el, id_mapper=mapper,
    ...     source_col="user", target_col="item", timestamp_col="timestamp",
    ...     intermediate_col="item", projected_col="user",
    ...     output_format="edgelist",
    ... )

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

    Time Complexity: O(E + Σ_g |g|²) — one Polars dedup over the input
    plus a per-intermediate-group quadratic for the upper-triangular
    pair enumeration. The pair enumeration is executed as a Polars
    self-join at the Rust level (no Python per-row loop).

    Space Complexity: O(E + E_proj). The intermediate self-join
    materializes all projection-edge candidates before the ``i < j``
    filter; for hub-heavy inputs this can be the peak.
    """
    # Validate output_format up front (consistent with project_bipartite).
    if output_format not in (None, "graph", "edgelist", "dataframe"):
        raise ValidationError(
            f"output_format must be 'graph', 'edgelist', 'dataframe', or None; "
            f"got {output_format!r}"
        )
    if output_format is None:
        # Default mirrors the historical return type of this function so
        # existing callers (test_installation.py, README examples) keep
        # working without code changes.
        output_format = "graph"

    import time as _time
    _t_start = _time.perf_counter()

    log_function_entry(
        "temporal_bipartite_to_unipartite",
        intermediate_col=intermediate_col,
        projected_col=projected_col,
        output_format=output_format,
    )

    with LoggingTimer("temporal_bipartite_to_unipartite"):
        # Step 1: Resolve input type → working DataFrame + which columns to use.
        # For EdgeList input the working frame is the EdgeList's coded df
        # (src/tgt/timestamp/optional weight); for DataFrame / file inputs the
        # working frame uses the caller's source_col / target_col directly.
        input_is_edgelist = isinstance(edgelist, EdgeList)
        if isinstance(edgelist, str):
            working_df = _load_edge_list(edgelist)
            inter_col_actual = intermediate_col
            proj_col_actual = projected_col
            ts_col_actual = timestamp_col
            wt_col_actual = weight_col
        elif isinstance(edgelist, pl.DataFrame):
            working_df = edgelist
            inter_col_actual = intermediate_col
            proj_col_actual = projected_col
            ts_col_actual = timestamp_col
            wt_col_actual = weight_col
        elif input_is_edgelist:
            if id_mapper is None:
                raise ValidationError(
                    "`id_mapper` is required when `edgelist` is an EdgeList"
                )
            # Map intermediate_col / projected_col (which are user-provided
            # logical column names) to the EdgeList's physical src/tgt columns.
            # source_col → src, target_col → tgt; mirrors project_bipartite's
            # `projection_mode='source'/'target' → src/tgt` convention.
            _side_of = {source_col: "src", target_col: "tgt"}
            if intermediate_col not in _side_of:
                raise ConfigurationError(
                    f"intermediate_col '{intermediate_col}' must be either "
                    f"'{source_col}' or '{target_col}'"
                )
            if projected_col not in _side_of:
                raise ConfigurationError(
                    f"projected_col '{projected_col}' must be either "
                    f"'{source_col}' or '{target_col}'"
                )
            working_df = edgelist.df
            inter_col_actual = _side_of[intermediate_col]
            proj_col_actual = _side_of[projected_col]
            ts_col_actual = timestamp_col  # Must be present on el.df.
            wt_col_actual = (
                "weight" if weight_col is not None and "weight" in working_df.columns
                else (weight_col if weight_col in working_df.columns else None)
            )
        else:
            raise ValidationError(
                f"`edgelist` must be a file path (str), Polars DataFrame, or "
                f"EdgeList; got {type(edgelist).__name__}"
            )

        input_rows = len(working_df)

        # Step 2: Validate required columns and intermediate/projected wiring.
        if input_is_edgelist:
            # For EdgeList input the side-mapping check already ran above; here
            # we just need to confirm the timestamp column lives on el.df
            # (only when a timestamp_col was requested).
            required_cols = [inter_col_actual, proj_col_actual]
            if ts_col_actual is not None:
                required_cols.append(ts_col_actual)
            missing_cols = [c for c in required_cols if c not in working_df.columns]
            if missing_cols:
                hint = (
                    f" Build it with build_edgelist_from_frame(..., "
                    f"passthrough_cols=['{timestamp_col}'], "
                    f"auto_weight=False, remove_duplicates=False) so the "
                    f"timestamp survives onto el.df."
                    if timestamp_col is not None else ""
                )
                raise DataFormatError(
                    f"EdgeList is missing required columns: {missing_cols}.{hint}"
                )
        else:
            required_cols = [source_col, target_col]
            if timestamp_col is not None:
                required_cols.append(timestamp_col)
            missing_cols = [c for c in required_cols if c not in working_df.columns]
            if missing_cols:
                raise DataFormatError(f"Missing required columns: {missing_cols}")

            if intermediate_col not in [source_col, target_col]:
                raise ConfigurationError(
                    f"intermediate_col '{intermediate_col}' must be either "
                    f"'{source_col}' or '{target_col}'"
                )
            if projected_col not in [source_col, target_col]:
                raise ConfigurationError(
                    f"projected_col '{projected_col}' must be either "
                    f"'{source_col}' or '{target_col}'"
                )
        if intermediate_col == projected_col:
            raise ConfigurationError(
                "intermediate_col and projected_col cannot be the same"
            )

        logger.info(
            "Processing temporal bipartite conversion: %d edges "
            "(intermediate=%s, projected=%s, input=%s)",
            input_rows, intermediate_col, projected_col,
            "EdgeList" if input_is_edgelist else "DataFrame",
        )

        try:
            # Step 3: Ensure timestamp is Datetime — needed for the
            # temporal weight subtraction below. Does not change row order.
            # Skipped when no timestamp_col was requested.
            if ts_col_actual is not None and working_df[ts_col_actual].dtype != pl.Datetime:
                try:
                    working_df = working_df.with_columns(
                        pl.col(ts_col_actual).str.to_datetime().alias(ts_col_actual)
                    )
                except Exception as e:
                    raise ValidationError(
                        f"Cannot parse timestamp column '{ts_col_actual}': {e}"
                    )

            # NOTE: trusts the caller's row order. Input MUST be pre-sorted
            # by intermediate_col (any order), then latest-first within each
            # intermediate-node group. We do not re-sort because callers
            # working with very large datasets often pre-sort once and reuse.
            if ts_col_actual is not None:
                logger.info(
                    "Assuming pre-sorted input: by %s (grouped), "
                    "then %s DESCENDING within each group",
                    inter_col_actual, ts_col_actual,
                )
            else:
                logger.info(
                    "Assuming pre-sorted input: by %s (grouped), "
                    "then latest-first within each group (no timestamp_col)",
                    inter_col_actual,
                )

            # Step 4: Vectorized temporal projection (Polars self-join).
            # Replaces the legacy per-group Python loop that built lists of
            # (source, target) tuples and edge_weights — the join runs at
            # the Rust level and scales to millions of input edges. For
            # EdgeList input proj_col_actual is "src"/"tgt" so node IDs
            # below are codes; we translate back to originals before any
            # downstream graph/edgelist build.
            unipartite_df = _compute_temporal_projection_frame(
                working_df,
                intermediate_col=inter_col_actual,
                projected_col=proj_col_actual,
                timestamp_col=ts_col_actual,
                weight_col=wt_col_actual,
                add_edge_weights=add_edge_weights,
                remove_self_loops=remove_self_loops,
            )

            if input_is_edgelist and unipartite_df.height > 0:
                # Translate the projection's coded src/tgt back to original
                # IDs so the dataframe output and the downstream
                # build_*_from_frame paths match the DataFrame-input path's
                # semantics. Rebuilding the frame from scratch (rather than
                # using replace_strict on a coded column) sidesteps the
                # heterogeneous-original-dtype problem that pl.Object would
                # otherwise create.
                src_codes = unipartite_df["source"].to_list()
                tgt_codes = unipartite_df["target"].to_list()
                src_originals = id_mapper.get_original_batch([int(c) for c in src_codes])
                tgt_originals = id_mapper.get_original_batch([int(c) for c in tgt_codes])
                unipartite_df = pl.DataFrame({
                    "source": src_originals,
                    "target": tgt_originals,
                    "weight": unipartite_df["weight"],
                })

            n_proj_edges = unipartite_df.height
            logger.info(
                "Created temporal unipartite projection: %d edges",
                n_proj_edges,
            )

            # Step 5: Dispatch by output_format.
            if output_format == "dataframe":
                # Sum weights across intermediate-groups so the frame has
                # one row per projected edge — matches the "1 row per
                # projection edge" semantics of project_bipartite's
                # dataframe output. Building a graph/edgelist below would
                # apply the same aggregation, so we do it explicitly here
                # for the frame branch.
                edge_df = (
                    unipartite_df
                    .group_by(["source", "target"])
                    .agg(pl.col("weight").sum())
                    .rename({"source": "source_id", "target": "target_id"})
                )
                _print_temporal_projection_summary(
                    verbose, _t_start, input_rows, edge_df.height,
                    output_format, intermediate_col, projected_col,
                )
                return edge_df

            if output_format == "edgelist":
                if n_proj_edges == 0:
                    logger.warning(
                        "No temporal edges created - all intermediate nodes had "
                        "≤1 unique projected node"
                    )
                    empty_el = _empty_edgelist(
                        directed=True, bipartite=False, weighted=True,
                        code_dtype=pl.UInt32,
                    )
                    empty_mapper = IDMapper()
                    _print_temporal_projection_summary(
                        verbose, _t_start, input_rows, 0,
                        output_format, intermediate_col, projected_col,
                    )
                    return empty_el, empty_mapper
                edge_list, id_mapper = build_edgelist_from_frame(
                    unipartite_df,
                    source_col="source",
                    target_col="target",
                    weight_col="weight",
                    directed=True,
                    allow_self_loops=not remove_self_loops,
                    auto_weight=False,
                    remove_duplicates=False,
                    verbose=False,
                )
                _print_temporal_projection_summary(
                    verbose, _t_start, input_rows, edge_list.number_of_edges(),
                    output_format, intermediate_col, projected_col,
                )
                return edge_list, id_mapper

            # Default: output_format == "graph".
            if n_proj_edges == 0:
                logger.warning(
                    "No temporal edges created - all intermediate nodes had "
                    "≤1 unique projected node"
                )
                empty_graph = nk.Graph(0, weighted=True, directed=True)
                empty_mapper = IDMapper()
                _print_temporal_projection_summary(
                    verbose, _t_start, input_rows, 0,
                    output_format, intermediate_col, projected_col,
                )
                return empty_graph, empty_mapper

            graph, id_mapper = build_graph_from_edgelist(
                unipartite_df,
                source_col="source",
                target_col="target",
                weight_col="weight",
                directed=True,
                allow_self_loops=not remove_self_loops,
                auto_weight=False,
                remove_duplicates=False,
                verbose=False,
            )

            logger.info(
                "Final temporal unipartite graph: %d nodes, %d edges",
                graph.numberOfNodes(), graph.numberOfEdges(),
            )
            _print_temporal_projection_summary(
                verbose, _t_start, input_rows, graph.numberOfEdges(),
                output_format, intermediate_col, projected_col,
            )
            return graph, id_mapper

        except Exception as e:
            if isinstance(e, (
                GraphConstructionError, ValidationError,
                ConfigurationError, DataFormatError,
            )):
                raise
            raise GraphConstructionError(
                f"Temporal bipartite-to-unipartite conversion failed: {str(e)}",
                operation="temporal_bipartite_to_unipartite",
                cause=e,
            )


def _compute_temporal_projection_frame(
    df: pl.DataFrame,
    intermediate_col: str,
    projected_col: str,
    timestamp_col: Optional[str],
    weight_col: Optional[str],
    add_edge_weights: bool,
    remove_self_loops: bool,
) -> pl.DataFrame:
    """Vectorized temporal-projection kernel used by :func:`temporal_bipartite_to_unipartite`.

    Replaces the legacy per-group Python loop (``for intermediate_node,
    group in edgelist.group_by(...)`` + ``np.triu_indices`` + appending to
    Python lists) with a single Polars self-join. Same citation-convention
    semantics: within each intermediate-node group, the caller's latest-first
    row order is preserved by :func:`pl.DataFrame.unique` with
    ``keep="first"``, then a self-join on ``intermediate_col`` paired with
    an ``i < j`` filter enumerates the upper-triangular pairs. Index ``i``
    is the later sharer (source), ``j`` is the earlier sharer (target).

    Parameters
    ----------
    df : pl.DataFrame
        Input edge frame, pre-sorted by ``intermediate_col`` (any order)
        then latest-first within each group. Must contain
        ``intermediate_col``, ``projected_col``, ``weight_col`` when
        non-None, and ``timestamp_col`` (already cast to ``pl.Datetime``
        by the caller) when non-None.
    timestamp_col : Optional[str]
        If non-None, the temporal-decay term ``1 / (1 + Δdays)`` is
        included in the weight; if None, the term drops out and edge
        direction is determined purely by row order.
    add_edge_weights : bool
        If True and ``timestamp_col`` is non-None, edge weight is
        ``(w_i + w_j) / 2 * 1 / (1 + Δdays)``. If True and
        ``timestamp_col`` is None, edge weight is ``(w_i + w_j) / 2``.
        If False, every edge gets weight ``1.0``.
    remove_self_loops : bool
        If True, drop edges where ``source == target`` (can happen when
        the same projected node co-occurs under different intermediate
        codes — rare but possible).

    Returns
    -------
    pl.DataFrame
        Columns: ``source``, ``target``, ``weight`` (Float64). Node IDs
        are the ORIGINAL ``projected_col`` values — no coding step yet,
        so the frame is ready to be handed to
        :func:`build_graph_from_edgelist` or :func:`build_edgelist_from_frame`.
    """
    # 1. Dedupe within each intermediate group: matches the legacy
    #    per-group `seen` set semantics. The pre-sorted descending-timestamp
    #    order means `keep="first"` keeps the latest occurrence of each
    #    projected node, identical to the legacy loop.
    deduped = df.unique(
        subset=[intermediate_col, projected_col],
        keep="first",
        maintain_order=True,
    )

    # 2. Normalize to (intermediate, node, [ts], w, rank). The per-intermediate
    #    rank starts at 0 for the latest sharer (input row 0 in the group)
    #    and increases for earlier sharers — same ordering as `unique_nodes`
    #    in the legacy loop, but now derivable from a Polars window function
    #    rather than a Python list. __ts is only carried when a
    #    timestamp_col was provided; without it, the decay factor drops
    #    out of the weight formula and edge direction is determined
    #    purely by the caller's row order.
    if weight_col is None:
        weight_expr = pl.lit(1.0).cast(pl.Float64).alias("__w")
    else:
        weight_expr = pl.col(weight_col).cast(pl.Float64).alias("__w")

    normalized_cols = [
        pl.col(intermediate_col).alias("__inter"),
        pl.col(projected_col).alias("__node"),
        weight_expr,
    ]
    if timestamp_col is not None:
        normalized_cols.append(pl.col(timestamp_col).alias("__ts"))

    normalized = deduped.select(normalized_cols).with_columns(
        pl.int_range(pl.len()).over("__inter").alias("__rank")
    )

    # 3. Self-join on the intermediate column. The `i < j` filter keeps
    #    only the strict-upper-triangular pairs — same as np.triu_indices(n, k=1)
    #    in the legacy kernel, but executed at the Rust level over all
    #    intermediate groups at once.
    left_cols = [
        pl.col("__inter"),
        pl.col("__node").alias("source"),
        pl.col("__w").alias("__w_i"),
        pl.col("__rank").alias("__i"),
    ]
    right_cols = [
        pl.col("__inter"),
        pl.col("__node").alias("target"),
        pl.col("__w").alias("__w_j"),
        pl.col("__rank").alias("__j"),
    ]
    if timestamp_col is not None:
        left_cols.append(pl.col("__ts").alias("__ts_i"))
        right_cols.append(pl.col("__ts").alias("__ts_j"))

    left = normalized.select(left_cols)
    right = normalized.select(right_cols)
    pairs = left.join(right, on="__inter").filter(pl.col("__i") < pl.col("__j"))

    if pairs.is_empty():
        return pl.DataFrame({
            "source": pl.Series([], dtype=df[projected_col].dtype),
            "target": pl.Series([], dtype=df[projected_col].dtype),
            "weight": pl.Series([], dtype=pl.Float64),
        })

    # 4. Edge weight — three cases:
    #      - add_edge_weights + timestamps  → (w_i + w_j)/2 * 1/(1+Δdays)
    #      - add_edge_weights, no timestamps → (w_i + w_j)/2 (decay drops out)
    #      - add_edge_weights=False         → 1.0
    if add_edge_weights and timestamp_col is not None:
        time_diff_days = (
            (pl.col("__ts_i") - pl.col("__ts_j"))
            .dt.total_seconds()
            .abs()
            .cast(pl.Float64)
            / 86400.0
        )
        weight_value = (
            ((pl.col("__w_i") + pl.col("__w_j")) / 2.0)
            * (1.0 / (1.0 + time_diff_days))
        ).alias("weight")
    elif add_edge_weights:
        weight_value = (
            (pl.col("__w_i") + pl.col("__w_j")) / 2.0
        ).alias("weight")
    else:
        weight_value = pl.lit(1.0).cast(pl.Float64).alias("weight")

    edges = pairs.select([
        pl.col("source"),
        pl.col("target"),
        weight_value,
    ])

    # 5. Optionally drop self-loops (matches the legacy
    #    `unipartite_df.filter(source != target)` step).
    if remove_self_loops:
        rows_before = edges.height
        edges = edges.filter(pl.col("source") != pl.col("target"))
        removed = rows_before - edges.height
        if removed > 0:
            logger.info("Removed %d self-loops", removed)

    return edges