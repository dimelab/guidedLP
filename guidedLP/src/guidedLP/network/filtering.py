"""
Network filtering module for the Guided Label Propagation library.

Provides two complementary filtering interfaces:

- :func:`filter_graph` applies one or more global criteria (degree bounds,
  weight bounds, component selection, node inclusion/exclusion, centrality
  thresholds) using mask combinations.
- :func:`filter_by_seed_proximity` prunes the graph to a neighborhood
  centered on a set of seed nodes via k-hop BFS, Personalized PageRank, or
  NetworkIt's Local Tightness Expansion. Each call returns ``(graph,
  id_mapper)`` so methods can be chained (e.g. ``khop`` then ``lte``).

Backbone-extraction methods live in :mod:`guidedLP.network.backboning`.
"""

from typing import List, Dict, Any, Optional, Tuple, Union, Set

import polars as pl
import networkit as nk
import numpy as np
import scipy.sparse as sp

from guidedLP.common.id_mapper import IDMapper
from guidedLP.common.edgelist import EdgeList
from guidedLP.common.exceptions import (
    ComputationError,
    ValidationError,
)
from guidedLP.common.logging_config import get_logger, log_function_entry, LoggingTimer
from guidedLP.network.construction import _build_induced_subgraph, _extract_edge_arrays

logger = get_logger(__name__)

# Available filter types
SUPPORTED_FILTER_TYPES = [
    "min_degree", "max_degree",
    "min_source_degree", "max_source_degree",
    "min_target_degree", "max_target_degree",
    "min_weight", "giant_component_only",
    "nodes", "exclude_nodes", "centrality",
]

# Partition-aware degree filter keys (frame-friendly subset).
PARTITION_DEGREE_FILTERS = {
    "min_source_degree", "max_source_degree",
    "min_target_degree", "max_target_degree",
}

# Seed-proximity methods
SUPPORTED_PROXIMITY_METHODS = ["khop", "ppr", "lte"]
SUPPORTED_DIRECTIONS = ["out", "in", "both"]


# Filter types that operate purely on the edge frame (no graph traversal).
FRAME_FRIENDLY_FILTERS = {
    "min_degree", "max_degree",
    "min_source_degree", "max_source_degree",
    "min_target_degree", "max_target_degree",
    "min_weight", "nodes", "exclude_nodes",
}
TRAVERSAL_REQUIRED_FILTERS = {"giant_component_only", "centrality"}


def filter_graph(
    edges: Union[nk.Graph, pl.DataFrame, EdgeList],
    id_mapper: Optional[IDMapper] = None,
    filters: Optional[Dict[str, Any]] = None,
    combine: str = "and",
    *,
    output_format: Optional[str] = None,
    protected_nodes: Optional[List[Any]] = None,
    keep_disconnected: bool = False,
) -> Union[Tuple[nk.Graph, IDMapper], Tuple[EdgeList, IDMapper], pl.DataFrame]:
    """
    Apply various filters to a network based on specified criteria.

    Accepts a NetworkIt graph (with an ``id_mapper``), a Polars edge frame
    (with columns ``source_id``, ``target_id``, ``weight``), or a coded
    :class:`EdgeList` (with an ``id_mapper``). The output format defaults to
    matching the input but can be forced via ``output_format``.
    ``frame/EdgeList → graph`` is intentionally not supported — call
    :func:`build_graph_from_edgelist` (frame) or :func:`edgelist_to_graph`
    (EdgeList) if you need a graph.

    Parameters
    ----------
    edges : nk.Graph, pl.DataFrame, or EdgeList
        Either a NetworkIt graph, a Polars edge frame with columns
        ``source_id``, ``target_id``, ``weight``, or a coded
        :class:`EdgeList` whose ``src``/``tgt`` columns are integer codes
        paired with ``id_mapper``.
    id_mapper : IDMapper, optional
        Required when ``edges`` is a Graph or EdgeList; ignored when it's a
        frame (frames already use original IDs). For EdgeList input the
        mapper is returned unchanged.
    filters : Dict[str, Any]
        Dictionary specifying filter criteria. Supported filters:
        - "min_degree": int - Minimum total degree threshold
        - "max_degree": int - Maximum total degree threshold
        - "min_source_degree": int - Minimum count of distinct targets per
          source node. Drops source nodes (and their edges) below the
          threshold. For bipartite frames this is the natural source-side
          partition filter; for unipartite directed graphs it's out-degree.
        - "max_source_degree": int - Maximum count of distinct targets per
          source node. Drops generic high-activity sources (e.g. spam users).
        - "min_target_degree" / "max_target_degree": int - Same as
          ``*_source_degree`` but for the target column / partition.
        - "min_weight": float - Minimum edge weight threshold
        - "giant_component_only": bool - Keep only largest connected component
          (**graph input only** — requires traversal)
        - "nodes": List[str] - Keep only these nodes (original IDs)
        - "exclude_nodes": List[str] - Remove these nodes (original IDs)
        - "centrality": Dict - Filter by centrality metrics
          {"metric": str, "min_value": float} (**graph input only**)
    combine : str, default "and"
        How to combine multiple filters within each category (node vs. edge):
        "and" or "or". Between categories the result is always AND-combined.
    output_format : str, optional
        ``"graph"``, ``"dataframe"``, or ``None`` (default — match input).
        ``output_format="graph"`` with a frame input is not supported.
    protected_nodes : list, optional
        Original IDs of nodes that should be exempt from filtering. A
        protected node always survives node-level filters (degree,
        component, centrality, etc.), and every edge incident to a
        protected node always survives edge-level filters (weight,
        partition degree). Protection is *localized*: if the other
        endpoint of a protected edge is dropped by some other criterion,
        the edge disappears with it. IDs not present in the graph / frame
        produce a warning and are skipped.
    keep_disconnected : bool, default False
        **EdgeList input only.** When ``False`` (default), drop nodes that
        have no surviving edges after filtering and renumber the remaining
        codes densely to ``0..K-1`` (the paired ``IDMapper`` is rebuilt to
        match). When ``True``, keep all nodes from the input mapper even if
        they end up isolated. Protected nodes always survive — they are
        retained as orphan entries in the mapper even when ``False``.
        Ignored for graph and DataFrame inputs (the graph path keeps
        isolated nodes regardless; frames have no node-count metadata).

    Returns
    -------
    Tuple[nk.Graph, IDMapper]
        Graph-output path; the filtered graph and the updated ID mapper.
    pl.DataFrame
        DataFrame-output path; the filtered edge frame.

    Examples
    --------
    Basic degree filtering:

    >>> filters = {"min_degree": 3, "max_degree": 50}
    >>> filtered_g, new_mapper = filter_graph(graph, mapper, filters)

    Multiple criteria with OR logic:

    >>> filters = {
    ...     "min_degree": 10,
    ...     "nodes": ["important_node1", "important_node2"]
    ... }
    >>> filtered_g, new_mapper = filter_graph(graph, mapper, filters, combine="or")

    Giant component extraction:

    >>> filters = {"giant_component_only": True}
    >>> filtered_g, new_mapper = filter_graph(graph, mapper, filters)

    Centrality-based filtering:

    >>> filters = {
    ...     "centrality": {"metric": "betweenness", "min_value": 0.01}
    ... }
    >>> filtered_g, new_mapper = filter_graph(graph, mapper, filters)

    Raises
    ------
    ValidationError
        If filter specifications are invalid or conflicting
    ComputationError
        If filtering results in an empty graph or other computation errors

    Notes
    -----
    Time Complexity:
        O(N + E) for most filters, O(N²) for component detection

    Space Complexity:
        O(N) for node masks, O(E) for edge operations

    The function creates boolean masks for each filter criterion and combines
    them according to the specified logic. For efficiency, component detection
    is only performed when necessary.
    """
    if filters is None:
        filters = {}
    _validate_filter_parameters(filters, combine)

    # Dispatch on input type.
    if isinstance(edges, EdgeList):
        return _filter_edges_edgelist(
            edges, id_mapper, filters, combine, output_format, protected_nodes,
            keep_disconnected=keep_disconnected,
        )

    if isinstance(edges, pl.DataFrame):
        if output_format == "graph":
            raise ValidationError(
                "output_format='graph' with a DataFrame input is not supported. "
                "Call build_graph_from_edgelist() on the returned frame instead."
            )
        return _filter_edges_frame(edges, filters, combine, protected_nodes)

    if not isinstance(edges, nk.Graph):
        raise ValidationError(
            f"`edges` must be a NetworkIt graph, Polars DataFrame, or EdgeList; "
            f"got {type(edges).__name__}"
        )
    if id_mapper is None:
        raise ValidationError("`id_mapper` is required when `edges` is a NetworkIt graph")

    if output_format not in (None, "graph", "dataframe"):
        raise ValidationError(
            f"output_format must be 'graph', 'dataframe', or None; got {output_format!r}"
        )

    graph = edges  # readability

    log_function_entry(
        "filter_graph",
        graph_nodes=graph.numberOfNodes(),
        graph_edges=graph.numberOfEdges(),
        filters=list(filters.keys()),
        combine=combine
    )

    # Handle empty graph
    if graph.numberOfNodes() == 0:
        logger.warning("Empty graph provided. Returning empty graph.")
        if output_format == "dataframe":
            from guidedLP.network.construction import graph_to_edges
            return graph_to_edges(graph, id_mapper)
        return graph, id_mapper

    with LoggingTimer("filter_graph", {"filters": list(filters.keys()), "nodes": graph.numberOfNodes()}):
        try:
            # Create masks for each filter
            node_masks = []
            edge_masks = []

            for filter_type, filter_value in filters.items():
                if filter_type in ["min_degree", "max_degree"]:
                    mask = _apply_degree_filter(graph, filter_type, filter_value)
                    node_masks.append(mask)

                elif filter_type in PARTITION_DEGREE_FILTERS:
                    # Partition-aware degree is naturally an edge predicate:
                    # only the failing role-side edges are dropped, not the node.
                    mask = _apply_partition_degree_filter(graph, filter_type, filter_value)
                    edge_masks.append(mask)

                elif filter_type == "min_weight":
                    mask = _apply_weight_filter(graph, filter_value)
                    edge_masks.append(mask)

                elif filter_type == "giant_component_only" and filter_value:
                    mask = _apply_component_filter(graph)
                    node_masks.append(mask)

                elif filter_type == "nodes":
                    mask = _apply_node_inclusion_filter(graph, id_mapper, filter_value)
                    node_masks.append(mask)

                elif filter_type == "exclude_nodes":
                    mask = _apply_node_exclusion_filter(graph, id_mapper, filter_value)
                    node_masks.append(mask)

                elif filter_type == "centrality":
                    mask = _apply_centrality_filter(graph, id_mapper, filter_value)
                    node_masks.append(mask)

            # Combine masks
            final_node_mask = _combine_masks(node_masks, combine) if node_masks else None
            final_edge_mask = _combine_masks(edge_masks, combine) if edge_masks else None

            # Apply node-protection overrides: force protected nodes (and the
            # other endpoint of each protected edge) to survive, and force all
            # incident edges to survive any edge-level filters.
            if protected_nodes:
                final_node_mask, final_edge_mask = _apply_node_protection(
                    graph, id_mapper, protected_nodes,
                    final_node_mask, final_edge_mask,
                )

            # Apply filters to create new graph
            filtered_graph, updated_mapper = _apply_masks_to_graph(
                graph, id_mapper, final_node_mask, final_edge_mask
            )

            # Validate result
            if filtered_graph.numberOfNodes() == 0:
                raise ComputationError(
                    "All nodes were filtered out. Consider relaxing filter criteria.",
                    context={"operation": "filter_graph", "filters": filters}
                )

            logger.info(
                f"Graph filtering completed: {graph.numberOfNodes()} → {filtered_graph.numberOfNodes()} nodes, "
                f"{graph.numberOfEdges()} → {filtered_graph.numberOfEdges()} edges"
            )

            if output_format == "dataframe":
                from guidedLP.network.construction import graph_to_edges
                return graph_to_edges(filtered_graph, updated_mapper)

            return filtered_graph, updated_mapper

        except Exception as e:
            raise ComputationError(
                f"Graph filtering failed: {str(e)}",
                context={
                    "operation": "filter_graph",
                    "filters": filters,
                    "error_type": "computation"
                }
            ) from e


def _filter_edges_frame(
    df: pl.DataFrame,
    filters: Dict[str, Any],
    combine: str,
    protected_nodes: Optional[List[Any]] = None,
) -> pl.DataFrame:
    """Frame-input branch of :func:`filter_graph`.

    Applies the frame-friendly filter types (``min_degree``, ``max_degree``,
    ``min_weight``, ``nodes``, ``exclude_nodes``). Raises
    :class:`ValidationError` for filters that require graph traversal
    (``giant_component_only``, ``centrality``) — pointing users to
    :func:`build_graph_from_edgelist` for the explicit conversion.

    Within each category (node-derived vs. edge-derived predicates) the
    user's ``combine`` choice applies. Between categories the predicates are
    always AND-combined — matching the graph-input path.

    ``protected_nodes`` (original IDs) are merged in at the very end: any
    edge with at least one protected endpoint is forced into the kept set,
    so the protected nodes' neighborhood passes through unfiltered.
    """
    required = {"source_id", "target_id", "weight"}
    missing = required - set(df.columns)
    if missing:
        raise ValidationError(
            f"Edge frame is missing required columns: {sorted(missing)}. "
            f"Expected {sorted(required)}."
        )

    # Reject traversal-required filters early with a clear message.
    needs_graph = TRAVERSAL_REQUIRED_FILTERS & set(filters.keys())
    if needs_graph:
        raise ValidationError(
            f"Filter type(s) {sorted(needs_graph)} require a graph object "
            "(connected-component or centrality computations are not expressible "
            "as a pure edge-frame operation). Build a graph first via "
            "build_graph_from_edgelist() and pass it to filter_graph()."
        )

    # Per-node degree (unweighted in+out count). Edge frames are agnostic to
    # direction here — we mirror the original graph-side semantic which simply
    # uses graph.degree(u).
    src = df.select(pl.col("source_id").alias("node"))
    tgt = df.select(pl.col("target_id").alias("node"))
    degrees = (
        pl.concat([src, tgt])
        .group_by("node")
        .agg(pl.len().alias("degree"))
    )

    surviving_node_sets: List[Set[Any]] = []
    edge_predicates: List["pl.Expr"] = []

    for filter_type, filter_value in filters.items():
        if filter_type == "min_degree":
            kept = set(degrees.filter(pl.col("degree") >= filter_value)["node"].to_list())
            surviving_node_sets.append(kept)
        elif filter_type == "max_degree":
            kept = set(degrees.filter(pl.col("degree") <= filter_value)["node"].to_list())
            surviving_node_sets.append(kept)
        elif filter_type in PARTITION_DEGREE_FILTERS:
            # Edge-level semantic: drop only edges where the failing node
            # appears in its filtered role (source or target).
            if filter_type in ("min_source_degree", "max_source_degree"):
                proj_col, other_col = "source_id", "target_id"
            else:
                proj_col, other_col = "target_id", "source_id"
            deg_df = (
                df.group_by(proj_col)
                  .agg(pl.col(other_col).n_unique().alias("d"))
            )
            if filter_type.startswith("min_"):
                failing = deg_df.filter(pl.col("d") < filter_value)[proj_col].to_list()
            else:
                failing = deg_df.filter(pl.col("d") > filter_value)[proj_col].to_list()
            edge_predicates.append(~pl.col(proj_col).is_in(failing))
        elif filter_type == "min_weight":
            edge_predicates.append(pl.col("weight") >= filter_value)
        elif filter_type == "nodes":
            surviving_node_sets.append(set(filter_value))
        elif filter_type == "exclude_nodes":
            all_nodes = set(degrees["node"].to_list())
            surviving_node_sets.append(all_nodes - set(filter_value))

    # Combine node-derived sets per user choice, then convert to an edge
    # predicate (both endpoints must survive).
    node_edge_pred: Optional["pl.Expr"] = None
    if surviving_node_sets:
        if combine == "and":
            surviving = set.intersection(*surviving_node_sets) if len(surviving_node_sets) > 1 else surviving_node_sets[0]
        else:
            surviving = set.union(*surviving_node_sets)
        surviving_list = list(surviving)
        node_edge_pred = (
            pl.col("source_id").is_in(surviving_list)
            & pl.col("target_id").is_in(surviving_list)
        )

    # Combine edge-derived predicates per user choice.
    edge_pred: Optional["pl.Expr"] = None
    if edge_predicates:
        edge_pred = edge_predicates[0]
        for p in edge_predicates[1:]:
            edge_pred = (edge_pred & p) if combine == "and" else (edge_pred | p)

    # Cross-category combine is always AND.
    if node_edge_pred is not None and edge_pred is not None:
        final = node_edge_pred & edge_pred
    elif node_edge_pred is not None:
        final = node_edge_pred
    elif edge_pred is not None:
        final = edge_pred
    else:
        final = None

    # Protection override: any edge with at least one protected endpoint
    # is forced kept, regardless of the assembled filter predicates.
    if protected_nodes:
        protected_dedup = list(dict.fromkeys(protected_nodes))
        all_nodes_in_frame = set(degrees["node"].to_list())
        missing = [n for n in protected_dedup if n not in all_nodes_in_frame]
        if missing:
            logger.warning(
                f"{len(missing)} of {len(protected_dedup)} protected nodes "
                f"not present in edge frame (first few: {missing[:5]})"
            )
        protection_pred = (
            pl.col("source_id").is_in(protected_dedup)
            | pl.col("target_id").is_in(protected_dedup)
        )
        final = protection_pred if final is None else (final | protection_pred)

    if final is None:
        return df

    out = df.filter(final)
    if out.height == 0:
        raise ComputationError(
            "All edges were filtered out. Consider relaxing filter criteria.",
            context={"operation": "filter_graph", "filters": filters},
        )
    return out


def _filter_edges_edgelist(
    edge_list: EdgeList,
    id_mapper: Optional[IDMapper],
    filters: Dict[str, Any],
    combine: str,
    output_format: Optional[str],
    protected_nodes: Optional[List[Any]],
    *,
    keep_disconnected: bool = False,
) -> Union[Tuple[EdgeList, IDMapper], pl.DataFrame]:
    """EdgeList-input branch of :func:`filter_graph`.

    Routes through the same vectorized Polars logic as the frame path. The
    EdgeList's ``src``/``tgt`` columns (integer codes) are renamed to
    ``source_id``/``target_id`` and handed to :func:`_filter_edges_frame`.
    User-supplied original IDs (``nodes`` / ``exclude_nodes`` / ``protected_nodes``)
    are translated to codes via ``id_mapper`` before filtering so they match
    the in-frame code values.

    When ``keep_disconnected=False`` (default), nodes that lose all their
    edges are dropped from the result: distinct codes appearing in the
    filtered frame (plus protected codes) are renumbered densely to
    ``0..K-1`` and a fresh ``IDMapper`` is built. When ``keep_disconnected=True``,
    the original mapper and ``n_nodes`` are preserved (isolated nodes
    survive as orphan mapper entries).
    """
    if id_mapper is None:
        raise ValidationError("`id_mapper` is required when `edges` is an EdgeList")

    if output_format == "graph":
        raise ValidationError(
            "output_format='graph' with an EdgeList input is not supported. "
            "Call edgelist_to_graph() on the returned EdgeList instead."
        )
    if output_format not in (None, "dataframe"):
        raise ValidationError(
            f"output_format must be 'dataframe' or None for EdgeList input; "
            f"got {output_format!r}"
        )

    # Reject traversal-required filters early, mirroring the frame path.
    needs_graph = TRAVERSAL_REQUIRED_FILTERS & set(filters.keys())
    if needs_graph:
        raise ValidationError(
            f"Filter type(s) {sorted(needs_graph)} require a graph object "
            "(connected-component or centrality computations are not expressible "
            "as a pure edge-frame operation). Materialize a graph via "
            "edgelist_to_graph() and pass it to filter_graph()."
        )

    log_function_entry(
        "filter_graph",
        edge_list_nodes=edge_list.number_of_nodes(),
        edge_list_edges=edge_list.number_of_edges(),
        filters=list(filters.keys()),
        combine=combine,
        input="edgelist",
    )

    if edge_list.number_of_edges() == 0:
        logger.warning("Empty EdgeList provided. Returning empty EdgeList.")
        if output_format == "dataframe":
            empty = edge_list.df.rename({"src": "source_id", "tgt": "target_id"})
            if "weight" not in empty.columns:
                empty = empty.with_columns(pl.lit(1.0).cast(pl.Float64).alias("weight"))
            return empty
        return edge_list, id_mapper

    # Build a frame in the column shape the frame helper expects. Codes pass
    # through unchanged — group_by/joins are dtype-agnostic. Synthesize
    # weight=1.0 if the EdgeList is unweighted so min_weight (if asked) and
    # the required-column check both work.
    edges_df = edge_list.df.rename({"src": "source_id", "tgt": "target_id"})
    synthesized_weight = "weight" not in edges_df.columns
    if synthesized_weight:
        edges_df = edges_df.with_columns(pl.lit(1.0).cast(pl.Float64).alias("weight"))

    # Translate any filter values that contain original IDs to integer codes
    # so they match the EdgeList's frame. Unresolvable IDs are warned and
    # skipped (matching the convention used elsewhere in the codebase).
    translated_filters = dict(filters)
    for key in ("nodes", "exclude_nodes"):
        if key in translated_filters:
            translated_filters[key] = _translate_originals_to_codes(
                translated_filters[key], id_mapper, edge_list.code_dtype, key
            )

    translated_protected: Optional[List[int]] = None
    if protected_nodes:
        translated_protected = _translate_originals_to_codes(
            protected_nodes, id_mapper, edge_list.code_dtype, "protected_nodes"
        )

    with LoggingTimer(
        "filter_graph",
        {"filters": list(filters.keys()), "edges": edge_list.number_of_edges(),
         "input": "edgelist"},
    ):
        filtered_df = _filter_edges_frame(
            edges_df, translated_filters, combine, translated_protected
        )

    logger.info(
        "Graph filtering completed (EdgeList): %d → %d edges",
        edge_list.number_of_edges(), filtered_df.height,
    )

    if output_format == "dataframe":
        return filtered_df

    if keep_disconnected:
        # Preserve the input mapper and n_nodes; nodes whose edges all got
        # filtered survive as orphan mapper entries.
        if synthesized_weight:
            filtered_df = filtered_df.drop("weight")
        filtered_df = filtered_df.rename({"source_id": "src", "target_id": "tgt"})
        new_edge_list = EdgeList(
            df=filtered_df,
            directed=edge_list.directed,
            bipartite=edge_list.bipartite,
            n_nodes=edge_list.n_nodes,
            code_dtype=edge_list.code_dtype,
        )
        return new_edge_list, id_mapper

    # Drop isolated nodes: renumber surviving codes densely and rebuild the
    # paired mapper. Protected codes are forced into the kept set even if
    # they have no surviving edges (matching the graph path's protection
    # override of node-level filters).
    return _build_induced_edgelist(
        edge_list,
        id_mapper,
        filtered_df,
        protected_codes=translated_protected,
        synthesized_weight=synthesized_weight,
    )


def _build_induced_edgelist(
    edge_list: EdgeList,
    id_mapper: IDMapper,
    filtered_df: pl.DataFrame,
    *,
    protected_codes: Optional[List[int]],
    synthesized_weight: bool,
) -> Tuple[EdgeList, IDMapper]:
    """Rebuild an EdgeList with dense codes ``0..K-1`` after edge filtering.

    Identifies all codes still appearing in the filtered frame, unions them
    with ``protected_codes`` (so explicitly-protected nodes survive as
    orphans even if all their edges were dropped), renumbers them densely,
    and rebuilds the paired mapper. Mirrors :func:`_build_induced_subgraph`
    for the EdgeList container.
    """
    src_codes = set(int(c) for c in filtered_df["source_id"].to_list())
    tgt_codes = set(int(c) for c in filtered_df["target_id"].to_list())
    keep_codes = src_codes | tgt_codes
    if protected_codes:
        keep_codes |= {int(c) for c in protected_codes}

    sorted_kept = sorted(keep_codes)
    n_kept = len(sorted_kept)

    new_mapper = IDMapper()
    for new_code, old_code in enumerate(sorted_kept):
        try:
            original = id_mapper.get_original(old_code)
        except KeyError:
            continue
        new_mapper.add_mapping(original, new_code)

    if synthesized_weight:
        filtered_df = filtered_df.drop("weight")

    if n_kept == 0:
        empty_df = filtered_df.head(0).rename({"source_id": "src", "target_id": "tgt"})
        return (
            EdgeList(
                df=empty_df,
                directed=edge_list.directed,
                bipartite=edge_list.bipartite,
                n_nodes=0,
                code_dtype=edge_list.code_dtype,
            ),
            new_mapper,
        )

    old_series = pl.Series("old", sorted_kept, dtype=edge_list.code_dtype)
    new_series = pl.Series("new", list(range(n_kept)), dtype=edge_list.code_dtype)

    renumbered = (
        filtered_df
        .with_columns([
            pl.col("source_id")
              .replace_strict(old=old_series, new=new_series)
              .alias("src"),
            pl.col("target_id")
              .replace_strict(old=old_series, new=new_series)
              .alias("tgt"),
        ])
        .drop(["source_id", "target_id"])
    )

    # Reorder columns to canonical EdgeList layout: src, tgt, weight, then
    # any passthrough columns the input carried (e.g. timestamp).
    leading = ["src", "tgt"]
    if "weight" in renumbered.columns:
        leading.append("weight")
    rest = [c for c in renumbered.columns if c not in leading]
    renumbered = renumbered.select(leading + rest)

    new_edge_list = EdgeList(
        df=renumbered,
        directed=edge_list.directed,
        bipartite=edge_list.bipartite,
        n_nodes=n_kept,
        code_dtype=edge_list.code_dtype,
    )
    return new_edge_list, new_mapper


def _translate_originals_to_codes(
    original_ids: List[Any],
    id_mapper: IDMapper,
    code_dtype: Any,
    label: str,
) -> List[int]:
    """Translate a list of original IDs to integer codes via ``id_mapper``.

    Deduplicates input, warns about IDs that aren't in the mapper, and casts
    the surviving codes to ``code_dtype`` so equality / ``is_in`` checks
    against the EdgeList's frame don't trip on dtype mismatch.
    """
    deduped = list(dict.fromkeys(original_ids))
    codes: List[int] = []
    missing: List[Any] = []
    for original in deduped:
        try:
            codes.append(int(id_mapper.get_internal(original)))
        except KeyError:
            missing.append(original)
    if missing:
        logger.warning(
            f"{len(missing)} of {len(deduped)} {label} not found in id_mapper "
            f"(first few: {missing[:5]})"
        )
    if not codes:
        return []
    return pl.Series(label, codes).cast(code_dtype).to_list()


# Helper functions for validation

def _validate_filter_parameters(filters: Dict[str, Any], combine: str) -> None:
    """Validate filter graph parameters."""
    if not filters:
        raise ValidationError("At least one filter must be specified")

    if combine not in ["and", "or"]:
        raise ValidationError("combine parameter must be 'and' or 'or'")

    for filter_type in filters.keys():
        if filter_type not in SUPPORTED_FILTER_TYPES:
            raise ValidationError(
                f"Unsupported filter type: {filter_type}. "
                f"Supported types: {SUPPORTED_FILTER_TYPES}"
            )

    # Check for conflicting degree filters (total degree)
    if "min_degree" in filters and "max_degree" in filters:
        if filters["min_degree"] > filters["max_degree"]:
            raise ValidationError("min_degree cannot be greater than max_degree")

    # Same conflict check on partition-aware degree pairs.
    for side in ("source", "target"):
        min_key, max_key = f"min_{side}_degree", f"max_{side}_degree"
        if min_key in filters and max_key in filters:
            if filters[min_key] > filters[max_key]:
                raise ValidationError(f"{min_key} cannot be greater than {max_key}")

    # Validate partition-aware degree thresholds are positive ints.
    for key in PARTITION_DEGREE_FILTERS:
        if key in filters:
            v = filters[key]
            if not isinstance(v, int) or isinstance(v, bool) or v < 0:
                raise ValidationError(
                    f"{key} threshold must be a non-negative integer, got {v!r}"
                )

    # Validate centrality filter format
    if "centrality" in filters:
        centrality_filter = filters["centrality"]
        if not isinstance(centrality_filter, dict):
            raise ValidationError("centrality filter must be a dictionary")
        if "metric" not in centrality_filter or "min_value" not in centrality_filter:
            raise ValidationError("centrality filter must have 'metric' and 'min_value' keys")


# Helper functions for filtering

def _apply_degree_filter(graph: nk.Graph, filter_type: str, threshold: int) -> np.ndarray:
    """Apply degree-based node filter."""
    n_nodes = graph.numberOfNodes()
    degrees = np.array([graph.degree(u) for u in range(n_nodes)])

    if filter_type == "min_degree":
        return degrees >= threshold
    elif filter_type == "max_degree":
        return degrees <= threshold
    else:
        raise ValueError(f"Unknown degree filter type: {filter_type}")


def _apply_partition_degree_filter(
    graph: nk.Graph, filter_type: str, threshold: int
) -> np.ndarray:
    """Build an *edge* mask for the ``min/max_source/target_degree`` filters.

    Counts distinct neighbors on the *other* side per node (the n_unique
    semantic), then drops only the edges in which a failing node appears on
    the filtered side. A node that fails as a source can still keep its
    target-side edges, and vice versa — the filter is partition-aware. Nodes
    that never appear on the filtered side are unaffected.
    """
    sources, targets, _ = _extract_edge_arrays(graph)
    n_edges = sources.size
    if n_edges == 0:
        return np.ones(0, dtype=bool)

    if filter_type in ("min_source_degree", "max_source_degree"):
        proj_arr, other_arr = sources, targets
    else:
        proj_arr, other_arr = targets, sources

    degrees = (
        pl.DataFrame({"node": proj_arr, "other": other_arr})
        .group_by("node")
        .agg(pl.col("other").n_unique().alias("d"))
    )

    if filter_type.startswith("min_"):
        failing_nodes = set(degrees.filter(pl.col("d") < threshold)["node"].to_list())
    else:
        failing_nodes = set(degrees.filter(pl.col("d") > threshold)["node"].to_list())

    if not failing_nodes:
        return np.ones(n_edges, dtype=bool)

    return np.array([int(n) not in failing_nodes for n in proj_arr], dtype=bool)


def _apply_weight_filter(graph: nk.Graph, min_weight: float) -> np.ndarray:
    """Apply weight-based edge filter."""
    if not graph.isWeighted():
        logger.warning("Weight filter applied to unweighted graph. All edges have weight 1.0")
        return np.ones(graph.numberOfEdges(), dtype=bool)

    _, _, weights = _extract_edge_arrays(graph)
    return weights >= min_weight


def _apply_component_filter(graph: nk.Graph) -> np.ndarray:
    """Apply giant component filter."""
    if graph.isDirected():
        # For directed graphs, use weakly connected components
        cc = nk.components.WeaklyConnectedComponents(graph)
    else:
        cc = nk.components.ConnectedComponents(graph)

    cc.run()
    component_sizes = cc.getComponentSizes()

    if not component_sizes:
        return np.zeros(graph.numberOfNodes(), dtype=bool)

    largest_component_id = max(component_sizes, key=component_sizes.get)

    mask = np.zeros(graph.numberOfNodes(), dtype=bool)
    for node in range(graph.numberOfNodes()):
        if cc.componentOfNode(node) == largest_component_id:
            mask[node] = True

    return mask


def _apply_node_inclusion_filter(
    graph: nk.Graph,
    id_mapper: IDMapper,
    node_list: List[str]
) -> np.ndarray:
    """Apply node inclusion filter."""
    mask = np.zeros(graph.numberOfNodes(), dtype=bool)

    for original_id in node_list:
        try:
            internal_id = id_mapper.get_internal(original_id)
            if internal_id < graph.numberOfNodes():
                mask[internal_id] = True
        except KeyError:
            logger.warning(f"Node {original_id} not found in graph")

    return mask


def _apply_node_exclusion_filter(
    graph: nk.Graph,
    id_mapper: IDMapper,
    node_list: List[str]
) -> np.ndarray:
    """Apply node exclusion filter."""
    mask = np.ones(graph.numberOfNodes(), dtype=bool)

    for original_id in node_list:
        try:
            internal_id = id_mapper.get_internal(original_id)
            if internal_id < graph.numberOfNodes():
                mask[internal_id] = False
        except KeyError:
            logger.warning(f"Node {original_id} not found in graph")

    return mask


def _apply_centrality_filter(
    graph: nk.Graph,
    id_mapper: IDMapper,
    centrality_config: Dict[str, Any]
) -> np.ndarray:
    """Apply centrality-based filter."""
    from .analysis import extract_centrality

    metric = centrality_config["metric"]
    min_value = centrality_config["min_value"]

    # Calculate centrality
    centrality_df = extract_centrality(
        graph, id_mapper, metrics=[metric], normalized=True, n_jobs=1
    )

    # Create mask based on threshold
    centrality_col = f"{metric}_centrality"
    centrality_values = centrality_df[centrality_col].to_list()

    mask = np.array(centrality_values) >= min_value
    return mask


def _apply_node_protection(
    graph: nk.Graph,
    id_mapper: IDMapper,
    protected_nodes: List[Any],
    node_mask: Optional[np.ndarray],
    edge_mask: Optional[np.ndarray],
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Force protected nodes (and their incident edges) through node/edge filters.

    Resolves original protected IDs to internal IDs (warns on unknown ones).
    Returns ``(node_mask, edge_mask)``: True entries are added for protected
    nodes (so they survive node-level filters) and for every edge with at
    least one protected endpoint (so they survive edge-level filters).

    Edges from a protected node to a *non-protected* node still disappear if
    the non-protected endpoint is dropped by some other filter — this is
    NetworkIt's natural cascade when ``removeNode`` is called and is
    consistent with the apply_backbone semantics. Protection is *localized*
    to the protected node; it does not propagate to its neighbors.
    """
    protected_internals: Set[int] = set()
    missing: List[Any] = []
    for original_id in protected_nodes:
        try:
            internal_id = id_mapper.get_internal(original_id)
        except KeyError:
            missing.append(original_id)
            continue
        if graph.hasNode(internal_id):
            protected_internals.add(internal_id)
        else:
            missing.append(original_id)

    if missing:
        logger.warning(
            f"{len(missing)} of {len(protected_nodes)} protected nodes "
            f"not present in graph (first few: {missing[:5]})"
        )

    if not protected_internals:
        return node_mask, edge_mask

    n_nodes = graph.numberOfNodes()

    if node_mask is not None:
        node_protect_mask = np.zeros(n_nodes, dtype=bool)
        for i in protected_internals:
            node_protect_mask[i] = True
        node_mask = node_mask | node_protect_mask

    if edge_mask is not None:
        sources, targets, _ = _extract_edge_arrays(graph)
        prot_arr = np.fromiter(protected_internals, dtype=np.int64)
        edge_protect_mask = np.isin(sources, prot_arr) | np.isin(targets, prot_arr)
        edge_mask = edge_mask | edge_protect_mask

    return node_mask, edge_mask


def _combine_masks(masks: List[np.ndarray], combine: str) -> np.ndarray:
    """Combine multiple boolean masks using AND or OR logic."""
    if not masks:
        return None

    if len(masks) == 1:
        return masks[0]

    combined = masks[0].copy()
    for mask in masks[1:]:
        if combine == "and":
            combined = combined & mask
        elif combine == "or":
            combined = combined | mask
        else:
            raise ValueError(f"Invalid combine operation: {combine}")

    return combined


def _apply_masks_to_graph(
    graph: nk.Graph,
    id_mapper: IDMapper,
    node_mask: Optional[np.ndarray],
    edge_mask: Optional[np.ndarray]
) -> Tuple[nk.Graph, IDMapper]:
    """Apply node and edge masks to create filtered graph."""
    # Start with all nodes if no node mask
    if node_mask is None:
        kept_nodes = set(range(graph.numberOfNodes()))
    else:
        kept_nodes = set(np.where(node_mask)[0])

    # Apply edge mask by removing filtered edges
    if edge_mask is not None:
        # Get edges that should be removed
        edges_to_remove = []
        edge_idx = 0
        for u, v in graph.iterEdges():
            if not edge_mask[edge_idx]:
                edges_to_remove.append((u, v))
            edge_idx += 1

        # Remove edges (this may isolate some nodes)
        filtered_graph = nk.Graph(graph)
        for u, v in edges_to_remove:
            if filtered_graph.hasEdge(u, v):
                filtered_graph.removeEdge(u, v)
    else:
        filtered_graph = nk.Graph(graph)

    # Remove nodes not in mask
    nodes_to_remove = []
    for node in range(graph.numberOfNodes()):
        if node not in kept_nodes:
            nodes_to_remove.append(node)

    for node in nodes_to_remove:
        if filtered_graph.hasNode(node):
            filtered_graph.removeNode(node)

    # Create updated ID mapper
    updated_mapper = IDMapper()
    for internal_id in range(graph.numberOfNodes()):
        if internal_id in kept_nodes and filtered_graph.hasNode(internal_id):
            try:
                original_id = id_mapper.get_original(internal_id)
                updated_mapper.add_mapping(original_id, internal_id)
            except KeyError:
                pass

    return filtered_graph, updated_mapper


# ---------------------------------------------------------------------------
# Seed-proximity filtering
# ---------------------------------------------------------------------------


def filter_by_seed_proximity(
    graph: nk.Graph,
    id_mapper: IDMapper,
    seeds: Union[List[Any], pl.DataFrame],
    method: str = "khop",
    *,
    # khop parameters
    hops: int = 2,
    direction: str = "both",
    # ppr parameters
    ppr_alpha: float = 0.85,
    top_n: Optional[int] = None,
    min_ppr: Optional[float] = None,
    max_iter: int = 100,
    tol: float = 1e-6,
    # lte parameters
    lte_alpha: float = 1.0,
    # common
    include_seeds: bool = True,
    seed_column: str = "node_id",
) -> Tuple[nk.Graph, IDMapper]:
    """
    Filter a graph to a neighborhood centered on a set of seed nodes.

    Three selection methods are supported. Each returns a fresh graph with
    contiguous internal IDs and a matching ``IDMapper``, so calls can be
    chained — e.g. run ``"khop"`` first to bound size, then ``"lte"`` on
    the result to keep only tightly-connected expansions.

    Parameters
    ----------
    graph : nk.Graph
        NetworkIt graph to filter.
    id_mapper : IDMapper
        Bidirectional mapping between original and internal node IDs.
    seeds : list or polars.DataFrame
        Seed nodes (original IDs). If a ``DataFrame``, the column named by
        ``seed_column`` is used. Duplicates are deduplicated.
    method : {"khop", "ppr", "lte"}, default "khop"
        Selection method:

        - ``"khop"``: BFS up to ``hops`` levels from the seed set.
          Predictable size, ignores edge strength.
        - ``"ppr"``: Personalized PageRank from the seed set. Keep nodes
          with top ``top_n`` mass and/or mass above ``min_ppr``. Aligned
          with the GLP propagation kernel — useful when filtering before
          running GLP.
        - ``"lte"``: NetworkIt's Local Tightness Expansion
          (``nk.scd.LocalTightnessExpansion``) expanded from the seed set
          as a single community. Adaptive; size depends on graph structure.

    hops : int, default 2
        Number of BFS hops for ``method="khop"``. ``hops=0`` keeps only
        the seed set itself.
    direction : {"out", "in", "both"}, default "both"
        Edge direction for ``"khop"`` and ``"ppr"`` on directed graphs.
        Ignored for undirected graphs and for ``"lte"`` (which always
        treats the graph as undirected).
    ppr_alpha : float, default 0.85
        Damping factor for Personalized PageRank. Higher values let mass
        spread further from seeds.
    top_n : int, optional
        For ``"ppr"``: keep at most this many nodes ranked by PPR mass.
    min_ppr : float, optional
        For ``"ppr"``: keep nodes with PPR mass at or above this threshold.
        At least one of ``top_n`` or ``min_ppr`` must be given.
    max_iter : int, default 100
        Maximum PPR iterations.
    tol : float, default 1e-6
        L∞ convergence tolerance for PPR.
    lte_alpha : float, default 1.0
        ``alpha`` parameter passed to ``nk.scd.LocalTightnessExpansion``.
    include_seeds : bool, default True
        Always retain seed nodes in the result, even when a method would
        otherwise exclude them (e.g. an isolated seed under ``"lte"``).
    seed_column : str, default "node_id"
        Column name used to pull seed IDs from a polars DataFrame.

    Returns
    -------
    Tuple[nk.Graph, IDMapper]
        Filtered graph with contiguous internal IDs ``0..K-1`` and a new
        ``IDMapper`` mapping original IDs to those contiguous IDs.

    Raises
    ------
    ValidationError
        If parameters are missing or out of range, seeds are empty, or all
        seeds are missing from the graph.
    ComputationError
        If filtering produces an empty graph or the underlying method fails.

    Examples
    --------
    Two-hop neighborhood from a list of seed IDs:

    >>> g2, m2 = filter_by_seed_proximity(
    ...     graph, mapper, ["alice", "bob"], method="khop", hops=2
    ... )

    Personalized PageRank keeping the top 500 nodes:

    >>> g2, m2 = filter_by_seed_proximity(
    ...     graph, mapper, seed_df, method="ppr", top_n=500
    ... )

    Chain k-hop with LTE — bound by 3 hops, then trim to the tight core:

    >>> g_hop, m_hop = filter_by_seed_proximity(
    ...     graph, mapper, seeds, method="khop", hops=3
    ... )
    >>> g_core, m_core = filter_by_seed_proximity(
    ...     g_hop, m_hop, seeds, method="lte"
    ... )

    Notes
    -----
    Time complexity:
        - ``khop``: ``O(|V'| + |E'|)`` where ``V'/E'`` are nodes/edges
          within the explored frontier.
        - ``ppr``: ``O(max_iter · nnz(A))`` for the sparse iteration.
        - ``lte``: depends on NetworkIt's SCD implementation; sublinear in
          ``|V|`` for typical seed-local expansions.

    The returned graph is rebuilt with contiguous internal IDs (0..K-1) so
    downstream matrix operations (e.g. GLP) work directly.

    Notes
    -----
    Frame input is intentionally not supported here — all three methods
    (``khop``, ``ppr``, ``lte``) are graph-traversal algorithms. Pass a
    NetworkIt graph; if you only have a frame, build one with
    :func:`build_graph_from_edgelist` first.
    """
    if not isinstance(graph, nk.Graph):
        raise ValidationError(
            "filter_by_seed_proximity requires a NetworkIt graph (its three "
            "methods are graph traversals). Convert your edge frame with "
            f"build_graph_from_edgelist() first. Got {type(graph).__name__}."
        )

    log_function_entry(
        "filter_by_seed_proximity",
        graph_nodes=graph.numberOfNodes(),
        graph_edges=graph.numberOfEdges(),
        method=method,
        n_seeds=(len(seeds) if hasattr(seeds, "__len__") else None),
    )

    _validate_proximity_parameters(
        method=method,
        hops=hops,
        direction=direction,
        ppr_alpha=ppr_alpha,
        top_n=top_n,
        min_ppr=min_ppr,
        max_iter=max_iter,
        tol=tol,
        lte_alpha=lte_alpha,
    )

    if graph.numberOfNodes() == 0:
        logger.warning("Empty graph provided. Returning empty graph.")
        return graph, id_mapper

    seed_originals = _normalize_seeds(seeds, seed_column)
    seed_internals = _resolve_seed_internals(seed_originals, id_mapper, graph)

    with LoggingTimer(
        "filter_by_seed_proximity",
        {"method": method, "nodes": graph.numberOfNodes(), "seeds": len(seed_internals)},
    ):
        try:
            if method == "khop":
                kept = _khop_select(graph, seed_internals, hops, direction)
            elif method == "ppr":
                kept = _ppr_select(
                    graph,
                    seed_internals,
                    alpha=ppr_alpha,
                    top_n=top_n,
                    min_ppr=min_ppr,
                    max_iter=max_iter,
                    tol=tol,
                    direction=direction,
                )
            elif method == "lte":
                kept = _lte_select(graph, seed_internals, lte_alpha)
            else:
                # Already validated, but be explicit.
                raise ValidationError(f"Unsupported method: {method}")

            if include_seeds:
                kept = kept | seed_internals

            new_graph, new_mapper = _build_induced_subgraph(graph, id_mapper, kept)

            if new_graph.numberOfNodes() == 0:
                raise ComputationError(
                    "Seed-proximity filtering produced an empty graph. "
                    "Try relaxing the method's parameters or supplying more seeds.",
                    context={"operation": "filter_by_seed_proximity", "method": method},
                )

            logger.info(
                f"Seed-proximity filter ({method}): "
                f"{graph.numberOfNodes()} → {new_graph.numberOfNodes()} nodes, "
                f"{graph.numberOfEdges()} → {new_graph.numberOfEdges()} edges, "
                f"{len(seed_internals)} seeds"
            )

            return new_graph, new_mapper

        except (ValidationError, ComputationError):
            raise
        except Exception as e:
            raise ComputationError(
                f"Seed-proximity filtering failed: {e}",
                context={
                    "operation": "filter_by_seed_proximity",
                    "method": method,
                    "error_type": "computation",
                },
            ) from e


# Validation ----------------------------------------------------------------


def _validate_proximity_parameters(
    *,
    method: str,
    hops: int,
    direction: str,
    ppr_alpha: float,
    top_n: Optional[int],
    min_ppr: Optional[float],
    max_iter: int,
    tol: float,
    lte_alpha: float,
) -> None:
    if method not in SUPPORTED_PROXIMITY_METHODS:
        raise ValidationError(
            f"Unsupported method: {method!r}. "
            f"Supported: {SUPPORTED_PROXIMITY_METHODS}"
        )

    if direction not in SUPPORTED_DIRECTIONS:
        raise ValidationError(
            f"Unsupported direction: {direction!r}. "
            f"Supported: {SUPPORTED_DIRECTIONS}"
        )

    if method == "khop":
        if not isinstance(hops, int) or hops < 0:
            raise ValidationError("hops must be a non-negative integer")

    if method == "ppr":
        if not (0.0 < ppr_alpha < 1.0):
            raise ValidationError("ppr_alpha must be in the open interval (0, 1)")
        if top_n is None and min_ppr is None:
            raise ValidationError(
                "For method='ppr', supply at least one of top_n or min_ppr."
            )
        if top_n is not None and top_n <= 0:
            raise ValidationError("top_n must be a positive integer")
        if min_ppr is not None and min_ppr < 0:
            raise ValidationError("min_ppr must be non-negative")
        if max_iter <= 0:
            raise ValidationError("max_iter must be positive")
        if tol <= 0:
            raise ValidationError("tol must be positive")

    if method == "lte":
        if lte_alpha <= 0:
            raise ValidationError("lte_alpha must be positive")


# Seed normalization --------------------------------------------------------


def _normalize_seeds(
    seeds: Union[List[Any], pl.DataFrame],
    seed_column: str,
) -> List[Any]:
    """Extract a deduplicated list of original seed IDs from list or DataFrame."""
    if isinstance(seeds, pl.DataFrame):
        if seed_column not in seeds.columns:
            raise ValidationError(
                f"Seed DataFrame is missing column {seed_column!r}. "
                f"Available columns: {seeds.columns}"
            )
        values = seeds.get_column(seed_column).to_list()
    elif isinstance(seeds, (list, tuple, set)):
        values = list(seeds)
    else:
        raise ValidationError(
            f"seeds must be a list, tuple, set, or polars.DataFrame; got {type(seeds).__name__}"
        )

    if not values:
        raise ValidationError("seeds must be non-empty")

    seen: Set[Any] = set()
    deduped: List[Any] = []
    for v in values:
        if v not in seen:
            seen.add(v)
            deduped.append(v)
    return deduped


def _resolve_seed_internals(
    seed_originals: List[Any],
    id_mapper: IDMapper,
    graph: nk.Graph,
) -> Set[int]:
    """Translate original seed IDs to internal IDs, warning on unknown seeds."""
    resolved: Set[int] = set()
    missing: List[Any] = []
    for original in seed_originals:
        try:
            internal = id_mapper.get_internal(original)
        except KeyError:
            missing.append(original)
            continue
        if graph.hasNode(internal):
            resolved.add(internal)
        else:
            missing.append(original)

    if missing:
        preview = missing[:5]
        logger.warning(
            f"{len(missing)} of {len(seed_originals)} seeds not present in graph "
            f"(first few: {preview})"
        )

    if not resolved:
        raise ValidationError(
            "None of the supplied seeds are present in the graph."
        )

    return resolved


# Selection methods ---------------------------------------------------------


def _iter_neighbors_directional(
    graph: nk.Graph,
    node: int,
    direction: str,
):
    """Yield neighbors of ``node`` according to ``direction``.

    For undirected graphs the direction is ignored. For directed graphs:

    - ``"out"``: outgoing neighbors only
    - ``"in"``:  incoming neighbors only
    - ``"both"``: union of both
    """
    if not graph.isDirected() or direction == "out":
        yield from graph.iterNeighbors(node)
        return

    if direction == "in":
        yield from graph.iterInNeighbors(node)
        return

    # direction == "both" on a directed graph
    seen: Set[int] = set()
    for v in graph.iterNeighbors(node):
        if v not in seen:
            seen.add(v)
            yield v
    for v in graph.iterInNeighbors(node):
        if v not in seen:
            seen.add(v)
            yield v


def _khop_select(
    graph: nk.Graph,
    seed_internals: Set[int],
    hops: int,
    direction: str,
) -> Set[int]:
    """BFS frontier expansion up to ``hops`` levels from the seed set."""
    visited: Set[int] = set(seed_internals)
    if hops == 0:
        return visited

    frontier: Set[int] = set(seed_internals)
    for _ in range(hops):
        next_frontier: Set[int] = set()
        for u in frontier:
            for v in _iter_neighbors_directional(graph, u, direction):
                if v not in visited:
                    next_frontier.add(v)
        if not next_frontier:
            break
        visited |= next_frontier
        frontier = next_frontier

    return visited


def _ppr_select(
    graph: nk.Graph,
    seed_internals: Set[int],
    *,
    alpha: float,
    top_n: Optional[int],
    min_ppr: Optional[float],
    max_iter: int,
    tol: float,
    direction: str,
) -> Set[int]:
    """Personalized PageRank from the seed set, then threshold."""
    n_nodes = graph.numberOfNodes()

    # Build adjacency in COO form. For directed graphs we respect direction;
    # for undirected we always symmetrize (matching GLP convention).
    rows: List[int] = []
    cols: List[int] = []
    weights: List[float] = []
    is_directed = graph.isDirected()
    is_weighted = graph.isWeighted()

    for u, v in graph.iterEdges():
        w = graph.weight(u, v) if is_weighted else 1.0
        if not is_directed:
            rows.append(u); cols.append(v); weights.append(w)
            rows.append(v); cols.append(u); weights.append(w)
        elif direction == "out":
            rows.append(u); cols.append(v); weights.append(w)
        elif direction == "in":
            rows.append(v); cols.append(u); weights.append(w)
        else:  # both -> symmetrize
            rows.append(u); cols.append(v); weights.append(w)
            rows.append(v); cols.append(u); weights.append(w)

    if not rows:
        # No edges: PPR mass concentrates entirely on seeds.
        return set(seed_internals)

    adj = sp.coo_matrix(
        (weights, (rows, cols)),
        shape=(n_nodes, n_nodes),
        dtype=np.float64,
    ).tocsr()

    # Row-normalize: P = D^{-1} A. Handle dangling rows by leaving them zero
    # (their mass is preserved by the teleport term).
    row_sums = np.asarray(adj.sum(axis=1)).ravel()
    nonzero = row_sums > 0
    inv = np.zeros_like(row_sums)
    inv[nonzero] = 1.0 / row_sums[nonzero]
    P = sp.diags(inv).dot(adj)

    # Seed distribution: uniform over the seed set, sums to 1.
    s = np.zeros(n_nodes, dtype=np.float64)
    seed_list = list(seed_internals)
    s[seed_list] = 1.0 / len(seed_list)

    pi = s.copy()
    for it in range(max_iter):
        pi_next = alpha * (P.T @ pi) + (1.0 - alpha) * s
        # Re-add mass lost from dangling rows so the vector stays a distribution.
        leaked = pi.sum() - pi_next.sum()
        if leaked > 0:
            pi_next += leaked * s
        diff = float(np.max(np.abs(pi_next - pi)))
        pi = pi_next
        if diff < tol:
            logger.debug(f"PPR converged after {it + 1} iterations (Δ={diff:.2e})")
            break
    else:
        logger.warning(
            f"PPR did not converge within {max_iter} iterations (final Δ={diff:.2e})"
        )

    # Apply thresholds. If both top_n and min_ppr are set, both must hold.
    candidates = np.arange(n_nodes)
    mask = np.ones(n_nodes, dtype=bool)
    if min_ppr is not None:
        mask &= pi >= min_ppr
    selected = candidates[mask]

    if top_n is not None and selected.size > top_n:
        order = np.argpartition(-pi[selected], top_n - 1)[:top_n]
        selected = selected[order]

    return set(int(i) for i in selected.tolist())


def _lte_select(
    graph: nk.Graph,
    seed_internals: Set[int],
    lte_alpha: float,
) -> Set[int]:
    """Expand the seed set via NetworkIt's Local Tightness Expansion.

    LTE operates on undirected graphs. For directed input we expand on an
    undirected view but apply the resulting node set back to the original
    graph (preserving its direction).
    """
    target = graph
    if graph.isDirected():
        target = nk.graphtools.toUndirected(graph)

    lte = nk.scd.LocalTightnessExpansion(target, alpha=lte_alpha)
    expanded = lte.expandOneCommunity(list(seed_internals))
    return set(int(v) for v in expanded)


# Subgraph rebuild ----------------------------------------------------------
# ``_build_induced_subgraph`` is now defined in
# :mod:`guidedLP.network.construction` so it can be shared with
# :mod:`guidedLP.network.reduction` (sampling, influence post-pass). It is
# imported above; this section is left as a marker only.
