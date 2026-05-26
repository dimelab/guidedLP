"""
Network backboning module for the Guided Label Propagation library.

This module implements network backbone extraction methods that identify the
most significant edges in weighted (or unweighted) networks. The single public
entry point is :func:`apply_backbone`, which dispatches on a ``method`` keyword
to one of:

- ``"disparity"`` — Serrano et al. (2009) disparity filter, fully vectorized.
- ``"noise_corrected"`` — Coscia & Neffke (2017) Bayesian noise-corrected
  backbone, vectorized in Polars.
- ``"bipartite_svn"`` — Tumminello et al. (2011) Statistically Validated
  Network filter for bipartite graphs, with Benjamini-Hochberg FDR correction
  and an optional node-retention post-filter.
- ``"weight"`` — simple weight threshold (with median fallback or explicit
  threshold / ``target_edges`` cap).
- ``"degree"`` — keep highest-degree nodes (median fallback or ``target_nodes``).
"""

from typing import Tuple, Optional, Dict, Any, Union, List, Set
import time as _time
import warnings

import numpy as np
import networkit as nk
import polars as pl
from scipy.stats import poisson

from guidedLP.common.id_mapper import IDMapper
from guidedLP.common.edgelist import EdgeList
from guidedLP.common.exceptions import (
    ComputationError,
    ConfigurationError,
    ValidationError,
)
from guidedLP.common.logging_config import get_logger, log_function_entry, LoggingTimer
from guidedLP.network.construction import _extract_edge_arrays

logger = get_logger(__name__)

# Canonical list of supported methods.
AVAILABLE_BACKBONE_METHODS = [
    "disparity",
    "noise_corrected",
    "bipartite_svn",
    "weight",
    "degree",
]
# Legacy alias retained for tests/old call sites.
AVAILABLE_METHODS = AVAILABLE_BACKBONE_METHODS


def apply_backbone(
    edges: Union[nk.Graph, pl.DataFrame, EdgeList],
    id_mapper: Optional[IDMapper] = None,
    method: str = "disparity",
    *,
    alpha: float = 0.05,
    threshold: float = 1.0,
    target_nodes: Optional[int] = None,
    target_edges: Optional[int] = None,
    target_fraction: Optional[float] = None,
    weight_threshold: Optional[float] = None,
    keep_disconnected: bool = False,
    correction: str = "fdr_bh",
    min_node_retention: Optional[float] = None,
    return_filtered_edges: bool = False,
    verbose: bool = True,
    directed: bool = False,
    output_format: Optional[str] = None,
    include_scores: bool = False,
    streaming: bool = False,
) -> Union[
    Tuple[nk.Graph, IDMapper],
    Tuple[nk.Graph, IDMapper, pl.DataFrame],
    Tuple[EdgeList, IDMapper],
    pl.DataFrame,
]:
    """
    Extract a network backbone by filtering edges with one of several methods.

    Accepts a NetworkIt graph (with an ``id_mapper``), a Polars edge frame
    (with columns ``source_id``, ``target_id``, ``weight``), or a coded
    :class:`EdgeList` (with an ``id_mapper``). The output format defaults to
    matching the input but can be forced via ``output_format``.
    ``frame/EdgeList → graph`` is intentionally not supported — call
    :func:`build_graph_from_edgelist` or :func:`edgelist_to_graph` on the
    returned object if you need a graph.

    Parameters
    ----------
    edges : nk.Graph, pl.DataFrame, or EdgeList
        Either a NetworkIt graph (weighted is required for
        ``method="disparity"`` and ``method="noise_corrected"``), a Polars
        edge frame with columns ``source_id``, ``target_id``, ``weight``, or
        a coded :class:`EdgeList` whose ``src``/``tgt`` columns are integer
        node codes paired with the supplied ``id_mapper``.
    id_mapper : IDMapper, optional
        Required when ``edges`` is a Graph or EdgeList; ignored when it's a
        frame. For EdgeList input the mapper is returned unchanged
        (isolated-node renumbering is not performed on the EdgeList path).
    method : str, default "disparity"
        Backbone extraction method. One of
        ``{"disparity", "noise_corrected", "bipartite_svn", "weight", "degree"}``.

        - ``"disparity"``: Serrano et al. disparity filter
          ``α_ij = (1 − p_ij)^(k−1)``. Edges with ``α < alpha`` are kept.
        - ``"noise_corrected"``: Coscia & Neffke (2017) Bayesian noise-corrected
          backbone. Compares observed edge weight to the configuration-model
          expectation and keeps edges whose lift exceeds ``threshold`` posterior
          standard deviations.
        - ``"bipartite_svn"``: Tumminello et al. (2011) SVN filter for
          bipartite graphs. Per-edge p-values under a Poisson configuration
          null with multiple-testing correction (``correction``) and an
          optional node-retention post-filter (``min_node_retention``).
        - ``"weight"``: keep edges with weight ≥ threshold. The threshold is
          (in priority order) ``weight_threshold`` if given, else the value
          at rank ``target_edges`` if given, else the median weight.
        - ``"degree"``: keep nodes with degree ≥ threshold. The threshold is
          (in priority order) the value at rank ``target_nodes`` if given,
          else the median degree.
    alpha : float, default 0.05
        Significance level for ``method="disparity"`` (disparity threshold)
        and ``method="bipartite_svn"`` (per-edge p-value cutoff). Must be in
        ``(0, 1)``.
    threshold : float, default 1.0
        Standard-deviation multiplier for ``method="noise_corrected"``. An
        edge is kept iff ``score − threshold · sdev_cij > 0`` — i.e. the
        observed weight exceeds the null expectation by at least ``threshold``
        posterior standard deviations. Must be ``> 0``.
    target_nodes : int, optional
        For ``method="degree"``: keep approximately this many top-degree nodes.
        Conflicts with ``target_edges``.
    target_edges : int, optional
        For ``method="weight"``: keep approximately this many highest-weight
        edges. For ``method="disparity"``: cap the kept set to this many
        edges (lowest α first). Conflicts with ``target_nodes``. See
        ``target_fraction`` for a method-agnostic way to request a specific
        kept-edge count.
    target_fraction : float, optional
        When set (in ``(0, 1]``), **overrides** ``alpha`` / ``threshold``
        for the three statistical methods (``disparity``,
        ``noise_corrected``, ``bipartite_svn``) and returns approximately
        ``ceil(target_fraction · |E_in|)`` edges, ranked by the method's
        per-edge significance margin (lowest α for disparity, highest
        ``score − threshold · sdev`` for noise_corrected, lowest p-value
        for bipartite_svn). Conflicts with ``target_edges`` /
        ``target_nodes``. Useful when you want a graph of a specific size
        rather than a statistically significant set — e.g. to feed a
        fixed-size backbone into downstream community detection or
        visualization. **Note**: the kept set is no longer "statistically
        significant" in any rigorous sense; you are asking for the
        method's top-K edges by score, nothing more.
    weight_threshold : float, optional
        For ``method="weight"``: explicit weight cutoff. Takes precedence over
        ``target_edges`` if both are set.
    keep_disconnected : bool, default False
        If True, keep isolated nodes that resulted from edge filtering. If
        False (default), drop them and renumber internal IDs.
    correction : str, default "fdr_bh"
        Multiple-testing correction for ``method="bipartite_svn"``. One of:

        - ``"fdr_bh"`` (recommended): Benjamini-Hochberg false-discovery rate.
          Scales gracefully to millions of edges.
        - ``"bonferroni"``: per-edge cutoff ``alpha / |E|``. Extremely
          conservative on large graphs.
        - ``"none"``: use ``alpha`` directly. Most permissive.
    min_node_retention : float, optional
        Only used by ``method="bipartite_svn"``. When set to a value in
        ``(0, 1]``, applied as a *post-filter* after per-edge SVN: any node
        whose retention ratio ``surviving_edges / original_edges`` is below
        this threshold is removed entirely (along with its remaining edges).
        Use this to eliminate generic high-degree nodes whose edges are
        mostly noise.
    return_filtered_edges : bool, default False
        Only meaningful when the effective output format is ``"graph"``: if
        True, returns an additional per-edge DataFrame. Schema depends on
        ``method``:

        - all methods: ``source_id``, ``target_id``, ``weight``, ``kept``
        - ``disparity``: also ``p_value``, ``alpha_score``
        - ``noise_corrected``: also ``score``, ``sdev_cij``
        - ``bipartite_svn``: also ``p_value``
        - ``weight`` / ``degree``: ``p_value`` and ``alpha_score`` are NaN

        Redundant when ``output_format="dataframe"`` (the primary return is
        already that frame).
    verbose : bool, default True
        Print a one-line summary at the end (timing + node/edge retention).
    directed : bool, default False
        Whether the underlying graph is directed. Only used when ``edges`` is
        a DataFrame (Graph and EdgeList carry this intrinsically).
    output_format : str, optional
        ``"graph"`` (returns ``(graph, mapper)`` or
        ``(graph, mapper, df)``), ``"dataframe"`` (returns the per-edge
        results frame), or ``None`` (default — matches input type, so an
        ``EdgeList`` input returns ``(EdgeList, IDMapper)``).
        ``output_format="graph"`` with a frame or EdgeList input is not
        supported.
    include_scores : bool, default False
        Only relevant when the function returns a DataFrame
        (frame-in or ``output_format="dataframe"``). Default
        (``False``) returns a lean frame: only ``source_id``,
        ``target_id``, ``weight`` columns, filtered to surviving edges.
        Set ``True`` to keep the full diagnostic frame (all edges, with
        method-specific score columns and the ``kept`` boolean).
        Independent of ``return_filtered_edges`` — that flag's graph-mode
        third-element frame is always the full diagnostic shape, regardless
        of ``include_scores``.
    streaming : bool, default False
        Collect the lazy per-method pipeline with Polars's streaming engine
        to bound peak memory by processing in batches. Slower than the
        in-memory engine on small inputs (per-batch overhead) but avoids
        OOM on very large graphs. Currently honored by ``method="noise_corrected"``;
        a no-op for other methods.

    Returns
    -------
    Tuple[nk.Graph, IDMapper]
        Graph-output path; the backbone graph and the updated ID mapper.
    Tuple[nk.Graph, IDMapper, pl.DataFrame]
        If ``return_filtered_edges=True``, additionally a per-edge results
        DataFrame (always the full diagnostic shape — all edges, all score
        columns, plus ``kept``).
    Tuple[EdgeList, IDMapper]
        EdgeList-output path; the backbone EdgeList (filtered to surviving
        edges, codes preserved) and the unchanged ID mapper.
    pl.DataFrame
        DataFrame-output path. Default: lean frame
        (``source_id``/``target_id``/``weight``, kept rows only). With
        ``include_scores=True``: full diagnostic frame, all edges, with
        method-specific score columns and ``kept``.

    Raises
    ------
    ValidationError
        If ``method`` is unknown, parameters are out of range, or required
        graph properties are missing (e.g. weighted graph for disparity).
    ConfigurationError
        If ``method="degree"`` is used without ``target_nodes`` and no
        sensible default applies, or required parameters are missing.
    ComputationError
        If extraction fails or produces an empty graph.

    Notes
    -----
    Time complexity:
        - ``disparity``: O(E) (fully vectorized in NumPy).
        - ``noise_corrected``: O(E) (vectorized in Polars).
        - ``bipartite_svn``: O(E + E log E) with BH correction.
        - ``weight``: O(E) (O(E log E) when sorting for ``target_edges``).
        - ``degree``: O(V + E).

    Examples
    --------
    >>> backbone, mapper = apply_backbone(g, m, method="disparity", alpha=0.05)
    >>> backbone, mapper = apply_backbone(g, m, method="noise_corrected", threshold=1.0)
    >>> backbone, mapper = apply_backbone(g, m, method="bipartite_svn", alpha=0.01)
    >>> backbone, mapper, edges = apply_backbone(
    ...     g, m, method="disparity", alpha=0.05, return_filtered_edges=True
    ... )

    References
    ----------
    Serrano, M. Á., Boguñá, M., & Vespignani, A. (2009). Extracting the
    multiscale backbone of complex weighted networks. PNAS, 106(16), 6483–6488.

    Coscia, M., & Neffke, F. M. H. (2017). Network backboning with noisy data.
    ICDE 2017.

    Tumminello, M., Miccichè, S., Lillo, F., Piilo, J., & Mantegna, R. N.
    (2011). Statistically validated networks in bipartite complex systems.
    PLoS ONE, 6(3), e17994.
    """
    _t_start = _time.perf_counter()

    # Dispatch on input type.
    if isinstance(edges, EdgeList):
        return _apply_backbone_edgelist_path(
            edge_list=edges,
            id_mapper=id_mapper,
            method=method,
            alpha=alpha,
            threshold=threshold,
            target_edges=target_edges,
            target_nodes=target_nodes,
            target_fraction=target_fraction,
            weight_threshold=weight_threshold,
            correction=correction,
            min_node_retention=min_node_retention,
            verbose=verbose,
            t_start=_t_start,
            output_format=output_format,
            include_scores=include_scores,
            streaming=streaming,
        )

    if isinstance(edges, pl.DataFrame):
        if output_format == "graph":
            raise ValidationError(
                "output_format='graph' with a DataFrame input is not supported. "
                "Call build_graph_from_edgelist() on the returned frame instead."
            )
        return _apply_backbone_frame_path(
            edges_df=edges,
            method=method,
            directed=directed,
            alpha=alpha,
            threshold=threshold,
            target_edges=target_edges,
            target_nodes=target_nodes,
            target_fraction=target_fraction,
            weight_threshold=weight_threshold,
            correction=correction,
            min_node_retention=min_node_retention,
            verbose=verbose,
            t_start=_t_start,
            include_scores=include_scores,
            streaming=streaming,
        )

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

    # Graph-input path. If the caller asked for a dataframe back, force the
    # per-edge df to be built and unwrap it from the result tuple at the end.
    want_df_only = output_format == "dataframe"
    return_edges_for_path = return_filtered_edges or want_df_only

    graph = edges  # readability for the rest of this function

    log_function_entry(
        "apply_backbone",
        graph_nodes=graph.numberOfNodes(),
        graph_edges=graph.numberOfEdges(),
        method=method,
        alpha=alpha,
        target_nodes=target_nodes,
        target_edges=target_edges,
    )

    _validate_backbone_parameters(
        method=method,
        alpha=alpha,
        threshold=threshold,
        target_nodes=target_nodes,
        target_edges=target_edges,
        target_fraction=target_fraction,
        weight_threshold=weight_threshold,
        correction=correction,
        min_node_retention=min_node_retention,
        graph=graph,
    )

    # Resolve target_fraction → top_k_override based on the input edge count.
    # The override path ignores alpha/threshold and ranks edges by the
    # method's per-edge significance margin (see each _*_on_edges helper).
    top_k_override: Optional[int] = None
    if target_fraction is not None:
        top_k_override = max(1, int(np.ceil(target_fraction * graph.numberOfEdges())))

    # Empty-graph short-circuit.
    if graph.numberOfNodes() == 0 or graph.numberOfEdges() == 0:
        warnings.warn("Empty graph provided. Returning empty graph.")
        empty_graph = nk.Graph(
            graph.numberOfNodes(),
            weighted=graph.isWeighted(),
            directed=graph.isDirected(),
        )
        empty_mapper = id_mapper if graph.numberOfNodes() > 0 else IDMapper()
        _print_backbone_summary(verbose, _t_start, graph, empty_graph, method)
        if want_df_only:
            empty_df = _empty_edges_df(method)
            return empty_df if include_scores else _slim_edges_df(empty_df)
        if return_filtered_edges:
            return empty_graph, empty_mapper, _empty_edges_df(method)
        return empty_graph, empty_mapper

    with LoggingTimer("apply_backbone", {
        "method": method,
        "nodes": graph.numberOfNodes(),
        "edges": graph.numberOfEdges(),
    }):
        try:
            if method == "disparity":
                result = _apply_disparity_filter(
                    graph, id_mapper, alpha, target_edges,
                    keep_disconnected, return_edges_for_path,
                    top_k_override=top_k_override,
                )
            elif method == "noise_corrected":
                result = _apply_noise_corrected_filter(
                    graph, id_mapper, threshold,
                    keep_disconnected, return_edges_for_path,
                    streaming=streaming,
                    top_k_override=top_k_override,
                )
            elif method == "bipartite_svn":
                result = _apply_bipartite_svn_filter(
                    graph, id_mapper, alpha, correction,
                    keep_disconnected, min_node_retention,
                    return_edges_for_path,
                    top_k_override=top_k_override,
                )
            elif method == "weight":
                result = _apply_weight_threshold(
                    graph, id_mapper, target_edges, weight_threshold,
                    keep_disconnected, return_edges_for_path,
                )
            elif method == "degree":
                result = _apply_degree_threshold(
                    graph, id_mapper, target_nodes,
                    keep_disconnected, return_edges_for_path,
                )
            else:
                # Defensive: _validate_backbone_parameters already guards this.
                raise ValidationError(f"Unsupported backbone method: {method}")

            if return_edges_for_path:
                backbone_graph, _, edge_results_df = result
            else:
                backbone_graph, _ = result

            logger.info(
                "Backbone extraction completed: %d → %d edges, %d → %d nodes",
                graph.numberOfEdges(), backbone_graph.numberOfEdges(),
                graph.numberOfNodes(), backbone_graph.numberOfNodes(),
            )

            _print_backbone_summary(verbose, _t_start, graph, backbone_graph, method)

            if want_df_only:
                return edge_results_df if include_scores else _slim_edges_df(edge_results_df)
            return result

        except (ConfigurationError, ComputationError, ValidationError):
            raise
        except Exception as e:
            raise ComputationError(
                f"Backbone extraction failed: {e}",
                context={
                    "operation": "apply_backbone",
                    "method": method,
                    "error_type": "computation",
                },
            ) from e


def _apply_backbone_frame_path(
    edges_df: pl.DataFrame,
    method: str,
    *,
    directed: bool,
    alpha: float,
    threshold: float,
    target_edges: Optional[int],
    target_nodes: Optional[int],
    target_fraction: Optional[float] = None,
    weight_threshold: Optional[float],
    correction: str,
    min_node_retention: Optional[float],
    verbose: bool,
    t_start: float,
    include_scores: bool = False,
    streaming: bool = False,
) -> pl.DataFrame:
    """Frame-input branch of :func:`apply_backbone`.

    Dispatches to the appropriate ``_*_on_edges`` helper. Returns either the
    lean filtered frame (``include_scores=False``, default) or the full
    diagnostic frame with score columns and the ``kept`` boolean
    (``include_scores=True``).
    """
    # Validate parameters that don't depend on a graph object.
    _validate_frame_input(edges_df, method)
    _validate_backbone_parameters(
        method=method,
        alpha=alpha,
        threshold=threshold,
        target_nodes=target_nodes,
        target_edges=target_edges,
        target_fraction=target_fraction,
        weight_threshold=weight_threshold,
        correction=correction,
        min_node_retention=min_node_retention,
        graph=None,  # frame path: no graph-side checks
    )

    n_in = edges_df.height
    if n_in == 0:
        warnings.warn("Empty edge frame provided. Returning empty result.")
        result_df = _empty_edges_df(method)
        _print_backbone_frame_summary(verbose, t_start, n_in, 0, method)
        return result_df if include_scores else _slim_edges_df(result_df)

    top_k_override: Optional[int] = None
    if target_fraction is not None:
        top_k_override = max(1, int(np.ceil(target_fraction * n_in)))

    with LoggingTimer("apply_backbone", {"method": method, "edges": n_in, "input": "frame"}):
        try:
            if method == "disparity":
                result_df = _disparity_on_edges(
                    edges_df, directed, alpha, target_edges,
                    top_k_override=top_k_override,
                )
            elif method == "noise_corrected":
                result_df = _noise_corrected_on_edges(
                    edges_df, directed, threshold, streaming=streaming,
                    top_k_override=top_k_override,
                )
            elif method == "bipartite_svn":
                # Treat as weighted if the column actually carries variation.
                w = edges_df["weight"].to_numpy()
                is_weighted_effective = w.size > 0 and not np.allclose(w, 1.0)
                result_df, _ = _bipartite_svn_on_edges(
                    edges_df, alpha, correction, min_node_retention, is_weighted_effective,
                    top_k_override=top_k_override,
                )
            elif method == "weight":
                result_df = _weight_threshold_on_edges(edges_df, target_edges, weight_threshold)
            elif method == "degree":
                result_df, _ = _degree_threshold_on_edges(edges_df, target_nodes)
            else:
                raise ValidationError(f"Unsupported backbone method: {method}")

            n_kept = int(result_df.filter(pl.col("kept")).height)
            logger.info(
                "Backbone extraction completed (frame): %d → %d edges kept",
                n_in, n_kept,
            )
            _print_backbone_frame_summary(verbose, t_start, n_in, n_kept, method)
            return result_df if include_scores else _slim_edges_df(result_df)

        except (ConfigurationError, ComputationError, ValidationError):
            raise
        except Exception as e:
            raise ComputationError(
                f"Backbone extraction failed: {e}",
                context={
                    "operation": "apply_backbone",
                    "method": method,
                    "error_type": "computation",
                },
            ) from e


def _validate_frame_input(df: pl.DataFrame, method: str) -> None:
    """Schema check for the DataFrame-input path of :func:`apply_backbone`."""
    required = {"source_id", "target_id", "weight"}
    missing = required - set(df.columns)
    if missing:
        raise ValidationError(
            f"Edge frame is missing required columns: {sorted(missing)}. "
            f"Expected {sorted(required)}."
        )


def _print_backbone_frame_summary(
    verbose: bool, t_start: float, n_in: int, n_kept: int, method: str
) -> None:
    """Frame-input variant of :func:`_print_backbone_summary`."""
    if not verbose:
        return
    dt = _time.perf_counter() - t_start
    pct = (100 * n_kept / n_in) if n_in else 0.0
    print(
        f"[apply_backbone] {dt:.2f}s | method={method} | frame: "
        f"{n_in:,} → {n_kept:,} edges kept ({pct:.1f}%)"
    )


def _apply_backbone_edgelist_path(
    edge_list: EdgeList,
    id_mapper: Optional[IDMapper],
    method: str,
    *,
    alpha: float,
    threshold: float,
    target_edges: Optional[int],
    target_nodes: Optional[int],
    target_fraction: Optional[float] = None,
    weight_threshold: Optional[float],
    correction: str,
    min_node_retention: Optional[float],
    verbose: bool,
    t_start: float,
    output_format: Optional[str],
    include_scores: bool,
    streaming: bool = False,
) -> Union[Tuple[EdgeList, IDMapper], pl.DataFrame]:
    """EdgeList-input branch of :func:`apply_backbone`.

    Routes through the same vectorized Polars helpers as the frame path.
    The EdgeList ``src``/``tgt`` columns (integer codes) are aliased to
    ``source_id``/``target_id`` and handed to the helpers as-is — the
    helpers do group-by / join operations that are dtype-agnostic, so codes
    stay UInt32/UInt64 throughout. The mapper is returned unchanged.
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

    _validate_backbone_parameters(
        method=method,
        alpha=alpha,
        threshold=threshold,
        target_nodes=target_nodes,
        target_edges=target_edges,
        target_fraction=target_fraction,
        weight_threshold=weight_threshold,
        correction=correction,
        min_node_retention=min_node_retention,
        graph=None,
    )

    want_df_only = output_format == "dataframe"
    n_in = edge_list.number_of_edges()

    top_k_override: Optional[int] = None
    if target_fraction is not None:
        top_k_override = max(1, int(np.ceil(target_fraction * n_in)))

    # Build a frame in the column shape the per-method helpers expect.
    # Codes pass through unchanged — Polars group_by/joins don't care about
    # dtype. Synthesize weight=1.0 if the EdgeList is unweighted.
    edges_df = edge_list.df.rename({"src": "source_id", "tgt": "target_id"})
    if "weight" not in edges_df.columns:
        edges_df = edges_df.with_columns(pl.lit(1.0).cast(pl.Float64).alias("weight"))

    if n_in == 0:
        warnings.warn("Empty EdgeList provided. Returning empty result.")
        _print_backbone_edgelist_summary(verbose, t_start, n_in, 0, method)
        if want_df_only:
            empty_df = _empty_edges_df(method)
            return empty_df if include_scores else _slim_edges_df(empty_df)
        return edge_list, id_mapper

    with LoggingTimer("apply_backbone", {"method": method, "edges": n_in, "input": "edgelist"}):
        try:
            if method == "disparity":
                result_df = _disparity_on_edges(
                    edges_df, edge_list.directed, alpha, target_edges,
                    top_k_override=top_k_override,
                )
            elif method == "noise_corrected":
                result_df = _noise_corrected_on_edges(
                    edges_df, edge_list.directed, threshold, streaming=streaming,
                    top_k_override=top_k_override,
                )
            elif method == "bipartite_svn":
                w = edges_df["weight"].to_numpy()
                is_weighted_effective = w.size > 0 and not np.allclose(w, 1.0)
                result_df, _ = _bipartite_svn_on_edges(
                    edges_df, alpha, correction, min_node_retention, is_weighted_effective,
                    top_k_override=top_k_override,
                )
            elif method == "weight":
                result_df = _weight_threshold_on_edges(
                    edges_df, target_edges, weight_threshold,
                )
            elif method == "degree":
                result_df, _ = _degree_threshold_on_edges(edges_df, target_nodes)
            else:
                raise ValidationError(f"Unsupported backbone method: {method}")

            n_kept = int(result_df.filter(pl.col("kept")).height)
            logger.info(
                "Backbone extraction completed (EdgeList): %d → %d edges kept",
                n_in, n_kept,
            )
            _print_backbone_edgelist_summary(verbose, t_start, n_in, n_kept, method)

            if want_df_only:
                return result_df if include_scores else _slim_edges_df(result_df)

            # Default: re-wrap surviving edges as an EdgeList. Use a semi-join
            # against the input EdgeList's frame so passthrough columns
            # (e.g. timestamp carried via build_edgelist_from_frame's
            # ``passthrough_cols``) and the original code dtype are
            # preserved through the backbone step. Every method's
            # ``result_df`` carries the input weight unmodified, so the
            # semi-join recovers all original per-row data.
            surviving_keys = (
                result_df.filter(pl.col("kept"))
                .select([
                    pl.col("source_id").alias("src").cast(edge_list.code_dtype),
                    pl.col("target_id").alias("tgt").cast(edge_list.code_dtype),
                ])
            )
            kept_df = (
                edge_list.df.lazy()
                .join(surviving_keys.lazy(), on=["src", "tgt"], how="semi")
                .collect()
            )

            new_edge_list = EdgeList(
                df=kept_df,
                directed=edge_list.directed,
                bipartite=edge_list.bipartite,
                n_nodes=edge_list.n_nodes,
                code_dtype=edge_list.code_dtype,
            )
            return new_edge_list, id_mapper

        except (ConfigurationError, ComputationError, ValidationError):
            raise
        except Exception as e:
            raise ComputationError(
                f"Backbone extraction failed: {e}",
                context={
                    "operation": "apply_backbone",
                    "method": method,
                    "error_type": "computation",
                },
            ) from e


def _print_backbone_edgelist_summary(
    verbose: bool, t_start: float, n_in: int, n_kept: int, method: str
) -> None:
    """EdgeList-input variant of :func:`_print_backbone_summary`."""
    if not verbose:
        return
    dt = _time.perf_counter() - t_start
    pct = (100 * n_kept / n_in) if n_in else 0.0
    print(
        f"[apply_backbone] {dt:.2f}s | method={method} | EdgeList: "
        f"{n_in:,} → {n_kept:,} edges kept ({pct:.1f}%)"
    )


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def _validate_backbone_parameters(
    *,
    method: str,
    alpha: float,
    threshold: float,
    target_nodes: Optional[int],
    target_edges: Optional[int],
    target_fraction: Optional[float] = None,
    weight_threshold: Optional[float],
    correction: str,
    min_node_retention: Optional[float],
    graph: Optional[nk.Graph],
) -> None:
    """Validate the cross-method parameter surface of :func:`apply_backbone`.

    ``graph`` is optional: when ``None`` (DataFrame-input path) any
    graph-side checks are skipped.
    """
    if method not in AVAILABLE_BACKBONE_METHODS:
        raise ValidationError(
            f"Invalid backbone method: {method}. "
            f"Available methods: {AVAILABLE_BACKBONE_METHODS}"
        )

    n_size_caps = sum(
        x is not None for x in (target_nodes, target_edges, target_fraction)
    )
    if n_size_caps > 1:
        raise ValidationError(
            "Specify at most one of target_nodes / target_edges / target_fraction"
        )

    if target_nodes is not None and target_nodes <= 0:
        raise ValidationError(f"target_nodes must be positive, got {target_nodes}")
    if target_edges is not None and target_edges <= 0:
        raise ValidationError(f"target_edges must be positive, got {target_edges}")
    if target_fraction is not None and not (0.0 < target_fraction <= 1.0):
        raise ValidationError(
            f"target_fraction must be in (0, 1], got {target_fraction}"
        )
    if target_fraction is not None and method not in (
        "disparity", "noise_corrected", "bipartite_svn",
    ):
        raise ValidationError(
            f"target_fraction is only supported by method='disparity', "
            f"'noise_corrected', or 'bipartite_svn'; got method={method!r}"
        )
    if weight_threshold is not None and weight_threshold <= 0:
        raise ValidationError(
            f"weight_threshold must be positive, got {weight_threshold}"
        )

    if method in ("disparity", "bipartite_svn"):
        if not (0.0 < alpha < 1.0):
            raise ValidationError(
                f"alpha must be between 0 and 1, got {alpha}"
            )

    if method == "noise_corrected":
        if threshold <= 0:
            raise ValidationError(
                f"threshold must be > 0 for noise_corrected, got {threshold}"
            )

    if method == "disparity" and graph is not None and not graph.isWeighted():
        raise ValidationError("Disparity filter requires a weighted graph")

    if method == "bipartite_svn":
        if correction not in ("fdr_bh", "bonferroni", "none"):
            raise ValidationError(
                f"Invalid correction value: {correction!r}. "
                f"Expected 'fdr_bh', 'bonferroni', or 'none'."
            )
        if (
            min_node_retention is not None
            and not (0.0 < min_node_retention <= 1.0)
        ):
            raise ValidationError(
                f"min_node_retention must be in (0.0, 1.0] or None, "
                f"got {min_node_retention!r}."
            )


# ---------------------------------------------------------------------------
# Disparity filter (Serrano et al. 2009)
# ---------------------------------------------------------------------------

def _disparity_on_edges(
    df: pl.DataFrame,
    directed: bool,
    alpha: float,
    target_edges: Optional[int] = None,
    *,
    top_k_override: Optional[int] = None,
) -> pl.DataFrame:
    """Serrano et al. (2009) disparity filter on an edge frame.

    Parameters
    ----------
    df : pl.DataFrame
        Edge frame with columns ``source_id``, ``target_id``, ``weight``.
    directed : bool
        Directed graphs use only the source-side disparity score; undirected
        graphs combine source- and target-side scores via element-wise min.
    alpha : float
        Significance level. An edge is kept iff its disparity score is below
        ``alpha`` (or exactly 1.0, the leaf-node sentinel). Ignored when
        ``top_k_override`` is set.
    target_edges : int, optional
        Cap the kept-edge count to this many lowest-α edges. Only takes
        effect *after* the α-filter, so callers wanting a fixed count
        regardless of α should use ``top_k_override`` instead.
    top_k_override : int, optional
        When set, ignore ``alpha`` / ``target_edges`` and keep exactly the
        ``top_k_override`` edges with the lowest ``alpha_score`` (most
        significant). Invalid edges (zero strength on either endpoint) are
        excluded.

    Returns
    -------
    pl.DataFrame
        Columns: ``source_id``, ``target_id``, ``weight``, ``p_value``,
        ``alpha_score``, ``kept``.
    """
    # Per-node aggregates.
    # Directed: strength = out-strength only (source-side group_by). Degree
    # for the disparity formula is the total in+out count — historically what
    # the numpy version did.
    # Undirected: strength and degree both summed across source and target sides
    # (each edge contributes to both endpoints).
    src_strength = df.group_by("source_id").agg(
        pl.col("weight").sum().alias("ws"),
        pl.len().alias("ks"),
    )
    tgt_strength = df.group_by("target_id").agg(
        pl.col("weight").sum().alias("wt"),
        pl.len().alias("kt"),
    )

    if directed:
        # Per-node "strength" used in the formula is out-strength.
        # Per-node "degree" is in+out, so we have to combine both sides.
        node_strength = src_strength.rename({"source_id": "node", "ws": "strength"}).drop("ks")
        node_degree = (
            pl.concat([
                src_strength.select(pl.col("source_id").alias("node"), pl.col("ks").alias("k")),
                tgt_strength.select(pl.col("target_id").alias("node"), pl.col("kt").alias("k")),
            ])
            .group_by("node")
            .agg(pl.col("k").sum().alias("degree"))
        )
    else:
        combined = (
            pl.concat([
                src_strength.select(
                    pl.col("source_id").alias("node"),
                    pl.col("ws").alias("strength"),
                    pl.col("ks").alias("degree"),
                ),
                tgt_strength.select(
                    pl.col("target_id").alias("node"),
                    pl.col("wt").alias("strength"),
                    pl.col("kt").alias("degree"),
                ),
            ])
            .group_by("node")
            .agg(
                pl.col("strength").sum().alias("strength"),
                pl.col("degree").sum().alias("degree"),
            )
        )
        node_strength = combined.select("node", "strength")
        node_degree = combined.select("node", "degree")

    node_stats = node_strength.join(node_degree, on="node", how="inner")

    work = (
        df.join(
            node_stats.rename({"node": "source_id", "strength": "s_u", "degree": "k_u"}),
            on="source_id", how="left",
        )
        .join(
            node_stats.rename({"node": "target_id", "strength": "s_v", "degree": "k_v"}),
            on="target_id", how="left",
        )
    )

    # Compute the source-side disparity α_uv. Guard s_u==0 by treating it as 1
    # downstream — those edges get marked invalid and dropped.
    work = work.with_columns([
        pl.when(pl.col("s_u") > 0)
            .then(pl.col("weight") / pl.col("s_u"))
            .otherwise(0.0)
            .alias("p_uv"),
    ])
    work = work.with_columns(
        _disparity_alpha_expr(
            1.0 - pl.col("p_uv"),
            pl.col("k_u").cast(pl.Float64) - 1.0,
        ).alias("alpha_uv")
    )

    if directed:
        work = work.with_columns([
            pl.col("alpha_uv").alias("alpha_score"),
            pl.col("p_uv").alias("p_value"),
        ])
    else:
        work = work.with_columns([
            pl.when(pl.col("s_v") > 0)
                .then(pl.col("weight") / pl.col("s_v"))
                .otherwise(0.0)
                .alias("p_vu"),
        ])
        work = work.with_columns(
            _disparity_alpha_expr(
                1.0 - pl.col("p_vu"),
                pl.col("k_v").cast(pl.Float64) - 1.0,
            ).alias("alpha_vu")
        )
        work = work.with_columns([
            pl.min_horizontal("alpha_uv", "alpha_vu").alias("alpha_score"),
            pl.min_horizontal("p_uv", "p_vu").alias("p_value"),
        ])

    # Invalid edges: zero strength on either endpoint. We clamp their score
    # to 1.0 and exclude them from the kept set.
    work = work.with_columns(
        ((pl.col("s_u") == 0) | (pl.col("s_v") == 0)).alias("_invalid")
    )
    work = work.with_columns([
        pl.when(pl.col("_invalid")).then(1.0).otherwise(pl.col("alpha_score")).alias("alpha_score"),
        pl.when(pl.col("_invalid")).then(1.0).otherwise(pl.col("p_value")).alias("p_value"),
    ])
    if top_k_override is not None:
        # Override path: ignore alpha / target_edges entirely. Rank valid
        # edges by ascending alpha_score (lowest = most significant) and
        # keep exactly top_k_override. Stable row_index disambiguates ties.
        work = work.with_row_index(name="_idx")
        kept_idx = (
            work.filter(~pl.col("_invalid"))
            .sort(["alpha_score", "_idx"])
            .head(top_k_override)
            .select("_idx")
        )
        work = work.with_columns(
            pl.col("_idx").is_in(kept_idx["_idx"]).alias("kept")
        ).drop("_idx")
    else:
        work = work.with_columns(
            (
                ((pl.col("alpha_score") < alpha) | (pl.col("alpha_score") >= 1.0))
                & ~pl.col("_invalid")
            ).alias("kept")
        )

        if target_edges is not None:
            n_kept = int(work.filter(pl.col("kept")).height)
            if n_kept > target_edges:
                # Rank by ascending alpha_score among valid edges; only the
                # target_edges lowest survive. row_index gives a stable ordering
                # that matches numpy's argsort behavior for ties.
                work = work.with_row_index(name="_idx")
                sorted_valid = (
                    work.filter(~pl.col("_invalid"))
                    .sort(["alpha_score", "_idx"])
                    .head(target_edges)
                    .select("_idx")
                )
                work = work.with_columns(
                    pl.col("_idx").is_in(sorted_valid["_idx"]).alias("kept")
                ).drop("_idx")

    return work.select([
        "source_id", "target_id", "weight",
        "p_value", "alpha_score", "kept",
    ])


def _disparity_alpha_expr(base: "pl.Expr", exponent: "pl.Expr") -> "pl.Expr":
    """Polars expression form of :func:`_disparity_alpha`. Same edge-case order:
    exponent ≤ 0 → 1.0; base ≤ 0 → 0.0; base ≥ 1 → 1.0; else exp(k·log(base))."""
    log_base = base.log()
    raw = (exponent * log_base).clip(lower_bound=-700.0).exp()
    return (
        pl.when(exponent <= 0).then(1.0)
        .when(base <= 0).then(0.0)
        .when(base >= 1).then(1.0)
        .otherwise(raw)
    )


def _apply_disparity_filter(
    graph: nk.Graph,
    id_mapper: IDMapper,
    alpha: float,
    target_edges: Optional[int],
    keep_disconnected: bool,
    return_filtered_edges: bool,
    *,
    top_k_override: Optional[int] = None,
) -> Union[Tuple[nk.Graph, IDMapper], Tuple[nk.Graph, IDMapper, pl.DataFrame]]:
    """Graph-shim around :func:`_disparity_on_edges`."""
    logger.debug("Applying disparity filter with alpha=%.3f", alpha)

    n_nodes = graph.numberOfNodes()
    directed = graph.isDirected()

    sources, targets, weights = _extract_edge_arrays(graph)

    if sources.size == 0:
        empty = nk.Graph(n_nodes, weighted=True, directed=directed)
        if return_filtered_edges:
            return empty, id_mapper, _empty_edges_df("disparity")
        return empty, id_mapper

    edge_df = pl.DataFrame({
        "source_id": sources,
        "target_id": targets,
        "weight": weights,
    })
    scored = _disparity_on_edges(
        edge_df, directed, alpha, target_edges,
        top_k_override=top_k_override,
    )

    kept_df = scored.filter(pl.col("kept"))
    backbone_graph, updated_mapper = _assemble_backbone(
        n_nodes, directed, weighted=True,
        kept_sources=kept_df["source_id"].to_numpy(),
        kept_targets=kept_df["target_id"].to_numpy(),
        kept_weights=kept_df["weight"].to_numpy(),
        keep_disconnected=keep_disconnected, id_mapper=id_mapper,
    )

    if return_filtered_edges:
        return backbone_graph, updated_mapper, _translate_edges_to_originals(
            scored, id_mapper
        )

    return backbone_graph, updated_mapper


def _disparity_alpha(base: np.ndarray, exponent: np.ndarray) -> np.ndarray:
    """Vectorized ``(1 − p)^(k − 1)`` matching :func:`_safe_power` edge cases.

    Elementwise, in precedence order:
    - exponent <= 0 -> 1.0 (leaf-node sentinel where k <= 1)
    - base <= 0     -> 0.0 (p = 1, the edge carries all of u's strength)
    - base >= 1     -> 1.0 (p = 0, zero-weight edge)
    - otherwise     -> exp(exponent * log(base)), clamped against underflow
    """
    result = np.ones_like(base, dtype=np.float64)
    formula_applies = exponent > 0
    compute_mask = formula_applies & (base > 0) & (base < 1)
    if np.any(compute_mask):
        log_result = exponent[compute_mask] * np.log(base[compute_mask])
        np.maximum(log_result, -700.0, out=log_result)
        result[compute_mask] = np.exp(log_result)
    result[formula_applies & (base <= 0)] = 0.0
    return result


def _safe_power(base: float, exponent: float) -> float:
    """Numerically stable power calculation for the disparity filter.

    Retained as a scalar helper for tests that exercise the original edge
    cases (very small base, large exponent, invalid inputs).
    """
    if base <= 0:
        return 0.0
    if base >= 1:
        return 1.0
    if exponent <= 0:
        return 1.0
    try:
        log_result = exponent * np.log(base)
        if log_result < -700:
            return 0.0
        return np.exp(log_result)
    except (OverflowError, ValueError):
        return 0.0


# ---------------------------------------------------------------------------
# Noise-corrected backbone (Coscia & Neffke 2017)
# ---------------------------------------------------------------------------

def _noise_corrected_on_edges(
    df: pl.DataFrame,
    directed: bool,
    threshold: float,
    *,
    streaming: bool = False,
    top_k_override: Optional[int] = None,
) -> pl.DataFrame:
    """Coscia & Neffke (2017) noise-corrected backbone on an edge frame.

    Parameters
    ----------
    df : pl.DataFrame
        Edge frame with columns ``source_id``, ``target_id``, ``weight``.
        Node-ID column types may be anything Polars can group_by on.
    directed : bool
        Whether the underlying graph is directed. For undirected graphs the
        per-node strength is computed by aggregating source- and target-side
        weight sums separately and combining them on V-row intermediates —
        the edge frame itself is never symmetrized, so peak working memory
        stays proportional to E (not 2E).
    threshold : float
        Posterior-standard-deviation multiplier. Edges are kept iff
        ``score − threshold · sdev_cij > 0``. Ignored when
        ``top_k_override`` is set.
    streaming : bool, default False
        Collect the lazy pipeline with Polars's streaming engine. Slower than
        the in-memory engine on small inputs (per-batch overhead) but bounds
        peak memory by processing in chunks — flip this on for graphs large
        enough that the in-memory engine would OOM.
    top_k_override : int, optional
        When set, ignore ``threshold`` and keep exactly the
        ``top_k_override`` edges with the highest significance margin
        ``score − threshold · sdev_cij`` (i.e. the edges most strongly
        favored by the noise-corrected lift). ``threshold`` is still used
        as the scaling factor on ``sdev_cij`` when computing the margin
        — pass the same value you'd use for a significance-based call.

    Returns
    -------
    pl.DataFrame
        Columns: ``source_id``, ``target_id``, ``weight``, ``score``,
        ``sdev_cij``, ``kept`` — one row per input edge. Orientation is
        preserved from the input frame (no canonical reordering on the
        undirected path — the noise-corrected score is symmetric in
        s_u/s_v, so input orientation is observationally equivalent).

    Raises
    ------
    ComputationError
        If the total edge weight is non-positive.
    """
    # n_total convention preserved from the previous symmetrized formulation:
    # undirected ⇒ 2 · Σw (the canonical "2m"); directed ⇒ Σw. Downstream
    # formulae are derived against this scale.
    total_w = float(df["weight"].sum())
    if total_w <= 0:
        raise ComputationError(
            "Noise-corrected backbone requires positive total edge weight",
            context={"operation": "noise_corrected"},
        )
    n_total = 2.0 * total_w if not directed else total_w

    # Per-endpoint strengths. Directed graphs distinguish out-strength (used
    # on the source side) from in-strength (used on the target side).
    # Undirected graphs use a single strength per node — sum of all incident
    # edge weights — computed without symmetrizing the edge frame by unioning
    # the two side-aggregates and re-grouping (a V-row, not 2E-row, op).
    if directed:
        u_strength = (
            df.group_by("source_id")
            .agg(pl.col("weight").sum().alias("s_u"))
        )
        v_strength = (
            df.group_by("target_id")
            .agg(pl.col("weight").sum().alias("s_v"))
        )
    else:
        src_part = (
            df.lazy()
            .group_by("source_id")
            .agg(pl.col("weight").sum().alias("s"))
            .rename({"source_id": "node"})
        )
        tgt_part = (
            df.lazy()
            .group_by("target_id")
            .agg(pl.col("weight").sum().alias("s"))
            .rename({"target_id": "node"})
        )
        node_strength = (
            pl.concat([src_part, tgt_part])
            .group_by("node")
            .agg(pl.col("s").sum().alias("strength"))
            .collect()
        )
        u_strength = node_strength.rename({"node": "source_id", "strength": "s_u"})
        v_strength = node_strength.rename({"node": "target_id", "strength": "s_v"})

    # Build the per-edge math as one lazy plan. The chained with_columns are
    # kept separate for readability; projection-pushdown in the optimizer
    # ensures only the columns referenced by the final select survive into
    # the collected frame, so the wide intermediate schema (alpha_prior,
    # beta_prior, expected_pij, …) doesn't pin memory.
    plan = (
        df.lazy()
        .join(u_strength.lazy(), on="source_id", how="left")
        .join(v_strength.lazy(), on="target_id", how="left")
        .with_columns([
            ((pl.col("s_u") * pl.col("s_v")) / (n_total * n_total))
                .alias("mean_prior_probability"),
            (n_total / (pl.col("s_u") * pl.col("s_v"))).alias("kappa"),
        ])
        .with_columns(
            ((pl.col("kappa") * pl.col("weight") - 1)
             / (pl.col("kappa") * pl.col("weight") + 1)).alias("score")
        )
        .with_columns(
            (
                (pl.col("s_u") * pl.col("s_v")
                 * (n_total - pl.col("s_u")) * (n_total - pl.col("s_v")))
                / ((n_total ** 4) * (n_total - 1))
            ).alias("var_prior_probability")
        )
        .with_columns([
            (
                (pl.col("mean_prior_probability") ** 2 / pl.col("var_prior_probability"))
                * (1 - pl.col("mean_prior_probability"))
                - pl.col("mean_prior_probability")
            ).alias("alpha_prior"),
            (
                (pl.col("mean_prior_probability") / pl.col("var_prior_probability"))
                * (1 - pl.col("mean_prior_probability") ** 2)
                - (1 - pl.col("mean_prior_probability"))
            ).alias("beta_prior"),
        ])
        .with_columns([
            (pl.col("alpha_prior") + pl.col("weight")).alias("alpha_post"),
            (n_total - pl.col("weight") + pl.col("beta_prior")).alias("beta_post"),
        ])
        .with_columns(
            (pl.col("alpha_post") / (pl.col("alpha_post") + pl.col("beta_post")))
                .alias("expected_pij")
        )
        .with_columns(
            (pl.col("expected_pij") * (1 - pl.col("expected_pij")) * n_total)
                .alias("variance_nij")
        )
        .with_columns(
            (
                (1.0 / (pl.col("s_u") * pl.col("s_v")))
                - (n_total * (pl.col("s_u") + pl.col("s_v"))
                   / ((pl.col("s_u") * pl.col("s_v")) ** 2))
            ).alias("d")
        )
        .with_columns(
            (
                pl.col("variance_nij")
                * ((2 * (pl.col("kappa") + pl.col("weight") * pl.col("d")))
                   / ((pl.col("kappa") * pl.col("weight") + 1) ** 2)) ** 2
            ).alias("variance_cij")
        )
        .with_columns(
            pl.col("variance_cij").clip(lower_bound=0.0).sqrt().alias("sdev_cij")
        )
        .with_columns(
            ((pl.col("score") - threshold * pl.col("sdev_cij")) > 0).alias("kept")
        )
        .select([
            "source_id",
            "target_id",
            "weight",
            "score",
            "sdev_cij",
            "kept",
        ])
    )

    scored = plan.collect(engine="streaming") if streaming else plan.collect()

    if top_k_override is not None:
        # Override path: rank by (score − threshold · sdev_cij) descending —
        # the same margin the default threshold filter uses, just sliced
        # at rank K instead of zero. Stable row_index breaks ties so the
        # output ordering is deterministic.
        margin = pl.col("score") - threshold * pl.col("sdev_cij")
        scored = scored.with_row_index(name="_idx")
        kept_idx = (
            scored
            .sort([margin.alias("__margin"), pl.col("_idx")],
                  descending=[True, False], nulls_last=True)
            .head(top_k_override)
            .select("_idx")
        )
        scored = (
            scored
            .with_columns(pl.col("_idx").is_in(kept_idx["_idx"]).alias("kept"))
            .drop("_idx")
        )

    return scored


def _apply_noise_corrected_filter(
    graph: nk.Graph,
    id_mapper: IDMapper,
    threshold: float,
    keep_disconnected: bool,
    return_filtered_edges: bool,
    *,
    streaming: bool = False,
    top_k_override: Optional[int] = None,
) -> Union[Tuple[nk.Graph, IDMapper], Tuple[nk.Graph, IDMapper, pl.DataFrame]]:
    """Graph-shim around :func:`_noise_corrected_on_edges`."""
    logger.debug("Applying noise-corrected backbone with threshold=%.3f", threshold)

    n_nodes = graph.numberOfNodes()
    directed = graph.isDirected()

    sources, targets, weights = _extract_edge_arrays(graph)

    if sources.size == 0:
        empty = nk.Graph(n_nodes, weighted=True, directed=directed)
        if return_filtered_edges:
            return empty, id_mapper, _empty_edges_df("noise_corrected")
        return empty, id_mapper

    edge_df = pl.DataFrame({
        "source_id": sources,
        "target_id": targets,
        "weight": weights,
    })
    scored = _noise_corrected_on_edges(
        edge_df, directed, threshold, streaming=streaming,
        top_k_override=top_k_override,
    )

    kept_df = scored.filter(pl.col("kept"))
    backbone_graph, updated_mapper = _assemble_backbone(
        n_nodes, directed, weighted=True,
        kept_sources=kept_df["source_id"].to_numpy(),
        kept_targets=kept_df["target_id"].to_numpy(),
        kept_weights=kept_df["weight"].to_numpy(),
        keep_disconnected=keep_disconnected, id_mapper=id_mapper,
    )

    if return_filtered_edges:
        return backbone_graph, updated_mapper, _translate_edges_to_originals(
            scored, id_mapper
        )

    return backbone_graph, updated_mapper


# ---------------------------------------------------------------------------
# Bipartite SVN filter (Tumminello et al. 2011)
# ---------------------------------------------------------------------------

def _bipartite_svn_on_edges(
    df: pl.DataFrame,
    alpha: float,
    correction: str,
    min_node_retention: Optional[float] = None,
    is_weighted_effective: bool = True,
    *,
    top_k_override: Optional[int] = None,
) -> Tuple[pl.DataFrame, Set[Any]]:
    """Tumminello et al. (2011) SVN filter on a bipartite edge frame.

    Parameters
    ----------
    df : pl.DataFrame
        Edge frame with columns ``source_id``, ``target_id``, ``weight``.
    alpha : float
        Per-edge significance level (subject to ``correction``). Ignored
        when ``top_k_override`` is set.
    correction : str
        One of ``"fdr_bh"``, ``"bonferroni"``, ``"none"``. Ignored when
        ``top_k_override`` is set (the override ranks by raw p-value).
    min_node_retention : float, optional
        Post-filter node-retention threshold in ``(0, 1]``. Nodes whose
        retention ratio falls below this are dropped; the corresponding
        ``kept`` entries are cleared, and the dropped-node IDs are returned
        as the second tuple element so the graph shim can remove them.
        Applied AFTER ``top_k_override`` when both are set.
    is_weighted_effective : bool, default True
        If False, treat all weights as 1 (unweighted Poisson tail).
    top_k_override : int, optional
        When set, ignore ``alpha`` and ``correction`` and keep exactly the
        ``top_k_override`` edges with the lowest raw p-value (most
        significant under the Poisson configuration null).

    Returns
    -------
    Tuple[pl.DataFrame, Set[Any]]
        - DataFrame with columns ``source_id``, ``target_id``, ``weight``,
          ``p_value``, ``kept``.
        - Set of node IDs flagged by ``min_node_retention`` (empty when the
          retention filter is not used). The graph shim uses this to drop
          those specific nodes even when ``keep_disconnected=True``.
    """
    # Strengths: each edge contributes to both endpoints (bipartite ⇒ undirected).
    incident = pl.concat([
        df.select(pl.col("source_id").alias("node"), pl.col("weight")),
        df.select(pl.col("target_id").alias("node"), pl.col("weight")),
    ])
    strengths = incident.group_by("node").agg(pl.col("weight").sum().alias("strength"))

    n_edges = df.height
    W_total = float(df["weight"].sum())
    if W_total <= 0:
        logger.warning("Bipartite SVN filter received zero total weight; nothing kept.")
        empty = df.with_columns([
            pl.lit(1.0, dtype=pl.Float64).alias("p_value"),
            pl.lit(False).alias("kept"),
        ]).select(["source_id", "target_id", "weight", "p_value", "kept"])
        return empty, set()

    work = (
        df.join(
            strengths.rename({"node": "source_id", "strength": "s_u"}),
            on="source_id", how="left",
        )
        .join(
            strengths.rename({"node": "target_id", "strength": "s_v"}),
            on="target_id", how="left",
        )
        .with_columns(((pl.col("s_u") * pl.col("s_v")) / W_total).alias("mu"))
    )

    # Poisson tail: scipy has no Polars equivalent; drop to numpy here only.
    mu_arr = work["mu"].to_numpy()
    w_arr = work["weight"].to_numpy()
    if is_weighted_effective:
        p_values = poisson.sf(np.ceil(w_arr) - 1, mu_arr)
    else:
        p_values = -np.expm1(-mu_arr)

    if top_k_override is not None:
        # Override path: rank by ascending p-value, keep top K. argpartition
        # avoids a full sort of the whole array — O(n) instead of O(n log n).
        k = min(top_k_override, n_edges)
        keep_mask = np.zeros(n_edges, dtype=bool)
        if k > 0:
            kept_idx = np.argpartition(p_values, k - 1)[:k]
            keep_mask[kept_idx] = True
    elif correction == "bonferroni":
        keep_mask = p_values <= (alpha / n_edges)
    elif correction == "fdr_bh":
        keep_mask = _benjamini_hochberg_mask(p_values, alpha)
    elif correction == "none":
        keep_mask = p_values <= alpha
    else:
        # Defensive: validated upstream.
        raise ValidationError(
            f"Invalid correction: {correction!r}. "
            f"Expected 'fdr_bh', 'bonferroni', or 'none'."
        )

    work = work.with_columns([
        pl.Series("p_value", p_values),
        pl.Series("kept", keep_mask),
    ])

    dropped_nodes: Set[Any] = set()
    if min_node_retention is not None:
        original_degree = incident.group_by("node").agg(
            pl.len().alias("original_degree")
        )
        surviving_incident = pl.concat([
            work.filter(pl.col("kept")).select(pl.col("source_id").alias("node")),
            work.filter(pl.col("kept")).select(pl.col("target_id").alias("node")),
        ])
        surviving_degree = surviving_incident.group_by("node").agg(
            pl.len().alias("surviving_degree")
        )
        retention = (
            original_degree
            .join(surviving_degree, on="node", how="left")
            .with_columns(pl.col("surviving_degree").fill_null(0))
            .with_columns(
                (pl.col("surviving_degree") / pl.col("original_degree")).alias("ratio")
            )
        )
        dropped_frame = retention.filter(pl.col("ratio") < min_node_retention)
        dropped_nodes = set(dropped_frame["node"].to_list())

        if dropped_nodes:
            work = work.with_columns(
                (
                    pl.col("kept")
                    & ~pl.col("source_id").is_in(list(dropped_nodes))
                    & ~pl.col("target_id").is_in(list(dropped_nodes))
                ).alias("kept")
            )

        logger.info(
            "Bipartite SVN filter: weighted=%s, alpha=%.3g, correction=%s, "
            "min_node_retention=%.2f — dropped %d nodes by retention; "
            "kept %d/%d edges (%.1f%%)",
            is_weighted_effective, alpha, correction, min_node_retention,
            len(dropped_nodes),
            int(work["kept"].sum()), n_edges,
            100.0 * int(work["kept"].sum()) / n_edges if n_edges else 0.0,
        )
    else:
        logger.info(
            "Bipartite SVN filter: weighted=%s, alpha=%.3g, correction=%s — "
            "kept %d/%d edges (%.1f%%)",
            is_weighted_effective, alpha, correction,
            int(work["kept"].sum()), n_edges,
            100.0 * int(work["kept"].sum()) / n_edges if n_edges else 0.0,
        )

    return (
        work.select(["source_id", "target_id", "weight", "p_value", "kept"]),
        dropped_nodes,
    )


def _apply_bipartite_svn_filter(
    graph: nk.Graph,
    id_mapper: IDMapper,
    alpha: float,
    correction: str,
    keep_disconnected: bool,
    min_node_retention: Optional[float],
    return_filtered_edges: bool,
    *,
    top_k_override: Optional[int] = None,
) -> Union[Tuple[nk.Graph, IDMapper], Tuple[nk.Graph, IDMapper, pl.DataFrame]]:
    """Graph-shim around :func:`_bipartite_svn_on_edges`.

    Builds the backbone with NetworkIt's ``removeNode`` (preserves internal
    IDs so bipartite partition info on the IDMapper stays consistent).
    """
    n_nodes = graph.numberOfNodes()
    if graph.numberOfEdges() == 0:
        if return_filtered_edges:
            return graph, id_mapper, _empty_edges_df("bipartite_svn")
        return graph, id_mapper

    u_arr, v_arr, w_arr = _extract_edge_arrays(graph)
    is_weighted_effective = graph.isWeighted() and not np.allclose(w_arr, 1.0)

    if w_arr.sum() <= 0:
        empty = _build_empty_like(graph)
        if return_filtered_edges:
            return empty, IDMapper(), _empty_edges_df("bipartite_svn")
        return empty, IDMapper()

    edge_df = pl.DataFrame({
        "source_id": u_arr,
        "target_id": v_arr,
        "weight": w_arr,
    })
    scored, dropped_internal_nodes = _bipartite_svn_on_edges(
        edge_df, alpha, correction, min_node_retention, is_weighted_effective,
        top_k_override=top_k_override,
    )

    kept_df = scored.filter(pl.col("kept"))
    kept_u = kept_df["source_id"].to_numpy()
    kept_v = kept_df["target_id"].to_numpy()
    kept_w = kept_df["weight"].to_numpy()

    # bipartite_svn historically uses the removeNode approach (preserves
    # internal IDs and bipartite partition info).
    backbone_graph = nk.Graph(
        n_nodes, directed=graph.isDirected(), weighted=graph.isWeighted()
    )
    for u, v, w in zip(kept_u, kept_v, kept_w):
        if graph.isWeighted():
            backbone_graph.addEdge(int(u), int(v), float(w))
        else:
            backbone_graph.addEdge(int(u), int(v))

    # Drop retention-flagged nodes explicitly (these may differ from
    # "no surviving edges" only when keep_disconnected=True).
    for node in dropped_internal_nodes:
        if backbone_graph.hasNode(int(node)):
            backbone_graph.removeNode(int(node))

    if not keep_disconnected:
        nodes_to_remove = [
            node for node in range(backbone_graph.numberOfNodes())
            if backbone_graph.hasNode(node) and backbone_graph.degree(node) == 0
        ]
        for node in nodes_to_remove:
            backbone_graph.removeNode(node)

    updated_mapper = IDMapper()
    surviving_originals: List[Any] = []
    for internal_id in range(graph.numberOfNodes()):
        if backbone_graph.hasNode(internal_id):
            try:
                original_id = id_mapper.get_original(internal_id)
            except KeyError:
                continue
            updated_mapper.add_mapping(original_id, internal_id)
            surviving_originals.append(original_id)

    if id_mapper.has_bipartite_partitions():
        survivors = set(surviving_originals)
        new_src = id_mapper.source_partition_originals & survivors
        new_tgt = id_mapper.target_partition_originals & survivors
        if new_src.isdisjoint(new_tgt):
            updated_mapper.set_bipartite_partitions(new_src, new_tgt)
        else:
            updated_mapper.source_partition_originals = new_src
            updated_mapper.target_partition_originals = new_tgt

    if return_filtered_edges:
        return backbone_graph, updated_mapper, _translate_edges_to_originals(
            scored, id_mapper
        )

    return backbone_graph, updated_mapper


def _benjamini_hochberg_mask(p_values: np.ndarray, alpha: float) -> np.ndarray:
    """Boolean mask: which p-values are rejected under Benjamini-Hochberg FDR."""
    n = p_values.shape[0]
    if n == 0:
        return np.zeros(0, dtype=bool)

    sorted_idx = np.argsort(p_values)
    sorted_p = p_values[sorted_idx]
    thresholds = np.arange(1, n + 1, dtype=np.float64) / n * alpha
    below = sorted_p <= thresholds

    if not below.any():
        return np.zeros(n, dtype=bool)

    last_reject_in_sorted = int(np.where(below)[0].max())
    reject_mask = np.zeros(n, dtype=bool)
    reject_mask[sorted_idx[: last_reject_in_sorted + 1]] = True
    return reject_mask


# ---------------------------------------------------------------------------
# Weight threshold
# ---------------------------------------------------------------------------

def _weight_threshold_on_edges(
    df: pl.DataFrame,
    target_edges: Optional[int] = None,
    weight_threshold: Optional[float] = None,
) -> pl.DataFrame:
    """Weight-threshold filter on an edge frame.

    Threshold selection (priority order):
      1. explicit ``weight_threshold`` if given,
      2. value at rank ``target_edges`` if given,
      3. median of all weights (fallback).

    Returns
    -------
    pl.DataFrame
        Columns: ``source_id``, ``target_id``, ``weight``, ``p_value`` (NaN),
        ``alpha_score`` (NaN), ``kept``.
    """
    weights = df["weight"].to_numpy()
    n_edges = weights.size

    if n_edges == 0:
        return df.with_columns([
            pl.lit(float("nan"), dtype=pl.Float64).alias("p_value"),
            pl.lit(float("nan"), dtype=pl.Float64).alias("alpha_score"),
            pl.lit(False).alias("kept"),
        ]).select(["source_id", "target_id", "weight", "p_value", "alpha_score", "kept"])

    if weight_threshold is not None:
        threshold = float(weight_threshold)
        keep_mask = weights >= threshold
        logger.debug("Weight threshold: %s (explicit)", threshold)
    elif target_edges is not None:
        # Exactly target_edges highest-weight edges; stable on ties.
        if target_edges >= n_edges:
            keep_mask = np.ones(n_edges, dtype=bool)
        else:
            sorted_idx = np.argsort(-weights, kind="stable")
            keep_mask = np.zeros(n_edges, dtype=bool)
            keep_mask[sorted_idx[:target_edges]] = True
        logger.debug("Weight threshold: top %d edges", target_edges)
    else:
        threshold = float(np.median(weights))
        keep_mask = weights >= threshold
        logger.debug("Weight threshold: %s (median fallback)", threshold)

    return df.with_columns([
        pl.lit(float("nan"), dtype=pl.Float64).alias("p_value"),
        pl.lit(float("nan"), dtype=pl.Float64).alias("alpha_score"),
        pl.Series("kept", keep_mask),
    ]).select(["source_id", "target_id", "weight", "p_value", "alpha_score", "kept"])


def _apply_weight_threshold(
    graph: nk.Graph,
    id_mapper: IDMapper,
    target_edges: Optional[int],
    weight_threshold: Optional[float],
    keep_disconnected: bool,
    return_filtered_edges: bool,
) -> Union[Tuple[nk.Graph, IDMapper], Tuple[nk.Graph, IDMapper, pl.DataFrame]]:
    """Graph-shim around :func:`_weight_threshold_on_edges`."""
    if not graph.isWeighted():
        logger.warning("Weight threshold applied to unweighted graph (all weights = 1.0)")

    n_nodes = graph.numberOfNodes()
    directed = graph.isDirected()

    sources, targets, weights = _extract_edge_arrays(graph)

    if sources.size == 0:
        empty = nk.Graph(n_nodes, weighted=True, directed=directed)
        if return_filtered_edges:
            return empty, id_mapper, _empty_edges_df("weight")
        return empty, id_mapper

    edge_df = pl.DataFrame({
        "source_id": sources,
        "target_id": targets,
        "weight": weights,
    })
    scored = _weight_threshold_on_edges(edge_df, target_edges, weight_threshold)

    kept_df = scored.filter(pl.col("kept"))
    backbone_graph, updated_mapper = _assemble_backbone(
        n_nodes, directed, weighted=True,
        kept_sources=kept_df["source_id"].to_numpy(),
        kept_targets=kept_df["target_id"].to_numpy(),
        kept_weights=kept_df["weight"].to_numpy(),
        keep_disconnected=keep_disconnected, id_mapper=id_mapper,
    )

    if return_filtered_edges:
        return backbone_graph, updated_mapper, _translate_edges_to_originals(
            scored, id_mapper
        )

    return backbone_graph, updated_mapper


# ---------------------------------------------------------------------------
# Degree threshold
# ---------------------------------------------------------------------------

def _degree_threshold_on_edges(
    df: pl.DataFrame,
    target_nodes: Optional[int] = None,
) -> Tuple[pl.DataFrame, pl.DataFrame]:
    """Degree-threshold filter on an edge frame.

    Treats degree as the unweighted in+out count (each edge contributes 1
    to both endpoints), matching the original graph-side semantics for both
    directed and undirected inputs.

    Threshold selection (priority order):
      1. value at rank ``target_nodes`` if given,
      2. median degree (fallback).

    Returns
    -------
    Tuple[pl.DataFrame, pl.DataFrame]
        - Edge frame with columns ``source_id``, ``target_id``, ``weight``,
          ``p_value`` (NaN), ``alpha_score`` (NaN), ``kept``.
        - Per-node frame with columns ``node``, ``degree``, ``kept_node``.
          The graph shim uses ``kept_node`` to preserve high-degree nodes
          that lost all their edges (``keep_disconnected=True`` semantics).
    """
    src = df.select(pl.col("source_id").alias("node"))
    tgt = df.select(pl.col("target_id").alias("node"))
    degrees = (
        pl.concat([src, tgt])
        .group_by("node")
        .agg(pl.len().alias("degree"))
    )

    deg_arr = degrees["degree"].to_numpy()
    n_nodes_in_edges = deg_arr.size

    if target_nodes is not None:
        if target_nodes >= n_nodes_in_edges:
            node_keep_mask = np.ones(n_nodes_in_edges, dtype=bool)
        else:
            sorted_idx = np.argsort(-deg_arr, kind="stable")
            node_keep_mask = np.zeros(n_nodes_in_edges, dtype=bool)
            node_keep_mask[sorted_idx[:target_nodes]] = True
        logger.debug("Degree threshold: top %d nodes", target_nodes)
    else:
        threshold = int(np.median(deg_arr)) if n_nodes_in_edges else 0
        node_keep_mask = deg_arr >= threshold
        logger.debug("Degree threshold: %d (median fallback)", threshold)

    node_df = degrees.with_columns(pl.Series("kept_node", node_keep_mask))
    kept_node_ids = node_df.filter(pl.col("kept_node"))["node"].to_list()

    edge_out = df.with_columns([
        pl.lit(float("nan"), dtype=pl.Float64).alias("p_value"),
        pl.lit(float("nan"), dtype=pl.Float64).alias("alpha_score"),
        (
            pl.col("source_id").is_in(kept_node_ids)
            & pl.col("target_id").is_in(kept_node_ids)
        ).alias("kept"),
    ]).select(["source_id", "target_id", "weight", "p_value", "alpha_score", "kept"])

    return edge_out, node_df


def _apply_degree_threshold(
    graph: nk.Graph,
    id_mapper: IDMapper,
    target_nodes: Optional[int],
    keep_disconnected: bool,
    return_filtered_edges: bool,
) -> Union[Tuple[nk.Graph, IDMapper], Tuple[nk.Graph, IDMapper, pl.DataFrame]]:
    """Graph-shim around :func:`_degree_threshold_on_edges`."""
    n_nodes = graph.numberOfNodes()
    directed = graph.isDirected()
    weighted = graph.isWeighted()

    sources, targets, weights = _extract_edge_arrays(graph)
    edge_df = pl.DataFrame({
        "source_id": sources,
        "target_id": targets,
        "weight": weights,
    })
    scored, node_df = _degree_threshold_on_edges(edge_df, target_nodes)

    kept_df = scored.filter(pl.col("kept"))
    kept_sources = kept_df["source_id"].to_numpy()
    kept_targets = kept_df["target_id"].to_numpy()
    kept_weights = kept_df["weight"].to_numpy()

    # `keep_disconnected` here controls whether high-degree nodes that
    # lost all their edges are preserved (True) or dropped (False).
    if keep_disconnected:
        node_survives = np.zeros(n_nodes, dtype=bool)
        if node_df.height > 0:
            kept_nodes = node_df.filter(pl.col("kept_node"))["node"].to_numpy().astype(np.int64)
            node_survives[kept_nodes] = True
    else:
        node_survives = np.zeros(n_nodes, dtype=bool)
        if kept_sources.size > 0:
            node_survives[kept_sources] = True
            node_survives[kept_targets] = True

    surviving_node_ids = np.where(node_survives)[0]
    node_mapping = {int(old): new for new, old in enumerate(surviving_node_ids)}

    backbone_graph = nk.Graph(
        len(surviving_node_ids), weighted=weighted, directed=directed
    )
    for u, v, w in zip(kept_sources, kept_targets, kept_weights):
        if weighted:
            backbone_graph.addEdge(node_mapping[int(u)], node_mapping[int(v)], float(w))
        else:
            backbone_graph.addEdge(node_mapping[int(u)], node_mapping[int(v)])

    updated_mapper = IDMapper()
    for old_internal, new_internal in node_mapping.items():
        try:
            updated_mapper.add_mapping(id_mapper.get_original(old_internal), new_internal)
        except KeyError:
            pass

    if return_filtered_edges:
        return backbone_graph, updated_mapper, _translate_edges_to_originals(
            scored, id_mapper
        )

    return backbone_graph, updated_mapper


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _assemble_backbone(
    n_nodes: int,
    directed: bool,
    weighted: bool,
    kept_sources: np.ndarray,
    kept_targets: np.ndarray,
    kept_weights: np.ndarray,
    keep_disconnected: bool,
    id_mapper: IDMapper,
) -> Tuple[nk.Graph, IDMapper]:
    """Construct the backbone graph and updated id mapper.

    If ``keep_disconnected`` is True, the backbone graph has the same node
    set as the original. Otherwise, isolated nodes are dropped and the
    surviving nodes are renumbered to a consecutive 0..N-1 internal-ID
    space (the new id_mapper reflects this renumbering).
    """
    if keep_disconnected:
        backbone_graph = nk.Graph(n_nodes, weighted=weighted, directed=directed)
        for u, v, w in zip(kept_sources, kept_targets, kept_weights):
            if weighted:
                backbone_graph.addEdge(int(u), int(v), float(w))
            else:
                backbone_graph.addEdge(int(u), int(v))
        return backbone_graph, id_mapper

    if kept_sources.size == 0:
        return nk.Graph(0, weighted=weighted, directed=directed), IDMapper()

    connected = sorted({int(x) for x in kept_sources} | {int(x) for x in kept_targets})
    node_mapping = {old: new for new, old in enumerate(connected)}

    backbone_graph = nk.Graph(len(connected), weighted=weighted, directed=directed)
    for u, v, w in zip(kept_sources, kept_targets, kept_weights):
        if weighted:
            backbone_graph.addEdge(node_mapping[int(u)], node_mapping[int(v)], float(w))
        else:
            backbone_graph.addEdge(node_mapping[int(u)], node_mapping[int(v)])

    updated_mapper = IDMapper()
    for old_internal, new_internal in node_mapping.items():
        updated_mapper.add_mapping(id_mapper.get_original(old_internal), new_internal)

    return backbone_graph, updated_mapper


def _build_empty_like(graph: nk.Graph) -> nk.Graph:
    """Create an empty graph with the same directed/weighted flags."""
    return nk.Graph(0, weighted=graph.isWeighted(), directed=graph.isDirected())


def _translate_edges_to_originals(
    df: pl.DataFrame, id_mapper: IDMapper
) -> pl.DataFrame:
    """Replace the ``source_id`` and ``target_id`` columns (assumed to hold
    NetworkIt internal integer IDs) with their original-ID counterparts.

    Used by the graph-input path of :func:`apply_backbone` when the caller
    asks for a per-edge results frame: the math is done on internal IDs but
    the surfaced frame must speak the user's ID space.
    """
    if df.height == 0:
        return df
    src_originals = id_mapper.get_original_batch(df["source_id"].to_list())
    tgt_originals = id_mapper.get_original_batch(df["target_id"].to_list())
    return df.with_columns([
        pl.Series("source_id", src_originals),
        pl.Series("target_id", tgt_originals),
    ])


def _slim_edges_df(df: pl.DataFrame) -> pl.DataFrame:
    """Trim a per-edge results frame to the lean
    (``source_id``, ``target_id``, ``weight``) shape, keeping only kept rows.

    Used to honour ``include_scores=False`` (the default) on the
    DataFrame-output paths of :func:`apply_backbone`, so callers can chain
    into the next pipeline stage without carrying score columns or the
    ``kept`` boolean. The dropped columns become garbage-collectable on the
    next reference, which is the point of trimming here rather than asking
    callers to do it after the fact.
    """
    if "kept" in df.columns:
        df = df.filter(pl.col("kept"))
    keep_cols = [c for c in ("source_id", "target_id", "weight") if c in df.columns]
    return df.select(keep_cols)


def _empty_edges_df(method: str) -> pl.DataFrame:
    """Return an empty edge-results DataFrame with the correct schema."""
    base = {"source_id": [], "target_id": [], "weight": [], "kept": []}
    if method == "disparity" or method in ("weight", "degree"):
        base["p_value"] = []
        base["alpha_score"] = []
    elif method == "noise_corrected":
        base["score"] = []
        base["sdev_cij"] = []
    elif method == "bipartite_svn":
        base["p_value"] = []
    return pl.DataFrame(base)


def _print_backbone_summary(
    verbose: bool,
    t_start: float,
    original_graph: nk.Graph,
    backbone_graph: nk.Graph,
    method: str,
) -> None:
    """One-line summary printed at the bottom of apply_backbone.

    Mirrors :func:`guidedLP.network.construction._print_backbone_summary`
    (duplicated to avoid the circular import).
    """
    if not verbose:
        return
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


# ---------------------------------------------------------------------------
# Summary statistics
# ---------------------------------------------------------------------------

def get_backbone_statistics(
    original_graph: nk.Graph,
    backbone_graph: nk.Graph,
    filtered_edges: Optional[pl.DataFrame] = None,
) -> Dict[str, Any]:
    """
    Compute summary statistics comparing original and backbone graphs.

    Parameters
    ----------
    original_graph : nk.Graph
        Graph before backbone extraction.
    backbone_graph : nk.Graph
        Graph after backbone extraction.
    filtered_edges : pl.DataFrame, optional
        Per-edge filtering details returned by ``apply_backbone`` when
        ``return_filtered_edges=True``. When provided, adds weight and (for
        the disparity filter) alpha-score statistics under
        ``weight_statistics`` and ``alpha_statistics`` keys.

    Returns
    -------
    Dict[str, Any]
        Always contains: ``original_nodes``, ``original_edges``,
        ``backbone_nodes``, ``backbone_edges``, ``node_retention``,
        ``edge_retention``, ``compression_ratio``, ``original_density``,
        ``backbone_density``, ``density_ratio``,
        ``node_retention_rate``, ``edge_retention_rate``.
    """
    n0 = original_graph.numberOfNodes()
    e0 = original_graph.numberOfEdges()
    n1 = backbone_graph.numberOfNodes()
    e1 = backbone_graph.numberOfEdges()

    node_retention = n1 / max(n0, 1)
    edge_retention = e1 / max(e0, 1)

    stats: Dict[str, Any] = {
        "original_nodes": n0,
        "original_edges": e0,
        "backbone_nodes": n1,
        "backbone_edges": e1,
        "node_retention": node_retention,
        "edge_retention": edge_retention,
        # Aliases used by some callers / tests:
        "node_retention_rate": node_retention,
        "edge_retention_rate": edge_retention,
        "compression_ratio": (n0 + e0) / max(n1 + e1, 1),
    }

    original_density = e0 / max(n0 * (n0 - 1) / 2, 1)
    backbone_density = e1 / max(n1 * (n1 - 1) / 2, 1)
    stats["original_density"] = original_density
    stats["backbone_density"] = backbone_density
    stats["density_ratio"] = backbone_density / max(original_density, 1e-10)

    if filtered_edges is not None:
        kept = filtered_edges.filter(pl.col("kept"))
        if kept.height > 0:
            stats["weight_statistics"] = {
                "kept_weight_mean": float(kept["weight"].mean()),
                "kept_weight_std": float(kept["weight"].std()),
                "kept_weight_min": float(kept["weight"].min()),
                "kept_weight_max": float(kept["weight"].max()),
            }
            if "alpha_score" in kept.columns:
                alpha_scores = kept["alpha_score"].drop_nulls()
                if len(alpha_scores) > 0:
                    stats["alpha_statistics"] = {
                        "alpha_mean": float(alpha_scores.mean()),
                        "alpha_std": float(alpha_scores.std()),
                        "alpha_min": float(alpha_scores.min()),
                        "alpha_max": float(alpha_scores.max()),
                    }

    return stats


# Backwards-compatible alias for the older `get_backbone_summary` name.
get_backbone_summary = get_backbone_statistics
