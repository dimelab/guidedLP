"""
Guided Label Propagation implementation.

This module implements the core guided label propagation algorithm for semi-supervised
network analysis. It uses matrix-based calculations with sparse operations for efficiency
and supports both directed and undirected graphs.

Mathematical Foundation:
- Initialize label matrix Y where Y[i,j] = 1 if node i has label j (seed), 0 otherwise
- Create transition matrix P = D^-1 A (row-normalized adjacency matrix)
- Iteratively update: F^(t+1) = α P F^(t) + (1-α) Y
- Continue until convergence: max|F^(t+1) - F^(t)| < threshold

For directed graphs, run propagation twice:
- Out-degree: Use A as-is (influence propagation)
- In-degree: Use A^T (receptivity propagation)
"""

from typing import Callable, Dict, List, Tuple, Union, Optional, Any
import math
import warnings
import numpy as np
import scipy.sparse as sp
import polars as pl
import networkit as nk

from guidedLP.common.id_mapper import IDMapper
from guidedLP.common.exceptions import (
    ValidationError,
    ConfigurationError,
    ConvergenceError,
    ComputationError,
    validate_parameter,
    require_positive,
    check_convergence
)
from guidedLP.common.logging_config import get_logger, LoggingTimer
from guidedLP.common.seed_input import SeedInput, normalize_seed_input

logger = get_logger(__name__)


# Edge-weight transforms ----------------------------------------------------
# A WeightTransform maps a raw edge weight to a propagation weight. It is
# applied at transition-matrix build time, not graph-construction time, so the
# same graph can be propagated under multiple transforms without rebuilding.
#
# Transforms must satisfy ``f(w) > 0`` for ``w > 0`` — the zero-degree
# handling in ``_create_transition_matrix`` treats a zero row-sum as an
# isolated node, so a transform that maps positive weights to zero will
# silently disconnect parts of the graph.

WeightTransform = Callable[[float], float]


def tanh_transform(scale: float = 100.0, offset: float = 1.0) -> WeightTransform:
    """
    Tanh-based weight compression (the historical ``stlp`` transform).

    Returns ``f(w) = tanh((w + offset) / scale)``. Saturates as ``w`` grows
    large, so a single very heavy edge cannot dominate its neighborhood.

    Parameters
    ----------
    scale : float, default 100.0
        Larger values delay saturation; smaller values compress harder.
    offset : float, default 1.0
        Shifts the input so small weights still produce nonzero output.

    Returns
    -------
    WeightTransform
    """
    return lambda w: math.tanh((w + offset) / scale)


def log1p_transform() -> WeightTransform:
    """
    Logarithmic compression: ``f(w) = log(1 + w)``.

    Gentler than tanh — no upper bound, but high weights still grow much
    more slowly than linear. Good first choice when edge weights span
    several orders of magnitude.
    """
    return math.log1p


def winsorize_transform(cap: float) -> WeightTransform:
    """
    Hard cap: ``f(w) = min(w, cap)``.

    Cheapest outlier control. Linear up to ``cap``, then flat. Useful when
    you want to neutralize a known number of anomalous high-weight edges
    without changing the rest of the weight distribution.
    """
    if cap <= 0:
        raise ConfigurationError(
            f"winsorize_transform requires cap > 0, got {cap}",
            parameter="cap", value=cap,
        )
    return lambda w: min(w, cap)


def guided_label_propagation(
    graph: nk.Graph,
    id_mapper: IDMapper,
    seed_labels: SeedInput,
    labels: List[str],
    alpha: float = 0.85,
    max_iterations: int = 100,
    convergence_threshold: float = 1e-6,
    normalize: bool = True,
    directional: bool = True,
    n_jobs: int = 1,
    enable_noise_category: bool = False,
    noise_ratio: float = 0.1,
    confidence_threshold: float = 0.0,
    seed_node_col: str = "node_id",
    seed_label_col: str = "label",
    weight_transform: Optional[WeightTransform] = None,
    random_seed: Optional[int] = 42,
) -> Union[pl.DataFrame, Tuple[pl.DataFrame, pl.DataFrame]]:
    """
    Propagate labels from seed nodes through network using guided label propagation.
    
    This function implements the guided label propagation algorithm using efficient
    sparse matrix operations. For directed graphs, it can compute both out-degree
    (influence) and in-degree (receptivity) propagation.
    
    Mathematical Algorithm:
    1. Initialize label matrix Y (n × k) where n=nodes, k=labels
    2. Create transition matrix P = D^-1 A (row-normalized adjacency)
    3. Iteratively update: F^(t+1) = α P F^(t) + (1-α) Y
    4. Check convergence: max|F^(t+1) - F^(t)| < threshold
    5. Return final probabilities with original node IDs
    
    Time Complexity: O(k × i × E) where k=labels, i=iterations, E=edges
    Space Complexity: O(n × k) for label probability matrix
    
    Parameters
    ----------
    graph : nk.Graph
        NetworkIt graph (directed or undirected, must be weighted)
    id_mapper : IDMapper
        Bidirectional mapping between original and internal node IDs
    seed_labels : SeedInput
        Known seed labels in any of four supported shapes (see
        :func:`guidedLP.common.normalize_seed_input`):

        - ``Dict[Any, str]`` — node_id → label, e.g.
          ``{"user_123": "left", "user_456": "right"}``
        - ``Dict[str, List[Any]]`` — label → list of node_ids, e.g.
          ``{"left": ["user_123"], "right": ["user_456"]}``
        - polars.DataFrame with node_id and label columns
        - pandas.DataFrame with node_id and label columns
    labels : List[str]
        Complete list of possible labels (must include all values in seed_labels)
    alpha : float, default 0.85
        Propagation coefficient (0-1). Higher values emphasize neighbor influence,
        lower values preserve seed labels more strongly
    max_iterations : int, default 100
        Maximum number of propagation iterations
    convergence_threshold : float, default 1e-6
        Stop when maximum probability change between iterations < threshold
    normalize : bool, default True
        Normalize final probabilities to sum to 1.0 per node
    directional : bool, default True
        For directed graphs, compute both out-degree and in-degree propagation
    n_jobs : int, default 1
        Number of parallel jobs (reserved for future multi-label parallelization)
    enable_noise_category : bool, default True
        Automatically add a "noise" category for nodes with weak label associations.
        Helps identify outlier nodes and improves classification confidence.
    noise_ratio : float, default 0.1
        Fraction of non-seed nodes to randomly assign as noise seeds (0.0-1.0).
        Only used when enable_noise_category=True.
    confidence_threshold : float, default 0.0
        Minimum probability threshold for classification. Nodes with max probability
        below this threshold are classified as "uncertain" (0.0-1.0).
    seed_node_col : str, default "node_id"
        Column name for node IDs when ``seed_labels`` is a DataFrame.
    seed_label_col : str, default "label"
        Column name for labels when ``seed_labels`` is a DataFrame.
    weight_transform : Optional[WeightTransform], default None
        Callable applied to each raw edge weight when the transition matrix
        is built. Use :func:`tanh_transform`, :func:`log1p_transform`, or
        :func:`winsorize_transform` to compress heavy-tailed weight
        distributions. Must satisfy ``f(w) > 0`` for ``w > 0`` — otherwise
        nodes whose edges all map to zero become treated as isolates.
    random_seed : Optional[int], default 42
        Seed for the local RNG used when generating noise seeds
        (``enable_noise_category=True``). Distinct values produce
        independent noise samples — :func:`ensemble_label_propagation`
        relies on this. Pass ``None`` to use the global :mod:`random`
        state. No effect when ``enable_noise_category=False``.

    Returns
    -------
    Union[pl.DataFrame, Tuple[pl.DataFrame, pl.DataFrame]]
        If directional=False OR graph is undirected:
            Single DataFrame with columns:
            - node_id: Original node ID
            - {label}_prob: Probability for each label (float)
            - dominant_label: Label with highest probability (str)
            - confidence: Maximum probability value (float)
            - is_seed: Whether node was in seed set (bool)
        
        If directional=True AND graph is directed:
            Tuple of (out_degree_df, in_degree_df) with same schema
    
    Raises
    ------
    ValidationError
        If seed_labels contains labels not in `labels` list, or if seeds
        reference nodes not in the graph
    ConfigurationError
        If alpha not in [0,1] or other parameter validation fails
    ConvergenceError
        If algorithm fails to converge within max_iterations
    ComputationError
        If matrix operations fail (e.g., due to numerical issues)
    
    Examples
    --------
    >>> # Basic usage
    >>> seeds = {"node1": "A", "node2": "B", "node3": "A"}
    >>> result = guided_label_propagation(graph, mapper, seeds, ["A", "B"])
    
    >>> # Directed graph with both propagation directions
    >>> out_result, in_result = guided_label_propagation(
    ...     directed_graph, mapper, seeds, ["A", "B"], directional=True
    ... )
    
    >>> # Lower alpha for stronger seed influence
    >>> result = guided_label_propagation(
    ...     graph, mapper, seeds, ["A", "B"], alpha=0.5
    ... )
    
    Notes
    -----
    - Seeds must be present in the graph (original IDs must map to internal IDs)
    - For disconnected components without seeds, nodes get uniform probability
    - Zero-degree nodes (isolates) retain their initial state
    - For directed graphs, out-degree measures influence, in-degree measures receptivity
    - Algorithm uses sparse matrix operations for memory efficiency on large graphs
    """
    
    # Normalize the seed input shape to the canonical Dict[node, label] before
    # any other processing — everything downstream assumes this shape.
    seed_labels = normalize_seed_input(seed_labels, seed_node_col, seed_label_col)

    logger.info(f"Starting guided label propagation with {len(seed_labels)} seeds, "
               f"{len(labels)} labels, alpha={alpha}")

    # Validate inputs
    _validate_inputs(graph, id_mapper, seed_labels, labels, alpha,
                    max_iterations, convergence_threshold, enable_noise_category,
                    noise_ratio, confidence_threshold)
    
    # Process labels and seeds with noise category support
    processed_labels, processed_seed_labels = _process_noise_category(
        graph, id_mapper, seed_labels, labels, enable_noise_category, noise_ratio,
        random_seed=random_seed,
    )
    
    # Check if graph is directed
    is_directed = graph.isDirected()
    logger.info(f"Graph type: {'directed' if is_directed else 'undirected'}, "
               f"nodes={graph.numberOfNodes()}, edges={graph.numberOfEdges()}")
    
    # For undirected graphs or directional=False, run single propagation
    if not is_directed or not directional:
        result = _run_single_propagation(
            graph, id_mapper, processed_seed_labels, processed_labels, alpha,
            max_iterations, convergence_threshold, normalize,
            direction="undirected" if not is_directed else "out_degree",
            weight_transform=weight_transform,
        )
        
        # Apply confidence thresholding if enabled
        if confidence_threshold > 0.0:
            result = _apply_confidence_threshold(result, confidence_threshold)
        
        logger.info("Completed single propagation")
        return result
    
    # For directed graphs with directional=True, run both directions
    else:
        logger.info("Running directional propagation for directed graph")
        
        # Out-degree propagation (influence)
        out_result = _run_single_propagation(
            graph, id_mapper, processed_seed_labels, processed_labels, alpha,
            max_iterations, convergence_threshold, normalize,
            direction="out_degree",
            weight_transform=weight_transform,
        )

        # In-degree propagation (receptivity) - use transposed adjacency
        in_result = _run_single_propagation(
            graph, id_mapper, processed_seed_labels, processed_labels, alpha,
            max_iterations, convergence_threshold, normalize,
            direction="in_degree",
            weight_transform=weight_transform,
        )
        
        # Apply confidence thresholding if enabled
        if confidence_threshold > 0.0:
            out_result = _apply_confidence_threshold(out_result, confidence_threshold)
            in_result = _apply_confidence_threshold(in_result, confidence_threshold)
        
        logger.info("Completed directional propagation")
        return out_result, in_result


def audience_composition_pass(
    graph: nk.Graph,
    id_mapper: IDMapper,
    forward_result: pl.DataFrame,
    labels: List[str],
    weight_transform: Optional[WeightTransform] = None,
    normalize: bool = True,
) -> pl.DataFrame:
    """
    Compute audience composition by one-hop aggregation of converged forward labels.

    For each node ``n``, computes the in-degree-weighted average of the
    forward-pass label probabilities of nodes pointing at ``n``:

    .. math::

        \\text{audience}[n, j] = \\frac{1}{\\deg_{in}(n)}
            \\sum_{m: m \\to n} w(m, n) \\cdot F_\\text{forward}[m, j]

    Semantically, this answers "what is the (forward-pass) label profile of
    nodes that point at ``n``?" — i.e., a measurement of audience or
    reception, given that the forward pass has labeled each node by its own
    outgoing structure.

    This is **distinct** from running :func:`guided_label_propagation` with
    ``directional=True``: the latter's in-degree pass is a full backward
    random walk starting from the *original* seeds, answering "is ``n``
    upstream of a seed?" The audience pass instead consumes the converged
    forward labels at every node and aggregates them one hop along incoming
    edges, without iteration.

    This recovers the semantics of the legacy ``stlp`` reverse-direction
    pass in ``guidedLP_OLD/network/net_utils.py``. See
    ``docs/architecture/glp.md`` for the broader semantic distinction.

    Time complexity: ``O(E + n·k)`` — single sparse matmul.
    Space complexity: ``O(n·k + E)``.

    Parameters
    ----------
    graph : nk.Graph
        Directed graph. Must be the same graph used to compute
        ``forward_result``.
    id_mapper : IDMapper
        Bidirectional mapping for the graph.
    forward_result : pl.DataFrame
        Output of :func:`guided_label_propagation` (single DataFrame, not the
        tuple form). Must contain columns ``node_id`` and one ``{label}_prob``
        per entry in ``labels``. The ``is_seed`` column, if present, is
        carried through to the output.
    labels : List[str]
        Labels to compute audience for. Each must have a corresponding
        ``{label}_prob`` column in ``forward_result``. To include the noise
        column, pass ``"noise"`` explicitly in this list.
    weight_transform : Optional[WeightTransform], default None
        Same edge-weight transform as for the forward pass. Pass the same
        callable that was used in :func:`guided_label_propagation` so the
        two passes are consistent.
    normalize : bool, default True
        Row-normalize the result so each node's probabilities sum to 1.
        Disable to read raw aggregated mass.

    Returns
    -------
    pl.DataFrame
        Same schema as :func:`guided_label_propagation` output:
        ``node_id``, ``{label}_prob`` columns, ``dominant_label``,
        ``confidence``, ``is_seed``. The ``is_seed`` column is carried
        through from ``forward_result`` (audience pass has no seeds of its
        own — this column identifies forward-pass seeds).

    Raises
    ------
    ValidationError
        If ``graph`` is undirected, ``forward_result`` is missing required
        columns, or ``forward_result.node_id`` references nodes not in
        ``id_mapper``.

    Examples
    --------
    >>> forward = guided_label_propagation(
    ...     graph, mapper, seeds, ["left", "right"], directional=False
    ... )
    >>> audience = audience_composition_pass(
    ...     graph, mapper, forward, ["left", "right"]
    ... )
    >>> # audience["left_prob"] now reads as: "fraction of in-neighbors that
    >>> # the forward pass labeled left"

    Notes
    -----
    - Nodes with zero in-degree receive a uniform distribution after
      normalization (no in-neighbors to aggregate from).
    - The historical ``stlp`` reverse pass also skipped iteration and
      disabled distance decay; this implementation matches that behavior
      exactly (one matmul, no ``α``, no ``Y`` term).
    """
    if not graph.isDirected():
        raise ValidationError(
            "audience_composition_pass requires a directed graph. For undirected "
            "graphs, the in-neighbor aggregation is symmetric with the forward "
            "pass and produces no additional information."
        )

    required_cols = {"node_id"} | {f"{label}_prob" for label in labels}
    missing = required_cols - set(forward_result.columns)
    if missing:
        raise ValidationError(
            f"forward_result missing required columns: {sorted(missing)}",
            details={"missing_columns": sorted(missing), "labels": labels},
        )

    n_nodes = graph.numberOfNodes()
    n_labels = len(labels)
    label_cols = [f"{label}_prob" for label in labels]

    with LoggingTimer("Audience composition pass"):
        # Reconstruct F_forward (n × k) indexed by internal node IDs
        F_forward = np.zeros((n_nodes, n_labels), dtype=np.float64)
        fwd_ids = forward_result["node_id"].to_list()
        fwd_probs = forward_result.select(label_cols).to_numpy()
        internal_ids = id_mapper.get_internal_batch(fwd_ids)
        F_forward[internal_ids] = fwd_probs

        # Build A^T (rows = receivers, cols = senders): for each original edge
        # u -> v, place weight at row v, col u.
        rows: List[int] = []
        cols: List[int] = []
        weights: List[float] = []
        for u, v in graph.iterEdges():
            w = graph.weight(u, v)
            if weight_transform is not None:
                w = weight_transform(w)
            rows.append(v)
            cols.append(u)
            weights.append(w)

        A_T = sp.coo_matrix(
            (weights, (rows, cols)),
            shape=(n_nodes, n_nodes),
            dtype=np.float64,
        ).tocsr()

        # Row-normalize by in-degree (mass per row of A^T = total incoming weight)
        in_deg = np.array(A_T.sum(axis=1)).flatten()
        zero_mask = (in_deg == 0)
        in_deg[zero_mask] = 1.0  # avoid div-by-zero; rows are all-zero anyway
        in_deg_inv = 1.0 / in_deg
        in_deg_inv[zero_mask] = 0.0
        A_T_norm = sp.diags(in_deg_inv).dot(A_T)

        # Single-hop aggregation. No iteration, no Y term, no alpha.
        F_audience = A_T_norm.dot(F_forward)

        # Build a Y-like indicator for is_seed pass-through. _create_results_dataframe
        # reads seed_mask = (Y.sum(axis=1) > 0), so we just need any nonzero entry
        # per forward-seed row.
        Y_passthrough = np.zeros_like(F_audience)
        if "is_seed" in forward_result.columns:
            seed_mask_input = forward_result["is_seed"].to_numpy()
            seed_internals = [
                internal_ids[i] for i, was_seed in enumerate(seed_mask_input)
                if was_seed
            ]
            if seed_internals:
                Y_passthrough[seed_internals, 0] = 1.0

        return _create_results_dataframe(
            F_audience, Y_passthrough, labels, id_mapper,
            normalize=normalize, converged_iteration=1, direction="audience",
        )


def ensemble_label_propagation(
    graph: nk.Graph,
    id_mapper: IDMapper,
    seed_labels: SeedInput,
    labels: List[str],
    n_epochs: int = 10,
    base_seed: int = 42,
    return_variance: bool = False,
    **glp_kwargs: Any,
) -> Union[pl.DataFrame, Tuple[pl.DataFrame, pl.DataFrame]]:
    """
    Average ``n_epochs`` independent GLP runs with resampled noise seeds.

    Repeatedly calls :func:`guided_label_propagation` with a per-epoch
    ``random_seed = base_seed + epoch_index`` and averages the resulting
    probability matrices using Welford's online algorithm. This recovers the
    bagging-style variance reduction from the legacy ``stlp`` implementation's
    ``epochs`` loop, where each epoch sampled fresh noise seeds.

    Only meaningful when ``enable_noise_category=True`` — otherwise each
    epoch produces an identical deterministic result and ensembling reduces
    to a single run. A warning is emitted in that case and a single GLP run
    is returned instead.

    Time complexity: ``O(n_epochs · cost(guided_label_propagation))``.
    Space complexity: ``O(n · k)`` — running mean/variance only, no
    per-epoch matrices retained.

    Parameters
    ----------
    graph, id_mapper, seed_labels, labels
        Forwarded to :func:`guided_label_propagation`. See that function's
        docstring for details.
    n_epochs : int, default 10
        Number of independent runs to average. Must be ``>= 2``.
    base_seed : int, default 42
        Each epoch ``i`` uses ``random_seed = base_seed + i``. Together with
        ``n_epochs`` this makes the ensemble fully deterministic.
    return_variance : bool, default False
        If True, the returned DataFrame(s) include ``{label}_prob_std``
        columns alongside the means (sample standard deviation across
        epochs, with Bessel's correction).
    **glp_kwargs
        Any other :func:`guided_label_propagation` keyword arguments. Any
        ``random_seed`` value is overridden per epoch.

    Returns
    -------
    Union[pl.DataFrame, Tuple[pl.DataFrame, pl.DataFrame]]
        Same shape as :func:`guided_label_propagation` — single DataFrame
        for undirected/``directional=False``, tuple for directed +
        ``directional=True``. ``{label}_prob`` columns hold averaged
        probabilities; ``dominant_label`` and ``confidence`` are recomputed
        from the averaged probabilities (not voted across epochs).

    Raises
    ------
    ConfigurationError
        If ``n_epochs < 2``.

    Examples
    --------
    >>> # 20 noise-resampled runs, with per-label std for confidence intervals
    >>> result = ensemble_label_propagation(
    ...     graph, mapper, seeds, ["left", "right"],
    ...     n_epochs=20, enable_noise_category=True, return_variance=True,
    ... )

    Notes
    -----
    - The label set passed in does not need to include ``"noise"``; when
      ``enable_noise_category=True`` the noise column is auto-added and
      averaged like any other label.
    - Probability averaging (not vote averaging) is what gives the bagging
      variance reduction. Recomputing ``dominant_label`` from averaged
      probabilities means a node can have a different dominant label in the
      ensemble than in any single epoch — this is correct, not a bug.
    - Epochs run serially. Parallelization is straightforward (each epoch
      is independent and reads ``graph`` read-only) but is not yet wired;
      see ``_ensemble_run_epoch`` for the natural ``ProcessPoolExecutor``
      entry point.
    """
    if n_epochs < 2:
        raise ConfigurationError(
            f"n_epochs must be >= 2 for ensembling, got {n_epochs}",
            parameter="n_epochs",
            value=n_epochs,
        )

    if not glp_kwargs.get("enable_noise_category", False):
        warnings.warn(
            "ensemble_label_propagation with enable_noise_category=False "
            "produces n_epochs identical runs (no noise resampling to ensemble "
            "over). Returning a single GLP run instead. Pass "
            "enable_noise_category=True to get the bagging effect.",
            UserWarning,
            stacklevel=2,
        )
        glp_kwargs.pop("random_seed", None)
        return guided_label_propagation(
            graph, id_mapper, seed_labels, labels, **glp_kwargs
        )

    # random_seed is set per epoch; drop any user-supplied value.
    glp_kwargs.pop("random_seed", None)

    # Normalize seed_labels once so is_seed in the output reflects the
    # user-supplied seeds only (noise seeds vary across epochs and would
    # otherwise leak epoch-0's random sample into the ensemble result).
    seed_node_col = glp_kwargs.get("seed_node_col", "node_id")
    seed_label_col = glp_kwargs.get("seed_label_col", "label")
    original_seed_ids = set(
        normalize_seed_input(seed_labels, seed_node_col, seed_label_col).keys()
    )

    logger.info(
        f"Starting ensemble propagation: n_epochs={n_epochs}, base_seed={base_seed}"
    )

    # Buffers populated on first epoch (we discover result shape from it).
    mean_out: Optional[np.ndarray] = None
    mean_in: Optional[np.ndarray] = None
    m2_out: Optional[np.ndarray] = None
    m2_in: Optional[np.ndarray] = None
    template_out: Optional[pl.DataFrame] = None
    template_in: Optional[pl.DataFrame] = None
    label_cols: Optional[List[str]] = None
    all_labels: Optional[List[str]] = None

    def _welford_update(
        mean: np.ndarray,
        m2: Optional[np.ndarray],
        new: np.ndarray,
        count: int,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """One step of Welford's online mean/variance update. ``count`` is 1-based."""
        delta = new - mean
        mean = mean + delta / count
        if m2 is not None:
            delta2 = new - mean
            m2 = m2 + delta * delta2
        return mean, m2

    for epoch in range(n_epochs):
        epoch_seed = base_seed + epoch
        logger.debug(f"Ensemble epoch {epoch + 1}/{n_epochs} (random_seed={epoch_seed})")

        result = guided_label_propagation(
            graph, id_mapper, seed_labels, labels,
            random_seed=epoch_seed, **glp_kwargs,
        )
        is_tuple = isinstance(result, tuple)
        out_df = result[0] if is_tuple else result
        in_df = result[1] if is_tuple else None

        if epoch == 0:
            # Discover the label column set from the first result. When
            # enable_noise_category is on, the result has an extra noise_prob
            # column that we need to average too.
            label_cols = [c for c in out_df.columns if c.endswith("_prob")]
            all_labels = [c[: -len("_prob")] for c in label_cols]
            template_out = out_df
            mean_out = out_df.select(label_cols).to_numpy().copy()
            m2_out = np.zeros_like(mean_out) if return_variance else None
            if is_tuple:
                template_in = in_df
                mean_in = in_df.select(label_cols).to_numpy().copy()
                m2_in = np.zeros_like(mean_in) if return_variance else None
            continue

        new_out = out_df.select(label_cols).to_numpy()
        mean_out, m2_out = _welford_update(mean_out, m2_out, new_out, epoch + 1)
        if is_tuple:
            new_in = in_df.select(label_cols).to_numpy()
            mean_in, m2_in = _welford_update(mean_in, m2_in, new_in, epoch + 1)

    def _assemble(
        template: pl.DataFrame,
        mean: np.ndarray,
        m2: Optional[np.ndarray],
    ) -> pl.DataFrame:
        dom_idx = np.argmax(mean, axis=1)
        dom_labels = [all_labels[i] for i in dom_idx]
        conf = mean.max(axis=1)

        cols: Dict[str, Any] = {"node_id": template["node_id"]}
        for i, col_name in enumerate(label_cols):
            cols[col_name] = mean[:, i]
        cols["dominant_label"] = dom_labels
        cols["confidence"] = conf
        # is_seed reflects user-supplied seeds only, not per-epoch noise seeds.
        cols["is_seed"] = [nid in original_seed_ids for nid in template["node_id"].to_list()]

        if m2 is not None:
            # Sample variance with Bessel's correction (n_epochs >= 2 enforced above).
            variance = m2 / (n_epochs - 1)
            std = np.sqrt(np.maximum(variance, 0.0))  # clip tiny negatives from FP
            for i, col_name in enumerate(label_cols):
                cols[f"{col_name}_std"] = std[:, i]

        return pl.DataFrame(cols)

    logger.info(f"Completed ensemble: averaged {n_epochs} epochs")

    if template_in is not None:
        return (
            _assemble(template_out, mean_out, m2_out),
            _assemble(template_in, mean_in, m2_in),
        )
    return _assemble(template_out, mean_out, m2_out)


def _validate_inputs(
    graph: nk.Graph,
    id_mapper: IDMapper,
    seed_labels: Dict[Any, str],
    labels: List[str],
    alpha: float,
    max_iterations: int,
    convergence_threshold: float,
    enable_noise_category: bool,
    noise_ratio: float,
    confidence_threshold: float
) -> None:
    """Validate all input parameters."""
    
    # Validate graph
    if graph.numberOfNodes() == 0:
        raise ValidationError("Graph has no nodes")
    
    if graph.numberOfEdges() == 0:
        warnings.warn("Graph has no edges - propagation will only affect seed nodes")
    
    # Validate alpha
    if not 0 <= alpha <= 1:
        raise ConfigurationError(
            f"Alpha must be between 0 and 1, got {alpha}",
            parameter="alpha",
            value=alpha
        )
    
    # Validate iteration parameters
    require_positive(max_iterations, "max_iterations")
    require_positive(convergence_threshold, "convergence_threshold")
    
    # Validate labels
    if not labels:
        raise ValidationError("Labels list cannot be empty")
    
    if len(set(labels)) != len(labels):
        raise ValidationError("Labels list contains duplicates")
    
    # Warn about single label scenarios
    if len(labels) == 1 and not enable_noise_category:
        warnings.warn(
            "GLP with single label provides limited discriminative power. "
            "Consider enabling noise category or adding additional labels.",
            UserWarning
        )
    
    # Validate noise category parameters
    if enable_noise_category:
        if not 0.0 <= noise_ratio <= 1.0:
            raise ConfigurationError(
                f"Noise ratio must be between 0.0 and 1.0, got {noise_ratio}",
                parameter="noise_ratio",
                value=noise_ratio
            )
    
    # Validate confidence threshold
    if not 0.0 <= confidence_threshold <= 1.0:
        raise ConfigurationError(
            f"Confidence threshold must be between 0.0 and 1.0, got {confidence_threshold}",
            parameter="confidence_threshold",
            value=confidence_threshold
        )
    
    # Validate seed_labels
    if not seed_labels:
        raise ValidationError("Seed labels dictionary cannot be empty")
    
    # Check that all seed labels are in the labels list
    seed_label_values = set(seed_labels.values())
    unknown_labels = seed_label_values - set(labels)
    if unknown_labels:
        raise ValidationError(
            f"Seed labels contain unknown labels: {unknown_labels}",
            details={"unknown_labels": list(unknown_labels), "valid_labels": labels}
        )
    
    # Check that all seed nodes exist in the graph (via ID mapper)
    missing_seeds = []
    for seed_id in seed_labels.keys():
        if not id_mapper.has_original(seed_id):
            missing_seeds.append(seed_id)
    
    if missing_seeds:
        raise ValidationError(
            f"Seed nodes not found in graph: {missing_seeds[:5]}{'...' if len(missing_seeds) > 5 else ''}",
            details={"missing_seeds_count": len(missing_seeds), "missing_seeds_sample": missing_seeds[:10]}
        )


def _run_single_propagation(
    graph: nk.Graph,
    id_mapper: IDMapper,
    seed_labels: Dict[Any, str],
    labels: List[str],
    alpha: float,
    max_iterations: int,
    convergence_threshold: float,
    normalize: bool,
    direction: str,
    weight_transform: Optional[WeightTransform] = None,
) -> pl.DataFrame:
    """Run a single propagation (either undirected, out-degree, or in-degree)."""

    with LoggingTimer(f"Running {direction} propagation"):

        # Setup matrices and initial conditions
        n_nodes = graph.numberOfNodes()
        n_labels = len(labels)

        logger.debug(f"Matrix dimensions: {n_nodes} nodes × {n_labels} labels")

        # Create initial label matrix Y (n × k)
        Y = _initialize_label_matrix(graph, id_mapper, seed_labels, labels)

        # Create transition matrix P
        P = _create_transition_matrix(graph, direction, weight_transform=weight_transform)
        
        # Initialize propagation matrix
        F = Y.copy()
        
        # Iterative propagation
        converged_iteration = _iterative_propagation(
            F, P, Y, alpha, max_iterations, convergence_threshold
        )
        
        # Post-process results
        result_df = _create_results_dataframe(
            F, Y, labels, id_mapper, normalize, converged_iteration, direction
        )
        
        logger.info(f"Propagation converged after {converged_iteration} iterations")
        return result_df


def _initialize_label_matrix(
    graph: nk.Graph,
    id_mapper: IDMapper,
    seed_labels: Dict[Any, str],
    labels: List[str]
) -> np.ndarray:
    """
    Create initial label matrix Y from seed labels.
    
    This function creates the initial n×k label matrix Y where:
    - Y[i,j] = 1.0 if node i is a seed with label j
    - Y[i,j] = 0.0 otherwise
    
    Parameters
    ----------
    graph : nk.Graph
        NetworkIt graph to get node count
    id_mapper : IDMapper
        Mapping between original and internal node IDs
    seed_labels : Dict[Any, str]
        Mapping from original node IDs to their labels
    labels : List[str]
        Complete list of possible labels
    
    Returns
    -------
    np.ndarray
        Initial label matrix Y of shape (n_nodes, n_labels)
    
    Examples
    --------
    >>> seeds = {"node_1": "A", "node_3": "B"}
    >>> labels = ["A", "B"]
    >>> Y = _initialize_label_matrix(graph, mapper, seeds, labels)
    >>> Y.shape
    (n_nodes, 2)
    """
    n_nodes = graph.numberOfNodes()
    n_labels = len(labels)
    label_to_idx = {label: i for i, label in enumerate(labels)}
    
    Y = np.zeros((n_nodes, n_labels), dtype=np.float64)
    
    # Set seed nodes
    for original_id, label in seed_labels.items():
        internal_id = id_mapper.get_internal(original_id)
        label_idx = label_to_idx[label]
        Y[internal_id, label_idx] = 1.0
    
    logger.debug(f"Initialized {len(seed_labels)} seed nodes in label matrix")
    return Y


def _create_transition_matrix(
    graph: nk.Graph,
    direction: str,
    weight_transform: Optional[WeightTransform] = None,
) -> sp.csr_matrix:
    """Create row-normalized transition matrix P = D^-1 A."""

    with LoggingTimer("Creating transition matrix"):

        n_nodes = graph.numberOfNodes()

        # Build adjacency matrix in COO format for efficiency
        row_indices = []
        col_indices = []
        edge_weights = []

        for u, v in graph.iterEdges():
            weight = graph.weight(u, v)
            if weight_transform is not None:
                weight = weight_transform(weight)

            if direction == "in_degree":
                # For in-degree: transpose the adjacency matrix (v -> u)
                row_indices.append(v)
                col_indices.append(u)
                edge_weights.append(weight)

                # Add reverse edge for undirected graphs
                if not graph.isDirected():
                    row_indices.append(u)
                    col_indices.append(v)
                    edge_weights.append(weight)

            else:  # out_degree or undirected
                # Standard adjacency matrix (u -> v)
                row_indices.append(u)
                col_indices.append(v)
                edge_weights.append(weight)

                # Add reverse edge for undirected graphs
                if not graph.isDirected():
                    row_indices.append(v)
                    col_indices.append(u)
                    edge_weights.append(weight)
        
        # Create sparse adjacency matrix
        adj_matrix = sp.coo_matrix(
            (edge_weights, (row_indices, col_indices)),
            shape=(n_nodes, n_nodes),
            dtype=np.float64
        ).tocsr()
        
        # Create row-normalized transition matrix P = D^-1 A
        # Calculate row sums (degrees)
        row_sums = np.array(adj_matrix.sum(axis=1)).flatten()
        
        # Handle zero-degree nodes
        zero_degree_mask = (row_sums == 0)
        if zero_degree_mask.any():
            logger.debug(f"Found {zero_degree_mask.sum()} zero-degree nodes")
            # For zero-degree nodes, use zero rows (no propagation)
            row_sums[zero_degree_mask] = 1.0  # Avoid division by zero
        
        # Create diagonal matrix for normalization
        row_sums_inv = 1.0 / row_sums
        row_sums_inv[zero_degree_mask] = 0.0  # Reset zero-degree nodes
        
        # Normalize: P = D^-1 A
        P = sp.diags(row_sums_inv).dot(adj_matrix)
        
        logger.debug(f"Transition matrix shape: {P.shape}, nnz: {P.nnz}")
        return P


def _propagate_iteration(
    F: np.ndarray,
    P: sp.csr_matrix,
    Y: np.ndarray,
    alpha: float
) -> np.ndarray:
    """
    Perform a single propagation iteration step.
    
    This function implements the core propagation update:
    F^(t+1) = α P F^(t) + (1-α) Y
    
    Parameters
    ----------
    F : np.ndarray
        Current label probability matrix (n_nodes × n_labels)
    P : sp.csr_matrix
        Row-normalized transition matrix (n_nodes × n_nodes)
    Y : np.ndarray
        Initial seed label matrix (n_nodes × n_labels)
    alpha : float
        Propagation coefficient (0-1)
    
    Returns
    -------
    np.ndarray
        Updated label probability matrix F^(t+1)
    
    Raises
    ------
    ComputationError
        If matrix operations fail due to numerical issues
    
    Examples
    --------
    >>> F_new = _propagate_iteration(F_old, P, Y, alpha=0.85)
    >>> F_new.shape == F_old.shape
    True
    
    Notes
    -----
    This function performs the mathematical core of label propagation:
    - P @ F: propagates current probabilities through network connections
    - α controls balance between neighbor influence vs. seed retention
    - (1-α) Y ensures seeds maintain their original labels
    """
    try:
        # Matrix multiplication: P @ F (propagate through network)
        propagated = P.dot(F)
        
        # Update: F = α * propagated + (1-α) * Y
        F_new = alpha * propagated + (1 - alpha) * Y
        
        return F_new
        
    except Exception as e:
        raise ComputationError(
            f"Matrix operation failed during propagation iteration",
            operation="matrix_multiplication",
            error_type="numerical",
            cause=e
        )


def _check_convergence(
    F_new: np.ndarray,
    F_old: np.ndarray,
    threshold: float
) -> Tuple[bool, float]:
    """
    Check if propagation has converged.
    
    Convergence is determined by checking if the maximum absolute change
    in any probability value is below the specified threshold.
    
    Parameters
    ----------
    F_new : np.ndarray
        New label probability matrix
    F_old : np.ndarray
        Previous label probability matrix
    threshold : float
        Convergence threshold
    
    Returns
    -------
    Tuple[bool, float]
        (has_converged, max_change) where:
        - has_converged: True if max_change < threshold
        - max_change: Maximum absolute change between F_new and F_old
    
    Examples
    --------
    >>> converged, change = _check_convergence(F_new, F_old, 1e-6)
    >>> if converged:
    ...     print(f"Converged with change {change}")
    
    Notes
    -----
    Uses the L∞ norm (maximum absolute difference) as the convergence criterion.
    This is appropriate for probability matrices where we want all values to
    stabilize within the threshold.
    """
    max_change = np.max(np.abs(F_new - F_old))
    has_converged = max_change < threshold
    
    return has_converged, max_change


def _create_results_dataframe(
    F: np.ndarray,
    Y: np.ndarray,
    labels: List[str],
    id_mapper: IDMapper,
    normalize: bool,
    converged_iteration: int,
    direction: str
) -> pl.DataFrame:
    """
    Convert propagation results to Polars DataFrame with original node IDs.
    
    This function post-processes the final probability matrix and creates
    a structured DataFrame with all necessary output columns.
    
    Parameters
    ----------
    F : np.ndarray
        Final label probability matrix (n_nodes × n_labels)
    Y : np.ndarray
        Initial seed label matrix (for seed identification)
    labels : List[str]
        List of label names
    id_mapper : IDMapper
        Mapping between original and internal node IDs
    normalize : bool
        Whether to normalize probabilities to sum to 1.0
    converged_iteration : int
        Number of iterations until convergence
    direction : str
        Propagation direction ("undirected", "out_degree", "in_degree")
    
    Returns
    -------
    pl.DataFrame
        Structured results with columns:
        - node_id: Original node ID
        - {label}_prob: Probability for each label
        - dominant_label: Label with highest probability
        - confidence: Maximum probability value
        - is_seed: Whether node was in seed set
    
    Examples
    --------
    >>> df = _create_results_dataframe(F, Y, ["A", "B"], mapper, True, 15, "undirected")
    >>> print(df.columns)
    ['node_id', 'A_prob', 'B_prob', 'dominant_label', 'confidence', 'is_seed']
    
    Notes
    -----
    - Handles probability normalization if requested
    - Identifies seed nodes from initial matrix Y
    - Maps internal node IDs back to original IDs
    - Calculates dominant label and confidence scores
    """
    with LoggingTimer("Post-processing results"):
        
        n_nodes, n_labels = F.shape
        
        # Normalize probabilities if requested
        if normalize:
            row_sums = F.sum(axis=1, keepdims=True)
            # Handle rows with zero probabilities
            zero_mask = (row_sums.flatten() == 0)
            if zero_mask.any():
                # For nodes with no probabilities, set uniform distribution
                uniform_prob = 1.0 / n_labels
                F[zero_mask, :] = uniform_prob
                row_sums = F.sum(axis=1, keepdims=True)
            
            F = F / row_sums
        
        # Calculate dominant label and confidence
        dominant_indices = np.argmax(F, axis=1)
        dominant_labels = [labels[idx] for idx in dominant_indices]
        confidence_scores = np.max(F, axis=1)
        
        # Determine which nodes were seeds
        seed_mask = (Y.sum(axis=1) > 0)
        
        # Create result data
        result_data = {}
        
        # Original node IDs
        original_ids = id_mapper.get_original_batch(list(range(n_nodes)))
        result_data["node_id"] = original_ids
        
        # Probability columns for each label
        for i, label in enumerate(labels):
            result_data[f"{label}_prob"] = F[:, i].tolist()
        
        # Additional columns
        result_data["dominant_label"] = dominant_labels
        result_data["confidence"] = confidence_scores.tolist()
        result_data["is_seed"] = seed_mask.tolist()
        
        # Create DataFrame
        df = pl.DataFrame(result_data)
        
        logger.info(f"Created result DataFrame with {len(df)} nodes")
        logger.debug(f"Label distribution: {df['dominant_label'].value_counts().to_dict()}")
        
        return df


def _iterative_propagation(
    F: np.ndarray,
    P: sp.csr_matrix,
    Y: np.ndarray,
    alpha: float,
    max_iterations: int,
    convergence_threshold: float
) -> int:
    """
    Perform iterative label propagation until convergence.
    
    This function runs the core iterative propagation loop using the helper
    functions for individual operations. It continues until convergence or
    maximum iterations are reached.
    
    Parameters
    ----------
    F : np.ndarray
        Initial label probability matrix (modified in-place)
    P : sp.csr_matrix
        Row-normalized transition matrix
    Y : np.ndarray
        Initial seed label matrix
    alpha : float
        Propagation coefficient
    max_iterations : int
        Maximum number of iterations
    convergence_threshold : float
        Convergence threshold for stopping
    
    Returns
    -------
    int
        Number of iterations until convergence
    
    Raises
    ------
    ConvergenceError
        If algorithm fails to converge within max_iterations
    """
    with LoggingTimer("Iterative propagation"):
        
        for iteration in range(max_iterations):
            F_prev = F.copy()
            
            # Perform single propagation iteration
            F_new = _propagate_iteration(F, P, Y, alpha)
            F[:] = F_new  # Update in-place
            
            # Check convergence
            has_converged, max_change = _check_convergence(F, F_prev, convergence_threshold)
            
            if iteration % 10 == 0 or has_converged:
                logger.debug(f"Iteration {iteration}: max_change = {max_change:.2e}")
            
            if has_converged:
                return iteration + 1
        
        # If we reach here, algorithm didn't converge
        final_change = max_change  # From last iteration
        
        # Check if result might still be usable
        if final_change <= 10 * convergence_threshold:
            warnings.warn(
                f"Propagation did not fully converge (final_change={final_change:.2e}, "
                f"threshold={convergence_threshold:.2e}) but result may be usable"
            )
            return max_iterations
        
        raise ConvergenceError(
            "Label propagation failed to converge",
            algorithm="guided_label_propagation",
            iterations=max_iterations,
            max_iterations=max_iterations,
            final_change=final_change,
            threshold=convergence_threshold
        )



def get_propagation_info(
    graph: nk.Graph,
    seed_labels: SeedInput,
    labels: List[str],
    seed_node_col: str = "node_id",
    seed_label_col: str = "label",
) -> Dict[str, Any]:
    """
    Get information about a potential propagation run without executing it.

    This utility function provides estimates and diagnostics for a guided label
    propagation run, helping users understand the computational requirements
    and potential issues before running the full algorithm.

    Parameters
    ----------
    graph : nk.Graph
        NetworkIt graph to analyze
    seed_labels : SeedInput
        Seed labels in any of four supported shapes (see
        :func:`guidedLP.common.normalize_seed_input`).
    labels : List[str]
        List of all possible labels
    seed_node_col : str, default "node_id"
        Column name for node IDs when ``seed_labels`` is a DataFrame.
    seed_label_col : str, default "label"
        Column name for labels when ``seed_labels`` is a DataFrame.
    
    Returns
    -------
    Dict[str, Any]
        Information dictionary containing:
        - graph_stats: Basic graph statistics
        - seed_stats: Seed set analysis
        - memory_estimate: Estimated memory requirements
        - computational_estimate: Expected computational complexity
        - potential_issues: List of potential problems
    
    Examples
    --------
    >>> info = get_propagation_info(graph, seeds, ["A", "B"])
    >>> print(f"Estimated memory: {info['memory_estimate']['total_mb']:.1f} MB")
    >>> if info['potential_issues']:
    ...     print("Warnings:", info['potential_issues'])
    """
    seed_labels = normalize_seed_input(seed_labels, seed_node_col, seed_label_col)

    n_nodes = graph.numberOfNodes()
    n_edges = graph.numberOfEdges()
    n_labels = len(labels)
    n_seeds = len(seed_labels)
    
    # Basic graph statistics
    graph_stats = {
        "nodes": n_nodes,
        "edges": n_edges,
        "labels": n_labels,
        "seeds": n_seeds,
        "is_directed": graph.isDirected(),
        "density": n_edges / (n_nodes * (n_nodes - 1) / 2) if n_nodes > 1 else 0,
        "seed_ratio": n_seeds / n_nodes if n_nodes > 0 else 0
    }
    
    # Seed set analysis
    seed_label_counts = {}
    for label in labels:
        seed_label_counts[label] = sum(1 for l in seed_labels.values() if l == label)
    
    seed_stats = {
        "seeds_per_label": seed_label_counts,
        "min_seeds_per_label": min(seed_label_counts.values()) if seed_label_counts else 0,
        "max_seeds_per_label": max(seed_label_counts.values()) if seed_label_counts else 0,
        "label_balance": min(seed_label_counts.values()) / max(seed_label_counts.values()) 
                       if seed_label_counts and max(seed_label_counts.values()) > 0 else 0
    }
    
    # Memory estimates (in MB)
    float64_size = 8  # bytes
    label_matrix_mb = (n_nodes * n_labels * float64_size) / (1024 * 1024)
    adjacency_matrix_mb = (n_edges * 3 * float64_size) / (1024 * 1024)  # COO format
    
    memory_estimate = {
        "label_matrix_mb": label_matrix_mb,
        "adjacency_matrix_mb": adjacency_matrix_mb,
        "total_mb": label_matrix_mb * 2 + adjacency_matrix_mb,  # F, Y, and P matrices
        "large_network": label_matrix_mb > 100  # Flag for networks > 100MB
    }
    
    # Computational estimates
    estimated_iterations = min(50, max(10, int(np.log10(n_nodes) * 10)))  # Rough heuristic
    computational_estimate = {
        "estimated_iterations": estimated_iterations,
        "ops_per_iteration": n_labels * n_edges,
        "total_ops_estimate": estimated_iterations * n_labels * n_edges,
        "expected_runtime_class": "fast" if n_edges < 10000 else "medium" if n_edges < 100000 else "slow"
    }
    
    # Potential issues
    potential_issues = []
    
    if n_seeds == 0:
        potential_issues.append("No seed nodes provided")
    
    if seed_stats["min_seeds_per_label"] == 0:
        missing_labels = [l for l, count in seed_label_counts.items() if count == 0]
        potential_issues.append(f"No seeds for labels: {missing_labels}")
    
    if seed_stats["label_balance"] < 0.1:
        potential_issues.append("Highly imbalanced seed set - consider balancing")
    
    if n_edges == 0:
        potential_issues.append("Graph has no edges - propagation will be limited")
    
    if memory_estimate["total_mb"] > 1000:
        potential_issues.append("High memory usage expected (>1GB)")
    
    if graph_stats["density"] < 0.001 and n_nodes > 1000:
        potential_issues.append("Very sparse graph - may have convergence issues")
    
    return {
        "graph_stats": graph_stats,
        "seed_stats": seed_stats,
        "memory_estimate": memory_estimate,
        "computational_estimate": computational_estimate,
        "potential_issues": potential_issues
    }


def _process_noise_category(
    graph: nk.Graph,
    id_mapper: IDMapper,
    seed_labels: Dict[Any, str],
    labels: List[str],
    enable_noise_category: bool,
    noise_ratio: float,
    random_seed: Optional[int] = 42,
) -> Tuple[List[str], Dict[Any, str]]:
    """
    Process labels and seed labels to include noise category if enabled.
    
    This function implements the noise category functionality from reference 
    implementations, automatically adding a "noise" category and generating
    noise seeds to improve classification robustness.
    
    Parameters
    ----------
    graph : nk.Graph
        NetworkIt graph
    id_mapper : IDMapper
        ID mapper for the graph
    seed_labels : Dict[Any, str]
        Original seed labels mapping
    labels : List[str]
        Original labels list
    enable_noise_category : bool
        Whether to add noise category
    noise_ratio : float
        Fraction of non-seed nodes to use as noise seeds
    
    Returns
    -------
    processed_labels : List[str]
        Labels list with noise category added if enabled
    processed_seed_labels : Dict[Any, str]
        Seed labels with noise seeds added if enabled
    
    Notes
    -----
    Noise seeds are randomly sampled from non-seed nodes to provide
    the algorithm with examples of nodes that don't belong to any
    specific category. This improves classification confidence and
    helps identify outlier nodes.
    """
    import random
    
    processed_labels = labels.copy()
    processed_seed_labels = seed_labels.copy()
    
    if not enable_noise_category:
        return processed_labels, processed_seed_labels
    
    # Add noise category if not already present
    if "noise" not in processed_labels:
        processed_labels.append("noise")
        logger.info("Added 'noise' category to labels")
    
    # Generate noise seeds if noise category was added
    if "noise" not in seed_labels.values():
        noise_seeds = _generate_noise_seeds(
            graph, id_mapper, seed_labels, noise_ratio, random_seed=random_seed
        )
        processed_seed_labels.update(noise_seeds)
        logger.info(f"Generated {len(noise_seeds)} noise seeds")
    
    return processed_labels, processed_seed_labels


def _generate_noise_seeds(
    graph: nk.Graph,
    id_mapper: IDMapper,
    seed_labels: Dict[Any, str],
    noise_ratio: float,
    random_seed: Optional[int] = 42,
) -> Dict[Any, str]:
    """
    Generate noise seeds from non-seed nodes.

    Randomly samples nodes that are not already in the seed set
    and assigns them the "noise" label. This provides the algorithm
    with examples of nodes that don't strongly belong to any category.

    Parameters
    ----------
    graph : nk.Graph
        NetworkIt graph
    id_mapper : IDMapper
        ID mapper for the graph
    seed_labels : Dict[Any, str]
        Existing seed labels
    noise_ratio : float
        Fraction of non-seed nodes to use as noise seeds
    random_seed : Optional[int], default 42
        Seed for the local RNG used to sample noise nodes. Use distinct values
        per call to draw independent noise samples (e.g., from
        :func:`ensemble_label_propagation`). Pass ``None`` to draw from the
        global :mod:`random` state.

    Returns
    -------
    Dict[Any, str]
        Mapping of selected nodes to "noise" label
    """
    import random

    # Get all nodes that are not already seeds
    all_nodes = set(range(graph.numberOfNodes()))
    seed_node_internals = {id_mapper.get_internal(seed_id) for seed_id in seed_labels.keys()}
    non_seed_nodes = all_nodes - seed_node_internals

    if not non_seed_nodes:
        logger.warning("No non-seed nodes available for noise seed generation")
        return {}

    # Calculate number of noise seeds
    n_existing_seeds = len(seed_labels)
    n_noise_seeds = max(1, int(noise_ratio * n_existing_seeds))
    n_noise_seeds = min(n_noise_seeds, len(non_seed_nodes))

    # Local RNG so concurrent callers (ensembling) don't fight over global state
    rng = random.Random(random_seed) if random_seed is not None else random
    selected_internal_ids = rng.sample(list(non_seed_nodes), n_noise_seeds)

    # Convert back to original IDs
    noise_seeds = {}
    for internal_id in selected_internal_ids:
        original_id = id_mapper.get_original(internal_id)
        noise_seeds[original_id] = "noise"

    logger.debug(f"Selected {len(noise_seeds)} noise seeds from {len(non_seed_nodes)} candidates")
    return noise_seeds


def _apply_confidence_threshold(
    result_df: pl.DataFrame,
    confidence_threshold: float
) -> pl.DataFrame:
    """
    Apply confidence thresholding to classification results.
    
    Nodes with maximum probability below the threshold are reclassified
    as "uncertain", providing a mechanism to identify low-confidence
    predictions that should not be trusted.
    
    Parameters
    ----------
    result_df : pl.DataFrame
        Results DataFrame from GLP
    confidence_threshold : float
        Minimum confidence threshold (0.0-1.0)
    
    Returns
    -------
    pl.DataFrame
        Results with uncertain classifications applied
    """
    if confidence_threshold <= 0.0:
        return result_df
    
    # Identify low-confidence nodes
    low_confidence_mask = result_df["confidence"] < confidence_threshold
    n_uncertain = low_confidence_mask.sum()
    
    if n_uncertain > 0:
        # Update dominant label for low-confidence nodes
        result_df = result_df.with_columns([
            pl.when(pl.col("confidence") < confidence_threshold)
            .then(pl.lit("uncertain"))
            .otherwise(pl.col("dominant_label"))
            .alias("dominant_label")
        ])
        
        logger.info(f"Classified {n_uncertain} nodes as 'uncertain' due to low confidence")
    
    return result_df