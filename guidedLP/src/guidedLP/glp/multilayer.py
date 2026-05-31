"""
Multilayer Guided Label Propagation.

Standalone module implementing supra-adjacency multilayer GLP. The caller
supplies several graphs over (a subset of) the same node set; this module
builds a supra-adjacency matrix from the per-layer adjacencies with
intra-layer importance weights ``gammas[ℓ]`` and inter-layer same-node
coupling ``coupling``, runs the standard GLP update on it, then aggregates
each node's layer-copies into a single per-node label distribution.

Designed to be standalone and easily removable: delete this file and the
two import lines in ``glp/__init__.py`` and the module is gone, with no
other code in the package touched.

Mathematical foundation:
- Per-layer block ``A_ℓ`` is the layer's (undirected) adjacency lifted to
  the union node set and scaled by ``gammas[ℓ]``.
- Inter-layer coupling adds an edge of weight ``coupling`` between every
  pair of layer-copies of the same node, i.e.
  ``coupling · ((J_LxL − I_L) ⊗ I_N_union)``.
- The supra-adjacency is the sum of the block-diagonal layer matrix and
  the coupling term. GLP then runs on its row-normalized transition.

See ``docs/plans/multilayer_glp_implementation.md`` for the full spec and
the rationale for the locked-in design choices.
"""

import random
from typing import Any, Callable, Dict, List, Literal, Optional, Sequence, Set, Tuple, Union

import networkit as nk
import numpy as np
import polars as pl
import scipy.sparse as sp

from guidedLP.common.exceptions import ValidationError
from guidedLP.common.id_mapper import IDMapper
from guidedLP.common.logging_config import LoggingTimer, get_logger
from guidedLP.common.seed_input import SeedInput, normalize_seed_input

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Phase 1: layer alignment, union IDMapper, and input validation.
# These are private helpers used by the public functions added in later
# phases (kernel construction, propagation, aggregation, and the public
# ``multilayer_guided_label_propagation`` entry point).
# ---------------------------------------------------------------------------


def _validate_layers_input(layers: Any) -> None:
    """
    Validate that ``layers`` is a non-empty list of ``(nk.Graph, IDMapper)``
    tuples with all graphs undirected and consistent with their mappers.

    Parameters
    ----------
    layers : Any
        Caller-supplied object expected to be a list (or tuple) of
        ``(nk.Graph, IDMapper)`` 2-tuples.

    Raises
    ------
    ValidationError
        - ``layers`` is not a list/tuple.
        - ``layers`` is empty.
        - Any element is not a 2-tuple of the expected types.
        - Any graph is directed (v1 supports undirected only).
        - Any graph's node count disagrees with its IDMapper size.
    """
    if not isinstance(layers, (list, tuple)):
        raise ValidationError(
            "layers must be a list of (nk.Graph, IDMapper) tuples",
            field="layers",
            expected="list of (nk.Graph, IDMapper) tuples",
            value=type(layers).__name__,
        )

    if len(layers) == 0:
        raise ValidationError(
            "layers must contain at least one (nk.Graph, IDMapper) tuple",
            field="layers",
            expected="non-empty list",
        )

    for i, layer in enumerate(layers):
        if not isinstance(layer, tuple) or len(layer) != 2:
            raise ValidationError(
                f"layers[{i}] must be a 2-tuple of (nk.Graph, IDMapper)",
                field=f"layers[{i}]",
                expected="2-tuple of (nk.Graph, IDMapper)",
                value=type(layer).__name__,
            )

        graph, mapper = layer

        if not isinstance(graph, nk.Graph):
            raise ValidationError(
                f"layers[{i}][0] must be an nk.Graph, got {type(graph).__name__}",
                field=f"layers[{i}][0]",
                expected="nk.Graph",
                value=type(graph).__name__,
            )

        if not isinstance(mapper, IDMapper):
            raise ValidationError(
                f"layers[{i}][1] must be an IDMapper, got {type(mapper).__name__}",
                field=f"layers[{i}][1]",
                expected="IDMapper",
                value=type(mapper).__name__,
            )

        if graph.isDirected():
            raise ValidationError(
                f"layers[{i}] graph is directed; multilayer GLP v1 only "
                "supports undirected layers. Convert to undirected or wait "
                "for v2 directional support.",
                field=f"layers[{i}]",
                expected="undirected graph",
            )

        if graph.numberOfNodes() != mapper.size():
            raise ValidationError(
                f"layers[{i}] graph and IDMapper have inconsistent sizes "
                f"({graph.numberOfNodes()} nodes vs mapper size {mapper.size()}). "
                "Each layer's IDMapper must cover exactly the graph's node set.",
                field=f"layers[{i}]",
                expected="graph.numberOfNodes() == mapper.size()",
            )


def _validate_and_default_gammas(
    gammas: Optional[Sequence[float]],
    n_layers: int,
) -> List[float]:
    """
    Validate ``gammas`` and apply the uniform default if ``None``.

    Parameters
    ----------
    gammas : Sequence[float] or None
        Per-layer importance weights. ``None`` means "use ``[1.0] * n_layers``".
    n_layers : int
        Number of layers; must equal ``len(gammas)`` when ``gammas`` is given.

    Returns
    -------
    list of float
        Validated per-layer weights, always a fresh list of plain floats.

    Raises
    ------
    ValidationError
        On length mismatch, non-numeric entry, or non-positive value.
    """
    if gammas is None:
        return [1.0] * n_layers

    gammas_list = list(gammas)

    if len(gammas_list) != n_layers:
        raise ValidationError(
            f"gammas length ({len(gammas_list)}) must equal number of "
            f"layers ({n_layers})",
            field="gammas",
            expected=f"length {n_layers}",
            value=len(gammas_list),
        )

    for i, g in enumerate(gammas_list):
        if isinstance(g, bool) or not isinstance(g, (int, float)):
            raise ValidationError(
                f"gammas[{i}] must be a number, got {type(g).__name__}",
                field=f"gammas[{i}]",
                expected="int or float",
                value=type(g).__name__,
            )
        if g <= 0:
            raise ValidationError(
                f"gammas[{i}] must be positive, got {g}",
                field=f"gammas[{i}]",
                expected="positive number",
                value=g,
            )

    return [float(g) for g in gammas_list]


def _build_union_mapper(layers: List[Tuple[nk.Graph, IDMapper]]) -> IDMapper:
    """
    Build a fresh IDMapper covering the union of original IDs across all layers.

    Internal IDs are assigned 0..N-1 in **sorted order of original IDs** so
    the output is deterministic across runs and platforms. Callers must pass
    layers whose originals are comparable (e.g. all strings, or all ints);
    mixed-type unions raise ``TypeError`` from ``sorted()``.

    Parameters
    ----------
    layers : list of (nk.Graph, IDMapper)
        Already-validated layer list. Mapper keys are unioned across layers.

    Returns
    -------
    IDMapper
        Mapper of size ``|⋃_ℓ originals(layer_ℓ)|``.
    """
    all_originals: set = set()
    for _graph, mapper in layers:
        all_originals.update(mapper.original_to_internal.keys())

    sorted_union = sorted(all_originals)
    return IDMapper.from_originals(sorted_union)


def _layer_to_union_lookup(
    layer_mapper: IDMapper,
    union_mapper: IDMapper,
) -> np.ndarray:
    """
    Build a dense lookup array from layer-internal IDs to union-internal IDs.

    Parameters
    ----------
    layer_mapper : IDMapper
        The layer's own internal-ID mapping (size ``N_ℓ``).
    union_mapper : IDMapper
        The union-internal-ID mapping built by :func:`_build_union_mapper`.

    Returns
    -------
    np.ndarray of int64, shape ``(N_ℓ,)``
        ``lookup[i] == j`` iff layer-internal ID ``i`` corresponds to the
        same original as union-internal ID ``j``.

    Raises
    ------
    KeyError
        If any of the layer's originals is missing from ``union_mapper``
        (should not happen when ``union_mapper`` was built from the same
        layer list).
    """
    n_layer = layer_mapper.size()
    if n_layer == 0:
        return np.empty(0, dtype=np.int64)

    originals_in_order = layer_mapper.get_original_batch(list(range(n_layer)))
    union_ids = union_mapper.get_internal_batch(originals_in_order)
    return np.asarray(union_ids, dtype=np.int64)


# ---------------------------------------------------------------------------
# Phase 2: supra-adjacency construction.
# Builds a (L·N_union × L·N_union) sparse matrix whose block ℓ along the
# diagonal is the layer's adjacency lifted to the union node set and scaled
# by gammas[ℓ], plus an off-block-diagonal contribution that couples every
# pair of layer-copies of the same node by ``coupling``.
# ---------------------------------------------------------------------------


def _build_supra_adjacency(
    layers: List[Tuple[nk.Graph, IDMapper]],
    gammas: Sequence[float],
    coupling: float,
    union_mapper: IDMapper,
) -> sp.csr_matrix:
    """
    Construct the supra-adjacency matrix for multilayer GLP.

    For each layer ℓ a CSR block ``A_ℓ`` of shape ``(N_union, N_union)`` is
    built by translating that layer's edge endpoints from layer-internal to
    union-internal IDs via :func:`_layer_to_union_lookup`, symmetrizing, and
    scaling weights by ``gammas[ℓ]``. The per-layer blocks are stacked into
    a block-diagonal matrix of shape ``(L·N_union, L·N_union)``.

    Inter-layer coupling is added on top: every pair of layer-copies of the
    same node ``u`` (positions ``u, u + N_union, ..., u + (L-1)·N_union``)
    is connected by an edge of weight ``coupling``. This is the kron
    product ``coupling · ((J_LxL − I_L) ⊗ I_N_union)``.

    Parameters
    ----------
    layers : list of (nk.Graph, IDMapper)
        Already-validated layer list.
    gammas : Sequence[float]
        Per-layer importance weights, length ``L = len(layers)``. Assumed
        validated by :func:`_validate_and_default_gammas`.
    coupling : float
        Inter-layer same-node coupling strength. ``coupling == 0`` skips
        the kron contribution entirely.
    union_mapper : IDMapper
        Union-internal mapping built by :func:`_build_union_mapper`.

    Returns
    -------
    sp.csr_matrix
        Supra-adjacency in CSR format. Shape ``(L·N_union, L·N_union)``,
        dtype float64, symmetric.

    Notes
    -----
    Total nonzeros are approximately
    ``Σ_ℓ 2·E_ℓ + L·(L−1)·N_union`` (each undirected layer edge contributes
    two entries; coupling contributes one entry per ordered cross-layer pair
    per node).

    Time complexity ``O(Σ_ℓ E_ℓ + L²·N_union)`` for matrix assembly.
    """
    N_union = union_mapper.size()
    L = len(layers)

    layer_blocks: List[sp.csr_matrix] = []
    for (graph, mapper), gamma in zip(layers, gammas):
        lookup = _layer_to_union_lookup(mapper, union_mapper)

        rows: List[int] = []
        cols: List[int] = []
        weights: List[float] = []
        for u, v in graph.iterEdges():
            w = graph.weight(u, v) * gamma
            u_union = int(lookup[u])
            v_union = int(lookup[v])
            rows.append(u_union)
            cols.append(v_union)
            weights.append(w)
            # Symmetrize. For self-loops (u == v), the entry is already in
            # place — don't double-count it.
            if u != v:
                rows.append(v_union)
                cols.append(u_union)
                weights.append(w)

        A_layer = sp.coo_matrix(
            (weights, (rows, cols)),
            shape=(N_union, N_union),
            dtype=np.float64,
        ).tocsr()
        layer_blocks.append(A_layer)

    A_block = sp.block_diag(layer_blocks, format="csr", dtype=np.float64)

    # Inter-layer coupling. Skip when there's only one layer or the
    # coupling weight is exactly zero — the kron term would be all-zero.
    if L > 1 and coupling != 0.0:
        coupling_pattern = np.ones((L, L), dtype=np.float64) - np.eye(L, dtype=np.float64)
        A_coupling = sp.kron(
            coupling_pattern,
            sp.eye(N_union, dtype=np.float64, format="csr"),
            format="csr",
        ) * coupling
        return (A_block + A_coupling).tocsr()

    return A_block


# ---------------------------------------------------------------------------
# Phase 3: propagation kernel.
# Builds the row-normalized supra-transition matrix, lifts seeds into the
# supra-node space (same seed active in every layer copy), and iterates the
# standard GLP update on the supra-matrix until convergence.
#
# Normalization of the final F matrix (so each row sums to 1) happens at
# output-formatting time, mirroring the reference ``guided_label_propagation``
# pattern where ``_normalize_F`` runs in ``_create_results_dataframe``.
# ---------------------------------------------------------------------------


def _supra_propagation_step(
    F: np.ndarray,
    P_supra: sp.csr_matrix,
    Y: np.ndarray,
    alpha: float,
) -> np.ndarray:
    """
    One iteration of the supra-GLP update: ``α · P_supra · F + (1−α) · Y``.

    Extracted as a single function so a v2 variant (e.g. Boutemine-Bouguessa
    adaptive quality-function update) can be swapped in without touching the
    runner.
    """
    return alpha * (P_supra @ F) + (1.0 - alpha) * Y


def _row_normalize_supra(A_supra: sp.csr_matrix) -> sp.csr_matrix:
    """
    Compute ``D^-1 A_supra`` (row-normalized supra-transition).

    Zero-degree rows in the supra-graph — isolated layer-copies such as a
    node absent from a layer when ``coupling == 0`` — are left as zero rows
    so they contribute no mass during propagation. Mirrors the convention in
    :func:`guidedLP.glp.propagation._create_transition_matrix`.
    """
    row_sums = np.asarray(A_supra.sum(axis=1)).flatten()
    zero_mask = (row_sums == 0.0)
    safe_row_sums = np.where(zero_mask, 1.0, row_sums)
    row_inv = 1.0 / safe_row_sums
    row_inv[zero_mask] = 0.0
    return sp.diags(row_inv).dot(A_supra).tocsr()


def _build_supra_seeds(
    seed_dict: Dict[Any, str],
    labels: List[str],
    union_mapper: IDMapper,
    n_layers: int,
) -> np.ndarray:
    """
    Lift seeds into the supra-Y indicator matrix.

    Each seed ``(orig_id, label)`` is replicated across all L layer-copies of
    the node: positions ``u_union + ℓ·N_union`` (for ``ℓ ∈ {0..L−1}``) get a
    ``1.0`` in the column corresponding to ``label``. This is the standard
    supra-adjacency lift for Dirichlet boundary conditions — every layer
    "knows" the seed.

    Parameters
    ----------
    seed_dict : Dict[Any, str]
        Canonical seed mapping. Caller is responsible for ensuring every
        ``orig_id`` is present in ``union_mapper`` and every label appears
        in ``labels`` (Phase 5 validates these up front).
    labels : List[str]
        Label vocabulary. Column order in the returned matrix.
    union_mapper : IDMapper
        Union-internal mapping.
    n_layers : int
        Number of layers L.

    Returns
    -------
    np.ndarray of float64, shape ``(L · N_union, len(labels))``
    """
    N_union = union_mapper.size()
    Y = np.zeros((n_layers * N_union, len(labels)), dtype=np.float64)
    label_to_idx = {label: i for i, label in enumerate(labels)}
    for orig_id, label in seed_dict.items():
        u_union = union_mapper.get_internal(orig_id)
        c_idx = label_to_idx[label]
        for ell in range(n_layers):
            Y[u_union + ell * N_union, c_idx] = 1.0
    return Y


def _run_multilayer_propagation(
    layers: List[Tuple[nk.Graph, IDMapper]],
    gammas: Sequence[float],
    coupling: float,
    union_mapper: IDMapper,
    seed_dict: Dict[Any, str],
    labels: List[str],
    alpha: float,
    max_iterations: int,
    convergence_threshold: float,
) -> Tuple[np.ndarray, int]:
    """
    Iterate the supra-GLP update to convergence.

    Builds the supra-adjacency via :func:`_build_supra_adjacency`,
    row-normalizes to obtain the supra-transition, lifts seeds via
    :func:`_build_supra_seeds`, and iterates
    ``F^(t+1) = α · P · F^(t) + (1 − α) · Y`` until
    ``L∞(F_new − F_old) < convergence_threshold`` or ``max_iterations``
    is reached.

    Parameters
    ----------
    layers, gammas, coupling, union_mapper
        Passed through to :func:`_build_supra_adjacency`.
    seed_dict : Dict[Any, str]
        Canonical ``original_id → label`` mapping.
    labels : List[str]
        Label vocabulary (column order of the output).
    alpha, max_iterations, convergence_threshold
        Standard GLP knobs. ``alpha ∈ [0, 1]`` is assumed.

    Returns
    -------
    F_supra : np.ndarray of float64, shape ``(L · N_union, len(labels))``
        Final propagated label-probability matrix in supra-node layout
        ``[layer_0 nodes..., layer_1 nodes..., ...]``. **Not row-normalized**
        — row-normalization (so each node's probabilities sum to 1) is
        applied at output-formatting time by Phase 4.
    n_iter : int
        Number of iterations actually run (``0`` if ``max_iterations == 0``;
        otherwise ``1 ≤ n_iter ≤ max_iterations``).
    """
    A_supra = _build_supra_adjacency(layers, gammas, coupling, union_mapper)
    P_supra = _row_normalize_supra(A_supra)

    L = len(layers)
    Y = _build_supra_seeds(seed_dict, labels, union_mapper, L)
    F = Y.copy()

    n_iter = 0
    change = 0.0
    for it in range(1, max_iterations + 1):
        F_new = _supra_propagation_step(F, P_supra, Y, alpha)
        change = float(np.max(np.abs(F_new - F)))
        F = F_new
        n_iter = it
        if change < convergence_threshold:
            break

    logger.debug(
        "multilayer GLP iterated %d times (final L_inf change=%.2e, supra shape=%s)",
        n_iter, change, F.shape,
    )

    return F, n_iter


# ---------------------------------------------------------------------------
# Phase 4: aggregation and output formatting.
# Collapses the supra-F (L · N_union × C) over the layer axis into a single
# per-node distribution, optionally row-normalizes, and emits a polars
# DataFrame with the same schema as ``guided_label_propagation``.
# ---------------------------------------------------------------------------


# Valid aggregation modes. Kept as a module constant so the public-function
# validator and the helper share one source of truth.
_VALID_AGGREGATIONS: Tuple[str, ...] = ("weighted_mean", "mean", "max")


def _aggregate_layer_copies(
    F_supra: np.ndarray,
    N_union: int,
    L: int,
    gammas: Sequence[float],
    aggregation: str,
) -> np.ndarray:
    """
    Collapse each node's L layer-copies into a single distribution.

    The supra-F is laid out as ``[layer_0_nodes..., layer_1_nodes..., ...]``,
    so reshape to ``(L, N_union, C)`` puts the layer index on axis 0 and
    each subsequent reduction operates over that axis.

    Parameters
    ----------
    F_supra : np.ndarray of float64, shape ``(L · N_union, C)``
        Supra-propagation output from :func:`_run_multilayer_propagation`.
    N_union : int
        Number of distinct nodes in the union node space.
    L : int
        Number of layers.
    gammas : Sequence[float]
        Per-layer importance weights (only used for ``"weighted_mean"``).
    aggregation : str
        One of ``"weighted_mean"``, ``"mean"``, ``"max"``.

    Returns
    -------
    np.ndarray of float64, shape ``(N_union, C)``
        Per-node aggregated label distribution.

    Raises
    ------
    ValidationError
        If ``aggregation`` is not a recognized mode. (Phase 5 also
        validates up-front; this is a defensive last-line check.)
    """
    if aggregation not in _VALID_AGGREGATIONS:
        raise ValidationError(
            f"Unknown aggregation mode '{aggregation}'",
            field="aggregation",
            value=aggregation,
            expected=f"one of {list(_VALID_AGGREGATIONS)}",
        )

    C = F_supra.shape[1]
    F_stacked = F_supra.reshape(L, N_union, C)

    if aggregation == "weighted_mean":
        gammas_arr = np.asarray(gammas, dtype=np.float64)
        # gammas_arr sums to a strictly positive value (Phase 1 validation
        # guarantees every entry > 0), so no zero-divide guard needed here.
        return np.einsum("l,lnc->nc", gammas_arr, F_stacked) / gammas_arr.sum()

    if aggregation == "mean":
        return F_stacked.mean(axis=0)

    # aggregation == "max"
    return F_stacked.max(axis=0)


def _normalize_F_rows(F: np.ndarray) -> np.ndarray:
    """
    Row-normalize ``F`` so each row sums to 1; zero-rows become uniform.

    Inlined (rather than imported from ``propagation._normalize_F``) so that
    the module stays self-contained and trivially removable per the "delete
    one file" design goal.

    Notes
    -----
    Mutates ``F`` in place when zero-rows are present. Always assign the
    return value.
    """
    n_labels = F.shape[1]
    row_sums = F.sum(axis=1, keepdims=True)
    zero_mask = (row_sums.flatten() == 0.0)
    if zero_mask.any():
        F[zero_mask, :] = 1.0 / n_labels
        row_sums = F.sum(axis=1, keepdims=True)
    return F / row_sums


def _format_output(
    F_aggregated: np.ndarray,
    union_mapper: IDMapper,
    labels: List[str],
    seed_dict: Dict[Any, str],
    normalize: bool,
    confidence_threshold: float,
    exclude_from_output: Optional[Set[Any]],
) -> pl.DataFrame:
    """
    Build the final per-node DataFrame mirroring ``guided_label_propagation``.

    Column order and dtypes match the reference exactly:
    ``node_id``, ``{label}_prob`` (one per label, float64), ``dominant_label``
    (str), ``confidence`` (float64), ``is_seed`` (bool).

    Parameters
    ----------
    F_aggregated : np.ndarray, shape ``(N_union, len(labels))``
        Per-node label distribution from :func:`_aggregate_layer_copies`.
    union_mapper : IDMapper
        Maps union-internal indices back to original IDs.
    labels : List[str]
        Label vocabulary; column order of the probability columns.
    seed_dict : Dict[Any, str]
        Canonical seed mapping. ``is_seed`` is True for original IDs in
        ``seed_dict.keys()`` (mirrors the reference single-pass behavior
        — if the caller pre-extends ``seed_dict`` with auto-generated noise
        seeds, those will be flagged too).
    normalize : bool
        Row-normalize ``F_aggregated`` so each node's probabilities sum to
        1. Matches the reference ``normalize`` semantics.
    confidence_threshold : float
        Nodes with row-max below this are relabeled ``"uncertain"``. Pass
        ``0.0`` to disable.
    exclude_from_output : set or None
        Original IDs to drop from the final DataFrame. Pass ``None`` or
        an empty set to keep everything.

    Returns
    -------
    pl.DataFrame
    """
    if normalize:
        F_aggregated = _normalize_F_rows(F_aggregated.copy())

    n_nodes, _n_labels = F_aggregated.shape

    dominant_indices = np.argmax(F_aggregated, axis=1)
    dominant_labels = [labels[idx] for idx in dominant_indices]
    confidence_scores = np.max(F_aggregated, axis=1)

    original_ids = union_mapper.get_original_batch(list(range(n_nodes)))
    seed_keys = set(seed_dict.keys())
    is_seed = [orig_id in seed_keys for orig_id in original_ids]

    result_data: Dict[str, Any] = {"node_id": original_ids}
    for i, label in enumerate(labels):
        result_data[f"{label}_prob"] = F_aggregated[:, i].tolist()
    result_data["dominant_label"] = dominant_labels
    result_data["confidence"] = confidence_scores.tolist()
    result_data["is_seed"] = is_seed

    df = pl.DataFrame(result_data)

    if confidence_threshold > 0.0:
        n_uncertain = int((df["confidence"] < confidence_threshold).sum())
        if n_uncertain > 0:
            df = df.with_columns(
                pl.when(pl.col("confidence") < confidence_threshold)
                .then(pl.lit("uncertain"))
                .otherwise(pl.col("dominant_label"))
                .alias("dominant_label")
            )
            logger.info(
                "Classified %d nodes as 'uncertain' (confidence < %.4f)",
                n_uncertain, confidence_threshold,
            )

    if exclude_from_output:
        n_before = len(df)
        df = df.filter(~pl.col("node_id").is_in(list(exclude_from_output)))
        n_dropped = n_before - len(df)
        if n_dropped > 0:
            logger.info(
                "Excluded %d node(s) from output (requested %d)",
                n_dropped, len(exclude_from_output),
            )

    return df


# ---------------------------------------------------------------------------
# Phase 5: public multilayer_guided_label_propagation entry point.
# Validates inputs, normalizes seeds, optionally extends labels/seed_dict
# with auto-noise, then wires Phases 1–4 together.
# ---------------------------------------------------------------------------


def _generate_union_noise_seeds(
    union_mapper: IDMapper,
    seed_dict: Dict[Any, str],
    noise_ratio: float,
    random_seed: Optional[int],
) -> Dict[Any, str]:
    """
    Sample noise seeds from non-seed nodes in the union node space.

    Mirrors ``propagation._generate_noise_seeds`` but operates against the
    union of original IDs across layers rather than a single graph.

    Returns
    -------
    Dict[Any, str]
        ``{original_id: "noise"}`` for each sampled node. Empty if there
        are no non-seed nodes available.
    """
    all_originals = set(union_mapper.original_to_internal.keys())
    non_seed_originals = sorted(all_originals - set(seed_dict.keys()))

    if not non_seed_originals:
        logger.warning("No non-seed nodes available for noise seed generation")
        return {}

    n_user_seeds = len(seed_dict)
    n_noise = max(1, int(noise_ratio * n_user_seeds))
    n_noise = min(n_noise, len(non_seed_originals))

    rng = random.Random(random_seed) if random_seed is not None else random
    selected = rng.sample(non_seed_originals, n_noise)
    return {orig: "noise" for orig in selected}


def multilayer_guided_label_propagation(
    layers: List[Tuple[nk.Graph, IDMapper]],
    seed_labels: SeedInput,
    labels: List[str],
    *,
    gammas: Optional[Sequence[float]] = None,
    coupling: float = 0.3,
    aggregation: Literal["weighted_mean", "mean", "max"] = "weighted_mean",
    alpha: float = 0.85,
    max_iterations: int = 100,
    convergence_threshold: float = 1e-6,
    normalize: bool = True,
    directional: bool = False,
    enable_noise_category: bool = False,
    noise_ratio: float = 0.1,
    confidence_threshold: float = 0.0,
    seed_node_col: str = "node_id",
    seed_label_col: str = "label",
    random_seed: Optional[int] = 42,
    exclude_from_output: Optional[Set[Any]] = None,
    verbose: bool = True,
) -> Union[pl.DataFrame, Tuple[pl.DataFrame, pl.DataFrame]]:
    """
    Run Guided Label Propagation jointly over multiple graphs that share
    (a subset of) the same node set.

    Builds a supra-adjacency matrix from the per-layer adjacencies with
    intra-layer block weights ``gammas[ℓ]`` and inter-layer same-node
    coupling ``coupling``, lifts the seed indicator to the supra-node
    space, runs the standard GLP update, then aggregates over each node's
    layer-copies to produce a single per-node label distribution.

    Parameters
    ----------
    layers : list of (nk.Graph, IDMapper) tuples
        One per layer. Layers may have different (overlapping) node sets;
        the function unions them. All graphs must be undirected for v1.
    seed_labels : SeedInput
        Same shape rules as ``guided_label_propagation`` — ``node_id → label``
        dict, ``label → node_ids`` dict, or a DataFrame with the configured
        columns.
    labels : list of str
        Label vocabulary. Must include every label appearing in ``seed_labels``.
    gammas : Sequence[float], optional
        Per-layer importance weights. Length must equal ``len(layers)``; all
        values must be positive. Default: uniform (all ``1.0``).
    coupling : float, default 0.3
        Inter-layer same-node coupling strength ``C``. Each node's ``L`` copies
        are connected pairwise by edges of weight ``coupling``.

        - ``C ≈ 0``: layers act nearly independently (evaporation risk).
        - ``C → ∞``: layer copies converge to identical values ("consensus
          collapse"). ``0.1–0.5`` is the usable range for most data.
        - Must be ``≥ 0``.
    aggregation : {"weighted_mean", "mean", "max"}, default "weighted_mean"
        How to combine each node's ``L`` layer-copies into a single output
        distribution.

        - ``"weighted_mean"``: ``Σ_ℓ γ_ℓ · F[(u,ℓ)] / Σ_ℓ γ_ℓ``
        - ``"mean"``: simple arithmetic mean over layers
        - ``"max"``: element-wise max across layers (winner-takes-all)
    alpha, max_iterations, convergence_threshold, normalize : ...
        Standard GLP knobs. Defaults match ``guided_label_propagation``
        exactly.
    directional : bool, default False
        Must be ``False`` in v1. Raises :class:`NotImplementedError` if
        ``True`` — supra-adjacency for directed graphs needs two passes
        (``A_ℓ`` and ``A_ℓᵀ`` blocks) and is deferred to v2.
    enable_noise_category : bool, default False
        Auto-add a ``"noise"`` label and sample non-seed nodes as noise
        seeds. Matches ``guided_label_propagation`` semantics.
    noise_ratio : float, default 0.1
        Multiplier on the user-seed count for noise sampling.
    confidence_threshold : float, default 0.0
        Nodes whose row-max probability falls below this are relabeled
        ``"uncertain"`` in ``dominant_label``. Must be in ``[0, 1]``.
    seed_node_col, seed_label_col : str
        Column names used when ``seed_labels`` is a DataFrame.
    random_seed : int or None, default 42
        Seed for the local RNG used to sample noise seeds.
    exclude_from_output : set or None, default None
        Original IDs to drop from the returned DataFrame after propagation.
    verbose : bool, default True
        Print a one-line summary at the end.

    Returns
    -------
    pl.DataFrame
        Schema matches ``guided_label_propagation``: ``node_id``,
        ``{label}_prob`` per label, ``dominant_label``, ``confidence``,
        ``is_seed``.

    Raises
    ------
    NotImplementedError
        If ``directional=True``.
    ValidationError
        On any malformed input. See :func:`_validate_layers_input`,
        :func:`_validate_and_default_gammas`, and inline checks.

    See Also
    --------
    make_multilayer_propagator
        Factory that wires a configured multilayer propagator for use with
        :func:`guidedLP.glp.train_test_split_validation`.
    """
    if directional:
        raise NotImplementedError(
            "directional=True is not supported in v1 of multilayer GLP. "
            "Use directional=False (undirected propagation) for now."
        )

    _validate_layers_input(layers)
    gammas_list = _validate_and_default_gammas(gammas, len(layers))

    if aggregation not in _VALID_AGGREGATIONS:
        raise ValidationError(
            f"aggregation must be one of {list(_VALID_AGGREGATIONS)}, "
            f"got '{aggregation}'",
            field="aggregation",
            value=aggregation,
            expected=f"one of {list(_VALID_AGGREGATIONS)}",
        )

    # coupling: non-negative number (bool excluded because bool is an int subclass).
    if isinstance(coupling, bool) or not isinstance(coupling, (int, float)):
        raise ValidationError(
            f"coupling must be a number, got {type(coupling).__name__}",
            field="coupling",
            value=type(coupling).__name__,
            expected="int or float",
        )
    if coupling < 0.0:
        raise ValidationError(
            f"coupling must be non-negative, got {coupling}",
            field="coupling",
            value=coupling,
            expected="non-negative number",
        )

    if not 0.0 <= alpha <= 1.0:
        raise ValidationError(
            f"alpha must be in [0, 1], got {alpha}",
            field="alpha",
            value=alpha,
            expected="number in [0, 1]",
        )
    if max_iterations <= 0:
        raise ValidationError(
            f"max_iterations must be positive, got {max_iterations}",
            field="max_iterations",
            value=max_iterations,
        )
    if convergence_threshold <= 0:
        raise ValidationError(
            f"convergence_threshold must be positive, got {convergence_threshold}",
            field="convergence_threshold",
            value=convergence_threshold,
        )
    if not 0.0 <= confidence_threshold <= 1.0:
        raise ValidationError(
            f"confidence_threshold must be in [0, 1], got {confidence_threshold}",
            field="confidence_threshold",
            value=confidence_threshold,
        )
    if enable_noise_category and noise_ratio < 0.0:
        raise ValidationError(
            f"noise_ratio must be non-negative, got {noise_ratio}",
            field="noise_ratio",
            value=noise_ratio,
        )

    if not labels:
        raise ValidationError("labels list cannot be empty", field="labels")
    if len(set(labels)) != len(labels):
        raise ValidationError("labels list contains duplicates", field="labels")

    seed_dict = normalize_seed_input(seed_labels, seed_node_col, seed_label_col)
    if not seed_dict:
        raise ValidationError(
            "seed_labels is empty after normalization",
            field="seed_labels",
        )

    unknown_labels = set(seed_dict.values()) - set(labels)
    if unknown_labels:
        raise ValidationError(
            f"seed labels contain unknown labels: {sorted(unknown_labels)}",
            field="seed_labels",
            details={
                "unknown_labels": sorted(unknown_labels),
                "valid_labels": labels,
            },
        )

    union_mapper = _build_union_mapper(layers)

    # Drop seeds that aren't present in any layer (warn-and-skip, mirroring
    # the single-layer reference). Only error if nothing usable remains.
    missing_seeds = [s for s in list(seed_dict.keys()) if not union_mapper.has_original(s)]
    if missing_seeds:
        for s in missing_seeds:
            del seed_dict[s]
        logger.warning(
            "%d seed node(s) not found in any layer and were skipped "
            "(first few: %s)",
            len(missing_seeds), missing_seeds[:5],
        )
    if not seed_dict:
        raise ValidationError(
            "None of the supplied seeds are present in any layer.",
            field="seed_labels",
            details={
                "missing_seeds_count": len(missing_seeds),
                "missing_seeds_sample": missing_seeds[:10],
            },
        )

    # Noise-category handling. Mirrors propagation._process_noise_category:
    # add the "noise" label (if not already present) and sample auto-noise
    # seeds from non-seed union nodes (skipped if the user already labeled
    # some seeds as "noise").
    processed_labels: List[str] = list(labels)
    if enable_noise_category:
        if "noise" not in processed_labels:
            processed_labels.append("noise")
            logger.info("Added 'noise' category to labels")
        if "noise" not in seed_dict.values():
            noise_seeds = _generate_union_noise_seeds(
                union_mapper, seed_dict, noise_ratio, random_seed,
            )
            seed_dict.update(noise_seeds)
            logger.info("Generated %d noise seed(s)", len(noise_seeds))

    with LoggingTimer("multilayer_guided_label_propagation"):
        F_supra, n_iter = _run_multilayer_propagation(
            layers, gammas_list, coupling, union_mapper, seed_dict, processed_labels,
            alpha, max_iterations, convergence_threshold,
        )

    F_aggregated = _aggregate_layer_copies(
        F_supra, union_mapper.size(), len(layers), gammas_list, aggregation,
    )

    df = _format_output(
        F_aggregated, union_mapper, processed_labels, seed_dict,
        normalize, confidence_threshold, exclude_from_output,
    )

    if verbose:
        total_edges = sum(g.numberOfEdges() for g, _ in layers)
        print(
            f"[multilayer_glp] {n_iter} iter | "
            f"layers={len(layers)} union_nodes={union_mapper.size():,} "
            f"edges={total_edges:,} | "
            f"alpha={alpha} coupling={coupling} aggregation={aggregation} | "
            f"seeds={len(seed_dict):,} labels={len(processed_labels)}"
        )

    return df


# ---------------------------------------------------------------------------
# Phase 6: factory that adapts the multilayer propagator to the
# ``train_test_split_validation`` protocol.
# ---------------------------------------------------------------------------


def make_multilayer_propagator(
    layers: List[Tuple[nk.Graph, IDMapper]],
    *,
    gammas: Optional[Sequence[float]] = None,
    coupling: float = 0.3,
    aggregation: Literal["weighted_mean", "mean", "max"] = "weighted_mean",
) -> Tuple[Callable[..., Union[pl.DataFrame, Tuple[pl.DataFrame, pl.DataFrame]]],
           IDMapper, nk.Graph]:
    """
    Build a multilayer-GLP propagator wired for ``train_test_split_validation``.

    Returns a closure with the signature
    ``(graph, id_mapper, seed_labels, labels, **glp_kwargs)`` expected by
    :func:`guidedLP.glp.train_test_split_validation`'s ``propagator=`` slot.
    The ``graph`` and ``id_mapper`` positional arguments are accepted for
    protocol compatibility but **ignored** — the multilayer configuration
    (layers, gammas, coupling, aggregation) is captured in the closure at
    factory-call time.

    Validation of ``layers``, ``gammas``, ``coupling``, and ``aggregation``
    happens eagerly here so misconfiguration surfaces at factory time
    rather than deep inside a validation loop.

    Parameters
    ----------
    layers : list of (nk.Graph, IDMapper) tuples
        Same shape as :func:`multilayer_guided_label_propagation`.
    gammas : Sequence[float], optional
        Per-layer importance weights. Default: uniform.
    coupling : float, default 0.3
        Inter-layer same-node coupling strength. Must be ``≥ 0``.
    aggregation : {"weighted_mean", "mean", "max"}, default "weighted_mean"

    Returns
    -------
    propagator : Callable
        Drop-in propagator for :func:`train_test_split_validation`.
    union_mapper : IDMapper
        Master IDMapper over the union of nodes from all layers. Pass this
        as ``id_mapper=`` to :func:`train_test_split_validation` so its seed
        splitter sees the full node universe.
    anchor_graph : nk.Graph
        Empty (no-edge) undirected graph sized to ``union_mapper``. Pass
        this as ``graph=`` to satisfy
        :func:`train_test_split_validation`'s protocol; the propagator
        ignores it.

    Raises
    ------
    ValidationError
        On any malformed layer / gamma / coupling / aggregation argument.

    Examples
    --------
    >>> from guidedLP.glp import make_multilayer_propagator
    >>> from guidedLP.glp import train_test_split_validation
    >>> propagator, union_mapper, anchor_graph = make_multilayer_propagator(
    ...     layers=[(url_g, url_m), (dom_g, dom_m), (emb_g, emb_m)],
    ...     gammas=[1.0, 1.0, 0.5],
    ...     coupling=0.3,
    ... )
    >>> metrics = train_test_split_validation(
    ...     graph=anchor_graph,
    ...     id_mapper=union_mapper,
    ...     seed_labels=seeds,
    ...     labels=["left", "right"],
    ...     test_size=0.2,
    ...     propagator=propagator,
    ...     alpha=0.85,
    ... )
    """
    # Eager validation — mirror multilayer_guided_label_propagation's checks
    # so misconfiguration is caught at factory-build time, not inside the
    # train/test loop.
    _validate_layers_input(layers)
    gammas_resolved = _validate_and_default_gammas(gammas, len(layers))

    if aggregation not in _VALID_AGGREGATIONS:
        raise ValidationError(
            f"aggregation must be one of {list(_VALID_AGGREGATIONS)}, "
            f"got '{aggregation}'",
            field="aggregation",
            value=aggregation,
            expected=f"one of {list(_VALID_AGGREGATIONS)}",
        )

    if isinstance(coupling, bool) or not isinstance(coupling, (int, float)):
        raise ValidationError(
            f"coupling must be a number, got {type(coupling).__name__}",
            field="coupling",
            value=type(coupling).__name__,
            expected="int or float",
        )
    if coupling < 0.0:
        raise ValidationError(
            f"coupling must be non-negative, got {coupling}",
            field="coupling",
            value=coupling,
            expected="non-negative number",
        )

    union_mapper = _build_union_mapper(layers)
    anchor_graph = nk.Graph(union_mapper.size(), weighted=False, directed=False)

    def propagator(
        graph: nk.Graph,                       # noqa: ARG001 — protocol slot
        id_mapper: IDMapper,                   # noqa: ARG001 — protocol slot
        seed_labels: SeedInput,
        labels: List[str],
        **glp_kwargs: Any,
    ) -> Union[pl.DataFrame, Tuple[pl.DataFrame, pl.DataFrame]]:
        # `graph` and `id_mapper` come from train_test_split_validation but
        # are intentionally ignored: the multilayer setup (layers, gammas,
        # coupling, aggregation) is fully captured by the enclosing closure.
        return multilayer_guided_label_propagation(
            layers=layers,
            seed_labels=seed_labels,
            labels=labels,
            gammas=gammas_resolved,
            coupling=coupling,
            aggregation=aggregation,
            **glp_kwargs,
        )

    return propagator, union_mapper, anchor_graph
