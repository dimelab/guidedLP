"""
Reduced graph export module for the Guided Label Propagation library.

For large networks, full-graph visualization in tools like Gephi is infeasible.
This module produces a smaller, information-preserving graph through one of
three approaches:

- ``reduce_by_grouping`` — Louvain community detection followed by quotient
  (supernode) graph construction. Each supernode aggregates its members'
  GLP results (mean per-label probabilities, majority dominant label,
  member count, pipe-delimited member IDs). Preserves mesoscale structure.
- ``reduce_by_influence`` — Network backboning via
  :func:`~guidedLP.network.backboning.apply_backbone`. Retains statistically
  significant edges; optionally further reduced by keeping top-degree nodes.
  Preserves the connectivity skeleton the network carries.
- ``reduce_by_sampling`` — Forest Fire sampling (Leskovec & Faloutsos 2006).
  Probabilistic BFS where each visited node burns a geometrically-distributed
  number of neighbors, with ambassador restart logic. Produces a connected,
  representative subgraph.

A convenience wrapper :func:`export_reduced_graph` dispatches to the
appropriate ``reduce_by_*`` and pipes the result through
:func:`~guidedLP.network.export.export_graph` (typically GEXF for Gephi).

GLP results are attached to nodes via the existing ``metadata`` parameter
of :func:`~guidedLP.network.export.export_graph`. Directional GLP results
(out-direction and in-direction frames) are supported via the
``glp_direction`` parameter on every function.

References
----------
Leskovec, J., & Faloutsos, C. (2006). Sampling from large graphs.
KDD '06: 631–636.

Coscia, M. (2021). The Atlas for the Aspiring Network Scientist.
Parts VII (Sampling/Backboning) and IX (Communities).
"""

from collections import defaultdict, deque
from pathlib import Path
from typing import (
    Any,
    Dict,
    List,
    Literal,
    Optional,
    Set,
    Tuple,
    Union,
)
import warnings

import networkit as nk
import numpy as np
import polars as pl

from guidedLP.common.exceptions import (
    ComputationError,
    ConfigurationError,
    ValidationError,
    validate_parameter,
)
from guidedLP.common.id_mapper import IDMapper
from guidedLP.common.logging_config import (
    LoggingTimer,
    get_logger,
    log_function_entry,
)
from guidedLP.network.backboning import apply_backbone
from guidedLP.network.construction import _build_induced_subgraph
from guidedLP.network.export import export_graph

logger = get_logger(__name__)

# Public type aliases ---------------------------------------------------------

GLPResult = Union[pl.DataFrame, Tuple[pl.DataFrame, pl.DataFrame]]
GLPDirection = Literal["out", "in", "both"]
ReductionMethod = Literal["grouping", "influence", "sampling"]

_REDUCTION_METHODS: List[str] = ["grouping", "influence", "sampling"]
_GLP_DIRECTIONS: List[str] = ["out", "in", "both"]
_BACKBONE_METHODS: List[str] = [
    "disparity",
    "noise_corrected",
    "bipartite_svn",
    "weight",
    "degree",
]
_RESOLUTION_SEARCH_OPTIONS: List[str] = ["fixed", "bisect"]


# ---------------------------------------------------------------------------
# Internal validation helpers
# ---------------------------------------------------------------------------


def _resolve_target_nodes(
    target_nodes: Optional[int],
    node_fraction: Optional[float],
    total: int,
    func_name: str,
) -> Optional[int]:
    """Return the resolved integer target node count, or None.

    Raises ``ValidationError`` if both are given or values are out of range.
    """
    if target_nodes is not None and node_fraction is not None:
        raise ValidationError(
            "Specify at most one of 'target_nodes' or 'node_fraction'",
            details={"function": func_name},
        )
    if target_nodes is not None:
        if not isinstance(target_nodes, (int, np.integer)) or target_nodes < 1:
            raise ValidationError(
                "'target_nodes' must be a positive integer",
                field="target_nodes",
                value=target_nodes,
            )
        return int(target_nodes)
    if node_fraction is not None:
        if not (0.0 < float(node_fraction) <= 1.0):
            raise ValidationError(
                "'node_fraction' must be in the interval (0, 1]",
                field="node_fraction",
                value=node_fraction,
            )
        return max(1, int(round(float(node_fraction) * total)))
    return None


def _resolve_target_edges(
    target_edges: Optional[int],
    edge_fraction: Optional[float],
    total: int,
    func_name: str,
) -> Optional[int]:
    """Return the resolved integer target edge count, or None."""
    if target_edges is not None and edge_fraction is not None:
        raise ValidationError(
            "Specify at most one of 'target_edges' or 'edge_fraction'",
            details={"function": func_name},
        )
    if target_edges is not None:
        if not isinstance(target_edges, (int, np.integer)) or target_edges < 1:
            raise ValidationError(
                "'target_edges' must be a positive integer",
                field="target_edges",
                value=target_edges,
            )
        return int(target_edges)
    if edge_fraction is not None:
        if not (0.0 < float(edge_fraction) <= 1.0):
            raise ValidationError(
                "'edge_fraction' must be in the interval (0, 1]",
                field="edge_fraction",
                value=edge_fraction,
            )
        return max(1, int(round(float(edge_fraction) * total)))
    return None


def _reject_unsupported_axes(
    target_edges: Optional[int],
    edge_fraction: Optional[float],
    method_name: str,
    *,
    target_nodes: Optional[int] = None,
    node_fraction: Optional[float] = None,
    forbid_node_axis: bool = False,
) -> None:
    """Raise ValidationError if non-native target axes are supplied.

    The plan documents which axes each method honors natively. Passing a
    non-native axis is rejected loudly rather than silently ignored.
    """
    if not forbid_node_axis and (
        target_edges is not None or edge_fraction is not None
    ):
        raise ValidationError(
            f"{method_name} does not honor 'target_edges' / 'edge_fraction'. "
            f"Use 'target_nodes' or 'node_fraction' instead.",
            details={"function": method_name},
        )
    if forbid_node_axis and (
        target_nodes is not None or node_fraction is not None
    ):
        # Reserved for future per-method node-axis rejection if needed.
        raise ValidationError(
            f"{method_name} does not honor 'target_nodes' / 'node_fraction'.",
            details={"function": method_name},
        )


# ---------------------------------------------------------------------------
# GLP result handling
# ---------------------------------------------------------------------------


def _merge_directional_glp(
    out_df: pl.DataFrame,
    in_df: pl.DataFrame,
) -> pl.DataFrame:
    """Merge directional GLP frames by prefixing non-key columns.

    Both frames are expected to share the same ``node_id`` universe (they come
    from a single :func:`guided_label_propagation` call on the same graph).
    Columns other than ``node_id`` are prefixed with ``out_`` and ``in_``
    respectively to avoid collisions on ``dominant_label``/``confidence``/
    ``is_seed``/``*_prob`` columns. A ``full`` join (with coalescing on
    ``node_id``) keeps every node even if the two frames diverge.
    """

    def _prefix(df: pl.DataFrame, prefix: str) -> pl.DataFrame:
        rename_map = {col: f"{prefix}_{col}" for col in df.columns if col != "node_id"}
        return df.rename(rename_map)

    out_pref = _prefix(out_df, "out")
    in_pref = _prefix(in_df, "in")

    try:
        return out_pref.join(in_pref, on="node_id", how="full", coalesce=True)
    except TypeError:
        # Older Polars: 'full' alias not available; use legacy 'outer'.
        merged = out_pref.join(in_pref, on="node_id", how="outer")
        if "node_id_right" in merged.columns:
            merged = merged.with_columns(
                pl.coalesce(["node_id", "node_id_right"]).alias("node_id")
            ).drop("node_id_right")
        return merged


def _resolve_glp_for_attach(
    glp_results: Optional[GLPResult],
    glp_direction: str,
) -> Optional[pl.DataFrame]:
    """Normalize a GLP result (single frame or directional tuple) for joining."""
    if glp_results is None:
        return None

    if isinstance(glp_results, tuple):
        if len(glp_results) != 2:
            raise ValidationError(
                "glp_results tuple must have length 2 (out_df, in_df)",
                field="glp_results",
            )
        out_df, in_df = glp_results
        if not isinstance(out_df, pl.DataFrame) or not isinstance(in_df, pl.DataFrame):
            raise ValidationError(
                "Both elements of glp_results tuple must be Polars DataFrames",
                field="glp_results",
            )
        validate_parameter(
            glp_direction, _GLP_DIRECTIONS, "glp_direction", "reduce_by_*"
        )
        if glp_direction == "out":
            return out_df
        if glp_direction == "in":
            return in_df
        return _merge_directional_glp(out_df, in_df)

    if not isinstance(glp_results, pl.DataFrame):
        raise ValidationError(
            "glp_results must be a Polars DataFrame or (out_df, in_df) tuple, "
            f"got {type(glp_results).__name__}",
            field="glp_results",
        )

    if "node_id" not in glp_results.columns:
        raise ValidationError(
            "glp_results DataFrame must contain a 'node_id' column",
            field="glp_results",
        )
    return glp_results


def _glp_prob_columns(glp_df: pl.DataFrame) -> List[str]:
    """Return the per-label probability columns of a GLP frame.

    A 'probability column' is any column whose name ends in ``_prob`` (single
    direction) or ``_prob`` after stripping an ``out_``/``in_`` prefix
    (directional merge). We don't strip prefixes here — the column names from
    the merged frame are taken verbatim.
    """
    return [c for c in glp_df.columns if c.endswith("_prob")]


def _attach_glp_to_nodes(
    node_ids: List[Any],
    glp_df: Optional[pl.DataFrame],
) -> pl.DataFrame:
    """Build a per-node attribute frame by left-joining GLP onto ``node_ids``.

    Warns when the GLP frame's node set diverges from the kept node set
    (mirrors ``_prepare_node_data`` in the export module).
    """
    base = pl.DataFrame({"node_id": node_ids})
    if glp_df is None:
        return base

    node_set = set(node_ids)
    glp_set = set(glp_df["node_id"].to_list())
    missing_in_glp = node_set - glp_set
    if missing_in_glp:
        sample = list(missing_in_glp)[:5]
        logger.warning(
            f"GLP results missing for {len(missing_in_glp)} kept nodes "
            f"(sample: {sample}). Those rows will have null GLP attributes."
        )

    extra_in_glp = glp_set - node_set
    if extra_in_glp:
        logger.debug(
            f"GLP results contain {len(extra_in_glp)} nodes not in the reduced graph; "
            f"those rows are dropped."
        )

    return base.join(glp_df, on="node_id", how="left")


# ---------------------------------------------------------------------------
# Louvain helpers (grouping)
# ---------------------------------------------------------------------------


def _run_louvain(
    graph: nk.Graph,
    resolution: float,
    random_seed: Optional[int],
) -> "nk.Partition":
    """Run NetworkIt's Louvain (PLM) at the given resolution and return the partition.

    NetworkIt's PLM seed is set via ``nk.setSeed`` at the module level when a
    ``random_seed`` is supplied. This is best-effort reproducibility — PLM is
    not strictly deterministic even at a fixed seed in all NetworkIt builds.
    """
    if random_seed is not None:
        try:
            nk.setSeed(int(random_seed), useThreadId=False)
        except Exception:
            # Older NetworkIt versions accept a single int; ignore on failure.
            pass
    plm = nk.community.PLM(graph, refine=True, gamma=float(resolution))
    plm.run()
    return plm.getPartition()


def _bisect_louvain_resolution(
    graph: nk.Graph,
    target_count: int,
    random_seed: Optional[int],
    *,
    low_gamma: float = 0.1,
    high_gamma: float = 10.0,
    max_iter: int = 8,
    tolerance: float = 0.10,
) -> Tuple["nk.Partition", float]:
    """Bisect Louvain's resolution (γ) until community count is near target.

    Best-effort: PLM is stochastic and the γ→community-count mapping is
    monotone-ish but not strictly so. Returns the partition closest to target
    after at most ``max_iter`` iterations.
    """
    best_partition = None
    best_K = -1
    best_diff = float("inf")
    best_gamma = (low_gamma + high_gamma) / 2.0

    for i in range(max_iter):
        gamma = (low_gamma + high_gamma) / 2.0
        partition = _run_louvain(
            graph, gamma, None if random_seed is None else int(random_seed) + i
        )
        K = partition.numberOfSubsets()
        diff = abs(K - target_count) / max(target_count, 1)

        if diff < best_diff:
            best_diff = diff
            best_partition = partition
            best_K = K
            best_gamma = gamma

        if diff <= tolerance:
            logger.debug(
                f"Louvain bisection hit target: γ={gamma:.4f}, K={K}, "
                f"target={target_count}, diff={diff:.3f}"
            )
            return partition, gamma

        if K < target_count:
            low_gamma = gamma
        else:
            high_gamma = gamma

    logger.warning(
        f"Louvain bisection did not converge to within {tolerance:.0%} of "
        f"target_nodes={target_count} after {max_iter} iterations. "
        f"Best: K={best_K} at γ={best_gamma:.4f} (diff={best_diff:.3f})."
    )
    return best_partition, best_gamma  # type: ignore[return-value]


def _build_quotient_graph(
    graph: nk.Graph,
    id_mapper: IDMapper,
    partition: "nk.Partition",
    *,
    keep_self_loops: bool,
    supernode_id_prefix: str,
) -> Tuple[nk.Graph, IDMapper, Dict[int, List[Any]]]:
    """Construct a supernode (quotient) graph from a partition.

    Returns the new graph, a fresh IDMapper keyed on synthetic supernode IDs
    (``f"{prefix}{k}"``), and a dict mapping each supernode's *internal* ID
    to the list of original member IDs (for downstream attribute aggregation).
    """
    # Map each partition's subset ID to a contiguous 0..K-1 range so we can
    # use them as NetworkIt internal node IDs directly. NetworkIt's
    # ``Partition.getSubsetIds()`` returns a set of arbitrary subset labels.
    subset_ids = sorted(partition.getSubsetIds())
    subset_to_super = {sid: i for i, sid in enumerate(subset_ids)}
    K = len(subset_ids)

    new_graph = nk.Graph(
        n=K,
        weighted=True,  # quotient graph always carries weights (aggregated)
        directed=graph.isDirected(),
    )

    # Aggregate inter-community edge weights. For directed graphs (u→v) and
    # (v→u) are independent keys; for undirected, NetworkIt's iterEdges()
    # already yields each edge once.
    edge_acc: Dict[Tuple[int, int], float] = defaultdict(float)
    is_weighted = graph.isWeighted()
    for u, v in graph.iterEdges():
        c_u = subset_to_super[partition.subsetOf(u)]
        c_v = subset_to_super[partition.subsetOf(v)]
        w = graph.weight(u, v) if is_weighted else 1.0
        edge_acc[(c_u, c_v)] += w

    # Sanity check on self-loop magnitudes (cf. plan pitfall #4).
    inter_weights = [w for (a, b), w in edge_acc.items() if a != b]
    if inter_weights:
        median_inter = float(np.median(inter_weights))
        for (a, b), w in edge_acc.items():
            if a == b and median_inter > 0 and w > 100.0 * median_inter:
                logger.warning(
                    f"Supernode {a} has a self-loop weight ({w:.2f}) more than 100× "
                    f"the median inter-community edge weight ({median_inter:.2f}). "
                    f"This often signals a resolution that's too coarse."
                )
                break

    for (cu, cv), w in edge_acc.items():
        if cu == cv and not keep_self_loops:
            continue
        new_graph.addEdge(cu, cv, w)

    # Build supernode IDMapper.
    supernode_mapper = IDMapper()
    for k in range(K):
        supernode_mapper.add_mapping(f"{supernode_id_prefix}{k}", k)

    # Build member_originals_by_community lookup. Iterating once over all
    # graph nodes via the IDMapper is O(N).
    member_originals_by_community: Dict[int, List[Any]] = defaultdict(list)
    for internal_id in range(graph.numberOfNodes()):
        if not graph.hasNode(internal_id):
            continue
        super_idx = subset_to_super[partition.subsetOf(internal_id)]
        try:
            orig = id_mapper.get_original(internal_id)
        except KeyError:
            continue
        member_originals_by_community[super_idx].append(orig)

    return new_graph, supernode_mapper, dict(member_originals_by_community)


def _aggregate_supernode_attrs(
    member_originals_by_community: Dict[int, List[Any]],
    K: int,
    supernode_id_prefix: str,
    glp_df: Optional[pl.DataFrame],
) -> pl.DataFrame:
    """Build the per-supernode attribute frame.

    Always includes ``node_id``, ``member_count``, and pipe-delimited
    ``member_ids``. If ``glp_df`` is provided, additionally aggregates:
    mean per-label probabilities, majority ``dominant_label`` (ties broken
    lexicographically), mean ``confidence``, and ``seed_fraction`` (mean of
    ``is_seed`` as a boolean).
    """
    # Build a flat membership frame: (node_id, community_idx)
    flat_rows_node_ids: List[Any] = []
    flat_rows_communities: List[int] = []
    for c_idx in range(K):
        members = member_originals_by_community.get(c_idx, [])
        flat_rows_node_ids.extend(members)
        flat_rows_communities.extend([c_idx] * len(members))

    membership = pl.DataFrame(
        {"node_id": flat_rows_node_ids, "__community": flat_rows_communities}
    )

    # Base aggregation: member_count and member_ids.
    base = membership.group_by("__community").agg(
        pl.col("node_id").len().alias("member_count"),
        pl.col("node_id").cast(pl.Utf8).alias("__member_id_list"),
    ).with_columns(
        pl.col("__member_id_list").list.join("|").alias("member_ids")
    ).drop("__member_id_list")

    if glp_df is not None:
        joined = membership.join(glp_df, on="node_id", how="left")

        agg_exprs: List[pl.Expr] = []
        # Per-label probability columns: take the mean.
        prob_cols = _glp_prob_columns(glp_df)
        for col in prob_cols:
            agg_exprs.append(pl.col(col).mean().alias(col))

        # dominant_label: majority vote with lexicographic tiebreak.
        for dom_col in ("dominant_label", "out_dominant_label", "in_dominant_label"):
            if dom_col in glp_df.columns:
                agg_exprs.append(
                    pl.col(dom_col).drop_nulls().mode().sort().first().alias(dom_col)
                )

        # confidence: mean.
        for conf_col in ("confidence", "out_confidence", "in_confidence"):
            if conf_col in glp_df.columns:
                agg_exprs.append(pl.col(conf_col).mean().alias(conf_col))

        # is_seed: mean (= fraction of seeds in the supernode).
        for seed_col in ("is_seed", "out_is_seed", "in_is_seed"):
            if seed_col in glp_df.columns:
                fraction_alias = (
                    "seed_fraction"
                    if seed_col == "is_seed"
                    else f"{seed_col.replace('is_seed', 'seed_fraction')}"
                )
                agg_exprs.append(
                    pl.col(seed_col).cast(pl.Float64).mean().alias(fraction_alias)
                )

        if agg_exprs:
            glp_agg = joined.group_by("__community").agg(agg_exprs)
            base = base.join(glp_agg, on="__community", how="left")

    # Map community index back to supernode_id (original ID).
    base = base.with_columns(
        (pl.lit(supernode_id_prefix) + pl.col("__community").cast(pl.Utf8)).alias(
            "node_id"
        )
    ).drop("__community")

    # Reorder so node_id comes first.
    cols = ["node_id"] + [c for c in base.columns if c != "node_id"]
    return base.select(cols)


# ---------------------------------------------------------------------------
# Forest fire (sampling)
# ---------------------------------------------------------------------------


def _forest_fire_burn(
    graph: nk.Graph,
    target: int,
    forward_burn_mean: float,
    backward_burn_mean: float,
    seed_internals: Optional[List[int]],
    max_restarts: int,
    random_seed: Optional[int],
) -> Set[int]:
    """Forest Fire sampling (Leskovec & Faloutsos 2006).

    Returns a set of internal node IDs of size ``target`` (or fewer if the
    graph is too small / disconnected to reach the target before
    ``max_restarts`` ambassadors are exhausted).
    """
    N = graph.numberOfNodes()
    if target >= N:
        return set(range(N))

    directed = graph.isDirected()
    visited: Set[int] = set()

    rng = np.random.default_rng(random_seed)

    # Ambassador order: user-specified seeds first, then a random permutation
    # of the remaining nodes. We never call rng.choice on a per-restart
    # filtered list (O(N) per restart) — a single shuffle handles it.
    if seed_internals:
        ambassador_order: List[int] = [int(s) for s in seed_internals]
    else:
        ambassador_order = []
    perm = rng.permutation(N)
    ambassador_order.extend(int(x) for x in perm)

    p_f = 1.0 / (1.0 + forward_burn_mean) if forward_burn_mean > 0 else None
    p_b = 1.0 / (1.0 + backward_burn_mean) if backward_burn_mean > 0 else None

    restarts = 0
    for amb in ambassador_order:
        if len(visited) >= target or restarts >= max_restarts:
            break
        if amb in visited:
            continue
        # Deterministic per-restart RNG so the BFS draws reproduce even when
        # restart count varies.
        burn_rng = (
            np.random.default_rng(int(random_seed) + restarts + 1)
            if random_seed is not None
            else rng
        )

        visited.add(amb)
        queue: deque[int] = deque([amb])

        while queue and len(visited) < target:
            u = queue.popleft()

            # Forward burn.
            if p_f is not None:
                out_unvisited = [
                    int(n) for n in graph.iterNeighbors(u) if n not in visited
                ]
                if out_unvisited:
                    n_fwd = max(0, int(burn_rng.geometric(p_f)) - 1)
                    n_fwd = min(n_fwd, len(out_unvisited))
                    if n_fwd > 0:
                        burned = burn_rng.choice(
                            out_unvisited, size=n_fwd, replace=False
                        )
                        for b in burned:
                            bi = int(b)
                            if bi not in visited:
                                visited.add(bi)
                                queue.append(bi)
                                if len(visited) >= target:
                                    break

            if not (directed and p_b is not None and len(visited) < target):
                continue

            # Backward burn (only meaningful for directed graphs).
            in_unvisited = [
                int(n) for n in graph.iterInNeighbors(u) if n not in visited
            ]
            if not in_unvisited:
                continue
            n_bwd = max(0, int(burn_rng.geometric(p_b)) - 1)
            n_bwd = min(n_bwd, len(in_unvisited))
            if n_bwd > 0:
                burned = burn_rng.choice(in_unvisited, size=n_bwd, replace=False)
                for b in burned:
                    bi = int(b)
                    if bi not in visited:
                        visited.add(bi)
                        queue.append(bi)
                        if len(visited) >= target:
                            break

        restarts += 1

    return visited


# ---------------------------------------------------------------------------
# Public reduce_by_grouping
# ---------------------------------------------------------------------------


def reduce_by_grouping(
    graph: nk.Graph,
    id_mapper: IDMapper,
    *,
    glp_results: Optional[GLPResult] = None,
    glp_direction: GLPDirection = "both",
    resolution: float = 1.0,
    target_nodes: Optional[int] = None,
    node_fraction: Optional[float] = None,
    target_edges: Optional[int] = None,
    edge_fraction: Optional[float] = None,
    resolution_search: Literal["fixed", "bisect"] = "fixed",
    keep_self_loops: bool = True,
    supernode_id_prefix: str = "community_",
    random_seed: Optional[int] = None,
) -> Tuple[nk.Graph, IDMapper, pl.DataFrame]:
    """
    Reduce ``graph`` by Louvain community detection + quotient graph construction.

    Each detected community becomes a supernode in the returned graph. Edges
    between supernodes carry the summed weight of inter-community edges; if
    ``keep_self_loops`` is True (default), the summed intra-community weight
    is attached as a supernode self-loop (Gephi renders this as a loop arc).

    Parameters
    ----------
    graph : nk.Graph
        Source NetworkIt graph.
    id_mapper : IDMapper
        Mapper for ``graph``'s original IDs.
    glp_results : pl.DataFrame or (pl.DataFrame, pl.DataFrame), optional
        Single-direction GLP frame or (out_df, in_df) tuple. When ``None``,
        the returned ``node_attrs_df`` contains only ``node_id``,
        ``member_count``, and ``member_ids``.
    glp_direction : {"out", "in", "both"}, default "both"
        Only consulted when ``glp_results`` is a tuple. "out"/"in" pick one
        frame; "both" merges them with ``out_``/``in_`` prefixes.
    resolution : float, default 1.0
        Louvain γ. Higher → more, smaller communities. Ignored when
        ``resolution_search="bisect"``.
    target_nodes : int, optional
        Desired number of supernodes. Mutually exclusive with ``node_fraction``.
    node_fraction : float, optional
        Desired supernode count as a fraction of ``graph.numberOfNodes()``.
    resolution_search : {"fixed", "bisect"}, default "fixed"
        ``"fixed"`` uses ``resolution`` as-is. ``"bisect"`` requires
        ``target_nodes`` or ``node_fraction`` and bisects γ ∈ [0.1, 10] over
        at most 8 iterations to land within ±10 % of the target. Bisection
        is best-effort: Louvain is stochastic, so the achieved supernode
        count is not guaranteed exact.
    keep_self_loops : bool, default True
        If True, attach summed intra-community edge weight as a supernode
        self-loop. Gephi can render these.
    supernode_id_prefix : str, default "community_"
        Synthetic supernode IDs are ``f"{prefix}{k}"`` for k=0..K-1.
    random_seed : int, optional
        Seed for Louvain. Best-effort reproducibility.

    Returns
    -------
    Tuple[nk.Graph, IDMapper, pl.DataFrame]
        The quotient graph, a fresh IDMapper keyed on synthetic supernode
        IDs, and a per-supernode attribute frame ready for export_graph's
        ``metadata`` argument.

    Raises
    ------
    ValidationError
        On conflicting/invalid target axes, invalid ``glp_direction``,
        or invalid ``resolution_search`` config (e.g. ``"bisect"`` without a
        target).
    ComputationError
        When the user demands more supernodes than the natural community
        structure can provide (e.g. ``target_nodes > 1`` when Louvain
        produces a single community).

    Notes
    -----
    Time complexity:
        Louvain at one γ is O(E). With bisection, multiply by up to 8.
        Quotient graph construction is O(E). Total: O(E) or O(8·E).
    Space complexity:
        O(N + K²) — quotient adjacency in the worst case.
    """
    log_function_entry(
        "reduce_by_grouping",
        graph_nodes=graph.numberOfNodes(),
        graph_edges=graph.numberOfEdges(),
        resolution=resolution,
        target_nodes=target_nodes,
        node_fraction=node_fraction,
        resolution_search=resolution_search,
    )

    # Validation: target_edges / edge_fraction are not native to grouping.
    _reject_unsupported_axes(
        target_edges=target_edges,
        edge_fraction=edge_fraction,
        method_name="reduce_by_grouping",
    )
    validate_parameter(
        resolution_search, _RESOLUTION_SEARCH_OPTIONS, "resolution_search", "reduce_by_grouping"
    )

    N = graph.numberOfNodes()
    target_count = _resolve_target_nodes(
        target_nodes, node_fraction, N, "reduce_by_grouping"
    )

    glp_df = _resolve_glp_for_attach(glp_results, glp_direction)

    # Empty graph.
    if N == 0:
        warnings.warn(
            "reduce_by_grouping: empty input graph; returning empty result.",
            UserWarning,
        )
        empty_graph = nk.Graph(0, weighted=True, directed=graph.isDirected())
        return empty_graph, IDMapper(), pl.DataFrame(
            {"node_id": [], "member_count": [], "member_ids": []}
        )

    # No-edge graph: every node would be its own community.
    if graph.numberOfEdges() == 0:
        warnings.warn(
            "reduce_by_grouping: graph has no edges; every node forms its own "
            "supernode (K == N).",
            UserWarning,
        )

    with LoggingTimer(
        "reduce_by_grouping",
        {"nodes": N, "edges": graph.numberOfEdges(), "target": target_count},
    ):
        if resolution_search == "bisect":
            if target_count is None:
                raise ValidationError(
                    "resolution_search='bisect' requires 'target_nodes' or "
                    "'node_fraction' to be specified.",
                    details={"function": "reduce_by_grouping"},
                )
            partition, _ = _bisect_louvain_resolution(graph, target_count, random_seed)
        else:
            partition = _run_louvain(graph, resolution, random_seed)

        K = partition.numberOfSubsets()

        if K == 1:
            warnings.warn(
                "reduce_by_grouping: Louvain detected a single community; "
                "returning a 1-supernode graph.",
                UserWarning,
            )
            if target_count is not None and target_count > 1:
                raise ComputationError(
                    f"Cannot produce {target_count} supernodes; only 1 community "
                    f"was detected. Try a larger 'resolution' or "
                    f"'resolution_search=\"bisect\"' with a larger target.",
                    operation="reduce_by_grouping",
                )

        # Resolution-search "fixed" sanity warning if target was given but
        # bisection wasn't requested.
        if (
            resolution_search == "fixed"
            and target_count is not None
            and abs(K - target_count) / max(target_count, 1) > 0.10
        ):
            logger.warning(
                f"reduce_by_grouping: achieved K={K} differs from "
                f"target={target_count} by more than 10%. Pass "
                f"resolution_search='bisect' to auto-tune γ."
            )

        new_graph, new_mapper, members_by_community = _build_quotient_graph(
            graph,
            id_mapper,
            partition,
            keep_self_loops=keep_self_loops,
            supernode_id_prefix=supernode_id_prefix,
        )

        attrs_df = _aggregate_supernode_attrs(
            members_by_community, K, supernode_id_prefix, glp_df
        )

    logger.info(
        f"reduce_by_grouping: {N} nodes → {K} supernodes, "
        f"{graph.numberOfEdges()} edges → {new_graph.numberOfEdges()} aggregated edges."
    )
    return new_graph, new_mapper, attrs_df


# ---------------------------------------------------------------------------
# Public reduce_by_influence
# ---------------------------------------------------------------------------


def reduce_by_influence(
    graph: nk.Graph,
    id_mapper: IDMapper,
    *,
    glp_results: Optional[GLPResult] = None,
    glp_direction: GLPDirection = "both",
    backbone_method: Literal[
        "disparity", "noise_corrected", "bipartite_svn", "weight", "degree"
    ] = "noise_corrected",
    alpha: float = 0.05,
    threshold: float = 1.0,
    target_nodes: Optional[int] = None,
    node_fraction: Optional[float] = None,
    target_edges: Optional[int] = None,
    edge_fraction: Optional[float] = None,
    keep_disconnected: bool = False,
) -> Tuple[nk.Graph, IDMapper, pl.DataFrame]:
    """
    Reduce ``graph`` by network backboning + optional top-degree node post-pass.

    Wraps :func:`~guidedLP.network.backboning.apply_backbone` with the chosen
    statistical filter, then (if a node target is provided) keeps only the
    top-degree nodes and rebuilds an induced subgraph.

    Parameters
    ----------
    graph : nk.Graph
        Source graph.
    id_mapper : IDMapper
        Mapper for ``graph``.
    glp_results : pl.DataFrame or (pl.DataFrame, pl.DataFrame), optional
        GLP results to attach. Directional tuples are merged per ``glp_direction``.
    glp_direction : {"out", "in", "both"}, default "both"
        See :func:`reduce_by_grouping`.
    backbone_method : str, default "noise_corrected"
        One of ``"disparity"``, ``"noise_corrected"``, ``"bipartite_svn"``,
        ``"weight"``, ``"degree"`` — passed through to ``apply_backbone``.
    alpha : float, default 0.05
        Significance level (used by ``"disparity"`` and ``"bipartite_svn"``).
    threshold : float, default 1.0
        Standard-deviation multiplier for ``"noise_corrected"``.
    target_edges : int, optional
        Native edge target for the backbone step (passed to ``apply_backbone``).
    edge_fraction : float, optional
        Edge target as a fraction of the input edge count.
    target_nodes : int, optional
        After backboning, keep only the top-``target_nodes`` highest-degree
        nodes and take their induced subgraph.
    node_fraction : float, optional
        Same as ``target_nodes`` but as a fraction of the *post-backbone*
        node count.
    keep_disconnected : bool, default False
        Forwarded to ``apply_backbone``.

    Returns
    -------
    Tuple[nk.Graph, IDMapper, pl.DataFrame]
        The reduced graph, its fresh IDMapper, and the GLP-attached
        ``node_attrs_df``.

    Raises
    ------
    ValidationError
        Conflicting target axes (e.g. both ``target_nodes`` and ``node_fraction``).
    ComputationError
        Propagated from ``apply_backbone`` (e.g. empty backbone).

    Notes
    -----
    Time complexity:
        Dominated by ``apply_backbone`` (O(E) or O(E log E) depending on method).
        Post-pass: O(N log N) for sorting degrees + O(E) for induced-subgraph
        construction.
    """
    log_function_entry(
        "reduce_by_influence",
        graph_nodes=graph.numberOfNodes(),
        graph_edges=graph.numberOfEdges(),
        backbone_method=backbone_method,
        target_edges=target_edges,
        target_nodes=target_nodes,
    )

    validate_parameter(
        backbone_method, _BACKBONE_METHODS, "backbone_method", "reduce_by_influence"
    )
    # Resolve targets (also runs cross-axis exclusivity validation).
    _resolve_target_edges(target_edges, edge_fraction, graph.numberOfEdges(), "reduce_by_influence")
    _resolve_target_nodes(target_nodes, node_fraction, graph.numberOfNodes(), "reduce_by_influence")

    N = graph.numberOfNodes()
    if N == 0:
        warnings.warn(
            "reduce_by_influence: empty input graph; returning empty result.",
            UserWarning,
        )
        empty = nk.Graph(0, weighted=graph.isWeighted(), directed=graph.isDirected())
        return empty, IDMapper(), pl.DataFrame({"node_id": []})

    glp_df = _resolve_glp_for_attach(glp_results, glp_direction)

    # Translate edge_fraction to absolute target_edges if needed; apply_backbone
    # accepts target_edges directly.
    abs_target_edges = _resolve_target_edges(
        target_edges, edge_fraction, graph.numberOfEdges(), "reduce_by_influence"
    )

    with LoggingTimer(
        "reduce_by_influence",
        {"nodes": N, "edges": graph.numberOfEdges(), "method": backbone_method},
    ):
        backbone_kwargs: Dict[str, Any] = {
            "method": backbone_method,
            "alpha": alpha,
            "threshold": threshold,
            "keep_disconnected": keep_disconnected,
            "verbose": False,
        }
        if abs_target_edges is not None and backbone_method in (
            "disparity",
            "weight",
        ):
            backbone_kwargs["target_edges"] = abs_target_edges

        result = apply_backbone(graph, id_mapper, **backbone_kwargs)
        # apply_backbone returns either (graph, mapper) or (graph, mapper, df).
        # We only need the first two.
        if len(result) == 2:
            backbone_graph, backbone_mapper = result
        else:
            backbone_graph, backbone_mapper = result[0], result[1]

        # Optional top-degree post-pass.
        abs_target_nodes = _resolve_target_nodes(
            target_nodes,
            node_fraction,
            backbone_graph.numberOfNodes(),
            "reduce_by_influence",
        )
        if (
            abs_target_nodes is not None
            and abs_target_nodes < backbone_graph.numberOfNodes()
        ):
            degrees = [
                (i, backbone_graph.degree(i))
                for i in range(backbone_graph.numberOfNodes())
                if backbone_graph.hasNode(i)
            ]
            degrees.sort(key=lambda t: (-t[1], t[0]))
            keep_internal = {t[0] for t in degrees[:abs_target_nodes]}
            backbone_graph, backbone_mapper = _build_induced_subgraph(
                backbone_graph, backbone_mapper, keep_internal
            )

        # Build node_attrs_df.
        kept_originals: List[Any] = [
            backbone_mapper.get_original(i)
            for i in range(backbone_graph.numberOfNodes())
            if backbone_graph.hasNode(i)
        ]
        attrs_df = _attach_glp_to_nodes(kept_originals, glp_df)

    logger.info(
        f"reduce_by_influence: {N} nodes / {graph.numberOfEdges()} edges → "
        f"{backbone_graph.numberOfNodes()} nodes / {backbone_graph.numberOfEdges()} edges "
        f"(method={backbone_method})."
    )
    return backbone_graph, backbone_mapper, attrs_df


# ---------------------------------------------------------------------------
# Public reduce_by_sampling
# ---------------------------------------------------------------------------


def reduce_by_sampling(
    graph: nk.Graph,
    id_mapper: IDMapper,
    *,
    glp_results: Optional[GLPResult] = None,
    glp_direction: GLPDirection = "both",
    target_nodes: Optional[int] = None,
    node_fraction: Optional[float] = None,
    target_edges: Optional[int] = None,
    edge_fraction: Optional[float] = None,
    forward_burn_mean: float = 1.5,
    backward_burn_mean: float = 0.4,
    seed_nodes: Optional[List[Any]] = None,
    max_restarts: Optional[int] = None,
    random_seed: Optional[int] = None,
) -> Tuple[nk.Graph, IDMapper, pl.DataFrame]:
    """
    Reduce ``graph`` by Forest Fire sampling (Leskovec & Faloutsos 2006).

    Forest Fire is a probabilistic BFS: at each visited node, draw the
    number of out-neighbors to "burn" from a geometric distribution with
    mean ``forward_burn_mean``. For directed graphs, an analogous backward
    burn (``backward_burn_mean``) covers in-neighbors. Burning is sampled
    *without replacement*. When the fire dies out (queue empty) and the
    target hasn't been reached, restart from a fresh random ambassador.

    With ``forward_burn_mean=1.5``, ``E[burns] ≈ 1.5``, which corresponds
    roughly to Leskovec's recommended ``p_f=0.7``. Increase the mean to
    spread the fire wider; decrease for tighter local sampling.

    Parameters
    ----------
    graph : nk.Graph
        Source graph.
    id_mapper : IDMapper
        Mapper for ``graph``.
    glp_results : pl.DataFrame or tuple, optional
        GLP results to attach.
    glp_direction : {"out", "in", "both"}, default "both"
        See :func:`reduce_by_grouping`.
    target_nodes : int, optional
        Number of nodes to keep. Required (or supply ``node_fraction``).
    node_fraction : float, optional
        Fraction of ``graph``'s nodes to keep.
    forward_burn_mean : float, default 1.5
        Geometric-distribution mean for forward burn. ``0`` disables it.
    backward_burn_mean : float, default 0.4
        Geometric-distribution mean for backward burn (directed graphs only).
        ``0`` disables it; ignored for undirected graphs.
    seed_nodes : list of original IDs, optional
        Initial ambassadors (in priority order). When exhausted, random
        ambassadors are drawn.
    max_restarts : int, optional
        Cap on the number of ambassador restarts. Default: ``target_nodes``
        (or its resolved value from ``node_fraction``).
    random_seed : int, optional
        Seed for reproducibility. Each restart re-seeds deterministically
        via ``random_seed + restart_count + 1``.

    Returns
    -------
    Tuple[nk.Graph, IDMapper, pl.DataFrame]
        The sampled induced subgraph, its fresh IDMapper, and the
        GLP-attached ``node_attrs_df``.

    Raises
    ------
    ValidationError
        Missing or conflicting target axes; ``target_edges``/``edge_fraction``
        rejected (not native to Forest Fire).
    ComputationError
        When ``max_restarts`` is exhausted before ``target_nodes`` is reached.

    Notes
    -----
    Time complexity:
        O(target_nodes · ⟨deg⟩) in expectation. Each visited node performs
        an O(deg) neighbor scan plus an O(burn_count) sample.
    """
    log_function_entry(
        "reduce_by_sampling",
        graph_nodes=graph.numberOfNodes(),
        graph_edges=graph.numberOfEdges(),
        target_nodes=target_nodes,
        forward_burn_mean=forward_burn_mean,
        backward_burn_mean=backward_burn_mean,
    )

    _reject_unsupported_axes(
        target_edges=target_edges,
        edge_fraction=edge_fraction,
        method_name="reduce_by_sampling",
    )

    if forward_burn_mean < 0 or backward_burn_mean < 0:
        raise ValidationError(
            "Burn means must be non-negative",
            details={"function": "reduce_by_sampling"},
        )

    N = graph.numberOfNodes()
    target_count = _resolve_target_nodes(
        target_nodes, node_fraction, N, "reduce_by_sampling"
    )

    if N == 0:
        warnings.warn(
            "reduce_by_sampling: empty input graph; returning empty result.",
            UserWarning,
        )
        empty = nk.Graph(0, weighted=graph.isWeighted(), directed=graph.isDirected())
        return empty, IDMapper(), pl.DataFrame({"node_id": []})

    if target_count is None:
        raise ValidationError(
            "reduce_by_sampling requires 'target_nodes' or 'node_fraction'",
            details={"function": "reduce_by_sampling"},
        )

    if target_count >= N:
        warnings.warn(
            f"reduce_by_sampling: target_nodes ({target_count}) >= "
            f"graph.numberOfNodes() ({N}); returning the input graph as-is.",
            UserWarning,
        )
        glp_df = _resolve_glp_for_attach(glp_results, glp_direction)
        kept_originals = [
            id_mapper.get_original(i)
            for i in range(N)
            if graph.hasNode(i)
        ]
        attrs_df = _attach_glp_to_nodes(kept_originals, glp_df)
        # Build a fresh mapper that is a copy of id_mapper for symmetry with
        # the reduction path (no shared mutable state).
        fresh = IDMapper()
        for orig, internal in id_mapper.original_to_internal.items():
            fresh.add_mapping(orig, internal)
        return graph, fresh, attrs_df

    glp_df = _resolve_glp_for_attach(glp_results, glp_direction)

    # Translate seed_nodes (originals) to internals.
    seed_internals: Optional[List[int]] = None
    if seed_nodes:
        seed_internals = []
        for s in seed_nodes:
            try:
                seed_internals.append(id_mapper.get_internal(s))
            except KeyError:
                logger.warning(
                    f"reduce_by_sampling: seed_node {s!r} not in id_mapper; skipping."
                )

    resolved_max_restarts = (
        int(max_restarts) if max_restarts is not None else max(1, target_count)
    )

    with LoggingTimer(
        "reduce_by_sampling",
        {"nodes": N, "target": target_count, "edges": graph.numberOfEdges()},
    ):
        visited = _forest_fire_burn(
            graph,
            target_count,
            forward_burn_mean,
            backward_burn_mean,
            seed_internals,
            resolved_max_restarts,
            random_seed,
        )

        if len(visited) < target_count:
            raise ComputationError(
                f"Forest Fire sampling terminated with {len(visited)} nodes "
                f"(target was {target_count}). The graph may be disconnected "
                f"with no component large enough, or max_restarts="
                f"{resolved_max_restarts} is too low.",
                operation="reduce_by_sampling",
            )

        new_graph, new_mapper = _build_induced_subgraph(graph, id_mapper, visited)

        kept_originals = [
            new_mapper.get_original(i)
            for i in range(new_graph.numberOfNodes())
            if new_graph.hasNode(i)
        ]
        attrs_df = _attach_glp_to_nodes(kept_originals, glp_df)

    logger.info(
        f"reduce_by_sampling: {N} nodes / {graph.numberOfEdges()} edges → "
        f"{new_graph.numberOfNodes()} nodes / {new_graph.numberOfEdges()} edges."
    )
    return new_graph, new_mapper, attrs_df


# ---------------------------------------------------------------------------
# Public export_reduced_graph
# ---------------------------------------------------------------------------


def export_reduced_graph(
    graph: nk.Graph,
    id_mapper: IDMapper,
    output_path: Union[str, Path],
    method: ReductionMethod,
    *,
    glp_results: Optional[GLPResult] = None,
    glp_direction: GLPDirection = "both",
    format: str = "gexf",
    extra_metadata: Optional[pl.DataFrame] = None,
    overwrite: bool = False,
    return_reduced: bool = False,
    **method_kwargs: Any,
) -> Union[Path, Tuple[Path, nk.Graph, IDMapper, pl.DataFrame]]:
    """
    Reduce ``graph`` by the chosen method, attach GLP results, and export to file.

    One-call wrapper around the three ``reduce_by_*`` functions and
    :func:`~guidedLP.network.export.export_graph`.

    Parameters
    ----------
    graph : nk.Graph
        Source graph.
    id_mapper : IDMapper
        Mapper for ``graph``.
    output_path : str or Path
        Destination file. Extension is auto-added per ``format`` if missing.
    method : {"grouping", "influence", "sampling"}
        Which reduction approach to apply.
    glp_results : pl.DataFrame or (pl.DataFrame, pl.DataFrame), optional
        GLP results to attach to the reduced graph's nodes.
    glp_direction : {"out", "in", "both"}, default "both"
        Directional GLP handling. Ignored when ``glp_results`` is a single frame.
    format : str, default "gexf"
        Output format. Forwarded to :func:`export_graph`.
    extra_metadata : pl.DataFrame, optional
        Additional per-node metadata (keyed by ``node_id``) to merge in.
        Useful for grouping reductions where you want to attach external
        annotations to specific supernodes by ID.
    overwrite : bool, default False
        Forwarded to :func:`export_graph`.
    return_reduced : bool, default False
        If True, also return the reduced graph, its mapper, and the attribute
        frame (useful for debugging / chaining).
    **method_kwargs
        Forwarded to the underlying ``reduce_by_*``. See each function for
        valid keyword arguments.

    Returns
    -------
    Path
        The output path written. When ``return_reduced=True``: a 4-tuple of
        ``(path, reduced_graph, reduced_mapper, node_attrs_df)``.

    Raises
    ------
    ValidationError
        Invalid ``method``, or per-method invalid arguments (forwarded).
    """
    log_function_entry(
        "export_reduced_graph",
        method=method,
        format=format,
        output_path=str(output_path),
        has_glp=glp_results is not None,
        has_extra_metadata=extra_metadata is not None,
    )

    validate_parameter(method, _REDUCTION_METHODS, "method", "export_reduced_graph")

    if method == "grouping":
        reduced_graph, reduced_mapper, attrs_df = reduce_by_grouping(
            graph,
            id_mapper,
            glp_results=glp_results,
            glp_direction=glp_direction,
            **method_kwargs,
        )
    elif method == "influence":
        reduced_graph, reduced_mapper, attrs_df = reduce_by_influence(
            graph,
            id_mapper,
            glp_results=glp_results,
            glp_direction=glp_direction,
            **method_kwargs,
        )
    else:  # sampling
        reduced_graph, reduced_mapper, attrs_df = reduce_by_sampling(
            graph,
            id_mapper,
            glp_results=glp_results,
            glp_direction=glp_direction,
            **method_kwargs,
        )

    # Merge extra_metadata if provided.
    final_metadata = attrs_df
    if extra_metadata is not None:
        if not isinstance(extra_metadata, pl.DataFrame):
            raise ValidationError(
                "extra_metadata must be a Polars DataFrame",
                field="extra_metadata",
            )
        if "node_id" not in extra_metadata.columns:
            raise ValidationError(
                "extra_metadata must contain a 'node_id' column",
                field="extra_metadata",
            )
        final_metadata = attrs_df.join(extra_metadata, on="node_id", how="left")

    out_str = str(output_path)
    export_graph(
        reduced_graph,
        reduced_mapper,
        out_str,
        format=format,
        metadata=final_metadata,
        overwrite=overwrite,
    )

    final_path = Path(out_str)
    if not final_path.suffix:
        ext = "csv" if format == "edgelist" else format
        final_path = final_path.with_suffix(f".{ext}")

    if return_reduced:
        return final_path, reduced_graph, reduced_mapper, final_metadata
    return final_path
