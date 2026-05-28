#!/usr/bin/env python3
"""
Animate guided label propagation on a downscaled subgraph (sketch).

Pipeline:
  1. Forest-Fire-sample the full graph down to ~hundreds–thousands of
     nodes via ``reduce_by_sampling``, seeded with the GLP seeds so they
     end up in the sampled subgraph.
  2. Re-run propagation on the sampled subgraph, capturing the label
     matrix F at every iteration (not just the final state).
  3. Compute a static 2-D layout (NetworkX spring layout).
  4. Render with Plotly: edges static, node colors/sizes animated per
     iteration, with a slider + play button.

This is a sketch, not production code:
  - Reaches into ``glp.propagation`` private helpers
    (``_initialize_label_matrix``, ``_create_transition_matrix``) so we
    can keep every iteration's F. The "right" productionization is to
    add a ``return_snapshots`` option to ``guided_label_propagation``.
  - Undirected / single-pass only. For directional GLP you would render
    two animations side by side.
  - Layout via NetworkX spring layout is fine up to a few thousand
    nodes; beyond that, swap to a faster layout (e.g. precomputed
    coordinates from Gephi via ``export_reduced_graph``).
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))

import networkit as nk
import numpy as np

from guidedLP.common.id_mapper import IDMapper
from guidedLP.common.seed_input import normalize_seed_input
from guidedLP.glp.propagation import (
    _create_transition_matrix,
    _initialize_label_matrix,
)
from guidedLP.network.reduction import reduce_by_sampling

try:
    import networkx as nx
    import plotly.graph_objects as go
except ImportError as e:
    raise SystemExit(
        "This demo needs networkx and plotly: pip install networkx plotly"
    ) from e


# ---------------------------------------------------------------------------
# 1. Propagation with per-iteration snapshots
# ---------------------------------------------------------------------------

def propagate_with_snapshots(
    graph: nk.Graph,
    id_mapper: IDMapper,
    seed_labels: Dict[Any, str],
    labels: List[str],
    *,
    alpha: float = 0.85,
    max_iterations: int = 30,
    convergence_threshold: float = 1e-6,
) -> List[np.ndarray]:
    """Run the GLP fixed-point and return ``[F_0, F_1, ..., F_T]``.

    ``F_0`` is the seed matrix Y (pre-propagation); ``F_t`` for ``t >= 1``
    is the post-iteration state. Same update rule as
    :func:`guided_label_propagation` but keeps every snapshot.
    """
    Y = _initialize_label_matrix(graph, id_mapper, seed_labels, labels)
    P = _create_transition_matrix(graph, direction="out_degree")

    F = Y.copy()
    snapshots: List[np.ndarray] = [F.copy()]
    for _ in range(max_iterations):
        F_next = alpha * (P @ F) + (1.0 - alpha) * Y
        delta = float(np.max(np.abs(F_next - F)))
        F = F_next
        snapshots.append(F.copy())
        if delta < convergence_threshold:
            break
    return snapshots


# ---------------------------------------------------------------------------
# 2. Layout (computed once on the reduced subgraph)
# ---------------------------------------------------------------------------

def compute_layout(
    graph: nk.Graph,
    id_mapper: IDMapper,
    seed: int = 0,
) -> Dict[Any, Tuple[float, float]]:
    """Spring layout via NetworkX, keyed by original IDs."""
    G = nx.DiGraph() if graph.isDirected() else nx.Graph()
    is_weighted = graph.isWeighted()
    for u, v in graph.iterEdges():
        G.add_edge(
            id_mapper.get_original(u),
            id_mapper.get_original(v),
            weight=graph.weight(u, v) if is_weighted else 1.0,
        )
    for i in range(graph.numberOfNodes()):
        if graph.hasNode(i):
            G.add_node(id_mapper.get_original(i))
    return nx.spring_layout(G, seed=seed, weight="weight")


# ---------------------------------------------------------------------------
# 3. Plotly animation
# ---------------------------------------------------------------------------

_PALETTE = [
    "#636EFA", "#EF553B", "#00CC96", "#AB63FA", "#FFA15A",
    "#19D3F3", "#FF6692", "#B6E880", "#FF97FF", "#FECB52",
]


def _dominant_and_confidence(F: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (dom_idx, confidence, is_unlabeled) per node for a single F."""
    row_sums = F.sum(axis=1)
    unlabeled = row_sums == 0
    safe = np.where(unlabeled, 1.0, row_sums)[:, None]
    F_norm = F / safe
    dom_idx = np.argmax(F_norm, axis=1)
    conf = F_norm.max(axis=1)
    conf[unlabeled] = 0.0
    return dom_idx, conf, unlabeled


def build_animation(
    graph: nk.Graph,
    id_mapper: IDMapper,
    labels: List[str],
    snapshots: List[np.ndarray],
    layout: Dict[Any, Tuple[float, float]],
    seed_ids: List[Any],
    *,
    title: str = "Guided label propagation",
    frame_duration_ms: int = 400,
) -> go.Figure:
    n_nodes = graph.numberOfNodes()
    label_to_color = {lbl: _PALETTE[i % len(_PALETTE)] for i, lbl in enumerate(labels)}
    seed_set = set(seed_ids)

    # Static edge trace (one polyline with None separators between segments).
    edge_x: List[Any] = []
    edge_y: List[Any] = []
    for u, v in graph.iterEdges():
        x0, y0 = layout[id_mapper.get_original(u)]
        x1, y1 = layout[id_mapper.get_original(v)]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]
    edge_trace = go.Scatter(
        x=edge_x, y=edge_y, mode="lines",
        line=dict(width=0.4, color="rgba(140,140,140,0.4)"),
        hoverinfo="none", showlegend=False,
    )

    # Per-node coordinates, ordered by internal ID.
    node_x = np.zeros(n_nodes)
    node_y = np.zeros(n_nodes)
    node_orig: List[Any] = [None] * n_nodes
    is_seed_arr = np.zeros(n_nodes, dtype=bool)
    for i in range(n_nodes):
        if not graph.hasNode(i):
            continue
        o = id_mapper.get_original(i)
        node_orig[i] = o
        node_x[i], node_y[i] = layout[o]
        is_seed_arr[i] = o in seed_set

    def node_trace_for(F: np.ndarray) -> go.Scatter:
        dom_idx, conf, unlabeled = _dominant_and_confidence(F)
        colors = [
            "rgba(200,200,200,0.35)" if unlabeled[i]
            else label_to_color[labels[dom_idx[i]]]
            for i in range(n_nodes)
        ]
        sizes = np.where(is_seed_arr, 14.0, 6.0 + 10.0 * conf)
        hover = [
            (f"id: {node_orig[i]}<br>"
             f"label: {'—' if unlabeled[i] else labels[dom_idx[i]]}<br>"
             f"confidence: {conf[i]:.2f}"
             f"{'<br><b>SEED</b>' if is_seed_arr[i] else ''}")
            for i in range(n_nodes)
        ]
        return go.Scatter(
            x=node_x, y=node_y, mode="markers",
            marker=dict(
                size=sizes, color=colors,
                line=dict(width=0.5, color="white"),
            ),
            text=hover, hoverinfo="text", showlegend=False,
        )

    frames = [
        go.Frame(data=[edge_trace, node_trace_for(F)], name=f"iter_{t}")
        for t, F in enumerate(snapshots)
    ]

    # Invisible scatter points so each label gets a legend entry.
    legend_traces = [
        go.Scatter(
            x=[None], y=[None], mode="markers",
            marker=dict(size=10, color=label_to_color[lbl]),
            name=lbl, showlegend=True,
        )
        for lbl in labels
    ]

    fig = go.Figure(
        data=[edge_trace, frames[0].data[1], *legend_traces],
        frames=frames,
        layout=go.Layout(
            title=title,
            xaxis=dict(visible=False),
            yaxis=dict(visible=False, scaleanchor="x", scaleratio=1),
            hovermode="closest",
            margin=dict(l=20, r=20, t=60, b=20),
            updatemenus=[dict(
                type="buttons", showactive=False,
                x=0.05, y=-0.05, xanchor="left", yanchor="top",
                buttons=[
                    dict(label="Play", method="animate",
                         args=[None, dict(
                             frame=dict(duration=frame_duration_ms, redraw=True),
                             fromcurrent=True, mode="immediate")]),
                    dict(label="Pause", method="animate",
                         args=[[None], dict(
                             frame=dict(duration=0, redraw=False),
                             mode="immediate")]),
                ],
            )],
            sliders=[dict(
                active=0, x=0.1, y=-0.02, len=0.85,
                xanchor="left", yanchor="top",
                currentvalue=dict(prefix="iteration: ", visible=True),
                steps=[
                    dict(label=str(t), method="animate",
                         args=[[f.name], dict(
                             mode="immediate",
                             frame=dict(duration=0, redraw=True),
                             transition=dict(duration=0))])
                    for t, f in enumerate(frames)
                ],
            )],
        ),
    )
    return fig


# ---------------------------------------------------------------------------
# 4. End-to-end demo
# ---------------------------------------------------------------------------

def animate_propagation(
    full_graph: nk.Graph,
    full_mapper: IDMapper,
    seed_labels_raw: Any,
    labels: List[str],
    *,
    target_nodes: int = 800,
    alpha: float = 0.85,
    max_iterations: int = 30,
    output_html: str | Path = "propagation_demo.html",
) -> Path:
    seed_dict = normalize_seed_input(seed_labels_raw)
    seed_ids = list(seed_dict.keys())

    sub_graph, sub_mapper, _ = reduce_by_sampling(
        full_graph,
        full_mapper,
        target_nodes=target_nodes,
        seed_nodes=seed_ids,
        random_seed=42,
    )

    surviving = {
        node_id: lbl for node_id, lbl in seed_dict.items()
        if node_id in sub_mapper.original_to_internal
    }
    if not surviving:
        raise RuntimeError(
            "No seeds survived sampling — raise target_nodes or pass more seeds."
        )

    snapshots = propagate_with_snapshots(
        sub_graph, sub_mapper, surviving, labels,
        alpha=alpha, max_iterations=max_iterations,
    )
    layout = compute_layout(sub_graph, sub_mapper)

    fig = build_animation(
        sub_graph, sub_mapper, labels, snapshots, layout,
        seed_ids=list(surviving.keys()),
        title=(f"GLP propagation — {sub_graph.numberOfNodes()} nodes, "
               f"{sub_graph.numberOfEdges()} edges, "
               f"{len(snapshots) - 1} iterations"),
    )

    out = Path(output_html)
    fig.write_html(out, include_plotlyjs="cdn", auto_play=False)
    return out


if __name__ == "__main__":
    # Plug your own loading code in here. Minimal example below uses
    # construction.build_graph_from_edgelist on an existing dataset.
    from guidedLP.network.construction import build_graph_from_edgelist

    DATA_DIR = Path(__file__).parent / "data"
    edges_path = DATA_DIR / "social_network_edges.csv"
    if not edges_path.exists():
        raise SystemExit(
            f"Expected demo edgelist at {edges_path}. Point this script at "
            "your own (graph, id_mapper, seeds) triple instead."
        )

    graph, mapper = build_graph_from_edgelist(
        str(edges_path), source_col="source", target_col="target",
        weight_col="weight", directed=False,
    )

    # Pick a handful of nodes per label as seeds. Replace with your own.
    rng = np.random.default_rng(0)
    node_ids = [mapper.get_original(i) for i in range(graph.numberOfNodes())]
    chosen = rng.choice(node_ids, size=10, replace=False)
    seeds = {nid: ("A" if k < 5 else "B") for k, nid in enumerate(chosen)}

    out_path = animate_propagation(
        graph, mapper, seeds, labels=["A", "B"],
        target_nodes=500, output_html=Path(__file__).parent / "output" / "propagation_demo.html",
    )
    print(f"Wrote {out_path}")
