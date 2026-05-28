"""End-to-end pipeline wrappers for guidedLP.

Pipelines compose the lower-level building blocks (``build_edgelist_from_frame``,
``apply_backbone``, ``project_bipartite`` / ``temporal_bipartite_to_unipartite``,
``guided_label_propagation``) into single-call workflows with explicit
memory management between stages.

Current pipelines:

- :func:`run_canonical_pipeline` — raw input → bipartite EdgeList →
  bipartite backbone → temporal projection → projection backbone.
  Produces a **directed** citation-style attribution graph (later
  sharer → earlier sharer). Useful for PageRank / HITS-flavored
  attribution analyses.

- :func:`run_undirected_bipartite_pipeline` — raw input → bipartite
  EdgeList → bipartite backbone → undirected shared-neighbor projection
  → projection backbone. Produces an **undirected** co-occurrence
  graph weighted by jaccard / count / overlap similarity. Useful for
  symmetric similarity / community-style analyses.
"""

from guidedLP.pipelines.canonical import (
    CanonicalPipelineResult,
    run_canonical_pipeline,
)
from guidedLP.pipelines.undirected_bipartite import (
    UndirectedBipartitePipelineResult,
    run_undirected_bipartite_pipeline,
)
from guidedLP.pipelines._runtime import StageStats

__all__ = [
    "CanonicalPipelineResult",
    "StageStats",
    "UndirectedBipartitePipelineResult",
    "run_canonical_pipeline",
    "run_undirected_bipartite_pipeline",
]
