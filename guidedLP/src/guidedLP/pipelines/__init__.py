"""End-to-end pipeline wrappers for guidedLP.

Pipelines compose the lower-level building blocks (``build_edgelist_from_frame``,
``apply_backbone``, ``project_bipartite`` / ``temporal_bipartite_to_unipartite``,
``guided_label_propagation``) into single-call workflows with explicit
memory management between stages.

Current pipelines:

- :func:`run_canonical_pipeline` — raw input → bipartite EdgeList →
  bipartite backbone → temporal projection → projection backbone.
  Useful for attribution-style analyses (e.g. user-content sharing
  networks).
"""

from guidedLP.pipelines.canonical import (
    CanonicalPipelineResult,
    run_canonical_pipeline,
)
from guidedLP.pipelines._runtime import StageStats

__all__ = [
    "CanonicalPipelineResult",
    "StageStats",
    "run_canonical_pipeline",
]
