"""
Data preprocessing module.

Helper utilities that convert typical raw inputs (e.g. social-media post tables)
into the bipartite edge-list shape that ``network.construction.build_graph_from_edgelist``
expects. Functions in this module take a Polars DataFrame describing senders and
their posts and produce a long-form DataFrame mapping each sender to the content
elements (URLs, domains, keywords, embedding features, ...) they shared, preserving
timestamps.

The output schemas are designed to drop directly into the rest of the library:

>>> from guidedLP.preprocessing import extract_domains
>>> from guidedLP.network import build_graph_from_edgelist
>>> edges = extract_domains(posts_df)               # ["sender", "domain", "datetime"]
>>> g, mapper = build_graph_from_edgelist(
...     edges, source_col="sender", target_col="domain", bipartite=True
... )

Two flavors of extraction are currently supported, both operating on a
``[sender, post, datetime]`` post table:

- **Literal text content** (``extract_urls``, ``extract_domains``,
  ``extract_keywords``) — vectorized regex over the post column.
- **Semantic embedding features** (``extract_embedding_features``) — each
  post is mapped to an embedding vector (encoded on-the-fly or supplied as a
  column), aggregated per sender, and unrolled into sender→dimension edges.
"""

from .text_extraction import (
    extract_urls,
    extract_domains,
    extract_keywords,
)
from .embedding_extraction import extract_embedding_features

__all__ = [
    "extract_urls",
    "extract_domains",
    "extract_keywords",
    "extract_embedding_features",
]
