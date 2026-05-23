"""
Network construction and analysis module.

This module provides core network analysis capabilities:
- Graph construction from edge lists (unipartite and bipartite)
- Bipartite graph projection
- Centrality measure calculations (degree, betweenness, closeness, eigenvector, pagerank, katz)
- Community detection using Louvain algorithm with consensus and stability analysis
- Network filtering by various criteria (degree, weight, centrality, components)
- Network backboning using disparity filter and other statistical methods
- Graph export functionality (GEXF, GraphML, edgelist, parquet formats)
"""

# Network construction functions
from .construction import (
    build_graph_from_edgelist,
    project_bipartite,
    get_graph_info,
    get_bipartite_info,
    validate_graph_construction
)

# Network analysis functions
from .analysis import (
    extract_centrality,
    get_centrality_summary,
    identify_central_nodes
)

# Community detection functions
from .communities import (
    detect_communities,
    get_community_summary,
    identify_stable_communities
)

# Network filtering and backboning functions
from .filtering import (
    filter_graph,
    apply_backbone,
    get_backbone_statistics
)

# Graph export functions
from .export import (
    export_graph,
    get_export_info
)