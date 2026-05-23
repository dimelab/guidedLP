#!/usr/bin/env python3
"""
Basic Network Analysis Example

This example demonstrates the fundamental network analysis workflow using the
Guided Label Propagation library. It shows how to:

1. Load network data from an edge list
2. Build a graph with proper ID mapping
3. Calculate multiple centrality metrics
4. Analyze network structure
5. Export results for further analysis

The example uses a social network dataset representing friendships and
acquaintances among a group of people.
"""

import sys
from pathlib import Path

# Add the src directory to the Python path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import polars as pl
import numpy as np
from src.network.construction import build_graph_from_edgelist
from src.network.analysis import extract_centrality
from src.network.communities import detect_communities
from src.network.export import export_graph


def main():
    """Main function demonstrating basic network analysis workflow."""
    
    print("=" * 60)
    print("Basic Network Analysis Example")
    print("=" * 60)
    
    # Step 1: Load the network data
    print("\n1. Loading Social Network Data")
    print("-" * 40)
    
    # Path to the sample social network data
    data_path = Path(__file__).parent / "data" / "social_network.csv"
    
    # Load and examine the data
    edge_data = pl.read_csv(data_path)
    print(f"Loaded {len(edge_data)} edges from {data_path}")
    print("\nFirst few edges:")
    print(edge_data.head())
    
    # Get basic statistics about the data
    unique_nodes = set(edge_data["source"].to_list() + edge_data["target"].to_list())
    print(f"\nNetwork contains {len(unique_nodes)} unique nodes")
    print(f"Edge weights range from {edge_data['weight'].min():.1f} to {edge_data['weight'].max():.1f}")
    
    # Step 2: Build the network graph
    print("\n2. Building Network Graph")
    print("-" * 40)
    
    # Build graph from edge list with weight information
    graph, id_mapper = build_graph_from_edgelist(
        edgelist=str(data_path),
        source_col="source",
        target_col="target",
        weight_col="weight",
        directed=False,  # Treat as undirected friendship network
        allow_self_loops=False  # Remove any self-connections
    )
    
    print(f"Built graph with {graph.numberOfNodes()} nodes and {graph.numberOfEdges()} edges")
    print(f"Graph is {'weighted' if graph.isWeighted() else 'unweighted'}")
    print(f"Graph is {'directed' if graph.isDirected() else 'undirected'}")
    
    # Calculate basic network statistics
    density = graph.numberOfEdges() / (graph.numberOfNodes() * (graph.numberOfNodes() - 1) / 2)
    print(f"Network density: {density:.3f}")
    
    # Step 3: Calculate centrality metrics
    print("\n3. Calculating Centrality Metrics")
    print("-" * 40)
    
    # Calculate multiple centrality measures to understand node importance
    centrality_metrics = [
        "degree",        # Number of connections
        "betweenness",   # How often a node lies on shortest paths
        "closeness",     # How close a node is to all others
        "eigenvector"    # Connections to well-connected nodes
    ]
    
    print(f"Calculating centrality metrics: {', '.join(centrality_metrics)}")
    
    centrality_df = extract_centrality(
        graph=graph,
        id_mapper=id_mapper,
        metrics=centrality_metrics
    )
    
    print(f"Calculated centrality for {len(centrality_df)} nodes")
    print("\nCentrality results:")
    print(centrality_df.sort("degree_centrality", descending=True))
    
    # Step 4: Identify most central nodes
    print("\n4. Analyzing Network Structure")
    print("-" * 40)
    
    # Find the most central nodes by different measures
    most_connected = centrality_df.sort("degree_centrality", descending=True).head(3)
    most_between = centrality_df.sort("betweenness_centrality", descending=True).head(3)
    most_close = centrality_df.sort("closeness_centrality", descending=True).head(3)
    
    print("Most connected nodes (by degree):")
    for row in most_connected.iter_rows(named=True):
        print(f"  {row['node_id']}: {row['degree_centrality']:.3f}")
    
    print("\nMost central nodes (by betweenness):")
    for row in most_between.iter_rows(named=True):
        print(f"  {row['node_id']}: {row['betweenness_centrality']:.3f}")
    
    print("\nMost accessible nodes (by closeness):")
    for row in most_close.iter_rows(named=True):
        print(f"  {row['node_id']}: {row['closeness_centrality']:.3f}")
    
    # Step 5: Detect communities
    print("\n5. Community Detection")
    print("-" * 40)
    
    try:
        # Detect communities using the default algorithm
        communities_df = detect_communities(
            graph=graph,
            id_mapper=id_mapper
        )
        
        print(f"Detected {communities_df['community'].n_unique()} communities")
        
        # Show community membership
        community_summary = communities_df.group_by("community").agg([
            pl.col("node_id").count().alias("size"),
            pl.col("node_id").list()
        ]).sort("size", descending=True)
        
        print("\nCommunity structure:")
        for row in community_summary.iter_rows(named=True):
            members = ", ".join(row["node_id"])
            print(f"  Community {row['community']} ({row['size']} members): {members}")
        
    except Exception as e:
        print(f"Community detection failed: {e}")
        print("This might be due to the small network size or other factors")
    
    # Step 6: Calculate additional network metrics
    print("\n6. Network-Level Analysis")
    print("-" * 40)
    
    # Calculate basic network metrics manually
    try:
        # Calculate average degree
        degrees = [graph.degree(i) for i in range(graph.numberOfNodes())]
        avg_degree = sum(degrees) / len(degrees)
        max_degree = max(degrees)
        
        print(f"Average degree: {avg_degree:.3f}")
        print(f"Maximum degree: {max_degree}")
        
        # Find most connected nodes
        degree_map = {id_mapper.get_original(i): graph.degree(i) for i in range(graph.numberOfNodes())}
        sorted_by_degree = sorted(degree_map.items(), key=lambda x: x[1], reverse=True)
        
        print("\nMost connected nodes (by raw degree):")
        for node_id, degree in sorted_by_degree[:3]:
            print(f"  {node_id}: {degree} connections")
            
    except Exception as e:
        print(f"Basic metrics calculation failed: {e}")
    
    # Set avg_clustering for later use
    avg_clustering = 0.0
    
    # Step 7: Export results
    print("\n7. Exporting Results")
    print("-" * 40)
    
    # Create output directory
    output_dir = Path(__file__).parent / "output"
    output_dir.mkdir(exist_ok=True)
    
    # Export the graph with centrality information
    graph_output = output_dir / "social_network.gexf"
    export_graph(
        graph=graph,
        id_mapper=id_mapper,
        output_path=str(graph_output),
        format="gexf",
        metadata=centrality_df,
        overwrite=True
    )
    print(f"Exported graph to: {graph_output}")
    
    # Export centrality results as CSV
    centrality_output = output_dir / "centrality_analysis.csv"
    centrality_df.write_csv(centrality_output)
    print(f"Exported centrality analysis to: {centrality_output}")
    
    # Export network summary
    summary_output = output_dir / "network_summary.txt"
    with open(summary_output, "w") as f:
        f.write("Social Network Analysis Summary\n")
        f.write("=" * 40 + "\n\n")
        f.write(f"Nodes: {graph.numberOfNodes()}\n")
        f.write(f"Edges: {graph.numberOfEdges()}\n")
        f.write(f"Density: {density:.3f}\n")
        f.write(f"Average clustering: {avg_clustering:.3f}\n")
        f.write("\nMost Central Nodes:\n")
        for row in most_between.head(5).iter_rows(named=True):
            f.write(f"  {row['node_id']}: {row['betweenness_centrality']:.3f}\n")
    
    print(f"Exported network summary to: {summary_output}")
    
    # Step 8: Provide analysis insights
    print("\n8. Analysis Insights")
    print("-" * 40)
    
    # Get the most central person
    top_central = most_between.row(0, named=True)
    most_connected_person = most_connected.row(0, named=True)
    
    print("Key findings from the social network analysis:")
    print(f"• Most influential person (betweenness): {top_central['node_id']}")
    print(f"• Most connected person (degree): {most_connected_person['node_id']}")
    print(f"• Network shows {'high' if avg_clustering > 0.3 else 'moderate' if avg_clustering > 0.1 else 'low'} clustering")
    print(f"• Network density of {density:.1%} suggests {'dense' if density > 0.3 else 'moderate' if density > 0.1 else 'sparse'} connectivity")
    
    print("\n" + "=" * 60)
    print("Network analysis complete!")
    print("Check the 'output' directory for exported files.")
    print("=" * 60)


if __name__ == "__main__":
    main()