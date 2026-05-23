#!/usr/bin/env python3
"""
Temporal Network Analysis Example

This example demonstrates temporal network analysis capabilities using the
Guided Label Propagation library. It shows how to:

1. Load temporal network data with timestamps
2. Create temporal slices (daily intervals)
3. Extract centrality metrics over time
4. Calculate temporal statistics and trends
5. Analyze cross-category connections over time
6. Visualize temporal patterns
7. Export temporal analysis results

The example uses communication data showing how people interact over time,
allowing us to understand the evolution of network structure and influence.
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta

# Add the src directory to the Python path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import polars as pl
import numpy as np
from src.timeseries.slicing import create_temporal_slices, align_node_ids_across_slices
from src.timeseries.temporal_metrics import extract_temporal_metrics, calculate_temporal_statistics
from src.timeseries.category_analysis import (
    analyze_cross_category_connections, 
    calculate_category_segregation_index,
    analyze_category_centrality_by_time
)

# Optional imports for visualization
try:
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Note: matplotlib not available. Visualization will be skipped.")


def create_temporal_visualizations(temporal_metrics, temporal_stats, category_connections, output_dir):
    """Create visualizations of temporal network patterns."""
    if not HAS_MATPLOTLIB:
        print("Skipping visualization - matplotlib not available")
        return
    
    try:
        # Create figure with subplots
        fig = plt.figure(figsize=(16, 12))
        
        # Plot 1: Individual node centrality over time
        ax1 = plt.subplot(2, 3, 1)
        
        # Plot degree centrality for top 3 most central nodes
        degree_data = temporal_metrics.filter(pl.col("degree").is_not_null())
        if not degree_data.is_empty():
            # Get top nodes by average degree
            avg_degree = degree_data.group_by("node_id").agg(
                pl.col("degree").mean().alias("avg_degree")
            ).sort("avg_degree", descending=True).head(3)
            
            for row in avg_degree.iter_rows(named=True):
                node_id = row["node_id"]
                node_data = degree_data.filter(pl.col("node_id") == node_id).sort("date")
                
                dates = [datetime.fromisoformat(str(d)) for d in node_data["date"].to_list()]
                values = node_data["degree"].to_list()
                
                ax1.plot(dates, values, marker='o', label=node_id, linewidth=2)
            
            ax1.set_title("Node Degree Centrality Over Time", fontweight='bold')
            ax1.set_xlabel("Date")
            ax1.set_ylabel("Degree Centrality")
            ax1.legend()
            ax1.grid(True, alpha=0.3)
        
        # Plot 2: Network-level metrics over time
        ax2 = plt.subplot(2, 3, 2)
        
        # Calculate daily network metrics
        daily_metrics = temporal_metrics.group_by("date").agg([
            pl.col("degree").mean().alias("avg_degree"),
            pl.col("betweenness").mean().alias("avg_betweenness"),
            pl.col("node_id").count().alias("active_nodes")
        ]).sort("date")
        
        if not daily_metrics.is_empty():
            dates = [datetime.fromisoformat(str(d)) for d in daily_metrics["date"].to_list()]
            
            ax2_twin = ax2.twinx()
            
            # Plot average degree
            line1 = ax2.plot(dates, daily_metrics["avg_degree"].to_list(), 
                           'b-o', label="Avg Degree", linewidth=2)
            
            # Plot active nodes on secondary axis
            line2 = ax2_twin.plot(dates, daily_metrics["active_nodes"].to_list(), 
                                'r-s', label="Active Nodes", linewidth=2)
            
            ax2.set_title("Daily Network Metrics", fontweight='bold')
            ax2.set_xlabel("Date")
            ax2.set_ylabel("Average Degree", color='b')
            ax2_twin.set_ylabel("Active Nodes", color='r')
            
            # Combine legends
            lines = line1 + line2
            labels = [l.get_label() for l in lines]
            ax2.legend(lines, labels, loc='upper left')
            
            ax2.grid(True, alpha=0.3)
        
        # Plot 3: Temporal statistics (trends)
        ax3 = plt.subplot(2, 3, 3)
        
        if not temporal_stats.is_empty():
            # Plot trend values for degree centrality
            trend_data = temporal_stats.filter(pl.col("degree_trend").is_not_null()).sort("degree_trend", descending=True)
            
            if len(trend_data) > 0:
                # Show top nodes with strongest trends (positive and negative)
                top_positive = trend_data.head(3)
                top_negative = trend_data.tail(3)
                
                nodes = top_positive["node_id"].to_list() + top_negative["node_id"].to_list()
                trends = top_positive["degree_trend"].to_list() + top_negative["degree_trend"].to_list()
                colors = ['green'] * len(top_positive) + ['red'] * len(top_negative)
                
                bars = ax3.barh(range(len(nodes)), trends, color=colors, alpha=0.7)
                ax3.set_yticks(range(len(nodes)))
                ax3.set_yticklabels(nodes)
                ax3.set_xlabel("Degree Centrality Trend")
                ax3.set_title("Node Centrality Trends", fontweight='bold')
                ax3.axvline(x=0, color='black', linestyle='-', alpha=0.5)
                ax3.grid(True, alpha=0.3)
        
        # Plot 4: Category connections over time
        ax4 = plt.subplot(2, 3, 4)
        
        if not category_connections.is_empty():
            # Plot connections between different age groups over time
            age_connections = category_connections.filter(
                pl.col("category_a") != pl.col("category_b")  # Only cross-category
            ).group_by("date").agg(
                pl.col("connection_strength").sum().alias("cross_category_strength")
            ).sort("date")
            
            if not age_connections.is_empty():
                dates = [datetime.fromisoformat(str(d)) for d in age_connections["date"].to_list()]
                strengths = age_connections["cross_category_strength"].to_list()
                
                ax4.plot(dates, strengths, 'g-o', linewidth=2, markersize=6)
                ax4.set_title("Cross-Category Connections", fontweight='bold')
                ax4.set_xlabel("Date")
                ax4.set_ylabel("Connection Strength")
                ax4.grid(True, alpha=0.3)
        
        # Plot 5: Communication patterns
        ax5 = plt.subplot(2, 3, 5)
        
        # Load original temporal data to show communication patterns
        data_path = Path(__file__).parent / "data" / "temporal_communications.csv"
        comm_data = pl.read_csv(data_path)
        
        # Group by date and communication type
        daily_comm = comm_data.with_columns(
            pl.col("timestamp").str.to_datetime().dt.date().alias("date")
        ).group_by(["date", "communication_type"]).agg(
            pl.col("message_count").sum().alias("total_messages")
        ).sort("date")
        
        # Plot communication types over time
        comm_types = daily_comm["communication_type"].unique().to_list()
        colors = ['blue', 'orange']
        
        for i, comm_type in enumerate(comm_types):
            type_data = daily_comm.filter(pl.col("communication_type") == comm_type)
            if not type_data.is_empty():
                dates = type_data["date"].to_list()
                messages = type_data["total_messages"].to_list()
                
                ax5.plot(dates, messages, 'o-', color=colors[i % len(colors)], 
                        label=comm_type, linewidth=2, markersize=4)
        
        ax5.set_title("Communication Patterns", fontweight='bold')
        ax5.set_xlabel("Date")
        ax5.set_ylabel("Total Messages")
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        
        # Rotate x-axis labels for better readability
        for ax in [ax1, ax2, ax4, ax5]:
            ax.tick_params(axis='x', rotation=45)
        
        # Plot 6: Network evolution summary
        ax6 = plt.subplot(2, 3, 6)
        
        # Show key metrics evolution
        if not daily_metrics.is_empty():
            dates = [datetime.fromisoformat(str(d)) for d in daily_metrics["date"].to_list()]
            
            # Normalize metrics for comparison
            degree_norm = np.array(daily_metrics["avg_degree"].to_list())
            degree_norm = (degree_norm - degree_norm.min()) / (degree_norm.max() - degree_norm.min() + 1e-8)
            
            betweenness_norm = np.array(daily_metrics["avg_betweenness"].to_list())
            betweenness_norm = (betweenness_norm - betweenness_norm.min()) / (betweenness_norm.max() - betweenness_norm.min() + 1e-8)
            
            ax6.plot(dates, degree_norm, 'b-o', label="Degree (normalized)", linewidth=2)
            ax6.plot(dates, betweenness_norm, 'r-s', label="Betweenness (normalized)", linewidth=2)
            
            ax6.set_title("Network Evolution Summary", fontweight='bold')
            ax6.set_xlabel("Date")
            ax6.set_ylabel("Normalized Centrality")
            ax6.legend()
            ax6.grid(True, alpha=0.3)
            ax6.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        # Save visualization
        viz_output = output_dir / "temporal_analysis.png"
        plt.savefig(viz_output, dpi=300, bbox_inches='tight')
        print(f"Saved temporal visualization to: {viz_output}")
        
        # Show plot if in interactive mode
        try:
            plt.show()
        except:
            pass  # May fail in non-interactive environments
            
    except Exception as e:
        print(f"Visualization failed: {e}")


def main():
    """Main function demonstrating temporal network analysis workflow."""
    
    print("=" * 80)
    print("Temporal Network Analysis Example")
    print("=" * 80)
    
    # Step 1: Load temporal data
    print("\n1. Loading Temporal Communication Data")
    print("-" * 60)
    
    # Load the temporal communication data
    data_path = Path(__file__).parent / "data" / "temporal_communications.csv"
    metadata_path = Path(__file__).parent / "data" / "user_metadata.csv"
    
    comm_data = pl.read_csv(data_path)
    metadata_df = pl.read_csv(metadata_path)
    
    print(f"Loaded {len(comm_data)} communication events")
    print(f"Time range: {comm_data['timestamp'].min()} to {comm_data['timestamp'].max()}")
    
    # Show data structure
    print(f"\nFirst few communication events:")
    print(comm_data.head())
    
    # Get basic statistics
    unique_users = set(comm_data["source"].to_list() + comm_data["target"].to_list())
    unique_days = comm_data.with_columns(
        pl.col("timestamp").str.to_datetime().dt.date()
    )["timestamp"].n_unique()
    
    print(f"\nDataset summary:")
    print(f"  Unique users: {len(unique_users)}")
    print(f"  Unique days: {unique_days}")
    print(f"  Total messages: {comm_data['message_count'].sum()}")
    
    # Step 2: Create temporal slices
    print("\n2. Creating Temporal Network Slices")
    print("-" * 60)
    
    # Create daily temporal slices
    print("Creating daily temporal slices...")
    
    temporal_graphs = create_temporal_slices(
        edgelist=str(data_path),
        timestamp_col="timestamp",
        slice_interval="daily",
        rolling_window=None,
        cumulative=False,
        weight_col="message_count",
        directed=False  # Treat communication as undirected relationships
    )
    
    print(f"Created {len(temporal_graphs)} temporal slices")
    
    # Show slice information
    print("\nTemporal slice information:")
    for i, (date, graph, mapper) in enumerate(temporal_graphs):
        print(f"  {date.strftime('%Y-%m-%d')}: {graph.numberOfNodes()} nodes, "
              f"{graph.numberOfEdges()} edges")
    
    # Step 3: Align node IDs across slices
    print("\n3. Aligning Node IDs Across Time Slices")
    print("-" * 60)
    
    aligned_graphs, global_mapper = align_node_ids_across_slices(temporal_graphs)
    
    print(f"Aligned {len(aligned_graphs)} temporal slices")
    print(f"Global network contains {len(global_mapper.original_to_internal)} unique nodes")
    
    # Recreate temporal_graphs with aligned mappers for consistency
    aligned_temporal_graphs = []
    for i, ((date, _, _), (date_aligned, graph_aligned)) in enumerate(zip(temporal_graphs, aligned_graphs)):
        aligned_temporal_graphs.append((date, graph_aligned, global_mapper))
    
    # Step 4: Extract temporal metrics
    print("\n4. Extracting Temporal Centrality Metrics")
    print("-" * 60)
    
    # Extract centrality metrics across time
    centrality_metrics = ["degree", "betweenness", "closeness"]
    
    print(f"Calculating centrality metrics: {', '.join(centrality_metrics)}")
    
    temporal_metrics = extract_temporal_metrics(
        temporal_graphs=aligned_temporal_graphs,
        metrics=centrality_metrics,
        n_jobs=1  # Use single job for reproducibility
    )
    
    print(f"Extracted metrics for {len(temporal_metrics)} node-time combinations")
    
    if not temporal_metrics.is_empty():
        print("\nSample temporal metrics:")
        print(temporal_metrics.head(10))
        
        # Show metrics evolution for a specific node
        sample_node = temporal_metrics["node_id"].unique().to_list()[0]
        node_evolution = temporal_metrics.filter(pl.col("node_id") == sample_node).sort("date")
        
        if len(node_evolution) > 1:
            print(f"\nCentrality evolution for {sample_node}:")
            for row in node_evolution.iter_rows(named=True):
                print(f"  {row['date']}: degree={row['degree']:.3f}, "
                      f"betweenness={row['betweenness']:.3f}")
    
    # Step 5: Calculate temporal statistics
    print("\n5. Calculating Temporal Statistics")
    print("-" * 60)
    
    # Calculate various temporal statistics
    statistics_to_calculate = ["mean", "std", "trend", "volatility", "growth"]
    
    print(f"Calculating statistics: {', '.join(statistics_to_calculate)}")
    
    temporal_stats = calculate_temporal_statistics(
        temporal_metrics=temporal_metrics,
        statistics=statistics_to_calculate
    )
    
    if not temporal_stats.is_empty():
        print(f"Calculated temporal statistics for {len(temporal_stats)} nodes")
        
        # Show nodes with strongest trends
        print("\nNodes with strongest degree centrality trends:")
        trend_analysis = temporal_stats.select([
            "node_id", "degree_mean", "degree_trend", "degree_volatility"
        ]).sort("degree_trend", descending=True)
        
        print("Top increasing influence:")
        for row in trend_analysis.head(3).iter_rows(named=True):
            print(f"  {row['node_id']}: trend={row['degree_trend']:.4f}, "
                  f"avg_degree={row['degree_mean']:.3f}")
        
        print("Top decreasing influence:")
        for row in trend_analysis.tail(3).iter_rows(named=True):
            print(f"  {row['node_id']}: trend={row['degree_trend']:.4f}, "
                  f"avg_degree={row['degree_mean']:.3f}")
        
        # Show most volatile nodes
        print("\nMost volatile nodes (by degree centrality):")
        volatile_nodes = temporal_stats.sort("degree_volatility", descending=True).head(3)
        for row in volatile_nodes.iter_rows(named=True):
            print(f"  {row['node_id']}: volatility={row['degree_volatility']:.3f}")
    
    # Step 6: Analyze cross-category connections
    print("\n6. Analyzing Cross-Category Connections")
    print("-" * 60)
    
    # Analyze connections between different age groups over time
    category_connections = analyze_cross_category_connections(
        temporal_graphs=aligned_temporal_graphs,
        metadata=metadata_df,
        category_column="age_group",
        edge_weight="sum"
    )
    
    if not category_connections.is_empty():
        print(f"Analyzed {len(category_connections)} category connection patterns")
        
        print("\nCross-category connection evolution:")
        for row in category_connections.sort("date").iter_rows(named=True):
            print(f"  {row['date']}: {row['category_a']} ↔ {row['category_b']} "
                  f"(strength: {row['connection_strength']:.1f})")
        
        # Calculate segregation index
        segregation_analysis = calculate_category_segregation_index(category_connections)
        
        if not segregation_analysis.is_empty():
            print("\nSegregation analysis:")
            for row in segregation_analysis.iter_rows(named=True):
                print(f"  {row['date']}: segregation_index={row['segregation_index']:.3f}")
            
            avg_segregation = segregation_analysis["segregation_index"].mean()
            print(f"\nAverage segregation index: {avg_segregation:.3f}")
            print(f"Network shows {'high' if avg_segregation > 0.7 else 'moderate' if avg_segregation > 0.3 else 'low'} segregation")
    
    # Step 7: Analyze category centrality over time
    print("\n7. Category Centrality Analysis")
    print("-" * 60)
    
    try:
        category_centrality = analyze_category_centrality_by_time(
            temporal_graphs=aligned_temporal_graphs,
            metadata=metadata_df,
            category_column="age_group",
            centrality_metric="degree"
        )
    except Exception as e:
        print(f"Category centrality analysis failed: {e}")
        category_centrality = pl.DataFrame()
    
    if not category_centrality.is_empty():
        print("Average centrality by age group over time:")
        
        # Show centrality evolution by category
        categories = category_centrality["category"].unique().to_list()
        for category in categories:
            cat_data = category_centrality.filter(pl.col("category") == category).sort("date")
            print(f"\n{category} group:")
            for row in cat_data.iter_rows(named=True):
                print(f"  {row['date']}: avg_centrality={row['mean_centrality']:.3f}, "
                      f"members={row['node_count']}")
    
    # Step 8: Identify temporal patterns
    print("\n8. Identifying Temporal Patterns")
    print("-" * 60)
    
    # Analyze network growth over time
    daily_growth = []
    for date, graph, mapper in aligned_temporal_graphs:
        daily_growth.append({
            "date": date,
            "nodes": graph.numberOfNodes(),
            "edges": graph.numberOfEdges(),
            "density": graph.numberOfEdges() / max(1, graph.numberOfNodes() * (graph.numberOfNodes() - 1) / 2)
        })
    
    growth_df = pl.DataFrame(daily_growth).sort("date")
    
    print("Network growth over time:")
    for row in growth_df.iter_rows(named=True):
        print(f"  {row['date'].strftime('%Y-%m-%d')}: "
              f"{row['nodes']} nodes, {row['edges']} edges, "
              f"density={row['density']:.3f}")
    
    # Identify peak activity days
    peak_activity = growth_df.sort("edges", descending=True).head(2)
    print(f"\nPeak activity days:")
    for row in peak_activity.iter_rows(named=True):
        print(f"  {row['date'].strftime('%Y-%m-%d')}: {row['edges']} edges")
    
    # Step 9: Export results
    print("\n9. Exporting Temporal Analysis Results")
    print("-" * 60)
    
    # Create output directory
    output_dir = Path(__file__).parent / "output"
    output_dir.mkdir(exist_ok=True)
    
    # Export temporal metrics
    metrics_output = output_dir / "temporal_centrality_metrics.csv"
    temporal_metrics.write_csv(metrics_output)
    print(f"Exported temporal metrics to: {metrics_output}")
    
    # Export temporal statistics
    if not temporal_stats.is_empty():
        stats_output = output_dir / "temporal_statistics.csv"
        temporal_stats.write_csv(stats_output)
        print(f"Exported temporal statistics to: {stats_output}")
    
    # Export category analysis
    if not category_connections.is_empty():
        category_output = output_dir / "category_connections.csv"
        category_connections.write_csv(category_output)
        print(f"Exported category connections to: {category_output}")
    
    # Export network growth data
    growth_output = output_dir / "network_growth.csv"
    growth_df.write_csv(growth_output)
    print(f"Exported network growth to: {growth_output}")
    
    # Export enriched temporal data (with metadata)
    if not temporal_metrics.is_empty():
        enriched_metrics = temporal_metrics.join(metadata_df, on="node_id", how="left")
        enriched_output = output_dir / "enriched_temporal_metrics.csv"
        enriched_metrics.write_csv(enriched_output)
        print(f"Exported enriched metrics to: {enriched_output}")
    
    # Step 10: Create visualizations
    print("\n10. Creating Temporal Visualizations")
    print("-" * 60)
    
    create_temporal_visualizations(temporal_metrics, temporal_stats, category_connections, output_dir)
    
    # Step 11: Generate summary insights
    print("\n11. Temporal Analysis Summary")
    print("-" * 60)
    
    print("Key findings from the temporal network analysis:")
    
    # Activity patterns
    total_days = len(temporal_graphs)
    active_days = sum(1 for _, graph, _ in temporal_graphs if graph.numberOfEdges() > 0)
    print(f"• Network was active {active_days}/{total_days} days")
    
    # Growth patterns
    if len(growth_df) > 1:
        node_growth = growth_df["nodes"].max() - growth_df["nodes"].min()
        edge_growth = growth_df["edges"].max() - growth_df["edges"].min()
        print(f"• Node growth: {node_growth} nodes added over time")
        print(f"• Edge growth: {edge_growth} edges added over time")
    
    # Most influential nodes
    if not temporal_stats.is_empty():
        most_influential = temporal_stats.sort("degree_mean", descending=True).head(1)
        for row in most_influential.iter_rows(named=True):
            print(f"• Most consistently influential: {row['node_id']} "
                  f"(avg degree: {row['degree_mean']:.3f})")
        
        # Fastest growing influence
        fastest_growing = temporal_stats.sort("degree_trend", descending=True).head(1)
        for row in fastest_growing.iter_rows(named=True):
            if row['degree_trend'] > 0:
                print(f"• Fastest growing influence: {row['node_id']} "
                      f"(trend: +{row['degree_trend']:.4f})")
    
    # Category patterns
    if not category_connections.is_empty() and 'avg_segregation' in locals():
        print(f"• Average age group segregation: {avg_segregation:.3f}")
        
        # Most connected categories
        top_connection = category_connections.sort("connection_strength", descending=True).head(1)
        for row in top_connection.iter_rows(named=True):
            print(f"• Strongest category connection: {row['category_a']} ↔ {row['category_b']} "
                  f"(strength: {row['connection_strength']:.1f})")
    
    print("\n" + "=" * 80)
    print("Temporal network analysis complete!")
    print("Check the 'output' directory for detailed results and visualizations.")
    print("=" * 80)


if __name__ == "__main__":
    main()