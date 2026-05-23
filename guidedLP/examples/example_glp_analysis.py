#!/usr/bin/env python3
"""
Guided Label Propagation Analysis Example

This example demonstrates how to use Guided Label Propagation (GLP) for 
semi-supervised community detection. It shows how to:

1. Load network and seed data
2. Run guided label propagation
3. Validate results with different approaches
4. Analyze prediction confidence and quality
5. Create visualizations (if matplotlib available)
6. Export results with community assignments

The example uses a social network with known community seeds and predicts
the community membership of unlabeled nodes.
"""

import sys
from pathlib import Path
import warnings

# Add the src directory to the Python path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import polars as pl
import numpy as np
from src.network.construction import build_graph_from_edgelist
from src.glp.propagation import guided_label_propagation
from src.glp.validation import train_test_split_validation, cross_validate
from src.glp.evaluation import analyze_label_distribution
from src.glp.utils import create_balanced_seed_set, get_seed_statistics

# Optional imports for visualization
try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Note: matplotlib not available. Visualization will be skipped.")


def create_visualization(graph, id_mapper, results, metadata_df, output_dir):
    """Create network visualization with community colors."""
    if not HAS_MATPLOTLIB:
        print("Skipping visualization - matplotlib not available")
        return
    
    try:
        import networkx as nx
        
        # Convert NetworkIt graph to NetworkX for visualization
        nx_graph = nx.Graph()
        
        # Add nodes
        for i in range(graph.numberOfNodes()):
            original_id = id_mapper.get_original(i)
            nx_graph.add_node(original_id)
        
        # Add edges
        for edge in graph.iterEdges():
            u_orig = id_mapper.get_original(edge[0])
            v_orig = id_mapper.get_original(edge[1])
            weight = graph.weight(edge[0], edge[1]) if graph.isWeighted() else 1.0
            nx_graph.add_edge(u_orig, v_orig, weight=weight)
        
        # Create layout
        pos = nx.spring_layout(nx_graph, k=2, iterations=50)
        
        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        # Plot 1: Network with predicted communities
        ax1.set_title("Predicted Communities (GLP Results)", fontsize=14, fontweight='bold')
        
        # Color map for communities
        communities = results["dominant_label"].unique().to_list()
        colors = plt.cm.Set3(np.linspace(0, 1, len(communities)))
        color_map = {comm: colors[i] for i, comm in enumerate(communities)}
        
        # Draw nodes colored by predicted community
        for row in results.iter_rows(named=True):
            node_id = row["node_id"]
            community = row["dominant_label"]
            confidence = row["confidence"]
            is_seed = row.get("is_seed", False)
            
            # Node size based on confidence
            size = 300 + (confidence * 500)
            
            # Node color based on community
            color = color_map[community]
            
            # Border for seed nodes
            edgecolor = 'black' if is_seed else 'gray'
            linewidth = 3 if is_seed else 1
            
            nx.draw_networkx_nodes(
                nx_graph, pos, nodelist=[node_id], 
                node_color=[color], node_size=size,
                edgecolors=edgecolor, linewidths=linewidth, ax=ax1
            )
        
        # Draw edges
        nx.draw_networkx_edges(nx_graph, pos, alpha=0.5, ax=ax1)
        
        # Draw labels
        nx.draw_networkx_labels(nx_graph, pos, font_size=8, ax=ax1)
        
        # Add legend for communities
        legend_elements = [patches.Patch(color=color_map[comm], label=comm) 
                          for comm in communities]
        legend_elements.append(patches.Patch(facecolor='white', edgecolor='black', 
                                           linewidth=3, label='Seed nodes'))
        ax1.legend(handles=legend_elements, loc='upper right')
        ax1.axis('off')
        
        # Plot 2: Ground truth communities (if available)
        ax2.set_title("Ground Truth Communities", fontsize=14, fontweight='bold')
        
        # Join with metadata to get true communities
        results_with_truth = results.join(metadata_df, on="node_id", how="left")
        
        if "known_community" in results_with_truth.columns:
            true_communities = results_with_truth["known_community"].unique().to_list()
            true_color_map = {comm: colors[i % len(colors)] for i, comm in enumerate(true_communities)}
            
            # Draw nodes colored by true community
            for row in results_with_truth.iter_rows(named=True):
                node_id = row["node_id"]
                true_community = row.get("known_community", "unknown")
                is_seed = row.get("is_seed", False)
                
                if true_community != "unknown" and true_community is not None:
                    color = true_color_map[true_community]
                    size = 400
                    
                    edgecolor = 'black' if is_seed else 'gray'
                    linewidth = 3 if is_seed else 1
                    
                    nx.draw_networkx_nodes(
                        nx_graph, pos, nodelist=[node_id],
                        node_color=[color], node_size=size,
                        edgecolors=edgecolor, linewidths=linewidth, ax=ax2
                    )
            
            # Draw edges and labels
            nx.draw_networkx_edges(nx_graph, pos, alpha=0.5, ax=ax2)
            nx.draw_networkx_labels(nx_graph, pos, font_size=8, ax=ax2)
            
            # Add legend
            true_legend = [patches.Patch(color=true_color_map[comm], label=comm) 
                          for comm in true_communities if comm != "unknown"]
            true_legend.append(patches.Patch(facecolor='white', edgecolor='black', 
                                           linewidth=3, label='Seed nodes'))
            ax2.legend(handles=true_legend, loc='upper right')
        else:
            ax2.text(0.5, 0.5, 'Ground truth not available', 
                    transform=ax2.transAxes, ha='center', va='center')
        
        ax2.axis('off')
        
        plt.tight_layout()
        
        # Save visualization
        viz_output = output_dir / "community_visualization.png"
        plt.savefig(viz_output, dpi=300, bbox_inches='tight')
        print(f"Saved visualization to: {viz_output}")
        
        # Show plot if in interactive mode
        try:
            plt.show()
        except:
            pass  # May fail in non-interactive environments
            
    except ImportError:
        print("NetworkX not available for visualization")
    except Exception as e:
        print(f"Visualization failed: {e}")


def main():
    """Main function demonstrating GLP analysis workflow."""
    
    print("=" * 70)
    print("Guided Label Propagation Analysis Example")
    print("=" * 70)
    
    # Step 1: Load network and metadata
    print("\n1. Loading Network and Metadata")
    print("-" * 50)
    
    # Load the network data
    network_path = Path(__file__).parent / "data" / "social_network.csv"
    metadata_path = Path(__file__).parent / "data" / "user_metadata.csv"
    seeds_path = Path(__file__).parent / "data" / "community_seeds.csv"
    
    # Load data
    network_data = pl.read_csv(network_path)
    metadata_df = pl.read_csv(metadata_path)
    seeds_df = pl.read_csv(seeds_path)
    
    print(f"Loaded network with {len(network_data)} edges")
    print(f"Loaded metadata for {len(metadata_df)} users")
    print(f"Loaded {len(seeds_df)} seed labels")
    
    print("\nSeed nodes and their communities:")
    for row in seeds_df.iter_rows(named=True):
        print(f"  {row['node_id']}: {row['community_label']}")
    
    # Step 2: Build the graph
    print("\n2. Building Network Graph")
    print("-" * 50)
    
    graph, id_mapper = build_graph_from_edgelist(
        edgelist=str(network_path),
        source_col="source",
        target_col="target",
        weight_col="weight",
        directed=False
    )
    
    print(f"Built graph: {graph.numberOfNodes()} nodes, {graph.numberOfEdges()} edges")
    
    # Step 3: Prepare seed labels
    print("\n3. Preparing Seed Labels")
    print("-" * 50)
    
    # Convert seeds to the format expected by GLP
    seed_labels = {}
    all_labels = set()
    
    for row in seeds_df.iter_rows(named=True):
        node_id = row["node_id"]
        label = row["community_label"]
        if node_id in id_mapper.original_to_internal:
            seed_labels[node_id] = label
            all_labels.add(label)
    
    labels_list = sorted(list(all_labels))
    print(f"Using {len(seed_labels)} seed nodes for {len(labels_list)} communities")
    print(f"Communities: {', '.join(labels_list)}")
    
    # Get seed statistics
    seed_stats = get_seed_statistics(seed_labels, labels_list)
    print(f"\nSeed statistics:")
    print(f"  Total seeds: {seed_stats['total_seeds']}")
    print(f"  Balance ratio: {seed_stats['balance_ratio']:.3f}")
    print(f"  Is balanced: {seed_stats['is_balanced']}")
    if 'label_counts' in seed_stats:
        print(f"  Label distribution: {seed_stats['label_counts']}")
    
    # Step 4: Run Guided Label Propagation
    print("\n4. Running Guided Label Propagation")
    print("-" * 50)
    
    # Run GLP with different alpha values to show effect
    alpha_values = [0.5, 0.85, 0.95]
    glp_results = {}
    
    for alpha in alpha_values:
        print(f"\nRunning GLP with alpha = {alpha}")
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # Suppress convergence warnings
            
            results = guided_label_propagation(
                graph=graph,
                id_mapper=id_mapper,
                seed_labels=seed_labels,
                labels=labels_list,
                alpha=alpha,
                max_iterations=100,
                convergence_threshold=1e-6,
                normalize=True,
                directional=False  # Use non-directional for simplicity
            )
        
        glp_results[alpha] = results
        
        # Show basic results
        avg_confidence = results["confidence"].mean()
        print(f"  Average confidence: {avg_confidence:.3f}")
        
        # Show community distribution
        community_counts = results.group_by("dominant_label").agg(
            pl.col("node_id").count().alias("count")
        ).sort("dominant_label")
        
        print("  Community assignments:")
        for row in community_counts.iter_rows(named=True):
            print(f"    {row['dominant_label']}: {row['count']} nodes")
    
    # Use the best alpha (0.85) for detailed analysis
    best_results = glp_results[0.85]
    
    # Step 5: Analyze Results Quality
    print("\n5. Analyzing Prediction Quality")
    print("-" * 50)
    
    # Analyze label distribution
    label_analysis = analyze_label_distribution(
        predictions=best_results,
        labels=labels_list
    )
    
    print("Label distribution analysis:")
    print(f"  Label entropy: {label_analysis['label_entropy']:.3f}")
    if 'total_entropy' in label_analysis:
        print(f"  Total entropy: {label_analysis['total_entropy']:.3f}")
    else:
        print(f"  Available keys: {list(label_analysis.keys())}")
    
    print("\nConfidence by community:")
    for community, confidence in label_analysis["confidence_by_label"].items():
        print(f"  {community}: {confidence:.3f}")
    
    # Identify low confidence nodes
    low_confidence_nodes = best_results.filter(pl.col("confidence") < 0.6)
    if len(low_confidence_nodes) > 0:
        print(f"\nNodes with low confidence (< 0.6): {len(low_confidence_nodes)}")
        for row in low_confidence_nodes.iter_rows(named=True):
            print(f"  {row['node_id']}: {row['confidence']:.3f} ({row['dominant_label']})")
    
    # Step 6: Validation
    print("\n6. Validation Analysis")
    print("-" * 50)
    
    try:
        # Perform train/test split validation
        print("Running train/test split validation...")
        
        validation_results = train_test_split_validation(
            graph=graph,
            id_mapper=id_mapper,
            seed_labels=seed_labels,
            labels=labels_list,
            test_ratio=0.3,
            alpha=0.85,
            random_seed=42
        )
        
        print(f"Validation accuracy: {validation_results['accuracy']:.3f}")
        print(f"Validation F1-score: {validation_results['f1_weighted']:.3f}")
        
        # Show per-class performance
        if 'classification_report' in validation_results:
            print("\nPer-community performance:")
            for label in labels_list:
                if label in validation_results['classification_report']:
                    metrics = validation_results['classification_report'][label]
                    print(f"  {label}: F1={metrics['f1-score']:.3f}, "
                          f"Precision={metrics['precision']:.3f}, "
                          f"Recall={metrics['recall']:.3f}")
    
    except Exception as e:
        print(f"Validation failed: {e}")
        print("This might be due to insufficient seed data for validation")
    
    # Step 7: Cross-validation (if enough data)
    try:
        print("\nRunning cross-validation...")
        cv_results = cross_validate(
            graph=graph,
            id_mapper=id_mapper,
            seed_labels=seed_labels,
            labels=labels_list,
            k_folds=3,
            alpha=0.85,
            random_seed=42
        )
        
        print(f"Cross-validation accuracy: {cv_results['mean_accuracy']:.3f} ± {cv_results['std_accuracy']:.3f}")
        print(f"Cross-validation F1-score: {cv_results['mean_f1_weighted']:.3f} ± {cv_results['std_f1_weighted']:.3f}")
        
    except Exception as e:
        print(f"Cross-validation failed: {e}")
    
    # Step 8: Compare with ground truth (if available)
    print("\n7. Ground Truth Comparison")
    print("-" * 50)
    
    # Add ground truth comparison
    results_with_metadata = best_results.join(metadata_df, on="node_id", how="left")
    
    if "known_community" in results_with_metadata.columns:
        # Calculate accuracy against ground truth
        correct_predictions = 0
        total_predictions = 0
        
        for row in results_with_metadata.iter_rows(named=True):
            predicted = row["dominant_label"]
            true_label = row.get("known_community")
            
            if true_label is not None and true_label != "unknown":
                total_predictions += 1
                if predicted == true_label:
                    correct_predictions += 1
        
        if total_predictions > 0:
            accuracy = correct_predictions / total_predictions
            print(f"Ground truth accuracy: {accuracy:.3f} ({correct_predictions}/{total_predictions})")
            
            # Show misclassified nodes
            misclassified = results_with_metadata.filter(
                (pl.col("dominant_label") != pl.col("known_community")) &
                (pl.col("known_community") != "unknown") &
                (pl.col("known_community").is_not_null())
            )
            
            if len(misclassified) > 0:
                print(f"\nMisclassified nodes ({len(misclassified)}):")
                for row in misclassified.iter_rows(named=True):
                    print(f"  {row['node_id']}: predicted={row['dominant_label']}, "
                          f"true={row['known_community']}, confidence={row['confidence']:.3f}")
        else:
            print("No ground truth available for comparison")
    
    # Step 8: Export results
    print("\n8. Exporting Results")
    print("-" * 50)
    
    # Create output directory
    output_dir = Path(__file__).parent / "output"
    output_dir.mkdir(exist_ok=True)
    
    # Export detailed results
    results_output = output_dir / "glp_community_results.csv"
    
    # Add seed indicator and metadata
    final_results = best_results.with_columns([
        pl.col("node_id").map_elements(
            lambda x: x in seed_labels,
            return_dtype=pl.Boolean
        ).alias("is_seed")
    ]).join(metadata_df, on="node_id", how="left")
    
    final_results.write_csv(results_output)
    print(f"Exported detailed results to: {results_output}")
    
    # Export alpha comparison
    alpha_comparison = []
    for alpha, results in glp_results.items():
        for row in results.iter_rows(named=True):
            alpha_comparison.append({
                "alpha": alpha,
                "node_id": row["node_id"],
                "predicted_community": row["dominant_label"],
                "confidence": row["confidence"]
            })
    
    alpha_df = pl.DataFrame(alpha_comparison)
    alpha_output = output_dir / "alpha_comparison.csv"
    alpha_df.write_csv(alpha_output)
    print(f"Exported alpha comparison to: {alpha_output}")
    
    # Step 9: Create visualization
    print("\n9. Creating Visualization")
    print("-" * 50)
    
    create_visualization(graph, id_mapper, final_results, metadata_df, output_dir)
    
    # Step 10: Summary insights
    print("\n10. Analysis Summary")
    print("-" * 50)
    
    print("Key findings from the GLP analysis:")
    
    # Best performing alpha
    best_alpha = max(glp_results.keys(), 
                    key=lambda a: glp_results[a]["confidence"].mean())
    print(f"• Best performing alpha: {best_alpha}")
    
    # Community sizes
    community_sizes = best_results.group_by("dominant_label").agg(
        pl.col("node_id").count().alias("size")
    ).sort("size", descending=True)
    
    largest_community = community_sizes.row(0, named=True)
    print(f"• Largest predicted community: {largest_community['dominant_label']} "
          f"({largest_community['size']} members)")
    
    # Confidence analysis
    high_confidence = len(best_results.filter(pl.col("confidence") > 0.8))
    print(f"• High confidence predictions (>0.8): {high_confidence}/{len(best_results)} "
          f"({high_confidence/len(best_results):.1%})")
    
    # Seed effectiveness
    seed_nodes = best_results.filter(pl.col("node_id").is_in(list(seed_labels.keys())))
    seed_confidence = seed_nodes["confidence"].mean()
    print(f"• Average seed node confidence: {seed_confidence:.3f}")
    
    if 'accuracy' in locals():
        print(f"• Ground truth accuracy: {accuracy:.1%}")
    
    print("\n" + "=" * 70)
    print("GLP analysis complete!")
    print("Check the 'output' directory for detailed results and visualizations.")
    print("=" * 70)


if __name__ == "__main__":
    main()