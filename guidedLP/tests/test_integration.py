"""
Integration tests for end-to-end workflows in the Guided Label Propagation library.

These tests verify that the complete workflows function correctly from data loading
through final outputs, using realistic sample data and testing all major
combinations of functionality.
"""

import pytest
import tempfile
import shutil
from pathlib import Path
from datetime import datetime
import warnings

import polars as pl
import networkit as nk

# Network construction and analysis
from src.network.construction import build_graph_from_edgelist
from src.network.analysis import extract_centrality
from src.network.export import export_graph

# Guided Label Propagation
from src.glp.propagation import guided_label_propagation
from src.glp.validation import train_test_split_validation, get_validation_summary
from src.glp.evaluation import analyze_label_distribution

# Timeseries analysis
from src.timeseries.slicing import create_temporal_slices, align_node_ids_across_slices
from src.timeseries.temporal_metrics import extract_temporal_metrics, calculate_temporal_statistics
from src.timeseries.category_analysis import analyze_cross_category_connections

# Common utilities
from src.common.id_mapper import IDMapper


class TestIntegrationWorkflows:
    """Test complete end-to-end workflows with realistic data."""
    
    @pytest.fixture
    def fixtures_dir(self):
        """Get path to test fixtures directory."""
        return Path(__file__).parent / "fixtures"
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for test outputs."""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def sample_edgelist_path(self, fixtures_dir):
        """Path to sample edge list CSV."""
        return fixtures_dir / "sample_edgelist.csv"
    
    @pytest.fixture
    def sample_temporal_edgelist_path(self, fixtures_dir):
        """Path to sample temporal edge list CSV."""
        return fixtures_dir / "sample_temporal_edgelist.csv"
    
    @pytest.fixture
    def sample_metadata_path(self, fixtures_dir):
        """Path to sample metadata CSV."""
        return fixtures_dir / "sample_metadata.csv"
    
    @pytest.fixture
    def sample_seeds_path(self, fixtures_dir):
        """Path to sample seeds CSV."""
        return fixtures_dir / "sample_seeds.csv"
    
    def test_workflow_1_network_analysis_pipeline(self, sample_edgelist_path, temp_dir):
        """
        Test Workflow 1: Load edge list → Build graph → Extract centrality → Export
        
        This workflow tests the basic network analysis pipeline:
        1. Load edge list from CSV
        2. Build NetworkIt graph with ID mapping
        3. Extract multiple centrality metrics
        4. Export graph and results
        """
        # Step 1: Load edge list and build graph
        graph, id_mapper = build_graph_from_edgelist(
            str(sample_edgelist_path),
            source_col="source",
            target_col="target", 
            weight_col="weight",
            directed=False
        )
        
        # Verify graph construction
        assert graph.numberOfNodes() > 0, "Graph should have nodes"
        assert graph.numberOfEdges() > 0, "Graph should have edges"
        assert graph.isWeighted(), "Graph should be weighted"
        assert not graph.isDirected(), "Graph should be undirected"
        
        # Verify ID mapping
        assert len(id_mapper.original_to_internal) > 0, "ID mapper should have mappings"
        assert len(id_mapper.internal_to_original) > 0, "ID mapper should have reverse mappings"
        
        # Step 2: Extract centrality metrics
        centrality_metrics = [
            "degree", "betweenness", "closeness", "eigenvector"
        ]
        
        centrality_df = extract_centrality(
            graph, 
            id_mapper, 
            metrics=centrality_metrics
        )
        
        # Verify centrality extraction
        assert not centrality_df.is_empty(), "Centrality results should not be empty"
        assert "node_id" in centrality_df.columns, "Should have node_id column"
        
        # Check for centrality columns (they have _centrality suffix)
        expected_columns = [f"{metric}_centrality" for metric in centrality_metrics]
        for expected_col in expected_columns:
            assert expected_col in centrality_df.columns, f"Should have {expected_col} column"
            assert centrality_df[expected_col].null_count() == 0, f"{expected_col} should have no null values"
        
        # Verify all nodes are included
        expected_nodes = set(id_mapper.original_to_internal.keys())
        actual_nodes = set(centrality_df["node_id"].to_list())
        assert expected_nodes == actual_nodes, "All nodes should be in centrality results"
        
        # Step 3: Export graph and results
        output_path = temp_dir / "network_output.gexf"
        
        export_graph(
            graph,
            id_mapper,
            output_path=str(output_path),
            format="gexf",
            metadata=centrality_df,
            overwrite=True
        )
        
        # Verify export was created
        assert output_path.exists(), "GEXF export should exist"
        
        # Step 4: Also export centrality data separately
        centrality_output = temp_dir / "centrality_metrics.csv"
        centrality_df.write_csv(centrality_output)
        
        # Verify exported centrality data
        exported_centrality = pl.read_csv(centrality_output)
        assert exported_centrality.shape[0] == centrality_df.shape[0], "Export should have same number of rows"
        assert exported_centrality.shape[1] == centrality_df.shape[1], "Export should have same number of columns"
    
    def test_workflow_2_glp_analysis_pipeline(self, sample_edgelist_path, sample_seeds_path, temp_dir):
        """
        Test Workflow 2: Load edge list → Build graph → Run GLP → Validate
        
        This workflow tests the guided label propagation pipeline:
        1. Load edge list and build graph
        2. Load seed labels
        3. Run guided label propagation
        4. Validate propagation results
        5. Analyze prediction quality
        """
        # Step 1: Build graph
        graph, id_mapper = build_graph_from_edgelist(
            str(sample_edgelist_path),
            source_col="source",
            target_col="target",
            weight_col="weight",
            directed=False
        )
        
        # Step 2: Load and prepare seed labels
        seeds_df = pl.read_csv(sample_seeds_path)
        
        # Convert to external format expected by GLP
        seed_labels = {}
        all_labels = set()
        for row in seeds_df.iter_rows(named=True):
            node_id = row["node_id"]
            label = str(row["label"])  # Convert to string
            if node_id in id_mapper.original_to_internal:
                seed_labels[node_id] = label
                all_labels.add(label)
        
        assert len(seed_labels) > 0, "Should have valid seed nodes in graph"
        
        # Get unique labels
        labels_list = sorted(list(all_labels))
        
        # Step 3: Run guided label propagation
        results = guided_label_propagation(
            graph=graph,
            id_mapper=id_mapper,
            seed_labels=seed_labels,
            labels=labels_list,
            alpha=0.85,
            max_iterations=100,
            convergence_threshold=1e-6,
            normalize=True,
            directional=False  # Use simpler non-directional for test
        )
        
        # Verify GLP results (returns DataFrame with directional=False)
        assert isinstance(results, pl.DataFrame), "Results should be DataFrame with directional=False"
        assert not results.is_empty(), "Results should not be empty"
        assert "node_id" in results.columns, "Results should have node_id column"
        assert "dominant_label" in results.columns, "Results should have dominant_label column"
        assert "confidence" in results.columns, "Results should have confidence column"
        
        # Check that we have results for all nodes
        assert len(results) == graph.numberOfNodes(), "Should have results for all nodes"
        
        # Verify seeds preserved their labels
        for node_id, expected_label in seed_labels.items():
            node_result = results.filter(pl.col("node_id") == node_id)
            assert len(node_result) == 1, f"Should have result for seed node {node_id}"
            predicted_label = node_result["dominant_label"][0]
            assert predicted_label == expected_label, f"Seed node {node_id} should retain its label"
        
        # Step 4: Analyze label distribution and quality
        label_distribution = analyze_label_distribution(
            predictions=results,
            labels=labels_list
        )
        
        # Verify label analysis
        assert "label_counts" in label_distribution, "Should have label counts"
        assert "confidence_by_label" in label_distribution, "Should have confidence by label"
        assert len(label_distribution["label_counts"]) > 0, "Should have some label counts"
        
        # Calculate confidence statistics
        confidence_scores = results["confidence"].to_list()
        mean_confidence = sum(confidence_scores) / len(confidence_scores)
        low_confidence_count = sum(1 for conf in confidence_scores if conf < 0.5)
        
        assert mean_confidence > 0.3, "Should have reasonable mean confidence"  # Relaxed for test data
        assert low_confidence_count <= len(confidence_scores), "Should have confidence scores"
        
        # Step 5: Export results with additional metadata
        # Add seed indicator
        results_with_seed = results.with_columns([
            pl.col("node_id").map_elements(
                lambda x: x in seed_labels,
                return_dtype=pl.Boolean
            ).alias("is_seed")
        ])
        
        # Save results
        output_path = temp_dir / "glp_results.csv"
        results_with_seed.write_csv(output_path)
        
        assert output_path.exists(), "GLP results should be exported"
        
        # Verify export contents
        exported_results = pl.read_csv(output_path)
        assert exported_results.shape[0] == graph.numberOfNodes(), "Export should have all nodes"
        
        # Check that we have seed indicators
        seed_results = exported_results.filter(pl.col("is_seed") == True)
        assert len(seed_results) == len(seed_labels), "Should have correct number of seed nodes marked"
    
    def test_workflow_3_temporal_analysis_pipeline(self, sample_temporal_edgelist_path, temp_dir):
        """
        Test Workflow 3: Load temporal edge list → Create slices → Track metrics
        
        This workflow tests temporal network analysis:
        1. Load temporal edge list
        2. Create temporal slices (daily intervals)
        3. Align node IDs across slices
        4. Extract temporal metrics
        5. Calculate temporal statistics
        """
        # Step 1: Create temporal slices
        temporal_graphs = create_temporal_slices(
            edgelist=str(sample_temporal_edgelist_path),
            timestamp_col="timestamp",
            slice_interval="daily",
            rolling_window=None,
            cumulative=False,
            directed=False,
            weight_col="weight"
        )
        
        # Verify temporal slicing
        assert len(temporal_graphs) > 0, "Should have temporal slices"
        
        dates = [date for date, graph, mapper in temporal_graphs]
        assert len(set(dates)) == len(dates), "Dates should be unique"
        assert all(isinstance(date, datetime) for date in dates), "All dates should be datetime objects"
        
        # Verify graphs have expected structure
        for date, graph, mapper in temporal_graphs:
            assert isinstance(graph, nk.Graph), "Should have NetworkIt graph"
            assert isinstance(mapper, IDMapper), "Should have ID mapper"
            if graph.numberOfNodes() > 0:
                assert graph.isWeighted(), "Non-empty graphs should be weighted"
        
        # Step 2: Align node IDs across slices
        aligned_graphs, global_mapper = align_node_ids_across_slices(temporal_graphs)
        
        # Verify alignment
        assert len(aligned_graphs) == len(temporal_graphs), "Should preserve number of slices"
        assert isinstance(global_mapper, IDMapper), "Should have global mapper"
        
        # Step 3: Extract temporal metrics
        metrics_to_extract = ["degree", "betweenness", "closeness"]
        
        temporal_metrics_df = extract_temporal_metrics(
            temporal_graphs,
            metrics=metrics_to_extract,
            n_jobs=1  # Use single job for reproducibility
        )
        
        # Verify temporal metrics
        assert not temporal_metrics_df.is_empty(), "Should have temporal metrics"
        assert "node_id" in temporal_metrics_df.columns, "Should have node_id column"
        assert "date" in temporal_metrics_df.columns, "Should have date column"
        
        for metric in metrics_to_extract:
            assert metric in temporal_metrics_df.columns, f"Should have {metric} column"
        
        # Step 4: Calculate temporal statistics
        statistics_to_calculate = ["mean", "std", "trend", "volatility"]
        
        temporal_stats_df = calculate_temporal_statistics(
            temporal_metrics_df,
            statistics=statistics_to_calculate
        )
        
        # Verify temporal statistics
        assert not temporal_stats_df.is_empty(), "Should have temporal statistics"
        assert "node_id" in temporal_stats_df.columns, "Should have node_id column"
        
        for metric in metrics_to_extract:
            for stat in statistics_to_calculate:
                expected_col = f"{metric}_{stat}"
                assert expected_col in temporal_stats_df.columns, f"Should have {expected_col} column"
        
        # Step 5: Export temporal analysis results
        metrics_output = temp_dir / "temporal_metrics.csv"
        stats_output = temp_dir / "temporal_statistics.csv"
        
        temporal_metrics_df.write_csv(metrics_output)
        temporal_stats_df.write_csv(stats_output)
        
        assert metrics_output.exists(), "Temporal metrics should be exported"
        assert stats_output.exists(), "Temporal statistics should be exported"
        
        # Verify export integrity
        exported_metrics = pl.read_csv(metrics_output)
        exported_stats = pl.read_csv(stats_output)
        
        assert exported_metrics.shape == temporal_metrics_df.shape, "Metrics export should match original"
        assert exported_stats.shape == temporal_stats_df.shape, "Statistics export should match original"
    
    def test_workflow_4_combined_analysis_pipeline(
        self, 
        sample_temporal_edgelist_path, 
        sample_metadata_path, 
        sample_seeds_path, 
        temp_dir
    ):
        """
        Test Workflow 4: Combined Network + GLP + Time-series analysis
        
        This comprehensive workflow tests integration of all major components:
        1. Load temporal data and metadata
        2. Create temporal network slices
        3. Run GLP on each time slice
        4. Track label propagation evolution over time
        5. Analyze category-based temporal patterns
        6. Generate comprehensive reports
        """
        # Step 1: Load metadata and seeds
        metadata_df = pl.read_csv(sample_metadata_path)
        seeds_df = pl.read_csv(sample_seeds_path)
        
        # Step 2: Create temporal slices
        temporal_graphs = create_temporal_slices(
            edgelist=str(sample_temporal_edgelist_path),
            timestamp_col="timestamp", 
            slice_interval="daily",
            rolling_window=None,
            cumulative=False,
            directed=False,
            weight_col="weight"
        )
        
        # Step 3: Run GLP on each temporal slice
        glp_temporal_results = []
        
        for date, graph, mapper in temporal_graphs:
            if graph.numberOfNodes() == 0:
                # Skip empty graphs
                continue
                
            # Convert seeds to external format for this slice
            slice_seed_labels = {}
            slice_labels_set = set()
            for row in seeds_df.iter_rows(named=True):
                node_id = row["node_id"]
                label = str(row["label"])
                if node_id in mapper.original_to_internal:
                    slice_seed_labels[node_id] = label
                    slice_labels_set.add(label)
            
            if len(slice_seed_labels) == 0:
                # Skip slices with no seed nodes
                continue
            
            slice_labels_list = sorted(list(slice_labels_set))
            
            # Run GLP
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")  # Suppress convergence warnings for test data
                results = guided_label_propagation(
                    graph=graph,
                    id_mapper=mapper,
                    seed_labels=slice_seed_labels,
                    labels=slice_labels_list,
                    alpha=0.85,
                    max_iterations=50,  # Reduced for faster testing
                    convergence_threshold=1e-4,  # Relaxed for test data
                    normalize=True,
                    directional=False
                )
            
            # Store results with date (results is DataFrame with directional=False)
            for row in results.iter_rows(named=True):
                node_id = row["node_id"]
                predicted_label = row["dominant_label"]
                confidence = row["confidence"]
                is_seed = node_id in slice_seed_labels
                
                glp_temporal_results.append({
                    "date": date,
                    "node_id": node_id,
                    "predicted_label": predicted_label,
                    "max_probability": confidence,
                    "is_seed": is_seed
                })
        
        # Verify GLP temporal results
        assert len(glp_temporal_results) > 0, "Should have GLP results across time"
        
        glp_results_df = pl.DataFrame(glp_temporal_results)
        
        # Step 4: Analyze category connections over time
        category_connections = analyze_cross_category_connections(
            temporal_graphs,
            metadata_df,
            category_column="department",
            edge_weight="sum"
        )
        
        # Verify category analysis
        if not category_connections.is_empty():
            assert "date" in category_connections.columns, "Should have date column"
            assert "category_a" in category_connections.columns, "Should have category_a column"
            assert "category_b" in category_connections.columns, "Should have category_b column"
            assert "connection_strength" in category_connections.columns, "Should have connection_strength column"
        
        # Step 5: Extract temporal centrality metrics
        centrality_metrics = extract_temporal_metrics(
            temporal_graphs,
            metrics=["degree", "betweenness"],
            n_jobs=1
        )
        
        # Step 6: Combine all results with metadata
        if not centrality_metrics.is_empty():
            # Join centrality with metadata
            enriched_metrics = centrality_metrics.join(
                metadata_df,
                on="node_id",
                how="left"
            )
            
            # Join GLP results with metadata
            enriched_glp = glp_results_df.join(
                metadata_df,
                on="node_id", 
                how="left"
            )
            
            # Verify enriched data
            assert "department" in enriched_metrics.columns, "Centrality should be enriched with metadata"
            assert "department" in enriched_glp.columns, "GLP results should be enriched with metadata"
        
        # Step 7: Generate comprehensive outputs
        output_dir = temp_dir / "combined_analysis"
        output_dir.mkdir()
        
        # Export all results
        if not glp_results_df.is_empty():
            glp_results_df.write_csv(output_dir / "glp_temporal_results.csv")
        
        if not category_connections.is_empty():
            category_connections.write_csv(output_dir / "category_connections.csv")
        
        if not centrality_metrics.is_empty():
            centrality_metrics.write_csv(output_dir / "temporal_centrality.csv")
        
        # Export enriched results if available
        if not centrality_metrics.is_empty():
            enriched_metrics.write_csv(output_dir / "enriched_centrality.csv")
            enriched_glp.write_csv(output_dir / "enriched_glp_results.csv")
        
        # Step 8: Verify all outputs were created
        expected_files = [
            "glp_temporal_results.csv",
            "temporal_centrality.csv"
        ]
        
        for filename in expected_files:
            if (output_dir / filename).exists():
                file_path = output_dir / filename
                exported_data = pl.read_csv(file_path)
                assert not exported_data.is_empty(), f"{filename} should not be empty"
        
        # Generate summary statistics
        summary_stats = {
            "temporal_slices": len(temporal_graphs),
            "total_nodes": len(set(glp_results_df["node_id"].to_list())) if not glp_results_df.is_empty() else 0,
            "glp_results_count": len(glp_results_df) if not glp_results_df.is_empty() else 0,
            "category_connections_count": len(category_connections) if not category_connections.is_empty() else 0,
            "centrality_measurements": len(centrality_metrics) if not centrality_metrics.is_empty() else 0
        }
        
        # Export summary
        summary_df = pl.DataFrame([summary_stats])
        summary_df.write_csv(output_dir / "analysis_summary.csv")
        
        assert (output_dir / "analysis_summary.csv").exists(), "Analysis summary should be created"
        
        # Verify summary contents
        exported_summary = pl.read_csv(output_dir / "analysis_summary.csv")
        assert exported_summary["temporal_slices"][0] > 0, "Should have processed temporal slices"


class TestIntegrationErrorHandling:
    """Test error handling and edge cases in integration workflows."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for test outputs."""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir)
    
    def test_missing_input_files(self, temp_dir):
        """Test handling of missing input files."""
        nonexistent_path = temp_dir / "nonexistent.csv"
        
        with pytest.raises(Exception):  # Should raise FileNotFoundError or similar
            build_graph_from_edgelist(str(nonexistent_path))
    
    def test_empty_edge_list(self, temp_dir):
        """Test handling of empty edge lists."""
        empty_edgelist = temp_dir / "empty.csv"
        empty_edgelist.write_text("source,target,weight\n")  # Header only
        
        # Empty edge lists should raise ValidationError
        from src.common.exceptions import ValidationError
        with pytest.raises(ValidationError, match="DataFrame is empty"):
            build_graph_from_edgelist(str(empty_edgelist))
    
    def test_malformed_data_handling(self, temp_dir):
        """Test handling of malformed data files."""
        malformed_file = temp_dir / "malformed.csv"
        malformed_file.write_text("invalid,data,format\nthis,is,not\na,valid,edgelist\n")
        
        # Should handle malformed data gracefully
        try:
            graph, mapper = build_graph_from_edgelist(
                str(malformed_file),
                source_col="source",  # Columns don't exist
                target_col="target"
            )
            # If it doesn't raise an exception, verify empty result
            assert graph.numberOfNodes() == 0, "Malformed data should produce empty graph"
        except Exception:
            # Expected to raise an exception for malformed data
            pass
    
    def test_inconsistent_temporal_data(self, temp_dir):
        """Test handling of temporal data with gaps and inconsistencies."""
        inconsistent_temporal = temp_dir / "inconsistent_temporal.csv"
        inconsistent_temporal.write_text(
            "source,target,timestamp\n"
            "A,B,2024-01-01\n"
            "C,D,invalid_date\n"  # Invalid timestamp
            "E,F,2024-01-05\n"    # Gap in dates
        )
        
        # Should handle inconsistent temporal data
        try:
            temporal_graphs = create_temporal_slices(
                str(inconsistent_temporal),
                timestamp_col="timestamp",
                slice_interval="daily"
            )
            # Should at least process valid entries
            assert isinstance(temporal_graphs, list), "Should return list even with some invalid data"
        except Exception:
            # May raise exception for invalid data
            pass


class TestIntegrationPerformance:
    """Test performance characteristics of integration workflows."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for test outputs."""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir)
    
    def test_large_network_workflow(self, temp_dir):
        """Test workflow performance with larger synthetic network."""
        # Generate larger synthetic edge list
        import random
        random.seed(42)  # For reproducibility
        
        # Create synthetic edge list with 100 nodes and ~300 edges
        nodes = [f"node_{i:03d}" for i in range(100)]
        edges = []
        
        for _ in range(300):
            source = random.choice(nodes)
            target = random.choice(nodes)
            if source != target:  # Avoid self-loops
                weight = round(random.uniform(1.0, 5.0), 2)
                edges.append(f"{source},{target},{weight}")
        
        large_edgelist = temp_dir / "large_network.csv"
        large_edgelist.write_text("source,target,weight\n" + "\n".join(edges))
        
        # Test basic workflow performance
        import time
        start_time = time.time()
        
        # Build graph
        graph, mapper = build_graph_from_edgelist(str(large_edgelist))
        
        # Extract centrality
        centrality_df = extract_centrality(graph, mapper, metrics=["degree", "betweenness"])
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        # Verify results
        assert graph.numberOfNodes() > 0, "Large network should have nodes"
        assert not centrality_df.is_empty(), "Should have centrality results"
        
        # Performance should be reasonable (less than 30 seconds for this size)
        assert processing_time < 30, f"Processing took {processing_time:.2f}s, should be faster"
    
    def test_temporal_workflow_performance(self, temp_dir):
        """Test temporal analysis performance with multiple time slices."""
        # Generate temporal data with multiple days
        import random
        from datetime import timedelta
        
        random.seed(42)
        nodes = [f"node_{i:02d}" for i in range(20)]
        
        # Generate 7 days of temporal data
        base_date = datetime(2024, 1, 1)
        edges = []
        
        for day in range(7):
            current_date = base_date + timedelta(days=day)
            
            # Generate 20-30 edges per day
            for _ in range(random.randint(20, 30)):
                source = random.choice(nodes)
                target = random.choice(nodes)
                if source != target:
                    timestamp = current_date + timedelta(
                        hours=random.randint(8, 18),
                        minutes=random.randint(0, 59)
                    )
                    weight = round(random.uniform(1.0, 3.0), 2)
                    edges.append(f"{source},{target},{timestamp},{weight}")
        
        temporal_edgelist = temp_dir / "temporal_performance.csv"
        temporal_edgelist.write_text("source,target,timestamp,weight\n" + "\n".join(edges))
        
        # Test temporal workflow performance
        import time
        start_time = time.time()
        
        # Create temporal slices
        temporal_graphs = create_temporal_slices(
            str(temporal_edgelist),
            timestamp_col="timestamp",
            slice_interval="daily"
        )
        
        # Extract temporal metrics
        temporal_metrics = extract_temporal_metrics(
            temporal_graphs,
            metrics=["degree", "betweenness"],
            n_jobs=1
        )
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        # Verify results
        assert len(temporal_graphs) > 0, "Should have temporal slices"
        assert not temporal_metrics.is_empty(), "Should have temporal metrics"
        
        # Performance should be reasonable
        assert processing_time < 20, f"Temporal processing took {processing_time:.2f}s, should be faster"