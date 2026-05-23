"""
Tests for noise category functionality in Guided Label Propagation.

This module tests the new noise category features including:
- Automatic noise category addition
- Noise seed generation
- Confidence thresholding  
- Single label scenario handling
"""

import pytest
import polars as pl
import numpy as np
from typing import Dict, List, Any

from src.network.construction import build_graph_from_edgelist
from src.glp.propagation import guided_label_propagation, _process_noise_category, _generate_noise_seeds, _apply_confidence_threshold
from src.common.exceptions import ValidationError, ConfigurationError


class TestNoiseCategoryFeatures:
    """Test suite for noise category functionality."""
    
    @pytest.fixture
    def simple_network_data(self):
        """Create simple test network data."""
        edges_data = [
            ("A1", "A2", 2.0), ("A2", "A3", 2.0), ("A3", "A1", 1.5),  # Community A
            ("B1", "B2", 2.0), ("B2", "B3", 2.0), ("B3", "B1", 1.5),  # Community B
            ("A1", "B1", 0.3),  # Weak inter-community connection
            ("OUT1", "OUT2", 1.0), ("OUT1", "A1", 0.2)  # Outlier nodes
        ]
        
        edges_df = pl.DataFrame({
            "source": [e[0] for e in edges_data],
            "target": [e[1] for e in edges_data],
            "weight": [e[2] for e in edges_data]
        })
        
        seed_labels = {"A1": "community_a", "B1": "community_b"}
        labels = ["community_a", "community_b"]
        
        return edges_df, seed_labels, labels
    
    @pytest.fixture
    def graph_and_mapper(self, simple_network_data):
        """Create graph and mapper from test data."""
        edges_df, seed_labels, labels = simple_network_data
        graph, mapper = build_graph_from_edgelist(edges_df, "source", "target", "weight")
        return graph, mapper, seed_labels, labels

    def test_noise_category_enabled(self, graph_and_mapper):
        """Test GLP with noise category enabled."""
        graph, mapper, seed_labels, labels = graph_and_mapper
        
        results = guided_label_propagation(
            graph=graph,
            id_mapper=mapper,
            seed_labels=seed_labels,
            labels=labels,
            enable_noise_category=True,
            noise_ratio=0.2
        )
        
        # Check that noise category was added
        assert "noise_prob" in results.columns
        
        # Check that some nodes were classified as noise
        noise_nodes = results.filter(pl.col("dominant_label") == "noise")
        assert len(noise_nodes) > 0
        
        # Check that results have expected structure
        expected_columns = ["node_id", "community_a_prob", "community_b_prob", "noise_prob", 
                          "dominant_label", "confidence", "is_seed"]
        assert set(results.columns) == set(expected_columns)

    def test_noise_category_disabled(self, graph_and_mapper):
        """Test GLP with noise category disabled."""
        graph, mapper, seed_labels, labels = graph_and_mapper
        
        results = guided_label_propagation(
            graph=graph,
            id_mapper=mapper,
            seed_labels=seed_labels,
            labels=labels,
            enable_noise_category=False
        )
        
        # Check that noise category was not added
        assert "noise_prob" not in results.columns
        
        # Check that only original labels appear
        unique_labels = set(results["dominant_label"].unique())
        assert unique_labels.issubset({"community_a", "community_b", "uncertain"})

    def test_confidence_thresholding(self, graph_and_mapper):
        """Test confidence thresholding functionality."""
        graph, mapper, seed_labels, labels = graph_and_mapper
        
        results = guided_label_propagation(
            graph=graph,
            id_mapper=mapper,
            seed_labels=seed_labels,
            labels=labels,
            confidence_threshold=0.7,
            enable_noise_category=True
        )
        
        # Check that some nodes were marked as uncertain
        uncertain_nodes = results.filter(pl.col("dominant_label") == "uncertain")
        
        # All uncertain nodes should have confidence below threshold
        if len(uncertain_nodes) > 0:
            max_uncertain_confidence = uncertain_nodes["confidence"].max()
            assert max_uncertain_confidence < 0.7

    def test_single_label_scenario(self, graph_and_mapper):
        """Test single label scenario with and without noise category."""
        graph, mapper, _, _ = graph_and_mapper
        
        single_seed_labels = {"A1": "important"}
        single_labels = ["important"]
        
        # With noise category (should work well)
        with pytest.warns(None) as warning_list:
            results_with_noise = guided_label_propagation(
                graph=graph,
                id_mapper=mapper,
                seed_labels=single_seed_labels,
                labels=single_labels,
                enable_noise_category=True
            )
        
        # Should have noise category
        assert "noise_prob" in results_with_noise.columns
        unique_labels_with_noise = set(results_with_noise["dominant_label"].unique())
        assert "noise" in unique_labels_with_noise
        
        # Without noise category (should show warning)
        with pytest.warns(UserWarning, match="single label provides limited discriminative power"):
            results_without_noise = guided_label_propagation(
                graph=graph,
                id_mapper=mapper,
                seed_labels=single_seed_labels,
                labels=single_labels,
                enable_noise_category=False
            )
        
        # Should not have noise category
        assert "noise_prob" not in results_without_noise.columns

    def test_noise_seed_generation(self, graph_and_mapper):
        """Test noise seed generation function."""
        graph, mapper, seed_labels, labels = graph_and_mapper
        
        noise_seeds = _generate_noise_seeds(graph, mapper, seed_labels, noise_ratio=0.3)
        
        # Check that noise seeds were generated
        assert len(noise_seeds) > 0
        
        # Check that all noise seeds have "noise" label
        assert all(label == "noise" for label in noise_seeds.values())
        
        # Check that noise seeds don't overlap with original seeds
        assert set(noise_seeds.keys()).isdisjoint(set(seed_labels.keys()))

    def test_process_noise_category(self, graph_and_mapper):
        """Test noise category processing function."""
        graph, mapper, seed_labels, labels = graph_and_mapper
        
        # Test with noise category enabled
        processed_labels, processed_seeds = _process_noise_category(
            graph, mapper, seed_labels, labels, enable_noise_category=True, noise_ratio=0.2
        )
        
        # Check that noise was added to labels
        assert "noise" in processed_labels
        assert len(processed_labels) == len(labels) + 1
        
        # Check that noise seeds were added
        noise_seed_count = sum(1 for label in processed_seeds.values() if label == "noise")
        assert noise_seed_count > 0
        
        # Test with noise category disabled
        processed_labels_no_noise, processed_seeds_no_noise = _process_noise_category(
            graph, mapper, seed_labels, labels, enable_noise_category=False, noise_ratio=0.2
        )
        
        # Check that nothing was added
        assert processed_labels_no_noise == labels
        assert processed_seeds_no_noise == seed_labels

    def test_apply_confidence_threshold(self):
        """Test confidence threshold application."""
        # Create test results DataFrame
        test_results = pl.DataFrame({
            "node_id": ["A", "B", "C", "D"],
            "community_a_prob": [0.8, 0.4, 0.2, 0.9],
            "community_b_prob": [0.2, 0.6, 0.8, 0.1],
            "dominant_label": ["community_a", "community_b", "community_b", "community_a"],
            "confidence": [0.8, 0.6, 0.8, 0.9],
            "is_seed": [True, False, False, False]
        })
        
        # Apply threshold of 0.7
        thresholded_results = _apply_confidence_threshold(test_results, 0.7)
        
        # Check that low-confidence nodes were marked as uncertain
        uncertain_nodes = thresholded_results.filter(pl.col("dominant_label") == "uncertain")
        assert len(uncertain_nodes) == 1  # Node B with confidence 0.6
        
        # Check that high-confidence nodes were unchanged
        confident_nodes = thresholded_results.filter(pl.col("confidence") >= 0.7)
        original_labels = confident_nodes["dominant_label"].to_list()
        assert "uncertain" not in original_labels

    def test_parameter_validation(self, graph_and_mapper):
        """Test parameter validation for noise category features."""
        graph, mapper, seed_labels, labels = graph_and_mapper
        
        # Test invalid noise_ratio
        with pytest.raises(ConfigurationError, match="Noise ratio must be between 0.0 and 1.0"):
            guided_label_propagation(
                graph, mapper, seed_labels, labels,
                enable_noise_category=True, noise_ratio=1.5
            )
        
        # Test invalid confidence_threshold
        with pytest.raises(ConfigurationError, match="Confidence threshold must be between 0.0 and 1.0"):
            guided_label_propagation(
                graph, mapper, seed_labels, labels,
                confidence_threshold=1.2
            )

    def test_noise_category_performance_impact(self, graph_and_mapper):
        """Test that noise category doesn't significantly impact performance."""
        graph, mapper, seed_labels, labels = graph_and_mapper
        
        import time
        
        # Run without noise category
        start_time = time.time()
        results_no_noise = guided_label_propagation(
            graph, mapper, seed_labels, labels, enable_noise_category=False
        )
        time_no_noise = time.time() - start_time
        
        # Run with noise category
        start_time = time.time()
        results_with_noise = guided_label_propagation(
            graph, mapper, seed_labels, labels, enable_noise_category=True
        )
        time_with_noise = time.time() - start_time
        
        # Noise category should not increase runtime by more than 50%
        assert time_with_noise < time_no_noise * 1.5
        
        # Both should return valid results
        assert len(results_no_noise) > 0
        assert len(results_with_noise) > 0

    def test_noise_category_reproducibility(self, graph_and_mapper):
        """Test that noise category results are reproducible."""
        graph, mapper, seed_labels, labels = graph_and_mapper
        
        # Run same configuration twice
        results1 = guided_label_propagation(
            graph, mapper, seed_labels, labels,
            enable_noise_category=True, noise_ratio=0.2
        )
        
        results2 = guided_label_propagation(
            graph, mapper, seed_labels, labels,
            enable_noise_category=True, noise_ratio=0.2
        )
        
        # Results should be identical (due to fixed random seed)
        assert results1.equals(results2)

    @pytest.mark.parametrize("noise_ratio", [0.0, 0.1, 0.2, 0.5])
    def test_noise_ratio_variations(self, graph_and_mapper, noise_ratio):
        """Test different noise ratio values."""
        graph, mapper, seed_labels, labels = graph_and_mapper
        
        results = guided_label_propagation(
            graph, mapper, seed_labels, labels,
            enable_noise_category=True, noise_ratio=noise_ratio
        )
        
        # Should always work without error
        assert len(results) > 0
        
        # If noise_ratio > 0, should have some noise seeds
        if noise_ratio > 0:
            assert "noise_prob" in results.columns

    @pytest.mark.parametrize("confidence_threshold", [0.0, 0.3, 0.5, 0.7, 0.9])
    def test_confidence_threshold_variations(self, graph_and_mapper, confidence_threshold):
        """Test different confidence threshold values."""
        graph, mapper, seed_labels, labels = graph_and_mapper
        
        results = guided_label_propagation(
            graph, mapper, seed_labels, labels,
            confidence_threshold=confidence_threshold
        )
        
        # Should always work without error
        assert len(results) > 0
        
        # If threshold > 0, might have uncertain nodes
        if confidence_threshold > 0:
            uncertain_count = (results["dominant_label"] == "uncertain").sum()
            # All uncertain nodes should have confidence below threshold
            if uncertain_count > 0:
                uncertain_nodes = results.filter(pl.col("dominant_label") == "uncertain")
                max_uncertain_confidence = uncertain_nodes["confidence"].max()
                assert max_uncertain_confidence < confidence_threshold