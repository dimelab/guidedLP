"""
Tests for network export module.

This module tests graph export functionality including various file formats,
metadata integration, and file content validation.
"""

import pytest
import polars as pl
import networkit as nk
import numpy as np
import os
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, Any, List

from src.network.export import (
    export_graph,
    get_export_info
)
from src.common.id_mapper import IDMapper
from src.common.exceptions import ValidationError, ComputationError


class TestExportGraph:
    """Test graph export functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        # Create test graph
        self.test_graph = nk.Graph(5, weighted=True, directed=False)
        self.test_graph.addEdge(0, 1, 2.5)
        self.test_graph.addEdge(1, 2, 1.0)
        self.test_graph.addEdge(2, 3, 3.0)
        self.test_graph.addEdge(3, 4, 1.5)
        self.test_graph.addEdge(4, 0, 2.0)
        
        # Create ID mapper
        self.test_mapper = IDMapper()
        node_names = ["Alice", "Bob", "Charlie", "David", "Eve"]
        for i, name in enumerate(node_names):
            self.test_mapper.add_mapping(name, i)
        
        # Create sample metadata
        self.metadata = pl.DataFrame({
            "node_id": ["Alice", "Bob", "Charlie", "David", "Eve"],
            "category": ["A", "B", "A", "C", "B"],
            "value": [1.0, 2.5, 1.8, 3.2, 2.1],
            "active": [True, False, True, True, False]
        })
        
        # Create directed graph for testing
        self.directed_graph = nk.Graph(3, weighted=True, directed=True)
        self.directed_graph.addEdge(0, 1, 1.0)
        self.directed_graph.addEdge(1, 2, 2.0)
        self.directed_graph.addEdge(2, 0, 0.5)
        
        self.directed_mapper = IDMapper()
        for i, name in enumerate(["X", "Y", "Z"]):
            self.directed_mapper.add_mapping(name, i)
        
        # Setup temporary directory for test files
        self.temp_dir = tempfile.mkdtemp()

    def teardown_method(self):
        """Clean up test files."""
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_export_gexf_basic(self):
        """Test basic GEXF export without metadata."""
        output_path = os.path.join(self.temp_dir, "test.gexf")
        
        export_graph(
            self.test_graph, self.test_mapper, output_path, format="gexf"
        )
        
        assert os.path.exists(output_path)
        
        # Parse and validate XML structure
        tree = ET.parse(output_path)
        root = tree.getroot()
        
        assert root.tag.endswith("gexf")
        
        # Check nodes
        nodes = root.find(".//{http://www.gexf.net/1.2draft}nodes")
        assert nodes is not None
        node_elements = nodes.findall(".//{http://www.gexf.net/1.2draft}node")
        assert len(node_elements) == 5
        
        # Check that original IDs are used
        node_ids = {node.get("id") for node in node_elements}
        expected_ids = {"Alice", "Bob", "Charlie", "David", "Eve"}
        assert node_ids == expected_ids
        
        # Check edges
        edges = root.find(".//{http://www.gexf.net/1.2draft}edges")
        assert edges is not None
        edge_elements = edges.findall(".//{http://www.gexf.net/1.2draft}edge")
        assert len(edge_elements) == 5

    def test_export_gexf_with_metadata(self):
        """Test GEXF export with metadata."""
        output_path = os.path.join(self.temp_dir, "test_meta.gexf")
        
        export_graph(
            self.test_graph, self.test_mapper, output_path,
            format="gexf", metadata=self.metadata
        )
        
        assert os.path.exists(output_path)
        
        # Parse XML and check attributes
        tree = ET.parse(output_path)
        root = tree.getroot()
        
        # Check attribute definitions
        attributes = root.find(".//{http://www.gexf.net/1.2draft}attributes")
        assert attributes is not None
        
        attr_elements = attributes.findall(".//{http://www.gexf.net/1.2draft}attribute")
        attr_titles = {attr.get("title") for attr in attr_elements}
        expected_attrs = {"category", "value", "active"}
        assert expected_attrs.issubset(attr_titles)
        
        # Check that nodes have attribute values
        nodes = root.findall(".//{http://www.gexf.net/1.2draft}node")
        alice_node = next(n for n in nodes if n.get("id") == "Alice")
        attvalues = alice_node.find(".//{http://www.gexf.net/1.2draft}attvalues")
        assert attvalues is not None

    def test_export_gexf_with_metrics(self):
        """Test GEXF export with centrality metrics."""
        output_path = os.path.join(self.temp_dir, "test_metrics.gexf")
        
        export_graph(
            self.test_graph, self.test_mapper, output_path,
            format="gexf", include_metrics=["degree", "betweenness"]
        )
        
        assert os.path.exists(output_path)
        
        # Parse XML and check that centrality attributes are present
        tree = ET.parse(output_path)
        root = tree.getroot()
        
        attributes = root.find(".//{http://www.gexf.net/1.2draft}attributes")
        attr_elements = attributes.findall(".//{http://www.gexf.net/1.2draft}attribute")
        attr_titles = {attr.get("title") for attr in attr_elements}
        
        assert "degree_centrality" in attr_titles
        assert "betweenness_centrality" in attr_titles

    def test_export_graphml_basic(self):
        """Test basic GraphML export."""
        output_path = os.path.join(self.temp_dir, "test.graphml")
        
        export_graph(
            self.test_graph, self.test_mapper, output_path, format="graphml"
        )
        
        assert os.path.exists(output_path)
        
        # Parse and validate XML structure
        tree = ET.parse(output_path)
        root = tree.getroot()
        
        assert root.tag.endswith("graphml")
        
        # Check graph element
        graph_elem = root.find(".//{http://graphml.graphdrawing.org/xmlns}graph")
        assert graph_elem is not None
        assert graph_elem.get("edgedefault") == "undirected"
        
        # Check nodes
        nodes = root.findall(".//{http://graphml.graphdrawing.org/xmlns}node")
        assert len(nodes) == 5
        
        # Check that original IDs are used
        node_ids = {node.get("id") for node in nodes}
        expected_ids = {"Alice", "Bob", "Charlie", "David", "Eve"}
        assert node_ids == expected_ids

    def test_export_graphml_with_metadata(self):
        """Test GraphML export with metadata."""
        output_path = os.path.join(self.temp_dir, "test_meta.graphml")
        
        export_graph(
            self.test_graph, self.test_mapper, output_path,
            format="graphml", metadata=self.metadata
        )
        
        assert os.path.exists(output_path)
        
        # Parse XML and check key definitions
        tree = ET.parse(output_path)
        root = tree.getroot()
        
        keys = root.findall(".//{http://graphml.graphdrawing.org/xmlns}key")
        key_names = {key.get("attr.name") for key in keys if key.get("for") == "node"}
        expected_keys = {"category", "value", "active"}
        assert expected_keys.issubset(key_names)

    def test_export_directed_graph_graphml(self):
        """Test GraphML export with directed graph."""
        output_path = os.path.join(self.temp_dir, "directed.graphml")
        
        export_graph(
            self.directed_graph, self.directed_mapper, output_path, format="graphml"
        )
        
        assert os.path.exists(output_path)
        
        # Check that graph is marked as directed
        tree = ET.parse(output_path)
        root = tree.getroot()
        graph_elem = root.find(".//{http://graphml.graphdrawing.org/xmlns}graph")
        assert graph_elem.get("edgedefault") == "directed"

    def test_export_edgelist_basic(self):
        """Test basic edgelist export."""
        output_path = os.path.join(self.temp_dir, "test.csv")
        
        export_graph(
            self.test_graph, self.test_mapper, output_path, format="edgelist"
        )
        
        assert os.path.exists(output_path)
        
        # Read and validate CSV
        df = pl.read_csv(output_path)
        
        assert "source" in df.columns
        assert "target" in df.columns
        assert "weight" in df.columns
        assert df.height == 5  # 5 edges
        
        # Check that original IDs are used
        all_nodes = set(df["source"].to_list() + df["target"].to_list())
        expected_ids = {"Alice", "Bob", "Charlie", "David", "Eve"}
        assert all_nodes == expected_ids

    def test_export_edgelist_with_metadata(self):
        """Test edgelist export with node metadata."""
        output_path = os.path.join(self.temp_dir, "test_enriched.csv")
        
        export_graph(
            self.test_graph, self.test_mapper, output_path,
            format="edgelist", metadata=self.metadata
        )
        
        assert os.path.exists(output_path)
        
        # Read and check that node attributes are included
        df = pl.read_csv(output_path)
        
        assert "source" in df.columns
        assert "target" in df.columns
        assert "weight" in df.columns
        
        # Check for source and target attributes
        source_attr_cols = [col for col in df.columns if col.startswith("source_")]
        target_attr_cols = [col for col in df.columns if col.startswith("target_")]
        
        assert len(source_attr_cols) > 0
        assert len(target_attr_cols) > 0

    def test_export_parquet_basic(self):
        """Test basic Parquet export."""
        output_path = os.path.join(self.temp_dir, "test.parquet")
        
        export_graph(
            self.test_graph, self.test_mapper, output_path, format="parquet"
        )
        
        # Check that both files are created
        nodes_path = os.path.join(self.temp_dir, "test_nodes.parquet")
        edges_path = os.path.join(self.temp_dir, "test_edges.parquet")
        
        assert os.path.exists(nodes_path)
        assert os.path.exists(edges_path)
        
        # Read and validate data
        nodes_df = pl.read_parquet(nodes_path)
        edges_df = pl.read_parquet(edges_path)
        
        assert "node_id" in nodes_df.columns
        assert nodes_df.height == 5
        
        assert "source" in edges_df.columns
        assert "target" in edges_df.columns
        assert "weight" in edges_df.columns
        assert edges_df.height == 5

    def test_export_parquet_with_metadata(self):
        """Test Parquet export with metadata."""
        output_path = os.path.join(self.temp_dir, "test_meta.parquet")
        
        export_graph(
            self.test_graph, self.test_mapper, output_path,
            format="parquet", metadata=self.metadata
        )
        
        nodes_path = os.path.join(self.temp_dir, "test_meta_nodes.parquet")
        nodes_df = pl.read_parquet(nodes_path)
        
        # Check that metadata columns are included
        expected_cols = {"node_id", "category", "value", "active"}
        assert expected_cols.issubset(set(nodes_df.columns))

    def test_auto_extension_addition(self):
        """Test automatic file extension addition."""
        # Test without extension
        output_path = os.path.join(self.temp_dir, "test_auto")
        
        export_graph(
            self.test_graph, self.test_mapper, output_path, format="gexf"
        )
        
        # Should create file with .gexf extension
        expected_path = output_path + ".gexf"
        assert os.path.exists(expected_path)

    def test_directory_creation(self):
        """Test automatic directory creation."""
        nested_dir = os.path.join(self.temp_dir, "nested", "subdir")
        output_path = os.path.join(nested_dir, "test.gexf")
        
        export_graph(
            self.test_graph, self.test_mapper, output_path, format="gexf"
        )
        
        assert os.path.exists(output_path)
        assert os.path.exists(nested_dir)

    def test_overwrite_protection(self):
        """Test file overwrite protection."""
        output_path = os.path.join(self.temp_dir, "existing.gexf")
        
        # Create existing file
        with open(output_path, 'w') as f:
            f.write("existing content")
        
        # Export should warn but not fail
        with pytest.warns(UserWarning, match="already exists"):
            export_graph(
                self.test_graph, self.test_mapper, output_path, format="gexf"
            )

    def test_overwrite_forced(self):
        """Test forced file overwrite."""
        output_path = os.path.join(self.temp_dir, "overwrite.gexf")
        
        # Create existing file
        with open(output_path, 'w') as f:
            f.write("old content")
        
        # Export with overwrite=True should not warn
        export_graph(
            self.test_graph, self.test_mapper, output_path,
            format="gexf", overwrite=True
        )
        
        # File should be replaced with new content
        assert os.path.exists(output_path)
        with open(output_path, 'r') as f:
            content = f.read()
            assert "gexf" in content  # Should contain GEXF structure

    def test_empty_graph_export(self):
        """Test export of empty graph."""
        empty_graph = nk.Graph(0)
        empty_mapper = IDMapper()
        
        for format in ["gexf", "graphml", "edgelist", "parquet"]:
            output_path = os.path.join(self.temp_dir, f"empty.{format}")
            
            export_graph(
                empty_graph, empty_mapper, output_path, format=format
            )
            
            if format == "parquet":
                # Check both parquet files
                nodes_path = os.path.join(self.temp_dir, "empty_nodes.parquet")
                edges_path = os.path.join(self.temp_dir, "empty_edges.parquet")
                assert os.path.exists(nodes_path)
                assert os.path.exists(edges_path)
            else:
                assert os.path.exists(output_path)

    def test_missing_metadata_warning(self):
        """Test warning when metadata is missing for some nodes."""
        # Create metadata missing one node
        partial_metadata = pl.DataFrame({
            "node_id": ["Alice", "Bob", "Charlie", "David"],  # Missing Eve
            "category": ["A", "B", "A", "C"]
        })
        
        output_path = os.path.join(self.temp_dir, "partial_meta.gexf")
        
        # Should log warning about missing metadata
        export_graph(
            self.test_graph, self.test_mapper, output_path,
            format="gexf", metadata=partial_metadata
        )
        
        assert os.path.exists(output_path)

    def test_weighted_vs_unweighted_export(self):
        """Test export of unweighted graph."""
        unweighted_graph = nk.Graph(3, weighted=False)
        unweighted_graph.addEdge(0, 1)
        unweighted_graph.addEdge(1, 2)
        
        mapper = IDMapper()
        for i, name in enumerate(["A", "B", "C"]):
            mapper.add_mapping(name, i)
        
        output_path = os.path.join(self.temp_dir, "unweighted.csv")
        
        export_graph(
            unweighted_graph, mapper, output_path, format="edgelist"
        )
        
        # Check that weights are 1.0 for unweighted graph
        df = pl.read_csv(output_path)
        weights = df["weight"].to_list()
        assert all(w == 1.0 for w in weights)

    def test_large_graph_performance(self):
        """Test export performance on moderately large graph."""
        # Create larger graph
        n_nodes = 100
        large_graph = nk.Graph(n_nodes, weighted=True)
        
        # Add random edges
        np.random.seed(42)
        for _ in range(200):
            u, v = np.random.choice(n_nodes, 2, replace=False)
            weight = np.random.uniform(0.1, 5.0)
            if not large_graph.hasEdge(u, v):
                large_graph.addEdge(u, v, weight)
        
        # Create mapper
        mapper = IDMapper()
        for i in range(n_nodes):
            mapper.add_mapping(f"node_{i:03d}", i)
        
        # Export should complete without errors
        output_path = os.path.join(self.temp_dir, "large.parquet")
        export_graph(
            large_graph, mapper, output_path, format="parquet"
        )
        
        # Verify files were created
        nodes_path = os.path.join(self.temp_dir, "large_nodes.parquet")
        edges_path = os.path.join(self.temp_dir, "large_edges.parquet")
        assert os.path.exists(nodes_path)
        assert os.path.exists(edges_path)


class TestParameterValidation:
    """Test parameter validation for export functions."""

    def setup_method(self):
        """Set up minimal test fixtures."""
        self.graph = nk.Graph(3, weighted=True)
        self.graph.addEdge(0, 1, 1.0)
        self.graph.addEdge(1, 2, 2.0)
        
        self.mapper = IDMapper()
        for i in range(3):
            self.mapper.add_mapping(f"node_{i}", i)
        
        self.temp_dir = tempfile.mkdtemp()

    def teardown_method(self):
        """Clean up test files."""
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_invalid_format(self):
        """Test error for invalid export format."""
        output_path = os.path.join(self.temp_dir, "test.txt")
        
        with pytest.raises(ValidationError) as exc_info:
            export_graph(self.graph, self.mapper, output_path, format="invalid")
        
        assert "Unsupported export format" in str(exc_info.value)

    def test_empty_output_path(self):
        """Test error for empty output path."""
        with pytest.raises(ValidationError) as exc_info:
            export_graph(self.graph, self.mapper, "", format="gexf")
        
        assert "output_path must be a non-empty string" in str(exc_info.value)

    def test_invalid_metadata_type(self):
        """Test error for invalid metadata type."""
        output_path = os.path.join(self.temp_dir, "test.gexf")
        
        with pytest.raises(ValidationError) as exc_info:
            export_graph(
                self.graph, self.mapper, output_path,
                format="gexf", metadata="invalid"
            )
        
        assert "metadata must be a Polars DataFrame" in str(exc_info.value)

    def test_metadata_missing_node_id_column(self):
        """Test error when metadata lacks node_id column."""
        output_path = os.path.join(self.temp_dir, "test.gexf")
        bad_metadata = pl.DataFrame({"name": ["A", "B", "C"]})
        
        with pytest.raises(ValidationError) as exc_info:
            export_graph(
                self.graph, self.mapper, output_path,
                format="gexf", metadata=bad_metadata
            )
        
        assert "must contain 'node_id' column" in str(exc_info.value)

    def test_invalid_metrics_type(self):
        """Test error for invalid include_metrics type."""
        output_path = os.path.join(self.temp_dir, "test.gexf")
        
        with pytest.raises(ValidationError) as exc_info:
            export_graph(
                self.graph, self.mapper, output_path,
                format="gexf", include_metrics="degree"
            )
        
        assert "include_metrics must be a list" in str(exc_info.value)

    def test_invalid_metric_names(self):
        """Test error for invalid metric names."""
        output_path = os.path.join(self.temp_dir, "test.gexf")
        
        with pytest.raises(ValidationError) as exc_info:
            export_graph(
                self.graph, self.mapper, output_path,
                format="gexf", include_metrics=["invalid_metric"]
            )
        
        assert "Invalid metrics" in str(exc_info.value)


class TestExportInfo:
    """Test export information functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.graph = nk.Graph(4, weighted=True)
        self.graph.addEdge(0, 1, 1.0)
        self.graph.addEdge(1, 2, 2.0)
        self.graph.addEdge(2, 3, 1.5)
        
        self.mapper = IDMapper()
        for i in range(4):
            self.mapper.add_mapping(f"node_{i}", i)
        
        self.metadata = pl.DataFrame({
            "node_id": ["node_0", "node_1", "node_2"],  # Missing node_3
            "category": ["A", "B", "A"]
        })

    def test_get_export_info_basic(self):
        """Test basic export info without metadata."""
        info = get_export_info(self.graph, self.mapper)
        
        assert "graph_info" in info
        assert info["graph_info"]["nodes"] == 4
        assert info["graph_info"]["edges"] == 3
        assert info["graph_info"]["weighted"] is True
        
        assert "export_attributes" in info
        assert "estimated_sizes" in info
        assert "missing_metadata_nodes" in info

    def test_get_export_info_with_metadata(self):
        """Test export info with metadata."""
        info = get_export_info(self.graph, self.mapper, metadata=self.metadata)
        
        assert "category" in info["export_attributes"]
        assert info["missing_metadata_nodes"] == 1  # node_3 missing

    def test_get_export_info_with_metrics(self):
        """Test export info with centrality metrics."""
        info = get_export_info(
            self.graph, self.mapper,
            include_metrics=["degree", "betweenness"]
        )
        
        assert "degree_centrality" in info["export_attributes"]
        assert "betweenness_centrality" in info["export_attributes"]

    def test_estimated_file_sizes(self):
        """Test that estimated file sizes are reasonable."""
        info = get_export_info(self.graph, self.mapper)
        
        sizes = info["estimated_sizes"]
        
        # Parquet should be smallest due to compression
        assert sizes["parquet"] < sizes["gexf"]
        assert sizes["parquet"] < sizes["graphml"]
        
        # All sizes should be positive
        assert all(size > 0 for size in sizes.values())


class TestEdgeCases:
    """Test edge cases and special scenarios."""

    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()

    def teardown_method(self):
        """Clean up test files."""
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_single_node_graph(self):
        """Test export of single node graph."""
        single_graph = nk.Graph(1)
        mapper = IDMapper()
        mapper.add_mapping("only_node", 0)
        
        output_path = os.path.join(self.temp_dir, "single.gexf")
        
        export_graph(single_graph, mapper, output_path, format="gexf")
        
        assert os.path.exists(output_path)
        
        # Verify structure
        tree = ET.parse(output_path)
        nodes = tree.findall(".//{http://www.gexf.net/1.2draft}node")
        assert len(nodes) == 1
        assert nodes[0].get("id") == "only_node"

    def test_disconnected_graph(self):
        """Test export of disconnected graph."""
        disconnected = nk.Graph(4)
        disconnected.addEdge(0, 1)  # Component 1
        disconnected.addEdge(2, 3)  # Component 2
        
        mapper = IDMapper()
        for i in range(4):
            mapper.add_mapping(f"comp_{i}", i)
        
        output_path = os.path.join(self.temp_dir, "disconnected.csv")
        
        export_graph(disconnected, mapper, output_path, format="edgelist")
        
        assert os.path.exists(output_path)
        
        # Should have 2 edges
        df = pl.read_csv(output_path)
        assert df.height == 2

    def test_self_loops(self):
        """Test export of graph with self-loops."""
        graph_with_loops = nk.Graph(3, weighted=True)
        graph_with_loops.addEdge(0, 1, 1.0)
        graph_with_loops.addEdge(1, 1, 2.0)  # Self-loop
        graph_with_loops.addEdge(1, 2, 1.5)
        
        mapper = IDMapper()
        for i in range(3):
            mapper.add_mapping(f"node_{i}", i)
        
        output_path = os.path.join(self.temp_dir, "loops.csv")
        
        export_graph(graph_with_loops, mapper, output_path, format="edgelist")
        
        assert os.path.exists(output_path)
        
        # Check that self-loop is included
        df = pl.read_csv(output_path)
        self_loops = df.filter(pl.col("source") == pl.col("target"))
        assert self_loops.height == 1

    def test_unicode_node_ids(self):
        """Test export with Unicode node IDs."""
        unicode_graph = nk.Graph(3)
        unicode_graph.addEdge(0, 1)
        unicode_graph.addEdge(1, 2)
        
        mapper = IDMapper()
        unicode_names = ["αβγ", "δεζ", "ηθι"]
        for i, name in enumerate(unicode_names):
            mapper.add_mapping(name, i)
        
        output_path = os.path.join(self.temp_dir, "unicode.gexf")
        
        export_graph(unicode_graph, mapper, output_path, format="gexf")
        
        assert os.path.exists(output_path)
        
        # Verify Unicode characters are preserved
        with open(output_path, 'r', encoding='utf-8') as f:
            content = f.read()
            for name in unicode_names:
                assert name in content

    def test_very_large_metadata(self):
        """Test export with metadata containing many columns."""
        graph = nk.Graph(3)
        graph.addEdge(0, 1)
        graph.addEdge(1, 2)
        
        mapper = IDMapper()
        for i in range(3):
            mapper.add_mapping(f"n_{i}", i)
        
        # Create metadata with many columns
        metadata_dict = {"node_id": ["n_0", "n_1", "n_2"]}
        for i in range(20):
            metadata_dict[f"attr_{i}"] = [f"val_{i}_{j}" for j in range(3)]
        
        large_metadata = pl.DataFrame(metadata_dict)
        
        output_path = os.path.join(self.temp_dir, "large_meta.gexf")
        
        export_graph(
            graph, mapper, output_path,
            format="gexf", metadata=large_metadata
        )
        
        assert os.path.exists(output_path)
        
        # Verify all attributes are included
        tree = ET.parse(output_path)
        attributes = tree.find(".//{http://www.gexf.net/1.2draft}attributes")
        attr_elements = attributes.findall(".//{http://www.gexf.net/1.2draft}attribute")
        assert len(attr_elements) == 20

    def test_mixed_data_types_metadata(self):
        """Test export with metadata containing mixed data types."""
        graph = nk.Graph(2)
        graph.addEdge(0, 1)
        
        mapper = IDMapper()
        mapper.add_mapping("A", 0)
        mapper.add_mapping("B", 1)
        
        # Mixed data types
        mixed_metadata = pl.DataFrame({
            "node_id": ["A", "B"],
            "int_col": [1, 2],
            "float_col": [1.5, 2.7],
            "bool_col": [True, False],
            "str_col": ["hello", "world"]
        })
        
        output_path = os.path.join(self.temp_dir, "mixed.gexf")
        
        export_graph(
            graph, mapper, output_path,
            format="gexf", metadata=mixed_metadata
        )
        
        assert os.path.exists(output_path)
        
        # All data types should be converted to strings in XML
        tree = ET.parse(output_path)
        attvalue_elements = tree.findall(".//{http://www.gexf.net/1.2draft}attvalue")
        assert len(attvalue_elements) > 0