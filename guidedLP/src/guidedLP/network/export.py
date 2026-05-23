"""
Network export module for the Guided Label Propagation library.

This module provides functionality for exporting NetworkIt graphs to various
formats including GEXF, GraphML, edgelist, and Parquet. Supports metadata
integration and automatic centrality calculation.
"""

import os
import warnings
from pathlib import Path
from typing import List, Dict, Any, Optional, Union, Set
import xml.etree.ElementTree as ET
from xml.dom import minidom

import polars as pl
import networkit as nk
import numpy as np

from guidedLP.common.id_mapper import IDMapper
from guidedLP.common.exceptions import (
    ComputationError,
    ConfigurationError,
    ValidationError,
    validate_parameter
)
from guidedLP.common.logging_config import get_logger, log_function_entry, LoggingTimer

logger = get_logger(__name__)

# Supported export formats
SUPPORTED_FORMATS = ["gexf", "graphml", "edgelist", "parquet"]


def export_graph(
    graph: nk.Graph,
    id_mapper: IDMapper,
    output_path: str,
    format: str = "gexf",
    metadata: Optional[pl.DataFrame] = None,
    include_metrics: Optional[List[str]] = None,
    overwrite: bool = False
) -> None:
    """
    Export graph with metadata to various file formats.
    
    This function exports NetworkIt graphs to standard network analysis formats
    while preserving original node IDs and optionally including metadata and
    centrality metrics. All exports maintain compatibility with original IDs
    for downstream analysis.
    
    Parameters
    ----------
    graph : nk.Graph
        NetworkIt graph object to export
    id_mapper : IDMapper
        Bidirectional mapping between original and internal node IDs
    output_path : str
        Path for output file (extension will be added if not present)
    format : str, default "gexf"
        Export format, one of:
        - "gexf": Gephi Exchange Format (XML-based, supports rich metadata)
        - "graphml": GraphML format (XML-based, widely compatible)
        - "edgelist": CSV edgelist with optional node attributes
        - "parquet": Binary format with separate edges and nodes tables
    metadata : pl.DataFrame, optional
        DataFrame with node metadata. Must contain 'node_id' column
        matching original node IDs from id_mapper
    include_metrics : List[str], optional
        List of centrality metrics to calculate and include.
        Available: ['degree', 'betweenness', 'closeness', 'eigenvector', 'pagerank', 'katz']
    overwrite : bool, default False
        If True, overwrite existing files without warning
    
    Returns
    -------
    None
        File is written to specified output_path
    
    Examples
    --------
    Basic export to GEXF:
    
    >>> export_graph(graph, mapper, "network.gexf")
    
    Export with metadata:
    
    >>> metadata = pl.DataFrame({
    ...     "node_id": ["A", "B", "C"],
    ...     "category": ["type1", "type2", "type1"],
    ...     "value": [1.0, 2.5, 1.8]
    ... })
    >>> export_graph(graph, mapper, "network.gexf", metadata=metadata)
    
    Export with centrality metrics:
    
    >>> export_graph(
    ...     graph, mapper, "network.graphml",
    ...     format="graphml",
    ...     include_metrics=["degree", "betweenness"]
    ... )
    
    Export to Parquet for large graphs:
    
    >>> export_graph(graph, mapper, "network.parquet", format="parquet")
    
    Raises
    ------
    ValidationError
        If format is unsupported or parameters are invalid
    ComputationError
        If export fails due to file I/O or format errors
    
    Notes
    -----
    Time Complexity:
        O(N + E) for most formats, O(N log N) for sorted operations
    
    Space Complexity:
        O(N + E) for storing graph data during export
    
    File Formats:
    - GEXF: Best for Gephi visualization, supports rich attributes
    - GraphML: Widely compatible, good for academic tools
    - Edgelist: Simple CSV format, good for basic analysis
    - Parquet: Efficient binary format for large graphs
    
    Original node IDs are preserved in all formats to maintain
    compatibility with external data sources and analysis workflows.
    """
    log_function_entry(
        "export_graph",
        graph_nodes=graph.numberOfNodes(),
        graph_edges=graph.numberOfEdges(),
        format=format,
        output_path=output_path,
        has_metadata=metadata is not None,
        include_metrics=include_metrics
    )
    
    # Validate parameters
    _validate_export_parameters(format, output_path, metadata, include_metrics)
    
    # Check file existence
    output_path = _prepare_output_path(output_path, format, overwrite)
    
    # Handle empty graph
    if graph.numberOfNodes() == 0:
        logger.warning("Empty graph provided. Creating empty export file.")
        _export_empty_graph(output_path, format)
        return
    
    with LoggingTimer("export_graph", {"format": format, "nodes": graph.numberOfNodes(), "edges": graph.numberOfEdges()}):
        try:
            # Prepare node data with metadata and metrics
            node_data = _prepare_node_data(graph, id_mapper, metadata, include_metrics)
            
            # Prepare edge data
            edge_data = _prepare_edge_data(graph, id_mapper)
            
            # Export to specified format
            if format == "gexf":
                _export_gexf(graph, id_mapper, node_data, edge_data, output_path)
            elif format == "graphml":
                _export_graphml(graph, id_mapper, node_data, edge_data, output_path)
            elif format == "edgelist":
                _export_edgelist(graph, id_mapper, node_data, edge_data, output_path)
            elif format == "parquet":
                _export_parquet(graph, id_mapper, node_data, edge_data, output_path)
            
            logger.info(f"Graph exported successfully to {output_path}")
            
        except Exception as e:
            raise ComputationError(
                f"Graph export failed: {str(e)}",
                context={
                    "operation": "export_graph",
                    "format": format,
                    "output_path": output_path,
                    "error_type": "export_failure"
                }
            ) from e


def _validate_export_parameters(
    format: str,
    output_path: str,
    metadata: Optional[pl.DataFrame],
    include_metrics: Optional[List[str]]
) -> None:
    """Validate export parameters."""
    if format not in SUPPORTED_FORMATS:
        raise ValidationError(
            f"Unsupported export format: {format}. "
            f"Supported formats: {SUPPORTED_FORMATS}"
        )
    
    if not output_path or not isinstance(output_path, str):
        raise ValidationError("output_path must be a non-empty string")
    
    if metadata is not None:
        if not isinstance(metadata, pl.DataFrame):
            raise ValidationError("metadata must be a Polars DataFrame")
        if "node_id" not in metadata.columns:
            raise ValidationError("metadata DataFrame must contain 'node_id' column")
    
    if include_metrics is not None:
        if not isinstance(include_metrics, list):
            raise ValidationError("include_metrics must be a list of strings")
        
        # Import here to avoid circular dependency
        from .analysis import AVAILABLE_METRICS
        invalid_metrics = [m for m in include_metrics if m not in AVAILABLE_METRICS]
        if invalid_metrics:
            raise ValidationError(
                f"Invalid metrics: {invalid_metrics}. "
                f"Available metrics: {AVAILABLE_METRICS}"
            )


def _prepare_output_path(output_path: str, format: str, overwrite: bool) -> str:
    """Prepare and validate output path."""
    # Add extension if not present
    path = Path(output_path)
    if not path.suffix:
        if format == "edgelist":
            path = path.with_suffix(".csv")
        else:
            path = path.with_suffix(f".{format}")
    
    output_path = str(path)
    
    # Check if file exists
    if os.path.exists(output_path) and not overwrite:
        warnings.warn(
            f"File {output_path} already exists. Use overwrite=True to replace it.",
            UserWarning
        )
    
    # Create directory if it doesn't exist
    os.makedirs(path.parent, exist_ok=True)
    
    return output_path


def _prepare_node_data(
    graph: nk.Graph,
    id_mapper: IDMapper,
    metadata: Optional[pl.DataFrame],
    include_metrics: Optional[List[str]]
) -> pl.DataFrame:
    """Prepare node data with metadata and metrics."""
    # Create base node data
    original_ids = []
    for internal_id in range(graph.numberOfNodes()):
        if graph.hasNode(internal_id):
            try:
                original_id = id_mapper.get_original(internal_id)
                original_ids.append(original_id)
            except KeyError:
                logger.warning(f"Internal ID {internal_id} not found in mapper")
                original_ids.append(f"unknown_{internal_id}")
    
    node_data = pl.DataFrame({"node_id": original_ids})
    
    # Add centrality metrics if requested
    if include_metrics:
        from .analysis import extract_centrality
        
        logger.debug(f"Calculating centrality metrics: {include_metrics}")
        centrality_df = extract_centrality(
            graph, id_mapper, metrics=include_metrics, normalized=True, n_jobs=1
        )
        
        # Merge with node data
        node_data = node_data.join(centrality_df, on="node_id", how="left")
    
    # Add user-provided metadata if available
    if metadata is not None:
        logger.debug(f"Adding metadata with {metadata.shape[1]} columns")
        
        # Check for missing nodes
        node_ids_set = set(node_data["node_id"].to_list())
        metadata_ids_set = set(metadata["node_id"].to_list())
        
        missing_in_metadata = node_ids_set - metadata_ids_set
        if missing_in_metadata:
            logger.warning(
                f"Metadata missing for {len(missing_in_metadata)} nodes: "
                f"{list(missing_in_metadata)[:5]}{'...' if len(missing_in_metadata) > 5 else ''}"
            )
        
        # Merge with metadata
        node_data = node_data.join(metadata, on="node_id", how="left")
    
    return node_data


def _prepare_edge_data(graph: nk.Graph, id_mapper: IDMapper) -> pl.DataFrame:
    """Prepare edge data with original IDs."""
    sources = []
    targets = []
    weights = []
    
    for u, v in graph.iterEdges():
        try:
            source_id = id_mapper.get_original(u)
            target_id = id_mapper.get_original(v)
            
            sources.append(source_id)
            targets.append(target_id)
            
            if graph.isWeighted():
                weights.append(graph.weight(u, v))
            else:
                weights.append(1.0)
                
        except KeyError as e:
            logger.warning(f"Node ID not found in mapper: {e}")
            continue
    
    edge_data = pl.DataFrame({
        "source": sources,
        "target": targets,
        "weight": weights
    })
    
    return edge_data


def _export_empty_graph(output_path: str, format: str) -> None:
    """Export empty graph in specified format."""
    if format == "gexf":
        _write_empty_gexf(output_path)
    elif format == "graphml":
        _write_empty_graphml(output_path)
    elif format == "edgelist":
        # Empty CSV with headers
        pl.DataFrame({"source": [], "target": [], "weight": []}).write_csv(output_path)
    elif format == "parquet":
        # Empty parquet files
        base_path = Path(output_path).with_suffix("")
        pl.DataFrame({"node_id": []}).write_parquet(f"{base_path}_nodes.parquet")
        pl.DataFrame({"source": [], "target": [], "weight": []}).write_parquet(f"{base_path}_edges.parquet")


def _export_gexf(
    graph: nk.Graph,
    id_mapper: IDMapper,
    node_data: pl.DataFrame,
    edge_data: pl.DataFrame,
    output_path: str
) -> None:
    """Export graph to GEXF format."""
    # Create root GEXF element
    gexf = ET.Element("gexf", xmlns="http://www.gexf.net/1.2draft", version="1.2")
    
    # Add meta information
    meta = ET.SubElement(gexf, "meta", lastmodifieddate=str(np.datetime64('today')))
    ET.SubElement(meta, "creator").text = "Guided Label Propagation Library"
    ET.SubElement(meta, "description").text = "Network exported from NetworkIt graph"
    
    # Create graph element
    graph_elem = ET.SubElement(gexf, "graph", mode="static", defaultedgetype="directed" if graph.isDirected() else "undirected")
    
    # Define attributes for nodes
    attributes = ET.SubElement(graph_elem, "attributes", **{"class": "node"})
    
    attr_id = 0
    attr_map = {}
    for col in node_data.columns:
        if col != "node_id":
            attr_elem = ET.SubElement(attributes, "attribute", id=str(attr_id), title=col, type="string")
            attr_map[col] = str(attr_id)
            attr_id += 1
    
    # Add nodes
    nodes_elem = ET.SubElement(graph_elem, "nodes")
    for row in node_data.iter_rows(named=True):
        node_id = str(row["node_id"])
        node_elem = ET.SubElement(nodes_elem, "node", id=node_id, label=node_id)
        
        # Add attributes
        if len(attr_map) > 0:
            attvalues = ET.SubElement(node_elem, "attvalues")
            for col, attr_id in attr_map.items():
                value = row.get(col)
                if value is not None:
                    ET.SubElement(attvalues, "attvalue", **{"for": attr_id, "value": str(value)})
    
    # Add edges
    edges_elem = ET.SubElement(graph_elem, "edges")
    for i, row in enumerate(edge_data.iter_rows(named=True)):
        edge_attrs = {
            "id": str(i),
            "source": str(row["source"]),
            "target": str(row["target"])
        }
        if graph.isWeighted():
            edge_attrs["weight"] = str(row["weight"])
        
        ET.SubElement(edges_elem, "edge", **edge_attrs)
    
    # Write to file with pretty formatting
    xml_str = ET.tostring(gexf, encoding='unicode')
    pretty_xml = minidom.parseString(xml_str).toprettyxml(indent="  ")
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(pretty_xml)


def _export_graphml(
    graph: nk.Graph,
    id_mapper: IDMapper,
    node_data: pl.DataFrame,
    edge_data: pl.DataFrame,
    output_path: str
) -> None:
    """Export graph to GraphML format."""
    # Create root GraphML element
    graphml = ET.Element("graphml", xmlns="http://graphml.graphdrawing.org/xmlns")
    
    # Define keys for attributes
    key_id = 0
    key_map = {}
    
    for col in node_data.columns:
        if col != "node_id":
            key_elem = ET.SubElement(graphml, "key", id=f"n{key_id}", **{"for": "node", "attr.name": col, "attr.type": "string"})
            key_map[col] = f"n{key_id}"
            key_id += 1
    
    # Add weight key for edges if graph is weighted
    if graph.isWeighted():
        ET.SubElement(graphml, "key", id="weight", **{"for": "edge", "attr.name": "weight", "attr.type": "double"})
    
    # Create graph element
    graph_elem = ET.SubElement(graphml, "graph", id="G", edgedefault="directed" if graph.isDirected() else "undirected")
    
    # Add nodes
    for row in node_data.iter_rows(named=True):
        node_id = str(row["node_id"])
        node_elem = ET.SubElement(graph_elem, "node", id=node_id)
        
        # Add data elements
        for col, key_id in key_map.items():
            value = row.get(col)
            if value is not None:
                data_elem = ET.SubElement(node_elem, "data", key=key_id)
                data_elem.text = str(value)
    
    # Add edges
    for i, row in enumerate(edge_data.iter_rows(named=True)):
        edge_elem = ET.SubElement(graph_elem, "edge", id=f"e{i}", source=str(row["source"]), target=str(row["target"]))
        
        if graph.isWeighted():
            data_elem = ET.SubElement(edge_elem, "data", key="weight")
            data_elem.text = str(row["weight"])
    
    # Write to file with pretty formatting
    xml_str = ET.tostring(graphml, encoding='unicode')
    pretty_xml = minidom.parseString(xml_str).toprettyxml(indent="  ")
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(pretty_xml)


def _export_edgelist(
    graph: nk.Graph,
    id_mapper: IDMapper,
    node_data: pl.DataFrame,
    edge_data: pl.DataFrame,
    output_path: str
) -> None:
    """Export graph to CSV edgelist format."""
    # If we have node attributes, create an enriched edgelist
    if node_data.shape[1] > 1:  # More than just node_id
        # Add source node attributes
        source_attrs = node_data.rename({"node_id": "source"})
        source_cols = {col: f"source_{col}" for col in source_attrs.columns if col != "source"}
        source_attrs = source_attrs.rename(source_cols)
        
        enriched_edges = edge_data.join(source_attrs, on="source", how="left")
        
        # Add target node attributes
        target_attrs = node_data.rename({"node_id": "target"})
        target_cols = {col: f"target_{col}" for col in target_attrs.columns if col != "target"}
        target_attrs = target_attrs.rename(target_cols)
        
        enriched_edges = enriched_edges.join(target_attrs, on="target", how="left")
        
        enriched_edges.write_csv(output_path)
    else:
        # Simple edgelist
        edge_data.write_csv(output_path)


def _export_parquet(
    graph: nk.Graph,
    id_mapper: IDMapper,
    node_data: pl.DataFrame,
    edge_data: pl.DataFrame,
    output_path: str
) -> None:
    """Export graph to Parquet format with separate node and edge files."""
    base_path = Path(output_path).with_suffix("")
    
    # Write nodes table
    nodes_path = f"{base_path}_nodes.parquet"
    node_data.write_parquet(nodes_path)
    
    # Write edges table
    edges_path = f"{base_path}_edges.parquet"
    edge_data.write_parquet(edges_path)
    
    logger.info(f"Parquet export created: {nodes_path} and {edges_path}")


def _write_empty_gexf(output_path: str) -> None:
    """Write empty GEXF file."""
    gexf = ET.Element("gexf", xmlns="http://www.gexf.net/1.2draft", version="1.2")
    meta = ET.SubElement(gexf, "meta", lastmodifieddate=str(np.datetime64('today')))
    ET.SubElement(meta, "creator").text = "Guided Label Propagation Library"
    
    graph_elem = ET.SubElement(gexf, "graph", mode="static", defaultedgetype="undirected")
    ET.SubElement(graph_elem, "nodes")
    ET.SubElement(graph_elem, "edges")
    
    xml_str = ET.tostring(gexf, encoding='unicode')
    pretty_xml = minidom.parseString(xml_str).toprettyxml(indent="  ")
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(pretty_xml)


def _write_empty_graphml(output_path: str) -> None:
    """Write empty GraphML file."""
    graphml = ET.Element("graphml", xmlns="http://graphml.graphdrawing.org/xmlns")
    graph_elem = ET.SubElement(graphml, "graph", id="G", edgedefault="undirected")
    
    xml_str = ET.tostring(graphml, encoding='unicode')
    pretty_xml = minidom.parseString(xml_str).toprettyxml(indent="  ")
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(pretty_xml)


def get_export_info(
    graph: nk.Graph,
    id_mapper: IDMapper,
    metadata: Optional[pl.DataFrame] = None,
    include_metrics: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Get information about what would be exported without actually exporting.
    
    Parameters
    ----------
    graph : nk.Graph
        NetworkIt graph object
    id_mapper : IDMapper
        ID mapper for original node IDs
    metadata : pl.DataFrame, optional
        Node metadata to include
    include_metrics : List[str], optional
        Centrality metrics to calculate
        
    Returns
    -------
    Dict[str, Any]
        Information about the export including node/edge counts,
        attributes, and estimated file sizes
    """
    info = {
        "graph_info": {
            "nodes": graph.numberOfNodes(),
            "edges": graph.numberOfEdges(),
            "directed": graph.isDirected(),
            "weighted": graph.isWeighted()
        },
        "export_attributes": [],
        "missing_metadata_nodes": 0
    }
    
    # Check what attributes would be included
    if include_metrics:
        info["export_attributes"].extend([f"{m}_centrality" for m in include_metrics])
    
    if metadata is not None:
        # Find attributes from metadata
        metadata_attrs = [col for col in metadata.columns if col != "node_id"]
        info["export_attributes"].extend(metadata_attrs)
        
        # Check for missing metadata
        node_ids = set()
        for internal_id in range(graph.numberOfNodes()):
            if graph.hasNode(internal_id):
                try:
                    original_id = id_mapper.get_original(internal_id)
                    node_ids.add(original_id)
                except KeyError:
                    pass
        
        metadata_ids = set(metadata["node_id"].to_list())
        missing = node_ids - metadata_ids
        info["missing_metadata_nodes"] = len(missing)
    
    # Estimate file sizes (rough approximations)
    base_size = graph.numberOfNodes() * 50 + graph.numberOfEdges() * 30  # bytes
    attr_size = len(info["export_attributes"]) * graph.numberOfNodes() * 20
    
    info["estimated_sizes"] = {
        "gexf": base_size + attr_size * 3,  # XML overhead
        "graphml": base_size + attr_size * 3,
        "edgelist": base_size + attr_size,
        "parquet": (base_size + attr_size) * 0.3  # Compression
    }
    
    return info