#!/usr/bin/env python3
"""
Installation test script for guidedLP package.

Run this script after installing to verify everything works correctly:
    python test_installation.py
"""

def test_installation():
    """Test that guidedLP package is properly installed and functional."""
    
    print("GuidedLP Installation Test")
    print("=" * 30)
    
    # Test 1: Import all key modules
    print("Test 1: Importing modules...")
    try:
        from guidedLP.network.construction import build_graph_from_edgelist, temporal_bipartite_to_unipartite
        from guidedLP.glp.propagation import guided_label_propagation
        from guidedLP.common.id_mapper import IDMapper
        from guidedLP.network.analysis import extract_centrality
        from guidedLP.timeseries.slicing import create_temporal_slices
        print("✅ All modules imported successfully")
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False
    
    # Test 2: Basic functionality
    print("\nTest 2: Basic functionality...")
    try:
        import polars as pl
        
        # Create simple test graph
        edges = pl.DataFrame({
            "source": ["A", "B", "C"],
            "target": ["B", "C", "A"],
            "weight": [1.0, 1.0, 1.0]
        })
        
        graph, mapper = build_graph_from_edgelist(edges, "source", "target", "weight")
        
        if graph.numberOfNodes() == 3 and graph.numberOfEdges() == 3:
            print("✅ Graph construction works")
        else:
            print("❌ Graph construction failed")
            return False
            
    except Exception as e:
        print(f"❌ Basic functionality failed: {e}")
        return False
    
    # Test 3: Temporal bipartite conversion
    print("\nTest 3: Temporal bipartite conversion...")
    try:
        temporal_data = pl.DataFrame({
            "user": ["Alice", "Bob"],
            "item": ["X", "X"],
            "timestamp": ["2024-01-01 09:00", "2024-01-01 11:00"]
        })
        
        temporal_graph, temporal_mapper = temporal_bipartite_to_unipartite(
            temporal_data,
            source_col="user",
            target_col="item",
            timestamp_col="timestamp",
            intermediate_col="item",
            projected_col="user"
        )
        
        if temporal_graph.numberOfNodes() == 2 and temporal_graph.numberOfEdges() == 1:
            print("✅ Temporal bipartite conversion works")
        else:
            print("❌ Temporal bipartite conversion failed")
            return False
            
    except Exception as e:
        print(f"❌ Temporal bipartite conversion failed: {e}")
        return False
    
    # Test 4: Package version
    print("\nTest 4: Package information...")
    try:
        import guidedLP
        print(f"✅ Package installed and accessible")
        
        # Try to get version
        try:
            version = guidedLP.__version__
            print(f"✅ Version: {version}")
        except AttributeError:
            print("⚠️  Version info not available (expected in development)")
            
    except Exception as e:
        print(f"❌ Package access failed: {e}")
        return False
    
    print("\n" + "=" * 30)
    print("🎉 Installation test PASSED!")
    print("\nYour guidedLP installation is working correctly.")
    print("You can now use the package in your projects:")
    print()
    print("  from guidedLP.network.construction import build_graph_from_edgelist")
    print("  from guidedLP.glp.propagation import guided_label_propagation")
    print("  from guidedLP.network.construction import temporal_bipartite_to_unipartite")
    print()
    print("See README.md for examples and documentation.")
    
    return True

if __name__ == "__main__":
    success = test_installation()
    exit(0 if success else 1)