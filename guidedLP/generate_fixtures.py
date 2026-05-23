#!/usr/bin/env python3
"""
Generate test fixtures for the Guided Label Propagation library.

This script creates realistic but small datasets for testing:
- sample_edgelist.csv: Network with 100 nodes and ~300 edges
- sample_temporal.csv: Temporal network data with timestamps
- sample_seeds.json: Seed nodes for GLP testing
- sample_metadata.csv: Node metadata with various attributes

The generated data follows realistic patterns but is designed to be
deterministic and suitable for testing.
"""

import json
import random
import csv
from datetime import datetime, timedelta
from pathlib import Path
import numpy as np

# Set random seeds for reproducibility
random.seed(42)
np.random.seed(42)

def generate_node_names(n_nodes=100):
    """Generate realistic node names with a mix of patterns."""
    names = []
    
    # Academic institutions (20%)
    institutions = [
        "MIT", "Stanford", "Harvard", "Berkeley", "CMU", "Caltech", "Princeton", 
        "Yale", "Columbia", "NYU", "UCLA", "USC", "UChicago", "Northwestern",
        "Cornell", "Brown", "Dartmouth", "Penn", "Duke", "Vanderbilt"
    ]
    
    # Companies (30%)
    companies = [
        "Google", "Apple", "Microsoft", "Amazon", "Meta", "Tesla", "Netflix",
        "Adobe", "Salesforce", "Oracle", "IBM", "Intel", "Nvidia", "Uber",
        "Airbnb", "Spotify", "Twitter", "LinkedIn", "Dropbox", "Slack",
        "Zoom", "Palantir", "Snowflake", "Databricks", "Stripe", "Square",
        "PayPal", "eBay", "Shopify", "Atlassian"
    ]
    
    # Research labs (20%)
    labs = [
        "CERN", "JPL", "NIST", "LLNL", "ANL", "ORNL", "PNNL", "SLAC",
        "Fermilab", "BNL", "LANL", "SNL", "LBNL", "NREL", "NIH", "CDC",
        "NASA_Ames", "NASA_Glenn", "ESA", "JAXA"
    ]
    
    # Startups and others (30%)
    others = [
        f"Startup_{i:02d}" for i in range(1, 21)
    ] + [
        f"Lab_{i:02d}" for i in range(1, 11)
    ]
    
    # Combine all categories
    all_options = institutions + companies + labs + others
    
    # Select nodes ensuring we get exactly n_nodes
    if n_nodes <= len(all_options):
        names = random.sample(all_options, n_nodes)
    else:
        # If we need more nodes than options, add numbered variants
        names = all_options.copy()
        for i in range(len(all_options), n_nodes):
            base_name = random.choice(all_options[:50])  # Use first 50 as base
            names.append(f"{base_name}_{i-len(all_options)+1:02d}")
    
    return sorted(names)  # Sort for deterministic ordering

def generate_communities(nodes):
    """Assign nodes to realistic communities based on their names."""
    communities = {}
    
    for node in nodes:
        if any(inst in node for inst in ["MIT", "Stanford", "Harvard", "Berkeley", "CMU", "Caltech"]):
            communities[node] = "top_universities"
        elif any(inst in node for inst in ["Princeton", "Yale", "Columbia", "NYU", "UCLA"]):
            communities[node] = "ivy_league"
        elif any(comp in node for comp in ["Google", "Apple", "Microsoft", "Amazon", "Meta"]):
            communities[node] = "big_tech"
        elif any(comp in node for comp in ["Tesla", "Uber", "Airbnb", "Netflix", "Spotify"]):
            communities[node] = "disruptors"
        elif any(lab in node for lab in ["CERN", "JPL", "NIST", "NASA", "ESA"]):
            communities[node] = "research_labs"
        elif "Startup" in node:
            communities[node] = "startups"
        elif "Lab" in node:
            communities[node] = "private_labs"
        else:
            # Assign remaining nodes to communities based on hash for determinism
            hash_val = hash(node) % 7
            community_map = {
                0: "top_universities", 1: "ivy_league", 2: "big_tech", 
                3: "disruptors", 4: "research_labs", 5: "startups", 6: "private_labs"
            }
            communities[node] = community_map[hash_val]
    
    return communities

def generate_network_edges(nodes, target_edges=300):
    """Generate realistic network edges with community structure."""
    communities = generate_communities(nodes)
    
    # Group nodes by community
    community_groups = {}
    for node, community in communities.items():
        if community not in community_groups:
            community_groups[community] = []
        community_groups[community].append(node)
    
    edges = set()
    edge_weights = {}
    
    # Generate intra-community edges (70% of edges)
    intra_target = int(target_edges * 0.7)
    intra_count = 0
    
    while intra_count < intra_target:
        # Select a community weighted by size
        community = random.choice(list(community_groups.keys()))
        community_nodes = community_groups[community]
        
        if len(community_nodes) < 2:
            continue
            
        # Select two different nodes from the same community
        node1, node2 = random.sample(community_nodes, 2)
        edge = tuple(sorted([node1, node2]))
        
        if edge not in edges:
            edges.add(edge)
            # Intra-community edges have higher weights (2.0-5.0)
            edge_weights[edge] = round(random.uniform(2.0, 5.0), 2)
            intra_count += 1
    
    # Generate inter-community edges (30% of edges)
    inter_target = target_edges - intra_count
    inter_count = 0
    
    while inter_count < inter_target:
        # Select two different communities
        communities_list = list(community_groups.keys())
        if len(communities_list) < 2:
            break
            
        comm1, comm2 = random.sample(communities_list, 2)
        
        # Select one node from each community
        node1 = random.choice(community_groups[comm1])
        node2 = random.choice(community_groups[comm2])
        edge = tuple(sorted([node1, node2]))
        
        if edge not in edges:
            edges.add(edge)
            # Inter-community edges have lower weights (0.5-2.5)
            edge_weights[edge] = round(random.uniform(0.5, 2.5), 2)
            inter_count += 1
    
    return list(edges), edge_weights

def generate_temporal_edges(base_edges, edge_weights, days=14):
    """Generate temporal network data from base network."""
    temporal_edges = []
    
    # Create a timeline over the specified days
    start_date = datetime(2024, 1, 1)
    
    for day in range(days):
        current_date = start_date + timedelta(days=day)
        
        # Each day, activate a subset of edges (30-70% of total edges)
        activation_rate = random.uniform(0.3, 0.7)
        active_edges = random.sample(base_edges, int(len(base_edges) * activation_rate))
        
        for edge in active_edges:
            source, target = edge
            base_weight = edge_weights[edge]
            
            # Generate 1-3 interactions per active edge per day
            n_interactions = random.randint(1, 3)
            
            for interaction in range(n_interactions):
                # Random time during the day
                hour = random.randint(8, 18)  # Business hours
                minute = random.randint(0, 59)
                timestamp = current_date.replace(hour=hour, minute=minute)
                
                # Interaction weight varies around base weight
                weight = max(0.1, base_weight + random.uniform(-0.5, 0.5))
                weight = round(weight, 2)
                
                # Add interaction type
                interaction_types = ["email", "meeting", "call", "collaboration", "message"]
                interaction_type = random.choice(interaction_types)
                
                temporal_edges.append({
                    "source": source,
                    "target": target,
                    "timestamp": timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                    "weight": weight,
                    "interaction_type": interaction_type
                })
    
    # Sort by timestamp for realism
    temporal_edges.sort(key=lambda x: x["timestamp"])
    
    return temporal_edges

def generate_seeds(nodes, communities, seeds_per_community=3):
    """Generate seed nodes for GLP testing."""
    community_groups = {}
    for node, community in communities.items():
        if community not in community_groups:
            community_groups[community] = []
        community_groups[community].append(node)
    
    seeds = {}
    
    for community, community_nodes in community_groups.items():
        # Select seeds_per_community nodes from each community
        n_seeds = min(seeds_per_community, len(community_nodes))
        selected_seeds = random.sample(community_nodes, n_seeds)
        
        for seed_node in selected_seeds:
            seeds[seed_node] = community
    
    return seeds

def generate_metadata(nodes, communities):
    """Generate realistic metadata for nodes."""
    metadata = []
    
    # Define realistic attributes
    sizes = ["small", "medium", "large", "very_large"]
    sectors = ["tech", "research", "education", "government", "startup", "biotech", "fintech"]
    regions = ["north_america", "europe", "asia", "oceania", "south_america"]
    founding_years = list(range(1950, 2025))
    
    for node in nodes:
        community = communities[node]
        
        # Determine attributes based on node name and community
        if any(inst in node for inst in ["MIT", "Stanford", "Harvard", "Berkeley"]):
            sector = "education"
            size = random.choice(["large", "very_large"])
            founding_year = random.choice(range(1861, 1950))  # Historical universities
        elif any(comp in node for comp in ["Google", "Apple", "Microsoft", "Amazon"]):
            sector = "tech"
            size = "very_large"
            founding_year = random.choice(range(1970, 2000))
        elif "Startup" in node:
            sector = random.choice(["tech", "biotech", "fintech"])
            size = random.choice(["small", "medium"])
            founding_year = random.choice(range(2010, 2024))
        elif any(lab in node for lab in ["CERN", "JPL", "NASA"]):
            sector = "research"
            size = random.choice(["medium", "large"])
            founding_year = random.choice(range(1950, 1980))
        else:
            sector = random.choice(sectors)
            size = random.choice(sizes)
            founding_year = random.choice(founding_years)
        
        # Assign region based on name patterns
        if any(x in node for x in ["MIT", "Harvard", "Stanford", "Google", "Apple"]):
            region = "north_america"
        elif any(x in node for x in ["CERN", "ESA"]):
            region = "europe"
        elif any(x in node for x in ["JAXA"]):
            region = "asia"
        else:
            region = random.choice(regions)
        
        # Calculate derived metrics
        age = 2024 - founding_year
        
        # Size to employee mapping
        size_to_employees = {
            "small": random.randint(10, 100),
            "medium": random.randint(100, 1000),
            "large": random.randint(1000, 10000),
            "very_large": random.randint(10000, 200000)
        }
        
        employees = size_to_employees[size]
        
        # Calculate influence score (combines age, size, and sector)
        sector_multiplier = {
            "tech": 1.5, "research": 1.3, "education": 1.2,
            "biotech": 1.1, "fintech": 1.1, "government": 1.0, "startup": 0.8
        }
        
        influence_score = round(
            (employees / 1000) * sector_multiplier[sector] * min(age / 20, 2.0), 2
        )
        
        metadata.append({
            "node_id": node,
            "community": community,
            "sector": sector,
            "size": size,
            "region": region,
            "founding_year": founding_year,
            "age": age,
            "employees": employees,
            "influence_score": influence_score,
            "is_startup": "startup" in sector or "Startup" in node,
            "is_academic": sector == "education",
            "is_research": sector == "research"
        })
    
    return metadata

def main():
    """Generate all test fixtures."""
    print("Generating test fixtures...")
    
    fixtures_dir = Path("tests/fixtures")
    fixtures_dir.mkdir(exist_ok=True)
    
    # Step 1: Generate nodes
    print("1. Generating node names...")
    nodes = generate_node_names(100)
    communities = generate_communities(nodes)
    
    print(f"   Created {len(nodes)} nodes across {len(set(communities.values()))} communities")
    
    # Step 2: Generate network edges
    print("2. Generating network edges...")
    edges, edge_weights = generate_network_edges(nodes, target_edges=300)
    
    print(f"   Created {len(edges)} edges")
    
    # Step 3: Create sample_edgelist.csv
    print("3. Writing sample_edgelist.csv...")
    with open(fixtures_dir / "sample_edgelist.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["source", "target", "weight", "edge_type"])
        
        for edge in edges:
            source, target = edge
            weight = edge_weights[edge]
            
            # Assign edge type based on weight
            if weight >= 3.5:
                edge_type = "strong"
            elif weight >= 2.0:
                edge_type = "medium" 
            else:
                edge_type = "weak"
            
            writer.writerow([source, target, weight, edge_type])
    
    # Step 4: Generate temporal data
    print("4. Generating temporal network data...")
    temporal_edges = generate_temporal_edges(edges, edge_weights, days=14)
    
    print(f"   Created {len(temporal_edges)} temporal interactions")
    
    # Step 5: Create sample_temporal.csv
    print("5. Writing sample_temporal.csv...")
    with open(fixtures_dir / "sample_temporal.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["source", "target", "timestamp", "weight", "interaction_type"])
        writer.writeheader()
        writer.writerows(temporal_edges)
    
    # Step 6: Generate seeds
    print("6. Generating seed nodes...")
    seeds = generate_seeds(nodes, communities, seeds_per_community=3)
    
    print(f"   Created {len(seeds)} seed nodes")
    
    # Step 7: Create sample_seeds.json
    print("7. Writing sample_seeds.json...")
    with open(fixtures_dir / "sample_seeds.json", "w") as f:
        json.dump(seeds, f, indent=2, sort_keys=True)
    
    # Step 8: Generate metadata
    print("8. Generating node metadata...")
    metadata = generate_metadata(nodes, communities)
    
    # Step 9: Create sample_metadata.csv
    print("9. Writing sample_metadata.csv...")
    with open(fixtures_dir / "sample_metadata.csv", "w", newline="") as f:
        fieldnames = [
            "node_id", "community", "sector", "size", "region",
            "founding_year", "age", "employees", "influence_score",
            "is_startup", "is_academic", "is_research"
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(metadata)
    
    # Step 10: Generate summary statistics
    print("\n" + "="*60)
    print("FIXTURE GENERATION SUMMARY")
    print("="*60)
    
    print(f"Nodes: {len(nodes)}")
    print(f"Edges: {len(edges)}")
    print(f"Temporal interactions: {len(temporal_edges)}")
    print(f"Seed nodes: {len(seeds)}")
    
    print(f"\nCommunity distribution:")
    community_counts = {}
    for community in communities.values():
        community_counts[community] = community_counts.get(community, 0) + 1
    
    for community, count in sorted(community_counts.items()):
        print(f"  {community}: {count} nodes")
    
    print(f"\nSeed distribution:")
    seed_communities = {}
    for node, community in seeds.items():
        seed_communities[community] = seed_communities.get(community, 0) + 1
    
    for community, count in sorted(seed_communities.items()):
        print(f"  {community}: {count} seeds")
    
    print(f"\nEdge weight distribution:")
    weights = list(edge_weights.values())
    print(f"  Min weight: {min(weights):.2f}")
    print(f"  Max weight: {max(weights):.2f}")
    print(f"  Mean weight: {sum(weights)/len(weights):.2f}")
    
    print(f"\nTemporal data span:")
    timestamps = [edge["timestamp"] for edge in temporal_edges]
    print(f"  From: {min(timestamps)}")
    print(f"  To: {max(timestamps)}")
    
    print(f"\nFiles created in {fixtures_dir}:")
    for file_path in sorted(fixtures_dir.glob("*.csv")) + sorted(fixtures_dir.glob("*.json")):
        size_kb = file_path.stat().st_size / 1024
        print(f"  {file_path.name}: {size_kb:.1f} KB")
    
    print("\n✅ Test fixtures generated successfully!")
    print("These files can now be used for testing across the library.")

if __name__ == "__main__":
    main()