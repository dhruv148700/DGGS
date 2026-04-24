#!/usr/bin/env python3
"""Test script for the DependencyGraph class."""

from scr.dependency_graph import DependencyGraph

# Create DependencyGraph instance
dep_graph = DependencyGraph()

# Load framework from out.aba
print("Loading framework from out.aba...")
dep_graph.create_from_file("outputs/aba-files/random_3_out.aba")

print(f"Assumptions: {dep_graph.assumptions}")
print(f"Contraries: {dep_graph.contrary}")
print(f"Number of rules: {len(dep_graph.rules)}")
print(f"All elements: {dep_graph.all_elements}")

# Create dependency graph
print("\nCreating dependency graph...")
dep_graph.create_dependency_graph(print_graph=True)

print(f"Graph nodes: {list(dep_graph.graph.nodes())}")
print(f"Graph edges: {list(dep_graph.graph.edges(data=True))[:10]}")  # First 10 edges
print(f"Total nodes: {dep_graph.graph.number_of_nodes()}")
print(f"Total edges: {dep_graph.graph.number_of_edges()}")

print("\nTest completed successfully!")
