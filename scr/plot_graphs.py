import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from data_utils import reindex_nodes, create_hetero_graph
import pathlib
import numpy as np
import sys

# 1.  Point to the parent directory …/dependency_graph
PARENT_DIR = pathlib.Path(__file__).resolve().parents[1]

# 2.  Push that path on the *front* of sys.path so imports see it first
sys.path.insert(0, str(PARENT_DIR))

from dependency_graph import DependencyGraph


def plot_hetero_graph(data_dict, rule_mapping, assmpt_mapping, non_assmpt_mapping):
    # Create reverse mappings
    reverse_rule = {v: k for k, v in rule_mapping.items()}
    reverse_assmpt = {v: k for k, v in assmpt_mapping.items()}
    reverse_non_assmpt = {v: k for k, v in non_assmpt_mapping.items()}
    
    # Create a directed graph
    G = nx.DiGraph()
    
    # Add nodes with attributes for type
    for rule_id, rule_name in reverse_rule.items():
        G.add_node(rule_name, node_type='rule')
    
    for assmpt_id, assmpt_name in reverse_assmpt.items():
        G.add_node(assmpt_name, node_type='assmpt')
    
    for non_assmpt_id, non_assmpt_name in reverse_non_assmpt.items():
        G.add_node(non_assmpt_name, node_type='non_assmpt')
    
    # Define edge label mapping
    edge_labels = {
        'supports': '+',
        'attacks': '-',
        'derives': 'd'
    }
    
    # Add edges with appropriate labels
    edge_attributes = {}
    
    for edge_type, (src_ids, dst_ids) in data_dict.items():
        if len(src_ids) == 0:  # Skip empty edge lists
            continue
            
        src_type, relation, dst_type = edge_type
        edge_label = edge_labels.get(relation, relation)
        
        for i in range(len(src_ids)):
            src_id = src_ids[i]
            dst_id = dst_ids[i]
            
            # Get original labels based on node type
            if src_type == 'rule':
                src_label = reverse_rule[src_id]
            elif src_type == 'assmpt':
                src_label = reverse_assmpt[src_id]
            elif src_type == 'non_assmpt':
                src_label = reverse_non_assmpt[src_id]
            
            if dst_type == 'rule':
                dst_label = reverse_rule[dst_id]
            elif dst_type == 'assmpt':
                dst_label = reverse_assmpt[dst_id]
            elif dst_type == 'non_assmpt':
                dst_label = reverse_non_assmpt[dst_id]
            
            G.add_edge(src_label, dst_label, relation=relation)
            edge_attributes[(src_label, dst_label)] = edge_label
    
    # Set up the plot
    plt.figure(figsize=(14, 12))
    
    # Define node positions using a layout algorithm
    pos = nx.spring_layout(G, seed=42, k=1.0)
    
    # Define node colors based on type
    color_map = {'rule': 'lightgreen', 'assmpt': 'lightblue', 'non_assmpt': 'lightpink'}
    node_colors = [color_map[G.nodes[node]['node_type']] for node in G.nodes()]
    
    node_size = 1600
    node_radius = np.sqrt(node_size) / 2  # Approximate radius based on node size

    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, alpha=1.0, node_size=node_size)
    
    # Draw edges with different styles based on relation
    edges_supports = [(u, v) for u, v, d in G.edges(data=True) if d['relation'] == 'supports']
    edges_attacks = [(u, v) for u, v, d in G.edges(data=True) if d['relation'] == 'attacks']
    edges_derives = [(u, v) for u, v, d in G.edges(data=True) if d['relation'] == 'derives']
    
    # Handle self-loops separately with special styling
    selfloop_edges_supports = [(u, v) for (u, v) in edges_supports if u == v]
    selfloop_edges_attacks = [(u, v) for (u, v) in edges_attacks if u == v]
    selfloop_edges_derives = [(u, v) for (u, v) in edges_derives if u == v]

    # Remove self-loops from regular edge lists
    edges_supports = [(u, v) for (u, v) in edges_supports if u != v]
    edges_attacks = [(u, v) for (u, v) in edges_attacks if u != v]
    edges_derives = [(u, v) for (u, v) in edges_derives if u != v]

    connection_style = 'arc3'

    nx.draw_networkx_edges(G, pos, edgelist=edges_supports, edge_color='green', arrows=True,
                           arrowsize=25, width=2, connectionstyle=connection_style, node_size=node_size)
    nx.draw_networkx_edges(G, pos, edgelist=edges_attacks, edge_color='red', arrows=True,
                           arrowsize=25, width=2, connectionstyle=connection_style, node_size=node_size)
    nx.draw_networkx_edges(G, pos, edgelist=edges_derives, edge_color='blue', arrows=True,
                           arrowsize=25, width=2, connectionstyle=connection_style, node_size=node_size)
    
    # Draw self-loops with larger radius to make them more visible
    # Use arc3 with a larger value for more pronounced curve
    nx.draw_networkx_edges(G, pos, edgelist=selfloop_edges_supports, edge_color='green', arrows=True,
                          arrowsize=25, width=2, connectionstyle='arc3,rad=0.3', node_size=node_size)
    nx.draw_networkx_edges(G, pos, edgelist=selfloop_edges_attacks, edge_color='red', arrows=True,
                           arrowsize=25, width=2, connectionstyle='arc3,rad=0.3', node_size=node_size)
    nx.draw_networkx_edges(G, pos, edgelist=selfloop_edges_derives, edge_color='blue', arrows=True,
                           arrowsize=25, width=2, connectionstyle='arc3,rad=0.3', node_size=node_size)
    

    # Draw node labels
    nx.draw_networkx_labels(G, pos, font_weight='bold', font_size=20)
    
    # Create custom edge label positions with controlled perpendicular offset
    edge_label_pos = {}
    
    for edge in G.edges():
        u, v = edge
        
        # Special handling for self-loops (where u == v)
        if u == v:
            # Get the position of the node
            node_x, node_y = pos[u]
            
            # For self-loops, place the label above and to the right of the node
            # Use a larger offset to ensure it doesn't overlap with the node
            offset = 0.25
            
            # Position the label in the top-right quadrant relative to the node
            edge_label_pos[edge] = (node_x, node_y + offset)
            continue
        
        # For regular edges (not self-loops)
        # Get the positions of both nodes
        x1, y1 = pos[u]
        x2, y2 = pos[v]
        
        # Calculate midpoint
        x_mid = (x1 + x2) / 2
        y_mid = (y1 + y2) / 2
        
        # Calculate the direction vector of the edge
        dx = x2 - x1
        dy = y2 - y1
        
        # Calculate the length of the direction vector
        length = np.sqrt(dx**2 + dy**2)
        
        # Normalize the direction vector
        if length > 0:
            dx, dy = dx/length, dy/length
        
        # Calculate both possible perpendicular vectors
        # First option: rotate 90 degrees counterclockwise (-dy, dx)
        # Second option: rotate 90 degrees clockwise (dy, -dx)
        perp_x1, perp_y1 = -dy, dx  # counterclockwise (often "above")
        perp_x2, perp_y2 = dy, -dx  # clockwise (often "below")
        
        # Set the length of the perpendicular offset
        offset_length = 0.06
        
        # Choose the perpendicular direction that points "up" (positive y) 
        # or "right" (positive x) in the majority of cases
        # If the edge is more horizontal (|dx| > |dy|), prioritize the "above" direction
        # If the edge is more vertical (|dy| > |dx|), prioritize the "right" direction
        if abs(dx) > abs(dy):  # More horizontal edge
            # For horizontal edges, choose the perpendicular that has a positive y component
            if perp_y1 > 0:
                perp_x, perp_y = perp_x1, perp_y1  # Use counterclockwise (above)
            else:
                perp_x, perp_y = perp_x2, perp_y2  # Use clockwise (above)
        else:  # More vertical edge
            # For vertical edges, choose the perpendicular that has a positive x component
            if perp_x1 > 0:
                perp_x, perp_y = perp_x1, perp_y1  # Use counterclockwise (right)
            else:
                perp_x, perp_y = perp_x2, perp_y2  # Use clockwise (right)
        
        # Apply the selected perpendicular offset
        edge_label_pos[edge] = (x_mid + perp_x * offset_length, y_mid + perp_y * offset_length)
    
    # Draw edge labels with perpendicular positions
    for edge, label_pos in edge_label_pos.items():
        if edge in edge_attributes:
            label = edge_attributes[edge]
            # Manually place the text perpendicular to the edge
            plt.text(label_pos[0], label_pos[1], 
                    label, 
                    horizontalalignment='center',
                    verticalalignment='center',
                    fontsize=18,
                    bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', boxstyle='round,pad=0.2'))
            

    # Create a legend
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='lightgreen', markersize=20, label='Rule'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='lightblue', markersize=20, label='Assumption'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='lightpink', markersize=20, label='Non-Assumption'),
        Line2D([0], [0], color='green', label='Support (+)'),
        Line2D([0], [0], color='red', label='Attack (-)'),
        Line2D([0], [0], color='blue', label='Derives (d)')
    ]
    
    plt.legend(handles=legend_elements, loc='lower left', fontsize=18)
    plt.title('Argumentation Framework Graph', fontsize=14)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(f"./data/reconstructed_graph")

if __name__ == "__main__":
    f_input = "./data/example_framework.txt"
    dep_graph = DependencyGraph()
    dep_graph.create_from_file(f_input)
    dep_graph.create_dependency_graph()
    (rules_mapping, assmpt_mapping, non_assmpt_mapping) = reindex_nodes(dep_graph) 
    hetero_graph, data_dict = create_hetero_graph(
            dep_graph.graph, 
            rules_mapping, 
            assmpt_mapping, 
            non_assmpt_mapping,
        )
    plot_hetero_graph(data_dict, rules_mapping, assmpt_mapping, non_assmpt_mapping)