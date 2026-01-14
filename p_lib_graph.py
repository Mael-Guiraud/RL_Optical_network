import networkx as nx
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

def plot_instance(instance):
    """
    Plots the instance as a layered simple graph.
    """

    # Create a simple Graph and prepare nodes
    G = nx.Graph()
    positions = {}
    labels = {}
    nlevels = len(instance)
    levels_nodes = []
    final_cycles_by_node = {}

    # Build nodes for each level
    for lvl, level in enumerate(instance):
        current_level_nodes = []
        if lvl == nlevels - 1:
            for i, subset in enumerate(level):
                aggregated_routes = set()
                for cycle in subset:
                    aggregated_routes.update(cycle)
                node_id = (lvl, i)
                G.add_node(node_id)
                labels[node_id] = str(sorted(aggregated_routes))
                current_level_nodes.append((node_id, aggregated_routes))
                final_cycles_by_node[node_id] = subset  # Store original cycles
        else:
            for i, subset in enumerate(level):
                aggregated_routes = set()
                for cycle in subset:
                    aggregated_routes.update(cycle)
                node_id = (lvl, i)
                G.add_node(node_id)
                labels[node_id] = str(sorted(aggregated_routes))
                current_level_nodes.append((node_id, aggregated_routes))
        levels_nodes.append(current_level_nodes)

    # Assign positions
    for lvl, nodes_in_level in enumerate(levels_nodes):
        count = len(nodes_in_level)
        for i, (node_id, _) in enumerate(nodes_in_level):
            x_pos = (i + 1) / (count + 1) if count > 0 else 0.5
            y_pos = -lvl
            positions[node_id] = (x_pos, y_pos)

    # Add edges between levels (shared routes on a simple graph)
    edge_labels = {}
    for lvl in range(nlevels - 1):
        for (u_id, u_routes) in levels_nodes[lvl]:
            for (v_id, v_routes) in levels_nodes[lvl + 1]:
                shared = u_routes.intersection(v_routes)
                if shared:
                    G.add_edge(u_id, v_id)
                    edge_labels[(u_id, v_id)] = ",".join(map(str, sorted(shared)))

    # Final level exits
    final_nodes = []
    vertical_gap = 1.0  # Increase spacing
    horizontal_offset = 0.2  # Spread final nodes
    for agg_node, cycles in final_cycles_by_node.items():
        x0, y0 = positions[agg_node]
        k = len(cycles)
        for idx, cycle in enumerate(cycles):
            final_node_id = ('final', agg_node, idx)
            G.add_node(final_node_id)
            offset = (idx - (k - 1) / 2) * horizontal_offset
            positions[final_node_id] = (x0 + offset, y0 - vertical_gap)
            labels[final_node_id] = ""
            G.add_edge(agg_node, final_node_id)
            edge_labels[(agg_node, final_node_id)] = str(cycle)
            final_nodes.append(final_node_id)

    # Draw graph
    plt.figure(figsize=(8, 6))
    ax = plt.gca()

    # Draw nodes
    node_colors = ['white' if n in final_nodes else 'skyblue' for n in G.nodes]
    nx.draw(G, pos=positions, with_labels=False, labels=labels, font_size=8, font_weight='bold',
            node_color=node_colors, node_size=750, edge_color="black", ax=ax)

    # Draw edge labels
    nx.draw_networkx_edge_labels(G, pos=positions, edge_labels=edge_labels, font_size=10, ax=ax, rotate=False)

    plt.title("Instance Graph")
    plt.axis('off')
    plt.show()

def plot_interactive_traffic(instance, Fs=None, weight_matrix=None):
    """
    Plot the instance graph with interactive packet selection.
    Allows users to select which packets to display on the graph.

    Parameters
    ----------
    instance : List of list of lists
        The network instance.
    Fs : List of list of lists, optional
        The scheduling matrix.
    weight_matrix : List of list of lists, optional
        The weight matrix.
    """
    import matplotlib.widgets as widgets
    mpl.use('macosx')

    # Create a simple Graph and prepare nodes
    G = nx.Graph()
    positions = {}
    labels = {}
    nlevels = len(instance)
    levels_nodes = []
    final_cycles_by_node = {}

    # Helper function to extract routes from a subset
    def extract_routes(subset):
        routes = set()
        for cycle in subset:
            for route in cycle:
                if isinstance(route, np.integer):
                    routes.add(int(route))
                else:
                    routes.add(route)
        return routes

    # Build nodes for each level (same as before)
    for lvl, level in enumerate(instance):
        current_level_nodes = []
        if lvl == nlevels - 1:
            for i, subset in enumerate(level):
                aggregated_routes = extract_routes(subset)
                node_id = f"L{lvl}_{i}"
                G.add_node(node_id)
                positions[node_id] = (lvl, i - len(level) / 2)
                labels[node_id] = str(sorted(list(aggregated_routes)))
                current_level_nodes.append(node_id)
                final_cycles_by_node[node_id] = aggregated_routes
        else:
            for i, subset in enumerate(level):
                node_id = f"L{lvl}_{i}"
                G.add_node(node_id)
                positions[node_id] = (lvl, i - len(level) / 2)
                aggregated_routes = extract_routes(subset)
                labels[node_id] = str(sorted(list(aggregated_routes)))
                current_level_nodes.append(node_id)
        levels_nodes.append(current_level_nodes)

    # Build edges between levels (same as before)
    edge_labels = {}
    final_nodes = []
    for lvl in range(nlevels - 1):
        for i, node_id in enumerate(levels_nodes[lvl]):
            for j, next_node_id in enumerate(levels_nodes[lvl + 1]):
                if lvl == nlevels - 2:
                    current_routes = set(int(x) for x in eval(labels[node_id]))
                    next_routes = final_cycles_by_node[next_node_id]
                    if current_routes & next_routes:
                        G.add_edge(node_id, next_node_id)
                        edge_labels[(node_id, next_node_id)] = ""
                        final_nodes.append(next_node_id)
                else:
                    current_routes = set(int(x) for x in eval(labels[node_id]))
                    next_routes = set(int(x) for x in eval(labels[next_node_id]))
                    if current_routes & next_routes:
                        G.add_edge(node_id, next_node_id)
                        edge_labels[(node_id, next_node_id)] = ""

    # Collect all packets information
    all_packets = []
    if Fs is not None and weight_matrix is not None:
        for (u, v) in G.edges():
            u_level = int(u.split('_')[0][1])
            if u_level < len(Fs):
                for route_idx in range(len(weight_matrix)):
                    if route_idx < len(weight_matrix) and u_level < len(weight_matrix[route_idx]):
                        weight = weight_matrix[route_idx][u_level][0] if isinstance(weight_matrix[route_idx][u_level], list) else weight_matrix[route_idx][u_level]
                        
                        # Find all packets for this route
                        for i in range(len(Fs[u_level])):
                            for j in range(len(Fs[u_level][i])):
                                packet_value = Fs[u_level][i][j][route_idx]
                                if isinstance(packet_value, np.ndarray):
                                    if packet_value.any() > 0:
                                        packet_info = {'route': route_idx, 'weight': weight, 'level': u_level, 'edge': (u, v)}
                                        if packet_info not in all_packets:
                                            all_packets.append(packet_info)
                                else:
                                    if packet_value > 0:
                                        packet_info = {'route': route_idx, 'weight': weight, 'level': u_level, 'edge': (u, v)}
                                        if packet_info not in all_packets:
                                            all_packets.append(packet_info)

    # Create the main figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 8), gridspec_kw={'width_ratios': [3, 1]})
    plt.subplots_adjust(right=0.85)

    # Function to update the plot based on selected packets
    def update_plot(selected_packets):
        ax1.clear()
        
        # Draw nodes
        node_colors = ['white' if n in final_nodes else 'skyblue' for n in G.nodes]
        nx.draw(G, pos=positions, with_labels=True, labels=labels, font_size=8,
                font_weight='bold', node_color=node_colors, node_size=750,
                edge_color='lightgray', ax=ax1, width=0.5)

        # Draw selected packets
        colors = plt.cm.rainbow(np.linspace(0, 1, len(selected_packets)))
        for idx, packet in enumerate(selected_packets):
            u, v = packet['edge']
            curve_rad = 0.15 * (1 + idx * 0.15)
            edge_color = colors[idx]
            
            nx.draw_networkx_edges(G, positions, edgelist=[(u, v)],
                                 edge_color=[edge_color], width=1.5,
                                 connectionstyle=f'arc3, rad={curve_rad}',
                                 arrows=True, arrowsize=5, ax=ax1)
            
            # Add label
            x1, y1 = positions[u]
            x2, y2 = positions[v]
            dx = x2 - x1
            dy = y2 - y1
            dist = np.sqrt(dx*dx + dy*dy)
            dx = dx / dist
            dy = dy / dist
            mid_x = (x1 + x2) / 2 - curve_rad * dy * 1.5
            mid_y = (y1 + y2) / 2 + curve_rad * dx * 1.5

            ax1.text(mid_x, mid_y, f'R{packet["route"]}:{packet["weight"]:.1f}',
                    color=edge_color, fontsize=7, ha='center', va='center',
                    bbox=dict(facecolor='white', edgecolor='none', alpha=0.8),
                    zorder=3)

        ax1.set_title("Interactive Traffic Visualization")
        ax1.axis('off')
        fig.canvas.draw_idle()

    # Create checkboxes for packet selection
    packet_labels = [f'Route {p["route"]} (L{p["level"]}, W:{p["weight"]:.1f})' for p in all_packets]
    checkbox_ax = plt.axes([0.87, 0.1, 0.1, 0.8])
    checkbox = widgets.CheckButtons(checkbox_ax, packet_labels, [False] * len(packet_labels))

    def on_checkbox_click(label):
        # Get all selected packets
        selected_indices = [i for i, status in enumerate(checkbox.get_status()) if status]
        selected_packets = [all_packets[i] for i in selected_indices]
        update_plot(selected_packets)

    checkbox.on_clicked(on_checkbox_click)

    # Initial plot with no packets selected
    update_plot([])
    plt.show()

def plot_instance_with_tracked_packets(instance, all_packets, base_curvature=0.2, max_curvature=0.8, single_packet_curvature=0.1, alternate_direction=True):
    """
    Plots the instance as a layered graph with packets from track_all_packets, 
    each packet having its own color during the route with curved edges for better readability.
    
    Parameters
    ----------
    instance : List of list of lists
        The network instance.
    all_packets : Dict[int, List[Dict]]
        Result from track_all_packets function, containing packet paths.
        Dictionary mapping route_id to a list of packets, where each packet is a dictionary:
        - packet_id: Unique identifier for the packet
        - path: List of tuples (level, router, cycle, time_slot) showing packet's path
    base_curvature : float, optional
        Minimum curvature for edges with multiple packets (default: 0.2)
    max_curvature : float, optional
        Maximum curvature for edges with multiple packets (default: 0.8)
    single_packet_curvature : float, optional
        Curvature for edges with only one packet (default: 0.1)
    alternate_direction : bool, optional
        Whether to alternate the direction of curvature for better visualization (default: True)
    """
    # Create a simple Graph and prepare nodes
    G = nx.Graph()
    positions = {}
    labels = {}
    nlevels = len(instance)
    levels_nodes = []
    final_cycles_by_node = {}
    
    # Build nodes for each level
    for lvl, level in enumerate(instance):
        current_level_nodes = []
        if lvl == nlevels - 1:
            for i, subset in enumerate(level):
                aggregated_routes = set()
                for cycle in subset:
                    aggregated_routes.update(cycle)
                node_id = (lvl, i)
                G.add_node(node_id)
                labels[node_id] = str(sorted(aggregated_routes))
                current_level_nodes.append((node_id, aggregated_routes))
                final_cycles_by_node[node_id] = subset  # Store original cycles
        else:
            for i, subset in enumerate(level):
                aggregated_routes = set()
                for cycle in subset:
                    aggregated_routes.update(cycle)
                node_id = (lvl, i)
                G.add_node(node_id)
                labels[node_id] = str(sorted(aggregated_routes))
                current_level_nodes.append((node_id, aggregated_routes))
        levels_nodes.append(current_level_nodes)
    
    # Assign positions
    for lvl, nodes_in_level in enumerate(levels_nodes):
        count = len(nodes_in_level)
        for i, (node_id, _) in enumerate(nodes_in_level):
            x_pos = (i + 1) / (count + 1) if count > 0 else 0.5
            y_pos = -lvl
            positions[node_id] = (x_pos, y_pos)
    
    # Add edges between levels (shared routes on a simple graph)
    edge_labels = {}
    for lvl in range(nlevels - 1):
        for (u_id, u_routes) in levels_nodes[lvl]:
            for (v_id, v_routes) in levels_nodes[lvl + 1]:
                shared = u_routes.intersection(v_routes)
                if shared:
                    G.add_edge(u_id, v_id)
                    edge_labels[(u_id, v_id)] = ",".join(map(str, sorted(shared)))
    
    # Final level exits
    final_nodes = []
    vertical_gap = 1.0
    horizontal_offset = 0.2
    for agg_node, cycles in final_cycles_by_node.items():
        x0, y0 = positions[agg_node]
        k = len(cycles)
        for idx, cycle in enumerate(cycles):
            final_node_id = ('final', agg_node, idx)
            G.add_node(final_node_id)
            offset = (idx - (k - 1) / 2) * horizontal_offset
            positions[final_node_id] = (x0 + offset, y0 - vertical_gap)
            labels[final_node_id] = ""
            G.add_edge(agg_node, final_node_id)
            edge_labels[(agg_node, final_node_id)] = str(cycle)
            final_nodes.append(final_node_id)
    
    # Draw graph
    plt.figure(figsize=(12, 10))
    ax = plt.gca()
    
    # Draw nodes
    node_colors = ['white' if n in final_nodes else 'skyblue' for n in G.nodes]
    nx.draw(G, pos=positions, with_labels=False, labels=labels, font_size=8, font_weight='bold',
            node_color=node_colors, node_size=750, edge_color="lightgray", alpha=0.5, ax=ax, width=1.0)
    
    # Draw edge labels
    nx.draw_networkx_edge_labels(G, pos=positions, edge_labels=edge_labels, font_size=10, ax=ax, rotate=False)
    
    # Create a colormap for packets
    total_packets = sum(len(packets) for packets in all_packets.values())
    colormap = plt.cm.get_cmap('tab20', total_packets)
    
    # Track which edges have been used by which packets to adjust curve
    edge_packet_count = {}
    
    # First pass to count packets per edge for curve adjustment
    for route_id, packets in all_packets.items():
        for packet in packets:
            packet_path = packet['path']
            for i in range(len(packet_path) - 1):
                level1, router1, _, _ = packet_path[i]
                level2, router2, _, _ = packet_path[i+1]
                
                node1 = (level1, router1)
                node2 = (level2, router2)
                
                if G.has_edge(node1, node2):
                    edge = tuple(sorted([node1, node2]))  # Ensure consistent edge representation
                    if edge not in edge_packet_count:
                        edge_packet_count[edge] = 0
                    edge_packet_count[edge] += 1
    
    # Plot each packet's path with a unique color and curved edges
    packet_counter = 0
    for route_id, packets in all_packets.items():
        for packet in packets:
            packet_path = packet['path']
            packet_id = packet['packet_id']
            
            # Create a list of edges representing the packet's path
            path_edges = []
            for i in range(len(packet_path) - 1):
                level1, router1, _, _ = packet_path[i]
                level2, router2, _, _ = packet_path[i+1]
                
                node1 = (level1, router1)
                node2 = (level2, router2)
                
                if G.has_edge(node1, node2):
                    path_edges.append((node1, node2))
            
            # Draw the packet's path with a unique color and curved edges
            if path_edges:
                color = colormap(packet_counter)
                
                # Draw each edge with a curved path
                for edge in path_edges:
                    n1, n2 = edge
                    x1, y1 = positions[n1]
                    x2, y2 = positions[n2]
                    
                    # Determine the edge index for this packet on this edge
                    edge_sorted = tuple(sorted([n1, n2]))
                    total_packets_on_edge = edge_packet_count[edge_sorted]
                    
                    # Calculate curvature based on packet position and total packets on this edge
                    if total_packets_on_edge > 1:
                        # Distribute packets evenly around the edge
                        packet_idx = packet_counter % total_packets_on_edge
                        # Scale curvature based on number of packets
                        curvature_step = (max_curvature - base_curvature) / (total_packets_on_edge - 1)
                        curvature = base_curvature + packet_idx * curvature_step
                        
                        # Alternate the direction of curvature for better visualization
                        if alternate_direction and packet_idx % 2 == 1:
                            curvature = -curvature
                    else:
                        # Single packet on edge - slight curve for consistency
                        curvature = single_packet_curvature
                    
                    # Create curved connection
                    # Calculate the midpoint
                    mid_x = (x1 + x2) / 2
                    mid_y = (y1 + y2) / 2
                    
                    # Calculate the perpendicular offset for the control point
                    dx = x2 - x1
                    dy = y2 - y1
                    dist = np.sqrt(dx**2 + dy**2)
                    
                    # Perpendicular direction
                    perpendicular_x = -dy / dist
                    perpendicular_y = dx / dist
                    
                    # Control point with curvature
                    control_x = mid_x + curvature * perpendicular_x
                    control_y = mid_y + curvature * perpendicular_y
                    
                    # Create curved path
                    curve_path = mpl.path.Path(
                        [(x1, y1), (control_x, control_y), (x2, y2)],
                        [mpl.path.Path.MOVETO, mpl.path.Path.CURVE3, mpl.path.Path.CURVE3]
                    )
                    
                    # Draw the curved edge
                    patch = mpl.patches.PathPatch(
                        curve_path, 
                        facecolor='none', 
                        edgecolor=color, 
                        linewidth=2.5, 
                        alpha=0.8,
                        label=f"Packet {packet_id} (Route {route_id})" if edge == path_edges[0] else ""
                    )
                    ax.add_patch(patch)
                
                # Highlight nodes in the path
                path_nodes = [(level, router) for level, router, _, _ in packet_path]
                nx.draw_networkx_nodes(
                    G, positions,
                    nodelist=path_nodes,
                    node_color=color,
                    node_size=300,
                    alpha=0.6
                )
                
            packet_counter += 1
    
    # Add a legend for packets - only include one entry per packet
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    if by_label:
        plt.legend(by_label.values(), by_label.keys(), loc='upper right', bbox_to_anchor=(1.15, 1))
    
    plt.title("Instance with Packet Routes")
    plt.axis('off')
    plt.tight_layout()
    plt.show()
