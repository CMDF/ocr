import networkx as nx
import re
import math
from collections import defaultdict
import json
from networkx.readwrite import json_graph
import matplotlib.pyplot as plt

# Transforms raw data into a list of graph node dictionaries.
def load_and_transform_data(data):
    transformed_results = []

    for page_info in data['pages']:
        page_index = page_info['page_index']
        boxes = page_info['boxes']

        if not boxes:
            continue

        for i, box in enumerate(boxes):
            node = {}
            coord = box['coordinate']

            node['id'] = f"pg{page_index}_box{i}"
            node['type'] = box['label']

            if box['cls_id'] == 99 and box['label'] == 'figure':
                node['type'] = 'figure'

            node['page'] = page_index
            node['conf'] = box['score']
            if 'text' in box:
                node['text'] = box['text']

            node['bbox'] = [
                coord[0],
                coord[1],
                coord[2],
                coord[3]
            ]

            transformed_results.append(node)

    return transformed_results

VALID_NODE_TYPES = [
    "doc_title", "paragraph_title", "section",
    "text", "abstract", "references", "sidebar_text",
    "formula", "algorithm",
    "table", "figure",
    "formula_number", "page_number", "number", "footnote", "chart_title"
]
IGNORED_NODE_TYPES = ["header", "footer", "header_image", "footer_image", "seal"]

# Returns the center coordinates (x, y) of a node's bounding box (bbox).
def _get_node_center(node):
    bbox = node['bbox']
    center_x = (bbox[0] + bbox[2]) / 2
    center_y = (bbox[1] + bbox[3]) / 2
    return center_x, center_y

# Calculates the Euclidean distance between the centers of two nodes.
def _get_distance(node1, node2):
    x1, y1 = _get_node_center(node1)
    x2, y2 = _get_node_center(node2)
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

# Adds 'sequence' edges by sorting nodes on a page by their y-coordinate.
def _add_sequence_edges(graph, page_nodes):
    sorted_nodes = sorted(page_nodes, key=lambda n: n['bbox'][1])
    for i in range(len(sorted_nodes) - 1):
        node1 = sorted_nodes[i]
        node2 = sorted_nodes[i + 1]
        if abs(node2['bbox'][1] - node1['bbox'][1]) < 0.2:
            graph.add_edge(node1['id'], node2['id'], type='sequence')

# Adds 'spatial' edges by finding the nearest 'up', 'down', 'left', and 'right' neighbor for each node on a page.
def _add_spatial_edges(graph, page_nodes):
    for node1 in page_nodes:
        center1_x, center1_y = _get_node_center(node1)
        neighbors = {'up': None, 'down': None, 'left': None, 'right': None}
        min_dists = {'up': float('inf'), 'down': float('inf'), 'left': float('inf'), 'right': float('inf')}

        for node2 in page_nodes:
            if node1['id'] == node2['id']: continue
            center2_x, center2_y = _get_node_center(node2)
            dist = _get_distance(node1, node2)

            if center2_y < center1_y and dist < min_dists['up']:
                min_dists['up'], neighbors['up'] = dist, node2
            elif center2_y > center1_y and dist < min_dists['down']:
                min_dists['down'], neighbors['down'] = dist, node2
            if center2_x < center1_x and dist < min_dists['left']:
                min_dists['left'], neighbors['left'] = dist, node2
            elif center2_x > center1_x and dist < min_dists['right']:
                min_dists['right'], neighbors['right'] = dist, node2

        for direction, neighbor_node in neighbors.items():
            if neighbor_node:
                graph.add_edge(node1['id'], neighbor_node['id'], type='spatial', dir=direction)

# Adds parent-child 'hierarchical' edges based on document titles, sections, etc.
def _add_hierarchical_edges(graph, all_nodes):
    sorted_nodes = sorted(all_nodes, key=lambda n: (n['page'], n['bbox'][1]))
    parent_stack = []
    for node in sorted_nodes:
        node_type = node['type']
        if node_type == 'doc_title':
            parent_stack.clear()
            parent_stack.append(node)
        elif node_type in ['paragraph_title', 'section']:
            while parent_stack and parent_stack[-1]['type'] not in ['doc_title']:
                parent_stack.pop()
            parent_stack.append(node)

        if parent_stack and node['id'] != parent_stack[-1]['id']:
            parent_node = parent_stack[-1]
            graph.add_edge(parent_node['id'], node['id'], type='hierarchical', rel='child')

# Builds the final document graph by adding nodes and orchestrating edge creation (sequence, spatial, hierarchical).
def build_document_graph(processed_data):
    G = nx.DiGraph()
    all_nodes = []
    for node_data in processed_data:
        if node_data['type'] in IGNORED_NODE_TYPES: continue

        node_attrs = node_data.copy()

        G.add_node(node_attrs['id'], **node_attrs)
        all_nodes.append(node_attrs)

    nodes_by_page = defaultdict(list)
    for node in all_nodes:
        nodes_by_page[node['page']].append(node)

    for page_num, page_nodes in nodes_by_page.items():
        _add_sequence_edges(G, page_nodes)
        _add_spatial_edges(G, page_nodes)

    _add_hierarchical_edges(G, all_nodes)

    return G

# Creates a {child_id: parent_id} map by traversing 'hierarchical' edges.
def _get_parent_section_map(graph):
    parent_map = {}
    for source_id, dest_id, attrs in graph.edges(data=True):
        if attrs.get('type') == 'hierarchical':
            parent_map[dest_id] = source_id

    return parent_map

# Finds reference pairs between text (source) and objects (target) and returns a {source_id: target_node} map.
def create_reference_pairs(graph):
    target_nodes = defaultdict(list)
    label_pattern = r'\b(Figure|Fig|Table|Formula|Algorithm)\.?\s*(\d+(\.\d+)?)'
    for node_id, attrs in graph.nodes(data=True):
        if attrs.get('type') in ['figure', 'table', 'formula', 'algorithm']:
            match = re.search(label_pattern, attrs.get('text', ''), re.IGNORECASE)
            if match:
                kind = match.group(1).lower()
                if kind == 'fig':
                    kind = 'figure'
                num = match.group(2)
                key = f"{kind}:{num}"
                target_nodes[key].append(attrs)

    parent_map = _get_parent_section_map(graph)

    reference_pairs = {}
    text_nodes = [attrs for _, attrs in graph.nodes(data=True) if attrs.get('type') == 'text']

    for source_node in text_nodes:
        ref_string = source_node.get('text', '')
        match = re.match(label_pattern, ref_string.strip(), re.IGNORECASE)

        if match:
            kind, num = match.group(1).lower(), match.group(2)
            if kind == 'fig':
                kind = 'figure'
            ref_key = f"{kind}:{num}"

            candidates = target_nodes.get(ref_key, [])
            if not candidates:
                continue

            best_match = None
            if len(candidates) > 1:
                source_parent = parent_map.get(source_node['id'])

                if source_parent:
                    for candidate in candidates:
                        if parent_map.get(candidate['id']) == source_parent:
                            best_match = candidate
                            break

                if not best_match:
                    best_match = min(candidates, key=lambda c: abs(c['page'] - source_node['page']))

            else:
                best_match = candidates[0]

            if best_match:
                reference_pairs[source_node['id']] = best_match

    return reference_pairs

# Finds and returns a list of nodes matching given key-value conditions.
def find_nodes(graph, **conditions):
    found_nodes = []

    for node_id, attrs in graph.nodes(data=True):
        match = True
        for key, condition_value in conditions.items():
            attr_value = attrs.get(key)

            if callable(condition_value):
                if not condition_value(attr_value):
                    match = False
                    break
            else:
                if attr_value != condition_value:
                    match = False
                    break

        if match:
            found_nodes.append(attrs)

    return found_nodes

# Serializes and saves a NetworkX graph to a JSON file.
def save_graph_to_json(graph, filepath: str):
    graph_data = json_graph.node_link_data(graph, edges='edges')

    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(graph_data, f, indent=4, ensure_ascii=False)

# Loads and deserializes a NetworkX graph object from a JSON file.
def load_graph_from_json(filepath='document_graph.json'):
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
            if 'links' in data and 'edges' not in data:
                data['edges'] = data.pop('links')
            return json_graph.node_link_graph(data, edges='edges')
    except FileNotFoundError:
        print(f"오류: '{filepath}' 파일을 찾을 수 없습니다.")
        return None

# Visualizes the graph using Matplotlib and saves it as 'graph.png'.
def save_graph_to_img(graph: nx.Graph):
    plt.figure(figsize=(30, 30))
    pos = nx.spring_layout(graph, k=0.5, iterations=100)
    node_color_map = {
        'doc_title': 'gold',
        'paragraph_title': 'orange',
        'text': 'skyblue',
        'figure': 'salmon',
        'formula': 'lightgreen',
        'number': 'lightgray'
    }

    edge_color_map = {
        'hierarchical': 'black',
        'spatial': 'darkgray',
        'ref:figure': 'red'
    }

    node_colors = [node_color_map.get(graph.nodes[n]['type'], 'gray') for n in graph.nodes()]
    edge_colors = [edge_color_map.get(graph.edges[e]['type'], 'gray') for e in graph.edges()]

    nx.draw_networkx(
        graph,
        pos=pos,
        node_size=1000,
        node_color=node_colors,
        edge_color=edge_colors,
        font_size=20,
        width=0.5,
        alpha=0.8
    )
    plt.savefig('graph.png', dpi=300, bbox_inches='tight')
    plt.show()