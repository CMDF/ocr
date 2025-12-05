import networkx as nx
import re, math, json
from collections import defaultdict
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

            if box['cls_id'] == 99 and box['label'] in ['image', 'table', 'figure', 'chart', 'algorithm']:
                node['type'] = box['label']

            node['page'] = page_index
            node['conf'] = box['score']
            if 'text' in box:
                node['text'] = box['text']
            if 'ref_info' in box:
                node['ref_info'] = box['ref_info']

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
    "table", "figure", "chart", "image",
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
    # length_validation = len(sorted_nodes) > 3
    # if length_validation:
    #     y_comparison_first_two = abs(sorted_nodes[0]['bbox'][1] - sorted_nodes[1]['bbox'][1]) < 0.0001 and sorted_nodes[0]['bbox'][1] != sorted_nodes[1]['bbox'][1]
    #     y_comparison_next_two = abs(sorted_nodes[1]['bbox'][1] - sorted_nodes[2]['bbox'][1]) < 0.0001 and sorted_nodes[1]['bbox'][1] != sorted_nodes[2]['bbox'][1]
    #
    #     if y_comparison_first_two and not y_comparison_next_two:
    #         left_boxes = []
    #         right_boxes = []
    #         for target_nodes in sorted_nodes:
    #             if sorted_nodes[0].get('bbox')[0] < 0.4:
    #                 left_boxes.append(target_nodes)
    #             else:
    #                 right_boxes.append(target_nodes)
    #         sorted_nodes = left_boxes + right_boxes

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
def _get_hierarchical_ancestors(graph, node_id):
    ancestors = {
        'paragraph_title': set(),
        'doc_title': set()
    }

    queue = list(graph.predecessors(node_id))
    visited = {node_id}

    while queue:
        parent_id = queue.pop(0)
        if parent_id in visited:
            continue
        visited.add(parent_id)

        try:
            edge_data = graph.get_edge_data(parent_id, node_id)
            node_attrs = graph.nodes[parent_id]
        except KeyError:
            continue

        if edge_data and edge_data.get('type') == 'hierarchical':
            node_type = node_attrs.get('type')

            if node_type in ['paragraph_title', 'section']:
                ancestors['paragraph_title'].add(parent_id)
            elif node_type == 'doc_title':
                ancestors['doc_title'].add(parent_id)

            for grand_parent_id in graph.predecessors(parent_id):
                if grand_parent_id not in visited:
                    queue.append(grand_parent_id)

        node_id = parent_id

    return ancestors

# Finds reference pairs between text (source) and objects (target) and returns a {source_id: target_node} map.
def create_reference_pairs(graph):
    target_nodes_list = []
    for node_id, attrs in graph.nodes(data=True):
        if attrs.get('type') in ['image', 'table', 'figure', 'chart', 'algorithm', 'formula']:
            node_data = attrs.copy()
            node_data['id'] = node_id
            sort_key = (
                attrs.get('page', 0),
                attrs.get('bbox', [0, 0, 0, 0])[1],
                attrs.get('bbox', [0, 0, 0, 0])[0]
            )
            target_nodes_list.append((sort_key, node_data))

    target_nodes_list.sort(key=lambda x: x[0])
    length_validation = len(target_nodes_list) > 3
    if length_validation:
        y_comparison_first_two = abs(target_nodes_list[0][1]['bbox'][1] - target_nodes_list[1][1]['bbox'][1]) < 0.0001 and target_nodes_list[0][1]['bbox'][1] != target_nodes_list[1][1]['bbox'][1]
        y_comparison_next_two = abs(target_nodes_list[1][1]['bbox'][1] - target_nodes_list[2][1]['bbox'][1]) < 0.0001 and target_nodes_list[1][1]['bbox'][1] != target_nodes_list[2][1]['bbox'][1]

        if y_comparison_first_two and not y_comparison_next_two:
            left_boxes = []
            right_boxes = []
            for target_nodes in target_nodes_list:
                if target_nodes_list[0][1].get('bbox')[0] < 0.4:
                    left_boxes.append(target_nodes)
                else:
                    right_boxes.append(target_nodes)
            target_nodes_list = left_boxes + right_boxes

    sorted_targets = [item[1] for item in target_nodes_list]
    targets_by_section = defaultdict(list)

    for target in sorted_targets:
        ancestors = _get_hierarchical_ancestors(graph, target['id'])
        parent_sections = ancestors.get('paragraph_title', set()) | \
                          ancestors.get('section', set()) | \
                          ancestors.get('doc_title', set())

        for sec_id in parent_sections:
            targets_by_section[sec_id].append(target)

    label_pattern = r'\b(Figure|Fig|Table|Formula|Algorithm|Chart|Equation|Eq)\s*\.?\s*\(?(\d+(\.\d+)?|[A-Za-z]+)'
    label_pattern_1 = r'\b(\d+(\.\d+)?|[A-Za-z]+)\s*\.?\s*(Figure|Fig|Table|Formula|Algorithm|Chart|Equation|Eq)'
    equation_pattern = r'\b(Equation|Eq)\s*\.?\s*\(?\s*(\d+)\s*\)?'
    pairs = []

    for source_id, source_attrs in graph.nodes(data=True):
        if 'ref_info' not in source_attrs:
            continue

        for ref_item in source_attrs['ref_info']:
            scope_candidates = sorted_targets

            best_match = None

            target_text = ref_item.get('figure_text', '')
            match = re.search(label_pattern, target_text, re.IGNORECASE)
            if not match:
                match = re.search(equation_pattern, target_text, re.IGNORECASE)

            if match:
                t_kind = match.group(1).lower()
                if t_kind == 'fig': t_kind = 'figure'
                if t_kind == 'eq' : t_kind = 'equation'
                t_num = match.group(2)

                for target in scope_candidates:
                    tm = re.search(label_pattern_1, target.get('text', ''), re.IGNORECASE)
                    if not tm:
                        tm = re.search(label_pattern, target.get('text', ''), re.IGNORECASE)
                        if not tm:
                            tm = re.search(r'\(\s*(\d+)\s*\)', target.get('text', ''))
                            if tm:
                                tn = tm.group(1)
                                tk = "equation"
                        else:
                            tk = tm.group(1).lower()
                            if tk == 'fig': tk = 'figure'
                            tn = tm.group(2)
                    else:
                        tk = tm.group(3).lower()
                        if tk == 'fig': tk = 'figure'
                        tn = tm.group(1)
                    if tm:
                        if t_kind == tk and t_num == tn:
                            best_match = target
                            break

            if best_match:
                pairs.append({
                    'source_id': source_id,
                    'page': source_attrs['page'],
                    'raw_text': ref_item['raw_text'],
                    'figure_text': ref_item['figure_text'],
                    'text_box': ref_item['text_box'],
                    'ref': best_match
                })

    return pairs

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
        'spatial': 'darkgray'
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