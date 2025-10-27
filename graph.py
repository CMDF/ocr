import networkx as nx
import re
import math
from collections import defaultdict
import json
from networkx.readwrite import json_graph
import matplotlib.pyplot as plt

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

def get_node_center(node):
    bbox = node['bbox']
    center_x = (bbox[0] + bbox[2]) / 2
    center_y = (bbox[1] + bbox[3]) / 2
    return center_x, center_y

def get_distance(node1, node2):
    x1, y1 = get_node_center(node1)
    x2, y2 = get_node_center(node2)
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

def extract_label_num(node):
    text = node.get("text", "")
    match = re.search(r'(figure|fig|table|formula|algorithm)\.?\s*(\d+)', text, re.IGNORECASE)
    if match:
        kind = match.group(1).lower()
        if kind == 'fig':
            kind = 'figure'
        num = match.group(2)
        return {"kind": kind, "num": num}
    return None

def extract_references(text):
    references = []
    matches = re.finditer(r'(Figure|Fig|Table|Formula|Algorithm)\.?\s*(\d+)', text, re.IGNORECASE)
    for match in matches:
        kind = match.group(1).lower()
        if kind == 'fig':
            kind = 'figure'
        num = match.group(2)
        references.append({"kind": kind, "num": num, "key": f"{kind}:{num}"})
    return references

def add_sequence_edges(graph, page_nodes):
    sorted_nodes = sorted(page_nodes, key=lambda n: n['bbox'][1])
    for i in range(len(sorted_nodes) - 1):
        node1 = sorted_nodes[i]
        node2 = sorted_nodes[i + 1]
        if abs(node2['bbox'][1] - node1['bbox'][1]) < 0.2:
            graph.add_edge(node1['id'], node2['id'], type='sequence')

def add_spatial_edges(graph, page_nodes):
    for node1 in page_nodes:
        center1_x, center1_y = get_node_center(node1)
        neighbors = {'up': None, 'down': None, 'left': None, 'right': None}
        min_dists = {'up': float('inf'), 'down': float('inf'), 'left': float('inf'), 'right': float('inf')}

        for node2 in page_nodes:
            if node1['id'] == node2['id']: continue
            center2_x, center2_y = get_node_center(node2)
            dist = get_distance(node1, node2)

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

def add_hierarchical_edges(graph, all_nodes):
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

def add_reference_edges(graph, all_nodes):
    target_nodes = defaultdict(list)

    label_pattern = r'\b(Figure|Fig|Table|Formula|Algorithm)\.?\s*(\d+(\.\d+)?)'

    for node in all_nodes:
        if node['type'] in ['figure', 'table', 'formula', 'algorithm']:
            caption_text = node.get("text", "")
            match = re.search(label_pattern, caption_text, re.IGNORECASE)

            if match:
                kind = match.group(1).lower()
                if kind == 'fig':
                    kind = 'figure'
                num = match.group(2)
                key = f"{kind}:{num}"
                target_nodes[key].append(node)

    for node in all_nodes:
        if node['type'] == 'text':
            ref_string = node.get("text", "")
            match = re.match(label_pattern, ref_string.strip(), re.IGNORECASE)

            if match:
                kind = match.group(1).lower()
                if kind == 'fig':
                    kind = 'figure'
                num = match.group(2)
                ref_key = f"{kind}:{num}"

                if ref_key in target_nodes:
                    candidates = target_nodes[ref_key]
                    best_target = min(
                        candidates,
                        key=lambda target: abs(target['page'] - node['page'])
                    )
                    graph.add_edge(
                        node['id'],
                        best_target['id'],
                        type=f"ref:{kind}",
                        attrs={'key': ref_key}
                    )

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
        add_sequence_edges(G, page_nodes)
        add_spatial_edges(G, page_nodes)

    add_hierarchical_edges(G, all_nodes)
    add_reference_edges(G, all_nodes)

    return G

def get_parent_section_map(graph):
    parent_map = {}
    for source_id, dest_id, attrs in graph.edges(data=True):
        if attrs.get('type') == 'hierarchical':
            parent_map[dest_id] = source_id

    return parent_map

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

    parent_map = get_parent_section_map(graph)

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

def get_referenced_nodes(graph, node_id):
    if not graph.has_node(node_id):
        print("그래프에 존재하지 않는 노드입니다.")
        return []
    referenced_nodes = []
    for _, destination_id, attrs in graph.edges(node_id, data=True):
        if attrs.get('type', '').startswith('ref:'):
            referenced_nodes.append(graph.nodes[destination_id])
    return referenced_nodes

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

def print_graph_details(graph):
    print("\n--- 📊 전체 그래프 상세 정보 ---")
    print(f"총 노드 수: {graph.number_of_nodes()}")
    print(f"총 엣지 수: {graph.number_of_edges()}")

    print("\n--- 노드 리스트 (ID와 속성) ---")
    for node_id, attrs in graph.nodes(data=True):
        print(f"Node '{node_id}': {attrs}")

    print("\n--- 엣지 리스트 (Source, Target, 속성) ---")
    for src, dst, attrs in graph.edges(data=True):
        print(f"Edge ('{src}' -> '{dst}'): {attrs}")

def save_graph_to_json(graph, filepath: str):
    graph_data = json_graph.node_link_data(graph, edges='edges')

    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(graph_data, f, indent=4, ensure_ascii=False)

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