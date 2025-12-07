import networkx as nx
import re, math, json
from collections import defaultdict
import matplotlib.pyplot as plt

def load_and_transform_data(data):
    transformed_results = []

    for page_info in data['pages']:
        page_index = page_info['page_index']
        boxes = page_info['boxes']

        if not boxes:
            continue

        for i, box in enumerate(boxes):
            node = {
                'id': f"pg{page_index}_box{i}",
                'type': box['label'],
                'page': page_index,
                'bbox': [
                    box['coordinate'][0],
                    box['coordinate'][1],
                    box['coordinate'][2],
                    box['coordinate'][3]
                ]
            }

            if 'text' in box:
                node['text'] = box['text']
            if 'ref_info' in box:
                node['ref_info'] = box['ref_info']
            if 'section_info' in box:
                node['section_info'] = box['section_info']

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

def _get_node_center(node):
    bbox = node['bbox']
    center_x = (bbox[0] + bbox[2]) / 2
    center_y = (bbox[1] + bbox[3]) / 2
    return center_x, center_y

def _get_distance(node1, node2):
    x1, y1 = _get_node_center(node1)
    x2, y2 = _get_node_center(node2)
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

def _add_sequence_edges(graph, page_nodes):
    sorted_nodes = sorted(page_nodes, key=lambda n: n['bbox'][1])

    left_boxes = []
    right_boxes = []
    for node in sorted_nodes:
        if node['bbox'][0] < 0.4:
            left_boxes.append(node)
        else:
            right_boxes.append(node)
    if len(right_boxes) > len(sorted_nodes)*0.3:
        sorted_nodes = left_boxes + right_boxes

    for i in range(len(sorted_nodes)-1):
        node1 = sorted_nodes[i]
        node2 = sorted_nodes[i+1]
        graph.add_edge(node1['id'], node2['id'], type='sequence')

def _add_hierarchical_edges(graph, all_nodes):
    section_node_ids = list(range(1, 30))

    for sec_id in section_node_ids:
        graph.add_node(f"Section_{sec_id}", node_type="section")

    for node_id, attrs in graph.nodes(data=True):
        if 'section_info' in attrs:
            target_section_node = f"Section_{int(float(attrs['section_info']))}"
            if graph.has_node(target_section_node):
                graph.add_edge(node_id, target_section_node, type='hierarchical')

def build_document_graph(processed_data):
    G = nx.DiGraph()
    all_nodes = []
    for node_data in processed_data:
        if node_data['type'] in IGNORED_NODE_TYPES:
            continue

        node_attrs = node_data.copy()
        G.add_node(node_attrs['id'], node_type='doc_component', **node_attrs)
        all_nodes.append(node_attrs)

    nodes_by_page = defaultdict(list)
    for node in all_nodes:
        nodes_by_page[node['page']].append(node)

    for page_num, page_nodes in nodes_by_page.items():
        _add_sequence_edges(G, page_nodes)

    _add_hierarchical_edges(G, all_nodes)

    return G

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

def find_target_with_name(scope_candidates, ref_item):
    label_pattern = r'\b(Figure|Fig|Table|Formula|Algorithm|Chart|Equation|Eq)\s*\.?\s*\(?(\d+(\.\d+)?|[A-Za-z]+)'
    label_pattern_1 = r'\b(\d+(\.\d+)?|[A-Za-z]+)\s*\.?\s*(Figure|Fig|Table|Formula|Algorithm|Chart|Equation|Eq)'
    equation_pattern = r'\b(Equation|Eq)\s*\.?\s*\(?\s*(\d+)\s*\)?'

    target_text = ref_item.get('figure_text', '')
    match = re.search(label_pattern, target_text, re.IGNORECASE)
    best_match = None

    if not match:
        match = re.search(equation_pattern, target_text, re.IGNORECASE)

    if match:
        t_kind = match.group(1).lower()
        if t_kind == 'fig': t_kind = 'figure'
        if t_kind == 'eq': t_kind = 'equation'
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

    return best_match

def create_reference_pairs(graph):
    target_nodes_list = []
    source_nodes_list = []
    for node_id, attrs in graph.nodes(data=True):
        if attrs.get('type') in ['image', 'table', 'figure', 'chart', 'algorithm', 'formula']:
            node_data = attrs.copy()
            node_data['id'] = node_id
            target_nodes_list.append(node_data)
        if attrs.get('type') == 'text':
            node_data = attrs.copy()
            node_data['id'] = node_id
            source_nodes_list.append(node_data)

    pairs = []

    for source_attrs in source_nodes_list:
        source_id = source_attrs.get('id')
        if 'ref_info' not in source_attrs:
            continue

        for ref_item in source_attrs['ref_info']:
            scope = []
            if ref_item['section_info']:
                try:
                    section_node_id = f"Section_{int(float(ref_item.get('section_info')[0]))}"
                    if not graph.has_node(section_node_id):
                        scope = []
                    for u, v, data in graph.in_edges(section_node_id, data=True):
                        if data.get('type') == 'hierarchical':
                            node_data = graph.nodes[u]
                            scope.append(node_data)
                except Exception:
                    scope = target_nodes_list
            else:
                scope = target_nodes_list

            best_match = find_target_with_name(scope, ref_item)

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

def save_graph_to_img(graph: nx.Graph):
    plt.figure(figsize=(30, 30))
    for u, v, d in graph.edges(data=True):
        if d.get('type') == 'sequence':
            graph[u][v]['weight'] = 2.0
        elif d.get('type') == 'hierarchical':
            graph[u][v]['weight'] = 0.1
        else:
            graph[u][v]['weight'] = 0.5

    pos = nx.spring_layout(graph, k=0.8, iterations=300, weight='weight', seed=42)

    node_color_map = {
        'doc_title'         : '#FFD700',
        'paragraph_title'   : '#FFA500',
        'text'              : '#87CEEB',
        'figure'            : '#FA8072',
        'formula'           : '#90EE90',
        'number'            : '#D3D3D3',
        'chart'             : '#FF0000',
        'table'             : '#FFC0CB',
        'algorithm'         : '#006400',
        'section'           : '#A9A9A9',
        'doc_component'     : '#DCDCDC'
    }

    node_colors = []
    for n in graph.nodes():
        node_attrs = graph.nodes[n]
        color_key = node_attrs.get('type', node_attrs.get('node_type', 'doc_component'))
        node_colors.append(node_color_map.get(color_key, '#DCDCDC'))

    nx.draw_networkx_nodes(
        graph, pos,
        node_color=node_colors,
        node_size=500,
        alpha=0.9,
        edgecolors='white',
        linewidths=1.0
    )

    sequence_edges = [(u, v) for u, v, d in graph.edges(data=True) if d.get('type') == 'sequence']
    hierarchical_edges = [(u, v) for u, v, d in graph.edges(data=True) if d.get('type') == 'hierarchical']
    other_edges = [(u, v) for u, v, d in graph.edges(data=True) if d.get('type') not in ['sequence', 'hierarchical']]

    nx.draw_networkx_edges(
        graph, pos,
        edgelist=sequence_edges,
        width=1.2,
        alpha=0.6,
        edge_color='brown',
        arrowsize=12,
        connectionstyle='arc3,rad=0.1'
    )

    nx.draw_networkx_edges(
        graph, pos,
        edgelist=hierarchical_edges,
        width=0.8,
        alpha=0.4,
        edge_color='gray',
        style='dashed',
        arrowsize=10,
        connectionstyle='arc3,rad=-0.1'
    )

    if other_edges:
        nx.draw_networkx_edges(
            graph, pos,
            edgelist=other_edges,
            width=0.5,
            alpha=0.2,
            edge_color='lightgray'
        )

    plt.axis('off')
    plt.tight_layout()
    plt.savefig('graph.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()

if __name__ == '__main__':
    with open('/home/gyupil/ocr/data/temp/Fast and secure IPC for microkernel.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
        graph = build_document_graph(load_and_transform_data(data))
        save_graph_to_img(graph)