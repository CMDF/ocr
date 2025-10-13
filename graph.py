def create_document_graph(doc_data):
    graph_nodes = []
    root_ids = []
    node_id_counter = 1

    current_doc_title_node = None
    current_paragraph_title_node = None

    for page in doc_data.get('pages', []):
        for box in page.get('boxes', []):

            new_node = {
                "node_id": node_id_counter,
                "label": box.get("label"),
                "page_index": page.get("page_index"),
                "parent_id": None,  # 나중에 결정
                "children_ids": [],
                "content_box": box
            }

            label = new_node["label"]

            if label == "doc_title":
                new_node["parent_id"] = None
                root_ids.append(new_node["node_id"])
                current_doc_title_node = new_node
                current_paragraph_title_node = None

            elif label == "paragraph_title":
                if current_doc_title_node:
                    new_node["parent_id"] = current_doc_title_node["node_id"]
                current_paragraph_title_node = new_node

            else:
                if current_paragraph_title_node:
                    new_node["parent_id"] = current_paragraph_title_node["node_id"]
                elif current_doc_title_node:
                    new_node["parent_id"] = current_doc_title_node["node_id"]

            graph_nodes.append(new_node)
            node_id_counter += 1

    node_map = {node["node_id"]: node for node in graph_nodes}
    for node in graph_nodes:
        parent_id = node.get("parent_id")
        if parent_id is not None and parent_id in node_map:
            parent_node = node_map[parent_id]
            parent_node["children_ids"].append(node["node_id"])

    document_graph = {
        "document_path": doc_data.get("document_path"),
        "graph_nodes": graph_nodes,
        "root_ids": root_ids
    }

    return document_graph

def display_document_tree(graph_data):
    nodes_map = {node['node_id']: node for node in graph_data.get('graph_nodes', [])}

    doc_path = graph_data.get("document_path", "경로 정보 없음")
    print(f"문서: {doc_path}\n")

    def print_node_recursively(node_id, prefix="", is_last=True):
        node = nodes_map.get(node_id)
        if not node:
            return

        connector = "└── " if is_last else "├── "

        label_info = f"{node['label']} (Page: {node['page_index']})"
        print(prefix + connector + label_info)

        children_ids = node.get('children_ids', [])
        for i, child_id in enumerate(children_ids):
            new_prefix = prefix + ("    " if is_last else "│   ")
            is_child_last = (i == len(children_ids) - 1)
            print_node_recursively(child_id, new_prefix, is_child_last)

    root_ids = graph_data.get('root_ids', [])
    if not root_ids:
        print("표시할 루트 노드가 없습니다.")
        return

    for i, root_id in enumerate(root_ids):
        is_root_last = (i == len(root_ids) - 1)
        print_node_recursively(root_id, prefix="", is_last=is_root_last)
        if not is_root_last:
            print("│")