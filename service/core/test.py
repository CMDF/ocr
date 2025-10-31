from graph import *

if __name__ == '__main__':
    graph = load_graph_from_json('document_graph.json')
    save_graph_to_img(graph)