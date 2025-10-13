from graph import *
from layout import *
import json

if __name__ == "__main__":
    layout_detection('/Users/neung_gae/Desktop/output')

    try:
        with open('document_structure.json', 'r', encoding='utf-8') as f:
            document_data = json.load(f)

        graph_data = create_document_graph(document_data)

        with open('document_graph.json', 'w', encoding='utf-8') as f:
            json.dump(graph_data, f, indent=2, ensure_ascii=False)

    except FileNotFoundError:
        print("입력 파일을 찾을 수 없습니다.")
    except Exception as e:
        print("처리 중 오류가 발생했습니다 - ", e)

    try:
        with open('document_graph.json', 'r', encoding='utf-8') as f:
            graph_json_data = json.load(f)

        display_document_tree(graph_json_data)

    except FileNotFoundError:
        print("입력 파일을 찾을 수 없습니다.")
    except Exception as e:
        print("처리 중 오류가 발생했습니다 - ", e)