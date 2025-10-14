from graph import *
from layout import *
from ocr import *
from crop import *
import json

if __name__ == "__main__":
    layout_detection('/home/gyupil/Downloads/Test.pdf')

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

    try:
        with open('document_structure.json', 'r', encoding='utf-8') as f:
            data = json.load(f)

        pages = data['pages']
        for page in pages:
            boxes = page['boxes']
            texts = [t for t in boxes if t['label'] == 'text']

            for text in texts:
                coord = text['coordinate']
                path = "/home/gyupil/mapping_test/Test/page_" + str(page['page_index'] + 1) + ".png"
                ocr(crop_image_by_bbox(path, coord))

    except FileNotFoundError:
        print("입력 파일을 찾을 수 없습니다.")
    except Exception as e:
        print("처리 중 오류가 발생했습니다 - ", e)