from graph import *
from layout import *
from ocr import *
from crop import *
import json

if __name__ == "__main__":
    layout_detection('/home/gyupil/Downloads/Test.pdf')
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
                paragraph = ocr(crop_image_by_bbox(path, coord))
                found, figure = find_figure_reference(paragraph)
                if found:
                    text['text'] = figure

        with open('document_structure.json', 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4, ensure_ascii=False)

    except FileNotFoundError:
        print("입력 파일을 찾을 수 없습니다.")
    except Exception as e:
        print("처리 중 오류가 발생했습니다 - ", e)

    try:
        with open('document_structure.json', 'r', encoding='utf-8') as f:
            data = json.load(f)

        graph = build_document_graph(load_and_transform_data(data))
        pairs = create_reference_pairs(graph)
        print(type(pairs), type(pairs[0]))
        save_graph_to_json(graph, 'document_graph.json')

    except FileNotFoundError:(
        print("파일을 찾을 수 없습니다."))
    except Exception as e:(
        print("오류가 발생했습니다 - ", e))

    ref_nodes = find_nodes(graph, type='text', text=lambda t:t)
    for node in ref_nodes:
        print("ID: {} | {} in page {}".format(node['id'], node['text'], node['page']))

    while True:
        ID = input("보고 싶은 figure의 아이디를 입력하세요.\n종료는 0을 입력하세요.\n>>> ")
        if ID == '0':
            break
        ref = pairs[ID]
        if ref:
            show(ref['bbox'], 'Test/page_' + str(ref['page'] + 1) + '.png')