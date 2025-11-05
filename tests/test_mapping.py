from service.core.graph import *
from service.core.layout import *
from service.core.ocr import *
from service.core.crop import *
import json
from pathlib import Path
from service.models.predict import predict_from_text
import joblib
import os
from pdf2image import convert_from_path
from PIL import Image, ImageDraw

def convert_pdf_to_png(pdf_path: str, output_folder: str = None):
    if not os.path.exists(pdf_path):
        print(" 파일을 찾을 수 없습니다.")
        return

    if output_folder is None:
        output_folder = os.path.splitext(os.path.basename(pdf_path))[0]

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    try:
        images = convert_from_path(pdf_path, dpi=300, use_cropbox=True)
    except Exception as e:
        print("오류가 발생했습니다 - ", e)
        return

    for i, image in enumerate(images):
        file_name = f"page_{i + 1}.png"
        save_path = os.path.join(output_folder, file_name)

        image.save(save_path, "PNG")

def draw_box_on_image(image_path: str, relative_coords: list, color: str = "red", width: int = 2):
    try:
        with Image.open(image_path) as img:
            draw = ImageDraw.Draw(img)
            img_width, img_height = img.size

            # Convert relative coordinates to absolute coordinates
            abs_x_min = relative_coords[0] * img_width
            abs_y_min = relative_coords[1] * img_height
            abs_x_max = relative_coords[2] * img_width
            abs_y_max = relative_coords[3] * img_height

            # Draw the rectangle
            draw.rectangle([(abs_x_min, abs_y_min), (abs_x_max, abs_y_max)], outline=color, width=width)

            # Save the new image
            img.save(image_path)
            print(f"Image saved to {image_path}")

    except FileNotFoundError:
        print(f"Error: Image file not found at {image_path}")
    except Exception as e:
        print(f"An error occurred: {e}")


my_pdf_file = "/home/gyupil/Downloads/Test.pdf"
convert_pdf_to_png(my_pdf_file)

crf = joblib.load(Path(__file__).parent.parent/"service"/"models"/"artifacts"/"figure_model.joblib")

if __name__ == "__main__":
    graph = load_graph_from_json()

    if graph is None:
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
                    path = "/home/gyupil/ocr/tests/Test/page_" + str(page['page_index'] + 1) + ".png"
                    paragraph = ocr(crop_image_by_bbox(path, coord))
                    print(paragraph)
                    draw_box_on_image(image_path=path, relative_coords=coord)
                    # spans, _, _ = predict_from_text(paragraph, crf)
                    # if len(spans) > 0:
                    #     text['text'] = spans[0]

            with open('document_structure.json', 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=4, ensure_ascii=False)

        except FileNotFoundError:
            print("입력 파일을 찾을 수 없습니다.")
        except Exception as e:
            print("처리 중 오류가 발생했습니다 - ", e)

    #     try:
    #         with open('document_structure.json', 'r', encoding='utf-8') as f:
    #             data = json.load(f)
    #
    #         graph = build_document_graph(load_and_transform_data(data))
    #         save_graph_to_json(graph, 'document_graph.json')
    #
    #     except FileNotFoundError:
    #         print("파일을 찾을 수 없습니다.")
    #     except Exception as e:
    #         print("오류가 발생했습니다 - ", e)
    #
    # pairs = create_reference_pairs(graph)
    # ref_nodes = find_nodes(graph, type='text', text=lambda t:t)
    # for node in ref_nodes:
    #     print("ID: {} | {} in page {}".format(node['id'], node['text'], node['page']))
    #
    # while True:
    #     ID = input("보고 싶은 figure의 아이디를 입력하세요.\n종료는 0을 입력하세요.\n>>> ")
    #     if ID == '0':
    #         break
    #     ref = pairs[ID]
    #     if ref:
    #         show(ref['bbox'], 'Test/page_' + str(ref['page'] + 1) + '.png')