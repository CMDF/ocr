from paddleocr import LayoutDetection
from service.core.pre import *
from pathlib import Path
import json
import os
import fitz
from config import debug
from PIL import Image, ImageDraw

model = LayoutDetection(model_name="PP-DocLayout-L")

def layout_detection(path):
    doc = fitz.open(path)
    page = doc.load_page(0)
    width_pnt = page.rect.width
    height_pnt = page.rect.height
    doc.close()

    width_px = width_pnt/72*144
    height_px = height_pnt/72*144

    output = model.predict(input=path,
                           batch_size=100,
                           layout_nms=True)

    if debug:
        for res in output:
            res.save_to_img(Path(__file__).parent.parent.parent/'data'/'debug')

    structured_document = {
        "document_path": None,
        "total_pages": 0,
        "pages": []
    }

    for res in output:
        data = res.json['res']
        if structured_document["document_path"] is None:
            structured_document["document_path"] = data.get("input_path", "Unknown")

        for box in data["boxes"]:
            coord = box['coordinate']
            coord = [
                coord[0]/width_px,
                coord[1]/height_px,
                coord[2]/width_px,
                coord[3]/height_px
            ]
            box["coordinate"] = coord
        processed_data_1 = remove_nested_boxes(data)
        folder_name = os.path.basename(path).split(".")[0]
        final_page_data = group_image_with_caption(processed_data_1, folder_name)

        structured_document['pages'].append({
            "page_index": final_page_data["page_index"],
            "boxes": final_page_data["boxes"]
        })

    structured_document['pages'].sort(key=lambda p: p["page_index"])

    structured_document["total_pages"] = len(structured_document['pages'])
    filename = os.path.basename(path).split(".")[0] + ".json"

    try:
        with open(Path(__file__).parent.parent.parent/'data'/'temp'/filename, 'w', encoding='utf-8') as f:
            json.dump(structured_document, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f">>> [Error] Failed to save structured json.({e})")


def det_debug(output: dict):
    def draw_bounding_box(image_path: str, rel_coord: list):
        try:
            with Image.open(image_path).convert("RGB") as img:
                width, height = img.size
                draw = ImageDraw.Draw(img)
                bbox_coords = [(math.ceil(rel_coord[0] * width), math.ceil(rel_coord[1] * height)),
                               (math.ceil(rel_coord[2] * width), math.ceil(rel_coord[3] * height))]
                draw.rectangle(bbox_coords, outline="black", width=3)
                img.save(image_path)
        except FileNotFoundError:
            print(f">>> Error: Image not found at {image_path}")
        except Exception as e:
            print(f">>> An error occurred: {e}")

    figures = output['figures']
    for figure in figures:
        filename = "yoochan-exprace_"+str(figure['page_num']) + "_res.png"
        path = str(Path(__file__).parent.parent.parent/'data'/'debug'/filename)
        coord = figure['figure_box']
        draw_bounding_box(path, coord)