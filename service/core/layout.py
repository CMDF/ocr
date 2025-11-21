from paddleocr import LayoutDetection
from service.core.pre import *
from pathlib import Path
import json
import os
import fitz

def layout_detection(path):
    doc = fitz.open(path)
    page = doc.load_page(0)
    width_pnt = page.rect.width
    height_pnt = page.rect.height
    doc.close()

    width_px = width_pnt/72*144
    height_px = height_pnt/72*144

    model = LayoutDetection(model_name="PP-DocLayout-L")
    output = model.predict(input=path,
                           batch_size=100,
                           layout_nms=True)

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

        structured_document["pages"].append({
            "page_index": final_page_data["page_index"],
            "boxes": final_page_data["boxes"]
        })

    structured_document["pages"].sort(key=lambda p: p["page_index"])

    structured_document["total_pages"] = len(structured_document["pages"])
    filename = os.path.basename(path).split(".")[0] + ".json"

    try:
        with open(Path(__file__).parent.parent.parent/'data'/'temp'/filename, 'w', encoding='utf-8') as f:
            json.dump(structured_document, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(">>> [Error] Failed to save structured json.")
