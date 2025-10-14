from paddleocr import LayoutDetection
from pre import *
import json

def layout_detection(path):
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

        processed_data_1 = group_image_with_caption(data)
        final_page_data = remove_nested_boxes(processed_data_1)

        structured_document["pages"].append({
            "page_index": final_page_data["page_index"],
            "boxes": final_page_data["boxes"]
        })

    structured_document["pages"].sort(key=lambda p: p["page_index"])

    structured_document["total_pages"] = len(structured_document["pages"])

    try:
        with open('document_structure.json', 'w', encoding='utf-8') as f:
            json.dump(structured_document, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print("최종 파일 저장 중 에러 발생 - ", e)
