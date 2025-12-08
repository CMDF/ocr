from paddleocr import LayoutDetection
from service.core.pre import *
from pathlib import Path
import json, os, fitz
from config import debug
from PIL import Image, ImageDraw
import re

model = LayoutDetection(model_name="PP-DocLayoutV2")

class HeaderParser:
    def __init__(self):
        self.patterns = {
            'part': re.compile(r'^(Part|PART)\s*([IVX0-9]+|[A-Z])\s*(.*)', re.IGNORECASE),
            'chapter': re.compile(r'^(Chapter|CHAPTER)\s*([0-9]+)\s*(.*)', re.IGNORECASE),
            'section_explicit': re.compile(r'^(Section|§)\s*([0-9]+)\s*(.*)', re.IGNORECASE),
            'section_numeric': re.compile(r'^([0-9]+\.[0-9]+)\s+(.*)'),
            'special': re.compile(r'^(Preface|Contents|Index|Bibliography|Appendix|Problems|Notes|Exercises)',
                                  re.IGNORECASE)
        }

        self.state = {
            'part': None,
            'chapter': None,
            'section_num': None,
            'section_title': None
        }

    def _get_priority(self, text):
        if self.patterns['part'].match(text): return 1
        if self.patterns['chapter'].match(text): return 2
        return 3

    def feed_page(self, header_list):
        if not header_list:
            return self._format_output()

        sorted_headers = sorted(header_list, key=lambda x: self._get_priority(x))

        for text in sorted_headers:
            clean_text = text.strip()

            if self.patterns['part'].match(clean_text):
                self.state['part'] = clean_text
                continue

            if self.patterns['chapter'].match(clean_text):
                self.state['chapter'] = clean_text
                self.state['section_num'] = None
                self.state['section_title'] = None
                continue

            sec_num, sec_title = None, None

            m_sec_exp = self.patterns['section_explicit'].match(clean_text)
            m_sec_num = self.patterns['section_numeric'].match(clean_text)

            if m_sec_exp:
                sec_num = m_sec_exp.group(2)
                sec_title = m_sec_exp.group(3).strip()
            elif m_sec_num:
                sec_num = m_sec_num.group(1)
                sec_title = m_sec_num.group(2).strip()

            if sec_num:
                if (self.state['section_num'] != sec_num) or \
                        (sec_title and (
                                not self.state['section_title'] or len(sec_title) > len(self.state['section_title']))):
                    self.state['section_num'] = sec_num
                    self.state['section_title'] = sec_title
                continue

            if self.patterns['special'].match(clean_text):
                self.state['section_title'] = clean_text

        return self._format_output()

    def _format_output(self):
        full_title = self.state['section_num'] if self.state['section_num'] else ""

        return full_title

parser = HeaderParser()

def layout_detection(path):
    doc = fitz.open(path)

    output = model.predict(input=path,
                           layout_nms=True)

    if debug:
        for res in output:
            res.save_to_img(Path(__file__).parent.parent.parent/'data'/'debug')

    structured_document = {
        "document_path": None,
        "total_pages": 0,
        "pages": []
    }

    for i, res in enumerate(output):
        data = res.json['res']
        page = doc.load_page(data['page_index'])
        width_pnt = page.rect.width
        height_pnt = page.rect.height
        width_px = width_pnt / 72 * 144
        height_px = height_pnt / 72 * 144
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

        section_nos = []
        if i > 0:
            previous_page_data = output[i-1].json['res']
            for j, box in enumerate(previous_page_data["boxes"]):
                if box['label'] in ['header', 'paragraph_title'] and box['coordinate'][1]/height_px < 0.17:
                    section_coord = [
                        box['coordinate'][0] / width_px,
                        box['coordinate'][1] / height_px,
                        box['coordinate'][2] / width_px,
                        box['coordinate'][3] / height_px
                    ]
                    filename = "page_" + str(previous_page_data['page_index']+1) + ".png"
                    try:
                        section = ocr(crop_image_by_bbox(str(Path(__file__).parent.parent.parent/"data"/"temp"/folder_name/filename), section_coord))
                        section_no = ""
                        for sec_res in section:
                            section_no = section_no + sec_res['rec_texts'][0]
                        section_nos.append(section_no)
                    except Exception:
                        pass

        data = res.json['res']
        for j, box in enumerate(data["boxes"]):
            if box['label'] in ['header', 'paragraph_title'] and box['coordinate'][1]/height_px < 0.17:
                section_coord = [
                    box['coordinate'][0] / width_px,
                    box['coordinate'][1] / height_px,
                    box['coordinate'][2] / width_px,
                    box['coordinate'][3] / height_px
                ]
                filename = "page_" + str(data['page_index']+1) + ".png"
                try:
                    section_no = ""
                    section = ocr(crop_image_by_bbox(str(Path(__file__).parent.parent.parent/"data"/"temp"/folder_name/filename), section_coord))
                    for sec_res in section:
                        section_no = section_no + sec_res['rec_texts'][0]
                    section_nos.append(section_no)
                except Exception:
                    pass

        page_section = parser.feed_page(section_nos)
        if page_section != "":
            for box in final_page_data["boxes"]:
                box['section_info'] = page_section

        structured_document['pages'].append({
            "page_index": final_page_data["page_index"],
            "boxes": final_page_data["boxes"]
        })

    doc.close()
    structured_document['pages'].sort(key=lambda p: p["page_index"])

    structured_document["total_pages"] = len(structured_document['pages'])
    filename = os.path.basename(path).split(".")[0] + ".json"

    try:
        with open(Path(__file__).parent.parent.parent/'data'/'temp'/filename, 'w', encoding='utf-8') as f:
            json.dump(structured_document, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f">>> [Error] Failed to save structured json.({e})")

import img2pdf
def det_debug(output: dict, folder_name: str, do: bool = debug):
    if not do:
        return
    def draw_bounding_box(image_path: str, rel_coord: list):
        try:
            with Image.open(image_path).convert("RGB") as img:
                width, height = img.size
                draw = ImageDraw.Draw(img)
                bbox_coords = [(math.ceil(rel_coord[0] * width), math.ceil(rel_coord[1] * height)),
                               (math.ceil(rel_coord[2] * width), math.ceil(rel_coord[3] * height))]
                draw.rectangle(bbox_coords, outline="red", width=4)
                img.save(image_path)
        except FileNotFoundError:
            print(f">>> Error: Image not found at {image_path}")
        except Exception as e:
            print(f">>> An error occurred: {e}")

    images_path = str(Path(__file__).parent.parent.parent/'data'/'temp'/folder_name)
    imgs = [
        os.path.join(images_path, f)
        for f in os.listdir(images_path)
        if f.lower().endswith(".png")
    ]

    for pair in output['matches']:
        filename = "page_" + str(pair['page_num'] + 1) + ".png"
        path = str(Path(__file__).parent.parent.parent/'data'/'temp'/folder_name/filename)
        coord = pair['text_box']
        draw_bounding_box(path, coord)
        filename = "page_" + str(pair['figure_page'] + 1) + ".png"
        path = str(Path(__file__).parent.parent.parent/'data'/'temp'/folder_name/filename)
        coord = pair['figure_box']
        draw_bounding_box(path, coord)

    imgs.sort(key=lambda p: int(((p.split('/')[-1]).split(".")[0]).split("_")[-1]))

    with open(Path(__file__).parent.parent.parent/'data'/'temp'/folder_name/'output.pdf', "wb") as f:
        f.write(img2pdf.convert(imgs))