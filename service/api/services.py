from service.core.layout import layout_detection
from service.core.ocr import ocr
from service.core.crop import crop_image_by_bbox
from service.core.post import correct
from service.core.graph import build_document_graph
from service.core.graph import load_and_transform_data
from service.core.graph import create_reference_pairs
from service.models.predict import predict_from_text
from pdf2image import convert_from_path
from pathlib import Path
import os
import json
import spacy

nlp = spacy.load("en_core_web_sm")

def _convert_pdf_to_png(pdf_path: str, output_folder: str = None):
    if not os.path.exists(pdf_path):
        print(" 파일을 찾을 수 없습니다.")
        return

    if output_folder is None:
        output_folder = Path(__file__).parent.parent.parent/'data'/'temp'/'Test'

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

def extract_infos_from_pdf(pdf_path: str):
    _convert_pdf_to_png(pdf_path)
    layout_detection(pdf_path)

    try:
        with open(Path(__file__).parent.parent.parent/'data'/'temp'/'document_structure.json', 'r', encoding='utf-8') as f:
            data = json.load(f)

        pages = data['pages']

        text_result = []
        figure_result = []
        for page in pages:
            page_text = ""
            boxes = page['boxes']
            texts = [t for t in boxes if t['label'] == 'text']
            figures = [f for f in boxes if f['label'] == 'figure']

            for text in texts:
                coord = text['coordinate']
                filename = "page_" + str(page['page_index'] + 1) + ".png"
                path = Path(__file__).parent.parent.parent/'data'/'temp'/'Test'/filename
                output = ocr(crop_image_by_bbox(str(path), coord))
                lines = correct(output[0])
                paragraph = ' '.join(lines)
                if paragraph != "":
                    doc = nlp(paragraph)
                    sentences = list(doc.sents)
                    for i, sentence in enumerate(sentences):
                        output, _, _ = predict_from_text(sentence.text.strip())
                        if output.ref_info:
                            text['ref_info'] = {'figure_text':  output.ref_info,
                                                'raw_text':     output.raw_texts,
                                                'section_info': output.section_info}
                page_text += paragraph

            for figure in figures:
                coord = figure['coordinate']
                figure_data = {'page_num': page['page_index']+1,'figure_box': coord, 'figure_type': 'figure'}
                figure_result.append(figure_data)

            page_data = {'page_num': page['page_index']+1, 'text': page_text}
            text_result.append(page_data)

        final_result = {'pages': text_result, 'figures': figure_result}
        result_json = json.dumps(final_result, ensure_ascii=False, indent=4)

        for file_name in os.listdir(Path(__file__).parent.parent.parent/'data'/'temp'/'Test'):
            file_path = os.path.join(Path(__file__).parent.parent.parent/'data'/'temp'/'Test', file_name)
            os.remove(file_path)

        os.removedirs(Path(__file__).parent.parent.parent/'data'/'temp'/'Test')

        graph = build_document_graph(load_and_transform_data(data))
        pairs = create_reference_pairs(graph)

        pair_result = []
        # TODO: save pairs(by DB format)

        os.remove(Path(__file__).parent.parent.parent/'data'/'temp'/'document_structure.json')

        return result_json

    except FileNotFoundError:
        print("Invalid path")