from service.core.layout import layout_detection, det_debug
from service.core.ocr import ocr
from service.core.crop import crop_image_by_bbox, show
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
import time
import fitz
import concurrent.futures
from functools import partial
from config import debug

nlp = spacy.load("en_core_web_sm")

def _convert_page_worker(page_num, pdf_path, output_folder, dpi, use_cropbox):
    try:
        image = convert_from_path(
            pdf_path,
            dpi=dpi,
            first_page=page_num,
            last_page=page_num,
            use_cropbox=use_cropbox,
            grayscale=True
        )[0]
        file_name = f"page_{page_num}.png"
        save_path = os.path.join(output_folder, file_name)
        image.save(save_path, "PNG")
        return save_path
    except Exception:
        print(f">>> [Error] Page {page_num} failed to convert to PNG.")
        return None

def _convert_pdf_to_png(pdf_path: str, output_folder: str = None):
    if not os.path.exists(pdf_path):
        print(">>> [Error] File doesn't exist.")
        return

    folder_name = os.path.basename(pdf_path).split(".")[0]
    if output_folder is None:
        output_folder = Path(__file__).parent.parent.parent/'data'/'temp'/folder_name

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    try:
        with fitz.open(pdf_path) as doc:
            page_count = doc.page_count
    except Exception:
        print(f">>> [Error] Fail to convert PDF to PNG.")
        return

    dpi = 300
    use_cropbox = True

    worker = partial(_convert_page_worker, pdf_path=pdf_path, output_folder=output_folder, dpi=dpi, use_cropbox=use_cropbox)

    with concurrent.futures.ProcessPoolExecutor() as executor:
        list(executor.map(worker, range(1, page_count+1)))

def extract_infos_from_pdf(pdf_path: str):
    folder_name = os.path.basename(pdf_path).split(".")[0]

    _convert_pdf_to_png(pdf_path)
    layout_detection(pdf_path)

    try:
        filename = os.path.basename(pdf_path).split(".")[0] + ".json"
        with open(Path(__file__).parent.parent.parent/'data'/'temp'/filename, 'r', encoding='utf-8') as f:
            data = json.load(f)

        pages = data['pages']

        text_result = []
        figure_result = []
        for page in pages:
            page_text = ""
            boxes = page['boxes']
            texts = [t for t in boxes if t['label'] == 'text']
            figures = [f for f in boxes if f['label'] in ['image', 'table', 'figure', 'chart', 'algorithm']]

            for text in texts:
                coord = text['coordinate']
                filename = "page_" + str(page['page_index'] + 1) + ".png"
                path = Path(__file__).parent.parent.parent/'data'/'temp'/folder_name/filename
                output = ocr(crop_image_by_bbox(str(path), coord))
                try:
                    lines = correct(output[0])
                except Exception:
                    lines = [""]
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
                figure_result.append({'page_num': page['page_index'],
                                      'figure_box': figure['coordinate'],
                                      'figure_type': figure['label']})

            page_data = {'page_num': page['page_index']+1, 'text': page_text}
            text_result.append(page_data)

        graph = build_document_graph(load_and_transform_data(data))
        pairs = create_reference_pairs(graph)

        pair_result = []
        for pair in pairs:
            pair_result.append({
                'figure_box': pair['ref']['bbox'],
                'figure_page': pair['ref']['page'],
                'page_num': pair['page'],
                'raw_text': pair['raw_text'],
                'figure_text': pair['figure_text']
            })

        filename = os.path.basename(pdf_path).split(".")[0] + ".json"
        folder_name = os.path.basename(pdf_path).split(".")[0]
        if not debug:
            os.remove(Path(__file__).parent.parent.parent/'data'/'temp'/filename)
            for file_name in os.listdir(Path(__file__).parent.parent.parent/'data'/'temp'/folder_name):
                file_path = os.path.join(Path(__file__).parent.parent.parent/'data'/'temp'/folder_name, file_name)
                os.remove(file_path)
            for file_name in os.listdir(Path(__file__).parent.parent.parent/'data'/'debug'):
                file_path = os.path.join(Path(__file__).parent.parent.parent/'data'/'debug', file_name)
                os.remove(file_path)

            os.removedirs(Path(__file__).parent.parent.parent/'data'/'temp'/folder_name)

        final_result = {'pages': text_result, 'figures': figure_result, 'matches': pair_result}
        result_json = json.dumps(final_result, ensure_ascii=False, indent=4)

        for text in final_result['pages']:
            print("page ", text['page_num'])
            print(text['text'])

        if debug:
            det_debug(final_result, folder_name)

        return result_json

    except FileNotFoundError:
        print(">>> [Error] Structured json file not found.")

if __name__ == "__main__":
    start = time.time()
    # output = extract_infos_from_pdf("/home/gyupil/Downloads/Introduction to Algorithms (Thomas H. Cormen, Charles E. Leiserson etc.) (Z-Library).pdf")
    output = extract_infos_from_pdf("/home/gyupil/Downloads/yoochan-exprace.pdf")
    # output = extract_infos_from_pdf("/home/gyupil/Downloads/Test.pdf")
    interval = time.time() - start
    print(f">>> Task completed in {int(interval/60)} minutes {int(interval%60)} seconds.")