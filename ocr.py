from paddleocr import TextRecognition
from paddleocr import PaddleOCR
from crop import *
import json

model = TextRecognition(model_name="PP-OCRv5_server_rec")
ocr_m = PaddleOCR()

def ocr(img):
    scale_factor = 2
    width, height = img.shape[1], img.shape[0]

    while width*scale_factor <= 4000 and height*scale_factor <= 4000:
        scale_factor *= 1.5
    scale_factor /= 1.5

    new_width = int(img.shape[1] * scale_factor)
    new_height = int(img.shape[0] * scale_factor)
    img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_CUBIC)

    output = ocr_m.predict(input=img)
    paragraph = ""
    for res in output:
        rec_texts = res['rec_texts']
        for rec_text in rec_texts:
            paragraph = paragraph + " " + rec_text
            if 'Fig' in rec_text:
                plt.imshow(img)
                plt.axis('on')
                plt.show()

    paragraph = paragraph[1:]
    
    return paragraph

with open('document_structure.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

pages = data['pages']
for page in pages:
    boxes = page['boxes']
    texts = [t for t in boxes if t['label'] == 'text']

    for text in texts:
        coord = text['coordinate']
        path = "/home/gyupil/mapping_test/Test/page_"+str(page['page_index']+1)+".png"
        ocr(crop_image_by_bbox(path, coord))
