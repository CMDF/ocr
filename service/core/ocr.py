import math
from paddleocr import PaddleOCR
import cv2

ocr_m = PaddleOCR(use_doc_unwarping=False,
                  use_doc_orientation_classify=False,
                  use_textline_orientation=False,
                  text_detection_model_name="PP-OCRv5_server_det",
                  text_recognition_model_name="PP-OCRv5_server_rec")

def ocr(img):
    width, height = img.shape[1], img.shape[0]
    if width <= 4000 and height <= 4000:
        scale_factor = 2
        while width * scale_factor <= 4000 and height * scale_factor <= 4000:
            scale_factor *= 1.5
        scale_factor /= 1.5
    else:
        scale_factor = 0.8
        while width * scale_factor >= 4000 or height * scale_factor >= 4000:
            scale_factor *= 0.7
        scale_factor /= 0.7

    new_width = math.floor(width * scale_factor)
    new_height = math.floor(height * scale_factor)
    img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_CUBIC)

    try:
        output = ocr_m.predict(input=img)
    except Exception:
        print(f">>> [Warning] GPU out of memory occurred (resolved)")
        scale_factor /= 1.5
        new_width = math.floor(width * scale_factor)
        new_height = math.floor(height * scale_factor)
        img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
        output = ocr_m.predict(input=img)

    return output