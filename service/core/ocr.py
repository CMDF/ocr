import math
from paddleocr import PaddleOCR
import matplotlib.pyplot as plt
import cv2

ocr_m = PaddleOCR(use_doc_unwarping=False,
                  use_doc_orientation_classify=False,
                  use_textline_orientation=False,
                  text_detection_model_name="PP-OCRv5_server_det",
                  text_recognition_model_name="PP-OCRv5_server_rec")

def ocr(img):
    scale_factor = 2
    width, height = img.shape[1], img.shape[0]

    while width*scale_factor <= 4000 and height*scale_factor <= 4000:
        scale_factor *= 1.5
    scale_factor /= 1.5

    new_width = math.floor(img.shape[1] * scale_factor)
    new_height = math.floor(img.shape[0] * scale_factor)
    img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_CUBIC)

    output = ocr_m.predict(input=img)

    if not output[0]['rec_texts']:
        plt.imshow(img)
        plt.show()

    return output