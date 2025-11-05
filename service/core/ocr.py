from paddleocr import PaddleOCR
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

    new_width = int(img.shape[1] * scale_factor)
    new_height = int(img.shape[0] * scale_factor)
    img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_CUBIC)

    output = ocr_m.predict(input=img)

    return output