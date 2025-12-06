from paddleocr import PaddleOCR
import cv2

ocr_m = PaddleOCR(use_doc_unwarping=False,
                  use_doc_orientation_classify=False,
                  use_textline_orientation=True,
                  text_detection_model_name="PP-OCRv5_server_det",
                  text_recognition_model_name="PP-OCRv5_server_rec")

def ocr(img):
    try:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    except Exception:
        raise

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img = clahe.apply(img)
    img = cv2.copyMakeBorder(img, 40, 40, 20, 20, cv2.BORDER_CONSTANT, value=255)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    output = ocr_m.predict(input=img)

    return output