import cv2
import matplotlib.pyplot as plt
import math

def crop_image_by_bbox(image_path: str, bbox: list):
    l_padding = 20
    r_padding = 29
    u_padding = 9
    d_padding = 8
    l_padding = 0
    r_padding = 0
    u_padding = 0
    d_padding = 0

    img = cv2.imread(str(image_path))
    if img is None:
        print("파일을 찾을 수 없거나 열 수 없습니다.")
        return None

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    height, width, _ = img_rgb.shape
    bbox = [
        bbox[0] * width,
        bbox[1] * height,
        bbox[2] * width,
        bbox[3] * height
    ]
    xmin, ymin, xmax, ymax = [math.ceil(box) for box in bbox]

    img_cropped = img_rgb[ymin-u_padding:ymax+d_padding, xmin-l_padding:xmax+r_padding]

    return img_cropped

def show(coordinate: list, path: str) -> None:
    image_file = path
    bounding_box = coordinate

    cropped_image_array = crop_image_by_bbox(image_file, bounding_box)

    if cropped_image_array is not None:
        plt.imshow(cropped_image_array)
        plt.axis('on')
        plt.show()