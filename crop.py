import cv2
import matplotlib.pyplot as plt

def crop_image_by_bbox(image_path: str, bbox: list):
    img = cv2.imread(image_path)
    if img is None:
        print("파일을 찾을 수 없거나 열 수 없습니다.")
        return

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    xmin, ymin, xmax, ymax = bbox

    img_cropped = img_rgb[ymin:ymax, xmin:xmax]

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    axes[0].imshow(img_rgb)
    rect_points = [
        (xmin, ymin),  # 왼쪽 위
        (xmax, ymin),  # 오른쪽 위
        (xmax, ymax),  # 오른쪽 아래
        (xmin, ymax),  # 왼쪽 아래
        (xmin, ymin)  # 닫기
    ]
    poly = plt.Polygon(rect_points, edgecolor='r', facecolor='none', linewidth=2)
    axes[0].add_patch(poly)
    axes[0].set_title('Original Image with Bounding Box')
    axes[0].axis('off')

    axes[1].imshow(img_cropped)
    axes[1].set_title('Cropped Image')
    axes[1].axis('off')

    plt.tight_layout()
    plt.show()

image_file = '/Users/neung_gae/Desktop/output/Test_27_res.png'

coordinate = list(map(int, [216.796997070313, 140.049926757813, 1000.78656005859, 750.095397949219]))

try:
    crop_image_by_bbox(image_file, coordinate)
except FileNotFoundError:
    print("파일을 찾을 수 없습니다.")
except Exception as e:
    print("오류가 발생했습니다 - ", e)