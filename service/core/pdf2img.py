from pdf2image import convert_from_path
import os

def convert_pdf_to_png(pdf_path: str, output_folder: str = None):
    if not os.path.exists(pdf_path):
        print(" 파일을 찾을 수 없습니다.")
        return

    if output_folder is None:
        output_folder = os.path.splitext(os.path.basename(pdf_path))[0]

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    try:
        images = convert_from_path(pdf_path, dpi=144, use_cropbox=True)
    except Exception as e:
        print("오류가 발생했습니다 - ", e)
        return

    for i, image in enumerate(images):
        file_name = f"page_{i + 1}.png"
        save_path = os.path.join(output_folder, file_name)

        image.save(save_path, "PNG")

if __name__ == "__main__":
    my_pdf_file = "/home/gyupil/Downloads/Test.pdf"
    convert_pdf_to_png(my_pdf_file)