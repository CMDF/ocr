from service.api.models import TempModel
from service.core.s3 import download_temp_file

def make_model():
    return TempModel(name="TempModel")

def download_pdf(bucket: str, object: str):
    try:
        with download_temp_file(bucket, object) as temp_file:
            return temp_file
    except Exception as e:
        print(e)
        raise ValueError({object})