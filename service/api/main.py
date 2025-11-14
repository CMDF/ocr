from fastapi import FastAPI
from pydantic import BaseModel
from fastapi import status
from pathlib import Path
import os
import pprint
from service.api.services import extract_infos_from_pdf
from service.core.s3 import download_file_from_presigned_url

app = FastAPI(
    title="Test API",
    description="Test API",
    version="1.0",
)
@app.get("/")
def read_root():
    return {"status": "ok"}

class S3model(BaseModel):
    file_url: str
    timeout: int

@app.post("/pages", status_code=status.HTTP_201_CREATED)
def read_pdf(bucket: S3model):
    download_file_from_presigned_url(bucket.file_url, Path(__file__).parent.parent.parent/"data"/"temp"/"Test.pdf")
    print(bucket.file_url)
    output = extract_infos_from_pdf(str(Path(__file__).parent.parent.parent/"data"/"temp"/"Test.pdf"))
    pprint.pprint(output)
    os.remove(Path(__file__).parent.parent.parent/"data"/"temp"/"Test.pdf")

    return output