from fastapi import FastAPI
from pydantic import BaseModel
from fastapi import status
from pathlib import Path
import os
import hashlib
import threading
from service.api.services import extract_infos_from_pdf
from service.core.s3 import download_file_from_presigned_url
from fastapi.responses import JSONResponse

app = FastAPI(
    title="Test API",
    description="Test API",
    version="1.0",
)

gpu_lock = threading.Lock()
@app.get("/")
def read_root():
    return {"status": "ok"}

class S3model(BaseModel):
    file_url: str
    timeout: int

processed_files = set()

@app.post("/pages", status_code=status.HTTP_201_CREATED)
def read_pdf(bucket: S3model):
    filename = hashlib.sha256(bucket.file_url.encode()).hexdigest() + ".pdf"

    if filename in processed_files:
        return JSONResponse(
            status_code = status.HTTP_200_OK,
            content = {"message": "This file has already been processed.",
                       "s3_url": bucket.file_url}
        )

    download_file_from_presigned_url(bucket.file_url, Path(__file__).parent.parent.parent/"data"/"temp"/filename)

    with gpu_lock:
        output = extract_infos_from_pdf(str(Path(__file__).parent.parent.parent/"data"/"temp"/filename))

    processed_files.add(filename)
    os.remove(Path(__file__).parent.parent.parent/"data"/"temp"/filename)

    return output