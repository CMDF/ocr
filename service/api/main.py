from fastapi import FastAPI, HTTPException
from fastapi import status
from pathlib import Path
import os
import hashlib
import asyncio
from service.api.services import extract_infos_from_pdf
from service.api.models import S3model
from service.core.s3 import download_file_from_presigned_url
from fastapi.responses import JSONResponse
from concurrent.futures import ThreadPoolExecutor

app = FastAPI(
    title="DOCent OCR API",
    description="OCR API server for DOCent",
    version="1.0",
)

@app.get("/")
def read_root():
    return {"status": "ok"}

processed_files = set()
processing_files = set()

gpu_lock = asyncio.Lock()
executor = ThreadPoolExecutor(max_workers=1)

@app.post("/pages", status_code=status.HTTP_201_CREATED)
async def read_pdf(bucket: S3model):
    filename = hashlib.sha256(bucket.file_url.encode()).hexdigest() + ".pdf"
    temp_path = Path(__file__).parent.parent.parent/"data"/"temp"/filename

    if filename in processed_files:
        return JSONResponse(
            status_code = status.HTTP_200_OK,
            content = {"message": "This file has already been processed.",
                       "s3_url": bucket.file_url}
        )

    if filename in processing_files:
        return JSONResponse(
            status_code = status.HTTP_200_OK,
            content = {"message": "This file has been processing.",
                       "s3_url": bucket.file_url}
        )

    processing_files.add(filename)

    temp_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        await asyncio.to_thread(download_file_from_presigned_url, bucket.file_url, temp_path)
        async with gpu_lock:
            output = await asyncio.get_running_loop().run_in_executor(
                None,
                extract_infos_from_pdf,
                str(temp_path)
            )

        processed_files.add(filename)

        return output

    except Exception as e:
        print(f"Error processing {filename}: {e}")
        raise HTTPException(status_code=500, detail="Processing failed")

    finally:
        if filename in processing_files:
            processing_files.remove(filename)

        if temp_path.exists():
            try:
                os.remove(temp_path)
            except OSError as e:
                print(f"Error deleting file {filename}: {e}")