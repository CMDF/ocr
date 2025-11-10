from fastapi import FastAPI
from pydantic import BaseModel
from service.api.services import extract_texts_from_pdf

app = FastAPI(
    title="Test API",
    description="Test API",
    version="1.0",
)
@app.get("/")
def read_root():
    return {"status": "ok"}

@app.get("/pages")
def read_pdf():
    return extract_texts_from_pdf("/home/gyupil/Downloads/Test.pdf")