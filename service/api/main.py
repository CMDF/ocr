from fastapi import FastAPI

app = FastAPI(
    title="Test API",
    description="Test API",
    version="1.0",
)
@app.get("/")
def read_root():
    return {"status": "ok"}