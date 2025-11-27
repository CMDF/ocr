from pydantic import BaseModel

class S3model(BaseModel):
    file_url: str
    timeout: int