from dotenv import load_dotenv

load_dotenv()

import uvicorn
from service.api.main import app

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, workers=1)