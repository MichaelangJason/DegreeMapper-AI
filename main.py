from typing import Union

from fastapi import FastAPI
import uvicorn
from dotenv import load_dotenv
import os

# Load environment variables at the very beginning
load_dotenv(verbose=True)

app = FastAPI()

@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.get("/items/{item_id}")
def read_item(item_id: int, q: Union[str, None] = None):
    return {"item_id": item_id, "q": q}

# if __name__ == "__main__":
#     # Get environment variables with explicit defaults
#     port = int(os.environ.get("APP_PORT", 8000))
#     host = os.environ.get("APP_HOST", "0.0.0.0")
    
#     # Print for debugging
#     print(f"Starting server on {host}:{port}")
    
    # uvicorn.run(app, host=host, port=port)