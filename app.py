from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from llm.huggingface import get_huggingface_embedding, get_huggingface_llm
from dotenv import load_dotenv
from database.chromadb import get_chroma_client
from database.mongodb import get_mongodb_client
import os

load_dotenv()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # initialization
    # client = get_chroma_client()
    # hugginface model init
    get_huggingface_embedding()
    if os.getenv("USE_LOCAL_LLM") == "true":
        get_huggingface_llm()
    yield
    # shutdown
    get_huggingface_embedding.cache_clear()
    if os.getenv("USE_LOCAL_LLM") == "true":
        get_huggingface_llm.cache_clear()

    await get_chroma_client().close()
    await get_mongodb_client().close()
    get_chroma_client.cache_clear()
    get_mongodb_client.cache_clear()

app = FastAPI(
    title="DegreeMapper AI API",
    description="API for degree planning and scheduling",
    version="0.1.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"message": "Hello from DegreeMapper AI!"}


