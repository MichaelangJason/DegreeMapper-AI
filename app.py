from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from dotenv import load_dotenv
from database.chromadb import get_chroma_client, ChromaDBClient
from database.mongodb import get_mongodb_client, MongoDBClient
from agents import router as agents_router
from database import router as database_router
from llm.huggingface import get_huggingface_embedding, get_huggingface_llm
from agents.graph import get_compiled_graph
from agents.enums import Model
import logging
import os

@asynccontextmanager
async def lifespan(app: FastAPI):
    # initialization
    # hugginface model init
    load_dotenv()
    get_huggingface_embedding()
    if os.getenv("USE_LOCAL_LLM") == "true":
        get_huggingface_llm()
    # initialize database clients
    await get_chroma_client().ensure_connection()
    await get_mongodb_client().ensure_connection()
    await get_compiled_graph(Model.OPENAI)
    yield
    # shutdown
    embedding = get_huggingface_embedding()
    if hasattr(embedding, "cleanup"):
        logging.info(embedding.cleanup())
    get_huggingface_embedding.cache_clear()

    if os.getenv("USE_LOCAL_LLM") == "true":
        llm = get_huggingface_llm()
        if hasattr(llm, "cleanup"):
            logging.info(llm.cleanup())
        get_huggingface_llm.cache_clear()

    await get_chroma_client().close()
    await get_mongodb_client().close()

    ChromaDBClient.reset_instance()
    MongoDBClient.reset_instance()
    
    # manually trigger garbage collection
    import gc
    gc.collect()

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

app.include_router(agents_router)
app.include_router(database_router)

@app.get("/")
async def root():
    return {"message": "Hello from DegreeMapper AI!"}


