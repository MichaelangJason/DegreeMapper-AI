from fastapi import APIRouter
from .chromadb import get_chroma_client
from .mongodb import get_mongodb_client
from database.enums import MongoCollection, ChromaCollection
router = APIRouter()

@router.get("/ping/mongodb")
async def ping_mongodb():
    client = get_mongodb_client()
    return await client.ensure_connection()

@router.get("/similarity/mongodb/courses")
async def similarity_mongodb_courses(query: str, n_results: int = 10):
    client = get_mongodb_client()
    return await client.asimilarify_search(MongoCollection.Course, query, n_results)

@router.get("/query/mongodb")
async def query_mongodb(query: str, n_results: int = 10):
    client = get_mongodb_client()
    return await client.query(MongoCollection.Course, query, n_results)

@router.get("/similarity/mongodb/programs")
async def similarity_mongodb_programs(query: str, n_results: int = 10):
    client = get_mongodb_client()
    return await client.asimilarify_search(MongoCollection.Program, query, n_results)

@router.get("/ping/chroma")
async def ping_chroma():
    client = get_chroma_client()
    return await client.ensure_connection()

@router.get("/similarity/chroma")
async def similarity_chroma(query: str, n_results: int = 10):
    client = get_chroma_client()
    return await client.asimilarity_search(ChromaCollection.Faculty ,query, n_results)
