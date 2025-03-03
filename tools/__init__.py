from langchain_core.tools import tool
from database.types import Course
from database.mongodb import get_mongodb_client
from database.enums import MongoCollection, ChromaCollection
from database.chromadb import get_chroma_client
from typing import List

@tool
async def get_course_info(course_id: str) -> Course:
    client = get_mongodb_client()
    
    result = await client.query(
        collection_name=MongoCollection.COURSES,
        query=course_id,
        n_results=1
    )
    
    if not result:
        raise ValueError(f"Course {course_id} not found")
    
    return Course(**result[0].metadata)

@tool
async def search_relevant_courses(query: str, n_results: int = 10) -> List[Course]:
    client = get_mongodb_client()
    
    result = await client.asimilarify_search(
        collection_name=MongoCollection.COURSES,
        query=query,
        n_results=n_results
    )

    if not result:
        raise ValueError(f"No relevant courses found for {query}")
    
    return [Course(**result[i].metadata) for i in range(n_results)]

@tool
async def get_knowledge_by_faculty(faculty: ChromaCollection, n_results: int = 10) -> List[str]:
    client = get_chroma_client()
    
    result = await client.asimilarity_search(
        query=faculty,
        n_results=n_results
    )

    return result
