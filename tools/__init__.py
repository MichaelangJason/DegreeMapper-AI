from langchain_core.tools import tool
from database.types import Course
from database.mongodb import get_mongodb_client
from database.enums import MongoCollection, ChromaCollection
from database.chromadb import get_chroma_client
from typing import List, Annotated, Optional

# @tool(description="Get the course information by course id, name, faculty, department, all optional")
@tool(description="Get the course information by course id")
async def get_course_info(
    course_id: Annotated[Optional[str], "The ID of the course, should be a string"] = None,
    # name: Annotated[Optional[str], "The name of the course, should be a string"] = None,
    # faculty: Annotated[Optional[str], "The faculty of the course, should be a string"] = None,
    # department: Annotated[Optional[str], "The department of the course, should be a string"] = None,
    n_results: Annotated[int, "The number of results to return, should be an integer"] = 3
) -> Course:
    client = get_mongodb_client()
    
    result = await client.query(
        collection_name=MongoCollection.Course,
        query=course_id,
        n_results=n_results
    )
    
    if not result:
        raise ValueError(f"Course {course_id} not found")
    
    return result

@tool(description="Search for courses by semantic similarity on the course overview, not the course id, name, faculty, department, etc.")
async def semantic_course_search(
    query: Annotated[str, "The query to search for, should be a string"], 
    n_results: Annotated[int, "The number of results to return, should be an integer"] = 10
) -> List[Course]:
    client = get_mongodb_client()
    
    result = await client.asimilarify_search(
        collection_name=MongoCollection.Course,
        query=query,
        n_results=n_results
    )

    if not result:
        raise ValueError(f"No relevant courses found for {query}")
    
    return result

@tool(description="get_knowledge_by_faculty")
async def get_knowledge_by_faculty(
    # faculty: Annotated[ChromaCollection, "The faculty to search for, should be a ChromaCollection"], 
    query: Annotated[str, "The query to search for, should be a string"],
    n_results: Annotated[int, "The number of results to return, should be an integer"] = 10
) -> List[str]:
    client = get_chroma_client()
    
    result = await client.asimilarity_search(
        collection_name=ChromaCollection.Faculty,
        query=query,
        n_results=n_results
    )

    return result

tools = [
    get_course_info,
    semantic_course_search,
    get_knowledge_by_faculty
]