from langchain_core.tools import tool
from database.types import Course, Program
from database.mongodb import get_mongodb_client
from database.enums import MongoCollection, ChromaCollection
from database.chromadb import get_chroma_client
from typing import List, Annotated, Optional
import logging

info_logger = logging.getLogger("uvicorn.info")

# @tool(description="Get the course information by course id, name, faculty, department, all optional")
@tool(description="Get the course information by course id")
async def get_course_info(
    course_id: Annotated[str, "The ID of the course, should be a string"],
    # name: Annotated[Optional[str], "The name of the course, should be a string"] = None,
    # faculty: Annotated[Optional[str], "The faculty of the course, should be a string"] = None,
    # department: Annotated[Optional[str], "The department of the course, should be a string"] = None,
    n_results: Annotated[int, "The number of results to return, should be an integer"] = 1
) -> Course:
    client = get_mongodb_client()

    course_id = course_id[:10] # format the course id to 10 characters max.
    info_logger.info(f"Searching for course with id: {course_id}, n_results: {n_results}")
    
    result = await client.query( # TODO: Add bulk query
        collection_name=MongoCollection.Course,
        query=course_id,
        n_results=n_results
    )
    
    # if not result:
    #     raise ValueError(f"Course {course_id} not found")
    info_logger.info(f"Found {len(result)} courses for {course_id}")
    return result

@tool(description="Search for courses by semantic similarity on the course overview, not the course id, name, faculty, department, etc.")
async def semantic_course_search(
    query: Annotated[str, "The query to search for, should be a string"], 
    n_results: Annotated[int, "The number of results to return, should be an integer"] = 10
) -> List[Course]:
    client = get_mongodb_client()

    info_logger.info(f"Searching for courses with query: {query}, n_results: {n_results}")
    
    result = await client.asimilarify_search(
        collection_name=MongoCollection.Course,
        query=query,
        n_results=n_results
    )

    # if not result:
    #     raise ValueError(f"No relevant courses found for {query}")
    info_logger.info(f"Found {len(result)} courses for {query}")
    return result

@tool(description="Search for programs by semantic similarity on the program overview, not the program id, name, faculty, department, etc.")
async def semantic_program_search(
    query: Annotated[str, "The query to search for, should be a string"],
    n_results: Annotated[int, "The number of results to return, should be an integer"] = 10
) -> List[Program]:
    client = get_mongodb_client()

    info_logger.info(f"Searching for programs with query: {query}, n_results: {n_results}")

    result = await client.asimilarify_search(
        collection_name=MongoCollection.Program,
        query=query,
        n_results=n_results
    )

    # if not result:
    #     raise ValueError(f"No relevant programs found for {query}")
    
    info_logger.info(f"Found {len(result)} programs for {query}")
    return result

@tool(description="Get general knowledges about McGill University and its faculties, departments, programs, courses, etc.")
async def get_knowledge_by_faculty(
    # faculty: Annotated[ChromaCollection, "The faculty to search for, should be a ChromaCollection"], 
    query: Annotated[str, "The query to search for, should be a string"],
    n_results: Annotated[int, "The number of results to return, should be an integer"] = 10
) -> List[str]:
    client = get_chroma_client()

    info_logger.info(f"Searching for knowledge by faculty with query: {query}, n_results: {n_results}")

    result = await client.asimilarity_search(
        collection_name=ChromaCollection.Faculty,
        query=query,
        n_results=n_results
    )

    return result

tools = [
    get_course_info,
    semantic_course_search,
    semantic_program_search,
    get_knowledge_by_faculty
]