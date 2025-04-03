from fastapi import APIRouter
from .mongodb import  get_async_mongodb_client, MongoDBClient
from .enums import MongoCollection, Department, Level
from agents.tools import search_course, search_program, query_mcgill
from langchain_core.messages import ToolMessage

router = APIRouter()

@router.get("/ping/mongodb")
async def ping_mongodb():
    client = MongoDBClient.get_instance()
    return await client.ensure_connection()

@router.get("/query/tools/{tool_name}")
async def test_tool(tool_name: str, query: str, n_results: int = 3):

    if tool_name == search_program.name:
        result: ToolMessage =  await search_program.ainvoke({
            "name": tool_name,
            "args": {
                "query": query,
                "department": [Department.BIOLOGY, Department.COMPUTER_SCIENCE],
                "n_results": n_results
            },
            "id": 124,
            "type": "tool_call"
        })
    elif tool_name == search_course.name:
        result: ToolMessage = await search_course.ainvoke({
            "name": tool_name,
            "args": {
                "query": query,
                "department": [],
                "level": Level.UGRAD,
                "n_results": n_results
            },
            "id": 123,
            "type": "tool_call"
        })
    elif tool_name == query_mcgill.name:
        result: ToolMessage = await query_mcgill.ainvoke({
            "name": tool_name,
            "args": {
                "query": query,
                "n_results": n_results
            },
            "id": 125,
            "type": "tool_call"
        })
    else:
        raise ValueError("invalid tool name:" + tool_name)

    return result.artifact



@router.get("/similarity/mongodb/programs")
async def similarity_mongodb_programs(query: str, n_results: int = 10):
    client = get_async_mongodb_client()
    return await client.asimilarify_search(MongoCollection.Program, query, n_results)