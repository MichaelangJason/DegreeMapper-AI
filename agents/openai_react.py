from langgraph.prebuilt import create_react_agent
from tools import tools
from langgraph.checkpoint.mongodb.aio import AsyncMongoDBSaver
from database.mongodb import get_mongodb_client
from async_lru import alru_cache
from llm.openai import get_openai_llm
from langgraph.graph.graph import CompiledGraph

@alru_cache(maxsize=1)
async def get_async_mongo_checkpointer():
    client = await get_mongodb_client().get_async_client()
    return AsyncMongoDBSaver(client)

@alru_cache(maxsize=1)
async def get_agent() -> CompiledGraph:
    """Get or create an agent instance for the given session ID"""
    
    mongo_memory = await get_async_mongo_checkpointer()
    llm = get_openai_llm()
    agent = create_react_agent(
        model=llm,  # Add your LLM here
        tools=tools,
        # prompt=prompt,
        checkpointer=mongo_memory
    )
    
    return agent

