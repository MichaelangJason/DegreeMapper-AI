from langgraph.prebuilt import ToolNode, create_react_agent
from tools import tools
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from pydantic import BaseModel, Field
from langgraph.checkpoint.mongodb.aio import AsyncMongoDBSaver
from database.mongodb import get_mongodb_client
from pymongo import AsyncMongoClient
from async_lru import alru_cache
from typing import Any
from llm.openai import get_openai_llm
from langgraph.graph.graph import CompiledGraph
from dotenv import load_dotenv
import os
from motor.motor_asyncio import AsyncIOMotorClient

load_dotenv()

@alru_cache(maxsize=1)
async def get_async_mongo_checkpointer():
    # client = await get_mongodb_client()._get_client()
    URI = os.getenv("MONGODB_URI")
    # client = AsyncIOMotorClient(URI)
    client = AsyncMongoClient(URI)
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

