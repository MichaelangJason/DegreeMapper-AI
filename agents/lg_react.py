from langgraph.prebuilt import ToolNode, create_react_agent
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field
from langgraph.checkpoint.mongodb import AsyncMongoDBSaver
from database.mongodb import get_mongodb_client