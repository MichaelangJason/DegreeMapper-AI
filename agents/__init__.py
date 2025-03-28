from fastapi import APIRouter, Response, status
from agents.openai_react import get_agent
from langchain_core.messages import HumanMessage
import logging
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Optional
import json
import uuid

info_logger = logging.getLogger("uvicorn.info")
error_logger = logging.getLogger("uvicorn.error")
warning_logger = logging.getLogger("uvicorn.warning")
router = APIRouter(prefix="/api")
class Request(BaseModel):
    messages: List[str]
    thread_id: Optional[str] = None

@router.post("/chat")
async def handle_chat(request: Request):
    # data = request.messages

    info_logger.info(request)
    
    async def stream_response():    
        thread_id = request.thread_id
        if not thread_id:
            thread_id = str(uuid.uuid4())

        user_input = request.messages[-1]

        agent = await get_agent()
        results = agent.astream(
            {"messages": [HumanMessage(content=user_input)]}, 
            config={"configurable": {"thread_id": thread_id, "recursion_limit": 10}},
            stream_mode="messages"
        )
        async for msg, metadata in results:
            if msg.content and "chatbot" in metadata.get("tags", []):
                yield f"event: message\ndata: {json.dumps({'content': msg.content, 'metadata': {"thread_id": metadata.get("thread_id", thread_id) }})}\n\n"

    return StreamingResponse(stream_response(), media_type="text/event-stream")

@router.get("/chat/{thread_id}")
async def get_chat(thread_id: str):
    agent = await get_agent()
    results = await agent.aget_state({"configurable": {"thread_id": thread_id}})

    if not results.values.get("messages", None) or len(results.values.get("messages", [])) == 0: # no messages in the thread
        error_logger.error(f"No attributes? {results.values.get('messages', None)}")
        error_logger.error(f"No messages? {len(results.values.get('messages', []))}")
        # error_logger.error(f"Results values: {results.values}")
        # error_logger.error(f"Results metadata: {results.metadata}")
        # error_logger.error(f"Results: {results.parent_config.get('configurable')}")

        return Response(status_code=status.HTTP_404_NOT_FOUND)
    
    info_logger.info(f"Messages in the thread {thread_id}: {results.values['messages']}")
    info_logger.info(f"Thread id: {results.config}")
    response = {
        "messages": [{"role": msg.type, "content": msg.content} for msg in results.values.get("messages") if msg.type != "tool"],
        "thread_id": results.config.get("configurable").get("thread_id")
    }
    
    return response