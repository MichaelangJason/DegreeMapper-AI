from fastapi import APIRouter, Response, status
from agents.openai_react import get_agent
from langchain_core.messages import HumanMessage, AIMessageChunk, AnyMessage
from langchain_core.runnables import RunnableConfig
from sse_starlette import EventSourceResponse
from pydantic import BaseModel
from typing import List, Optional
from langgraph.types import Command
from .enums import Node
from .prompts import Prompts
import logging
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
    info_logger.info(request)
    
    # TODO: update to current setting with proper filter
    async def stream_response():    
        thread_id = request.thread_id
        if not thread_id:
            thread_id = str(uuid.uuid4()) # generate a new thread

        user_input = request.messages[-1] # TODO: user_input validation

        agent = await get_agent()
        config: RunnableConfig = { "configurable": { "thread_id": thread_id, "recursion_limit": 25 } }
        state = await agent.aget_state(config=config)

        info_logger.info(state)
        
        # resume if interrupted for human input
        should_resume = state.values.get("interrupted", False)

        # input = Command(
        #     # update={"messages": [HumanMessage(content=user_input)]} if not should_resume else {},
        #     resume=user_input if should_resume else None,
        #     goto=Node.INTERACTIVE_QUERY.value if should_resume else Node.CONTEXT_MANAGER.value
        # )
        input = ""
        if should_resume:
            input = Command(resume=user_input)
        else:
            messages: List[AnyMessage] = [HumanMessage(content=user_input)]
            if state.created_at is None:
                messages.insert(0, Prompts.get_intro_message())
            input = { "messages": messages }

        # print(input)

        stream = agent.astream_events(
            input=input,
            config=config,
            version="v2",
            include_names=[Node.PERSONA_RESPONDER.value, Node.INTERACTIVE_QUERY.value] # final response only
        )

        async for event in stream:
            # only chat model stream 

            # if (event["name"] == Node.INTERACTIVE_QUERY.value): info_logger.info(event)
            if event["event"] not in ("on_chat_model_stream", "on_chain_start"): continue

            # if (event["name"] == Node.PERSONA_RESPONDER.value and event["event"] != "on_chat_model_stream"): continue
            if event["event"] == "on_chat_model_stream":
                chunk: AIMessageChunk = event["data"]["chunk"]
                content = chunk.content
                event_type = "llm_response"
            elif event["event"] == "on_chain_start" and \
                 event["name"] == Node.INTERACTIVE_QUERY.value:
                
                input = event["data"]["input"]
                if isinstance(input, Command) or \
                   not input.get("interrupted", False): 
                    continue
                
                # interrupted
                content = event["data"]["input"].get("ask_user_call", {})
                if (content == {}):
                    raise Exception("Question not existing")
                event_type = "question"
            else:
                continue


            # print(event)
            # print(chunk)
            # print(content)
            
            result = {
                "data": {
                    "content": content,
                    "metadata": {
                        "thread_id": thread_id
                    }
                },
                "event": event_type
            }

            yield result


        # results = agent.astream(
        #     {"messages": [HumanMessage(content=user_input)]}, 
        #     config={"configurable": {"thread_id": thread_id, "recursion_limit": 10}},
        #     stream_mode="messages"
        # )
        # async for msg, metadata in results:
        #     if msg.content and "chatbot" in metadata.get("tags", []):
        #         yield f"event: message\ndata: {json.dumps({'content': msg.content, 'metadata': {"thread_id": metadata.get("thread_id", thread_id) }})}\n\n"

    return EventSourceResponse(stream_response())

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