from fastapi import APIRouter, WebSocket
from agents.lg_react import get_agent
from langchain_core.messages import HumanMessage
import logging
from websockets.exceptions import ConnectionClosed
import json

info_logger = logging.getLogger("uvicorn.info")
error_logger = logging.getLogger("uvicorn.error")
warning_logger = logging.getLogger("uvicorn.warning")
router = APIRouter()

@router.post("/chat/{thread_id}")
async def chat(thread_id: str, message: str):
    agent = await get_agent()
    config = {"configurable": {"thread_id": "thread_" + thread_id}}
    info_logger.info(config)
    info_logger.info(agent)
    results = await agent.ainvoke({"messages": [HumanMessage(content=message)]}, config=config)
    # print(results)
    return results

@router.websocket("/ws/{thread_id}")
async def chat_websocket(*, websocket: WebSocket, thread_id: str):
    try:
        await websocket.accept()
        agent = await get_agent()
        config = {"configurable": {"thread_id": "thread_" + thread_id}}
        info_logger.info(config)
        
        while True:
            data = await websocket.receive_text()
            info_logger.info(data)
            if data == "quit":
                break
            results = agent.astream({"messages": [HumanMessage(content=data)]}, config=config, stream_mode="values")
            info_logger.info(results)
            async for message_chunk in results:
                # await websocket.send_text(message_chunk.content)
                    await websocket.send_json(message_chunk["messages"][-1].content)

        await websocket.close()
    except ConnectionClosed:
        info_logger.info("Connection closed")
    

@router.websocket("/ws")
async def websocket_endpoint(*, websocket: WebSocket):
    await websocket.accept()
    while True:
        data = await websocket.receive_text()
        await websocket.send_text(f"Message text was: {data}")