from langgraph.graph.graph import CompiledGraph
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.mongodb.aio import AsyncMongoDBSaver
from database.mongodb import get_mongodb_client
from llm.openai import get_openai_llm
from async_lru import alru_cache
from .enums import Model, Node
from .types import OverallState
from .nodes import ToolExecutionHandler, InteractiveQuery, ContextManager, PersonaResponder
from .tools import tools
import logging

info_logger = logging.getLogger("uvicorn.info")

@alru_cache(maxsize=2)
async def get_compiled_graph(model: Model) -> CompiledGraph:
    if model == Model.OPENAI:
        get_llm = get_openai_llm

        ctxmanager_config = { "model": "gpt-4o-mini", "tag": Node.CONTEXT_MANAGER.value}
        persona_responder_config = { "model": "gpt-4o", "tag": Node.PERSONA_RESPONDER.value }

    elif model == Model.DEEPSEEK:
        get_llm = get_openai_llm
        
        ctxmanager_config = { "model": "gpt-4o-mini", "tag": Node.CONTEXT_MANAGER.value}
        persona_responder_config = { "model": "gpt-4o", "tag": Node.PERSONA_RESPONDER.value }
    # elif model == Model.HUGGINGFACE:  # TODO: add local huggingface model support
    #     llm = get_huggingface_llm()
    else:
        raise ValueError(f"Invalid model: {model}")

    mongodb_client = await get_mongodb_client().get_async_client()
    checkpointer = AsyncMongoDBSaver(mongodb_client)
    compiled_graph = graph.compile(checkpointer=checkpointer, debug=True)
    # compiled_graph.get_graph().draw_mermaid_png(output_file_path="graph.png")

    def route_tools(state: OverallState) -> str:
        if not state.get("tool_calls", None):
            return "persona_responder"
        else:
            return "execution_handler"

    graph = StateGraph(OverallState)
    graph.add_node(Node.EXECUTION_HANDLER.value, ToolExecutionHandler(tools=tools))
    graph.add_node(Node.INTERACTIVE_QUERY.value, InteractiveQuery())
    graph.add_node(Node.CONTEXT_MANAGER.value, ContextManager(get_llm=get_llm, llm_config=ctxmanager_config, tools=tools))
    graph.add_node(Node.PERSONA_RESPONDER.value, PersonaResponder(get_llm=get_llm, llm_config=persona_responder_config))

    graph.add_edge(START, Node.PERSONA_RESPONDER.value)
    graph.add_conditional_edges(Node.CONTEXT_MANAGER.value, route_tools)
    graph.add_edge(Node.EXECUTION_HANDLER.value, Node.INTERACTIVE_QUERY.value)
    graph.add_edge(Node.INTERACTIVE_QUERY, Node.CONTEXT_MANAGER.value)
    graph.add_edge(Node.PERSONA_RESPONDER.value, END)

    compiled_graph = graph.compile(checkpointer=checkpointer, debug=True)

    return compiled_graph

