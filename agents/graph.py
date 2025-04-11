from langgraph.graph.graph import CompiledGraph
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.mongodb.aio import AsyncMongoDBSaver
from database.mongodb import get_mongodb_client
from llm import get_deepseek_llm, get_openai_llm
from async_lru import alru_cache
from .enums import Model, Node
from .nodes import *
from .tools import tools
import logging

info_logger = logging.getLogger("uvicorn.info")

@alru_cache(maxsize=2)
async def get_compiled_graph(model: Model) -> CompiledGraph:
    if model == Model.OPENAI:
        get_llm = get_openai_llm

        ctxmanager_config = { "model": "gpt-4o-mini" }
        persona_responder_config = { "model": "gpt-4o-mini", "temperature": 0.7 }

    elif model == Model.DEEPSEEK:
        get_llm = get_deepseek_llm
        
        ctxmanager_config = { "model": "deepseek-reasoner" }
        persona_responder_config = { "model": "deepseek-reasoner" }
    # elif model == Model.HUGGINGFACE:  # TODO: add local huggingface model support
    #     llm = get_huggingface_llm()
    else:
        raise ValueError(f"Invalid model: {model}")

    mongodb_client = await get_mongodb_client().get_async_client()
    checkpointer = AsyncMongoDBSaver(mongodb_client)

    def route_tools(state: OverallState) -> str:
        if state.get("tool_calls", []) == []:
            return Node.PERSONA_RESPONDER.value
        else:
            return Node.EXECUTION_HANDLER.value

    graph = StateGraph(OverallState)
    graph.add_node(Node.EXECUTION_HANDLER.value, ToolExecutionHandler(tools=tools))
    graph.add_node(Node.INTERACTIVE_QUERY.value, InteractiveQuery())
    graph.add_node(Node.CONTEXT_MANAGER.value, ContextManager(get_llm=get_llm, llm_config=ctxmanager_config, tools=tools))
    graph.add_node(Node.PERSONA_RESPONDER.value, PersonaResponder(get_llm=get_llm, llm_config=persona_responder_config))

    # the edge here can be replace by Command from langgraph.types
    graph.add_edge(START, Node.CONTEXT_MANAGER.value)
    graph.add_conditional_edges(Node.CONTEXT_MANAGER.value, route_tools, {
        Node.PERSONA_RESPONDER.value: Node.PERSONA_RESPONDER.value,
        Node.EXECUTION_HANDLER.value: Node.EXECUTION_HANDLER.value
    })
    graph.add_edge(Node.EXECUTION_HANDLER.value, Node.INTERACTIVE_QUERY.value)
    graph.add_edge(Node.INTERACTIVE_QUERY.value, Node.CONTEXT_MANAGER.value)
    graph.add_edge(Node.PERSONA_RESPONDER.value, END)

    compiled_graph = graph.compile(checkpointer=checkpointer, debug=False)
    # compiled_graph.get_graph().draw_mermaid_png(output_file_path="graph.png")

    return compiled_graph

