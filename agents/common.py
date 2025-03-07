from typing import Annotated, List, Dict, Any, Optional, Callable, TypedDict
from langgraph.graph.message import add_messages, Messages
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts.chat import MessagesPlaceholder # https://python.langchain.com/api_reference/core/prompts/langchain_core.prompts.chat.MessagesPlaceholder.html
from langchain_core.messages import SystemMessage, ToolMessage
from langchain_core.language_models import BaseChatModel
from tools import tools
from functools import partial
import json
from llm.openai import get_openai_llm
from langgraph.graph.graph import CompiledGraph
from langgraph.graph import StateGraph, START, END
from .enums import Model
from database.mongodb import get_mongodb_client
from langgraph.checkpoint.mongodb.aio import AsyncMongoDBSaver
from async_lru import alru_cache
import logging

info_logger = logging.getLogger("uvicorn.info")

class UserInfo(TypedDict, total=False):
    name: Annotated[Optional[str], "The name of the user"]
    program: Annotated[Optional[str], "The program the user is interested in or is taking"]
    degree: Annotated[Optional[str], "The degree the user is interested in or is pursuing"]
    level: Annotated[Optional[str], "The level of the user, either undergraduate or graduate"]
    courses_id_taken: Annotated[Optional[List[str]], "The courses the user has taken, represented by their ids"]
    interests: Annotated[Optional[List[str]], "The interests of the user"]
    dislikes: Annotated[Optional[List[str]], "The dislikes of the user"]
    notes: Annotated[Optional[List[str]], "The notes of the user"]

class Term(TypedDict):
    pass

def user_preference_reducer(prev: UserInfo, update: UserInfo) -> UserInfo:
    if prev is None or prev == {}:
        return update
    if update is None or update == {}:
        return prev
    # delete any empty fields in the update
    update = {k: v for k, v in update.items() if v is not None}
    return {**prev, **update} # merge the two dictionaries

class OverallState(TypedDict, total=False):
    # can defined by inheriting from MessagesState of langgraph.graph, https://langchain-ai.github.io/langgraph/concepts/low_level/#messagesstate
    messages: Annotated[List[Messages], add_messages(format="langchain-openai")] 
    # should store information about user's program, degree, interests, etc., 
    # should be updated as the agent interacts with the user. Pass {} at the beginning
    # not persistent across sessions for now, unlike ChatGPT's memory.
    user_info: Annotated[UserInfo, user_preference_reducer]
    # current user plan from frontend
    user_plan: List[Term]


# first node in the graph
async def info_updater(state: OverallState, get_llm: Callable[..., BaseChatModel], llm_config: Dict[str, Any]) -> OverallState:
    """
    this function ask the llm to extract the possible information from the user's message before answering.
    this can return none if the user's message is not related to any fields expected in the user_info.
    """

    # the JSON output here can also be done with the method mentioned here: https://github.com/langchain-ai/langchainjs/issues/4555
    llm = get_llm(**llm_config).with_structured_output(UserInfo)

    if not "messages" in state \
        or not isinstance(state["messages"], List) \
        or state["messages"][-1].type != "human":
        
        info_logger.info(f"state: {state}")
        # info_logger.info(f"Last message should be a Human Message: {state['messages'][-1]}")
        info_logger.info(f"hasattr? {hasattr(state, 'messages')}")
        info_logger.info(f"isinstance? {isinstance(state['messages'], List)}")

        info_logger.info(f"last message type: {state['messages'][-1].type}, is it Human Message? {state['messages'][-1].type == 'human'}")
        raise ValueError("Last message should be a Human Message")

    user_input = state["messages"][-1]
    system_message = SystemMessage(content=
                      "You are helping the next agent to extract possible information from the user's input. "
                      "You will be given the user's input, the previous user_info json, and the expected json schema to extract. "
                      "User's input can be unrelated to any of the fields expected in the json schema. "
                      "In that case, you should return empty json object."
                      "If there is no new information, you should return the previous user_info json unchanged."
                      "Else you should update the corresponding fields in the user_info, either by adding new values or overwriting the previous ones."
                      "The user could input an abbreviation of the program, degree, or level, you should expand them to the full form."
                      "you should not make up any information, only use the information provided in the user's input."
                      "Before updating the program, degree, you should check whether the information exists with the tool provided."
                      "degree can only be undergraduate or graduate."
                      "Here is the previous user_info json: {prev_user_info}"
                      )

    prev_user_info = state.get("user_info", {})
    prompt = ChatPromptTemplate.from_messages([system_message, user_input]).invoke({
        "prev_user_info": json.dumps(prev_user_info)
    })

    result = await llm.ainvoke(input=prompt)
    info_logger.info(f"result: {result}")

    return { "user_info": result }
    

# for now it only answers questions, not helping with Recommendation and Frontend Interaction.
async def chatbot(state: OverallState, get_llm: Callable[..., BaseChatModel], llm_config: Dict[str, Any]) -> Dict[str, List[str]]:
    """
    this function should take in the state and return a message to be added to the conversation history.
    """
    # llm = get_openai_llm("gpt-4o").bind_tools(tools=tools) # use a larger model for this task
    llm = get_llm(**llm_config).bind_tools(tools)

    template = ChatPromptTemplate.from_messages([
        SystemMessage(content=
                      "You are a helpful assistant that can answer questions and help with tasks. "
                      "If something is not in the context, use the appropriate tool to get the information you need. "
                      "If you don't know the answer based on the context, just say so. Don't make up an answer of information that is not provided in the context."
                      ),
        SystemMessage(content=
                      "You will be given the information about the user, answer the question based on the information provided."
                      "Here is the informations about the user: {user_info}"
                      ),
        MessagesPlaceholder("history") # TODO: should be later optimized to summarize the conversation history if it's too long.
    ])

    prompt = template.invoke({
        "user_info": json.dumps(state["user_info"]),
        "history": state["messages"]
    })

    message = await llm.ainvoke(input=prompt) # TODO: can this be streamed?

    return {"messages": [message]} # will be appended to the conversation history

# tool node, this is explicitly defined for the tool calling for more control over the tool calling process
# you can use ToolNode from langgraph.prebuilt to create a tool node
class BasicToolNode:
    """A node that runs the tools requested in the last AIMessage."""

    def __init__(self, tools: list) -> None:
        self.tools_by_name = {tool.name: tool for tool in tools}

    async def __call__(self, inputs: dict):
        if messages := inputs.get("messages", []):
            message = messages[-1] # last message
        else:
            raise ValueError("No message found in input")
        outputs = []
        for tool_call in message.tool_calls:
            tool_result = await self.tools_by_name[tool_call["name"]].ainvoke(
                tool_call["args"]
            )
            outputs.append(
                ToolMessage(
                    content=json.dumps(tool_result),
                    name=tool_call["name"],
                    tool_call_id=tool_call["id"],
                )
            )
        return {"messages": outputs}

# definition from LangGraph's example
def route_tools(
    state: OverallState,
    tool_node_name: str = "tools_chatbot",
    end_node_name: str = END
):
    """
    Use in the conditional_edge to route to the ToolNode if the last message
    has tool calls. Otherwise, route to the end.
    """
    if isinstance(state, list):
        ai_message = state[-1]
    elif messages := state.get("messages", []):
        ai_message = messages[-1]
    else:
        raise ValueError(f"No messages found in input state to tool_edge: {state}")
    if hasattr(ai_message, "tool_calls") and len(ai_message.tool_calls) > 0:
        return tool_node_name
    return end_node_name
    

@alru_cache(maxsize=1)
async def get_async_mongo_checkpointer():
    client = await get_mongodb_client().get_async_client()
    return AsyncMongoDBSaver(client)


@alru_cache(maxsize=2)
async def get_compiled_graph(model: Model) -> CompiledGraph:
    if model == Model.OPENAI:
        get_llm = get_openai_llm
    # elif model == Model.HUGGINGFACE:  # TODO: add local huggingface model support
    #     llm = get_huggingface_llm()
        chatbot_config = {"model": "gpt-4o", "tag": "chatbot"}
        info_updater_config = {"model": "gpt-4o-mini", "tag": "info_updater"}
    else:
        raise ValueError(f"Invalid model: {model}")
    
    graph = StateGraph(OverallState)
    graph.add_node("info_updater", partial(info_updater, get_llm=get_llm, llm_config=info_updater_config))
    graph.add_node("chatbot", partial(chatbot, get_llm=get_llm, llm_config=chatbot_config))
    graph.add_node("tools_chatbot", BasicToolNode(tools)) # can also be replaced with ToolNode from langgraph.prebuilt

    graph.add_edge(START, "info_updater")
    graph.add_edge("info_updater", "chatbot")
    graph.add_edge("tools_chatbot", "chatbot")
    graph.add_conditional_edges(
        "chatbot",
        route_tools, # can also be replaced with tools_condition from langgraph.prebuilt
        {
            "tools_chatbot": "tools_chatbot",
            END: END
        }
    )

    checkpointer = await get_async_mongo_checkpointer()
    compiled_graph = graph.compile(checkpointer=checkpointer, debug=True)
    compiled_graph.get_graph().draw_mermaid_png(output_file_path="graph.png")

    return compiled_graph

