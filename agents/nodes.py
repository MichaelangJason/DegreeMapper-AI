
from typing import Dict, Any, Callable
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.tools import BaseTool
from langchain_core.language_models import BaseChatModel
from langgraph.types import interrupt
from .types import OverallState, ToolNodeOutput
from .prompts import Prompts

class ToolExecutionHandler:
    """ 
    A node that runs the tools requested in the state.
    It clears the toolCalls after running the tools.
    """
    def __init__(self, tools: list):
        self.tools: Dict[str, BaseTool] = {tool.name: tool for tool in tools}

    async def __call__(self, state: OverallState) -> ToolNodeOutput:
        if not state.get("tool_calls", []):
            raise ValueError("No tool calls found in the state")
        
        tool_calls = state["tool_calls"]
        results = []
        
        # execute the tool calls
        for tool_call in tool_calls:
            tool_name = tool_call["name"]

            # skip ask_user and update_user_info
            # TODO: properly handle this part
            if tool_name not in self.tools:
                raise ValueError(f"Tool {tool_name} not found")
            elif tool_name == "ask_user":
                ask_user_call = tool_call
                continue
            elif tool_name == "update_user_info":
                user_info_update = tool_call
                continue
            elif tool_name == "update_contexts":
                contexts_update = tool_call
                continue

            tool = self.tools[tool_name]
            args = tool_call["args"]

            tool_result = await tool.ainvoke(args)
            results.append(tool_result)

        return {
            "contexts": {
                "content": results,
                "op": "append"
            },
            "user_info": user_info_update if user_info_update else None,
            "ask_user_call": ask_user_call if ask_user_call else None,
        }

class InteractiveQuery:
    """
    A node that asks the user for input.
    """
    async def __call__(self, state: ToolNodeOutput) -> OverallState:
        if not state.get("interrupted", False):
            # return the updates
            return {
                "user_info": state.get("user_info", None),
                "contexts": state.get("contexts", [])
            }
        else:
            question = state.get("ask_user_call")
            answer = interrupt(question)
            
            # TODO: add answer to messages, process option if there is any
            contexts = state.get("contexts")

            return {
                "messages": [AIMessage(content=question), HumanMessage(content=answer)], # update conversation to state history
                "user_info": state.get("user_info", None),
                "contexts": contexts,
                "interrupted": False, # resumed from interrupt
                "tool_calls": None # clear tool_calls
            }

class ContextManager:
    def __init__(self, get_llm: Callable[..., BaseChatModel], llm_config: Dict[str, Any], tools: list[BaseTool]=[]) -> None:
        self.llm = get_llm(**llm_config);
        self.tools = tools;
        
    async def __call__(self, state: OverallState) -> OverallState:
        user_query = state["messages"][-1];

        if (isinstance(user_query), HumanMessage):
            raise TypeError("The last message should be a HumanMessage")

        llm = self.llm.bind_tools(self.tools) if len(self.tools) > 0 else self.llm
        contexts = state["contexts"]
        user_info = state["user_info"]

        response = await llm.ainvoke(Prompts.get_manager_prompt(contexts=contexts, user_info=user_info))

        if hasattr(response, "tool_calls") and len(response.tool_calls) > 0:
            return {
                "tool_calls": response.tool_calls
            }
        else:
            return {
                "messages": [response]
            }

class PersonaResponder:
    def __init__(self, get_llm: Callable[..., BaseChatModel], llm_config: Dict[str, Any]) -> None:
        self.llm = get_llm(**llm_config);
    
    async def __call__(self, state: OverallState) -> OverallState:
        # answer user's question based on the context

        llm = self.llm
        chat_history = state["messages"]
        contexts = state["contexts"]
        user_info = state["user_info"]
        
        response = await llm.ainvoke(Prompts.get_persona_prompt(
            chat_history=chat_history, 
            contexts=contexts, 
            user_info=user_info
        ))

        return {
            "messages": [response]
        }

__all__ = [
    ToolExecutionHandler,
    InteractiveQuery,
    ContextManager,
    PersonaResponder
]

