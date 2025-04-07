from typing import Dict, Any, Callable, Annotated, TypedDict, List, Optional
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolCall, ToolMessage, RemoveMessage
from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool
from langgraph.graph.message import add_messages, Messages
from langgraph.graph import END
from langgraph.types import interrupt, Command
from .tools import ask_user, update_context, search_course, search_program, query_mcgill_knowledges, generate_base_plan
from .types import Context, ContextUpdateDict, Question
from .reducer import context_reducer
from .prompts import Prompts
from .enums import Node
import logging
import json
import traceback

error_logger = logging.getLogger("uvicorn.error")
info_logger = logging.getLogger("uvicorn.info")


class OverallState(TypedDict, total=False):
    messages: Annotated[List[Messages], add_messages]
    # user_info: Annotated[UserInfo, user_info_reducer]
    contexts: Annotated[Context, context_reducer]

    # The tool calls to be executed
    tool_calls: List[ToolCall]

    # this controls whether we directly invoke the agent with the input or use a Command to resume from the interrupted state
    # "Whether the agent is interrupted, waiting for user's input"
    interrupted: bool

class ToolNodeOutput(TypedDict):
    contexts_update: Annotated[List[ContextUpdateDict], "The tool results as contexts"]
    # user_info: Annotated[UserInfo, "intermediate user info"]
    ask_user_call: Annotated[Optional[Question], "The question to be asked to the user"]
    interrupted: Annotated[bool, "change the agent state if any ask user call"]

class ToolExecutionHandler:
    """ 
    A node that runs the tools requested in the state.
    It clears the toolCalls after running the tools.
    """
    def __init__(self, tools: list):
        self.tools: Dict[str, BaseTool] = {tool.name: tool for tool in tools}

    async def __call__(self, state: OverallState) -> ToolNodeOutput:
        if not (tool_calls := state.get("tool_calls", [])):
            raise ValueError("No tool calls found in the state")
        
        contexts_update: List[ContextUpdateDict] = []
        # user_info_update = None
        ask_user_call = None
        
        # execute the tool calls
        for tool_call in tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]
            tool_call_id = tool_call["id"]

            # TODO: properly handle this part
            if tool_name not in self.tools or tool_call_id is None:
                raise ValueError(f"Tool {tool_name} not found")
            elif tool_name == ask_user.name:
                # error_logger.error(args)
                ask_user_call: Question = tool_args
                continue
            # elif tool_name == update_user_info.name:
            #     error_logger.error(args)
            #     user_info_update: UserInfo = args
            #     continue
            elif tool_name == update_context.name:
                # error_logger.error(args)
                contexts_update.extend(tool_args.get("updates", []))
                continue

            tool = self.tools[tool_name]
            try: 
                # error_logger.error(tool_call)
                result: ToolMessage = await tool.ainvoke(tool_call)
                # error_logger.error(result)
                
                artifact: List = result.artifact if result.artifact else []

                # no results for this query.
                if len(artifact) == 0:
                    contexts_update.append({
                        "context_id": tool_call_id,
                        "new_value": f"No results for tool call {tool_name} with args: {json.dumps(tool_args)}",
                        "type": "no_result",
                        "op": "update"
                    })

                for r in artifact:
                    context_id = ""
                    if tool_name == search_program.name:
                        context_id = f"{r["faculty"]} - {r["name"]}"
                        context_type = "program"
                    elif tool_name == search_course.name:
                        context_id = r["id"]
                        context_type = "course"
                    elif tool_name == query_mcgill_knowledges.name:
                        context_id = r.get("id", None)
                        context_type = "general"
                    elif tool_name == generate_base_plan.name:
                        context_id = "new_plan"
                        context_type = "plan"
                    else:
                        raise ValueError("tool name invalid: ", tool_name)

                    if context_id is None:
                        raise ValueError("tool result error: ", r)

                    contexts_update.append({
                        "context_id": context_id,
                        "new_value": r,
                        "op": "update",
                        "type": context_type
                    })
                
            except Exception as e:
                formatted_traceback = ''.join(traceback.format_exception(type(e), e, e.__traceback__))
                error_logger.error(formatted_traceback)
                # error_logger.error(f"Error calling {tool_name} with arguments:\n\n{tool_args}\n\nraised the following error:\n\n{type(e)}: {e}")
                return Command(
                    update={ "messages": [
                        SystemMessage(
                            content=f"Calling {tool_name} with arguments:\n\n{tool_args}\n\nraised the following error:\n\n{type(e)}: {e}",
                            tool_call_id=tool_call_id
                            )] 
                        },
                    goto=Node.CONTEXT_MANAGER.value
                )

            if not isinstance(result, ToolMessage):
                error_logger.error(type(result))
                error_logger.error(result)
                raise ValueError("Results from tool calls should be ToolMessage")
            if result.artifact is None:
                raise ValueError("Results should contain artifact")

        return {
                "contexts_update": contexts_update,
                # "user_info": user_info_update,
                "ask_user_call": ask_user_call,
                "interrupted": bool(ask_user_call) # Unbound Variable
            }

class InteractiveQuery:
    """
    A node that asks the user for input.
    """
    async def __call__(self, state: ToolNodeOutput) -> OverallState:
        if not state.get("interrupted", False):
            # return the updates
            return {
                # "user_info": state.get("user_info", None),
                "contexts": state.get("contexts_update", {}),
                "contexts_update": None,
            }
        else:
            ask_user_call = state.get("ask_user_call", None)
            if ask_user_call is None:
                return Command(
                    update={ 
                        "messages": [ToolMessage(content="Calling ask_user with None Value")],
                        "interrupted": False,
                        "contexts": state.get("contexts_update", {}),
                        "contexts_update": None
                        # "user_info": state.get("user_info", None)
                    },
                    goto=Node.CONTEXT_MANAGER.value
                )


            answer = interrupt(ask_user_call)
            
            # TODO: add answer to messages, process option if there is any

            return {
                "messages": [AIMessage(content=json.dumps(ask_user_call), message_type="question"), HumanMessage(content=answer)], # update conversation to state history
                # "user_info": state.get("user_info", None),
                "contexts": state.get("contexts_update", []),
                "contexts_update": None,
                "interrupted": False, # resumed from interrupt
            }

class ContextManager:
    def __init__(self, get_llm: Callable[..., BaseChatModel], llm_config: Dict[str, Any], tools: list[BaseTool]=[]) -> None:
        self.llm = get_llm(**llm_config);
        self.tools = tools;
        
    async def __call__(self, state: OverallState) -> OverallState:
        contexts = state.get("contexts", {})
        chat_history = state.get("messages", [])
        made_tool_call = len(state.get("tool_calls", [])) > 0
        user_query = chat_history[-1];
        fail_call_id = None

        if isinstance(user_query, ToolMessage):
            info_logger.info(f"[Last Tool Call failed, no contexts are updated, try different args] \n error: {user_query.content}")
            fail_call_id = user_query.id
        elif not isinstance(user_query, HumanMessage):
            error_logger.error(user_query)
            raise TypeError("The last message should be a HumanMessage")

        llm = self.llm.bind_tools(self.tools) if len(self.tools) > 0 else self.llm
        llm = llm.with_config({ "run_name": Node.CONTEXT_MANAGER.value })

        # user_info = state.get("user_info", {})

        prompt = Prompts.get_manager_prompt(
                contexts=contexts, 
                chat_history=chat_history,
                made_tool_call=made_tool_call
            )

        response: AIMessage = await llm.ainvoke(prompt)
        # response.type = "assistant"
        # info_logger.info(response.id)
        

        if hasattr(response, "tool_calls") and len(response.tool_calls) > 0:
            return {
                "tool_calls": response.tool_calls,
                "messages": [RemoveMessage(fail_call_id)] if fail_call_id else []
            }
        else:
            return {
                "messages": [response] + ([RemoveMessage(fail_call_id)] if fail_call_id else []),
                "tool_calls": [] # clear tool calls
            }

class PersonaResponder:
    def __init__(self, get_llm: Callable[..., BaseChatModel], llm_config: Dict[str, Any]) -> None:
        self.llm = get_llm(**llm_config);
    
    async def __call__(self, state: OverallState) -> OverallState:
        # answer user's question based on the context

        llm = self.llm.with_config({ "run_name": Node.PERSONA_RESPONDER.value })
        chat_history = state["messages"]
        assistant_message: AIMessage = state["messages"][-1]
        # info_logger.info(assistant_message)
        # info_logger.info(assistant_message.id)
        # info_logger.info(type(assistant_message))
        contexts = state["contexts"]
        # user_info = state["user_info"]

        prompt = Prompts.get_persona_prompt(
            chat_history=chat_history, 
            contexts=contexts, 
            # user_info=user_info
        )
        
        response: AIMessage = await llm.ainvoke(prompt)

        # print(response)
        # for m in state["messages"]:
        #     info_logger.info(m)
        #     info_logger.info(m.id)

        # return {
        #     "messages": [RemoveMessage(id=assistant_message.id), response],
        # }
        return Command(
            update={
                "messages": [RemoveMessage(id=assistant_message.id), response],
                "contexts": [{
                    "context_id": None,
                    "new_value": None,
                    "type": None,
                    "op": "clear"
                }]
            },
            goto=END
        )


__all__ = [
    "ToolExecutionHandler",
    "InteractiveQuery",
    "ContextManager",
    "PersonaResponder",
    "OverallState",
    "ToolNodeOutput"
]

