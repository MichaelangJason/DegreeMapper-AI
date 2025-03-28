from langgraph.graph.message import add_messages, Messages
from langchain_core.messages import ToolCall
from typing import TypedDict, Annotated, Optional, List
from .reducer import user_preference_reducer

class UserInfo(TypedDict, total=False):
    name: Annotated[Optional[str], "The name of the user"]
    program: Annotated[Optional[str], "The program the user is interested in or is taking"]
    degree: Annotated[Optional[str], "The degree the user is interested in or is pursuing"]
    level: Annotated[Optional[str], "The level of the user, either undergraduate or graduate"]
    courses_id_taken: Annotated[Optional[List[str]], "The courses the user has taken, represented by their ids"]
    interests: Annotated[Optional[List[str]], "The interests of the user"]
    dislikes: Annotated[Optional[List[str]], "The dislikes of the user"]
    notes: Annotated[Optional[List[str]], "The notes of the user"]

class Option(TypedDict):
    content: Annotated[str, "The content of the option"]
    id: Annotated[str, "The id of the option"]

class Question(TypedDict):
    question: Annotated[str, "The question to be asked to the user"]
    options: Annotated[Optional[List[Option]], "The options to be chosen from"]

class Context(TypedDict):
    id: Annotated[str, "The id of the context"]
    content: Annotated[str, "The content of the context"]

class ContextUpdate(Context):
    op: Annotated[str, "The operation to be performed on the context"]

class OverallState(TypedDict, total=False):
    messages: Annotated[List[Messages], add_messages(format="langchain-openai")]
    user_info: Annotated[UserInfo, user_preference_reducer]
    contexts: Annotated[List[Context], "The contexts of the user's input"]
    tool_calls: Annotated[List[ToolCall], "The tool calls to be executed"]
    # this controls whether we directly invoke the agent with the input or use a Command to resume from the interrupted state
    interrupted: Annotated[bool, "Whether the agent is interrupted, waiting for user's input"]

class ToolNodeOutput(TypedDict):
    contexts: Annotated[List[ContextUpdate], "The tool results as contexts"]
    user_info: Annotated[Optional[UserInfo], "intermediate user info"]
    ask_user_call: Annotated[Optional[Question], "The question to be asked to the user"]
    interrupted: Annotated[bool, "change the agent state if any ask user call"]