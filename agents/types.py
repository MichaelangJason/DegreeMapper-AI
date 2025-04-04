from typing import TypedDict, Annotated, Optional, List, Literal, Any, Dict, Tuple
from database.enums import Level, Degree
from typing_extensions import NotRequired
try:
    from pydantic import BaseModel, Field
except ImportError:
    # Define a placeholder if pydantic is not available
    class BaseModel:
        pass
    def Field(*args, **kwargs):
        return None


CourseId = str
Program = str

class UserInfo(TypedDict, total=False):
    name: Annotated[Optional[str], "The name of the user"]
    program: Annotated[Optional[Program], "The program the user is interested in or is taking"]
    degree: Annotated[Optional[Degree], "The degree the user is interested in or is pursuing"]
    level: Annotated[Optional[Level], "The level of the user, either undergraduate or graduate"]
    courses_id_taken: Annotated[Optional[List[CourseId]], "The courses the user has taken, represented by their ids"]
    interests: Annotated[Optional[List[str]], "The interests of the user"]
    dislikes: Annotated[Optional[List[str]], "The dislikes of the user"]
    notes: Annotated[Optional[List[str]], "The notes of the user"]

class Option(TypedDict):
    content: Annotated[str, "The content of the option"]
    id: Annotated[str, "The id of the option, can be equal to content"]

class Question(TypedDict):
    question: Annotated[str, "The question to be asked to the user"]
    options: Annotated[List[Option], "The options to be chosen from"]

# class Context(TypedDict):
#     id: Annotated[str, "The id of the context"]
#     content: Annotated[Any, "The content of the context"]

ContextId = str
class ContextDict(TypedDict):
    type: Literal["user_info", "course", "program", "general"]
    value: Any
Context = Dict[ContextId, ContextDict]
ContextUpdate = Dict[ContextId, Tuple[Any | None, Literal["update", "delete"]]]

# TypedDict version of ContextUpdate with arbitrary string keys
class ContextUpdateDict(TypedDict):
    context_id: Annotated[ContextId, "the id of the context"]
    new_value: Annotated[Any | None, "the value of the context, either None to delete or any json serializable"]
    type: Literal["user_info", "course", "program", "general"]
    op: Literal["update", "delete"]

class Term(TypedDict):
    id: Annotated[str, "Term id provided from frontend"]
    name: Annotated[str, "Term name provided from frontend"]
    courses: Annotated[List[CourseId], "Term course id, must exists in database"]