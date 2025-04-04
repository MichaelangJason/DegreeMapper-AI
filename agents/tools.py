from typing import List, Annotated, Dict, Tuple, Any, Literal
from langchain_core.tools import BaseTool, tool
from .types import UserInfo, Option, Term, ContextId, ContextUpdateDict
from database.types import Course, Program
from database.mongodb import MongoDBClient
from database.enums import Level, Faculty, Department, Degree, MongoCollection

@tool(response_format="content_and_artifact")
async def search_program(
  query: Annotated[str, "The query string"],
  n_results: Annotated[int, "Number of results expected"] = 3,
  # show_requirement: Annotated[bool, "Whether to include inner requirements of a program"] = False, # whether to include sections or not
  level: Annotated[Level, "the level of program to see, default is undergraduate"] = Level.UGRAD,
  # TODO: rethink about this?
  faculty: Annotated[List[Faculty], "the faculty filter, default to empty"] = [],
  department: Annotated[List[Department], "the department filter, default to empty"] = [],
  degree: Annotated[List[Degree], "the degree filter, default to empty"] = []
) -> List[Program]:
  """
  Search programs offered at McGill. Multiple filters applicable with default values.
  Make sure you are using correct enum value for filters like faculty or department.
  This tool is ONLY used to search program information.
  """
  client = MongoDBClient.get_instance()
  results = await client.hybrid_search(
    query=query,
    collection=MongoCollection.Program,
    n_results=n_results,
    filter={
      "level": [Level.UGRAD.value, Level.GRAD.value] if level == Level.ALL else level.value,
      "faculty": [f.value for f in faculty],
      "department": [d.value for d in department],
      "degree": [d.value for d in degree]
    },
    # proj={ "sections": 0 } if not show_requirement else {}
  )

  return "search_program resutls", [Program(r) for r in results];

@tool(response_format="content_and_artifact")
async def search_course(
  query: Annotated[str, "The query string"],
  n_credits_limit: Annotated[float | List[float], "You can pass either a float as an upperbound for the courses fetched, common ones are 3 or 4. Or you can pass a list of float for exact filter. Default is empty list meaning no filter"] = [],
  n_results: Annotated[int, "Number of results expected"] = 3,
  # less_detail: Annotated[bool, "Whether to include"] = True,
  level: Annotated[Level, "the level of program to see (all=0, undergraduate=1, graduate=2) default is undergraduate"] = Level.UGRAD,
  # TODO: rethink about this
  faculty: Annotated[List[Faculty], "the faculty filter, default to empty"] = [],
  department: Annotated[List[Department], "the department filter, default to empty"] = [],
) -> List[Course]:
  """
  Search courses offered at McGill. Multiple filters applicable with default values.
  Make sure you are using correct enum value for filters like faculty or department.
  This tool is ONLY used to search relevant course information
  """
  client = MongoDBClient.get_instance()
  results = await client.hybrid_search(
    query=query,
    n_results=n_results,
    collection=MongoCollection.Course,
    filter={
      "level": 0 if level == Level.ALL else [0, level.value],
      "credits": n_credits_limit,
      "faculty": [f.value for f in faculty],
      "department": [d.value for d in department]
    }
  )

  return "search_course_result", [Course(r) for r in results];

@tool(response_format="content_and_artifact")
async def query_mcgill_knowledges(
  query: Annotated[str, "query mcgill knowledge db for info"],
  n_results: Annotated[int, "Number of results expected"] = 3
) -> List[Dict]:
  """
  Semantically query McGill knowledges database.
  """
  client = MongoDBClient.get_instance()
  results = await client.hybrid_search(
    query=query,
    collection=MongoCollection.General,
    n_results=n_results
  )

  return "query_mcgill results", results;

@tool
def update_user_info(new_values: UserInfo):
  """
  Use this tool to manage (update values) of the user info fields.
  UserInfo is a json and you can updates it with partial fields. too
  """
  return new_values;

@tool
def update_context(updates: List[ContextUpdateDict]):
  """
  Use this tool to manage the contexts you want to keep.
  context is a key, value pair. where keys are context name, and value can be any serializable json object
  """
  return updates;

@tool
def ask_user(
  question: Annotated[str, "the question you will ask"], 
  options: Annotated[List[str], "the predefined answer you expect from user, will be listed as options"]
):
  """
  Use this tool to ask user for missing informations, you can provide options (predefined answers) for the user to answer but this is not required.
  Notice that each Option must contain an id. 
  If you want the user to choose between different programs or courses, you must first retrieve the docs from database,
  use their name as option and their 'id' or 'name' as id.
  """
  return question, options;

@tool
def generate_plan(terms: List[Term]):
  """
  Generate a plan. 
  Include only the terms you want to updtes in order.
  """
  return terms;

tools: List[BaseTool] = [
  search_program,
  search_course,
  query_mcgill_knowledges,
  # update_user_info,
  update_context,
  ask_user,
  generate_plan
]