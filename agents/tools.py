from typing import List, Annotated
from langchain_core.tools import BaseTool, tool
from database.types import Course, Program
from database.mongodb import MongoDBClient
from database.enums import AcademicLevel, Faculty, Department, Degree, MongoCollection, CourseLevel
from database.utils import generate_course_id_pipeline
from .utils import parse_req
from .types import ContextUpdateDict, CourseId, Term, Plan
import logging

info_logger = logging.getLogger("uvicorn.info")

@tool(response_format="content_and_artifact")
async def search_program(
  query: Annotated[str, "The query string"],
  n_results: Annotated[int, "Number of results expected"] = 3,
  # show_requirement: Annotated[bool, "Whether to include inner requirements of a program"] = False, # whether to include sections or not
  level: Annotated[AcademicLevel, "the academic level of program to see, default is undergraduate"] = AcademicLevel.UGRAD,
  # TODO: rethink about this?
  faculty: Annotated[List[Faculty], "the faculty filter, default to empty"] = [],
  department: Annotated[List[Department], "the department filter, default to empty"] = [],
  degree: Annotated[List[Degree], "the degree filter, default to empty"] = []
):
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
      "level": [1, 2] if level == AcademicLevel.ALL else (1 if level == AcademicLevel.UGRAD else 2),
      "faculty": [f.value for f in faculty],
      "department": [d.value for d in department],
      "degree": [d.value for d in degree]
    },
    # proj={ "sections": 0 } if not show_requirement else {}
  )

  return "search_program resutls", [Program(**r) for r in results];

@tool(response_format="content_and_artifact")
async def search_course(
  query: Annotated[str, "The query string"],
  n_credits_limit: Annotated[float | List[float], "You can pass either a float as an upperbound for the courses fetched, common ones are 3 or 4. Or you can pass a list of float for exact filter. Default is empty list meaning no filter"] = [],
  n_results: Annotated[int, "Number of results expected"] = 3,
  # less_detail: Annotated[bool, "Whether to include"] = True,
  course_level: Annotated[List[CourseLevel], "the course levels to filter, default to empty"] = [],
  academic_level: Annotated[AcademicLevel, "the academic level of course to see, default is undergraduate"] = AcademicLevel.UGRAD,
  # TODO: rethink about this
  faculty: Annotated[List[Faculty], "the faculty filter, default to empty"] = [],
  department: Annotated[List[Department], "the department filter, default to empty"] = [],
):
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
      "academicLevel": 0 if academic_level == AcademicLevel.ALL else [0, 1 if academic_level == AcademicLevel.UGRAD else 2],
      "courseLevel": [l.value for l in course_level],
      "credits": n_credits_limit,
      "faculty": [f.value for f in faculty],
      "department": [d.value for d in department]
    }
  )

  return "search_course_result", [Course(**r) for r in results];

@tool(response_format="content_and_artifact")
async def query_mcgill_knowledges(
  query: Annotated[str, "query mcgill knowledge db for info"],
  n_results: Annotated[int, "Number of results expected"] = 3
):
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

@tool(response_format="content_and_artifact")
async def generate_base_plan(
  required_course_ids: Annotated[List[CourseId], "course id to be put in the plan"], 
  complementary_course_ids: Annotated[List[CourseId], "course id to be put in the plan as complementary courses"],
  target_credits: Annotated[int, "the total credits required for the program"],
  faculties: Annotated[List[Faculty], "the faculty of the program(s)"],
  departments: Annotated[List[Department], "the departments to fetch complementary courses from, must includes the same departments as the program(s)"],
  per_term_credits: Annotated[int, "An upperbound for each term, represent workloads"] = 15,
  # complementary course settings
  course_levels: Annotated[List[CourseLevel], "the course level to fetch complementary courses from, default is emtpy"] = [],
  academic_level: Annotated[AcademicLevel, "the academic level to fetch complementary courses from, must be set to the same academic level as the program(s)"] = AcademicLevel.UGRAD,
):
  """
  Given course id, generate a basic plan, you must adjust the basic plan to meets user need.
  The result is deterministic, meaning same course_ids (provided order does not matter) always return same results based on workload
  The plan generation considers prerequisites/co-requisites/anti-requisites.
  """

  # combine required and complementary course ids
  course_ids = required_course_ids + complementary_course_ids
  if len(course_ids) == 0:
    return "plan", [Plan(terms={}, notes={}, total_credits=0)];
  # sort by level
  course_ids = list(map(lambda c: c.lower().replace(" ", ""), course_ids))
  course_ids = sorted(course_ids, key=lambda id: id[4:])

  # fetch informations
  coll = await MongoDBClient.get_instance().get_async_collection(MongoCollection.Course)
  pipeline = generate_course_id_pipeline(course_ids)
  # info_logger.info(f"pipeline: {pipeline}")
  results = await coll.aggregate(pipeline=pipeline)
  results = await results.to_list()
  courses: List[Course] = [Course(**c) for c in results]
  courses.sort(key=lambda c: c["id"][4:])

  # info_logger.info(f"courseids: {course_ids}")
  # info_logger.info(f"result courses: {[c['id'] for c in courses]}")

  plan: Plan = {
    "terms": {
      "term_1": Term(
        id="term_1",
        name="Term 1",
        course_ids=[],
        total_credits=0,
      )
    },
    "notes": {},
    "total_credits": 0
  }

  terms = plan["terms"]
  notes = plan["notes"]

  # verify if any missing course ids
  fetched_course_ids: List[str] = list(map(lambda c: c["id"], courses))
  if len(fetched_course_ids) != len(course_ids):
    diff = [id for id in course_ids if id not in fetched_course_ids]
    notes.update({ "invalid_course_ids": diff })

  added_additional_courses = False
  possible_future_courses: List[CourseId] = []
  planned_courses: List[CourseId] = []

  while len(courses) > 0:
    course = courses.pop(0)
    prereq_ids, _ = parse_req(course["prerequisites"])
    coreq_ids, _ = parse_req(course["corequisites"])
    antireq_ids, _ = parse_req(course["restrictions"])



    # info_logger.info(f"current course: {course['id']}")
    # info_logger.info(f"prereq_ids: {prereq_ids}")
    # info_logger.info(f"coreq_ids: {coreq_ids}")
    # info_logger.info(f"antireq_ids: {antireq_ids}")
    # subject_code = course["id"][:4]

    # find the first term after any prereq and before any antireq that:
    # 1. has enough credits
    # 2. preferred if has a coreq

    planned = False
    # find the first term that has enough credits
    first_possible_term = None
    possible_term_with_coreq = None
    plannable = True

    # first verify if any antireq exists in any term
    # this means we cannot plan this course in any term
    for term in terms.values():
      if any(antireq_id in term["course_ids"] for antireq_id in antireq_ids):
        notes.update({ "unplannable_course": notes.get("unplannable_course", {}) })
        notes["unplannable_course"][course["id"]] = "antireq not met: " + ", ".join(antireq_ids)
        plannable = False
        break

    # info_logger.info(f"plannable: {plannable}")
    last_prereq_term_idx = 0
    for i, term in enumerate(reversed(list(terms.values()))):
      if any(prereq_id in term["course_ids"] for prereq_id in prereq_ids):
        last_prereq_term_idx = len(terms) - 1 - i
        break

    # info_logger.info(f"last_prereq_term_idx: {last_prereq_term_idx}")

    if not plannable:
      continue

    for term in list(terms.values())[last_prereq_term_idx:]: # order guaranteed
      # check if enough remaining credits
      if term["total_credits"] + course["credits"] > per_term_credits:
        continue

      # check if any prereq in this term
      if any(prereq_id in term["course_ids"] for prereq_id in prereq_ids):
        continue;
      
      # check if any coreq in this term
      if any(coreq_id in term["course_ids"] for coreq_id in coreq_ids):
        possible_term_with_coreq = term
        break

      # can be planned in this term
      if first_possible_term is None:
        first_possible_term = term
    
    if possible_term_with_coreq is not None:
      first_possible_term = possible_term_with_coreq
    
    if first_possible_term is not None:
      first_possible_term["course_ids"].append(course["id"])
      first_possible_term["total_credits"] += course["credits"]
      plan["total_credits"] += course["credits"]
      possible_future_courses.extend(course["futureCourses"])
      planned_courses.append(course["id"])
      planned = True

    # if not planned, create a new term
    if not planned:
      terms.update({
        f"term_{len(terms) + 1}": Term(
          id=f"term_{len(terms) + 1}",
          name=f"Term {len(terms) + 1}",
          course_ids=[course["id"]],
          total_credits=course["credits"],
        )
      })
      plan["total_credits"] += course["credits"]
      possible_future_courses.extend(course["futureCourses"])
      planned_courses.append(course["id"])
    
    if plan["total_credits"] >= target_credits:
      break

    # add additional complementary courses if not enough credits
    if len(courses) == 0 and plan["total_credits"] < target_credits and not added_additional_courses:
      # remove course levels after the level in the last term

      # gather future courses available and filter out by faculty and department
      results = await coll.aggregate(
        pipeline=generate_course_id_pipeline(
          included_ids=possible_future_courses,
          excluded_ids=planned_courses,
          faculties=faculties,
          departments=departments,
          excluded_levels=[CourseLevel.LEVEL_000, CourseLevel.LEVEL_100],
          included_levels=course_levels,
          academic_level=academic_level
        )
      )
      results = await results.to_list()
      courses = [Course(**c) for c in results]
      courses = list(filter(lambda c: c["credits"] + plan["total_credits"] <= target_credits, courses))

  return "plan", [plan];

tools: List[BaseTool] = [
  search_program,
  search_course,
  query_mcgill_knowledges,
  update_context,
  ask_user,
  generate_base_plan
]