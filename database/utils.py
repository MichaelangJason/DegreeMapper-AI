from .enums import MongoCollection, MongoIndex, Faculty, Department, CourseLevel, AcademicLevel
from typing import Any, Dict, List, Tuple
import logging

info_logger = logging.getLogger("uvicorn.info")

SEARCH_WEIGHTS = {
        collection: {
            MongoIndex.VECTOR: vector_index_weight,
            MongoIndex.FULL_TEXT: search_index_weight  
        } for collection, 
              (vector_index_weight, search_index_weight) 
          in zip(MongoCollection, [
            (1, 1), # course
            (1, 2), # programs
            (4, 1), # general
          ]) 
        }

RECIPROCAL_C = 60

def generate_vector_search_filter(
    filters: Dict[str, Any] = {}
):
  filter: Dict[str, Dict[str, Any] | List[Dict[str, Any]]] = {}

  for k, v in filters.items():
    if isinstance(v, list) and len(v) == 0:
      continue
    if (k == "course_level"):
      pass
    elif (k == "credits") and isinstance(v, float):
      filter.update({ k: { "$lte": v } })
    else:
      filter.update({ k: { "$in": v if isinstance(v, list) else [v] } })

  return filter
  
  # this is confusing lol, we can use a better format
  # filter = {
  #   "$or" if isinstance(v, list) else k: ([{ k: val } for val in v] if isinstance(v, list) else { "$eq": v })
  #   for k, v in filters.items()
  # }
  # print(filter)
  # return filter

def generate_search_filter(
    filters: Dict[str, Any] = {}
):
  filter: List[Dict[str, Any]] = []

  for k, v in filters.items():
    if isinstance(v, list) and len(v) == 0:
      continue
    elif (k == "credits") and isinstance(v, float):
      filter.append({ "range": { "path": k, "lte": v } })
    else:
      filter.append({ "in": { "path": k, "value": v if isinstance(v, list) else [v] } })
      
  return filter

def generate_search_stage(
    query: str,
    collection: MongoCollection
):
  def map_to_search(*queries: Tuple[str, str]):
    return [{
                # "text": {
                #   "query": query,
                #   "path": field,
                #   "fuzzy": { "maxEdits": 2 },
                # }
                "queryString": {
                  "defaultPath": field,
                  "query": query
                }
            }
            for (field, query) in list(queries)]
      
  
  if collection == MongoCollection.Course:
    # preprocess queries
    query_name = query.lower()[:100]+"~2";
    query_id_full = query.lower().replace(" ", "")[:10];
    query_id_code = query_id_full[:7]
    query_id = f"{query_id_full}~2 OR {query_id_code}d1~2 OR {query_id_code}d2~2 OR {query_id_code}j1~2 OR {query_id_code}j2~2 OR {query_id_code}j3~2"
    # print(query_id)
    return map_to_search(
      ("id", query_id),
      ("name", query_name)
    )
  elif collection == MongoCollection.General:
    return map_to_search(("content", query+"~2"))
  elif collection == MongoCollection.Program:
    query_name = query.lower()[:200]+"~2"
    return map_to_search(("name", query_name))
  

def generate_course_id_pipeline(
  included_ids: List[str],
  excluded_ids: List[str] = [],
  faculties: List[Faculty] = [],
  departments: List[Department] = [],
  excluded_levels: List[CourseLevel] = [],
  included_levels: List[CourseLevel] = [],
  academic_level: AcademicLevel = AcademicLevel.UGRAD
):

  filters = []
  must_not = []


  if len(faculties) > 0:
    filters.append({
      "in": {
        "path": "faculty",
        "value": [f.value for f in faculties]
      }
    })
  if len(departments) > 0:
    filters.append({
      "in": {
        "path": "department",
        "value": [d.value for d in departments]
      }
    })
  if len(included_levels) > 0:
    filters.append({
      "in": {
        "path": "courseLevel",
        "value": [l.value for l in included_levels]
      }
    })
  filters.append({
    "in": {
      "path": "academicLevel",
      "value": [0] if academic_level == AcademicLevel.ALL else [0, 1 if academic_level == AcademicLevel.UGRAD else 2]
    }
  })
  
  if len(excluded_ids) > 0:
    must_not.append({
      "text": {
        "path": "id",
        "query": excluded_ids
      }
    })
  if len(excluded_levels) > 0:
    must_not.append({
      "in": {
        "path": "courseLevel",
        "value": [l.value for l in excluded_levels]
      }
    })
  
  # info_logger.info(f"included_ids: {included_ids}")
  # info_logger.info(f"filters: {filters}")
  # info_logger.info(f"must_not: {must_not}")
  
  return [
    {
      "$search": {
        "index": "full_text_index",
        "compound": {
          "must": [
            {
              "text": {
                "path": "id",
                "query": included_ids
              }
            }
          ],
          "mustNot": must_not,
          "filter": filters
        }
      }
    },
    {
      "$addFields": {
        "nameSubstring": { "$substr": ["$id", 4, -1]}
      }
    },
    {
      "$sort": {
        "nameSubstring": 1 # ascending order
      }
    },
    {
      "$project": {
        "prerequisites": 1,
        "corequisites": 1,
        "restrictions": 1,
        "futureCourses": 1,
        "notes": 1,
        "credits": 1,
        "name": 1,
        "id": 1,
      }
    },
    {
      "$project": {
        "_id": 0
      }
    }
  ]