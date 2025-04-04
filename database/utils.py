from .enums import MongoCollection, MongoIndex
from typing import Any, Dict, List, Tuple

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