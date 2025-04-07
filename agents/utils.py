from typing import List, Tuple
from database.types import Requisites
from database.enums import CourseLevel
from .types import CreditGroup, CourseId
import re

def parse_req(req: Requisites) -> Tuple[List[CourseId], List[CreditGroup]]:
  # remove credits requirements from Requisites
  rest, credits_req = pop_substrings(req['parsed'], r'[0-9]{1,2}-[0-9]{1,}(-[a-zA-Z]{4})+')
  
  credits_groups: List[CreditGroup] = []
  for r in credits_req:
    items = r.split(sep="-")
    credits_requirement = items[0]
    course_levels = [map_course_level(char) for char in items[1]]
    subject_codes = items[2:]

    credits_groups.append(CreditGroup(
      credits_requirement=credits_requirement,
      course_levels=course_levels,
      subject_codes=subject_codes
    ))

  _, course_ids = pop_substrings(rest, r'[^+\|\-\(\)\/]+')

  return course_ids, credits_groups
  
def map_course_level(level: str):
  return CourseLevel["LEVEL_" + level[0].upper() + "00"]

def pop_substrings(source: str, pattern: re.Pattern):
  matches = re.finditer(pattern, source, flags=re.IGNORECASE | re.MULTILINE)
  rest = re.sub(pattern, '', source, flags=re.IGNORECASE | re.MULTILINE)

  matches = [m.group().lower().replace(" ", "") for m in matches if m.group().strip() != ""]

  return rest, matches