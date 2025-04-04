from pydantic import BaseModel # not serializable?
from typing import List, Optional, Dict
from typing_extensions import TypedDict
from .enums import Level, Faculty, Degree, Department

class Prerequisites(TypedDict, total=False):
    raw: str
    parsed: str

class Corequisites(TypedDict, total=False):
    raw: str
    parsed: str

class Restrictions(TypedDict, total=False):
    raw: str
    parsed: str  # restricted taken or taking courses

class Course(TypedDict, total=False):
    id: str
    name: str
    credits: float  # Using float since it's a number in TypeScript
    faculty: str
    department: str
    level: int
    terms: List[str]
    overview: Optional[str] = None
    instructors: Optional[str] = None
    notes: Optional[List[str]] = None
    prerequisites: Optional[Prerequisites] = None
    corequisites: Optional[Corequisites] = None
    restrictions: Optional[Restrictions] = None
    futureCourses: Optional[List[str]] = None

    class Config:
        from_attributes = True  # For ORM compatibility

class Program(TypedDict, total=False):
    url: str
    degree: str
    name: str
    level: str
    faculty: str
    department: str
    overview: str
    sections: Dict[str, str | List[str]]
