from typing import List, Dict
from typing_extensions import TypedDict

class Requisites(TypedDict):
    raw: str
    parsed: str

class Course(TypedDict, total=False):
    id: str
    name: str
    credits: float  # Using float since it's a number in TypeScript
    faculty: str
    department: str
    academicLevel: int
    courseLevel: str
    terms: List[str]
    overview: str
    instructors: str
    notes: List[str]
    prerequisites: Requisites
    corequisites: Requisites
    restrictions: Requisites
    futureCourses: List[str]

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
