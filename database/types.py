from pydantic import BaseModel
from typing import List, Optional, Dict

class Prerequisites(BaseModel):
    raw: str
    parsed: str

class Corequisites(BaseModel):
    raw: str
    parsed: str

class Restrictions(BaseModel):
    raw: str
    parsed: str  # restricted taken or taking courses

class Course(BaseModel):
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

class Program(BaseModel):
    url: str
    degree: str
    name: str
    level: str
    faculty: str
    department: str
    overview: str
    sections: Dict[str, str | List[str]]
  