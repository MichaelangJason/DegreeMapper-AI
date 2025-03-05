from enum import Enum

class ChromaCollection(Enum):
    Faculty = "faculty"
    Course = "course"
    Program = "program"

class MongoCollection(Enum):
    ChatHistory = "chat_history"
    Course = "courses_2024_2025"
    Faculty = "faculty"
    Program = "programs_2024_2025"

class MongoVectorIndex(Enum):
    Course = "vector_index"
    Faculty = "vector_index"
