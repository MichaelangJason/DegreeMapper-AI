import enum

class ChromaCollection(enum.Enum):
    Faculty = "faculty"
    Course = "course"
    Program = "program"

class MongoCollection(enum.Enum):
    ChatHistory = "chat_history"
    Course = "course"
    Faculty = "faculty"
    Program = "program"
