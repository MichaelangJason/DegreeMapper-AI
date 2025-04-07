from enum import Enum

class Model(Enum):
    OPENAI = "openai"
    HUGGINGFACE = "huggingface"
    DEEPSEEK = "deepseek"

class Node(Enum):
    CONTEXT_MANAGER = "context_manager"
    PERSONA_RESPONDER = "persona_responder"
    EXECUTION_HANDLER = "execution_handler"
    INTERACTIVE_QUERY = "interactive_query"