from dotenv import load_dotenv
from langchain_deepseek import ChatDeepSeek
from functools import lru_cache
import os
from typing import Optional

load_dotenv()

@lru_cache(maxsize=1)
def get_deepseek_llm(model: str = "deepseek-chat", tag: Optional[str] = None, **kwargs):
  API_KEY = os.getenv("DEEPSEEK_API_KEY")

  if API_KEY is None:
    raise ValueError("DEEPSEEK_API_KEY is not set")
  
  return ChatDeepSeek(
    model=model,
    temperature=0.3,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    api_key=API_KEY,
    tags=[tag],
    **kwargs
  )