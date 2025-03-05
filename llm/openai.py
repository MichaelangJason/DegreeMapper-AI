from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from functools import lru_cache
import os
load_dotenv()

# cache only one here since Embedding model is using cpu
@lru_cache(maxsize=1)
def get_openai_llm(model: str = "gpt-4o"):
  API_KEY = os.getenv("OPENAI_API_KEY")

  if API_KEY is None:
    raise ValueError("OPENAI_API_KEY is not set")

  return ChatOpenAI(
      model=model,
      temperature=0,
      max_tokens=None,
      timeout=None,
      max_retries=2,
      api_key=API_KEY,
  )

