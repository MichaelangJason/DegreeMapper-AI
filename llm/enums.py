import enum

class HF_LLM(enum.Enum):
  BGE = "BAAI/bge-m3"
  DEEPSEEK = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
  QWEN = "Qwen/Qwen2.5-7B-Instruct-1M" # very large context window
  LLAMA = "meta-llama/Llama-3.1-8B"
  CHATGPT = "gpt-4o-mini"

class HF_EMBEDDING(enum.Enum):
  BGE = "BAAI/bge-m3"
  DEEPSEEK = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
  QWEN = "Qwen/Qwen2.5-7B-Instruct-1M" # very large context window
  LLAMA = "meta-llama/Llama-3.1-8B"
  CHATGPT = "gpt-4o-mini"
  REMOTE = "remote"
