from langchain_huggingface import HuggingFaceEmbeddings, ChatHuggingFace, HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TextStreamer, pipeline
from llm.enums import HF_LLM, HF_EMBEDDING
import os
from functools import lru_cache
import logging
from bson.binary import Binary, BinaryVectorDtype

info_logger = logging.getLogger("uvicorn.info")
error_logger = logging.getLogger("uvicorn.error")
warning_logger = logging.getLogger("uvicorn.warning")

# since local LLM and embedding model are resource heavy and does not support async/parallel processing, we will use a singleton pattern to ensure that the same instance is used across the app
@lru_cache(maxsize=2)
def get_huggingface_embedding(model: HF_EMBEDDING = HF_EMBEDDING.BGE):
    """Get singleton instance of HuggingFace embedding model"""
    device = os.getenv("EMBEDDING_DEVICE") or "cpu"
    model = os.getenv("EMBEDDING_MODEL") or "BAAI/bge-m3"
    max_length = int(os.getenv("EMBEDDING_MAX_LENGTH") or 8192)
    truncation = bool(os.getenv("EMBEDDING_TRUNCATION") or True)

    info_logger.info(f"Loading embedding model {model} on {device}")

    embedding = HuggingFaceEmbeddings(
        model_name=model,
        model_kwargs={"device": device},
        encode_kwargs={
            "convert_to_tensor": True, 
            "max_length": max_length, 
            "truncation": truncation
        }
    )

    return embedding

def generate_bson_vector(vector, vector_dtype=BinaryVectorDtype.FLOAT32):
    return Binary.from_vector(vector, vector_dtype)

@lru_cache(maxsize=1)
def get_huggingface_llm():
    """Get singleton instance of HuggingFace LLM"""
    model_name = os.getenv("LOCAL_LLM_MODEL") or HF_LLM.DEEPSEEK.value
    
    # Initialize tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=BitsAndBytesConfig(load_in_8bit=True),
        torch_dtype="auto",
    )
    
    # Setup pipeline
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        streamer=TextStreamer(tokenizer),
        max_new_tokens=2096,
        do_sample=True,
        temperature=0.1,  # slightly random
        top_p=0.95,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
    )

    # Create LangChain wrapper
    llm = HuggingFacePipeline(pipeline=pipe)
    return ChatHuggingFace(llm=llm)
