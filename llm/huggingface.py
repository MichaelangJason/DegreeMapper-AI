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

# monkey patch the cleanup method to the HuggingFaceEmbeddings class
def _cleanup_embedding_model(self: HuggingFaceEmbeddings):
    """
    Custom cleanup method for HuggingFaceEmbeddings.
    Inspects the SentenceTransformers object to determine the appropriate cleanup strategy.
    """
    try:
        # Check if we have access to the SentenceTransformers client
        if not hasattr(self, "_client"):
            info_logger.warning("No client attribute found on HuggingFaceEmbeddings")
            return False
            
        # Get the SentenceTransformers model
        st_model = self._client
        
        # Get device information
        device = getattr(st_model, "device", None)
        info_logger.info(f"SentenceTransformers model is on device: {device}")
        
        # Clean up based on device
        if str(device).startswith("cuda"):
            # GPU cleanup
            info_logger.info("Performing CUDA cleanup")
            try:
                import torch
                # Clear CUDA cache
                torch.cuda.empty_cache()
                # Move model to CPU to free GPU memory
                for param in st_model.parameters():
                    param.data = param.data.cpu()
                    if param.grad is not None:
                        param.grad.data = param.grad.data.cpu()
            except Exception as e:
                error_logger.error(f"Error during CUDA cleanup: {e}")
        
        # Clean up thread pool if it exists (common in SentenceTransformers)
        if hasattr(st_model, "_pool"):
            info_logger.info("Shutting down thread pool")
            st_model._pool.shutdown(wait=True)
        
        # Check for tokenizer cleanup
        if hasattr(st_model, "tokenizer"):
            info_logger.info("Cleaning up tokenizer")
            del st_model.tokenizer
        
        # Clean up any loky semaphores that might have been created
        try:
            import os
            import glob
            semaphore_pattern = "/dev/shm/loky-*"
            semaphore_files = glob.glob(semaphore_pattern)
            if semaphore_files:
                info_logger.info(f"Cleaning up {len(semaphore_files)} loky semaphore files")
                for semaphore_file in semaphore_files:
                    try:
                        os.unlink(semaphore_file)
                    except Exception as e:
                        error_logger.error(f"Error removing semaphore file {semaphore_file}: {e}")
        except Exception as e:
            error_logger.error(f"Error during loky semaphore cleanup: {e}")
        
        # Force garbage collection
        import gc
        gc.collect()
        
        info_logger.info("HuggingFaceEmbeddings cleanup completed successfully")
        return True
        
    except Exception as e:
        error_logger.error(f"Error during HuggingFaceEmbeddings cleanup: {e}")
        return False

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

    # monkey patch the cleanup method to the HuggingFaceEmbeddings class
    # embedding.__class__.cleanup = _cleanup_embedding_model
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
