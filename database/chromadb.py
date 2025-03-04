from functools import lru_cache
from chromadb import AsyncHttpClient, HttpClient
from dotenv import load_dotenv
from llm.huggingface import get_huggingface_embedding
from langchain_chroma import Chroma
import os
from database.enums import ChromaCollection
from chromadb.api.types import QueryResult, IncludeEnum
from typing import Dict, Optional
from async_lru import alru_cache

load_dotenv()

class ChromaDBClient:
    _instance: Optional['ChromaDBClient'] = None
    _initialized: bool = False
    
    @classmethod
    def get_instance(cls, host: str = None, port: int = None):
        """Get or create the singleton instance of ChromaDBClient"""
        # If instance doesn't exist, create it
        if cls._instance is None:
            # Get default values if not provided
            host = host or os.getenv("CHROMA_HOST") or "localhost"
            port = port or int(os.getenv("CHROMA_PORT") or 8000)
            
            cls._instance = cls.__new__(cls)
            cls._instance._init(host, port)
        # If instance exists but new parameters are provided, log a warning
        elif (host is not None or port is not None):
            import logging
            logging.warning(
                "ChromaDBClient instance already exists. "
                "Ignoring new connection parameters. "
                "Use reset_instance() first to create a new instance with different parameters."
            )
        return cls._instance
    
    @classmethod
    def reset_instance(cls):
        """Reset the singleton instance"""
        if cls._instance is not None:
            cls._instance = None
    
    def __init__(self):
        """
        Prevent direct instantiation.
        Use get_instance() class method instead.
        """
        if self._initialized:
            return
        raise RuntimeError("Use ChromaDBClient.get_instance() instead of direct instantiation")
    
    def _init(self, host: str, port: int):
        """
        Internal initialization method.
        """
        self.host = host
        self.port = port
        self.embedding_function = get_huggingface_embedding()
        # self.embedding_function = None
        self._client = None
        self._stores: Dict[str, Chroma] = {}
        self.__class__._initialized = True

    async def _get_client(self):
        """Get or create AsyncHttpClient"""
        if self._client is None:
            # self._client = await AsyncHttpClient(host=self.host, port=self.port)
            self._client = HttpClient(host=self.host, port=self.port)
        return self._client

    async def get_store(self, collection_name: ChromaCollection) -> Chroma:
        """Get or create Chroma store for specific collection"""
        collection_key = collection_name.value
        
        if collection_key not in self._stores:
            client = await self._get_client()
            # print(client)
            self._stores[collection_key] = Chroma(
                client=client,
                collection_name=collection_key,
                embedding_function=self.embedding_function
            )
        
        return self._stores[collection_key]

    async def ensure_connection(self) -> bool:
        """Check connection health and reinitialize if needed"""
        try:
            self._client.heartbeat()
            # print("Client created")
            return True
        except Exception as e:
            # print(e)
            # Reset client and stores on connection failure
            self._client = None
            self._stores.clear()
            await self._get_client()
            # Will be reinitialized on next use
            return False

    async def asimilarity_search(self, 
          collection_name: ChromaCollection, 
          query: str, 
          n_results: int = 10) -> QueryResult:
        await self.ensure_connection()
        store = await self.get_store(collection_name)
        # print(store)

        results = await store.asimilarity_search(
            query=query,
            k=n_results
        )
        # print(results)
        results = [
            {
                **doc.metadata,
                "content": doc.page_content
            }
            for doc in results
        ]
        return results

    async def close(self):
        """Clean up resources"""
        try:
            self._client.clear_system_cache()
        except Exception as e:
            print(e)
        self._client = None
        self._stores.clear()

def get_chroma_client():
    """
    Get the singleton instance of ChromaDBClient.
    This function is kept for backward compatibility.
    """
    return ChromaDBClient.get_instance()
    
    