from functools import lru_cache
from chromadb import AsyncHttpClient
from dotenv import load_dotenv
from llm.huggingface import get_huggingface_embedding
from langchain_chroma import Chroma
import os
from database.enums import ChromaCollection
from chromadb.api.types import QueryResult, IncludeEnum
from typing import Dict

load_dotenv()

class ChromaDBClient:
    def __init__(self, host: str, port: int):
        self.host = host
        self.port = port
        self.embedding_function = get_huggingface_embedding()
        self._client = None
        self._stores: Dict[str, Chroma] = {}

    async def _get_client(self):
        """Get or create AsyncHttpClient"""
        if self._client is None:
            self._client = await AsyncHttpClient(host=self.host, port=self.port)
        return self._client

    async def get_store(self, collection_name: ChromaCollection) -> Chroma:
        """Get or create Chroma store for specific collection"""
        collection_key = collection_name.value
        
        if collection_key not in self._stores:
            client = await self._get_client()
            self._stores[collection_key] = Chroma(
                client=client,
                collection_name=collection_key,
                embedding_function=self.embedding_function
            )
        
        return self._stores[collection_key]

    async def ensure_connection(self):
        """Check connection health and reinitialize if needed"""
        try:
            if self._client:
                await self._client.heartbeat()
        except Exception:
            # Reset client and stores on connection failure
            self._client = None
            self._stores.clear()
            # Will be reinitialized on next use

    async def asimilarity_search(self, 
          collection_name: ChromaCollection, 
          query: str, 
          n_results: int = 10) -> QueryResult:
        await self.ensure_connection()
        store = await self.get_store(collection_name)
        return await store.asimilarity_search(
            query=query,
            k=n_results
        )

    async def close(self):
        """Clean up resources"""
        if self._client:
            await self._client.close()
            self._client = None
        self._stores.clear()

@lru_cache(maxsize=1)
def get_chroma_client():
    host = os.getenv("CHROMA_HOST") or "localhost"
    port = os.getenv("CHROMA_PORT") or 8000
    return ChromaDBClient(host=host, port=port)
    
    