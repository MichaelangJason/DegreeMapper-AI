from langchain_mongodb import MongoDBAtlasVectorSearch, MongoDBChatMessageHistory
from dotenv import load_dotenv
from functools import lru_cache
import os
from pymongo import AsyncMongoClient
from typing import Dict
from database.enums import MongoCollection  # Assuming you'll create this enum

load_dotenv()

class MongoDBClient:
    def __init__(self, uri: str, database_name: str):
        self.uri = uri
        self.database_name = database_name
        self._client = None
        self._stores: Dict[str, MongoDBAtlasVectorSearch] = {}

    async def _get_client(self):
        """Get or create MongoClient"""
        if self._client is None:
            self._client = await AsyncMongoClient(self.uri)
            # Verify connection
            await self._client.admin.command('ping')
        return self._client

    async def get_store(self, collection_name: MongoCollection) -> MongoDBAtlasVectorSearch:
        """Get or create vector store for specific collection"""
        collection_key = collection_name.value
        
        if collection_key not in self._stores:
            client = await self._get_client()
            self._stores[collection_key] = MongoDBAtlasVectorSearch(
                client=client,
                database_name=self.database_name,
                collection_name=collection_key
            )
        
        return self._stores[collection_key]

    async def ensure_connection(self):
        """Check connection health and reinitialize if needed"""
        try:
            if self._client:
                await self._client.admin.command('ping')
        except Exception:
            # Reset client and stores on connection failure
            if self._client:
                self._client.close()
            self._client = None
            self._stores.clear()
            # Will be reinitialized on next use

    async def get_chat_history(self, session_id: str) -> MongoDBChatMessageHistory:
        """Get chat history for a specific session"""
        await self.ensure_connection()
        return MongoDBChatMessageHistory(
            connection_string=self.uri,
            database_name=self.database_name,
            collection_name="chat_history",
            session_id=session_id
        )

    async def asimilarify_search(self, 
            collection_name: MongoCollection, 
            query: str, 
            n_results: int = 10):
        """Search for similar documents"""
        await self.ensure_connection()
        store = await self.get_store(collection_name)
        return await store.asimilarity_search(
            query=query,
            k=n_results
        )

    async def query(self,
            collection_name: MongoCollection,
            query: str,
            n_results: int = 10):
        """Search for documents"""
        await self.ensure_connection()
        store = await self.get_store(collection_name)
        return await store.query(query, n_results)

    async def insert_documents(self, 
            collection_name: MongoCollection, 
            documents: list[dict]):
        """Insert new documents"""
        await self.ensure_connection()
        store = await self.get_store(collection_name)
        return await store.aadd_documents(documents)

    async def close(self):
        """Clean up resources"""
        if self._client:
            self._client.close()
            self._client = None
        self._stores.clear()

@lru_cache(maxsize=1)
def get_mongodb_client():
    connection_string = os.getenv("MONGODB_URI")
    database_name = os.getenv("MONGODB_DATABASE_NAME")

    if not all([connection_string, database_name]):
        raise ValueError(
            "MONGODB_CONNECTION_STRING and MONGODB_DATABASE_NAME must be set"
        )
    
    return MongoDBClient(
        uri=connection_string,
        database_name=database_name
    )
