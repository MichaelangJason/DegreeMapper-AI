from langchain_mongodb import MongoDBAtlasVectorSearch, MongoDBChatMessageHistory
from dotenv import load_dotenv
from functools import lru_cache
import os
from pymongo import AsyncMongoClient, MongoClient
from typing import Dict
from database.enums import MongoCollection, MongoVectorIndex
from database.chromadb import get_huggingface_embedding
load_dotenv()

from .types import Course

class MongoDBClient:
    def __init__(self, uri: str, database_name: str):
        self.uri = uri
        self.database_name = database_name
        self._client: MongoClient = None
        self._async_client: AsyncMongoClient = None
        self._stores: Dict[str, MongoDBAtlasVectorSearch] = {}

    async def _get_client(self):
        """Get or create MongoClient"""
        if self._client is None:
            self._client = MongoClient(self.uri)
            # Verify connection
            # await self._client.admin.command('ping')
            self._client.admin.command('ping')
            # print("Client created")
        return self._client

    async def _get_async_client(self):
        """Get or create AsyncMongoClient"""
        if self._async_client is None:
            self._async_client = AsyncMongoClient(self.uri)
            # Verify connection
            await self._async_client.admin.command('ping')
            # print("Async client created")
        return self._async_client

    async def get_store(self, collection: MongoCollection) -> MongoDBAtlasVectorSearch:
        """Get or create vector store for specific collection"""
        collection_key = collection.value
        ef = get_huggingface_embedding()
        
        if collection_key not in self._stores:
            # print("Collection not found, creating new collection")
            client = await self._get_client()
            db = client[self.database_name]
            collection = db[collection_key]

            self._stores[collection_key] = MongoDBAtlasVectorSearch(
                collection=collection,
                index_name="vector_index",
                text_key="overview",
                embedding_key="embeddings",
                embedding=ef,
                relevance_score_fn="dotProduct"
            )
            # print("Store created")
        
        return self._stores[collection_key]

    async def ensure_connection(self) -> bool:
        """Check connection health and reinitialize if needed"""
        try:
            if self._client is not None:
                # print("Client found")
                # await self._client.admin.command('ping')
                self._client.admin.command('ping')
            else:
                # print("Client not found, creating new client")
                await self.get_store(MongoCollection.Course)
            return True
        
        except Exception as e:
            # print(e)
            # Reset client and stores on connection failure
            if self._client is not None:
                # await self._client.aclose()
                self._client.close()
            self._client = None
            self._stores.clear()
            # Will be reinitialized on next use
            return False

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
        # print("start search")
        # Use the synchronous version since the async version has issues
        results =await store.asimilarity_search(
            query=query,
            k=n_results
        )
        results = [
            Course(**{
                **doc.metadata,
                "overview": doc.page_content
            })
            for doc in results
        ]
        return results
    
    async def query(self,
            collection_name: MongoCollection,
            query: str,
            n_results: int = 10):
        """Search for documents"""
        await self.ensure_connection()

        client = await self._get_async_client()
        db = client[self.database_name]
        collection = db[collection_name.value]
        
        # mongo atlas aggregation pipeline
        results = await collection.aggregate([
            {
                "$search": {
                    "index": "search_index",
                    "text": {
                        "query": query,
                        "path": "id",
                        "fuzzy": {} # use default fuzzy setting
                    }
                }
            },
            {
                "$limit": n_results
            },
            {
                "$project": {
                    "_id": 0,
                    "embeddings": 0,
                    "__v": 0,
                    "createdAt": 0,
                    "updatedAt": 0
                }
            }
        ])

        results = await results.to_list()
        # print(results)
        return [Course(**result) for result in results]

    async def insert_documents(self, 
            collection_name: MongoCollection, 
            documents: list[dict]):
        """Insert new documents"""
        await self.ensure_connection()
        store = await self.get_store(collection_name)
        return await store.aadd_documents(documents)

    async def close(self):
        """Clean up resources"""
        if self._client is not None:
            await self._client.aclose()
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
