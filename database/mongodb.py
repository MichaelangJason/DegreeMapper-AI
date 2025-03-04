from langchain_mongodb import MongoDBAtlasVectorSearch
from dotenv import load_dotenv
from functools import lru_cache
from pymongo import AsyncMongoClient, MongoClient
from typing import Dict, Optional
from database.enums import MongoCollection
from database.chromadb import get_huggingface_embedding
from .types import Course
import os

load_dotenv()


class MongoDBClient:
    _instance: Optional["MongoDBClient"] = None
    _initialized: bool = False
    
    @classmethod
    def get_instance(cls, uri: str = None, database_name: str = None):
        """Get or create the singleton instance of MongoDBClient"""
        if cls._instance is None:
            if uri is None:
                uri = os.getenv("MONGODB_URI")
            if database_name is None:
                database_name = os.getenv("MONGODB_DATABASE_NAME")
                
            if not all([uri, database_name]):
                raise ValueError(
                    "MONGODB_URI and MONGODB_DATABASE_NAME must be set"
                )
                
            cls._instance = cls.__new__(cls)
            cls._instance._init(uri, database_name)
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
        raise RuntimeError("Use MongoDBClient.get_instance() instead of direct instantiation")
    
    def _init(self, uri: str, database_name: str):
        """
        Internal initialization method.
        """
        self.uri = uri
        self.database_name = database_name
        self._client: MongoClient = None # for langchain vector store usage
        self._async_client: AsyncMongoClient = None # for query with atlas search
        self._stores: Dict[str, MongoDBAtlasVectorSearch] = {}
        self.__class__._initialized = True

    def get_client(self):
        """Get or create MongoClient"""
        if self._client is None:
            self._client = MongoClient(self.uri)
            # Verify connection
            self._client.admin.command('ping')
            # print("Client created")
        return self._client

    async def get_async_client(self):
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
        await self.ensure_connection()
        
        if collection_key not in self._stores:
            # print("Collection not found, creating new collection")
            client = self.get_client()
            db = client[self.database_name]
            collection = db[collection_key]

            self._stores[collection_key] = MongoDBAtlasVectorSearch(
                collection=collection,
                index_name="vector_index",
                text_key="overview", # TODO: should also apply to other collections
                embedding_key="embeddings",
                embedding=ef,
                relevance_score_fn="dotProduct"
            )
            # print("Store created")
        
        return self._stores[collection_key]

    async def ensure_connection(self, async_client: bool = False) -> bool:
        """Check connection health and reinitialize if needed"""
        try:
            if async_client:
                await self._async_client.admin.command('ping')
            else:
                self._client.admin.command('ping')
            return True
        
        except Exception as e:
            # print(e)

            if async_client:
                if self._async_client is not None:
                    await self._async_client.aclose()
                self._async_client = None
                await self.get_async_client()
            else:
                if self._client is not None:
                    self._client.close()
                self._client = None
                self._stores.clear() # only clear if the sync client connection is stale
                self.get_client()
            
            return False

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
        # cast to Course object, should support other collections as well
        if collection_name == MongoCollection.Course:
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
        await self.ensure_connection(async_client=True)

        # use async client to query
        client = await self.get_async_client()
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
            self._client.close()
            self._client = None
        if self._async_client is not None:
            await self._async_client.aclose()
            self._async_client = None
        self._stores.clear()


def get_mongodb_client():
    """
    Get the singleton instance of MongoDBClient.
    """
    return MongoDBClient.get_instance()


# will be increased with more collections being put into used
@lru_cache(maxsize=1)
def get_mongodb_vector_store(collection_name: MongoCollection):
    uri = os.getenv("MONGODB_URI")
    database_name = os.getenv("MONGODB_DATABASE_NAME")

    if not all([uri, database_name]):
        raise ValueError(
            "MONGODB_CONNECTION_STRING and MONGODB_DATABASE_NAME must be set"
        )
    
    client = MongoClient(uri)
    db = client[database_name]
    collection = db[collection_name.value]

    return MongoDBAtlasVectorSearch(
        collection=collection,
        index_name="vector_index",
        text_key="overview", # TODO: should also apply to other collections
        embedding_key="embeddings",
        embedding=get_huggingface_embedding(),
        relevance_score_fn="dotProduct"
    )

@lru_cache(maxsize=1)
def get_async_mongodb_client():
    uri = os.getenv("MONGODB_URI")

    if not uri:
        raise ValueError(
            "MONGODB_CONNECTION_STRING must be set"
        )
    
    return AsyncMongoClient(uri)