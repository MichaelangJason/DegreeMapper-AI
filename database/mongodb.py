from langchain_mongodb import MongoDBAtlasVectorSearch
from dotenv import load_dotenv
from functools import lru_cache
from pymongo import AsyncMongoClient, MongoClient
from typing import Dict, Optional, Any
from llm.huggingface import get_huggingface_embedding, generate_bson_vector
from .enums import MongoCollection, MongoIndex
from .types import Course
from .utils import SEARCH_WEIGHTS, RECIPROCAL_C, generate_vector_search_filter, generate_search_filter, generate_search_stage
import os
import logging
from warnings import deprecated

load_dotenv()

info_logger = logging.getLogger("uvicorn.info")

class MongoDBClient:
    _instance: Optional["MongoDBClient"] = None
    _initialized: bool = False
    _search_weights = SEARCH_WEIGHTS
    
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
        self.__class__._search_weights = SEARCH_WEIGHTS

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
    
    @deprecated("Will be moved hybrid search for all")
    async def query(self,
            collection: MongoCollection,
            query: str,
            n_results: int = 10):
        """Search for documents"""
        await self.ensure_connection(async_client=True)

        # use async client to query, get corresponding collection
        client = await self.get_async_client()
        db = client[self.database_name]
        collection = db[collection.value]
        
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

    async def hybrid_search(
            self,
            query: str,
            collection: MongoCollection,
            n_results = 3,
            *,
            filter: Dict[str, Any] = {}, # use to filter out documents by criteria
            proj: Dict[str, Any] = {}, # use to keep wanted fields
        ):
        await self.ensure_connection(async_client=True)

        client = await self.get_async_client()
        db = client[self.database_name]
        coll = db[collection.value]

        ef = get_huggingface_embedding()
        embeddings = await ef.aembed_query(query)
        embeddings = generate_bson_vector(embeddings)
        vector_weight = self._search_weights[collection][MongoIndex.VECTOR]
        full_text_weight = self._search_weights[collection][MongoIndex.FULL_TEXT]

        vector_pipeline = [
            # 1. vector search
            {
                "$vectorSearch": {
                    "index": "vector_index",
                    "path": "embeddings",
                    "queryVector": embeddings,
                    "numCandidates": 100,
                    "limit": 20,
                    "filter": generate_vector_search_filter(filter) # can be empty dict
                }
            },
            # add vectorScore
            {
                "$addFields": {
                    "vector_search_score": { "$meta": "vectorSearchScore" }
                }
            },
            # group into one doc with docs fields for ranks
            {
                "$group": {
                    "_id": None,
                    "docs": {"$push": "$$ROOT"}
                }
            },
            # unwind to get the rank of the document
            {
                "$unwind": {
                    "path": "$docs",
                    "includeArrayIndex": "rank"
                }
            },
            # compute verctor search score with reciprocal rank
            {
                "$addFields": {
                    "vs_score": { "$multiply": [ vector_weight, { "$divide": [ "$docs.vector_search_score", { "$add": [ "$rank", RECIPROCAL_C ]}]}]}
                }
            },
            # replace by original id id
            {
                "$set": {
                    "_id": "$docs._id" 
                }
            }
        ]
        

        # 2. full text search
        full_text_pipeline = [
            # $search stage, supports text
            {   
                "$search": {
                    "index": "full_text_index",
                    "compound": {
                        "should": generate_search_stage(query, collection=collection),
                        "filter": generate_search_filter(filter)
                    }
                }
            },
            # limit to 20 results
            {
                "$limit": 20
            },
            # add search score
            {
                "$addFields": {
                    "search_score": { "$meta": "searchScore" },
                }
            },
            # push
            {
                "$group": {
                    "_id": None,
                    "docs": {"$push": "$$ROOT"}
                }
            },
            # add rank field
            {
                "$unwind": {
                    "path": "$docs", 
                    "includeArrayIndex": "rank"
                }
            },
            # compute full text search score
            {
                "$addFields": {
                    "fts_score": { "$multiply": [ full_text_weight, { "$divide": [ "$docs.search_score", { "$add": [ "$rank", RECIPROCAL_C ]}]}]}
                }
            },
            # replace _id by original one
            {
                "$set": {
                    "_id": "$docs._id"
                }
            }
        ]

        # union, unpack the values and apply any projection
        pipeline = [
            # vector search results
            *vector_pipeline,
            # union with full text search
            {
                "$unionWith": {
                    "coll": collection.value,
                    "pipeline": full_text_pipeline
                }
            },
            # group 2 results
            {
                "$group": {
                    "_id": "$_id",
                    "docs": {"$first": "$docs"},
                    "vs_score": {"$max": "$vs_score"},
                    "fts_score": {"$max": "$fts_score"},
                    "vector_search_score": { "$max": "$docs.vector_search_score" },
                    "search_score": { "$max": "$docs.search_score" }
                }
            },
            # filter out null one
            {
                "$set": {
                    "vs_score": {"$ifNull": ["$vs_score", 0]},
                    "fts_score": {"$ifNull": ["$fts_score", 0]},
                    "vector_search_score": {"$ifNull": ["$vector_search_score", 0]},
                    "search_score": {"$ifNull": ["$search_score", 0]}
                }
            },
            # compute score
            {
                "$addFields": {
                    "score": { "$add": ["$vs_score", "$fts_score"] }
                }
            },
            # sort
            {
                "$sort": { "score": -1 }
            },
            # limit to n_results
            {
                "$limit": n_results
            },
            # extract other fields to result document
            {
                "$replaceRoot": {
                    "newRoot": {
                        "$mergeObjects": ["$$ROOT", "$docs"]
                    }
                }
            },
            {
                "$project": {
                    "_id": 0,
                    "embeddings": 0,
                    "docs": 0,
                    "createdAt": 0,
                    "updatedAt": 0,
                    "__v": 0
                }
            }
        ]

        if (proj): pipeline.append({ "$project": proj })

        response = await coll.aggregate(pipeline=pipeline)
        results = await response.to_list()
        
        return results

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