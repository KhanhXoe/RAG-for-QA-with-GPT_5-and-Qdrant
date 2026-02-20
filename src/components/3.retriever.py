from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any
from qdrant_client import QdrantClient
from qdrant_client.http import models as rest
from qdrant_client.http.models import Filter, FieldCondition, MatchText
from openai import OpenAI
import numpy as np
import uuid
from ..models import SearchResult, Document
from ..config import Settings
from .base_component import BaseComponent
from ..utils.cache import MemoryCache

class BaseRetriever(BaseComponent):
    """Base class for retrieving relevant context"""
    def __init__(self):
        super().__init__(name="retriever")
    
    def _execute(self, query: str, keywords: List[str]) -> List[SearchResult]:
        return self.retrieve(query, keywords)
    
    @abstractmethod
    def retrieve(self, query: str, keywords: List[str]) -> List[SearchResult]:
        pass

class VectorRetriever(BaseRetriever):
    def __init__(
        self,
        collection_name: str,
        embedding_model: str = "text-embedding-3-large",
        url: Optional[str] = None,
        cache_ttl: int = 3600
    ):
        super().__init__()
        settings = Settings()
        self.db_client = QdrantClient(url=url)
        self.openai_client = OpenAI(api_key=settings.openai_api_key)
        self.embedding_model = embedding_model
        self.collection_name = collection_name
        self.embedding_cache = MemoryCache(ttl=cache_ttl)
        
        try:
            self.db_client.get_collection(collection_name)
        except:
            self.db_client.create_collection(
                collection_name=collection_name,
                vectors_config=rest.VectorParams(
                    size=1536,  # Embedding dimension
                    distance=rest.Distance.COSINE
                )
            )
    
    def retrieve(self, query: str, keywords: List[str]) -> List[SearchResult]:
        """Combine semantic and keyword search results."""
        semantic_results =  self.semantic_search(query)
        keyword_results =  self.keyword_search(keywords)
        return self.rerank(semantic_results, keyword_results)
    
    def add_documents(self, documents: List[Document]) -> None:
        embeddings = [self._get_embedding(doc.text) for doc in documents]
        # Prepare points for Qdrant
        points = [
            rest.PointStruct(
                id=str(uuid.uuid4()),
                vector=embedding.tolist(),
                payload={
                    "text": doc.text,
                    **(doc.metadata or {})
                }
            )
            for doc, embedding in zip(documents, embeddings)
        ]
        
        self.db_client.upsert(
            collection_name=self.collection_name,
            points=points
        )
    
    def semantic_search(self, query: str, top_k: int = 5) -> List[SearchResult]:
        query_vector = self._get_embedding(query)
        
        results = self.db_client.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            limit=top_k
        )
        
        return [
            SearchResult(
                text=hit.payload["text"],
                metadata={k: v for k, v in hit.payload.items() if k != "text"},
                score=hit.score
            )
            for hit in results
        ]
    
    def keyword_search(self, keywords: List[str], top_k: int = 5) -> List[SearchResult]:
        keyword_conditions = [
            FieldCondition(
                key="text",
                match=MatchText(text=keyword)
            )
            for keyword in keywords
        ]
        
        results = (self.db_client.scroll(
            collection_name=self.collection_name,
            scroll_filter=Filter(
                should=keyword_conditions
            ),
            limit=top_k
        ))[0]
        
        return [
            SearchResult(
                text=point.payload.get("text", "") if point.payload else "",
                metadata={k: v for k, v in point.payload.items() if k != "text"} if point.payload else {},
                score=1.0
            )
            for point in results
            if point.payload
        ]
    
    def _get_embedding(self, text: str) -> np.ndarray:
        cached = self.embedding_cache.get(text)
        if cached is not None:
            return cached
        
        response = self.openai_client.embeddings.create(
            model=self.embedding_model,
            input=text
        )
        embedding = np.array(response.data[0].embedding)
        
        self.embedding_cache.set(text, embedding)
        return embedding
    
    def rerank(self, semantic_results: List[SearchResult], 
                      keyword_results: List[SearchResult]) -> List[SearchResult]:
        """Merge results using weighted scoring approach."""
        merged = {}
        semantic_weight = 0.7
        keyword_weight = 0.3
        
        for result in semantic_results:
            merged[result.text] = SearchResult(
                text=result.text,
                metadata=result.metadata,
                score=result.score * semantic_weight
            )
        
        for result in keyword_results:
            if result.text in merged:
                merged[result.text].score += result.score * keyword_weight
            else:
                merged[result.text] = SearchResult(
                    text=result.text,
                    metadata=result.metadata,
                    score=result.score * keyword_weight
                )
        
        return sorted(merged.values(), key=lambda x: x.score, reverse=True) 
