from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from openai import OpenAI
import json
from .base_component import BaseComponent
from ..config import Settings
from ..utils.cache import MemoryCache

@dataclass
class ReformulatedQuery:
    refined_text: str
    keywords: List[str]

class BaseQueryReformulator(BaseComponent, ABC):
    """Base class for query reformulation"""
    def __init__(self):
        super().__init__(name="reformulator")
    
    def _execute(self, query: str) -> ReformulatedQuery:
        """Execute reformulation"""
        return self.reformulate(query)
    
    @abstractmethod
    def reformulate(self, query: str) -> ReformulatedQuery:
        """Reformulate the query and generate keywords."""
        pass

class LLMQueryReformulator(BaseQueryReformulator):
    def __init__(self, model: str = "gpt-5-2025-08-07", cache_ttl: int = 3600):
        super().__init__()
        settings = Settings() # type: ignore
        self.client = OpenAI(api_key=settings.openai_api_key)
        self.model = model
        self.cache = MemoryCache(ttl=cache_ttl)

    def reformulate(self, query: str) -> ReformulatedQuery:
        cached = self.cache.get(query)
        if cached is not None:
            return cached
        
        prompt = f"""
        Given the user query, reformulate it to be more precise and extract key search terms.
        Return EXACTLY your response in this JSON format:
        {{
            "refined_query": "reformulated question",
            "keywords": ["key1", "key2", "key3"]
        }}

        Only return the JSON object, no other text. The number of keywords should be between 2 and 6.

        User Query: {query}"""
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{
                    "role": "user",
                    "content": prompt
                }],
                temperature=0,
                response_format={ "type": "json_object" }
            )
            
            result = json.loads(response.choices[0].message.content) 3 # type: ignore
            keywords = result.get("keywords", [])
            if len(keywords) > 6:
                keywords = keywords[:6]
            elif len(keywords) < 2:
                keywords = query.split()[:6] 
            
            return ReformulatedQuery(
                refined_text=result.get("refined_query", query),
                keywords=keywords
            )
        except Exception as e:
            print(f"Reformulator error: {e}")
            return ReformulatedQuery(
                refined_text=query,
                keywords=query.split()[:6]
            ) 