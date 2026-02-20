from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from openai import OpenAI
import json
from .base_component import BaseComponent
from ..config import Settings
from ..utils.cache import QueryCache

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
        self.cache = QueryCache(ttl=cache_ttl)

    def reformulate(self, query: str) -> ReformulatedQuery:
        cached = self.cache.get(query, cache_type="reformulator")
        if cached is not None:
            return cached
        
        prompt = f"""
        Given the user query, reformulate it to be more precise and extract key search terms.
        Return EXACTLY your response in this JSON format:
        {{
            "refined_query": "reformulated question",
            "keywords": ["key1", "key2", "key3"]
        }}

        ONLY return the JSON object, NO other text. The number of keywords should be between 2 and 6.

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
            
            result = json.loads(response.choices[0].message.content)
            keywords = result.get("keywords", [])
            if len(keywords) > 6:
                keywords = keywords[:6]
            elif len(keywords) < 2:
                keywords = query.split()[:6] 
            
            final_result = ReformulatedQuery(
                refined_text=result.get("refined_query", query),
                keywords=keywords
            )
            
            self.cache.set(query, final_result, cache_type="reformulator")
            return final_result
            
        except Exception as e:
            print(f"Reformulator error: {e}")
            return ReformulatedQuery(
                refined_text=query,
                keywords=query.split()[:6]
            ) 