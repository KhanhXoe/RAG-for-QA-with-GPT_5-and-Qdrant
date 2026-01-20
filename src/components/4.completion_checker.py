from abc import abstractmethod
from openai import OpenAI
from typing import List, Dict, Any
import json
from .base_component import BaseComponent
from ..config import Settings
from ..models import SearchResult

class BaseCompletionChecker(BaseComponent):
    """Base class for checking if query can be answered with context"""
    def __init__(self):
        super().__init__(name="completion_checker")
    
    def _execute(self, query: str, context: List[Dict[str, Any]]) -> float:
        """Execute completion check"""
        return self.check_completion(query, context)
    
    @abstractmethod
    def check_completion(self, query: str, context: List[Dict[str, Any]]) -> float:
        """Check if the query can be answered with the given context."""
        pass

class LLMCompletionChecker(BaseCompletionChecker):
    def __init__(self, model: str = "gpt-4-turbo-preview"):
        super().__init__()
        try:
            settings = Settings()
        except Exception:
            from ..config import Settings as SettingsClass
            settings = SettingsClass(_env_file="")
        self.client = OpenAI(api_key=settings.openai_api_key)
        self.model = model
    
    def check_completion(self, query: str, context: List[SearchResult]) -> float:
        formatted_context = "\n\n".join([
            f"Context {i+1}:\n{result.text}"
            for i, result in enumerate(context)
        ])
        
        prompt = """Analyze if the given context contains sufficient information to answer the query.
        Return your response in JSON format:
        {{
            "score": float between 0.0 and 1.0,
            "reasoning": "brief explanation"
        }}
        
        Score guidelines:
        - 1.0: context perfectly contains all needed information
        - 0.0: context has no relevant information
        - Values in between: partial information
        
        Consider:
        - Relevance of context to the query
        - Completeness of information
        - Reliability of information
        
        Context:
        {context}
        
        Query: {query}"""
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{
                    "role": "user", 
                    "content": prompt.format(context=formatted_context, query=query)
                }],
                temperature=0,
                max_tokens=150,
                response_format={ "type": "json_object" }
            )
            
            result = json.loads(response.choices[0].message.content) # type: ignore
            score = float(result.get("score", 0.0))
            reasoning = result.get("reasoning", "")
            
            # Log reasoning for debugging
            if score < 0.5:
                print(f"Low completion score ({score}): {reasoning}")
            
            return max(0.0, min(1.0, score))  # Clamp to [0, 1]
        except Exception as e:
            print(f"Completion checker error: {e}")
            return 0.5  # Default to middle value on error 