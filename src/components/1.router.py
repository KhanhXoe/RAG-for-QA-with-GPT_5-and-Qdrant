from abc import abstractmethod
from enum import Enum
from openai import OpenAI
from typing import Optional, Tuple
from datetime import datetime
from uuid import uuid4

from .base_component import BaseComponent
from ..config import Settings
from ..models import QueryIntent
from ..logging import StepLog

class BaseRequestRouter(BaseComponent):
    """Base class for routing user queries"""
    def __init__(self):
        super().__init__(name="router")
    
    def _execute(self, query: str) -> Tuple[QueryIntent, StepLog]:
        """Execute routing"""
        return self.route_query(query)
    
    @abstractmethod
    def route_query(self, query: str) -> Tuple[QueryIntent, StepLog]:
        """Determine the intent of the query."""
        pass

class LLMRequestRouter(BaseRequestRouter):
    def __init__(self):
        super().__init__()
        
        settings = Settings() # type: ignore
        self.client = OpenAI(api_key=settings.openai_api_key)
        self.model = settings.router_model
        
    def route_query(self, query: str) -> Tuple[QueryIntent, StepLog]:
        route_prompt = f"""You are a query router. 
        Analyze the following query and determine how it should be handled.
        Return EXACTLY ONE of these values (nothing else): ANSWER, CLARIFY, or REJECT
        
        Guidelines:
        - ANSWER: If the query is clear, specific, and can be answered with factual information
        - CLARIFY: If the query is ambiguous, vague, or needs more context
        - REJECT: If the query is inappropriate, harmful, or completely out of scope
        
        Query: {query}
        
        Decision:"""
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": route_prompt.format(query=query)}],
                temperature=0,
                max_tokens=9
            )
            decision = response.choices[0].message.content.strip().upper()            
            if decision in ["ANSWER", "CLARIFY", "REJECT"]:
                route_log = StepLog(
                    step_id=uuid4().hex,
                    step_name="route_query",
                    input={"query": query},
                    output={"decision": decision},
                    metadata={},
                    timestamp=datetime.now(),
                    duration_ms=0.0,
                    success=True,
                    error=None
                )
                return QueryIntent[decision], route_log
            else:
                route_log = StepLog(
                    step_id=uuid4().hex,
                    step_name="route_query",
                    input={"query": query},
                    output={"decision": None},
                    metadata={},
                    timestamp=datetime.now(),
                    duration_ms=0.0,
                    success=False,
                    error="Invalid response from router"
                )
                return QueryIntent.CLARIFY, route_log
        except Exception as e:
            print(f"Router error: {e}")
            route_log = StepLog(
                step_id=uuid4().hex,
                step_name="route_query",
                input={"query": query},
                output={"decision": None},
                metadata={},
                timestamp=datetime.now(),
                duration_ms=0.0,
                success=False,
                error=str(e)
            )
            return QueryIntent.ANSWER, route_log