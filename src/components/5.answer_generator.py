from abc import abstractmethod
from typing import List, Dict, Any
from openai import OpenAI
from .base_component import BaseComponent
from ..config import Settings
from ..models import RAGResponse, Citation, SearchResult
import json

class BaseAnswerGenerator(BaseComponent):
    """Base class for generating answers from context"""
    def __init__(self):
        super().__init__(name="answer_generator")
    
    def _execute(self, query: str, context: List[Dict[str, Any]]) -> RAGResponse:
        """Execute answer generation"""
        return self.generate_answer(query, context)
    
    @abstractmethod
    def generate_answer(self, query: str, context: List[Dict[str, Any]]) -> RAGResponse:
        """Generate an answer using the retrieved context."""
        pass

class LLMAnswerGenerator(BaseAnswerGenerator):
    def __init__(self, model: str = "gpt-4-turbo-preview"):
        super().__init__()
        try:
            settings = Settings()
        except Exception:
            from ..config import Settings as SettingsClass
            settings = SettingsClass(_env_file="")
        self.client = OpenAI(api_key=settings.openai_api_key)
        self.model = model
    
    def generate_answer(self, query: str, context: List[SearchResult]) -> RAGResponse:
        # Format context for the prompt
        formatted_context = "\n\n".join([
            f"Context {i+1}:\n{result.text}"
            for i, result in enumerate(context)
        ])
        
        prompt = """Using the provided context, answer the question. Your response must be in JSON format with these fields:
        1. "answer": Your detailed response
        2. "citations": A list of objects, each with:
           - "text": The relevant quote from the context
           - "relevance_score": A float between 0-1 indicating how relevant this citation is
        3. "confidence_score": A float between 0-1 indicating your overall confidence
        
        Only use information from the provided context. If you're unsure, reflect that in the confidence score.
        
        Context:
        {context}
        
        Question: {query}
        
        Respond with only the JSON object, no other text."""
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{
                    "role": "user", 
                    "content": prompt.format(context=formatted_context, query=query)
                }],
                temperature=0,
                max_tokens=2000,  # Increased for longer answers
                response_format={ "type": "json_object" }
            )
            
            result = response.choices[0].message.content
            parsed = json.loads(result)
            
            # Match citations with original context metadata
            citations = []
            for i, cite in enumerate(parsed.get("citations", [])):
                # Find matching context
                matching_context = None
                for ctx in context:
                    if cite["text"] in ctx.text:
                        matching_context = ctx
                        break
                
                citations.append(Citation(
                    text=cite["text"],
                    metadata=matching_context.metadata if matching_context else {},
                    relevance_score=cite.get("relevance_score", 0.5)
                ))
            
            return RAGResponse(
                answer=parsed.get("answer", "Unable to generate answer."),
                citations=citations,
                confidence_score=float(parsed.get("confidence_score", 0.5))
            )
        except Exception as e:
            print(f"Answer generation error: {e}")
            # Return error response
            return RAGResponse(
                answer=f"Sorry, I encountered an error generating the answer: {str(e)}",
                citations=[],
                confidence_score=0.0
            ) 