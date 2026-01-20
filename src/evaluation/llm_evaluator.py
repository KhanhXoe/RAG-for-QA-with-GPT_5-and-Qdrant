from typing import Dict, Any
from openai import OpenAI
from .base import BaseEvaluator, EvaluationResult
from ..config import Settings

class LLMEvaluator(BaseEvaluator):
    def __init__(self, model: str = "gpt-4-turbo-preview"):
        settings = Settings()
        self.client = OpenAI(api_key=settings.openai_api_key)
        self.model = model
        
        # Component-specific evaluation prompts
        self.prompts = {
            "router": """Evaluate the routing decision. Consider:
                1. Was the intent classification appropriate?
                2. Was the confidence reasonable?
                
                Input: {input}
                Output: {output}
                
                Provide:
                1. Score (0-1)
                2. Detailed feedback
                Return as JSON: {{"score": float, "feedback": "string"}}
                """,
            "reformulator": """Evaluate the query reformulation. Consider:
                1. Is the refined query clearer?
                2. Are the keywords relevant?
                
                Input: {input}
                Output: {output}
                
                Provide:
                1. Score (0-1)
                2. Detailed feedback
                Return as JSON: {{"score": float, "feedback": "string"}}
                """
            # ... other component prompts ...
        }
    
    async def evaluate(self,
                      component_name: str,
                      input_data: Dict[str, Any],
                      output_data: Dict[str, Any]) -> EvaluationResult:
        prompt = self.prompts.get(component_name, "Evaluate the component execution")
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{
                "role": "user",
                "content": prompt.format(
                    input=str(input_data),
                    output=str(output_data)
                )
            }],
            response_format={ "type": "json_object" }
        )
        
        result = response.choices[0].message.content
        return EvaluationResult(
            score=result["score"],
            feedback=result["feedback"],
            metadata={"model": self.model}
        ) 