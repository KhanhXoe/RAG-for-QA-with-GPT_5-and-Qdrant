from typing import Dict, Any, Callable
from .base import BaseEvaluator, EvaluationResult

class ScriptEvaluator(BaseEvaluator):
    """Evaluator that uses custom Python functions for evaluation"""
    
    def __init__(self, evaluation_functions: Dict[str, Callable]):
        """
        Initialize with component-specific evaluation functions.
        Each function should take (input_data, output_data) and return (score, feedback)
        """
        self.evaluation_functions = evaluation_functions
    
    async def evaluate(self,
                      component_name: str,
                      input_data: Dict[str, Any],
                      output_data: Dict[str, Any]) -> EvaluationResult:
        if component_name not in self.evaluation_functions:
            raise ValueError(f"No evaluation function defined for {component_name}")
        
        eval_func = self.evaluation_functions[component_name]
        score, feedback = eval_func(input_data, output_data)
        
        return EvaluationResult(
            score=score,
            feedback=feedback,
            metadata={
                "evaluation_type": "script",
                "function_name": eval_func.__name__
            }
        ) 