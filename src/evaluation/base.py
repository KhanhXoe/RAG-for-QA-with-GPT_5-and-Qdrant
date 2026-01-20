from abc import ABC, abstractmethod
from typing import Dict, Any
from dataclasses import dataclass

@dataclass
class EvaluationResult:
    score: float  # 0-1 score
    feedback: str  # Detailed feedback
    metadata: Dict[str, Any]  # Additional evaluation metadata

class BaseEvaluator(ABC):
    @abstractmethod
    async def evaluate(self, 
                      component_name: str,
                      input_data: Dict[str, Any],
                      output_data: Dict[str, Any]) -> EvaluationResult:
        """Evaluate component execution"""
        pass 