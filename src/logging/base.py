from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from datetime import datetime
import json
import uuid

@dataclass
class StepLog:
    step_id: str
    step_name: str
    input: Dict[str, Any]
    output: Dict[str, Any]
    metadata: Dict[str, Any]
    timestamp: datetime
    duration_ms: float
    success: bool
    error: Optional[str] = None

@dataclass
class WorkflowLog:
    workflow_id: str
    query: str
    step_ids: List[str]
    start_time: datetime
    end_time: datetime
    success: bool
    final_response: Optional[Dict[str, Any]] = None

class BaseLogger(ABC):
    @abstractmethod
    def log_step(self, step_log: StepLog) -> None:
        """Log a single step in the workflow"""
        pass
    
    @abstractmethod
    def log_workflow(self, workflow_log: WorkflowLog) -> None:
        """Log the entire workflow completion"""
        pass
    
    @abstractmethod
    def get_workflow_logs(
        self, 
        workflow_id: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None) -> List[WorkflowLog]:
        """Retrieve workflow logs with optional filtering"""
        pass