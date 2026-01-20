from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime
import uuid
from ..logging.base import WorkflowLog, StepLog

class BaseWorkflow(ABC):
    def __init__(self, name: str, metadata: Optional[Dict[str, Any]] = None):
        self.name = name
        self.metadata = metadata or {}
    
    @abstractmethod
    def _execute(self, *args, **kwargs) -> Tuple[Any, List[StepLog]]:
        """Execute workflow and return (result, step_logs)"""
        pass
    
    def execute(self, *args, **kwargs) -> Tuple[Any, WorkflowLog]:
        """Execute with logging"""
        workflow_id = str(uuid.uuid4())
        start_time = datetime.now()
        
        try:
            result, step_logs = self._execute(*args, **kwargs)
            workflow_log = WorkflowLog(
                workflow_id=workflow_id,
                query=args[0] if args else "",  # Assuming first arg is query
                step_ids=[log.step_id for log in step_logs],
                start_time=start_time,
                end_time=datetime.now(),
                success=True,
                final_response=result if result else None
            )
            return result, workflow_log
            
        except Exception as e:
            workflow_log = WorkflowLog(
                workflow_id=workflow_id,
                query=args[0] if args else "",
                step_ids=[],
                start_time=start_time,
                end_time=datetime.now(),
                success=False,
                final_response=None
            )
            raise e 