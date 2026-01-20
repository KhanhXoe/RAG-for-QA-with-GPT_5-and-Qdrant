from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple, Optional
from datetime import datetime
import uuid
from ..logging.base import StepLog

class BaseComponent(ABC):
    """Base class for all RAG workflow components"""
    def __init__(self, name: str, metadata: Optional[Dict[str, Any]] = None):
        self.name = name
        self.metadata = metadata or {}
    
    @abstractmethod
    def _execute(self, *args, **kwargs) -> Any:
        """Internal execution method to be implemented by components"""
        pass
    
    def execute(self, *args, **kwargs) -> Tuple[Any, StepLog]:
        """Execute with logging. Don't override this."""
        start_time = datetime.now()
        step_id = str(uuid.uuid4())
        
        try:
            result = self._execute(*args, **kwargs)
            duration = (datetime.now() - start_time).total_seconds() * 1000
            
            log = StepLog(
                step_id=step_id,
                step_name=self.name,
                input={"args": args, "kwargs": kwargs},
                output={"result": result},
                metadata=self.metadata,
                timestamp=start_time,
                duration_ms=duration,
                success=True
            )
            return result, log
            
        except Exception as e:
            duration = (datetime.now() - start_time).total_seconds() * 1000
            log = StepLog(
                step_id=step_id,
                step_name=self.name,
                input={"args": args, "kwargs": kwargs},
                output={},
                metadata={"error_type": e.__class__.__name__},
                timestamp=start_time,
                duration_ms=duration,
                success=False,
                error=str(e)
            )
            raise e