import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional, Dict, Any
from enum import Enum
from dataclasses import asdict, is_dataclass
import gzip
import shutil
from .base import BaseLogger, StepLog, WorkflowLog

class LoggingEncoder(json.JSONEncoder):
    def default(self, o):
        if is_dataclass(o) and not isinstance(o, type):
            return asdict(o)
        if isinstance(o, Enum):
            return o.value
        if isinstance(o, datetime):
            return o.isoformat()
        return super().default(o)

class JsonLogger(BaseLogger):
    def __init__(
        self, 
        log_dir: str = "logs", 
        max_log_size_mb: int = 100,
        retention_days: int = 3,
        auto_cleanup: bool = True
    ):
        self.log_dir = Path(log_dir)
        self.step_dir = self.log_dir / "steps"
        self.workflow_dir = self.log_dir / "workflows"
        self.archive_dir = self.log_dir / "archive"
        self.max_log_size = max_log_size_mb * 1024 * 1024  # Convert to bytes
        self.retention_days = retention_days
        
        # Create directories if they don't exist
        self.step_dir.mkdir(parents=True, exist_ok=True)
        self.workflow_dir.mkdir(parents=True, exist_ok=True)
        self.archive_dir.mkdir(parents=True, exist_ok=True)
        
        # Auto cleanup old logs on init
        if auto_cleanup:
            self.cleanup_old_logs()
    
    def cleanup_old_logs(self) -> Dict[str, int]:
        cutoff_time = datetime.now() - timedelta(days=self.retention_days)
        deleted_counts = {"steps": 0, "workflows": 0, "archives": 0}
        
        # Cleanup step logs
        for log_file in self.step_dir.glob("*.json"):
            if self._is_file_older_than(log_file, cutoff_time):
                log_file.unlink()
                deleted_counts["steps"] += 1
        
        # Cleanup workflow logs
        for log_file in self.workflow_dir.glob("*.json"):
            if self._is_file_older_than(log_file, cutoff_time):
                log_file.unlink()
                deleted_counts["workflows"] += 1
        
        # Cleanup archived logs
        for archive_file in self.archive_dir.glob("*.json.gz"):
            if self._is_file_older_than(archive_file, cutoff_time):
                archive_file.unlink()
                deleted_counts["archives"] += 1
        
        total = sum(deleted_counts.values())
        if total > 0:
            print(f"[JsonLogger] Cleaned up {total} old logs: {deleted_counts}")
        
        return deleted_counts
    
    def _is_file_older_than(self, file_path: Path, cutoff_time: datetime) -> bool:
        """Check if file modification time is older than cutoff"""
        try:
            file_mtime = datetime.fromtimestamp(file_path.stat().st_mtime)
            return file_mtime < cutoff_time
        except Exception:
            return False
    
    def _rotate_if_needed(self, file_path: Path):
        """Rotate log file if it exceeds size limit"""
        if file_path.exists() and file_path.stat().st_size > self.max_log_size:
            # Archive old file
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            archive_path = self.archive_dir / f"{file_path.stem}_{timestamp}.json.gz"
            
            with open(file_path, 'rb') as f_in:
                with gzip.open(archive_path, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
            
            # Remove original
            file_path.unlink()
    
    def log_step(self, step_log: StepLog) -> None:
        """Log a single step to a JSON file"""
        step_data = {
            "step_id": step_log.step_id,
            "step_name": step_log.step_name,
            "input": step_log.input,
            "output": step_log.output,
            "metadata": step_log.metadata,
            "timestamp": step_log.timestamp.isoformat(),
            "duration_ms": step_log.duration_ms,
            "success": step_log.success,
            "error": step_log.error
        }
        
        file_path = self.step_dir / f"{step_log.step_id}.json"
        self._rotate_if_needed(file_path)
        
        with open(file_path, 'w') as f:
            json.dump(step_data, f, indent=2, cls=LoggingEncoder)
    
    def log_workflow(self, workflow_log: WorkflowLog) -> None:
        """Log the entire workflow completion to a JSON file"""
        workflow_data = {
            "workflow_id": workflow_log.workflow_id,
            "query": workflow_log.query,
            "step_ids": workflow_log.step_ids,
            "start_time": workflow_log.start_time.isoformat(),
            "end_time": workflow_log.end_time.isoformat(),
            "success": workflow_log.success,
            "final_response": workflow_log.final_response
        }
        
        file_path = self.workflow_dir / f"{workflow_log.workflow_id}.json"
        self._rotate_if_needed(file_path)
        
        with open(file_path, 'w') as f:
            json.dump(workflow_data, f, indent=2, cls=LoggingEncoder)
    
    def get_workflow_logs(
        self, 
        workflow_id: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None) -> List[WorkflowLog]:
        """Get workflow logs with optional filtering"""
        if workflow_id:
            workflow_files = [self.workflow_dir / f"{workflow_id}.json"]
        else:
            workflow_files = list(self.workflow_dir.glob("*.json"))
        
        workflows = []
        for wf_file in workflow_files:
            if wf_file.exists():
                try:
                    with open(wf_file) as f:
                        data = json.load(f)
                        start = datetime.fromisoformat(data["start_time"])
                        end = datetime.fromisoformat(data["end_time"])
                        if start_time and start < start_time:
                            continue
                        if end_time and end > end_time:
                            continue
                        
                        workflows.append(WorkflowLog(
                            workflow_id=data["workflow_id"],
                            query=data["query"],
                            step_ids=data["step_ids"],
                            start_time=start,
                            end_time=end,
                            success=data["success"],
                            final_response=data.get("final_response")
                        ))
                except Exception as e:
                    print(f"Error reading workflow log {wf_file}: {e}")
                    continue
        
        return workflows 