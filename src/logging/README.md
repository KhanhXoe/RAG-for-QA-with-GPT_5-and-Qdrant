# Logging System

A flexible and extensible logging system designed for RAG (Retrieval-Augmented Generation) workflows, providing structured event logging, visualization, and analysis capabilities.

## Architecture

### Logs Storage Directory Structure

logs/

├── steps/     
└── workflows/ 


### 1. Base Logging System
- `BaseLogger`: Abstract base class defining the logging interface
- `StepLog`: Dataclass for logging individual workflow steps
- `WorkflowLog`: Dataclass for logging complete workflow executions
- `JsonLogger`: Default implementation using JSON file storage


### 2. Visualization Dashboard
- Built with Streamlit
- Real-time workflow monitoring
- Performance metrics and analysis


## Usage

### Standard Logging

```python
from src.logging import JsonLogger
logger = JsonLogger()
# Log workflow execution
logger.log_workflow(workflow_log)
# Log individual step
logger.log_step(step_log)
```

### Metadata Logging

The logging system supports rich metadata through the `metadata` field in `StepLog`. You can log any relevant information to a component by accessing the component's `metadata` attribute through `self.metadata` in the component's `_execute` method.

```python
metadata = {
    # Performance metrics
    "latency_ms": 120,
    "memory_usage_mb": 256,
    
    # Component configuration
    "model_name": "gpt-3.5-turbo",
    "temperature": 0.7,
    "chunk_size": 512,
    
    # Business metrics
    "customer_id": "cust-789",
    "session_id": "sess-012",
    
    # Debug information
    "cache_hit": True,
    "num_tokens": 156
}

self.metadata = metadata
```

This will then get logged to the `metadata` field in the `StepLog` dataclass.

### Implementing Custom Logger

You can implement your own logging backend by inheriting from `BaseLogger`. For example, you could implement a logging backend that logs to a SQL database or a NoSQL database (like MongoDB) by defining the `log_step` and `log_workflow` methods. Note, to visualize the logs, you will need to implement the `get_workflow_logs` method and make updates the the viz dashboard processing of the logs as well.


## Visualization Dashboard

The visualization dashboard is built with Streamlit. It allows you to visualize the logs and analyze the performance of your workflow. viz/base.py contains the code for the dashboard. This component is also dockerized and included in the docker-compose.yml file, but can be run separately and is provided with a standalone Dockerfile and requirements.txt file.

By default, the dashboard will run on port 8501. You can access it at `http://localhost:8501`.
 


