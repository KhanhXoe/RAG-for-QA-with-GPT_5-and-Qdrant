from fastapi import FastAPI, HTTPException, Depends, Header, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import uvicorn
from datetime import datetime
from collections import defaultdict
import time
from .config import Settings
from .logging.json_logger import JsonLogger
from .components import (
    LLMRequestRouter,
    LLMQueryReformulator,
    VectorRetriever,
    LLMCompletionChecker,
    LLMAnswerGenerator
)
from .workflow import RAGWorkflow
from .models import RAGResponse, Document

app = FastAPI()

logger = JsonLogger()
security = HTTPBearer(auto_error=False)

class RateLimiter:
    def __init__(self, max_requests: int = 100, window: int = 60):
        self.max_requests = max_requests
        self.window = window  # seconds
        self.requests = defaultdict(list)
    
    def is_allowed(self, client_id: str) -> bool:
        now = time.time()
        self.requests[client_id] = [
            req_time for req_time in self.requests[client_id]
            if now - req_time < self.window
        ]
        
        if len(self.requests[client_id]) >= self.max_requests:
            return False
        
        self.requests[client_id].append(now)
        return True

rate_limiter = RateLimiter(max_requests=100, window=60)

async def verify_api_key(credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)):
    if not credentials:
        return "anonymous"
    return credentials.credentials

try:
    settings = Settings()
except Exception as e:
    print(f"Error loading settings: {e}")
    raise

retriever = VectorRetriever(
    collection_name=settings.qdrant_collection_name,
    url=settings.qdrant_url
)

router = LLMRequestRouter(model=settings.router_model)
reformulator = LLMQueryReformulator(model=settings.reformulator_model)
completion_checker = LLMCompletionChecker(model=settings.completion_model)
answer_generator = LLMAnswerGenerator(model=settings.answer_model)

workflow = RAGWorkflow(
    router=router,
    reformulator=reformulator,
    retriever=retriever,
    completion_checker=completion_checker,
    answer_generator=answer_generator,
    completion_threshold=settings.completion_threshold
)

class QueryRequest(BaseModel):
    query: str

class DocumentRequest(BaseModel):
    documents: List[Document]

@app.post("/query")
async def process_query(request: QueryRequest, api_key: str = Depends(verify_api_key)):
    """Process a query through the RAG workflow"""
    if not rate_limiter.is_allowed(api_key):
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit exceeded. Please try again later."
        )
    
    workflow_log = None
    try:
        response, workflow_log = workflow.execute(request.query)
        logger.log_workflow(workflow_log)
        
        if not response:
            raise HTTPException(
                status_code=400, 
                detail={
                    "error": "Could not process query",
                    "reason": "Query may need clarification or is out of scope",
                    "workflow_id": workflow_log.workflow_id
                }
            )
        return response
    except HTTPException:
        raise
    except Exception as e:
        if workflow_log:
            logger.log_workflow(workflow_log)
        raise HTTPException(
            status_code=500,
            detail={"error": "Internal server error", "message": str(e)}
        )

@app.post("/documents")
async def add_documents(
    request: DocumentRequest, 
    api_key: str = Depends(verify_api_key)
) -> dict:
    """Add documents to the retriever with batch processing"""
    # Rate limiting
    if not rate_limiter.is_allowed(api_key):
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit exceeded"
        )
    
    try:
        documents = [
            Document(text=doc.text, metadata=doc.metadata) 
            for doc in request.documents
        ]
        
        # Batch processing for large document sets
        batch_size = 50
        total_added = 0
        
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            retriever.add_documents(batch)
            total_added += len(batch)
        
        return {
            "status": "success", 
            "message": f"Added {total_added} documents",
            "total_documents": total_added
        }
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail={"error": "Failed to add documents", "message": str(e)}
        )
@app.get("/health")
def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}

@app.get("/logs/workflows")
async def get_workflow_logs(
    workflow_id: Optional[str] = None,
    start_time: Optional[datetime] = None,
    end_time: Optional[datetime] = None
):
    """Get workflow logs with optional filtering"""
    return logger.get_workflow_logs(workflow_id, start_time, end_time)

if __name__ == "__main__":
    uvicorn.run("src.api:app", host="0.0.0.0", port=8029, reload=True)