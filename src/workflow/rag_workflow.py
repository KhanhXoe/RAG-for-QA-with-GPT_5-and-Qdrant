from typing import Optional, Tuple, List, Dict, Any
import sys
import time
from ..models import RAGResponse, QueryIntent
from ..components import (
    BaseRequestRouter, LLMRequestRouter,
    BaseQueryReformulator, LLMQueryReformulator,
    BaseRetriever, VectorRetriever,
    BaseCompletionChecker, LLMCompletionChecker,
    BaseAnswerGenerator, LLMAnswerGenerator
)
from .base import BaseWorkflow
from ..logging.base import StepLog
from ..logging.json_logger import JsonLogger

logger = JsonLogger()

class RAGWorkflow(BaseWorkflow):
    def __init__(
        self,
        router: LLMRequestRouter,
        reformulator: LLMQueryReformulator,
        retriever: VectorRetriever,
        completion_checker: LLMCompletionChecker,
        answer_generator: LLMAnswerGenerator,
        completion_threshold: float = 0.7,
        max_retries: int = 3,
        metadata: Optional[Dict[str, Any]] = None
    ):
        super().__init__(name="rag_workflow", metadata=metadata)
        self.router = router
        self.reformulator = reformulator
        self.retriever = retriever
        self.completion_checker = completion_checker
        self.answer_generator = answer_generator
        self.completion_threshold = completion_threshold
        self.max_retries = max_retries
    
    def _execute_with_retry(self, component, *args, max_retries=None, **kwargs) -> Tuple[Any, ...]:
        """Execute component with retry logic"""
        retries = max_retries if max_retries is not None else self.max_retries
        last_error: Optional[Exception] = None
        
        for attempt in range(retries):
            try:
                return component._execute(*args, **kwargs)
            except Exception as e:
                last_error = e
                if attempt < retries - 1:
                    wait_time = 2 ** attempt  # Exponential backoff
                    print(f"Retry {attempt + 1}/{retries} after {wait_time}s: {e}")
                    time.sleep(wait_time)
                else:
                    raise last_error
        raise last_error  # type: ignore
    
    def _execute(self, query: str) -> Tuple[Optional[RAGResponse], List[StepLog]]:
        step_logs: List[StepLog] = []
        
        # Route with retry
        intent, route_log = self._execute_with_retry(self.router, query)
        logger.log_step(route_log)
        step_logs.append(route_log)
        if intent != QueryIntent.ANSWER:
            return None, step_logs
        
        # Reformulate with retry
        reformulated, reform_log = self._execute_with_retry(self.reformulator, query)
        logger.log_step(reform_log)
        step_logs.append(reform_log)
        
        # Retrieve with retry
        context, retrieve_log = self._execute_with_retry(
            self.retriever,
            reformulated.refined_text,
            reformulated.keywords
        )
        step_logs.append(retrieve_log)
        logger.log_step(retrieve_log)
        
        # Check completion with retry
        completion_score, check_log = self._execute_with_retry(
            self.completion_checker, query, context
        )
        logger.log_step(check_log)
        step_logs.append(check_log)

        if completion_score < self.completion_threshold:
            return None, step_logs
        
        # Generate answer with retry
        response, generate_log = self._execute_with_retry(
            self.answer_generator, query, context
        )
        logger.log_step(generate_log)
        step_logs.append(generate_log)
        return response, step_logs 