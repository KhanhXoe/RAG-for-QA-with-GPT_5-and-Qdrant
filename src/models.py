from dataclasses import dataclass
from typing import List, Dict, Optional
import numpy as np
from enum import Enum

@dataclass
class Document:
    text: str
    metadata: Dict = None

@dataclass
class SearchResult:
    text: str
    metadata: Dict
    score: float

@dataclass
class Citation:
    text: str
    metadata: Dict
    relevance_score: float

@dataclass
class RAGResponse:
    answer: str
    citations: List[Citation]
    confidence_score: float 

class QueryIntent(Enum):
    ANSWER = "answer"    
    CLARIFY = "clarify"
    REJECT = "reject"