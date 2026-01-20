"""Simple caching mechanism for queries and embeddings"""
from typing import Any, Optional, Dict
import hashlib
import json
import time
from pathlib import Path
import pickle


class QueryCache:
    """Simple file-based cache for query results and embeddings"""
    
    def __init__(self, cache_dir: str = "cache", ttl: int = 3600):
        """
        Args:
            cache_dir: Directory to store cache files
            ttl: Time to live in seconds (default 1 hour)
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.ttl = ttl
        
        # Separate subdirectories for different cache types
        self.query_cache_dir = self.cache_dir / "queries"
        self.embedding_cache_dir = self.cache_dir / "embeddings"
        
        self.query_cache_dir.mkdir(exist_ok=True)
        self.embedding_cache_dir.mkdir(exist_ok=True)
    
    def _get_cache_key(self, data: Any) -> str:
        """Generate cache key from data"""
        if isinstance(data, str):
            content = data
        else:
            content = json.dumps(data, sort_keys=True)
        return hashlib.md5(content.encode()).hexdigest()
    
    def _is_expired(self, file_path: Path) -> bool:
        """Check if cache file is expired"""
        if not file_path.exists():
            return True
        
        file_age = time.time() - file_path.stat().st_mtime
        return file_age > self.ttl
    
    def get(self, key: str, cache_type: str = "query") -> Optional[Any]:
        """Get cached value"""
        cache_dir = self.query_cache_dir if cache_type == "query" else self.embedding_cache_dir
        cache_key = self._get_cache_key(key)
        cache_file = cache_dir / f"{cache_key}.pkl"
        
        if self._is_expired(cache_file):
            return None
        
        try:
            with open(cache_file, 'rb') as f:
                return pickle.load(f)
        except Exception:
            return None
    
    def set(self, key: str, value: Any, cache_type: str = "query") -> None:
        """Set cached value"""
        cache_dir = self.query_cache_dir if cache_type == "query" else self.embedding_cache_dir
        cache_key = self._get_cache_key(key)
        cache_file = cache_dir / f"{cache_key}.pkl"
        
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(value, f)
        except Exception as e:
            print(f"Cache write error: {e}")
    
    def clear(self, cache_type: Optional[str] = None) -> None:
        """Clear cache files"""
        if cache_type == "query":
            dirs = [self.query_cache_dir]
        elif cache_type == "embedding":
            dirs = [self.embedding_cache_dir]
        else:
            dirs = [self.query_cache_dir, self.embedding_cache_dir]
        
        for cache_dir in dirs:
            for cache_file in cache_dir.glob("*.pkl"):
                cache_file.unlink()
    
    def clear_expired(self) -> int:
        """Clear expired cache files and return count"""
        count = 0
        for cache_dir in [self.query_cache_dir, self.embedding_cache_dir]:
            for cache_file in cache_dir.glob("*.pkl"):
                if self._is_expired(cache_file):
                    cache_file.unlink()
                    count += 1
        return count


class MemoryCache:
    """Simple in-memory cache with TTL"""
    
    def __init__(self, ttl: int = 3600):
        self.cache: Dict[str, tuple[Any, float]] = {}
        self.ttl = ttl
    
    def _get_cache_key(self, data: Any) -> str:
        """Generate cache key from data"""
        if isinstance(data, str):
            content = data
        else:
            content = json.dumps(data, sort_keys=True)
        return hashlib.md5(content.encode()).hexdigest()
    
    def get(self, key: str) -> Optional[Any]:
        """Get cached value"""
        cache_key = self._get_cache_key(key)
        
        if cache_key in self.cache:
            value, timestamp = self.cache[cache_key]
            if time.time() - timestamp < self.ttl:
                return value
            else:
                del self.cache[cache_key]
        
        return None
    
    def set(self, key: str, value: Any) -> None:
        """Set cached value"""
        cache_key = self._get_cache_key(key)
        self.cache[cache_key] = (value, time.time())
    
    def clear(self) -> None:
        """Clear all cache"""
        self.cache.clear()
    
    def clear_expired(self) -> int:
        """Clear expired entries"""
        now = time.time()
        expired_keys = [
            k for k, (_, timestamp) in self.cache.items()
            if now - timestamp >= self.ttl
        ]
        
        for key in expired_keys:
            del self.cache[key]
        
        return len(expired_keys)
