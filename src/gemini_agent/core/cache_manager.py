import hashlib
import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, Optional, List
import diskcache

logger = logging.getLogger(__name__)


class CacheManager:
    """
    Unified caching system for embeddings, LLM responses, tool results, and state.
    Uses diskcache for persistent, process-safe storage.
    """

    def __init__(self, cache_dir: Path):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Initialize specialized caches
        self.embedding_cache = diskcache.Cache(str(self.cache_dir / "embeddings"))
        self.llm_cache = diskcache.Cache(str(self.cache_dir / "llm_responses"))
        self.tool_cache = diskcache.Cache(str(self.cache_dir / "tool_results"))
        self.state_cache = diskcache.Cache(str(self.cache_dir / "state"))

    def _generate_key(self, *args: Any) -> str:
        """Generates a stable MD5 hash for a set of arguments."""
        hasher = hashlib.md5()
        for arg in args:
            if isinstance(arg, (dict, list)):
                hasher.update(json.dumps(arg, sort_keys=True).encode("utf-8"))
            else:
                hasher.update(str(arg).encode("utf-8"))
        return hasher.hexdigest()

    # --- Embedding Cache ---
    def get_embedding(self, text: str) -> Optional[List[float]]:
        key = self._generate_key(text)
        return self.embedding_cache.get(key)

    def set_embedding(self, text: str, embedding: List[float]):
        key = self._generate_key(text)
        self.embedding_cache.set(key, embedding)

    # --- LLM Response Cache ---
    def get_llm_response(self, prompt_data: Dict[str, Any]) -> Optional[Any]:
        key = self._generate_key(prompt_data)
        return self.llm_cache.get(key)

    def set_llm_response(
        self, prompt_data: Dict[str, Any], response: Any, expire: int = 3600
    ):
        key = self._generate_key(prompt_data)
        self.llm_cache.set(key, response, expire=expire)

    # --- Tool Result Cache ---
    def get_tool_result(self, tool_name: str, args: Dict[str, Any]) -> Optional[Any]:
        key = self._generate_key(tool_name, args)
        return self.tool_cache.get(key)

    def set_tool_result(
        self, tool_name: str, args: Dict[str, Any], result: Any, expire: int = 300
    ):
        key = self._generate_key(tool_name, args)
        self.tool_cache.set(key, result, expire=expire)

    # --- State & Preference Cache ---
    def get_state(self, key: str) -> Optional[Any]:
        return self.state_cache.get(key)

    def set_state(self, key: str, value: Any, expire: Optional[int] = None):
        self.state_cache.set(key, value, expire=expire)

    def clear_all(self):
        """Clears all caches."""
        self.embedding_cache.clear()
        self.llm_cache.clear()
        self.tool_cache.clear()
        self.state_cache.clear()
        logger.info("All caches cleared.")

    def close(self):
        """Closes all cache databases."""
        self.embedding_cache.close()
        self.llm_cache.close()
        self.tool_cache.close()
        self.state_cache.close()
