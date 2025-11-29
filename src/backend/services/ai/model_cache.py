"""
Model caching and memory management.

Provides caching functionality to avoid reloading models and
manage memory efficiently across model operations.
"""

import gc
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import torch

from backend.config.logging import get_logger

logger = get_logger(__name__)


@dataclass
class CachedModel:
    """Information about a cached model."""

    name: str
    model: Any
    model_type: str
    device: str
    loaded_at: float = field(default_factory=time.time)
    last_used: float = field(default_factory=time.time)
    memory_mb: float = 0.0
    use_count: int = 0

    def update_usage(self) -> None:
        """Update last used time and increment use count."""
        self.last_used = time.time()
        self.use_count += 1


class ModelCache:
    """
    LRU cache for loaded AI models.

    Provides:
    - Automatic memory management with configurable limits
    - LRU eviction when memory is constrained
    - Model lifecycle tracking
    """

    def __init__(
        self,
        max_memory_gb: float = 24.0,
        max_models: int = 10,
        eviction_threshold: float = 0.9,
    ):
        """
        Initialize the model cache.

        Args:
            max_memory_gb: Maximum memory to use for cached models
            max_models: Maximum number of models to keep in cache
            eviction_threshold: Memory usage ratio that triggers eviction
        """
        self.max_memory_gb = max_memory_gb
        self.max_models = max_models
        self.eviction_threshold = eviction_threshold
        self._cache: OrderedDict[str, CachedModel] = OrderedDict()
        self._memory_used_mb: float = 0.0

    def get(self, model_name: str) -> Optional[Any]:
        """
        Get a model from cache.

        Args:
            model_name: Name of the model

        Returns:
            Cached model or None if not found
        """
        if model_name not in self._cache:
            return None

        cached = self._cache[model_name]
        cached.update_usage()
        # Move to end (most recently used)
        self._cache.move_to_end(model_name)

        logger.debug(
            f"Cache hit for model '{model_name}' (use count: {cached.use_count})"
        )
        return cached.model

    def put(
        self,
        model_name: str,
        model: Any,
        model_type: str,
        device: str,
        memory_mb: float = 0.0,
    ) -> None:
        """
        Put a model in the cache.

        Args:
            model_name: Name of the model
            model: The model object
            model_type: Type of model (text, image, etc.)
            device: Device the model is loaded on
            memory_mb: Estimated memory usage in MB
        """
        # Check if we need to evict models
        self._ensure_capacity(memory_mb)

        # Remove existing entry if present
        if model_name in self._cache:
            self.remove(model_name)

        cached = CachedModel(
            name=model_name,
            model=model,
            model_type=model_type,
            device=device,
            memory_mb=memory_mb,
        )

        self._cache[model_name] = cached
        self._memory_used_mb += memory_mb

        logger.debug(
            f"Cached model '{model_name}' ({model_type}, {memory_mb:.0f}MB on {device})"
        )

    def remove(self, model_name: str) -> bool:
        """
        Remove a model from cache.

        Args:
            model_name: Name of the model

        Returns:
            True if model was removed
        """
        if model_name not in self._cache:
            return False

        cached = self._cache.pop(model_name)
        self._memory_used_mb -= cached.memory_mb

        # Clean up model resources
        self._cleanup_model(cached)

        logger.debug(f"Removed model '{model_name}' from cache")
        return True

    def _cleanup_model(self, cached: CachedModel) -> None:
        """Clean up model resources."""
        try:
            # Try to move model to CPU first to free GPU memory
            if hasattr(cached.model, "to") and "cuda" in cached.device:
                cached.model.to("cpu")

            # Delete the model
            del cached.model

            # Force garbage collection
            gc.collect()

            # Clear CUDA cache if available
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        except Exception as e:
            logger.warning(f"Error cleaning up model '{cached.name}': {e}")

    def _ensure_capacity(self, required_mb: float) -> None:
        """
        Ensure there's enough capacity for a new model.

        Args:
            required_mb: Memory required for new model
        """
        max_memory_mb = self.max_memory_gb * 1024
        threshold_mb = max_memory_mb * self.eviction_threshold

        # Check if we need to evict based on memory
        while self._memory_used_mb + required_mb > threshold_mb and self._cache:
            self._evict_lru()

        # Check if we need to evict based on model count
        while len(self._cache) >= self.max_models and self._cache:
            self._evict_lru()

    def _evict_lru(self) -> None:
        """Evict the least recently used model."""
        if not self._cache:
            return

        # Get the first item (least recently used)
        lru_name = next(iter(self._cache))
        logger.info(f"Evicting LRU model '{lru_name}' from cache")
        self.remove(lru_name)

    def clear(self) -> None:
        """Clear all models from cache."""
        model_names = list(self._cache.keys())
        for name in model_names:
            self.remove(name)

        self._memory_used_mb = 0.0
        logger.info("Cleared model cache")

    def contains(self, model_name: str) -> bool:
        """Check if a model is in the cache."""
        return model_name in self._cache

    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.

        Returns:
            Dictionary with cache statistics
        """
        return {
            "cached_models": len(self._cache),
            "max_models": self.max_models,
            "memory_used_mb": self._memory_used_mb,
            "max_memory_gb": self.max_memory_gb,
            "memory_usage_percent": (
                (self._memory_used_mb / (self.max_memory_gb * 1024)) * 100
                if self.max_memory_gb > 0
                else 0
            ),
            "models": [
                {
                    "name": c.name,
                    "type": c.model_type,
                    "device": c.device,
                    "memory_mb": c.memory_mb,
                    "use_count": c.use_count,
                    "age_seconds": time.time() - c.loaded_at,
                }
                for c in self._cache.values()
            ],
        }

    def get_model_info(self, model_name: str) -> Optional[Dict[str, Any]]:
        """
        Get information about a cached model.

        Args:
            model_name: Name of the model

        Returns:
            Model information or None if not found
        """
        if model_name not in self._cache:
            return None

        cached = self._cache[model_name]
        return {
            "name": cached.name,
            "type": cached.model_type,
            "device": cached.device,
            "memory_mb": cached.memory_mb,
            "loaded_at": cached.loaded_at,
            "last_used": cached.last_used,
            "use_count": cached.use_count,
        }


# Global model cache instance
_model_cache: Optional[ModelCache] = None


def get_model_cache(
    max_memory_gb: float = 24.0,
    max_models: int = 10,
) -> ModelCache:
    """
    Get the global model cache instance.

    Args:
        max_memory_gb: Maximum memory to use for cached models
        max_models: Maximum number of models to keep in cache

    Returns:
        ModelCache singleton instance
    """
    global _model_cache
    if _model_cache is None:
        _model_cache = ModelCache(
            max_memory_gb=max_memory_gb,
            max_models=max_models,
        )
    return _model_cache
