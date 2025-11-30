"""
Redis Cache Service for Gator Platform

Provides a centralized caching layer for:
- Persona data caching
- API response caching
- Rate limiting data (Redis backend)
- Session management
- Content generation queue tracking
"""

import json
import hashlib
from typing import Any, Optional, TypeVar, Callable, Union
from functools import wraps
from uuid import UUID

from backend.config.logging import get_logger
from backend.config.settings import get_settings

logger = get_logger(__name__)

T = TypeVar("T")

# Try to import redis.asyncio (redis-py 4.2+)
try:
    import redis.asyncio as redis_async
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    redis_async = None


class CacheConfig:
    """Cache configuration constants."""

    # TTL values in seconds
    DEFAULT_TTL = 300  # 5 minutes
    SHORT_TTL = 60  # 1 minute
    MEDIUM_TTL = 600  # 10 minutes
    LONG_TTL = 3600  # 1 hour
    VERY_LONG_TTL = 86400  # 24 hours

    # Key prefixes for namespacing
    PREFIX_PERSONA = "persona:"
    PREFIX_USER = "user:"
    PREFIX_SESSION = "session:"
    PREFIX_RATE_LIMIT = "rate_limit:"
    PREFIX_GENERATION = "generation:"
    PREFIX_ACD = "acd:"
    PREFIX_API = "api:"
    PREFIX_LOCK = "lock:"


class CacheService:
    """
    Async Redis cache service with connection pooling.

    Provides methods for caching various data types with
    configurable TTL and automatic serialization.
    """

    _instance: Optional["CacheService"] = None
    _redis: Optional[Any] = None

    def __new__(cls):
        """Singleton pattern for cache service."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    async def connect(self) -> None:
        """
        Initialize Redis connection pool.

        Should be called at application startup.
        """
        if not REDIS_AVAILABLE:
            logger.warning("Redis library not available. Cache will be disabled.")
            return

        if self._redis is None:
            settings = get_settings()
            try:
                self._redis = redis_async.from_url(
                    settings.redis_url,
                    encoding="utf-8",
                    decode_responses=True,
                )
                # Test connection
                await self._redis.ping()
                logger.info("Redis cache connected successfully")
            except Exception as e:
                logger.warning(f"Redis connection failed: {e}. Cache will be disabled.")
                self._redis = None

    async def disconnect(self) -> None:
        """
        Close Redis connection pool.

        Should be called at application shutdown.
        """
        if self._redis:
            await self._redis.close()
            self._redis = None
            logger.info("Redis cache disconnected")

    @property
    def is_connected(self) -> bool:
        """Check if Redis is connected."""
        return self._redis is not None

    async def get(self, key: str) -> Optional[Any]:
        """
        Get a value from cache.

        Args:
            key: Cache key

        Returns:
            Cached value or None if not found
        """
        if not self._redis:
            return None

        try:
            value = await self._redis.get(key)
            if value:
                return json.loads(value)
            return None
        except Exception as e:
            logger.error(f"Cache get error for key {key}: {e}")
            return None

    async def set(
        self,
        key: str,
        value: Any,
        ttl: int = CacheConfig.DEFAULT_TTL,
    ) -> bool:
        """
        Set a value in cache with TTL.

        Args:
            key: Cache key
            value: Value to cache (must be JSON serializable)
            ttl: Time-to-live in seconds

        Returns:
            True if successful, False otherwise
        """
        if not self._redis:
            return False

        try:
            serialized = json.dumps(value, default=str)
            await self._redis.setex(key, ttl, serialized)
            return True
        except Exception as e:
            logger.error(f"Cache set error for key {key}: {e}")
            return False

    async def delete(self, key: str) -> bool:
        """
        Delete a key from cache.

        Args:
            key: Cache key

        Returns:
            True if key was deleted, False otherwise
        """
        if not self._redis:
            return False

        try:
            await self._redis.delete(key)
            return True
        except Exception as e:
            logger.error(f"Cache delete error for key {key}: {e}")
            return False

    async def delete_pattern(self, pattern: str) -> int:
        """
        Delete all keys matching a pattern.

        Args:
            pattern: Redis key pattern (e.g., "persona:*")

        Returns:
            Number of keys deleted
        """
        if not self._redis:
            return 0

        try:
            keys = []
            async for key in self._redis.scan_iter(match=pattern):
                keys.append(key)

            if keys:
                return await self._redis.delete(*keys)
            return 0
        except Exception as e:
            logger.error(f"Cache delete pattern error for {pattern}: {e}")
            return 0

    async def exists(self, key: str) -> bool:
        """
        Check if a key exists in cache.

        Args:
            key: Cache key

        Returns:
            True if key exists, False otherwise
        """
        if not self._redis:
            return False

        try:
            return await self._redis.exists(key) > 0
        except Exception as e:
            logger.error(f"Cache exists error for key {key}: {e}")
            return False

    async def ttl(self, key: str) -> int:
        """
        Get remaining TTL for a key.

        Args:
            key: Cache key

        Returns:
            TTL in seconds, -1 if no TTL, -2 if key doesn't exist
        """
        if not self._redis:
            return -2

        try:
            return await self._redis.ttl(key)
        except Exception as e:
            logger.error(f"Cache TTL error for key {key}: {e}")
            return -2

    async def incr(self, key: str, amount: int = 1) -> Optional[int]:
        """
        Increment a counter in cache.

        Args:
            key: Cache key
            amount: Amount to increment

        Returns:
            New value or None on error
        """
        if not self._redis:
            return None

        try:
            return await self._redis.incrby(key, amount)
        except Exception as e:
            logger.error(f"Cache incr error for key {key}: {e}")
            return None

    async def decr(self, key: str, amount: int = 1) -> Optional[int]:
        """
        Decrement a counter in cache.

        Args:
            key: Cache key
            amount: Amount to decrement

        Returns:
            New value or None on error
        """
        if not self._redis:
            return None

        try:
            return await self._redis.decrby(key, amount)
        except Exception as e:
            logger.error(f"Cache decr error for key {key}: {e}")
            return None

    async def setex_if_not_exists(
        self,
        key: str,
        value: Any,
        ttl: int,
    ) -> bool:
        """
        Set a value only if key doesn't exist (atomic).

        Useful for distributed locks.

        Args:
            key: Cache key
            value: Value to set
            ttl: Time-to-live in seconds

        Returns:
            True if key was set, False if key already existed
        """
        if not self._redis:
            return False

        try:
            serialized = json.dumps(value, default=str)
            result = await self._redis.set(key, serialized, ex=ttl, nx=True)
            return result is not None
        except Exception as e:
            logger.error(f"Cache setnx error for key {key}: {e}")
            return False

    # ========================================
    # Persona-specific cache methods
    # ========================================

    async def get_persona(self, persona_id: Union[str, UUID]) -> Optional[dict]:
        """Get cached persona data."""
        key = f"{CacheConfig.PREFIX_PERSONA}{persona_id}"
        return await self.get(key)

    async def set_persona(
        self,
        persona_id: Union[str, UUID],
        data: dict,
        ttl: int = CacheConfig.MEDIUM_TTL,
    ) -> bool:
        """Cache persona data."""
        key = f"{CacheConfig.PREFIX_PERSONA}{persona_id}"
        return await self.set(key, data, ttl)

    async def invalidate_persona(self, persona_id: Union[str, UUID]) -> bool:
        """Invalidate cached persona data."""
        key = f"{CacheConfig.PREFIX_PERSONA}{persona_id}"
        return await self.delete(key)

    # ========================================
    # User session cache methods
    # ========================================

    async def get_user_session(self, session_id: str) -> Optional[dict]:
        """Get user session data."""
        key = f"{CacheConfig.PREFIX_SESSION}{session_id}"
        return await self.get(key)

    async def set_user_session(
        self,
        session_id: str,
        data: dict,
        ttl: int = CacheConfig.LONG_TTL,
    ) -> bool:
        """Cache user session data."""
        key = f"{CacheConfig.PREFIX_SESSION}{session_id}"
        return await self.set(key, data, ttl)

    async def delete_user_session(self, session_id: str) -> bool:
        """Delete user session."""
        key = f"{CacheConfig.PREFIX_SESSION}{session_id}"
        return await self.delete(key)

    # ========================================
    # Rate limiting cache methods
    # ========================================

    async def check_rate_limit(
        self,
        identifier: str,
        limit: int,
        window_seconds: int,
    ) -> tuple[bool, int, int]:
        """
        Check and update rate limit using sliding window.

        Args:
            identifier: Unique identifier (e.g., user_id, IP)
            limit: Maximum requests allowed
            window_seconds: Time window in seconds

        Returns:
            Tuple of (allowed, remaining_requests, reset_time)
        """
        if not self._redis:
            return True, limit, 0

        key = f"{CacheConfig.PREFIX_RATE_LIMIT}{identifier}"

        try:
            current = await self._redis.get(key)
            count = int(current) if current else 0

            if count >= limit:
                ttl = await self._redis.ttl(key)
                return False, 0, max(0, ttl)

            pipe = self._redis.pipeline()
            pipe.incr(key)
            if count == 0:
                pipe.expire(key, window_seconds)
            await pipe.execute()

            return True, limit - count - 1, window_seconds
        except Exception as e:
            logger.error(f"Rate limit check error: {e}")
            return True, limit, 0

    # ========================================
    # Generation queue tracking
    # ========================================

    async def add_generation_task(
        self,
        task_id: str,
        persona_id: str,
        content_type: str,
        ttl: int = CacheConfig.LONG_TTL,
    ) -> bool:
        """Track an active generation task."""
        key = f"{CacheConfig.PREFIX_GENERATION}{task_id}"
        data = {
            "persona_id": persona_id,
            "content_type": content_type,
            "status": "processing",
        }
        return await self.set(key, data, ttl)

    async def update_generation_status(
        self,
        task_id: str,
        status: str,
        result: Optional[dict] = None,
    ) -> bool:
        """Update generation task status."""
        key = f"{CacheConfig.PREFIX_GENERATION}{task_id}"
        data = await self.get(key)
        if data:
            data["status"] = status
            if result:
                data["result"] = result
            return await self.set(key, data)
        return False

    async def get_generation_status(self, task_id: str) -> Optional[dict]:
        """Get generation task status."""
        key = f"{CacheConfig.PREFIX_GENERATION}{task_id}"
        return await self.get(key)

    # ========================================
    # ACD context caching
    # ========================================

    async def cache_acd_context(
        self,
        context_id: str,
        data: dict,
        ttl: int = CacheConfig.MEDIUM_TTL,
    ) -> bool:
        """Cache ACD context data for quick access."""
        key = f"{CacheConfig.PREFIX_ACD}context:{context_id}"
        return await self.set(key, data, ttl)

    async def get_cached_acd_context(self, context_id: str) -> Optional[dict]:
        """Get cached ACD context."""
        key = f"{CacheConfig.PREFIX_ACD}context:{context_id}"
        return await self.get(key)

    async def cache_acd_patterns(
        self,
        domain: str,
        patterns: dict,
        ttl: int = CacheConfig.LONG_TTL,
    ) -> bool:
        """Cache learned ACD patterns for a domain."""
        key = f"{CacheConfig.PREFIX_ACD}patterns:{domain}"
        return await self.set(key, patterns, ttl)

    async def get_cached_acd_patterns(self, domain: str) -> Optional[dict]:
        """Get cached ACD patterns for a domain."""
        key = f"{CacheConfig.PREFIX_ACD}patterns:{domain}"
        return await self.get(key)

    # ========================================
    # API response caching
    # ========================================

    def _generate_cache_key(self, endpoint: str, params: dict) -> str:
        """Generate a unique cache key for API responses."""
        params_str = json.dumps(params, sort_keys=True, default=str)
        params_hash = hashlib.md5(params_str.encode()).hexdigest()[:8]
        return f"{CacheConfig.PREFIX_API}{endpoint}:{params_hash}"

    async def cache_api_response(
        self,
        endpoint: str,
        params: dict,
        response: dict,
        ttl: int = CacheConfig.SHORT_TTL,
    ) -> bool:
        """Cache API response."""
        key = self._generate_cache_key(endpoint, params)
        return await self.set(key, response, ttl)

    async def get_cached_api_response(
        self,
        endpoint: str,
        params: dict,
    ) -> Optional[dict]:
        """Get cached API response."""
        key = self._generate_cache_key(endpoint, params)
        return await self.get(key)

    # ========================================
    # Distributed locking
    # ========================================

    async def acquire_lock(
        self,
        lock_name: str,
        lock_timeout: int = 30,
    ) -> bool:
        """
        Acquire a distributed lock.

        Args:
            lock_name: Unique lock name
            lock_timeout: Lock timeout in seconds

        Returns:
            True if lock acquired, False otherwise
        """
        key = f"{CacheConfig.PREFIX_LOCK}{lock_name}"
        return await self.setex_if_not_exists(key, "locked", lock_timeout)

    async def release_lock(self, lock_name: str) -> bool:
        """
        Release a distributed lock.

        Args:
            lock_name: Lock name to release

        Returns:
            True if lock released
        """
        key = f"{CacheConfig.PREFIX_LOCK}{lock_name}"
        return await self.delete(key)

    async def extend_lock(
        self,
        lock_name: str,
        additional_seconds: int,
    ) -> bool:
        """
        Extend lock timeout.

        Args:
            lock_name: Lock name
            additional_seconds: Additional seconds to add

        Returns:
            True if lock extended
        """
        if not self._redis:
            return False

        key = f"{CacheConfig.PREFIX_LOCK}{lock_name}"
        try:
            return await self._redis.expire(key, additional_seconds)
        except Exception as e:
            logger.error(f"Lock extend error: {e}")
            return False

    # ========================================
    # Cache statistics
    # ========================================

    async def get_stats(self) -> dict:
        """Get cache statistics."""
        if not self._redis:
            return {"connected": False}

        try:
            info = await self._redis.info("memory")
            keys = await self._redis.dbsize()

            return {
                "connected": True,
                "used_memory": info.get("used_memory_human", "unknown"),
                "used_memory_peak": info.get("used_memory_peak_human", "unknown"),
                "total_keys": keys,
            }
        except Exception as e:
            logger.error(f"Cache stats error: {e}")
            return {"connected": True, "error": str(e)}


# Global cache instance
cache = CacheService()


def cached(
    ttl: int = CacheConfig.DEFAULT_TTL,
    key_prefix: str = "",
    key_builder: Optional[Callable[..., str]] = None,
):
    """
    Decorator for caching function results.

    Args:
        ttl: Cache TTL in seconds
        key_prefix: Prefix for cache key
        key_builder: Optional function to build cache key from args

    Usage:
        @cached(ttl=300, key_prefix="persona")
        async def get_persona(persona_id: str):
            ...
    """

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        async def wrapper(*args, **kwargs) -> T:
            # Generate cache key
            if key_builder:
                cache_key = key_builder(*args, **kwargs)
            else:
                # Default key from function name and args
                args_str = json.dumps(
                    {"args": args, "kwargs": kwargs},
                    sort_keys=True,
                    default=str,
                )
                args_hash = hashlib.md5(args_str.encode()).hexdigest()[:12]
                cache_key = f"{key_prefix}{func.__name__}:{args_hash}"

            # Try to get from cache
            cached_result = await cache.get(cache_key)
            if cached_result is not None:
                return cached_result

            # Execute function
            result = await func(*args, **kwargs)

            # Cache result
            if result is not None:
                await cache.set(cache_key, result, ttl)

            return result

        return wrapper

    return decorator


async def init_cache() -> None:
    """Initialize cache connection at application startup."""
    await cache.connect()


async def close_cache() -> None:
    """Close cache connection at application shutdown."""
    await cache.disconnect()
