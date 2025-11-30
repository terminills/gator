"""
Cache Management API Routes

Provides endpoints for monitoring and managing the Redis cache.
"""

from fastapi import APIRouter, Depends, HTTPException

from backend.config.logging import get_logger
from backend.services.cache_service import cache, CacheConfig

logger = get_logger(__name__)

router = APIRouter(prefix="/api/v1/cache", tags=["cache"])


@router.get("/status")
async def get_cache_status():
    """
    Get cache connection status and statistics.

    Returns:
        Cache status including connection state and memory usage
    """
    try:
        stats = await cache.get_stats()
        return {
            "status": "connected" if stats.get("connected") else "disconnected",
            **stats,
        }
    except Exception as e:
        logger.error(f"Failed to get cache status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health")
async def cache_health_check():
    """
    Check if cache is healthy and responsive.

    Returns:
        Health status
    """
    if not cache.is_connected:
        return {
            "status": "degraded",
            "message": "Cache is not connected. Application is running without caching.",
        }

    try:
        # Test with a simple set/get operation
        test_key = f"{CacheConfig.PREFIX_API}health_check"
        await cache.set(test_key, {"test": "value"}, ttl=10)
        result = await cache.get(test_key)
        await cache.delete(test_key)

        if result and result.get("test") == "value":
            return {"status": "healthy", "message": "Cache is operational"}
        else:
            return {"status": "degraded", "message": "Cache read/write mismatch"}
    except Exception as e:
        logger.error(f"Cache health check failed: {e}")
        return {"status": "unhealthy", "message": str(e)}


@router.delete("/invalidate/{prefix}")
async def invalidate_cache_by_prefix(prefix: str):
    """
    Invalidate all cache entries matching a prefix.

    Args:
        prefix: Cache key prefix to invalidate (e.g., 'persona', 'api')

    Returns:
        Number of keys deleted
    """
    try:
        # Map friendly names to actual prefixes
        prefix_map = {
            "persona": CacheConfig.PREFIX_PERSONA,
            "user": CacheConfig.PREFIX_USER,
            "session": CacheConfig.PREFIX_SESSION,
            "generation": CacheConfig.PREFIX_GENERATION,
            "acd": CacheConfig.PREFIX_ACD,
            "api": CacheConfig.PREFIX_API,
        }

        actual_prefix = prefix_map.get(prefix, prefix)
        pattern = f"{actual_prefix}*"

        deleted = await cache.delete_pattern(pattern)
        return {
            "status": "success",
            "prefix": prefix,
            "keys_deleted": deleted,
        }
    except Exception as e:
        logger.error(f"Failed to invalidate cache: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/key/{key}")
async def delete_cache_key(key: str):
    """
    Delete a specific cache key.

    Args:
        key: Full cache key to delete

    Returns:
        Deletion status
    """
    try:
        success = await cache.delete(key)
        return {
            "status": "success" if success else "not_found",
            "key": key,
            "deleted": success,
        }
    except Exception as e:
        logger.error(f"Failed to delete cache key: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/ttl/{key}")
async def get_cache_key_ttl(key: str):
    """
    Get the TTL (time-to-live) for a cache key.

    Args:
        key: Cache key to check

    Returns:
        TTL in seconds
    """
    try:
        ttl = await cache.ttl(key)
        exists = ttl >= -1

        return {
            "key": key,
            "exists": exists,
            "ttl": ttl if ttl >= 0 else None,
            "has_expiry": ttl > 0,
        }
    except Exception as e:
        logger.error(f"Failed to get TTL: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/config")
async def get_cache_config():
    """
    Get cache configuration constants.

    Returns:
        Cache configuration values
    """
    return {
        "ttl_values": {
            "default": CacheConfig.DEFAULT_TTL,
            "short": CacheConfig.SHORT_TTL,
            "medium": CacheConfig.MEDIUM_TTL,
            "long": CacheConfig.LONG_TTL,
            "very_long": CacheConfig.VERY_LONG_TTL,
        },
        "prefixes": {
            "persona": CacheConfig.PREFIX_PERSONA,
            "user": CacheConfig.PREFIX_USER,
            "session": CacheConfig.PREFIX_SESSION,
            "rate_limit": CacheConfig.PREFIX_RATE_LIMIT,
            "generation": CacheConfig.PREFIX_GENERATION,
            "acd": CacheConfig.PREFIX_ACD,
            "api": CacheConfig.PREFIX_API,
            "lock": CacheConfig.PREFIX_LOCK,
        },
    }
