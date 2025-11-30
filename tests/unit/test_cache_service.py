"""
Tests for the Redis Cache Service.

These tests verify the cache service functionality with mocked Redis.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
import json

from backend.services.cache_service import (
    CacheService,
    CacheConfig,
    cache,
    cached,
)


class TestCacheConfig:
    """Test cache configuration constants."""

    def test_ttl_values(self):
        """Test TTL constants are defined correctly."""
        assert CacheConfig.DEFAULT_TTL == 300
        assert CacheConfig.SHORT_TTL == 60
        assert CacheConfig.MEDIUM_TTL == 600
        assert CacheConfig.LONG_TTL == 3600
        assert CacheConfig.VERY_LONG_TTL == 86400

    def test_prefix_values(self):
        """Test prefix constants are defined correctly."""
        assert CacheConfig.PREFIX_PERSONA == "persona:"
        assert CacheConfig.PREFIX_USER == "user:"
        assert CacheConfig.PREFIX_SESSION == "session:"
        assert CacheConfig.PREFIX_RATE_LIMIT == "rate_limit:"
        assert CacheConfig.PREFIX_GENERATION == "generation:"
        assert CacheConfig.PREFIX_ACD == "acd:"
        assert CacheConfig.PREFIX_API == "api:"
        assert CacheConfig.PREFIX_LOCK == "lock:"


class TestCacheServiceSingleton:
    """Test singleton pattern for cache service."""

    def test_singleton_pattern(self):
        """Test that CacheService returns the same instance."""
        service1 = CacheService()
        service2 = CacheService()
        assert service1 is service2


class TestCacheServiceNotConnected:
    """Test cache service behavior when Redis is not connected."""

    @pytest.fixture
    def cache_service(self):
        """Create a cache service without Redis connection."""
        service = CacheService()
        service._redis = None
        return service

    @pytest.mark.asyncio
    async def test_get_returns_none_when_disconnected(self, cache_service):
        """Test get returns None when Redis is not connected."""
        result = await cache_service.get("test_key")
        assert result is None

    @pytest.mark.asyncio
    async def test_set_returns_false_when_disconnected(self, cache_service):
        """Test set returns False when Redis is not connected."""
        result = await cache_service.set("test_key", {"value": 1})
        assert result is False

    @pytest.mark.asyncio
    async def test_delete_returns_false_when_disconnected(self, cache_service):
        """Test delete returns False when Redis is not connected."""
        result = await cache_service.delete("test_key")
        assert result is False

    @pytest.mark.asyncio
    async def test_exists_returns_false_when_disconnected(self, cache_service):
        """Test exists returns False when Redis is not connected."""
        result = await cache_service.exists("test_key")
        assert result is False

    @pytest.mark.asyncio
    async def test_is_connected_property(self, cache_service):
        """Test is_connected property returns False."""
        assert cache_service.is_connected is False


class TestCacheServiceConnected:
    """Test cache service with mocked Redis connection."""

    @pytest.fixture
    def mock_redis(self):
        """Create a mock Redis instance."""
        mock = AsyncMock()
        return mock

    @pytest.fixture
    def cache_service(self, mock_redis):
        """Create a cache service with mocked Redis."""
        service = CacheService()
        service._redis = mock_redis
        return service

    @pytest.mark.asyncio
    async def test_get_success(self, cache_service, mock_redis):
        """Test successful get operation."""
        mock_redis.get.return_value = '{"key": "value"}'

        result = await cache_service.get("test_key")

        assert result == {"key": "value"}
        mock_redis.get.assert_called_once_with("test_key")

    @pytest.mark.asyncio
    async def test_get_not_found(self, cache_service, mock_redis):
        """Test get returns None when key doesn't exist."""
        mock_redis.get.return_value = None

        result = await cache_service.get("missing_key")

        assert result is None

    @pytest.mark.asyncio
    async def test_set_success(self, cache_service, mock_redis):
        """Test successful set operation."""
        result = await cache_service.set("test_key", {"value": 123}, ttl=600)

        assert result is True
        mock_redis.setex.assert_called_once_with(
            "test_key", 600, '{"value": 123}'
        )

    @pytest.mark.asyncio
    async def test_delete_success(self, cache_service, mock_redis):
        """Test successful delete operation."""
        result = await cache_service.delete("test_key")

        assert result is True
        mock_redis.delete.assert_called_once_with("test_key")

    @pytest.mark.asyncio
    async def test_exists_true(self, cache_service, mock_redis):
        """Test exists returns True when key exists."""
        mock_redis.exists.return_value = 1

        result = await cache_service.exists("test_key")

        assert result is True

    @pytest.mark.asyncio
    async def test_exists_false(self, cache_service, mock_redis):
        """Test exists returns False when key doesn't exist."""
        mock_redis.exists.return_value = 0

        result = await cache_service.exists("test_key")

        assert result is False

    @pytest.mark.asyncio
    async def test_incr(self, cache_service, mock_redis):
        """Test increment operation."""
        mock_redis.incrby.return_value = 5

        result = await cache_service.incr("counter", 2)

        assert result == 5
        mock_redis.incrby.assert_called_once_with("counter", 2)

    @pytest.mark.asyncio
    async def test_decr(self, cache_service, mock_redis):
        """Test decrement operation."""
        mock_redis.decrby.return_value = 3

        result = await cache_service.decr("counter", 2)

        assert result == 3
        mock_redis.decrby.assert_called_once_with("counter", 2)

    @pytest.mark.asyncio
    async def test_ttl(self, cache_service, mock_redis):
        """Test TTL retrieval."""
        mock_redis.ttl.return_value = 120

        result = await cache_service.ttl("test_key")

        assert result == 120

    @pytest.mark.asyncio
    async def test_setex_if_not_exists_success(self, cache_service, mock_redis):
        """Test setnx operation when key doesn't exist."""
        mock_redis.set.return_value = True

        result = await cache_service.setex_if_not_exists(
            "lock_key", "locked", 30
        )

        assert result is True
        mock_redis.set.assert_called_once()

    @pytest.mark.asyncio
    async def test_setex_if_not_exists_already_exists(self, cache_service, mock_redis):
        """Test setnx operation when key already exists."""
        mock_redis.set.return_value = None

        result = await cache_service.setex_if_not_exists(
            "lock_key", "locked", 30
        )

        assert result is False


class TestCacheServicePersonaMethods:
    """Test persona-specific cache methods."""

    @pytest.fixture
    def mock_redis(self):
        """Create a mock Redis instance."""
        return AsyncMock()

    @pytest.fixture
    def cache_service(self, mock_redis):
        """Create a cache service with mocked Redis."""
        service = CacheService()
        service._redis = mock_redis
        return service

    @pytest.mark.asyncio
    async def test_get_persona(self, cache_service, mock_redis):
        """Test get_persona method."""
        persona_data = {"id": "123", "name": "Test"}
        mock_redis.get.return_value = json.dumps(persona_data)

        result = await cache_service.get_persona("123")

        assert result == persona_data
        mock_redis.get.assert_called_once_with("persona:123")

    @pytest.mark.asyncio
    async def test_set_persona(self, cache_service, mock_redis):
        """Test set_persona method."""
        persona_data = {"id": "123", "name": "Test"}

        result = await cache_service.set_persona("123", persona_data)

        assert result is True
        mock_redis.setex.assert_called_once()

    @pytest.mark.asyncio
    async def test_invalidate_persona(self, cache_service, mock_redis):
        """Test invalidate_persona method."""
        result = await cache_service.invalidate_persona("123")

        assert result is True
        mock_redis.delete.assert_called_once_with("persona:123")


class TestCacheServiceSessionMethods:
    """Test session-specific cache methods."""

    @pytest.fixture
    def mock_redis(self):
        """Create a mock Redis instance."""
        return AsyncMock()

    @pytest.fixture
    def cache_service(self, mock_redis):
        """Create a cache service with mocked Redis."""
        service = CacheService()
        service._redis = mock_redis
        return service

    @pytest.mark.asyncio
    async def test_get_user_session(self, cache_service, mock_redis):
        """Test get_user_session method."""
        session_data = {"user_id": "456", "expires": "2024-01-01"}
        mock_redis.get.return_value = json.dumps(session_data)

        result = await cache_service.get_user_session("session_abc")

        assert result == session_data
        mock_redis.get.assert_called_once_with("session:session_abc")

    @pytest.mark.asyncio
    async def test_set_user_session(self, cache_service, mock_redis):
        """Test set_user_session method."""
        session_data = {"user_id": "456"}

        result = await cache_service.set_user_session("session_abc", session_data)

        assert result is True
        mock_redis.setex.assert_called_once()

    @pytest.mark.asyncio
    async def test_delete_user_session(self, cache_service, mock_redis):
        """Test delete_user_session method."""
        result = await cache_service.delete_user_session("session_abc")

        assert result is True
        mock_redis.delete.assert_called_once_with("session:session_abc")


class TestCacheServiceLocking:
    """Test distributed locking methods."""

    @pytest.fixture
    def mock_redis(self):
        """Create a mock Redis instance."""
        return AsyncMock()

    @pytest.fixture
    def cache_service(self, mock_redis):
        """Create a cache service with mocked Redis."""
        service = CacheService()
        service._redis = mock_redis
        return service

    @pytest.mark.asyncio
    async def test_acquire_lock_success(self, cache_service, mock_redis):
        """Test successful lock acquisition."""
        mock_redis.set.return_value = True

        result = await cache_service.acquire_lock("my_lock", 30)

        assert result is True
        mock_redis.set.assert_called_once()

    @pytest.mark.asyncio
    async def test_acquire_lock_already_locked(self, cache_service, mock_redis):
        """Test failed lock acquisition when already locked."""
        mock_redis.set.return_value = None

        result = await cache_service.acquire_lock("my_lock", 30)

        assert result is False

    @pytest.mark.asyncio
    async def test_release_lock(self, cache_service, mock_redis):
        """Test lock release."""
        result = await cache_service.release_lock("my_lock")

        assert result is True
        mock_redis.delete.assert_called_once_with("lock:my_lock")

    @pytest.mark.asyncio
    async def test_extend_lock(self, cache_service, mock_redis):
        """Test lock extension."""
        mock_redis.expire.return_value = True

        result = await cache_service.extend_lock("my_lock", 60)

        assert result is True
        mock_redis.expire.assert_called_once_with("lock:my_lock", 60)


class TestCacheServiceRateLimiting:
    """Test rate limiting methods."""

    @pytest.fixture
    def mock_redis(self):
        """Create a mock Redis instance."""
        mock = MagicMock()
        mock.get = AsyncMock()
        mock.ttl = AsyncMock()
        # Create a proper pipeline mock
        mock_pipeline = MagicMock()
        mock_pipeline.incr = MagicMock(return_value=mock_pipeline)
        mock_pipeline.expire = MagicMock(return_value=mock_pipeline)
        mock_pipeline.execute = AsyncMock(return_value=[1, True])
        mock.pipeline = MagicMock(return_value=mock_pipeline)
        return mock

    @pytest.fixture
    def cache_service(self, mock_redis):
        """Create a cache service with mocked Redis."""
        service = CacheService()
        service._redis = mock_redis
        return service

    @pytest.mark.asyncio
    async def test_rate_limit_allowed(self, cache_service, mock_redis):
        """Test rate limit when under limit."""
        mock_redis.get.return_value = "5"

        allowed, remaining, reset = await cache_service.check_rate_limit(
            "user:123", 10, 60
        )

        assert allowed is True
        assert remaining == 4  # 10 - 5 - 1

    @pytest.mark.asyncio
    async def test_rate_limit_exceeded(self, cache_service, mock_redis):
        """Test rate limit when exceeded."""
        mock_redis.get.return_value = "10"
        mock_redis.ttl.return_value = 30

        allowed, remaining, reset = await cache_service.check_rate_limit(
            "user:123", 10, 60
        )

        assert allowed is False
        assert remaining == 0
        assert reset == 30

    @pytest.mark.asyncio
    async def test_rate_limit_first_request(self, cache_service, mock_redis):
        """Test rate limit on first request."""
        mock_redis.get.return_value = None

        allowed, remaining, reset = await cache_service.check_rate_limit(
            "user:123", 10, 60
        )

        assert allowed is True
        assert remaining == 9  # 10 - 0 - 1


class TestCacheServiceStats:
    """Test cache statistics methods."""

    @pytest.fixture
    def mock_redis(self):
        """Create a mock Redis instance."""
        return AsyncMock()

    @pytest.fixture
    def cache_service(self, mock_redis):
        """Create a cache service with mocked Redis."""
        service = CacheService()
        service._redis = mock_redis
        return service

    @pytest.mark.asyncio
    async def test_get_stats_connected(self, cache_service, mock_redis):
        """Test stats retrieval when connected."""
        mock_redis.info.return_value = {
            "used_memory_human": "1.5M",
            "used_memory_peak_human": "2.0M",
        }
        mock_redis.dbsize.return_value = 100

        stats = await cache_service.get_stats()

        assert stats["connected"] is True
        assert stats["used_memory"] == "1.5M"
        assert stats["total_keys"] == 100

    @pytest.mark.asyncio
    async def test_get_stats_not_connected(self, cache_service):
        """Test stats retrieval when not connected."""
        cache_service._redis = None

        stats = await cache_service.get_stats()

        assert stats["connected"] is False


class TestCacheDecorator:
    """Test the @cached decorator."""

    @pytest.mark.asyncio
    async def test_cached_decorator_hit(self):
        """Test decorator returns cached value."""
        with patch.object(cache, "get", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = {"cached": True}

            @cached(ttl=300, key_prefix="test:")
            async def my_function(arg1):
                return {"computed": True}

            result = await my_function("test_arg")

            assert result == {"cached": True}

    @pytest.mark.asyncio
    async def test_cached_decorator_miss(self):
        """Test decorator computes and caches on miss."""
        with patch.object(cache, "get", new_callable=AsyncMock) as mock_get, \
             patch.object(cache, "set", new_callable=AsyncMock) as mock_set:
            mock_get.return_value = None

            @cached(ttl=300, key_prefix="test:")
            async def my_function(arg1):
                return {"computed": True}

            result = await my_function("test_arg")

            assert result == {"computed": True}
            mock_set.assert_called_once()
