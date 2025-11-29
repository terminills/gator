"""
Tests for Rate Limiting Middleware

Tests for the sliding window rate limiter and middleware.
"""

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock

from backend.api.rate_limiting import (
    SlidingWindowRateLimiter,
    RateLimitConfig,
    RateLimitExceeded,
    DEFAULT_RATE_LIMIT_CONFIG,
)


class TestSlidingWindowRateLimiter:
    """Test the sliding window rate limiter."""

    @pytest.fixture
    def limiter(self):
        """Create a rate limiter with small limits for testing."""
        return SlidingWindowRateLimiter(
            requests_per_window=5,
            window_seconds=60,
        )

    @pytest.mark.asyncio
    async def test_allows_requests_under_limit(self, limiter):
        """Test that requests under the limit are allowed."""
        key = "test-client-1"

        for i in range(5):
            allowed, remaining, _ = await limiter.is_allowed(key)
            assert allowed is True
            assert remaining == 4 - i

    @pytest.mark.asyncio
    async def test_blocks_requests_over_limit(self, limiter):
        """Test that requests over the limit are blocked."""
        key = "test-client-2"

        # Use up all allowed requests
        for _ in range(5):
            allowed, _, _ = await limiter.is_allowed(key)
            assert allowed is True

        # Next request should be blocked
        allowed, remaining, retry_after = await limiter.is_allowed(key)
        assert allowed is False
        assert remaining == 0
        assert retry_after > 0

    @pytest.mark.asyncio
    async def test_different_keys_have_separate_limits(self, limiter):
        """Test that different clients have separate rate limits."""
        key1 = "client-a"
        key2 = "client-b"

        # Exhaust limit for key1
        for _ in range(5):
            await limiter.is_allowed(key1)

        # key1 should be blocked
        allowed, _, _ = await limiter.is_allowed(key1)
        assert allowed is False

        # key2 should still be allowed
        allowed, remaining, _ = await limiter.is_allowed(key2)
        assert allowed is True
        assert remaining == 4

    @pytest.mark.asyncio
    async def test_reset_clears_limit(self, limiter):
        """Test that reset clears the rate limit for a key."""
        key = "test-client-3"

        # Use up all requests
        for _ in range(5):
            await limiter.is_allowed(key)

        # Should be blocked
        allowed, _, _ = await limiter.is_allowed(key)
        assert allowed is False

        # Reset the limit
        limiter.reset(key)

        # Should be allowed again
        allowed, remaining, _ = await limiter.is_allowed(key)
        assert allowed is True
        assert remaining == 4

    @pytest.mark.asyncio
    async def test_returns_retry_after_when_blocked(self, limiter):
        """Test that retry_after is returned when rate limited."""
        key = "test-client-4"

        # Exhaust limit
        for _ in range(5):
            await limiter.is_allowed(key)

        # Check retry_after
        _, _, retry_after = await limiter.is_allowed(key)
        assert retry_after > 0
        assert retry_after <= 60  # Should be within the window


class TestRateLimitConfig:
    """Test the rate limit configuration."""

    @pytest.fixture
    def config(self):
        """Create a test configuration."""
        return RateLimitConfig(
            default_rate=100,
            default_window=60,
            endpoint_limits={
                "/api/v1/auth/login": (10, 60),
                "/api/v1/content/*": (5, 60),
            },
        )

    def test_returns_default_for_unknown_path(self, config):
        """Test that default limits are returned for unknown paths."""
        rate, window = config.get_limit("/api/v1/unknown")
        assert rate == 100
        assert window == 60

    def test_returns_exact_match_limit(self, config):
        """Test exact path matching."""
        rate, window = config.get_limit("/api/v1/auth/login")
        assert rate == 10
        assert window == 60

    def test_returns_wildcard_match_limit(self, config):
        """Test wildcard path matching."""
        rate, window = config.get_limit("/api/v1/content/generate")
        assert rate == 5
        assert window == 60

    def test_wildcard_does_not_match_partial(self, config):
        """Test that wildcard only matches complete prefix."""
        # "/api/v1/content" without trailing parts should use default
        # since our wildcard is "/api/v1/content/*"
        rate, window = config.get_limit("/api/v1/content")
        # Should return default since "/api/v1/content" != "/api/v1/content/"
        assert rate == 100


class TestDefaultRateLimitConfig:
    """Test the default rate limit configuration."""

    def test_auth_endpoints_have_lower_limits(self):
        """Test that auth endpoints have restrictive limits."""
        login_rate, _ = DEFAULT_RATE_LIMIT_CONFIG.get_limit("/api/v1/auth/login")
        register_rate, _ = DEFAULT_RATE_LIMIT_CONFIG.get_limit("/api/v1/auth/register")

        assert login_rate <= 10  # At most 10 login attempts per minute
        assert register_rate <= 5  # At most 5 registrations per minute

    def test_content_generation_has_lower_limits(self):
        """Test that content generation endpoints have restrictive limits."""
        rate, _ = DEFAULT_RATE_LIMIT_CONFIG.get_limit("/api/v1/content/generate")
        assert rate <= 10  # Resource-intensive endpoints

    def test_public_endpoints_have_higher_limits(self):
        """Test that public endpoints have permissive limits."""
        rate, _ = DEFAULT_RATE_LIMIT_CONFIG.get_limit("/api/v1/public/gallery")
        assert rate >= 100

    def test_health_endpoint_has_high_limit(self):
        """Test that health check has high limit."""
        rate, _ = DEFAULT_RATE_LIMIT_CONFIG.get_limit("/health")
        assert rate >= 100


class TestRateLimitExceeded:
    """Test the rate limit exceeded exception."""

    def test_default_message(self):
        """Test default exception message."""
        exc = RateLimitExceeded()
        assert exc.message == "Rate limit exceeded"
        assert exc.retry_after == 60

    def test_custom_message(self):
        """Test custom exception message."""
        exc = RateLimitExceeded(
            message="Custom limit message",
            retry_after=120,
        )
        assert exc.message == "Custom limit message"
        assert exc.retry_after == 120


class TestConcurrentAccess:
    """Test rate limiter under concurrent access."""

    @pytest.mark.asyncio
    async def test_concurrent_requests_are_tracked(self):
        """Test that concurrent requests are properly counted."""
        limiter = SlidingWindowRateLimiter(
            requests_per_window=10,
            window_seconds=60,
        )
        key = "concurrent-client"

        # Make 10 concurrent requests
        async def make_request():
            return await limiter.is_allowed(key)

        results = await asyncio.gather(*[make_request() for _ in range(10)])

        # All should be allowed
        allowed_count = sum(1 for allowed, _, _ in results if allowed)
        assert allowed_count == 10

        # Next request should be blocked
        allowed, _, _ = await limiter.is_allowed(key)
        assert allowed is False
