"""
Rate Limiting Middleware

In-memory rate limiting for API endpoints.
Uses a sliding window algorithm for smooth rate limiting.

For production use with multiple workers, consider Redis-based rate limiting.
"""

import asyncio
from collections import defaultdict
from datetime import datetime, timezone
from typing import Callable, Dict, List, Optional, Tuple

from fastapi import Request, Response
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

from backend.config.logging import get_logger
from backend.config.settings import get_settings

logger = get_logger(__name__)
settings = get_settings()


class RateLimitExceeded(Exception):
    """Raised when rate limit is exceeded."""

    def __init__(
        self,
        message: str = "Rate limit exceeded",
        retry_after: int = 60,
    ):
        self.message = message
        self.retry_after = retry_after
        super().__init__(self.message)


class SlidingWindowRateLimiter:
    """
    In-memory sliding window rate limiter.

    Uses a sliding window algorithm for smoother rate limiting compared
    to fixed windows. Stores timestamps of requests and counts requests
    within the window.

    Note: This implementation is suitable for single-worker deployments.
    For multi-worker deployments, use Redis-based rate limiting.
    """

    def __init__(
        self,
        requests_per_window: int = 100,
        window_seconds: int = 60,
        cleanup_interval: int = 300,
    ):
        """
        Initialize the rate limiter.

        Args:
            requests_per_window: Maximum requests allowed per window
            window_seconds: Size of the sliding window in seconds
            cleanup_interval: Seconds between cleanup of expired entries
        """
        self.requests_per_window = requests_per_window
        self.window_seconds = window_seconds
        self.cleanup_interval = cleanup_interval

        # Store request timestamps: {key: [timestamp1, timestamp2, ...]}
        self._requests: Dict[str, List[float]] = defaultdict(list)
        self._last_cleanup = datetime.now(timezone.utc).timestamp()
        self._lock = asyncio.Lock()

    async def is_allowed(self, key: str) -> Tuple[bool, int, int]:
        """
        Check if a request is allowed under the rate limit.

        Args:
            key: Unique identifier for the rate limit bucket (e.g., IP, user ID)

        Returns:
            Tuple of (allowed, remaining_requests, reset_seconds)
        """
        async with self._lock:
            now = datetime.now(timezone.utc).timestamp()

            # Cleanup old entries periodically
            if now - self._last_cleanup > self.cleanup_interval:
                await self._cleanup(now)
                self._last_cleanup = now

            # Get window boundaries
            window_start = now - self.window_seconds

            # Filter out old requests outside the window
            self._requests[key] = [
                ts for ts in self._requests[key] if ts > window_start
            ]

            # Check if we're under the limit
            current_count = len(self._requests[key])

            if current_count >= self.requests_per_window:
                # Calculate when the oldest request will expire
                oldest_request = min(self._requests[key]) if self._requests[key] else now
                reset_seconds = int(oldest_request + self.window_seconds - now)
                return False, 0, max(1, reset_seconds)

            # Record this request
            self._requests[key].append(now)

            remaining = self.requests_per_window - current_count - 1
            return True, remaining, self.window_seconds

    async def _cleanup(self, current_time: float) -> None:
        """Remove expired entries from all buckets."""
        window_start = current_time - self.window_seconds
        keys_to_delete = []

        for key, timestamps in self._requests.items():
            # Filter out old requests
            self._requests[key] = [ts for ts in timestamps if ts > window_start]

            # Mark empty buckets for deletion
            if not self._requests[key]:
                keys_to_delete.append(key)

        # Delete empty buckets
        for key in keys_to_delete:
            del self._requests[key]

        if keys_to_delete:
            logger.debug(f"Rate limiter cleanup: removed {len(keys_to_delete)} buckets")

    def reset(self, key: str) -> None:
        """Reset rate limit for a specific key."""
        if key in self._requests:
            del self._requests[key]


class RateLimitConfig:
    """Configuration for rate limiting different endpoints."""

    def __init__(
        self,
        default_rate: int = 100,
        default_window: int = 60,
        endpoint_limits: Optional[Dict[str, Tuple[int, int]]] = None,
    ):
        """
        Initialize rate limit configuration.

        Args:
            default_rate: Default requests per window
            default_window: Default window in seconds
            endpoint_limits: Dict of path patterns to (rate, window) tuples
        """
        self.default_rate = default_rate
        self.default_window = default_window
        self.endpoint_limits = endpoint_limits or {}

    def get_limit(self, path: str) -> Tuple[int, int]:
        """
        Get rate limit for a specific path.

        Args:
            path: Request path

        Returns:
            Tuple of (requests_per_window, window_seconds)
        """
        # Check for exact matches first
        if path in self.endpoint_limits:
            return self.endpoint_limits[path]

        # Check for prefix matches
        for pattern, limit in self.endpoint_limits.items():
            if pattern.endswith("*") and path.startswith(pattern[:-1]):
                return limit

        return self.default_rate, self.default_window


# Default configuration
DEFAULT_RATE_LIMIT_CONFIG = RateLimitConfig(
    default_rate=100,  # 100 requests per minute by default
    default_window=60,
    endpoint_limits={
        # Content generation endpoints - more restrictive
        "/api/v1/content/*": (10, 60),  # 10 per minute
        "/api/v1/personas/*/generate*": (10, 60),

        # Auth endpoints - prevent brute force
        "/api/v1/auth/login": (10, 60),  # 10 login attempts per minute
        "/api/v1/auth/register": (5, 60),  # 5 registrations per minute
        "/api/v1/auth/refresh": (30, 60),  # 30 refresh attempts per minute

        # ACD learning endpoints - resource intensive
        "/api/v1/acd/learning/*": (5, 60),
        "/api/v1/acd/memory/consolidate": (2, 60),

        # Search endpoints
        "/api/v1/search/*": (30, 60),  # 30 searches per minute

        # Public endpoints - more permissive
        "/api/v1/public/*": (200, 60),
        "/health": (300, 60),
    },
)


class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    FastAPI middleware for rate limiting.

    Applies rate limiting based on client IP address by default.
    Can be extended to use user ID for authenticated requests.
    """

    def __init__(
        self,
        app: ASGIApp,
        config: Optional[RateLimitConfig] = None,
        key_func: Optional[Callable[[Request], str]] = None,
        enabled: bool = True,
    ):
        """
        Initialize the middleware.

        Args:
            app: FastAPI application
            config: Rate limit configuration
            key_func: Function to extract rate limit key from request
            enabled: Whether rate limiting is enabled
        """
        super().__init__(app)
        self.config = config or DEFAULT_RATE_LIMIT_CONFIG
        self.enabled = enabled

        # Custom key function or default to IP
        self.key_func = key_func or self._get_client_ip

        # Create rate limiters for each unique limit configuration
        # Note: This dictionary is bounded by the number of unique rate/window
        # combinations in the config (typically < 10), not by requests
        self._limiters: Dict[Tuple[int, int], SlidingWindowRateLimiter] = {}

    def _get_limiter(self, rate: int, window: int) -> SlidingWindowRateLimiter:
        """Get or create a rate limiter for the given configuration."""
        key = (rate, window)
        if key not in self._limiters:
            self._limiters[key] = SlidingWindowRateLimiter(
                requests_per_window=rate,
                window_seconds=window,
            )
        return self._limiters[key]

    def _get_client_ip(self, request: Request) -> str:
        """
        Extract client IP from request.

        Handles X-Forwarded-For header for proxy setups.

        Args:
            request: FastAPI request

        Returns:
            Client IP address
        """
        # Check X-Forwarded-For header (for reverse proxy setups)
        forwarded = request.headers.get("X-Forwarded-For")
        if forwarded:
            # Get the first IP in the chain (original client)
            return forwarded.split(",")[0].strip()

        # Fall back to direct client IP
        if request.client:
            return request.client.host

        return "unknown"

    async def dispatch(
        self,
        request: Request,
        call_next: Callable,
    ) -> Response:
        """
        Process request and apply rate limiting.

        Args:
            request: Incoming request
            call_next: Next middleware/handler

        Returns:
            Response from handler or rate limit error
        """
        # Skip if disabled
        if not self.enabled:
            return await call_next(request)

        # Skip rate limiting for certain paths
        skip_paths = ["/docs", "/redoc", "/openapi.json", "/static"]
        if any(request.url.path.startswith(p) for p in skip_paths):
            return await call_next(request)

        # Get rate limit for this path
        rate, window = self.config.get_limit(request.url.path)

        # Get rate limit key (IP by default)
        key = self.key_func(request)

        # Get the appropriate limiter
        limiter = self._get_limiter(rate, window)

        # Check if request is allowed
        allowed, remaining, retry_after = await limiter.is_allowed(key)

        if not allowed:
            logger.warning(
                f"Rate limit exceeded for {key} on {request.url.path}",
                extra={
                    "client_ip": key,
                    "path": request.url.path,
                    "retry_after": retry_after,
                },
            )

            return JSONResponse(
                status_code=429,
                content={
                    "error": "RATE_LIMIT_EXCEEDED",
                    "message": "Too many requests. Please slow down.",
                    "retry_after": retry_after,
                },
                headers={
                    "Retry-After": str(retry_after),
                    "X-RateLimit-Limit": str(rate),
                    "X-RateLimit-Remaining": "0",
                    "X-RateLimit-Reset": str(retry_after),
                },
            )

        # Call the actual handler
        response = await call_next(request)

        # Add rate limit headers to response
        response.headers["X-RateLimit-Limit"] = str(rate)
        response.headers["X-RateLimit-Remaining"] = str(remaining)
        response.headers["X-RateLimit-Reset"] = str(retry_after)

        return response


def get_rate_limiter_for_user(request: Request) -> str:
    """
    Extract rate limit key based on authenticated user.

    Falls back to IP if not authenticated.

    Args:
        request: FastAPI request

    Returns:
        User ID or client IP
    """
    # Try to get user ID from request state (set by auth middleware)
    user = getattr(request.state, "user", None)
    if user and hasattr(user, "id"):
        return f"user:{user.id}"

    # Fall back to IP
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        return f"ip:{forwarded.split(',')[0].strip()}"

    if request.client:
        return f"ip:{request.client.host}"

    return "ip:unknown"


def create_rate_limit_middleware(
    enabled: bool = True,
    use_user_auth: bool = False,
) -> RateLimitMiddleware:
    """
    Factory function to create rate limit middleware.

    Args:
        enabled: Whether rate limiting is enabled
        use_user_auth: Whether to use user ID for rate limiting

    Returns:
        Configured RateLimitMiddleware instance
    """
    key_func = get_rate_limiter_for_user if use_user_auth else None

    return RateLimitMiddleware(
        app=None,  # Will be set by FastAPI
        config=DEFAULT_RATE_LIMIT_CONFIG,
        key_func=key_func,
        enabled=enabled,
    )
