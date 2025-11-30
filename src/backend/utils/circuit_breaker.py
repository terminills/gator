"""
Circuit Breaker Utility for Production Hardening

Implements the circuit breaker pattern to prevent cascading failures
when external services are unavailable or overloaded.

Usage:
    from backend.utils.circuit_breaker import CircuitBreaker, circuit_breaker
    
    # As a decorator
    @circuit_breaker(failure_threshold=5, recovery_timeout=60)
    async def call_external_service():
        ...
    
    # As a context manager
    async with CircuitBreaker("service_name") as breaker:
        result = await call_external_service()
"""

import asyncio
import time
from enum import Enum
from functools import wraps
from typing import Any, Callable, Dict, Optional, TypeVar

from backend.config.logging import get_logger
from backend.config.settings import get_settings

logger = get_logger(__name__)
settings = get_settings()

T = TypeVar("T")


class CircuitState(str, Enum):
    """Circuit breaker states."""

    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Failing, rejecting calls
    HALF_OPEN = "half_open"  # Testing if service recovered


class CircuitBreakerError(Exception):
    """Raised when circuit breaker is open."""

    def __init__(self, name: str, message: str = "Circuit breaker is open"):
        self.name = name
        self.message = message
        super().__init__(f"{name}: {message}")


class CircuitBreaker:
    """
    Circuit breaker implementation for external service calls.

    Prevents cascading failures by tracking failures and temporarily
    blocking calls to services that are failing repeatedly.

    Attributes:
        name: Identifier for this circuit breaker
        failure_threshold: Number of failures before opening circuit
        recovery_timeout: Seconds before attempting recovery
        state: Current circuit state
    """

    # Class-level registry of circuit breakers for monitoring
    _registry: Dict[str, "CircuitBreaker"] = {}

    def __init__(
        self,
        name: str,
        failure_threshold: Optional[int] = None,
        recovery_timeout: Optional[int] = None,
    ):
        """
        Initialize circuit breaker.

        Args:
            name: Unique identifier for this circuit breaker
            failure_threshold: Failures before opening (default from settings)
            recovery_timeout: Seconds before recovery attempt (default from settings)
        """
        self.name = name
        self.failure_threshold = (
            failure_threshold or settings.circuit_breaker_failure_threshold
        )
        self.recovery_timeout = (
            recovery_timeout or settings.circuit_breaker_recovery_timeout
        )
        self.failure_count = 0
        self.last_failure_time: Optional[float] = None
        self.state = CircuitState.CLOSED
        self._lock = asyncio.Lock()

        # Register for monitoring
        CircuitBreaker._registry[name] = self

    async def __aenter__(self) -> "CircuitBreaker":
        """Async context manager entry."""
        await self._check_state()
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> bool:
        """Async context manager exit."""
        if exc_type is not None:
            await self._record_failure()
        else:
            await self._record_success()
        return False  # Don't suppress exceptions

    async def _check_state(self) -> None:
        """Check circuit state and raise if open."""
        async with self._lock:
            if self.state == CircuitState.OPEN:
                if self._should_attempt_recovery():
                    self.state = CircuitState.HALF_OPEN
                    logger.info(f"Circuit '{self.name}' entering half-open state")
                else:
                    raise CircuitBreakerError(
                        self.name,
                        f"Service unavailable (retry in {self._time_until_recovery():.0f}s)",
                    )

    def _should_attempt_recovery(self) -> bool:
        """Check if enough time has passed for recovery attempt."""
        if self.last_failure_time is None:
            return True
        return time.time() - self.last_failure_time > self.recovery_timeout

    def _time_until_recovery(self) -> float:
        """Calculate seconds until recovery attempt."""
        if self.last_failure_time is None:
            return 0
        elapsed = time.time() - self.last_failure_time
        return max(0, self.recovery_timeout - elapsed)

    async def _record_failure(self) -> None:
        """Record a failure and potentially open the circuit."""
        async with self._lock:
            self.failure_count += 1
            self.last_failure_time = time.time()

            if self.state == CircuitState.HALF_OPEN:
                # Failed during recovery test, go back to open
                self.state = CircuitState.OPEN
                logger.warning(
                    f"Circuit '{self.name}' re-opened after failed recovery attempt"
                )
            elif self.failure_count >= self.failure_threshold:
                self.state = CircuitState.OPEN
                logger.error(
                    f"Circuit '{self.name}' opened after {self.failure_count} failures"
                )

    async def _record_success(self) -> None:
        """Record a success and potentially close the circuit."""
        async with self._lock:
            if self.state == CircuitState.HALF_OPEN:
                # Successfully completed call during recovery
                self.state = CircuitState.CLOSED
                self.failure_count = 0
                logger.info(f"Circuit '{self.name}' closed after successful recovery")
            elif self.state == CircuitState.CLOSED:
                # Reset failure count on success
                self.failure_count = 0

    def get_status(self) -> Dict[str, Any]:
        """Get current circuit breaker status."""
        return {
            "name": self.name,
            "state": self.state.value,
            "failure_count": self.failure_count,
            "failure_threshold": self.failure_threshold,
            "recovery_timeout": self.recovery_timeout,
            "last_failure": self.last_failure_time,
            "time_until_recovery": (
                self._time_until_recovery()
                if self.state == CircuitState.OPEN
                else None
            ),
        }

    @classmethod
    def get_all_status(cls) -> Dict[str, Dict[str, Any]]:
        """Get status of all registered circuit breakers."""
        return {name: breaker.get_status() for name, breaker in cls._registry.items()}

    @classmethod
    def reset_all(cls) -> None:
        """Reset all circuit breakers (for testing)."""
        for breaker in cls._registry.values():
            breaker.state = CircuitState.CLOSED
            breaker.failure_count = 0
            breaker.last_failure_time = None


def circuit_breaker(
    name: Optional[str] = None,
    failure_threshold: Optional[int] = None,
    recovery_timeout: Optional[int] = None,
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """
    Circuit breaker decorator for async functions.

    Args:
        name: Circuit breaker identifier (defaults to function name)
        failure_threshold: Failures before opening circuit
        recovery_timeout: Seconds before recovery attempt

    Returns:
        Decorated function with circuit breaker protection

    Example:
        @circuit_breaker(name="ollama", failure_threshold=3)
        async def call_ollama(prompt: str) -> str:
            async with httpx.AsyncClient() as client:
                response = await client.post(...)
                return response.json()
    """

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        cb_name = name or func.__name__
        breaker = CircuitBreaker(
            cb_name,
            failure_threshold=failure_threshold,
            recovery_timeout=recovery_timeout,
        )

        @wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> T:
            async with breaker:
                return await func(*args, **kwargs)

        return wrapper

    return decorator
