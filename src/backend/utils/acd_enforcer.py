"""
ACD Enforcer - Centralized ACD Enforcement Wrapper

Provides consistent ACD context tracking across all operations
to prevent ACD drift and ensure comprehensive tracing.

Usage:
    from backend.utils.acd_enforcer import ACDEnforcer, acd_tracked
    
    # As a context manager
    async with ACDEnforcer(db).track_operation(
        phase="IMAGE_GENERATION",
        complexity="HIGH"
    ) as context:
        result = await generate_image(...)
        context.update_metadata({"model_used": model_name})
    
    # As a decorator
    @acd_tracked(phase="TEXT_GENERATION")
    async def generate_text(prompt: str, db: AsyncSession) -> str:
        ...
"""

from contextlib import asynccontextmanager
from datetime import datetime, timezone
from functools import wraps
from typing import Any, AsyncGenerator, Callable, Dict, Optional, TypeVar
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from backend.config.logging import get_logger, correlation_id
from backend.config.settings import get_settings
from backend.models.acd import (
    ACDContextCreate,
    ACDContextResponse,
    ACDContextUpdate,
    ACDTraceArtifactCreate,
    AIComplexity,
    AIPhase,
    AIQueueStatus,
    AIState,
)
from backend.services.acd_service import ACDService

logger = get_logger(__name__)
settings = get_settings()

T = TypeVar("T")


class ACDContextWrapper:
    """
    Wrapper around ACD context for easier manipulation during operations.

    Provides methods to update metadata, record errors, and manage state
    during the lifetime of an operation.
    """

    def __init__(
        self,
        context: Optional[ACDContextResponse],
        acd_service: Optional[ACDService],
        enabled: bool = True,
    ):
        self.context = context
        self.acd_service = acd_service
        self.enabled = enabled and context is not None
        self._start_time = datetime.now(timezone.utc)
        self._metadata: Dict[str, Any] = {}

    @property
    def id(self) -> Optional[UUID]:
        """Get context ID if available."""
        return self.context.id if self.context else None

    def update_metadata(self, metadata: Dict[str, Any]) -> None:
        """
        Add metadata to be persisted with the context.

        Args:
            metadata: Key-value pairs to add to context metadata
        """
        self._metadata.update(metadata)

    async def _persist_metadata(self) -> None:
        """Persist accumulated metadata to the context."""
        if not self.enabled or not self.acd_service:
            return

        try:
            # Calculate duration
            duration = (datetime.now(timezone.utc) - self._start_time).total_seconds()
            self._metadata["duration_seconds"] = duration

            await self.acd_service.update_context(
                self.context.id,
                ACDContextUpdate(ai_metadata=self._metadata),
            )
        except Exception as e:
            logger.warning(f"Failed to persist ACD metadata: {e}")


class ACDEnforcer:
    """
    Central ACD enforcement wrapper for consistent context tracking.

    Ensures all operations are properly tracked in the ACD system
    with appropriate context creation, trace artifacts, and state management.
    """

    def __init__(self, db_session: AsyncSession):
        """
        Initialize ACD enforcer.

        Args:
            db_session: Database session for ACD operations
        """
        self.db = db_session
        self.enabled = settings.acd_enabled
        self._acd_service: Optional[ACDService] = None

    @property
    def acd_service(self) -> ACDService:
        """Lazy initialization of ACD service."""
        if self._acd_service is None:
            self._acd_service = ACDService(self.db)
        return self._acd_service

    @asynccontextmanager
    async def track_operation(
        self,
        phase: str,
        complexity: str = "MEDIUM",
        content_id: Optional[UUID] = None,
        persona_id: Optional[UUID] = None,
        **metadata: Any,
    ) -> AsyncGenerator[ACDContextWrapper, None]:
        """
        Context manager for ACD-tracked operations.

        Automatically creates context on entry, updates state on exit,
        and handles errors appropriately.

        Args:
            phase: AI phase (e.g., "IMAGE_GENERATION", "TEXT_GENERATION")
            complexity: Task complexity level
            content_id: Optional content ID being processed
            persona_id: Optional persona ID involved
            **metadata: Additional metadata to store

        Yields:
            ACDContextWrapper for the operation

        Example:
            async with enforcer.track_operation(
                phase="IMAGE_GENERATION",
                complexity="HIGH",
                persona_id=persona.id
            ) as ctx:
                result = await generate_image()
                ctx.update_metadata({"model": "sdxl", "steps": 30})
        """
        if not self.enabled:
            yield ACDContextWrapper(None, None, enabled=False)
            return

        context: Optional[ACDContextResponse] = None
        wrapper: Optional[ACDContextWrapper] = None

        try:
            # Create ACD context
            context = await self.acd_service.create_context(
                ACDContextCreate(
                    ai_phase=phase,
                    ai_complexity=complexity,
                    ai_state=AIState.PROCESSING,
                    ai_queue_status=AIQueueStatus.IN_PROGRESS,
                    content_id=content_id,
                    ai_started=datetime.now(timezone.utc),
                    ai_context={
                        "persona_id": str(persona_id) if persona_id else None,
                        "correlation_id": correlation_id.get("") or None,
                        **metadata,
                    },
                )
            )

            logger.debug(f"Created ACD context {context.id} for {phase}")

            wrapper = ACDContextWrapper(context, self.acd_service, enabled=True)
            yield wrapper

            # Success - update state to DONE
            await self.acd_service.update_context(
                context.id,
                ACDContextUpdate(
                    ai_state=AIState.DONE,
                    ai_queue_status=AIQueueStatus.COMPLETED,
                ),
            )

            # Persist any accumulated metadata
            await wrapper._persist_metadata()

            logger.debug(f"ACD context {context.id} completed successfully")

        except Exception as e:
            # Error - update state to FAILED and create trace artifact
            if context:
                try:
                    await self.acd_service.update_context(
                        context.id,
                        ACDContextUpdate(
                            ai_state=AIState.FAILED,
                            ai_queue_status=AIQueueStatus.ABANDONED,
                            runtime_err=str(e),
                        ),
                    )

                    # Create trace artifact for the error
                    await self.acd_service.create_trace_artifact(
                        ACDTraceArtifactCreate(
                            session_id=str(context.id),
                            event_type="operation_error",
                            error_message=str(e),
                            acd_context_id=context.id,
                            environment={
                                "phase": phase,
                                "complexity": complexity,
                                "correlation_id": correlation_id.get("") or None,
                            },
                        )
                    )

                    logger.error(f"ACD context {context.id} failed: {e}")

                except Exception as acd_error:
                    logger.error(f"Failed to update ACD context on error: {acd_error}")

            raise

    async def create_trace_artifact(
        self,
        context_id: Optional[UUID],
        event_type: str,
        message: str,
        **metadata: Any,
    ) -> None:
        """
        Create a trace artifact for debugging and auditing.

        Args:
            context_id: Optional ACD context ID
            event_type: Type of event (e.g., "decision", "handoff", "error")
            message: Human-readable message
            **metadata: Additional metadata to store
        """
        if not self.enabled:
            return

        try:
            corr = correlation_id.get("") or None
            await self.acd_service.create_trace_artifact(
                ACDTraceArtifactCreate(
                    session_id=str(context_id) if context_id else corr or "unknown",
                    event_type=event_type,
                    error_message=message,
                    acd_context_id=context_id,
                    environment={
                        "correlation_id": corr,
                        **metadata,
                    },
                )
            )
        except Exception as e:
            logger.warning(f"Failed to create trace artifact: {e}")


def acd_tracked(
    phase: str,
    complexity: str = "MEDIUM",
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """
    Decorator for ACD-tracked async functions.

    The decorated function must have a `db` parameter of type AsyncSession.

    Args:
        phase: AI phase for tracking
        complexity: Task complexity level

    Returns:
        Decorated function with ACD tracking

    Example:
        @acd_tracked(phase="TEXT_GENERATION", complexity="LOW")
        async def generate_caption(prompt: str, db: AsyncSession) -> str:
            ...
    """

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> T:
            # Find db session in args or kwargs
            db_session: Optional[AsyncSession] = None

            # Check kwargs first
            if "db" in kwargs:
                db_session = kwargs["db"]
            elif "db_session" in kwargs:
                db_session = kwargs["db_session"]
            else:
                # Check args for AsyncSession type
                for arg in args:
                    if isinstance(arg, AsyncSession):
                        db_session = arg
                        break

            if db_session is None:
                # Can't track without db session, just call function
                logger.warning(
                    f"@acd_tracked: No db session found for {func.__name__}, "
                    "skipping ACD tracking"
                )
                return await func(*args, **kwargs)

            enforcer = ACDEnforcer(db_session)
            async with enforcer.track_operation(phase=phase, complexity=complexity):
                return await func(*args, **kwargs)

        return wrapper

    return decorator
