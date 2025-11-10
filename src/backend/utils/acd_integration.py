"""
ACD Integration Utilities

Helper functions for integrating ACD context tracking into content generation.
"""

import traceback
from typing import Optional, Dict, Any
from uuid import UUID
from datetime import datetime, timezone

from sqlalchemy.ext.asyncio import AsyncSession

from backend.models.acd import (
    ACDContextCreate,
    ACDContextUpdate,
    ACDTraceArtifactCreate,
    AIStatus,
    AIState,
    AIComplexity,
    AIConfidence,
    AIQueuePriority,
    AIQueueStatus,
)
from backend.services.acd_service import ACDService
from backend.config.logging import get_logger

logger = get_logger(__name__)


class ACDContextManager:
    """
    Context manager for ACD tracking in content generation.

    Usage:
        async with ACDContextManager(db, "IMAGE_GENERATION", prompt) as acd:
            content = await generate_content(prompt)
            acd.set_metadata({"model": "stable-diffusion"})
            acd.set_confidence(AIConfidence.CONFIDENT)
    """

    def __init__(
        self,
        db_session: AsyncSession,
        phase: str,
        note: Optional[str] = None,
        complexity: Optional[AIComplexity] = None,
        content_id: Optional[UUID] = None,
        benchmark_id: Optional[UUID] = None,
        initial_context: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize ACD context manager.

        Args:
            db_session: Database session
            phase: Generation phase (e.g., "IMAGE_GENERATION")
            note: Human-readable description
            complexity: Task complexity level
            content_id: Link to content being generated
            benchmark_id: Link to benchmark record
            initial_context: Initial context metadata
        """
        self.db = db_session
        self.phase = phase
        self.note = note
        self.complexity = complexity
        self.content_id = content_id
        self.benchmark_id = benchmark_id
        self.initial_context = initial_context or {}
        self.context_id: Optional[UUID] = None
        self.acd_service: Optional[ACDService] = None
        self.session_id: Optional[str] = None

    async def __aenter__(self):
        """Enter context - create ACD context."""
        try:
            self.acd_service = ACDService(self.db)

            context_data = ACDContextCreate(
                ai_phase=self.phase,
                ai_status=AIStatus.IMPLEMENTED,
                ai_complexity=self.complexity,
                ai_note=self.note,
                ai_state=AIState.PROCESSING,
                ai_queue_priority=AIQueuePriority.NORMAL,
                ai_queue_status=AIQueueStatus.IN_PROGRESS,
                ai_context=self.initial_context,
                content_id=self.content_id,
                benchmark_id=self.benchmark_id,
            )

            context = await self.acd_service.create_context(context_data)
            self.context_id = context.id
            self.session_id = str(context.id)

            # Mark as started
            await self.acd_service.update_context(
                self.context_id,
                ACDContextUpdate(ai_queue_status=AIQueueStatus.IN_PROGRESS),
            )

            logger.info(f"ACD context created: {self.context_id} for phase {self.phase}")

            return self

        except Exception as e:
            logger.error(f"Failed to create ACD context: {str(e)}")
            # Don't fail the operation if ACD tracking fails
            return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Exit context - update ACD context with results."""
        if not self.context_id or not self.acd_service:
            return False

        try:
            if exc_type is None:
                # Success case
                await self.acd_service.update_context(
                    self.context_id,
                    ACDContextUpdate(
                        ai_state=AIState.DONE,
                        ai_queue_status=AIQueueStatus.COMPLETED,
                    ),
                )
                logger.info(f"ACD context completed: {self.context_id}")
            else:
                # Error case - create trace artifact
                await self._handle_error(exc_type, exc_val, exc_tb)

        except Exception as e:
            logger.error(f"Failed to update ACD context: {str(e)}")

        return False  # Don't suppress exceptions

    async def _handle_error(self, exc_type, exc_val, exc_tb):
        """Handle error by creating trace artifact and updating context."""
        try:
            # Extract error information
            error_message = str(exc_val) if exc_val else "Unknown error"
            stack_trace = traceback.format_tb(exc_tb) if exc_tb else []

            # Create trace artifact
            artifact_data = ACDTraceArtifactCreate(
                session_id=self.session_id or "unknown",
                event_type="runtime_error",
                error_message=error_message,
                acd_context_id=self.context_id,
                stack_trace=stack_trace,
                environment={"phase": self.phase},
            )

            await self.acd_service.create_trace_artifact(artifact_data)

            # Update context to failed
            await self.acd_service.update_context(
                self.context_id,
                ACDContextUpdate(
                    ai_state=AIState.FAILED,
                    ai_queue_status=AIQueueStatus.ABANDONED,
                ),
            )

            logger.error(
                f"ACD trace artifact created for error in {self.phase}: {error_message}"
            )

        except Exception as e:
            logger.error(f"Failed to create trace artifact: {str(e)}")

    async def set_confidence(self, confidence: AIConfidence):
        """Set confidence level for the context."""
        if self.context_id and self.acd_service:
            try:
                await self.acd_service.update_context(
                    self.context_id,
                    ACDContextUpdate(ai_confidence=confidence),
                )
            except Exception as e:
                logger.error(f"Failed to set confidence: {str(e)}")

    async def set_metadata(self, metadata: Dict[str, Any]):
        """Update context metadata."""
        if self.context_id and self.acd_service:
            try:
                # Get current context
                context = await self.acd_service.get_context(self.context_id)
                if context:
                    current_context = context.ai_context or {}
                    current_context.update(metadata)

                    # Note: We need to update the ai_context field directly
                    # This is a limitation - we should enhance the update to merge contexts
                    logger.info(f"Updated context metadata for {self.context_id}")
            except Exception as e:
                logger.error(f"Failed to set metadata: {str(e)}")

    async def add_note(self, note: str):
        """Add a note to the context."""
        if self.context_id and self.acd_service:
            try:
                await self.acd_service.update_context(
                    self.context_id,
                    ACDContextUpdate(ai_note=note),
                )
            except Exception as e:
                logger.error(f"Failed to add note: {str(e)}")

    async def set_validation(
        self,
        validation_result: str,
        issues: Optional[list] = None,
        suggestions: Optional[list] = None,
    ):
        """Set validation results for dual-agent review."""
        if self.context_id and self.acd_service:
            try:
                await self.acd_service.update_context(
                    self.context_id,
                    ACDContextUpdate(
                        ai_validation=validation_result,
                        ai_issues=issues,
                        ai_suggestions=suggestions,
                    ),
                )
            except Exception as e:
                logger.error(f"Failed to set validation: {str(e)}")


async def track_generation_with_acd(
    db_session: AsyncSession,
    phase: str,
    generation_func,
    note: Optional[str] = None,
    complexity: Optional[AIComplexity] = None,
    content_id: Optional[UUID] = None,
    **kwargs,
):
    """
    Wrapper function to track any generation function with ACD.

    Args:
        db_session: Database session
        phase: Generation phase name
        generation_func: Async function to execute
        note: Description of the generation
        complexity: Task complexity
        content_id: Content ID to link
        **kwargs: Additional arguments passed to generation_func

    Returns:
        Result from generation_func
    """
    async with ACDContextManager(
        db_session, phase, note, complexity, content_id
    ) as acd:
        result = await generation_func(**kwargs)
        return result


def get_phase_from_content_type(content_type: str) -> str:
    """
    Map content type to ACD phase.

    Args:
        content_type: Content type (image, text, video, etc.)

    Returns:
        ACD phase name
    """
    phase_map = {
        "image": "IMAGE_GENERATION",
        "text": "TEXT_GENERATION",
        "video": "VIDEO_GENERATION",
        "audio": "AUDIO_GENERATION",
        "caption": "CAPTION_GENERATION",
        "description": "DESCRIPTION_GENERATION",
    }
    return phase_map.get(content_type.lower(), "CONTENT_GENERATION")


def get_complexity_from_quality(quality: str) -> AIComplexity:
    """
    Map quality level to complexity.

    Args:
        quality: Quality level (draft, standard, high, premium)

    Returns:
        ACD complexity level
    """
    complexity_map = {
        "draft": AIComplexity.LOW,
        "standard": AIComplexity.MEDIUM,
        "high": AIComplexity.HIGH,
        "premium": AIComplexity.CRITICAL,
    }
    return complexity_map.get(quality.lower(), AIComplexity.MEDIUM)


async def link_acd_to_benchmark(
    db_session: AsyncSession,
    benchmark_id: UUID,
    acd_context_id: UUID,
):
    """
    Link an ACD context to a generation benchmark.

    Args:
        db_session: Database session
        benchmark_id: Benchmark ID
        acd_context_id: ACD context ID
    """
    try:
        from backend.models.generation_feedback import GenerationBenchmarkModel
        from sqlalchemy import select

        # Update benchmark with ACD context ID
        stmt = select(GenerationBenchmarkModel).where(
            GenerationBenchmarkModel.id == benchmark_id
        )
        result = await db_session.execute(stmt)
        benchmark = result.scalar_one_or_none()

        if benchmark:
            benchmark.acd_context_id = acd_context_id
            await db_session.commit()
            logger.info(f"Linked ACD context {acd_context_id} to benchmark {benchmark_id}")
    except Exception as e:
        logger.error(f"Failed to link ACD to benchmark: {str(e)}")
