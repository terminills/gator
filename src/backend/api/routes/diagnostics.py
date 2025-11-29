"""
Diagnostics API Routes

Comprehensive diagnostics for AI activity and content generation.
Shows real-time AI agent activity, generation attempts, success/failure rates.
"""

from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy import and_, desc, func, or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.config.logging import get_logger
from backend.database.connection import get_db_session
from backend.models.acd import ACDContextModel, ACDTraceArtifactModel
from backend.models.content import ContentModel
from backend.models.persona import PersonaModel
from backend.services.acd_service import ACDService

logger = get_logger(__name__)

router = APIRouter(
    prefix="/api/v1/diagnostics",
    tags=["diagnostics"],
)


class AIActivitySummary(BaseModel):
    """Summary of AI activity."""

    total_generations: int = 0
    successful_generations: int = 0
    failed_generations: int = 0
    fallback_generations: int = 0
    success_rate: float = 0.0

    by_content_type: Dict[str, int] = {}
    by_persona: Dict[str, int] = {}

    active_contexts: int = 0
    completed_contexts: int = 0
    failed_contexts: int = 0

    recent_errors: List[Dict[str, Any]] = []


class ContentGenerationAttempt(BaseModel):
    """Detailed information about a content generation attempt."""

    content_id: Optional[UUID]
    persona_id: UUID
    persona_name: str
    content_type: str
    status: str
    created_at: datetime

    # ACD context details
    has_acd_context: bool = False
    acd_context_id: Optional[UUID] = None
    ai_phase: Optional[str] = None
    ai_state: Optional[str] = None
    ai_confidence: Optional[str] = None
    ai_note: Optional[str] = None

    # Generation details
    prompt: Optional[str] = None
    quality: Optional[str] = None
    using_fallback: bool = False
    error_message: Optional[str] = None
    file_path: Optional[str] = None


class AIModelActivity(BaseModel):
    """Activity for a specific AI model."""

    model_name: str
    provider: str
    total_calls: int = 0
    successful_calls: int = 0
    failed_calls: int = 0
    avg_generation_time: Optional[float] = None


@router.get("/ai-activity", status_code=200)
async def get_ai_activity(
    hours: int = Query(24, ge=1, le=168, description="Time window in hours"),
    db: AsyncSession = Depends(get_db_session),
) -> AIActivitySummary:
    """
    Get comprehensive AI activity summary.

    Shows all AI generation activity including:
    - Total generations (successful, failed, fallback)
    - Breakdown by content type and persona
    - Active/completed/failed ACD contexts
    - Recent errors with details

    Args:
        hours: Time window in hours (1-168, default 24)
        db: Database session

    Returns:
        Comprehensive AI activity summary
    """
    try:
        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=hours)

        # Get content generation statistics
        content_stmt = select(ContentModel).where(
            ContentModel.created_at >= cutoff_time
        )
        content_result = await db.execute(content_stmt)
        contents = content_result.scalars().all()

        total_generations = len(contents)
        successful = 0
        failed = 0
        fallback = 0
        by_content_type = {}
        by_persona = {}

        for content in contents:
            # Count by type
            ct = content.content_type
            by_content_type[ct] = by_content_type.get(ct, 0) + 1

            # Count by persona
            persona_id_str = str(content.persona_id)
            by_persona[persona_id_str] = by_persona.get(persona_id_str, 0) + 1

            # Determine if successful, failed, or fallback
            gen_params = content.generation_params or {}

            # Check if this used a fallback (template-based generation)
            is_fallback = gen_params.get("fallback", False) or gen_params.get(
                "template_based", False
            )

            # Check if there was an error (but fallback succeeded)
            has_error = gen_params.get("error") or gen_params.get("fallback_reason")

            if is_fallback:
                fallback += 1
            elif has_error:
                failed += 1
            else:
                successful += 1

        # Get ACD context statistics
        acd_stmt = select(ACDContextModel).where(
            ACDContextModel.created_at >= cutoff_time
        )
        acd_result = await db.execute(acd_stmt)
        acd_contexts = acd_result.scalars().all()

        active_contexts = 0
        completed_contexts = 0
        failed_contexts = 0

        for context in acd_contexts:
            if context.ai_state in ["PROCESSING", "READY", "BLOCKED", "PAUSED"]:
                active_contexts += 1
            elif context.ai_state == "DONE":
                completed_contexts += 1
            elif context.ai_state in ["FAILED", "CANCELLED"]:
                failed_contexts += 1

        # Get recent errors from trace artifacts
        trace_stmt = (
            select(ACDTraceArtifactModel)
            .where(ACDTraceArtifactModel.timestamp >= cutoff_time)
            .order_by(desc(ACDTraceArtifactModel.timestamp))
            .limit(10)
        )
        trace_result = await db.execute(trace_stmt)
        traces = trace_result.scalars().all()

        recent_errors = []
        for trace in traces:
            recent_errors.append(
                {
                    "timestamp": trace.timestamp.isoformat(),
                    "event_type": trace.event_type,
                    "error_message": trace.error_message,
                    "error_file": trace.error_file,
                    "error_line": trace.error_line,
                    "acd_context_id": (
                        str(trace.acd_context_id) if trace.acd_context_id else None
                    ),
                }
            )

        # Calculate success rate
        success_rate = (
            (successful / total_generations * 100) if total_generations > 0 else 0.0
        )

        return AIActivitySummary(
            total_generations=total_generations,
            successful_generations=successful,
            failed_generations=failed,
            fallback_generations=fallback,
            success_rate=round(success_rate, 2),
            by_content_type=by_content_type,
            by_persona=by_persona,
            active_contexts=active_contexts,
            completed_contexts=completed_contexts,
            failed_contexts=failed_contexts,
            recent_errors=recent_errors,
        )

    except Exception as e:
        logger.error(f"Failed to get AI activity summary: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Failed to get AI activity: {str(e)}"
        )


@router.get("/generation-attempts", status_code=200)
async def get_generation_attempts(
    hours: int = Query(24, ge=1, le=168, description="Time window in hours"),
    content_type: Optional[str] = Query(None, description="Filter by content type"),
    persona_id: Optional[UUID] = Query(None, description="Filter by persona"),
    status: Optional[str] = Query(
        None, description="Filter by status (success, failed, fallback)"
    ),
    limit: int = Query(50, ge=1, le=200, description="Maximum results"),
    db: AsyncSession = Depends(get_db_session),
) -> List[ContentGenerationAttempt]:
    """
    Get detailed list of content generation attempts.

    Shows every generation attempt with full context:
    - Content details (type, persona, prompt)
    - ACD context (state, confidence, notes)
    - Success/failure/fallback status
    - Error messages if failed

    Args:
        hours: Time window in hours
        content_type: Optional filter by content type
        persona_id: Optional filter by persona
        status: Optional filter by status
        limit: Maximum results to return
        db: Database session

    Returns:
        List of generation attempts with full details
    """
    try:
        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=hours)

        # Build query
        stmt = (
            select(ContentModel, PersonaModel)
            .join(PersonaModel, ContentModel.persona_id == PersonaModel.id)
            .where(ContentModel.created_at >= cutoff_time)
        )

        if content_type:
            stmt = stmt.where(ContentModel.content_type == content_type)

        if persona_id:
            stmt = stmt.where(ContentModel.persona_id == persona_id)

        stmt = stmt.order_by(desc(ContentModel.created_at)).limit(limit)

        result = await db.execute(stmt)
        rows = result.all()

        attempts = []
        for content, persona in rows:
            gen_params = content.generation_params or {}

            # Determine status
            has_error = bool(gen_params.get("error"))
            is_fallback = gen_params.get("fallback", False) or gen_params.get(
                "template_based", False
            )

            if has_error and not is_fallback:
                attempt_status = "failed"
            elif is_fallback:
                attempt_status = "fallback"
            else:
                attempt_status = "success"

            # Filter by status if requested
            if status and attempt_status != status:
                continue

            # Get ACD context ID from generation params
            acd_context_id = gen_params.get("acd_context_id")

            # Fetch ACD context details if available
            acd_context = None
            if acd_context_id:
                try:
                    acd_service = ACDService(db)
                    from uuid import UUID as UUIDType

                    acd_id = (
                        UUIDType(acd_context_id)
                        if isinstance(acd_context_id, str)
                        else acd_context_id
                    )
                    acd_context = await acd_service.get_context(acd_id)
                except Exception as e:
                    logger.warning(
                        f"Failed to fetch ACD context {acd_context_id}: {str(e)}"
                    )

            attempts.append(
                ContentGenerationAttempt(
                    content_id=content.id,
                    persona_id=persona.id,
                    persona_name=persona.name,
                    content_type=content.content_type,
                    status=attempt_status,
                    created_at=content.created_at,
                    has_acd_context=bool(acd_context),
                    acd_context_id=acd_context.id if acd_context else None,
                    ai_phase=acd_context.ai_phase if acd_context else None,
                    ai_state=acd_context.ai_state if acd_context else None,
                    ai_confidence=acd_context.ai_confidence if acd_context else None,
                    ai_note=acd_context.ai_note if acd_context else None,
                    prompt=gen_params.get("prompt"),
                    quality=gen_params.get("quality"),
                    using_fallback=is_fallback,
                    error_message=gen_params.get("error"),
                    file_path=content.file_path,
                )
            )

        return attempts

    except Exception as e:
        logger.error(f"Failed to get generation attempts: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Failed to get generation attempts: {str(e)}"
        )


@router.get("/ai-models", status_code=200)
async def get_ai_model_activity(
    hours: int = Query(24, ge=1, le=168, description="Time window in hours"),
    db: AsyncSession = Depends(get_db_session),
) -> List[AIModelActivity]:
    """
    Get AI model usage statistics.

    Shows which AI models are being called and their performance:
    - Model name and provider
    - Total calls (successful/failed)
    - Average generation time

    Args:
        hours: Time window in hours
        db: Database session

    Returns:
        List of AI model activity statistics
    """
    try:
        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=hours)

        # Get all content with model information
        stmt = select(ContentModel).where(ContentModel.created_at >= cutoff_time)
        result = await db.execute(stmt)
        contents = result.scalars().all()

        # Aggregate by model
        model_stats = {}

        for content in contents:
            gen_params = content.generation_params or {}
            model_name = gen_params.get("model", "unknown")
            provider = gen_params.get("provider", "unknown")

            key = f"{provider}:{model_name}"

            if key not in model_stats:
                model_stats[key] = {
                    "model_name": model_name,
                    "provider": provider,
                    "total_calls": 0,
                    "successful_calls": 0,
                    "failed_calls": 0,
                    "total_time": 0.0,
                    "time_count": 0,
                }

            stats = model_stats[key]
            stats["total_calls"] += 1

            if gen_params.get("error"):
                stats["failed_calls"] += 1
            else:
                stats["successful_calls"] += 1

            # Track generation time if available
            gen_time = gen_params.get("generation_time_seconds") or gen_params.get(
                "total_time_seconds"
            )
            if gen_time:
                stats["total_time"] += gen_time
                stats["time_count"] += 1

        # Convert to response format
        activities = []
        for stats in model_stats.values():
            avg_time = None
            if stats["time_count"] > 0:
                avg_time = round(stats["total_time"] / stats["time_count"], 2)

            activities.append(
                AIModelActivity(
                    model_name=stats["model_name"],
                    provider=stats["provider"],
                    total_calls=stats["total_calls"],
                    successful_calls=stats["successful_calls"],
                    failed_calls=stats["failed_calls"],
                    avg_generation_time=avg_time,
                )
            )

        # Sort by total calls
        activities.sort(key=lambda x: x.total_calls, reverse=True)

        return activities

    except Exception as e:
        logger.error(f"Failed to get AI model activity: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Failed to get AI model activity: {str(e)}"
        )


@router.get("/acd-contexts", status_code=200)
async def get_acd_contexts(
    hours: int = Query(24, ge=1, le=168, description="Time window in hours"),
    phase: Optional[str] = Query(None, description="Filter by phase"),
    state: Optional[str] = Query(None, description="Filter by state"),
    limit: int = Query(50, ge=1, le=200, description="Maximum results"),
    db: AsyncSession = Depends(get_db_session),
):
    """
    Get ACD contexts with full details.

    Shows all AI agent contexts including:
    - Phase, state, confidence
    - Context metadata and notes
    - Linked content
    - Timestamps

    Args:
        hours: Time window in hours
        phase: Optional filter by phase
        state: Optional filter by state
        limit: Maximum results
        db: Database session

    Returns:
        List of ACD contexts
    """
    try:
        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=hours)

        # Build query
        stmt = select(ACDContextModel).where(ACDContextModel.created_at >= cutoff_time)

        if phase:
            stmt = stmt.where(ACDContextModel.ai_phase == phase)

        if state:
            stmt = stmt.where(ACDContextModel.ai_state == state)

        stmt = stmt.order_by(desc(ACDContextModel.created_at)).limit(limit)

        result = await db.execute(stmt)
        contexts = result.scalars().all()

        # Convert to response format
        acd_service = ACDService(db)
        response_contexts = []
        for context in contexts:
            from backend.models.acd import ACDContextResponse

            response_contexts.append(ACDContextResponse.model_validate(context))

        return response_contexts

    except Exception as e:
        logger.error(f"Failed to get ACD contexts: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Failed to get ACD contexts: {str(e)}"
        )
