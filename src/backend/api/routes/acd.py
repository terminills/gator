"""
ACD (Autonomous Continuous Development) API Routes

Endpoints for managing ACD context metadata and trace artifacts.
"""

from typing import List, Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession

from backend.config.logging import get_logger
from backend.database.connection import get_db_session
from backend.models.acd import (
    ACDContextCreate,
    ACDContextResponse,
    ACDContextUpdate,
    ACDStats,
    ACDTraceArtifactCreate,
    ACDTraceArtifactResponse,
    ACDValidationReport,
    GenerationRating,
    HILRatingCreate,
    HILRatingResponse,
    HILRatingStats,
    LoRAIncompatibilityFlag,
    MisgenerationPattern,
    MisgenerationTag,
    RecommendedConfiguration,
    WorkflowEffectiveness,
)
from backend.services.acd_service import ACDService
from backend.services.hil_rating_service import HILRatingService

logger = get_logger(__name__)

router = APIRouter(prefix="/api/v1/acd", tags=["acd"])


@router.get("/contexts", response_model=List[ACDContextResponse])
async def list_contexts(
    limit: int = Query(
        10, ge=1, le=100, description="Maximum number of contexts to return"
    ),
    offset: int = Query(0, ge=0, description="Number of contexts to skip"),
    db: AsyncSession = Depends(get_db_session),
):
    """
    List ACD contexts with pagination.

    Args:
        limit: Maximum number of contexts to return (1-100)
        offset: Number of contexts to skip

    Returns:
        List of context records
    """
    try:
        service = ACDService(db)
        contexts = await service.list_contexts(limit=limit, offset=offset)
        return contexts
    except Exception as e:
        logger.error(f"Failed to list ACD contexts: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/contexts/", response_model=ACDContextResponse, status_code=201)
async def create_context(
    context_data: ACDContextCreate,
    db: AsyncSession = Depends(get_db_session),
):
    """
    Create a new ACD context for content generation.

    Args:
        context_data: Context metadata

    Returns:
        Created context record
    """
    try:
        service = ACDService(db)
        context = await service.create_context(context_data)
        return context
    except Exception as e:
        logger.error(f"Failed to create ACD context: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/contexts/{context_id}", response_model=ACDContextResponse)
async def get_context(
    context_id: UUID,
    db: AsyncSession = Depends(get_db_session),
):
    """
    Get an ACD context by ID.

    Args:
        context_id: UUID of the context

    Returns:
        Context record
    """
    try:
        service = ACDService(db)
        context = await service.get_context(context_id)
        if not context:
            raise HTTPException(status_code=404, detail="Context not found")
        return context
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get ACD context: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/contexts/{context_id}", response_model=ACDContextResponse)
async def update_context(
    context_id: UUID,
    update_data: ACDContextUpdate,
    db: AsyncSession = Depends(get_db_session),
):
    """
    Update an ACD context.

    Args:
        context_id: UUID of the context
        update_data: Fields to update

    Returns:
        Updated context record
    """
    try:
        service = ACDService(db)
        context = await service.update_context(context_id, update_data)
        if not context:
            raise HTTPException(status_code=404, detail="Context not found")
        return context
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to update ACD context: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/contexts/benchmark/{benchmark_id}", response_model=ACDContextResponse)
async def get_context_by_benchmark(
    benchmark_id: UUID,
    db: AsyncSession = Depends(get_db_session),
):
    """
    Get ACD context associated with a benchmark.

    Args:
        benchmark_id: UUID of the benchmark

    Returns:
        Context record
    """
    try:
        service = ACDService(db)
        context = await service.get_context_by_benchmark(benchmark_id)
        if not context:
            raise HTTPException(status_code=404, detail="Context not found")
        return context
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get ACD context: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/contexts/content/{content_id}", response_model=ACDContextResponse)
async def get_context_by_content(
    content_id: UUID,
    db: AsyncSession = Depends(get_db_session),
):
    """
    Get ACD context associated with content.

    Args:
        content_id: UUID of the content

    Returns:
        Context record
    """
    try:
        service = ACDService(db)
        context = await service.get_context_by_content(content_id)
        if not context:
            raise HTTPException(status_code=404, detail="Context not found")
        return context
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get ACD context: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post(
    "/contexts/{context_id}/assign",
    response_model=ACDContextResponse,
)
async def assign_context_to_agent(
    context_id: UUID,
    agent_name: str = Query(..., description="Name of the agent to assign"),
    reason: Optional[str] = Query(None, description="Reason for assignment"),
    db: AsyncSession = Depends(get_db_session),
):
    """
    Assign a context to an agent.

    Args:
        context_id: UUID of the context
        agent_name: Name of the agent
        reason: Reason for assignment

    Returns:
        Updated context record
    """
    try:
        service = ACDService(db)
        context = await service.assign_to_agent(context_id, agent_name, reason)
        if not context:
            raise HTTPException(status_code=404, detail="Context not found")
        return context
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to assign context: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post(
    "/trace-artifacts/", response_model=ACDTraceArtifactResponse, status_code=201
)
async def create_trace_artifact(
    artifact_data: ACDTraceArtifactCreate,
    db: AsyncSession = Depends(get_db_session),
):
    """
    Create a trace artifact for error tracking.

    Args:
        artifact_data: Trace artifact data

    Returns:
        Created trace artifact
    """
    try:
        service = ACDService(db)
        artifact = await service.create_trace_artifact(artifact_data)
        return artifact
    except Exception as e:
        logger.error(f"Failed to create trace artifact: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get(
    "/trace-artifacts/session/{session_id}",
    response_model=List[ACDTraceArtifactResponse],
)
async def get_trace_artifacts_by_session(
    session_id: str,
    db: AsyncSession = Depends(get_db_session),
):
    """
    Get all trace artifacts for a session.

    Args:
        session_id: Session identifier

    Returns:
        List of trace artifacts
    """
    try:
        service = ACDService(db)
        artifacts = await service.get_trace_artifacts_by_session(session_id)
        return artifacts
    except Exception as e:
        logger.error(f"Failed to get trace artifacts: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/stats/", response_model=ACDStats)
async def get_acd_stats(
    hours: int = Query(24, ge=1, le=168, description="Time window in hours"),
    phase: Optional[str] = Query(None, description="Filter by phase"),
    db: AsyncSession = Depends(get_db_session),
):
    """
    Get ACD context statistics.

    Args:
        hours: Time window in hours (1-168)
        phase: Optional phase filter

    Returns:
        Aggregate statistics
    """
    try:
        service = ACDService(db)
        stats = await service.get_stats(hours, phase)
        return stats
    except Exception as e:
        logger.error(f"Failed to get ACD stats: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/validation-report/", response_model=ACDValidationReport)
async def get_validation_report(
    db: AsyncSession = Depends(get_db_session),
):
    """
    Generate a validation report for all ACD contexts.

    Returns:
        Validation report with metadata, contexts, and issues
    """
    try:
        service = ACDService(db)
        report = await service.generate_validation_report()
        return report
    except Exception as e:
        logger.error(f"Failed to generate validation report: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/process-queue/", status_code=202)
async def process_queue(
    max_contexts: int = Query(
        10, ge=1, le=100, description="Maximum contexts to process"
    ),
    phase: Optional[str] = Query(None, description="Filter by phase"),
    db: AsyncSession = Depends(get_db_session),
):
    """
    Process queued ACD contexts and trigger content generation.

    This endpoint actually triggers content generation for queued ACD contexts,
    fixing the issue where ACD only logged to database without generating content.

    Args:
        max_contexts: Maximum number of contexts to process (1-100)
        phase: Optional phase filter (e.g., "IMAGE_GENERATION", "TEXT_GENERATION")

    Returns:
        Processing results with success/failure counts
    """
    try:
        service = ACDService(db)
        results = await service.process_queued_contexts(
            max_contexts=max_contexts, phase_filter=phase
        )
        return {
            "status": "processing_complete",
            "summary": results,
        }
    except Exception as e:
        logger.error(f"Failed to process queue: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================
# HIL (Human-in-the-Loop) Rating Endpoints
# ============================================================


@router.post("/rate/{context_id}", response_model=HILRatingResponse, tags=["hil"])
async def rate_generation(
    context_id: UUID,
    rating_data: HILRatingCreate,
    db: AsyncSession = Depends(get_db_session),
):
    """
    Submit a human rating for generated content.

    This rating is stored and used to:
    - Train the correlation engine on quality patterns
    - Build a knowledge base of working configurations
    - Identify problematic LoRA/model combinations

    Args:
        context_id: UUID of the ACD context to rate
        rating_data: Rating data including score, tags, and notes

    Returns:
        HILRatingResponse with the saved rating information
    """
    try:
        service = HILRatingService(db)
        result = await service.rate_generation(context_id, rating_data)
        return result
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to rate generation: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/ratings", response_model=List[HILRatingResponse], tags=["hil"])
async def get_recent_ratings(
    limit: int = Query(50, ge=1, le=100, description="Maximum ratings to return"),
    rating: Optional[GenerationRating] = Query(
        None, description="Filter by specific rating"
    ),
    db: AsyncSession = Depends(get_db_session),
):
    """
    Get recent ratings.

    Args:
        limit: Maximum number of ratings to return
        rating: Optional filter by specific rating level

    Returns:
        List of recent ratings
    """
    try:
        service = HILRatingService(db)
        results = await service.get_recent_ratings(limit=limit, rating_filter=rating)
        return results
    except Exception as e:
        logger.error(f"Failed to get ratings: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/ratings/stats", response_model=HILRatingStats, tags=["hil"])
async def get_rating_stats(
    time_window_hours: int = Query(
        168, ge=1, description="Time window in hours (default 1 week)"
    ),
    db: AsyncSession = Depends(get_db_session),
):
    """
    Get overall HIL rating statistics.

    Returns statistics including:
    - Total rated vs unrated contexts
    - Rating distribution
    - Average rating
    - Most common misgeneration tags
    - Breakdown by model and domain

    Args:
        time_window_hours: Time window to analyze

    Returns:
        HILRatingStats with overall statistics
    """
    try:
        service = HILRatingService(db)
        stats = await service.get_rating_stats(time_window_hours=time_window_hours)
        return stats
    except Exception as e:
        logger.error(f"Failed to get rating stats: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get(
    "/workflow-effectiveness", response_model=WorkflowEffectiveness, tags=["hil"]
)
async def get_workflow_effectiveness(
    workflow_id: Optional[str] = Query(None, description="Workflow ID to analyze"),
    model_id: Optional[str] = Query(None, description="Model ID to analyze"),
    lora_ids: Optional[str] = Query(
        None, description="Comma-separated list of LoRA IDs"
    ),
    time_window_hours: int = Query(
        720, ge=1, description="Time window in hours (default 30 days)"
    ),
    db: AsyncSession = Depends(get_db_session),
):
    """
    Analyze effectiveness of specific workflows/models/LoRAs.

    Returns:
    - Average rating for this configuration
    - Common misgeneration tags
    - Recommended alternatives
    - Success rate over time

    Args:
        workflow_id: Optional workflow to filter by
        model_id: Optional model to filter by
        lora_ids: Optional comma-separated list of LoRAs to filter by
        time_window_hours: Time window to analyze (default 30 days)

    Returns:
        WorkflowEffectiveness analysis
    """
    try:
        # Parse lora_ids if provided
        parsed_lora_ids = None
        if lora_ids:
            parsed_lora_ids = [lid.strip() for lid in lora_ids.split(",")]

        service = HILRatingService(db)
        result = await service.get_workflow_effectiveness(
            workflow_id=workflow_id,
            model_id=model_id,
            lora_ids=parsed_lora_ids,
            time_window_hours=time_window_hours,
        )
        return result
    except Exception as e:
        logger.error(f"Failed to get workflow effectiveness: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get(
    "/best-configs",
    response_model=List[RecommendedConfiguration],
    tags=["hil"],
)
async def get_best_configurations(
    content_type: Optional[str] = Query(None, description="Content type filter"),
    style: Optional[str] = Query(None, description="Style filter"),
    min_rating: float = Query(
        4.0, ge=1.0, le=5.0, description="Minimum average rating threshold"
    ),
    limit: int = Query(10, ge=1, le=50, description="Maximum configurations to return"),
    db: AsyncSession = Depends(get_db_session),
):
    """
    Get best-rated configurations for content generation.

    Returns configurations that consistently produce highly-rated content,
    learned from HIL feedback.

    Args:
        content_type: Optional content type to filter by
        style: Optional style to filter by
        min_rating: Minimum average rating threshold
        limit: Maximum number of configurations to return

    Returns:
        List of recommended configurations
    """
    try:
        service = HILRatingService(db)
        results = await service.get_best_configurations(
            content_type=content_type,
            style=style,
            min_rating=min_rating,
            limit=limit,
        )
        return results
    except Exception as e:
        logger.error(f"Failed to get best configurations: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/flag-incompatibility", tags=["hil"])
async def flag_lora_incompatibility(
    flag_data: LoRAIncompatibilityFlag,
    db: AsyncSession = Depends(get_db_session),
):
    """
    Flag incompatible LoRA combinations discovered through HIL.

    System learns to avoid these combinations in future generations.

    Args:
        flag_data: LoRA incompatibility flag data

    Returns:
        Confirmation of the flag being recorded
    """
    try:
        service = HILRatingService(db)
        result = await service.flag_lora_incompatibility(flag_data)
        return result
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to flag incompatibility: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get(
    "/misgeneration-patterns",
    response_model=List[MisgenerationPattern],
    tags=["hil"],
)
async def get_misgeneration_patterns(
    tag: Optional[MisgenerationTag] = Query(
        None, description="Specific tag to analyze"
    ),
    time_window_hours: int = Query(
        168, ge=1, description="Time window in hours (default 1 week)"
    ),
    db: AsyncSession = Depends(get_db_session),
):
    """
    Analyze patterns in misgenerated content.

    Identifies:
    - Common causes of specific misgeneration types
    - Correlations between settings and failures
    - Trends over time (improving or degrading)

    Args:
        tag: Optional specific tag to analyze
        time_window_hours: Time window to analyze

    Returns:
        List of misgeneration patterns
    """
    try:
        service = HILRatingService(db)
        results = await service.get_misgeneration_patterns(
            tag=tag, time_window_hours=time_window_hours
        )
        return results
    except Exception as e:
        logger.error(f"Failed to get misgeneration patterns: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
