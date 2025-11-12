"""
ACD (Autonomous Continuous Development) API Routes

Endpoints for managing ACD context metadata and trace artifacts.
"""

from typing import List, Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession

from backend.database.connection import get_db_session
from backend.services.acd_service import ACDService
from backend.models.acd import (
    ACDContextCreate,
    ACDContextUpdate,
    ACDContextResponse,
    ACDTraceArtifactCreate,
    ACDTraceArtifactResponse,
    ACDValidationReport,
    ACDStats,
)
from backend.config.logging import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/api/v1/acd", tags=["acd"])


@router.get("/contexts", response_model=List[ACDContextResponse])
async def list_contexts(
    limit: int = Query(10, ge=1, le=100, description="Maximum number of contexts to return"),
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
