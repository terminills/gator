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
    MemoryCreate,
    MemoryRecallRequest,
    MemoryResponse,
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


# ============================================================
# Correlation Engine Endpoints
# ============================================================


@router.get("/correlations/similar/{context_id}", tags=["correlations"])
async def find_similar_contexts(
    context_id: UUID,
    similarity_threshold: float = Query(
        0.7, ge=0.0, le=1.0, description="Minimum similarity score"
    ),
    max_results: int = Query(10, ge=1, le=50, description="Maximum results to return"),
    time_window_hours: Optional[int] = Query(
        None, ge=1, description="Optional time window in hours"
    ),
    db: AsyncSession = Depends(get_db_session),
):
    """
    Find historically similar contexts to inform decisions.

    Uses domain/subdomain matching, phase matching, model/workflow matching,
    and outcome correlation to find similar contexts.

    Args:
        context_id: UUID of the context to find similar ones for
        similarity_threshold: Minimum similarity score (0-1)
        max_results: Maximum number of results
        time_window_hours: Optional time window filter

    Returns:
        List of similar contexts with similarity scores
    """
    from backend.services.acd_correlation_engine import ACDCorrelationEngine

    try:
        # Get the source context
        service = ACDService(db)
        source_context = await service.get_context(context_id)
        if not source_context:
            raise HTTPException(status_code=404, detail="Context not found")

        # Get source context model for correlation engine
        from sqlalchemy import select
        from backend.models.acd import ACDContextModel

        stmt = select(ACDContextModel).where(ACDContextModel.id == context_id)
        result = await db.execute(stmt)
        context_model = result.scalar_one_or_none()

        if not context_model:
            raise HTTPException(status_code=404, detail="Context not found")

        correlation_engine = ACDCorrelationEngine(db)
        similar = await correlation_engine.find_similar_contexts(
            context_model,
            similarity_threshold=similarity_threshold,
            max_results=max_results,
            time_window_hours=time_window_hours,
        )

        return {
            "source_context_id": str(context_id),
            "similar_contexts": similar,
            "count": len(similar),
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to find similar contexts: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/correlations/patterns", tags=["correlations"])
async def get_success_patterns(
    domain: Optional[str] = Query(None, description="Domain filter"),
    time_window_hours: int = Query(
        168, ge=1, description="Time window in hours (default 1 week)"
    ),
    min_rating: int = Query(
        4, ge=1, le=5, description="Minimum HIL rating for success"
    ),
    db: AsyncSession = Depends(get_db_session),
):
    """
    Analyze successful contexts to extract winning patterns.

    Returns:
    - Common prompt structures
    - Optimal parameter combinations
    - Best-performing model/quality settings
    - Recommendations based on patterns

    Args:
        domain: Optional domain filter
        time_window_hours: Time window for analysis
        min_rating: Minimum HIL rating to consider successful

    Returns:
        Success patterns and recommendations
    """
    from backend.services.acd_correlation_engine import ACDCorrelationEngine
    from backend.models.acd import AIDomain

    try:
        correlation_engine = ACDCorrelationEngine(db)

        # Convert domain string to enum if provided
        domain_enum = None
        if domain:
            try:
                domain_enum = AIDomain(domain)
            except ValueError:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid domain: {domain}. Valid domains: {[d.value for d in AIDomain]}"
                )

        patterns = await correlation_engine.extract_success_patterns(
            domain=domain_enum,
            time_window_hours=time_window_hours,
            min_rating=min_rating,
        )

        return patterns

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get success patterns: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/correlations/learn/{context_id}", tags=["correlations"])
async def learn_from_outcome(
    context_id: UUID,
    outcome: dict,
    db: AsyncSession = Depends(get_db_session),
):
    """
    Update ACD's knowledge base from generation outcomes.

    Learns:
    - What prompts lead to high engagement
    - Which model combinations work best
    - Failure patterns to avoid

    Args:
        context_id: UUID of the context to learn from
        outcome: Outcome data including engagement_metrics, quality_score, success

    Returns:
        Learning results and insights
    """
    from backend.services.acd_correlation_engine import ACDCorrelationEngine

    try:
        correlation_engine = ACDCorrelationEngine(db)
        results = await correlation_engine.learn_from_outcome(context_id, outcome)
        return results

    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to learn from outcome: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/correlations/insights", tags=["correlations"])
async def get_correlation_insights(
    domain: Optional[str] = Query(None, description="Domain filter"),
    time_window_hours: int = Query(
        168, ge=1, description="Time window in hours (default 1 week)"
    ),
    db: AsyncSession = Depends(get_db_session),
):
    """
    Get cross-context correlation insights.

    Returns correlation statistics, high-correlation pairs,
    and domain-level analysis.

    Args:
        domain: Optional domain filter
        time_window_hours: Time window for analysis

    Returns:
        Correlation insights and patterns
    """
    from backend.services.acd_correlation_engine import ACDCorrelationEngine
    from backend.models.acd import AIDomain

    try:
        correlation_engine = ACDCorrelationEngine(db)

        # Convert domain string to enum if provided
        domain_enum = None
        if domain:
            try:
                domain_enum = AIDomain(domain)
            except ValueError:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid domain: {domain}"
                )

        insights = await correlation_engine.get_correlation_insights(
            domain=domain_enum,
            time_window_hours=time_window_hours,
        )

        return insights

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get correlation insights: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================
# Memory System Endpoints
# ============================================================


@router.post("/memory", response_model=MemoryResponse, tags=["memory"])
async def store_memory(
    memory_data: MemoryCreate,
    db: AsyncSession = Depends(get_db_session),
):
    """
    Store a memory with importance weighting.

    Memory types:
    - WORKING: Current task context
    - SHORT_TERM: Recent generations (24h)
    - LONG_TERM: Persistent patterns
    - EPISODIC: Specific memorable outcomes

    Args:
        memory_data: Memory data to store

    Returns:
        Stored memory response
    """
    from backend.services.acd_memory_system import ACDMemorySystem

    try:
        memory_system = ACDMemorySystem(db)
        result = await memory_system.store_memory(memory_data)
        return result

    except Exception as e:
        logger.error(f"Failed to store memory: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/memory/recall", tags=["memory"])
async def recall_memories(
    recall_request: MemoryRecallRequest,
    db: AsyncSession = Depends(get_db_session),
):
    """
    Recall relevant memories using semantic search.

    Prioritizes:
    - Recency (recent memories weighted higher)
    - Importance (marked important memories)
    - Relevance (semantic similarity to query)

    Args:
        recall_request: Recall request with query and filters

    Returns:
        List of relevant memories
    """
    from backend.services.acd_memory_system import ACDMemorySystem

    try:
        memory_system = ACDMemorySystem(db)
        results = await memory_system.recall(recall_request)
        return {
            "query": recall_request.query,
            "memories": results,
            "count": len(results),
        }

    except Exception as e:
        logger.error(f"Failed to recall memories: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/memory/consolidate", tags=["memory"])
async def consolidate_memories(
    batch_size: int = Query(100, ge=1, le=1000, description="Max memories to process"),
    db: AsyncSession = Depends(get_db_session),
):
    """
    Trigger memory consolidation.

    Consolidation:
    - Promotes frequently accessed short-term memories to long-term
    - Promotes high-importance memories to episodic
    - Marks low-value memories as consolidated

    Args:
        batch_size: Maximum memories to process per consolidation

    Returns:
        Consolidation results
    """
    from backend.services.acd_memory_system import ACDMemorySystem

    try:
        memory_system = ACDMemorySystem(db)
        results = await memory_system.consolidate(batch_size=batch_size)
        return results

    except Exception as e:
        logger.error(f"Failed to consolidate memories: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/memory/stats", tags=["memory"])
async def get_memory_stats(
    db: AsyncSession = Depends(get_db_session),
):
    """
    Get statistics about the memory system.

    Returns counts by memory type, consolidation status,
    and other memory system metrics.

    Returns:
        Memory system statistics
    """
    from backend.services.acd_memory_system import ACDMemorySystem

    try:
        memory_system = ACDMemorySystem(db)
        stats = await memory_system.get_memory_stats()
        return stats

    except Exception as e:
        logger.error(f"Failed to get memory stats: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/memory/{memory_id}/reinforce", tags=["memory"])
async def reinforce_memory(
    memory_id: UUID,
    boost: float = Query(0.1, ge=0.01, le=0.5, description="Importance boost amount"),
    db: AsyncSession = Depends(get_db_session),
):
    """
    Reinforce a memory by increasing its importance.

    Used when a memory proves useful or relevant.

    Args:
        memory_id: UUID of the memory to reinforce
        boost: Amount to increase importance

    Returns:
        Updated memory
    """
    from backend.services.acd_memory_system import ACDMemorySystem

    try:
        memory_system = ACDMemorySystem(db)
        result = await memory_system.reinforce(memory_id, boost=boost)

        if not result:
            raise HTTPException(status_code=404, detail="Memory not found")

        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to reinforce memory: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/memory/{memory_id}", tags=["memory"])
async def forget_memory(
    memory_id: UUID,
    db: AsyncSession = Depends(get_db_session),
):
    """
    Mark a memory as forgotten (reduce importance to 0).

    The memory is not deleted, but its importance is reduced
    so it won't appear in recalls.

    Args:
        memory_id: UUID of the memory to forget

    Returns:
        Success status
    """
    from backend.services.acd_memory_system import ACDMemorySystem

    try:
        memory_system = ACDMemorySystem(db)
        success = await memory_system.forget(memory_id)

        if not success:
            raise HTTPException(status_code=404, detail="Memory not found")

        return {"status": "forgotten", "memory_id": str(memory_id)}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to forget memory: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================
# Self-Improvement Endpoints
# ============================================================


@router.get("/self-improvement/evaluate", tags=["self-improvement"])
async def evaluate_decisions(
    time_window_hours: int = Query(24, ge=1, description="Time window in hours"),
    domain: Optional[str] = Query(None, description="Domain filter"),
    db: AsyncSession = Depends(get_db_session),
):
    """
    Evaluate ACD decision quality.

    Returns:
    - Decision accuracy rate
    - Areas of consistent failure
    - Opportunities for improvement

    Args:
        time_window_hours: Time window to analyze
        domain: Optional domain filter

    Returns:
        Decision analysis results
    """
    from backend.services.acd_self_improvement import ACDSelfImprovement
    from backend.models.acd import AIDomain

    try:
        improvement_service = ACDSelfImprovement(db)

        domain_enum = None
        if domain:
            try:
                domain_enum = AIDomain(domain)
            except ValueError:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid domain: {domain}"
                )

        analysis = await improvement_service.evaluate_decisions(
            time_window_hours=time_window_hours,
            domain=domain_enum,
        )

        return {
            "total_decisions": analysis.total_decisions,
            "correct_decisions": analysis.correct_decisions,
            "accuracy_rate": analysis.accuracy_rate,
            "by_domain": analysis.by_domain,
            "failure_patterns": analysis.failure_patterns,
            "improvement_opportunities": analysis.improvement_opportunities,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to evaluate decisions: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/self-improvement/update-weights", tags=["self-improvement"])
async def update_decision_weights(
    time_window_hours: int = Query(24, ge=1, description="Time window in hours"),
    db: AsyncSession = Depends(get_db_session),
):
    """
    Update ACD decision weights based on outcomes.

    Uses reinforcement learning principles:
    - Increase weight for successful patterns
    - Decrease weight for failure patterns

    Args:
        time_window_hours: Time window to analyze

    Returns:
        Summary of weight adjustments made
    """
    from backend.services.acd_self_improvement import ACDSelfImprovement

    try:
        improvement_service = ACDSelfImprovement(db)

        # First evaluate to get analysis
        analysis = await improvement_service.evaluate_decisions(
            time_window_hours=time_window_hours
        )

        # Then update weights
        results = await improvement_service.update_decision_weights(analysis)
        return results

    except Exception as e:
        logger.error(f"Failed to update decision weights: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/self-improvement/suggestions", tags=["self-improvement"])
async def get_improvement_suggestions(
    time_window_hours: int = Query(168, ge=1, description="Time window in hours"),
    db: AsyncSession = Depends(get_db_session),
):
    """
    Generate actionable improvement suggestions.

    Includes:
    - Failure mitigation recommendations
    - Domain improvement suggestions
    - Configuration adjustment recommendations

    Args:
        time_window_hours: Time window to analyze

    Returns:
        List of improvement suggestions
    """
    from backend.services.acd_self_improvement import ACDSelfImprovement

    try:
        improvement_service = ACDSelfImprovement(db)
        suggestions = await improvement_service.generate_improvement_suggestions(
            time_window_hours=time_window_hours
        )

        return {
            "suggestions": [s.to_dict() for s in suggestions],
            "count": len(suggestions),
        }

    except Exception as e:
        logger.error(f"Failed to get improvement suggestions: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/self-improvement/metrics", tags=["self-improvement"])
async def get_improvement_metrics(
    time_window_hours: int = Query(168, ge=1, description="Time window in hours"),
    db: AsyncSession = Depends(get_db_session),
):
    """
    Get metrics on self-improvement effectiveness over time.

    Returns:
    - Outcome score trends
    - Improvement rate
    - Number of improvements applied

    Args:
        time_window_hours: Time window to analyze

    Returns:
        Improvement metrics
    """
    from backend.services.acd_self_improvement import ACDSelfImprovement

    try:
        improvement_service = ACDSelfImprovement(db)
        metrics = await improvement_service.get_improvement_metrics(
            time_window_hours=time_window_hours
        )

        return metrics

    except Exception as e:
        logger.error(f"Failed to get improvement metrics: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================
# Cross-Thinking Endpoints
# ============================================================


@router.get("/cross-thinking/analyze", tags=["cross-thinking"])
async def analyze_cross_domain_patterns(
    domains: str = Query(
        ...,
        description="Comma-separated list of domains to analyze"
    ),
    time_window_hours: int = Query(168, ge=1, description="Time window in hours"),
    db: AsyncSession = Depends(get_db_session),
):
    """
    Analyze patterns that span multiple domains.

    Finds correlations between different types of tasks,
    such as how image generation success relates to text prompts.

    Args:
        domains: Comma-separated list of domains to analyze
        time_window_hours: Time window for analysis

    Returns:
        Cross-domain analysis results
    """
    from backend.services.acd_cross_thinking import ACDCrossThinking
    from backend.models.acd import AIDomain

    try:
        cross_thinking = ACDCrossThinking(db)

        # Parse domains
        domain_list = []
        for d in domains.split(","):
            d = d.strip()
            try:
                domain_list.append(AIDomain(d))
            except ValueError:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid domain: {d}"
                )

        if len(domain_list) < 2:
            raise HTTPException(
                status_code=400,
                detail="At least 2 domains required for cross-domain analysis"
            )

        analysis = await cross_thinking.analyze_cross_domain_patterns(
            domains=domain_list,
            time_window_hours=time_window_hours,
        )

        return {
            "domains_analyzed": analysis.domains_analyzed,
            "cross_patterns": analysis.cross_patterns,
            "correlations": analysis.correlations,
            "insights": analysis.insights,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to analyze cross-domain patterns: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/cross-thinking/suggest", tags=["cross-thinking"])
async def suggest_domain_combinations(
    goal: str = Query(..., description="High-level goal description"),
    max_suggestions: int = Query(3, ge=1, le=10, description="Maximum suggestions"),
    db: AsyncSession = Depends(get_db_session),
):
    """
    Suggest domain combinations for complex tasks.

    Example: For "viral social media post", suggests:
    - TEXT_GENERATION for caption
    - IMAGE_GENERATION for visual
    - ANALYSIS for hashtag optimization

    Args:
        goal: High-level goal description
        max_suggestions: Maximum suggestions to return

    Returns:
        List of domain combination suggestions
    """
    from backend.services.acd_cross_thinking import ACDCrossThinking

    try:
        cross_thinking = ACDCrossThinking(db)
        suggestions = await cross_thinking.suggest_domain_combinations(
            goal=goal,
            max_suggestions=max_suggestions,
        )

        return {
            "goal": goal,
            "suggestions": [s.to_dict() for s in suggestions],
            "count": len(suggestions),
        }

    except Exception as e:
        logger.error(f"Failed to suggest domain combinations: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/cross-thinking/orchestrate", tags=["cross-thinking"])
async def orchestrate_multi_domain_task(
    goal: str = Query(..., description="Task goal"),
    requirements: Optional[str] = Query(None, description="Comma-separated requirements"),
    priority: str = Query("normal", description="Task priority"),
    db: AsyncSession = Depends(get_db_session),
):
    """
    Create execution plan for a multi-domain task.

    Handles:
    - Dependency ordering
    - Parallel execution opportunities
    - Fallback strategies

    Args:
        goal: Task goal description
        requirements: Optional comma-separated requirements
        priority: Task priority

    Returns:
        Orchestration plan with execution details
    """
    from backend.services.acd_cross_thinking import ACDCrossThinking, ComplexTask

    try:
        cross_thinking = ACDCrossThinking(db)

        # Parse requirements
        req_list = []
        if requirements:
            req_list = [r.strip() for r in requirements.split(",")]

        task = ComplexTask(
            goal=goal,
            requirements=req_list,
            priority=priority,
        )

        plan = await cross_thinking.orchestrate_multi_domain_task(task)

        return plan.to_dict()

    except Exception as e:
        logger.error(f"Failed to orchestrate task: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/cross-thinking/recommendations/{domain}", tags=["cross-thinking"])
async def get_cross_domain_recommendations(
    domain: str,
    db: AsyncSession = Depends(get_db_session),
):
    """
    Get recommendations for complementary domain tasks.

    Args:
        domain: Current domain

    Returns:
        Recommendations for related domain tasks
    """
    from backend.services.acd_cross_thinking import ACDCrossThinking
    from backend.models.acd import AIDomain

    try:
        cross_thinking = ACDCrossThinking(db)

        try:
            domain_enum = AIDomain(domain)
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid domain: {domain}"
            )

        recommendations = await cross_thinking.get_cross_domain_recommendations(
            current_domain=domain_enum
        )

        return recommendations

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get recommendations: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================
# Learning Task Trigger Endpoints
# ============================================================


@router.post("/learning/trigger-cycle", tags=["learning"])
async def trigger_learning_cycle(
    db: AsyncSession = Depends(get_db_session),
):
    """
    Trigger a complete ACD learning cycle.

    This runs all learning tasks:
    1. Memory consolidation
    2. HIL learning
    3. Decision weight updates
    4. Cross-domain analysis
    5. Improvement suggestions

    Note: This endpoint triggers the tasks but doesn't wait for completion.

    Returns:
        Task trigger confirmation
    """
    try:
        from backend.tasks.acd_tasks import run_acd_learning_cycle

        # Trigger the task asynchronously
        task = run_acd_learning_cycle.delay()

        return {
            "status": "triggered",
            "task_id": str(task.id),
            "message": "ACD learning cycle has been triggered",
        }

    except Exception as e:
        logger.error(f"Failed to trigger learning cycle: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
