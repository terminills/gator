"""
Reasoning Orchestrator API Routes

Endpoints for the ACD reasoning orchestrator - the basal ganglia of the system.
"""

from typing import Optional, Dict, Any
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Body
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from backend.database.connection import get_db_session
from backend.services.reasoning_orchestrator import (
    ReasoningOrchestrator,
    OrchestrationDecision,
    DecisionType,
    DecisionConfidence,
)
from backend.services.acd_service import ACDService
from backend.config.logging import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/api/v1/reasoning", tags=["reasoning_orchestrator"])


class OrchestrationRequest(BaseModel):
    """Request for orchestration decision."""
    
    context_id: UUID = Field(description="ACD context ID to orchestrate")
    current_agent: Optional[str] = Field(None, description="Name of current agent")
    additional_context: Optional[Dict[str, Any]] = Field(None, description="Extra context")


class OrchestrationResponse(BaseModel):
    """Response with orchestration decision."""
    
    context_id: UUID
    decision_type: str
    confidence: str
    reasoning: str
    target_agent: Optional[str] = None
    action_plan: Dict[str, Any]
    learned_patterns: list[str]
    risk_assessment: Optional[str] = None
    timestamp: str


class ExecutionRequest(BaseModel):
    """Request to execute an orchestration decision."""
    
    context_id: UUID = Field(description="ACD context ID")
    execute_immediately: bool = Field(True, description="Execute decision immediately")


class ExecutionResponse(BaseModel):
    """Response from decision execution."""
    
    context_id: UUID
    executed: bool
    decision_type: str
    message: str


class LearningRequest(BaseModel):
    """Request to record learning from outcome."""
    
    context_id: UUID = Field(description="ACD context ID")
    success: bool = Field(description="Whether outcome was successful")
    outcome_metadata: Optional[Dict[str, Any]] = Field(None, description="Outcome details")


class LearningResponse(BaseModel):
    """Response from learning operation."""
    
    context_id: UUID
    learned: bool
    message: str


@router.post("/orchestrate", response_model=OrchestrationResponse)
async def orchestrate_decision(
    request: OrchestrationRequest,
    db: AsyncSession = Depends(get_db_session),
):
    """
    Request an orchestration decision from the reasoning model.
    
    This is the core "basal ganglia" endpoint that evaluates an ACD context
    and decides on the best action to take.
    
    Args:
        request: Orchestration request with context ID
        
    Returns:
        OrchestrationResponse with recommended action
    """
    try:
        orchestrator = ReasoningOrchestrator(db)
        acd_service = ACDService(db)
        
        # Get the ACD context
        context = await acd_service.get_context(request.context_id)
        if not context:
            raise HTTPException(
                status_code=404,
                detail=f"ACD context {request.context_id} not found"
            )
        
        # Make orchestration decision
        decision = await orchestrator.orchestrate_decision(
            context=context,
            current_agent=request.current_agent,
            additional_context=request.additional_context,
        )
        
        # Convert to response
        return OrchestrationResponse(
            context_id=request.context_id,
            decision_type=decision.decision_type.value,
            confidence=decision.confidence.value,
            reasoning=decision.reasoning,
            target_agent=decision.target_agent,
            action_plan=decision.action_plan,
            learned_patterns=decision.learned_patterns,
            risk_assessment=decision.risk_assessment,
            timestamp=decision.timestamp.isoformat(),
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Orchestration failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/orchestrate-and-execute", response_model=ExecutionResponse)
async def orchestrate_and_execute(
    request: OrchestrationRequest,
    db: AsyncSession = Depends(get_db_session),
):
    """
    Request orchestration decision and immediately execute it.
    
    This combines decision-making and execution in one call for convenience.
    
    Args:
        request: Orchestration request
        
    Returns:
        ExecutionResponse with execution result
    """
    try:
        orchestrator = ReasoningOrchestrator(db)
        acd_service = ACDService(db)
        
        # Get context
        context = await acd_service.get_context(request.context_id)
        if not context:
            raise HTTPException(
                status_code=404,
                detail=f"ACD context {request.context_id} not found"
            )
        
        # Make decision
        decision = await orchestrator.orchestrate_decision(
            context=context,
            current_agent=request.current_agent,
            additional_context=request.additional_context,
        )
        
        # Execute decision
        executed = await orchestrator.execute_decision(context, decision)
        
        return ExecutionResponse(
            context_id=request.context_id,
            executed=executed,
            decision_type=decision.decision_type.value,
            message=(
                f"Decision executed: {decision.decision_type.value}" if executed
                else f"Decision made but execution failed: {decision.decision_type.value}"
            )
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Orchestrate and execute failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/execute", response_model=ExecutionResponse)
async def execute_decision(
    request: ExecutionRequest,
    db: AsyncSession = Depends(get_db_session),
):
    """
    Execute a previously made orchestration decision.
    
    Looks up the decision from context metadata and executes it.
    
    Args:
        request: Execution request
        
    Returns:
        ExecutionResponse with result
    """
    try:
        orchestrator = ReasoningOrchestrator(db)
        acd_service = ACDService(db)
        
        # Get context
        context = await acd_service.get_context(request.context_id)
        if not context:
            raise HTTPException(
                status_code=404,
                detail=f"ACD context {request.context_id} not found"
            )
        
        # Check if decision exists in metadata
        if not context.ai_metadata or "orchestration_decision" not in context.ai_metadata:
            raise HTTPException(
                status_code=400,
                detail="No orchestration decision found in context metadata"
            )
        
        decision_data = context.ai_metadata["orchestration_decision"]
        
        # Reconstruct decision object
        decision = OrchestrationDecision(
            decision_type=DecisionType(decision_data["type"]),
            confidence=DecisionConfidence(decision_data["confidence"]),
            reasoning=decision_data["reasoning"],
            target_agent=decision_data.get("target_agent"),
            action_plan=decision_data.get("action_plan", {}),
            learned_patterns=decision_data.get("learned_patterns", []),
        )
        
        # Execute
        executed = await orchestrator.execute_decision(context, decision)
        
        return ExecutionResponse(
            context_id=request.context_id,
            executed=executed,
            decision_type=decision.decision_type.value,
            message=(
                f"Decision executed: {decision.decision_type.value}" if executed
                else f"Execution failed for: {decision.decision_type.value}"
            )
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Execution failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/learn", response_model=LearningResponse)
async def learn_from_outcome(
    request: LearningRequest,
    db: AsyncSession = Depends(get_db_session),
):
    """
    Record learning from an orchestration outcome.
    
    This reinforces successful patterns or inhibits failed patterns,
    implementing the learning function of the basal ganglia.
    
    Args:
        request: Learning request with outcome
        
    Returns:
        LearningResponse confirming learning
    """
    try:
        orchestrator = ReasoningOrchestrator(db)
        
        # Record learning
        await orchestrator.learn_from_outcome(
            context_id=request.context_id,
            success=request.success,
            outcome_metadata=request.outcome_metadata,
        )
        
        return LearningResponse(
            context_id=request.context_id,
            learned=True,
            message=(
                f"Pattern {'reinforced' if request.success else 'inhibited'} "
                f"for context {request.context_id}"
            )
        )
        
    except Exception as e:
        logger.error(f"Learning failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/stats", response_model=Dict[str, Any])
async def get_orchestration_stats(
    hours: int = 24,
    db: AsyncSession = Depends(get_db_session),
):
    """
    Get orchestration statistics.
    
    Shows decision patterns, success rates, and learning trends.
    
    Args:
        hours: Time window for stats
        
    Returns:
        Statistics dictionary
    """
    try:
        orchestrator = ReasoningOrchestrator(db)
        acd_service = ACDService(db)
        
        # Get all contexts with orchestration decisions in time window
        from datetime import datetime, timezone, timedelta
        cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
        
        contexts = await acd_service.get_contexts(
            created_after=cutoff,
            limit=1000,
        )
        
        # Analyze decisions
        decision_counts = {}
        confidence_counts = {}
        success_counts = {"successful": 0, "failed": 0, "pending": 0}
        handoff_counts = {}
        
        for context in contexts:
            if not context.ai_metadata or "orchestration_decision" not in context.ai_metadata:
                continue
            
            decision_data = context.ai_metadata["orchestration_decision"]
            
            # Count decision types
            decision_type = decision_data.get("type", "UNKNOWN")
            decision_counts[decision_type] = decision_counts.get(decision_type, 0) + 1
            
            # Count confidence levels
            confidence = decision_data.get("confidence", "UNKNOWN")
            confidence_counts[confidence] = confidence_counts.get(confidence, 0) + 1
            
            # Count outcomes
            if context.ai_validation in ["APPROVED", "CONDITIONALLY_APPROVED"]:
                success_counts["successful"] += 1
            elif context.ai_validation == "REJECTED":
                success_counts["failed"] += 1
            else:
                success_counts["pending"] += 1
            
            # Count handoffs
            if context.ai_handoff_requested:
                handoff_to = context.ai_handoff_to or "unknown"
                handoff_counts[handoff_to] = handoff_counts.get(handoff_to, 0) + 1
        
        total_decisions = len([c for c in contexts if c.ai_metadata and "orchestration_decision" in c.ai_metadata])
        total_completed = success_counts["successful"] + success_counts["failed"]
        
        return {
            "time_window_hours": hours,
            "total_decisions": total_decisions,
            "decision_types": decision_counts,
            "confidence_levels": confidence_counts,
            "outcomes": success_counts,
            "success_rate": (
                success_counts["successful"] / total_completed * 100
                if total_completed > 0 else 0
            ),
            "handoffs": handoff_counts,
            "learning_enabled": True,
            "basal_ganglia_active": True,
        }
        
    except Exception as e:
        logger.error(f"Stats query failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
