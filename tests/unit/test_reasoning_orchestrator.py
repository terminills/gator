"""
Tests for Reasoning Orchestrator - Basal Ganglia of ACD

Tests the decision-making, pattern learning, and coordination capabilities.
"""

import pytest
from datetime import datetime, timezone, timedelta
from uuid import uuid4

from backend.services.reasoning_orchestrator import (
    ReasoningOrchestrator,
    DecisionType,
    DecisionConfidence,
)
from backend.services.acd_service import ACDService
from backend.models.acd import (
    ACDContextCreate,
    ACDContextUpdate,
    AIStatus,
    AIState,
    AIComplexity,
    AIConfidence,
    AIValidation,
)


@pytest.mark.asyncio
async def test_orchestrate_simple_task(db_session):
    """Test orchestration of a simple, low-complexity task."""
    # Create a simple task context
    acd_service = ACDService(db_session)
    context = await acd_service.create_context(
        ACDContextCreate(
            ai_phase="TEXT_GENERATION",
            ai_status=AIStatus.IMPLEMENTED,
            ai_complexity=AIComplexity.LOW,
            ai_confidence=AIConfidence.CONFIDENT,
            ai_state=AIState.READY,
            ai_note="Simple text generation task",
        )
    )
    
    # Orchestrate decision
    orchestrator = ReasoningOrchestrator(db_session)
    decision = await orchestrator.orchestrate_decision(context)
    
    # Should execute locally for simple, confident tasks
    assert decision.decision_type == DecisionType.EXECUTE_LOCALLY
    assert decision.confidence in [DecisionConfidence.HIGH, DecisionConfidence.MEDIUM]
    assert "confidence" in decision.reasoning.lower()
    assert decision.action_plan is not None


@pytest.mark.asyncio
async def test_orchestrate_high_complexity_task(db_session):
    """Test orchestration of high-complexity task with low confidence."""
    # Create high-complexity task
    acd_service = ACDService(db_session)
    context = await acd_service.create_context(
        ACDContextCreate(
            ai_phase="VIDEO_GENERATION",
            ai_status=AIStatus.IMPLEMENTED,
            ai_complexity=AIComplexity.CRITICAL,
            ai_confidence=AIConfidence.UNCERTAIN,
            ai_state=AIState.READY,
            ai_note="Complex video generation requiring expertise",
        )
    )
    
    # Orchestrate decision
    orchestrator = ReasoningOrchestrator(db_session)
    decision = await orchestrator.orchestrate_decision(context)
    
    # Should escalate due to high complexity and low confidence
    assert decision.decision_type == DecisionType.HANDOFF_ESCALATION
    assert decision.target_agent is not None
    assert "escalation" in decision.reasoning.lower() or "complexity" in decision.reasoning.lower()


@pytest.mark.asyncio
async def test_orchestrate_with_errors(db_session):
    """Test orchestration when errors are present."""
    # Create context with errors
    acd_service = ACDService(db_session)
    context = await acd_service.create_context(
        ACDContextCreate(
            ai_phase="IMAGE_GENERATION",
            ai_status=AIStatus.IMPLEMENTED,
            ai_complexity=AIComplexity.MEDIUM,
            ai_confidence=AIConfidence.UNCERTAIN,
            ai_state=AIState.FAILED,
            ai_note="Image generation failed",
        )
    )
    
    # Update with error
    await acd_service.update_context(
        context.id,
        ACDContextUpdate(runtime_err="CUDA out of memory")
    )
    
    # Re-fetch context
    context = await acd_service.get_context(context.id)
    
    # Orchestrate decision
    orchestrator = ReasoningOrchestrator(db_session)
    decision = await orchestrator.orchestrate_decision(context)
    
    # Should either retry or request assistance
    assert decision.decision_type in [
        DecisionType.RETRY_WITH_LEARNING,
        DecisionType.REQUEST_REVIEW,
        DecisionType.HANDOFF_ESCALATION,
    ]


@pytest.mark.asyncio
async def test_orchestrate_retry_limit_exceeded(db_session):
    """Test orchestration when retry limit is exceeded."""
    # Create context with high retry count
    acd_service = ACDService(db_session)
    context = await acd_service.create_context(
        ACDContextCreate(
            ai_phase="TEXT_GENERATION",
            ai_status=AIStatus.IMPLEMENTED,
            ai_complexity=AIComplexity.MEDIUM,
            ai_state=AIState.FAILED,
            ai_metadata={"retry_count": 5},  # Exceeds default limit of 3
        )
    )
    
    # Orchestrate decision
    orchestrator = ReasoningOrchestrator(db_session)
    decision = await orchestrator.orchestrate_decision(context)
    
    # Should defer to human after retries exhausted
    assert decision.decision_type == DecisionType.DEFER_TO_HUMAN
    assert "retry limit" in decision.reasoning.lower() or "human" in decision.reasoning.lower()


@pytest.mark.asyncio
async def test_orchestrate_blocked_task(db_session):
    """Test orchestration of a blocked task."""
    # Create blocked context
    acd_service = ACDService(db_session)
    context = await acd_service.create_context(
        ACDContextCreate(
            ai_phase="IMAGE_GENERATION",
            ai_status=AIStatus.IMPLEMENTED,
            ai_complexity=AIComplexity.MEDIUM,
            ai_state=AIState.BLOCKED,
            ai_started=datetime.now(timezone.utc) - timedelta(minutes=10),
            ai_note="Task blocked for 10 minutes",
        )
    )
    
    # Orchestrate decision
    orchestrator = ReasoningOrchestrator(db_session)
    decision = await orchestrator.orchestrate_decision(context)
    
    # Should request collaboration or assistance
    assert decision.decision_type in [
        DecisionType.REQUEST_COLLABORATION,
        DecisionType.HANDOFF_ESCALATION,
        DecisionType.REQUEST_REVIEW,
    ]


@pytest.mark.asyncio
async def test_execute_handoff_decision(db_session):
    """Test execution of a handoff decision."""
    # Create context
    acd_service = ACDService(db_session)
    context = await acd_service.create_context(
        ACDContextCreate(
            ai_phase="IMAGE_GENERATION",
            ai_status=AIStatus.IMPLEMENTED,
            ai_complexity=AIComplexity.HIGH,
            ai_confidence=AIConfidence.UNCERTAIN,
            ai_state=AIState.READY,
        )
    )
    
    # Orchestrate decision
    orchestrator = ReasoningOrchestrator(db_session)
    decision = await orchestrator.orchestrate_decision(context, current_agent="basic_generator")
    
    # Execute decision
    success = await orchestrator.execute_decision(context, decision)
    
    # Verify execution
    assert success is True
    
    # Check context was updated
    updated_context = await acd_service.get_context(context.id)
    assert updated_context is not None
    
    # Verify decision was logged in metadata
    assert updated_context.ai_metadata is not None
    assert "orchestration_decision" in updated_context.ai_metadata


@pytest.mark.asyncio
async def test_learn_from_successful_outcome(db_session):
    """Test learning from a successful outcome."""
    # Create and complete successful context
    acd_service = ACDService(db_session)
    context = await acd_service.create_context(
        ACDContextCreate(
            ai_phase="TEXT_GENERATION",
            ai_status=AIStatus.IMPLEMENTED,
            ai_complexity=AIComplexity.MEDIUM,
            ai_state=AIState.DONE,
            ai_validation=AIValidation.APPROVED.value,
        )
    )
    
    # Make decision (which logs to metadata)
    orchestrator = ReasoningOrchestrator(db_session)
    decision = await orchestrator.orchestrate_decision(context)
    
    # Re-fetch context with decision metadata
    context = await acd_service.get_context(context.id)
    
    # Learn from successful outcome
    await orchestrator.learn_from_outcome(
        context.id,
        success=True,
        outcome_metadata={"engagement_rate": 8.5}
    )
    
    # Pattern cache should be cleared to force refresh
    # (pattern reinforcement)
    cache_key = f"{context.ai_phase}_{context.ai_complexity}"
    assert cache_key not in orchestrator._pattern_cache


@pytest.mark.asyncio
async def test_learn_from_failed_outcome(db_session):
    """Test learning from a failed outcome."""
    # Create failed context
    acd_service = ACDService(db_session)
    context = await acd_service.create_context(
        ACDContextCreate(
            ai_phase="IMAGE_GENERATION",
            ai_status=AIStatus.IMPLEMENTED,
            ai_complexity=AIComplexity.HIGH,
            ai_state=AIState.FAILED,
            ai_validation=AIValidation.REJECTED.value,
        )
    )
    
    # Make decision
    orchestrator = ReasoningOrchestrator(db_session)
    decision = await orchestrator.orchestrate_decision(context)
    
    # Re-fetch context
    context = await acd_service.get_context(context.id)
    
    # Learn from failed outcome
    await orchestrator.learn_from_outcome(
        context.id,
        success=False,
        outcome_metadata={"failure_reason": "GPU out of memory"}
    )
    
    # Check failed decision was recorded
    updated_context = await acd_service.get_context(context.id)
    assert updated_context.ai_metadata is not None
    
    if "failed_decisions" in updated_context.ai_metadata:
        failed_decisions = updated_context.ai_metadata["failed_decisions"]
        assert len(failed_decisions) > 0
        assert "failure_reason" in str(failed_decisions)


@pytest.mark.asyncio
async def test_pattern_learning_with_successful_history(db_session):
    """Test that orchestrator learns from successful patterns."""
    acd_service = ACDService(db_session)
    
    # Create multiple successful contexts with same characteristics
    for i in range(5):
        context = await acd_service.create_context(
            ACDContextCreate(
                ai_phase="TEXT_GENERATION",
                ai_status=AIStatus.IMPLEMENTED,
                ai_complexity=AIComplexity.MEDIUM,
                ai_state=AIState.DONE,
                ai_validation=AIValidation.APPROVED.value,
                ai_assigned_to="text_specialist",
                ai_strategy="use_gpt4_for_quality",
                ai_pattern="high_quality_text",
            )
        )
    
    # Create new similar task
    new_context = await acd_service.create_context(
        ACDContextCreate(
            ai_phase="TEXT_GENERATION",
            ai_status=AIStatus.IMPLEMENTED,
            ai_complexity=AIComplexity.MEDIUM,
            ai_confidence=AIConfidence.UNCERTAIN,
            ai_state=AIState.READY,
        )
    )
    
    # Orchestrate - should learn from patterns
    orchestrator = ReasoningOrchestrator(db_session)
    decision = await orchestrator.orchestrate_decision(
        new_context,
        current_agent="basic_generator"
    )
    
    # Should have learned patterns
    assert len(decision.learned_patterns) > 0 or decision.target_agent == "text_specialist"


@pytest.mark.asyncio
async def test_orchestrator_with_uncertain_confidence(db_session):
    """Test orchestration with uncertain confidence."""
    acd_service = ACDService(db_session)
    context = await acd_service.create_context(
        ACDContextCreate(
            ai_phase="SOCIAL_MEDIA_CONTENT",
            ai_status=AIStatus.IMPLEMENTED,
            ai_complexity=AIComplexity.MEDIUM,
            ai_confidence=AIConfidence.UNCERTAIN,
            ai_state=AIState.READY,
        )
    )
    
    orchestrator = ReasoningOrchestrator(db_session)
    decision = await orchestrator.orchestrate_decision(context)
    
    # Should request review for uncertain tasks
    assert decision.decision_type in [
        DecisionType.REQUEST_REVIEW,
        DecisionType.HANDOFF_SPECIALIZATION,
        DecisionType.EXECUTE_LOCALLY,
    ]
    assert decision.reasoning is not None
    assert len(decision.reasoning) > 0


@pytest.mark.asyncio
async def test_complexity_evaluation(db_session):
    """Test complexity scoring logic."""
    acd_service = ACDService(db_session)
    orchestrator = ReasoningOrchestrator(db_session)
    
    # Test low complexity
    low_context = await acd_service.create_context(
        ACDContextCreate(
            ai_phase="TEXT_GENERATION",
            ai_status=AIStatus.IMPLEMENTED,
            ai_complexity=AIComplexity.LOW,
            ai_state=AIState.READY,
        )
    )
    
    situation = await orchestrator._assess_situation(low_context, None)
    complexity_score = orchestrator._evaluate_complexity(low_context, situation)
    
    assert 0.0 <= complexity_score <= 0.5  # Low complexity
    
    # Test high complexity with errors
    high_context = await acd_service.create_context(
        ACDContextCreate(
            ai_phase="VIDEO_GENERATION",
            ai_status=AIStatus.IMPLEMENTED,
            ai_complexity=AIComplexity.CRITICAL,
            ai_state=AIState.BLOCKED,
            runtime_err="Complex error",
        )
    )
    
    situation = await orchestrator._assess_situation(high_context, None)
    complexity_score = orchestrator._evaluate_complexity(high_context, situation)
    
    assert complexity_score >= 0.8  # High complexity with errors


@pytest.mark.asyncio
async def test_confidence_evaluation_with_patterns(db_session):
    """Test confidence scoring with learned patterns."""
    acd_service = ACDService(db_session)
    orchestrator = ReasoningOrchestrator(db_session)
    
    # Create successful pattern
    await acd_service.create_context(
        ACDContextCreate(
            ai_phase="IMAGE_GENERATION",
            ai_status=AIStatus.IMPLEMENTED,
            ai_complexity=AIComplexity.MEDIUM,
            ai_state=AIState.DONE,
            ai_validation=AIValidation.APPROVED.value,
        )
    )
    
    # Create context to evaluate
    context = await acd_service.create_context(
        ACDContextCreate(
            ai_phase="IMAGE_GENERATION",
            ai_status=AIStatus.IMPLEMENTED,
            ai_complexity=AIComplexity.MEDIUM,
            ai_confidence=AIConfidence.UNCERTAIN,
            ai_state=AIState.READY,
        )
    )
    
    # Query patterns and evaluate confidence
    patterns = await orchestrator._query_relevant_patterns(context)
    confidence_score = orchestrator._evaluate_confidence(context, patterns)
    
    # Confidence should be boosted by successful patterns
    # Base uncertain = 0.3, pattern boost should increase it
    assert confidence_score > 0.3
