"""
Tests for ACD (Autonomous Continuous Development) Integration

Tests the ACD context management system and its integration
with the content generation feedback loop.
"""

import pytest
from datetime import datetime, timezone
from uuid import uuid4

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
    AIValidation,
)
from backend.services.acd_service import ACDService


@pytest.mark.asyncio
async def test_create_acd_context(test_db_session):
    """Test creating an ACD context."""
    service = ACDService(test_db_session)

    context_data = ACDContextCreate(
        ai_phase="IMAGE_GENERATION",
        ai_status=AIStatus.IMPLEMENTED,
        ai_complexity=AIComplexity.MEDIUM,
        ai_note="Testing ACD context creation",
        ai_confidence=AIConfidence.CONFIDENT,
        ai_state=AIState.READY,
        ai_queue_priority=AIQueuePriority.NORMAL,
        ai_queue_status=AIQueueStatus.QUEUED,
    )

    context = await service.create_context(context_data)

    assert context is not None
    assert context.ai_phase == "IMAGE_GENERATION"
    assert context.ai_status == "IMPLEMENTED"
    assert context.ai_complexity == "MEDIUM"
    assert context.ai_confidence == "CONFIDENT"
    assert context.ai_state == "READY"


@pytest.mark.asyncio
async def test_get_acd_context(test_db_session):
    """Test retrieving an ACD context by ID."""
    service = ACDService(test_db_session)

    # Create context
    context_data = ACDContextCreate(
        ai_phase="TEXT_GENERATION",
        ai_status=AIStatus.PARTIAL,
        ai_note="Test context for retrieval",
    )
    created_context = await service.create_context(context_data)

    # Retrieve context
    retrieved_context = await service.get_context(created_context.id)

    assert retrieved_context is not None
    assert retrieved_context.id == created_context.id
    assert retrieved_context.ai_phase == "TEXT_GENERATION"
    assert retrieved_context.ai_status == "PARTIAL"


@pytest.mark.asyncio
async def test_update_acd_context(test_db_session):
    """Test updating an ACD context."""
    service = ACDService(test_db_session)

    # Create context
    context_data = ACDContextCreate(
        ai_phase="CONTENT_MODERATION",
        ai_status=AIStatus.NOT_STARTED,
        ai_state=AIState.READY,
    )
    context = await service.create_context(context_data)

    # Update context
    update_data = ACDContextUpdate(
        ai_status=AIStatus.IMPLEMENTED,
        ai_state=AIState.DONE,
        ai_confidence=AIConfidence.VALIDATED,
        ai_validation=AIValidation.APPROVED,
    )
    updated_context = await service.update_context(context.id, update_data)

    assert updated_context is not None
    assert updated_context.ai_status == "IMPLEMENTED"
    assert updated_context.ai_state == "DONE"
    assert updated_context.ai_confidence == "VALIDATED"
    assert updated_context.ai_validation == "APPROVED"


@pytest.mark.asyncio
async def test_assign_to_agent(test_db_session):
    """Test assigning a context to an agent."""
    service = ACDService(test_db_session)

    # Create context
    context_data = ACDContextCreate(
        ai_phase="PROMPT_ENHANCEMENT",
        ai_status=AIStatus.PARTIAL,
    )
    context = await service.create_context(context_data)

    # Assign to agent
    assigned_context = await service.assign_to_agent(
        context.id, "content_generation_agent", "Specialized in prompt enhancement"
    )

    assert assigned_context is not None
    assert assigned_context.ai_assigned_to == "content_generation_agent"
    assert assigned_context.ai_assignment_reason == "Specialized in prompt enhancement"
    assert assigned_context.ai_assigned_at is not None


@pytest.mark.asyncio
async def test_create_trace_artifact(test_db_session):
    """Test creating a trace artifact for error tracking."""
    service = ACDService(test_db_session)

    # Create a context first
    context_data = ACDContextCreate(
        ai_phase="IMAGE_GENERATION",
        ai_status=AIStatus.IMPLEMENTED,
    )
    context = await service.create_context(context_data)

    # Create trace artifact
    artifact_data = ACDTraceArtifactCreate(
        session_id="test_session_123",
        event_type="runtime_error",
        error_message="Out of memory during image generation",
        error_file="/path/to/generation_service.py",
        error_line=250,
        error_function="generate_image",
        acd_context_id=context.id,
        stack_trace=[
            "File generation_service.py, line 250, in generate_image",
            "File model.py, line 100, in forward",
        ],
        environment={"gpu_memory": "8GB", "model": "stable-diffusion-xl"},
    )

    artifact = await service.create_trace_artifact(artifact_data)

    assert artifact is not None
    assert artifact.session_id == "test_session_123"
    assert artifact.event_type == "runtime_error"
    assert artifact.error_message == "Out of memory during image generation"
    assert artifact.acd_context_id == context.id


@pytest.mark.asyncio
async def test_get_trace_artifacts_by_session(test_db_session):
    """Test retrieving trace artifacts by session ID."""
    service = ACDService(test_db_session)

    session_id = "test_session_456"

    # Create multiple trace artifacts
    for i in range(3):
        artifact_data = ACDTraceArtifactCreate(
            session_id=session_id,
            event_type="validation_error",
            error_message=f"Validation error {i}",
        )
        await service.create_trace_artifact(artifact_data)

    # Retrieve artifacts
    artifacts = await service.get_trace_artifacts_by_session(session_id)

    assert len(artifacts) == 3
    assert all(a.session_id == session_id for a in artifacts)


@pytest.mark.asyncio
async def test_get_acd_stats(test_db_session):
    """Test retrieving ACD statistics."""
    service = ACDService(test_db_session)

    # Create multiple contexts with different states
    phases = ["IMAGE_GENERATION", "TEXT_GENERATION", "IMAGE_GENERATION"]
    statuses = [AIStatus.IMPLEMENTED, AIStatus.PARTIAL, AIStatus.IMPLEMENTED]
    states = [AIState.DONE, AIState.PROCESSING, AIState.DONE]

    for phase, status, state in zip(phases, statuses, states):
        context_data = ACDContextCreate(
            ai_phase=phase,
            ai_status=status,
            ai_state=state,
        )
        await service.create_context(context_data)

    # Get statistics
    stats = await service.get_stats(hours=24)

    assert stats.total_contexts == 3
    assert stats.by_phase["IMAGE_GENERATION"] == 2
    assert stats.by_phase["TEXT_GENERATION"] == 1
    assert stats.completed_contexts == 2
    assert stats.active_contexts == 1


@pytest.mark.asyncio
async def test_validation_report(test_db_session):
    """Test generating a validation report."""
    service = ACDService(test_db_session)

    # Create contexts
    for i in range(5):
        context_data = ACDContextCreate(
            ai_phase=f"PHASE_{i}",
            ai_status=AIStatus.IMPLEMENTED,
            ai_state=AIState.DONE if i < 3 else AIState.PROCESSING,
        )
        await service.create_context(context_data)

    # Generate validation report
    report = await service.generate_validation_report()

    assert report is not None
    assert report.metadata["acd_metadata_found"] == 5
    assert len(report.acd_contexts) == 5
    assert report.metadata["acd_schema_version"] == "1.1.0"


@pytest.mark.asyncio
async def test_context_with_benchmark_link(test_db_session):
    """Test creating ACD context linked to a benchmark."""
    service = ACDService(test_db_session)

    benchmark_id = uuid4()

    context_data = ACDContextCreate(
        benchmark_id=benchmark_id,
        ai_phase="QUALITY_ASSESSMENT",
        ai_status=AIStatus.IMPLEMENTED,
        ai_note="Linked to benchmark for quality tracking",
    )

    context = await service.create_context(context_data)

    assert context.benchmark_id == benchmark_id

    # Retrieve by benchmark
    retrieved = await service.get_context_by_benchmark(benchmark_id)
    assert retrieved is not None
    assert retrieved.id == context.id


@pytest.mark.asyncio
async def test_context_with_content_link(test_db_session):
    """Test creating ACD context linked to content."""
    service = ACDService(test_db_session)

    content_id = uuid4()

    context_data = ACDContextCreate(
        content_id=content_id,
        ai_phase="CONTENT_GENERATION",
        ai_status=AIStatus.IMPLEMENTED,
        ai_note="Linked to generated content",
    )

    context = await service.create_context(context_data)

    assert context.content_id == content_id

    # Retrieve by content
    retrieved = await service.get_context_by_content(content_id)
    assert retrieved is not None
    assert retrieved.id == context.id


@pytest.mark.asyncio
async def test_acd_context_dependencies(test_db_session):
    """Test ACD context with dependencies."""
    service = ACDService(test_db_session)

    context_data = ACDContextCreate(
        ai_phase="POST_PROCESSING",
        ai_status=AIStatus.IMPLEMENTED,
        ai_dependencies=["IMAGE_GENERATION", "QUALITY_CHECK"],
        ai_pattern="sequential_pipeline",
        ai_strategy="Apply filters and enhancements after generation",
    )

    context = await service.create_context(context_data)

    assert context.ai_dependencies is not None
    # Note: JSON fields return as dict/list directly from database
    # We need to check if the service properly stores them


@pytest.mark.asyncio
async def test_acd_context_with_metadata(test_db_session):
    """Test ACD context with extended metadata."""
    service = ACDService(test_db_session)

    context_data = ACDContextCreate(
        ai_phase="STYLE_TRANSFER",
        ai_status=AIStatus.EXPERIMENTAL,
        ai_context={
            "source_style": "impressionist",
            "target_style": "modern",
            "blend_factor": 0.7,
        },
        ai_metadata={
            "experiment_id": "exp_001",
            "researcher": "content_team",
        },
    )

    context = await service.create_context(context_data)

    assert context.ai_context is not None
    assert context.ai_metadata is not None


@pytest.mark.asyncio
async def test_acd_agent_reassignment(test_db_session):
    """Test reassigning context to different agents."""
    service = ACDService(test_db_session)

    # Create context
    context_data = ACDContextCreate(
        ai_phase="COMPLEX_TASK",
        ai_status=AIStatus.PARTIAL,
    )
    context = await service.create_context(context_data)

    # First assignment
    context = await service.assign_to_agent(context.id, "agent_1", "Initial assignment")
    assert context.ai_assigned_to == "agent_1"

    # Reassignment
    context = await service.assign_to_agent(context.id, "agent_2", "Needs expert")
    assert context.ai_assigned_to == "agent_2"
    assert context.ai_previous_assignee == "agent_1"
