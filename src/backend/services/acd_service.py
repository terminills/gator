"""
ACD (Autonomous Continuous Development) Service

Manages ACD context metadata for content generation tasks,
enabling autonomous decision-making and continuous improvement.
"""

import asyncio
from typing import Dict, List, Optional, Any
from uuid import UUID
from datetime import datetime, timezone, timedelta

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, and_

from backend.models.acd import (
    ACDContextModel,
    ACDTraceArtifactModel,
    ACDContextCreate,
    ACDContextUpdate,
    ACDContextResponse,
    ACDTraceArtifactCreate,
    ACDTraceArtifactResponse,
    ACDValidationReport,
    ACDStats,
    AIStatus,
    AIState,
    AIQueueStatus,
)
from backend.config.logging import get_logger

logger = get_logger(__name__)


class ACDService:
    """
    Service for managing ACD context metadata.

    Provides comprehensive context management for content generation,
    enabling autonomous operation and continuous learning.
    """

    def __init__(self, db_session: AsyncSession):
        """
        Initialize ACD service.

        Args:
            db_session: Database session for persistence
        """
        self.db = db_session

    async def create_context(
        self, context_data: ACDContextCreate
    ) -> ACDContextResponse:
        """
        Create a new ACD context for content generation.

        Args:
            context_data: Context metadata to create

        Returns:
            Created context record
        """
        try:
            # Convert enums to strings for database storage
            context_dict = context_data.model_dump()
            if "ai_status" in context_dict and hasattr(
                context_dict["ai_status"], "value"
            ):
                context_dict["ai_status"] = context_dict["ai_status"].value
            if "ai_complexity" in context_dict and context_dict["ai_complexity"]:
                if hasattr(context_dict["ai_complexity"], "value"):
                    context_dict["ai_complexity"] = context_dict["ai_complexity"].value
            if "ai_state" in context_dict and hasattr(context_dict["ai_state"], "value"):
                context_dict["ai_state"] = context_dict["ai_state"].value
            if "ai_confidence" in context_dict and context_dict["ai_confidence"]:
                if hasattr(context_dict["ai_confidence"], "value"):
                    context_dict["ai_confidence"] = context_dict["ai_confidence"].value
            if "ai_queue_priority" in context_dict and context_dict["ai_queue_priority"]:
                if hasattr(context_dict["ai_queue_priority"], "value"):
                    context_dict["ai_queue_priority"] = context_dict[
                        "ai_queue_priority"
                    ].value
            if "ai_queue_status" in context_dict and context_dict["ai_queue_status"]:
                if hasattr(context_dict["ai_queue_status"], "value"):
                    context_dict["ai_queue_status"] = context_dict[
                        "ai_queue_status"
                    ].value

            context = ACDContextModel(**context_dict)

            self.db.add(context)
            await self.db.commit()
            await self.db.refresh(context)

            logger.info(
                f"Created ACD context {context.id}: "
                f"phase={context.ai_phase}, "
                f"status={context.ai_status}, "
                f"state={context.ai_state}"
            )

            return ACDContextResponse.model_validate(context)

        except Exception as e:
            logger.error(f"Failed to create ACD context: {str(e)}")
            await self.db.rollback()
            raise

    async def list_contexts(
        self, limit: int = 10, offset: int = 0
    ) -> List[ACDContextResponse]:
        """
        List ACD contexts with pagination.

        Args:
            limit: Maximum number of contexts to return
            offset: Number of contexts to skip

        Returns:
            List of context records
        """
        try:
            stmt = (
                select(ACDContextModel)
                .order_by(ACDContextModel.created_at.desc())
                .limit(limit)
                .offset(offset)
            )
            result = await self.db.execute(stmt)
            contexts = result.scalars().all()

            return [ACDContextResponse.model_validate(c) for c in contexts]

        except Exception as e:
            logger.error(f"Failed to list ACD contexts: {str(e)}")
            return []

    async def get_context(self, context_id: UUID) -> Optional[ACDContextResponse]:
        """
        Get an ACD context by ID.

        Args:
            context_id: UUID of the context

        Returns:
            Context record if found, None otherwise
        """
        try:
            stmt = select(ACDContextModel).where(ACDContextModel.id == context_id)
            result = await self.db.execute(stmt)
            context = result.scalar_one_or_none()

            if context:
                return ACDContextResponse.model_validate(context)
            return None

        except Exception as e:
            logger.error(f"Failed to get ACD context {context_id}: {str(e)}")
            return None

    async def update_context(
        self, context_id: UUID, update_data: ACDContextUpdate
    ) -> Optional[ACDContextResponse]:
        """
        Update an ACD context.

        Args:
            context_id: UUID of the context to update
            update_data: Fields to update

        Returns:
            Updated context record if found, None otherwise
        """
        try:
            stmt = select(ACDContextModel).where(ACDContextModel.id == context_id)
            result = await self.db.execute(stmt)
            context = result.scalar_one_or_none()

            if not context:
                logger.warning(f"ACD context {context_id} not found for update")
                return None

            # Update fields
            update_dict = update_data.model_dump(exclude_unset=True)
            for key, value in update_dict.items():
                # Convert enums to strings
                if value and hasattr(value, "value"):
                    value = value.value
                setattr(context, key, value)

            await self.db.commit()
            await self.db.refresh(context)

            logger.info(f"Updated ACD context {context_id}")

            return ACDContextResponse.model_validate(context)

        except Exception as e:
            logger.error(f"Failed to update ACD context {context_id}: {str(e)}")
            await self.db.rollback()
            raise

    async def get_context_by_benchmark(
        self, benchmark_id: UUID
    ) -> Optional[ACDContextResponse]:
        """
        Get ACD context associated with a benchmark.

        Args:
            benchmark_id: UUID of the benchmark

        Returns:
            Context record if found, None otherwise
        """
        try:
            stmt = select(ACDContextModel).where(
                ACDContextModel.benchmark_id == benchmark_id
            )
            result = await self.db.execute(stmt)
            context = result.scalar_one_or_none()

            if context:
                return ACDContextResponse.model_validate(context)
            return None

        except Exception as e:
            logger.error(
                f"Failed to get ACD context for benchmark {benchmark_id}: {str(e)}"
            )
            return None

    async def get_context_by_content(
        self, content_id: UUID
    ) -> Optional[ACDContextResponse]:
        """
        Get ACD context associated with content.

        Args:
            content_id: UUID of the content

        Returns:
            Context record if found, None otherwise
        """
        try:
            stmt = select(ACDContextModel).where(
                ACDContextModel.content_id == content_id
            )
            result = await self.db.execute(stmt)
            context = result.scalar_one_or_none()

            if context:
                return ACDContextResponse.model_validate(context)
            return None

        except Exception as e:
            logger.error(
                f"Failed to get ACD context for content {content_id}: {str(e)}"
            )
            return None

    async def create_trace_artifact(
        self, artifact_data: ACDTraceArtifactCreate
    ) -> ACDTraceArtifactResponse:
        """
        Create a trace artifact for error tracking.

        Args:
            artifact_data: Trace artifact data

        Returns:
            Created trace artifact
        """
        try:
            artifact = ACDTraceArtifactModel(**artifact_data.model_dump())

            self.db.add(artifact)
            await self.db.commit()
            await self.db.refresh(artifact)

            logger.info(
                f"Created trace artifact {artifact.id}: "
                f"type={artifact.event_type}, "
                f"session={artifact.session_id}"
            )

            return ACDTraceArtifactResponse.model_validate(artifact)

        except Exception as e:
            logger.error(f"Failed to create trace artifact: {str(e)}")
            await self.db.rollback()
            raise

    async def get_trace_artifacts_by_session(
        self, session_id: str
    ) -> List[ACDTraceArtifactResponse]:
        """
        Get all trace artifacts for a session.

        Args:
            session_id: Session identifier

        Returns:
            List of trace artifacts
        """
        try:
            stmt = (
                select(ACDTraceArtifactModel)
                .where(ACDTraceArtifactModel.session_id == session_id)
                .order_by(ACDTraceArtifactModel.timestamp.desc())
            )
            result = await self.db.execute(stmt)
            artifacts = result.scalars().all()

            return [
                ACDTraceArtifactResponse.model_validate(a) for a in artifacts
            ]

        except Exception as e:
            logger.error(
                f"Failed to get trace artifacts for session {session_id}: {str(e)}"
            )
            return []

    async def get_stats(
        self, hours: int = 24, phase: Optional[str] = None
    ) -> ACDStats:
        """
        Get ACD context statistics.

        Args:
            hours: Time window for stats
            phase: Filter by specific phase

        Returns:
            Aggregate statistics
        """
        try:
            cutoff_time = datetime.now(timezone.utc) - timedelta(hours=hours)

            # Build base query
            stmt = select(ACDContextModel).where(
                ACDContextModel.created_at >= cutoff_time
            )

            if phase:
                stmt = stmt.where(ACDContextModel.ai_phase == phase)

            result = await self.db.execute(stmt)
            contexts = result.scalars().all()

            if not contexts:
                return ACDStats()

            # Calculate statistics
            total = len(contexts)
            by_phase = {}
            by_status = {}
            by_state = {}
            by_confidence = {}
            by_validation = {}
            active_count = 0
            completed_count = 0
            failed_count = 0
            total_completion_time = 0
            completion_count = 0

            for context in contexts:
                # Count by phase
                phase = context.ai_phase
                by_phase[phase] = by_phase.get(phase, 0) + 1

                # Count by status
                status = context.ai_status
                by_status[status] = by_status.get(status, 0) + 1

                # Count by state
                state = context.ai_state
                by_state[state] = by_state.get(state, 0) + 1

                # Count by confidence
                if context.ai_confidence:
                    conf = context.ai_confidence
                    by_confidence[conf] = by_confidence.get(conf, 0) + 1

                # Count by validation
                if context.ai_validation:
                    val = context.ai_validation
                    by_validation[val] = by_validation.get(val, 0) + 1

                # Count active/completed/failed
                if context.ai_state in [
                    "PROCESSING",
                    "READY",
                    "BLOCKED",
                    "PAUSED",
                ]:
                    active_count += 1
                elif context.ai_state == "DONE":
                    completed_count += 1
                    # Calculate completion time if we have started time
                    if context.ai_started:
                        duration = (context.updated_at - context.ai_started).total_seconds()
                        total_completion_time += duration
                        completion_count += 1
                elif context.ai_state in ["FAILED", "CANCELLED"]:
                    failed_count += 1

            return ACDStats(
                total_contexts=total,
                by_phase=by_phase,
                by_status=by_status,
                by_state=by_state,
                by_confidence=by_confidence,
                by_validation=by_validation,
                avg_completion_time=(
                    total_completion_time / completion_count
                    if completion_count > 0
                    else None
                ),
                active_contexts=active_count,
                completed_contexts=completed_count,
                failed_contexts=failed_count,
            )

        except Exception as e:
            logger.error(f"Failed to get ACD stats: {str(e)}")
            return ACDStats()

    async def generate_validation_report(self) -> ACDValidationReport:
        """
        Generate a validation report for all ACD contexts.

        Returns:
            Validation report with metadata, contexts, and issues
        """
        try:
            # Get all contexts
            stmt = select(ACDContextModel).order_by(ACDContextModel.created_at.desc())
            result = await self.db.execute(stmt)
            contexts = result.scalars().all()

            # Count metadata
            total_contexts = len(contexts)
            by_phase = {}
            errors = []
            warnings = []

            for context in contexts:
                # Count by phase
                phase = context.ai_phase
                by_phase[phase] = by_phase.get(phase, 0) + 1

                # Validate required fields
                if not context.ai_status:
                    errors.append(
                        {
                            "context_id": str(context.id),
                            "message": "Missing required field: ai_status",
                            "severity": "error",
                        }
                    )

                # Check for inconsistencies
                if context.ai_state == "DONE" and context.ai_status not in [
                    "IMPLEMENTED",
                    "FIXED",
                ]:
                    warnings.append(
                        {
                            "context_id": str(context.id),
                            "message": f"State is DONE but status is {context.ai_status}",
                            "severity": "warning",
                        }
                    )

            # Build report
            report = ACDValidationReport(
                metadata={
                    "acd_schema_version": "1.1.0",
                    "files_processed": 0,  # Not applicable for database records
                    "acd_metadata_found": total_contexts,
                    "errors": len(errors),
                    "warnings": len(warnings),
                    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                },
                acd_contexts=[ACDContextResponse.model_validate(c) for c in contexts],
                errors=errors,
                warnings=warnings,
            )

            logger.info(
                f"Generated validation report: {total_contexts} contexts, "
                f"{len(errors)} errors, {len(warnings)} warnings"
            )

            return report

        except Exception as e:
            logger.error(f"Failed to generate validation report: {str(e)}")
            raise

    async def assign_to_agent(
        self, context_id: UUID, agent_name: str, reason: Optional[str] = None
    ) -> Optional[ACDContextResponse]:
        """
        Assign a context to an agent.

        Args:
            context_id: UUID of the context
            agent_name: Name of the agent to assign
            reason: Reason for assignment

        Returns:
            Updated context record
        """
        try:
            stmt = select(ACDContextModel).where(ACDContextModel.id == context_id)
            result = await self.db.execute(stmt)
            context = result.scalar_one_or_none()

            if not context:
                return None

            # Update assignment
            previous_assignee = context.ai_assigned_to
            context.ai_assigned_to = agent_name
            context.ai_assigned_at = datetime.now(timezone.utc)
            context.ai_assignment_reason = reason

            if previous_assignee:
                context.ai_previous_assignee = previous_assignee
                # Update history
                history = context.ai_assignment_history or []
                history.append(f"{previous_assignee} -> {agent_name}")
                context.ai_assignment_history = history

            await self.db.commit()
            await self.db.refresh(context)

            logger.info(
                f"Assigned context {context_id} to agent {agent_name}"
            )

            return ACDContextResponse.model_validate(context)

        except Exception as e:
            logger.error(f"Failed to assign context {context_id}: {str(e)}")
            await self.db.rollback()
            raise
