"""
ACD Learning Background Tasks

Celery tasks for automated ACD learning and improvement:
- Memory consolidation
- Decision weight updates
- Cross-domain pattern analysis
- Improvement suggestion generation
"""

import asyncio
from datetime import datetime, timezone

from celery import shared_task

from backend.celery_app import app
from backend.config.logging import get_logger

logger = get_logger(__name__)


def run_async(coro):
    """Helper to run async code in sync context."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


@app.task(
    name="backend.tasks.acd_tasks.consolidate_memories",
    bind=True,
    max_retries=3,
    default_retry_delay=300,
)
def consolidate_memories(self, batch_size: int = 100):
    """
    Consolidate ACD memories from short-term to long-term storage.

    This task:
    - Promotes frequently accessed memories to long-term
    - Promotes high-importance memories to episodic
    - Marks low-value memories as consolidated

    Args:
        batch_size: Maximum memories to process per run

    Returns:
        Consolidation results summary
    """
    try:
        logger.info("Starting ACD memory consolidation task")

        async def _consolidate():
            from backend.database.connection import get_async_session
            from backend.services.acd_memory_system import ACDMemorySystem

            async with get_async_session() as session:
                memory_system = ACDMemorySystem(session)
                results = await memory_system.consolidate(batch_size=batch_size)
                return results

        results = run_async(_consolidate())

        logger.info(
            f"Memory consolidation complete: {results.get('total_processed', 0)} processed"
        )
        return results

    except Exception as e:
        logger.error(f"Memory consolidation failed: {e}")
        raise self.retry(exc=e)


@app.task(
    name="backend.tasks.acd_tasks.update_decision_weights",
    bind=True,
    max_retries=3,
    default_retry_delay=300,
)
def update_decision_weights(self, time_window_hours: int = 24):
    """
    Update ACD decision weights based on recent outcomes.

    This task:
    - Evaluates decisions from the specified time window
    - Adjusts learning weights based on outcomes
    - Applies reinforcement learning principles

    Args:
        time_window_hours: Time window to analyze

    Returns:
        Summary of weight adjustments
    """
    try:
        logger.info(f"Starting decision weight update (window: {time_window_hours}h)")

        async def _update_weights():
            from backend.database.connection import get_async_session
            from backend.services.acd_self_improvement import ACDSelfImprovement

            async with get_async_session() as session:
                improvement_service = ACDSelfImprovement(session)
                analysis = await improvement_service.evaluate_decisions(
                    time_window_hours=time_window_hours
                )
                results = await improvement_service.update_decision_weights(analysis)
                return results

        results = run_async(_update_weights())

        logger.info(
            f"Decision weight update complete: {results.get('total_adjusted', 0)} adjusted"
        )
        return results

    except Exception as e:
        logger.error(f"Decision weight update failed: {e}")
        raise self.retry(exc=e)


@app.task(
    name="backend.tasks.acd_tasks.analyze_cross_domain_patterns",
    bind=True,
    max_retries=2,
    default_retry_delay=600,
)
def analyze_cross_domain_patterns(self, time_window_hours: int = 168):
    """
    Analyze patterns across domains for cross-domain learning.

    This task:
    - Finds correlations between different domains
    - Identifies successful domain combinations
    - Generates insights for workflow optimization

    Args:
        time_window_hours: Time window to analyze (default 1 week)

    Returns:
        Cross-domain analysis results
    """
    try:
        logger.info(f"Starting cross-domain pattern analysis (window: {time_window_hours}h)")

        async def _analyze():
            from backend.database.connection import get_async_session
            from backend.services.acd_cross_thinking import ACDCrossThinking
            from backend.models.acd import AIDomain

            async with get_async_session() as session:
                cross_thinking = ACDCrossThinking(session)

                # Analyze common domain combinations
                domains = [
                    AIDomain.TEXT_GENERATION,
                    AIDomain.IMAGE_GENERATION,
                    AIDomain.VIDEO_GENERATION,
                    AIDomain.AUDIO_GENERATION,
                    AIDomain.CODE_GENERATION,
                    AIDomain.ANALYSIS,
                ]

                analysis = await cross_thinking.analyze_cross_domain_patterns(
                    domains=domains,
                    time_window_hours=time_window_hours,
                )

                return {
                    "domains_analyzed": analysis.domains_analyzed,
                    "patterns_found": len(analysis.cross_patterns),
                    "correlations_found": len(analysis.correlations),
                    "insights": analysis.insights,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }

        results = run_async(_analyze())

        logger.info(
            f"Cross-domain analysis complete: "
            f"{results.get('patterns_found', 0)} patterns, "
            f"{results.get('correlations_found', 0)} correlations"
        )
        return results

    except Exception as e:
        logger.error(f"Cross-domain pattern analysis failed: {e}")
        raise self.retry(exc=e)


@app.task(
    name="backend.tasks.acd_tasks.generate_improvement_suggestions",
    bind=True,
    max_retries=2,
    default_retry_delay=600,
)
def generate_improvement_suggestions(self, time_window_hours: int = 168):
    """
    Generate improvement suggestions for human review.

    This task:
    - Analyzes decision quality over time
    - Identifies failure patterns
    - Generates actionable improvement suggestions

    Args:
        time_window_hours: Time window to analyze (default 1 week)

    Returns:
        List of improvement suggestions
    """
    try:
        logger.info(f"Generating improvement suggestions (window: {time_window_hours}h)")

        async def _generate():
            from backend.database.connection import get_async_session
            from backend.services.acd_self_improvement import ACDSelfImprovement

            async with get_async_session() as session:
                improvement_service = ACDSelfImprovement(session)
                suggestions = await improvement_service.generate_improvement_suggestions(
                    time_window_hours=time_window_hours
                )

                return {
                    "suggestions": [s.to_dict() for s in suggestions],
                    "count": len(suggestions),
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }

        results = run_async(_generate())

        logger.info(f"Generated {results.get('count', 0)} improvement suggestions")
        return results

    except Exception as e:
        logger.error(f"Improvement suggestion generation failed: {e}")
        raise self.retry(exc=e)


@app.task(
    name="backend.tasks.acd_tasks.learn_from_hil_ratings",
    bind=True,
    max_retries=3,
    default_retry_delay=300,
)
def learn_from_hil_ratings(self, time_window_hours: int = 24):
    """
    Learn from Human-in-the-Loop ratings to improve future generations.

    This task:
    - Analyzes recent HIL ratings
    - Updates correlation patterns based on ratings
    - Adjusts model/workflow effectiveness scores

    Args:
        time_window_hours: Time window to analyze

    Returns:
        Learning summary
    """
    try:
        logger.info(f"Learning from HIL ratings (window: {time_window_hours}h)")

        async def _learn():
            from backend.database.connection import get_async_session
            from backend.services.acd_correlation_engine import ACDCorrelationEngine
            from backend.models.acd import ACDContextModel
            from sqlalchemy import and_, select
            from datetime import timedelta

            async with get_async_session() as session:
                cutoff = datetime.now(timezone.utc) - timedelta(hours=time_window_hours)

                # Get recently rated contexts
                stmt = select(ACDContextModel).where(
                    and_(
                        ACDContextModel.hil_rated_at >= cutoff,
                        ACDContextModel.hil_rating.isnot(None),
                    )
                )
                result = await session.execute(stmt)
                rated_contexts = result.scalars().all()

                correlation_engine = ACDCorrelationEngine(session)
                learned_count = 0

                for ctx in rated_contexts:
                    try:
                        # Create outcome based on rating
                        outcome = {
                            "quality_score": ctx.hil_rating / 5.0,
                            "success": ctx.hil_rating >= 3,
                        }

                        await correlation_engine.learn_from_outcome(ctx.id, outcome)
                        learned_count += 1
                    except Exception as learn_error:
                        logger.warning(
                            f"Failed to learn from context {ctx.id}: {learn_error}"
                        )

                return {
                    "contexts_processed": len(rated_contexts),
                    "contexts_learned": learned_count,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }

        results = run_async(_learn())

        logger.info(
            f"HIL learning complete: {results.get('contexts_learned', 0)} contexts learned"
        )
        return results

    except Exception as e:
        logger.error(f"HIL learning failed: {e}")
        raise self.retry(exc=e)


@app.task(
    name="backend.tasks.acd_tasks.extract_success_patterns",
    bind=True,
    max_retries=2,
    default_retry_delay=600,
)
def extract_success_patterns(self, time_window_hours: int = 168, min_rating: int = 4):
    """
    Extract success patterns from highly-rated generations.

    This task:
    - Analyzes successful generations
    - Identifies common patterns
    - Generates recommendations

    Args:
        time_window_hours: Time window to analyze (default 1 week)
        min_rating: Minimum rating to consider successful

    Returns:
        Success patterns and recommendations
    """
    try:
        logger.info(
            f"Extracting success patterns (window: {time_window_hours}h, min_rating: {min_rating})"
        )

        async def _extract():
            from backend.database.connection import get_async_session
            from backend.services.acd_correlation_engine import ACDCorrelationEngine

            async with get_async_session() as session:
                correlation_engine = ACDCorrelationEngine(session)
                patterns = await correlation_engine.extract_success_patterns(
                    domain=None,  # All domains
                    time_window_hours=time_window_hours,
                    min_rating=min_rating,
                )

                return patterns

        results = run_async(_extract())

        logger.info(
            f"Success pattern extraction complete: "
            f"{results.get('total_successful', 0)} successful contexts analyzed"
        )
        return results

    except Exception as e:
        logger.error(f"Success pattern extraction failed: {e}")
        raise self.retry(exc=e)


@app.task(
    name="backend.tasks.acd_tasks.run_acd_learning_cycle",
    bind=True,
)
def run_acd_learning_cycle(self):
    """
    Run a complete ACD learning cycle.

    This orchestration task runs all learning tasks in sequence:
    1. Memory consolidation
    2. HIL learning
    3. Decision weight updates
    4. Cross-domain pattern analysis
    5. Improvement suggestion generation

    Returns:
        Summary of all learning tasks
    """
    try:
        logger.info("Starting complete ACD learning cycle")

        results = {
            "start_time": datetime.now(timezone.utc).isoformat(),
            "tasks": {},
        }

        # 1. Memory consolidation
        try:
            results["tasks"]["memory_consolidation"] = consolidate_memories(batch_size=100)
        except Exception as e:
            results["tasks"]["memory_consolidation"] = {"error": str(e)}

        # 2. HIL learning
        try:
            results["tasks"]["hil_learning"] = learn_from_hil_ratings(time_window_hours=24)
        except Exception as e:
            results["tasks"]["hil_learning"] = {"error": str(e)}

        # 3. Decision weight updates
        try:
            results["tasks"]["weight_updates"] = update_decision_weights(time_window_hours=24)
        except Exception as e:
            results["tasks"]["weight_updates"] = {"error": str(e)}

        # 4. Cross-domain analysis
        try:
            results["tasks"]["cross_domain_analysis"] = analyze_cross_domain_patterns(
                time_window_hours=168
            )
        except Exception as e:
            results["tasks"]["cross_domain_analysis"] = {"error": str(e)}

        # 5. Improvement suggestions
        try:
            results["tasks"]["improvement_suggestions"] = generate_improvement_suggestions(
                time_window_hours=168
            )
        except Exception as e:
            results["tasks"]["improvement_suggestions"] = {"error": str(e)}

        results["end_time"] = datetime.now(timezone.utc).isoformat()

        logger.info("ACD learning cycle complete")
        return results

    except Exception as e:
        logger.error(f"ACD learning cycle failed: {e}")
        return {"error": str(e)}
