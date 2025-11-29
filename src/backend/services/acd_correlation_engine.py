"""
ACD Correlation Engine

Enables ACD to find patterns across different contexts
and learn from successful generations.
"""

from collections import Counter
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional
from uuid import UUID

from sqlalchemy import and_, func, or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.config.logging import get_logger
from backend.models.acd import (
    ACDContextModel,
    ACDContextResponse,
    AIDomain,
    DOMAIN_COMPATIBILITY,
)

logger = get_logger(__name__)


class ACDCorrelationEngine:
    """
    Enables ACD to find patterns across different contexts
    and learn from successful generations.

    Key capabilities:
    - Find historically similar contexts to inform decisions
    - Extract success patterns from past generations
    - Learn from outcomes to improve future decisions
    - Correlate contexts across compatible domains
    """

    def __init__(self, db_session: AsyncSession):
        """
        Initialize the correlation engine.

        Args:
            db_session: Database session for persistence
        """
        self.db = db_session

    async def find_similar_contexts(
        self,
        context: ACDContextModel,
        similarity_threshold: float = 0.7,
        max_results: int = 10,
        time_window_hours: Optional[int] = None,
    ) -> List[ACDContextResponse]:
        """
        Find historically similar contexts to inform decisions.

        Uses:
        - Domain/subdomain matching
        - Phase matching
        - Model/workflow matching
        - Outcome correlation (success patterns)

        Args:
            context: The context to find similar ones for
            similarity_threshold: Minimum similarity score (0-1)
            max_results: Maximum number of results to return
            time_window_hours: Optional time window to limit search

        Returns:
            List of similar contexts sorted by similarity score
        """
        try:
            # Build base query
            conditions = [
                ACDContextModel.id != context.id,  # Exclude self
            ]

            # Add time window filter if specified
            if time_window_hours:
                cutoff = datetime.now(timezone.utc) - timedelta(hours=time_window_hours)
                conditions.append(ACDContextModel.created_at >= cutoff)

            # Get candidate contexts - start with same domain
            if context.ai_domain:
                domain_conditions = [ACDContextModel.ai_domain == context.ai_domain]

                # Also include compatible domains
                compatible = DOMAIN_COMPATIBILITY.get(
                    AIDomain(context.ai_domain) if context.ai_domain else None, []
                )
                if compatible:
                    domain_conditions.extend(
                        [
                            ACDContextModel.ai_domain == d.value
                            for d in compatible
                        ]
                    )
                conditions.append(or_(*domain_conditions))

            stmt = (
                select(ACDContextModel)
                .where(and_(*conditions))
                .order_by(ACDContextModel.created_at.desc())
                .limit(max_results * 5)  # Get more to filter by similarity
            )

            result = await self.db.execute(stmt)
            candidates = result.scalars().all()

            # Calculate similarity scores
            scored_contexts = []
            for candidate in candidates:
                score = self._calculate_similarity(context, candidate)
                if score >= similarity_threshold:
                    scored_contexts.append((candidate, score))

            # Sort by score descending
            scored_contexts.sort(key=lambda x: x[1], reverse=True)

            # Convert to response objects
            similar_contexts = [
                ACDContextResponse.model_validate(ctx)
                for ctx, _ in scored_contexts[:max_results]
            ]

            logger.info(
                f"Found {len(similar_contexts)} similar contexts for {context.id}"
            )
            return similar_contexts

        except Exception as e:
            logger.error(f"Failed to find similar contexts: {e}")
            return []

    def _calculate_similarity(
        self,
        context_a: ACDContextModel,
        context_b: ACDContextModel,
    ) -> float:
        """
        Calculate similarity score between two contexts.

        Components:
        - Domain match: 0.25
        - Subdomain match: 0.15
        - Phase match: 0.20
        - Model match: 0.15
        - Workflow match: 0.10
        - Parameter similarity: 0.15

        Args:
            context_a: First context
            context_b: Second context

        Returns:
            Similarity score between 0 and 1
        """
        score = 0.0

        # Domain match (0.25)
        if context_a.ai_domain and context_b.ai_domain:
            if context_a.ai_domain == context_b.ai_domain:
                score += 0.25
            elif context_b.ai_domain in [
                d.value for d in DOMAIN_COMPATIBILITY.get(
                    AIDomain(context_a.ai_domain), []
                )
            ]:
                score += 0.15  # Compatible domain

        # Subdomain match (0.15)
        if context_a.ai_subdomain and context_b.ai_subdomain:
            if context_a.ai_subdomain == context_b.ai_subdomain:
                score += 0.15

        # Phase match (0.20)
        if context_a.ai_phase == context_b.ai_phase:
            score += 0.20

        # Model match (0.15)
        if context_a.model_id and context_b.model_id:
            if context_a.model_id == context_b.model_id:
                score += 0.15

        # Workflow match (0.10)
        if context_a.workflow_id and context_b.workflow_id:
            if context_a.workflow_id == context_b.workflow_id:
                score += 0.10

        # LoRA overlap (0.15)
        if context_a.lora_ids and context_b.lora_ids:
            loras_a = set(context_a.lora_ids)
            loras_b = set(context_b.lora_ids)
            if loras_a and loras_b:
                overlap = len(loras_a & loras_b) / len(loras_a | loras_b)
                score += 0.15 * overlap

        return score

    async def extract_success_patterns(
        self,
        domain: Optional[AIDomain] = None,
        time_window_hours: int = 168,  # 1 week
        min_rating: int = 4,
    ) -> Dict[str, Any]:
        """
        Analyze successful contexts to extract winning patterns.

        Returns:
        - Common prompt structures
        - Optimal parameter combinations
        - Best-performing model/quality settings
        - Timing patterns for engagement

        Args:
            domain: Optional domain filter
            time_window_hours: Time window for analysis
            min_rating: Minimum HIL rating to consider successful

        Returns:
            Dictionary of success patterns
        """
        try:
            cutoff = datetime.now(timezone.utc) - timedelta(hours=time_window_hours)

            conditions = [
                ACDContextModel.created_at >= cutoff,
                ACDContextModel.hil_rating >= min_rating,
                ACDContextModel.ai_state == "DONE",
            ]

            if domain:
                conditions.append(ACDContextModel.ai_domain == domain.value)

            stmt = select(ACDContextModel).where(and_(*conditions))
            result = await self.db.execute(stmt)
            contexts = result.scalars().all()

            if not contexts:
                return {
                    "total_successful": 0,
                    "patterns": {},
                    "recommendations": [],
                }

            # Analyze patterns
            patterns = {
                "models": Counter(),
                "workflows": Counter(),
                "loras": Counter(),
                "phases": Counter(),
                "quality_settings": Counter(),
                "parameters": [],
            }

            for ctx in contexts:
                if ctx.model_id:
                    patterns["models"][ctx.model_id] += 1
                if ctx.workflow_id:
                    patterns["workflows"][ctx.workflow_id] += 1
                if ctx.lora_ids:
                    for lora in ctx.lora_ids:
                        patterns["loras"][lora] += 1
                if ctx.ai_phase:
                    patterns["phases"][ctx.ai_phase] += 1
                if ctx.generation_params:
                    patterns["parameters"].append(ctx.generation_params)

            # Generate recommendations
            recommendations = []

            # Best model
            if patterns["models"]:
                best_model, count = patterns["models"].most_common(1)[0]
                recommendations.append({
                    "type": "model",
                    "value": best_model,
                    "confidence": count / len(contexts),
                    "reason": f"Used in {count} successful generations",
                })

            # Best workflow
            if patterns["workflows"]:
                best_workflow, count = patterns["workflows"].most_common(1)[0]
                recommendations.append({
                    "type": "workflow",
                    "value": best_workflow,
                    "confidence": count / len(contexts),
                    "reason": f"Used in {count} successful generations",
                })

            # Top LoRAs
            if patterns["loras"]:
                for lora, count in patterns["loras"].most_common(3):
                    recommendations.append({
                        "type": "lora",
                        "value": lora,
                        "confidence": count / len(contexts),
                        "reason": f"Used in {count} successful generations",
                    })

            # Analyze common parameters
            common_params = self._extract_common_parameters(patterns["parameters"])

            return {
                "total_successful": len(contexts),
                "domain": domain.value if domain else "all",
                "time_window_hours": time_window_hours,
                "patterns": {
                    "top_models": dict(patterns["models"].most_common(5)),
                    "top_workflows": dict(patterns["workflows"].most_common(5)),
                    "top_loras": dict(patterns["loras"].most_common(5)),
                    "phases": dict(patterns["phases"]),
                    "common_parameters": common_params,
                },
                "recommendations": recommendations,
            }

        except Exception as e:
            logger.error(f"Failed to extract success patterns: {e}")
            return {
                "total_successful": 0,
                "patterns": {},
                "recommendations": [],
                "error": str(e),
            }

    def _extract_common_parameters(
        self, parameters_list: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Extract common parameters from successful generations.

        Args:
            parameters_list: List of generation parameter dictionaries

        Returns:
            Dictionary of common parameters and their typical values
        """
        if not parameters_list:
            return {}

        common = {}

        # Common parameter keys to track
        param_keys = [
            "steps", "cfg_scale", "sampler", "seed", "width", "height",
            "clip_skip", "denoising_strength", "negative_prompt"
        ]

        for key in param_keys:
            values = []
            for params in parameters_list:
                if isinstance(params, dict) and key in params:
                    values.append(params[key])

            if values:
                # For numeric values, compute average
                if all(isinstance(v, (int, float)) for v in values):
                    common[key] = {
                        "average": sum(values) / len(values),
                        "min": min(values),
                        "max": max(values),
                        "count": len(values),
                    }
                else:
                    # For non-numeric, find most common
                    counter = Counter(str(v) for v in values)
                    most_common = counter.most_common(1)[0]
                    common[key] = {
                        "most_common": most_common[0],
                        "frequency": most_common[1] / len(values),
                        "count": len(values),
                    }

        return common

    async def learn_from_outcome(
        self,
        context_id: UUID,
        outcome: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Update ACD's knowledge base from generation outcomes.

        Learns:
        - What prompts lead to high engagement
        - Which model combinations work best
        - Failure patterns to avoid

        Args:
            context_id: UUID of the context to learn from
            outcome: Outcome data including engagement metrics, quality scores

        Returns:
            Learning results and insights
        """
        try:
            # Find the context
            stmt = select(ACDContextModel).where(ACDContextModel.id == context_id)
            result = await self.db.execute(stmt)
            context = result.scalar_one_or_none()

            if not context:
                raise ValueError(f"Context {context_id} not found")

            # Extract outcome data
            engagement_metrics = outcome.get("engagement_metrics", {})
            quality_score = outcome.get("quality_score")
            success = outcome.get("success", True)

            # Update context with outcome
            if engagement_metrics:
                context.engagement_metrics = engagement_metrics

            if quality_score is not None:
                context.content_quality_score = quality_score

            # Calculate outcome score
            outcome_score = self._calculate_outcome_score(
                engagement_metrics, quality_score, success
            )
            context.outcome_score = outcome_score

            # Adjust learning weight based on outcome
            if outcome_score >= 0.8:
                context.learning_weight = 1.5  # Learn more from excellent outcomes
                context.memory_importance = 0.8
            elif outcome_score >= 0.6:
                context.learning_weight = 1.0  # Normal learning
                context.memory_importance = 0.5
            elif outcome_score < 0.3:
                context.learning_weight = 1.2  # Learn from failures too
                context.memory_importance = 0.6  # Remember failures

            await self.db.commit()
            await self.db.refresh(context)

            # Find and update correlations
            correlations = await self._update_correlations(context)

            logger.info(
                f"Learned from outcome for context {context_id}: "
                f"score={outcome_score:.2f}, correlations={len(correlations)}"
            )

            return {
                "context_id": str(context_id),
                "outcome_score": outcome_score,
                "learning_weight": context.learning_weight,
                "memory_importance": context.memory_importance,
                "correlations_updated": len(correlations),
            }

        except Exception as e:
            logger.error(f"Failed to learn from outcome: {e}")
            await self.db.rollback()
            raise

    def _calculate_outcome_score(
        self,
        engagement_metrics: Dict[str, Any],
        quality_score: Optional[float],
        success: bool,
    ) -> float:
        """
        Calculate a normalized outcome score.

        Args:
            engagement_metrics: Social engagement metrics
            quality_score: Quality assessment score (0-1)
            success: Whether generation was successful

        Returns:
            Outcome score between 0 and 1
        """
        if not success:
            return 0.0

        score = 0.5  # Base score for successful generation

        # Quality component (0.3)
        if quality_score is not None:
            score += 0.3 * quality_score

        # Engagement component (0.2)
        if engagement_metrics:
            engagement_score = 0.0

            # Normalize engagement metrics
            likes = engagement_metrics.get("likes", 0)
            comments = engagement_metrics.get("comments", 0)
            shares = engagement_metrics.get("shares", 0)
            views = engagement_metrics.get("views", 1)  # Avoid division by zero

            if views > 0:
                engagement_rate = (likes + comments * 2 + shares * 3) / views
                # Cap at reasonable rate
                engagement_score = min(1.0, engagement_rate * 10)

            score += 0.2 * engagement_score

        return min(1.0, score)

    async def _update_correlations(
        self, context: ACDContextModel
    ) -> List[Dict[str, Any]]:
        """
        Find and update correlations with similar contexts.

        Args:
            context: The context to find correlations for

        Returns:
            List of correlation updates made
        """
        correlations = []

        try:
            # Find similar successful contexts
            similar = await self.find_similar_contexts(
                context,
                similarity_threshold=0.5,
                max_results=20,
                time_window_hours=168,  # Last week
            )

            if not similar:
                return correlations

            # Update correlation scores
            correlation_scores = {}
            related_ids = []

            for sim_ctx in similar:
                # Get the actual context for comparison
                stmt = select(ACDContextModel).where(ACDContextModel.id == sim_ctx.id)
                result = await self.db.execute(stmt)
                sim_model = result.scalar_one_or_none()

                if sim_model:
                    score = self._calculate_similarity(context, sim_model)
                    correlation_scores[str(sim_ctx.id)] = score
                    related_ids.append(str(sim_ctx.id))

                    correlations.append({
                        "related_id": str(sim_ctx.id),
                        "score": score,
                    })

            # Update context with correlations
            context.correlation_scores = correlation_scores
            context.related_contexts = related_ids

            await self.db.commit()

        except Exception as e:
            logger.error(f"Failed to update correlations: {e}")

        return correlations

    async def get_correlation_insights(
        self,
        domain: Optional[AIDomain] = None,
        time_window_hours: int = 168,
    ) -> Dict[str, Any]:
        """
        Get cross-context correlation insights.

        Args:
            domain: Optional domain filter
            time_window_hours: Time window for analysis

        Returns:
            Correlation insights and patterns
        """
        try:
            cutoff = datetime.now(timezone.utc) - timedelta(hours=time_window_hours)

            conditions = [
                ACDContextModel.created_at >= cutoff,
                ACDContextModel.correlation_scores.isnot(None),
            ]

            if domain:
                conditions.append(ACDContextModel.ai_domain == domain.value)

            stmt = select(ACDContextModel).where(and_(*conditions))
            result = await self.db.execute(stmt)
            contexts = result.scalars().all()

            if not contexts:
                return {
                    "total_contexts": 0,
                    "avg_correlations": 0,
                    "insights": [],
                }

            # Analyze correlations
            total_correlations = 0
            high_correlation_pairs = []
            domain_correlations = Counter()

            for ctx in contexts:
                if ctx.correlation_scores:
                    scores = ctx.correlation_scores
                    total_correlations += len(scores)

                    for related_id, score in scores.items():
                        if score >= 0.7:
                            high_correlation_pairs.append({
                                "context_a": str(ctx.id),
                                "context_b": related_id,
                                "score": score,
                            })

                if ctx.ai_domain:
                    domain_correlations[ctx.ai_domain] += 1

            avg_correlations = (
                total_correlations / len(contexts) if contexts else 0
            )

            return {
                "total_contexts": len(contexts),
                "total_correlations": total_correlations,
                "avg_correlations_per_context": round(avg_correlations, 2),
                "high_correlation_pairs": high_correlation_pairs[:20],
                "contexts_by_domain": dict(domain_correlations),
                "time_window_hours": time_window_hours,
            }

        except Exception as e:
            logger.error(f"Failed to get correlation insights: {e}")
            return {
                "total_contexts": 0,
                "error": str(e),
            }
