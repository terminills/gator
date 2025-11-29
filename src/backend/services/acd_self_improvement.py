"""
ACD Self-Improvement Service

Enables ACD to improve its own decision-making over time through:
- Decision quality evaluation
- Weight adjustment based on outcomes
- Improvement suggestion generation
"""

from collections import Counter
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional
from uuid import UUID

from sqlalchemy import and_, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.config.logging import get_logger
from backend.models.acd import (
    ACDContextModel,
    ACDContextResponse,
    AIDomain,
    AIState,
)

logger = get_logger(__name__)


class DecisionAnalysis:
    """Analysis of ACD decision quality."""

    def __init__(
        self,
        total_decisions: int = 0,
        correct_decisions: int = 0,
        accuracy_rate: float = 0.0,
        by_domain: Dict[str, Dict[str, Any]] = None,
        failure_patterns: List[Dict[str, Any]] = None,
        improvement_opportunities: List[Dict[str, Any]] = None,
    ):
        self.total_decisions = total_decisions
        self.correct_decisions = correct_decisions
        self.accuracy_rate = accuracy_rate
        self.by_domain = by_domain or {}
        self.failure_patterns = failure_patterns or []
        self.improvement_opportunities = improvement_opportunities or []


class Suggestion:
    """Actionable improvement suggestion."""

    def __init__(
        self,
        suggestion_type: str,
        title: str,
        description: str,
        priority: str = "medium",
        estimated_impact: float = 0.0,
        affected_domains: List[str] = None,
        implementation_notes: str = None,
    ):
        self.suggestion_type = suggestion_type
        self.title = title
        self.description = description
        self.priority = priority
        self.estimated_impact = estimated_impact
        self.affected_domains = affected_domains or []
        self.implementation_notes = implementation_notes

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.suggestion_type,
            "title": self.title,
            "description": self.description,
            "priority": self.priority,
            "estimated_impact": self.estimated_impact,
            "affected_domains": self.affected_domains,
            "implementation_notes": self.implementation_notes,
        }


class ACDSelfImprovement:
    """
    Enables ACD to improve its own decision-making over time.

    Key capabilities:
    - Evaluate decision quality against actual outcomes
    - Identify areas of consistent failure
    - Update decision weights based on results
    - Generate actionable improvement suggestions
    """

    # Thresholds for analysis
    MIN_SAMPLES_FOR_ANALYSIS = 10
    SUCCESS_THRESHOLD = 0.7  # 70% outcome score = success
    HIGH_CONFIDENCE_THRESHOLD = 0.8
    LOW_CONFIDENCE_THRESHOLD = 0.4

    # Weight adjustment parameters
    LEARNING_RATE = 0.1
    MAX_WEIGHT_ADJUSTMENT = 0.2
    MIN_WEIGHT = 0.1
    MAX_WEIGHT = 2.0

    def __init__(self, db_session: AsyncSession):
        """
        Initialize the self-improvement service.

        Args:
            db_session: Database session for persistence
        """
        self.db = db_session

    async def evaluate_decisions(
        self,
        time_window_hours: int = 24,
        domain: Optional[AIDomain] = None,
    ) -> DecisionAnalysis:
        """
        Analyze ACD decisions and their outcomes.

        Returns:
        - Decision accuracy rate
        - Areas of consistent failure
        - Opportunities for improvement

        Args:
            time_window_hours: Time window to analyze
            domain: Optional domain filter

        Returns:
            DecisionAnalysis with evaluation results
        """
        try:
            cutoff = datetime.now(timezone.utc) - timedelta(hours=time_window_hours)

            conditions = [
                ACDContextModel.created_at >= cutoff,
                ACDContextModel.ai_state.in_([
                    AIState.DONE.value,
                    AIState.FAILED.value,
                    AIState.CANCELLED.value,
                ]),
            ]

            if domain:
                conditions.append(ACDContextModel.ai_domain == domain.value)

            stmt = select(ACDContextModel).where(and_(*conditions))
            result = await self.db.execute(stmt)
            contexts = result.scalars().all()

            if not contexts:
                return DecisionAnalysis()

            # Analyze decisions
            total = len(contexts)
            successful = 0
            by_domain: Dict[str, Dict[str, Any]] = {}
            failure_reasons: List[str] = []
            confidence_vs_outcome: List[tuple] = []

            for ctx in contexts:
                # Determine if decision was successful
                is_success = (
                    ctx.ai_state == AIState.DONE.value
                    and (ctx.outcome_score is None or ctx.outcome_score >= self.SUCCESS_THRESHOLD)
                    and (ctx.hil_rating is None or ctx.hil_rating >= 3)
                )

                if is_success:
                    successful += 1

                # Track by domain
                domain_key = ctx.ai_domain or "unknown"
                if domain_key not in by_domain:
                    by_domain[domain_key] = {
                        "total": 0,
                        "successful": 0,
                        "avg_confidence": [],
                        "avg_outcome": [],
                    }
                by_domain[domain_key]["total"] += 1
                if is_success:
                    by_domain[domain_key]["successful"] += 1
                if ctx.ai_confidence:
                    confidence_value = self._confidence_to_numeric(ctx.ai_confidence)
                    by_domain[domain_key]["avg_confidence"].append(confidence_value)
                if ctx.outcome_score is not None:
                    by_domain[domain_key]["avg_outcome"].append(ctx.outcome_score)

                # Track failures
                if not is_success:
                    reason = ctx.runtime_err or ctx.compiler_err or "Unknown failure"
                    failure_reasons.append(reason[:200])

                # Track confidence vs outcome for calibration
                if ctx.ai_confidence and ctx.outcome_score is not None:
                    confidence_vs_outcome.append((
                        self._confidence_to_numeric(ctx.ai_confidence),
                        ctx.outcome_score,
                    ))

            # Calculate accuracy
            accuracy_rate = successful / total if total > 0 else 0.0

            # Calculate domain averages
            for domain_key, data in by_domain.items():
                data["accuracy"] = (
                    data["successful"] / data["total"]
                    if data["total"] > 0 else 0.0
                )
                data["avg_confidence"] = (
                    sum(data["avg_confidence"]) / len(data["avg_confidence"])
                    if data["avg_confidence"] else None
                )
                data["avg_outcome"] = (
                    sum(data["avg_outcome"]) / len(data["avg_outcome"])
                    if data["avg_outcome"] else None
                )

            # Identify failure patterns
            failure_patterns = self._analyze_failure_patterns(failure_reasons)

            # Identify improvement opportunities
            improvement_opportunities = self._identify_improvements(
                by_domain, accuracy_rate, confidence_vs_outcome
            )

            logger.info(
                f"Decision evaluation: {successful}/{total} successful "
                f"({accuracy_rate:.1%} accuracy)"
            )

            return DecisionAnalysis(
                total_decisions=total,
                correct_decisions=successful,
                accuracy_rate=accuracy_rate,
                by_domain=by_domain,
                failure_patterns=failure_patterns,
                improvement_opportunities=improvement_opportunities,
            )

        except Exception as e:
            logger.error(f"Failed to evaluate decisions: {e}")
            return DecisionAnalysis()

    def _confidence_to_numeric(self, confidence: str) -> float:
        """Convert confidence level to numeric value."""
        confidence_map = {
            "VALIDATED": 1.0,
            "CONFIDENT": 0.8,
            "UNCERTAIN": 0.5,
            "HYPOTHESIS": 0.3,
            "EXPERIMENTAL": 0.2,
        }
        return confidence_map.get(confidence, 0.5)

    def _analyze_failure_patterns(
        self, failure_reasons: List[str]
    ) -> List[Dict[str, Any]]:
        """Identify common patterns in failures."""
        if not failure_reasons:
            return []

        # Categorize failures
        categories: Dict[str, int] = Counter()
        for reason in failure_reasons:
            reason_lower = reason.lower()
            if "timeout" in reason_lower:
                categories["timeout"] += 1
            elif "model" in reason_lower and ("not found" in reason_lower or "unavailable" in reason_lower):
                categories["model_unavailable"] += 1
            elif "memory" in reason_lower or "oom" in reason_lower:
                categories["out_of_memory"] += 1
            elif "validation" in reason_lower or "invalid" in reason_lower:
                categories["validation_error"] += 1
            elif "permission" in reason_lower or "access" in reason_lower:
                categories["permission_error"] += 1
            elif "network" in reason_lower or "connection" in reason_lower:
                categories["network_error"] += 1
            else:
                categories["other"] += 1

        # Build pattern list
        patterns = []
        total_failures = sum(categories.values())
        for category, count in categories.most_common(5):
            if count > 0:
                patterns.append({
                    "category": category,
                    "count": count,
                    "percentage": count / total_failures * 100,
                    "suggested_action": self._get_action_for_failure(category),
                })

        return patterns

    def _get_action_for_failure(self, category: str) -> str:
        """Get suggested action for a failure category."""
        actions = {
            "timeout": "Increase timeout limits or optimize processing",
            "model_unavailable": "Implement fallback models or check model availability",
            "out_of_memory": "Reduce batch sizes or optimize memory usage",
            "validation_error": "Improve input validation and error messages",
            "permission_error": "Review access controls and permissions",
            "network_error": "Add retry logic and connection pooling",
            "other": "Investigate specific error patterns",
        }
        return actions.get(category, "Review and categorize this error type")

    def _identify_improvements(
        self,
        by_domain: Dict[str, Dict[str, Any]],
        overall_accuracy: float,
        confidence_vs_outcome: List[tuple],
    ) -> List[Dict[str, Any]]:
        """Identify specific improvement opportunities."""
        improvements = []

        # Check for underperforming domains
        for domain, data in by_domain.items():
            if data["total"] >= self.MIN_SAMPLES_FOR_ANALYSIS:
                if data["accuracy"] < overall_accuracy - 0.1:
                    improvements.append({
                        "type": "domain_underperformance",
                        "domain": domain,
                        "current_accuracy": data["accuracy"],
                        "target_accuracy": overall_accuracy,
                        "priority": "high" if data["accuracy"] < 0.5 else "medium",
                    })

        # Check for confidence calibration issues
        if len(confidence_vs_outcome) >= self.MIN_SAMPLES_FOR_ANALYSIS:
            avg_confidence = sum(c for c, _ in confidence_vs_outcome) / len(confidence_vs_outcome)
            avg_outcome = sum(o for _, o in confidence_vs_outcome) / len(confidence_vs_outcome)

            if avg_confidence > avg_outcome + 0.2:
                improvements.append({
                    "type": "overconfidence",
                    "description": "System is overconfident in its predictions",
                    "avg_confidence": avg_confidence,
                    "avg_outcome": avg_outcome,
                    "priority": "medium",
                })
            elif avg_confidence < avg_outcome - 0.2:
                improvements.append({
                    "type": "underconfidence",
                    "description": "System is underconfident in its predictions",
                    "avg_confidence": avg_confidence,
                    "avg_outcome": avg_outcome,
                    "priority": "low",
                })

        return improvements

    async def update_decision_weights(
        self,
        analysis: DecisionAnalysis,
    ) -> Dict[str, Any]:
        """
        Adjust internal decision weights based on outcomes.

        Uses reinforcement learning principles:
        - Increase weight for successful patterns
        - Decrease weight for failure patterns
        - Explore new approaches periodically

        Args:
            analysis: Decision analysis results

        Returns:
            Summary of weight adjustments made
        """
        try:
            adjustments = []

            # Get recent contexts with clear outcomes
            cutoff = datetime.now(timezone.utc) - timedelta(hours=24)
            stmt = select(ACDContextModel).where(
                and_(
                    ACDContextModel.created_at >= cutoff,
                    ACDContextModel.outcome_score.isnot(None),
                    ACDContextModel.improvement_applied.is_(False),
                )
            )
            result = await self.db.execute(stmt)
            contexts = result.scalars().all()

            for ctx in contexts:
                # Calculate weight adjustment based on outcome
                current_weight = ctx.learning_weight or 1.0
                outcome = ctx.outcome_score or 0.5

                # Adjust towards target based on outcome
                if outcome >= self.SUCCESS_THRESHOLD:
                    # Successful - increase weight (cap at MAX_WEIGHT)
                    adjustment = self.LEARNING_RATE * (1 - current_weight / self.MAX_WEIGHT)
                else:
                    # Unsuccessful - decrease weight (floor at MIN_WEIGHT)
                    # Normalize to ensure negative adjustment for failures
                    weight_range = self.MAX_WEIGHT - self.MIN_WEIGHT
                    normalized_position = (current_weight - self.MIN_WEIGHT) / weight_range
                    adjustment = -self.LEARNING_RATE * normalized_position

                # Cap adjustment
                adjustment = max(-self.MAX_WEIGHT_ADJUSTMENT,
                               min(self.MAX_WEIGHT_ADJUSTMENT, adjustment))

                new_weight = max(self.MIN_WEIGHT,
                               min(self.MAX_WEIGHT, current_weight + adjustment))

                # Apply adjustment
                ctx.learning_weight = new_weight
                ctx.improvement_applied = True
                ctx.improvement_notes = (
                    f"Weight adjusted from {current_weight:.3f} to {new_weight:.3f} "
                    f"based on outcome score {outcome:.3f}"
                )

                adjustments.append({
                    "context_id": str(ctx.id),
                    "old_weight": current_weight,
                    "new_weight": new_weight,
                    "adjustment": adjustment,
                    "outcome": outcome,
                })

            await self.db.commit()

            logger.info(f"Updated weights for {len(adjustments)} contexts")

            return {
                "total_adjusted": len(adjustments),
                "adjustments": adjustments[:20],  # Return first 20
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

        except Exception as e:
            logger.error(f"Failed to update decision weights: {e}")
            await self.db.rollback()
            return {"error": str(e)}

    async def generate_improvement_suggestions(
        self,
        time_window_hours: int = 168,  # 1 week
    ) -> List[Suggestion]:
        """
        Generate actionable improvements for human review.

        Includes:
        - Code changes to improve patterns
        - Configuration adjustments
        - New capability requests

        Args:
            time_window_hours: Time window to analyze

        Returns:
            List of improvement suggestions
        """
        try:
            suggestions = []

            # Evaluate decisions for context
            analysis = await self.evaluate_decisions(time_window_hours=time_window_hours)

            # Generate suggestions based on failure patterns
            for pattern in analysis.failure_patterns:
                if pattern["count"] >= 5:  # Only suggest for repeated patterns
                    suggestion = Suggestion(
                        suggestion_type="failure_mitigation",
                        title=f"Address {pattern['category']} failures",
                        description=(
                            f"{pattern['category']} errors account for "
                            f"{pattern['percentage']:.1f}% of failures"
                        ),
                        priority="high" if pattern["percentage"] > 30 else "medium",
                        estimated_impact=pattern["percentage"] / 100,
                        implementation_notes=pattern["suggested_action"],
                    )
                    suggestions.append(suggestion)

            # Generate suggestions based on domain performance
            for domain, data in analysis.by_domain.items():
                if data["total"] >= self.MIN_SAMPLES_FOR_ANALYSIS:
                    if data["accuracy"] < 0.6:
                        suggestion = Suggestion(
                            suggestion_type="domain_improvement",
                            title=f"Improve {domain} domain performance",
                            description=(
                                f"Domain {domain} has only {data['accuracy']:.1%} accuracy "
                                f"across {data['total']} decisions"
                            ),
                            priority="high",
                            estimated_impact=0.2,
                            affected_domains=[domain],
                            implementation_notes=(
                                "Consider adding specialized handling, "
                                "better prompts, or dedicated models for this domain"
                            ),
                        )
                        suggestions.append(suggestion)

            # Generate suggestions based on improvement opportunities
            for opp in analysis.improvement_opportunities:
                if opp["type"] == "overconfidence":
                    suggestion = Suggestion(
                        suggestion_type="calibration",
                        title="Calibrate confidence levels",
                        description=(
                            f"System confidence ({opp['avg_confidence']:.2f}) exceeds "
                            f"actual outcomes ({opp['avg_outcome']:.2f})"
                        ),
                        priority=opp.get("priority", "medium"),
                        estimated_impact=0.15,
                        implementation_notes=(
                            "Reduce baseline confidence levels, add more validation "
                            "checks, or implement uncertainty quantification"
                        ),
                    )
                    suggestions.append(suggestion)

            # Check for low HIL ratings patterns
            low_rating_suggestions = await self._analyze_low_ratings()
            suggestions.extend(low_rating_suggestions)

            logger.info(f"Generated {len(suggestions)} improvement suggestions")

            return suggestions

        except Exception as e:
            logger.error(f"Failed to generate improvement suggestions: {e}")
            return []

    async def _analyze_low_ratings(self) -> List[Suggestion]:
        """Analyze contexts with low HIL ratings for improvement opportunities."""
        suggestions = []

        try:
            cutoff = datetime.now(timezone.utc) - timedelta(hours=168)
            stmt = select(ACDContextModel).where(
                and_(
                    ACDContextModel.created_at >= cutoff,
                    ACDContextModel.hil_rating.isnot(None),
                    ACDContextModel.hil_rating <= 2,  # Poor or Failed
                )
            )
            result = await self.db.execute(stmt)
            low_rated = result.scalars().all()

            if len(low_rated) >= 5:
                # Analyze common tags in low-rated content
                all_tags = []
                for ctx in low_rated:
                    if ctx.hil_rating_tags:
                        all_tags.extend(ctx.hil_rating_tags)

                if all_tags:
                    tag_counts = Counter(all_tags)
                    most_common_tag, count = tag_counts.most_common(1)[0]

                    suggestion = Suggestion(
                        suggestion_type="quality_improvement",
                        title=f"Address recurring {most_common_tag} issues",
                        description=(
                            f"'{most_common_tag}' appears in {count} "
                            f"low-rated generations ({count / len(low_rated) * 100:.0f}%)"
                        ),
                        priority="high",
                        estimated_impact=count / len(low_rated) * 0.3,
                        implementation_notes=(
                            f"Focus on resolving {most_common_tag} issues in generation pipeline"
                        ),
                    )
                    suggestions.append(suggestion)

        except Exception as e:
            logger.error(f"Failed to analyze low ratings: {e}")

        return suggestions

    async def get_improvement_metrics(
        self,
        time_window_hours: int = 168,
    ) -> Dict[str, Any]:
        """
        Get metrics on self-improvement effectiveness.

        Args:
            time_window_hours: Time window to analyze

        Returns:
            Metrics on improvement over time
        """
        try:
            cutoff = datetime.now(timezone.utc) - timedelta(hours=time_window_hours)
            mid_point = datetime.now(timezone.utc) - timedelta(hours=time_window_hours / 2)

            # Get first half
            first_half_stmt = select(ACDContextModel).where(
                and_(
                    ACDContextModel.created_at >= cutoff,
                    ACDContextModel.created_at < mid_point,
                    ACDContextModel.outcome_score.isnot(None),
                )
            )
            first_result = await self.db.execute(first_half_stmt)
            first_half = first_result.scalars().all()

            # Get second half
            second_half_stmt = select(ACDContextModel).where(
                and_(
                    ACDContextModel.created_at >= mid_point,
                    ACDContextModel.outcome_score.isnot(None),
                )
            )
            second_result = await self.db.execute(second_half_stmt)
            second_half = second_result.scalars().all()

            # Calculate averages
            first_avg = (
                sum(c.outcome_score for c in first_half if c.outcome_score) / len(first_half)
                if first_half else None
            )
            second_avg = (
                sum(c.outcome_score for c in second_half if c.outcome_score) / len(second_half)
                if second_half else None
            )

            # Determine trend
            improvement = None
            trend = "insufficient_data"
            if first_avg is not None and second_avg is not None:
                improvement = second_avg - first_avg
                if improvement > 0.05:
                    trend = "improving"
                elif improvement < -0.05:
                    trend = "declining"
                else:
                    trend = "stable"

            # Count improvements applied
            improved_stmt = select(func.count()).where(
                and_(
                    ACDContextModel.created_at >= cutoff,
                    ACDContextModel.improvement_applied.is_(True),
                )
            )
            improved_result = await self.db.execute(improved_stmt)
            improvements_applied = improved_result.scalar() or 0

            return {
                "time_window_hours": time_window_hours,
                "first_period_avg": first_avg,
                "second_period_avg": second_avg,
                "improvement": improvement,
                "trend": trend,
                "first_period_count": len(first_half),
                "second_period_count": len(second_half),
                "improvements_applied": improvements_applied,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

        except Exception as e:
            logger.error(f"Failed to get improvement metrics: {e}")
            return {"error": str(e)}
