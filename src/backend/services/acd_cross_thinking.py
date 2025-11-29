"""
ACD Cross-Thinking Service

Enables reasoning that spans multiple domains for complex multi-step tasks.
"""

import uuid as uuid_module
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
    AISubdomain,
    DOMAIN_COMPATIBILITY,
)

logger = get_logger(__name__)


class CrossDomainAnalysis:
    """Analysis of patterns spanning multiple domains."""

    def __init__(
        self,
        domains_analyzed: List[str] = None,
        cross_patterns: List[Dict[str, Any]] = None,
        correlations: List[Dict[str, Any]] = None,
        insights: List[str] = None,
    ):
        self.domains_analyzed = domains_analyzed or []
        self.cross_patterns = cross_patterns or []
        self.correlations = correlations or []
        self.insights = insights or []


class DomainCombination:
    """Suggested domain combination for a task."""

    def __init__(
        self,
        domains: List[AIDomain],
        subdomains: List[AISubdomain] = None,
        sequence: str = "sequential",  # sequential, parallel, hybrid
        confidence: float = 0.5,
        reasoning: str = None,
        example_workflow: Dict[str, Any] = None,
    ):
        self.domains = domains
        self.subdomains = subdomains or []
        self.sequence = sequence
        self.confidence = confidence
        self.reasoning = reasoning
        self.example_workflow = example_workflow

    def to_dict(self) -> Dict[str, Any]:
        return {
            "domains": [d.value for d in self.domains],
            "subdomains": [s.value for s in self.subdomains] if self.subdomains else [],
            "sequence": self.sequence,
            "confidence": self.confidence,
            "reasoning": self.reasoning,
            "example_workflow": self.example_workflow,
        }


class OrchestrationPlan:
    """Execution plan for multi-domain tasks."""

    def __init__(
        self,
        task_id: str,
        steps: List[Dict[str, Any]] = None,
        parallel_groups: List[List[str]] = None,
        dependencies: Dict[str, List[str]] = None,
        estimated_duration_seconds: int = None,
        fallback_strategies: List[Dict[str, Any]] = None,
    ):
        self.task_id = task_id
        self.steps = steps or []
        self.parallel_groups = parallel_groups or []
        self.dependencies = dependencies or {}
        self.estimated_duration_seconds = estimated_duration_seconds
        self.fallback_strategies = fallback_strategies or []

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "steps": self.steps,
            "parallel_groups": self.parallel_groups,
            "dependencies": self.dependencies,
            "estimated_duration_seconds": self.estimated_duration_seconds,
            "fallback_strategies": self.fallback_strategies,
        }


class ComplexTask:
    """Definition of a complex multi-domain task."""

    def __init__(
        self,
        goal: str,
        requirements: List[str] = None,
        constraints: Dict[str, Any] = None,
        priority: str = "normal",
        deadline: Optional[datetime] = None,
    ):
        self.goal = goal
        self.requirements = requirements or []
        self.constraints = constraints or {}
        self.priority = priority
        self.deadline = deadline


class ACDCrossThinking:
    """
    Enables reasoning that spans multiple domains.

    Key capabilities:
    - Analyze patterns across domains
    - Suggest domain combinations for complex tasks
    - Create execution plans for multi-domain operations
    - Learn from cross-domain successes and failures
    """

    # Correlation thresholds for pattern detection
    CORRELATION_THRESHOLD = 0.2  # Minimum difference for correlation detection
    STRONG_CORRELATION_THRESHOLD = 0.1  # Threshold for strong/positive correlation

    # Common goal keywords to domain mappings
    GOAL_DOMAIN_HINTS = {
        "social media": [AIDomain.TEXT_GENERATION, AIDomain.IMAGE_GENERATION],
        "post": [AIDomain.TEXT_GENERATION, AIDomain.IMAGE_GENERATION],
        "video": [AIDomain.VIDEO_GENERATION, AIDomain.AUDIO_GENERATION],
        "content": [AIDomain.TEXT_GENERATION, AIDomain.IMAGE_GENERATION],
        "analysis": [AIDomain.ANALYSIS, AIDomain.TEXT_GENERATION],
        "code": [AIDomain.CODE_GENERATION, AIDomain.PLANNING],
        "report": [AIDomain.TEXT_GENERATION, AIDomain.ANALYSIS],
        "marketing": [AIDomain.TEXT_GENERATION, AIDomain.IMAGE_GENERATION, AIDomain.ANALYSIS],
        "presentation": [AIDomain.TEXT_GENERATION, AIDomain.IMAGE_GENERATION],
        "tutorial": [AIDomain.TEXT_GENERATION, AIDomain.VIDEO_GENERATION],
        "automation": [AIDomain.CODE_GENERATION, AIDomain.SYSTEM_OPERATIONS],
    }

    def __init__(self, db_session: AsyncSession):
        """
        Initialize the cross-thinking service.

        Args:
            db_session: Database session for persistence
        """
        self.db = db_session

    async def analyze_cross_domain_patterns(
        self,
        domains: List[AIDomain],
        time_window_hours: int = 168,  # 1 week
    ) -> CrossDomainAnalysis:
        """
        Find patterns that span multiple domains.

        Example: Image generation success correlates with
        specific text generation prompts.

        Args:
            domains: Domains to analyze
            time_window_hours: Time window for analysis

        Returns:
            CrossDomainAnalysis with patterns and insights
        """
        try:
            cutoff = datetime.now(timezone.utc) - timedelta(hours=time_window_hours)

            # Get contexts from all specified domains
            domain_contexts: Dict[str, List[ACDContextModel]] = {}

            for domain in domains:
                stmt = select(ACDContextModel).where(
                    and_(
                        ACDContextModel.created_at >= cutoff,
                        ACDContextModel.ai_domain == domain.value,
                    )
                )
                result = await self.db.execute(stmt)
                domain_contexts[domain.value] = result.scalars().all()

            # Analyze patterns between domains
            cross_patterns = []
            correlations = []
            insights = []

            # Check for temporal correlations (tasks that happen close together)
            for i, domain_a in enumerate(domains):
                for domain_b in domains[i + 1:]:
                    contexts_a = domain_contexts.get(domain_a.value, [])
                    contexts_b = domain_contexts.get(domain_b.value, [])

                    pattern = self._find_temporal_patterns(
                        contexts_a, contexts_b, domain_a, domain_b
                    )
                    if pattern:
                        cross_patterns.append(pattern)

                    correlation = self._calculate_success_correlation(
                        contexts_a, contexts_b, domain_a, domain_b
                    )
                    if correlation:
                        correlations.append(correlation)

            # Generate insights from patterns
            insights = self._generate_insights(cross_patterns, correlations)

            logger.info(
                f"Cross-domain analysis: {len(cross_patterns)} patterns, "
                f"{len(correlations)} correlations, {len(insights)} insights"
            )

            return CrossDomainAnalysis(
                domains_analyzed=[d.value for d in domains],
                cross_patterns=cross_patterns,
                correlations=correlations,
                insights=insights,
            )

        except Exception as e:
            logger.error(f"Failed to analyze cross-domain patterns: {e}")
            return CrossDomainAnalysis()

    def _find_temporal_patterns(
        self,
        contexts_a: List[ACDContextModel],
        contexts_b: List[ACDContextModel],
        domain_a: AIDomain,
        domain_b: AIDomain,
    ) -> Optional[Dict[str, Any]]:
        """Find patterns in timing between domain tasks."""
        if not contexts_a or not contexts_b:
            return None

        # Check for sequential patterns (A followed by B within 1 hour)
        sequential_count = 0
        successful_sequential = 0

        for ctx_a in contexts_a:
            for ctx_b in contexts_b:
                if ctx_a.created_at and ctx_b.created_at:
                    time_diff = (ctx_b.created_at - ctx_a.created_at).total_seconds()
                    if 0 < time_diff < 3600:  # B within 1 hour after A
                        sequential_count += 1
                        # Check if both were successful
                        a_success = (
                            ctx_a.outcome_score is not None
                            and ctx_a.outcome_score >= 0.7
                        )
                        b_success = (
                            ctx_b.outcome_score is not None
                            and ctx_b.outcome_score >= 0.7
                        )
                        if a_success and b_success:
                            successful_sequential += 1

        if sequential_count >= 3:  # Minimum threshold
            return {
                "type": "sequential_pattern",
                "domain_a": domain_a.value,
                "domain_b": domain_b.value,
                "occurrences": sequential_count,
                "success_rate": (
                    successful_sequential / sequential_count
                    if sequential_count > 0 else 0
                ),
                "description": (
                    f"{domain_a.value} tasks are often followed by "
                    f"{domain_b.value} tasks ({sequential_count} occurrences)"
                ),
            }

        return None

    def _calculate_success_correlation(
        self,
        contexts_a: List[ACDContextModel],
        contexts_b: List[ACDContextModel],
        domain_a: AIDomain,
        domain_b: AIDomain,
    ) -> Optional[Dict[str, Any]]:
        """Calculate correlation between success rates of two domains."""
        if len(contexts_a) < 5 or len(contexts_b) < 5:
            return None

        # Get success rates
        a_successful = sum(
            1 for c in contexts_a
            if c.outcome_score is not None and c.outcome_score >= 0.7
        )
        b_successful = sum(
            1 for c in contexts_b
            if c.outcome_score is not None and c.outcome_score >= 0.7
        )

        a_rate = a_successful / len(contexts_a)
        b_rate = b_successful / len(contexts_b)

        # Simple correlation check - if rates are similar, they may be correlated
        rate_diff = abs(a_rate - b_rate)

        if rate_diff < self.CORRELATION_THRESHOLD:  # Similar performance
            return {
                "domain_a": domain_a.value,
                "domain_b": domain_b.value,
                "domain_a_success_rate": a_rate,
                "domain_b_success_rate": b_rate,
                "correlation_type": (
                    "positive" if rate_diff < self.STRONG_CORRELATION_THRESHOLD else "weak"
                ),
                "description": (
                    f"{domain_a.value} and {domain_b.value} have similar "
                    f"success rates ({a_rate:.1%} vs {b_rate:.1%})"
                ),
            }

        return None

    def _generate_insights(
        self,
        patterns: List[Dict[str, Any]],
        correlations: List[Dict[str, Any]],
    ) -> List[str]:
        """Generate actionable insights from patterns and correlations."""
        insights = []

        # Insights from sequential patterns
        for pattern in patterns:
            if pattern.get("type") == "sequential_pattern":
                if pattern.get("success_rate", 0) > 0.7:
                    insights.append(
                        f"High success rate ({pattern['success_rate']:.0%}) when "
                        f"{pattern['domain_a']} is followed by {pattern['domain_b']} - "
                        f"consider automating this workflow"
                    )
                elif pattern.get("success_rate", 0) < 0.4:
                    insights.append(
                        f"Low success rate ({pattern['success_rate']:.0%}) for "
                        f"{pattern['domain_a']} → {pattern['domain_b']} sequence - "
                        f"review task dependencies"
                    )

        # Insights from correlations
        for corr in correlations:
            if corr.get("correlation_type") == "positive":
                insights.append(
                    f"{corr['domain_a']} and {corr['domain_b']} performance is linked - "
                    f"improvements in one may benefit the other"
                )

        return insights

    async def suggest_domain_combinations(
        self,
        goal: str,
        max_suggestions: int = 3,
    ) -> List[DomainCombination]:
        """
        Suggest domain combinations for complex tasks.

        Example: For "viral social media post":
        - TEXT_GENERATION for caption
        - IMAGE_GENERATION for visual
        - ANALYSIS for hashtag optimization

        Args:
            goal: High-level goal description
            max_suggestions: Maximum suggestions to return

        Returns:
            List of domain combinations
        """
        try:
            suggestions = []
            goal_lower = goal.lower()

            # Find matching domain hints
            matched_domains = []
            for keyword, domains in self.GOAL_DOMAIN_HINTS.items():
                if keyword in goal_lower:
                    matched_domains.extend(domains)

            # Remove duplicates while preserving order
            seen = set()
            unique_domains = []
            for d in matched_domains:
                if d not in seen:
                    seen.add(d)
                    unique_domains.append(d)

            if unique_domains:
                # Primary suggestion based on keywords
                primary = DomainCombination(
                    domains=unique_domains[:4],  # Max 4 domains
                    sequence="sequential",
                    confidence=0.8,
                    reasoning=f"Based on keywords in goal: {goal}",
                    example_workflow={
                        "steps": [
                            {"domain": d.value, "order": i + 1}
                            for i, d in enumerate(unique_domains[:4])
                        ],
                    },
                )
                suggestions.append(primary)

            # Check historical patterns for similar goals
            historical_combo = await self._find_historical_combinations(goal)
            if historical_combo:
                suggestions.append(historical_combo)

            # Add compatible domain suggestions
            if unique_domains:
                for primary_domain in unique_domains[:2]:
                    compatible = DOMAIN_COMPATIBILITY.get(primary_domain, [])
                    if compatible:
                        combo = DomainCombination(
                            domains=[primary_domain] + compatible[:2],
                            sequence="parallel",
                            confidence=0.6,
                            reasoning=f"Compatible domains for {primary_domain.value}",
                        )
                        suggestions.append(combo)

            # Deduplicate and limit
            unique_suggestions = []
            seen_combos = set()
            for s in suggestions:
                combo_key = tuple(sorted(d.value for d in s.domains))
                if combo_key not in seen_combos:
                    seen_combos.add(combo_key)
                    unique_suggestions.append(s)

            return unique_suggestions[:max_suggestions]

        except Exception as e:
            logger.error(f"Failed to suggest domain combinations: {e}")
            return []

    async def _find_historical_combinations(
        self,
        goal: str,
    ) -> Optional[DomainCombination]:
        """Find successful domain combinations from history."""
        try:
            cutoff = datetime.now(timezone.utc) - timedelta(hours=720)  # 30 days

            # Find successful contexts with high outcome scores
            stmt = select(ACDContextModel).where(
                and_(
                    ACDContextModel.created_at >= cutoff,
                    ACDContextModel.outcome_score >= 0.8,
                    ACDContextModel.ai_domain.isnot(None),
                )
            )
            result = await self.db.execute(stmt)
            successful = result.scalars().all()

            if len(successful) < 5:
                return None

            # Find common domain combinations
            domain_counts: Dict[str, int] = {}
            for ctx in successful:
                if ctx.ai_domain:
                    domain_counts[ctx.ai_domain] = domain_counts.get(ctx.ai_domain, 0) + 1

            # Get top domains
            top_domains = sorted(
                domain_counts.items(),
                key=lambda x: x[1],
                reverse=True
            )[:3]

            if top_domains:
                domains = []
                for d, _ in top_domains:
                    try:
                        domains.append(AIDomain(d))
                    except ValueError:
                        pass

                if domains:
                    return DomainCombination(
                        domains=domains,
                        sequence="sequential",
                        confidence=0.7,
                        reasoning="Based on historical success patterns",
                    )

        except Exception as e:
            logger.error(f"Failed to find historical combinations: {e}")

        return None

    async def orchestrate_multi_domain_task(
        self,
        task: ComplexTask,
    ) -> OrchestrationPlan:
        """
        Create execution plan spanning multiple domains.

        Handles:
        - Dependency ordering
        - Parallel execution opportunities
        - Fallback strategies

        Args:
            task: Complex task definition

        Returns:
            OrchestrationPlan with execution details
        """
        try:
            task_id = str(uuid_module.uuid4())

            # Get domain suggestions for the goal
            combinations = await self.suggest_domain_combinations(task.goal)

            if not combinations:
                # Default plan with text generation
                combinations = [
                    DomainCombination(
                        domains=[AIDomain.TEXT_GENERATION],
                        confidence=0.5,
                    )
                ]

            primary_combo = combinations[0]

            # Build execution steps
            steps = []
            dependencies: Dict[str, List[str]] = {}
            parallel_groups: List[List[str]] = []

            for i, domain in enumerate(primary_combo.domains):
                step_id = f"step_{i + 1}"
                step = {
                    "id": step_id,
                    "order": i + 1,
                    "domain": domain.value,
                    "action": self._get_action_for_domain(domain, task.goal),
                    "estimated_duration_seconds": self._estimate_duration(domain),
                    "retry_count": 2,
                    "priority": task.priority,
                }
                steps.append(step)

                # Set dependencies (sequential by default)
                if i > 0:
                    dependencies[step_id] = [f"step_{i}"]

            # Identify parallelization opportunities
            parallel_groups = self._identify_parallel_groups(primary_combo.domains)

            # Calculate total duration
            total_duration = sum(s.get("estimated_duration_seconds", 60) for s in steps)

            # Build fallback strategies
            fallbacks = [
                {
                    "trigger": "step_failure",
                    "action": "retry_with_simpler_params",
                    "max_retries": 2,
                },
                {
                    "trigger": "timeout",
                    "action": "skip_optional_steps",
                    "timeout_seconds": total_duration * 2,
                },
                {
                    "trigger": "resource_unavailable",
                    "action": "use_fallback_model",
                },
            ]

            plan = OrchestrationPlan(
                task_id=task_id,
                steps=steps,
                parallel_groups=parallel_groups,
                dependencies=dependencies,
                estimated_duration_seconds=total_duration,
                fallback_strategies=fallbacks,
            )

            logger.info(
                f"Created orchestration plan {task_id} with {len(steps)} steps"
            )

            return plan

        except Exception as e:
            logger.error(f"Failed to orchestrate multi-domain task: {e}")
            return OrchestrationPlan(task_id=str(uuid_module.uuid4()))

    def _get_action_for_domain(self, domain: AIDomain, goal: str) -> str:
        """Get action description for a domain."""
        actions = {
            AIDomain.TEXT_GENERATION: f"Generate text content for: {goal[:50]}",
            AIDomain.IMAGE_GENERATION: f"Generate image for: {goal[:50]}",
            AIDomain.VIDEO_GENERATION: f"Generate video for: {goal[:50]}",
            AIDomain.AUDIO_GENERATION: f"Generate audio for: {goal[:50]}",
            AIDomain.CODE_GENERATION: f"Generate code for: {goal[:50]}",
            AIDomain.ANALYSIS: f"Analyze for: {goal[:50]}",
            AIDomain.PLANNING: f"Create plan for: {goal[:50]}",
            AIDomain.SYSTEM_OPERATIONS: f"Execute system task for: {goal[:50]}",
        }
        return actions.get(domain, f"Execute {domain.value} task")

    def _estimate_duration(self, domain: AIDomain) -> int:
        """Estimate duration in seconds for a domain task."""
        durations = {
            AIDomain.TEXT_GENERATION: 30,
            AIDomain.IMAGE_GENERATION: 120,
            AIDomain.VIDEO_GENERATION: 300,
            AIDomain.AUDIO_GENERATION: 60,
            AIDomain.CODE_GENERATION: 45,
            AIDomain.ANALYSIS: 30,
            AIDomain.PLANNING: 20,
            AIDomain.SYSTEM_OPERATIONS: 15,
        }
        return durations.get(domain, 60)

    def _identify_parallel_groups(
        self,
        domains: List[AIDomain],
    ) -> List[List[str]]:
        """Identify domains that can run in parallel."""
        parallel_groups = []

        # Check for compatible domains that don't depend on each other
        # Audio and image can often run in parallel
        if AIDomain.IMAGE_GENERATION in domains and AIDomain.AUDIO_GENERATION in domains:
            img_idx = domains.index(AIDomain.IMAGE_GENERATION)
            audio_idx = domains.index(AIDomain.AUDIO_GENERATION)
            parallel_groups.append([f"step_{img_idx + 1}", f"step_{audio_idx + 1}"])

        return parallel_groups

    async def get_cross_domain_recommendations(
        self,
        current_domain: AIDomain,
    ) -> Dict[str, Any]:
        """
        Get recommendations for complementary domain tasks.

        Args:
            current_domain: The current domain being worked on

        Returns:
            Recommendations for related domain tasks
        """
        try:
            compatible = DOMAIN_COMPATIBILITY.get(current_domain, [])

            recommendations = {
                "current_domain": current_domain.value,
                "compatible_domains": [d.value for d in compatible],
                "suggested_workflows": [],
            }

            # Add workflow suggestions
            for compat_domain in compatible[:3]:
                workflow = {
                    "domains": [current_domain.value, compat_domain.value],
                    "description": (
                        f"Combine {current_domain.value} with {compat_domain.value} "
                        f"for enhanced results"
                    ),
                    "confidence": 0.7,
                }
                recommendations["suggested_workflows"].append(workflow)

            return recommendations

        except Exception as e:
            logger.error(f"Failed to get cross-domain recommendations: {e}")
            return {"current_domain": current_domain.value, "error": str(e)}
