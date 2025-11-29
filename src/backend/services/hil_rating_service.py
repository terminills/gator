"""
HIL (Human-in-the-Loop) Rating Service

Human-in-the-Loop rating system for content generation quality.
Allows humans to rate generated content, tag misgenerated content,
and train the system to learn which workflows, LoRAs, models,
and parameter combinations work best together.
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
    GenerationRating,
    HILRatingCreate,
    HILRatingResponse,
    HILRatingStats,
    LoRAIncompatibilityFlag,
    MisgenerationPattern,
    MisgenerationTag,
    RecommendedConfiguration,
    WorkflowEffectiveness,
)

logger = get_logger(__name__)

# Mapping from GenerationRating to numeric value
RATING_VALUES = {
    GenerationRating.EXCELLENT: 5,
    GenerationRating.GOOD: 4,
    GenerationRating.ACCEPTABLE: 3,
    GenerationRating.POOR: 2,
    GenerationRating.FAILED: 1,
}

# Reverse mapping for O(1) lookups
VALUE_TO_RATING = {v: k for k, v in RATING_VALUES.items()}


class HILRatingService:
    """
    Human-in-the-Loop rating system for content generation quality.

    Enables humans to rate generated content, tag misgeneration issues,
    and train the system to learn optimal configurations.
    """

    def __init__(self, db_session: AsyncSession):
        """
        Initialize HIL Rating service.

        Args:
            db_session: Database session for persistence
        """
        self.db = db_session

    async def rate_generation(
        self,
        context_id: UUID,
        rating_data: HILRatingCreate,
    ) -> HILRatingResponse:
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

        Raises:
            ValueError: If context not found
        """
        # Find the context
        stmt = select(ACDContextModel).where(ACDContextModel.id == context_id)
        result = await self.db.execute(stmt)
        context = result.scalar_one_or_none()

        if not context:
            raise ValueError(f"Context with ID {context_id} not found")

        # Convert rating to numeric value
        numeric_rating = RATING_VALUES.get(rating_data.rating, 3)

        # Convert tags to list of strings if provided
        tags_list = None
        if rating_data.tags:
            tags_list = [tag.value for tag in rating_data.tags]

        # Update context with rating
        context.hil_rating = numeric_rating
        context.hil_rating_tags = tags_list
        context.hil_rating_notes = rating_data.notes
        context.hil_rated_by = rating_data.rater_id
        context.hil_rated_at = datetime.now(timezone.utc)

        # Update learning weight based on rating
        # Higher rated content gets higher learning weight
        context.learning_weight = numeric_rating / 5.0

        # Update outcome score
        context.outcome_score = numeric_rating / 5.0

        await self.db.commit()
        await self.db.refresh(context)

        logger.info(
            f"Rated context {context_id}: {rating_data.rating.value} "
            f"({numeric_rating}/5)"
        )

        return HILRatingResponse(
            context_id=context.id,
            rating=context.hil_rating,
            rating_label=rating_data.rating.value,
            tags=context.hil_rating_tags,
            notes=context.hil_rating_notes,
            rated_by=context.hil_rated_by,
            rated_at=context.hil_rated_at,
        )

    async def get_workflow_effectiveness(
        self,
        workflow_id: Optional[str] = None,
        model_id: Optional[str] = None,
        lora_ids: Optional[List[str]] = None,
        time_window_hours: int = 720,  # 30 days default
    ) -> WorkflowEffectiveness:
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
            lora_ids: Optional list of LoRAs to filter by
            time_window_hours: Time window to analyze (default 30 days)

        Returns:
            WorkflowEffectiveness analysis
        """
        cutoff = datetime.now(timezone.utc) - timedelta(hours=time_window_hours)

        # Build query conditions
        conditions = [
            ACDContextModel.created_at >= cutoff,
            ACDContextModel.hil_rating.isnot(None),
        ]

        if workflow_id:
            conditions.append(ACDContextModel.workflow_id == workflow_id)
        if model_id:
            conditions.append(ACDContextModel.model_id == model_id)

        # Query rated contexts
        stmt = select(ACDContextModel).where(and_(*conditions))
        result = await self.db.execute(stmt)
        contexts = result.scalars().all()

        # If filtering by LoRAs, do additional filtering
        if lora_ids:
            lora_ids_set = set(lora_ids)
            contexts = [
                c
                for c in contexts
                if c.lora_ids and lora_ids_set.issubset(set(c.lora_ids))
            ]

        # Calculate statistics
        total_generations = len(contexts)
        if total_generations == 0:
            return WorkflowEffectiveness(
                workflow_id=workflow_id,
                model_id=model_id,
                lora_ids=lora_ids,
                total_generations=0,
                rated_generations=0,
            )

        # Rating distribution
        rating_dist: Dict[str, int] = {}
        for rating in GenerationRating:
            rating_dist[rating.value] = 0

        all_tags: List[str] = []
        ratings_with_time: List[tuple] = []  # (created_at, rating)

        for ctx in contexts:
            if ctx.hil_rating:
                # Use O(1) reverse lookup
                rating_enum = VALUE_TO_RATING.get(ctx.hil_rating)
                if rating_enum:
                    rating_dist[rating_enum.value] = (
                        rating_dist.get(rating_enum.value, 0) + 1
                    )
                # Store rating with timestamp for trend analysis
                ratings_with_time.append((ctx.created_at, ctx.hil_rating))

            if ctx.hil_rating_tags:
                all_tags.extend(ctx.hil_rating_tags)

        # Sort by time for accurate trend analysis
        ratings_with_time.sort(
            key=lambda x: x[0] or datetime.min.replace(tzinfo=timezone.utc)
        )
        ratings = [r[1] for r in ratings_with_time]

        # Calculate average rating
        avg_rating = sum(ratings) / len(ratings) if ratings else None

        # Count common tags
        tag_counts = Counter(all_tags)
        common_tags = [
            {"tag": tag, "count": count} for tag, count in tag_counts.most_common(10)
        ]

        # Calculate success rate (GOOD or EXCELLENT)
        success_count = sum(1 for r in ratings if r >= 4)
        success_rate = (success_count / len(ratings) * 100) if ratings else None

        # Determine trend (simplified)
        trend = None
        if len(ratings) >= 10:
            first_half = ratings[: len(ratings) // 2]
            second_half = ratings[len(ratings) // 2 :]
            first_avg = sum(first_half) / len(first_half)
            second_avg = sum(second_half) / len(second_half)
            if second_avg > first_avg + 0.2:
                trend = "improving"
            elif second_avg < first_avg - 0.2:
                trend = "degrading"
            else:
                trend = "stable"

        return WorkflowEffectiveness(
            workflow_id=workflow_id,
            model_id=model_id,
            lora_ids=lora_ids,
            total_generations=total_generations,
            rated_generations=len(ratings),
            average_rating=avg_rating,
            rating_distribution=rating_dist,
            common_tags=common_tags,
            success_rate=success_rate,
            trend=trend,
        )

    async def get_best_configurations(
        self,
        content_type: Optional[str] = None,
        style: Optional[str] = None,
        min_rating: float = 4.0,
        limit: int = 10,
    ) -> List[RecommendedConfiguration]:
        """
        Get best-rated configurations for a content type.

        Returns configurations that consistently produce
        highly-rated content, learned from HIL feedback.

        Args:
            content_type: Optional content type to filter by
            style: Optional style to filter by
            min_rating: Minimum average rating threshold
            limit: Maximum number of configurations to return

        Returns:
            List of recommended configurations
        """
        # Query all rated contexts
        stmt = select(ACDContextModel).where(
            and_(
                ACDContextModel.hil_rating.isnot(None),
                ACDContextModel.model_id.isnot(None),
            )
        )
        result = await self.db.execute(stmt)
        contexts = result.scalars().all()

        # Group by model_id and aggregate ratings
        config_stats: Dict[str, Dict[str, Any]] = {}

        for ctx in contexts:
            key = ctx.model_id or "unknown"

            if key not in config_stats:
                config_stats[key] = {
                    "model_id": ctx.model_id,
                    "workflow_ids": set(),
                    "lora_ids_sets": [],
                    "ratings": [],
                    "content_types": set(),
                    "styles": set(),
                    "generation_params": [],
                }

            stats = config_stats[key]
            if ctx.hil_rating:
                stats["ratings"].append(ctx.hil_rating)
            if ctx.workflow_id:
                stats["workflow_ids"].add(ctx.workflow_id)
            if ctx.lora_ids:
                stats["lora_ids_sets"].append(tuple(sorted(ctx.lora_ids)))
            if ctx.ai_domain:
                stats["content_types"].add(ctx.ai_domain)
            if ctx.ai_subdomain:
                stats["styles"].add(ctx.ai_subdomain)
            if ctx.generation_params:
                stats["generation_params"].append(ctx.generation_params)

        # Calculate averages and filter
        recommendations: List[RecommendedConfiguration] = []

        for key, stats in config_stats.items():
            if not stats["ratings"]:
                continue

            avg_rating = sum(stats["ratings"]) / len(stats["ratings"])

            if avg_rating < min_rating:
                continue

            # Find most common LoRA combination
            lora_counter = Counter(stats["lora_ids_sets"])
            most_common_loras = (
                list(lora_counter.most_common(1)[0][0]) if lora_counter else None
            )

            # Find most common workflow
            workflow_id = (
                list(stats["workflow_ids"])[0] if stats["workflow_ids"] else None
            )

            # Calculate confidence based on sample size
            total_ratings = len(stats["ratings"])
            confidence = min(1.0, total_ratings / 50)  # Max confidence at 50 ratings

            recommendations.append(
                RecommendedConfiguration(
                    workflow_id=workflow_id,
                    model_id=stats["model_id"],
                    lora_ids=most_common_loras,
                    average_rating=avg_rating,
                    total_ratings=total_ratings,
                    content_types=list(stats["content_types"]),
                    styles=list(stats["styles"]),
                    confidence=confidence,
                )
            )

        # Sort by average rating (descending), then by total ratings
        recommendations.sort(
            key=lambda x: (x.average_rating, x.total_ratings), reverse=True
        )

        return recommendations[:limit]

    async def flag_lora_incompatibility(
        self,
        flag_data: LoRAIncompatibilityFlag,
    ) -> Dict[str, Any]:
        """
        Flag incompatible LoRA combinations discovered through HIL.

        System learns to avoid these combinations in future generations.

        Args:
            flag_data: LoRA incompatibility flag data

        Returns:
            Confirmation of the flag being recorded
        """
        # Find the context
        stmt = select(ACDContextModel).where(ACDContextModel.id == flag_data.context_id)
        result = await self.db.execute(stmt)
        context = result.scalar_one_or_none()

        if not context:
            raise ValueError(f"Context with ID {flag_data.context_id} not found")

        # Store incompatibility in the context's cross_domain_insights
        insights = context.cross_domain_insights or {}
        incompatibilities = insights.get("lora_incompatibilities", [])
        incompatibilities.append(
            {
                "lora_a": flag_data.lora_a,
                "lora_b": flag_data.lora_b,
                "severity": flag_data.severity,
                "notes": flag_data.notes,
                "flagged_at": datetime.now(timezone.utc).isoformat(),
            }
        )
        insights["lora_incompatibilities"] = incompatibilities
        context.cross_domain_insights = insights

        # Also add LORA_CONFLICT tag if not present
        tags = context.hil_rating_tags or []
        if MisgenerationTag.LORA_CONFLICT.value not in tags:
            tags.append(MisgenerationTag.LORA_CONFLICT.value)
            context.hil_rating_tags = tags

        await self.db.commit()

        logger.info(
            f"Flagged LoRA incompatibility: {flag_data.lora_a} + {flag_data.lora_b}"
        )

        return {
            "status": "flagged",
            "lora_a": flag_data.lora_a,
            "lora_b": flag_data.lora_b,
            "context_id": str(flag_data.context_id),
            "severity": flag_data.severity,
        }

    async def get_misgeneration_patterns(
        self,
        tag: Optional[MisgenerationTag] = None,
        time_window_hours: int = 168,  # 1 week default
    ) -> List[MisgenerationPattern]:
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
        cutoff = datetime.now(timezone.utc) - timedelta(hours=time_window_hours)

        # Query contexts with rating tags
        stmt = select(ACDContextModel).where(
            and_(
                ACDContextModel.created_at >= cutoff,
                ACDContextModel.hil_rating_tags.isnot(None),
            )
        )
        result = await self.db.execute(stmt)
        contexts = result.scalars().all()

        # Collect all tags and associated metadata
        tag_data: Dict[str, Dict[str, Any]] = {}

        for ctx in contexts:
            if not ctx.hil_rating_tags:
                continue

            for t in ctx.hil_rating_tags:
                if tag and t != tag.value:
                    continue

                if t not in tag_data:
                    tag_data[t] = {
                        "count": 0,
                        "models": [],
                        "loras": [],
                        "contexts": [],
                    }

                tag_data[t]["count"] += 1
                if ctx.model_id:
                    tag_data[t]["models"].append(ctx.model_id)
                if ctx.lora_ids:
                    tag_data[t]["loras"].extend(ctx.lora_ids)
                tag_data[t]["contexts"].append(ctx)

        # Calculate total misgenerations
        total_misgenerations = sum(d["count"] for d in tag_data.values())

        # Build pattern analysis
        patterns: List[MisgenerationPattern] = []

        for tag_name, data in tag_data.items():
            percentage = (
                (data["count"] / total_misgenerations * 100)
                if total_misgenerations > 0
                else 0
            )

            # Find most associated models
            model_counts = Counter(data["models"])
            associated_models = [model for model, _ in model_counts.most_common(5)]

            # Find most associated LoRAs
            lora_counts = Counter(data["loras"])
            associated_loras = [lora for lora, _ in lora_counts.most_common(5)]

            # Determine trend
            trend = "stable"
            contexts_list = data["contexts"]
            if len(contexts_list) >= 5:
                # Use timezone-aware datetime for comparison
                contexts_list.sort(
                    key=lambda x: x.created_at
                    or datetime.min.replace(tzinfo=timezone.utc)
                )
                mid = len(contexts_list) // 2
                first_half_count = mid
                second_half_count = len(contexts_list) - mid
                if second_half_count > first_half_count * 1.3:
                    trend = "increasing"
                elif second_half_count < first_half_count * 0.7:
                    trend = "decreasing"

            # Generate suggestions based on tag
            suggestions = self._get_suggestions_for_tag(tag_name)

            patterns.append(
                MisgenerationPattern(
                    tag=tag_name,
                    count=data["count"],
                    percentage=percentage,
                    associated_models=associated_models,
                    associated_loras=associated_loras,
                    trend=trend,
                    suggestions=suggestions,
                )
            )

        # Sort by count descending
        patterns.sort(key=lambda x: x.count, reverse=True)

        return patterns

    def _get_suggestions_for_tag(self, tag: str) -> List[str]:
        """Get suggestions for avoiding a specific misgeneration tag."""
        suggestions_map = {
            MisgenerationTag.ANATOMY_ERROR.value: [
                "Use a model trained on anatomical accuracy",
                "Reduce CFG scale to allow more flexibility",
                "Try adding negative prompts for body parts",
            ],
            MisgenerationTag.STYLE_MISMATCH.value: [
                "Check LoRA compatibility with base model",
                "Adjust LoRA weight to reduce style conflicts",
                "Use a style-specific checkpoint",
            ],
            MisgenerationTag.PROMPT_IGNORED.value: [
                "Increase CFG scale for stronger prompt adherence",
                "Simplify the prompt to key elements",
                "Try using prompt weighting syntax",
            ],
            MisgenerationTag.ARTIFACT.value: [
                "Reduce noise strength",
                "Use a higher quality VAE",
                "Try different samplers (Euler a, DPM++)",
            ],
            MisgenerationTag.LORA_CONFLICT.value: [
                "Test LoRAs individually before combining",
                "Reduce combined LoRA weights",
                "Use LoRAs trained on same base model",
            ],
            MisgenerationTag.QUALITY_LOW.value: [
                "Increase step count",
                "Use a higher resolution",
                "Enable hires fix for upscaling",
            ],
        }
        return suggestions_map.get(tag, ["Review generation parameters"])

    async def get_rating_stats(
        self,
        time_window_hours: int = 168,  # 1 week default
    ) -> HILRatingStats:
        """
        Get overall HIL rating statistics.

        Args:
            time_window_hours: Time window to analyze

        Returns:
            HILRatingStats with overall statistics
        """
        cutoff = datetime.now(timezone.utc) - timedelta(hours=time_window_hours)

        # Count total rated and unrated
        rated_stmt = select(func.count()).where(
            and_(
                ACDContextModel.created_at >= cutoff,
                ACDContextModel.hil_rating.isnot(None),
            )
        )
        rated_result = await self.db.execute(rated_stmt)
        total_rated = rated_result.scalar() or 0

        unrated_stmt = select(func.count()).where(
            and_(
                ACDContextModel.created_at >= cutoff,
                ACDContextModel.hil_rating.is_(None),
            )
        )
        unrated_result = await self.db.execute(unrated_stmt)
        total_unrated = unrated_result.scalar() or 0

        # Get all rated contexts for detailed stats
        contexts_stmt = select(ACDContextModel).where(
            and_(
                ACDContextModel.created_at >= cutoff,
                ACDContextModel.hil_rating.isnot(None),
            )
        )
        contexts_result = await self.db.execute(contexts_stmt)
        contexts = contexts_result.scalars().all()

        # Calculate rating distribution
        rating_dist: Dict[str, int] = {}
        for rating in GenerationRating:
            rating_dist[rating.value] = 0

        all_ratings: List[int] = []
        all_tags: List[str] = []
        by_model: Dict[str, Dict[str, Any]] = {}
        by_domain: Dict[str, Dict[str, Any]] = {}

        for ctx in contexts:
            if ctx.hil_rating:
                all_ratings.append(ctx.hil_rating)
                # Use O(1) reverse lookup
                rating_enum = VALUE_TO_RATING.get(ctx.hil_rating)
                if rating_enum:
                    rating_dist[rating_enum.value] = (
                        rating_dist.get(rating_enum.value, 0) + 1
                    )

            if ctx.hil_rating_tags:
                all_tags.extend(ctx.hil_rating_tags)

            # Group by model
            if ctx.model_id:
                if ctx.model_id not in by_model:
                    by_model[ctx.model_id] = {"ratings": [], "count": 0}
                by_model[ctx.model_id]["ratings"].append(ctx.hil_rating)
                by_model[ctx.model_id]["count"] += 1

            # Group by domain
            if ctx.ai_domain:
                if ctx.ai_domain not in by_domain:
                    by_domain[ctx.ai_domain] = {"ratings": [], "count": 0}
                by_domain[ctx.ai_domain]["ratings"].append(ctx.hil_rating)
                by_domain[ctx.ai_domain]["count"] += 1

        # Calculate average
        avg_rating = sum(all_ratings) / len(all_ratings) if all_ratings else None

        # Count common tags
        tag_counts = Counter(all_tags)
        most_common_tags = [
            {"tag": tag, "count": count} for tag, count in tag_counts.most_common(10)
        ]

        # Calculate averages for by_model and by_domain
        ratings_by_model = {}
        for model_id, data in by_model.items():
            ratings = [r for r in data["ratings"] if r is not None]
            ratings_by_model[model_id] = {
                "count": data["count"],
                "average": sum(ratings) / len(ratings) if ratings else None,
            }

        ratings_by_domain = {}
        for domain, data in by_domain.items():
            ratings = [r for r in data["ratings"] if r is not None]
            ratings_by_domain[domain] = {
                "count": data["count"],
                "average": sum(ratings) / len(ratings) if ratings else None,
            }

        return HILRatingStats(
            total_rated=total_rated,
            total_unrated=total_unrated,
            rating_distribution=rating_dist,
            average_rating=avg_rating,
            most_common_tags=most_common_tags,
            ratings_by_model=ratings_by_model,
            ratings_by_domain=ratings_by_domain,
        )

    async def get_recent_ratings(
        self,
        limit: int = 50,
        rating_filter: Optional[GenerationRating] = None,
    ) -> List[HILRatingResponse]:
        """
        Get recent ratings.

        Args:
            limit: Maximum number of ratings to return
            rating_filter: Optional filter by specific rating

        Returns:
            List of recent ratings
        """
        conditions = [ACDContextModel.hil_rating.isnot(None)]

        if rating_filter:
            conditions.append(
                ACDContextModel.hil_rating == RATING_VALUES[rating_filter]
            )

        stmt = (
            select(ACDContextModel)
            .where(and_(*conditions))
            .order_by(ACDContextModel.hil_rated_at.desc())
            .limit(limit)
        )

        result = await self.db.execute(stmt)
        contexts = result.scalars().all()

        ratings: List[HILRatingResponse] = []
        for ctx in contexts:
            # Use O(1) reverse lookup
            rating_enum = VALUE_TO_RATING.get(ctx.hil_rating)
            rating_label = rating_enum.value if rating_enum else "UNKNOWN"

            ratings.append(
                HILRatingResponse(
                    context_id=ctx.id,
                    rating=ctx.hil_rating,
                    rating_label=rating_label,
                    tags=ctx.hil_rating_tags,
                    notes=ctx.hil_rating_notes,
                    rated_by=ctx.hil_rated_by,
                    rated_at=ctx.hil_rated_at,
                )
            )

        return ratings
