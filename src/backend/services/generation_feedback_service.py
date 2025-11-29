"""
Generation Feedback Service

Handles recording of AI generation benchmarks and human feedback for continuous improvement.
"""

import asyncio
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional
from uuid import UUID

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.config.logging import get_logger
from backend.models.acd import ACDContextUpdate, AIConfidence, AIValidation
from backend.models.generation_feedback import (
    BenchmarkStats,
    FeedbackRating,
    FeedbackSubmission,
    GenerationBenchmarkCreate,
    GenerationBenchmarkModel,
    GenerationBenchmarkResponse,
)
from backend.services.acd_service import ACDService

logger = get_logger(__name__)


class GenerationFeedbackService:
    """
    Service for tracking AI generation performance and human feedback.

    Enables learning and optimization of model selection and prompt enhancement
    based on real-world usage and human evaluations.
    """

    def __init__(self, db_session: AsyncSession):
        """
        Initialize feedback service.

        Args:
            db_session: Database session for persistence
        """
        self.db = db_session

    async def record_benchmark(
        self, benchmark_data: GenerationBenchmarkCreate
    ) -> GenerationBenchmarkResponse:
        """
        Record a new generation benchmark.

        Args:
            benchmark_data: Benchmark data to record

        Returns:
            Created benchmark record
        """
        try:
            benchmark = GenerationBenchmarkModel(**benchmark_data.model_dump())

            self.db.add(benchmark)
            await self.db.commit()
            await self.db.refresh(benchmark)

            logger.info(
                f"Recorded benchmark {benchmark.id}: "
                f"model={benchmark.model_selected}, "
                f"time={benchmark.generation_time_seconds:.2f}s, "
                f"quality={benchmark.quality_requested}"
            )

            return GenerationBenchmarkResponse.model_validate(benchmark)

        except Exception as e:
            logger.error(f"Failed to record benchmark: {str(e)}")
            await self.db.rollback()
            raise

    async def submit_feedback(
        self, feedback: FeedbackSubmission
    ) -> GenerationBenchmarkResponse:
        """
        Submit human feedback for a generation benchmark.
        Updates both the benchmark record and linked ACD context for learning.

        Args:
            feedback: Human feedback data

        Returns:
            Updated benchmark record
        """
        try:
            stmt = select(GenerationBenchmarkModel).where(
                GenerationBenchmarkModel.id == feedback.benchmark_id
            )
            result = await self.db.execute(stmt)
            benchmark = result.scalar_one_or_none()

            if not benchmark:
                raise ValueError(f"Benchmark {feedback.benchmark_id} not found")

            # Update with feedback
            benchmark.human_rating = feedback.rating.value
            benchmark.human_feedback_text = feedback.feedback_text
            benchmark.feedback_timestamp = datetime.now(timezone.utc)

            # Store issues if provided
            if feedback.issues:
                if not benchmark.content_features:
                    benchmark.content_features = {}
                benchmark.content_features["reported_issues"] = feedback.issues

            await self.db.commit()
            await self.db.refresh(benchmark)

            # Update linked ACD context with feedback (close the learning loop)
            if benchmark.acd_context_id:
                await self._update_acd_with_feedback(benchmark, feedback)

            logger.info(
                f"Feedback recorded for benchmark {benchmark.id}: "
                f"rating={feedback.rating.value}, "
                f"model={benchmark.model_selected}"
            )

            return GenerationBenchmarkResponse.model_validate(benchmark)

        except Exception as e:
            logger.error(f"Failed to submit feedback: {str(e)}")
            await self.db.rollback()
            raise

    async def _update_acd_with_feedback(
        self, benchmark: GenerationBenchmarkModel, feedback: FeedbackSubmission
    ):
        """
        Update ACD context with human feedback to close the learning loop.

        Args:
            benchmark: The benchmark record with feedback
            feedback: The submitted feedback data
        """
        try:
            acd_service = ACDService(self.db)

            # Map feedback rating to validation status
            validation_map = {
                FeedbackRating.EXCELLENT: AIValidation.APPROVED,
                FeedbackRating.GOOD: AIValidation.APPROVED,
                FeedbackRating.ACCEPTABLE: AIValidation.CONDITIONALLY_APPROVED,
                FeedbackRating.POOR: AIValidation.REJECTED,
                FeedbackRating.UNACCEPTABLE: AIValidation.REJECTED,
            }

            # Map rating to confidence
            confidence_map = {
                FeedbackRating.EXCELLENT: AIConfidence.VALIDATED,
                FeedbackRating.GOOD: AIConfidence.VALIDATED,
                FeedbackRating.ACCEPTABLE: AIConfidence.CONFIDENT,
                FeedbackRating.POOR: AIConfidence.UNCERTAIN,
                FeedbackRating.UNACCEPTABLE: AIConfidence.UNCERTAIN,
            }

            validation = validation_map.get(feedback.rating, AIValidation.PENDING)
            confidence = confidence_map.get(feedback.rating, AIConfidence.CONFIDENT)

            # Prepare update data
            update_data = ACDContextUpdate(
                ai_validation=validation,
                ai_confidence=confidence,
                human_override=(
                    feedback.feedback_text if feedback.feedback_text else None
                ),
                ai_issues=feedback.issues if feedback.issues else None,
            )

            # If highly rated, extract pattern for learning
            if feedback.rating in [FeedbackRating.EXCELLENT, FeedbackRating.GOOD]:
                # Extract pattern from successful generation
                pattern = f"{benchmark.content_type}_{benchmark.model_selected}"
                strategy = f"Model: {benchmark.model_selected}, Quality: {benchmark.quality_requested}, Rating: {feedback.rating.value}"

                update_data.ai_pattern = pattern
                update_data.ai_strategy = strategy

            await acd_service.update_context(benchmark.acd_context_id, update_data)

            logger.info(
                f"Updated ACD context {benchmark.acd_context_id} with feedback: "
                f"validation={validation.value}, confidence={confidence.value}"
            )

        except Exception as e:
            # Don't fail feedback submission if ACD update fails
            logger.error(f"Failed to update ACD context with feedback: {str(e)}")

    async def get_benchmark_stats(
        self,
        hours: int = 24,
        model_name: Optional[str] = None,
        content_type: Optional[str] = None,
    ) -> BenchmarkStats:
        """
        Get benchmark statistics for analysis.

        Args:
            hours: Time window for stats
            model_name: Filter by specific model
            content_type: Filter by content type

        Returns:
            Aggregate statistics
        """
        try:
            cutoff_time = datetime.now(timezone.utc) - timedelta(hours=hours)

            # Build base query
            stmt = select(GenerationBenchmarkModel).where(
                GenerationBenchmarkModel.created_at >= cutoff_time
            )

            if model_name:
                stmt = stmt.where(GenerationBenchmarkModel.model_selected == model_name)
            if content_type:
                stmt = stmt.where(GenerationBenchmarkModel.content_type == content_type)

            result = await self.db.execute(stmt)
            benchmarks = result.scalars().all()

            if not benchmarks:
                return BenchmarkStats()

            # Calculate statistics
            total = len(benchmarks)
            by_model = {}
            by_rating = {}
            total_time = 0
            total_quality = 0
            quality_count = 0
            feedback_count = 0
            success_count = 0
            fallback_count = 0

            for benchmark in benchmarks:
                # Count by model
                model = benchmark.model_selected
                by_model[model] = by_model.get(model, 0) + 1

                # Count by rating
                if benchmark.human_rating:
                    rating = benchmark.human_rating
                    by_rating[rating] = by_rating.get(rating, 0) + 1
                    feedback_count += 1

                # Aggregate metrics
                total_time += benchmark.generation_time_seconds

                if benchmark.quality_score is not None:
                    total_quality += benchmark.quality_score
                    quality_count += 1

                if not benchmark.had_errors:
                    success_count += 1

                if benchmark.fallback_used:
                    fallback_count += 1

            return BenchmarkStats(
                total_generations=total,
                by_model=by_model,
                by_rating=by_rating,
                avg_generation_time=total_time / total if total > 0 else 0,
                avg_quality_score=(
                    total_quality / quality_count if quality_count > 0 else None
                ),
                success_rate=success_count / total if total > 0 else 0,
                fallback_rate=fallback_count / total if total > 0 else 0,
                feedback_count=feedback_count,
            )

        except Exception as e:
            logger.error(f"Failed to get benchmark stats: {str(e)}")
            return BenchmarkStats()

    async def get_model_performance_comparison(
        self, content_type: str, hours: int = 168  # 7 days default
    ) -> Dict[str, Dict[str, Any]]:
        """
        Compare performance of different models for a content type.

        Args:
            content_type: Type of content to analyze
            hours: Time window for analysis

        Returns:
            Dict mapping model names to their performance metrics
        """
        try:
            cutoff_time = datetime.now(timezone.utc) - timedelta(hours=hours)

            stmt = select(GenerationBenchmarkModel).where(
                GenerationBenchmarkModel.created_at >= cutoff_time,
                GenerationBenchmarkModel.content_type == content_type,
            )

            result = await self.db.execute(stmt)
            benchmarks = result.scalars().all()

            # Group by model
            model_stats = {}

            for benchmark in benchmarks:
                model = benchmark.model_selected
                if model not in model_stats:
                    model_stats[model] = {
                        "count": 0,
                        "total_time": 0,
                        "success_count": 0,
                        "ratings": [],
                        "quality_scores": [],
                    }

                stats = model_stats[model]
                stats["count"] += 1
                stats["total_time"] += benchmark.generation_time_seconds

                if not benchmark.had_errors:
                    stats["success_count"] += 1

                if benchmark.human_rating:
                    # Convert rating to numeric score
                    rating_scores = {
                        "excellent": 5,
                        "good": 4,
                        "acceptable": 3,
                        "poor": 2,
                        "unacceptable": 1,
                    }
                    score = rating_scores.get(benchmark.human_rating, 3)
                    stats["ratings"].append(score)

                if benchmark.quality_score is not None:
                    stats["quality_scores"].append(benchmark.quality_score)

            # Calculate averages
            comparison = {}
            for model, stats in model_stats.items():
                comparison[model] = {
                    "generations": stats["count"],
                    "avg_time": stats["total_time"] / stats["count"],
                    "success_rate": stats["success_count"] / stats["count"],
                    "avg_human_rating": (
                        sum(stats["ratings"]) / len(stats["ratings"])
                        if stats["ratings"]
                        else None
                    ),
                    "avg_quality_score": (
                        sum(stats["quality_scores"]) / len(stats["quality_scores"])
                        if stats["quality_scores"]
                        else None
                    ),
                }

            return comparison

        except Exception as e:
            logger.error(f"Failed to compare model performance: {str(e)}")
            return {}

    async def get_prompt_enhancement_insights(
        self, content_type: str, min_rating: str = "good", hours: int = 168  # 7 days
    ) -> List[Dict[str, Any]]:
        """
        Analyze successful prompt enhancements for learning.

        Args:
            content_type: Type of content to analyze
            min_rating: Minimum human rating to consider successful
            hours: Time window for analysis

        Returns:
            List of successful prompt enhancement patterns
        """
        try:
            cutoff_time = datetime.now(timezone.utc) - timedelta(hours=hours)

            # Get highly-rated generations with prompt enhancements
            rating_threshold = (
                ["excellent", "good"] if min_rating == "good" else ["excellent"]
            )

            stmt = select(GenerationBenchmarkModel).where(
                GenerationBenchmarkModel.created_at >= cutoff_time,
                GenerationBenchmarkModel.content_type == content_type,
                GenerationBenchmarkModel.enhanced_prompt.isnot(None),
                GenerationBenchmarkModel.human_rating.in_(rating_threshold),
            )

            result = await self.db.execute(stmt)
            benchmarks = result.scalars().all()

            insights = []
            for benchmark in benchmarks:
                insights.append(
                    {
                        "original_prompt": benchmark.prompt,
                        "enhanced_prompt": benchmark.enhanced_prompt,
                        "rating": benchmark.human_rating,
                        "model": benchmark.model_selected,
                        "keywords": benchmark.prompt_keywords,
                        "generation_time": benchmark.generation_time_seconds,
                    }
                )

            logger.info(
                f"Found {len(insights)} successful prompt enhancements for {content_type}"
            )

            return insights

        except Exception as e:
            logger.error(f"Failed to get prompt insights: {str(e)}")
            return []
