"""
Pattern Analysis Utilities for ACD Learning

Helper functions to analyze patterns from successful content generations,
extract insights from social media engagement, and provide recommendations.
"""

from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta, timezone
from collections import defaultdict
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, and_, desc

from backend.models.acd import ACDContextModel, AIValidation, AIConfidence
from backend.models.social_media_post import SocialMediaPostModel, PostStatus
from backend.models.generation_feedback import GenerationBenchmarkModel, FeedbackRating
from backend.config.logging import get_logger

logger = get_logger(__name__)


class PatternAnalyzer:
    """
    Analyzes patterns from ACD contexts, social engagement, and feedback
    to extract insights for improving future content generation.
    """

    def __init__(self, db_session: AsyncSession):
        """
        Initialize pattern analyzer.

        Args:
            db_session: Database session
        """
        self.db = db_session

    async def get_successful_patterns(
        self,
        persona_id: Optional[UUID] = None,
        platform: Optional[str] = None,
        min_engagement_rate: float = 5.0,
        days: int = 30,
    ) -> List[Dict[str, Any]]:
        """
        Get patterns from successful content.

        Args:
            persona_id: Filter by persona
            platform: Filter by social platform
            min_engagement_rate: Minimum engagement rate to consider successful
            days: Time window in days

        Returns:
            List of successful patterns with metadata
        """
        try:
            cutoff = datetime.now(timezone.utc) - timedelta(days=days)

            # Query high-performing social posts
            stmt = select(
                SocialMediaPostModel,
                ACDContextModel
            ).join(
                ACDContextModel,
                SocialMediaPostModel.acd_context_id == ACDContextModel.id,
                isouter=True
            ).where(
                and_(
                    SocialMediaPostModel.status == PostStatus.PUBLISHED.value,
                    SocialMediaPostModel.engagement_rate >= min_engagement_rate,
                    SocialMediaPostModel.published_at >= cutoff,
                )
            )

            if persona_id:
                stmt = stmt.where(SocialMediaPostModel.persona_id == persona_id)
            if platform:
                stmt = stmt.where(SocialMediaPostModel.platform == platform)

            stmt = stmt.order_by(desc(SocialMediaPostModel.engagement_rate))

            result = await self.db.execute(stmt)
            rows = result.all()

            patterns = []
            for post, acd_context in rows:
                pattern = {
                    "post_id": post.id,
                    "platform": post.platform,
                    "engagement_rate": post.engagement_rate,
                    "hashtags": post.hashtags,
                    "published_hour": post.published_at.hour if post.published_at else None,
                    "genuine_users": post.genuine_user_count,
                    "likes": post.likes_count,
                    "comments": post.comments_count,
                    "shares": post.shares_count,
                }

                if acd_context:
                    pattern["acd_pattern"] = acd_context.ai_pattern
                    pattern["acd_strategy"] = acd_context.ai_strategy
                    pattern["acd_context"] = acd_context.ai_context
                    pattern["phase"] = acd_context.ai_phase

                patterns.append(pattern)

            logger.info(
                f"Found {len(patterns)} successful patterns "
                f"(engagement >= {min_engagement_rate}%)"
            )

            return patterns

        except Exception as e:
            logger.error(f"Failed to get successful patterns: {str(e)}")
            return []

    async def get_common_failure_patterns(
        self,
        persona_id: Optional[UUID] = None,
        days: int = 30,
    ) -> List[Dict[str, Any]]:
        """
        Analyze common failure patterns to avoid them.

        Args:
            persona_id: Filter by persona
            days: Time window in days

        Returns:
            List of common failure patterns
        """
        try:
            cutoff = datetime.now(timezone.utc) - timedelta(days=days)

            # Query failed or low-performing content from ACD
            stmt = select(ACDContextModel).where(
                and_(
                    ACDContextModel.created_at >= cutoff,
                    ACDContextModel.ai_validation == AIValidation.REJECTED,
                )
            )

            result = await self.db.execute(stmt)
            failed_contexts = result.scalars().all()

            # Analyze failure patterns
            failure_reasons = defaultdict(int)
            failure_phases = defaultdict(int)
            failure_examples = []

            for context in failed_contexts:
                if context.runtime_err:
                    failure_reasons["runtime_error"] += 1
                if context.compiler_err:
                    failure_reasons["compiler_error"] += 1
                if context.ai_issues:
                    for issue in context.ai_issues:
                        failure_reasons[f"issue_{issue}"] += 1

                failure_phases[context.ai_phase] += 1

                failure_examples.append({
                    "phase": context.ai_phase,
                    "complexity": context.ai_complexity,
                    "error": context.runtime_err or context.compiler_err,
                    "issues": context.ai_issues,
                    "note": context.ai_note,
                })

            patterns = []
            for reason, count in failure_reasons.items():
                patterns.append({
                    "failure_type": reason,
                    "occurrence_count": count,
                    "percentage": (count / len(failed_contexts) * 100) if failed_contexts else 0,
                })

            logger.info(f"Analyzed {len(failed_contexts)} failure contexts")

            return {
                "failure_patterns": patterns,
                "failures_by_phase": dict(failure_phases),
                "total_failures": len(failed_contexts),
                "examples": failure_examples[:10],  # Top 10 examples
            }

        except Exception as e:
            logger.error(f"Failed to analyze failure patterns: {str(e)}")
            return {}

    async def get_optimal_posting_times(
        self,
        persona_id: UUID,
        platform: str,
        days: int = 90,
    ) -> Dict[int, float]:
        """
        Determine optimal posting times based on engagement data.

        Args:
            persona_id: Persona ID
            platform: Social platform
            days: Time window in days

        Returns:
            Dict mapping hour (0-23) to average engagement rate
        """
        try:
            cutoff = datetime.now(timezone.utc) - timedelta(days=days)

            stmt = select(SocialMediaPostModel).where(
                and_(
                    SocialMediaPostModel.persona_id == persona_id,
                    SocialMediaPostModel.platform == platform,
                    SocialMediaPostModel.status == PostStatus.PUBLISHED.value,
                    SocialMediaPostModel.published_at >= cutoff,
                    SocialMediaPostModel.engagement_rate.isnot(None),
                )
            )

            result = await self.db.execute(stmt)
            posts = result.scalars().all()

            # Group by hour
            hour_engagement = defaultdict(list)
            for post in posts:
                if post.published_at:
                    hour = post.published_at.hour
                    hour_engagement[hour].append(post.engagement_rate)

            # Calculate averages
            optimal_times = {}
            for hour, rates in hour_engagement.items():
                optimal_times[hour] = sum(rates) / len(rates)

            # Sort by engagement rate
            sorted_times = dict(sorted(optimal_times.items(), key=lambda x: x[1], reverse=True))

            logger.info(
                f"Analyzed posting times for {len(posts)} posts, "
                f"found {len(sorted_times)} different hours"
            )

            return sorted_times

        except Exception as e:
            logger.error(f"Failed to get optimal posting times: {str(e)}")
            return {}

    async def get_effective_hashtags(
        self,
        persona_id: UUID,
        platform: str,
        min_posts: int = 5,
        days: int = 90,
    ) -> List[Tuple[str, float, int]]:
        """
        Identify hashtags that correlate with high engagement.

        Args:
            persona_id: Persona ID
            platform: Social platform
            min_posts: Minimum number of posts to consider hashtag
            days: Time window in days

        Returns:
            List of (hashtag, avg_engagement_rate, post_count)
        """
        try:
            cutoff = datetime.now(timezone.utc) - timedelta(days=days)

            stmt = select(SocialMediaPostModel).where(
                and_(
                    SocialMediaPostModel.persona_id == persona_id,
                    SocialMediaPostModel.platform == platform,
                    SocialMediaPostModel.status == PostStatus.PUBLISHED.value,
                    SocialMediaPostModel.published_at >= cutoff,
                    SocialMediaPostModel.hashtags.isnot(None),
                )
            )

            result = await self.db.execute(stmt)
            posts = result.scalars().all()

            # Analyze hashtag performance
            hashtag_stats = defaultdict(lambda: {"rates": [], "count": 0})
            
            for post in posts:
                if post.hashtags and post.engagement_rate:
                    for hashtag in post.hashtags:
                        hashtag_stats[hashtag]["rates"].append(post.engagement_rate)
                        hashtag_stats[hashtag]["count"] += 1

            # Calculate averages and filter by min_posts
            effective_hashtags = []
            for hashtag, stats in hashtag_stats.items():
                if stats["count"] >= min_posts:
                    avg_rate = sum(stats["rates"]) / len(stats["rates"])
                    effective_hashtags.append((hashtag, avg_rate, stats["count"]))

            # Sort by engagement rate
            effective_hashtags.sort(key=lambda x: x[1], reverse=True)

            logger.info(
                f"Analyzed {len(posts)} posts, "
                f"found {len(effective_hashtags)} effective hashtags"
            )

            return effective_hashtags

        except Exception as e:
            logger.error(f"Failed to get effective hashtags: {str(e)}")
            return []

    async def get_content_performance_summary(
        self,
        persona_id: UUID,
        days: int = 30,
    ) -> Dict[str, Any]:
        """
        Get comprehensive performance summary for a persona.

        Args:
            persona_id: Persona ID
            days: Time window in days

        Returns:
            Performance summary with metrics and recommendations
        """
        try:
            cutoff = datetime.now(timezone.utc) - timedelta(days=days)

            # Get social media performance
            social_stmt = select(SocialMediaPostModel).where(
                and_(
                    SocialMediaPostModel.persona_id == persona_id,
                    SocialMediaPostModel.status == PostStatus.PUBLISHED.value,
                    SocialMediaPostModel.published_at >= cutoff,
                )
            )
            social_result = await self.db.execute(social_stmt)
            social_posts = social_result.scalars().all()

            # Get generation feedback
            feedback_stmt = select(GenerationBenchmarkModel).where(
                GenerationBenchmarkModel.created_at >= cutoff
            )
            feedback_result = await self.db.execute(feedback_stmt)
            benchmarks = feedback_result.scalars().all()

            # Calculate metrics
            total_posts = len(social_posts)
            if total_posts == 0:
                return {"message": "No posts in time window"}

            total_engagement = sum(
                (p.likes_count + p.comments_count + p.shares_count + p.saves_count)
                for p in social_posts
            )
            avg_engagement_rate = sum(
                p.engagement_rate for p in social_posts if p.engagement_rate
            ) / total_posts if total_posts > 0 else 0

            # Platform breakdown
            platform_performance = defaultdict(lambda: {"posts": 0, "avg_engagement": 0.0})
            for post in social_posts:
                platform_performance[post.platform]["posts"] += 1
                if post.engagement_rate:
                    platform_performance[post.platform]["avg_engagement"] += post.engagement_rate

            for platform in platform_performance:
                count = platform_performance[platform]["posts"]
                platform_performance[platform]["avg_engagement"] /= count

            # Top performing posts
            top_posts = sorted(
                social_posts,
                key=lambda p: p.engagement_rate if p.engagement_rate else 0,
                reverse=True
            )[:5]

            # Generate recommendations
            recommendations = []
            
            best_platform = max(
                platform_performance.items(),
                key=lambda x: x[1]["avg_engagement"]
            )[0] if platform_performance else None
            
            if best_platform:
                recommendations.append(f"Focus more on {best_platform} (highest engagement)")

            if avg_engagement_rate < 3.0:
                recommendations.append("Overall engagement below target - review content strategy")
            
            return {
                "time_window_days": days,
                "total_posts": total_posts,
                "total_engagement": total_engagement,
                "avg_engagement_rate": avg_engagement_rate,
                "platform_performance": dict(platform_performance),
                "top_posts": [
                    {
                        "id": str(p.id),
                        "platform": p.platform,
                        "engagement_rate": p.engagement_rate,
                        "hashtags": p.hashtags,
                    }
                    for p in top_posts
                ],
                "recommendations": recommendations,
            }

        except Exception as e:
            logger.error(f"Failed to get performance summary: {str(e)}")
            return {"error": str(e)}

    async def suggest_content_improvements(
        self,
        persona_id: UUID,
        platform: str,
    ) -> List[str]:
        """
        Generate specific content improvement suggestions.

        Args:
            persona_id: Persona ID
            platform: Social platform

        Returns:
            List of actionable suggestions
        """
        try:
            suggestions = []

            # Get optimal posting times
            optimal_times = await self.get_optimal_posting_times(persona_id, platform)
            if optimal_times:
                best_hour = list(optimal_times.keys())[0]
                suggestions.append(
                    f"Post around {best_hour}:00 for best engagement "
                    f"({optimal_times[best_hour]:.1f}% avg rate)"
                )

            # Get effective hashtags
            hashtags = await self.get_effective_hashtags(persona_id, platform)
            if hashtags:
                top_3 = [h[0] for h in hashtags[:3]]
                suggestions.append(
                    f"Use high-performing hashtags: #{', #'.join(top_3)}"
                )

            # Get successful patterns
            patterns = await self.get_successful_patterns(persona_id, platform, days=30)
            if patterns:
                avg_engagement = sum(p["engagement_rate"] for p in patterns) / len(patterns)
                suggestions.append(
                    f"Replicate elements from {len(patterns)} high-performing posts "
                    f"(avg {avg_engagement:.1f}% engagement)"
                )

            if not suggestions:
                suggestions.append("Collect more data for personalized recommendations")

            return suggestions

        except Exception as e:
            logger.error(f"Failed to generate suggestions: {str(e)}")
            return ["Error generating suggestions"]
