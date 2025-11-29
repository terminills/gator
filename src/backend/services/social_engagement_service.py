"""
Social Media Engagement Tracking Service

Handles real-time tracking of social media engagement metrics,
filters out bot/persona interactions, and integrates with ACD for learning.
"""

from datetime import datetime, timedelta, timezone
from typing import Any, Dict
from uuid import UUID

from sqlalchemy import and_, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.config.logging import get_logger
from backend.models.acd import ACDContextUpdate, AIConfidence, AIValidation
from backend.models.persona import PersonaModel
from backend.models.social_media_post import (
    EngagementAnalysis,
    EngagementMetrics,
    PostStatus,
    SocialMediaPostCreate,
    SocialMediaPostModel,
    SocialMediaPostResponse,
    SocialPlatform,
)
from backend.services.acd_service import ACDService
from backend.services.social_media_clients import (
    FacebookClient,
    InstagramClient,
    TikTokClient,
    TwitterClient,
)

logger = get_logger(__name__)


class SocialEngagementService:
    """
    Service for tracking social media engagement and learning from interactions.

    Monitors real-time metrics from social platforms, filters out bot/persona
    interactions, and updates ACD contexts for continuous learning.
    """

    def __init__(self, db_session: AsyncSession):
        """
        Initialize engagement tracking service.

        Args:
            db_session: Database session for persistence
        """
        self.db = db_session
        self.platform_clients = {
            SocialPlatform.INSTAGRAM: InstagramClient(),
            SocialPlatform.FACEBOOK: FacebookClient(),
            SocialPlatform.TWITTER: TwitterClient(),
            SocialPlatform.TIKTOK: TikTokClient(),
        }

    async def create_post_record(
        self, post_data: SocialMediaPostCreate
    ) -> SocialMediaPostResponse:
        """
        Create a social media post record.

        Args:
            post_data: Post creation data

        Returns:
            Created post record
        """
        try:
            post = SocialMediaPostModel(**post_data.model_dump())
            self.db.add(post)
            await self.db.commit()
            await self.db.refresh(post)

            logger.info(
                f"Created social media post record: {post.id} "
                f"for platform {post.platform}"
            )

            return SocialMediaPostResponse.model_validate(post)

        except Exception as e:
            logger.error(f"Failed to create post record: {str(e)}")
            await self.db.rollback()
            raise

    async def update_post_metrics(
        self, post_id: UUID, metrics: EngagementMetrics
    ) -> SocialMediaPostResponse:
        """
        Update engagement metrics for a post.

        Args:
            post_id: Post ID
            metrics: New engagement metrics

        Returns:
            Updated post record
        """
        try:
            stmt = select(SocialMediaPostModel).where(
                SocialMediaPostModel.id == post_id
            )
            result = await self.db.execute(stmt)
            post = result.scalar_one_or_none()

            if not post:
                raise ValueError(f"Post {post_id} not found")

            # Update metrics
            post.likes_count = metrics.likes_count
            post.comments_count = metrics.comments_count
            post.shares_count = metrics.shares_count
            post.saves_count = metrics.saves_count
            post.impressions = metrics.impressions
            post.reach = metrics.reach
            post.video_views = metrics.video_views
            post.video_completion_rate = metrics.video_completion_rate
            post.click_through_rate = metrics.click_through_rate

            # Update filtered metrics
            post.bot_interaction_count = metrics.bot_interaction_count
            post.persona_interaction_count = metrics.persona_interaction_count
            post.genuine_user_count = metrics.genuine_user_count

            # Store detailed data
            if metrics.top_comments:
                post.top_comments = metrics.top_comments
            if metrics.engagement_timeline:
                post.engagement_timeline = metrics.engagement_timeline
            if metrics.demographic_insights:
                post.demographic_insights = metrics.demographic_insights

            # Calculate engagement rate
            if post.reach > 0:
                total_engagement = (
                    post.likes_count
                    + post.comments_count
                    + post.shares_count
                    + post.saves_count
                )
                post.engagement_rate = (total_engagement / post.reach) * 100

            post.last_metrics_update = datetime.now(timezone.utc)

            await self.db.commit()
            await self.db.refresh(post)

            # Update ACD context with engagement data
            if post.acd_context_id:
                await self._update_acd_with_engagement(post, metrics)

            logger.info(
                f"Updated metrics for post {post_id}: "
                f"likes={post.likes_count}, engagement_rate={post.engagement_rate:.2f}%"
            )

            return SocialMediaPostResponse.model_validate(post)

        except Exception as e:
            logger.error(f"Failed to update post metrics: {str(e)}")
            await self.db.rollback()
            raise

    async def fetch_latest_metrics(
        self, post_id: UUID, account_access_token: str
    ) -> EngagementMetrics:
        """
        Fetch latest metrics from social platform.

        Args:
            post_id: Post ID in our database
            account_access_token: Platform access token

        Returns:
            Latest engagement metrics with bot filtering applied
        """
        try:
            stmt = select(SocialMediaPostModel).where(
                SocialMediaPostModel.id == post_id
            )
            result = await self.db.execute(stmt)
            post = result.scalar_one_or_none()

            if not post or not post.platform_post_id:
                raise ValueError(f"Post {post_id} not found or not published")

            # Get platform client
            platform = SocialPlatform(post.platform)
            client = self.platform_clients.get(platform)

            if not client:
                raise ValueError(f"Platform {platform} not supported")

            # Create account object for API calls
            from backend.services.social_media_service import SocialAccount

            account = SocialAccount(
                platform=platform,
                account_id=str(post.persona_id),
                access_token=account_access_token,
            )

            # Fetch raw metrics from platform
            raw_metrics = await client.get_engagement_metrics(
                account, post.platform_post_id
            )

            # Apply filtering to remove bot/persona interactions
            filtered_metrics = await self._filter_engagement_metrics(
                post, raw_metrics, account
            )

            return filtered_metrics

        except Exception as e:
            logger.error(f"Failed to fetch metrics: {str(e)}")
            raise

    async def _filter_engagement_metrics(
        self,
        post: SocialMediaPostModel,
        raw_metrics: Dict[str, Any],
        account: Any,
    ) -> EngagementMetrics:
        """
        Filter engagement metrics to remove bot and other AI persona interactions.

        Args:
            post: Post record
            raw_metrics: Raw metrics from platform
            account: Social media account credentials

        Returns:
            Filtered engagement metrics with genuine user interactions
        """
        try:
            # Extract raw counts
            total_likes = raw_metrics.get("likes", raw_metrics.get("like_count", 0))
            total_comments = raw_metrics.get(
                "comments", raw_metrics.get("reply_count", 0)
            )
            total_shares = raw_metrics.get("shares", raw_metrics.get("share_count", 0))
            total_saves = raw_metrics.get("saves", 0)
            impressions = raw_metrics.get("impressions", 0)
            reach = raw_metrics.get("reach", 0)
            video_views = raw_metrics.get(
                "video_views", raw_metrics.get("view_count", 0)
            )

            # Fetch detailed interaction data for filtering
            # This would require additional API calls to get user data
            # For now, we'll use heuristics and pattern detection

            # Identify bot patterns
            bot_count = await self._detect_bot_interactions(post, raw_metrics)

            # Identify other AI personas
            persona_count = await self._detect_persona_interactions(post, raw_metrics)

            # Calculate genuine user count
            total_interactions = total_likes + total_comments
            genuine_count = max(0, total_interactions - bot_count - persona_count)

            # Adjust metrics based on filtering
            filtering_ratio = (
                genuine_count / total_interactions if total_interactions > 0 else 1.0
            )

            filtered_likes = int(total_likes * filtering_ratio)
            filtered_comments = int(total_comments * filtering_ratio)
            filtered_shares = int(total_shares * filtering_ratio)

            return EngagementMetrics(
                likes_count=filtered_likes,
                comments_count=filtered_comments,
                shares_count=filtered_shares,
                saves_count=total_saves,
                impressions=impressions,
                reach=reach,
                video_views=video_views,
                video_completion_rate=raw_metrics.get("video_completion_rate"),
                click_through_rate=raw_metrics.get("click_through_rate"),
                bot_interaction_count=bot_count,
                persona_interaction_count=persona_count,
                genuine_user_count=genuine_count,
                top_comments=raw_metrics.get("top_comments"),
                engagement_timeline=raw_metrics.get("engagement_timeline"),
                demographic_insights=raw_metrics.get("demographic_insights"),
            )

        except Exception as e:
            logger.error(f"Failed to filter metrics: {str(e)}")
            # Return unfiltered metrics on error
            return EngagementMetrics(
                likes_count=raw_metrics.get("likes", 0),
                comments_count=raw_metrics.get("comments", 0),
                shares_count=raw_metrics.get("shares", 0),
                saves_count=raw_metrics.get("saves", 0),
                impressions=raw_metrics.get("impressions", 0),
                reach=raw_metrics.get("reach", 0),
            )

    async def _detect_bot_interactions(
        self, post: SocialMediaPostModel, raw_metrics: Dict[str, Any]
    ) -> int:
        """
        Detect bot interactions using pattern analysis.

        Heuristics include:
        - Very rapid engagement (< 1 second)
        - Generic comments
        - New accounts with no profile
        - Similar timing patterns
        """
        bot_indicators = 0

        # Analyze engagement timeline for bot patterns
        timeline = raw_metrics.get("engagement_timeline", {})
        if timeline:
            # Check for unnaturally rapid engagement spikes
            for hour, count in timeline.items():
                if count > 100:  # Threshold for suspicious activity
                    bot_indicators += int(count * 0.1)  # Estimate 10% are bots

        # Analyze comments for bot patterns
        comments = raw_metrics.get("top_comments", [])
        for comment in comments:
            comment_text = comment.get("text", "").lower()
            # Generic bot phrases
            bot_phrases = [
                "check out my profile",
                "click my link",
                "dm for collab",
                "follow back",
                "🔥🔥🔥",  # Only emojis
            ]
            if any(phrase in comment_text for phrase in bot_phrases):
                bot_indicators += 1

        return bot_indicators

    async def _detect_persona_interactions(
        self, post: SocialMediaPostModel, raw_metrics: Dict[str, Any]
    ) -> int:
        """
        Detect interactions from other AI personas.

        Strategy:
        - Check against known AI persona accounts in database
        - Pattern detection for AI-generated comments
        - Cross-reference with persona network
        """
        try:
            # Get all known personas
            stmt = select(PersonaModel).where(
                PersonaModel.id != post.persona_id, PersonaModel.is_active == True
            )
            result = await self.db.execute(stmt)
            known_personas = result.scalars().all()

            # Extract social handles from personas
            persona_handles = set()
            for persona in known_personas:
                # Personas might have social media handles in metadata
                if hasattr(persona, "social_accounts"):
                    for account in persona.social_accounts or []:
                        persona_handles.add(account.get("username", "").lower())

            # Check comments against known personas
            persona_count = 0
            comments = raw_metrics.get("top_comments", [])
            for comment in comments:
                username = comment.get("username", "").lower()
                if username in persona_handles:
                    persona_count += 1

            return persona_count

        except Exception as e:
            logger.error(f"Failed to detect persona interactions: {str(e)}")
            return 0

    async def _update_acd_with_engagement(
        self, post: SocialMediaPostModel, metrics: EngagementMetrics
    ):
        """
        Update ACD context with engagement data for learning.

        Args:
            post: Social media post
            metrics: Engagement metrics
        """
        try:
            acd_service = ACDService(self.db)

            # Calculate engagement quality score
            total_engagement = (
                metrics.likes_count
                + metrics.comments_count
                + metrics.shares_count
                + metrics.saves_count
            )
            genuine_ratio = (
                metrics.genuine_user_count / total_engagement
                if total_engagement > 0
                else 0
            )

            # Determine validation based on engagement quality
            if post.engagement_rate and post.engagement_rate > 5.0:  # > 5% is excellent
                validation = AIValidation.APPROVED
                confidence = AIConfidence.VALIDATED
            elif post.engagement_rate and post.engagement_rate > 2.0:  # > 2% is good
                validation = AIValidation.CONDITIONALLY_APPROVED
                confidence = AIConfidence.CONFIDENT
            else:
                validation = AIValidation.ANALYZED
                confidence = AIConfidence.UNCERTAIN

            # Prepare update data
            update_data = ACDContextUpdate(
                ai_validation=validation,
                ai_confidence=confidence,
                ai_metadata={
                    "social_metrics": {
                        "platform": post.platform,
                        "likes": metrics.likes_count,
                        "comments": metrics.comments_count,
                        "shares": metrics.shares_count,
                        "engagement_rate": post.engagement_rate,
                        "genuine_user_count": metrics.genuine_user_count,
                        "bot_filtered_count": metrics.bot_interaction_count,
                        "persona_filtered_count": metrics.persona_interaction_count,
                        "genuine_ratio": genuine_ratio,
                    }
                },
            )

            # Extract patterns from high-performing posts
            if post.engagement_rate and post.engagement_rate > 5.0:
                # Extract successful patterns
                patterns = []
                if post.hashtags:
                    patterns.append(f"hashtags:{','.join(post.hashtags[:3])}")
                if post.published_at:
                    hour = post.published_at.hour
                    patterns.append(f"published_hour:{hour}")

                update_data.ai_pattern = f"{post.platform}_high_engagement"
                update_data.ai_strategy = (
                    f"High engagement on {post.platform}: "
                    f"{post.engagement_rate:.2f}% engagement rate, "
                    f"patterns: {', '.join(patterns)}"
                )

            await acd_service.update_context(post.acd_context_id, update_data)

            logger.info(
                f"Updated ACD context {post.acd_context_id} with engagement data: "
                f"validation={validation.value}, engagement_rate={post.engagement_rate:.2f}%"
            )

        except Exception as e:
            logger.error(f"Failed to update ACD with engagement: {str(e)}")

    async def analyze_post_performance(self, post_id: UUID) -> EngagementAnalysis:
        """
        Analyze post performance and generate recommendations.

        Args:
            post_id: Post ID

        Returns:
            Detailed engagement analysis with recommendations
        """
        try:
            stmt = select(SocialMediaPostModel).where(
                SocialMediaPostModel.id == post_id
            )
            result = await self.db.execute(stmt)
            post = result.scalar_one_or_none()

            if not post:
                raise ValueError(f"Post {post_id} not found")

            # Calculate metrics
            total_engagement = (
                post.likes_count
                + post.comments_count
                + post.shares_count
                + post.saves_count
            )
            genuine_engagement = post.genuine_user_count
            engagement_rate = post.engagement_rate or 0.0

            # Get persona average for comparison
            avg_stmt = select(func.avg(SocialMediaPostModel.engagement_rate)).where(
                SocialMediaPostModel.persona_id == post.persona_id,
                SocialMediaPostModel.platform == post.platform,
                SocialMediaPostModel.status == PostStatus.PUBLISHED.value,
            )
            avg_result = await self.db.execute(avg_stmt)
            persona_avg = avg_result.scalar() or 0.0

            performance_vs_average = (
                ((engagement_rate - persona_avg) / persona_avg * 100)
                if persona_avg > 0
                else 0.0
            )

            # Identify top performing elements
            top_elements = []
            if post.hashtags and post.engagement_rate and post.engagement_rate > 3.0:
                top_elements.append(f"Hashtags: {', '.join(post.hashtags[:3])}")
            if post.published_at:
                hour = post.published_at.hour
                if 10 <= hour <= 14 or 18 <= hour <= 21:  # Peak hours
                    top_elements.append(f"Posted during peak time ({hour}:00)")

            # Generate recommendations
            recommendations = []
            if engagement_rate < persona_avg:
                recommendations.append(
                    "Consider posting during peak hours (10am-2pm or 6pm-9pm)"
                )
                recommendations.append("Try trending hashtags relevant to content")
            if post.comments_count < post.likes_count * 0.1:
                recommendations.append(
                    "Add engaging questions to caption to encourage comments"
                )
            if post.shares_count < total_engagement * 0.05:
                recommendations.append(
                    "Create more shareable content (tips, quotes, infographics)"
                )

            # Sentiment analysis (basic for now)
            sentiment = {
                "positive": 0.7,
                "neutral": 0.2,
                "negative": 0.1,
            }  # Placeholder

            return EngagementAnalysis(
                post_id=post.id,
                platform=SocialPlatform(post.platform),
                total_engagement=total_engagement,
                genuine_engagement=genuine_engagement,
                engagement_rate=engagement_rate,
                performance_vs_average=performance_vs_average,
                top_performing_elements=top_elements,
                sentiment_analysis=sentiment,
                best_performing_time=(
                    f"{post.published_at.hour}:00" if post.published_at else None
                ),
                recommendations=recommendations,
            )

        except Exception as e:
            logger.error(f"Failed to analyze post performance: {str(e)}")
            raise

    async def sync_all_recent_posts(
        self, hours: int = 24, access_tokens: Dict[UUID, str] = None
    ) -> Dict[str, Any]:
        """
        Sync engagement metrics for all recent posts.

        Args:
            hours: How many hours back to sync
            access_tokens: Map of persona_id to access token

        Returns:
            Summary of sync operation
        """
        try:
            cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)

            # Get recent published posts
            stmt = select(SocialMediaPostModel).where(
                and_(
                    SocialMediaPostModel.status == PostStatus.PUBLISHED.value,
                    SocialMediaPostModel.published_at >= cutoff,
                )
            )
            result = await self.db.execute(stmt)
            posts = result.scalars().all()

            synced = 0
            failed = 0

            for post in posts:
                try:
                    # Skip if no access token provided
                    if not access_tokens or post.persona_id not in access_tokens:
                        continue

                    # Fetch and update metrics
                    metrics = await self.fetch_latest_metrics(
                        post.id, access_tokens[post.persona_id]
                    )
                    await self.update_post_metrics(post.id, metrics)
                    synced += 1

                except Exception as e:
                    logger.error(f"Failed to sync post {post.id}: {str(e)}")
                    failed += 1

            logger.info(
                f"Synced engagement metrics: {synced} successful, {failed} failed"
            )

            return {
                "total_posts": len(posts),
                "synced": synced,
                "failed": failed,
                "time_window_hours": hours,
            }

        except Exception as e:
            logger.error(f"Failed to sync posts: {str(e)}")
            raise
