"""
ACD Understanding Service

Provides system understanding and recall capabilities for the diagnostics chat.
This service enables the Gator agent to understand and discuss the system's
current state, ACD contexts, and overall platform health.

The ACD (Autonomous Continuous Development) system provides timestamped recall
enabling the LLM to reason about:
- Content performance and engagement metrics
- PPV upsell conversions and revenue patterns
- Social media engagement (likes, comments, shares)
- User churn and retention analysis
- Traffic patterns and conversion funnels
- Historical trends for intelligent forecasting
"""

from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional
from uuid import UUID

from sqlalchemy import and_, desc, select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.config.logging import get_logger
from backend.models.acd import (
    ACDContextModel,
    ACDTraceArtifactModel,
)
from backend.models.content import ContentModel
from backend.models.conversation import ConversationModel
from backend.models.message import MessageModel
from backend.models.persona import PersonaModel
from backend.models.ppv_offer import PPVOfferModel, PPVOfferStatus
from backend.models.social_media_post import SocialMediaPostModel
from backend.models.user import UserModel

logger = get_logger(__name__)


class ACDUnderstandingService:
    """
    Service for providing comprehensive system understanding and recall
    capabilities for the ACD system.

    Enables the Gator agent to:
    - Understand current system state
    - Recall historical ACD contexts and their outcomes
    - Explain what the system is doing and why
    - Provide insights on system health and performance
    """

    def __init__(self, db_session: AsyncSession):
        """
        Initialize ACD understanding service.

        Args:
            db_session: Database session for persistence
        """
        self.db = db_session

    async def get_system_understanding(
        self, hours: int = 24, include_details: bool = True
    ) -> Dict[str, Any]:
        """
        Get a comprehensive understanding of the current system state.

        This method aggregates information from all system components
        to provide a complete picture of what's happening, including:
        - System health and ACD activity
        - Content performance metrics
        - Social media engagement
        - PPV conversions and revenue
        - User engagement and churn indicators

        Args:
            hours: Time window to analyze (default 24 hours)
            include_details: Whether to include detailed breakdowns

        Returns:
            Comprehensive system understanding dictionary
        """
        try:
            cutoff_time = datetime.now(timezone.utc) - timedelta(hours=hours)

            understanding = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "analysis_window_hours": hours,
                "system_state": await self._get_system_state(cutoff_time),
                "acd_summary": await self._get_acd_summary(cutoff_time),
                "content_activity": await self._get_content_activity(cutoff_time),
                "persona_status": await self._get_persona_status(),
                "error_analysis": await self._get_error_analysis(cutoff_time),
                # Business Intelligence
                "engagement_metrics": await self._get_engagement_metrics(cutoff_time),
                "ppv_performance": await self._get_ppv_performance(cutoff_time),
                "user_activity": await self._get_user_activity(cutoff_time),
                "traffic_analysis": await self._get_traffic_analysis(cutoff_time),
                "churn_indicators": await self._get_churn_indicators(cutoff_time),
                "recommendations": [],
            }

            # Generate recommendations based on the analysis
            understanding["recommendations"] = self._generate_recommendations(
                understanding
            )

            return understanding

        except Exception as e:
            logger.error(f"Failed to get system understanding: {str(e)}")
            return {
                "error": str(e),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

    async def _get_system_state(self, cutoff_time: datetime) -> Dict[str, Any]:
        """Get overall system state."""
        try:
            # Check ACD contexts state distribution
            stmt = select(ACDContextModel).where(
                ACDContextModel.created_at >= cutoff_time
            )
            result = await self.db.execute(stmt)
            contexts = result.scalars().all()

            # Categorize contexts by state
            state_counts = {}
            for ctx in contexts:
                state = ctx.ai_state or "UNKNOWN"
                state_counts[state] = state_counts.get(state, 0) + 1

            # Calculate health indicators
            total = len(contexts)
            processing = state_counts.get("PROCESSING", 0)
            done = state_counts.get("DONE", 0)
            failed = state_counts.get("FAILED", 0)

            # Determine overall health
            if total == 0:
                health_status = "IDLE"
                health_score = 100
            elif failed / max(total, 1) > 0.5:
                health_status = "CRITICAL"
                health_score = max(0, 100 - (failed / total * 100))
            elif failed / max(total, 1) > 0.2:
                health_status = "DEGRADED"
                health_score = max(0, 100 - (failed / total * 50))
            elif processing > 10:
                health_status = "BUSY"
                health_score = 80
            else:
                health_status = "HEALTHY"
                health_score = 100 - min(failed / max(total, 1) * 20, 20)

            return {
                "health_status": health_status,
                "health_score": round(health_score, 1),
                "total_contexts": total,
                "state_distribution": state_counts,
                "active_processing": processing,
                "completed_successfully": done,
                "failures": failed,
            }

        except Exception as e:
            logger.error(f"Error getting system state: {str(e)}")
            return {"health_status": "UNKNOWN", "error": str(e)}

    async def _get_acd_summary(self, cutoff_time: datetime) -> Dict[str, Any]:
        """Get ACD contexts summary."""
        try:
            stmt = (
                select(ACDContextModel)
                .where(ACDContextModel.created_at >= cutoff_time)
                .order_by(desc(ACDContextModel.created_at))
            )

            result = await self.db.execute(stmt)
            contexts = result.scalars().all()

            # Analyze by phase/domain
            by_phase = {}
            by_domain = {}
            by_confidence = {}

            for ctx in contexts:
                # Count by phase
                phase = ctx.ai_phase or "UNKNOWN"
                by_phase[phase] = by_phase.get(phase, 0) + 1

                # Count by domain
                domain = ctx.ai_domain or "UNCLASSIFIED"
                by_domain[domain] = by_domain.get(domain, 0) + 1

                # Count by confidence
                confidence = ctx.ai_confidence or "NOT_SET"
                by_confidence[confidence] = by_confidence.get(confidence, 0) + 1

            # Get most recent contexts for recall
            recent_contexts = []
            for ctx in contexts[:10]:
                recent_contexts.append(
                    {
                        "id": str(ctx.id),
                        "phase": ctx.ai_phase,
                        "state": ctx.ai_state,
                        "status": ctx.ai_status,
                        "domain": ctx.ai_domain,
                        "note": ctx.ai_note,
                        "created_at": (
                            ctx.created_at.isoformat() if ctx.created_at else None
                        ),
                    }
                )

            return {
                "total_contexts": len(contexts),
                "by_phase": by_phase,
                "by_domain": by_domain,
                "by_confidence": by_confidence,
                "recent_contexts": recent_contexts,
            }

        except Exception as e:
            logger.error(f"Error getting ACD summary: {str(e)}")
            return {"error": str(e)}

    async def _get_content_activity(self, cutoff_time: datetime) -> Dict[str, Any]:
        """Get content generation activity."""
        try:
            stmt = select(ContentModel).where(ContentModel.created_at >= cutoff_time)
            result = await self.db.execute(stmt)
            contents = result.scalars().all()

            # Analyze content by type
            by_type = {}
            successful = 0
            failed = 0
            with_fallback = 0

            for content in contents:
                # Count by type
                ct = content.content_type or "UNKNOWN"
                by_type[ct] = by_type.get(ct, 0) + 1

                # Analyze generation params for success/failure
                gen_params = content.generation_params or {}
                if gen_params.get("error"):
                    failed += 1
                elif gen_params.get("fallback") or gen_params.get("template_based"):
                    with_fallback += 1
                else:
                    successful += 1

            return {
                "total_content": len(contents),
                "successful": successful,
                "failed": failed,
                "with_fallback": with_fallback,
                "by_type": by_type,
                "success_rate": round(successful / max(len(contents), 1) * 100, 1),
            }

        except Exception as e:
            logger.error(f"Error getting content activity: {str(e)}")
            return {"error": str(e)}

    async def _get_persona_status(self) -> Dict[str, Any]:
        """Get persona status information."""
        try:
            stmt = select(PersonaModel)
            result = await self.db.execute(stmt)
            personas = result.scalars().all()

            active_count = sum(1 for p in personas if p.is_active)

            persona_info = []
            for p in personas[:10]:  # Limit to 10 for summary
                persona_info.append(
                    {
                        "id": str(p.id),
                        "name": p.name,
                        "is_active": p.is_active,
                        "content_themes": p.content_themes or [],
                        "post_count": p.post_count or 0,
                    }
                )

            return {
                "total_personas": len(personas),
                "active_personas": active_count,
                "inactive_personas": len(personas) - active_count,
                "personas": persona_info,
            }

        except Exception as e:
            logger.error(f"Error getting persona status: {str(e)}")
            return {"error": str(e)}

    async def _get_error_analysis(self, cutoff_time: datetime) -> Dict[str, Any]:
        """Get error analysis from trace artifacts."""
        try:
            stmt = (
                select(ACDTraceArtifactModel)
                .where(ACDTraceArtifactModel.timestamp >= cutoff_time)
                .order_by(desc(ACDTraceArtifactModel.timestamp))
            )

            result = await self.db.execute(stmt)
            artifacts = result.scalars().all()

            # Analyze errors by type
            by_event_type = {}
            by_error_code = {}

            for artifact in artifacts:
                # Count by event type
                event_type = artifact.event_type or "UNKNOWN"
                by_event_type[event_type] = by_event_type.get(event_type, 0) + 1

                # Count by error code
                error_code = artifact.error_code or "UNCLASSIFIED"
                by_error_code[error_code] = by_error_code.get(error_code, 0) + 1

            # Get most recent errors
            recent_errors = []
            for artifact in artifacts[:5]:
                recent_errors.append(
                    {
                        "id": str(artifact.id),
                        "event_type": artifact.event_type,
                        "error_message": artifact.error_message,
                        "error_file": artifact.error_file,
                        "error_line": artifact.error_line,
                        "timestamp": (
                            artifact.timestamp.isoformat()
                            if artifact.timestamp
                            else None
                        ),
                    }
                )

            return {
                "total_errors": len(artifacts),
                "by_event_type": by_event_type,
                "by_error_code": by_error_code,
                "recent_errors": recent_errors,
            }

        except Exception as e:
            logger.error(f"Error getting error analysis: {str(e)}")
            return {"error": str(e)}

    # ============================================================
    # Business Intelligence Methods - Engagement, PPV, Traffic, Churn
    # ============================================================

    async def _get_engagement_metrics(self, cutoff_time: datetime) -> Dict[str, Any]:
        """
        Get social media engagement metrics for LLM reasoning.

        Includes likes, comments, shares, impressions across all platforms
        to enable intelligent analysis of content performance.
        """
        try:
            stmt = (
                select(SocialMediaPostModel)
                .where(SocialMediaPostModel.created_at >= cutoff_time)
                .order_by(desc(SocialMediaPostModel.created_at))
            )

            result = await self.db.execute(stmt)
            posts = result.scalars().all()

            if not posts:
                return {
                    "total_posts": 0,
                    "message": "No social media posts in analysis window",
                }

            # Aggregate metrics
            total_likes = sum(p.likes_count or 0 for p in posts)
            total_comments = sum(p.comments_count or 0 for p in posts)
            total_shares = sum(p.shares_count or 0 for p in posts)
            total_saves = sum(p.saves_count or 0 for p in posts)
            total_impressions = sum(p.impressions or 0 for p in posts)
            total_reach = sum(p.reach or 0 for p in posts)
            total_video_views = sum(p.video_views or 0 for p in posts)

            # Calculate engagement rates
            engagement_rates = [p.engagement_rate for p in posts if p.engagement_rate]
            avg_engagement_rate = (
                sum(engagement_rates) / len(engagement_rates) if engagement_rates else 0
            )

            # Track genuine vs bot interactions
            genuine_users = sum(p.genuine_user_count or 0 for p in posts)
            bot_interactions = sum(p.bot_interaction_count or 0 for p in posts)
            persona_interactions = sum(p.persona_interaction_count or 0 for p in posts)

            # Performance analysis
            performance_scores = [
                p.compared_to_average for p in posts if p.compared_to_average
            ]
            avg_performance = (
                sum(performance_scores) / len(performance_scores)
                if performance_scores
                else 0
            )

            # By platform breakdown
            by_platform = {}
            for post in posts:
                platform = post.platform or "unknown"
                if platform not in by_platform:
                    by_platform[platform] = {
                        "posts": 0,
                        "likes": 0,
                        "comments": 0,
                        "shares": 0,
                        "impressions": 0,
                    }
                by_platform[platform]["posts"] += 1
                by_platform[platform]["likes"] += post.likes_count or 0
                by_platform[platform]["comments"] += post.comments_count or 0
                by_platform[platform]["shares"] += post.shares_count or 0
                by_platform[platform]["impressions"] += post.impressions or 0

            # Top performing posts
            top_posts = sorted(
                posts, key=lambda p: (p.engagement_rate or 0), reverse=True
            )[:5]
            top_performing = [
                {
                    "id": str(p.id),
                    "platform": p.platform,
                    "likes": p.likes_count,
                    "comments": p.comments_count,
                    "engagement_rate": p.engagement_rate,
                    "performance_percentile": p.performance_percentile,
                }
                for p in top_posts
            ]

            return {
                "total_posts": len(posts),
                "total_engagement": {
                    "likes": total_likes,
                    "comments": total_comments,
                    "shares": total_shares,
                    "saves": total_saves,
                    "impressions": total_impressions,
                    "reach": total_reach,
                    "video_views": total_video_views,
                },
                "avg_engagement_rate": round(avg_engagement_rate, 2),
                "avg_performance_vs_baseline": round(avg_performance, 2),
                "interaction_quality": {
                    "genuine_users": genuine_users,
                    "bot_interactions": bot_interactions,
                    "persona_interactions": persona_interactions,
                    "genuine_ratio": round(
                        genuine_users / max(genuine_users + bot_interactions, 1) * 100,
                        1,
                    ),
                },
                "by_platform": by_platform,
                "top_performing_posts": top_performing,
            }

        except Exception as e:
            logger.error(f"Error getting engagement metrics: {str(e)}")
            return {"error": str(e), "total_posts": 0}

    async def _get_ppv_performance(self, cutoff_time: datetime) -> Dict[str, Any]:
        """
        Get PPV (Pay-Per-View) upsell performance metrics.

        Enables LLM reasoning about conversion rates, revenue,
        and optimal upselling strategies.
        """
        try:
            stmt = select(PPVOfferModel).where(PPVOfferModel.created_at >= cutoff_time)

            result = await self.db.execute(stmt)
            offers = result.scalars().all()

            if not offers:
                return {
                    "total_offers": 0,
                    "message": "No PPV offers in analysis window",
                }

            # Count by status
            total = len(offers)
            accepted = sum(1 for o in offers if o.status == PPVOfferStatus.ACCEPTED)
            declined = sum(1 for o in offers if o.status == PPVOfferStatus.DECLINED)
            pending = sum(1 for o in offers if o.status == PPVOfferStatus.PENDING)
            expired = sum(1 for o in offers if o.status == PPVOfferStatus.EXPIRED)

            # Calculate conversion rate
            responded = accepted + declined
            conversion_rate = (accepted / responded * 100) if responded > 0 else 0

            # Revenue analysis
            accepted_offers = [o for o in offers if o.status == PPVOfferStatus.ACCEPTED]
            total_revenue = sum(float(o.price or 0) for o in accepted_offers)
            avg_price = (
                sum(float(o.price or 0) for o in offers) / total if total > 0 else 0
            )
            avg_accepted_price = (
                total_revenue / len(accepted_offers) if accepted_offers else 0
            )

            # By offer type
            by_type = {}
            for offer in offers:
                offer_type = offer.offer_type.value if offer.offer_type else "unknown"
                if offer_type not in by_type:
                    by_type[offer_type] = {"total": 0, "accepted": 0, "revenue": 0}
                by_type[offer_type]["total"] += 1
                if offer.status == PPVOfferStatus.ACCEPTED:
                    by_type[offer_type]["accepted"] += 1
                    by_type[offer_type]["revenue"] += float(offer.price or 0)

            # Calculate conversion by type
            for ot in by_type:
                by_type[ot]["conversion_rate"] = round(
                    by_type[ot]["accepted"] / max(by_type[ot]["total"], 1) * 100, 1
                )

            # By persona
            by_persona = {}
            for offer in offers:
                persona_id = str(offer.persona_id) if offer.persona_id else "unknown"
                if persona_id not in by_persona:
                    by_persona[persona_id] = {"total": 0, "accepted": 0, "revenue": 0}
                by_persona[persona_id]["total"] += 1
                if offer.status == PPVOfferStatus.ACCEPTED:
                    by_persona[persona_id]["accepted"] += 1
                    by_persona[persona_id]["revenue"] += float(offer.price or 0)

            return {
                "total_offers": total,
                "status_breakdown": {
                    "accepted": accepted,
                    "declined": declined,
                    "pending": pending,
                    "expired": expired,
                },
                "conversion_rate": round(conversion_rate, 1),
                "revenue": {
                    "total": round(total_revenue, 2),
                    "avg_offer_price": round(avg_price, 2),
                    "avg_accepted_price": round(avg_accepted_price, 2),
                },
                "by_offer_type": by_type,
                "offers_by_persona_count": len(by_persona),
                "top_converting_types": sorted(
                    [(k, v["conversion_rate"]) for k, v in by_type.items()],
                    key=lambda x: x[1],
                    reverse=True,
                )[:3],
            }

        except Exception as e:
            logger.error(f"Error getting PPV performance: {str(e)}")
            return {"error": str(e), "total_offers": 0}

    async def _get_user_activity(self, cutoff_time: datetime) -> Dict[str, Any]:
        """
        Get user activity metrics for engagement analysis.

        Tracks active users, conversation patterns, and engagement depth.
        """
        try:
            # Get conversations in time window
            conv_stmt = select(ConversationModel).where(
                ConversationModel.created_at >= cutoff_time
            )
            conv_result = await self.db.execute(conv_stmt)
            conversations = conv_result.scalars().all()

            # Get messages in time window
            msg_stmt = select(MessageModel).where(
                MessageModel.created_at >= cutoff_time
            )
            msg_result = await self.db.execute(msg_stmt)
            messages = msg_result.scalars().all()

            # Get active users
            user_stmt = select(UserModel).where(UserModel.is_active.is_(True))
            user_result = await self.db.execute(user_stmt)
            users = user_result.scalars().all()

            # Analyze message patterns
            # Default to False (user message) if is_ai_generated attribute doesn't exist
            user_messages = [
                m for m in messages if not getattr(m, "is_ai_generated", False)
            ]
            ai_messages = [m for m in messages if getattr(m, "is_ai_generated", False)]

            # Unique active users (those who sent messages)
            active_user_ids = set(m.sender_id for m in user_messages if m.sender_id)

            # Calculate avg messages per conversation
            conv_message_counts = {}
            for msg in messages:
                conv_id = str(msg.conversation_id) if msg.conversation_id else "unknown"
                conv_message_counts[conv_id] = conv_message_counts.get(conv_id, 0) + 1

            avg_messages_per_conv = (
                sum(conv_message_counts.values()) / len(conv_message_counts)
                if conv_message_counts
                else 0
            )

            return {
                "total_users": len(users),
                "active_users": len(active_user_ids),
                "conversations": {
                    "total": len(conversations),
                    "avg_messages_per_conversation": round(avg_messages_per_conv, 1),
                },
                "messages": {
                    "total": len(messages),
                    "user_messages": len(user_messages),
                    "ai_messages": len(ai_messages),
                },
                "engagement_depth": round(
                    len(user_messages) / max(len(active_user_ids), 1), 1
                ),
            }

        except Exception as e:
            logger.error(f"Error getting user activity: {str(e)}")
            return {"error": str(e)}

    async def _get_traffic_analysis(self, cutoff_time: datetime) -> Dict[str, Any]:
        """
        Analyze traffic patterns and content funnel.

        Shows how content moves through the system and converts to engagement.
        """
        try:
            # Content created
            content_stmt = select(ContentModel).where(
                ContentModel.created_at >= cutoff_time
            )
            content_result = await self.db.execute(content_stmt)
            contents = content_result.scalars().all()

            # Social posts (content distributed)
            posts_stmt = select(SocialMediaPostModel).where(
                SocialMediaPostModel.created_at >= cutoff_time
            )
            posts_result = await self.db.execute(posts_stmt)
            posts = posts_result.scalars().all()

            # Calculate funnel metrics
            content_created = len(contents)
            content_published = sum(1 for c in contents if c.is_published)
            posts_published = sum(1 for p in posts if p.status == "published")

            # Impressions from published content
            total_impressions = sum(p.impressions or 0 for p in posts)
            total_reach = sum(p.reach or 0 for p in posts)

            # Calculate conversion rates through funnel
            publish_rate = (content_published / max(content_created, 1)) * 100
            distribution_rate = (posts_published / max(content_published, 1)) * 100

            return {
                "content_funnel": {
                    "created": content_created,
                    "published": content_published,
                    "distributed_to_social": posts_published,
                    "publish_rate": round(publish_rate, 1),
                    "distribution_rate": round(distribution_rate, 1),
                },
                "reach_metrics": {
                    "total_impressions": total_impressions,
                    "total_reach": total_reach,
                    "avg_impressions_per_post": round(
                        total_impressions / max(posts_published, 1), 1
                    ),
                },
                "content_velocity": round(
                    content_created / max(24, 1), 2
                ),  # per hour average
            }

        except Exception as e:
            logger.error(f"Error getting traffic analysis: {str(e)}")
            return {"error": str(e)}

    async def _get_churn_indicators(self, cutoff_time: datetime) -> Dict[str, Any]:
        """
        Analyze user churn indicators and retention patterns.

        Enables LLM to reason about user retention and engagement drop-offs.
        """
        try:
            # Get all users
            user_stmt = select(UserModel)
            user_result = await self.db.execute(user_stmt)
            users = user_result.scalars().all()

            if not users:
                return {"message": "No users to analyze"}

            # Analyze activity patterns
            now = datetime.now(timezone.utc)

            # Categorize users by last activity
            active_24h = 0
            active_7d = 0
            active_30d = 0
            inactive_30d = 0

            for user in users:
                last_active = user.updated_at or user.created_at
                if last_active:
                    days_since_active = (now - last_active).days
                    if days_since_active <= 1:
                        active_24h += 1
                    if days_since_active <= 7:
                        active_7d += 1
                    if days_since_active <= 30:
                        active_30d += 1
                    else:
                        inactive_30d += 1

            total_users = len(users)

            # Calculate retention metrics
            retention_7d = (active_7d / total_users * 100) if total_users > 0 else 0
            retention_30d = (active_30d / total_users * 100) if total_users > 0 else 0
            churn_rate = (inactive_30d / total_users * 100) if total_users > 0 else 0

            # Analyze PPV acceptance as engagement indicator
            ppv_stmt = select(PPVOfferModel).where(
                PPVOfferModel.created_at >= cutoff_time
            )
            ppv_result = await self.db.execute(ppv_stmt)
            ppv_offers = ppv_result.scalars().all()

            ppv_acceptance_rate = 0
            if ppv_offers:
                accepted = sum(
                    1 for o in ppv_offers if o.status == PPVOfferStatus.ACCEPTED
                )
                total_responded = sum(
                    1
                    for o in ppv_offers
                    if o.status in [PPVOfferStatus.ACCEPTED, PPVOfferStatus.DECLINED]
                )
                ppv_acceptance_rate = (accepted / max(total_responded, 1)) * 100

            # Risk indicators
            risk_level = "LOW"
            risk_factors = []

            if churn_rate > 30:
                risk_level = "HIGH"
                risk_factors.append(
                    f"High churn rate: {round(churn_rate, 1)}% inactive"
                )
            elif churn_rate > 15:
                risk_level = "MEDIUM"
                risk_factors.append(f"Moderate churn: {round(churn_rate, 1)}% inactive")

            if retention_7d < 50:
                if risk_level == "LOW":
                    risk_level = "MEDIUM"
                risk_factors.append(f"Low 7-day retention: {round(retention_7d, 1)}%")

            if ppv_acceptance_rate < 10 and ppv_offers:
                risk_factors.append(
                    f"Low PPV conversion: {round(ppv_acceptance_rate, 1)}%"
                )

            return {
                "user_activity_breakdown": {
                    "total_users": total_users,
                    "active_24h": active_24h,
                    "active_7d": active_7d,
                    "active_30d": active_30d,
                    "inactive_30d_plus": inactive_30d,
                },
                "retention_metrics": {
                    "retention_7d": round(retention_7d, 1),
                    "retention_30d": round(retention_30d, 1),
                    "churn_rate_30d": round(churn_rate, 1),
                },
                "engagement_indicators": {
                    "ppv_acceptance_rate": round(ppv_acceptance_rate, 1),
                },
                "risk_assessment": {
                    "level": risk_level,
                    "factors": risk_factors,
                },
            }

        except Exception as e:
            logger.error(f"Error getting churn indicators: {str(e)}")
            return {"error": str(e)}

    def _generate_recommendations(
        self, understanding: Dict[str, Any]
    ) -> List[Dict[str, str]]:
        """
        Generate intelligent recommendations based on comprehensive system understanding.

        Analyzes system health, engagement metrics, PPV performance, and churn indicators
        to provide actionable insights.
        """
        recommendations = []

        # Check system state
        system_state = understanding.get("system_state", {})
        health_status = system_state.get("health_status", "UNKNOWN")
        failures = system_state.get("failures", 0)

        if health_status == "CRITICAL":
            recommendations.append(
                {
                    "priority": "HIGH",
                    "category": "SYSTEM_HEALTH",
                    "message": "System is in critical state with high failure rate. Check error logs and investigate failing contexts.",
                    "action": "Review recent trace artifacts and identify root causes.",
                }
            )
        elif health_status == "DEGRADED":
            recommendations.append(
                {
                    "priority": "MEDIUM",
                    "category": "SYSTEM_HEALTH",
                    "message": "System performance is degraded. Some operations are failing.",
                    "action": "Monitor error trends and consider scaling resources.",
                }
            )

        if failures > 0:
            recommendations.append(
                {
                    "priority": "MEDIUM",
                    "category": "ERROR_RESOLUTION",
                    "message": f"{failures} ACD contexts have failed in the analysis window.",
                    "action": "Use the error analysis to identify patterns and fix recurring issues.",
                }
            )

        # Check content activity
        content_activity = understanding.get("content_activity", {})
        success_rate = content_activity.get("success_rate", 100)

        if success_rate < 80:
            recommendations.append(
                {
                    "priority": "MEDIUM",
                    "category": "CONTENT_GENERATION",
                    "message": f"Content generation success rate is {success_rate}%. Below optimal threshold.",
                    "action": "Check AI model configuration and ensure models are properly loaded.",
                }
            )

        # Check persona status
        persona_status = understanding.get("persona_status", {})
        total_personas = persona_status.get("total_personas", 0)

        if total_personas == 0:
            recommendations.append(
                {
                    "priority": "LOW",
                    "category": "SETUP",
                    "message": "No personas configured in the system.",
                    "action": "Create at least one persona to enable content generation.",
                }
            )

        # ============================================================
        # Business Intelligence Recommendations
        # ============================================================

        # Engagement recommendations
        engagement = understanding.get("engagement_metrics", {})
        if engagement and not engagement.get("error"):
            avg_engagement = engagement.get("avg_engagement_rate", 0)
            if avg_engagement < 2:
                recommendations.append(
                    {
                        "priority": "MEDIUM",
                        "category": "ENGAGEMENT",
                        "message": f"Average engagement rate is {avg_engagement}%, below industry standard of 3-6%.",
                        "action": "Analyze top-performing content and replicate successful elements. Consider adjusting posting times and content types.",
                    }
                )

            interaction_quality = engagement.get("interaction_quality", {})
            genuine_ratio = interaction_quality.get("genuine_ratio", 100)
            if genuine_ratio < 70:
                recommendations.append(
                    {
                        "priority": "LOW",
                        "category": "ENGAGEMENT_QUALITY",
                        "message": f"Only {genuine_ratio}% of interactions are from genuine users. High bot/persona interaction rate.",
                        "action": "Review content strategy and consider adjusting targeting to reach more real users.",
                    }
                )

        # PPV recommendations
        ppv = understanding.get("ppv_performance", {})
        if ppv and not ppv.get("error") and ppv.get("total_offers", 0) > 0:
            conversion_rate = ppv.get("conversion_rate", 0)
            if conversion_rate < 15:
                recommendations.append(
                    {
                        "priority": "MEDIUM",
                        "category": "PPV_CONVERSION",
                        "message": f"PPV conversion rate is {conversion_rate}%, below optimal 15-25% range.",
                        "action": "Review offer pricing, timing, and descriptions. Consider A/B testing different offer types.",
                    }
                )
            elif conversion_rate > 30:
                recommendations.append(
                    {
                        "priority": "INFO",
                        "category": "PPV_SUCCESS",
                        "message": f"Excellent PPV conversion rate of {conversion_rate}%! Revenue optimization is working well.",
                        "action": "Maintain current strategy and consider expanding offer variety.",
                    }
                )

            # Check for offer type optimization
            top_types = ppv.get("top_converting_types", [])
            if top_types and len(top_types) > 1:
                best_type, best_rate = top_types[0]
                worst_type, worst_rate = top_types[-1]
                if best_rate > worst_rate + 20:
                    recommendations.append(
                        {
                            "priority": "LOW",
                            "category": "PPV_OPTIMIZATION",
                            "message": f"'{best_type}' converts at {best_rate}% while '{worst_type}' only converts at {worst_rate}%.",
                            "action": f"Focus more on {best_type} offers and reduce or improve {worst_type} offers.",
                        }
                    )

        # Churn recommendations
        churn = understanding.get("churn_indicators", {})
        if churn and not churn.get("error"):
            risk = churn.get("risk_assessment", {})
            risk_level = risk.get("level", "LOW")
            risk_factors = risk.get("factors", [])

            if risk_level == "HIGH":
                recommendations.append(
                    {
                        "priority": "HIGH",
                        "category": "USER_RETENTION",
                        "message": f"High churn risk detected. Factors: {'; '.join(risk_factors[:2])}",
                        "action": "Implement re-engagement campaigns. Review user experience and content freshness.",
                    }
                )
            elif risk_level == "MEDIUM":
                recommendations.append(
                    {
                        "priority": "MEDIUM",
                        "category": "USER_RETENTION",
                        "message": f"Moderate churn indicators. Factors: {'; '.join(risk_factors[:2])}",
                        "action": "Monitor engagement trends closely. Consider loyalty incentives for at-risk users.",
                    }
                )

            retention = churn.get("retention_metrics", {})
            retention_7d = retention.get("retention_7d", 100)
            if retention_7d < 60:
                recommendations.append(
                    {
                        "priority": "MEDIUM",
                        "category": "RETENTION",
                        "message": f"7-day retention is {retention_7d}%, indicating users may not be returning.",
                        "action": "Improve onboarding experience and create compelling daily content hooks.",
                    }
                )

        # Traffic funnel recommendations
        traffic = understanding.get("traffic_analysis", {})
        if traffic and not traffic.get("error"):
            funnel = traffic.get("content_funnel", {})
            publish_rate = funnel.get("publish_rate", 100)
            distribution_rate = funnel.get("distribution_rate", 100)

            if publish_rate < 70:
                recommendations.append(
                    {
                        "priority": "LOW",
                        "category": "CONTENT_FUNNEL",
                        "message": f"Only {publish_rate}% of created content is being published.",
                        "action": "Review content moderation process and quality thresholds.",
                    }
                )

            if distribution_rate < 50:
                recommendations.append(
                    {
                        "priority": "LOW",
                        "category": "DISTRIBUTION",
                        "message": f"Only {distribution_rate}% of published content reaches social media.",
                        "action": "Check social media integration settings and posting schedules.",
                    }
                )

        # If no issues found
        if not recommendations:
            recommendations.append(
                {
                    "priority": "INFO",
                    "category": "STATUS",
                    "message": "System is operating normally. All metrics within healthy ranges.",
                    "action": "Continue monitoring for any changes. Consider A/B testing to optimize further.",
                }
            )

        return recommendations

    async def recall_context(self, context_id: UUID) -> Optional[Dict[str, Any]]:
        """
        Recall detailed information about a specific ACD context.

        Args:
            context_id: UUID of the context to recall

        Returns:
            Detailed context information or None if not found
        """
        try:
            stmt = select(ACDContextModel).where(ACDContextModel.id == context_id)
            result = await self.db.execute(stmt)
            context = result.scalar_one_or_none()

            if not context:
                return None

            # Get related trace artifacts
            artifact_stmt = (
                select(ACDTraceArtifactModel)
                .where(ACDTraceArtifactModel.acd_context_id == context_id)
                .order_by(desc(ACDTraceArtifactModel.timestamp))
            )

            artifact_result = await self.db.execute(artifact_stmt)
            artifacts = artifact_result.scalars().all()

            return {
                "context": {
                    "id": str(context.id),
                    "phase": context.ai_phase,
                    "status": context.ai_status,
                    "state": context.ai_state,
                    "domain": context.ai_domain,
                    "subdomain": context.ai_subdomain,
                    "complexity": context.ai_complexity,
                    "confidence": context.ai_confidence,
                    "note": context.ai_note,
                    "assigned_to": context.ai_assigned_to,
                    "metadata": context.ai_metadata,
                    "context_data": context.ai_context,
                    "created_at": (
                        context.created_at.isoformat() if context.created_at else None
                    ),
                    "updated_at": (
                        context.updated_at.isoformat() if context.updated_at else None
                    ),
                },
                "trace_artifacts": [
                    {
                        "id": str(a.id),
                        "event_type": a.event_type,
                        "error_message": a.error_message,
                        "error_code": a.error_code,
                        "timestamp": a.timestamp.isoformat() if a.timestamp else None,
                    }
                    for a in artifacts
                ],
            }

        except Exception as e:
            logger.error(f"Error recalling context {context_id}: {str(e)}")
            return None

    async def search_contexts(
        self, query: str, limit: int = 20, filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for ACD contexts based on query and filters.

        Args:
            query: Search query (searches in phase, note, and status)
            limit: Maximum number of results
            filters: Optional filters (phase, state, domain, etc.)

        Returns:
            List of matching contexts
        """
        try:
            stmt = select(ACDContextModel)

            # Apply filters
            conditions = []

            if query:
                # Search in multiple fields
                query_pattern = f"%{query}%"
                conditions.append(
                    ACDContextModel.ai_phase.ilike(query_pattern)
                    | ACDContextModel.ai_note.ilike(query_pattern)
                    | ACDContextModel.ai_status.ilike(query_pattern)
                )

            if filters:
                if filters.get("phase"):
                    conditions.append(ACDContextModel.ai_phase == filters["phase"])
                if filters.get("state"):
                    conditions.append(ACDContextModel.ai_state == filters["state"])
                if filters.get("domain"):
                    conditions.append(ACDContextModel.ai_domain == filters["domain"])
                if filters.get("status"):
                    conditions.append(ACDContextModel.ai_status == filters["status"])

            if conditions:
                stmt = stmt.where(and_(*conditions))

            stmt = stmt.order_by(desc(ACDContextModel.created_at)).limit(limit)

            result = await self.db.execute(stmt)
            contexts = result.scalars().all()

            return [
                {
                    "id": str(ctx.id),
                    "phase": ctx.ai_phase,
                    "status": ctx.ai_status,
                    "state": ctx.ai_state,
                    "domain": ctx.ai_domain,
                    "note": ctx.ai_note,
                    "confidence": ctx.ai_confidence,
                    "created_at": (
                        ctx.created_at.isoformat() if ctx.created_at else None
                    ),
                }
                for ctx in contexts
            ]

        except Exception as e:
            logger.error(f"Error searching contexts: {str(e)}")
            return []

    async def get_explanation(self, topic: str) -> str:
        """
        Get an explanation of a specific ACD concept or system component.

        Args:
            topic: The topic to explain

        Returns:
            Explanation string
        """
        explanations = {
            "acd": """ACD (Autonomous Continuous Development) is the system's self-aware
metadata framework. It tracks every operation, including content generation,
error handling, and agent coordination. Each ACD context represents a unit of
work with its phase, state, confidence level, and outcomes.""",
            "phase": """AI Phase represents the type of operation being performed.
Common phases include: IMAGE_GENERATION, TEXT_GENERATION, VIDEO_GENERATION,
AUDIO_GENERATION, and SYSTEM_OPERATIONS. Each phase has specific requirements
and expected outcomes.""",
            "state": """AI State tracks the current processing status of a context.
States include: PROCESSING (actively working), READY (queued for work),
DONE (completed successfully), FAILED (encountered error), BLOCKED (waiting
for external input), PAUSED (temporarily suspended), and CANCELLED.""",
            "confidence": """AI Confidence indicates how certain the system is about
its actions or outputs. Levels include: CONFIDENT (high certainty),
UNCERTAIN (low certainty), HYPOTHESIS (speculative), VALIDATED (confirmed
by review), and EXPERIMENTAL (testing new approaches).""",
            "domain": """AI Domain is a top-level classification that separates
different types of work. Domains include CODE_GENERATION, TEXT_GENERATION,
IMAGE_GENERATION, VIDEO_GENERATION, AUDIO_GENERATION, MULTIMODAL_SEMANTICS,
SYSTEM_OPERATIONS, HUMAN_INTERFACE, PLANNING, and ANALYSIS.""",
            "trace_artifact": """Trace Artifacts capture diagnostic information
about errors and issues. Each artifact includes the error message, code,
file location, stack trace, and related ACD context. This enables
comprehensive error analysis and debugging.""",
            "gator_agent": """The Gator Agent is the AI assistant interface that
helps users understand and interact with the system. It has access to
ACD understanding and recall capabilities, allowing it to explain
system behavior, diagnose issues, and provide recommendations.""",
            "system_understanding": """System Understanding is the ability to
comprehensively analyze the current state of all system components,
including ACD contexts, content generation status, persona configurations,
and error patterns. This enables intelligent recommendations.""",
        }

        topic_lower = topic.lower().replace(" ", "_").replace("-", "_")

        if topic_lower in explanations:
            return explanations[topic_lower]

        # Try partial matching
        for key, explanation in explanations.items():
            if topic_lower in key or key in topic_lower:
                return explanation

        return f"No explanation available for '{topic}'. Available topics: {', '.join(explanations.keys())}"

    async def get_domain_summary(self, domain: str) -> Dict[str, Any]:
        """
        Get a summary of activities for a specific AI domain.

        Args:
            domain: The AI domain to analyze

        Returns:
            Summary of domain activities
        """
        try:
            cutoff_time = datetime.now(timezone.utc) - timedelta(hours=24)

            stmt = (
                select(ACDContextModel)
                .where(
                    and_(
                        ACDContextModel.ai_domain == domain,
                        ACDContextModel.created_at >= cutoff_time,
                    )
                )
                .order_by(desc(ACDContextModel.created_at))
            )

            result = await self.db.execute(stmt)
            contexts = result.scalars().all()

            # Analyze by state
            by_state = {}
            by_subdomain = {}

            for ctx in contexts:
                state = ctx.ai_state or "UNKNOWN"
                by_state[state] = by_state.get(state, 0) + 1

                subdomain = ctx.ai_subdomain or "UNCLASSIFIED"
                by_subdomain[subdomain] = by_subdomain.get(subdomain, 0) + 1

            return {
                "domain": domain,
                "total_contexts": len(contexts),
                "by_state": by_state,
                "by_subdomain": by_subdomain,
                "recent_contexts": [
                    {
                        "id": str(ctx.id),
                        "phase": ctx.ai_phase,
                        "state": ctx.ai_state,
                        "subdomain": ctx.ai_subdomain,
                        "note": ctx.ai_note,
                        "created_at": (
                            ctx.created_at.isoformat() if ctx.created_at else None
                        ),
                    }
                    for ctx in contexts[:5]
                ],
            }

        except Exception as e:
            logger.error(f"Error getting domain summary for {domain}: {str(e)}")
            return {"error": str(e)}

    # ============================================================
    # Scheduling Context Methods - For LLM-driven scheduler
    # ============================================================

    async def get_scheduling_context(self, hours: int = 24) -> Dict[str, Any]:
        """
        Get comprehensive scheduling context for LLM reasoning.

        This enables the LLM to make intelligent scheduling decisions based on:
        - Current queue state and priorities
        - Historical performance patterns
        - Resource availability
        - Optimal timing patterns learned from past operations

        Args:
            hours: Time window for historical analysis

        Returns:
            Comprehensive scheduling context for LLM reasoning
        """
        try:
            cutoff_time = datetime.now(timezone.utc) - timedelta(hours=hours)
            now = datetime.now(timezone.utc)

            # Get all contexts for analysis
            stmt = (
                select(ACDContextModel)
                .where(ACDContextModel.created_at >= cutoff_time)
                .order_by(desc(ACDContextModel.created_at))
            )

            result = await self.db.execute(stmt)
            contexts = result.scalars().all()

            # Analyze queue state
            queue_state = await self._analyze_queue_state(contexts)

            # Analyze timing patterns
            timing_patterns = await self._analyze_timing_patterns(contexts, hours)

            # Analyze resource usage patterns
            resource_patterns = await self._analyze_resource_patterns(contexts)

            # Get pending scheduled items
            pending_scheduled = await self._get_pending_scheduled(now)

            # Get orchestration groups
            orchestration_groups = await self._get_orchestration_groups(contexts)

            # Calculate scheduling recommendations
            recommendations = self._generate_scheduling_recommendations(
                queue_state, timing_patterns, resource_patterns, pending_scheduled
            )

            return {
                "timestamp": now.isoformat(),
                "analysis_window_hours": hours,
                "queue_state": queue_state,
                "timing_patterns": timing_patterns,
                "resource_patterns": resource_patterns,
                "pending_scheduled": pending_scheduled,
                "orchestration_groups": orchestration_groups,
                "scheduling_recommendations": recommendations,
            }

        except Exception as e:
            logger.error(f"Error getting scheduling context: {str(e)}")
            return {"error": str(e)}

    async def _analyze_queue_state(self, contexts: list) -> Dict[str, Any]:
        """Analyze current queue state for scheduling decisions."""
        queued = [c for c in contexts if c.ai_queue_status == "QUEUED"]
        in_progress = [c for c in contexts if c.ai_queue_status == "IN_PROGRESS"]

        # Analyze by priority
        by_priority = {}
        for ctx in queued:
            priority = ctx.ai_queue_priority or "NORMAL"
            by_priority[priority] = by_priority.get(priority, 0) + 1

        # Analyze by domain
        by_domain = {}
        for ctx in queued:
            domain = ctx.ai_domain or "UNCLASSIFIED"
            by_domain[domain] = by_domain.get(domain, 0) + 1

        # Calculate wait times
        wait_times = []
        now = datetime.now(timezone.utc)
        for ctx in queued:
            if ctx.created_at:
                wait_time = (now - ctx.created_at).total_seconds() / 60  # minutes
                wait_times.append(wait_time)

        avg_wait = sum(wait_times) / len(wait_times) if wait_times else 0
        max_wait = max(wait_times) if wait_times else 0

        return {
            "total_queued": len(queued),
            "total_in_progress": len(in_progress),
            "by_priority": by_priority,
            "by_domain": by_domain,
            "avg_wait_time_minutes": round(avg_wait, 1),
            "max_wait_time_minutes": round(max_wait, 1),
            "queue_depth": len(queued) + len(in_progress),
        }

    async def _analyze_timing_patterns(
        self, contexts: list, hours: int
    ) -> Dict[str, Any]:
        """Analyze timing patterns for optimal scheduling."""
        # Group completions by hour
        completions_by_hour = {}
        successes_by_hour = {}

        for ctx in contexts:
            if ctx.ai_state == "DONE" and ctx.updated_at:
                hour = ctx.updated_at.hour
                completions_by_hour[hour] = completions_by_hour.get(hour, 0) + 1
                successes_by_hour[hour] = successes_by_hour.get(hour, 0) + 1
            elif ctx.ai_state == "FAILED" and ctx.updated_at:
                hour = ctx.updated_at.hour
                completions_by_hour[hour] = completions_by_hour.get(hour, 0) + 1

        # Calculate success rate by hour
        success_rate_by_hour = {}
        for hour in completions_by_hour:
            total = completions_by_hour[hour]
            successes = successes_by_hour.get(hour, 0)
            success_rate_by_hour[hour] = (
                round(successes / total * 100, 1) if total > 0 else 0
            )

        # Find optimal hours (highest success rate with sufficient volume)
        optimal_hours = []
        for hour, rate in sorted(
            success_rate_by_hour.items(), key=lambda x: x[1], reverse=True
        ):
            if completions_by_hour.get(hour, 0) >= 3:  # Minimum volume threshold
                optimal_hours.append({"hour": hour, "success_rate": rate})

        # Analyze average completion time by phase
        completion_times_by_phase = {}
        for ctx in contexts:
            if ctx.ai_state == "DONE" and ctx.ai_started and ctx.updated_at:
                phase = ctx.ai_phase or "UNKNOWN"
                duration = (ctx.updated_at - ctx.ai_started).total_seconds()
                if phase not in completion_times_by_phase:
                    completion_times_by_phase[phase] = []
                completion_times_by_phase[phase].append(duration)

        avg_times_by_phase = {
            phase: round(sum(times) / len(times), 1)
            for phase, times in completion_times_by_phase.items()
        }

        return {
            "completions_by_hour": completions_by_hour,
            "success_rate_by_hour": success_rate_by_hour,
            "optimal_hours": optimal_hours[:5],  # Top 5 optimal hours
            "avg_completion_time_by_phase": avg_times_by_phase,
        }

    async def _analyze_resource_patterns(self, contexts: list) -> Dict[str, Any]:
        """Analyze resource usage patterns for capacity planning."""
        # Analyze resource usage from contexts that have it recorded
        resource_data = []
        for ctx in contexts:
            if ctx.actual_resources:
                resource_data.append(ctx.actual_resources)

        # Calculate averages
        if not resource_data:
            return {
                "message": "No resource data available yet",
                "avg_cpu": None,
                "avg_memory": None,
                "avg_gpu": None,
            }

        # Aggregate resource metrics
        avg_cpu = sum(r.get("cpu", 0) for r in resource_data) / len(resource_data)
        avg_memory = sum(r.get("memory", 0) for r in resource_data) / len(resource_data)
        avg_gpu = sum(r.get("gpu", 0) for r in resource_data) / len(resource_data)

        return {
            "samples": len(resource_data),
            "avg_cpu_percent": round(avg_cpu, 1),
            "avg_memory_mb": round(avg_memory, 1),
            "avg_gpu_percent": round(avg_gpu, 1),
        }

    async def _get_pending_scheduled(self, now: datetime) -> Dict[str, Any]:
        """Get pending scheduled operations."""
        try:
            # Get items scheduled for future execution
            stmt = (
                select(ACDContextModel)
                .where(
                    and_(
                        ACDContextModel.scheduled_for is not None,
                        ACDContextModel.scheduled_for > now,
                        ACDContextModel.ai_state.in_(["READY", "QUEUED"]),
                    )
                )
                .order_by(ACDContextModel.scheduled_for)
            )

            result = await self.db.execute(stmt)
            scheduled = result.scalars().all()

            # Group by schedule type
            by_type = {}
            for ctx in scheduled:
                stype = ctx.schedule_type or "SCHEDULED"
                by_type[stype] = by_type.get(stype, 0) + 1

            return {
                "total_pending": len(scheduled),
                "by_schedule_type": by_type,
                "next_scheduled": [
                    {
                        "id": str(ctx.id),
                        "phase": ctx.ai_phase,
                        "scheduled_for": (
                            ctx.scheduled_for.isoformat() if ctx.scheduled_for else None
                        ),
                        "schedule_type": ctx.schedule_type,
                        "optimization_goal": ctx.schedule_optimization_goal,
                    }
                    for ctx in scheduled[:10]
                ],
            }

        except Exception as e:
            logger.error(f"Error getting pending scheduled: {str(e)}")
            return {"error": str(e)}

    async def _get_orchestration_groups(self, contexts: list) -> Dict[str, Any]:
        """Get orchestration group information."""
        groups = {}
        for ctx in contexts:
            if ctx.orchestration_group:
                group = ctx.orchestration_group
                if group not in groups:
                    groups[group] = {
                        "total": 0,
                        "completed": 0,
                        "in_progress": 0,
                        "queued": 0,
                    }
                groups[group]["total"] += 1
                if ctx.ai_state == "DONE":
                    groups[group]["completed"] += 1
                elif ctx.ai_state == "PROCESSING":
                    groups[group]["in_progress"] += 1
                elif ctx.ai_queue_status == "QUEUED":
                    groups[group]["queued"] += 1

        return {
            "total_groups": len(groups),
            "groups": groups,
        }

    def _generate_scheduling_recommendations(
        self,
        queue_state: Dict[str, Any],
        timing_patterns: Dict[str, Any],
        resource_patterns: Dict[str, Any],
        pending_scheduled: Dict[str, Any],
    ) -> List[Dict[str, str]]:
        """Generate intelligent scheduling recommendations for LLM."""
        recommendations = []

        # Queue depth recommendations
        queue_depth = queue_state.get("queue_depth", 0)
        if queue_depth > 50:
            recommendations.append(
                {
                    "priority": "HIGH",
                    "category": "QUEUE_MANAGEMENT",
                    "insight": f"Queue depth is {queue_depth}. Consider increasing parallelism or deferring low-priority items.",
                    "action": "Increase worker threads or reschedule DEFERRED priority items.",
                }
            )

        # Wait time recommendations
        max_wait = queue_state.get("max_wait_time_minutes", 0)
        if max_wait > 30:
            recommendations.append(
                {
                    "priority": "MEDIUM",
                    "category": "WAIT_TIME",
                    "insight": f"Maximum wait time is {max_wait:.0f} minutes. Some items may be stuck.",
                    "action": "Review stuck items and consider priority escalation.",
                }
            )

        # Timing optimization
        optimal_hours = timing_patterns.get("optimal_hours", [])
        if optimal_hours:
            best_hour = optimal_hours[0]
            recommendations.append(
                {
                    "priority": "INFO",
                    "category": "TIMING_OPTIMIZATION",
                    "insight": f"Hour {best_hour['hour']}:00 has {best_hour['success_rate']}% success rate - optimal for scheduling.",
                    "action": f"Schedule high-priority operations around {best_hour['hour']}:00 UTC.",
                }
            )

        # Pending scheduled items
        pending = pending_scheduled.get("total_pending", 0)
        if pending > 100:
            recommendations.append(
                {
                    "priority": "MEDIUM",
                    "category": "SCHEDULE_BACKLOG",
                    "insight": f"{pending} items pending in schedule. May indicate scheduling issues.",
                    "action": "Review scheduled items and consolidate where possible.",
                }
            )

        return recommendations

    async def record_schedule_feedback(
        self,
        context_id: UUID,
        feedback_type: str,
        feedback_data: Dict[str, Any],
        effectiveness_score: float,
    ) -> bool:
        """
        Record feedback for a scheduled operation to enable learning.

        This allows the LLM scheduler to learn from outcomes and improve
        future scheduling decisions.

        Args:
            context_id: The ACD context ID
            feedback_type: Type of feedback (from ScheduleFeedbackType enum)
            feedback_data: Detailed feedback data
            effectiveness_score: 0-100 score of how effective the scheduling was

        Returns:
            True if feedback was recorded successfully
        """
        try:
            stmt = select(ACDContextModel).where(ACDContextModel.id == context_id)
            result = await self.db.execute(stmt)
            context = result.scalar_one_or_none()

            if not context:
                return False

            context.schedule_feedback_type = feedback_type
            context.schedule_feedback_data = feedback_data
            context.schedule_effectiveness_score = effectiveness_score

            await self.db.commit()

            logger.info(
                f"Recorded schedule feedback for context {context_id}: "
                f"type={feedback_type}, score={effectiveness_score}"
            )

            return True

        except Exception as e:
            logger.error(f"Error recording schedule feedback: {str(e)}")
            await self.db.rollback()
            return False

    async def get_scheduling_learning_data(
        self, domain: Optional[str] = None, limit: int = 100
    ) -> Dict[str, Any]:
        """
        Get historical scheduling data for LLM learning.

        This provides the LLM with past scheduling decisions and their outcomes
        to enable continuous improvement of scheduling strategies.

        Args:
            domain: Optional domain filter
            limit: Maximum number of records

        Returns:
            Historical scheduling data with outcomes
        """
        try:
            stmt = select(ACDContextModel).where(
                ACDContextModel.schedule_feedback_data is not None
            )

            if domain:
                stmt = stmt.where(ACDContextModel.ai_domain == domain)

            stmt = stmt.order_by(desc(ACDContextModel.updated_at)).limit(limit)

            result = await self.db.execute(stmt)
            contexts = result.scalars().all()

            learning_data = []
            for ctx in contexts:
                learning_data.append(
                    {
                        "id": str(ctx.id),
                        "domain": ctx.ai_domain,
                        "phase": ctx.ai_phase,
                        "schedule_type": ctx.schedule_type,
                        "optimization_goal": ctx.schedule_optimization_goal,
                        "scheduled_for": (
                            ctx.scheduled_for.isoformat() if ctx.scheduled_for else None
                        ),
                        "actual_start": (
                            ctx.ai_started.isoformat() if ctx.ai_started else None
                        ),
                        "decision_source": ctx.schedule_decision_source,
                        "reasoning": ctx.schedule_reasoning,
                        "feedback_type": ctx.schedule_feedback_type,
                        "feedback_data": ctx.schedule_feedback_data,
                        "effectiveness_score": ctx.schedule_effectiveness_score,
                        "final_state": ctx.ai_state,
                    }
                )

            # Calculate aggregate metrics
            scores = [
                d["effectiveness_score"]
                for d in learning_data
                if d["effectiveness_score"]
            ]
            avg_effectiveness = sum(scores) / len(scores) if scores else 0

            return {
                "total_records": len(learning_data),
                "avg_effectiveness_score": round(avg_effectiveness, 1),
                "learning_data": learning_data,
            }

        except Exception as e:
            logger.error(f"Error getting scheduling learning data: {str(e)}")
            return {"error": str(e)}

    # ============================================================
    # Intelligent Task Recommendation and Agent Routing
    # ============================================================

    async def recommend_next_task(
        self, optimization_goal: str = "BALANCED", persona_id: Optional[UUID] = None
    ) -> Dict[str, Any]:
        """
        Use system understanding to recommend the next optimal task.

        This enables LLM-driven scheduling by providing:
        - Which persona should act next
        - What content type is optimal
        - Which platform to target
        - When to execute

        Args:
            optimization_goal: What to optimize for (MAX_ENGAGEMENT, MAX_REACH, etc.)
            persona_id: Optional specific persona to recommend for

        Returns:
            Task recommendation with reasoning
        """
        try:
            # Get current system state
            understanding = await self.get_system_understanding(hours=24)
            scheduling_context = await self.get_scheduling_context(hours=24)

            # Analyze queue state
            queue_state = scheduling_context.get("queue_state", {})
            queue_depth = queue_state.get("queue_depth", 0)

            # Analyze timing patterns
            timing_patterns = scheduling_context.get("timing_patterns", {})
            optimal_hours = timing_patterns.get("optimal_hours", [])

            # Get engagement metrics
            engagement = understanding.get("engagement_metrics", {})
            ppv_performance = understanding.get("ppv_performance", {})

            # Get persona performance
            persona_status = understanding.get("persona_status", {})
            personas = persona_status.get("personas", [])

            # Build recommendation
            recommendation = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "optimization_goal": optimization_goal,
                "queue_state": {
                    "depth": queue_depth,
                    "recommendation": "proceed" if queue_depth < 20 else "wait",
                },
            }

            # Recommend persona
            if persona_id:
                recommendation["persona"] = {
                    "id": str(persona_id),
                    "source": "specified",
                }
            elif personas:
                # Select persona with best recent performance or lowest activity
                active_personas = [p for p in personas if p.get("is_active")]
                if active_personas:
                    # Sort by post_count (lower = needs more activity)
                    sorted_personas = sorted(
                        active_personas, key=lambda p: p.get("post_count", 0)
                    )
                    selected = sorted_personas[0]
                    recommendation["persona"] = {
                        "id": selected["id"],
                        "name": selected["name"],
                        "source": "load_balanced",
                        "reasoning": f"Selected {selected['name']} - lowest recent activity ({selected.get('post_count', 0)} posts)",
                    }

            # Recommend content type based on optimization goal
            content_type_map = {
                "MAX_ENGAGEMENT": "proactive_topic",  # Topics generate discussion
                "MAX_REACH": "image",  # Images spread fastest
                "MAX_CONVERSION": "ppv_offer",  # Direct revenue
                "MIN_COST": "text",  # Lowest resource usage
                "BALANCED": "proactive_topic",
            }

            recommendation["content_type"] = {
                "type": content_type_map.get(optimization_goal, "proactive_topic"),
                "reasoning": f"Optimal for {optimization_goal} goal",
            }

            # Recommend platform
            by_platform = engagement.get("by_platform", {})
            if by_platform:
                # Find platform with best engagement
                best_platform = max(
                    by_platform.items(),
                    key=lambda x: x[1].get("likes", 0) + x[1].get("comments", 0),
                )
                recommendation["platform"] = {
                    "platform": best_platform[0],
                    "reasoning": f"Best engagement on {best_platform[0]}",
                }
            else:
                recommendation["platform"] = {
                    "platform": "instagram",
                    "reasoning": "Default platform (no historical data)",
                }

            # Recommend timing
            now = datetime.now(timezone.utc)
            if optimal_hours:
                best_hour = optimal_hours[0]["hour"]
                current_hour = now.hour

                if abs(best_hour - current_hour) <= 2:
                    recommendation["timing"] = {
                        "execute": "now",
                        "reasoning": f"Current time is near optimal hour ({best_hour}:00)",
                    }
                else:
                    recommendation["timing"] = {
                        "execute": "scheduled",
                        "scheduled_hour": best_hour,
                        "reasoning": f"Schedule for {best_hour}:00 UTC for {optimal_hours[0]['success_rate']}% success rate",
                    }
            else:
                recommendation["timing"] = {
                    "execute": "now",
                    "reasoning": "No historical timing data - execute immediately",
                }

            # Add PPV opportunity if conversion rate is high
            ppv_conversion = ppv_performance.get("conversion_rate", 0)
            if ppv_conversion > 20:
                recommendation["ppv_opportunity"] = {
                    "suggested": True,
                    "conversion_rate": ppv_conversion,
                    "reasoning": f"High PPV conversion rate ({ppv_conversion}%) - consider upsell opportunity",
                }

            # Overall confidence
            recommendation["confidence"] = {
                "score": 0.8 if optimal_hours and personas else 0.5,
                "factors": {
                    "has_timing_data": bool(optimal_hours),
                    "has_engagement_data": bool(by_platform),
                    "has_persona_data": bool(personas),
                    "queue_clear": queue_depth < 20,
                },
            }

            return recommendation

        except Exception as e:
            logger.error(f"Error recommending next task: {str(e)}")
            return {"error": str(e)}

    async def route_to_agent(
        self, task: Dict[str, Any], required_capabilities: List[str]
    ) -> Dict[str, Any]:
        """
        Intelligent agent routing based on task requirements.

        Routes tasks to the optimal agent based on:
        - Agent capabilities
        - Current load
        - Past performance
        - Task complexity

        Args:
            task: Task details including type, content, requirements
            required_capabilities: List of required agent capabilities

        Returns:
            Routing decision with agent assignment and reasoning
        """
        try:
            # Get current agent states from ACD contexts
            stmt = select(ACDContextModel).where(
                and_(
                    ACDContextModel.ai_state == "PROCESSING",
                    ACDContextModel.ai_assigned_agent_id is not None,
                )
            )
            result = await self.db.execute(stmt)
            active_contexts = result.scalars().all()

            # Analyze agent load
            agent_load = {}
            for ctx in active_contexts:
                agent_id = str(ctx.ai_assigned_agent_id)
                agent_load[agent_id] = agent_load.get(agent_id, 0) + 1

            # Find historical performance by agent
            perf_stmt = (
                select(ACDContextModel)
                .where(
                    and_(
                        ACDContextModel.ai_state == "DONE",
                        ACDContextModel.ai_assigned_agent_id is not None,
                    )
                )
                .order_by(desc(ACDContextModel.updated_at))
                .limit(100)
            )

            perf_result = await self.db.execute(perf_stmt)
            completed_contexts = perf_result.scalars().all()

            # Calculate agent success rates
            agent_performance = {}
            for ctx in completed_contexts:
                agent_id = str(ctx.ai_assigned_agent_id)
                if agent_id not in agent_performance:
                    agent_performance[agent_id] = {"total": 0, "success": 0}
                agent_performance[agent_id]["total"] += 1
                # Success if confidence is HIGH or status is COMPLETED
                if ctx.ai_confidence == "HIGH" or ctx.ai_status == "COMPLETED":
                    agent_performance[agent_id]["success"] += 1

            # Calculate success rates
            for agent_id in agent_performance:
                perf = agent_performance[agent_id]
                perf["success_rate"] = (
                    perf["success"] / perf["total"] if perf["total"] > 0 else 0
                )

            # Build routing decision
            routing = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "task": {
                    "type": task.get("type"),
                    "domain": task.get("domain"),
                    "complexity": task.get("complexity", "MEDIUM"),
                },
                "required_capabilities": required_capabilities,
                "agent_analysis": {
                    "total_active_agents": len(set(agent_load.keys())),
                    "total_active_tasks": sum(agent_load.values()),
                },
            }

            # Select best agent
            if agent_performance:
                # Score agents: high success rate, low current load
                agent_scores = []
                for agent_id, perf in agent_performance.items():
                    current_load = agent_load.get(agent_id, 0)
                    score = perf["success_rate"] * 0.7 - (current_load * 0.1)
                    agent_scores.append(
                        {
                            "agent_id": agent_id,
                            "score": score,
                            "success_rate": perf["success_rate"],
                            "current_load": current_load,
                        }
                    )

                # Sort by score descending
                agent_scores.sort(key=lambda x: x["score"], reverse=True)

                if agent_scores:
                    best = agent_scores[0]
                    routing["assignment"] = {
                        "agent_id": best["agent_id"],
                        "confidence": "HIGH" if best["score"] > 0.5 else "MEDIUM",
                        "reasoning": f"Best scoring agent (score: {best['score']:.2f}, success: {best['success_rate']:.0%}, load: {best['current_load']})",
                    }
                    routing["alternatives"] = agent_scores[1:3]  # Top 3 alternatives
                else:
                    routing["assignment"] = {
                        "agent_id": None,
                        "confidence": "LOW",
                        "reasoning": "No agent performance data available - assign to any available agent",
                    }
            else:
                routing["assignment"] = {
                    "agent_id": None,
                    "confidence": "LOW",
                    "reasoning": "No historical agent data - use round-robin assignment",
                }

            return routing

        except Exception as e:
            logger.error(f"Error routing to agent: {str(e)}")
            return {"error": str(e)}

    async def analyze_system_state(self) -> Dict[str, Any]:
        """
        Get comprehensive system state analysis for LLM reasoning.

        Returns insights on:
        - Bottlenecks (what's blocked?)
        - Resource utilization (queue depth, processing load)
        - Performance trends (what's working?)
        - Recommended actions (what should we do next?)

        This is the "brain" method that connects all system data.
        """
        try:
            # Get all component analyses
            understanding = await self.get_system_understanding(hours=24)
            scheduling_context = await self.get_scheduling_context(hours=24)

            # Build comprehensive state analysis
            analysis = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "overall_health": understanding.get("system_state", {}).get(
                    "health_status", "UNKNOWN"
                ),
            }

            # Bottleneck analysis
            bottlenecks = []

            queue_state = scheduling_context.get("queue_state", {})
            if queue_state.get("queue_depth", 0) > 50:
                bottlenecks.append(
                    {
                        "type": "QUEUE_BACKUP",
                        "severity": "HIGH",
                        "description": f"Queue depth is {queue_state['queue_depth']} - processing is falling behind",
                        "suggestion": "Increase parallelism or defer low-priority tasks",
                    }
                )

            if queue_state.get("max_wait_time_minutes", 0) > 60:
                bottlenecks.append(
                    {
                        "type": "STUCK_TASKS",
                        "severity": "HIGH",
                        "description": f"Max wait time is {queue_state['max_wait_time_minutes']}min - some tasks may be stuck",
                        "suggestion": "Review stuck tasks and check for errors",
                    }
                )

            error_analysis = understanding.get("error_analysis", {})
            if error_analysis.get("total_errors", 0) > 10:
                bottlenecks.append(
                    {
                        "type": "ERROR_RATE",
                        "severity": "MEDIUM",
                        "description": f"{error_analysis['total_errors']} errors in last 24h",
                        "suggestion": "Review error patterns and fix recurring issues",
                    }
                )

            analysis["bottlenecks"] = bottlenecks

            # Resource utilization
            analysis["resource_utilization"] = {
                "queue_depth": queue_state.get("queue_depth", 0),
                "processing_tasks": queue_state.get("total_in_progress", 0),
                "pending_scheduled": scheduling_context.get(
                    "pending_scheduled", {}
                ).get("total_pending", 0),
                "status": (
                    "HEALTHY"
                    if queue_state.get("queue_depth", 0) < 30
                    else "OVERLOADED"
                ),
            }

            # Performance trends
            content_activity = understanding.get("content_activity", {})
            engagement = understanding.get("engagement_metrics", {})
            ppv = understanding.get("ppv_performance", {})

            analysis["performance_trends"] = {
                "content_success_rate": content_activity.get("success_rate", 0),
                "avg_engagement_rate": engagement.get("avg_engagement_rate", 0),
                "ppv_conversion_rate": ppv.get("conversion_rate", 0),
                "trend": (
                    "IMPROVING"
                    if content_activity.get("success_rate", 0) > 80
                    else "NEEDS_ATTENTION"
                ),
            }

            # Recommended actions (prioritized)
            actions = []

            if bottlenecks:
                for b in bottlenecks:
                    actions.append(
                        {
                            "priority": 1 if b["severity"] == "HIGH" else 2,
                            "action": b["suggestion"],
                            "reason": b["description"],
                        }
                    )

            # Add proactive recommendations
            churn = understanding.get("churn_indicators", {})
            if churn.get("risk_assessment", {}).get("level") in ["HIGH", "CRITICAL"]:
                actions.append(
                    {
                        "priority": 1,
                        "action": "Implement re-engagement campaign",
                        "reason": f"Churn risk is {churn['risk_assessment']['level']}",
                    }
                )

            if ppv.get("conversion_rate", 0) < 15:
                actions.append(
                    {
                        "priority": 2,
                        "action": "Optimize PPV offers - review pricing and timing",
                        "reason": f"PPV conversion rate is only {ppv.get('conversion_rate', 0)}%",
                    }
                )

            # Sort by priority
            actions.sort(key=lambda x: x["priority"])
            analysis["recommended_actions"] = actions[:5]  # Top 5

            # Next task recommendation
            analysis["next_task_recommendation"] = await self.recommend_next_task()

            return analysis

        except Exception as e:
            logger.error(f"Error analyzing system state: {str(e)}")
            return {"error": str(e)}
