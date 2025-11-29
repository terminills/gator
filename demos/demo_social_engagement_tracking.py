#!/usr/bin/env python3
"""
Demo: Social Media Engagement Tracking with ACD Integration

Demonstrates how social media engagement metrics are tracked, filtered,
and integrated with ACD for continuous learning.
"""

import asyncio
from datetime import datetime, timedelta, timezone
from uuid import uuid4

from backend.database.connection import get_async_session
from backend.services.social_engagement_service import SocialEngagementService
from backend.models.social_media_post import (
    SocialMediaPostCreate,
    EngagementMetrics,
    SocialPlatform,
)
from backend.models.acd import ACDContextCreate, AIStatus, AIState, AIComplexity
from backend.services.acd_service import ACDService


async def demo_social_engagement_tracking():
    """Demo social media engagement tracking and learning integration."""

    print("\n" + "=" * 80)
    print("DEMO: Social Media Engagement Tracking with ACD Integration")
    print("=" * 80)

    async with get_async_session() as session:
        engagement_service = SocialEngagementService(session)
        acd_service = ACDService(session)

        # Demo 1: Create ACD context for content generation
        print("\n" + "-" * 80)
        print("DEMO 1: Content Generation with ACD Context")
        print("-" * 80)

        acd_context = await acd_service.create_context(
            ACDContextCreate(
                ai_phase="SOCIAL_MEDIA_CONTENT",
                ai_status=AIStatus.IMPLEMENTED,
                ai_complexity=AIComplexity.MEDIUM,
                ai_note="Creating Instagram post with lifestyle theme",
                ai_state=AIState.DONE,
                ai_context={
                    "platform": "instagram",
                    "content_theme": "lifestyle",
                    "target_audience": "millennials",
                },
            )
        )
        print(f"\n✓ Created ACD context: {acd_context.id}")
        print(f"  Phase: {acd_context.ai_phase}")
        print(f"  Complexity: {acd_context.ai_complexity}")

        # Demo 2: Create social media post record
        print("\n" + "-" * 80)
        print("DEMO 2: Track Published Social Media Post")
        print("-" * 80)

        # Simulate a persona and content
        persona_id = uuid4()
        content_id = uuid4()

        post_data = SocialMediaPostCreate(
            content_id=content_id,
            persona_id=persona_id,
            platform=SocialPlatform.INSTAGRAM,
            caption="Living my best life! 🌟 #lifestyle #motivation #positivevibes",
            hashtags=["lifestyle", "motivation", "positivevibes"],
            acd_context_id=acd_context.id,
        )

        post = await engagement_service.create_post_record(post_data)
        print(f"\n✓ Created post record: {post.id}")
        print(f"  Platform: {post.platform}")
        print(f"  Caption: {post.caption}")
        print(f"  Hashtags: {', '.join(post.hashtags or [])}")
        print(f"  Linked to ACD context: {post.acd_context_id}")

        # Demo 3: Simulate engagement metrics with bot filtering
        print("\n" + "-" * 80)
        print("DEMO 3: Update Engagement Metrics (with Bot Filtering)")
        print("-" * 80)

        # Simulate raw metrics from platform
        print("\n📊 Raw metrics from platform:")
        raw_metrics = {
            "likes": 1250,
            "comments": 87,
            "shares": 43,
            "saves": 156,
            "impressions": 15000,
            "reach": 8500,
        }
        for metric, value in raw_metrics.items():
            print(f"  {metric}: {value}")

        # Apply filtering
        print("\n🔍 After filtering bots and AI personas:")
        filtered_metrics = EngagementMetrics(
            likes_count=1100,  # Filtered 150 bot likes
            comments_count=75,  # Filtered 12 bot comments
            shares_count=43,
            saves_count=156,
            impressions=15000,
            reach=8500,
            bot_interaction_count=162,  # Detected bots
            persona_interaction_count=5,  # Other AI personas
            genuine_user_count=1175,  # Real users
            engagement_timeline={
                "09:00": 45,
                "10:00": 120,
                "11:00": 180,
                "12:00": 210,
                "13:00": 165,
                "14:00": 95,
            },
        )

        updated_post = await engagement_service.update_post_metrics(
            post.id, filtered_metrics
        )

        print(f"  Likes: {updated_post.likes_count} (filtered {162 - updated_post.likes_count} bots)")
        print(f"  Comments: {updated_post.comments_count}")
        print(f"  Shares: {updated_post.shares_count}")
        print(f"  Saves: {updated_post.saves_count}")
        print(f"  Engagement Rate: {updated_post.engagement_rate:.2f}%")
        print(f"  Genuine Users: {updated_post.genuine_user_count}")
        print(f"  Bot Interactions Filtered: {updated_post.bot_interaction_count}")
        print(f"  AI Persona Interactions: {updated_post.persona_interaction_count}")

        # Demo 4: Check ACD context update
        print("\n" + "-" * 80)
        print("DEMO 4: ACD Context Updated with Engagement Data")
        print("-" * 80)

        updated_acd = await acd_service.get_context(acd_context.id)
        print(f"\n✓ ACD context automatically updated with engagement metrics")
        print(f"  Validation: {updated_acd.ai_validation}")
        print(f"  Confidence: {updated_acd.ai_confidence}")

        if updated_acd.ai_metadata and "social_metrics" in updated_acd.ai_metadata:
            social_metrics = updated_acd.ai_metadata["social_metrics"]
            print(f"\n  Social Metrics stored in ACD:")
            print(f"    Platform: {social_metrics.get('platform')}")
            print(f"    Engagement Rate: {social_metrics.get('engagement_rate'):.2f}%")
            print(f"    Genuine User Count: {social_metrics.get('genuine_user_count')}")
            print(f"    Bot Filtered: {social_metrics.get('bot_filtered_count')}")
            print(f"    Genuine Ratio: {social_metrics.get('genuine_ratio'):.2%}")

        if updated_acd.ai_pattern:
            print(f"\n  Pattern Extracted: {updated_acd.ai_pattern}")
            print(f"  Strategy: {updated_acd.ai_strategy}")

        # Demo 5: Analyze post performance
        print("\n" + "-" * 80)
        print("DEMO 5: Performance Analysis & Recommendations")
        print("-" * 80)

        analysis = await engagement_service.analyze_post_performance(post.id)
        print(f"\n📈 Post Performance Analysis:")
        print(f"  Total Engagement: {analysis.total_engagement}")
        print(f"  Genuine Engagement: {analysis.genuine_engagement}")
        print(f"  Engagement Rate: {analysis.engagement_rate:.2f}%")
        print(f"  vs Average: {analysis.performance_vs_average:+.1f}%")

        if analysis.top_performing_elements:
            print(f"\n  🌟 Top Performing Elements:")
            for element in analysis.top_performing_elements:
                print(f"    • {element}")

        if analysis.recommendations:
            print(f"\n  💡 AI Recommendations:")
            for rec in analysis.recommendations:
                print(f"    • {rec}")

        # Demo 6: Show learning feedback loop
        print("\n" + "-" * 80)
        print("DEMO 6: Learning Feedback Loop")
        print("-" * 80)

        print("\n🔄 Feedback Loop Components:")
        print("  1. Content Generated → Linked to ACD context")
        print("  2. Posted to Social Media → Tracked in database")
        print("  3. Real-time Metrics Collected → Bot filtering applied")
        print("  4. ACD Context Updated → Validation & confidence set")
        print("  5. Patterns Extracted → For future content generation")
        print("  6. Recommendations Generated → Improve next post")

        print("\n✓ Complete feedback loop operational!")
        print("  The system now learns from:")
        print("    • Which content gets genuine engagement")
        print("    • What time to post for best results")
        print("    • Which hashtags drive real user interaction")
        print("    • What caption styles resonate with audience")
        print("    • Which platforms work best for each persona")

        print("\n" + "=" * 80)
        print("Demo completed successfully!")
        print("=" * 80)


async def demo_multi_platform_tracking():
    """Demo tracking across multiple social platforms."""

    print("\n" + "=" * 80)
    print("BONUS DEMO: Multi-Platform Engagement Tracking")
    print("=" * 80)

    async with get_async_session() as session:
        engagement_service = SocialEngagementService(session)

        platforms = [
            (SocialPlatform.INSTAGRAM, 1200, 8.5),
            (SocialPlatform.FACEBOOK, 850, 6.2),
            (SocialPlatform.TWITTER, 450, 4.1),
            (SocialPlatform.TIKTOK, 3500, 12.3),
        ]

        print("\n📊 Engagement Across Platforms:")
        print("-" * 80)

        for platform, likes, engagement_rate in platforms:
            print(f"\n  {platform.value.upper()}")
            print(f"    Likes: {likes}")
            print(f"    Engagement Rate: {engagement_rate}%")
            print(f"    Status: {'✓ High Performance' if engagement_rate > 7.0 else '• Average Performance'}")

        print("\n💡 Platform Insights:")
        print("  • TikTok: Highest engagement - use for viral content")
        print("  • Instagram: Strong performance - maintain consistent posting")
        print("  • Facebook: Moderate engagement - test different formats")
        print("  • Twitter: Lower engagement - focus on trending topics")

        print("\n" + "=" * 80)


if __name__ == "__main__":
    print("\nStarting Social Media Engagement Tracking Demo...")
    print("This demonstrates Phase 2 ACD integration with real-time social signals")

    asyncio.run(demo_social_engagement_tracking())
    asyncio.run(demo_multi_platform_tracking())

    print("\n✅ All demos completed successfully!")
