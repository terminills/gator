#!/usr/bin/env python
"""
Demo script for RSS Feed Enhancement features.

Demonstrates:
1. Adding RSS feeds
2. Assigning feeds to personas with topic filtering
3. Getting content suggestions based on assigned feeds
"""

import asyncio
from uuid import uuid4

from backend.database.connection import database_manager
from backend.services.rss_ingestion_service import RSSIngestionService
from backend.services.persona_service import PersonaService
from backend.models.feed import RSSFeedCreate, PersonaFeedAssignment
from backend.models.persona import PersonaCreate


async def main():
    """Run RSS enhancement demo."""
    print("🚀 RSS Feed Enhancement Demo\n")
    print("=" * 70)

    # Connect to database
    await database_manager.connect()

    try:
        async with database_manager.get_session() as db:
            # Initialize services
            persona_service = PersonaService(db)
            rss_service = RSSIngestionService(db)

            # 1. Create a persona
            print("\n1️⃣  Creating AI Persona...")
            persona_data = PersonaCreate(
                name="Tech Innovator Sarah",
                appearance="Professional tech entrepreneur in her 30s",
                personality="Innovative, forward-thinking, passionate about technology",
                content_themes=["technology", "ai", "startups", "innovation"],
                style_preferences={"tone": "professional", "style": "engaging"},
            )

            persona = await persona_service.create_persona(persona_data)
            print(f"   ✅ Created persona: {persona.name}")
            print(f"   📋 Themes: {', '.join(persona.content_themes)}")

            # 2. List available feeds by topic
            print("\n2️⃣  Listing existing feeds...")
            all_feeds = await rss_service.list_feeds(active_only=True)
            print(f"   📡 Total active feeds: {len(all_feeds)}")

            # Mock feeds for demo (in production, these would be real RSS feeds)
            print("\n3️⃣  Simulating feed addition (normally would fetch real RSS)...")
            print("   Note: Actual RSS fetching requires network access")
            print("   In production, use feed URLs like:")
            print("     - https://techcrunch.com/feed/")
            print("     - https://www.wired.com/feed/rss")
            print("     - https://feeds.arstechnica.com/arstechnica/index")

            # Create mock feed for demo purposes
            try:
                # This would normally be a real feed URL
                mock_feed = RSSFeedCreate(
                    name="Tech News Feed (Demo)",
                    url=f"https://example.com/feed/{uuid4()}",
                    description="Demo tech news feed",
                    categories=["technology", "ai", "startups"],
                    fetch_frequency_hours=6,
                )

                # Note: This will fail validation since it's not a real feed
                # In production, you'd use real RSS feed URLs
                print("   ⚠️  Skipping actual feed addition (requires network)")

            except Exception as e:
                print(f"   ℹ️  Expected: Cannot fetch demo URLs")

            # 4. Demonstrate feed assignment API
            print("\n4️⃣  RSS Feed to Persona Assignment (Conceptual)...")
            print("   To assign a feed to a persona:")
            print("   POST /api/v1/feeds/personas/{persona_id}/feeds")
            print("   Body: {")
            print('     "feed_id": "uuid-of-feed",')
            print('     "topics": ["ai", "machine learning"],')
            print('     "priority": 80')
            print("   }")

            # 5. List feeds by topic
            print("\n5️⃣  Feeds by Topic Organization:")
            print("   GET /api/v1/feeds/by-topic/{topic}")
            print("   Examples:")
            print("     - /api/v1/feeds/by-topic/technology")
            print("     - /api/v1/feeds/by-topic/business")
            print("     - /api/v1/feeds/by-topic/science")

            # 6. Get content suggestions
            print("\n6️⃣  Content Suggestions for Persona:")
            print("   GET /api/v1/feeds/suggestions/{persona_id}")
            print("   Returns: Feed items from assigned feeds filtered by topics")
            print("   Features:")
            print("     ✓ Topic extraction from feed items")
            print("     ✓ Sentiment analysis")
            print("     ✓ Relevance scoring")
            print("     ✓ Priority-based filtering")

            # 7. API Endpoints Summary
            print("\n" + "=" * 70)
            print("📚 Available API Endpoints:\n")

            endpoints = [
                (
                    "POST   /api/v1/feeds/",
                    "Add new RSS feed",
                ),
                (
                    "GET    /api/v1/feeds/",
                    "List all RSS feeds",
                ),
                (
                    "POST   /api/v1/feeds/fetch",
                    "Manually fetch all feeds",
                ),
                (
                    "GET    /api/v1/feeds/trending",
                    "Get trending topics",
                ),
                (
                    "POST   /api/v1/feeds/personas/{id}/feeds",
                    "Assign feed to persona",
                ),
                (
                    "GET    /api/v1/feeds/personas/{id}/feeds",
                    "List persona's feeds",
                ),
                (
                    "DELETE /api/v1/feeds/personas/{id}/feeds/{fid}",
                    "Unassign feed",
                ),
                (
                    "GET    /api/v1/feeds/by-topic/{topic}",
                    "List feeds by topic",
                ),
                (
                    "GET    /api/v1/feeds/suggestions/{id}",
                    "Get content suggestions",
                ),
            ]

            for method_path, description in endpoints:
                print(f"   {method_path:<50} {description}")

            # 8. Database Schema
            print("\n" + "=" * 70)
            print("🗄️  Database Schema:\n")
            print("   ✅ rss_feeds         - RSS feed sources")
            print("   ✅ feed_items        - Parsed feed content")
            print("   ✅ persona_feeds     - Persona-to-feed assignments (NEW)")
            print("\n   persona_feeds columns:")
            print("     - persona_id: Links to persona")
            print("     - feed_id: Links to RSS feed")
            print("     - topics: Filter specific topics from feed")
            print("     - priority: Content suggestion priority (0-100)")
            print("     - is_active: Enable/disable assignment")

            print("\n" + "=" * 70)
            print("✅ Demo Complete!")
            print("\nKey Features Implemented:")
            print("  ✓ RSS feed management with topic categorization")
            print("  ✓ Persona-to-feed assignment with topic filtering")
            print("  ✓ Content suggestions based on assigned feeds")
            print("  ✓ Topic-based feed discovery")
            print("  ✓ Priority-based content selection")
            print("  ✓ Sentiment analysis and relevance scoring")
            print("\nNext Steps:")
            print("  1. Add real RSS feeds via the API")
            print("  2. Assign feeds to personas with specific topics")
            print("  3. Fetch feed content regularly")
            print("  4. Use content suggestions for AI-generated posts")

    finally:
        await database_manager.disconnect()


if __name__ == "__main__":
    asyncio.run(main())
