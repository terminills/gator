#!/usr/bin/env python3
"""
Interactive Content and Audience Segmentation Demo

Demonstrates the new Q2 2025 features: Interactive Content and Audience Segmentation.
"""

import asyncio
import sys
from pathlib import Path
from datetime import datetime, timedelta
from uuid import UUID

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from backend.database.connection import database_manager
from backend.services.persona_service import PersonaService
from backend.services.interactive_content_service import InteractiveContentService
from backend.services.audience_segment_service import AudienceSegmentService
from backend.models.interactive_content import InteractiveContentType
from backend.models.audience_segment import PersonalizationStrategy
from backend.config.logging import setup_logging, get_logger

# Setup logging
setup_logging()
logger = get_logger(__name__)


def print_header(text):
    """Print a formatted header."""
    print(f"\n{'='*60}")
    print(f"  {text}")
    print(f"{'='*60}\n")


def print_success(text):
    """Print success message."""
    print(f"✅ {text}")


def print_info(text, indent=0):
    """Print info message."""
    print(f"{'  ' * indent}ℹ️  {text}")


async def demo_interactive_content(session, persona_id):
    """Demonstrate interactive content features."""
    print_header("🎯 Interactive Content Demo")
    
    service = InteractiveContentService(session)
    
    # 1. Create a poll
    print_info("Creating a poll about workout preferences...")
    poll = await service.create_content(
        persona_id=str(persona_id),
        content_type=InteractiveContentType.POLL,
        title="What's your favorite workout?",
        question="Choose your preferred type of workout:",
        options=[
            {"text": "Running"},
            {"text": "Yoga"},
            {"text": "Weight Training"},
            {"text": "Swimming"},
        ],
    )
    print_success(f"Poll created with ID: {poll.id}")
    print_info(f"Title: {poll.title}", 1)
    print_info(f"Options: {len(poll.options)}", 1)
    
    # 2. Publish the poll
    print_info("Publishing the poll...")
    published_poll = await service.publish_content(str(poll.id))
    print_success(f"Poll published! Status: {published_poll.status}")
    
    # 3. Simulate responses
    print_info("Simulating user responses...")
    await service.submit_response(str(poll.id), {"option_id": 1})
    await service.submit_response(str(poll.id), {"option_id": 1})
    await service.submit_response(str(poll.id), {"option_id": 2})
    await service.submit_response(str(poll.id), {"option_id": 3})
    await service.submit_response(str(poll.id), {"option_id": 1})
    print_success("5 responses submitted")
    
    # 4. Increment view counts
    for _ in range(20):
        await service.increment_view_count(str(poll.id))
    print_success("20 views recorded")
    
    # 5. Get statistics
    stats = await service.get_content_stats(str(poll.id))
    print_info("Poll Statistics:")
    print_info(f"Views: {stats['total_views']}", 1)
    print_info(f"Responses: {stats['total_responses']}", 1)
    print_info(f"Response Rate: {stats['response_rate']}%", 1)
    print_info("Top Options:", 1)
    for i, option in enumerate(stats['top_options'][:3], 1):
        print_info(f"{i}. {option['text']}: {option['votes']} votes ({option['percentage']}%)", 2)
    
    # 6. Create a story
    print_info("\nCreating a story with 24-hour expiration...")
    story = await service.create_content(
        persona_id=str(persona_id),
        content_type=InteractiveContentType.STORY,
        title="Morning Motivation",
        description="Starting the day with positive energy!",
        media_url="https://example.com/morning-story.jpg",
    )
    await service.publish_content(str(story.id))
    print_success(f"Story created and published! ID: {story.id}")
    print_info(f"Expires at: {story.expires_at.strftime('%Y-%m-%d %H:%M:%S')}", 1)
    
    # 7. List all interactive content
    print_info("\nListing all interactive content...")
    all_content = await service.list_content(persona_id=str(persona_id))
    print_success(f"Found {len(all_content)} interactive content items")
    for content in all_content:
        print_info(f"- {content.content_type.upper()}: {content.title} ({content.status})", 1)
    
    return poll.id


async def demo_audience_segmentation(session, persona_id):
    """Demonstrate audience segmentation features."""
    print_header("👥 Audience Segmentation Demo")
    
    service = AudienceSegmentService(session)
    
    # 1. Create segments
    print_info("Creating audience segments...")
    
    tech_segment = await service.create_segment(
        persona_id=str(persona_id),
        segment_name="Tech Enthusiasts",
        description="Users interested in technology and gadgets",
        criteria={
            "age_range": [25, 45],
            "interests": ["technology", "gadgets", "AI", "programming"],
            "engagement_level": "high"
        },
        strategy=PersonalizationStrategy.HYBRID,
    )
    print_success(f"Created segment: {tech_segment.segment_name}")
    print_info(f"ID: {tech_segment.id}", 1)
    print_info(f"Strategy: {tech_segment.strategy}", 1)
    
    fitness_segment = await service.create_segment(
        persona_id=str(persona_id),
        segment_name="Fitness Focused",
        description="Users who prioritize health and fitness",
        criteria={
            "age_range": [20, 55],
            "interests": ["fitness", "health", "nutrition", "wellness"],
            "engagement_level": "medium"
        },
        strategy=PersonalizationStrategy.BEHAVIORAL,
    )
    print_success(f"Created segment: {fitness_segment.segment_name}")
    print_info(f"ID: {fitness_segment.id}", 1)
    print_info(f"Strategy: {fitness_segment.strategy}", 1)
    
    # 2. List segments
    print_info("\nListing all segments...")
    segments = await service.list_segments(persona_id=str(persona_id))
    print_success(f"Found {len(segments)} segments")
    for seg in segments:
        print_info(f"- {seg.segment_name} ({seg.status})", 1)
        print_info(f"Members: {seg.member_count}", 2)
    
    # 3. Update segment with performance metrics
    print_info("\nUpdating segment with performance metrics...")
    updated_segment = await service.update_segment(
        str(tech_segment.id),
        performance_metrics={
            "avg_engagement_rate": 35.5,
            "conversion_rate": 8.2,
            "avg_view_duration": 120.5,
        },
        estimated_size=1500,
        member_count=350,
    )
    print_success("Segment updated with performance metrics")
    print_info(f"Engagement Rate: {updated_segment.performance_metrics['avg_engagement_rate']}%", 1)
    print_info(f"Member Count: {updated_segment.member_count}", 1)
    
    # 4. Run segment analysis
    print_info("\nRunning segment analysis...")
    await service.analyze_segment(str(tech_segment.id))
    print_success("Segment analysis complete")
    
    # 5. Get analytics
    analytics = await service.get_segment_analytics(str(tech_segment.id))
    print_info("Segment Analytics:")
    print_info(f"Name: {analytics['segment_name']}", 1)
    print_info(f"Members: {analytics['member_count']}", 1)
    print_info("Performance Summary:", 1)
    for key, value in analytics['performance_summary'].items():
        print_info(f"- {key}: {value}", 2)
    
    if analytics['recommendations']:
        print_info("Recommendations:", 1)
        for rec in analytics['recommendations']:
            print_info(f"- {rec}", 2)
    
    return tech_segment.id, fitness_segment.id


async def demo_integration(session, persona_id, poll_id, segment_id):
    """Demonstrate integration between features."""
    print_header("🔗 Feature Integration Demo")
    
    segment_service = AudienceSegmentService(session)
    
    # Create personalized content (linking poll to segment)
    print_info("Creating personalized content for segment...")
    
    # Note: This would normally link to real content, but for demo we'll create the mapping
    from backend.models.content import ContentModel
    from uuid import uuid4
    
    # Create demo content
    demo_content = ContentModel(
        id=uuid4(),
        persona_id=UUID(persona_id),
        content_type="poll",
        title="Tech Poll",
        content_rating="sfw",
        moderation_status="approved",
        is_published=True,
    )
    session.add(demo_content)
    await session.commit()
    await session.refresh(demo_content)
    
    personalized = await segment_service.create_personalized_content(
        content_id=str(demo_content.id),
        segment_id=str(segment_id),
        variant_id="variant_a",
        is_control=False,
    )
    print_success("Personalized content created")
    print_info(f"Content ID: {personalized.content_id}", 1)
    print_info(f"Segment ID: {personalized.segment_id}", 1)
    print_info(f"Variant: {personalized.variant_id}", 1)
    
    # Update performance
    print_info("\nUpdating personalized content performance...")
    updated = await segment_service.update_content_performance(
        str(personalized.id),
        {
            "views": 500,
            "engagement": 125,
            "conversions": 35,
        }
    )
    print_success("Performance updated")
    print_info(f"Views: {updated.view_count}", 1)
    print_info(f"Engagement: {updated.engagement_count}", 1)
    print_info(f"Conversions: {updated.conversion_count}", 1)
    print_info(f"Engagement Rate: {updated.engagement_rate}%", 1)


async def main():
    """Run the demo."""
    print("\n🎭 Gator AI Platform - Q2 2025 Features Demo")
    print("=" * 60)
    print("Demonstrating: Interactive Content & Audience Segmentation")
    print("=" * 60)
    
    try:
        # Connect to database
        print_info("Connecting to database...")
        await database_manager.connect()
        print_success("Database connected")
        
        # Get or create a test persona
        async with database_manager.session_factory() as session:
            persona_service = PersonaService(session)
            personas = await persona_service.list_personas(limit=1)
            
            if personas:
                persona = personas[0]
                print_success(f"Using existing persona: {persona.name}")
            else:
                print_info("No personas found, creating demo persona...")
                from backend.models.persona import PersonaCreate
                persona_data = PersonaCreate(
                    name="Demo Persona",
                    appearance="A friendly AI assistant",
                    personality="Helpful and engaging",
                    content_themes=["technology", "fitness", "lifestyle"],
                )
                persona_response = await persona_service.create_persona(persona_data)
                # Get the actual persona model
                persona = await persona_service.get_persona(persona_response.id)
                print_success(f"Created demo persona: {persona.name}")
            
            persona_id = str(persona.id)
            
            # Run demos
            poll_id = await demo_interactive_content(session, persona_id)
            segment_id, _ = await demo_audience_segmentation(session, persona_id)
            await demo_integration(session, persona_id, poll_id, segment_id)
        
        # Disconnect
        await database_manager.disconnect()
        
        print_header("✨ Demo Complete")
        print_success("All Q2 2025 features demonstrated successfully!")
        print("\n📚 Documentation:")
        print("  - Interactive Content: docs/INTERACTIVE_CONTENT_IMPLEMENTATION.md")
        print("  - Audience Segmentation: docs/AUDIENCE_SEGMENTATION_IMPLEMENTATION.md")
        print("\n🚀 API Endpoints:")
        print("  - Interactive: http://localhost:8000/api/v1/interactive/*")
        print("  - Segments: http://localhost:8000/api/v1/segments/*")
        print("\n📖 Interactive API Docs: http://localhost:8000/docs")
        
    except Exception as e:
        logger.error(f"Demo failed: {str(e)}")
        print(f"\n❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
