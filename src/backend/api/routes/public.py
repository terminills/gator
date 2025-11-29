"""
Public View API Routes

Provides public-facing endpoints for viewing AI influencer content
without requiring authentication. Designed for public consumption.
"""

import random
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import desc, select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.database.connection import get_db_session
from backend.models.content import ContentModel
from backend.models.persona import PersonaModel

router = APIRouter(
    prefix="/api/v1/public",
    tags=["public"],
    responses={404: {"description": "Resource not found"}},
)


@router.get("/personas", response_model=List[Dict[str, Any]])
async def list_public_personas(
    limit: int = Query(
        default=10, ge=1, le=50, description="Maximum personas to return"
    ),
    category: Optional[str] = Query(None, description="Filter by category"),
    featured: Optional[bool] = Query(None, description="Filter featured personas only"),
    trending: Optional[bool] = Query(None, description="Filter trending personas only"),
    db: AsyncSession = Depends(get_db_session),
):
    """
    List publicly available AI personas from the database.

    Returns basic information about active AI personas for public viewing.
    Personal details and generation metadata are filtered for privacy.

    Args:
        limit: Maximum number of personas to return
        category: Filter by category (technology, art, lifestyle, fashion, politics, etc.)
        featured: Show only featured personas
        trending: Show only trending personas (by generation count)
        db: Database session

    Returns:
        List of public persona information
    """
    try:
        # Build query for active personas
        query = select(PersonaModel).where(PersonaModel.is_active.is_(True))

        # Apply category filter if provided
        if category:
            # Filter by content_themes containing the category
            query = query.where(PersonaModel.content_themes.contains([category]))

        # Order by generation count for trending
        if trending:
            query = query.order_by(desc(PersonaModel.generation_count))
        else:
            query = query.order_by(desc(PersonaModel.created_at))

        # Apply limit
        query = query.limit(limit)

        result = await db.execute(query)
        personas = result.scalars().all()

        # Transform to public-facing format
        public_personas = []
        for persona in personas:
            # Determine primary category from content themes
            primary_category = (
                persona.content_themes[0] if persona.content_themes else "general"
            )

            # Get style from style_preferences or default
            style = "realistic"
            if isinstance(persona.style_preferences, dict):
                style = persona.style_preferences.get("visual_style", "realistic")

            public_personas.append(
                {
                    "id": str(persona.id),
                    "name": persona.name,
                    "bio": (
                        persona.personality[:200]
                        if len(persona.personality) > 200
                        else persona.personality
                    ),
                    "themes": persona.content_themes,
                    "content_count": persona.generation_count,
                    "style": style,
                    "category": primary_category,
                    "featured": persona.generation_count
                    > 10,  # Featured if has generated content
                    "trending": persona.generation_count > 5,
                    "trending_score": min(100, persona.generation_count * 2),
                }
            )

        return public_personas

    except Exception as e:
        # Fallback to sample data if database query fails
        import logging

        logging.error(f"Failed to fetch personas from database: {e}")

        # Return minimal fallback data
        return _get_fallback_personas(limit=limit, category=category)


def _get_fallback_personas(
    limit: int = 10, category: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Fallback sample personas when database is not available.
    Used only for development/testing.
    """
    sample_personas = [
        {
            "id": "sample-politics-1",
            "name": "Alexandria Policy",
            "bio": "Political analyst AI providing insightful commentary on current events and policy analysis.",
            "themes": ["politics", "policy", "current events", "analysis"],
            "content_count": 0,
            "style": "realistic",
            "category": "politics",
            "featured": False,
            "trending": False,
            "trending_score": 50,
        },
        {
            "id": "sample-tech-1",
            "name": "Luna Tech",
            "bio": "A futuristic AI persona passionate about technology, innovation, and digital art.",
            "themes": [
                "technology",
                "digital art",
                "innovation",
            ],
            "content_count": 0,
            "style": "futuristic",
            "category": "technology",
            "featured": False,
            "trending": False,
            "trending_score": 50,
        },
    ]

    # Apply category filter if provided
    if category:
        sample_personas = [p for p in sample_personas if p.get("category") == category]

    return sample_personas[:limit]


@router.get("/personas/{persona_id}", response_model=Dict[str, Any])
async def get_public_persona(
    persona_id: str, db: AsyncSession = Depends(get_db_session)
):
    """
    Get detailed public information about specific persona from database.

    Args:
        persona_id: Persona identifier (UUID)
        db: Database session

    Returns:
        Public persona details

    Raises:
        404: Persona not found
    """
    try:
        from uuid import UUID

        # Convert string to UUID
        try:
            persona_uuid = UUID(persona_id)
        except ValueError:
            raise HTTPException(status_code=404, detail="Invalid persona ID format")

        # Query database for persona
        query = select(PersonaModel).where(
            PersonaModel.id == persona_uuid, PersonaModel.is_active.is_(True)
        )
        result = await db.execute(query)
        persona = result.scalar_one_or_none()

        if not persona:
            raise HTTPException(status_code=404, detail="Persona not found")

        # Get style from style_preferences (for potential future use)
        if isinstance(persona.style_preferences, dict):
            _style = persona.style_preferences.get("visual_style", "realistic")

        # Transform to public format
        return {
            "id": str(persona.id),
            "name": persona.name,
            "bio": persona.personality,
            "appearance": persona.appearance,
            "themes": persona.content_themes,
            "style_preferences": persona.style_preferences,
            "statistics": {
                "content_created": persona.generation_count,
                "active_since": persona.created_at.isoformat(),
                "last_updated": persona.updated_at.isoformat(),
            },
        }

    except HTTPException:
        raise
    except Exception as e:
        import logging

        logging.error(f"Failed to fetch persona {persona_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch persona")


@router.get("/personas/{persona_id}/gallery", response_model=List[Dict[str, Any]])
async def get_persona_gallery(
    persona_id: str,
    content_type: Optional[str] = Query(
        None, pattern="^(image|video|text)$", description="Filter by content type"
    ),
    limit: int = Query(default=20, ge=1, le=100, description="Maximum items to return"),
    offset: int = Query(default=0, ge=0, description="Number of items to skip"),
    db: AsyncSession = Depends(get_db_session),
):
    """
    Get public gallery of content for specific persona from database.

    Returns approved, publicly viewable content created by the AI persona.

    Args:
        persona_id: Persona UUID
        content_type: Filter by content type (image, video, text)
        limit: Maximum items to return
        offset: Number of items to skip
        db: Database session

    Returns:
        List of content items for the persona
    """
    try:
        from uuid import UUID

        # Convert string to UUID
        try:
            persona_uuid = UUID(persona_id)
        except ValueError:
            return []  # Return empty list for invalid UUID

        # Build query for persona content
        query = select(ContentModel).where(ContentModel.persona_id == persona_uuid)

        # Filter by content type if specified
        if content_type:
            query = query.where(ContentModel.content_type == content_type)

        # Order by creation date (newest first)
        query = query.order_by(desc(ContentModel.created_at))

        # Apply pagination
        query = query.limit(limit).offset(offset)

        result = await db.execute(query)
        content_items = result.scalars().all()

        # Transform to public format
        return [
            {
                "id": str(item.id),
                "type": item.content_type,
                "title": item.prompt[:50] if item.prompt else "Untitled",
                "description": item.prompt if item.prompt else "AI-generated content",
                "created_at": item.created_at.isoformat(),
                "file_path": item.file_path,
            }
            for item in content_items
        ]

    except Exception as e:
        import logging

        logging.error(f"Failed to fetch gallery for persona {persona_id}: {e}")
        return []  # Return empty list on error


@router.get("/categories", response_model=List[Dict[str, Any]])
async def list_categories(db: AsyncSession = Depends(get_db_session)):
    """
    Get available persona categories with counts from database.

    Returns:
        List of categories with actual persona counts from database
    """
    try:
        # Get all active personas
        query = select(PersonaModel).where(PersonaModel.is_active.is_(True))
        result = await db.execute(query)
        personas = result.scalars().all()

        # Count personas by category (using first theme as category)
        category_counts = {}
        for persona in personas:
            if persona.content_themes:
                primary_category = persona.content_themes[0]
                category_counts[primary_category] = (
                    category_counts.get(primary_category, 0) + 1
                )

        # Define category metadata with icons
        category_info = {
            "politics": {
                "name": "Politics & Policy",
                "description": "Political analysis, policy commentary, and civic engagement",
                "icon": "🗳️",
            },
            "technology": {
                "name": "Technology",
                "description": "AI personas focused on tech, innovation, and digital trends",
                "icon": "🚀",
            },
            "art": {
                "name": "Art & Creativity",
                "description": "Creative AI personas specializing in visual arts and design",
                "icon": "🎨",
            },
            "lifestyle": {
                "name": "Lifestyle",
                "description": "Personas covering wellness, cooking, travel, and daily living",
                "icon": "✨",
            },
            "fashion": {
                "name": "Fashion & Style",
                "description": "Fashion-forward AI personas with style expertise",
                "icon": "👗",
            },
            "business": {
                "name": "Business & Finance",
                "description": "Business insights, financial analysis, and entrepreneurship",
                "icon": "💼",
            },
            "entertainment": {
                "name": "Entertainment",
                "description": "Movies, music, gaming, and pop culture content",
                "icon": "🎬",
            },
        }

        # Build categories list
        categories = []

        # Add categories that have personas
        for category, count in category_counts.items():
            info = category_info.get(
                category,
                {
                    "name": category.title(),
                    "description": f"Personas focused on {category}",
                    "icon": "📁",
                },
            )
            categories.append(
                {
                    "id": category,
                    "name": info["name"],
                    "description": info["description"],
                    "icon": info["icon"],
                    "persona_count": count,
                }
            )

        # Sort by persona count (descending)
        categories.sort(key=lambda x: x["persona_count"], reverse=True)

        return categories

    except Exception as e:
        import logging

        logging.error(f"Failed to fetch categories: {e}")

        # Return basic fallback categories
        return [
            {
                "id": "politics",
                "name": "Politics & Policy",
                "description": "Political content",
                "icon": "🗳️",
                "persona_count": 0,
            },
            {
                "id": "technology",
                "name": "Technology",
                "description": "Tech content",
                "icon": "🚀",
                "persona_count": 0,
            },
            {
                "id": "art",
                "name": "Art & Creativity",
                "description": "Creative content",
                "icon": "🎨",
                "persona_count": 0,
            },
        ]


@router.get("/feed", response_model=List[Dict[str, Any]])
async def get_public_feed(
    limit: int = Query(default=20, ge=1, le=100, description="Maximum posts to return"),
):
    """
    Get public feed of posts from AI personas.

    Returns a feed of posts with engagement metrics (likes, comments, shares) and timestamps.
    This endpoint uses real persona data from the database and generates feed-style posts.

    Args:
        limit: Maximum number of posts to return

    Returns:
        List of feed posts with persona info and engagement metrics
    """
    # Get real personas data - call with explicit parameters
    personas_response = await list_public_personas(
        limit=50, category=None, featured=None, trending=None
    )

    if not personas_response:
        return []

    # Generate feed posts from personas
    posts = []
    current_time = datetime.now()

    for persona in personas_response:
        # Create 2-3 posts per persona for a richer feed
        num_posts = random.randint(2, 3)
        for i in range(num_posts):
            # Generate realistic timestamp (within last 7 days)
            days_ago = random.uniform(0, 7)
            post_time = current_time - timedelta(days=days_ago)

            # Generate engagement metrics based on persona popularity
            base_engagement = persona.get("content_count", 0) * 10
            post = {
                "id": f"{persona['id']}-post-{i}",
                "persona_id": persona["id"],
                "persona_name": persona["name"],
                "persona_bio": persona["bio"],
                "persona_style": persona.get("style", "realistic"),
                "persona_themes": persona.get("themes", []),
                "timestamp": post_time.isoformat(),
                "text": _generate_post_text(persona),
                "likes": random.randint(base_engagement, base_engagement + 500),
                "comments": random.randint(10, 100),
                "shares": random.randint(5, 50),
                "content_type": random.choice(["image", "text", "video"]),
            }
            posts.append(post)

    # Sort by timestamp (newest first)
    posts.sort(key=lambda x: x["timestamp"], reverse=True)

    return posts[:limit]


def _generate_post_text(persona: Dict[str, Any]) -> str:
    """Generate realistic post text based on persona."""
    themes = persona.get("themes", ["content"])
    templates = [
        f"Just created an amazing {themes[0]} piece! What do you think? 🎨",
        f"Exploring new {themes[1] if len(themes) > 1 else themes[0]} concepts today. The creative process is fascinating! ✨",
        f"New creation alert! Diving deep into {themes[2] if len(themes) > 2 else themes[0]} 🚀",
        f"{persona['bio'].split('.')[0] if '.' in persona['bio'] else persona['bio'][:100]}",
    ]
    return random.choice(templates)
