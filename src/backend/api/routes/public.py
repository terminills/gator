"""
Public View API Routes

Provides public-facing endpoints for viewing AI influencer content
without requiring authentication. Designed for public consumption.
"""

from typing import List, Optional, Dict, Any

from fastapi import APIRouter, Query

router = APIRouter(
    prefix="/public",
    tags=["public"],
    responses={404: {"description": "Resource not found"}},
)


@router.get("/personas", response_model=List[Dict[str, Any]])
async def list_public_personas(
    limit: int = Query(default=10, ge=1, le=50, description="Maximum personas to return"),
):
    """
    List publicly available AI personas.
    
    Returns basic information about active AI personas for public viewing.
    Personal details and generation metadata are filtered for privacy.
    
    Args:
        limit: Maximum number of personas to return
    
    Returns:
        List of public persona information
    """
    # Mock data for demonstration - in real app this would query database
    mock_personas = [
        {
            "id": "persona-1",
            "name": "Luna Tech",
            "bio": "A futuristic AI persona passionate about technology, innovation, and digital art. Creates stunning tech-inspired content with a sleek modern aesthetic.",
            "themes": ["technology", "digital art", "innovation", "futurism", "cyberpunk"],
            "content_count": 42,
            "style": "futuristic"
        },
        {
            "id": "persona-2", 
            "name": "Aria Creative",
            "bio": "An artistic AI with a love for vibrant colors, abstract compositions, and creative storytelling. Specializes in whimsical and imaginative content.",
            "themes": ["art", "creativity", "abstract", "colors", "storytelling"],
            "content_count": 38,
            "style": "artistic"
        },
        {
            "id": "persona-3",
            "name": "Nova Vintage",
            "bio": "Nostalgic AI creator focused on retro aesthetics, vintage photography, and timeless elegance. Brings classic beauty into the modern world.",
            "themes": ["vintage", "retro", "photography", "elegance", "nostalgia"],
            "content_count": 25,
            "style": "vintage"
        }
    ]
    
    return mock_personas[:limit]


@router.get("/personas/{persona_id}", response_model=Dict[str, Any])
async def get_public_persona(persona_id: str):
    """
    Get detailed public information about specific persona.
    
    Args:
        persona_id: Persona identifier
    
    Returns:
        Public persona details
    """
    # Mock data - in real app would query database
    mock_personas = {
        "persona-1": {
            "id": "persona-1",
            "name": "Luna Tech",
            "bio": "A futuristic AI persona passionate about technology, innovation, and digital art. Creates stunning tech-inspired content with a sleek modern aesthetic that blends cutting-edge design with artistic vision.",
            "appearance": "Sleek, modern aesthetic with digital elements",
            "themes": ["technology", "digital art", "innovation", "futurism", "cyberpunk", "AI", "robotics"],
            "style_preferences": {
                "visual_style": "futuristic",
                "color_palette": "neon",
                "lighting": "dramatic"
            },
            "statistics": {
                "content_created": 42,
                "active_since": "2024-01-15T00:00:00Z",
                "last_updated": "2024-01-20T12:00:00Z"
            }
        },
        "persona-2": {
            "id": "persona-2",
            "name": "Aria Creative", 
            "bio": "An artistic AI with a love for vibrant colors, abstract compositions, and creative storytelling. Specializes in whimsical and imaginative content that sparks joy and wonder.",
            "appearance": "Colorful, expressive with artistic flair",
            "themes": ["art", "creativity", "abstract", "colors", "storytelling", "imagination", "whimsical"],
            "style_preferences": {
                "visual_style": "artistic",
                "color_palette": "vibrant",
                "lighting": "natural"
            },
            "statistics": {
                "content_created": 38,
                "active_since": "2024-01-10T00:00:00Z", 
                "last_updated": "2024-01-19T15:30:00Z"
            }
        },
        "persona-3": {
            "id": "persona-3",
            "name": "Nova Vintage",
            "bio": "Nostalgic AI creator focused on retro aesthetics, vintage photography, and timeless elegance. Brings classic beauty into the modern world with sophisticated charm.",
            "appearance": "Classic, elegant with vintage styling",
            "themes": ["vintage", "retro", "photography", "elegance", "nostalgia", "classic", "timeless"],
            "style_preferences": {
                "visual_style": "vintage",
                "color_palette": "sepia",
                "lighting": "soft"
            },
            "statistics": {
                "content_created": 25,
                "active_since": "2024-01-05T00:00:00Z",
                "last_updated": "2024-01-18T09:15:00Z"
            }
        }
    }
    
    if persona_id not in mock_personas:
        return {"error": "Persona not found"}
        
    return mock_personas[persona_id]


@router.get("/personas/{persona_id}/gallery", response_model=List[Dict[str, Any]])
async def get_persona_gallery(
    persona_id: str,
    content_type: Optional[str] = Query(None, regex="^(image|video|text)$", description="Filter by content type"),
    limit: int = Query(default=20, ge=1, le=100, description="Maximum items to return"),
    offset: int = Query(default=0, ge=0, description="Number of items to skip"),
):
    """
    Get public gallery of content for specific persona.
    
    Returns approved, publicly viewable content created by the AI persona.
    """
    # Mock gallery data
    mock_content = [
        {
            "id": "content-1",
            "type": "image",
            "title": "Neon Cityscape",
            "description": "A stunning futuristic cityscape with neon lights and cyberpunk aesthetics",
            "created_at": "2024-01-20T10:00:00Z"
        },
        {
            "id": "content-2", 
            "type": "image",
            "title": "Digital Dreams",
            "description": "Abstract digital art exploring the intersection of technology and imagination",
            "created_at": "2024-01-19T14:30:00Z"
        },
        {
            "id": "content-3",
            "type": "video",
            "title": "Tech Innovation",
            "description": "A short video showcasing cutting-edge technology concepts",
            "created_at": "2024-01-18T16:45:00Z"
        },
        {
            "id": "content-4",
            "type": "text",
            "title": "The Future is Now",
            "description": "A thoughtful piece about emerging technologies and their impact on society",
            "created_at": "2024-01-17T09:15:00Z"
        }
    ]
    
    # Filter by content type if specified
    if content_type:
        mock_content = [item for item in mock_content if item["type"] == content_type]
    
    # Apply pagination
    start = offset
    end = offset + limit
    
    return mock_content[start:end]