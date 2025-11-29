"""
Friend Groups API Routes

API endpoints for managing persona friend groups, interactions,
and collaborative content like duets and reels.
"""

from typing import List
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.ext.asyncio import AsyncSession

from backend.config.logging import get_logger
from backend.database.connection import get_db_session
from backend.models.friend_groups import (
    DuetRequestCreate,
    DuetRequestResponse,
    FriendGroupCreate,
    FriendGroupResponse,
    FriendGroupUpdate,
    PersonaInteractionCreate,
    PersonaInteractionResponse,
)
from backend.services.friend_groups_service import FriendGroupsService
from backend.services.reel_generation_service import ReelGenerationService

logger = get_logger(__name__)
router = APIRouter(prefix="/friend-groups", tags=["friend-groups"])


def get_friend_groups_service(
    db: AsyncSession = Depends(get_db_session),
) -> FriendGroupsService:
    """Dependency injection for friend groups service."""
    return FriendGroupsService(db)


def get_reel_service(
    db: AsyncSession = Depends(get_db_session),
) -> ReelGenerationService:
    """Dependency injection for reel generation service."""
    return ReelGenerationService(db)


# Friend Group Management


@router.post(
    "/", response_model=FriendGroupResponse, status_code=status.HTTP_201_CREATED
)
async def create_friend_group(
    group_data: FriendGroupCreate,
    service: FriendGroupsService = Depends(get_friend_groups_service),
):
    """
    Create a new friend group.

    Friend groups allow personas to form social networks and interact
    with each other's content across platforms.
    """
    try:
        return await service.create_friend_group(group_data)
    except Exception as e:
        logger.error(f"Failed to create friend group: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create friend group: {str(e)}",
        )


@router.get("/", response_model=List[FriendGroupResponse])
async def list_friend_groups(
    active_only: bool = Query(True, description="Only return active groups"),
    persona_id: UUID = Query(None, description="Filter by persona membership"),
    limit: int = Query(50, ge=1, le=100),
    service: FriendGroupsService = Depends(get_friend_groups_service),
):
    """
    List friend groups.

    Can filter by active status and persona membership.
    """
    try:
        return await service.list_friend_groups(
            active_only=active_only, persona_id=persona_id, limit=limit
        )
    except Exception as e:
        logger.error(f"Failed to list friend groups: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to list friend groups",
        )


@router.get("/{group_id}", response_model=FriendGroupResponse)
async def get_friend_group(
    group_id: UUID,
    service: FriendGroupsService = Depends(get_friend_groups_service),
):
    """Get friend group by ID."""
    group = await service.get_friend_group(group_id)
    if not group:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Friend group {group_id} not found",
        )
    return group


@router.put("/{group_id}", response_model=FriendGroupResponse)
async def update_friend_group(
    group_id: UUID,
    update_data: FriendGroupUpdate,
    service: FriendGroupsService = Depends(get_friend_groups_service),
):
    """Update friend group settings."""
    group = await service.update_friend_group(group_id, update_data)
    if not group:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Friend group {group_id} not found",
        )
    return group


# Group Membership Management


@router.post("/{group_id}/members/{persona_id}")
async def add_group_member(
    group_id: UUID,
    persona_id: UUID,
    role: str = Query("member", description="Role in group"),
    service: FriendGroupsService = Depends(get_friend_groups_service),
):
    """Add a persona to a friend group."""
    success = await service.add_persona_to_group(group_id, persona_id, role)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Group or persona not found",
        )
    return {
        "message": "Persona added to group",
        "group_id": str(group_id),
        "persona_id": str(persona_id),
    }


@router.delete("/{group_id}/members/{persona_id}")
async def remove_group_member(
    group_id: UUID,
    persona_id: UUID,
    service: FriendGroupsService = Depends(get_friend_groups_service),
):
    """Remove a persona from a friend group."""
    success = await service.remove_persona_from_group(group_id, persona_id)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Group or persona not found",
        )
    return {"message": "Persona removed from group"}


@router.get("/{group_id}/members")
async def get_group_members(
    group_id: UUID,
    service: FriendGroupsService = Depends(get_friend_groups_service),
):
    """Get all personas in a friend group."""
    members = await service.get_group_members(group_id)
    return {
        "group_id": str(group_id),
        "member_count": len(members),
        "members": [
            {
                "id": str(m.id),
                "name": m.name,
                "appearance": (
                    m.appearance[:100] + "..."
                    if len(m.appearance) > 100
                    else m.appearance
                ),
            }
            for m in members
        ],
    }


# Persona Interactions


@router.post(
    "/interactions",
    response_model=PersonaInteractionResponse,
    status_code=status.HTTP_201_CREATED,
)
async def create_interaction(
    interaction_data: PersonaInteractionCreate,
    service: FriendGroupsService = Depends(get_friend_groups_service),
):
    """
    Create a persona interaction (like, comment, share, etc.).

    Interactions can be between any personas, not just those in friend groups.
    """
    try:
        return await service.create_interaction(interaction_data)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
    except Exception as e:
        logger.error(f"Failed to create interaction: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create interaction",
        )


@router.get(
    "/interactions/content/{content_id}",
    response_model=List[PersonaInteractionResponse],
)
async def get_content_interactions(
    content_id: UUID,
    limit: int = Query(50, ge=1, le=100),
    service: FriendGroupsService = Depends(get_friend_groups_service),
):
    """Get all interactions for a piece of content."""
    return await service.get_content_interactions(content_id, limit=limit)


@router.get(
    "/interactions/persona/{persona_id}",
    response_model=List[PersonaInteractionResponse],
)
async def get_persona_interactions(
    persona_id: UUID,
    as_source: bool = Query(
        True, description="Get interactions made by persona (vs received)"
    ),
    limit: int = Query(50, ge=1, le=100),
    service: FriendGroupsService = Depends(get_friend_groups_service),
):
    """Get interactions by or targeting a persona."""
    return await service.get_persona_interactions(
        persona_id, as_source=as_source, limit=limit
    )


@router.post("/{group_id}/auto-interact/{content_id}")
async def generate_auto_interactions(
    group_id: UUID,
    content_id: UUID,
    service: FriendGroupsService = Depends(get_friend_groups_service),
):
    """
    Auto-generate interactions from group members for content.

    Creates likes, comments, shares based on group settings and
    interaction frequency configuration.
    """
    interactions = await service.generate_auto_interactions(group_id, content_id)
    return {
        "message": f"Generated {len(interactions)} auto-interactions",
        "count": len(interactions),
        "interactions": interactions,
    }


# Reel and Duet Generation


@router.post("/reels/single", status_code=status.HTTP_201_CREATED)
async def generate_single_reel(
    persona_id: UUID,
    prompt: str,
    duration: float = Query(15.0, ge=5.0, le=60.0, description="Duration in seconds"),
    quality: str = Query("high", description="Quality preset"),
    service: ReelGenerationService = Depends(get_reel_service),
):
    """
    Generate a single-persona reel (short-form video).

    Creates a vertical video (1080x1920) suitable for TikTok, Instagram Reels, etc.
    """
    try:
        from backend.services.video_processing_service import VideoQuality

        video_quality = VideoQuality(quality)

        result = await service.generate_single_reel(
            persona_id=persona_id,
            prompt=prompt,
            duration=duration,
            quality=video_quality,
        )

        return {
            "message": "Single reel generated successfully",
            "reel": result,
        }
    except Exception as e:
        logger.error(f"Failed to generate single reel: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate reel: {str(e)}",
        )


@router.post(
    "/reels/duet",
    response_model=DuetRequestResponse,
    status_code=status.HTTP_201_CREATED,
)
async def create_duet_request(
    duet_data: DuetRequestCreate,
    service: ReelGenerationService = Depends(get_reel_service),
):
    """
    Create a duet/reaction reel request.

    Generates a split-screen or overlay video with the original content
    and reactions from one or more personas.

    Supports:
    - side_by_side: Split screen (2 personas)
    - reaction: Original video with small reaction overlay
    - grid: 2x2 grid for multiple reactions (up to 4 personas)
    """
    try:
        duet_request = await service.create_duet_request(
            original_content_id=duet_data.original_content_id,
            participant_personas=duet_data.participant_personas,
            duet_type=duet_data.duet_type,
            layout_config=duet_data.layout_config,
        )

        return DuetRequestResponse.model_validate(duet_request)
    except Exception as e:
        logger.error(f"Failed to create duet request: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create duet request: {str(e)}",
        )


@router.post("/reels/duet/{request_id}/process")
async def process_duet_request(
    request_id: UUID,
    service: ReelGenerationService = Depends(get_reel_service),
):
    """
    Process a pending duet request and generate the duet video.

    This endpoint triggers the actual video generation for a duet request.
    """
    try:
        result = await service.process_duet_request(request_id)

        return {
            "message": "Duet reel generated successfully",
            "request_id": str(request_id),
            "reel": result,
        }
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e),
        )
    except Exception as e:
        logger.error(f"Failed to process duet request: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process duet request: {str(e)}",
        )


@router.post("/reels/duet/direct")
async def generate_duet_reel_direct(
    original_content_id: UUID,
    participant_persona_ids: List[UUID],
    duet_type: str = Query("side_by_side", description="Layout type"),
    service: ReelGenerationService = Depends(get_reel_service),
):
    """
    Generate a duet reel directly without creating a request.

    For immediate duet generation when you don't need request tracking.
    """
    try:
        result = await service.generate_duet_reel(
            original_content_id=original_content_id,
            participant_persona_ids=participant_persona_ids,
            duet_type=duet_type,
        )

        return {
            "message": "Duet reel generated successfully",
            "reel": result,
        }
    except Exception as e:
        logger.error(f"Failed to generate duet reel: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate duet reel: {str(e)}",
        )
