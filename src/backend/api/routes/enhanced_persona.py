"""
Enhanced Persona Creator API Routes

API endpoints for the advanced persona creation wizard with presets,
feature selection, and 4-image preview generation.
"""

from typing import Any, Dict, List
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.ext.asyncio import AsyncSession

from backend.config.logging import get_logger
from backend.database.connection import get_db_session
from backend.models.persona import PersonaResponse
from backend.services.enhanced_persona_creator import (
    EnhancedPersonaCreate,
    EnhancedPersonaCreatorService,
    FacePreview,
    PersonalitySelection,
    PhysicalFeaturesSelection,
)

logger = get_logger(__name__)
router = APIRouter(prefix="/enhanced-persona", tags=["enhanced-persona"])


def get_creator_service(
    db: AsyncSession = Depends(get_db_session),
) -> EnhancedPersonaCreatorService:
    """Dependency injection for enhanced persona creator service."""
    return EnhancedPersonaCreatorService(db)


@router.get("/presets")
async def get_presets():
    """
    Get available persona presets.

    Returns preset templates for fitness influencers, fashion influencers,
    gaming streamers, tech reviewers, lifestyle bloggers, and food creators.
    """
    service = EnhancedPersonaCreatorService(None)  # No DB needed for presets
    return {
        "presets": service.get_available_presets(),
        "count": len(service.get_available_presets()),
    }


@router.get("/feature-options")
async def get_feature_options():
    """
    Get available physical feature options.

    Returns dropdown options for body type, hair color, hair style, eye color,
    skin tone, age range, gender, and ethnicity.
    """
    service = EnhancedPersonaCreatorService(None)
    return {
        "features": service.get_feature_options(),
        "categories": list(service.get_feature_options().keys()),
    }


@router.get("/personality-options")
async def get_personality_options():
    """
    Get available personality trait options.

    Returns options for energy level, communication style, authenticity,
    expertise level, and engagement style.
    """
    service = EnhancedPersonaCreatorService(None)
    return {
        "traits": service.get_personality_options(),
        "categories": list(service.get_personality_options().keys()),
    }


@router.post("/preview-appearance")
async def preview_appearance(
    physical_features: PhysicalFeaturesSelection,
    preset_id: str = Query(None, description="Optional preset ID"),
):
    """
    Preview appearance description from feature selections.

    Returns the detailed appearance description that will be used for
    image generation, without actually generating images.
    """
    service = EnhancedPersonaCreatorService(None)
    description = service.build_appearance_description(physical_features, preset_id)
    return {
        "appearance_description": description,
        "physical_features": physical_features.model_dump(),
    }


@router.post("/preview-personality")
async def preview_personality(
    personality_selection: PersonalitySelection,
    preset_id: str = Query(None, description="Optional preset ID"),
):
    """
    Preview personality description from trait selections.

    Returns the personality description that will be used for the persona.
    """
    service = EnhancedPersonaCreatorService(None)
    description = service.build_personality_description(
        personality_selection, preset_id
    )
    return {
        "personality_description": description,
        "personality_selection": personality_selection.model_dump(),
    }


@router.post("/generate-face-previews", response_model=List[FacePreview])
async def generate_face_previews(
    physical_features: PhysicalFeaturesSelection,
    preset_id: str = Query(None, description="Optional preset ID"),
    count: int = Query(4, ge=1, le=8, description="Number of previews to generate"),
    quality: str = Query("high", description="Quality preset"),
    service: EnhancedPersonaCreatorService = Depends(get_creator_service),
):
    """
    Generate face preview options based on feature selections.

    Creates multiple face images (default 4) for the user to choose from.
    Each preview uses a different seed for variation while matching
    the specified physical features.

    This is the key step where users see visual options before finalizing
    their persona's appearance.
    """
    try:
        # Build appearance description
        appearance_description = service.build_appearance_description(
            physical_features, preset_id
        )

        # Generate previews
        previews = await service.generate_face_previews(
            appearance_description, count=count, quality=quality
        )

        return previews

    except Exception as e:
        logger.error(f"Failed to generate face previews: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate previews: {str(e)}",
        )


@router.post(
    "/create-with-preview",
    response_model=PersonaResponse,
    status_code=status.HTTP_201_CREATED,
)
async def create_persona_with_preview(
    creation_data: EnhancedPersonaCreate,
    selected_preview_id: int = Query(..., description="ID of selected preview (0-3)"),
    preview_image_path: str = Query(..., description="Path to selected preview image"),
    service: EnhancedPersonaCreatorService = Depends(get_creator_service),
):
    """
    Create persona with selected face preview as locked base image.

    Final step of persona creation wizard. Takes all the selections
    (preset, features, personality, content themes) and the user's
    chosen preview image, then creates the persona with appearance locked.

    The selected face becomes the base_image for consistent future content.
    """
    try:
        persona = await service.create_persona_with_preview(
            creation_data=creation_data,
            selected_preview_id=selected_preview_id,
            preview_image_path=preview_image_path,
        )

        return PersonaResponse.model_validate(persona)

    except Exception as e:
        logger.error(f"Failed to create persona: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create persona: {str(e)}",
        )


@router.post(
    "/quick-create", response_model=PersonaResponse, status_code=status.HTTP_201_CREATED
)
async def quick_create_from_preset(
    name: str,
    preset_id: str,
    generate_previews: bool = Query(True, description="Generate face previews"),
    auto_select_preview: bool = Query(False, description="Auto-select first preview"),
    service: EnhancedPersonaCreatorService = Depends(get_creator_service),
):
    """
    Quick create persona from preset with default features.

    Simplified creation flow that uses preset defaults and optionally
    generates face previews. If auto_select_preview is True, automatically
    uses the first generated preview without user selection.
    """
    try:
        from backend.services.enhanced_persona_creator import PERSONA_PRESETS

        if preset_id not in PERSONA_PRESETS:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid preset ID: {preset_id}",
            )

        preset = PERSONA_PRESETS[preset_id]

        # Use preset defaults for physical features
        physical_features = PhysicalFeaturesSelection(
            body_type=preset["physical_defaults"].get("body_type", "average"),
            hair_color="brown",
            hair_style="shoulder-length",
            eye_color="brown",
            skin_tone="medium",
            age_range=preset["physical_defaults"].get("age_range", "25-35"),
            gender="female",
            ethnicity="mixed",
        )

        # Use moderate personality traits
        personality_selection = PersonalitySelection(
            energy_level="moderate energy",
            communication_style="friendly",
            authenticity="relatable",
            expertise_level="knowledgeable",
            engagement_style="conversational",
        )

        # Generate previews if requested
        preview_path = None
        if generate_previews:
            appearance_desc = service.build_appearance_description(
                physical_features, preset_id
            )
            previews = await service.generate_face_previews(
                appearance_desc, count=1 if auto_select_preview else 4
            )

            if previews and auto_select_preview:
                preview_path = previews[0].image_path

        # Create persona
        creation_data = EnhancedPersonaCreate(
            name=name,
            preset_id=preset_id,
            physical_features=physical_features,
            personality_selection=personality_selection,
            content_themes=preset["content_themes"],
            platform_focus=preset["style_preferences"]["platform_focus"],
        )

        if preview_path:
            persona = await service.create_persona_with_preview(
                creation_data=creation_data,
                selected_preview_id=0,
                preview_image_path=preview_path,
            )
        else:
            # Create without preview (for testing or quick setup)
            appearance_desc = service.build_appearance_description(
                physical_features, preset_id
            )
            personality_desc = service.build_personality_description(
                personality_selection, preset_id
            )

            from backend.models.persona import PersonaModel

            persona = PersonaModel(
                name=name,
                appearance=appearance_desc,
                personality=personality_desc,
                content_themes=preset["content_themes"],
                style_preferences=preset["style_preferences"],
            )

            service.db.add(persona)
            await service.db.commit()
            await service.db.refresh(persona)

        return PersonaResponse.model_validate(persona)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to quick create persona: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create persona: {str(e)}",
        )
