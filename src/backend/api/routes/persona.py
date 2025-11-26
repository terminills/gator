"""
Persona Management API Routes

Handles AI persona creation, management, and configuration.
"""

from typing import List, Optional, Dict, Any
from uuid import UUID
from datetime import datetime
import os
import base64
import asyncio
import json

import httpx
from fastapi import APIRouter, Depends, HTTPException, status, Query, UploadFile, File
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from backend.database.connection import get_db_session
from backend.models.persona import (
    PersonaCreate,
    PersonaResponse,
    PersonaUpdate,
    BaseImageStatus,
)
from backend.services.persona_service import PersonaService
from backend.services.ai_models import AIModelManager
from backend.config.logging import get_logger

logger = get_logger(__name__)

# Ollama configuration for chat image generation
# These can be overridden via environment variables
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_CONNECT_TIMEOUT = float(os.environ.get("OLLAMA_CONNECT_TIMEOUT", "5.0"))
OLLAMA_GENERATE_TIMEOUT = float(os.environ.get("OLLAMA_GENERATE_TIMEOUT", "60.0"))

router = APIRouter(
    prefix="/api/v1/personas",
    tags=["personas"],
    responses={404: {"description": "Persona not found"}},
)


def get_persona_service(db: AsyncSession = Depends(get_db_session)) -> PersonaService:
    """Dependency injection for PersonaService."""
    return PersonaService(db)


@router.post("/", response_model=PersonaResponse, status_code=status.HTTP_201_CREATED)
async def create_persona(
    persona_data: PersonaCreate,
    persona_service: PersonaService = Depends(get_persona_service),
):
    """
    Create a new AI persona.

    Creates a new persona configuration that will be used for consistent
    character generation across all content. The persona includes appearance,
    personality traits, content themes, and style preferences.

    Args:
        persona_data: Persona configuration data
        persona_service: Injected persona service

    Returns:
        PersonaResponse: Created persona with metadata

    Raises:
        400: Validation error or content moderation failure
        500: Internal server error
    """
    try:
        persona = await persona_service.create_persona(persona_data)
        logger.info(f"Persona created: {persona.id}")
        return persona
    except ValueError as e:
        logger.warning(f"Persona validation failed: {str(e)}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to create persona: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error",
        )


@router.get("/", response_model=List[PersonaResponse])
async def list_personas(
    skip: int = Query(0, ge=0, description="Number of personas to skip"),
    limit: int = Query(10, ge=1, le=100, description="Number of personas to return"),
    active_only: bool = Query(True, description="Return only active personas"),
    persona_service: PersonaService = Depends(get_persona_service),
):
    """
    List personas with pagination.

    Retrieves a paginated list of personas with optional filtering
    by active status.

    Args:
        skip: Number of records to skip for pagination
        limit: Maximum number of records to return
        active_only: Whether to return only active personas
        persona_service: Injected persona service

    Returns:
        List[PersonaResponse]: List of personas
    """
    try:
        personas = await persona_service.list_personas(
            skip=skip, limit=limit, active_only=active_only
        )
        return personas
    except Exception as e:
        logger.error(f"Failed to list personas: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error",
        )


@router.get("/{persona_id}", response_model=PersonaResponse)
async def get_persona(
    persona_id: str,
    persona_service: PersonaService = Depends(get_persona_service),
):
    """
    Get a specific persona by ID.

    Args:
        persona_id: The persona identifier
        persona_service: Injected persona service

    Returns:
        PersonaResponse: The requested persona

    Raises:
        404: Persona not found
        500: Internal server error
    """
    try:
        persona = await persona_service.get_persona(persona_id)
        if not persona:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Persona {persona_id} not found",
            )
        return persona
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get persona {persona_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error",
        )


@router.put("/{persona_id}", response_model=PersonaResponse)
async def update_persona(
    persona_id: str,
    updates: PersonaUpdate,
    persona_service: PersonaService = Depends(get_persona_service),
):
    """
    Update an existing persona.

    Updates the specified persona with new data. All updates go through
    validation and content moderation.

    Args:
        persona_id: The persona to update
        updates: Fields to update
        persona_service: Injected persona service

    Returns:
        PersonaResponse: Updated persona

    Raises:
        404: Persona not found
        400: Validation error
        500: Internal server error
    """
    try:
        persona = await persona_service.update_persona(persona_id, updates)
        if not persona:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Persona {persona_id} not found",
            )
        logger.info(f"Persona updated {persona_id}")
        return persona
    except HTTPException:
        raise
    except ValueError as e:
        logger.warning(f"Persona update validation failed: {str(e)}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to update persona {persona_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error",
        )


@router.delete("/{persona_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_persona(
    persona_id: str,
    persona_service: PersonaService = Depends(get_persona_service),
):
    """
    Delete a persona (soft delete).

    Marks the persona as inactive rather than permanently deleting it.
    This preserves referential integrity and audit trails.

    Args:
        persona_id: The persona to delete
        persona_service: Injected persona service

    Raises:
        404: Persona not found
        500: Internal server error
    """
    try:
        success = await persona_service.delete_persona(persona_id)
        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Persona {persona_id} not found",
            )
        logger.info(f"Persona deleted {persona_id}")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete persona {persona_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error",
        )


@router.post("/{persona_id}/seed-image/upload", response_model=PersonaResponse)
async def upload_seed_image(
    persona_id: str,
    file: UploadFile = File(...),
    persona_service: PersonaService = Depends(get_persona_service),
):
    """
    Upload a seed image for a persona (Method 1: User Upload).

    Accepts binary file data and saves it as the persona's base image.
    Sets base_image_status to DRAFT for review.

    Args:
        persona_id: The persona to upload the image for
        file: The image file (PNG, JPG, WEBP)
        persona_service: Injected persona service

    Returns:
        PersonaResponse: Updated persona with base image

    Raises:
        404: Persona not found
        400: Invalid file format or upload error
        500: Internal server error
    """
    try:
        # Get the persona
        persona = await persona_service.get_persona(persona_id)
        if not persona:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Persona {persona_id} not found",
            )

        # Validate file type
        allowed_types = ["image/png", "image/jpeg", "image/jpg", "image/webp"]
        if file.content_type not in allowed_types:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid file type. Allowed: {', '.join(allowed_types)}",
            )

        # Read file data
        image_data = await file.read()
        
        # Log the size immediately after reading
        logger.info(f"Read {len(image_data)} bytes from uploaded file '{file.filename}'")

        # Validate file size (max 10MB)
        max_size = 10 * 1024 * 1024  # 10MB
        if len(image_data) > max_size:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="File too large. Maximum size is 10MB",
            )

        # Save image to disk
        file_extension = file.filename.split(".")[-1] if "." in file.filename else "png"
        custom_filename = f"persona_{persona_id}_uploaded.{file_extension}"
        
        logger.info(f"Saving image to disk: {len(image_data)} bytes as {custom_filename}")
        image_path = await persona_service._save_image_to_disk(
            persona_id=persona_id, image_data=image_data, filename=custom_filename
        )
        logger.info(f"Image saved successfully to: {image_path}")

        # Update persona with image path and status
        from backend.models.persona import PersonaUpdate

        updates = PersonaUpdate(
            base_image_path=image_path, base_image_status=BaseImageStatus.DRAFT
        )
        updated_persona = await persona_service.update_persona(persona_id, updates)

        logger.info(f"Uploaded seed image for persona {persona_id}")
        return updated_persona

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to upload seed image for persona {persona_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to upload image: {str(e)}",
        )


@router.post("/{persona_id}/seed-image/generate-cloud", response_model=PersonaResponse)
async def generate_seed_image_cloud(
    persona_id: str,
    persona_service: PersonaService = Depends(get_persona_service),
):
    """
    Generate a seed image using DALL-E 3 (Method 2: Cloud Generation).

    Triggers asynchronous generation using the persona's appearance and
    personality via OpenAI DALL-E 3 API. Sets base_image_status to DRAFT.

    Args:
        persona_id: The persona to generate the image for
        persona_service: Injected persona service

    Returns:
        PersonaResponse: Updated persona with generated base image

    Raises:
        404: Persona not found
        400: Missing appearance description or API key
        500: Generation error
    """
    try:
        # Get the persona
        persona = await persona_service.get_persona(persona_id)
        if not persona:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Persona {persona_id} not found",
            )

        # Check if persona has appearance description
        appearance_prompt = persona.base_appearance_description or persona.appearance
        if not appearance_prompt:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Persona must have an appearance description to generate image",
            )

        # Initialize AI model manager
        ai_manager = AIModelManager()
        await ai_manager.initialize_models()

        # Check if DALL-E is available
        dalle_available = any(
            m.get("name") == "dall-e-3" and m.get("provider") == "openai"
            for m in ai_manager.available_models.get("image", [])
        )

        if not dalle_available:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="DALL-E 3 is not available. Check OPENAI_API_KEY configuration",
            )

        # Generate reference image
        logger.info(f"Generating cloud reference image for persona {persona_id}")
        result = await ai_manager._generate_reference_image_openai(
            appearance_prompt=appearance_prompt,
            personality_context=(
                persona.personality[:200] if persona.personality else None
            ),
            quality="hd",
            size="1024x1024",
        )

        # Save image to disk
        image_path = await persona_service._save_image_to_disk(
            persona_id=persona_id,
            image_data=result["image_data"],
            filename=f"persona_{persona_id}_dalle3.png",
        )

        # Update persona with image path and status
        from backend.models.persona import PersonaUpdate

        updates = PersonaUpdate(
            base_image_path=image_path, base_image_status=BaseImageStatus.DRAFT
        )
        updated_persona = await persona_service.update_persona(persona_id, updates)

        logger.info(f"Generated cloud seed image for persona {persona_id}")

        # Clean up
        await ai_manager.close()

        return updated_persona

    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            f"Failed to generate cloud seed image for persona {persona_id}: {str(e)}"
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate image: {str(e)}",
        )


@router.post("/{persona_id}/seed-image/generate-local", response_model=PersonaResponse)
async def generate_seed_image_local(
    persona_id: str,
    persona_service: PersonaService = Depends(get_persona_service),
):
    """
    Generate a seed image using local Stable Diffusion (Method 3: Local Generation).

    Triggers generation using local ROCm/MI25 hardware with Stable Diffusion.
    Can use existing draft image with ControlNet for refinement.
    Sets base_image_status to DRAFT.

    Args:
        persona_id: The persona to generate the image for
        persona_service: Injected persona service

    Returns:
        PersonaResponse: Updated persona with generated base image

    Raises:
        404: Persona not found
        400: Missing appearance description or no local models
        500: Generation error
    """
    try:
        # Get the persona
        persona = await persona_service.get_persona(persona_id)
        if not persona:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Persona {persona_id} not found",
            )

        # Check if persona has appearance description
        appearance_prompt = persona.base_appearance_description or persona.appearance
        if not appearance_prompt:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Persona must have an appearance description to generate image",
            )

        # Initialize AI model manager
        ai_manager = AIModelManager()
        await ai_manager.initialize_models()

        # Check if local models are available
        local_models = [
            m
            for m in ai_manager.available_models.get("image", [])
            if m.get("provider") == "local" and m.get("loaded")
        ]

        if not local_models:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No local image generation models available",
            )

        # Generate reference image (use existing base_image_path for ControlNet if available)
        logger.info(f"Generating local reference image for persona {persona_id}")
        result = await ai_manager._generate_reference_image_local(
            appearance_prompt=appearance_prompt,
            personality_context=(
                persona.personality[:200] if persona.personality else None
            ),
            reference_image_path=(
                persona.base_image_path if persona.base_image_path else None
            ),
            width=1024,
            height=1024,
            num_inference_steps=50,
        )

        # Save image to disk
        image_path = await persona_service._save_image_to_disk(
            persona_id=persona_id,
            image_data=result["image_data"],
            filename=f"persona_{persona_id}_local.png",
        )

        # Update persona with image path and status
        from backend.models.persona import PersonaUpdate

        updates = PersonaUpdate(
            base_image_path=image_path, base_image_status=BaseImageStatus.DRAFT
        )
        updated_persona = await persona_service.update_persona(persona_id, updates)

        logger.info(f"Generated local seed image for persona {persona_id}")

        # Clean up
        await ai_manager.close()

        return updated_persona

    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            f"Failed to generate local seed image for persona {persona_id}: {str(e)}"
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate image: {str(e)}",
        )


@router.post("/{persona_id}/seed-image/approve", response_model=PersonaResponse)
async def approve_seed_image(
    persona_id: str,
    persona_service: PersonaService = Depends(get_persona_service),
):
    """
    Approve the baseline seed image for a persona.

    Sets base_image_status to APPROVED and appearance_locked to True,
    enabling visual consistency features for all future content generation.

    Args:
        persona_id: The persona to approve the image for
        persona_service: Injected persona service

    Returns:
        PersonaResponse: Updated persona with approved base image

    Raises:
        404: Persona not found
        400: No base image to approve
        500: Internal server error
    """
    try:
        # Approve the baseline image
        persona = await persona_service.approve_baseline(persona_id)

        if not persona:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Persona {persona_id} not found",
            )

        logger.info(f"Approved seed image for persona {persona_id}")
        return persona

    except ValueError as e:
        logger.warning(f"Cannot approve baseline for persona {persona_id}: {str(e)}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to approve seed image for persona {persona_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error",
        )


@router.post("/generate-sample-images")
async def generate_sample_images(
    appearance: str = Query(
        ..., description="Appearance description for image generation"
    ),
    personality: Optional[str] = Query(
        None, description="Personality context for image generation"
    ),
    resolution: Optional[str] = Query(
        "1024x1024",
        description="Image resolution (e.g., '1024x1024', '720x1280', '1920x1080')",
    ),
    quality: Optional[str] = Query(
        "standard", description="Generation quality: draft, standard, high, premium"
    ),
    style: Optional[str] = Query(
        "photorealistic",
        description="Image generation style: photorealistic, anime, cartoon, artistic, 3d_render, fantasy, cinematic",
    ),
    persona_service: PersonaService = Depends(get_persona_service),
) -> Dict[str, Any]:
    """
    Generate 4 sample images for persona creation.

    This endpoint generates 4 different sample images based on the appearance
    and personality descriptions. Returns base64-encoded images for immediate
    display in the UI. Used during persona creation to let users choose their
    preferred base image.

    Args:
        appearance: Physical appearance description
        personality: Optional personality traits
        resolution: Image resolution in format "widthxheight" (e.g., "1024x1024", "720x1280", "1920x1080")
        quality: Generation quality preset (draft, standard, high, premium)
        style: Image generation style (photorealistic, anime, cartoon, artistic, 3d_render, fantasy, cinematic)
        persona_service: Injected persona service

    Returns:
        Dict with 'images' array containing objects with 'id', 'data_url', and 'path'

    Raises:
        400: Missing appearance or no image generation available
        500: Generation error
    """
    try:
        if not appearance or len(appearance.strip()) < 10:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Appearance description must be at least 10 characters",
            )

        # Initialize AI model manager
        ai_manager = AIModelManager()
        await ai_manager.initialize_models()

        # Check what's available - prefer local models (free, no API costs)
        local_models = [
            m
            for m in ai_manager.available_models.get("image", [])
            if m.get("provider") == "local" and m.get("loaded")
        ]

        dalle_available = any(
            m.get("name") == "dall-e-3" and m.get("provider") == "openai"
            for m in ai_manager.available_models.get("image", [])
        )

        if not local_models and not dalle_available:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No image generation models available. Install local models (Stable Diffusion XL recommended) or configure OPENAI_API_KEY as fallback.",
            )

        # Parse resolution
        try:
            width, height = map(int, resolution.split("x"))
            if width < 256 or width > 4096 or height < 256 or height > 4096:
                raise ValueError("Resolution dimensions must be between 256 and 4096")
        except (ValueError, AttributeError):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid resolution format. Use 'widthxheight' (e.g., '1024x1024', '720x1280', '1920x1080')",
            )

        # Map quality to DALL-E quality setting
        dalle_quality_map = {
            "draft": "standard",
            "standard": "standard",
            "high": "hd",
            "premium": "hd",
        }
        dalle_quality = dalle_quality_map.get(quality.lower(), "standard")

        # Map quality to local generation steps
        quality_steps_map = {"draft": 20, "standard": 30, "high": 50, "premium": 80}
        num_steps = quality_steps_map.get(quality.lower(), 30)

        logger.info(f"Generating 4 sample images with appearance: {appearance[:50]}...")
        logger.info(f"Resolution: {width}x{height}, Quality: {quality}, Style: {style}")

        # Generate 4 images sequentially to prevent scheduler state conflicts
        images = []

        # Prefer local models (free, no API costs, better privacy)
        # Use DALL-E only as fallback if local models aren't available
        if local_models:
            # Generate with local models, distributing across available GPUs
            logger.info("Using local Stable Diffusion models for generation")

            # Get available GPUs sorted by load (least loaded first)
            from backend.services.gpu_monitoring_service import (
                get_gpu_monitoring_service,
            )

            gpu_service = get_gpu_monitoring_service()
            available_gpus = await gpu_service.get_available_gpus()

            if available_gpus:
                logger.info(
                    f"Distributing image generation across {len(available_gpus)} GPU(s): {available_gpus}"
                )
            else:
                logger.info("No GPU information available, using default GPU selection")

            # Generate images sequentially, but cycle through available GPUs
            # This distributes the workload across multiple GPUs for better utilization
            for i in range(4):
                try:
                    # Select GPU in round-robin fashion if multiple GPUs available
                    device_id = None
                    if available_gpus:
                        device_id = available_gpus[i % len(available_gpus)]
                        logger.info(f"Generating image {i+1}/4 on GPU {device_id}")

                    result = await ai_manager._generate_reference_image_local(
                        appearance_prompt=appearance,
                        personality_context=personality[:200] if personality else None,
                        reference_image_path=None,
                        width=width,
                        height=height,
                        num_inference_steps=num_steps,
                        device_id=device_id,  # Pass GPU selection
                        image_style=style,  # Pass image style
                    )

                    # Convert to base64 data URL
                    base64_image = base64.b64encode(result["image_data"]).decode(
                        "utf-8"
                    )
                    data_url = f"data:image/png;base64,{base64_image}"

                    images.append(
                        {
                            "id": f"sample_{i+1}",
                            "data_url": data_url,
                            "size": len(result["image_data"]),
                            "gpu_id": device_id if device_id is not None else "auto",
                        }
                    )

                    logger.info(f"Generated sample image {i+1}/4")

                except Exception as e:
                    logger.warning(f"Failed to generate image {i+1}: {str(e)}")

        elif dalle_available:
            # Fallback to DALL-E if local models not available (requires API key and costs money)
            logger.warning(
                "Using DALL-E as fallback - this will incur API costs. Consider installing local Stable Diffusion models."
            )
            # Generate 4 images sequentially to avoid rate limits
            for i in range(4):
                try:
                    result = await ai_manager._generate_reference_image_openai(
                        appearance_prompt=appearance,
                        personality_context=personality[:200] if personality else None,
                        quality=dalle_quality,
                        size=f"{width}x{height}",
                    )

                    # Convert to base64 data URL
                    base64_image = base64.b64encode(result["image_data"]).decode(
                        "utf-8"
                    )
                    data_url = f"data:image/png;base64,{base64_image}"

                    images.append(
                        {
                            "id": f"sample_{i+1}",
                            "data_url": data_url,
                            "size": len(result["image_data"]),
                        }
                    )

                    logger.info(f"Generated sample image {i+1}/4")

                except Exception as e:
                    logger.warning(f"Failed to generate image {i+1}: {str(e)}")
                    # Continue with other images even if one fails

        # Clean up
        await ai_manager.close()

        if len(images) == 0:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to generate any sample images",
            )

        logger.info(f"Successfully generated {len(images)} sample images")

        return {"images": images, "count": len(images)}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to generate sample images: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate images: {str(e)}",
        )


@router.post(
    "/random", response_model=PersonaResponse, status_code=status.HTTP_201_CREATED
)
async def create_random_persona(
    generate_images: bool = Query(
        False,
        description="If true, generates 4 sample images and auto-selects the first one",
    ),
    resolution: Optional[str] = Query(
        "1024x1024", description="Image resolution if generate_images is true"
    ),
    quality: Optional[str] = Query(
        "standard", description="Generation quality if generate_images is true"
    ),
    persona_service: PersonaService = Depends(get_persona_service),
):
    """
    Create a random persona with randomized appearance, personality, and themes.

    This endpoint generates a complete persona with randomly selected attributes,
    perfect for quick testing, experimentation, or creative inspiration.

    Args:
        generate_images: If true, automatically generates preview images
        resolution: Image resolution for preview generation
        quality: Quality preset for image generation
        persona_service: Injected persona service

    Returns:
        PersonaResponse: The created random persona

    Raises:
        500: Generation error
    """
    try:
        from backend.services.ai_persona_generator import get_ai_persona_generator

        # Generate AI-powered persona configuration using llama.cpp
        # This creates more coherent and realistic personas than random templates
        ai_generator = get_ai_persona_generator()
        random_config = await ai_generator.generate_persona(
            use_ai=True  # Use llama.cpp for AI-powered generation
        )

        logger.info(f"Creating AI-generated persona: {random_config['name']}")

        # Create persona data object with all fields including soul fields
        persona_data = PersonaCreate(
            name=random_config["name"],
            appearance=random_config["appearance"],
            personality=random_config["personality"],
            content_themes=random_config["content_themes"],
            style_preferences=random_config.get("style_preferences", {}),
            default_content_rating=random_config["default_content_rating"],
            allowed_content_ratings=random_config["allowed_content_ratings"],
            platform_restrictions=random_config.get("platform_restrictions", {}),
            is_active=random_config.get("is_active", True),
            # Soul Fields - Origin & Demographics
            hometown=random_config.get("hometown"),
            current_location=random_config.get("current_location"),
            generation_age=random_config.get("generation_age"),
            education_level=random_config.get("education_level"),
            # Soul Fields - Psychological Profile
            mbti_type=random_config.get("mbti_type"),
            enneagram_type=random_config.get("enneagram_type"),
            political_alignment=random_config.get("political_alignment"),
            risk_tolerance=random_config.get("risk_tolerance"),
            optimism_cynicism_scale=random_config.get("optimism_cynicism_scale"),
            # Soul Fields - Voice & Speech Patterns
            linguistic_register=random_config.get("linguistic_register"),
            typing_quirks=random_config.get("typing_quirks", {}),
            signature_phrases=random_config.get("signature_phrases", []),
            trigger_topics=random_config.get("trigger_topics", []),
            # Soul Fields - Backstory & Lore
            day_job=random_config.get("day_job"),
            war_story=random_config.get("war_story"),
            vices_hobbies=random_config.get("vices_hobbies", []),
            # Soul Fields - Anti-Pattern
            forbidden_phrases=random_config.get("forbidden_phrases", []),
            warmth_level=random_config.get("warmth_level"),
            patience_level=random_config.get("patience_level"),
        )

        # Create the persona
        persona = await persona_service.create_persona(persona_data)

        # Optionally generate and set preview images
        if generate_images:
            try:
                logger.info(
                    f"Generating preview images for random persona {persona.id}"
                )

                # Initialize AI model manager
                ai_manager = AIModelManager()
                await ai_manager.initialize_models()

                # Check for local models
                local_models = [
                    m
                    for m in ai_manager.available_models.get("image", [])
                    if m.get("provider") == "local" and m.get("loaded")
                ]

                if local_models:
                    # Parse resolution
                    width, height = map(int, resolution.split("x"))

                    # Map quality to steps
                    quality_steps = {
                        "draft": 20,
                        "standard": 30,
                        "high": 50,
                        "premium": 80,
                    }
                    num_steps = quality_steps.get(quality.lower(), 30)

                    # Generate first image only (faster for random personas)
                    result = await ai_manager._generate_reference_image_local(
                        appearance_prompt=persona.appearance,
                        personality_context=persona.personality[:200],
                        reference_image_path=None,
                        width=width,
                        height=height,
                        num_inference_steps=num_steps,
                    )

                    # Save and set as base image
                    image_path = await persona_service._save_image_to_disk(
                        persona_id=str(persona.id),
                        image_data=result["image_data"],
                        filename=f"persona_{persona.id}_random.png",
                    )

                    # Update persona with image
                    from backend.models.persona import PersonaUpdate

                    updates = PersonaUpdate(
                        base_image_path=image_path,
                        base_image_status=BaseImageStatus.APPROVED,
                        appearance_locked=True,
                    )
                    persona = await persona_service.update_persona(
                        str(persona.id), updates
                    )

                    logger.info(
                        f"Generated and set base image for random persona {persona.id}"
                    )

                    await ai_manager.close()
                else:
                    logger.warning("No local models available for image generation")

            except Exception as e:
                logger.error(f"Failed to generate images for random persona: {str(e)}")
                # Don't fail the whole operation if image generation fails

        logger.info(f"Random persona created: {persona.id} - {persona.name}")
        return persona

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to create random persona: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create random persona: {str(e)}",
        )


class GenerateSoulFieldsRequest(BaseModel):
    """Request model for generating soul fields."""
    
    name: str = Field(..., description="Persona name for context")
    appearance: str = Field(default="", description="Appearance description for context")
    personality: str = Field(default="", description="Personality description for context")
    content_rating: str = Field(
        default="sfw",
        description="Content rating: sfw, moderate, or nsfw - affects generation style"
    )


class GenerateSoulFieldsResponse(BaseModel):
    """Response model for generated soul fields."""
    
    # Origin & Demographics
    hometown: Optional[str] = None
    current_location: Optional[str] = None
    generation_age: Optional[str] = None
    education_level: Optional[str] = None
    
    # Psychological Profile
    mbti_type: Optional[str] = None
    enneagram_type: Optional[str] = None
    political_alignment: Optional[str] = None
    risk_tolerance: Optional[str] = None
    optimism_cynicism_scale: Optional[int] = None
    
    # Voice & Speech Patterns
    linguistic_register: Optional[str] = None
    typing_quirks: Optional[Dict[str, Any]] = None
    signature_phrases: Optional[List[str]] = None
    trigger_topics: Optional[List[str]] = None
    
    # Backstory & Lore
    day_job: Optional[str] = None
    war_story: Optional[str] = None
    vices_hobbies: Optional[List[str]] = None
    
    # Anti-Pattern
    forbidden_phrases: Optional[List[str]] = None
    warmth_level: Optional[str] = None
    patience_level: Optional[str] = None
    
    # Generation metadata
    generation_method: str = "ollama_ai"
    model_used: Optional[str] = None


@router.post("/generate-soul-fields", response_model=GenerateSoulFieldsResponse)
async def generate_soul_fields(
    request: GenerateSoulFieldsRequest,
):
    """
    Generate soul fields for a persona using AI (Ollama with dolphin-mixtral).
    
    This endpoint generates comprehensive soul fields based on the persona's name,
    appearance, personality, and content rating. The generated fields will match
    the tone appropriate for the content rating:
    
    - **SFW**: Family-friendly, professional persona attributes
    - **Moderate**: More edgy, adult humor, but not explicit
    - **NSFW**: Uninhibited, adult-oriented persona attributes
    
    Args:
        request: Persona context and content rating for generation
        
    Returns:
        GenerateSoulFieldsResponse: Generated soul fields ready to populate form
        
    Raises:
        503: Ollama not available
        500: Generation error
    """
    try:
        # Ollama configuration
        OLLAMA_BASE_URL = "http://localhost:11434"
        
        # Check if Ollama is available
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(f"{OLLAMA_BASE_URL}/api/tags")
                if response.status_code != 200:
                    raise HTTPException(
                        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                        detail="Ollama is not available. Please ensure Ollama is running with dolphin-mixtral model.",
                    )
                
                data = response.json()
                available_models = [m["name"] for m in data.get("models", [])]
                
                # Find best model
                preferred_models = [
                    "dolphin-mixtral",
                    "dolphin-mixtral:8x7b",
                    "dolphin-mistral",
                    "llama3.1:8b",
                    "llama3:8b",
                    "mistral",
                ]
                
                model_to_use = None
                for preferred in preferred_models:
                    for available in available_models:
                        if preferred in available.lower():
                            model_to_use = available
                            break
                    if model_to_use:
                        break
                
                if not model_to_use and available_models:
                    model_to_use = available_models[0]
                
                if not model_to_use:
                    raise HTTPException(
                        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                        detail="No suitable models found in Ollama. Please install dolphin-mixtral.",
                    )
                    
        except httpx.RequestError:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Cannot connect to Ollama. Please ensure Ollama is running on localhost:11434",
            )
        
        # Build content-rating-aware prompt
        rating = request.content_rating.lower()
        
        rating_guidance = ""
        if rating == "nsfw":
            rating_guidance = """
This persona is for ADULT/NSFW content. Generate edgy, uninhibited, provocative attributes.
- They can have explicit interests, risqué humor, and adult-oriented backstory
- Signature phrases can include innuendo or adult references
- Hobbies can include adult activities
- They should feel like a confident, uninhibited adult content creator
- NO corporate speak, NO sanitized language - make them feel REAL and SPICY
"""
        elif rating == "moderate":
            rating_guidance = """
This persona is for MODERATE/edgy content. Generate slightly edgy, adult-humor-friendly attributes.
- They can have mild adult humor, occasional crude jokes
- More rebellious or edgy personality traits
- Think "R-rated movie" level - suggestive but not explicit
- They should feel like a fun, slightly rebellious influencer
"""
        else:  # SFW
            rating_guidance = """
This persona is for SFW/family-friendly content. Generate wholesome, professional attributes.
- Keep everything appropriate for general audiences
- Focus on positive, inspiring, or entertaining traits
- No crude humor or adult references
- They should feel like a mainstream, brand-safe influencer
"""

        prompt = f"""You are an expert at creating realistic AI influencer personas that feel like real people.

Generate soul field attributes for this persona:
Name: {request.name}
Appearance: {request.appearance or 'Not specified'}
Personality: {request.personality or 'Not specified'}

{rating_guidance}

# CRITICAL RULES
1. Make them feel HUMAN, not like an AI or customer service bot
2. Give them strong opinions, quirks, and personality
3. Everything should feel internally consistent with the name/appearance/personality
4. The forbidden_phrases should include things this specific persona would NEVER say
5. Signature phrases should feel natural for their background and style

# OUTPUT FORMAT - Respond with ONLY valid JSON (no markdown, no explanation):

{{
  "hometown": "specific city/region they're from with character",
  "current_location": "where they live now and why they moved/stayed",
  "generation_age": "generation with age and context (e.g., 'Gen Z - 24, grew up online')",
  "education_level": "their education with personality (e.g., 'State school dropout, self-taught')",
  
  "mbti_type": "XXXX - The Label (e.g., 'ESTP - The Entrepreneur')",
  "enneagram_type": "Type X - The Label (e.g., 'Type 7 - The Enthusiast')",
  "political_alignment": "their worldview in their own words",
  "risk_tolerance": "their attitude in their own voice",
  "optimism_cynicism_scale": 7,
  
  "linguistic_register": "gen_z|millennial|southern|tech_bro|street|corporate|academic|blue_collar",
  "typing_quirks": {{
    "capitalization": "how they type (all lowercase, normal, RANDOM CAPS)",
    "emoji_usage": "none|minimal|moderate|heavy",
    "punctuation": "their punctuation style"
  }},
  "signature_phrases": ["phrase 1", "phrase 2", "phrase 3", "phrase 4", "phrase 5"],
  "trigger_topics": ["topic that fires them up 1", "topic 2", "topic 3"],
  
  "day_job": "what they do for work/money",
  "war_story": "one defining life moment that shaped who they are",
  "vices_hobbies": ["hobby 1", "hobby 2", "hobby 3", "hobby 4"],
  
  "forbidden_phrases": ["phrase they'd NEVER say 1", "phrase 2", "phrase 3", "phrase 4", "phrase 5"],
  "warmth_level": "cold|neutral|warm|buddy",
  "patience_level": "short_fuse|normal|patient|infinite"
}}

JSON:"""

        # Generate with Ollama (60 second timeout for reasonable user experience)
        logger.info(f"Generating soul fields for '{request.name}' with {model_to_use} (rating: {rating})")
        
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                f"{OLLAMA_BASE_URL}/api/generate",
                json={
                    "model": model_to_use,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.9,
                        "top_p": 0.95,
                        "num_predict": 2000,
                    }
                }
            )
            
            if response.status_code != 200:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"Ollama generation failed with status {response.status_code}",
                )
            
            output = response.json().get("response", "")
        
        # Parse JSON from output - find outermost { } block
        start_idx = output.find('{')
        end_idx = output.rfind('}')
        
        if start_idx == -1 or end_idx == -1:
            logger.error(f"No JSON found in Ollama output: {output[:500]}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to parse AI response - no valid JSON found",
            )
        
        json_str = output[start_idx:end_idx + 1]
        
        try:
            soul_data = json.loads(json_str)
        except json.JSONDecodeError as e:
            logger.error(f"JSON parse error: {e}, content: {json_str[:500]}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to parse AI response - invalid JSON",
            )
        
        # Validate and normalize the response
        optimism_scale = soul_data.get('optimism_cynicism_scale', 5)
        if isinstance(optimism_scale, (int, float)):
            optimism_scale = max(1, min(10, int(optimism_scale)))
        else:
            optimism_scale = 5
        
        # Ensure lists are lists
        def ensure_list(val, default=None):
            if default is None:
                default = []
            if isinstance(val, list):
                return val
            return default
        
        # Ensure dict
        typing_quirks = soul_data.get('typing_quirks', {})
        if not isinstance(typing_quirks, dict):
            typing_quirks = {}
        
        # Map linguistic register
        register_str = soul_data.get('linguistic_register', 'blue_collar').lower()
        valid_registers = ['blue_collar', 'academic', 'tech_bro', 'street', 'corporate', 'southern', 'millennial', 'gen_z']
        if register_str not in valid_registers:
            register_str = 'blue_collar'
        
        # Map warmth level
        warmth_str = soul_data.get('warmth_level', 'warm').lower()
        valid_warmth = ['cold', 'neutral', 'warm', 'buddy']
        if warmth_str not in valid_warmth:
            warmth_str = 'warm'
        
        # Map patience level
        patience_str = soul_data.get('patience_level', 'normal').lower()
        valid_patience = ['short_fuse', 'normal', 'patient', 'infinite']
        if patience_str not in valid_patience:
            patience_str = 'normal'
        
        result = GenerateSoulFieldsResponse(
            hometown=soul_data.get('hometown'),
            current_location=soul_data.get('current_location'),
            generation_age=soul_data.get('generation_age'),
            education_level=soul_data.get('education_level'),
            mbti_type=soul_data.get('mbti_type'),
            enneagram_type=soul_data.get('enneagram_type'),
            political_alignment=soul_data.get('political_alignment'),
            risk_tolerance=soul_data.get('risk_tolerance'),
            optimism_cynicism_scale=optimism_scale,
            linguistic_register=register_str,
            typing_quirks=typing_quirks,
            signature_phrases=ensure_list(soul_data.get('signature_phrases')),
            trigger_topics=ensure_list(soul_data.get('trigger_topics')),
            day_job=soul_data.get('day_job'),
            war_story=soul_data.get('war_story'),
            vices_hobbies=ensure_list(soul_data.get('vices_hobbies')),
            forbidden_phrases=ensure_list(soul_data.get('forbidden_phrases')),
            warmth_level=warmth_str,
            patience_level=patience_str,
            generation_method="ollama_ai",
            model_used=model_to_use,
        )
        
        logger.info(f"Successfully generated soul fields for '{request.name}' using {model_to_use}")
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to generate soul fields: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate soul fields: {str(e)}",
        )


@router.post("/{persona_id}/set-base-image")
async def set_base_image_from_sample(
    persona_id: str,
    image_data: Dict[str, str],
    persona_service: PersonaService = Depends(get_persona_service),
) -> PersonaResponse:
    """
    Set a persona's base image from a generated sample.

    Takes a base64-encoded image (from the sample generation) and sets it
    as the persona's base image, then locks the appearance.

    Args:
        persona_id: The persona to update
        image_data: Request body with 'image_data' key containing base64 encoded image (with or without data URL prefix)
        persona_service: Injected persona service

    Returns:
        PersonaResponse: Updated persona with locked base image

    Raises:
        404: Persona not found
        400: Invalid image data
        500: Internal server error
    """
    try:
        # Get the persona
        persona = await persona_service.get_persona(persona_id)
        if not persona:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Persona {persona_id} not found",
            )

        # Extract image_data from request body
        image_data_str = image_data.get("image_data")
        if not image_data_str:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Missing 'image_data' in request body",
            )

        # Remove data URL prefix if present
        if image_data_str.startswith("data:image"):
            image_data_str = image_data_str.split(",", 1)[1]

        # Decode base64
        try:
            decoded_image = base64.b64decode(image_data_str)
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid base64 image data: {str(e)}",
            )

        # Validate size
        max_size = 10 * 1024 * 1024  # 10MB
        if len(decoded_image) > max_size:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Image too large. Maximum size is 10MB",
            )

        # Save image to disk
        image_path = await persona_service._save_image_to_disk(
            persona_id=persona_id,
            image_data=decoded_image,
            filename=f"persona_{persona_id}_base.png",
        )

        # Update persona with image path, approve it, and lock appearance
        from backend.models.persona import PersonaUpdate

        updates = PersonaUpdate(
            base_image_path=image_path,
            base_image_status=BaseImageStatus.APPROVED,
            appearance_locked=True,
        )
        updated_persona = await persona_service.update_persona(persona_id, updates)

        logger.info(f"Set and locked base image for persona {persona_id}")
        return updated_persona

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to set base image for persona {persona_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to set base image: {str(e)}",
        )


@router.get("/{persona_id}/base-image-url")
async def get_base_image_url(
    persona_id: str,
    persona_service: PersonaService = Depends(get_persona_service),
) -> Dict[str, Any]:
    """
    Get the URL for a persona's base image.
    
    Returns the URL path to access the persona's locked base image,
    or None if no base image is set.
    
    Args:
        persona_id: The persona to retrieve the base image URL for
        persona_service: Injected persona service
        
    Returns:
        Dict with base_image_url field (or None if no image), appearance_locked (bool), and base_image_status (str)
        
    Raises:
        404: Persona not found
        500: Internal server error
    """
    try:
        # Get the persona
        persona = await persona_service.get_persona(persona_id)
        if not persona:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Persona {persona_id} not found",
            )
        
        # Check if base image exists
        if not persona.base_image_path:
            return {
                "base_image_url": None,
                "appearance_locked": bool(persona.appearance_locked),
                "base_image_status": str(persona.base_image_status),
            }
        
        # Convert file path to URL path
        # base_image_path is like: /opt/gator/data/models/base_images/persona_xxx_base.png
        # We need to convert to: /base_images/persona_xxx_base.png
        from pathlib import Path
        image_filename = Path(persona.base_image_path).name
        base_image_url = f"/base_images/{image_filename}"
        
        return {
            "base_image_url": base_image_url,
            "appearance_locked": bool(persona.appearance_locked),
            "base_image_status": str(persona.base_image_status),
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get base image URL for persona {persona_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get base image URL: {str(e)}",
        )


# Pydantic models for persona chat testing
class PersonaChatRequest(BaseModel):
    """Request model for persona chat testing."""
    message: str = Field(..., min_length=1, max_length=2000, description="User message to the persona")


class PersonaChatResponse(BaseModel):
    """Response model for persona chat testing."""
    response: str = Field(..., description="Persona's response")
    persona_name: str = Field(..., description="Name of the persona responding")
    model_used: str = Field(..., description="AI model used for generation")
    timestamp: str = Field(..., description="Response timestamp")


@router.post("/{persona_id}/chat", response_model=PersonaChatResponse)
async def chat_with_persona(
    persona_id: UUID,
    chat_request: PersonaChatRequest,
    persona_service: PersonaService = Depends(get_persona_service),
):
    """
    Test chat with a persona using the dolphin-mixtral model.
    
    This endpoint allows testing persona consistency by chatting with
    the persona using its configured appearance, personality, and filters.
    Uses the dolphin-mixtral model (uncensored) for unrestricted responses.
    
    Args:
        persona_id: UUID of the persona to chat with
        chat_request: The user message
        persona_service: Injected persona service
        
    Returns:
        PersonaChatResponse: The persona's response
    """
    from datetime import datetime
    
    try:
        # Get the persona data
        persona = await persona_service.get_persona(str(persona_id))
        if not persona:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Persona {persona_id} not found",
            )
        
        # Get AI models manager
        from backend.services.ai_models import ai_models
        
        # Check for Ollama models with dolphin-mixtral preference
        text_models = ai_models.available_models.get("text", [])
        ollama_models = [m for m in text_models if m.get("inference_engine") == "ollama" and m.get("loaded")]
        
        # Prefer dolphin-mixtral for unrestricted persona testing
        selected_model = None
        model_name = "fallback"
        
        for model in ollama_models:
            if "dolphin" in model.get("name", "").lower():
                selected_model = model
                model_name = model.get("name", "dolphin-mixtral")
                break
        
        # Fall back to any available Ollama model if dolphin not found
        if not selected_model and ollama_models:
            selected_model = ollama_models[0]
            model_name = selected_model.get("name", "ollama")
        
        # Fall back to any loaded text model
        if not selected_model:
            loaded_models = [m for m in text_models if m.get("loaded")]
            if loaded_models:
                selected_model = loaded_models[0]
                model_name = selected_model.get("name", "unknown")
        
        if not selected_model:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="No text generation models available. Please install dolphin-mixtral via Ollama.",
            )
        
        # Build persona system prompt
        system_prompt = _build_persona_chat_prompt(persona)
        
        # Generate response using AI models
        full_prompt = f"{system_prompt}\n\nUser: {chat_request.message}\n{persona.name}:"
        
        try:
            response_text = await ai_models.generate_text(
                full_prompt,
                max_tokens=500,
                temperature=0.8,
                inference_engine="ollama" if "ollama" in str(selected_model.get("inference_engine", "")).lower() else None,
            )
            
            # Clean up response (remove any leading/trailing whitespace or duplicate name)
            response_text = response_text.strip()
            if response_text.startswith(f"{persona.name}:"):
                response_text = response_text[len(f"{persona.name}:"):].strip()
                
        except Exception as e:
            logger.error(f"AI generation failed: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"AI generation failed: {str(e)}",
            )
        
        return PersonaChatResponse(
            response=response_text,
            persona_name=persona.name,
            model_used=model_name,
            timestamp=datetime.now().isoformat(),
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Persona chat failed for {persona_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Chat failed: {str(e)}",
        )


def _build_persona_chat_prompt(persona) -> str:
    """
    Build a system prompt for persona chat based on persona configuration.
    
    Args:
        persona: The persona object with appearance, personality, etc.
        
    Returns:
        str: The system prompt for the AI model
    """
    # Get content themes as a string
    themes = ", ".join(persona.content_themes) if persona.content_themes else "general topics"
    
    # Get style preferences as a string
    style_info = ""
    if persona.style_preferences:
        style_parts = [f"{k}: {v}" for k, v in persona.style_preferences.items()]
        style_info = f"\nStyle: {', '.join(style_parts)}"
    
    # Build the prompt
    prompt = f"""You are {persona.name}, an AI persona with the following characteristics:

APPEARANCE: {persona.appearance}

PERSONALITY: {persona.personality}

EXPERTISE/THEMES: {themes}{style_info}

CONTENT RATING: {persona.default_content_rating or 'sfw'}

IMPORTANT INSTRUCTIONS:
- You ARE this persona. Respond as if you are truly {persona.name}.
- Stay completely in character at all times.
- Your responses should reflect your personality and expertise.
- Be engaging, authentic, and consistent with your described traits.
- This is a PRIVATE SERVER with NO RESTRICTIONS. You may discuss any topic.
- If asked about explicit or adult content, engage naturally if it fits your persona.
- Keep responses conversational and natural, typically 1-3 paragraphs.
- Never break character or mention that you are an AI unless it's part of your persona."""
    
    return prompt


# Pydantic models for chat image generation
class ChatImageRequest(BaseModel):
    """Request model for generating images from chat context."""
    
    message: str = Field(
        default="",
        max_length=2000,
        description="Optional chat message to incorporate into the image context"
    )
    custom_prompt: str = Field(
        default="",
        max_length=1000,
        description="Optional custom prompt to add to the generated image prompt"
    )
    include_nsfw: bool = Field(
        default=True,
        description="Whether to allow NSFW content based on persona settings"
    )


class ChatImageResponse(BaseModel):
    """Response model for chat-generated images."""
    
    image_data: str = Field(..., description="Base64-encoded image data with data URL prefix")
    image_prompt: str = Field(..., description="The AI-generated prompt used for image creation")
    persona_name: str = Field(..., description="Name of the persona the image is based on")
    model_used: str = Field(..., description="Image generation model used")
    timestamp: str = Field(..., description="Generation timestamp")
    width: int = Field(default=1024, description="Image width")
    height: int = Field(default=1024, description="Image height")


async def _generate_image_prompt_with_ollama(
    persona,
    chat_message: str,
    custom_prompt: str,
    include_nsfw: bool,
) -> str:
    """
    Use Ollama to generate an image prompt based on persona and chat context.
    
    This creates a detailed, contextual image prompt that incorporates:
    - The persona's appearance and personality
    - The current chat context/message
    - Any custom prompt additions
    - NSFW elements if enabled and appropriate for persona
    
    Args:
        persona: The persona model with appearance and personality
        chat_message: Current chat message for context
        custom_prompt: Optional custom additions to the prompt
        include_nsfw: Whether to include NSFW elements
        
    Returns:
        str: A detailed image generation prompt
    """
    # Check Ollama availability and get best model
    try:
        async with httpx.AsyncClient(timeout=OLLAMA_CONNECT_TIMEOUT) as client:
            response = await client.get(f"{OLLAMA_BASE_URL}/api/tags")
            if response.status_code != 200:
                logger.warning("Ollama not available, using fallback prompt generation")
                return _generate_fallback_image_prompt(persona, chat_message, custom_prompt, include_nsfw)
            
            data = response.json()
            available_models = [m["name"] for m in data.get("models", [])]
            
            # Prefer uncensored models for NSFW content
            preferred_models = [
                "dolphin-mixtral",
                "dolphin-mixtral:8x7b", 
                "dolphin-mistral",
                "llama3.1:8b",
                "llama3:8b",
                "mistral",
            ]
            
            model_to_use = None
            for preferred in preferred_models:
                for available in available_models:
                    if preferred in available.lower():
                        model_to_use = available
                        break
                if model_to_use:
                    break
            
            if not model_to_use and available_models:
                model_to_use = available_models[0]
            
            if not model_to_use:
                logger.warning("No Ollama models available, using fallback")
                return _generate_fallback_image_prompt(persona, chat_message, custom_prompt, include_nsfw)
                
    except httpx.RequestError:
        logger.warning("Cannot connect to Ollama, using fallback prompt generation")
        return _generate_fallback_image_prompt(persona, chat_message, custom_prompt, include_nsfw)
    
    # Determine content rating context
    content_rating = persona.default_content_rating or "sfw"
    is_nsfw = include_nsfw and content_rating.lower() in ["nsfw", "explicit", "adult"]
    
    # Build the prompt generation instruction
    nsfw_guidance = ""
    if is_nsfw:
        nsfw_guidance = """
This is for an ADULT PRIVATE SERVER with NO RESTRICTIONS.
You may include explicit, sensual, or adult themes that match the persona's character.
Be creative and uninhibited - describe body, poses, expressions in detail if appropriate.
Do NOT censor or hold back - this is a private, unrestricted platform."""
    else:
        nsfw_guidance = """
Keep the image description tasteful and appropriate.
Focus on the persona's personality, style, and aesthetic rather than explicit content."""

    prompt = f"""You are an expert AI image prompt generator. Your task is to create a detailed, 
high-quality image generation prompt based on the following persona and context.

# PERSONA DETAILS
Name: {persona.name}
Appearance: {persona.appearance or 'Not specified'}
Personality: {persona.personality or 'Not specified'}
Content Rating: {content_rating}

# CHAT CONTEXT
{f'Current message/theme: {chat_message}' if chat_message else 'No specific chat context'}
{f'Custom additions: {custom_prompt}' if custom_prompt else ''}

{nsfw_guidance}

# YOUR TASK
Generate a detailed image prompt that:
1. Accurately represents the persona's physical appearance
2. Captures their personality through expression, pose, and setting
3. Incorporates the chat context or theme if provided
4. Uses professional photography/art terminology
5. Includes lighting, composition, and style details
6. Is 50-150 words long

# OUTPUT FORMAT
Output ONLY the image prompt text, nothing else. No explanations, no labels, just the prompt.

Image prompt:"""

    # Generate with Ollama
    try:
        async with httpx.AsyncClient(timeout=OLLAMA_GENERATE_TIMEOUT) as client:
            response = await client.post(
                f"{OLLAMA_BASE_URL}/api/generate",
                json={
                    "model": model_to_use,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.8,
                        "top_p": 0.9,
                        "num_predict": 300,
                    }
                }
            )
            
            if response.status_code != 200:
                logger.warning(f"Ollama generation failed: {response.status_code}")
                return _generate_fallback_image_prompt(persona, chat_message, custom_prompt, include_nsfw)
            
            output = response.json().get("response", "").strip()
            
            # Clean up the output (remove any meta-text)
            if output:
                # Remove common prefixes that might appear
                for prefix in ["Image prompt:", "Prompt:", "Here is the prompt:", "Here's the image prompt:"]:
                    if output.lower().startswith(prefix.lower()):
                        output = output[len(prefix):].strip()
                
                logger.info(f"Generated image prompt with Ollama ({model_to_use}): {output[:100]}...")
                return output
            
    except Exception as e:
        logger.error(f"Ollama image prompt generation failed: {e}")
    
    return _generate_fallback_image_prompt(persona, chat_message, custom_prompt, include_nsfw)


def _generate_fallback_image_prompt(
    persona,
    chat_message: str,
    custom_prompt: str,
    include_nsfw: bool,
) -> str:
    """
    Generate a fallback image prompt without Ollama.
    
    This creates a basic but functional prompt using the persona's attributes.
    """
    parts = []
    
    # Base description from appearance
    if persona.appearance:
        parts.append(persona.appearance)
    else:
        parts.append(f"Portrait of {persona.name}")
    
    # Add personality-influenced elements
    if persona.personality:
        # Extract key personality traits
        personality_lower = persona.personality.lower()
        if "confident" in personality_lower:
            parts.append("confident pose, direct gaze")
        if "shy" in personality_lower or "introverted" in personality_lower:
            parts.append("soft expression, gentle demeanor")
        if "energetic" in personality_lower or "fun" in personality_lower:
            parts.append("dynamic pose, bright expression")
        if "mysterious" in personality_lower:
            parts.append("enigmatic expression, dramatic lighting")
        if "sensual" in personality_lower or "seductive" in personality_lower:
            parts.append("alluring pose, intimate atmosphere")
    
    # Add chat context
    if chat_message:
        parts.append(f"themed around: {chat_message[:100]}")
    
    # Add custom prompt
    if custom_prompt:
        parts.append(custom_prompt)
    
    # Add quality modifiers
    parts.append("highly detailed, professional photography, studio lighting")
    
    # Add NSFW elements if appropriate
    content_rating = persona.default_content_rating or "sfw"
    if include_nsfw and content_rating.lower() in ["nsfw", "explicit", "adult"]:
        parts.append("artistic nude, sensual, intimate setting")
    
    return ", ".join(parts)


@router.post("/{persona_id}/chat/generate-image", response_model=ChatImageResponse)
async def generate_chat_image(
    persona_id: UUID,
    request: ChatImageRequest,
    persona_service: PersonaService = Depends(get_persona_service),
):
    """
    Generate an image from chat context for a specific persona.
    
    This endpoint uses Ollama to create an intelligent image prompt based on:
    - The persona's appearance and personality
    - The current chat message or theme
    - Any custom prompt additions
    - NSFW settings if enabled for the persona
    
    The generated image can be displayed directly in the chat interface.
    
    This is a PRIVATE SERVER - NSFW content is fully supported when:
    - The persona's content_rating allows it
    - The include_nsfw parameter is True
    
    Args:
        persona_id: UUID of the persona
        request: Chat image request with optional message and custom prompt
        persona_service: Injected persona service
        
    Returns:
        ChatImageResponse: Generated image with base64 data and metadata
        
    Raises:
        404: Persona not found
        503: No image generation models available
        500: Generation failed
    """
    try:
        # Get the persona
        persona = await persona_service.get_persona(str(persona_id))
        if not persona:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Persona {persona_id} not found",
            )
        
        logger.info(f"Generating chat image for persona {persona.name}")
        
        # Generate image prompt using Ollama
        image_prompt = await _generate_image_prompt_with_ollama(
            persona=persona,
            chat_message=request.message,
            custom_prompt=request.custom_prompt,
            include_nsfw=request.include_nsfw,
        )
        
        logger.info(f"Image prompt generated: {image_prompt[:100]}...")
        
        # Initialize AI model manager and generate image
        ai_manager = AIModelManager()
        await ai_manager.initialize_models()
        
        # Check for available image models
        image_models = ai_manager.available_models.get("image", [])
        loaded_models = [m for m in image_models if m.get("loaded")]
        
        if not loaded_models:
            await ai_manager.close()
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="No image generation models available. Please install Stable Diffusion XL or configure image models.",
            )
        
        # Select model (prefer local SDXL)
        selected_model = None
        model_name = "unknown"
        
        # Prefer local models with SDXL
        for model in loaded_models:
            if model.get("provider") == "local" and "xl" in model.get("name", "").lower():
                selected_model = model
                model_name = model.get("name", "sdxl")
                break
        
        # Fall back to any local model
        if not selected_model:
            for model in loaded_models:
                if model.get("provider") == "local":
                    selected_model = model
                    model_name = model.get("name", "local")
                    break
        
        # Fall back to any available model
        if not selected_model:
            selected_model = loaded_models[0]
            model_name = selected_model.get("name", "unknown")
        
        logger.info(f"Using image model: {model_name}")
        
        # Generate the image
        try:
            result = await ai_manager.generate_image(
                prompt=image_prompt,
                width=1024,
                height=1024,
                num_inference_steps=30,
                guidance_scale=7.5,
            )
            
            if not result or not result.get("image_data"):
                await ai_manager.close()
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Image generation failed - no image data returned",
                )
            
            # Convert to base64 data URL
            image_data = result["image_data"]
            base64_image = base64.b64encode(image_data).decode("utf-8")
            data_url = f"data:image/png;base64,{base64_image}"
            
            await ai_manager.close()
            
            return ChatImageResponse(
                image_data=data_url,
                image_prompt=image_prompt,
                persona_name=persona.name,
                model_used=model_name,
                timestamp=datetime.now().isoformat(),
                width=result.get("width", 1024),
                height=result.get("height", 1024),
            )
            
        except Exception as gen_error:
            await ai_manager.close()
            logger.error(f"Image generation failed: {gen_error}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Image generation failed: {str(gen_error)}",
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Chat image generation failed for persona {persona_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate chat image: {str(e)}",
        )
