"""
Persona Management API Routes

Handles AI persona creation, management, and configuration.
"""

from typing import List, Optional, Dict, Any
from uuid import UUID
import os
import base64
import asyncio

from fastapi import APIRouter, Depends, HTTPException, status, Query, UploadFile, File
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
        image_path = await persona_service._save_image_to_disk(
            persona_id=persona_id, image_data=image_data, filename=custom_filename
        )

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
        from backend.services.persona_randomizer import PersonaRandomizer

        # Generate random persona configuration
        random_config = PersonaRandomizer.generate_complete_random_persona(
            detailed=True
        )

        logger.info(f"Creating random persona: {random_config['name']}")

        # Create persona data object
        persona_data = PersonaCreate(
            name=random_config["name"],
            appearance=random_config["appearance"],
            personality=random_config["personality"],
            content_themes=random_config["content_themes"],
            style_preferences=random_config["style_preferences"],
            default_content_rating=random_config["default_content_rating"],
            allowed_content_ratings=random_config["allowed_content_ratings"],
            platform_restrictions=random_config["platform_restrictions"],
            is_active=random_config["is_active"],
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


@router.post("/{persona_id}/set-base-image")
async def set_base_image_from_sample(
    persona_id: str,
    image_data: str = Query(..., description="Base64 encoded image data"),
    persona_service: PersonaService = Depends(get_persona_service),
) -> PersonaResponse:
    """
    Set a persona's base image from a generated sample.

    Takes a base64-encoded image (from the sample generation) and sets it
    as the persona's base image, then locks the appearance.

    Args:
        persona_id: The persona to update
        image_data: Base64 encoded image data (with or without data URL prefix)
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

        # Remove data URL prefix if present
        if image_data.startswith("data:image"):
            image_data = image_data.split(",", 1)[1]

        # Decode base64
        try:
            decoded_image = base64.b64decode(image_data)
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
