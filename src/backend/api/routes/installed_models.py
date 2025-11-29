"""
Installed Model Routes

API endpoints for managing installed AI models, including
CivitAI metadata enrichment, trigger word management, and model lookup.
"""

from typing import List, Optional

from fastapi import APIRouter, HTTPException, Query, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from backend.config.logging import get_logger
from backend.database.connection import get_db_session
from backend.models.installed_model import (
    InstalledModelCreate,
    InstalledModelUpdate,
    InstalledModelResponse,
    TriggerWordResponse,
    ModelsByTriggerResponse,
)
from backend.services.installed_model_service import InstalledModelService
from backend.services.settings_service import get_db_setting

logger = get_logger(__name__)
router = APIRouter(prefix="/installed-models", tags=["installed-models"])


@router.get("/", response_model=List[InstalledModelResponse])
async def list_installed_models(
    model_type: Optional[str] = Query(None, description="Filter by model type (Checkpoint, LORA, etc.)"),
    source: Optional[str] = Query(None, description="Filter by source (civitai, huggingface, local)"),
    base_model: Optional[str] = Query(None, description="Filter by base model (SD 1.5, SDXL 1.0, etc.)"),
    is_active: Optional[bool] = Query(True, description="Filter by active status"),
    limit: int = Query(100, ge=1, le=500, description="Maximum results"),
    offset: int = Query(0, ge=0, description="Results offset"),
    db: AsyncSession = Depends(get_db_session),
):
    """
    List all installed AI models.
    
    Returns models with filtering options for type, source, and base model.
    """
    service = InstalledModelService(db)
    return await service.list_models(
        model_type=model_type,
        source=source,
        base_model=base_model,
        is_active=is_active,
        limit=limit,
        offset=offset,
    )


@router.get("/triggers", response_model=List[TriggerWordResponse])
async def get_all_trigger_words(
    db: AsyncSession = Depends(get_db_session),
):
    """
    Get all unique trigger words from installed models.
    
    Returns aggregated trigger words with the models that use them.
    Useful for building trigger word dropdowns in the UI.
    """
    service = InstalledModelService(db)
    return await service.get_all_trigger_words()


@router.get("/by-trigger/{trigger_word}", response_model=ModelsByTriggerResponse)
async def get_models_by_trigger(
    trigger_word: str,
    model_type: Optional[str] = Query(None, description="Filter by model type"),
    base_model: Optional[str] = Query(None, description="Filter by base model"),
    db: AsyncSession = Depends(get_db_session),
):
    """
    Find models that have a specific trigger word.
    
    Returns all models that can be activated by the given trigger word.
    """
    service = InstalledModelService(db)
    models = await service.find_models_by_trigger(
        trigger_word=trigger_word,
        model_type=model_type,
        base_model=base_model,
    )
    
    return ModelsByTriggerResponse(
        trigger_word=trigger_word,
        models=models,
        total_count=len(models),
    )


@router.get("/{model_id}", response_model=InstalledModelResponse)
async def get_installed_model(
    model_id: str,
    db: AsyncSession = Depends(get_db_session),
):
    """
    Get details of a specific installed model.
    """
    service = InstalledModelService(db)
    model = await service.get_model(model_id)
    
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")
    
    return model


@router.post("/", response_model=InstalledModelResponse)
async def create_installed_model(
    model_data: InstalledModelCreate,
    db: AsyncSession = Depends(get_db_session),
):
    """
    Register a new installed model.
    
    Creates a database record for an installed model with its metadata.
    """
    service = InstalledModelService(db)
    
    # Check if model with same path already exists
    existing = await service.get_model_by_path(model_data.file_path)
    if existing:
        raise HTTPException(
            status_code=400,
            detail=f"Model with path {model_data.file_path} already exists"
        )
    
    return await service.create_model(model_data)


@router.put("/{model_id}", response_model=InstalledModelResponse)
async def update_installed_model(
    model_id: str,
    update_data: InstalledModelUpdate,
    db: AsyncSession = Depends(get_db_session),
):
    """
    Update an installed model's metadata.
    
    Can be used to manually set trigger words, descriptions, or other metadata.
    """
    service = InstalledModelService(db)
    model = await service.update_model(model_id, update_data)
    
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")
    
    return model


@router.delete("/{model_id}")
async def delete_installed_model(
    model_id: str,
    db: AsyncSession = Depends(get_db_session),
):
    """
    Delete an installed model record.
    
    Note: This only removes the database record, not the actual model file.
    """
    service = InstalledModelService(db)
    deleted = await service.delete_model(model_id)
    
    if not deleted:
        raise HTTPException(status_code=404, detail="Model not found")
    
    return {"success": True, "message": "Model record deleted"}


@router.post("/{model_id}/enrich-from-civitai", response_model=InstalledModelResponse)
async def enrich_model_from_civitai(
    model_id: str,
    civitai_version_id: int = Query(..., description="CivitAI model version ID"),
    db: AsyncSession = Depends(get_db_session),
):
    """
    Enrich a model's metadata from CivitAI API.
    
    Fetches detailed information from CivitAI including:
    - Description and display name
    - Trigger/trained words
    - Base model information
    - NSFW status
    - Sample images
    """
    service = InstalledModelService(db)
    
    try:
        model = await service.enrich_from_civitai(model_id, civitai_version_id)
        
        if not model:
            raise HTTPException(status_code=404, detail="Model not found")
        
        return model
        
    except Exception as e:
        logger.error(f"Failed to enrich model from CivitAI: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to fetch CivitAI data: {str(e)}"
        )


@router.post("/{model_id}/enrich-from-huggingface", response_model=InstalledModelResponse)
async def enrich_model_from_huggingface(
    model_id: str,
    huggingface_repo_id: str = Query(..., description="HuggingFace repository ID (e.g., 'stabilityai/stable-diffusion-xl-base-1.0')"),
    db: AsyncSession = Depends(get_db_session),
):
    """
    Enrich a model's metadata from HuggingFace API.
    
    Fetches detailed information from HuggingFace including:
    - Description and display name
    - Base model information
    - Library and pipeline tag
    - Downloads and likes stats
    - Example prompts from model card
    """
    service = InstalledModelService(db)
    
    try:
        model = await service.enrich_from_huggingface(model_id, huggingface_repo_id)
        
        if not model:
            raise HTTPException(status_code=404, detail="Model not found")
        
        return model
        
    except Exception as e:
        logger.error(f"Failed to enrich model from HuggingFace: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to fetch HuggingFace data: {str(e)}"
        )


@router.post("/{model_id}/set-trigger-words", response_model=InstalledModelResponse)
async def set_model_trigger_words(
    model_id: str,
    trigger_words: List[str],
    db: AsyncSession = Depends(get_db_session),
):
    """
    Set trigger words for a model.
    
    Replaces the current trigger words with the provided list.
    """
    service = InstalledModelService(db)
    
    update_data = InstalledModelUpdate(trigger_words=trigger_words)
    model = await service.update_model(model_id, update_data)
    
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")
    
    return model


@router.post("/{model_id}/add-trigger-word", response_model=InstalledModelResponse)
async def add_model_trigger_word(
    model_id: str,
    trigger_word: str = Query(..., description="Trigger word to add"),
    db: AsyncSession = Depends(get_db_session),
):
    """
    Add a trigger word to a model.
    
    Appends a new trigger word without removing existing ones.
    """
    service = InstalledModelService(db)
    
    # Get current model
    model = await service.get_model(model_id)
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")
    
    # Add new trigger word if not already present
    current_triggers = model.trigger_words or []
    if trigger_word.lower().strip() not in [t.lower().strip() for t in current_triggers]:
        current_triggers.append(trigger_word.strip())
    
    update_data = InstalledModelUpdate(trigger_words=current_triggers)
    updated_model = await service.update_model(model_id, update_data)
    
    return updated_model


@router.post("/scan-directory")
async def scan_models_directory(
    directory: Optional[str] = Query(None, description="Directory to scan (default: ai_model_path setting)"),
    model_type: Optional[str] = Query(None, description="Model type for found models"),
    db: AsyncSession = Depends(get_db_session),
):
    """
    Scan a directory for model files and register them.
    
    Finds .safetensors, .ckpt, .pt, .bin files and creates database records.
    Also reads metadata files if present (from CivitAI downloads).
    """
    service = InstalledModelService(db)
    
    # Get directory from settings if not provided
    if not directory:
        directory = await get_db_setting("ai_model_path")
        if not directory:
            directory = "./models"
    
    results = await service.scan_and_register_models(
        models_directory=directory,
        model_type=model_type,
    )
    
    return {
        "success": True,
        "directory": directory,
        **results
    }


@router.post("/increment-usage/{model_id}")
async def increment_model_usage(
    model_id: str,
    db: AsyncSession = Depends(get_db_session),
):
    """
    Increment usage count for a model.
    
    Called when a model is used for generation.
    """
    service = InstalledModelService(db)
    await service.increment_usage(model_id)
    
    return {"success": True, "message": "Usage count incremented"}


@router.get("/types/list", response_model=List[str])
async def get_model_types():
    """Get list of available model types."""
    return [
        "Checkpoint",
        "LORA",
        "TextualInversion",
        "Hypernetwork",
        "Controlnet",
        "VAE",
        "AestheticGradient",
        "Poses",
        "Other",
    ]


@router.get("/base-models/list", response_model=List[str])
async def get_base_models():
    """Get list of known base model architectures."""
    return [
        "SD 1.5",
        "SD 2.0",
        "SD 2.1",
        "SDXL 0.9",
        "SDXL 1.0",
        "SD 3",
        "Flux.1",
        "Flux.1 D",
        "Flux.1 S",
        "Other",
    ]
