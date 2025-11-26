"""
CivitAI Model Management Routes

Provides API endpoints for browsing, downloading, and managing models from CivitAI.
"""

from typing import List, Optional
from pathlib import Path

from fastapi import APIRouter, HTTPException, Query, Depends
from pydantic import BaseModel, Field

from backend.config.logging import get_logger
from backend.config.settings import get_settings
from backend.services.settings_service import get_db_setting
from backend.utils.civitai_utils import (
    CivitAIClient,
    list_civitai_models,
    download_civitai_model,
    CivitAIModelType,
    CivitAIBaseModel,
)

logger = get_logger(__name__)
router = APIRouter(prefix="/civitai", tags=["civitai"])


class CivitAISearchRequest(BaseModel):
    """Request to search CivitAI models."""
    
    query: Optional[str] = Field(None, description="Search query")
    model_types: Optional[List[str]] = Field(
        None,
        description="Filter by model types (Checkpoint, LORA, etc.)"
    )
    base_models: Optional[List[str]] = Field(
        None,
        description="Filter by base models (SDXL 1.0, SD 1.5, etc.)"
    )
    limit: int = Field(20, ge=1, le=100, description="Number of results")
    page: int = Field(1, ge=1, description="Page number")
    nsfw: bool = Field(False, description="Include NSFW models")


class CivitAIDownloadRequest(BaseModel):
    """Request to download a CivitAI model."""
    
    model_version_id: int = Field(..., description="CivitAI model version ID")
    file_type: Optional[str] = Field(
        None,
        description="File type filter (Model, VAE, Pruned Model)"
    )
    output_directory: Optional[str] = Field(
        None,
        description="Custom output directory (relative to models path)"
    )


class CivitAIModelInfo(BaseModel):
    """CivitAI model information."""
    
    id: int
    name: str
    description: Optional[str]
    type: str
    nsfw: bool
    tags: List[str]
    creator: dict
    stats: dict
    model_versions: List[dict]


class DownloadStatus(BaseModel):
    """Status of a model download."""
    
    success: bool
    file_path: Optional[str]
    metadata: Optional[dict]
    error: Optional[str]


@router.get("/search", response_model=dict)
async def search_models(
    query: Optional[str] = Query(None, description="Search query"),
    model_types: Optional[str] = Query(
        None,
        description="Comma-separated model types (Checkpoint,LORA,etc.)"
    ),
    base_models: Optional[str] = Query(
        None,
        description="Comma-separated base models (SDXL 1.0,SD 1.5,etc.)"
    ),
    limit: int = Query(20, ge=1, le=100, description="Number of results"),
    page: int = Query(1, ge=1, description="Page number (ignored when query is provided)"),
    cursor: Optional[str] = Query(None, description="Cursor for pagination (use with query parameter)"),
    sort: str = Query("Highest Rated", description="Sort order"),
    period: str = Query("AllTime", description="Time period"),
    nsfw: bool = Query(True, description="Include NSFW models (enabled by default for private server)"),
):
    """
    Search models from CivitAI (alias for /models endpoint).
    
    NSFW is enabled by default for private server mode.
    Returns paginated list of models matching the search criteria.
    
    Note: When using 'query' parameter, page-based pagination is not supported
    by CivitAI API. Use 'cursor' parameter instead for pagination with search queries.
    The response metadata will contain 'nextCursor' for the next page of results.
    """
    # Call the list_models function directly
    return await list_models(
        query=query,
        model_types=model_types,
        base_models=base_models,
        limit=limit,
        page=page,
        cursor=cursor,
        sort=sort,
        period=period,
        nsfw=nsfw,
    )


@router.get("/models", response_model=dict)
async def list_models(
    query: Optional[str] = Query(None, description="Search query"),
    model_types: Optional[str] = Query(
        None,
        description="Comma-separated model types (Checkpoint,LORA,etc.)"
    ),
    base_models: Optional[str] = Query(
        None,
        description="Comma-separated base models (SDXL 1.0,SD 1.5,etc.)"
    ),
    limit: int = Query(20, ge=1, le=100, description="Number of results"),
    page: int = Query(1, ge=1, description="Page number (ignored when query is provided)"),
    cursor: Optional[str] = Query(None, description="Cursor for pagination (use with query parameter)"),
    sort: str = Query("Highest Rated", description="Sort order"),
    period: str = Query("AllTime", description="Time period"),
    nsfw: bool = Query(False, description="Include NSFW models"),
):
    """
    List models from CivitAI with filtering options.
    
    Returns paginated list of models matching the search criteria.
    
    Note: When using 'query' parameter, page-based pagination is not supported
    by CivitAI API. Use 'cursor' parameter instead for pagination with search queries.
    The response metadata will contain 'nextCursor' for the next page of results.
    """
    try:
        # Get CivitAI API key from database settings (stored via admin panel)
        api_key = await get_db_setting("civitai_api_key")
        
        # Check NSFW preference from database, default to True for private server mode
        allow_nsfw = await get_db_setting("civitai_allow_nsfw")
        if allow_nsfw is None:
            allow_nsfw = True  # Default to True for private server
        
        # For private server, NSFW is enabled by default
        if not allow_nsfw and nsfw:
            raise HTTPException(
                status_code=403,
                detail="NSFW models are disabled in settings"
            )
        
        # Parse model types and base models
        type_list = model_types.split(",") if model_types and model_types.strip() else None
        base_list = base_models.split(",") if base_models and base_models.strip() else None
        
        # Convert to enums
        type_enums = None
        if type_list:
            type_enums = []
            for t in type_list:
                try:
                    type_enums.append(CivitAIModelType(t))
                except ValueError:
                    logger.warning(f"Invalid model type: {t}")
        
        base_enums = None
        if base_list:
            base_enums = []
            for b in base_list:
                try:
                    base_enums.append(CivitAIBaseModel(b))
                except ValueError:
                    logger.warning(f"Invalid base model: {b}")
        
        # Create client and list models
        client = CivitAIClient(api_key=api_key)
        result = await client.list_models(
            limit=limit,
            page=page,
            query=query,
            model_types=type_enums,
            base_models=base_enums,
            sort=sort,
            period=period,
            nsfw=nsfw,
            cursor=cursor,
        )
        
        return result
        
    except Exception as e:
        logger.error(f"Failed to list CivitAI models: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to fetch models from CivitAI: {str(e)}"
        )


@router.get("/models/{model_id}", response_model=dict)
async def get_model_details(model_id: int):
    """
    Get detailed information about a specific CivitAI model.
    
    Returns model details including all versions, files, and metadata.
    """
    try:
        # Get CivitAI API key from database settings
        api_key = await get_db_setting("civitai_api_key")
        
        client = CivitAIClient(api_key=api_key)
        model_info = await client.get_model_details(model_id)
        
        return model_info
        
    except Exception as e:
        logger.error(f"Failed to get model details: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to fetch model details: {str(e)}"
        )


@router.get("/model-versions/{version_id}", response_model=dict)
async def get_model_version(version_id: int):
    """
    Get information about a specific model version.
    
    Returns version details including download URLs and file information.
    """
    try:
        # Get CivitAI API key from database settings
        api_key = await get_db_setting("civitai_api_key")
        
        client = CivitAIClient(api_key=api_key)
        version_info = await client.get_model_version(version_id)
        
        return version_info
        
    except Exception as e:
        logger.error(f"Failed to get version details: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to fetch version details: {str(e)}"
        )


@router.post("/download", response_model=DownloadStatus)
async def download_model(request: CivitAIDownloadRequest):
    """
    Download a model from CivitAI.
    
    Downloads the specified model version and saves metadata for tracking.
    """
    try:
        # Get CivitAI API key from database settings
        api_key = await get_db_setting("civitai_api_key")
        
        # Check NSFW settings from database - default to True for private server mode
        allow_nsfw = await get_db_setting("civitai_allow_nsfw")
        if allow_nsfw is None:
            allow_nsfw = True  # Default to True for private server
        
        # Get model info to check NSFW status
        client = CivitAIClient(api_key=api_key)
        version_info = await client.get_model_version(request.model_version_id)
        
        is_nsfw = version_info.get("model", {}).get("nsfw", False)
        if is_nsfw and not allow_nsfw:
            raise HTTPException(
                status_code=403,
                detail="NSFW models are disabled in settings"
            )
        
        # Determine output directory - get ai_model_path from database or use default
        settings = get_settings()
        ai_model_path = await get_db_setting("ai_model_path")
        if ai_model_path is None:
            ai_model_path = getattr(settings, "ai_model_path", "./models")
        models_path = Path(ai_model_path)
        
        if request.output_directory:
            output_dir = models_path / request.output_directory
        else:
            # Use civitai subdirectory by default
            output_dir = models_path / "civitai"
        
        # Download the model
        logger.info(f"Downloading CivitAI model version {request.model_version_id}")
        
        file_path, metadata = await download_civitai_model(
            model_version_id=request.model_version_id,
            output_dir=output_dir,
            api_key=api_key,
            file_type=request.file_type,
        )
        
        return DownloadStatus(
            success=True,
            file_path=str(file_path),
            metadata=metadata,
            error=None,
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to download model: {str(e)}")
        return DownloadStatus(
            success=False,
            file_path=None,
            metadata=None,
            error=str(e),
        )


@router.get("/local-models", response_model=List[dict])
async def list_local_civitai_models():
    """
    List CivitAI models that have been downloaded locally.
    
    Returns list of downloaded models with metadata and usage tracking info.
    """
    try:
        # Get ai_model_path from database or use default
        settings = get_settings()
        ai_model_path = await get_db_setting("ai_model_path")
        if ai_model_path is None:
            ai_model_path = getattr(settings, "ai_model_path", "./models")
        models_path = Path(ai_model_path)
        civitai_dir = models_path / "civitai"
        
        if not civitai_dir.exists():
            return []
        
        local_models = []
        
        # Find all metadata files
        for metadata_file in civitai_dir.glob("*_metadata.json"):
            try:
                import json
                
                with open(metadata_file, "r") as f:
                    metadata = json.load(f)
                
                # Check if model file exists
                model_file = metadata_file.parent / metadata.get("file_name", "")
                if model_file.exists():
                    metadata["local_path"] = str(model_file)
                    metadata["size_mb"] = model_file.stat().st_size / (1024 * 1024)
                    metadata["metadata_file"] = str(metadata_file)
                    local_models.append(metadata)
                    
            except Exception as e:
                logger.warning(f"Failed to read metadata from {metadata_file}: {e}")
        
        return local_models
        
    except Exception as e:
        logger.error(f"Failed to list local models: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to list local models: {str(e)}"
        )


@router.get("/types", response_model=List[str])
async def get_model_types():
    """Get list of available model types on CivitAI."""
    return [t.value for t in CivitAIModelType]


@router.get("/base-models", response_model=List[str])
async def get_base_models():
    """Get list of available base model architectures on CivitAI."""
    return [b.value for b in CivitAIBaseModel]
