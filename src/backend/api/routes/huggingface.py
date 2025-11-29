"""
HuggingFace Model Management Routes

Provides API endpoints for browsing, downloading, and managing models from HuggingFace Hub.
"""

from typing import List, Optional
from pathlib import Path

from fastapi import APIRouter, HTTPException, Query, Depends
from pydantic import BaseModel, Field

from backend.config.logging import get_logger
from backend.config.settings import get_settings
from backend.services.settings_service import get_db_setting
from backend.utils.huggingface_utils import (
    HuggingFaceClient,
    search_huggingface_models,
    download_huggingface_model,
    HuggingFaceModelType,
    HuggingFaceLibrary,
    POPULAR_DIFFUSION_REPOS,
)

logger = get_logger(__name__)
router = APIRouter(prefix="/huggingface", tags=["huggingface"])


class HuggingFaceSearchRequest(BaseModel):
    """Request to search HuggingFace models."""
    
    query: Optional[str] = Field(None, description="Search query")
    author: Optional[str] = Field(None, description="Filter by author/organization")
    task: Optional[str] = Field(None, description="Filter by task (e.g., text-to-image)")
    library: Optional[str] = Field(None, description="Filter by library (e.g., diffusers)")
    tags: Optional[List[str]] = Field(None, description="Filter by tags")
    limit: int = Field(20, ge=1, le=100, description="Number of results")


class HuggingFaceDownloadRequest(BaseModel):
    """Request to download a HuggingFace model."""
    
    repo_id: str = Field(..., description="HuggingFace repository ID (e.g., 'stabilityai/stable-diffusion-xl-base-1.0')")
    filename: Optional[str] = Field(None, description="Specific file to download (if None, downloads entire repo)")
    revision: Optional[str] = Field(None, description="Git revision (branch, tag, or commit)")
    output_directory: Optional[str] = Field(None, description="Custom output directory (relative to models path)")


class HuggingFaceModelInfo(BaseModel):
    """HuggingFace model information."""
    
    id: str
    modelId: str
    author: Optional[str]
    sha: Optional[str]
    private: bool
    gated: Optional[str]
    pipeline_tag: Optional[str]
    library_name: Optional[str]
    tags: List[str]
    downloads: int
    likes: int
    created_at: Optional[str]
    last_modified: Optional[str]


class DownloadStatus(BaseModel):
    """Status of a model download."""
    
    success: bool
    file_path: Optional[str]
    metadata: Optional[dict]
    error: Optional[str]


@router.get("/search", response_model=List[dict])
async def search_models(
    query: Optional[str] = Query(None, description="Search query"),
    author: Optional[str] = Query(None, description="Filter by author/organization"),
    task: Optional[str] = Query(None, description="Filter by task (e.g., text-to-image, text-generation)"),
    library: Optional[str] = Query(None, description="Filter by library (e.g., diffusers, transformers)"),
    tags: Optional[str] = Query(None, description="Comma-separated tags to filter by"),
    limit: int = Query(20, ge=1, le=100, description="Number of results"),
    sort: str = Query("downloads", description="Sort by (downloads, likes, created_at, lastModified)"),
):
    """
    Search models on HuggingFace Hub.
    
    Returns list of models matching the search criteria.
    """
    try:
        # Get HuggingFace API token from database settings
        api_token = await get_db_setting("hugging_face_token")
        
        client = HuggingFaceClient(api_token=api_token)
        
        # Parse tags
        tags_list = tags.split(",") if tags else None
        
        results = await client.search_models(
            query=query,
            author=author,
            task=task,
            library=library,
            tags=tags_list,
            limit=limit,
            sort=sort,
        )
        
        return results
        
    except Exception as e:
        logger.error(f"Failed to search HuggingFace models: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to search models: {str(e)}"
        )


@router.get("/models/{repo_id:path}", response_model=dict)
async def get_model_info(repo_id: str):
    """
    Get detailed information about a specific HuggingFace model.
    
    Args:
        repo_id: Repository ID (e.g., 'stabilityai/stable-diffusion-xl-base-1.0')
    
    Returns model details including files, tags, and metadata.
    """
    try:
        api_token = await get_db_setting("hugging_face_token")
        
        client = HuggingFaceClient(api_token=api_token)
        model_info = await client.get_model_info(repo_id)
        
        return model_info
        
    except Exception as e:
        logger.error(f"Failed to get model info: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to fetch model info: {str(e)}"
        )


@router.get("/models/{repo_id:path}/files", response_model=List[dict])
async def list_model_files(
    repo_id: str,
    revision: Optional[str] = Query(None, description="Git revision (branch, tag, or commit)"),
):
    """
    List files in a HuggingFace model repository.
    
    Returns list of files with their sizes and types.
    """
    try:
        api_token = await get_db_setting("hugging_face_token")
        
        client = HuggingFaceClient(api_token=api_token)
        files = await client.list_model_files(repo_id, revision=revision)
        
        return files
        
    except Exception as e:
        logger.error(f"Failed to list model files: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to list files: {str(e)}"
        )


@router.post("/download", response_model=DownloadStatus)
async def download_model(request: HuggingFaceDownloadRequest):
    """
    Download a model from HuggingFace Hub.
    
    Downloads the specified model and saves metadata for tracking.
    """
    try:
        api_token = await get_db_setting("hugging_face_token")
        
        # Get ai_model_path from database or use default
        settings = get_settings()
        ai_model_path = await get_db_setting("ai_model_path")
        if ai_model_path is None:
            ai_model_path = getattr(settings, "ai_model_path", "./models")
        models_path = Path(ai_model_path)
        
        if request.output_directory:
            output_dir = models_path / request.output_directory
        else:
            # Use huggingface subdirectory by default
            output_dir = models_path / "huggingface"
        
        logger.info(f"Downloading HuggingFace model {request.repo_id}")
        
        client = HuggingFaceClient(api_token=api_token)
        
        file_path, metadata = await client.download_model_with_hub(
            repo_id=request.repo_id,
            output_path=output_dir,
            filename=request.filename,
            revision=request.revision,
        )
        
        return DownloadStatus(
            success=True,
            file_path=str(file_path),
            metadata=metadata,
            error=None,
        )
        
    except Exception as e:
        logger.error(f"Failed to download model: {str(e)}")
        return DownloadStatus(
            success=False,
            file_path=None,
            metadata=None,
            error=str(e),
        )


@router.get("/popular/diffusion", response_model=List[dict])
async def get_popular_diffusion_models(
    limit: int = Query(20, ge=1, le=100, description="Number of results"),
):
    """
    Get popular text-to-image diffusion models.
    
    Returns a curated list of popular Stable Diffusion and related models.
    """
    try:
        api_token = await get_db_setting("hugging_face_token")
        
        client = HuggingFaceClient(api_token=api_token)
        models = await client.get_popular_diffusion_models(limit=limit)
        
        return models
        
    except Exception as e:
        logger.error(f"Failed to get popular models: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to fetch popular models: {str(e)}"
        )


@router.get("/popular/lora", response_model=List[dict])
async def get_popular_lora_models(
    limit: int = Query(20, ge=1, le=100, description="Number of results"),
):
    """
    Get popular LoRA models for Stable Diffusion.
    
    Returns popular LoRA adapters from HuggingFace.
    """
    try:
        api_token = await get_db_setting("hugging_face_token")
        
        client = HuggingFaceClient(api_token=api_token)
        models = await client.get_popular_lora_models(limit=limit)
        
        return models
        
    except Exception as e:
        logger.error(f"Failed to get LoRA models: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to fetch LoRA models: {str(e)}"
        )


@router.get("/popular/text-generation", response_model=List[dict])
async def get_popular_text_generation_models(
    limit: int = Query(20, ge=1, le=100, description="Number of results"),
):
    """
    Get popular text generation models (LLMs).
    
    Returns popular language models from HuggingFace.
    """
    try:
        api_token = await get_db_setting("hugging_face_token")
        
        client = HuggingFaceClient(api_token=api_token)
        models = await client.search_models(
            task="text-generation",
            limit=limit,
            sort="downloads",
        )
        
        return models
        
    except Exception as e:
        logger.error(f"Failed to get text generation models: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to fetch text generation models: {str(e)}"
        )


@router.get("/tasks", response_model=List[str])
async def get_available_tasks():
    """Get list of available task types for filtering."""
    return [t.value for t in HuggingFaceModelType]


@router.get("/libraries", response_model=List[str])
async def get_available_libraries():
    """Get list of available library types for filtering."""
    return [l.value for l in HuggingFaceLibrary]


@router.get("/recommended-repos", response_model=List[str])
async def get_recommended_repos():
    """Get list of recommended/popular model repositories."""
    return POPULAR_DIFFUSION_REPOS
