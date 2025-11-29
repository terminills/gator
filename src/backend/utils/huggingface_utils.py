"""
HuggingFace Model Integration Utilities

Provides utilities for searching, listing and downloading models from HuggingFace Hub.
Supports both REST API and huggingface_hub Python SDK approaches.

Documentation:
- REST API: https://huggingface.co/docs/hub/api
- Python SDK: https://huggingface.co/docs/huggingface_hub/
"""

import asyncio
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum

import httpx

from backend.config.logging import get_logger

logger = get_logger(__name__)

# HuggingFace API Configuration
HUGGINGFACE_API_BASE = "https://huggingface.co/api"
HUGGINGFACE_CDN_BASE = "https://huggingface.co"


class HuggingFaceModelType(str, Enum):
    """Model types/tasks available on HuggingFace relevant to image generation."""
    TEXT_TO_IMAGE = "text-to-image"
    IMAGE_TO_IMAGE = "image-to-image"
    UNCONDITIONAL_IMAGE_GENERATION = "unconditional-image-generation"
    TEXT_GENERATION = "text-generation"
    TEXT2TEXT_GENERATION = "text2text-generation"
    FEATURE_EXTRACTION = "feature-extraction"
    IMAGE_CLASSIFICATION = "image-classification"
    IMAGE_SEGMENTATION = "image-segmentation"
    OBJECT_DETECTION = "object-detection"
    DEPTH_ESTIMATION = "depth-estimation"


class HuggingFaceLibrary(str, Enum):
    """Libraries/frameworks for models."""
    DIFFUSERS = "diffusers"
    TRANSFORMERS = "transformers"
    PYTORCH = "pytorch"
    SAFETENSORS = "safetensors"
    GGUF = "gguf"


class HuggingFaceClient:
    """
    Client for interacting with HuggingFace Hub API.
    
    Supports both authenticated and unauthenticated requests.
    """
    
    def __init__(self, api_token: Optional[str] = None):
        """
        Initialize HuggingFace client.
        
        Args:
            api_token: Optional HuggingFace API token for authenticated requests
        """
        self.api_token = api_token
        self.base_url = HUGGINGFACE_API_BASE
        
    def _get_headers(self) -> Dict[str, str]:
        """Get headers for API requests."""
        headers = {
            "Content-Type": "application/json",
        }
        if self.api_token:
            headers["Authorization"] = f"Bearer {self.api_token}"
        return headers
    
    async def search_models(
        self,
        query: Optional[str] = None,
        author: Optional[str] = None,
        task: Optional[str] = None,
        library: Optional[str] = None,
        tags: Optional[List[str]] = None,
        limit: int = 20,
        sort: str = "downloads",
        direction: str = "-1",
        full: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        Search models on HuggingFace Hub.
        
        Args:
            query: Search query string
            author: Filter by author/organization
            task: Filter by task (e.g., text-to-image)
            library: Filter by library (e.g., diffusers)
            tags: Filter by tags
            limit: Maximum number of results
            sort: Sort field (downloads, likes, created_at, lastModified)
            direction: Sort direction (-1 for descending, 1 for ascending)
            full: Whether to return full model info
            
        Returns:
            List of model dictionaries
        """
        try:
            params = {
                "limit": min(limit, 100),
                "sort": sort,
                "direction": direction,
            }
            
            if query:
                params["search"] = query
            if author:
                params["author"] = author
            if task:
                params["pipeline_tag"] = task
            if library:
                params["library"] = library
            if tags:
                params["tags"] = ",".join(tags)
            if full:
                params["full"] = "true"
            
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(
                    f"{self.base_url}/models",
                    params=params,
                    headers=self._get_headers(),
                )
                response.raise_for_status()
                return response.json()
                
        except httpx.HTTPError as e:
            logger.error(f"Failed to search HuggingFace models: {str(e)}")
            raise
    
    async def get_model_info(self, repo_id: str) -> Dict[str, Any]:
        """
        Get detailed information about a specific model.
        
        Args:
            repo_id: HuggingFace model repository ID (e.g., "stabilityai/stable-diffusion-xl-base-1.0")
            
        Returns:
            Dictionary with model details
        """
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(
                    f"{self.base_url}/models/{repo_id}",
                    headers=self._get_headers(),
                )
                response.raise_for_status()
                return response.json()
                
        except httpx.HTTPError as e:
            logger.error(f"Failed to get model info for {repo_id}: {str(e)}")
            raise
    
    async def list_model_files(
        self,
        repo_id: str,
        revision: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        List files in a model repository.
        
        Args:
            repo_id: HuggingFace model repository ID
            revision: Git revision (branch, tag, or commit)
            
        Returns:
            List of file information dictionaries
        """
        try:
            url = f"{self.base_url}/models/{repo_id}/tree/{revision or 'main'}"
            
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(
                    url,
                    headers=self._get_headers(),
                )
                response.raise_for_status()
                return response.json()
                
        except httpx.HTTPError as e:
            logger.error(f"Failed to list files for {repo_id}: {str(e)}")
            raise
    
    async def download_model_file(
        self,
        repo_id: str,
        filename: str,
        output_path: Path,
        revision: Optional[str] = None,
        progress_callback: Optional[callable] = None,
    ) -> Tuple[Path, Dict[str, Any]]:
        """
        Download a specific file from a HuggingFace model repository.
        
        Args:
            repo_id: HuggingFace model repository ID
            filename: Name of the file to download
            output_path: Directory where file should be saved
            revision: Git revision (branch, tag, or commit)
            progress_callback: Optional callback function(downloaded_bytes, total_bytes)
            
        Returns:
            Tuple of (downloaded_file_path, model_metadata)
        """
        try:
            # Get model info for metadata
            model_info = await self.get_model_info(repo_id)
            
            # Construct download URL
            revision = revision or "main"
            download_url = f"{HUGGINGFACE_CDN_BASE}/{repo_id}/resolve/{revision}/{filename}"
            
            # Add auth token if available
            headers = self._get_headers()
            
            # Ensure output directory exists
            output_path.mkdir(parents=True, exist_ok=True)
            
            # Generate output filename
            output_file = output_path / filename
            
            logger.info(f"📥 Downloading {filename} from HuggingFace...")
            logger.info(f"   Repository: {repo_id}")
            logger.info(f"   Revision: {revision}")
            
            async with httpx.AsyncClient(timeout=None, follow_redirects=True) as client:
                async with client.stream("GET", download_url, headers=headers) as response:
                    response.raise_for_status()
                    
                    total_size = int(response.headers.get("content-length", 0))
                    downloaded_size = 0
                    
                    with open(output_file, "wb") as f:
                        async for chunk in response.aiter_bytes(chunk_size=8192):
                            f.write(chunk)
                            downloaded_size += len(chunk)
                            
                            if progress_callback:
                                progress_callback(downloaded_size, total_size)
                            
                            # Log progress every 10%
                            if total_size > 10 and downloaded_size % (total_size // 10) < 8192:
                                progress_pct = (downloaded_size / total_size) * 100
                                logger.info(f"   Progress: {progress_pct:.1f}%")
            
            logger.info(f"✅ Downloaded successfully to {output_file}")
            
            # Build metadata
            metadata = {
                "source": "huggingface",
                "repo_id": repo_id,
                "revision": revision,
                "filename": filename,
                "model_id": model_info.get("id"),
                "model_name": model_info.get("id", "").split("/")[-1],
                "author": model_info.get("author"),
                "pipeline_tag": model_info.get("pipeline_tag"),
                "library_name": model_info.get("library_name"),
                "tags": model_info.get("tags", []),
                "downloads": model_info.get("downloads"),
                "likes": model_info.get("likes"),
                "private": model_info.get("private", False),
                "gated": model_info.get("gated", False),
                "card_data": model_info.get("cardData", {}),
                "download_url": download_url,
            }
            
            # Extract trigger words from model card if available
            card_data = model_info.get("cardData", {})
            if card_data:
                # Some models have trigger words in widget examples
                widget = card_data.get("widget", [])
                if widget and isinstance(widget, list):
                    example_prompts = [w.get("text", "") for w in widget if w.get("text")]
                    metadata["example_prompts"] = example_prompts
                
                # Check for base_model info
                if card_data.get("base_model"):
                    metadata["base_model"] = card_data.get("base_model")
            
            # Save metadata file alongside the model
            metadata_filename = output_file.stem + "_metadata.json"
            metadata_file = output_path / metadata_filename
            
            try:
                import json
                with open(metadata_file, "w") as f:
                    json.dump(metadata, f, indent=2)
                logger.info(f"   📄 Metadata saved to {metadata_file}")
            except Exception as meta_error:
                logger.warning(f"   ⚠️ Failed to save metadata file: {meta_error}")
            
            return output_file, metadata
            
        except Exception as e:
            logger.error(f"Failed to download from HuggingFace: {str(e)}")
            raise
    
    async def download_model_with_hub(
        self,
        repo_id: str,
        output_path: Path,
        revision: Optional[str] = None,
        filename: Optional[str] = None,
        allow_patterns: Optional[List[str]] = None,
        ignore_patterns: Optional[List[str]] = None,
    ) -> Tuple[Path, Dict[str, Any]]:
        """
        Download a model using the huggingface_hub library.
        
        This method uses the official HuggingFace Hub Python SDK which handles
        caching, resumable downloads, and better authentication.
        
        Args:
            repo_id: HuggingFace model repository ID
            output_path: Directory where model should be saved
            revision: Git revision (branch, tag, or commit)
            filename: Specific file to download (if None, downloads entire repo)
            allow_patterns: Patterns of files to include
            ignore_patterns: Patterns of files to exclude
            
        Returns:
            Tuple of (downloaded_path, model_metadata)
        """
        try:
            from huggingface_hub import hf_hub_download, snapshot_download
            
            # Get model info for metadata
            model_info = await self.get_model_info(repo_id)
            
            # Set up token for authentication
            token = self.api_token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_TOKEN")
            
            output_path.mkdir(parents=True, exist_ok=True)
            
            if filename:
                # Download single file
                logger.info(f"📥 Downloading {filename} from {repo_id}...")
                downloaded_path = hf_hub_download(
                    repo_id=repo_id,
                    filename=filename,
                    revision=revision,
                    local_dir=str(output_path),
                    token=token,
                )
                downloaded_path = Path(downloaded_path)
            else:
                # Download entire repository
                logger.info(f"📥 Downloading entire repository {repo_id}...")
                downloaded_path = snapshot_download(
                    repo_id=repo_id,
                    revision=revision,
                    local_dir=str(output_path),
                    token=token,
                    allow_patterns=allow_patterns,
                    ignore_patterns=ignore_patterns or ["*.md", "*.txt", ".gitattributes"],
                )
                downloaded_path = Path(downloaded_path)
            
            logger.info(f"✅ Downloaded to {downloaded_path}")
            
            # Build metadata
            metadata = {
                "source": "huggingface",
                "repo_id": repo_id,
                "revision": revision or "main",
                "filename": filename,
                "model_id": model_info.get("id"),
                "model_name": model_info.get("id", "").split("/")[-1],
                "author": model_info.get("author"),
                "pipeline_tag": model_info.get("pipeline_tag"),
                "library_name": model_info.get("library_name"),
                "tags": model_info.get("tags", []),
                "downloads": model_info.get("downloads"),
                "likes": model_info.get("likes"),
                "private": model_info.get("private", False),
                "gated": model_info.get("gated", False),
            }
            
            return downloaded_path, metadata
            
        except ImportError:
            logger.warning("huggingface_hub not installed, falling back to direct download")
            if filename:
                return await self.download_model_file(
                    repo_id=repo_id,
                    filename=filename,
                    output_path=output_path,
                    revision=revision,
                )
            else:
                raise ValueError("filename required when huggingface_hub is not installed")
        except Exception as e:
            logger.error(f"Failed to download model: {str(e)}")
            raise
    
    async def get_popular_diffusion_models(
        self,
        limit: int = 20,
    ) -> List[Dict[str, Any]]:
        """
        Get popular text-to-image diffusion models.
        
        Returns:
            List of popular diffusion models
        """
        return await self.search_models(
            task="text-to-image",
            library="diffusers",
            limit=limit,
            sort="downloads",
        )
    
    async def get_popular_lora_models(
        self,
        limit: int = 20,
    ) -> List[Dict[str, Any]]:
        """
        Get popular LoRA models for Stable Diffusion.
        
        Returns:
            List of popular LoRA models
        """
        return await self.search_models(
            tags=["lora", "stable-diffusion"],
            limit=limit,
            sort="downloads",
        )


async def search_huggingface_models(
    query: Optional[str] = None,
    task: Optional[str] = None,
    library: Optional[str] = None,
    limit: int = 20,
    api_token: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Convenience function to search HuggingFace models.
    
    Args:
        query: Search query
        task: Filter by task type
        library: Filter by library
        limit: Maximum number of results
        api_token: Optional HuggingFace API token
        
    Returns:
        List of model dictionaries
    """
    client = HuggingFaceClient(api_token=api_token)
    return await client.search_models(
        query=query,
        task=task,
        library=library,
        limit=limit,
    )


async def download_huggingface_model(
    repo_id: str,
    output_dir: Path,
    filename: Optional[str] = None,
    revision: Optional[str] = None,
    api_token: Optional[str] = None,
) -> Tuple[Path, Dict[str, Any]]:
    """
    Convenience function to download a HuggingFace model.
    
    Args:
        repo_id: HuggingFace model repository ID
        output_dir: Directory to save the model
        filename: Specific file to download
        revision: Git revision
        api_token: Optional HuggingFace API token
        
    Returns:
        Tuple of (downloaded_path, model_metadata)
    """
    client = HuggingFaceClient(api_token=api_token)
    return await client.download_model_with_hub(
        repo_id=repo_id,
        output_path=output_dir,
        filename=filename,
        revision=revision,
    )


# Common diffusion model repositories
POPULAR_DIFFUSION_REPOS = [
    "stabilityai/stable-diffusion-xl-base-1.0",
    "stabilityai/stable-diffusion-2-1",
    "runwayml/stable-diffusion-v1-5",
    "CompVis/stable-diffusion-v1-4",
    "stabilityai/sdxl-turbo",
    "stabilityai/stable-diffusion-3-medium-diffusers",
    "black-forest-labs/FLUX.1-schnell",
    "black-forest-labs/FLUX.1-dev",
]
