"""
CivitAI Model Integration Utilities

Provides utilities for listing and downloading models from CivitAI.
Supports both REST API and Python SDK approaches.

Documentation:
- REST API: https://developer.civitai.com/docs/api/public-rest
- Python SDK: https://developer.civitai.com/docs/api/python-sdk
"""

import hashlib
import re
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import httpx

from backend.config.logging import get_logger

logger = get_logger(__name__)

# CivitAI API Configuration
CIVITAI_API_BASE = "https://civitai.com/api/v1"
CIVITAI_CDN_BASE = "https://civitai.com/api/download/models"

# Default value for unknown model names
UNKNOWN_MODEL_NAME = "Unknown"


class CivitAIModelType(str, Enum):
    """Model types available on CivitAI."""

    CHECKPOINT = "Checkpoint"
    TEXTUAL_INVERSION = "TextualInversion"
    HYPERNETWORK = "Hypernetwork"
    AESTHETIC_GRADIENT = "AestheticGradient"
    LORA = "LORA"
    CONTROLNET = "Controlnet"
    POSES = "Poses"


class CivitAIBaseModel(str, Enum):
    """Base model architectures."""

    SD_1_5 = "SD 1.5"
    SD_2_0 = "SD 2.0"
    SD_2_1 = "SD 2.1"
    SDXL_0_9 = "SDXL 0.9"
    SDXL_1_0 = "SDXL 1.0"
    SD_3 = "SD 3"
    FLUX_1 = "Flux.1"
    OTHER = "Other"


class CivitAIClient:
    """
    Client for interacting with CivitAI API.

    Supports both authenticated and unauthenticated requests.
    """

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize CivitAI client.

        Args:
            api_key: Optional CivitAI API key for authenticated requests
        """
        self.api_key = api_key
        self.base_url = CIVITAI_API_BASE

    def _get_headers(self) -> Dict[str, str]:
        """Get headers for API requests."""
        headers = {
            "Content-Type": "application/json",
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    async def list_models(
        self,
        limit: int = 20,
        page: int = 1,
        query: Optional[str] = None,
        model_types: Optional[List[CivitAIModelType]] = None,
        base_models: Optional[List[CivitAIBaseModel]] = None,
        sort: str = "Highest Rated",
        period: str = "AllTime",
        nsfw: bool = True,  # Default to True for private server mode
        cursor: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        List models from CivitAI.

        Args:
            limit: Number of models to return (max 100)
            page: Page number for pagination (only used when query is not provided)
            query: Search query string
            model_types: Filter by model types
            base_models: Filter by base model architectures
            sort: Sort order (Highest Rated, Most Downloaded, Newest)
            period: Time period (AllTime, Year, Month, Week, Day)
            nsfw: Include NSFW models (default True for private server)
            cursor: Cursor for pagination (required when using query parameter)

        Returns:
            Dictionary with 'items' (list of models) and 'metadata' (pagination info)

        Note:
            CivitAI API does not allow using 'page' parameter with 'query' parameter.
            When a query is provided, cursor-based pagination must be used instead.
        """
        try:
            params = {
                "limit": min(limit, 100),
                "sort": sort,
                "period": period,
                "nsfw": nsfw,
            }

            if query:
                # When query is provided, use cursor-based pagination
                # CivitAI API does not allow page param with query
                params["query"] = query
                if cursor:
                    params["cursor"] = cursor
            else:
                # When no query, use page-based pagination
                params["page"] = page

            if model_types:
                params["types"] = ",".join([t.value for t in model_types])

            if base_models:
                params["baseModels"] = ",".join([b.value for b in base_models])

            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(
                    f"{self.base_url}/models",
                    params=params,
                    headers=self._get_headers(),
                )
                response.raise_for_status()
                return response.json()

        except httpx.HTTPError as e:
            logger.error(f"Failed to list CivitAI models: {str(e)}")
            raise

    async def get_model_details(self, model_id: int) -> Dict[str, Any]:
        """
        Get detailed information about a specific model.

        Args:
            model_id: CivitAI model ID

        Returns:
            Dictionary with model details including versions, files, and metadata
        """
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(
                    f"{self.base_url}/models/{model_id}",
                    headers=self._get_headers(),
                )
                response.raise_for_status()
                return response.json()

        except httpx.HTTPError as e:
            logger.error(f"Failed to get model {model_id} details: {str(e)}")
            raise

    async def get_model_version(self, version_id: int) -> Dict[str, Any]:
        """
        Get information about a specific model version.

        Args:
            version_id: CivitAI model version ID

        Returns:
            Dictionary with version details
        """
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(
                    f"{self.base_url}/model-versions/{version_id}",
                    headers=self._get_headers(),
                )
                response.raise_for_status()
                return response.json()

        except httpx.HTTPError as e:
            logger.error(f"Failed to get model version {version_id}: {str(e)}")
            raise

    async def download_model(
        self,
        model_version_id: int,
        output_path: Path,
        file_type: Optional[str] = None,
        progress_callback: Optional[callable] = None,
    ) -> Tuple[Path, Dict[str, Any]]:
        """
        Download a model from CivitAI.

        Args:
            model_version_id: CivitAI model version ID to download
            output_path: Directory where model should be saved
            file_type: Optional file type filter (e.g., "Model", "VAE", "Pruned Model")
            progress_callback: Optional callback function(downloaded_bytes, total_bytes)

        Returns:
            Tuple of (downloaded_file_path, model_metadata)
        """
        try:
            # Get version details to find download URL
            version_info = await self.get_model_version(model_version_id)

            # Extract model metadata for detailed logging
            model_info = version_info.get("model", {})
            model_name = model_info.get("name", UNKNOWN_MODEL_NAME)
            is_nsfw = model_info.get("nsfw", False)

            # Check for access requirements that might cause 401 errors
            # CivitAI models can have various access restrictions
            availability = version_info.get("availability", "Public")
            early_access_deadline = version_info.get("earlyAccessEndsAt")
            requires_auth = (
                early_access_deadline is not None or availability != "Public"
            )

            # Log detailed authentication and model information
            logger.info(f"🔍 CivitAI Download Request Details:")
            logger.info(f"   Model: {model_name}")
            logger.info(f"   Version ID: {model_version_id}")
            logger.info(f"   NSFW: {is_nsfw}")
            logger.info(f"   Availability: {availability}")
            if early_access_deadline:
                logger.info(f"   Early Access Until: {early_access_deadline}")
            logger.info(
                f"   API Key Configured: {'Yes' if self.api_key else 'NO - This may cause 401 errors!'}"
            )
            if requires_auth and not self.api_key:
                logger.warning(
                    f"   ⚠️  This model may require authentication (early access or restricted)"
                )

            # Find the appropriate file to download
            files = version_info.get("files", [])
            if not files:
                raise ValueError(f"No files found for model version {model_version_id}")

            # Filter by file type if specified
            if file_type:
                files = [f for f in files if f.get("type") == file_type]
                if not files:
                    raise ValueError(f"No files of type '{file_type}' found")

            # Use the primary file (usually the first one)
            file_info = files[0]
            download_url = file_info.get("downloadUrl")

            # Log file-specific access information
            # Some files have their own access restrictions
            file_visibility = file_info.get("visibility", "Public")
            logger.info(f"   File Visibility: {file_visibility}")

            if not download_url:
                # Construct download URL manually
                download_url = f"{CIVITAI_CDN_BASE}/{model_version_id}"
                if file_type:
                    download_url += f"?type={file_type}"
                logger.info(
                    f"   Download URL: Constructed manually (no downloadUrl in file info)"
                )
            else:
                logger.info(f"   Download URL: Using provided downloadUrl from API")

            # Add API key to download URL if available
            # Per CivitAI docs: token can be passed as query param OR Authorization header
            # We do BOTH for maximum compatibility
            if self.api_key:
                separator = "&" if "?" in download_url else "?"
                download_url += f"{separator}token={self.api_key}"
                logger.info(
                    f"   Authentication: Token added to URL + Authorization header"
                )
            else:
                logger.warning(
                    f"   ⚠️  Authentication: NO API KEY - Downloads may fail for restricted models!"
                )
                logger.warning(f"   ⚠️  To fix: Add your CivitAI API key in Settings")

            # Log the download URL (redact token for security)
            # Pattern handles tokens at end of URL (no trailing &) and tokens with special characters
            safe_url = re.sub(r"token=[^&\s]*", "token=***REDACTED***", download_url)
            logger.info(f"   Request URL: {safe_url}")

            # Get filename from file info, or generate one from model name
            filename = file_info.get("name")
            if not filename:
                # Use model name for the filename instead of just version ID
                if model_name and model_name != UNKNOWN_MODEL_NAME:
                    # Sanitize model name to be filesystem-safe
                    safe_name = re.sub(
                        r'[<>:"/\\|?*]', "", model_name
                    )  # Remove invalid chars
                    safe_name = safe_name.strip()
                    safe_name = re.sub(
                        r"\s+", "_", safe_name
                    )  # Replace whitespace with underscores
                    safe_name = safe_name[
                        :50
                    ]  # Limit length to avoid overly long filenames
                    filename = f"{safe_name}_{model_version_id}.safetensors"
                else:
                    filename = f"model_{model_version_id}.safetensors"
            output_file = output_path / filename

            # Ensure output directory exists
            output_path.mkdir(parents=True, exist_ok=True)

            logger.info(f"📥 Downloading {filename} from CivitAI...")
            logger.info(f"   Version ID: {model_version_id}")
            logger.info(f"   Size: {file_info.get('sizeKB', 0) / 1024:.2f} MB")

            # Download file with progress tracking
            # Enable follow_redirects to handle 307 redirects from CivitAI to Cloudflare storage
            # IMPORTANT: Include Authorization header for NSFW models that require authentication
            async with httpx.AsyncClient(timeout=None, follow_redirects=True) as client:
                try:
                    async with client.stream(
                        "GET", download_url, headers=self._get_headers()
                    ) as response:
                        # Log response status for debugging
                        logger.info(f"   HTTP Response: {response.status_code}")
                        response.raise_for_status()

                        total_size = int(response.headers.get("content-length", 0))
                        downloaded_size = 0

                        with open(output_file, "wb") as f:
                            async for chunk in response.aiter_bytes(chunk_size=8192):
                                f.write(chunk)
                                downloaded_size += len(chunk)

                                if progress_callback:
                                    progress_callback(downloaded_size, total_size)

                                # Log progress every 10% (avoid division by zero for small files)
                                if (
                                    total_size > 10
                                    and downloaded_size % (total_size // 10) < 8192
                                ):
                                    progress_pct = (downloaded_size / total_size) * 100
                                    logger.info(f"   Progress: {progress_pct:.1f}%")
                except httpx.HTTPStatusError as http_err:
                    # Provide detailed error information for debugging
                    status_code = http_err.response.status_code
                    logger.error(f"❌ HTTP Error {status_code} during download")
                    logger.error(f"   URL: {safe_url}")

                    if status_code == 401:
                        logger.error(
                            "   🔐 401 Unauthorized - Authentication required!"
                        )
                        logger.error("   Possible causes:")
                        if not self.api_key:
                            logger.error(
                                "   1. No CivitAI API key configured - Add your API key in Settings"
                            )
                        else:
                            logger.error(
                                "   1. API key may be invalid or expired - Check your CivitAI API key"
                            )
                        logger.error(
                            "   2. Model may require you to accept terms on CivitAI website first"
                        )
                        logger.error(
                            "   3. Model may be early access or require special permissions"
                        )
                        logger.error(
                            "   4. Your CivitAI account may not have access to this model"
                        )
                        logger.error(f"   Model NSFW status: {is_nsfw}")
                        logger.error(f"   Model availability: {availability}")
                    elif status_code == 403:
                        logger.error("   🚫 403 Forbidden - Access denied!")
                        logger.error(
                            "   The model may be restricted or your account lacks permissions"
                        )
                    elif status_code == 404:
                        logger.error("   🔍 404 Not Found - Model file not found")
                        logger.error(
                            "   The model version may have been removed or the URL is incorrect"
                        )

                    raise

            logger.info(f"✅ Downloaded successfully to {output_file}")

            # Verify file hash if provided
            # CivitAI API returns hashes as a dictionary: {"SHA256": "hash", "AutoV2": "hash", ...}
            expected_hash = None
            hashes = file_info.get("hashes", {})
            if isinstance(hashes, dict):
                expected_hash = hashes.get("SHA256")
            elif isinstance(hashes, list):
                # Handle legacy list format for backwards compatibility
                for hash_info in hashes:
                    if (
                        isinstance(hash_info, dict)
                        and hash_info.get("type") == "SHA256"
                    ):
                        expected_hash = hash_info.get("hash")
                        break

            if expected_hash:
                logger.info("   Verifying file integrity...")
                actual_hash = await self._calculate_file_hash(output_file)
                if actual_hash.lower() != expected_hash.lower():
                    logger.error(
                        f"   ❌ Hash mismatch! Expected: {expected_hash}, Got: {actual_hash}"
                    )
                    logger.warning(f"   🗑️  Deleting corrupted file: {output_file}")
                    output_file.unlink()
                    raise ValueError(
                        "Downloaded file hash does not match expected hash"
                    )
                logger.info("   ✅ File integrity verified")

            # Build metadata
            metadata = {
                "source": "civitai",
                "model_id": version_info.get("modelId"),
                "version_id": model_version_id,
                "version_name": version_info.get("name"),
                "model_name": version_info.get("model", {}).get("name"),
                "base_model": version_info.get("baseModel"),
                "file_name": filename,
                "file_size_kb": file_info.get("sizeKB"),
                "file_hash": expected_hash,
                "download_url": download_url.split("?")[0],  # Remove token
                "trained_words": version_info.get("trainedWords", []),
                "license": version_info.get("model", {}).get("allowCommercialUse"),
                "nsfw": version_info.get("model", {}).get("nsfw", False),
                "type": version_info.get("model", {}).get("type", "Checkpoint"),
                "description": version_info.get("description", ""),
            }

            # Save metadata file alongside the model
            # This enables AIModelManager to detect and register the model
            metadata_filename = output_file.stem + "_metadata.json"
            metadata_file = output_path / metadata_filename

            try:
                import json

                with open(metadata_file, "w") as f:
                    json.dump(metadata, f, indent=2)
                logger.info(f"   📄 Metadata saved to {metadata_file}")
            except Exception as meta_error:
                logger.warning(f"   ⚠️ Failed to save metadata file: {meta_error}")
                # Don't fail the download if metadata save fails

            return output_file, metadata

        except Exception as e:
            logger.error(f"Failed to download model {model_version_id}: {str(e)}")
            raise

    async def _calculate_file_hash(self, file_path: Path) -> str:
        """Calculate SHA256 hash of a file."""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()

    async def search_models(
        self,
        query: str,
        limit: int = 20,
        nsfw: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        Search for models by name or description.

        Args:
            query: Search query
            limit: Maximum number of results
            nsfw: Include NSFW models

        Returns:
            List of model dictionaries
        """
        result = await self.list_models(
            limit=limit,
            query=query,
            nsfw=nsfw,
        )
        return result.get("items", [])


async def download_civitai_model(
    model_version_id: int,
    output_dir: Path,
    api_key: Optional[str] = None,
    file_type: Optional[str] = None,
) -> Tuple[Path, Dict[str, Any]]:
    """
    Convenience function to download a CivitAI model.

    Args:
        model_version_id: CivitAI model version ID
        output_dir: Directory to save the model
        api_key: Optional CivitAI API key
        file_type: Optional file type filter

    Returns:
        Tuple of (downloaded_file_path, model_metadata)
    """
    client = CivitAIClient(api_key=api_key)
    return await client.download_model(
        model_version_id=model_version_id,
        output_path=output_dir,
        file_type=file_type,
    )


async def list_civitai_models(
    query: Optional[str] = None,
    model_types: Optional[List[str]] = None,
    limit: int = 20,
    api_key: Optional[str] = None,
    nsfw: bool = False,
) -> List[Dict[str, Any]]:
    """
    Convenience function to list CivitAI models.

    Args:
        query: Search query
        model_types: Filter by model types
        limit: Maximum number of results
        api_key: Optional CivitAI API key
        nsfw: Include NSFW models

    Returns:
        List of model dictionaries
    """
    client = CivitAIClient(api_key=api_key)

    # Convert string types to enum if provided
    type_enums = None
    if model_types:
        type_enums = []
        for t in model_types:
            try:
                type_enums.append(CivitAIModelType(t))
            except ValueError:
                logger.warning(f"Invalid model type: {t}")

    result = await client.list_models(
        limit=limit,
        query=query,
        model_types=type_enums,
        nsfw=nsfw,
    )

    return result.get("items", [])
