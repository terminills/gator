"""
Installed Model Service

Service layer for managing installed AI models including
CivitAI metadata enrichment, trigger word management, and model lookup.
"""

import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from sqlalchemy import delete, func, or_, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from backend.config.logging import get_logger
from backend.models.installed_model import (
    InstalledModelCreate,
    InstalledModelModel,
    InstalledModelResponse,
    InstalledModelUpdate,
    TriggerWordResponse,
)
from backend.services.settings_service import get_db_setting
from backend.utils.civitai_utils import CivitAIClient

logger = get_logger(__name__)


class InstalledModelService:
    """
    Service for managing installed AI models.

    Provides functionality for:
    - CRUD operations on installed models
    - Enriching models with CivitAI metadata
    - Managing trigger words
    - Finding models by trigger word
    """

    def __init__(self, db: AsyncSession):
        self.db = db

    async def create_model(
        self, model_data: InstalledModelCreate
    ) -> InstalledModelResponse:
        """
        Create a new installed model record.

        Args:
            model_data: Model creation data

        Returns:
            Created model response
        """
        model = InstalledModelModel(
            id=uuid.uuid4(),
            name=model_data.name,
            display_name=model_data.display_name,
            model_type=model_data.model_type,
            source=model_data.source,
            file_path=model_data.file_path,
            file_name=model_data.file_name,
            file_size_mb=model_data.file_size_mb,
            file_hash=model_data.file_hash,
            civitai_model_id=model_data.civitai_model_id,
            civitai_version_id=model_data.civitai_version_id,
            civitai_version_name=model_data.civitai_version_name,
            civitai_url=model_data.civitai_url,
            huggingface_repo_id=model_data.huggingface_repo_id,
            huggingface_revision=model_data.huggingface_revision,
            huggingface_filename=model_data.huggingface_filename,
            huggingface_url=model_data.huggingface_url,
            description=model_data.description,
            base_model=model_data.base_model,
            trigger_words=model_data.trigger_words,
            trained_words=model_data.trained_words,
            recommended_weight=model_data.recommended_weight,
            recommended_steps=model_data.recommended_steps,
            recommended_sampler=model_data.recommended_sampler,
            recommended_cfg_scale=model_data.recommended_cfg_scale,
            default_positive_prompt=model_data.default_positive_prompt,
            default_negative_prompt=model_data.default_negative_prompt,
            is_nsfw=model_data.is_nsfw,
            is_active=model_data.is_active,
            extra_metadata=model_data.extra_metadata or {},
        )

        self.db.add(model)
        await self.db.commit()
        await self.db.refresh(model)

        logger.info(f"Created installed model: {model.name} ({model.id})")

        return self._to_response(model)

    async def get_model(self, model_id: str) -> Optional[InstalledModelResponse]:
        """
        Get a model by ID.

        Args:
            model_id: Model UUID

        Returns:
            Model response or None if not found
        """
        result = await self.db.execute(
            select(InstalledModelModel).where(
                InstalledModelModel.id == uuid.UUID(model_id)
            )
        )
        model = result.scalars().first()

        if model:
            return self._to_response(model)
        return None

    async def get_model_by_path(
        self, file_path: str
    ) -> Optional[InstalledModelResponse]:
        """
        Get a model by file path.

        Args:
            file_path: Full path to the model file

        Returns:
            Model response or None if not found
        """
        result = await self.db.execute(
            select(InstalledModelModel).where(
                InstalledModelModel.file_path == file_path
            )
        )
        model = result.scalars().first()

        if model:
            return self._to_response(model)
        return None

    async def get_model_by_civitai_version(
        self, version_id: int
    ) -> Optional[InstalledModelResponse]:
        """
        Get a model by CivitAI version ID.

        Args:
            version_id: CivitAI version ID

        Returns:
            Model response or None if not found
        """
        result = await self.db.execute(
            select(InstalledModelModel).where(
                InstalledModelModel.civitai_version_id == version_id
            )
        )
        model = result.scalars().first()

        if model:
            return self._to_response(model)
        return None

    async def list_models(
        self,
        model_type: Optional[str] = None,
        source: Optional[str] = None,
        base_model: Optional[str] = None,
        is_active: Optional[bool] = True,
        limit: int = 100,
        offset: int = 0,
    ) -> List[InstalledModelResponse]:
        """
        List installed models with optional filtering.

        Args:
            model_type: Filter by model type (Checkpoint, LORA, etc.)
            source: Filter by source (civitai, huggingface, local)
            base_model: Filter by base model (SD 1.5, SDXL 1.0, etc.)
            is_active: Filter by active status
            limit: Maximum results
            offset: Results offset

        Returns:
            List of model responses
        """
        query = select(InstalledModelModel)

        if model_type:
            query = query.where(InstalledModelModel.model_type == model_type)
        if source:
            query = query.where(InstalledModelModel.source == source)
        if base_model:
            query = query.where(InstalledModelModel.base_model == base_model)
        if is_active is not None:
            query = query.where(InstalledModelModel.is_active == is_active)

        query = query.order_by(InstalledModelModel.name).limit(limit).offset(offset)

        result = await self.db.execute(query)
        models = result.scalars().all()

        return [self._to_response(m) for m in models]

    async def update_model(
        self, model_id: str, update_data: InstalledModelUpdate
    ) -> Optional[InstalledModelResponse]:
        """
        Update an installed model.

        Args:
            model_id: Model UUID
            update_data: Update data

        Returns:
            Updated model response or None if not found
        """
        result = await self.db.execute(
            select(InstalledModelModel).where(
                InstalledModelModel.id == uuid.UUID(model_id)
            )
        )
        model = result.scalars().first()

        if not model:
            return None

        # Update fields that are provided
        update_dict = update_data.model_dump(exclude_unset=True)
        for field, value in update_dict.items():
            if hasattr(model, field):
                setattr(model, field, value)

        await self.db.commit()
        await self.db.refresh(model)

        logger.info(f"Updated installed model: {model.name} ({model.id})")

        return self._to_response(model)

    async def delete_model(self, model_id: str) -> bool:
        """
        Delete an installed model.

        Args:
            model_id: Model UUID

        Returns:
            True if deleted, False if not found
        """
        result = await self.db.execute(
            delete(InstalledModelModel).where(
                InstalledModelModel.id == uuid.UUID(model_id)
            )
        )
        await self.db.commit()

        if result.rowcount > 0:
            logger.info(f"Deleted installed model: {model_id}")
            return True
        return False

    async def enrich_from_civitai(
        self, model_id: str, civitai_version_id: int
    ) -> Optional[InstalledModelResponse]:
        """
        Enrich a model's metadata from CivitAI API.

        Args:
            model_id: Model UUID
            civitai_version_id: CivitAI version ID to fetch details from

        Returns:
            Updated model response or None if not found
        """
        # Get the model
        result = await self.db.execute(
            select(InstalledModelModel).where(
                InstalledModelModel.id == uuid.UUID(model_id)
            )
        )
        model = result.scalars().first()

        if not model:
            return None

        # Get CivitAI API key
        api_key = await get_db_setting("civitai_api_key")

        # Fetch details from CivitAI
        client = CivitAIClient(api_key=api_key)

        try:
            version_info = await client.get_model_version(civitai_version_id)

            # Update model with CivitAI data
            model.civitai_version_id = civitai_version_id
            model.civitai_version_name = version_info.get("name")
            model.civitai_model_id = version_info.get("modelId")

            # Get trained words (trigger words)
            trained_words = version_info.get("trainedWords", [])
            if trained_words:
                model.trained_words = trained_words
                # If no trigger words set, use trained words
                if not model.trigger_words:
                    model.trigger_words = trained_words

            # Get base model
            model.base_model = version_info.get("baseModel")

            # Get description from model info
            model_info = version_info.get("model", {})
            if model_info:
                model.description = model_info.get("description", model.description)
                model.is_nsfw = model_info.get("nsfw", model.is_nsfw)
                model.display_name = model_info.get("name", model.display_name)
                model.civitai_url = (
                    f"https://civitai.com/models/{model.civitai_model_id}"
                )

            # Store additional metadata
            model.extra_metadata = {
                **(model.extra_metadata or {}),
                "civitai_enriched_at": datetime.utcnow().isoformat(),
                "civitai_stats": version_info.get("stats", {}),
                "civitai_images": [
                    img.get("url")
                    for img in version_info.get("images", [])[:5]
                    if img.get("url")
                ],
            }

            # Get file info for recommended settings
            files = version_info.get("files", [])
            if files:
                file_info = files[0]
                # Look for recommended settings in metadata
                if file_info.get("metadata"):
                    file_meta = file_info.get("metadata")
                    # Extract fp and size settings if available
                    model.extra_metadata["file_metadata"] = file_meta

            model.source = "civitai"

            await self.db.commit()
            await self.db.refresh(model)

            logger.info(
                f"Enriched model {model.name} with CivitAI data "
                f"(model_id={model.civitai_model_id}, version_id={civitai_version_id})"
            )

            return self._to_response(model)

        except Exception as e:
            logger.error(f"Failed to enrich model from CivitAI: {str(e)}")
            raise

    async def enrich_from_huggingface(
        self, model_id: str, huggingface_repo_id: str
    ) -> Optional[InstalledModelResponse]:
        """
        Enrich a model's metadata from HuggingFace API.

        Args:
            model_id: Model UUID
            huggingface_repo_id: HuggingFace repository ID (e.g., "stabilityai/stable-diffusion-xl-base-1.0")

        Returns:
            Updated model response or None if not found
        """
        from backend.utils.huggingface_utils import HuggingFaceClient

        # Get the model
        result = await self.db.execute(
            select(InstalledModelModel).where(
                InstalledModelModel.id == uuid.UUID(model_id)
            )
        )
        model = result.scalars().first()

        if not model:
            return None

        # Get HuggingFace API token
        api_token = await get_db_setting("hugging_face_token")

        # Fetch details from HuggingFace
        client = HuggingFaceClient(api_token=api_token)

        try:
            model_info = await client.get_model_info(huggingface_repo_id)

            # Update model with HuggingFace data
            model.huggingface_repo_id = huggingface_repo_id
            model.huggingface_url = f"https://huggingface.co/{huggingface_repo_id}"

            # Get model name from repo ID
            model_name = (
                huggingface_repo_id.split("/")[-1]
                if "/" in huggingface_repo_id
                else huggingface_repo_id
            )
            if not model.display_name:
                model.display_name = model_name

            # Get description from model card if available
            if model_info.get("cardData", {}).get("description"):
                model.description = model_info["cardData"]["description"]

            # Get library/framework type to infer model type
            library_name = model_info.get("library_name")
            pipeline_tag = model_info.get("pipeline_tag")
            tags = model_info.get("tags", [])

            # Infer base model from tags or card data
            if not model.base_model:
                card_data = model_info.get("cardData", {})
                if card_data.get("base_model"):
                    model.base_model = card_data["base_model"]
                elif (
                    "sdxl" in model_name.lower()
                    or "stable-diffusion-xl" in model_name.lower()
                ):
                    model.base_model = "SDXL 1.0"
                elif (
                    "sd-1.5" in model_name.lower()
                    or "stable-diffusion-v1-5" in model_name.lower()
                ):
                    model.base_model = "SD 1.5"
                elif "flux" in model_name.lower():
                    model.base_model = "Flux.1"

            # Extract trigger words from widget examples if available
            widget = model_info.get("cardData", {}).get("widget", [])
            if widget and isinstance(widget, list):
                example_prompts = [w.get("text", "") for w in widget if w.get("text")]
                if example_prompts:
                    model.extra_metadata = {
                        **(model.extra_metadata or {}),
                        "example_prompts": example_prompts,
                    }

            # Store additional metadata
            model.extra_metadata = {
                **(model.extra_metadata or {}),
                "huggingface_enriched_at": datetime.utcnow().isoformat(),
                "author": model_info.get("author"),
                "pipeline_tag": pipeline_tag,
                "library_name": library_name,
                "tags": tags,
                "downloads": model_info.get("downloads"),
                "likes": model_info.get("likes"),
                "private": model_info.get("private", False),
                "gated": model_info.get("gated", False),
            }

            model.source = "huggingface"

            await self.db.commit()
            await self.db.refresh(model)

            logger.info(
                f"Enriched model {model.name} with HuggingFace data "
                f"(repo_id={huggingface_repo_id})"
            )

            return self._to_response(model)

        except Exception as e:
            logger.error(f"Failed to enrich model from HuggingFace: {str(e)}")
            raise

    async def get_all_trigger_words(self) -> List[TriggerWordResponse]:
        """
        Get all unique trigger words from all models.

        Returns:
            List of trigger word responses with associated models
        """
        result = await self.db.execute(
            select(InstalledModelModel).where(InstalledModelModel.is_active.is_(True))
        )
        models = result.scalars().all()

        # Aggregate trigger words
        trigger_map: Dict[str, List[str]] = {}
        usage_map: Dict[str, int] = {}

        for model in models:
            if model.trigger_words:
                for trigger in model.trigger_words:
                    trigger_lower = trigger.lower().strip()
                    if trigger_lower:
                        if trigger_lower not in trigger_map:
                            trigger_map[trigger_lower] = []
                            usage_map[trigger_lower] = 0
                        trigger_map[trigger_lower].append(model.name)
                        usage_map[trigger_lower] += model.usage_count

        return [
            TriggerWordResponse(
                trigger_word=trigger,
                source="model",
                models=model_names,
                usage_count=usage_map[trigger],
            )
            for trigger, model_names in sorted(trigger_map.items())
        ]

    async def find_models_by_trigger(
        self,
        trigger_word: str,
        model_type: Optional[str] = None,
        base_model: Optional[str] = None,
    ) -> List[InstalledModelResponse]:
        """
        Find models that have a specific trigger word.

        Args:
            trigger_word: The trigger word to search for
            model_type: Optional filter by model type
            base_model: Optional filter by base model

        Returns:
            List of matching models
        """
        # Get all active models
        query = select(InstalledModelModel).where(
            InstalledModelModel.is_active.is_(True)
        )

        if model_type:
            query = query.where(InstalledModelModel.model_type == model_type)
        if base_model:
            query = query.where(InstalledModelModel.base_model == base_model)

        result = await self.db.execute(query)
        models = result.scalars().all()

        # Filter by trigger word (case-insensitive)
        trigger_lower = trigger_word.lower().strip()
        matching_models = []

        for model in models:
            if model.trigger_words:
                model_triggers = [t.lower().strip() for t in model.trigger_words]
                if trigger_lower in model_triggers:
                    matching_models.append(model)

        return [self._to_response(m) for m in matching_models]

    async def increment_usage(self, model_id: str) -> None:
        """
        Increment usage count for a model.

        Args:
            model_id: Model UUID
        """
        await self.db.execute(
            update(InstalledModelModel)
            .where(InstalledModelModel.id == uuid.UUID(model_id))
            .values(
                usage_count=InstalledModelModel.usage_count + 1,
                last_used_at=datetime.utcnow(),
            )
        )
        await self.db.commit()

    async def scan_and_register_models(
        self,
        models_directory: str,
        model_type: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Scan a directory for model files and register them in the database.

        Args:
            models_directory: Directory to scan
            model_type: Model type for all found models (if not specified, will be guessed)

        Returns:
            Dict with scan results
        """
        results = {
            "scanned_count": 0,
            "registered_count": 0,
            "skipped_count": 0,
            "errors": [],
        }

        models_path = Path(models_directory)
        if not models_path.exists():
            logger.warning(f"Models directory does not exist: {models_directory}")
            return results

        # Find model files
        extensions = [".safetensors", ".ckpt", ".pt", ".bin"]
        for ext in extensions:
            for model_file in models_path.glob(f"**/*{ext}"):
                results["scanned_count"] += 1

                try:
                    # Check if already registered
                    existing = await self.get_model_by_path(str(model_file))
                    if existing:
                        results["skipped_count"] += 1
                        continue

                    # Determine model type based on directory or extension
                    detected_type = model_type
                    if not detected_type:
                        parent_dir = model_file.parent.name.lower()
                        if "lora" in parent_dir:
                            detected_type = "LORA"
                        elif "checkpoint" in parent_dir or "model" in parent_dir:
                            detected_type = "Checkpoint"
                        elif "vae" in parent_dir:
                            detected_type = "VAE"
                        elif "embedding" in parent_dir or "textual" in parent_dir:
                            detected_type = "TextualInversion"
                        else:
                            detected_type = "Other"

                    # Check for metadata file
                    metadata_file = (
                        model_file.parent / f"{model_file.stem}_metadata.json"
                    )
                    metadata = {}
                    civitai_model_id = None
                    civitai_version_id = None
                    trained_words = []

                    if metadata_file.exists():
                        import json

                        with open(metadata_file, "r") as f:
                            metadata = json.load(f)

                        civitai_model_id = metadata.get("model_id")
                        civitai_version_id = metadata.get("version_id")
                        trained_words = metadata.get("trained_words", [])

                    # Create model entry
                    model_data = InstalledModelCreate(
                        name=model_file.stem,
                        display_name=metadata.get("model_name", model_file.stem),
                        model_type=detected_type,
                        source="civitai" if civitai_version_id else "local",
                        file_path=str(model_file),
                        file_name=model_file.name,
                        file_size_mb=model_file.stat().st_size / (1024 * 1024),
                        civitai_model_id=civitai_model_id,
                        civitai_version_id=civitai_version_id,
                        civitai_version_name=metadata.get("version_name"),
                        description=metadata.get("description"),
                        base_model=metadata.get("base_model"),
                        trigger_words=trained_words,
                        trained_words=trained_words,
                        is_nsfw=metadata.get("nsfw", False),
                        huggingface_repo_id=metadata.get("repo_id"),
                        huggingface_revision=metadata.get("revision"),
                        huggingface_filename=metadata.get("filename"),
                        extra_metadata=metadata,
                    )

                    await self.create_model(model_data)
                    results["registered_count"] += 1

                except Exception as e:
                    logger.error(f"Error registering model {model_file}: {str(e)}")
                    results["errors"].append(str(e))

        logger.info(
            f"Model scan complete: {results['registered_count']} registered, "
            f"{results['skipped_count']} skipped, {len(results['errors'])} errors"
        )

        return results

    def _to_response(self, model: InstalledModelModel) -> InstalledModelResponse:
        """Convert model to response."""
        return InstalledModelResponse(
            id=str(model.id),
            name=model.name,
            display_name=model.display_name,
            model_type=model.model_type,
            source=model.source,
            file_path=model.file_path,
            file_name=model.file_name,
            file_size_mb=model.file_size_mb,
            file_hash=model.file_hash,
            civitai_model_id=model.civitai_model_id,
            civitai_version_id=model.civitai_version_id,
            civitai_version_name=model.civitai_version_name,
            civitai_url=model.civitai_url,
            huggingface_repo_id=model.huggingface_repo_id,
            huggingface_revision=model.huggingface_revision,
            huggingface_filename=model.huggingface_filename,
            huggingface_url=model.huggingface_url,
            description=model.description,
            base_model=model.base_model,
            trigger_words=model.trigger_words or [],
            trained_words=model.trained_words,
            recommended_weight=model.recommended_weight,
            recommended_steps=model.recommended_steps,
            recommended_sampler=model.recommended_sampler,
            recommended_cfg_scale=model.recommended_cfg_scale,
            default_positive_prompt=model.default_positive_prompt,
            default_negative_prompt=model.default_negative_prompt,
            is_nsfw=model.is_nsfw,
            is_active=model.is_active,
            usage_count=model.usage_count,
            last_used_at=model.last_used_at,
            extra_metadata=model.extra_metadata,
            created_at=model.created_at,
            updated_at=model.updated_at,
        )
