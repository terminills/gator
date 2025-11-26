"""
Persona Management Service

Core business logic for AI persona management including validation,
moderation, and lifecycle management.
"""

import os
import shutil
from typing import List, Optional, Union
from datetime import datetime, timezone
from uuid import UUID
from pathlib import Path

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update
from sqlalchemy.exc import IntegrityError

from backend.models.persona import (
    PersonaModel,
    PersonaCreate,
    PersonaResponse,
    PersonaUpdate,
    BaseImageStatus,
)
from backend.config.logging import get_logger

logger = get_logger(__name__)


class PersonaService:
    """
    Service for managing AI personas.

    Handles the core business logic for persona management including
    creation, validation, and lifecycle management. Follows the
    patterns established in the template service.
    """

    def __init__(self, db_session: AsyncSession):
        """
        Initialize the service with a database session.

        Args:
            db_session: Async SQLAlchemy session
        """
        self.db = db_session

    async def create_persona(self, persona_data: PersonaCreate) -> PersonaResponse:
        """
        Create a new AI persona with validation.

        Args:
            persona_data: The persona configuration data

        Returns:
            PersonaResponse: The created persona with metadata

        Raises:
            ValueError: If persona data fails validation
        """
        try:
            # Create database model instance
            db_persona = PersonaModel(
                name=persona_data.name,
                appearance=persona_data.appearance,
                personality=persona_data.personality,
                content_themes=persona_data.content_themes,
                style_preferences=persona_data.style_preferences,
                default_content_rating=persona_data.default_content_rating.value,
                allowed_content_ratings=[
                    rating.value for rating in persona_data.allowed_content_ratings
                ],
                platform_restrictions=persona_data.platform_restrictions,
                base_appearance_description=persona_data.base_appearance_description,
                base_image_path=persona_data.base_image_path,
                appearance_locked=persona_data.appearance_locked,
                base_image_status=persona_data.base_image_status.value,
                image_style=persona_data.image_style.value,
                default_image_resolution=persona_data.default_image_resolution,
                default_video_resolution=persona_data.default_video_resolution,
                post_style=persona_data.post_style,
                video_types=persona_data.video_types,
                nsfw_model_preference=persona_data.nsfw_model_preference,
                generation_quality=persona_data.generation_quality,
                # ==================== PERSONA SOUL FIELDS ====================
                # Origin & Demographics
                hometown=persona_data.hometown,
                current_location=persona_data.current_location,
                generation_age=persona_data.generation_age,
                education_level=persona_data.education_level,
                # Psychological Profile
                mbti_type=persona_data.mbti_type,
                enneagram_type=persona_data.enneagram_type,
                political_alignment=persona_data.political_alignment,
                risk_tolerance=persona_data.risk_tolerance,
                optimism_cynicism_scale=persona_data.optimism_cynicism_scale,
                # Voice & Speech Patterns
                linguistic_register=persona_data.linguistic_register.value,
                typing_quirks=persona_data.typing_quirks,
                signature_phrases=persona_data.signature_phrases,
                trigger_topics=persona_data.trigger_topics,
                # Backstory & Lore
                day_job=persona_data.day_job,
                war_story=persona_data.war_story,
                vices_hobbies=persona_data.vices_hobbies,
                # Anti-Pattern
                forbidden_phrases=persona_data.forbidden_phrases,
                warmth_level=persona_data.warmth_level.value,
                patience_level=persona_data.patience_level.value,
            )

            # Add to session and commit
            self.db.add(db_persona)
            await self.db.commit()
            await self.db.refresh(db_persona)

            logger.info(f"Created new persona {db_persona.id}: {db_persona.name}")

            # Convert to response model
            return PersonaResponse.model_validate(db_persona)

        except IntegrityError as e:
            await self.db.rollback()
            logger.error(f"Database integrity error creating persona: {str(e)}")
            raise ValueError("Failed to create persona due to data constraints")

        except Exception as e:
            await self.db.rollback()
            logger.error(f"Unexpected error creating persona: {str(e)}")
            raise ValueError(f"Persona creation failed: {str(e)}")

    async def get_persona(
        self, persona_id: Union[str, UUID]
    ) -> Optional[PersonaResponse]:
        """
        Retrieve a persona by ID.

        Args:
            persona_id: The unique persona identifier (string UUID or UUID object)

        Returns:
            PersonaResponse: The requested persona, or None if not found
        """
        try:
            # Convert to UUID object if needed
            import uuid

            if isinstance(persona_id, str):
                try:
                    uuid_id = uuid.UUID(persona_id)
                except ValueError:
                    # Invalid UUID format
                    logger.debug(f"Invalid UUID format for persona_id: {persona_id}")
                    return None
            elif isinstance(persona_id, uuid.UUID):
                uuid_id = persona_id
            else:
                logger.debug(f"Invalid persona_id type: {type(persona_id)}")
                return None

            stmt = select(PersonaModel).where(PersonaModel.id == uuid_id)
            result = await self.db.execute(stmt)
            db_persona = result.scalar_one_or_none()

            if not db_persona:
                logger.debug(f"Persona not found: {persona_id}")
                return None

            return PersonaResponse.model_validate(db_persona)

        except Exception as e:
            logger.error(f"Failed to get persona {persona_id}: {str(e)}")
            raise

    async def list_personas(
        self, skip: int = 0, limit: int = 10, active_only: bool = True
    ) -> List[PersonaResponse]:
        """
        List personas with pagination and filtering.

        Args:
            skip: Number of records to skip
            limit: Maximum number of records to return
            active_only: Whether to return only active personas

        Returns:
            List[PersonaResponse]: List of personas
        """
        try:
            stmt = select(PersonaModel)

            if active_only:
                stmt = stmt.where(PersonaModel.is_active == True)

            stmt = (
                stmt.offset(skip).limit(limit).order_by(PersonaModel.created_at.desc())
            )

            result = await self.db.execute(stmt)
            db_personas = result.scalars().all()

            return [PersonaResponse.model_validate(persona) for persona in db_personas]

        except Exception as e:
            logger.error(f"Failed to list personas: {str(e)}")
            raise

    async def update_persona(
        self, persona_id: Union[str, UUID], updates: PersonaUpdate
    ) -> Optional[PersonaResponse]:
        """
        Update an existing persona.

        Args:
            persona_id: The persona to update
            updates: Fields to update

        Returns:
            PersonaResponse: Updated persona, or None if not found

        Raises:
            ValueError: If updates fail validation
        """
        try:
            # First check if persona exists
            existing = await self.get_persona(persona_id)
            if not existing:
                return None

            # Build update dictionary with only provided fields
            update_data = {}
            if updates.name is not None:
                update_data["name"] = updates.name
            if updates.appearance is not None:
                update_data["appearance"] = updates.appearance
            if updates.personality is not None:
                update_data["personality"] = updates.personality
            if updates.content_themes is not None:
                update_data["content_themes"] = updates.content_themes
            if updates.style_preferences is not None:
                update_data["style_preferences"] = updates.style_preferences
            if updates.default_content_rating is not None:
                update_data["default_content_rating"] = updates.default_content_rating.value
            if updates.allowed_content_ratings is not None:
                update_data["allowed_content_ratings"] = [
                    rating.value for rating in updates.allowed_content_ratings
                ]
            if updates.platform_restrictions is not None:
                update_data["platform_restrictions"] = updates.platform_restrictions
            if updates.is_active is not None:
                update_data["is_active"] = updates.is_active
            if updates.base_appearance_description is not None:
                update_data["base_appearance_description"] = (
                    updates.base_appearance_description
                )
            if updates.base_image_path is not None:
                update_data["base_image_path"] = updates.base_image_path
            if updates.appearance_locked is not None:
                update_data["appearance_locked"] = updates.appearance_locked
            if updates.base_image_status is not None:
                update_data["base_image_status"] = updates.base_image_status.value
            if updates.image_style is not None:
                update_data["image_style"] = updates.image_style.value
            if updates.default_image_resolution is not None:
                update_data["default_image_resolution"] = updates.default_image_resolution
            if updates.default_video_resolution is not None:
                update_data["default_video_resolution"] = updates.default_video_resolution
            if updates.post_style is not None:
                update_data["post_style"] = updates.post_style
            if updates.video_types is not None:
                update_data["video_types"] = updates.video_types
            if updates.nsfw_model_preference is not None:
                update_data["nsfw_model_preference"] = updates.nsfw_model_preference
            if updates.generation_quality is not None:
                update_data["generation_quality"] = updates.generation_quality

            # ==================== PERSONA SOUL FIELDS ====================
            # Origin & Demographics
            if updates.hometown is not None:
                update_data["hometown"] = updates.hometown
            if updates.current_location is not None:
                update_data["current_location"] = updates.current_location
            if updates.generation_age is not None:
                update_data["generation_age"] = updates.generation_age
            if updates.education_level is not None:
                update_data["education_level"] = updates.education_level

            # Psychological Profile
            if updates.mbti_type is not None:
                update_data["mbti_type"] = updates.mbti_type
            if updates.enneagram_type is not None:
                update_data["enneagram_type"] = updates.enneagram_type
            if updates.political_alignment is not None:
                update_data["political_alignment"] = updates.political_alignment
            if updates.risk_tolerance is not None:
                update_data["risk_tolerance"] = updates.risk_tolerance
            if updates.optimism_cynicism_scale is not None:
                update_data["optimism_cynicism_scale"] = updates.optimism_cynicism_scale

            # Voice & Speech Patterns
            if updates.linguistic_register is not None:
                update_data["linguistic_register"] = updates.linguistic_register.value
            if updates.typing_quirks is not None:
                update_data["typing_quirks"] = updates.typing_quirks
            if updates.signature_phrases is not None:
                update_data["signature_phrases"] = updates.signature_phrases
            if updates.trigger_topics is not None:
                update_data["trigger_topics"] = updates.trigger_topics

            # Backstory & Lore
            if updates.day_job is not None:
                update_data["day_job"] = updates.day_job
            if updates.war_story is not None:
                update_data["war_story"] = updates.war_story
            if updates.vices_hobbies is not None:
                update_data["vices_hobbies"] = updates.vices_hobbies

            # Anti-Pattern
            if updates.forbidden_phrases is not None:
                update_data["forbidden_phrases"] = updates.forbidden_phrases
            if updates.warmth_level is not None:
                update_data["warmth_level"] = updates.warmth_level.value
            if updates.patience_level is not None:
                update_data["patience_level"] = updates.patience_level.value

            if not update_data:
                # No updates provided, return existing persona
                return existing

            # Add updated timestamp
            update_data["updated_at"] = datetime.now(timezone.utc)

            # Perform update
            import uuid

            if isinstance(persona_id, str):
                uuid_id = uuid.UUID(persona_id)
            elif isinstance(persona_id, uuid.UUID):
                uuid_id = persona_id
            else:
                raise ValueError(f"Invalid persona_id type: {type(persona_id)}")

            stmt = (
                update(PersonaModel)
                .where(PersonaModel.id == uuid_id)
                .values(**update_data)
            )

            await self.db.execute(stmt)
            await self.db.commit()
            
            # Expire session cache to ensure fresh data on next query
            # This is necessary because Core update() doesn't update the session identity map
            self.db.expire_all()

            logger.info(f"Updated persona {persona_id}: {list(update_data.keys())}")

            # Return updated persona
            return await self.get_persona(persona_id)

        except IntegrityError as e:
            await self.db.rollback()
            logger.error(
                f"Database integrity error updating persona {persona_id}: {str(e)}"
            )
            raise ValueError("Failed to update persona due to data constraints")

        except Exception as e:
            await self.db.rollback()
            logger.error(f"Failed to update persona {persona_id}: {str(e)}")
            raise

    async def delete_persona(self, persona_id: Union[str, UUID]) -> bool:
        """
        Soft delete a persona by marking it inactive.

        Args:
            persona_id: The persona to delete

        Returns:
            bool: True if deletion successful, False if persona not found
        """
        try:
            # Check if persona exists
            existing = await self.get_persona(persona_id)
            if not existing:
                return False

            # Soft delete by marking inactive
            import uuid

            if isinstance(persona_id, str):
                uuid_id = uuid.UUID(persona_id)
            elif isinstance(persona_id, uuid.UUID):
                uuid_id = persona_id
            else:
                raise ValueError(f"Invalid persona_id type: {type(persona_id)}")

            stmt = (
                update(PersonaModel)
                .where(PersonaModel.id == uuid_id)
                .values(is_active=False, updated_at=datetime.now(timezone.utc))
            )

            result = await self.db.execute(stmt)
            await self.db.commit()

            if result.rowcount > 0:
                logger.info(f"Soft deleted persona: {persona_id}")
                return True

            return False

        except Exception as e:
            await self.db.rollback()
            logger.error(f"Failed to delete persona {persona_id}: {str(e)}")
            raise

    async def increment_generation_count(self, persona_id: Union[str, UUID]) -> bool:
        """
        Increment the generation count for a persona.

        This method is called when content is generated using this persona.

        Args:
            persona_id: The persona to update (string UUID or UUID object)

        Returns:
            bool: True if successful, False if persona not found
        """
        try:
            # Convert to UUID object if needed
            import uuid

            if isinstance(persona_id, str):
                try:
                    uuid_id = uuid.UUID(persona_id)
                except ValueError:
                    logger.debug(f"Invalid UUID format for persona_id: {persona_id}")
                    return False
            elif isinstance(persona_id, uuid.UUID):
                uuid_id = persona_id
            else:
                logger.debug(f"Invalid persona_id type: {type(persona_id)}")
                return False

            stmt = (
                update(PersonaModel)
                .where(PersonaModel.id == uuid_id)
                .values(
                    generation_count=PersonaModel.generation_count + 1,
                    updated_at=datetime.now(timezone.utc),
                )
            )

            result = await self.db.execute(stmt)
            await self.db.commit()

            if result.rowcount > 0:
                logger.debug(f"Incremented generation count for persona: {persona_id}")
                return True

            return False

        except Exception as e:
            await self.db.rollback()
            logger.error(f"Failed to increment generation count {persona_id}: {str(e)}")
            return False

    async def _save_image_to_disk(
        self,
        persona_id: Union[str, UUID],
        image_data: bytes,
        filename: Optional[str] = None,
    ) -> str:
        """
        Save image data to disk for a persona.

        Args:
            persona_id: The persona this image belongs to
            image_data: Raw image bytes
            filename: Optional custom filename

        Returns:
            str: Path to the saved image file

        Raises:
            ValueError: If image save fails
        """
        try:
            # Create base images directory if it doesn't exist
            base_images_dir = Path("/opt/gator/data/models/base_images")
            base_images_dir.mkdir(parents=True, exist_ok=True)

            # Generate filename if not provided
            if filename is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"persona_{persona_id}_{timestamp}.png"

            # Log the size before writing
            logger.info(f"Writing {len(image_data)} bytes to disk for persona {persona_id}")

            # Save image to disk
            file_path = base_images_dir / filename
            bytes_written = 0
            with open(file_path, "wb") as f:
                bytes_written = f.write(image_data)

            # Verify the write was successful
            actual_file_size = file_path.stat().st_size
            logger.info(
                f"Saved base image for persona {persona_id}: {file_path} "
                f"(wrote {bytes_written} bytes, file size: {actual_file_size} bytes)"
            )
            
            # Check for size mismatch
            if actual_file_size != len(image_data):
                logger.error(
                    f"⚠️  SIZE MISMATCH: Expected {len(image_data)} bytes, "
                    f"but file is {actual_file_size} bytes!"
                )
            
            return str(file_path)

        except Exception as e:
            logger.error(f"Failed to save image to disk: {str(e)}")
            raise ValueError(f"Failed to save image: {str(e)}")

    async def approve_baseline(
        self, persona_id: Union[str, UUID]
    ) -> Optional[PersonaResponse]:
        """
        Approve the baseline image for a persona.

        Sets base_image_status to APPROVED and appearance_locked to True,
        enabling visual consistency features.

        Args:
            persona_id: The persona to approve

        Returns:
            PersonaResponse: Updated persona, or None if not found

        Raises:
            ValueError: If persona doesn't have a base image
        """
        try:
            # Get the persona
            persona = await self.get_persona(persona_id)
            if not persona:
                return None

            # Check if persona has a base image
            if not persona.base_image_path:
                raise ValueError(
                    "Cannot approve baseline: persona does not have a base image"
                )

            # Convert to UUID object if needed
            import uuid

            if isinstance(persona_id, str):
                uuid_id = uuid.UUID(persona_id)
            elif isinstance(persona_id, uuid.UUID):
                uuid_id = persona_id
            else:
                raise ValueError(f"Invalid persona_id type: {type(persona_id)}")

            # Update status to APPROVED and lock appearance
            stmt = (
                update(PersonaModel)
                .where(PersonaModel.id == uuid_id)
                .values(
                    base_image_status=BaseImageStatus.APPROVED.value,
                    appearance_locked=True,
                    updated_at=datetime.now(timezone.utc),
                )
            )

            await self.db.execute(stmt)
            await self.db.commit()
            
            # Expire session cache to ensure fresh data on next query
            # This is necessary because Core update() doesn't update the session identity map
            self.db.expire_all()

            logger.info(f"Approved baseline image for persona {persona_id}")

            # Return updated persona
            return await self.get_persona(persona_id)

        except Exception as e:
            await self.db.rollback()
            logger.error(
                f"Failed to approve baseline for persona {persona_id}: {str(e)}"
            )
            raise
