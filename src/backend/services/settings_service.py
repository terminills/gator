"""
Settings Service

Manages system settings stored in the database.
Provides CRUD operations for configuration values.
"""

from typing import Dict, List, Optional, Any
from uuid import UUID
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update, delete
from sqlalchemy.exc import IntegrityError

from backend.models.settings import (
    SystemSettingModel,
    SettingCreate,
    SettingUpdate,
    SettingResponse,
    SettingCategory,
)
from backend.config.logging import get_logger

logger = get_logger(__name__)


class SettingsService:
    """Service for managing system settings in database."""

    def __init__(self, db_session: AsyncSession):
        """
        Initialize settings service.

        Args:
            db_session: Database session
        """
        self.db = db_session

    async def get_setting(self, key: str) -> Optional[SettingResponse]:
        """
        Get a setting by key.

        Args:
            key: Setting key

        Returns:
            Setting if found, None otherwise
        """
        try:
            stmt = select(SystemSettingModel).where(
                SystemSettingModel.key == key,
                SystemSettingModel.is_active == True
            )
            result = await self.db.execute(stmt)
            setting = result.scalar_one_or_none()

            if setting:
                return SettingResponse.model_validate(setting)
            return None

        except Exception as e:
            logger.error(f"Error getting setting {key}: {e}")
            return None

    async def get_settings_by_category(
        self, category: SettingCategory
    ) -> List[SettingResponse]:
        """
        Get all settings in a category.

        Args:
            category: Setting category

        Returns:
            List of settings
        """
        try:
            stmt = select(SystemSettingModel).where(
                SystemSettingModel.category == category.value,
                SystemSettingModel.is_active == True
            ).order_by(SystemSettingModel.key)

            result = await self.db.execute(stmt)
            settings = result.scalars().all()

            return [SettingResponse.model_validate(s) for s in settings]

        except Exception as e:
            logger.error(f"Error getting settings for category {category}: {e}")
            return []

    async def list_all_settings(self) -> List[SettingResponse]:
        """
        List all active settings.

        Returns:
            List of all settings
        """
        try:
            stmt = select(SystemSettingModel).where(
                SystemSettingModel.is_active == True
            ).order_by(SystemSettingModel.category, SystemSettingModel.key)

            result = await self.db.execute(stmt)
            settings = result.scalars().all()

            return [SettingResponse.model_validate(s) for s in settings]

        except Exception as e:
            logger.error(f"Error listing all settings: {e}")
            return []

    async def create_setting(self, setting_data: SettingCreate) -> Optional[SettingResponse]:
        """
        Create a new setting.

        Args:
            setting_data: Setting creation data

        Returns:
            Created setting or None if failed
        """
        try:
            db_setting = SystemSettingModel(
                key=setting_data.key,
                category=setting_data.category.value,
                value=setting_data.value,
                description=setting_data.description,
                is_sensitive=setting_data.is_sensitive,
            )

            self.db.add(db_setting)
            await self.db.commit()
            await self.db.refresh(db_setting)

            logger.info(f"Created setting: {setting_data.key}")
            return SettingResponse.model_validate(db_setting)

        except IntegrityError:
            await self.db.rollback()
            logger.error(f"Setting key already exists: {setting_data.key}")
            return None
        except Exception as e:
            await self.db.rollback()
            logger.error(f"Error creating setting: {e}")
            return None

    async def update_setting(
        self, key: str, setting_data: SettingUpdate
    ) -> Optional[SettingResponse]:
        """
        Update an existing setting.

        Args:
            key: Setting key
            setting_data: Update data

        Returns:
            Updated setting or None if failed
        """
        try:
            # Build update dict
            update_data = {}
            if setting_data.value is not None:
                update_data["value"] = setting_data.value
            if setting_data.description is not None:
                update_data["description"] = setting_data.description
            if setting_data.is_sensitive is not None:
                update_data["is_sensitive"] = setting_data.is_sensitive

            if not update_data:
                return await self.get_setting(key)

            # Update setting
            stmt = (
                update(SystemSettingModel)
                .where(SystemSettingModel.key == key)
                .values(**update_data)
                .returning(SystemSettingModel)
            )

            result = await self.db.execute(stmt)
            await self.db.commit()

            updated_setting = result.scalar_one_or_none()
            if updated_setting:
                logger.info(f"Updated setting: {key}")
                return SettingResponse.model_validate(updated_setting)

            return None

        except Exception as e:
            await self.db.rollback()
            logger.error(f"Error updating setting {key}: {e}")
            return None

    async def delete_setting(self, key: str) -> bool:
        """
        Soft delete a setting (mark as inactive).

        Args:
            key: Setting key

        Returns:
            True if deleted, False otherwise
        """
        try:
            stmt = (
                update(SystemSettingModel)
                .where(SystemSettingModel.key == key)
                .values(is_active=False)
            )

            result = await self.db.execute(stmt)
            await self.db.commit()

            if result.rowcount > 0:
                logger.info(f"Deleted setting: {key}")
                return True

            return False

        except Exception as e:
            await self.db.rollback()
            logger.error(f"Error deleting setting {key}: {e}")
            return False

    async def upsert_setting(self, setting_data: SettingCreate) -> SettingResponse:
        """
        Create or update a setting.

        Args:
            setting_data: Setting data

        Returns:
            Setting response
        """
        # Try to get existing setting
        existing = await self.get_setting(setting_data.key)

        if existing:
            # Update existing
            update_data = SettingUpdate(
                value=setting_data.value,
                description=setting_data.description,
                is_sensitive=setting_data.is_sensitive,
            )
            result = await self.update_setting(setting_data.key, update_data)
            return result

        # Create new
        return await self.create_setting(setting_data)
