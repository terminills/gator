"""
Settings API Routes

Provides endpoints for managing system settings stored in the database.
"""

from typing import Any, Dict, List

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from backend.config.logging import get_logger
from backend.database.connection import get_db_session
from backend.models.settings import (
    SettingCategory,
    SettingCreate,
    SettingResponse,
    SettingUpdate,
)
from backend.services.settings_service import SettingsService

logger = get_logger(__name__)
router = APIRouter(prefix="/settings", tags=["settings"])


def get_settings_service(
    db: AsyncSession = Depends(get_db_session),
) -> SettingsService:
    """Dependency injection for SettingsService."""
    return SettingsService(db)


@router.get("/", response_model=List[SettingResponse])
async def list_settings(
    category: SettingCategory = None,
    service: SettingsService = Depends(get_settings_service),
):
    """
    List all settings or filter by category.

    Args:
        category: Optional category filter
        service: Settings service

    Returns:
        List of settings
    """
    if category:
        return await service.get_settings_by_category(category)
    return await service.list_all_settings()


@router.get("/{key}", response_model=SettingResponse)
async def get_setting(
    key: str,
    service: SettingsService = Depends(get_settings_service),
):
    """
    Get a specific setting by key.

    Args:
        key: Setting key
        service: Settings service

    Returns:
        Setting details

    Raises:
        404: Setting not found
    """
    setting = await service.get_setting(key)
    if not setting:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=f"Setting '{key}' not found"
        )
    return setting


@router.post("/", response_model=SettingResponse, status_code=status.HTTP_201_CREATED)
async def create_setting(
    setting_data: SettingCreate,
    service: SettingsService = Depends(get_settings_service),
):
    """
    Create a new setting.

    Args:
        setting_data: Setting creation data
        service: Settings service

    Returns:
        Created setting

    Raises:
        409: Setting key already exists
        500: Creation failed
    """
    result = await service.create_setting(setting_data)
    if not result:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Setting '{setting_data.key}' already exists",
        )
    return result


@router.put("/{key}", response_model=SettingResponse)
async def update_setting(
    key: str,
    setting_data: SettingUpdate,
    service: SettingsService = Depends(get_settings_service),
):
    """
    Update an existing setting.

    Args:
        key: Setting key
        setting_data: Update data
        service: Settings service

    Returns:
        Updated setting

    Raises:
        404: Setting not found
        500: Update failed
    """
    result = await service.update_setting(key, setting_data)
    if not result:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=f"Setting '{key}' not found"
        )
    return result


@router.delete("/{key}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_setting(
    key: str,
    service: SettingsService = Depends(get_settings_service),
):
    """
    Delete a setting (soft delete - marks as inactive).

    Args:
        key: Setting key
        service: Settings service

    Raises:
        404: Setting not found
    """
    result = await service.delete_setting(key)
    if not result:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=f"Setting '{key}' not found"
        )


@router.post("/bulk-update", response_model=Dict[str, Any])
async def bulk_update_settings(
    settings: Dict[str, Any],
    service: SettingsService = Depends(get_settings_service),
):
    """
    Update multiple settings at once.

    Args:
        settings: Dict of key-value pairs to update
        service: Settings service

    Returns:
        Summary of updates
    """
    results = {
        "updated": [],
        "failed": [],
    }

    for key, value in settings.items():
        try:
            # Determine category from key prefix or name
            if key.startswith("ipmi_"):
                category = SettingCategory.IPMI
            elif key.startswith("smtp_") or "email" in key.lower():
                category = SettingCategory.EMAIL
            elif key.startswith("aws_"):
                category = SettingCategory.CLOUD
            elif key.startswith("godaddy_") or "dns" in key.lower():
                category = SettingCategory.DNS
            elif (
                "facebook" in key.lower()
                or "instagram" in key.lower()
                or "twitter" in key.lower()
            ):
                category = SettingCategory.SOCIAL_MEDIA
            elif key.startswith("nsfw_") or key in ["civitai_allow_nsfw"]:
                category = SettingCategory.CONTENT
            else:
                category = SettingCategory.AI_MODELS  # Default

            setting_data = SettingCreate(
                key=key,
                category=category,
                value=value,
                is_sensitive="api_key" in key.lower()
                or "secret" in key.lower()
                or "token" in key.lower()
                or "password" in key.lower(),
            )

            result = await service.upsert_setting(setting_data)
            if result:
                results["updated"].append(key)
            else:
                results["failed"].append(key)

        except Exception as e:
            logger.error(f"Failed to update setting {key}: {e}")
            results["failed"].append(key)

    return {
        "success": len(results["failed"]) == 0,
        "message": f"Updated {len(results['updated'])} settings, {len(results['failed'])} failed",
        "results": results,
    }
