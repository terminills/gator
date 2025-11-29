"""
Branding API Routes

Provides endpoints for site branding and customization.
Stores branding in database for dynamic updates without restarts.
"""


from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.config.logging import get_logger
from backend.database.connection import get_db_session
from backend.models.branding import (
    BrandingCreate,
    BrandingModel,
    BrandingResponse,
    BrandingUpdate,
)

logger = get_logger(__name__)

router = APIRouter(
    prefix="/api/v1/branding",
    tags=["branding"],
)


@router.get("", response_model=BrandingResponse)
async def get_branding(db: AsyncSession = Depends(get_db_session)):
    """
    Get current site branding configuration from database.

    Returns customizable branding for the current installation.
    Each tenant/site can have their own branding while powered by Gator software.

    Args:
        db: Database session

    Returns:
        BrandingResponse: Site branding configuration
    """
    try:
        # Get active branding from database
        query = select(BrandingModel).where(BrandingModel.is_active == True).limit(1)
        result = await db.execute(query)
        branding = result.scalar_one_or_none()

        if not branding:
            # Create default branding if none exists
            branding = BrandingModel(
                site_name="AI Content Platform",
                site_icon="🤖",
                instance_name="My AI Platform",
                site_tagline="AI-Powered Content Generation",
                primary_color="#667eea",
                accent_color="#10b981",
                is_active=True,
            )
            db.add(branding)
            await db.commit()
            await db.refresh(branding)
            logger.info("Created default branding configuration")

        # Convert to response model
        return BrandingResponse(
            id=str(branding.id),
            site_name=branding.site_name,
            site_icon=branding.site_icon,
            instance_name=branding.instance_name,
            site_tagline=branding.site_tagline,
            primary_color=branding.primary_color,
            accent_color=branding.accent_color,
            logo_url=branding.logo_url,
            favicon_url=branding.favicon_url,
            custom_css=branding.custom_css,
            created_at=branding.created_at,
            updated_at=branding.updated_at,
        )

    except Exception as e:
        logger.error(f"Failed to get branding: {e}")
        # Return default branding as fallback
        from datetime import datetime, timezone

        return BrandingResponse(
            id="default",
            site_name="AI Content Platform",
            site_icon="🤖",
            instance_name="My AI Platform",
            site_tagline="AI-Powered Content Generation",
            primary_color="#667eea",
            accent_color="#10b981",
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
        )


@router.put("", response_model=BrandingResponse)
async def update_branding(
    branding_update: BrandingUpdate, db: AsyncSession = Depends(get_db_session)
):
    """
    Update site branding configuration in database.

    Updates the active branding record with new values.
    Changes take effect immediately without restarting.

    Args:
        branding_update: New branding values
        db: Database session

    Returns:
        Updated branding configuration
    """
    try:
        # Get active branding
        query = select(BrandingModel).where(BrandingModel.is_active == True).limit(1)
        result = await db.execute(query)
        branding = result.scalar_one_or_none()

        if not branding:
            raise HTTPException(status_code=404, detail="No active branding found")

        # Update fields that were provided
        update_data = branding_update.model_dump(exclude_unset=True)
        for field, value in update_data.items():
            setattr(branding, field, value)

        await db.commit()
        await db.refresh(branding)

        logger.info(f"Updated branding configuration: {update_data.keys()}")

        return BrandingResponse(
            id=str(branding.id),
            site_name=branding.site_name,
            site_icon=branding.site_icon,
            instance_name=branding.instance_name,
            site_tagline=branding.site_tagline,
            primary_color=branding.primary_color,
            accent_color=branding.accent_color,
            logo_url=branding.logo_url,
            favicon_url=branding.favicon_url,
            custom_css=branding.custom_css,
            created_at=branding.created_at,
            updated_at=branding.updated_at,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to update branding: {e}")
        await db.rollback()
        raise HTTPException(status_code=500, detail="Failed to update branding")


@router.post("", response_model=BrandingResponse)
async def create_branding(
    branding_data: BrandingCreate, db: AsyncSession = Depends(get_db_session)
):
    """
    Create new branding configuration.

    Note: Only one active branding configuration should exist.
    This endpoint is mainly for initial setup.

    Args:
        branding_data: Branding configuration
        db: Database session

    Returns:
        Created branding configuration
    """
    try:
        # Deactivate any existing branding
        await db.execute(select(BrandingModel).where(BrandingModel.is_active == True))

        # Create new branding
        branding = BrandingModel(**branding_data.model_dump(), is_active=True)
        db.add(branding)
        await db.commit()
        await db.refresh(branding)

        logger.info("Created new branding configuration")

        return BrandingResponse(
            id=str(branding.id),
            site_name=branding.site_name,
            site_icon=branding.site_icon,
            instance_name=branding.instance_name,
            site_tagline=branding.site_tagline,
            primary_color=branding.primary_color,
            accent_color=branding.accent_color,
            logo_url=branding.logo_url,
            favicon_url=branding.favicon_url,
            custom_css=branding.custom_css,
            created_at=branding.created_at,
            updated_at=branding.updated_at,
        )

    except Exception as e:
        logger.error(f"Failed to create branding: {e}")
        await db.rollback()
        raise HTTPException(status_code=500, detail="Failed to create branding")
