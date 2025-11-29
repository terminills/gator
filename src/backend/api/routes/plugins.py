"""
Plugin/Marketplace API Routes

REST API endpoints for plugin discovery, installation, and management.
"""

from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import and_, desc, select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.config.logging import get_logger
from backend.database.connection import get_db_session
from backend.models.plugin import (
    PluginInstallation,
    PluginInstallationSchema,
    PluginInstallRequest,
    PluginModel,
    PluginReview,
    PluginReviewRequest,
    PluginReviewSchema,
    PluginSchema,
    PluginUpdateRequest,
)
from backend.plugins import PluginStatus, PluginType
from backend.plugins.manager import plugin_manager

logger = get_logger(__name__)

router = APIRouter(prefix="/api/v1", tags=["plugins"])


@router.get("/plugins/marketplace", response_model=List[PluginSchema])
async def list_marketplace_plugins(
    plugin_type: Optional[PluginType] = None,
    featured: Optional[bool] = None,
    search: Optional[str] = None,
    skip: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=100),
    db: AsyncSession = Depends(get_db_session),
):
    """
    List plugins available in the marketplace.

    Args:
        plugin_type: Filter by plugin type
        featured: Filter featured plugins only
        search: Search in name and description
        skip: Number of records to skip
        limit: Maximum number of records to return
        db: Database session

    Returns:
        List of marketplace plugins
    """
    try:
        # Build query
        query = select(PluginModel).where(PluginModel.deprecated.is_(False))

        # Apply filters
        if plugin_type:
            query = query.where(PluginModel.plugin_type == plugin_type)

        if featured:
            query = query.where(PluginModel.featured.is_(True))

        if search:
            search_pattern = f"%{search}%"
            query = query.where(
                (PluginModel.name.ilike(search_pattern))
                | (PluginModel.description.ilike(search_pattern))
            )

        # Order by featured, rating, downloads
        query = (
            query.order_by(
                desc(PluginModel.featured),
                desc(PluginModel.rating),
                desc(PluginModel.downloads),
            )
            .offset(skip)
            .limit(limit)
        )

        result = await db.execute(query)
        plugins = result.scalars().all()

        return [PluginSchema.from_orm(p) for p in plugins]

    except Exception as e:
        logger.error(f"Failed to list marketplace plugins: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to fetch plugins")


@router.get("/plugins/marketplace/{plugin_slug}", response_model=PluginSchema)
async def get_marketplace_plugin(
    plugin_slug: str,
    db: AsyncSession = Depends(get_db_session),
):
    """
    Get details of a specific marketplace plugin.

    Args:
        plugin_slug: Plugin slug identifier
        db: Database session

    Returns:
        Plugin details
    """
    try:
        result = await db.execute(
            select(PluginModel).where(PluginModel.slug == plugin_slug)
        )
        plugin = result.scalar_one_or_none()

        if not plugin:
            raise HTTPException(status_code=404, detail="Plugin not found")

        return PluginSchema.from_orm(plugin)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get plugin {plugin_slug}: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to fetch plugin")


@router.get("/plugins/installed", response_model=List[PluginInstallationSchema])
async def list_installed_plugins(
    enabled: Optional[bool] = None,
    db: AsyncSession = Depends(get_db_session),
):
    """
    List installed plugins for the current user/tenant.

    Args:
        enabled: Filter by enabled status
        db: Database session

    Returns:
        List of installed plugins
    """
    try:
        # Build query
        query = select(PluginInstallation)

        # Apply filters
        if enabled is not None:
            query = query.where(PluginInstallation.enabled == enabled)

        # NOTE: User/tenant filtering requires authentication system
        # Once authentication is implemented, add:
        # from backend.api.dependencies import get_current_user
        # user = Depends(get_current_user)
        # query = query.where(PluginInstallation.user_id == user.id)
        # For multi-tenancy: query = query.where(PluginInstallation.tenant_id == user.tenant_id)

        result = await db.execute(query)
        installations = result.scalars().all()

        return [PluginInstallationSchema.from_orm(i) for i in installations]

    except Exception as e:
        logger.error(f"Failed to list installed plugins: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to fetch installed plugins")


@router.post(
    "/plugins/install", response_model=PluginInstallationSchema, status_code=201
)
async def install_plugin(
    request: PluginInstallRequest,
    db: AsyncSession = Depends(get_db_session),
):
    """
    Install a plugin from the marketplace.

    Args:
        request: Plugin installation request
        db: Database session

    Returns:
        Plugin installation details
    """
    try:
        # Get plugin from marketplace
        result = await db.execute(
            select(PluginModel).where(PluginModel.slug == request.plugin_slug)
        )
        plugin = result.scalar_one_or_none()

        if not plugin:
            raise HTTPException(
                status_code=404, detail="Plugin not found in marketplace"
            )

        # Check if already installed
        existing = await db.execute(
            select(PluginInstallation).where(
                PluginInstallation.plugin_slug == request.plugin_slug
            )
        )
        if existing.scalar_one_or_none():
            raise HTTPException(status_code=409, detail="Plugin already installed")

        # Create installation record
        installation = PluginInstallation(
            plugin_id=plugin.id,
            plugin_slug=plugin.slug,
            plugin_version=plugin.version,
            config=request.config,
            status=PluginStatus.INSTALLED,
        )

        db.add(installation)
        await db.commit()
        await db.refresh(installation)

        # Increment download count
        plugin.downloads += 1
        await db.commit()

        logger.info(f"Plugin installed: {request.plugin_slug}")

        return PluginInstallationSchema.from_orm(installation)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to install plugin {request.plugin_slug}: {str(e)}")
        await db.rollback()
        raise HTTPException(status_code=500, detail="Failed to install plugin")


@router.put(
    "/plugins/installed/{installation_id}", response_model=PluginInstallationSchema
)
async def update_plugin_installation(
    installation_id: str,
    request: PluginUpdateRequest,
    db: AsyncSession = Depends(get_db_session),
):
    """
    Update plugin installation configuration or status.

    Args:
        installation_id: Installation ID
        request: Update request
        db: Database session

    Returns:
        Updated installation details
    """
    try:
        result = await db.execute(
            select(PluginInstallation).where(PluginInstallation.id == installation_id)
        )
        installation = result.scalar_one_or_none()

        if not installation:
            raise HTTPException(status_code=404, detail="Plugin installation not found")

        # Update fields
        if request.config is not None:
            installation.config = request.config

        if request.enabled is not None:
            installation.enabled = request.enabled

        await db.commit()
        await db.refresh(installation)

        logger.info(f"Plugin installation updated: {installation_id}")

        return PluginInstallationSchema.from_orm(installation)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to update plugin installation: {str(e)}")
        await db.rollback()
        raise HTTPException(status_code=500, detail="Failed to update plugin")


@router.delete("/plugins/installed/{installation_id}", status_code=204)
async def uninstall_plugin(
    installation_id: str,
    db: AsyncSession = Depends(get_db_session),
):
    """
    Uninstall a plugin.

    Args:
        installation_id: Installation ID
        db: Database session
    """
    try:
        result = await db.execute(
            select(PluginInstallation).where(PluginInstallation.id == installation_id)
        )
        installation = result.scalar_one_or_none()

        if not installation:
            raise HTTPException(status_code=404, detail="Plugin installation not found")

        await db.delete(installation)
        await db.commit()

        logger.info(f"Plugin uninstalled: {installation_id}")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to uninstall plugin: {str(e)}")
        await db.rollback()
        raise HTTPException(status_code=500, detail="Failed to uninstall plugin")


@router.get(
    "/plugins/marketplace/{plugin_slug}/reviews",
    response_model=List[PluginReviewSchema],
)
async def list_plugin_reviews(
    plugin_slug: str,
    skip: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=100),
    db: AsyncSession = Depends(get_db_session),
):
    """
    List reviews for a plugin.

    Args:
        plugin_slug: Plugin slug
        skip: Number of records to skip
        limit: Maximum number of records to return
        db: Database session

    Returns:
        List of plugin reviews
    """
    try:
        # Get plugin
        result = await db.execute(
            select(PluginModel).where(PluginModel.slug == plugin_slug)
        )
        plugin = result.scalar_one_or_none()

        if not plugin:
            raise HTTPException(status_code=404, detail="Plugin not found")

        # Get reviews
        result = await db.execute(
            select(PluginReview)
            .where(PluginReview.plugin_id == plugin.id)
            .order_by(desc(PluginReview.helpful_count), desc(PluginReview.created_at))
            .offset(skip)
            .limit(limit)
        )
        reviews = result.scalars().all()

        return [PluginReviewSchema.from_orm(r) for r in reviews]

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to list plugin reviews: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to fetch reviews")


@router.post(
    "/plugins/marketplace/{plugin_slug}/reviews",
    response_model=PluginReviewSchema,
    status_code=201,
)
async def create_plugin_review(
    plugin_slug: str,
    request: PluginReviewRequest,
    db: AsyncSession = Depends(get_db_session),
):
    """
    Create a review for a plugin.

    Args:
        plugin_slug: Plugin slug
        request: Review request
        db: Database session

    Returns:
        Created review
    """
    try:
        # Get plugin
        result = await db.execute(
            select(PluginModel).where(PluginModel.slug == plugin_slug)
        )
        plugin = result.scalar_one_or_none()

        if not plugin:
            raise HTTPException(status_code=404, detail="Plugin not found")

        # NOTE: User ID extraction requires authentication system
        # Once authentication is implemented, add:
        # from backend.api.dependencies import get_current_user
        # current_user = Depends(get_current_user)
        # user_id = current_user.id
        # For now, using placeholder user_id for demo purposes
        user_id = "demo-user"

        # Check if user already reviewed
        existing = await db.execute(
            select(PluginReview).where(
                and_(
                    PluginReview.plugin_id == plugin.id,
                    PluginReview.user_id == user_id,
                )
            )
        )
        if existing.scalar_one_or_none():
            raise HTTPException(
                status_code=409, detail="You have already reviewed this plugin"
            )

        # Create review
        review = PluginReview(
            plugin_id=plugin.id,
            user_id=user_id,
            rating=request.rating,
            title=request.title,
            review_text=request.review_text,
        )

        db.add(review)

        # Update plugin rating
        plugin.rating_count += 1
        plugin.rating = (
            (plugin.rating * (plugin.rating_count - 1)) + request.rating
        ) / plugin.rating_count

        await db.commit()
        await db.refresh(review)

        logger.info(f"Review created for plugin: {plugin_slug}")

        return PluginReviewSchema.from_orm(review)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to create review: {str(e)}")
        await db.rollback()
        raise HTTPException(status_code=500, detail="Failed to create review")


@router.get("/plugins/manager/status")
async def get_plugin_manager_status():
    """
    Get plugin manager status and loaded plugins.

    Returns:
        Plugin manager status
    """
    try:
        plugins = plugin_manager.list_plugins()

        return {
            "loaded_plugins": len(plugins),
            "plugins": plugins,
            "hooks": {
                hook_name: len(plugin_ids)
                for hook_name, plugin_ids in plugin_manager.hooks.items()
            },
        }

    except Exception as e:
        logger.error(f"Failed to get plugin manager status: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to get manager status")
