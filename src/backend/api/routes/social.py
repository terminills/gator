"""
Social Media Integration API Routes

Handles social media account management, content publishing, and engagement tracking.
"""

from typing import List, Dict, Any
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from pydantic import BaseModel

from backend.database.connection import get_db_session
from backend.services.social_media_service import (
    SocialMediaService, SocialAccount, PostRequest, PostResponse, PlatformType
)
from backend.config.logging import get_logger

logger = get_logger(__name__)

router = APIRouter(
    prefix="/api/v1/social",
    tags=["social-media"],
    responses={404: {"description": "Resource not found"}},
)


class AccountRequest(BaseModel):
    """Request model for adding social media account."""
    platform: PlatformType
    account_id: str
    access_token: str
    refresh_token: str = None
    account_name: str = ""


def get_social_service(
    db: AsyncSession = Depends(get_db_session)
) -> SocialMediaService:
    """Dependency injection for SocialMediaService."""
    return SocialMediaService(db)


@router.post("/accounts", status_code=status.HTTP_201_CREATED)
async def add_social_account(
    account_request: AccountRequest,
    social_service: SocialMediaService = Depends(get_social_service),
):
    """
    Add social media account for publishing.
    
    Registers a social media account with the platform for automated
    content publishing and engagement tracking.
    
    Args:
        account_request: Social media account configuration
        social_service: Injected social media service
    
    Returns:
        Success confirmation
    
    Raises:
        400: Invalid account credentials
        500: Account registration failed
    """
    try:
        account = SocialAccount(
            platform=account_request.platform,
            account_id=account_request.account_id,
            access_token=account_request.access_token,
            refresh_token=account_request.refresh_token,
            account_name=account_request.account_name
        )
        
        success = await social_service.add_account(account)
        if not success:
            raise ValueError("Invalid account credentials")
        
        logger.info("Social media account added", 
                   platform=account_request.platform,
                   account_id=account_request.account_id)
        
        return {
            "message": "Social media account added successfully",
            "platform": account_request.platform,
            "account_id": account_request.account_id
        }
        
    except ValueError as e:
        logger.warning("Social account validation error", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error("Social account registration failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to add social media account"
        )


@router.post("/publish", response_model=List[PostResponse])
async def publish_content(
    request: PostRequest,
    social_service: SocialMediaService = Depends(get_social_service),
):
    """
    Publish content to social media platforms.
    
    Publishes generated content to specified social media platforms
    with appropriate formatting and optimization for each platform.
    
    Args:
        request: Publishing request with content and platform details
        social_service: Injected social media service
    
    Returns:
        List[PostResponse]: Results for each platform
    
    Raises:
        400: Invalid publishing request
        404: Content not found
        500: Publishing failed
    """
    try:
        results = await social_service.publish_content(request)
        
        # Log publishing results
        for result in results:
            logger.info("Content published", 
                       content_id=request.content_id,
                       platform=result.platform,
                       status=result.status,
                       post_id=result.post_id)
        
        return results
        
    except ValueError as e:
        logger.warning("Publishing validation error", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error("Content publishing failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Content publishing failed"
        )


@router.post("/schedule", response_model=List[str])
async def schedule_content(
    request: PostRequest,
    social_service: SocialMediaService = Depends(get_social_service),
):
    """
    Schedule content for future publishing.
    
    Schedules content to be published at a specified future time
    across multiple social media platforms.
    
    Args:
        request: Publishing request with schedule time
        social_service: Injected social media service
    
    Returns:
        List[str]: Schedule IDs for tracking
    
    Raises:
        400: Invalid schedule request
        500: Scheduling failed
    """
    try:
        schedule_ids = await social_service.schedule_post(request)
        
        logger.info("Content scheduled", 
                   content_id=request.content_id,
                   schedule_time=request.schedule_time,
                   platforms=request.platforms,
                   schedule_count=len(schedule_ids))
        
        return schedule_ids
        
    except ValueError as e:
        logger.warning("Scheduling validation error", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error("Content scheduling failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Content scheduling failed"
        )


@router.get("/metrics/{platform}/{post_id}")
async def get_engagement_metrics(
    platform: PlatformType,
    post_id: str,
    social_service: SocialMediaService = Depends(get_social_service),
):
    """
    Get engagement metrics for published post.
    
    Retrieves engagement data (likes, comments, shares, views) 
    for a specific post on a social media platform.
    
    Args:
        platform: Social media platform
        post_id: Platform-specific post identifier
        social_service: Injected social media service
    
    Returns:
        Dict with engagement metrics
    
    Raises:
        400: Unsupported platform
        404: Post not found
        500: Metrics retrieval failed
    """
    try:
        metrics = await social_service.get_engagement_metrics(post_id, platform)
        
        logger.info("Engagement metrics retrieved", 
                   platform=platform,
                   post_id=post_id)
        
        return {
            "platform": platform,
            "post_id": post_id,
            "metrics": metrics
        }
        
    except Exception as e:
        logger.error("Metrics retrieval failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve engagement metrics"
        )


@router.get("/platforms")
async def list_supported_platforms():
    """
    List supported social media platforms.
    
    Returns information about all social media platforms
    supported by the integration service.
    
    Returns:
        Dict with platform information
    """
    platforms = {
        "instagram": {
            "name": "Instagram",
            "supported_content": ["image", "video"],
            "max_caption_length": 2200,
            "supports_scheduling": True
        },
        "facebook": {
            "name": "Facebook", 
            "supported_content": ["image", "video", "text"],
            "max_caption_length": 63206,
            "supports_scheduling": True
        },
        "twitter": {
            "name": "Twitter/X",
            "supported_content": ["image", "video", "text"],
            "max_caption_length": 280,
            "supports_scheduling": True
        },
        "tiktok": {
            "name": "TikTok",
            "supported_content": ["video"],
            "max_caption_length": 2200,
            "supports_scheduling": False
        },
        "linkedin": {
            "name": "LinkedIn",
            "supported_content": ["image", "video", "text"],
            "max_caption_length": 3000,
            "supports_scheduling": True
        }
    }
    
    return {
        "supported_platforms": platforms,
        "total_platforms": len(platforms)
    }