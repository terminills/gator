"""
Social Media Integration Service

Handles publishing content to various social media platforms,
scheduling posts, and tracking engagement metrics.
"""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from uuid import UUID
from enum import Enum
import json
from dataclasses import dataclass

from sqlalchemy.ext.asyncio import AsyncSession
from pydantic import BaseModel
import httpx

from backend.models.content import ContentModel, ContentResponse
from backend.models.persona import PersonaModel
from backend.config.logging import get_logger

logger = get_logger(__name__)


class PlatformType(str, Enum):
    """Supported social media platforms."""
    INSTAGRAM = "instagram"
    FACEBOOK = "facebook"
    TWITTER = "twitter"
    TIKTOK = "tiktok"
    LINKEDIN = "linkedin"
    CUSTOM = "custom"


class PostStatus(str, Enum):
    """Post status types."""
    DRAFT = "draft"
    SCHEDULED = "scheduled"
    PUBLISHED = "published"
    FAILED = "failed"


@dataclass
class SocialAccount:
    """Social media account configuration."""
    platform: PlatformType
    account_id: str
    access_token: str
    refresh_token: Optional[str] = None
    account_name: str = ""
    is_active: bool = True
    settings: Dict[str, Any] = None


class PostRequest(BaseModel):
    """Request to publish content to social media."""
    content_id: UUID
    platforms: List[PlatformType]
    caption: Optional[str] = None
    hashtags: List[str] = []
    schedule_time: Optional[datetime] = None
    platform_specific: Dict[PlatformType, Dict[str, Any]] = {}


class PostResponse(BaseModel):
    """Response from social media posting."""
    platform: PlatformType
    post_id: Optional[str]
    status: PostStatus
    published_at: Optional[datetime]
    platform_url: Optional[str]
    error_message: Optional[str] = None
    engagement_metrics: Dict[str, int] = {}


class SocialMediaService:
    """
    Service for social media platform integration.
    
    Handles content publishing, scheduling, and engagement tracking
    across multiple social media platforms.
    """
    
    def __init__(self, db_session: AsyncSession):
        """
        Initialize social media service.
        
        Args:
            db_session: Database session for persistence
        """
        self.db = db_session
        self.accounts: Dict[str, SocialAccount] = {}
        self.http_client = httpx.AsyncClient(timeout=30.0)
        
        # Import platform clients
        from backend.services.social_media_clients import create_platform_client
        
        # Platform-specific clients
        self.platform_clients = {
            PlatformType.INSTAGRAM: create_platform_client(PlatformType.INSTAGRAM),
            PlatformType.FACEBOOK: create_platform_client(PlatformType.FACEBOOK),
            PlatformType.TWITTER: create_platform_client(PlatformType.TWITTER),
            PlatformType.TIKTOK: create_platform_client(PlatformType.TIKTOK),
            PlatformType.LINKEDIN: create_platform_client(PlatformType.LINKEDIN),
        }
    
    async def add_account(self, account: SocialAccount) -> bool:
        """
        Add social media account for posting.
        
        Args:
            account: Social media account configuration
            
        Returns:
            bool: True if account was added successfully
        """
        try:
            # Validate account credentials
            client = self.platform_clients.get(account.platform)
            if client and hasattr(client, 'validate_credentials'):
                is_valid = await client.validate_credentials(account)
                if not is_valid:
                    raise ValueError(f"Invalid credentials for {account.platform}")
            
            # Store account configuration
            account_key = f"{account.platform}_{account.account_id}"
            self.accounts[account_key] = account
            
            logger.info(f"Social media account added platform={account.platform} account_id={account.account_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to add social media account: {str(e)"), 
                        platform=account.platform)
            return False
    
    async def publish_content(self, request: PostRequest) -> List[PostResponse]:
        """
        Publish content to specified social media platforms.
        
        Args:
            request: Publishing request with content and platform details
            
        Returns:
            List of PostResponse objects with results for each platform
        """
        responses = []
        
        try:
            # Get content from database
            content = await self._get_content(request.content_id)
            if not content:
                raise ValueError(f"Content not found: {request.content_id}")
            
            # Get associated persona
            persona = await self._get_persona(content.persona_id)
            if not persona:
                raise ValueError(f"Persona not found: {content.persona_id}")
            
            # Publish to each platform
            for platform in request.platforms:
                response = await self._publish_to_platform(
                    platform, content, persona, request
                )
                responses.append(response)
                
                # Log publishing attempt
                logger.info(f"Content published to platform content_id={request.content_id} platform={platform} status={response.status}")
            
            return responses
            
        except Exception as e:
            logger.error(f"Failed to publish content: {str(e)"), 
                        content_id=request.content_id)
            
            # Return error responses for all platforms
            return [
                PostResponse(
                    platform=platform,
                    post_id=None,
                    status=PostStatus.FAILED,
                    published_at=None,
                    platform_url=None,
                    error_message=str(e)
                )
                for platform in request.platforms
            ]
    
    async def schedule_post(self, request: PostRequest) -> List[str]:
        """
        Schedule content for future publishing.
        
        Args:
            request: Publishing request with schedule time
            
        Returns:
            List of schedule IDs for tracking
        """
        try:
            if not request.schedule_time:
                raise ValueError("Schedule time is required")
            
            if request.schedule_time <= datetime.utcnow():
                raise ValueError("Schedule time must be in the future")
            
            # Store scheduled post (in production, use a job queue like Celery)
            schedule_ids = []
            
            for platform in request.platforms:
                schedule_id = f"schedule_{platform}_{datetime.utcnow().timestamp()}"
                
                # TODO: Implement actual scheduling with job queue
                # For now, just log the scheduling request
                logger.info(f"Post scheduled schedule_id={schedule_id} platform={platform} schedule_time={request.schedule_time} content_id={request.content_id}")
                
                schedule_ids.append(schedule_id)
            
            return schedule_ids
            
        except Exception as e:
            logger.error(f"Failed to schedule post: {str(e)"))
            return []
    
    async def get_engagement_metrics(self, post_id: str, platform: PlatformType) -> Dict[str, Any]:
        """
        Get engagement metrics for a published post.
        
        Args:
            post_id: Platform-specific post ID
            platform: Social media platform
            
        Returns:
            Dictionary with engagement metrics
        """
        try:
            client = self.platform_clients.get(platform)
            if not client or not hasattr(client, 'get_metrics'):
                return {"error": "Platform not supported for metrics"}
            
            metrics = await client.get_metrics(post_id)
            return metrics
            
        except Exception as e:
            logger.error(f"Failed to get engagement metrics: {str(e)"), 
                        post_id=post_id, 
                        platform=platform)
            return {"error": str(e)}
    
    async def _publish_to_platform(
        self, 
        platform: PlatformType, 
        content: ContentResponse, 
        persona: PersonaModel,
        request: PostRequest
    ) -> PostResponse:
        """Publish content to a specific platform."""
        try:
            # Get platform client
            client = self.platform_clients.get(platform)
            if not client:
                return PostResponse(
                    platform=platform,
                    post_id=None,
                    status=PostStatus.FAILED,
                    published_at=None,
                    platform_url=None,
                    error_message=f"Platform {platform} not supported"
                )
            
            # Get account for platform
            account = self._get_account_for_platform(platform)
            if not account:
                return PostResponse(
                    platform=platform,
                    post_id=None,
                    status=PostStatus.FAILED,
                    published_at=None,
                    platform_url=None,
                    error_message=f"No account configured for {platform}"
                )
            
            # Prepare post content
            post_data = await self._prepare_post_data(content, persona, request, platform)
            
            # Publish via platform client
            result = await client.publish_post(account, post_data)
            
            return PostResponse(
                platform=platform,
                post_id=result.get("post_id"),
                status=PostStatus.PUBLISHED if result.get("success") else PostStatus.FAILED,
                published_at=datetime.utcnow() if result.get("success") else None,
                platform_url=result.get("url"),
                error_message=result.get("error")
            )
            
        except Exception as e:
            return PostResponse(
                platform=platform,
                post_id=None,
                status=PostStatus.FAILED,
                published_at=None,
                platform_url=None,
                error_message=str(e)
            )
    
    def _get_account_for_platform(self, platform: PlatformType) -> Optional[SocialAccount]:
        """Get the first active account for a platform."""
        for account in self.accounts.values():
            if account.platform == platform and account.is_active:
                return account
        return None
    
    async def _prepare_post_data(
        self, 
        content: ContentResponse, 
        persona: PersonaModel,
        request: PostRequest,
        platform: PlatformType
    ) -> Dict[str, Any]:
        """Prepare post data for specific platform."""
        
        # Base post data
        post_data = {
            "content_type": content.content_type,
            "file_path": content.file_path,
            "caption": request.caption or self._generate_caption(content, persona),
            "hashtags": request.hashtags or self._generate_hashtags(content, persona),
        }
        
        # Add platform-specific customizations
        if platform in request.platform_specific:
            post_data.update(request.platform_specific[platform])
        
        # Platform-specific optimizations
        if platform == PlatformType.INSTAGRAM:
            post_data["caption"] = post_data["caption"][:2200]  # Instagram limit
        elif platform == PlatformType.TWITTER:
            post_data["caption"] = post_data["caption"][:280]   # Twitter limit
        elif platform == PlatformType.LINKEDIN:
            post_data["caption"] = post_data["caption"][:3000]  # LinkedIn limit
        
        return post_data
    
    def _generate_caption(self, content: ContentResponse, persona: PersonaModel) -> str:
        """Generate appropriate caption based on content and persona."""
        
        # Extract personality traits for tone
        personality_traits = persona.personality.split(", ")
        tone = personality_traits[0] if personality_traits else "professional"
        
        if content.content_type == "image":
            caption = f"Sharing some inspiration today! 🌟"
        elif content.content_type == "video":
            caption = f"Excited to share this with you all! 🎬"
        else:
            caption = content.description or "Thought you might find this interesting!"
        
        # Add personality-based ending
        if "creative" in tone.lower():
            caption += " What's inspiring you today?"
        elif "professional" in tone.lower():
            caption += " Thoughts on this?"
        else:
            caption += " Let me know what you think!"
        
        return caption
    
    def _generate_hashtags(self, content: ContentResponse, persona: PersonaModel) -> List[str]:
        """Generate relevant hashtags based on content and persona themes."""
        hashtags = []
        
        # Add hashtags based on persona themes
        for theme in persona.content_themes[:3]:  # Limit to top 3 themes
            hashtags.append(f"#{theme.replace(' ', '').lower()}")
        
        # Add content type specific hashtags
        if content.content_type == "image":
            hashtags.extend(["#photography", "#visual", "#inspiration"])
        elif content.content_type == "video":
            hashtags.extend(["#video", "#content", "#storytelling"])
        else:
            hashtags.extend(["#content", "#thoughts", "#sharing"])
        
        # Add personality-based hashtags
        if "tech" in persona.personality.lower():
            hashtags.append("#technology")
        if "creative" in persona.personality.lower():
            hashtags.append("#creativity")
        
        return hashtags[:10]  # Most platforms have hashtag limits
    
    async def _get_content(self, content_id: UUID) -> Optional[ContentResponse]:
        """Get content from database."""
        try:
            from sqlalchemy import select
            stmt = select(ContentModel).where(ContentModel.id == content_id)
            result = await self.db.execute(stmt)
            content = result.scalar_one_or_none()
            
            if content:
                return ContentResponse.model_validate(content)
            return None
            
        except Exception as e:
            logger.error(f"Error retrieving content: {str(e)"))
            return None
    
    async def _get_persona(self, persona_id: UUID) -> Optional[PersonaModel]:
        """Get persona from database."""
        try:
            from sqlalchemy import select
            stmt = select(PersonaModel).where(PersonaModel.id == persona_id)
            result = await self.db.execute(stmt)
            return result.scalar_one_or_none()
            
        except Exception as e:
            logger.error(f"Error retrieving persona: {str(e)"))
            return None

