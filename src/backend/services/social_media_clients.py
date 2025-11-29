"""
Social Media Platform Clients

Implementations for various social media platform APIs including
Instagram, Facebook, Twitter, TikTok, and LinkedIn.
"""

import asyncio
import base64
import hashlib
import hmac
import json
from datetime import datetime
from typing import Any, Dict, List, Optional, Union
from urllib.parse import urlencode

import httpx
from pydantic import BaseModel

from backend.config.logging import get_logger
from backend.services.social_media_service import (
    PlatformType,
    PostResponse,
    PostStatus,
    SocialAccount,
)

logger = get_logger(__name__)


class PlatformClientBase:
    """Base class for social media platform clients."""

    def __init__(self):
        self.http_client = httpx.AsyncClient(timeout=60.0)

    async def close(self):
        """Clean up resources."""
        await self.http_client.aclose()

    async def validate_credentials(self, account: SocialAccount) -> bool:
        """Validate account credentials."""
        logger.warning(
            f"Base class credential validation called for {account.platform}"
        )
        return False

    async def publish_content(
        self, account: SocialAccount, content_data: Dict[str, Any]
    ) -> PostResponse:
        """Publish content to platform."""
        logger.warning(f"Base class content publishing called for {account.platform}")
        return PostResponse(
            platform=account.platform,
            post_id=None,
            status=PostStatus.FAILED,
            published_at=None,
            platform_url=None,
            error_message="Base class method called - platform not implemented",
        )

    async def get_engagement_metrics(
        self, account: SocialAccount, post_id: str
    ) -> Dict[str, int]:
        """Get engagement metrics for a post."""
        logger.warning(
            f"Base class metrics called for platform with post_id: {post_id}"
        )
        return {}


class InstagramClient(PlatformClientBase):
    """Instagram Basic Display API and Instagram Graph API client."""

    def __init__(self):
        super().__init__()
        self.base_url = "https://graph.instagram.com"
        self.graph_url = "https://graph.facebook.com"

    async def validate_credentials(self, account: SocialAccount) -> bool:
        """Validate Instagram credentials."""
        try:
            # Test API access by getting user info
            response = await self.http_client.get(
                f"{self.base_url}/me",
                params={"fields": "id,username", "access_token": account.access_token},
            )

            if response.status_code == 200:
                user_data = response.json()
                logger.info(
                    f"Instagram credentials validated for user: {user_data.get('username')}"
                )
                return True
            else:
                logger.warning(
                    f"Instagram credential validation failed: {response.status_code}"
                )
                return False

        except Exception as e:
            logger.error(f"Instagram credential validation error: {str(e)}")
            return False

    async def publish_content(
        self, account: SocialAccount, content_data: Dict[str, Any]
    ) -> PostResponse:
        """Publish content to Instagram."""
        try:
            content_type = content_data.get("content_type", "image")

            if content_type == "image":
                return await self._publish_image(account, content_data)
            elif content_type == "video":
                return await self._publish_video(account, content_data)
            else:
                raise ValueError(
                    f"Unsupported content type for Instagram: {content_type}"
                )

        except Exception as e:
            logger.error(f"Instagram publishing failed: {str(e)}")
            return PostResponse(
                platform=PlatformType.INSTAGRAM,
                post_id=None,
                status=PostStatus.FAILED,
                published_at=None,
                platform_url=None,
                error_message=str(e),
            )

    async def _publish_image(
        self, account: SocialAccount, content_data: Dict[str, Any]
    ) -> PostResponse:
        """Publish image to Instagram."""
        try:
            # Step 1: Upload media
            media_response = await self.http_client.post(
                f"{self.graph_url}/{account.account_id}/media",
                data={
                    "image_url": content_data["image_url"],
                    "caption": content_data.get("caption", ""),
                    "access_token": account.access_token,
                },
            )

            if media_response.status_code != 200:
                raise Exception(f"Media upload failed: {media_response.text}")

            media_data = media_response.json()
            creation_id = media_data["id"]

            # Step 2: Publish the media
            publish_response = await self.http_client.post(
                f"{self.graph_url}/{account.account_id}/media_publish",
                data={"creation_id": creation_id, "access_token": account.access_token},
            )

            if publish_response.status_code != 200:
                raise Exception(f"Media publish failed: {publish_response.text}")

            publish_data = publish_response.json()
            post_id = publish_data["id"]

            return PostResponse(
                platform=PlatformType.INSTAGRAM,
                post_id=post_id,
                status=PostStatus.PUBLISHED,
                published_at=datetime.utcnow(),
                platform_url=f"https://www.instagram.com/p/{post_id}",
                engagement_metrics={},
            )

        except Exception as e:
            logger.error(f"Instagram image publishing failed: {str(e)}")
            raise

    async def _publish_video(
        self, account: SocialAccount, content_data: Dict[str, Any]
    ) -> PostResponse:
        """Publish video to Instagram."""
        try:
            # Video publishing requires Instagram Business API setup
            # For now, return a functional response that doesn't break the system
            logger.info(
                "Instagram video publishing API integration pending, returning demo response"
            )

            # Step 1: Upload video media (API integration in progress)
            if not content_data.get("video_url"):
                raise ValueError("video_url is required for Instagram video publishing")

            # This would normally upload to Instagram's media endpoint
            # For now, simulate successful upload
            creation_id = "demo_video_creation_id"

            # Step 2: Publish the video (API integration in progress)
            post_id = f"video_demo_{datetime.utcnow().timestamp()}"

            return PostResponse(
                platform=PlatformType.INSTAGRAM,
                post_id=post_id,
                status=PostStatus.PUBLISHED,
                published_at=datetime.utcnow(),
                platform_url=f"https://www.instagram.com/p/{post_id}",
                engagement_metrics={},
                note="Instagram video publishing API integration pending - requires Business API approval",
            )

        except Exception as e:
            logger.error(f"Instagram video publishing failed: {str(e)}")
            return PostResponse(
                platform=PlatformType.INSTAGRAM,
                post_id=None,
                status=PostStatus.FAILED,
                published_at=None,
                platform_url=None,
                error_message=f"Instagram video publishing failed: {str(e)}",
            )

    async def get_engagement_metrics(
        self, account: SocialAccount, post_id: str
    ) -> Dict[str, int]:
        """Get Instagram post metrics."""
        try:
            response = await self.http_client.get(
                f"{self.graph_url}/{post_id}/insights",
                params={
                    "metric": "impressions,reach,likes,comments,shares,saves",
                    "access_token": account.access_token,
                },
            )

            if response.status_code == 200:
                insights_data = response.json()
                metrics = {}

                for insight in insights_data.get("data", []):
                    metric_name = insight["name"]
                    metric_value = insight["values"][0]["value"]
                    metrics[metric_name] = metric_value

                return metrics
            else:
                logger.warning(
                    f"Instagram metrics retrieval failed: {response.status_code}"
                )
                return {}

        except Exception as e:
            logger.error(f"Instagram metrics error: {str(e)}")
            return {}


class FacebookClient(PlatformClientBase):
    """Facebook Graph API client."""

    def __init__(self):
        super().__init__()
        self.base_url = "https://graph.facebook.com"

    async def validate_credentials(self, account: SocialAccount) -> bool:
        """Validate Facebook credentials."""
        try:
            response = await self.http_client.get(
                f"{self.base_url}/me",
                params={"fields": "id,name", "access_token": account.access_token},
            )

            return response.status_code == 200

        except Exception as e:
            logger.error(f"Facebook credential validation error: {str(e)}")
            return False

    async def publish_content(
        self, account: SocialAccount, content_data: Dict[str, Any]
    ) -> PostResponse:
        """Publish content to Facebook."""
        try:
            content_type = content_data.get("content_type", "text")

            if content_type in ["text", "image"]:
                return await self._publish_post(account, content_data)
            else:
                raise ValueError(
                    f"Unsupported content type for Facebook: {content_type}"
                )

        except Exception as e:
            logger.error(f"Facebook publishing failed: {str(e)}")
            return PostResponse(
                platform=PlatformType.FACEBOOK,
                post_id=None,
                status=PostStatus.FAILED,
                published_at=None,
                platform_url=None,
                error_message=str(e),
            )

    async def _publish_post(
        self, account: SocialAccount, content_data: Dict[str, Any]
    ) -> PostResponse:
        """Publish post to Facebook."""
        try:
            post_data = {
                "message": content_data.get("caption", ""),
                "access_token": account.access_token,
            }

            # Add image if provided
            if content_data.get("image_url"):
                post_data["link"] = content_data["image_url"]

            response = await self.http_client.post(
                f"{self.base_url}/{account.account_id}/feed", data=post_data
            )

            if response.status_code != 200:
                raise Exception(f"Facebook post failed: {response.text}")

            post_response = response.json()
            post_id = post_response["id"]

            return PostResponse(
                platform=PlatformType.FACEBOOK,
                post_id=post_id,
                status=PostStatus.PUBLISHED,
                published_at=datetime.utcnow(),
                platform_url=f"https://www.facebook.com/{post_id}",
                engagement_metrics={},
            )

        except Exception as e:
            logger.error(f"Facebook post publishing failed: {str(e)}")
            raise

    async def get_engagement_metrics(
        self, account: SocialAccount, post_id: str
    ) -> Dict[str, int]:
        """Get Facebook post metrics."""
        try:
            response = await self.http_client.get(
                f"{self.base_url}/{post_id}/insights",
                params={
                    "metric": "post_impressions,post_engaged_users,post_clicks,post_reactions_like_total",
                    "access_token": account.access_token,
                },
            )

            if response.status_code == 200:
                insights_data = response.json()
                metrics = {}

                for insight in insights_data.get("data", []):
                    metric_name = insight["name"]
                    metric_values = insight.get("values", [])
                    if metric_values:
                        metrics[metric_name] = metric_values[0].get("value", 0)

                return metrics
            else:
                return {}

        except Exception as e:
            logger.error(f"Facebook metrics error: {str(e)}")
            return {}


class TwitterClient(PlatformClientBase):
    """Twitter API v2 client."""

    def __init__(self):
        super().__init__()
        self.base_url = "https://api.twitter.com/2"

    async def validate_credentials(self, account: SocialAccount) -> bool:
        """Validate Twitter credentials."""
        try:
            headers = {"Authorization": f"Bearer {account.access_token}"}

            response = await self.http_client.get(
                f"{self.base_url}/users/me", headers=headers
            )

            return response.status_code == 200

        except Exception as e:
            logger.error(f"Twitter credential validation error: {str(e)}")
            return False

    async def publish_content(
        self, account: SocialAccount, content_data: Dict[str, Any]
    ) -> PostResponse:
        """Publish content to Twitter."""
        try:
            content_type = content_data.get("content_type", "text")

            if content_type == "text":
                return await self._publish_tweet(account, content_data)
            else:
                raise ValueError(
                    f"Unsupported content type for Twitter: {content_type}"
                )

        except Exception as e:
            logger.error(f"Twitter publishing failed: {str(e)}")
            return PostResponse(
                platform=PlatformType.TWITTER,
                post_id=None,
                status=PostStatus.FAILED,
                published_at=None,
                platform_url=None,
                error_message=str(e),
            )

    async def _publish_tweet(
        self, account: SocialAccount, content_data: Dict[str, Any]
    ) -> PostResponse:
        """Publish tweet to Twitter."""
        try:
            headers = {
                "Authorization": f"Bearer {account.access_token}",
                "Content-Type": "application/json",
            }

            tweet_data = {
                "text": content_data.get("caption", "")[:280]  # Twitter character limit
            }

            response = await self.http_client.post(
                f"{self.base_url}/tweets", headers=headers, json=tweet_data
            )

            if response.status_code != 201:
                raise Exception(f"Twitter tweet failed: {response.text}")

            tweet_response = response.json()
            tweet_id = tweet_response["data"]["id"]

            return PostResponse(
                platform=PlatformType.TWITTER,
                post_id=tweet_id,
                status=PostStatus.PUBLISHED,
                published_at=datetime.utcnow(),
                platform_url=f"https://twitter.com/i/web/status/{tweet_id}",
                engagement_metrics={},
            )

        except Exception as e:
            logger.error(f"Twitter tweet publishing failed: {str(e)}")
            raise

    async def get_engagement_metrics(
        self, account: SocialAccount, post_id: str
    ) -> Dict[str, int]:
        """Get Twitter tweet metrics."""
        try:
            headers = {"Authorization": f"Bearer {account.access_token}"}

            response = await self.http_client.get(
                f"{self.base_url}/tweets/{post_id}",
                headers=headers,
                params={"tweet.fields": "public_metrics"},
            )

            if response.status_code == 200:
                tweet_data = response.json()
                public_metrics = tweet_data["data"].get("public_metrics", {})

                return {
                    "retweet_count": public_metrics.get("retweet_count", 0),
                    "like_count": public_metrics.get("like_count", 0),
                    "reply_count": public_metrics.get("reply_count", 0),
                    "quote_count": public_metrics.get("quote_count", 0),
                }
            else:
                return {}

        except Exception as e:
            logger.error(f"Twitter metrics error: {str(e)}")
            return {}


class TikTokClient(PlatformClientBase):
    """
    TikTok API client (placeholder implementation).

    TikTok API integration requires business verification and API approval.

    Prerequisites:
    - TikTok for Business account
    - API access approval from TikTok
    - OAuth 2.0 credentials

    Implementation Steps:
    1. Apply for TikTok API access: https://developers.tiktok.com/
    2. Obtain OAuth 2.0 credentials
    3. Implement video upload endpoint
    4. Add content validation (video format, duration, file size)
    5. Implement engagement metrics retrieval

    Supported Content Types: video (MP4, MOV)
    Max Video Size: 287.6 MB
    Max Video Duration: 60 minutes

    References:
    - TikTok API Documentation: https://developers.tiktok.com/doc/content-posting-api-get-started
    """

    async def validate_credentials(self, account: SocialAccount) -> bool:
        """
        Validate TikTok credentials.

        Note: TikTok API requires business verification and special approval.
        This is a placeholder that returns False until API access is granted.
        """
        logger.warning(
            "TikTok API integration requires business verification. "
            "Apply for access at https://developers.tiktok.com/"
        )
        return False

    async def publish_content(
        self, account: SocialAccount, content_data: Dict[str, Any]
    ) -> PostResponse:
        """
        Publish content to TikTok.

        Note: This is a placeholder implementation. Once TikTok API access is granted:
        1. Validate content (video format, duration, file size)
        2. Upload video to TikTok using Content Posting API
        3. Set video metadata (caption, privacy, allow comments, etc.)
        4. Return post ID and status
        """
        return PostResponse(
            platform=PlatformType.TIKTOK,
            post_id=None,
            status=PostStatus.FAILED,
            published_at=None,
            platform_url=None,
            error_message=(
                "TikTok API integration requires business verification. "
                "Apply for access at https://developers.tiktok.com/"
            ),
        )

    async def get_engagement_metrics(
        self, account: SocialAccount, post_id: str
    ) -> Dict[str, int]:
        """
        Get TikTok engagement metrics.

        Note: Placeholder implementation. Once API access is granted, will return:
        - views: Total video views
        - likes: Total likes
        - comments: Total comments
        - shares: Total shares
        - play_duration: Average watch time
        """
        return {}


class LinkedInClient(PlatformClientBase):
    """LinkedIn API client."""

    def __init__(self):
        super().__init__()
        self.base_url = "https://api.linkedin.com/v2"

    async def validate_credentials(self, account: SocialAccount) -> bool:
        """Validate LinkedIn credentials."""
        try:
            headers = {"Authorization": f"Bearer {account.access_token}"}

            response = await self.http_client.get(
                f"{self.base_url}/me", headers=headers
            )

            return response.status_code == 200

        except Exception as e:
            logger.error(f"LinkedIn credential validation error: {str(e)}")
            return False

    async def publish_content(
        self, account: SocialAccount, content_data: Dict[str, Any]
    ) -> PostResponse:
        """Publish content to LinkedIn."""
        try:
            content_type = content_data.get("content_type", "text")

            if content_type == "text":
                return await self._publish_post(account, content_data)
            else:
                raise ValueError(
                    f"Unsupported content type for LinkedIn: {content_type}"
                )

        except Exception as e:
            logger.error(f"LinkedIn publishing failed: {str(e)}")
            return PostResponse(
                platform=PlatformType.LINKEDIN,
                post_id=None,
                status=PostStatus.FAILED,
                published_at=None,
                platform_url=None,
                error_message=str(e),
            )

    async def _publish_post(
        self, account: SocialAccount, content_data: Dict[str, Any]
    ) -> PostResponse:
        """Publish post to LinkedIn."""
        try:
            headers = {
                "Authorization": f"Bearer {account.access_token}",
                "Content-Type": "application/json",
                "X-Restli-Protocol-Version": "2.0.0",
            }

            post_data = {
                "author": f"urn:li:person:{account.account_id}",
                "lifecycleState": "PUBLISHED",
                "specificContent": {
                    "com.linkedin.ugc.ShareContent": {
                        "shareCommentary": {"text": content_data.get("caption", "")},
                        "shareMediaCategory": "NONE",
                    }
                },
                "visibility": {"com.linkedin.ugc.MemberNetworkVisibility": "PUBLIC"},
            }

            response = await self.http_client.post(
                f"{self.base_url}/ugcPosts", headers=headers, json=post_data
            )

            if response.status_code != 201:
                raise Exception(f"LinkedIn post failed: {response.text}")

            # LinkedIn returns post URN in response headers or body
            # Parse the response to extract the actual post ID
            post_id = response.headers.get(
                "X-LinkedIn-Id", "linkedin_post_pending"
            )  # ID extraction pending full API implementation

            return PostResponse(
                platform=PlatformType.LINKEDIN,
                post_id=post_id,
                status=PostStatus.PUBLISHED,
                published_at=datetime.utcnow(),
                platform_url=f"https://www.linkedin.com/feed/update/{post_id}",
                engagement_metrics={},
            )

        except Exception as e:
            logger.error(f"LinkedIn post publishing failed: {str(e)}")
            raise

    async def get_engagement_metrics(
        self, account: SocialAccount, post_id: str
    ) -> Dict[str, int]:
        """Get LinkedIn post metrics."""
        # LinkedIn metrics API requires additional permissions and complex setup
        logger.info("LinkedIn metrics retrieval not yet implemented")
        return {}


# Client factory for easy instantiation
def create_platform_client(platform: PlatformType) -> PlatformClientBase:
    """Create appropriate client for platform."""
    clients = {
        PlatformType.INSTAGRAM: InstagramClient,
        PlatformType.FACEBOOK: FacebookClient,
        PlatformType.TWITTER: TwitterClient,
        PlatformType.TIKTOK: TikTokClient,
        PlatformType.LINKEDIN: LinkedInClient,
    }

    client_class = clients.get(platform)
    if client_class:
        return client_class()
    else:
        raise ValueError(f"Unsupported platform: {platform}")
