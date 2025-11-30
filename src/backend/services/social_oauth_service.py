"""
Social Media OAuth Service

Handles OAuth2 authentication flows for various social media platforms.
Supports Instagram, Facebook, Twitter, TikTok, and LinkedIn.
"""

import secrets
import uuid
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, Optional
from urllib.parse import urlencode

import httpx
from pydantic import BaseModel
from sqlalchemy import Column, DateTime, String, Text
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import declarative_base

from backend.config.logging import get_logger
from backend.config.settings import get_settings
from backend.database.connection import Base
from backend.services.social_media_service import PlatformType

logger = get_logger(__name__)
settings = get_settings()


class OAuthState(str, Enum):
    """OAuth flow state."""

    PENDING = "pending"
    AUTHORIZED = "authorized"
    FAILED = "failed"
    EXPIRED = "expired"


class OAuthTokenModel(Base):
    """SQLAlchemy model for storing OAuth tokens."""

    __tablename__ = "oauth_tokens"

    id = Column(String(36), primary_key=True)
    user_id = Column(String(36), nullable=False, index=True)
    platform = Column(String(50), nullable=False, index=True)
    access_token = Column(Text, nullable=False)
    refresh_token = Column(Text, nullable=True)
    token_type = Column(String(50), default="Bearer")
    expires_at = Column(DateTime(timezone=True), nullable=True)
    scope = Column(Text, nullable=True)
    account_id = Column(String(255), nullable=True)
    account_name = Column(String(255), nullable=True)
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)
    updated_at = Column(
        DateTime(timezone=True), default=datetime.utcnow, onupdate=datetime.utcnow
    )


class OAuthConfig(BaseModel):
    """OAuth configuration for a platform."""

    client_id: str
    client_secret: str
    redirect_uri: str
    authorize_url: str
    token_url: str
    scopes: list[str]
    extra_params: Dict[str, str] = {}


class OAuthAuthorizationResponse(BaseModel):
    """Response from OAuth authorization URL generation."""

    authorization_url: str
    state: str
    platform: PlatformType


class OAuthTokenResponse(BaseModel):
    """Response containing OAuth tokens."""

    access_token: str
    token_type: str = "Bearer"
    expires_in: Optional[int] = None
    refresh_token: Optional[str] = None
    scope: Optional[str] = None
    account_id: Optional[str] = None
    account_name: Optional[str] = None


class SocialOAuthService:
    """
    Service for handling OAuth2 flows with social media platforms.

    Supports authorization code flow for:
    - Instagram (via Facebook)
    - Facebook
    - Twitter (OAuth 2.0)
    - TikTok
    - LinkedIn

    Note on State Storage:
        OAuth states are stored in-memory with 10-minute expiration for simplicity.
        For production horizontal scaling, states should be stored in Redis using
        the CacheService. The current implementation is suitable for single-instance
        deployments and development environments.
    """

    # In-memory state storage with 10-minute TTL
    # TODO: Migrate to Redis (CacheService) for horizontal scaling
    _oauth_states: Dict[str, Dict[str, Any]] = {}

    def __init__(self, db_session: AsyncSession):
        """
        Initialize OAuth service.

        Args:
            db_session: Database session for token storage
        """
        self.db = db_session
        self.http_client = httpx.AsyncClient(timeout=30.0)

    async def close(self):
        """Clean up resources."""
        await self.http_client.aclose()

    def _get_platform_config(self, platform: PlatformType) -> OAuthConfig:
        """
        Get OAuth configuration for a platform.

        Args:
            platform: Social media platform

        Returns:
            OAuthConfig for the platform

        Raises:
            ValueError: If platform is not supported
        """
        base_url = settings.base_url or "http://localhost:8000"

        configs = {
            PlatformType.INSTAGRAM: OAuthConfig(
                client_id=settings.instagram_client_id or "",
                client_secret=settings.instagram_client_secret or "",
                redirect_uri=f"{base_url}/api/v1/oauth/instagram/callback",
                authorize_url="https://api.instagram.com/oauth/authorize",
                token_url="https://api.instagram.com/oauth/access_token",
                scopes=["user_profile", "user_media"],
            ),
            PlatformType.FACEBOOK: OAuthConfig(
                client_id=settings.facebook_client_id or "",
                client_secret=settings.facebook_client_secret or "",
                redirect_uri=f"{base_url}/api/v1/oauth/facebook/callback",
                authorize_url="https://www.facebook.com/v18.0/dialog/oauth",
                token_url="https://graph.facebook.com/v18.0/oauth/access_token",
                scopes=[
                    "public_profile",
                    "pages_manage_posts",
                    "pages_read_engagement",
                ],
            ),
            PlatformType.TWITTER: OAuthConfig(
                client_id=settings.twitter_client_id or "",
                client_secret=settings.twitter_client_secret or "",
                redirect_uri=f"{base_url}/api/v1/oauth/twitter/callback",
                authorize_url="https://twitter.com/i/oauth2/authorize",
                token_url="https://api.twitter.com/2/oauth2/token",
                scopes=["tweet.read", "tweet.write", "users.read", "offline.access"],
                extra_params={"code_challenge_method": "S256"},
            ),
            PlatformType.TIKTOK: OAuthConfig(
                client_id=settings.tiktok_client_id or "",
                client_secret=settings.tiktok_client_secret or "",
                redirect_uri=f"{base_url}/api/v1/oauth/tiktok/callback",
                authorize_url="https://www.tiktok.com/v2/auth/authorize/",
                token_url="https://open.tiktokapis.com/v2/oauth/token/",
                scopes=["user.info.basic", "video.upload", "video.list"],
            ),
            PlatformType.LINKEDIN: OAuthConfig(
                client_id=settings.linkedin_client_id or "",
                client_secret=settings.linkedin_client_secret or "",
                redirect_uri=f"{base_url}/api/v1/oauth/linkedin/callback",
                authorize_url="https://www.linkedin.com/oauth/v2/authorization",
                token_url="https://www.linkedin.com/oauth/v2/accessToken",
                scopes=["r_liteprofile", "r_emailaddress", "w_member_social"],
            ),
        }

        if platform not in configs:
            raise ValueError(f"OAuth not supported for platform: {platform}")

        return configs[platform]

    def generate_authorization_url(
        self, platform: PlatformType, user_id: str
    ) -> OAuthAuthorizationResponse:
        """
        Generate OAuth authorization URL for a platform.

        Creates a state token and constructs the authorization URL
        that the user should be redirected to.

        Args:
            platform: Social media platform
            user_id: ID of the user initiating OAuth

        Returns:
            OAuthAuthorizationResponse with authorization URL and state

        Raises:
            ValueError: If platform is not supported
        """
        config = self._get_platform_config(platform)

        # Generate secure state token
        state = secrets.token_urlsafe(32)

        # Store state with metadata (TTL: 10 minutes)
        self._oauth_states[state] = {
            "platform": platform,
            "user_id": user_id,
            "created_at": datetime.utcnow(),
            "expires_at": datetime.utcnow() + timedelta(minutes=10),
        }

        # Build authorization URL
        params = {
            "client_id": config.client_id,
            "redirect_uri": config.redirect_uri,
            "response_type": "code",
            "scope": " ".join(config.scopes),
            "state": state,
        }
        params.update(config.extra_params)

        authorization_url = f"{config.authorize_url}?{urlencode(params)}"

        logger.info(
            f"Generated OAuth authorization URL for {platform} user_id={user_id}"
        )

        return OAuthAuthorizationResponse(
            authorization_url=authorization_url,
            state=state,
            platform=platform,
        )

    async def exchange_code_for_token(
        self, platform: PlatformType, code: str, state: str
    ) -> OAuthTokenResponse:
        """
        Exchange authorization code for access token.

        Validates the state, exchanges the code for tokens, and stores
        the tokens in the database.

        Args:
            platform: Social media platform
            code: Authorization code from callback
            state: State token for validation

        Returns:
            OAuthTokenResponse with access token

        Raises:
            ValueError: If state is invalid or expired
            httpx.HTTPError: If token exchange fails
        """
        # Validate state
        state_data = self._oauth_states.get(state)
        if not state_data:
            raise ValueError("Invalid OAuth state")

        if datetime.utcnow() > state_data["expires_at"]:
            del self._oauth_states[state]
            raise ValueError("OAuth state has expired")

        if state_data["platform"] != platform:
            raise ValueError("OAuth state platform mismatch")

        # Clean up used state
        user_id = state_data["user_id"]
        del self._oauth_states[state]

        config = self._get_platform_config(platform)

        # Exchange code for token
        token_data = {
            "client_id": config.client_id,
            "client_secret": config.client_secret,
            "code": code,
            "grant_type": "authorization_code",
            "redirect_uri": config.redirect_uri,
        }

        response = await self.http_client.post(
            config.token_url,
            data=token_data,
            headers={"Accept": "application/json"},
        )

        if response.status_code != 200:
            logger.error(f"Token exchange failed: {response.text}")
            raise ValueError(f"Token exchange failed: {response.text}")

        token_response = response.json()

        # Get account info
        account_info = await self._get_account_info(platform, token_response)

        # Create response
        result = OAuthTokenResponse(
            access_token=token_response.get("access_token"),
            token_type=token_response.get("token_type", "Bearer"),
            expires_in=token_response.get("expires_in"),
            refresh_token=token_response.get("refresh_token"),
            scope=token_response.get("scope"),
            account_id=account_info.get("id"),
            account_name=account_info.get("name"),
        )

        # Store token in database
        await self._store_token(user_id, platform, result)

        logger.info(
            f"OAuth token exchange successful for {platform} user_id={user_id}"
        )

        return result

    async def _get_account_info(
        self, platform: PlatformType, token_data: Dict[str, Any]
    ) -> Dict[str, str]:
        """
        Get account information from the platform.

        Args:
            platform: Social media platform
            token_data: Token response data

        Returns:
            Dict with account id and name
        """
        access_token = token_data.get("access_token", "")
        account_info = {"id": "", "name": ""}

        try:
            if platform == PlatformType.INSTAGRAM:
                response = await self.http_client.get(
                    "https://graph.instagram.com/me",
                    params={"fields": "id,username", "access_token": access_token},
                )
                if response.status_code == 200:
                    data = response.json()
                    account_info = {"id": data.get("id"), "name": data.get("username")}

            elif platform == PlatformType.FACEBOOK:
                response = await self.http_client.get(
                    "https://graph.facebook.com/me",
                    params={"fields": "id,name", "access_token": access_token},
                )
                if response.status_code == 200:
                    data = response.json()
                    account_info = {"id": data.get("id"), "name": data.get("name")}

            elif platform == PlatformType.TWITTER:
                response = await self.http_client.get(
                    "https://api.twitter.com/2/users/me",
                    headers={"Authorization": f"Bearer {access_token}"},
                )
                if response.status_code == 200:
                    data = response.json().get("data", {})
                    account_info = {"id": data.get("id"), "name": data.get("username")}

            elif platform == PlatformType.TIKTOK:
                response = await self.http_client.get(
                    "https://open.tiktokapis.com/v2/user/info/",
                    params={"fields": "open_id,display_name"},
                    headers={"Authorization": f"Bearer {access_token}"},
                )
                if response.status_code == 200:
                    data = response.json().get("data", {}).get("user", {})
                    account_info = {
                        "id": data.get("open_id"),
                        "name": data.get("display_name"),
                    }

            elif platform == PlatformType.LINKEDIN:
                response = await self.http_client.get(
                    "https://api.linkedin.com/v2/me",
                    headers={"Authorization": f"Bearer {access_token}"},
                )
                if response.status_code == 200:
                    data = response.json()
                    account_info = {
                        "id": data.get("id"),
                        "name": f"{data.get('localizedFirstName', '')} {data.get('localizedLastName', '')}".strip(),
                    }

        except Exception as e:
            logger.warning(f"Failed to get account info for {platform}: {e}")

        return account_info

    async def _store_token(
        self, user_id: str, platform: PlatformType, token: OAuthTokenResponse
    ) -> None:
        """
        Store OAuth token in database.

        Args:
            user_id: User ID
            platform: Social media platform
            token: Token response to store
        """
        expires_at = None
        if token.expires_in:
            expires_at = datetime.utcnow() + timedelta(seconds=token.expires_in)

        oauth_token = OAuthTokenModel(
            id=str(uuid.uuid4()),
            user_id=user_id,
            platform=platform.value,
            access_token=token.access_token,
            refresh_token=token.refresh_token,
            token_type=token.token_type,
            expires_at=expires_at,
            scope=token.scope,
            account_id=token.account_id,
            account_name=token.account_name,
        )

        self.db.add(oauth_token)
        await self.db.commit()

    async def refresh_token(
        self, platform: PlatformType, user_id: str
    ) -> Optional[OAuthTokenResponse]:
        """
        Refresh an expired OAuth token.

        Args:
            platform: Social media platform
            user_id: User ID

        Returns:
            New OAuthTokenResponse if successful, None otherwise
        """
        from sqlalchemy import select

        # Get existing token
        stmt = select(OAuthTokenModel).where(
            OAuthTokenModel.user_id == user_id,
            OAuthTokenModel.platform == platform.value,
        )
        result = await self.db.execute(stmt)
        existing_token = result.scalar_one_or_none()

        if not existing_token or not existing_token.refresh_token:
            return None

        config = self._get_platform_config(platform)

        # Request new token
        refresh_data = {
            "client_id": config.client_id,
            "client_secret": config.client_secret,
            "refresh_token": existing_token.refresh_token,
            "grant_type": "refresh_token",
        }

        try:
            response = await self.http_client.post(
                config.token_url,
                data=refresh_data,
                headers={"Accept": "application/json"},
            )

            if response.status_code != 200:
                logger.error(f"Token refresh failed: {response.text}")
                return None

            token_response = response.json()

            # Update stored token
            existing_token.access_token = token_response.get("access_token")
            if token_response.get("refresh_token"):
                existing_token.refresh_token = token_response.get("refresh_token")
            if token_response.get("expires_in"):
                existing_token.expires_at = datetime.utcnow() + timedelta(
                    seconds=token_response.get("expires_in")
                )
            existing_token.updated_at = datetime.utcnow()

            await self.db.commit()

            logger.info(f"Token refreshed for {platform} user_id={user_id}")

            return OAuthTokenResponse(
                access_token=token_response.get("access_token"),
                token_type=token_response.get("token_type", "Bearer"),
                expires_in=token_response.get("expires_in"),
                refresh_token=token_response.get("refresh_token"),
            )

        except Exception as e:
            logger.error(f"Token refresh error: {e}")
            return None

    async def get_user_tokens(self, user_id: str) -> list[Dict[str, Any]]:
        """
        Get all OAuth tokens for a user.

        Args:
            user_id: User ID

        Returns:
            List of token information (without sensitive data)
        """
        from sqlalchemy import select

        stmt = select(OAuthTokenModel).where(OAuthTokenModel.user_id == user_id)
        result = await self.db.execute(stmt)
        tokens = result.scalars().all()

        return [
            {
                "id": token.id,
                "platform": token.platform,
                "account_id": token.account_id,
                "account_name": token.account_name,
                "expires_at": token.expires_at.isoformat() if token.expires_at else None,
                "created_at": token.created_at.isoformat() if token.created_at else None,
                "is_expired": token.expires_at < datetime.utcnow()
                if token.expires_at
                else False,
            }
            for token in tokens
        ]

    async def revoke_token(self, user_id: str, platform: PlatformType) -> bool:
        """
        Revoke OAuth token for a platform.

        Args:
            user_id: User ID
            platform: Social media platform

        Returns:
            True if token was revoked, False otherwise
        """
        from sqlalchemy import delete

        stmt = delete(OAuthTokenModel).where(
            OAuthTokenModel.user_id == user_id,
            OAuthTokenModel.platform == platform.value,
        )
        result = await self.db.execute(stmt)
        await self.db.commit()

        logger.info(f"Token revoked for {platform} user_id={user_id}")

        return result.rowcount > 0

    async def get_valid_access_token(
        self, user_id: str, platform: PlatformType
    ) -> Optional[str]:
        """
        Get a valid access token, refreshing if necessary.

        Args:
            user_id: User ID
            platform: Social media platform

        Returns:
            Valid access token or None
        """
        from sqlalchemy import select

        stmt = select(OAuthTokenModel).where(
            OAuthTokenModel.user_id == user_id,
            OAuthTokenModel.platform == platform.value,
        )
        result = await self.db.execute(stmt)
        token = result.scalar_one_or_none()

        if not token:
            return None

        # Check if token is expired
        if token.expires_at and token.expires_at < datetime.utcnow():
            # Try to refresh
            refreshed = await self.refresh_token(platform, user_id)
            if refreshed:
                return refreshed.access_token
            return None

        return token.access_token
