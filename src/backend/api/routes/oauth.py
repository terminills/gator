"""
Social Media OAuth API Routes

Handles OAuth2 authentication flows for social media platforms.
Supports Instagram, Facebook, Twitter, TikTok, and LinkedIn.
"""

from typing import List

from fastapi import APIRouter, Depends, HTTPException, Query, status
from fastapi.responses import RedirectResponse
from sqlalchemy.ext.asyncio import AsyncSession

from backend.api.routes.auth import get_current_user
from backend.config.logging import get_logger
from backend.database.connection import get_db_session
from backend.models.user import UserResponse
from backend.services.social_media_service import PlatformType
from backend.services.social_oauth_service import (
    OAuthAuthorizationResponse,
    OAuthTokenResponse,
    SocialOAuthService,
)

logger = get_logger(__name__)

router = APIRouter(
    prefix="/api/v1/oauth",
    tags=["oauth"],
    responses={
        401: {"description": "Unauthorized"},
        404: {"description": "Resource not found"},
    },
)


def get_oauth_service(
    db: AsyncSession = Depends(get_db_session),
) -> SocialOAuthService:
    """Dependency injection for SocialOAuthService."""
    return SocialOAuthService(db)


# =============================================================================
# OAuth Flow Endpoints
# =============================================================================


@router.get("/authorize/{platform}", response_model=OAuthAuthorizationResponse)
async def initiate_oauth(
    platform: PlatformType,
    current_user: UserResponse = Depends(get_current_user),
    oauth_service: SocialOAuthService = Depends(get_oauth_service),
):
    """
    Initiate OAuth authorization flow for a social media platform.

    Returns the authorization URL that the user should be redirected to.
    After authorization, the platform will redirect back to our callback URL.

    Args:
        platform: Social media platform to connect

    Returns:
        OAuthAuthorizationResponse with authorization URL and state
    """
    try:
        result = oauth_service.generate_authorization_url(
            platform, str(current_user.id)
        )
        logger.info(
            f"OAuth flow initiated for {platform} by user {current_user.username}"
        )
        return result

    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
    except Exception as e:
        logger.error(f"OAuth initiation error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to initiate OAuth flow",
        )


@router.get("/{platform}/callback")
async def oauth_callback(
    platform: PlatformType,
    code: str = Query(..., description="Authorization code from platform"),
    state: str = Query(..., description="State token for validation"),
    error: str = Query(None, description="Error from platform"),
    error_description: str = Query(None, description="Error description"),
    oauth_service: SocialOAuthService = Depends(get_oauth_service),
):
    """
    Handle OAuth callback from social media platform.

    This endpoint receives the authorization code from the platform
    after the user grants permission. It exchanges the code for tokens.

    Args:
        platform: Social media platform
        code: Authorization code
        state: State token for CSRF validation
        error: Error code if authorization failed
        error_description: Error description

    Returns:
        Redirect to success/error page or token response
    """
    # Handle OAuth errors from platform
    if error:
        logger.warning(f"OAuth error from {platform}: {error} - {error_description}")
        return RedirectResponse(
            url=f"/oauth/error?platform={platform.value}&error={error}&description={error_description or 'Unknown error'}"
        )

    try:
        token_response = await oauth_service.exchange_code_for_token(
            platform, code, state
        )

        logger.info(f"OAuth callback successful for {platform}")

        # Redirect to success page with account info
        return RedirectResponse(
            url=f"/oauth/success?platform={platform.value}&account={token_response.account_name or 'Connected'}"
        )

    except ValueError as e:
        logger.warning(f"OAuth callback validation error: {e}")
        return RedirectResponse(url=f"/oauth/error?platform={platform.value}&error={e}")

    except Exception as e:
        logger.error(f"OAuth callback error: {e}")
        return RedirectResponse(
            url=f"/oauth/error?platform={platform.value}&error=Token exchange failed"
        )


@router.post("/{platform}/token", response_model=OAuthTokenResponse)
async def exchange_token(
    platform: PlatformType,
    code: str = Query(..., description="Authorization code"),
    state: str = Query(..., description="State token"),
    oauth_service: SocialOAuthService = Depends(get_oauth_service),
):
    """
    Exchange authorization code for access token (API endpoint).

    Alternative to the callback redirect - returns token directly.
    Useful for mobile apps and SPAs.

    Args:
        platform: Social media platform
        code: Authorization code
        state: State token

    Returns:
        OAuthTokenResponse with access token
    """
    try:
        return await oauth_service.exchange_code_for_token(platform, code, state)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
    except Exception as e:
        logger.error(f"Token exchange error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Token exchange failed",
        )


# =============================================================================
# Token Management Endpoints
# =============================================================================


@router.get("/tokens", response_model=List[dict])
async def list_connected_accounts(
    current_user: UserResponse = Depends(get_current_user),
    oauth_service: SocialOAuthService = Depends(get_oauth_service),
):
    """
    List all connected social media accounts for the current user.

    Returns:
        List of connected accounts with platform and status information
    """
    tokens = await oauth_service.get_user_tokens(str(current_user.id))
    return tokens


@router.post("/{platform}/refresh", response_model=OAuthTokenResponse)
async def refresh_platform_token(
    platform: PlatformType,
    current_user: UserResponse = Depends(get_current_user),
    oauth_service: SocialOAuthService = Depends(get_oauth_service),
):
    """
    Refresh OAuth token for a connected platform.

    Attempts to refresh the access token using the stored refresh token.

    Args:
        platform: Social media platform

    Returns:
        OAuthTokenResponse with new access token
    """
    result = await oauth_service.refresh_token(platform, str(current_user.id))

    if not result:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unable to refresh token for {platform}. Please reconnect.",
        )

    return result


@router.delete("/{platform}")
async def disconnect_platform(
    platform: PlatformType,
    current_user: UserResponse = Depends(get_current_user),
    oauth_service: SocialOAuthService = Depends(get_oauth_service),
):
    """
    Disconnect a social media account.

    Revokes and removes the stored OAuth token for the platform.

    Args:
        platform: Social media platform to disconnect

    Returns:
        Success message
    """
    success = await oauth_service.revoke_token(str(current_user.id), platform)

    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No connected account found for {platform}",
        )

    logger.info(
        f"User {current_user.username} disconnected from {platform}"
    )

    return {
        "message": f"Successfully disconnected from {platform}",
        "platform": platform,
    }


# =============================================================================
# Utility Endpoints
# =============================================================================


@router.get("/platforms")
async def list_supported_oauth_platforms():
    """
    List social media platforms that support OAuth integration.

    Returns:
        List of platforms with OAuth support status
    """
    return {
        "platforms": [
            {
                "platform": PlatformType.INSTAGRAM,
                "name": "Instagram",
                "oauth_support": True,
                "scopes": ["user_profile", "user_media"],
            },
            {
                "platform": PlatformType.FACEBOOK,
                "name": "Facebook",
                "oauth_support": True,
                "scopes": [
                    "public_profile",
                    "pages_manage_posts",
                    "pages_read_engagement",
                ],
            },
            {
                "platform": PlatformType.TWITTER,
                "name": "Twitter/X",
                "oauth_support": True,
                "scopes": ["tweet.read", "tweet.write", "users.read", "offline.access"],
            },
            {
                "platform": PlatformType.TIKTOK,
                "name": "TikTok",
                "oauth_support": True,
                "scopes": ["user.info.basic", "video.upload", "video.list"],
            },
            {
                "platform": PlatformType.LINKEDIN,
                "name": "LinkedIn",
                "oauth_support": True,
                "scopes": ["r_liteprofile", "r_emailaddress", "w_member_social"],
            },
        ]
    }


@router.get("/status/{platform}")
async def get_connection_status(
    platform: PlatformType,
    current_user: UserResponse = Depends(get_current_user),
    oauth_service: SocialOAuthService = Depends(get_oauth_service),
):
    """
    Get OAuth connection status for a specific platform.

    Args:
        platform: Social media platform

    Returns:
        Connection status with account info
    """
    access_token = await oauth_service.get_valid_access_token(
        str(current_user.id), platform
    )

    if access_token:
        tokens = await oauth_service.get_user_tokens(str(current_user.id))
        platform_token = next(
            (t for t in tokens if t["platform"] == platform.value), None
        )

        return {
            "platform": platform,
            "connected": True,
            "account_name": platform_token.get("account_name") if platform_token else None,
            "expires_at": platform_token.get("expires_at") if platform_token else None,
        }

    return {
        "platform": platform,
        "connected": False,
        "account_name": None,
        "expires_at": None,
    }
