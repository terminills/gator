"""
Authentication API Routes

Handles user authentication, registration, and token management.
Implements JWT-based authentication with access and refresh tokens.
"""

from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.ext.asyncio import AsyncSession

from backend.config.logging import get_logger
from backend.database.connection import get_db_session
from backend.exceptions import (
    AuthenticationError,
    InvalidTokenError,
    TokenExpiredError,
)
from backend.models.user import UserResponse
from backend.services.auth_service import (
    AuthService,
    LoginRequest,
    PasswordChangeRequest,
    RefreshTokenRequest,
    RegisterRequest,
    TokenResponse,
)

logger = get_logger(__name__)

router = APIRouter(
    prefix="/api/v1/auth",
    tags=["authentication"],
    responses={
        401: {"description": "Unauthorized"},
        403: {"description": "Forbidden"},
    },
)

# HTTP Bearer token scheme
security = HTTPBearer(auto_error=False)


def get_auth_service(db: AsyncSession = Depends(get_db_session)) -> AuthService:
    """Dependency injection for AuthService."""
    return AuthService(db)


async def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
    auth_service: AuthService = Depends(get_auth_service),
) -> UserResponse:
    """
    Dependency to get the current authenticated user.

    Extracts the JWT token from the Authorization header and validates it.

    Args:
        credentials: HTTP Bearer credentials
        auth_service: Auth service instance

    Returns:
        UserResponse for the authenticated user

    Raises:
        HTTPException: If authentication fails
    """
    if not credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
            headers={"WWW-Authenticate": "Bearer"},
        )

    try:
        user = await auth_service.get_current_user(credentials.credentials)
        return user
    except TokenExpiredError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token has expired",
            headers={"WWW-Authenticate": "Bearer"},
        )
    except InvalidTokenError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=str(e),
            headers={"WWW-Authenticate": "Bearer"},
        )
    except AuthenticationError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=str(e),
            headers={"WWW-Authenticate": "Bearer"},
        )


async def get_current_user_optional(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
    auth_service: AuthService = Depends(get_auth_service),
) -> Optional[UserResponse]:
    """
    Dependency to optionally get the current user.

    Returns None if not authenticated instead of raising an exception.
    Useful for endpoints that work both with and without authentication.

    Args:
        credentials: HTTP Bearer credentials
        auth_service: Auth service instance

    Returns:
        UserResponse if authenticated, None otherwise
    """
    if not credentials:
        return None

    try:
        return await auth_service.get_current_user(credentials.credentials)
    except (TokenExpiredError, InvalidTokenError, AuthenticationError):
        return None


# =============================================================================
# Authentication Endpoints
# =============================================================================


@router.post(
    "/register",
    response_model=TokenResponse,
    status_code=status.HTTP_201_CREATED,
)
async def register(
    register_request: RegisterRequest,
    auth_service: AuthService = Depends(get_auth_service),
):
    """
    Register a new user account.

    Creates a new user with the provided credentials and returns
    authentication tokens for immediate access.

    Args:
        register_request: Registration data (username, email, password)

    Returns:
        TokenResponse with access token, refresh token, and user info
    """
    try:
        result = await auth_service.register(register_request)
        logger.info(f"New user registered: {register_request.username}")
        return result
    except AuthenticationError as e:
        logger.warning(f"Registration failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
    except Exception as e:
        logger.error(f"Registration error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Registration failed due to an internal error",
        )


@router.post("/login", response_model=TokenResponse)
async def login(
    login_request: LoginRequest,
    auth_service: AuthService = Depends(get_auth_service),
):
    """
    Authenticate user and return tokens.

    Validates user credentials and returns access and refresh tokens.
    The access token should be included in subsequent API requests.

    Args:
        login_request: Login credentials (username/email and password)

    Returns:
        TokenResponse with access token, refresh token, and user info
    """
    try:
        result = await auth_service.login(login_request)
        return result
    except AuthenticationError as e:
        logger.warning(f"Login failed for {login_request.username}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=str(e),
            headers={"WWW-Authenticate": "Bearer"},
        )
    except Exception as e:
        logger.error(f"Login error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Login failed due to an internal error",
        )


@router.post("/refresh", response_model=TokenResponse)
async def refresh_tokens(
    refresh_request: RefreshTokenRequest,
    auth_service: AuthService = Depends(get_auth_service),
):
    """
    Refresh authentication tokens.

    Uses a valid refresh token to obtain a new access token and
    optionally a new refresh token.

    Args:
        refresh_request: Refresh token

    Returns:
        TokenResponse with new access token, refresh token, and user info
    """
    try:
        result = await auth_service.refresh_tokens(refresh_request.refresh_token)
        return result
    except TokenExpiredError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Refresh token has expired. Please login again.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    except InvalidTokenError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=str(e),
            headers={"WWW-Authenticate": "Bearer"},
        )
    except AuthenticationError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=str(e),
            headers={"WWW-Authenticate": "Bearer"},
        )
    except Exception as e:
        logger.error(f"Token refresh error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Token refresh failed due to an internal error",
        )


@router.get("/me", response_model=UserResponse)
async def get_me(
    current_user: UserResponse = Depends(get_current_user),
):
    """
    Get current authenticated user's profile.

    Requires a valid access token in the Authorization header.

    Returns:
        UserResponse with current user's profile information
    """
    return current_user


@router.post("/change-password", status_code=status.HTTP_200_OK)
async def change_password(
    password_request: PasswordChangeRequest,
    current_user: UserResponse = Depends(get_current_user),
    auth_service: AuthService = Depends(get_auth_service),
):
    """
    Change the current user's password.

    Requires the current password for verification and sets a new password.

    Args:
        password_request: Current and new password

    Returns:
        Success message
    """
    try:
        await auth_service.change_password(
            str(current_user.id),
            password_request.current_password,
            password_request.new_password,
        )
        return {"message": "Password changed successfully"}
    except AuthenticationError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
    except Exception as e:
        logger.error(f"Password change error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Password change failed due to an internal error",
        )


@router.post("/logout", status_code=status.HTTP_200_OK)
async def logout(
    current_user: UserResponse = Depends(get_current_user),
):
    """
    Logout the current user.

    In a stateless JWT system, the client should discard the tokens.
    This endpoint is provided for API completeness and can be extended
    to implement token blacklisting if needed.

    Returns:
        Success message
    """
    logger.info(f"User logged out: {current_user.username}")
    return {"message": "Logged out successfully"}


@router.get("/verify", status_code=status.HTTP_200_OK)
async def verify_token(
    current_user: UserResponse = Depends(get_current_user),
):
    """
    Verify that the current token is valid.

    Useful for checking authentication status before making requests.

    Returns:
        Token validity status and user info
    """
    return {
        "valid": True,
        "user_id": str(current_user.id),
        "username": current_user.username,
    }
