"""
Authentication Service

Handles user authentication, JWT token generation, and verification.
Implements secure authentication flow with access and refresh tokens.
"""

from datetime import datetime, timedelta, timezone
from typing import Optional, Tuple
from uuid import UUID

from jose import JWTError, jwt
from passlib.context import CryptContext
from pydantic import BaseModel, Field
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from backend.config.logging import get_logger
from backend.config.settings import get_settings
from backend.exceptions import (
    AuthenticationError,
    InvalidTokenError,
    TokenExpiredError,
)
from backend.models.user import UserModel, UserResponse

logger = get_logger(__name__)
settings = get_settings()


# Password hashing configuration
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


# =============================================================================
# Token Models
# =============================================================================


class TokenPayload(BaseModel):
    """JWT token payload."""

    sub: str  # User ID
    exp: datetime  # Expiration time
    type: str = "access"  # Token type: access or refresh
    iat: Optional[datetime] = None  # Issued at


class TokenPair(BaseModel):
    """Access and refresh token pair."""

    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int = Field(description="Access token expiration in seconds")


class TokenResponse(BaseModel):
    """Token response for API."""

    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int
    user: UserResponse


class LoginRequest(BaseModel):
    """Login request model."""

    username: str = Field(min_length=3, max_length=50)
    password: str = Field(min_length=8, max_length=128)


class RegisterRequest(BaseModel):
    """Registration request model."""

    username: str = Field(
        min_length=3,
        max_length=50,
        pattern=r"^[a-zA-Z0-9_]+$",
        description="Unique username (alphanumeric and underscores only)",
    )
    email: str = Field(description="User email address")
    password: str = Field(
        min_length=8,
        max_length=128,
        description="Password (minimum 8 characters)",
    )
    display_name: Optional[str] = Field(None, max_length=100)


class RefreshTokenRequest(BaseModel):
    """Refresh token request model."""

    refresh_token: str


class PasswordChangeRequest(BaseModel):
    """Password change request model."""

    current_password: str
    new_password: str = Field(min_length=8, max_length=128)


# =============================================================================
# Authentication Service
# =============================================================================


class AuthService:
    """
    Authentication service for user login, registration, and token management.

    Implements JWT-based authentication with:
    - Access tokens (short-lived, for API access)
    - Refresh tokens (long-lived, for obtaining new access tokens)
    - Secure password hashing with bcrypt
    """

    # Token expiration settings
    ACCESS_TOKEN_EXPIRE_MINUTES = settings.jwt_expire_minutes
    REFRESH_TOKEN_EXPIRE_DAYS = 7

    def __init__(self, db_session: AsyncSession):
        """Initialize the service with a database session."""
        self.db = db_session
        self.secret_key = settings.secret_key
        self.algorithm = settings.jwt_algorithm

    # =========================================================================
    # Password Handling
    # =========================================================================

    @staticmethod
    def hash_password(password: str) -> str:
        """Hash a password using bcrypt."""
        return pwd_context.hash(password)

    @staticmethod
    def verify_password(plain_password: str, hashed_password: str) -> bool:
        """Verify a password against a hash."""
        return pwd_context.verify(plain_password, hashed_password)

    # =========================================================================
    # Token Generation
    # =========================================================================

    def create_access_token(
        self,
        user_id: str,
        expires_delta: Optional[timedelta] = None,
    ) -> str:
        """
        Create a JWT access token.

        Args:
            user_id: User ID to encode in the token
            expires_delta: Optional custom expiration time

        Returns:
            Encoded JWT access token
        """
        if expires_delta:
            expire = datetime.now(timezone.utc) + expires_delta
        else:
            expire = datetime.now(timezone.utc) + timedelta(
                minutes=self.ACCESS_TOKEN_EXPIRE_MINUTES
            )

        payload = {
            "sub": str(user_id),
            "exp": expire,
            "iat": datetime.now(timezone.utc),
            "type": "access",
        }

        return jwt.encode(payload, self.secret_key, algorithm=self.algorithm)

    def create_refresh_token(
        self,
        user_id: str,
        expires_delta: Optional[timedelta] = None,
    ) -> str:
        """
        Create a JWT refresh token.

        Args:
            user_id: User ID to encode in the token
            expires_delta: Optional custom expiration time

        Returns:
            Encoded JWT refresh token
        """
        if expires_delta:
            expire = datetime.now(timezone.utc) + expires_delta
        else:
            expire = datetime.now(timezone.utc) + timedelta(
                days=self.REFRESH_TOKEN_EXPIRE_DAYS
            )

        payload = {
            "sub": str(user_id),
            "exp": expire,
            "iat": datetime.now(timezone.utc),
            "type": "refresh",
        }

        return jwt.encode(payload, self.secret_key, algorithm=self.algorithm)

    def create_token_pair(self, user_id: str) -> TokenPair:
        """
        Create both access and refresh tokens.

        Args:
            user_id: User ID to encode in the tokens

        Returns:
            TokenPair with both tokens
        """
        access_token = self.create_access_token(user_id)
        refresh_token = self.create_refresh_token(user_id)

        return TokenPair(
            access_token=access_token,
            refresh_token=refresh_token,
            expires_in=self.ACCESS_TOKEN_EXPIRE_MINUTES * 60,
        )

    # =========================================================================
    # Token Verification
    # =========================================================================

    def verify_token(
        self,
        token: str,
        expected_type: str = "access",
    ) -> TokenPayload:
        """
        Verify and decode a JWT token.

        Args:
            token: JWT token to verify
            expected_type: Expected token type (access or refresh)

        Returns:
            Decoded token payload

        Raises:
            InvalidTokenError: If token is invalid
            TokenExpiredError: If token has expired
        """
        try:
            payload = jwt.decode(
                token,
                self.secret_key,
                algorithms=[self.algorithm],
            )

            # Verify token type
            token_type = payload.get("type", "access")
            if token_type != expected_type:
                raise InvalidTokenError(
                    f"Invalid token type. Expected {expected_type}, got {token_type}"
                )

            return TokenPayload(
                sub=payload["sub"],
                exp=datetime.fromtimestamp(payload["exp"], tz=timezone.utc),
                type=token_type,
                iat=datetime.fromtimestamp(payload.get("iat", 0), tz=timezone.utc)
                if payload.get("iat")
                else None,
            )

        except jwt.ExpiredSignatureError:
            raise TokenExpiredError("Token has expired")
        except JWTError as e:
            raise InvalidTokenError(f"Invalid token: {str(e)}")

    # =========================================================================
    # User Authentication
    # =========================================================================

    async def authenticate_user(
        self,
        username: str,
        password: str,
    ) -> Optional[UserModel]:
        """
        Authenticate a user by username/email and password.

        Args:
            username: Username or email
            password: Plain text password

        Returns:
            UserModel if authentication succeeds, None otherwise
        """
        try:
            # Try to find user by username or email
            stmt = select(UserModel).where(
                (UserModel.username == username.lower())
                | (UserModel.email == username.lower())
            )
            result = await self.db.execute(stmt)
            user = result.scalar_one_or_none()

            if not user:
                logger.warning(f"Authentication failed: User not found: {username}")
                return None

            # Check if user has a password hash
            if not hasattr(user, "password_hash") or not user.password_hash:
                logger.warning(
                    f"Authentication failed: No password set for user: {username}"
                )
                return None

            # Verify password
            if not self.verify_password(password, user.password_hash):
                logger.warning(
                    f"Authentication failed: Invalid password for user: {username}"
                )
                return None

            # Check if user is active
            if not user.is_active:
                logger.warning(
                    f"Authentication failed: User account is deactivated: {username}"
                )
                return None

            return user

        except Exception as e:
            logger.error(f"Authentication error: {str(e)}")
            return None

    async def login(self, login_request: LoginRequest) -> TokenResponse:
        """
        Authenticate user and return tokens.

        Args:
            login_request: Login credentials

        Returns:
            TokenResponse with tokens and user info

        Raises:
            AuthenticationError: If authentication fails
        """
        user = await self.authenticate_user(
            login_request.username,
            login_request.password,
        )

        if not user:
            raise AuthenticationError("Invalid username or password")

        # Update last active timestamp
        await self._update_last_active(user.id)

        # Create tokens
        token_pair = self.create_token_pair(str(user.id))

        logger.info(f"User logged in successfully: {user.username}")

        return TokenResponse(
            access_token=token_pair.access_token,
            refresh_token=token_pair.refresh_token,
            token_type=token_pair.token_type,
            expires_in=token_pair.expires_in,
            user=UserResponse.model_validate(user),
        )

    async def register(self, register_request: RegisterRequest) -> TokenResponse:
        """
        Register a new user and return tokens.

        Args:
            register_request: Registration data

        Returns:
            TokenResponse with tokens and user info

        Raises:
            AuthenticationError: If registration fails
        """
        try:
            # Check if username already exists
            stmt = select(UserModel).where(
                UserModel.username == register_request.username.lower()
            )
            result = await self.db.execute(stmt)
            if result.scalar_one_or_none():
                raise AuthenticationError("Username already exists")

            # Check if email already exists
            stmt = select(UserModel).where(
                UserModel.email == register_request.email.lower()
            )
            result = await self.db.execute(stmt)
            if result.scalar_one_or_none():
                raise AuthenticationError("Email already exists")

            # Create new user
            password_hash = self.hash_password(register_request.password)

            new_user = UserModel(
                username=register_request.username.lower(),
                email=register_request.email.lower(),
                password_hash=password_hash,
                display_name=register_request.display_name,
                is_active=True,
                receive_dm_notifications=True,
                allow_ppv_offers=True,
                last_active_at=datetime.now(timezone.utc),
            )

            self.db.add(new_user)
            await self.db.commit()
            await self.db.refresh(new_user)

            # Create tokens
            token_pair = self.create_token_pair(str(new_user.id))

            logger.info(f"New user registered: {new_user.username}")

            return TokenResponse(
                access_token=token_pair.access_token,
                refresh_token=token_pair.refresh_token,
                token_type=token_pair.token_type,
                expires_in=token_pair.expires_in,
                user=UserResponse.model_validate(new_user),
            )

        except AuthenticationError:
            raise
        except Exception as e:
            await self.db.rollback()
            logger.error(f"Registration failed: {str(e)}")
            raise AuthenticationError(f"Registration failed: {str(e)}")

    async def refresh_tokens(self, refresh_token: str) -> TokenResponse:
        """
        Refresh access token using a refresh token.

        Args:
            refresh_token: Valid refresh token

        Returns:
            TokenResponse with new tokens

        Raises:
            InvalidTokenError: If refresh token is invalid
            TokenExpiredError: If refresh token has expired
            AuthenticationError: If user not found or inactive
        """
        # Verify refresh token
        payload = self.verify_token(refresh_token, expected_type="refresh")

        # Get user
        user = await self.get_user_by_id(payload.sub)
        if not user:
            raise AuthenticationError("User not found")

        if not user.is_active:
            raise AuthenticationError("User account is deactivated")

        # Update last active timestamp
        await self._update_last_active(user.id)

        # Create new tokens
        token_pair = self.create_token_pair(str(user.id))

        logger.info(f"Tokens refreshed for user: {user.username}")

        return TokenResponse(
            access_token=token_pair.access_token,
            refresh_token=token_pair.refresh_token,
            token_type=token_pair.token_type,
            expires_in=token_pair.expires_in,
            user=UserResponse.model_validate(user),
        )

    async def change_password(
        self,
        user_id: str,
        current_password: str,
        new_password: str,
    ) -> bool:
        """
        Change user's password.

        Args:
            user_id: User ID
            current_password: Current password for verification
            new_password: New password to set

        Returns:
            True if password changed successfully

        Raises:
            AuthenticationError: If current password is invalid
        """
        user = await self._get_user_model(user_id)
        if not user:
            raise AuthenticationError("User not found")

        # Verify current password
        if not self.verify_password(current_password, user.password_hash):
            raise AuthenticationError("Current password is incorrect")

        # Update password
        new_hash = self.hash_password(new_password)
        stmt = (
            update(UserModel)
            .where(UserModel.id == user_id)
            .values(password_hash=new_hash, updated_at=datetime.now(timezone.utc))
        )
        await self.db.execute(stmt)
        await self.db.commit()

        logger.info(f"Password changed for user: {user.username}")
        return True

    # =========================================================================
    # Helper Methods
    # =========================================================================

    async def get_user_by_id(self, user_id: str) -> Optional[UserResponse]:
        """Get user by ID and return response model."""
        user = await self._get_user_model(user_id)
        if user:
            return UserResponse.model_validate(user)
        return None

    async def _get_user_model(self, user_id: str) -> Optional[UserModel]:
        """Get user model by ID."""
        try:
            stmt = select(UserModel).where(UserModel.id == user_id)
            result = await self.db.execute(stmt)
            return result.scalar_one_or_none()
        except Exception as e:
            logger.error(f"Failed to get user {user_id}: {str(e)}")
            return None

    async def _update_last_active(self, user_id: UUID) -> None:
        """Update user's last active timestamp."""
        try:
            stmt = (
                update(UserModel)
                .where(UserModel.id == user_id)
                .values(last_active_at=datetime.now(timezone.utc))
            )
            await self.db.execute(stmt)
            await self.db.commit()
        except Exception as e:
            logger.error(f"Failed to update last active for {user_id}: {str(e)}")

    async def get_current_user(self, token: str) -> UserResponse:
        """
        Get current user from access token.

        Args:
            token: Access token

        Returns:
            UserResponse for the current user

        Raises:
            InvalidTokenError: If token is invalid
            AuthenticationError: If user not found
        """
        payload = self.verify_token(token, expected_type="access")

        user = await self.get_user_by_id(payload.sub)
        if not user:
            raise AuthenticationError("User not found")

        return user
