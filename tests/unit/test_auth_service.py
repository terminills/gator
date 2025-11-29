"""
Tests for Authentication Service and Routes

Tests for JWT-based authentication with access and refresh tokens.
"""

import pytest
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch

from backend.exceptions import (
    AuthenticationError,
    InvalidTokenError,
    TokenExpiredError,
)
from backend.services.auth_service import (
    AuthService,
    LoginRequest,
    RegisterRequest,
    TokenPayload,
)


class TestPasswordHashing:
    """Test password hashing utilities."""

    def test_hash_password(self):
        """Test password is properly hashed."""
        password = "test_password_123"
        hashed = AuthService.hash_password(password)

        # Hash should be different from original
        assert hashed != password

        # Hash should be bcrypt format
        assert hashed.startswith("$2")

    def test_verify_password_correct(self):
        """Test correct password verification."""
        password = "secure_password_456"
        hashed = AuthService.hash_password(password)

        assert AuthService.verify_password(password, hashed) is True

    def test_verify_password_incorrect(self):
        """Test incorrect password fails verification."""
        password = "correct_password"
        wrong_password = "wrong_password"
        hashed = AuthService.hash_password(password)

        assert AuthService.verify_password(wrong_password, hashed) is False


class TestTokenGeneration:
    """Test JWT token generation."""

    @pytest.fixture
    def auth_service(self):
        """Create auth service with mock DB."""
        mock_db = AsyncMock()
        return AuthService(mock_db)

    def test_create_access_token(self, auth_service):
        """Test access token creation."""
        user_id = "test-user-id-123"
        token = auth_service.create_access_token(user_id)

        # Token should be a non-empty string
        assert isinstance(token, str)
        assert len(token) > 0

        # Should be able to verify it
        payload = auth_service.verify_token(token, expected_type="access")
        assert payload.sub == user_id
        assert payload.type == "access"

    def test_create_refresh_token(self, auth_service):
        """Test refresh token creation."""
        user_id = "test-user-id-456"
        token = auth_service.create_refresh_token(user_id)

        # Token should be a non-empty string
        assert isinstance(token, str)
        assert len(token) > 0

        # Should be able to verify it
        payload = auth_service.verify_token(token, expected_type="refresh")
        assert payload.sub == user_id
        assert payload.type == "refresh"

    def test_create_token_pair(self, auth_service):
        """Test creating both access and refresh tokens."""
        user_id = "test-user-id-789"
        token_pair = auth_service.create_token_pair(user_id)

        assert token_pair.access_token
        assert token_pair.refresh_token
        assert token_pair.token_type == "bearer"
        assert token_pair.expires_in > 0

        # Access token should verify as access type
        access_payload = auth_service.verify_token(
            token_pair.access_token, expected_type="access"
        )
        assert access_payload.sub == user_id

        # Refresh token should verify as refresh type
        refresh_payload = auth_service.verify_token(
            token_pair.refresh_token, expected_type="refresh"
        )
        assert refresh_payload.sub == user_id

    def test_access_token_custom_expiry(self, auth_service):
        """Test access token with custom expiry."""
        user_id = "test-user"
        custom_expiry = timedelta(hours=2)
        token = auth_service.create_access_token(user_id, expires_delta=custom_expiry)

        payload = auth_service.verify_token(token)

        # Expiry should be approximately 2 hours from now
        expected_exp = datetime.now(timezone.utc) + custom_expiry
        assert abs((payload.exp - expected_exp).total_seconds()) < 5


class TestTokenVerification:
    """Test JWT token verification."""

    @pytest.fixture
    def auth_service(self):
        """Create auth service with mock DB."""
        mock_db = AsyncMock()
        return AuthService(mock_db)

    def test_verify_valid_access_token(self, auth_service):
        """Test verifying a valid access token."""
        user_id = "valid-user"
        token = auth_service.create_access_token(user_id)

        payload = auth_service.verify_token(token, expected_type="access")

        assert payload.sub == user_id
        assert payload.type == "access"

    def test_verify_valid_refresh_token(self, auth_service):
        """Test verifying a valid refresh token."""
        user_id = "valid-user"
        token = auth_service.create_refresh_token(user_id)

        payload = auth_service.verify_token(token, expected_type="refresh")

        assert payload.sub == user_id
        assert payload.type == "refresh"

    def test_verify_wrong_token_type(self, auth_service):
        """Test verifying token with wrong type fails."""
        user_id = "user"
        access_token = auth_service.create_access_token(user_id)

        # Try to verify access token as refresh token
        with pytest.raises(InvalidTokenError) as exc_info:
            auth_service.verify_token(access_token, expected_type="refresh")

        assert "Invalid token type" in str(exc_info.value)

    def test_verify_invalid_token(self, auth_service):
        """Test verifying invalid token fails."""
        with pytest.raises(InvalidTokenError):
            auth_service.verify_token("invalid.token.here")

    def test_verify_expired_token(self, auth_service):
        """Test verifying expired token fails."""
        user_id = "user"
        # Create token that expires immediately
        token = auth_service.create_access_token(
            user_id, expires_delta=timedelta(seconds=-1)
        )

        with pytest.raises(TokenExpiredError):
            auth_service.verify_token(token)


class TestLoginRequest:
    """Test login request validation."""

    def test_valid_login_request(self):
        """Test valid login request."""
        request = LoginRequest(username="testuser", password="password123")
        assert request.username == "testuser"
        assert request.password == "password123"

    def test_short_username_fails(self):
        """Test username too short fails validation."""
        with pytest.raises(ValueError):
            LoginRequest(username="ab", password="password123")

    def test_short_password_fails(self):
        """Test password too short fails validation."""
        with pytest.raises(ValueError):
            LoginRequest(username="testuser", password="short")


class TestRegisterRequest:
    """Test register request validation."""

    def test_valid_register_request(self):
        """Test valid register request."""
        request = RegisterRequest(
            username="newuser",
            email="user@example.com",
            password="securepass123",
            display_name="New User",
        )
        assert request.username == "newuser"
        assert request.email == "user@example.com"
        assert request.password == "securepass123"
        assert request.display_name == "New User"

    def test_invalid_username_characters(self):
        """Test username with invalid characters fails."""
        with pytest.raises(ValueError):
            RegisterRequest(
                username="user@name",
                email="user@example.com",
                password="password123",
            )

    def test_valid_username_with_underscore(self):
        """Test username with underscore is valid."""
        request = RegisterRequest(
            username="user_name",
            email="user@example.com",
            password="password123",
        )
        assert request.username == "user_name"


class TestTokenPayload:
    """Test token payload model."""

    def test_token_payload_creation(self):
        """Test creating token payload."""
        now = datetime.now(timezone.utc)
        payload = TokenPayload(
            sub="user-123",
            exp=now + timedelta(hours=1),
            type="access",
            iat=now,
        )

        assert payload.sub == "user-123"
        assert payload.type == "access"
        assert payload.iat == now
