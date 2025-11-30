"""
Unit tests for Social OAuth Service.

Tests the OAuth2 authentication flow for social media platforms.
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

from backend.services.social_oauth_service import (
    SocialOAuthService,
    OAuthState,
    OAuthConfig,
    OAuthAuthorizationResponse,
    OAuthTokenResponse,
)
from backend.services.social_media_service import PlatformType


class TestOAuthConfig:
    """Tests for OAuthConfig model."""

    def test_oauth_config_creation(self):
        """Test creating OAuth configuration."""
        config = OAuthConfig(
            client_id="test_client_id",
            client_secret="test_secret",
            redirect_uri="http://localhost:8000/callback",
            authorize_url="https://example.com/oauth/authorize",
            token_url="https://example.com/oauth/token",
            scopes=["read", "write"],
        )

        assert config.client_id == "test_client_id"
        assert config.client_secret == "test_secret"
        assert config.redirect_uri == "http://localhost:8000/callback"
        assert config.scopes == ["read", "write"]
        assert config.extra_params == {}

    def test_oauth_config_with_extra_params(self):
        """Test OAuth config with extra parameters."""
        config = OAuthConfig(
            client_id="test",
            client_secret="secret",
            redirect_uri="http://localhost/callback",
            authorize_url="https://example.com/auth",
            token_url="https://example.com/token",
            scopes=["read"],
            extra_params={"code_challenge_method": "S256"},
        )

        assert config.extra_params == {"code_challenge_method": "S256"}


class TestOAuthAuthorizationResponse:
    """Tests for OAuth authorization response."""

    def test_authorization_response_creation(self):
        """Test creating authorization response."""
        response = OAuthAuthorizationResponse(
            authorization_url="https://example.com/auth?state=xyz",
            state="xyz123",
            platform=PlatformType.INSTAGRAM,
        )

        assert response.authorization_url == "https://example.com/auth?state=xyz"
        assert response.state == "xyz123"
        assert response.platform == PlatformType.INSTAGRAM


class TestOAuthTokenResponse:
    """Tests for OAuth token response."""

    def test_token_response_minimal(self):
        """Test minimal token response."""
        response = OAuthTokenResponse(
            access_token="access_token_123",
        )

        assert response.access_token == "access_token_123"
        assert response.token_type == "Bearer"
        assert response.expires_in is None
        assert response.refresh_token is None

    def test_token_response_full(self):
        """Test full token response."""
        response = OAuthTokenResponse(
            access_token="access_token_123",
            token_type="Bearer",
            expires_in=3600,
            refresh_token="refresh_token_456",
            scope="read write",
            account_id="user123",
            account_name="testuser",
        )

        assert response.access_token == "access_token_123"
        assert response.refresh_token == "refresh_token_456"
        assert response.expires_in == 3600
        assert response.account_name == "testuser"


class TestOAuthState:
    """Tests for OAuth state enum."""

    def test_oauth_states(self):
        """Test OAuth state values."""
        assert OAuthState.PENDING == "pending"
        assert OAuthState.AUTHORIZED == "authorized"
        assert OAuthState.FAILED == "failed"
        assert OAuthState.EXPIRED == "expired"


class TestSocialOAuthService:
    """Tests for SocialOAuthService."""

    @pytest.fixture
    def mock_db_session(self):
        """Create mock database session."""
        session = AsyncMock()
        session.add = MagicMock()
        session.commit = AsyncMock()
        session.execute = AsyncMock()
        return session

    @pytest.fixture
    def oauth_service(self, mock_db_session):
        """Create OAuth service instance."""
        return SocialOAuthService(mock_db_session)

    def test_generate_authorization_url_instagram(self, oauth_service):
        """Test generating Instagram authorization URL."""
        result = oauth_service.generate_authorization_url(
            PlatformType.INSTAGRAM, "user123"
        )

        assert isinstance(result, OAuthAuthorizationResponse)
        assert result.platform == PlatformType.INSTAGRAM
        assert "api.instagram.com/oauth/authorize" in result.authorization_url
        assert "state=" in result.authorization_url
        assert len(result.state) > 0

    def test_generate_authorization_url_facebook(self, oauth_service):
        """Test generating Facebook authorization URL."""
        result = oauth_service.generate_authorization_url(
            PlatformType.FACEBOOK, "user123"
        )

        assert result.platform == PlatformType.FACEBOOK
        assert "facebook.com" in result.authorization_url

    def test_generate_authorization_url_twitter(self, oauth_service):
        """Test generating Twitter authorization URL."""
        result = oauth_service.generate_authorization_url(
            PlatformType.TWITTER, "user123"
        )

        assert result.platform == PlatformType.TWITTER
        assert "twitter.com" in result.authorization_url

    def test_generate_authorization_url_tiktok(self, oauth_service):
        """Test generating TikTok authorization URL."""
        result = oauth_service.generate_authorization_url(
            PlatformType.TIKTOK, "user123"
        )

        assert result.platform == PlatformType.TIKTOK
        assert "tiktok.com" in result.authorization_url

    def test_generate_authorization_url_linkedin(self, oauth_service):
        """Test generating LinkedIn authorization URL."""
        result = oauth_service.generate_authorization_url(
            PlatformType.LINKEDIN, "user123"
        )

        assert result.platform == PlatformType.LINKEDIN
        assert "linkedin.com" in result.authorization_url

    def test_generate_authorization_stores_state(self, oauth_service):
        """Test that state is stored for validation."""
        result = oauth_service.generate_authorization_url(
            PlatformType.INSTAGRAM, "user123"
        )

        # State should be stored
        assert result.state in oauth_service._oauth_states
        state_data = oauth_service._oauth_states[result.state]
        assert state_data["platform"] == PlatformType.INSTAGRAM
        assert state_data["user_id"] == "user123"

    def test_unsupported_platform_raises_error(self, oauth_service):
        """Test that unsupported platform raises ValueError."""
        with pytest.raises(ValueError, match="OAuth not supported"):
            oauth_service._get_platform_config(PlatformType.CUSTOM)

    @pytest.mark.asyncio
    async def test_exchange_code_invalid_state(self, oauth_service):
        """Test code exchange with invalid state."""
        with pytest.raises(ValueError, match="Invalid OAuth state"):
            await oauth_service.exchange_code_for_token(
                PlatformType.INSTAGRAM, "code123", "invalid_state"
            )

    @pytest.mark.asyncio
    async def test_exchange_code_expired_state(self, oauth_service):
        """Test code exchange with expired state."""
        # Generate state then manually expire it
        result = oauth_service.generate_authorization_url(
            PlatformType.INSTAGRAM, "user123"
        )
        oauth_service._oauth_states[result.state]["expires_at"] = (
            datetime.utcnow() - timedelta(minutes=1)
        )

        with pytest.raises(ValueError, match="expired"):
            await oauth_service.exchange_code_for_token(
                PlatformType.INSTAGRAM, "code123", result.state
            )

    @pytest.mark.asyncio
    async def test_exchange_code_platform_mismatch(self, oauth_service):
        """Test code exchange with platform mismatch."""
        result = oauth_service.generate_authorization_url(
            PlatformType.INSTAGRAM, "user123"
        )

        with pytest.raises(ValueError, match="mismatch"):
            await oauth_service.exchange_code_for_token(
                PlatformType.FACEBOOK, "code123", result.state
            )

    @pytest.mark.asyncio
    async def test_get_user_tokens_empty(self, oauth_service, mock_db_session):
        """Test getting tokens when user has none."""
        mock_scalars = MagicMock()
        mock_scalars.all.return_value = []
        mock_result = MagicMock()
        mock_result.scalars.return_value = mock_scalars
        mock_db_session.execute = AsyncMock(return_value=mock_result)

        tokens = await oauth_service.get_user_tokens("user123")

        assert tokens == []

    @pytest.mark.asyncio
    async def test_revoke_token_success(self, oauth_service, mock_db_session):
        """Test revoking token successfully."""
        mock_result = MagicMock()
        mock_result.rowcount = 1
        mock_db_session.execute = AsyncMock(return_value=mock_result)

        result = await oauth_service.revoke_token("user123", PlatformType.INSTAGRAM)

        assert result is True
        mock_db_session.commit.assert_called_once()

    @pytest.mark.asyncio
    async def test_revoke_token_not_found(self, oauth_service, mock_db_session):
        """Test revoking non-existent token."""
        mock_result = MagicMock()
        mock_result.rowcount = 0
        mock_db_session.execute = AsyncMock(return_value=mock_result)

        result = await oauth_service.revoke_token("user123", PlatformType.INSTAGRAM)

        assert result is False

    @pytest.mark.asyncio
    async def test_get_valid_access_token_not_found(
        self, oauth_service, mock_db_session
    ):
        """Test getting token when none exists."""
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        mock_db_session.execute = AsyncMock(return_value=mock_result)

        token = await oauth_service.get_valid_access_token(
            "user123", PlatformType.INSTAGRAM
        )

        assert token is None

    @pytest.mark.asyncio
    async def test_close_cleans_up_http_client(self, oauth_service):
        """Test that close() cleans up HTTP client."""
        oauth_service.http_client = AsyncMock()

        await oauth_service.close()

        oauth_service.http_client.aclose.assert_called_once()


class TestOAuthTokenModel:
    """Tests for OAuth token database model."""

    def test_oauth_token_model_import(self):
        """Test importing OAuth token model."""
        from backend.services.social_oauth_service import OAuthTokenModel

        assert OAuthTokenModel.__tablename__ == "oauth_tokens"

    def test_oauth_token_model_columns(self):
        """Test OAuth token model has required columns."""
        from backend.services.social_oauth_service import OAuthTokenModel

        # Check table has expected columns
        columns = [c.name for c in OAuthTokenModel.__table__.columns]
        assert "id" in columns
        assert "user_id" in columns
        assert "platform" in columns
        assert "access_token" in columns
        assert "refresh_token" in columns
        assert "expires_at" in columns
        assert "account_id" in columns
        assert "account_name" in columns
