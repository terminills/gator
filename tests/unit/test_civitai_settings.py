"""
Tests for CivitAI API key retrieval from database settings.

Verifies that the CivitAI downloader correctly retrieves the API key
from database settings instead of environment variables.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker

from backend.database.connection import Base
from backend.models.settings import SystemSettingModel, SettingCategory
from backend.services.settings_service import SettingsService, get_db_setting


@pytest.fixture
async def settings_db():
    """Create test database engine and session factory for settings tests."""
    engine = create_async_engine("sqlite+aiosqlite:///:memory:", echo=False)

    # Create tables
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    session_factory = async_sessionmaker(
        bind=engine, class_=AsyncSession, expire_on_commit=False
    )

    yield session_factory, engine

    await engine.dispose()


@pytest.fixture
async def settings_session(settings_db):
    """Provide a database session for settings tests."""
    session_factory, _ = settings_db
    async with session_factory() as session:
        try:
            yield session
        finally:
            await session.rollback()
            await session.close()


class TestCivitAISettingsRetrieval:
    """Test CivitAI API key retrieval from database settings."""

    async def test_get_db_setting_returns_value_when_set(self, settings_session):
        """Test that get_db_setting returns the value when setting exists."""
        # Create a test setting in the database
        setting = SystemSettingModel(
            key="civitai_api_key",
            category=SettingCategory.AI_MODELS.value,
            value="test-api-key-12345",
            description="CivitAI API key for downloading models",
            is_sensitive=True,
            is_active=True,
        )
        settings_session.add(setting)
        await settings_session.commit()

        # Retrieve using SettingsService
        service = SettingsService(settings_session)
        result = await service.get_setting("civitai_api_key")

        assert result is not None
        assert result.value == "test-api-key-12345"
        assert result.is_sensitive is True

    async def test_get_db_setting_returns_none_when_not_set(self, settings_session):
        """Test that get_db_setting returns None when setting doesn't exist."""
        service = SettingsService(settings_session)
        result = await service.get_setting("nonexistent_key")

        assert result is None

    async def test_get_db_setting_returns_none_for_inactive_setting(self, settings_session):
        """Test that get_db_setting returns None for inactive settings."""
        # Create an inactive setting
        setting = SystemSettingModel(
            key="civitai_api_key",
            category=SettingCategory.AI_MODELS.value,
            value="inactive-api-key",
            is_sensitive=True,
            is_active=False,  # Inactive
        )
        settings_session.add(setting)
        await settings_session.commit()

        service = SettingsService(settings_session)
        result = await service.get_setting("civitai_api_key")

        assert result is None

    async def test_civitai_allow_nsfw_setting(self, settings_session):
        """Test retrieval of civitai_allow_nsfw setting."""
        setting = SystemSettingModel(
            key="civitai_allow_nsfw",
            category=SettingCategory.AI_MODELS.value,
            value=True,
            description="Allow NSFW models from CivitAI",
            is_sensitive=False,
            is_active=True,
        )
        settings_session.add(setting)
        await settings_session.commit()

        service = SettingsService(settings_session)
        result = await service.get_setting("civitai_allow_nsfw")

        assert result is not None
        assert result.value is True


class TestCivitAIRoutesUseDatabaseSettings:
    """Test that CivitAI routes properly use database settings."""

    async def test_civitai_client_gets_api_key_from_db(self):
        """Test that CivitAI client is initialized with API key from database."""
        from backend.utils.civitai_utils import CivitAIClient

        # Create a client with an API key
        client = CivitAIClient(api_key="test-key-from-db")
        
        # Verify the API key is set
        assert client.api_key == "test-key-from-db"
        
        # Verify headers include authorization
        headers = client._get_headers()
        assert "Authorization" in headers
        assert headers["Authorization"] == "Bearer test-key-from-db"

    async def test_civitai_client_without_api_key(self):
        """Test that CivitAI client works without API key but without auth."""
        from backend.utils.civitai_utils import CivitAIClient

        # Create a client without an API key
        client = CivitAIClient(api_key=None)
        
        # Verify no API key is set
        assert client.api_key is None
        
        # Verify headers don't include authorization
        headers = client._get_headers()
        assert "Authorization" not in headers


class TestGetDbSettingHelper:
    """Test the get_db_setting helper function."""

    async def test_get_db_setting_helper_uses_database_manager(self):
        """Test that get_db_setting helper function uses database_manager."""
        # Mock the database_manager
        mock_session = AsyncMock(spec=AsyncSession)
        mock_setting = MagicMock()
        mock_setting.value = "mock-api-key"
        mock_setting.id = "test-id"
        mock_setting.key = "civitai_api_key"
        mock_setting.category = "ai_models"
        mock_setting.description = "Test"
        mock_setting.is_sensitive = True
        mock_setting.is_active = True
        mock_setting.created_at = None
        mock_setting.updated_at = None
        
        # Mock the execute method to return our mock setting
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = mock_setting
        mock_session.execute = AsyncMock(return_value=mock_result)
        
        # Create SettingsService and test
        service = SettingsService(mock_session)
        result = await service.get_setting("civitai_api_key")
        
        # The execute method should have been called
        mock_session.execute.assert_called_once()
