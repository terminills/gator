"""
Tests for TikTok Client implementation.

Tests the TikTok Content Posting API client including:
- Credential validation
- Video content validation
- Content publishing workflow
- Engagement metrics retrieval
"""

import pytest
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

from backend.services.social_media_clients import (
    TikTokClient,
    PlatformType,
    PostStatus,
    SocialAccount,
)


class TestTikTokClient:
    """Test suite for TikTok client."""

    @pytest.fixture
    def tiktok_client(self):
        """Create TikTok client instance."""
        return TikTokClient()

    @pytest.fixture
    def mock_account(self):
        """Create mock social account with TikTok credentials."""
        account = MagicMock(spec=SocialAccount)
        account.access_token = "test_access_token_123"
        account.account_id = "test_account_id"
        account.platform = PlatformType.TIKTOK
        return account

    def test_initialization(self, tiktok_client):
        """Test TikTok client initialization."""
        assert tiktok_client.base_url == "https://open.tiktokapis.com/v2"
        assert tiktok_client.MAX_VIDEO_SIZE_BYTES == 287_600 * 1024
        assert tiktok_client.MAX_VIDEO_DURATION_SECONDS == 3600
        assert "mp4" in tiktok_client.SUPPORTED_FORMATS
        assert "mov" in tiktok_client.SUPPORTED_FORMATS

    def test_video_content_validation_missing_source(self, tiktok_client):
        """Test video validation fails when no source is provided."""
        content_data = {"caption": "Test video"}
        is_valid, error_msg = tiktok_client._validate_video_content(content_data)
        
        assert not is_valid
        assert "video_url or video_path is required" in error_msg

    def test_video_content_validation_unsupported_format(self, tiktok_client):
        """Test video validation fails for unsupported formats."""
        content_data = {"video_url": "https://example.com/video.avi"}
        is_valid, error_msg = tiktok_client._validate_video_content(content_data)
        
        assert not is_valid
        assert "Unsupported video format" in error_msg

    def test_video_content_validation_valid_mp4(self, tiktok_client):
        """Test video validation passes for valid MP4."""
        content_data = {"video_url": "https://example.com/video.mp4"}
        is_valid, error_msg = tiktok_client._validate_video_content(content_data)
        
        assert is_valid
        assert error_msg == ""

    def test_video_content_validation_file_too_large(self, tiktok_client):
        """Test video validation fails when file is too large."""
        content_data = {
            "video_url": "https://example.com/video.mp4",
            "file_size": 300 * 1024 * 1024,  # 300 MB - exceeds limit
        }
        is_valid, error_msg = tiktok_client._validate_video_content(content_data)
        
        assert not is_valid
        assert "exceeds maximum size" in error_msg

    def test_video_content_validation_duration_too_long(self, tiktok_client):
        """Test video validation fails when duration exceeds limit."""
        content_data = {
            "video_url": "https://example.com/video.mp4",
            "duration": 4000,  # 66+ minutes - exceeds 60 min limit
        }
        is_valid, error_msg = tiktok_client._validate_video_content(content_data)
        
        assert not is_valid
        assert "exceeds maximum duration" in error_msg

    @pytest.mark.asyncio
    async def test_validate_credentials_missing_token(self, tiktok_client):
        """Test credential validation fails when token is missing."""
        account = MagicMock(spec=SocialAccount)
        account.access_token = None
        
        result = await tiktok_client.validate_credentials(account)
        
        assert result is False

    @pytest.mark.asyncio
    async def test_validate_credentials_success(self, tiktok_client, mock_account):
        """Test credential validation succeeds with valid response."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "error": {"code": "ok"},
            "data": {"user": {"display_name": "TestUser", "open_id": "123"}},
        }

        with patch.object(
            tiktok_client.http_client, "get", return_value=mock_response
        ) as mock_get:
            result = await tiktok_client.validate_credentials(mock_account)

        assert result is True
        mock_get.assert_called_once()

    @pytest.mark.asyncio
    async def test_validate_credentials_api_error(self, tiktok_client, mock_account):
        """Test credential validation handles API errors."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "error": {"code": "access_token_invalid", "message": "Invalid token"},
        }

        with patch.object(
            tiktok_client.http_client, "get", return_value=mock_response
        ):
            result = await tiktok_client.validate_credentials(mock_account)

        assert result is False

    @pytest.mark.asyncio
    async def test_publish_content_invalid_video(self, tiktok_client, mock_account):
        """Test publishing fails with invalid video content."""
        content_data = {"caption": "No video source"}
        
        response = await tiktok_client.publish_content(mock_account, content_data)
        
        assert response.status == PostStatus.FAILED
        assert response.platform == PlatformType.TIKTOK
        assert "video_url or video_path is required" in response.error_message

    @pytest.mark.asyncio
    async def test_publish_content_success_flow(self, tiktok_client, mock_account):
        """Test successful content publishing workflow."""
        content_data = {
            "video_url": "https://example.com/video.mp4",
            "caption": "Test video caption",
            "privacy_level": "PUBLIC_TO_EVERYONE",
        }

        # Mock init response
        init_response = MagicMock()
        init_response.status_code = 200
        init_response.json.return_value = {
            "error": {"code": "ok"},
            "data": {
                "publish_id": "test_publish_123",
                "upload_url": "https://upload.tiktok.com/test",
            },
        }

        # Mock status response
        status_response = MagicMock()
        status_response.status_code = 200
        status_response.json.return_value = {
            "error": {"code": "ok"},
            "data": {
                "status": "PUBLISH_COMPLETE",
                "share_url": "https://tiktok.com/@user/video/123",
            },
        }

        with patch.object(
            tiktok_client.http_client, "post", side_effect=[init_response, status_response]
        ):
            response = await tiktok_client.publish_content(mock_account, content_data)

        assert response.post_id == "test_publish_123"
        assert response.status == PostStatus.PUBLISHED
        assert response.platform == PlatformType.TIKTOK

    @pytest.mark.asyncio
    async def test_get_engagement_metrics_success(self, tiktok_client, mock_account):
        """Test successful engagement metrics retrieval."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "error": {"code": "ok"},
            "data": {
                "videos": [
                    {
                        "id": "test_video_123",
                        "view_count": 1000,
                        "like_count": 100,
                        "comment_count": 50,
                        "share_count": 25,
                    }
                ]
            },
        }

        with patch.object(
            tiktok_client.http_client, "post", return_value=mock_response
        ):
            metrics = await tiktok_client.get_engagement_metrics(
                mock_account, "test_video_123"
            )

        assert metrics["views"] == 1000
        assert metrics["likes"] == 100
        assert metrics["comments"] == 50
        assert metrics["shares"] == 25

    @pytest.mark.asyncio
    async def test_get_engagement_metrics_no_video(self, tiktok_client, mock_account):
        """Test metrics retrieval when video not found."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "error": {"code": "ok"},
            "data": {"videos": []},
        }

        with patch.object(
            tiktok_client.http_client, "post", return_value=mock_response
        ):
            metrics = await tiktok_client.get_engagement_metrics(
                mock_account, "nonexistent_video"
            )

        assert metrics == {}

    @pytest.mark.asyncio
    async def test_get_engagement_metrics_api_error(self, tiktok_client, mock_account):
        """Test metrics retrieval handles API errors gracefully."""
        with patch.object(
            tiktok_client.http_client, "post", side_effect=Exception("Network error")
        ):
            metrics = await tiktok_client.get_engagement_metrics(
                mock_account, "test_video"
            )

        assert metrics == {}


class TestTikTokClientEdgeCases:
    """Edge case tests for TikTok client."""

    @pytest.fixture
    def tiktok_client(self):
        """Create TikTok client instance."""
        return TikTokClient()

    def test_validate_mov_format(self, tiktok_client):
        """Test MOV format is accepted."""
        content_data = {"video_path": "/videos/test.mov"}
        is_valid, _ = tiktok_client._validate_video_content(content_data)
        assert is_valid

    def test_validate_webm_format(self, tiktok_client):
        """Test WebM format is accepted."""
        content_data = {"video_url": "https://example.com/video.webm"}
        is_valid, _ = tiktok_client._validate_video_content(content_data)
        assert is_valid

    def test_validate_exact_size_limit(self, tiktok_client):
        """Test video exactly at size limit passes."""
        content_data = {
            "video_url": "https://example.com/video.mp4",
            "file_size": tiktok_client.MAX_VIDEO_SIZE_BYTES,  # Exactly at limit
        }
        is_valid, _ = tiktok_client._validate_video_content(content_data)
        assert is_valid

    def test_validate_exact_duration_limit(self, tiktok_client):
        """Test video exactly at duration limit passes."""
        content_data = {
            "video_url": "https://example.com/video.mp4",
            "duration": tiktok_client.MAX_VIDEO_DURATION_SECONDS,  # Exactly 60 min
        }
        is_valid, _ = tiktok_client._validate_video_content(content_data)
        assert is_valid
