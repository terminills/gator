"""
Tests for Path Utilities

Tests the centralized path management system.
"""

import os
from pathlib import Path
from unittest.mock import patch

import pytest

from backend.utils.paths import (
    GatorPaths,
    PathSettings,
    get_base_images_dir,
    get_data_dir,
    get_generated_content_dir,
    get_models_dir,
    get_paths,
    get_project_root,
    get_uploads_dir,
)


class TestGatorPaths:
    """Tests for GatorPaths class."""

    def test_default_data_dir(self):
        """Test that default data directory is set correctly."""
        paths = GatorPaths()
        assert paths.data_dir == Path("/opt/gator/data")

    def test_custom_data_dir(self):
        """Test that custom data directory can be set via settings."""
        settings = PathSettings(data_dir="/custom/data")
        paths = GatorPaths(settings=settings)
        assert paths.data_dir == Path("/custom/data")

    def test_models_dir_default(self):
        """Test that models directory is under data directory by default."""
        paths = GatorPaths()
        assert paths.models_dir == Path("/opt/gator/data/models")

    def test_models_dir_override(self):
        """Test that models directory can be overridden."""
        settings = PathSettings(models_dir="/custom/models")
        paths = GatorPaths(settings=settings)
        assert paths.models_dir == Path("/custom/models")

    def test_base_images_dir_default(self):
        """Test that base images directory is under models by default."""
        paths = GatorPaths()
        assert paths.base_images_dir == Path("/opt/gator/data/models/base_images")

    def test_generated_content_dir_default(self):
        """Test that generated content directory is under data by default."""
        paths = GatorPaths()
        assert paths.generated_content_dir == Path("/opt/gator/data/generated_content")

    def test_generated_content_dir_override(self):
        """Test that generated content directory can be overridden."""
        settings = PathSettings(generated_content_dir="/custom/content")
        paths = GatorPaths(settings=settings)
        assert paths.generated_content_dir == Path("/custom/content")

    def test_uploads_dir_default(self):
        """Test that uploads directory is under data by default."""
        paths = GatorPaths()
        assert paths.uploads_dir == Path("/opt/gator/data/uploads")

    def test_backups_dir_default(self):
        """Test that backups directory is under data by default."""
        paths = GatorPaths()
        assert paths.backups_dir == Path("/opt/gator/data/backups")

    def test_logs_dir_default(self):
        """Test that logs directory is under data by default."""
        paths = GatorPaths()
        assert paths.logs_dir == Path("/opt/gator/data/logs")

    def test_temp_dir_default(self):
        """Test that temp directory is under data by default."""
        paths = GatorPaths()
        assert paths.temp_dir == Path("/opt/gator/data/temp")

    def test_subdirectories(self):
        """Test that subdirectories are correctly computed."""
        paths = GatorPaths()
        assert paths.checkpoints_dir == paths.models_dir / "checkpoints"
        assert paths.loras_dir == paths.models_dir / "loras"
        assert paths.embeddings_dir == paths.models_dir / "embeddings"
        assert paths.generated_images_dir == paths.generated_content_dir / "images"
        assert paths.generated_videos_dir == paths.generated_content_dir / "videos"
        assert paths.generated_text_dir == paths.generated_content_dir / "text"

    def test_project_root_detection(self):
        """Test that project root is correctly detected."""
        paths = GatorPaths()
        # Project root should contain pyproject.toml
        project_root = paths.project_root
        assert (project_root / "pyproject.toml").exists()

    def test_frontend_dir(self):
        """Test that frontend directory is relative to project root."""
        paths = GatorPaths()
        assert paths.frontend_dir == paths.project_root / "frontend" / "public"

    def test_admin_panel_dir(self):
        """Test that admin panel directory is relative to project root."""
        paths = GatorPaths()
        assert paths.admin_panel_dir == paths.project_root / "admin_panel"


class TestPathResolution:
    """Tests for path resolution methods."""

    def test_resolve_absolute_path(self):
        """Test that absolute paths are returned as-is."""
        paths = GatorPaths()
        absolute = Path("/some/absolute/path")
        assert paths.resolve_path(str(absolute)) == absolute

    def test_resolve_relative_path(self):
        """Test that relative paths are resolved against base."""
        paths = GatorPaths()
        relative = "some/relative/path"
        resolved = paths.resolve_path(relative)
        assert resolved == (paths.data_dir / relative).resolve()

    def test_resolve_relative_path_custom_base(self):
        """Test that relative paths can use custom base."""
        paths = GatorPaths()
        relative = "some/file.txt"
        custom_base = Path("/custom/base")
        resolved = paths.resolve_path(relative, base=custom_base)
        assert resolved == (custom_base / relative).resolve()

    def test_get_relative_path(self):
        """Test getting relative path from absolute."""
        paths = GatorPaths()
        absolute = paths.data_dir / "models" / "test.bin"
        relative = paths.get_relative_path(absolute)
        assert relative == Path("models/test.bin")

    def test_get_relative_path_not_under_base(self):
        """Test that paths not under base are returned as-is."""
        paths = GatorPaths()
        absolute = Path("/some/other/path")
        result = paths.get_relative_path(absolute)
        assert result == absolute


class TestUrlPaths:
    """Tests for URL path generation."""

    def test_get_url_path_generated_content(self, tmp_path):
        """Test URL path for generated content."""
        settings = PathSettings(generated_content_dir=str(tmp_path / "content"))
        paths = GatorPaths(settings=settings)
        file_path = paths.generated_content_dir / "images" / "test.jpg"
        url = paths.get_url_path(file_path)
        assert url == "/generated_content/images/test.jpg"

    def test_get_url_path_base_images(self, tmp_path):
        """Test URL path for base images."""
        settings = PathSettings(base_images_dir=str(tmp_path / "base"))
        paths = GatorPaths(settings=settings)
        file_path = paths.base_images_dir / "persona1.jpg"
        url = paths.get_url_path(file_path)
        assert url == "/base_images/persona1.jpg"


class TestConvenienceFunctions:
    """Tests for convenience functions."""

    def test_get_paths_returns_cached_instance(self):
        """Test that get_paths returns a cached instance."""
        # Clear cache first
        get_paths.cache_clear()
        paths1 = get_paths()
        paths2 = get_paths()
        assert paths1 is paths2

    def test_get_project_root(self):
        """Test get_project_root convenience function."""
        root = get_project_root()
        assert (root / "pyproject.toml").exists()

    def test_get_data_dir(self):
        """Test get_data_dir convenience function."""
        data_dir = get_data_dir()
        assert data_dir == Path("/opt/gator/data")

    def test_get_models_dir(self):
        """Test get_models_dir convenience function."""
        models_dir = get_models_dir()
        assert models_dir == Path("/opt/gator/data/models")

    def test_get_generated_content_dir(self):
        """Test get_generated_content_dir convenience function."""
        content_dir = get_generated_content_dir()
        assert content_dir == Path("/opt/gator/data/generated_content")

    def test_get_base_images_dir(self):
        """Test get_base_images_dir convenience function."""
        base_images = get_base_images_dir()
        assert base_images == Path("/opt/gator/data/models/base_images")

    def test_get_uploads_dir(self):
        """Test get_uploads_dir convenience function."""
        uploads = get_uploads_dir()
        assert uploads == Path("/opt/gator/data/uploads")


class TestEnvironmentOverrides:
    """Tests for environment variable overrides."""

    def test_env_override_data_dir(self):
        """Test that GATOR_DATA_DIR env var overrides default."""
        with patch.dict(os.environ, {"GATOR_DATA_DIR": "/env/data"}):
            settings = PathSettings()
            paths = GatorPaths(settings=settings)
            assert paths.data_dir == Path("/env/data")

    def test_env_override_models_dir(self):
        """Test that GATOR_MODELS_DIR env var overrides default."""
        with patch.dict(os.environ, {"GATOR_MODELS_DIR": "/env/models"}):
            settings = PathSettings()
            paths = GatorPaths(settings=settings)
            assert paths.models_dir == Path("/env/models")


class TestFindFile:
    """Tests for file finding functionality."""

    def test_find_file_not_found(self):
        """Test that find_file returns None for non-existent files."""
        paths = GatorPaths()
        result = paths.find_file("nonexistent_file_xyz.bin")
        assert result is None

    def test_find_file_in_search_dirs(self, tmp_path):
        """Test finding a file in search directories."""
        # Create a test file
        test_file = tmp_path / "test_model.bin"
        test_file.write_bytes(b"test data")

        settings = PathSettings(models_dir=str(tmp_path))
        paths = GatorPaths(settings=settings)

        result = paths.find_file("test_model.bin", search_dirs=[tmp_path])
        assert result == test_file

    def test_find_model(self, tmp_path):
        """Test finding a model file."""
        # Create a test model file
        model_file = tmp_path / "llama.gguf"
        model_file.write_bytes(b"model data")

        settings = PathSettings(models_dir=str(tmp_path))
        paths = GatorPaths(settings=settings)

        result = paths.find_model("llama.gguf")
        assert result == model_file


class TestEnsureDirectories:
    """Tests for directory creation."""

    def test_ensure_directories(self, tmp_path):
        """Test that ensure_directories creates all required directories."""
        settings = PathSettings(data_dir=str(tmp_path / "data"))
        paths = GatorPaths(settings=settings)

        # Initially directories should not exist
        assert not paths.data_dir.exists()

        # Create directories
        paths.ensure_directories()

        # Verify all directories were created
        assert paths.data_dir.exists()
        assert paths.models_dir.exists()
        assert paths.checkpoints_dir.exists()
        assert paths.loras_dir.exists()
        assert paths.embeddings_dir.exists()
        assert paths.base_images_dir.exists()
        assert paths.generated_content_dir.exists()
        assert paths.generated_images_dir.exists()
        assert paths.generated_videos_dir.exists()
        assert paths.generated_text_dir.exists()
        assert paths.uploads_dir.exists()
        assert paths.backups_dir.exists()
        assert paths.logs_dir.exists()
        assert paths.temp_dir.exists()


class TestPathRepr:
    """Tests for string representation."""

    def test_repr(self):
        """Test that __repr__ returns a useful string."""
        paths = GatorPaths()
        repr_str = repr(paths)
        assert "GatorPaths" in repr_str
        assert "project_root" in repr_str
        assert "data_dir" in repr_str
        assert "models_dir" in repr_str
