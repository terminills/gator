"""
Path Utilities for Gator AI Platform

Centralized path management to ensure consistent file path handling across the codebase.
This module provides a single source of truth for all paths used in the application,
preventing issues with multiple paths referencing the same files.

Usage:
    from backend.utils.paths import GatorPaths

    paths = GatorPaths()
    models_dir = paths.models_dir
    content_dir = paths.generated_content_dir
"""

from functools import lru_cache
from pathlib import Path
from typing import List, Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class PathSettings(BaseSettings):
    """
    Path configuration settings with environment variable support.

    All paths can be overridden via environment variables with
    the prefix 'GATOR_' (e.g., GATOR_DATA_DIR).
    """

    model_config = SettingsConfigDict(
        env_prefix="GATOR_",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Base directories
    data_dir: str = Field(
        default="/opt/gator/data",
        description="Base data directory for all Gator data",
    )

    # Override individual paths (optional)
    models_dir: Optional[str] = Field(
        default=None,
        description="Override: AI models directory",
    )
    generated_content_dir: Optional[str] = Field(
        default=None,
        description="Override: Generated content directory",
    )
    uploads_dir: Optional[str] = Field(
        default=None,
        description="Override: User uploads directory",
    )
    base_images_dir: Optional[str] = Field(
        default=None,
        description="Override: Base images directory for personas",
    )
    backups_dir: Optional[str] = Field(
        default=None,
        description="Override: Backups directory",
    )
    logs_dir: Optional[str] = Field(
        default=None,
        description="Override: Logs directory",
    )
    temp_dir: Optional[str] = Field(
        default=None,
        description="Override: Temporary files directory",
    )


class GatorPaths:
    """
    Centralized path management for the Gator AI Platform.

    This class provides a single source of truth for all paths used in the application.
    Paths are computed lazily and cached for performance.

    Standard Directory Structure:
        /opt/gator/data/           # Base data directory (configurable)
        ├── models/                # AI models (checkpoints, LoRAs, etc.)
        │   ├── checkpoints/       # Model checkpoints
        │   ├── loras/             # LoRA models
        │   ├── base_images/       # Base images for personas
        │   └── embeddings/        # Text embeddings
        ├── generated_content/     # Generated AI content
        │   ├── images/            # Generated images
        │   ├── videos/            # Generated videos
        │   └── text/              # Generated text content
        ├── uploads/               # User uploads
        ├── backups/               # Database and content backups
        ├── logs/                  # Application logs
        └── temp/                  # Temporary files

    Attributes:
        project_root: Root directory of the Gator project
        data_dir: Base data directory
        models_dir: AI models directory
        generated_content_dir: Generated content directory
        uploads_dir: User uploads directory
        base_images_dir: Base images for personas
        backups_dir: Backups directory
        logs_dir: Application logs directory
        temp_dir: Temporary files directory
    """

    def __init__(self, settings: Optional[PathSettings] = None):
        """
        Initialize GatorPaths with optional custom settings.

        Args:
            settings: Custom path settings (uses defaults if not provided)
        """
        self._settings = settings or PathSettings()
        self._project_root: Optional[Path] = None

    @property
    def project_root(self) -> Path:
        """
        Get the project root directory.

        Computed by finding the directory containing pyproject.toml or
        falling back to the parent of the src directory.
        """
        if self._project_root is None:
            self._project_root = self._find_project_root()
        return self._project_root

    def _find_project_root(self) -> Path:
        """Find the project root by looking for pyproject.toml."""
        # Start from this file's location
        current = Path(__file__).resolve()

        # Walk up looking for pyproject.toml
        for parent in [current] + list(current.parents):
            if (parent / "pyproject.toml").exists():
                return parent

        # Fallback: assume src/backend/utils/paths.py structure
        # Go up 4 levels: paths.py -> utils -> backend -> src -> root
        return Path(__file__).resolve().parents[3]

    @property
    def data_dir(self) -> Path:
        """Base data directory for all Gator data."""
        return Path(self._settings.data_dir)

    @property
    def models_dir(self) -> Path:
        """Directory for AI models (checkpoints, LoRAs, etc.)."""
        if self._settings.models_dir:
            return Path(self._settings.models_dir)
        return self.data_dir / "models"

    @property
    def checkpoints_dir(self) -> Path:
        """Directory for model checkpoints."""
        return self.models_dir / "checkpoints"

    @property
    def loras_dir(self) -> Path:
        """Directory for LoRA models."""
        return self.models_dir / "loras"

    @property
    def embeddings_dir(self) -> Path:
        """Directory for text embeddings."""
        return self.models_dir / "embeddings"

    @property
    def base_images_dir(self) -> Path:
        """Directory for persona base images."""
        if self._settings.base_images_dir:
            return Path(self._settings.base_images_dir)
        return self.models_dir / "base_images"

    @property
    def generated_content_dir(self) -> Path:
        """Directory for generated AI content."""
        if self._settings.generated_content_dir:
            return Path(self._settings.generated_content_dir)
        return self.data_dir / "generated_content"

    @property
    def generated_images_dir(self) -> Path:
        """Directory for generated images."""
        return self.generated_content_dir / "images"

    @property
    def generated_videos_dir(self) -> Path:
        """Directory for generated videos."""
        return self.generated_content_dir / "videos"

    @property
    def generated_text_dir(self) -> Path:
        """Directory for generated text content."""
        return self.generated_content_dir / "text"

    @property
    def uploads_dir(self) -> Path:
        """Directory for user uploads."""
        if self._settings.uploads_dir:
            return Path(self._settings.uploads_dir)
        return self.data_dir / "uploads"

    @property
    def backups_dir(self) -> Path:
        """Directory for backups."""
        if self._settings.backups_dir:
            return Path(self._settings.backups_dir)
        return self.data_dir / "backups"

    @property
    def logs_dir(self) -> Path:
        """Directory for application logs."""
        if self._settings.logs_dir:
            return Path(self._settings.logs_dir)
        return self.data_dir / "logs"

    @property
    def temp_dir(self) -> Path:
        """Directory for temporary files."""
        if self._settings.temp_dir:
            return Path(self._settings.temp_dir)
        return self.data_dir / "temp"

    # Frontend paths (relative to project root)
    @property
    def frontend_dir(self) -> Path:
        """Frontend public directory."""
        return self.project_root / "frontend" / "public"

    @property
    def admin_panel_dir(self) -> Path:
        """Admin panel directory."""
        return self.project_root / "admin_panel"

    # Database path
    @property
    def database_path(self) -> Path:
        """Default SQLite database path."""
        return self.project_root / "gator.db"

    def ensure_directories(self) -> None:
        """
        Create all required directories if they don't exist.

        This should be called during application startup to ensure
        all necessary directories are available.
        """
        directories = [
            self.data_dir,
            self.models_dir,
            self.checkpoints_dir,
            self.loras_dir,
            self.embeddings_dir,
            self.base_images_dir,
            self.generated_content_dir,
            self.generated_images_dir,
            self.generated_videos_dir,
            self.generated_text_dir,
            self.uploads_dir,
            self.backups_dir,
            self.logs_dir,
            self.temp_dir,
        ]

        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)

    def get_relative_path(
        self, absolute_path: Path, base: Optional[Path] = None
    ) -> Path:
        """
        Get a relative path from an absolute path.

        Args:
            absolute_path: The absolute path to convert
            base: Base directory for relative path (defaults to data_dir)

        Returns:
            Relative path from the base directory
        """
        if base is None:
            base = self.data_dir
        try:
            return absolute_path.relative_to(base)
        except ValueError:
            # Path is not relative to base, return as-is
            return absolute_path

    def resolve_path(self, path: str, base: Optional[Path] = None) -> Path:
        """
        Resolve a path string to an absolute Path.

        If the path is relative, it's resolved against the base directory.
        If the path is absolute, it's returned as-is.

        Args:
            path: Path string to resolve
            base: Base directory for relative paths (defaults to data_dir)

        Returns:
            Resolved absolute Path
        """
        if base is None:
            base = self.data_dir

        p = Path(path)
        if p.is_absolute():
            return p
        return (base / p).resolve()

    def find_file(
        self,
        filename: str,
        search_dirs: Optional[List[Path]] = None,
    ) -> Optional[Path]:
        """
        Search for a file in multiple directories.

        Args:
            filename: Name of the file to find
            search_dirs: Directories to search (defaults to common directories)

        Returns:
            Path to the file if found, None otherwise
        """
        if search_dirs is None:
            search_dirs = [
                self.models_dir,
                self.checkpoints_dir,
                self.loras_dir,
                self.generated_content_dir,
                self.uploads_dir,
                self.base_images_dir,
            ]

        for directory in search_dirs:
            candidate = directory / filename
            if candidate.exists():
                return candidate

        return None

    def find_model(self, model_name: str) -> Optional[Path]:
        """
        Search for a model file by name.

        Searches in models_dir, checkpoints_dir, and loras_dir.

        Args:
            model_name: Name of the model file

        Returns:
            Path to the model if found, None otherwise
        """
        return self.find_file(
            model_name,
            search_dirs=[
                self.models_dir,
                self.checkpoints_dir,
                self.loras_dir,
            ],
        )

    def get_url_path(self, file_path: Path, mount_point: str = "/static") -> str:
        """
        Convert a file path to a URL path for serving.

        Args:
            file_path: Absolute path to the file
            mount_point: URL mount point for the static files

        Returns:
            URL path string
        """
        try:
            # Try to make it relative to generated_content_dir first
            relative = file_path.relative_to(self.generated_content_dir)
            return f"/generated_content/{relative}"
        except ValueError:
            pass

        try:
            # Try base_images_dir
            relative = file_path.relative_to(self.base_images_dir)
            return f"/base_images/{relative}"
        except ValueError:
            pass

        try:
            # Try uploads_dir
            relative = file_path.relative_to(self.uploads_dir)
            return f"/uploads/{relative}"
        except ValueError:
            pass

        # Fallback to mount_point
        return f"{mount_point}/{file_path.name}"

    def __repr__(self) -> str:
        return (
            f"GatorPaths(\n"
            f"  project_root={self.project_root},\n"
            f"  data_dir={self.data_dir},\n"
            f"  models_dir={self.models_dir},\n"
            f"  generated_content_dir={self.generated_content_dir}\n"
            f")"
        )


@lru_cache()
def get_paths() -> GatorPaths:
    """
    Get a cached GatorPaths instance.

    This is the recommended way to access paths throughout the application.
    The instance is cached to avoid re-computing paths on every call.

    Returns:
        GatorPaths: Configured paths instance

    Example:
        from backend.utils.paths import get_paths

        paths = get_paths()
        model_path = paths.models_dir / "my_model.safetensors"
    """
    return GatorPaths()


# Convenience aliases for common paths
def get_project_root() -> Path:
    """Get the project root directory."""
    return get_paths().project_root


def get_data_dir() -> Path:
    """Get the base data directory."""
    return get_paths().data_dir


def get_models_dir() -> Path:
    """Get the AI models directory."""
    return get_paths().models_dir


def get_generated_content_dir() -> Path:
    """Get the generated content directory."""
    return get_paths().generated_content_dir


def get_base_images_dir() -> Path:
    """Get the base images directory."""
    return get_paths().base_images_dir


def get_uploads_dir() -> Path:
    """Get the uploads directory."""
    return get_paths().uploads_dir
