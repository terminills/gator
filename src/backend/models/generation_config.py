"""
Generation Configuration Models

Models for configuring content generation parameters including resolution,
quality, and other options.
"""

from enum import Enum
from typing import Optional, Dict, Any
from pydantic import BaseModel, Field


class ImageResolution(str, Enum):
    """
    Standard image resolution options.
    
    Provides common aspect ratios and resolutions for various content types.
    """
    # Square formats
    SD_512 = "512x512"  # Standard Definition square
    HD_1024 = "1024x1024"  # High Definition square
    
    # Portrait formats (9:16 for social media)
    PORTRAIT_SD = "576x1024"  # Portrait SD
    PORTRAIT_HD = "720x1280"  # Portrait 720p
    PORTRAIT_FHD = "1080x1920"  # Portrait 1080p (Full HD)
    
    # Landscape formats (16:9 for standard content)
    LANDSCAPE_SD = "1024x576"  # Landscape SD
    LANDSCAPE_HD = "1280x720"  # Landscape 720p (HD)
    LANDSCAPE_FHD = "1920x1080"  # Landscape 1080p (Full HD)
    LANDSCAPE_2K = "2560x1440"  # Landscape 2K (QHD)
    LANDSCAPE_4K = "3840x2160"  # Landscape 4K (UHD)
    
    # Instagram/Social formats
    INSTAGRAM_POST = "1080x1080"  # Instagram square post
    INSTAGRAM_STORY = "1080x1920"  # Instagram story/reel
    INSTAGRAM_LANDSCAPE = "1080x566"  # Instagram landscape post
    
    # Custom formats
    CUSTOM = "custom"  # Allow custom width x height


class VideoResolution(str, Enum):
    """
    Standard video resolution options.
    """
    # Standard definitions
    SD_480 = "640x480"  # Standard Definition
    SD_576 = "720x576"  # PAL Standard Definition
    
    # High definitions
    HD_720 = "1280x720"  # 720p HD
    FHD_1080 = "1920x1080"  # 1080p Full HD
    QHD_1440 = "2560x1440"  # 1440p QHD/2K
    UHD_4K = "3840x2160"  # 4K UHD
    
    # Portrait/Vertical video for social media
    VERTICAL_720 = "720x1280"  # Vertical 720p
    VERTICAL_1080 = "1080x1920"  # Vertical 1080p (TikTok, Reels, Stories)
    
    # Custom
    CUSTOM = "custom"


class QualityPreset(str, Enum):
    """Quality presets for generation."""
    DRAFT = "draft"  # Fast generation, lower quality
    STANDARD = "standard"  # Balanced quality and speed
    HIGH = "high"  # High quality, slower generation
    PREMIUM = "premium"  # Maximum quality, slowest generation


class AspectRatio(str, Enum):
    """Common aspect ratios."""
    SQUARE = "1:1"  # Square (Instagram post, profile pictures)
    PORTRAIT = "9:16"  # Portrait (Stories, Reels, TikTok)
    LANDSCAPE = "16:9"  # Landscape (YouTube, standard video)
    WIDESCREEN = "21:9"  # Widescreen cinematic
    INSTAGRAM_LANDSCAPE = "1.91:1"  # Instagram landscape


class ImageGenerationConfig(BaseModel):
    """Configuration for image generation."""
    
    resolution: ImageResolution = Field(
        default=ImageResolution.HD_1024,
        description="Output image resolution"
    )
    custom_width: Optional[int] = Field(
        default=None,
        ge=256,
        le=4096,
        description="Custom width when resolution is CUSTOM"
    )
    custom_height: Optional[int] = Field(
        default=None,
        ge=256,
        le=4096,
        description="Custom height when resolution is CUSTOM"
    )
    quality: QualityPreset = Field(
        default=QualityPreset.HIGH,
        description="Generation quality preset"
    )
    num_inference_steps: Optional[int] = Field(
        default=None,
        ge=10,
        le=150,
        description="Number of inference steps (overrides quality preset)"
    )
    guidance_scale: Optional[float] = Field(
        default=None,
        ge=1.0,
        le=20.0,
        description="Guidance scale for generation"
    )
    seed: Optional[int] = Field(
        default=None,
        description="Random seed for reproducibility"
    )
    
    def get_dimensions(self) -> tuple[int, int]:
        """
        Get the width and height dimensions.
        
        Returns:
            Tuple of (width, height)
        """
        if self.resolution == ImageResolution.CUSTOM:
            if self.custom_width and self.custom_height:
                return (self.custom_width, self.custom_height)
            else:
                # Default to 1024x1024 if custom not specified
                return (1024, 1024)
        else:
            # Parse resolution string (e.g., "1920x1080")
            width, height = map(int, self.resolution.value.split('x'))
            return (width, height)
    
    def get_quality_params(self) -> Dict[str, Any]:
        """
        Get quality parameters based on preset.
        
        Returns:
            Dictionary with num_inference_steps and guidance_scale
        """
        quality_map = {
            QualityPreset.DRAFT: {"num_inference_steps": 20, "guidance_scale": 6.0},
            QualityPreset.STANDARD: {"num_inference_steps": 30, "guidance_scale": 7.5},
            QualityPreset.HIGH: {"num_inference_steps": 50, "guidance_scale": 8.0},
            QualityPreset.PREMIUM: {"num_inference_steps": 100, "guidance_scale": 9.0},
        }
        
        params = quality_map.get(self.quality, quality_map[QualityPreset.HIGH])
        
        # Override with custom values if provided
        if self.num_inference_steps is not None:
            params["num_inference_steps"] = self.num_inference_steps
        if self.guidance_scale is not None:
            params["guidance_scale"] = self.guidance_scale
            
        return params


class VideoGenerationConfig(BaseModel):
    """Configuration for video generation."""
    
    resolution: VideoResolution = Field(
        default=VideoResolution.FHD_1080,
        description="Output video resolution"
    )
    custom_width: Optional[int] = Field(
        default=None,
        ge=256,
        le=4096,
        description="Custom width when resolution is CUSTOM"
    )
    custom_height: Optional[int] = Field(
        default=None,
        ge=256,
        le=4096,
        description="Custom height when resolution is CUSTOM"
    )
    fps: int = Field(
        default=30,
        ge=24,
        le=60,
        description="Frames per second"
    )
    duration: float = Field(
        default=3.0,
        ge=0.5,
        le=30.0,
        description="Video duration in seconds"
    )
    quality: QualityPreset = Field(
        default=QualityPreset.HIGH,
        description="Generation quality preset"
    )
    
    def get_dimensions(self) -> tuple[int, int]:
        """
        Get the width and height dimensions.
        
        Returns:
            Tuple of (width, height)
        """
        if self.resolution == VideoResolution.CUSTOM:
            if self.custom_width and self.custom_height:
                return (self.custom_width, self.custom_height)
            else:
                # Default to 1920x1080 if custom not specified
                return (1920, 1080)
        else:
            # Parse resolution string (e.g., "1920x1080")
            width, height = map(int, self.resolution.value.split('x'))
            return (width, height)


# Resolution helper functions
def get_resolution_from_dimensions(width: int, height: int) -> Optional[ImageResolution]:
    """
    Get the ImageResolution enum from width and height.
    
    Args:
        width: Image width in pixels
        height: Image height in pixels
        
    Returns:
        ImageResolution enum value or None if no match
    """
    resolution_str = f"{width}x{height}"
    for res in ImageResolution:
        if res.value == resolution_str:
            return res
    return None


def get_aspect_ratio(width: int, height: int) -> str:
    """
    Calculate aspect ratio from dimensions.
    
    Args:
        width: Image width in pixels
        height: Image height in pixels
        
    Returns:
        Aspect ratio as string (e.g., "16:9", "1:1")
    """
    from math import gcd
    
    divisor = gcd(width, height)
    ratio_w = width // divisor
    ratio_h = height // divisor
    
    return f"{ratio_w}:{ratio_h}"


def get_recommended_resolutions_for_platform(platform: str) -> list[ImageResolution]:
    """
    Get recommended resolutions for a specific platform.
    
    Args:
        platform: Platform name (e.g., "instagram", "youtube", "tiktok")
        
    Returns:
        List of recommended ImageResolution values
    """
    platform_resolutions = {
        "instagram": [
            ImageResolution.INSTAGRAM_POST,
            ImageResolution.INSTAGRAM_STORY,
            ImageResolution.INSTAGRAM_LANDSCAPE,
        ],
        "tiktok": [
            ImageResolution.PORTRAIT_FHD,
            ImageResolution.PORTRAIT_HD,
        ],
        "youtube": [
            ImageResolution.LANDSCAPE_FHD,
            ImageResolution.LANDSCAPE_HD,
            ImageResolution.LANDSCAPE_4K,
        ],
        "twitter": [
            ImageResolution.LANDSCAPE_FHD,
            ImageResolution.HD_1024,
        ],
        "facebook": [
            ImageResolution.LANDSCAPE_FHD,
            ImageResolution.HD_1024,
        ],
        "onlyfans": [
            ImageResolution.LANDSCAPE_FHD,
            ImageResolution.PORTRAIT_FHD,
            ImageResolution.HD_1024,
        ],
    }
    
    return platform_resolutions.get(
        platform.lower(),
        [ImageResolution.LANDSCAPE_FHD, ImageResolution.HD_1024]
    )
