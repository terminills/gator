"""
Video model handlers for video generation.

Handles video generation using various backends:
- Local: Stable Video Diffusion, AnimateDiff
- Cloud: Runway ML
"""

import asyncio
import os
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx

from backend.config.logging import get_logger
from backend.config.settings import get_settings

from .base import BaseModelHandler, ModelCapabilities, ModelType
from .gpu_manager import get_gpu_manager
from .model_cache import get_model_cache
from .image_models import get_image_handler

logger = get_logger(__name__)


class VideoModelHandler(BaseModelHandler):
    """
    Handler for video generation models.

    Supports multiple backends:
    - Stable Video Diffusion (local)
    - AnimateDiff (local)
    - Frame-by-frame generation (using image models)
    - Runway ML API (cloud)
    """

    def __init__(self):
        """Initialize the video model handler."""
        super().__init__(ModelType.VIDEO)
        self.settings = get_settings()
        self.gpu_manager = get_gpu_manager()
        self.model_cache = get_model_cache()

        # Backend availability
        self.svd_available = False
        self.animatediff_available = False
        self.runway_available = False

        # Model paths
        self.model_dir = Path(
            os.environ.get("AI_MODEL_PATH", "/opt/gator/data/models/video")
        )
        self.output_dir = Path(
            os.environ.get("CONTENT_OUTPUT_PATH", "generated_content/videos")
        )

    async def initialize(self) -> None:
        """Initialize the video model handler and discover available models."""
        if self._initialized:
            return

        logger.info("Initializing video model handler...")

        # Detect available backends
        await self._detect_backends()

        # Discover local models
        await self._discover_local_models()

        # Register cloud models
        await self._register_cloud_models()

        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self._initialized = True
        logger.info(
            f"Video model handler initialized with {len(self.available_models)} models"
        )

    async def _detect_backends(self) -> None:
        """Detect available inference backends."""
        # Check SVD
        try:
            from diffusers import StableVideoDiffusionPipeline  # noqa: F401

            self.svd_available = True
            logger.info("✓ Stable Video Diffusion is available")
        except ImportError:
            logger.debug("SVD not available")

        # Check Runway API
        runway_key = os.environ.get("RUNWAY_API_KEY")
        if runway_key:
            self.runway_available = True
            self.runway_api_key = runway_key
            logger.info("✓ Runway API key configured")

    async def _discover_local_models(self) -> None:
        """Discover local video models."""
        if not self.model_dir.exists():
            logger.debug(f"Model directory does not exist: {self.model_dir}")
            return

        # Find video model directories
        for model_dir in self.model_dir.iterdir():
            if model_dir.is_dir() and (model_dir / "model_index.json").exists():
                model_name = model_dir.name

                # Determine model type
                model_type = "svd"
                if "animate" in model_name.lower():
                    model_type = "animatediff"

                self.available_models[model_name] = {
                    "name": model_name,
                    "path": str(model_dir),
                    "source": "local",
                    "backend": model_type,
                    "capabilities": ModelCapabilities(
                        max_resolution=(1024, 576),
                        supported_formats=["mp4", "gif"],
                    ).to_dict(),
                }
                logger.debug(f"Found video model: {model_name}")

        # Add frame-by-frame option (uses image models)
        self.available_models["frame-by-frame"] = {
            "name": "frame-by-frame",
            "source": "synthetic",
            "backend": "frame-by-frame",
            "description": "Generate video frame-by-frame using image models",
            "capabilities": ModelCapabilities(
                supported_formats=["mp4", "gif"],
            ).to_dict(),
        }

    async def _register_cloud_models(self) -> None:
        """Register cloud API models."""
        if self.runway_available:
            self.available_models["runway:gen3"] = {
                "name": "gen3",
                "source": "runway",
                "backend": "runway",
                "capabilities": ModelCapabilities(
                    max_resolution=(1280, 768),
                    supported_formats=["mp4"],
                ).to_dict(),
            }

    async def generate(
        self,
        prompt: str,
        model: Optional[str] = None,
        width: int = 1024,
        height: int = 576,
        num_frames: int = 25,
        fps: int = 7,
        seed: Optional[int] = None,
        init_image: Optional[str] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Generate a video.

        Args:
            prompt: Text prompt for video generation
            model: Model name (optional, will select best available)
            width: Video width
            height: Video height
            num_frames: Number of frames to generate
            fps: Frames per second
            seed: Random seed for reproducibility
            init_image: Path to initial image for img2vid
            **kwargs: Additional generation parameters

        Returns:
            Dictionary with video path and metadata
        """
        if not self._initialized:
            await self.initialize()

        # Select model if not specified
        if not model:
            model = self._select_best_model(**kwargs)

        if not model:
            raise ValueError("No video model available for generation")

        model_info = self.available_models.get(model)
        if not model_info:
            raise ValueError(f"Model not found: {model}")

        backend = model_info.get("backend", "svd")

        logger.info(f"Generating video with model '{model}' using {backend}")

        if backend == "svd":
            return await self._generate_svd(
                prompt=prompt,
                model_info=model_info,
                width=width,
                height=height,
                num_frames=num_frames,
                fps=fps,
                seed=seed,
                init_image=init_image,
                **kwargs,
            )
        elif backend == "frame-by-frame":
            return await self._generate_frame_by_frame(
                prompt=prompt,
                width=width,
                height=height,
                num_frames=num_frames,
                fps=fps,
                seed=seed,
                **kwargs,
            )
        elif backend == "runway":
            return await self._generate_runway(
                prompt=prompt,
                model_info=model_info,
                **kwargs,
            )
        else:
            raise ValueError(f"Unsupported backend: {backend}")

    def _select_best_model(self, **kwargs) -> Optional[str]:
        """Select the best available model for generation."""
        if not self.available_models:
            return None

        # Prefer local SVD models
        svd_models = [
            name
            for name, info in self.available_models.items()
            if info.get("backend") == "svd"
        ]
        if svd_models:
            return svd_models[0]

        # Fall back to frame-by-frame
        if "frame-by-frame" in self.available_models:
            return "frame-by-frame"

        # Fall back to cloud
        if self.runway_available:
            return "runway:gen3"

        return None

    async def _generate_svd(
        self,
        prompt: str,
        model_info: Dict[str, Any],
        width: int,
        height: int,
        num_frames: int,
        fps: int,
        seed: Optional[int],
        init_image: Optional[str],
        **kwargs,
    ) -> Dict[str, Any]:
        """Generate video using Stable Video Diffusion."""
        import torch
        from diffusers import StableVideoDiffusionPipeline
        from diffusers.utils import export_to_video
        from PIL import Image

        model_path = model_info.get("path")

        # Check cache
        cache_key = f"svd:{model_path}"
        pipe = self.model_cache.get(cache_key)

        if pipe is None:
            device = self.gpu_manager.select_device_for_model(
                cache_key, required_memory_gb=16.0
            )

            logger.info(f"Loading SVD pipeline from {model_path}")

            pipe = StableVideoDiffusionPipeline.from_pretrained(
                model_path,
                torch_dtype=torch.float16,
                variant="fp16",
            )
            pipe = pipe.to(device)

            self.model_cache.put(
                cache_key,
                pipe,
                model_type="video",
                device=device,
                memory_mb=16000,
            )

        # Load or generate init image
        if init_image and Path(init_image).exists():
            image = Image.open(init_image).resize((width, height))
        else:
            # Generate init image using image handler
            image_handler = get_image_handler()
            image_result = await image_handler.generate(
                prompt=prompt,
                width=width,
                height=height,
            )
            image = Image.open(image_result["path"])

        # Set up generator
        generator = None
        if seed is not None:
            generator = torch.Generator(device=pipe.device).manual_seed(seed)

        # Generate video
        loop = asyncio.get_event_loop()
        frames = await loop.run_in_executor(
            None,
            lambda: pipe(
                image,
                decode_chunk_size=8,
                generator=generator,
                num_frames=num_frames,
            ).frames[0],
        )

        # Export to video
        output_filename = f"{uuid.uuid4()}.mp4"
        output_path = self.output_dir / output_filename

        await loop.run_in_executor(
            None,
            lambda: export_to_video(frames, str(output_path), fps=fps),
        )

        return {
            "path": str(output_path),
            "width": width,
            "height": height,
            "num_frames": num_frames,
            "fps": fps,
            "model": model_info.get("name"),
            "seed": seed,
        }

    async def _generate_frame_by_frame(
        self,
        prompt: str,
        width: int,
        height: int,
        num_frames: int,
        fps: int,
        seed: Optional[int],
        motion_prompt_suffix: str = ", cinematic motion, smooth animation",
        **kwargs,
    ) -> Dict[str, Any]:
        """Generate video by creating frames with image model."""
        try:
            import cv2
            import numpy as np
        except ImportError:
            raise RuntimeError("OpenCV is required for frame-by-frame generation")

        from PIL import Image

        image_handler = get_image_handler()

        frames = []
        current_seed = seed if seed is not None else 42

        for i in range(num_frames):
            # Vary the prompt slightly for each frame
            frame_prompt = (
                f"{prompt}{motion_prompt_suffix}, frame {i + 1} of {num_frames}"
            )

            # Generate frame
            result = await image_handler.generate(
                prompt=frame_prompt,
                width=width,
                height=height,
                seed=current_seed + i,
            )

            # Load frame
            frame = Image.open(result["path"])
            frame_array = np.array(frame)
            frame_bgr = cv2.cvtColor(frame_array, cv2.COLOR_RGB2BGR)
            frames.append(frame_bgr)

        # Create video
        output_filename = f"{uuid.uuid4()}.mp4"
        output_path = self.output_dir / output_filename

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

        for frame in frames:
            out.write(frame)

        out.release()

        return {
            "path": str(output_path),
            "width": width,
            "height": height,
            "num_frames": num_frames,
            "fps": fps,
            "model": "frame-by-frame",
            "seed": seed,
        }

    async def _generate_runway(
        self,
        prompt: str,
        model_info: Dict[str, Any],
        **kwargs,
    ) -> Dict[str, Any]:
        """Generate video using Runway ML API."""
        payload = {
            "text_prompt": prompt,
        }

        try:
            async with httpx.AsyncClient(timeout=300.0) as client:
                # Start generation
                response = await client.post(
                    "https://api.runwayml.com/v1/generations",
                    json=payload,
                    headers={
                        "Authorization": f"Bearer {self.runway_api_key}",
                        "Content-Type": "application/json",
                    },
                )
                response.raise_for_status()
                data = response.json()
                generation_id = data.get("id")

                # Poll for completion
                while True:
                    await asyncio.sleep(5)
                    status_response = await client.get(
                        f"https://api.runwayml.com/v1/generations/{generation_id}",
                        headers={
                            "Authorization": f"Bearer {self.runway_api_key}",
                        },
                    )
                    status_data = status_response.json()

                    if status_data.get("status") == "completed":
                        video_url = status_data.get("output", {}).get("video_url")
                        if video_url:
                            # Download video
                            video_response = await client.get(video_url)
                            video_data = video_response.content

                            output_filename = f"{uuid.uuid4()}.mp4"
                            output_path = self.output_dir / output_filename
                            with open(output_path, "wb") as f:
                                f.write(video_data)

                            return {
                                "path": str(output_path),
                                "model": "runway:gen3",
                            }

                    elif status_data.get("status") == "failed":
                        raise RuntimeError(
                            f"Runway generation failed: {status_data.get('error')}"
                        )

        except Exception as e:
            raise RuntimeError(f"Runway generation failed: {e}")

    async def get_available_models(self) -> List[Dict[str, Any]]:
        """Get list of available video models."""
        if not self._initialized:
            await self.initialize()

        return [
            {
                "name": name,
                **info,
            }
            for name, info in self.available_models.items()
        ]


# Global video model handler instance
_video_handler: Optional[VideoModelHandler] = None


def get_video_handler() -> VideoModelHandler:
    """Get the global video model handler instance."""
    global _video_handler
    if _video_handler is None:
        _video_handler = VideoModelHandler()
    return _video_handler
