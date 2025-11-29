"""
Image model handlers for image generation.

Handles image generation using various backends:
- Local: Stable Diffusion, SDXL, FLUX via diffusers or ComfyUI
- Cloud: OpenAI DALL-E, Stability AI
"""

import asyncio
import base64
import io
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
from .model_loader import (
    disable_safety_checker,
    filter_scheduler_config,
    is_flux_model,
)

logger = get_logger(__name__)

# Default guidance scale values
DEFAULT_GUIDANCE_SCALE = 7.5
DEFAULT_FLUX_GUIDANCE_SCALE = 3.5

# NSFW model keywords
NSFW_MODEL_KEYWORDS = [
    "nsfw",
    "realistic",
    "photon",
    "dreamshaper",
    "realvis",
    "juggernaut",
    "explicit",
    "adult",
]


class ImageModelHandler(BaseModelHandler):
    """
    Handler for image generation models.

    Supports multiple backends:
    - diffusers (HuggingFace pipelines)
    - ComfyUI (advanced workflow execution)
    - OpenAI DALL-E API
    - Stability AI API
    """

    def __init__(self):
        """Initialize the image model handler."""
        super().__init__(ModelType.IMAGE)
        self.settings = get_settings()
        self.gpu_manager = get_gpu_manager()
        self.model_cache = get_model_cache()

        # Backend availability
        self.diffusers_available = False
        self.comfyui_available = False
        self.comfyui_url: Optional[str] = None
        self.openai_available = False

        # Model paths
        self.model_dir = Path(
            os.environ.get("AI_MODEL_PATH", "/opt/gator/data/models/image")
        )
        self.output_dir = Path(
            os.environ.get("CONTENT_OUTPUT_PATH", "generated_content/images")
        )

    async def initialize(self) -> None:
        """Initialize the image model handler and discover available models."""
        if self._initialized:
            return

        logger.info("Initializing image model handler...")

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
            f"Image model handler initialized with {len(self.available_models)} models"
        )

    async def _detect_backends(self) -> None:
        """Detect available inference backends."""
        # Check diffusers
        try:
            import diffusers  # noqa: F401

            self.diffusers_available = True
            logger.info("✓ diffusers is available")
        except ImportError:
            logger.debug("diffusers not available")

        # Check ComfyUI
        comfyui_urls = [
            os.environ.get("COMFYUI_URL"),
            "http://localhost:8188",
            "http://127.0.0.1:8188",
        ]

        for url in comfyui_urls:
            if not url:
                continue
            try:
                async with httpx.AsyncClient(timeout=5.0) as client:
                    response = await client.get(f"{url}/system_stats")
                    if response.status_code == 200:
                        self.comfyui_available = True
                        self.comfyui_url = url
                        logger.info(f"✓ ComfyUI is available at {url}")
                        break
            except Exception:
                continue

        # Check cloud APIs
        if self.settings.openai_api_key:
            self.openai_available = True
            logger.info("✓ OpenAI API key configured for DALL-E")

    async def _discover_local_models(self) -> None:
        """Discover local image models."""
        if not self.model_dir.exists():
            logger.debug(f"Model directory does not exist: {self.model_dir}")
            return

        # Find safetensors/checkpoint models
        model_files = list(self.model_dir.glob("**/*.safetensors")) + list(
            self.model_dir.glob("**/*.ckpt")
        )

        for model_path in model_files:
            model_name = model_path.stem
            model_info = self._analyze_model(model_name, model_path)
            self.available_models[model_name] = model_info
            logger.debug(f"Found local image model: {model_name}")

        # Find diffusers format models
        for model_dir in self.model_dir.iterdir():
            if model_dir.is_dir() and (model_dir / "model_index.json").exists():
                model_name = model_dir.name
                if model_name not in self.available_models:
                    model_info = self._analyze_model(model_name, model_dir)
                    self.available_models[model_name] = model_info
                    logger.debug(f"Found diffusers model: {model_name}")

    def _analyze_model(self, model_name: str, model_path: Path) -> Dict[str, Any]:
        """Analyze a model and determine its capabilities."""
        model_lower = model_name.lower()

        # Determine if NSFW capable
        is_nsfw = any(kw in model_lower for kw in NSFW_MODEL_KEYWORDS)

        # Determine model type
        model_info = {
            "name": model_name,
            "path": str(model_path),
        }

        if is_flux_model(model_info):
            model_info.update(
                {
                    "model_type": "flux",
                    "source": "local",
                    "backend": "comfyui" if self.comfyui_available else "diffusers",
                    "default_guidance": DEFAULT_FLUX_GUIDANCE_SCALE,
                    "capabilities": ModelCapabilities(
                        supports_nsfw=is_nsfw,
                        max_resolution=(1024, 1024),
                    ).to_dict(),
                }
            )
        elif "sdxl" in model_lower or "xl" in model_lower:
            model_info.update(
                {
                    "model_type": "sdxl",
                    "source": "local",
                    "backend": "diffusers",
                    "default_guidance": DEFAULT_GUIDANCE_SCALE,
                    "capabilities": ModelCapabilities(
                        supports_nsfw=is_nsfw,
                        max_resolution=(1024, 1024),
                    ).to_dict(),
                }
            )
        else:
            model_info.update(
                {
                    "model_type": "sd15",
                    "source": "local",
                    "backend": "diffusers",
                    "default_guidance": DEFAULT_GUIDANCE_SCALE,
                    "capabilities": ModelCapabilities(
                        supports_nsfw=is_nsfw,
                        max_resolution=(512, 512),
                    ).to_dict(),
                }
            )

        return model_info

    async def _register_cloud_models(self) -> None:
        """Register cloud API models."""
        if self.openai_available:
            openai_models = [
                ("dall-e-3", (1024, 1024)),
                ("dall-e-2", (1024, 1024)),
            ]
            for model_name, max_res in openai_models:
                self.available_models[f"openai:{model_name}"] = {
                    "name": model_name,
                    "source": "openai",
                    "backend": "openai",
                    "capabilities": ModelCapabilities(
                        max_resolution=max_res,
                    ).to_dict(),
                }

    async def generate(
        self,
        prompt: str,
        model: Optional[str] = None,
        width: int = 1024,
        height: int = 1024,
        num_inference_steps: int = 30,
        guidance_scale: Optional[float] = None,
        negative_prompt: Optional[str] = None,
        seed: Optional[int] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Generate an image using the selected model.

        Args:
            prompt: Text prompt for image generation
            model: Model name (optional, will select best available)
            width: Image width
            height: Image height
            num_inference_steps: Number of denoising steps
            guidance_scale: Classifier-free guidance scale
            negative_prompt: Negative prompt for things to avoid
            seed: Random seed for reproducibility
            **kwargs: Additional generation parameters

        Returns:
            Dictionary with image path, base64 data, and metadata
        """
        if not self._initialized:
            await self.initialize()

        # Select model if not specified
        if not model:
            model = self._select_best_model(**kwargs)

        if not model:
            raise ValueError("No image model available for generation")

        model_info = self.available_models.get(model)
        if not model_info:
            raise ValueError(f"Model not found: {model}")

        backend = model_info.get("backend", "diffusers")

        # Use model's default guidance if not specified
        if guidance_scale is None:
            guidance_scale = model_info.get("default_guidance", DEFAULT_GUIDANCE_SCALE)

        logger.info(f"Generating image with model '{model}' using {backend}")

        if backend == "comfyui":
            return await self._generate_comfyui(
                prompt=prompt,
                model_info=model_info,
                width=width,
                height=height,
                steps=num_inference_steps,
                guidance_scale=guidance_scale,
                negative_prompt=negative_prompt,
                seed=seed,
                **kwargs,
            )
        elif backend == "diffusers":
            return await self._generate_diffusers(
                prompt=prompt,
                model_info=model_info,
                width=width,
                height=height,
                steps=num_inference_steps,
                guidance_scale=guidance_scale,
                negative_prompt=negative_prompt,
                seed=seed,
                **kwargs,
            )
        elif backend == "openai":
            return await self._generate_openai(
                prompt=prompt,
                model_info=model_info,
                width=width,
                height=height,
                **kwargs,
            )
        else:
            raise ValueError(f"Unsupported backend: {backend}")

    def _select_best_model(
        self, prefer_nsfw: bool = False, prefer_quality: bool = True, **kwargs
    ) -> Optional[str]:
        """Select the best available model for generation."""
        if not self.available_models:
            return None

        # Prefer local models over cloud
        local_models = [
            (name, info)
            for name, info in self.available_models.items()
            if info.get("source") == "local"
        ]

        if prefer_nsfw:
            # Prefer NSFW-capable models
            nsfw_models = [
                (name, info)
                for name, info in local_models
                if info.get("capabilities", {}).get("supports_nsfw", False)
            ]
            if nsfw_models:
                # Prefer SDXL for quality
                sdxl_models = [
                    (name, info)
                    for name, info in nsfw_models
                    if info.get("model_type") == "sdxl"
                ]
                if sdxl_models:
                    return sdxl_models[0][0]
                return nsfw_models[0][0]

        if prefer_quality and local_models:
            # Prefer SDXL models
            sdxl_models = [
                (name, info)
                for name, info in local_models
                if info.get("model_type") == "sdxl"
            ]
            if sdxl_models:
                return sdxl_models[0][0]

        if local_models:
            return local_models[0][0]

        # Fall back to cloud models
        if self.openai_available:
            return "openai:dall-e-3"

        return None

    async def _generate_diffusers(
        self,
        prompt: str,
        model_info: Dict[str, Any],
        width: int,
        height: int,
        steps: int,
        guidance_scale: float,
        negative_prompt: Optional[str],
        seed: Optional[int],
        **kwargs,
    ) -> Dict[str, Any]:
        """Generate image using diffusers library."""
        import torch
        from diffusers import (
            DPMSolverMultistepScheduler,
            StableDiffusionPipeline,
            StableDiffusionXLPipeline,
        )

        model_path = model_info.get("path")
        model_type = model_info.get("model_type", "sd15")

        # Check cache first
        cache_key = f"diffusers:{model_path}"
        pipe = self.model_cache.get(cache_key)

        if pipe is None:
            # Load the pipeline
            device = self.gpu_manager.select_device_for_model(
                cache_key, required_memory_gb=8.0
            )

            logger.info(f"Loading {model_type} pipeline from {model_path}")

            if model_type == "sdxl":
                pipe = StableDiffusionXLPipeline.from_single_file(
                    model_path,
                    torch_dtype=torch.float16,
                    use_safetensors=True,
                )
            else:
                pipe = StableDiffusionPipeline.from_single_file(
                    model_path,
                    torch_dtype=torch.float16,
                    use_safetensors=True,
                )

            # Configure scheduler
            scheduler_config = filter_scheduler_config(pipe.scheduler.config)
            pipe.scheduler = DPMSolverMultistepScheduler.from_config(scheduler_config)

            # Disable safety checker for NSFW
            disable_safety_checker(pipe)

            # Move to device
            pipe = pipe.to(device)

            # Cache the pipeline
            self.model_cache.put(
                cache_key,
                pipe,
                model_type="image",
                device=device,
                memory_mb=8000,
            )

        # Set up generator for reproducibility
        generator = None
        if seed is not None:
            generator = torch.Generator(device=pipe.device).manual_seed(seed)

        # Generate image
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            lambda: pipe(
                prompt=prompt,
                negative_prompt=negative_prompt or "",
                width=width,
                height=height,
                num_inference_steps=steps,
                guidance_scale=guidance_scale,
                generator=generator,
            ),
        )

        image = result.images[0]

        # Save image
        output_filename = f"{uuid.uuid4()}.png"
        output_path = self.output_dir / output_filename
        image.save(output_path)

        # Convert to base64
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        base64_data = base64.b64encode(buffer.getvalue()).decode("utf-8")

        return {
            "path": str(output_path),
            "base64": base64_data,
            "width": width,
            "height": height,
            "model": model_info.get("name"),
            "seed": seed,
        }

    async def _generate_comfyui(
        self,
        prompt: str,
        model_info: Dict[str, Any],
        width: int,
        height: int,
        steps: int,
        guidance_scale: float,
        negative_prompt: Optional[str],
        seed: Optional[int],
        **kwargs,
    ) -> Dict[str, Any]:
        """Generate image using ComfyUI."""
        if not self.comfyui_url:
            raise RuntimeError("ComfyUI not available")

        # Build workflow
        workflow = self._build_comfyui_workflow(
            prompt=prompt,
            model_name=model_info.get("name"),
            width=width,
            height=height,
            steps=steps,
            guidance_scale=guidance_scale,
            negative_prompt=negative_prompt,
            seed=seed or 0,
        )

        # Queue prompt
        async with httpx.AsyncClient(timeout=300.0) as client:
            # Queue the workflow
            response = await client.post(
                f"{self.comfyui_url}/prompt",
                json={"prompt": workflow},
            )
            response.raise_for_status()
            queue_response = response.json()
            prompt_id = queue_response.get("prompt_id")

            # Wait for completion
            while True:
                await asyncio.sleep(1)
                history_response = await client.get(
                    f"{self.comfyui_url}/history/{prompt_id}"
                )
                history = history_response.json()

                if prompt_id in history:
                    outputs = history[prompt_id].get("outputs", {})
                    if outputs:
                        # Find image output
                        for node_id, node_output in outputs.items():
                            if "images" in node_output:
                                image_info = node_output["images"][0]
                                # Get image
                                img_response = await client.get(
                                    f"{self.comfyui_url}/view",
                                    params={
                                        "filename": image_info["filename"],
                                        "subfolder": image_info.get("subfolder", ""),
                                        "type": image_info.get("type", "output"),
                                    },
                                )
                                image_data = img_response.content

                                # Save locally
                                output_filename = f"{uuid.uuid4()}.png"
                                output_path = self.output_dir / output_filename
                                with open(output_path, "wb") as f:
                                    f.write(image_data)

                                return {
                                    "path": str(output_path),
                                    "base64": base64.b64encode(image_data).decode(
                                        "utf-8"
                                    ),
                                    "width": width,
                                    "height": height,
                                    "model": model_info.get("name"),
                                    "seed": seed,
                                }

                        raise RuntimeError("No image output found in ComfyUI response")

    def _build_comfyui_workflow(
        self,
        prompt: str,
        model_name: str,
        width: int,
        height: int,
        steps: int,
        guidance_scale: float,
        negative_prompt: Optional[str],
        seed: int,
    ) -> Dict[str, Any]:
        """Build a basic ComfyUI workflow."""
        return {
            "1": {
                "class_type": "CheckpointLoaderSimple",
                "inputs": {"ckpt_name": model_name},
            },
            "2": {
                "class_type": "CLIPTextEncode",
                "inputs": {
                    "text": prompt,
                    "clip": ["1", 1],
                },
            },
            "3": {
                "class_type": "CLIPTextEncode",
                "inputs": {
                    "text": negative_prompt or "",
                    "clip": ["1", 1],
                },
            },
            "4": {
                "class_type": "EmptyLatentImage",
                "inputs": {
                    "width": width,
                    "height": height,
                    "batch_size": 1,
                },
            },
            "5": {
                "class_type": "KSampler",
                "inputs": {
                    "model": ["1", 0],
                    "positive": ["2", 0],
                    "negative": ["3", 0],
                    "latent_image": ["4", 0],
                    "seed": seed,
                    "steps": steps,
                    "cfg": guidance_scale,
                    "sampler_name": "euler",
                    "scheduler": "normal",
                    "denoise": 1.0,
                },
            },
            "6": {
                "class_type": "VAEDecode",
                "inputs": {
                    "samples": ["5", 0],
                    "vae": ["1", 2],
                },
            },
            "7": {
                "class_type": "SaveImage",
                "inputs": {
                    "images": ["6", 0],
                    "filename_prefix": "gator",
                },
            },
        }

    async def _generate_openai(
        self,
        prompt: str,
        model_info: Dict[str, Any],
        width: int,
        height: int,
        **kwargs,
    ) -> Dict[str, Any]:
        """Generate image using OpenAI DALL-E API."""
        model_name = model_info.get("name", "dall-e-3")

        # Determine size
        if model_name == "dall-e-3":
            if width >= 1792 or height >= 1792:
                size = "1792x1024" if width > height else "1024x1792"
            else:
                size = "1024x1024"
        else:
            size = "1024x1024" if width >= 1024 else "512x512"

        payload = {
            "model": model_name,
            "prompt": prompt,
            "n": 1,
            "size": size,
            "response_format": "b64_json",
        }

        if model_name == "dall-e-3":
            payload["quality"] = kwargs.get("quality", "standard")

        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                response = await client.post(
                    "https://api.openai.com/v1/images/generations",
                    json=payload,
                    headers={
                        "Authorization": f"Bearer {self.settings.openai_api_key}",
                        "Content-Type": "application/json",
                    },
                )
                response.raise_for_status()
                data = response.json()

                image_data = base64.b64decode(data["data"][0]["b64_json"])

                # Save locally
                output_filename = f"{uuid.uuid4()}.png"
                output_path = self.output_dir / output_filename
                with open(output_path, "wb") as f:
                    f.write(image_data)

                # Parse size for response
                w, h = map(int, size.split("x"))

                return {
                    "path": str(output_path),
                    "base64": data["data"][0]["b64_json"],
                    "width": w,
                    "height": h,
                    "model": model_name,
                    "revised_prompt": data["data"][0].get("revised_prompt"),
                }

        except Exception as e:
            raise RuntimeError(f"OpenAI image generation failed: {e}")

    async def get_available_models(self) -> List[Dict[str, Any]]:
        """Get list of available image models."""
        if not self._initialized:
            await self.initialize()

        return [
            {
                "name": name,
                **info,
            }
            for name, info in self.available_models.items()
        ]


# Global image model handler instance
_image_handler: Optional[ImageModelHandler] = None


def get_image_handler() -> ImageModelHandler:
    """Get the global image model handler instance."""
    global _image_handler
    if _image_handler is None:
        _image_handler = ImageModelHandler()
    return _image_handler
