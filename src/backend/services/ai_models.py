"""
AI Model Integration Service

Handles integration with various AI models for content generation including:
- Local models: Llama 3.1, Qwen2.5, Stable Diffusion XL, Coqui XTTS-v2
- Cloud APIs: OpenAI, Anthropic, ElevenLabs
- Hardware optimization for AMD ROCm and CUDA
"""

import asyncio
import os
import io
import base64
import subprocess
import platform
import shutil
import time
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
from pathlib import Path
import json

import httpx
from PIL import Image
import torch

from backend.config.logging import get_logger
from backend.config.settings import get_settings
from backend.utils.model_detection import (
    find_comfyui_installation,
    check_inference_engine_available,
)

logger = get_logger(__name__)


class AIModelManager:
    """
    Manager for AI model integrations.

    Supports both local models (Llama, SDXL, XTTS) and cloud APIs (OpenAI, Anthropic).
    Automatically detects hardware capabilities and optimizes model selection.
    """

    def __init__(self):
        self.settings = get_settings()
        self.models_loaded = False
        self.available_models = {
            "text": [],
            "image": [],
            "voice": [],
            "video": [],
            "audio": [],
        }
        self.http_client = httpx.AsyncClient(
            timeout=300.0
        )  # Long timeout for AI generation

        # Hardware detection
        self.gpu_type = self._detect_gpu_type()
        self.gpu_memory_gb = self._get_gpu_memory()
        self.cpu_cores = os.cpu_count()

        # Local model paths
        self.models_dir = Path("./models")
        self.models_dir.mkdir(exist_ok=True)

        # Cache for loaded diffusion pipelines
        self._loaded_pipelines = {}
        
        # Lazy loading configuration
        # Models marked as lazy will only be loaded when first requested
        self.lazy_load_enabled = os.environ.get("AI_MODELS_LAZY_LOAD", "false").lower() == "true"
        self.lazy_load_models = set()  # Models configured for lazy loading
        
        # Configure which models should use lazy loading
        # Typically large models or those used infrequently
        if self.lazy_load_enabled:
            self.lazy_load_models.update([
                "llama-3.1-70b",  # 140GB - very large
                "qwen2.5-72b",    # 144GB - very large
                "flux.1-dev",     # 12GB - less frequently used
            ])

        # Model configurations based on recommendations
        self.local_model_configs = {
            "text": {
                "llama-3.1-70b": {
                    "model_id": "meta-llama/Llama-3.1-70B",
                    "size_gb": 140,
                    "min_gpu_memory_gb": 48,
                    "min_ram_gb": 64,
                    "inference_engine": "vllm",
                    "quant_options": ["Q4_K_M", "FP16", "BF16"],
                    "description": "Best general local base model",
                },
                "qwen2.5-72b": {
                    "model_id": "Qwen/Qwen2.5-72B",
                    "size_gb": 144,
                    "min_gpu_memory_gb": 48,
                    "min_ram_gb": 64,
                    "inference_engine": "vllm",
                    "description": "Stronger tools/code, longer context",
                },
                "mixtral-8x7b": {
                    "model_id": "mistralai/Mixtral-8x7B-Instruct-v0.1",
                    "size_gb": 90,
                    "min_gpu_memory_gb": 24,
                    "min_ram_gb": 32,
                    "inference_engine": "vllm",
                    "description": "Fast per token, solid instruction following at lower VRAM",
                },
                "llama-3.1-8b": {
                    "model_id": "meta-llama/Llama-3.1-8B-Instruct",
                    "size_gb": 16,
                    "min_gpu_memory_gb": 8,
                    "min_ram_gb": 16,
                    "inference_engine": "vllm",
                    "description": "Snappy persona worker for fast mode",
                },
            },
            "image": {
                "stable-diffusion-v1-5": {
                    "model_id": "runwayml/stable-diffusion-v1-5",
                    "size_gb": 4,
                    "min_gpu_memory_gb": 4,
                    "min_ram_gb": 8,
                    "inference_engine": "diffusers",
                    "description": "Fast and efficient base model for image generation",
                },
                "sdxl-1.0": {
                    "model_id": "stabilityai/stable-diffusion-xl-base-1.0",
                    "size_gb": 7,
                    "min_gpu_memory_gb": 8,
                    "min_ram_gb": 16,
                    "inference_engine": "diffusers",
                    "description": "High quality SDXL model, more VRAM required",
                },
                "flux.1-dev": {
                    "model_id": "black-forest-labs/FLUX.1-dev",
                    "size_gb": 12,
                    "min_gpu_memory_gb": 12,
                    "min_ram_gb": 24,
                    "inference_engine": "comfyui",
                    "description": "Very good quality; verify license for commercial use",
                },
            },
            "voice": {
                "xtts-v2": {
                    "model_id": "coqui/XTTS-v2",
                    "size_gb": 2,
                    "min_gpu_memory_gb": 4,
                    "min_ram_gb": 8,
                    "description": "Multilingual, cloning, runs locally; best all-around",
                },
                "piper": {
                    "model_id": "rhasspy/piper",
                    "size_gb": 0.1,
                    "min_gpu_memory_gb": 0,
                    "min_ram_gb": 2,
                    "description": "Ultralight, CPU-friendly for systems TTS",
                },
            },
        }

    def _detect_gpu_type(self) -> str:
        """Detect GPU type (CUDA, ROCm, or CPU)."""
        try:
            if torch.cuda.is_available():
                # Check for AMD ROCm
                try:
                    import subprocess

                    result = subprocess.run(
                        ["rocm-smi", "--showproduct"],
                        capture_output=True,
                        text=True,
                        timeout=5,
                    )
                    if result.returncode == 0 and "MI" in result.stdout:
                        return "rocm"
                except (subprocess.TimeoutExpired, FileNotFoundError):
                    pass

                # Default to CUDA
                return "cuda"
        except Exception:
            pass
        return "cpu"

    def _get_gpu_memory(self) -> float:
        """Get total GPU memory in GB."""
        if not torch.cuda.is_available():
            return 0.0

        total_memory = 0
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            total_memory += props.total_memory

        return total_memory / (1024**3)  # Convert to GB

    def _get_system_requirements(self) -> Dict[str, Any]:
        """Get current system capabilities."""
        import psutil

        return {
            "gpu_type": self.gpu_type,
            "gpu_memory_gb": self.gpu_memory_gb,
            "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
            "ram_gb": psutil.virtual_memory().total / (1024**3),
            "cpu_cores": self.cpu_cores,
            "disk_space_gb": shutil.disk_usage(self.models_dir).free / (1024**3),
            "platform": platform.platform(),
        }

    def _should_lazy_load(self, model_name: str) -> bool:
        """
        Determine if a model should be lazy loaded.
        
        Args:
            model_name: Name of the model to check
            
        Returns:
            True if model should be lazy loaded, False otherwise
        """
        if not self.lazy_load_enabled:
            return False
        
        return model_name in self.lazy_load_models
    
    async def _lazy_load_model(self, model_name: str, category: str) -> bool:
        """
        Load a specific model on-demand.
        
        Args:
            model_name: Name of the model to load
            category: Category of the model (text, image, voice, video)
            
        Returns:
            True if model loaded successfully, False otherwise
        """
        try:
            logger.info(f"Lazy loading model: {model_name} (category: {category})")
            
            # Find the model in available models
            category_models = self.available_models.get(category, [])
            model_index = None
            for i, model in enumerate(category_models):
                if model.get("name") == model_name:
                    model_index = i
                    break
            
            if model_index is None:
                logger.warning(f"Model {model_name} not found in {category} models")
                return False
            
            # Load the model based on its inference engine
            # This would trigger actual model loading logic
            # For now, we mark it as loaded
            self.available_models[category][model_index]["loaded"] = True
            logger.info(f"Successfully lazy loaded model: {model_name}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to lazy load model {model_name}: {e}")
            return False

    async def initialize_models(self) -> None:
        """
        Initialize and load AI models based on available hardware and configuration.
        
        With lazy loading enabled (AI_MODELS_LAZY_LOAD=true), large or infrequently
        used models will be marked as available but not loaded until first use,
        reducing startup time and memory usage.
        """
        try:
            sys_req = self._get_system_requirements()
            logger.info(f"Initializing AI models with system: {sys_req}")
            
            if self.lazy_load_enabled:
                logger.info(
                    f"Lazy loading enabled for models: {', '.join(self.lazy_load_models)}"
                )

            # Initialize local models based on hardware
            await self._initialize_local_text_models()
            await self._initialize_local_image_models()
            await self._initialize_local_voice_models()

            # Initialize cloud API models as fallbacks
            await self._initialize_cloud_text_models()
            await self._initialize_cloud_image_models()
            await self._initialize_cloud_voice_models()

            # Initialize video models (still mostly placeholder)
            await self._initialize_video_models()

            self.models_loaded = True
            logger.info(
                f"AI model initialization complete",
                extra={"available_models": self.available_models},
            )

        except Exception as e:
            logger.error(f"Failed to initialize AI models: {str(e)}")
            raise

    async def _initialize_local_text_models(self) -> None:
        """Initialize local text generation models based on hardware capabilities."""
        try:
            sys_req = self._get_system_requirements()

            # Select appropriate models based on available resources
            for model_name, config in self.local_model_configs["text"].items():
                can_run = sys_req["gpu_memory_gb"] >= config.get(
                    "min_gpu_memory_gb", 0
                ) and sys_req["ram_gb"] >= config.get("min_ram_gb", 0)

                if can_run:
                    # Check both model path formats for compatibility
                    # 1. Category subdirectory: ./models/text/model-name/
                    # 2. Direct: ./models/model-name/
                    model_path_with_category = self.models_dir / "text" / model_name
                    model_path_direct = self.models_dir / model_name

                    # Prefer category subdirectory if it exists, fallback to direct
                    if model_path_with_category.exists():
                        model_path = model_path_with_category
                        is_downloaded = True
                    elif model_path_direct.exists():
                        model_path = model_path_direct
                        is_downloaded = True
                    else:
                        model_path = (
                            model_path_with_category  # Default for future downloads
                        )
                        is_downloaded = False

                    inference_engine = config.get("inference_engine", "transformers")

                    # Check if inference engine is available
                    engine_available = await self._check_inference_engine(
                        inference_engine
                    )
                    
                    # Check if this model should be lazy loaded
                    should_lazy_load = self._should_lazy_load(model_name)
                    
                    # If lazy loading, mark as not loaded initially even if available
                    if should_lazy_load and is_downloaded and engine_available:
                        is_actually_loaded = False
                        logger.info(
                            f"Local text model {model_name} configured for lazy loading"
                        )
                    else:
                        is_actually_loaded = is_downloaded and engine_available

                    self.available_models["text"].append(
                        {
                            "name": model_name,
                            "type": "text-generation",
                            "model_id": config["model_id"],
                            "provider": "local",
                            "inference_engine": inference_engine,
                            "loaded": is_actually_loaded,
                            "can_load": can_run and engine_available,
                            "lazy_load": should_lazy_load,
                            "size_gb": config["size_gb"],
                            "description": config["description"],
                            "device": "cuda" if sys_req["gpu_memory_gb"] > 0 else "cpu",
                            "quant_options": config.get("quant_options", []),
                            "path": str(model_path),
                        }
                    )

                    if should_lazy_load:
                        logger.info(
                            f"Local text model {model_name} will be lazy loaded on first use"
                        )
                    elif is_downloaded and engine_available:
                        logger.info(
                            f"Local text model {model_name} ready at {model_path}"
                        )
                    elif is_downloaded and not engine_available:
                        logger.warning(
                            f"Local text model {model_name} found at {model_path} but inference engine {inference_engine} not available"
                        )
                    elif can_run and engine_available:
                        logger.info(f"Local text model {model_name} can be downloaded")
                    else:
                        logger.info(
                            f"Local text model {model_name} needs setup: engine={engine_available}, downloaded={is_downloaded}"
                        )

        except Exception as e:
            logger.error(f"Failed to initialize local text models: {str(e)}")

    async def _initialize_local_image_models(self) -> None:
        """Initialize local image generation models."""
        try:
            sys_req = self._get_system_requirements()

            for model_name, config in self.local_model_configs["image"].items():
                can_run = sys_req["gpu_memory_gb"] >= config.get(
                    "min_gpu_memory_gb", 0
                ) and sys_req["ram_gb"] >= config.get("min_ram_gb", 0)

                if can_run:
                    # Check both model path formats for compatibility
                    # 1. Category subdirectory: ./models/image/model-name/
                    # 2. Direct: ./models/model-name/
                    model_path_with_category = self.models_dir / "image" / model_name
                    model_path_direct = self.models_dir / model_name

                    # Prefer category subdirectory if it exists, fallback to direct
                    if model_path_with_category.exists():
                        model_path = model_path_with_category
                        is_downloaded = True
                    elif model_path_direct.exists():
                        model_path = model_path_direct
                        is_downloaded = True
                    else:
                        model_path = (
                            model_path_with_category  # Default for future downloads
                        )
                        is_downloaded = False

                    inference_engine = config.get("inference_engine", "diffusers")
                    engine_available = await self._check_inference_engine(
                        inference_engine
                    )

                    self.available_models["image"].append(
                        {
                            "name": model_name,
                            "type": "text-to-image",
                            "model_id": config["model_id"],
                            "provider": "local",
                            "inference_engine": inference_engine,
                            "loaded": is_downloaded and engine_available,
                            "can_load": can_run and engine_available,
                            "size_gb": config["size_gb"],
                            "description": config["description"],
                            "device": "cuda" if sys_req["gpu_memory_gb"] > 0 else "cpu",
                            "path": str(model_path),
                        }
                    )

                    if is_downloaded and engine_available:
                        logger.info(
                            f"Local image model {model_name} ready at {model_path}"
                        )
                    elif is_downloaded and not engine_available:
                        logger.warning(
                            f"Local image model {model_name} found at {model_path} but inference engine {inference_engine} not available"
                        )

        except Exception as e:
            logger.error(f"Failed to initialize local image models: {str(e)}")

    async def _initialize_local_voice_models(self) -> None:
        """Initialize local voice synthesis models."""
        try:
            sys_req = self._get_system_requirements()

            for model_name, config in self.local_model_configs["voice"].items():
                can_run = sys_req["ram_gb"] >= config.get("min_ram_gb", 0)

                if can_run:
                    # Check both model path formats for compatibility
                    # 1. Category subdirectory: ./models/voice/model-name/
                    # 2. Direct: ./models/model-name/
                    model_path_with_category = self.models_dir / "voice" / model_name
                    model_path_direct = self.models_dir / model_name

                    # Prefer category subdirectory if it exists, fallback to direct
                    if model_path_with_category.exists():
                        model_path = model_path_with_category
                        is_downloaded = True
                    elif model_path_direct.exists():
                        model_path = model_path_direct
                        is_downloaded = True
                    else:
                        model_path = (
                            model_path_with_category  # Default for future downloads
                        )
                        is_downloaded = False

                    self.available_models["voice"].append(
                        {
                            "name": model_name,
                            "type": "text-to-speech",
                            "model_id": config["model_id"],
                            "provider": "local",
                            "loaded": is_downloaded,
                            "can_load": can_run,
                            "size_gb": config["size_gb"],
                            "description": config["description"],
                            "device": (
                                "cuda"
                                if config.get("min_gpu_memory_gb", 0)
                                <= sys_req["gpu_memory_gb"]
                                else "cpu"
                            ),
                            "path": str(model_path),
                        }
                    )

                    if is_downloaded:
                        logger.info(
                            f"Local voice model {model_name} ready at {model_path}"
                        )

        except Exception as e:
            logger.error(f"Failed to initialize local voice models: {str(e)}")

    async def _check_inference_engine(self, engine: str) -> bool:
        """Check if inference engine is available."""
        # Use comprehensive detection logic from model_detection utility
        return check_inference_engine_available(engine, base_dir=self.models_dir.parent)

    async def _initialize_cloud_text_models(self) -> None:
        """Initialize cloud-based text generation models."""
        try:
            # Add OpenAI GPT as API option
            if (
                hasattr(self.settings, "openai_api_key")
                and self.settings.openai_api_key
            ):
                self.available_models["text"].extend(
                    [
                        {
                            "name": "gpt-4",
                            "type": "text-generation",
                            "provider": "openai",
                            "loaded": True,
                            "max_tokens": 4096,
                            "description": "OpenAI GPT-4 cloud API",
                        },
                        {
                            "name": "gpt-3.5-turbo",
                            "type": "text-generation",
                            "provider": "openai",
                            "loaded": True,
                            "max_tokens": 4096,
                            "description": "OpenAI GPT-3.5 Turbo cloud API",
                        },
                    ]
                )
                logger.info("OpenAI text models available")

            # Add Anthropic Claude as option
            if (
                hasattr(self.settings, "anthropic_api_key")
                and self.settings.anthropic_api_key
            ):
                self.available_models["text"].append(
                    {
                        "name": "claude-3-sonnet",
                        "type": "text-generation",
                        "provider": "anthropic",
                        "loaded": True,
                        "max_tokens": 4096,
                        "description": "Anthropic Claude-3 Sonnet cloud API",
                    }
                )
                logger.info("Anthropic Claude available")

        except Exception as e:
            logger.error(f"Failed to initialize cloud text models: {str(e)}")

    async def _initialize_cloud_image_models(self) -> None:
        """Initialize cloud-based image generation models."""
        try:
            # Add OpenAI DALL-E as fallback if API key available
            if (
                hasattr(self.settings, "openai_api_key")
                and self.settings.openai_api_key
            ):
                self.available_models["image"].append(
                    {
                        "name": "dall-e-3",
                        "type": "text-to-image",
                        "provider": "openai",
                        "loaded": True,
                        "description": "OpenAI DALL-E 3 cloud API",
                    }
                )
                logger.info("OpenAI DALL-E 3 API available")

        except Exception as e:
            logger.error(f"Failed to initialize cloud image models: {str(e)}")

    async def _initialize_cloud_voice_models(self) -> None:
        """Initialize cloud-based voice synthesis models."""
        try:
            # Check for ElevenLabs API
            if (
                hasattr(self.settings, "elevenlabs_api_key")
                and self.settings.elevenlabs_api_key
            ):
                self.available_models["voice"].append(
                    {
                        "name": "elevenlabs-tts",
                        "type": "text-to-speech",
                        "provider": "elevenlabs",
                        "loaded": True,
                        "description": "ElevenLabs voice synthesis cloud API",
                    }
                )
                logger.info("ElevenLabs voice synthesis available")

            # Check for OpenAI TTS
            if (
                hasattr(self.settings, "openai_api_key")
                and self.settings.openai_api_key
            ):
                self.available_models["voice"].append(
                    {
                        "name": "openai-tts",
                        "type": "text-to-speech",
                        "provider": "openai",
                        "loaded": True,
                        "description": "OpenAI TTS cloud API",
                    }
                )
                logger.info("OpenAI TTS available")

        except Exception as e:
            logger.error(f"Failed to initialize cloud voice models: {str(e)}")

    async def _initialize_video_models(self) -> None:
        """Initialize video generation models."""
        try:
            # Add Stable Video Diffusion configuration (local model)
            self.available_models["video"].append(
                {
                    "name": "stable-video-diffusion",
                    "type": "image-to-video",
                    "model_id": "stabilityai/stable-video-diffusion-img2vid-xt",
                    "provider": "local",
                    "loaded": False,
                    "can_load": True,
                    "min_vram_gb": 24,
                    "description": "Stable Video Diffusion - Image to Video generation",
                    "features": ["image2video", "frame_interpolation"],
                    "max_duration": 4.0,  # seconds
                    "resolution": "576x1024",
                    "timeline": "Q1-Q2 2025",
                }
            )

            # Add advanced frame-by-frame video generation (using video processing service)
            self.available_models["video"].append(
                {
                    "name": "frame-by-frame-generator",
                    "type": "multi-frame-video",
                    "provider": "local",
                    "loaded": True,  # Always available through video processing service
                    "can_load": True,
                    "description": "Frame-by-frame video generation with transitions",
                    "features": [
                        "multi_scene",
                        "transitions",
                        "audio_sync",
                        "storyboarding",
                    ],
                    "supported_transitions": [
                        "fade",
                        "crossfade",
                        "wipe",
                        "slide",
                        "zoom",
                        "dissolve",
                    ],
                    "timeline": "Q2-Q3 2025",
                }
            )

            # Add Runway ML configuration (cloud API)
            self.available_models["video"].append(
                {
                    "name": "runway-gen2",
                    "type": "text-to-video",
                    "provider": "runway",
                    "loaded": bool(os.environ.get("RUNWAY_API_KEY")),
                    "can_load": bool(os.environ.get("RUNWAY_API_KEY")),
                    "description": "Runway Gen-2 - Cloud-based video generation",
                    "features": [
                        "text2video",
                        "image2video",
                        "4k_output",
                        "custom_training",
                    ],
                    "max_duration": 18.0,  # seconds
                    "api_required": True,
                    "timeline": "Q1 2025",
                }
            )

            logger.info(
                f"Initialized {len(self.available_models['video'])} video models"
            )

        except Exception as e:
            logger.error(f"Failed to initialize video models: {str(e)}")

    async def _select_optimal_model(
        self,
        prompt: str,
        content_type: str,
        available_models: List[Dict[str, Any]],
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Intelligently select the optimal model for the given content request.

        Uses a lightweight analysis to match content requirements with model capabilities.
        With 60GB VRAM available, we can be smart about model selection.

        Args:
            prompt: The content generation prompt
            content_type: Type of content (image, text, video, etc.)
            available_models: List of available models to choose from
            **kwargs: Additional parameters like quality, style, etc.

        Returns:
            Selected model dictionary
        """
        if not available_models:
            raise ValueError(f"No {content_type} generation models available")

        # If only one model available, use it
        if len(available_models) == 1:
            return available_models[0]

        # Simple heuristic-based selection (can be enhanced with ML later)
        quality = kwargs.get("quality", "standard")

        # For image models
        if content_type == "image":
            # Keywords that indicate need for high quality
            high_quality_keywords = [
                "detailed",
                "professional",
                "portrait",
                "high quality",
                "photorealistic",
                "8k",
                "4k",
                "masterpiece",
            ]

            # Keywords that indicate speed is acceptable
            speed_keywords = ["quick", "draft", "simple", "sketch", "concept"]

            prompt_lower = prompt.lower()
            needs_quality = any(
                kw in prompt_lower for kw in high_quality_keywords
            ) or quality in ["high", "hd", "premium"]
            needs_speed = (
                any(kw in prompt_lower for kw in speed_keywords) or quality == "draft"
            )

            # Model preference based on needs
            # SDXL models: Best quality but slower
            # SD 1.5: Faster, good quality
            # Flux: Highest quality, needs most resources

            if needs_quality:
                # Prefer SDXL or Flux for quality
                for model in available_models:
                    if "xl" in model["name"].lower() or "flux" in model["name"].lower():
                        logger.info(
                            f"Selected {model['name']} for high-quality generation"
                        )
                        return model

            if needs_speed:
                # Prefer SD 1.5 for speed
                for model in available_models:
                    if "v1-5" in model["name"] or "1.5" in model["name"]:
                        logger.info(f"Selected {model['name']} for fast generation")
                        return model

            # Default: prefer local models by size (larger = better quality typically)
            local_models = [m for m in available_models if m.get("provider") == "local"]
            if local_models:
                # Sort by size_gb descending for quality, or ascending for speed
                if needs_speed:
                    sorted_models = sorted(
                        local_models, key=lambda x: x.get("size_gb", 0)
                    )
                else:
                    sorted_models = sorted(
                        local_models, key=lambda x: x.get("size_gb", 0), reverse=True
                    )
                logger.info(
                    f"Selected {sorted_models[0]['name']} based on size optimization"
                )
                return sorted_models[0]

        # For text models
        elif content_type == "text":
            # Longer prompts or complex tasks need larger models
            prompt_length = len(prompt.split())
            max_tokens = kwargs.get("max_tokens", 500)

            complexity_keywords = [
                "analyze",
                "explain",
                "detailed",
                "comprehensive",
                "essay",
            ]
            needs_large_model = (
                prompt_length > 100
                or max_tokens > 800
                or any(kw in prompt.lower() for kw in complexity_keywords)
            )

            if needs_large_model:
                # Prefer 70B+ models
                for model in available_models:
                    if "70b" in model["name"].lower() or "72b" in model["name"].lower():
                        logger.info(
                            f"Selected {model['name']} for complex text generation"
                        )
                        return model
            else:
                # Prefer smaller, faster models for simple tasks
                for model in available_models:
                    if "8b" in model["name"].lower():
                        logger.info(
                            f"Selected {model['name']} for fast text generation"
                        )
                        return model

        # Default: return first available model
        logger.info(f"Using default model selection: {available_models[0]['name']}")
        return available_models[0]

    async def generate_image(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """Generate image from text prompt using intelligently selected optimal model."""
        start_time = time.time()
        benchmark_data = None
        model = None
        had_errors = False
        error_message = None

        try:
            # Find best available image model (prefer local first)
            local_models = [
                m
                for m in self.available_models["image"]
                if m.get("provider") == "local" and m.get("loaded", False)
            ]
            cloud_models = [
                m
                for m in self.available_models["image"]
                if m.get("provider") in ["openai"] and m.get("loaded", False)
            ]

            # Prefer local models, fallback to cloud
            available_models = local_models + cloud_models

            if not available_models:
                raise ValueError("No image generation models available")

            # Use intelligent model selection
            model = await self._select_optimal_model(
                prompt=prompt,
                content_type="image",
                available_models=available_models,
                **kwargs,
            )

            # Record model selection reasoning for benchmark
            selection_reasoning = (
                model.get("selection_reason")
                or f"Selected based on quality={kwargs.get('quality', 'standard')}"
            )

            # Perform generation
            generation_start = time.time()
            if model.get("provider") == "openai":
                result = await self._generate_image_openai(prompt, **kwargs)
            elif model.get("provider") == "local":
                result = await self._generate_image_local(prompt, model, **kwargs)
            else:
                raise ValueError(f"Unsupported image model: {model['name']}")

            generation_time = time.time() - generation_start
            total_time = time.time() - start_time

            # Add timing info to result
            result["generation_time_seconds"] = generation_time
            result["total_time_seconds"] = total_time
            result["benchmark_data"] = {
                "model_selected": model["name"],
                "model_provider": model.get("provider", "unknown"),
                "selection_reasoning": selection_reasoning,
                "available_models": [m["name"] for m in available_models],
            }

            return result

        except Exception as e:
            had_errors = True
            error_message = str(e)
            logger.error(f"Image generation failed: {str(e)}")
            raise

    async def generate_images_batch(
        self, prompts: List[str], **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Generate multiple images from text prompts using available GPUs in parallel.

        Distributes batch requests across multiple detected GPU devices for improved performance.
        Uses parallel processing to leverage multi-card hardware (e.g., ROCm/MI25 setup).

        Args:
            prompts: List of text prompts for image generation
            **kwargs: Additional generation parameters

        Returns:
            List[Dict[str, Any]]: List of generated image results
        """
        try:
            if not prompts:
                return []

            # Find best available local image model
            local_models = [
                m
                for m in self.available_models["image"]
                if m.get("provider") == "local" and m.get("loaded", False)
            ]

            if not local_models:
                # Fallback to sequential generation with cloud models
                logger.warning(
                    "No local models available, falling back to sequential generation"
                )
                results = []
                for prompt in prompts:
                    result = await self.generate_image(prompt, **kwargs)
                    results.append(result)
                return results

            model = local_models[0]

            # Detect available GPU devices
            gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 0

            if gpu_count <= 1:
                # Single GPU or CPU - sequential processing
                logger.info(
                    f"Using single device for batch generation ({gpu_count} GPU detected)"
                )
                results = []
                for prompt in prompts:
                    result = await self._generate_image_local(prompt, model, **kwargs)
                    results.append(result)
                return results

            # Multi-GPU parallel processing
            logger.info(f"Using {gpu_count} GPUs for parallel batch generation")

            # Distribute prompts across available GPUs
            tasks = []
            for i, prompt in enumerate(prompts):
                gpu_id = i % gpu_count
                task = self._generate_image_on_device(prompt, model, gpu_id, **kwargs)
                tasks.append(task)

            # Execute all tasks in parallel
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Process results and handle any exceptions
            processed_results = []
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.error(
                        f"Batch generation failed for prompt {i}: {str(result)}"
                    )
                    processed_results.append(
                        {"error": str(result), "prompt": prompts[i], "status": "failed"}
                    )
                else:
                    processed_results.append(result)

            return processed_results

        except Exception as e:
            logger.error(f"Batch image generation failed: {str(e)}")
            raise

    async def _generate_image_on_device(
        self, prompt: str, model: Dict[str, Any], device_id: int, **kwargs
    ) -> Dict[str, Any]:
        """
        Generate image on a specific GPU device.

        Args:
            prompt: Text prompt for generation
            model: Model configuration
            device_id: GPU device ID to use
            **kwargs: Additional generation parameters

        Returns:
            Dict[str, Any]: Generated image result
        """
        try:
            # Override device in kwargs
            kwargs_with_device = kwargs.copy()
            kwargs_with_device["device_id"] = device_id

            logger.debug(f"Generating image on GPU {device_id}: {prompt[:50]}...")

            # Generate image using specified device
            result = await self._generate_image_local(
                prompt, model, **kwargs_with_device
            )

            # Add device info to result
            result["device_id"] = device_id

            return result

        except Exception as e:
            logger.error(f"Image generation failed on device {device_id}: {str(e)}")
            raise

    async def generate_text(self, prompt: str, **kwargs) -> str:
        """Generate text from prompt using intelligently selected optimal model."""
        try:
            # Find best available text model (prefer local first for privacy/speed)
            local_models = [
                m
                for m in self.available_models["text"]
                if m.get("provider") == "local" and m.get("loaded", False)
            ]
            cloud_models = [
                m
                for m in self.available_models["text"]
                if m.get("provider") in ["openai", "anthropic"]
                and m.get("loaded", False)
            ]

            # Prefer local models, fallback to cloud
            available_models = local_models + cloud_models

            if not available_models:
                raise ValueError("No text generation models available")

            # Use intelligent model selection
            model = await self._select_optimal_model(
                prompt=prompt,
                content_type="text",
                available_models=available_models,
                **kwargs,
            )

            if model.get("provider") == "openai":
                return await self._generate_text_openai(prompt, model["name"], **kwargs)
            elif model.get("provider") == "anthropic":
                return await self._generate_text_anthropic(prompt, **kwargs)
            elif model.get("provider") == "local":
                return await self._generate_text_local(prompt, model, **kwargs)
            else:
                raise ValueError(f"Unsupported text model: {model['name']}")

        except Exception as e:
            logger.error(f"Text generation failed: {str(e)}")
            raise

    async def generate_voice(self, text: str, **kwargs) -> Dict[str, Any]:
        """Generate voice from text using best available model."""
        try:
            # Find best available voice model (prefer local for privacy)
            local_models = [
                m
                for m in self.available_models["voice"]
                if m.get("provider") == "local" and m.get("loaded", False)
            ]
            cloud_models = [
                m
                for m in self.available_models["voice"]
                if m.get("provider") in ["elevenlabs", "openai"]
                and m.get("loaded", False)
            ]

            # Prefer local models, fallback to cloud
            available_models = local_models + cloud_models

            if not available_models:
                raise ValueError("No voice generation models available")

            model = available_models[0]

            if model.get("provider") == "elevenlabs":
                return await self._generate_voice_elevenlabs(text, **kwargs)
            elif model.get("provider") == "openai":
                return await self._generate_voice_openai(text, **kwargs)
            elif model.get("provider") == "local":
                return await self._generate_voice_local(text, model, **kwargs)
            else:
                raise ValueError(f"Unsupported voice model: {model['name']}")

        except Exception as e:
            logger.error(f"Voice generation failed: {str(e)}")
            raise

    # Local model generation methods
    async def _generate_text_local(
        self, prompt: str, model: Dict[str, Any], **kwargs
    ) -> str:
        """Generate text using local models."""
        try:
            model_name = model["name"]
            inference_engine = model.get("inference_engine", "transformers")

            if inference_engine == "vllm":
                return await self._generate_text_vllm(prompt, model, **kwargs)
            elif inference_engine == "transformers":
                return await self._generate_text_transformers(prompt, model, **kwargs)
            else:
                raise ValueError(f"Unsupported inference engine: {inference_engine}")

        except Exception as e:
            logger.error(f"Local text generation failed: {str(e)}")
            # Fallback to placeholder
            return f"[Local text generation failed: {str(e)}]"

    async def _generate_image_local(
        self, prompt: str, model: Dict[str, Any], **kwargs
    ) -> Dict[str, Any]:
        """Generate image using local models."""
        try:
            model_name = model["name"]
            inference_engine = model.get("inference_engine", "diffusers")

            if inference_engine == "comfyui":
                return await self._generate_image_comfyui(prompt, model, **kwargs)
            elif inference_engine == "diffusers":
                return await self._generate_image_diffusers(prompt, model, **kwargs)
            else:
                raise ValueError(f"Unsupported inference engine: {inference_engine}")

        except Exception as e:
            logger.error(f"Local image generation failed: {str(e)}")
            # Return placeholder
            return {
                "image_data": b"",
                "format": "PNG",
                "model": model_name,
                "error": str(e),
            }

    async def _generate_voice_local(
        self, text: str, model: Dict[str, Any], **kwargs
    ) -> Dict[str, Any]:
        """Generate voice using local models."""
        try:
            model_name = model["name"]

            if model_name == "xtts-v2":
                return await self._generate_voice_xtts(text, **kwargs)
            elif model_name == "piper":
                return await self._generate_voice_piper(text, **kwargs)
            else:
                raise ValueError(f"Unsupported local voice model: {model_name}")

        except Exception as e:
            logger.error(f"Local voice generation failed: {str(e)}")
            return {
                "audio_data": b"",
                "format": "WAV",
                "model": model_name,
                "error": str(e),
            }

    # vLLM integration
    async def _generate_text_vllm(
        self, prompt: str, model: Dict[str, Any], **kwargs
    ) -> str:
        """Generate text using vLLM for high-performance inference."""
        try:
            # This would integrate with vLLM in production
            # For now, return a placeholder indicating the capability
            return f"[vLLM generation with {model['name']}: {prompt[:100]}...]"
        except Exception as e:
            logger.error(f"vLLM generation failed: {str(e)}")
            raise

    async def _generate_text_transformers(
        self, prompt: str, model: Dict[str, Any], **kwargs
    ) -> str:
        """Generate text using Transformers library."""
        try:
            # This would integrate with transformers in production
            return f"[Transformers generation with {model['name']}: {prompt[:100]}...]"
        except Exception as e:
            logger.error(f"Transformers generation failed: {str(e)}")
            raise

    # ComfyUI integration
    async def _generate_image_comfyui(
        self, prompt: str, model: Dict[str, Any], **kwargs
    ) -> Dict[str, Any]:
        """Generate image using ComfyUI workflow."""
        try:
            # ComfyUI integration requires ComfyUI installation and API setup
            # This is a placeholder for future implementation
            logger.warning(
                f"ComfyUI integration not yet implemented. "
                f"Model {model['name']} requires ComfyUI setup. "
                f"Consider using diffusers-based models instead."
            )
            return {
                "image_data": b"",  # Empty data - ComfyUI not integrated yet
                "format": "PNG",
                "model": model["name"],
                "workflow": "comfyui",
                "provider": "local",
                "status": "not_implemented",
                "note": f"ComfyUI support planned for future release. Use diffusers models for now.",
            }
        except Exception as e:
            logger.error(f"ComfyUI generation failed: {str(e)}")
            raise

    async def _generate_image_diffusers(
        self, prompt: str, model: Dict[str, Any], **kwargs
    ) -> Dict[str, Any]:
        """Generate image using Diffusers library with optional reference image for consistency."""
        try:
            from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler

            model_name = model["name"]
            model_id = model["model_id"]
            # Use the path from model info if available (handles both formats)
            model_path = Path(
                model.get("path", str(self.models_dir / "image" / model_name))
            )

            # Check if reference image is provided for visual consistency
            reference_image_path = kwargs.get("reference_image_path")
            use_controlnet = kwargs.get("use_controlnet", False)

            # Get device ID if specified for multi-GPU support
            device_id = kwargs.get("device_id", None)

            # Load or get cached pipeline (with device-specific caching for multi-GPU)
            if device_id is not None:
                pipeline_key = f"diffusers_{model_name}_gpu{device_id}"
            else:
                pipeline_key = f"diffusers_{model_name}"

            if pipeline_key not in self._loaded_pipelines:
                logger.info(
                    f"Loading diffusion model: {model_name} on device {device_id if device_id is not None else 'default'}"
                )

                # Determine device
                if torch.cuda.is_available():
                    if device_id is not None:
                        device = f"cuda:{device_id}"
                    else:
                        device = "cuda"
                else:
                    device = "cpu"

                # Try to load from local path first, fallback to HuggingFace Hub
                if model_path.exists():
                    logger.info(f"Loading model from local path: {model_path}")
                    pipe = StableDiffusionPipeline.from_pretrained(
                        str(model_path),
                        torch_dtype=(
                            torch.float16 if "cuda" in device else torch.float32
                        ),
                        safety_checker=None,  # Disable for performance
                    )
                else:
                    logger.info(f"Loading model from HuggingFace Hub: {model_id}")
                    pipe = StableDiffusionPipeline.from_pretrained(
                        model_id,
                        torch_dtype=(
                            torch.float16 if "cuda" in device else torch.float32
                        ),
                        safety_checker=None,  # Disable for performance
                    )
                    # Save to local path for future use
                    if not model_path.exists():
                        model_path.mkdir(parents=True, exist_ok=True)
                        pipe.save_pretrained(str(model_path))
                        logger.info(f"Model saved to: {model_path}")

                # Use DPM-Solver++ for faster inference
                pipe.scheduler = DPMSolverMultistepScheduler.from_config(
                    pipe.scheduler.config
                )
                pipe = pipe.to(device)

                # Enable memory optimizations
                if "cuda" in device:
                    pipe.enable_attention_slicing()
                    # Try to enable xformers if available
                    try:
                        pipe.enable_xformers_memory_efficient_attention()
                    except Exception:
                        logger.warning(
                            "xformers not available, using default attention"
                        )

                self._loaded_pipelines[pipeline_key] = pipe
                logger.info(f"Model {model_name} loaded successfully on {device}")

            pipe = self._loaded_pipelines[pipeline_key]

            # Get generation parameters
            num_inference_steps = kwargs.get("num_inference_steps", 25)
            guidance_scale = kwargs.get("guidance_scale", 7.5)
            width = kwargs.get("width", 512)
            height = kwargs.get("height", 512)
            negative_prompt = kwargs.get(
                "negative_prompt", "ugly, blurry, low quality, distorted"
            )
            seed = kwargs.get("seed", None)

            # Set seed for reproducibility if provided
            generator = None
            if seed is not None:
                # Use the specific device for generator if device_id is provided
                if device_id is not None and torch.cuda.is_available():
                    gen_device = f"cuda:{device_id}"
                else:
                    gen_device = "cuda" if torch.cuda.is_available() else "cpu"
                generator = torch.Generator(device=gen_device).manual_seed(seed)

            # Handle reference image for visual consistency
            if reference_image_path and use_controlnet:
                logger.info(
                    f"Visual consistency enabled with reference: {reference_image_path}"
                )
                logger.info(
                    "Note: ControlNet support requires additional setup. Using prompt-based consistency for now."
                )
                # In production, this would load the reference image and use ControlNet
                # For now, we enhance the prompt with consistency instructions
                prompt = f"maintaining consistent appearance from reference, {prompt}"

            logger.info(f"Generating image with prompt: {prompt[:100]}...")

            # Generate image (run in thread pool to avoid blocking)
            loop = asyncio.get_event_loop()
            image = await loop.run_in_executor(
                None,
                lambda: pipe(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    width=width,
                    height=height,
                    generator=generator,
                ).images[0],
            )

            # Convert PIL Image to bytes
            img_byte_arr = io.BytesIO()
            image.save(img_byte_arr, format="PNG")
            image_data = img_byte_arr.getvalue()

            logger.info(f"Image generated successfully: {len(image_data)} bytes")

            return {
                "image_data": image_data,
                "format": "PNG",
                "width": width,
                "height": height,
                "model": model_name,
                "library": "diffusers",
                "provider": "local",
                "prompt": prompt,
                "negative_prompt": negative_prompt,
                "num_inference_steps": num_inference_steps,
                "guidance_scale": guidance_scale,
                "seed": seed,
                "reference_image_used": bool(reference_image_path),
            }
        except Exception as e:
            logger.error(f"Diffusers generation failed: {str(e)}")
            raise

    # XTTS-v2 integration
    async def _generate_voice_xtts(self, text: str, **kwargs) -> Dict[str, Any]:
        """Generate voice using Coqui XTTS-v2."""
        try:
            # Integration with XTTS-v2 in progress
            return {
                "audio_data": b"",  # Empty data - XTTS-v2 integration pending
                "format": "WAV",
                "model": "xtts-v2",
                "voice_cloned": kwargs.get("clone_voice", False),
                "note": f"XTTS-v2 generation: {text[:50]}...",
            }
        except Exception as e:
            logger.error(f"XTTS-v2 generation failed: {str(e)}")
            raise

    # Piper TTS integration
    async def _generate_voice_piper(self, text: str, **kwargs) -> Dict[str, Any]:
        """Generate voice using Piper TTS."""
        try:
            # Integration with Piper TTS in progress
            return {
                "audio_data": b"",  # Empty data - Piper TTS integration pending
                "format": "WAV",
                "model": "piper",
                "voice": kwargs.get("voice", "default"),
                "note": f"Piper TTS generation: {text[:50]}...",
            }
        except Exception as e:
            logger.error(f"Piper generation failed: {str(e)}")
            raise

    async def _generate_image_openai(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """Generate image using OpenAI DALL-E with optional reference image support."""
        try:
            # Check if reference image is provided for visual consistency
            reference_image_path = kwargs.get("reference_image_path")
            if reference_image_path:
                logger.info(
                    f"Visual consistency enabled with reference: {reference_image_path}"
                )
                logger.info(
                    "Note: DALL-E 3 doesn't support image-to-image. Using prompt-based consistency."
                )
                # Enhance prompt with consistency instructions
                prompt = f"maintaining consistent appearance and style, {prompt}"

            response = await self.http_client.post(
                "https://api.openai.com/v1/images/generations",
                headers={
                    "Authorization": f"Bearer {self.settings.openai_api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": "dall-e-3",
                    "prompt": prompt,
                    "size": kwargs.get("size", "1024x1024"),
                    "quality": kwargs.get("quality", "standard"),
                    "n": 1,
                },
            )

            if response.status_code != 200:
                raise Exception(f"OpenAI API error: {response.text}")

            result = response.json()
            image_url = result["data"][0]["url"]

            # Download the image
            image_response = await self.http_client.get(image_url)
            image_data = image_response.content

            return {
                "image_data": image_data,
                "format": "PNG",
                "width": 1024,
                "height": 1024,
                "model": "dall-e-3",
                "provider": "openai",
            }

        except Exception as e:
            logger.error(f"OpenAI image generation failed: {str(e)}")
            raise

    async def _generate_image_stable_diffusion(
        self, prompt: str, **kwargs
    ) -> Dict[str, Any]:
        """Generate image using Stable Diffusion (functional fallback implementation)."""
        try:
            # Return a functional placeholder that doesn't break the system
            # In production, this would use the actual Stable Diffusion pipeline
            logger.info(
                f"Stable Diffusion not available, returning placeholder for: {prompt[:100]}..."
            )

            # Create a basic placeholder image response
            return {
                "image_data": b"",  # Empty bytes as placeholder
                "format": "PNG",
                "width": 512,
                "height": 512,
                "model": "stable-diffusion-placeholder",
                "provider": "local",
                "prompt": prompt,
                "error": None,
                "status": "placeholder",
                "note": "Stable Diffusion requires model download and setup. This is a functional placeholder.",
            }
        except Exception as e:
            logger.error(f"Stable Diffusion placeholder failed: {str(e)}")
            return {
                "image_data": b"",
                "format": "PNG",
                "width": 512,
                "height": 512,
                "model": "stable-diffusion-error",
                "provider": "local",
                "error": str(e),
            }

    async def _generate_text_openai(self, prompt: str, model: str, **kwargs) -> str:
        """Generate text using OpenAI models."""
        try:
            response = await self.http_client.post(
                "https://api.openai.com/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.settings.openai_api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": kwargs.get("max_tokens", 1000),
                    "temperature": kwargs.get("temperature", 0.7),
                },
            )

            if response.status_code != 200:
                raise Exception(f"OpenAI API error: {response.text}")

            result = response.json()
            return result["choices"][0]["message"]["content"].strip()

        except Exception as e:
            logger.error(f"OpenAI text generation failed: {str(e)}")
            raise

    async def _generate_text_anthropic(self, prompt: str, **kwargs) -> str:
        """Generate text using Anthropic Claude."""
        try:
            response = await self.http_client.post(
                "https://api.anthropic.com/v1/messages",
                headers={
                    "x-api-key": self.settings.anthropic_api_key,
                    "Content-Type": "application/json",
                    "anthropic-version": "2023-06-01",
                },
                json={
                    "model": "claude-3-sonnet-20240229",
                    "max_tokens": kwargs.get("max_tokens", 1000),
                    "messages": [{"role": "user", "content": prompt}],
                },
            )

            if response.status_code != 200:
                raise Exception(f"Anthropic API error: {response.text}")

            result = response.json()
            return result["content"][0]["text"].strip()

        except Exception as e:
            logger.error(f"Anthropic text generation failed: {str(e)}")
            raise

    async def _generate_voice_elevenlabs(self, text: str, **kwargs) -> Dict[str, Any]:
        """Generate voice using ElevenLabs."""
        try:
            voice_id = kwargs.get("voice_id", "21m00Tcm4TlvDq8ikWAM")  # Default voice

            response = await self.http_client.post(
                f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}",
                headers={
                    "xi-api-key": self.settings.elevenlabs_api_key,
                    "Content-Type": "application/json",
                },
                json={
                    "text": text,
                    "model_id": "eleven_monolingual_v1",
                    "voice_settings": {
                        "stability": kwargs.get("stability", 0.5),
                        "similarity_boost": kwargs.get("similarity_boost", 0.5),
                    },
                },
            )

            if response.status_code != 200:
                raise Exception(f"ElevenLabs API error: {response.text}")

            audio_data = response.content

            return {
                "audio_data": audio_data,
                "format": "MP3",
                "voice_id": voice_id,
                "provider": "elevenlabs",
            }

        except Exception as e:
            logger.error(f"ElevenLabs voice generation failed: {str(e)}")
            raise

    async def _generate_voice_openai(self, text: str, **kwargs) -> Dict[str, Any]:
        """Generate voice using OpenAI TTS."""
        try:
            response = await self.http_client.post(
                "https://api.openai.com/v1/audio/speech",
                headers={
                    "Authorization": f"Bearer {self.settings.openai_api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": "tts-1",
                    "input": text,
                    "voice": kwargs.get("voice", "alloy"),
                },
            )

            if response.status_code != 200:
                raise Exception(f"OpenAI TTS API error: {response.text}")

            audio_data = response.content

            return {
                "audio_data": audio_data,
                "format": "MP3",
                "voice": kwargs.get("voice", "alloy"),
                "provider": "openai",
            }

        except Exception as e:
            logger.error(f"OpenAI voice generation failed: {str(e)}")
            raise

    async def get_available_models(self) -> Dict[str, List[Dict[str, Any]]]:
        """Get list of available models by type."""
        return self.available_models.copy()

    async def get_system_info(self) -> Dict[str, Any]:
        """Get comprehensive system information for model compatibility."""
        sys_req = self._get_system_requirements()

        # Add inference engine availability
        engines = {}
        for engine in ["vllm", "comfyui", "diffusers", "transformers"]:
            engines[engine] = await self._check_inference_engine(engine)

        return {
            **sys_req,
            "inference_engines": engines,
            "recommended_models": self._get_recommended_models(sys_req),
        }

    def _get_recommended_models(self, sys_req: Dict[str, Any]) -> Dict[str, List[str]]:
        """Get recommended models based on system capabilities."""
        recommendations = {"text": [], "image": [], "voice": []}

        # Text model recommendations
        if sys_req["gpu_memory_gb"] >= 48 and sys_req["ram_gb"] >= 64:
            recommendations["text"].extend(["llama-3.1-70b", "qwen2.5-72b"])
        elif sys_req["gpu_memory_gb"] >= 24 and sys_req["ram_gb"] >= 32:
            recommendations["text"].append("mixtral-8x7b")
        elif sys_req["gpu_memory_gb"] >= 8 and sys_req["ram_gb"] >= 16:
            recommendations["text"].append("llama-3.1-8b")

        # Image model recommendations
        if sys_req["gpu_memory_gb"] >= 12 and sys_req["ram_gb"] >= 24:
            recommendations["image"].append("flux.1-dev")
        elif sys_req["gpu_memory_gb"] >= 8 and sys_req["ram_gb"] >= 16:
            recommendations["image"].append("sdxl-1.0")

        # Voice model recommendations
        if sys_req["gpu_memory_gb"] >= 4 and sys_req["ram_gb"] >= 8:
            recommendations["voice"].append("xtts-v2")
        recommendations["voice"].append("piper")  # Always available

        return recommendations

    async def install_model(self, model_type: str, model_name: str) -> Dict[str, Any]:
        """Install a specific model."""
        try:
            if model_type not in self.local_model_configs:
                raise ValueError(f"Invalid model type: {model_type}")

            if model_name not in self.local_model_configs[model_type]:
                raise ValueError(f"Unknown model: {model_name}")

            config = self.local_model_configs[model_type][model_name]
            model_path = self.models_dir / model_name

            # Check system requirements
            sys_req = self._get_system_requirements()
            if sys_req["gpu_memory_gb"] < config.get("min_gpu_memory_gb", 0) or sys_req[
                "ram_gb"
            ] < config.get("min_ram_gb", 0):
                return {
                    "success": False,
                    "error": "Insufficient system resources",
                    "requirements": {
                        "min_gpu_memory_gb": config.get("min_gpu_memory_gb", 0),
                        "min_ram_gb": config.get("min_ram_gb", 0),
                    },
                    "available": {
                        "gpu_memory_gb": sys_req["gpu_memory_gb"],
                        "ram_gb": sys_req["ram_gb"],
                    },
                }

            # Simulate installation for demo purposes
            model_path.mkdir(exist_ok=True)
            (model_path / "model.safetensors").touch()  # Model stub file for testing

            return {
                "success": True,
                "model": model_name,
                "type": model_type,
                "path": str(model_path),
                "size_gb": config["size_gb"],
            }

        except Exception as e:
            logger.error(f"Model installation failed: {str(e)}")
            return {"success": False, "error": str(e)}

    async def _generate_reference_image_openai(
        self,
        appearance_prompt: str,
        personality_context: Optional[str] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Generate a high-quality reference image using DALL-E 3.

        This method is specifically designed for creating baseline character
        reference images for visual consistency. It uses DALL-E 3 for
        high-quality, single-subject portrait generation.

        Args:
            appearance_prompt: Detailed appearance description
            personality_context: Optional personality traits to inform the image
            **kwargs: Additional generation parameters (quality, size)

        Returns:
            Dict with image_data, format, width, height, and metadata

        Raises:
            Exception: If generation fails
        """
        try:
            # Build enhanced prompt for character reference
            prompt_parts = [
                "Professional character portrait,",
                "highly detailed, photorealistic,",
                appearance_prompt,
            ]

            if personality_context:
                prompt_parts.append(f"expressing {personality_context}")

            prompt_parts.append("high quality studio lighting, centered composition")
            full_prompt = " ".join(prompt_parts)

            logger.info(
                f"Generating reference image with DALL-E 3: {full_prompt[:100]}..."
            )

            response = await self.http_client.post(
                "https://api.openai.com/v1/images/generations",
                headers={
                    "Authorization": f"Bearer {self.settings.openai_api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": "dall-e-3",
                    "prompt": full_prompt,
                    "size": kwargs.get(
                        "size", "1024x1024"
                    ),  # Default to high resolution
                    "quality": kwargs.get(
                        "quality", "hd"
                    ),  # Use HD quality for reference images
                    "n": 1,
                },
            )

            if response.status_code != 200:
                raise Exception(f"OpenAI API error: {response.text}")

            result = response.json()
            image_url = result["data"][0]["url"]

            # Download the generated image
            image_response = await self.http_client.get(image_url)
            image_data = image_response.content

            logger.info("Successfully generated reference image with DALL-E 3")

            return {
                "image_data": image_data,
                "format": "PNG",
                "width": 1024,
                "height": 1024,
                "model": "dall-e-3",
                "provider": "openai",
                "prompt": full_prompt,
            }

        except Exception as e:
            logger.error(f"Failed to generate reference image with DALL-E: {str(e)}")
            raise

    async def _generate_reference_image_local(
        self,
        appearance_prompt: str,
        personality_context: Optional[str] = None,
        reference_image_path: Optional[str] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Generate a reference image using local Stable Diffusion.

        This method uses local MI25/ROCm hardware for cost-effective
        reference image generation. Supports ControlNet for refining
        draft images.

        Args:
            appearance_prompt: Detailed appearance description
            personality_context: Optional personality traits
            reference_image_path: Optional draft image for ControlNet refinement
            **kwargs: Additional generation parameters

        Returns:
            Dict with image_data, format, width, height, and metadata

        Raises:
            Exception: If generation fails
        """
        try:
            # Build enhanced prompt for character reference
            prompt_parts = [
                "professional character portrait, highly detailed,",
                appearance_prompt,
            ]

            if personality_context:
                prompt_parts.append(f"expressing {personality_context}")

            prompt_parts.append("high quality, centered composition, studio lighting")
            full_prompt = " ".join(prompt_parts)

            logger.info(f"Generating reference image locally: {full_prompt[:100]}...")

            # Use high-resolution parameters for reference images
            generation_kwargs = {
                "width": kwargs.get("width", 1024),
                "height": kwargs.get("height", 1024),
                "num_inference_steps": kwargs.get(
                    "num_inference_steps", 50
                ),  # Higher quality
                "guidance_scale": kwargs.get("guidance_scale", 8.0),
                "negative_prompt": kwargs.get(
                    "negative_prompt",
                    "ugly, blurry, low quality, distorted, deformed, bad anatomy",
                ),
                "seed": kwargs.get("seed"),
            }

            # Enable ControlNet if reference image provided
            if reference_image_path:
                logger.info(f"Using ControlNet with reference: {reference_image_path}")
                generation_kwargs["reference_image_path"] = reference_image_path
                generation_kwargs["use_controlnet"] = True

            # Generate using the diffusers pipeline
            result = await self._generate_image_diffusers(
                prompt=full_prompt,
                model=self._get_best_local_image_model(),
                **generation_kwargs,
            )

            logger.info("Successfully generated reference image locally")

            return result

        except Exception as e:
            logger.error(f"Failed to generate reference image locally: {str(e)}")
            raise

    def _get_best_local_image_model(self) -> Dict[str, Any]:
        """Get the best available local image model."""
        local_models = [
            m
            for m in self.available_models.get("image", [])
            if m.get("provider") == "local" and m.get("loaded")
        ]

        if not local_models:
            raise Exception("No local image models available")

        # Prefer SDXL models for best quality
        for model in local_models:
            if "xl" in model.get("name", "").lower():
                return model

        return local_models[0]

    async def generate_video(
        self, prompt: str, video_type: str = "single_frame", **kwargs
    ) -> Dict[str, Any]:
        """
        Generate video using best available model.

        Args:
            prompt: Text prompt or list of prompts for video generation
            video_type: Type of video generation (single_frame, multi_frame, storyboard)
            **kwargs: Additional generation parameters

        Returns:
            Dict with video metadata and file path
        """
        try:
            # Find best available video model
            available_models = [
                m for m in self.available_models["video"] if m.get("loaded", False)
            ]

            if not available_models:
                logger.warning(
                    "No video models available, using frame-by-frame generator"
                )
                # Frame-by-frame is always available
                available_models = [
                    m
                    for m in self.available_models["video"]
                    if m.get("name") == "frame-by-frame-generator"
                ]

            if not available_models:
                raise ValueError("No video generation models available")

            model = available_models[0]
            model_name = model.get("name")

            # Route to appropriate video generation method
            if model_name == "frame-by-frame-generator":
                return await self._generate_video_frame_by_frame(prompt, **kwargs)
            elif model_name == "stable-video-diffusion":
                return await self._generate_video_svd(prompt, **kwargs)
            elif model_name == "runway-gen2":
                return await self._generate_video_runway(prompt, **kwargs)
            else:
                raise ValueError(f"Unsupported video model: {model_name}")

        except Exception as e:
            logger.error(f"Video generation failed: {str(e)}")
            raise

    async def _generate_video_frame_by_frame(
        self, prompt: Union[str, List[str]], **kwargs
    ) -> Dict[str, Any]:
        """
        Generate video using frame-by-frame generation with transitions.

        This is the Q2-Q3 2025 advanced feature implementation.
        Uses AI image generation for each frame.
        """
        try:
            from backend.services.video_processing_service import (
                VideoProcessingService,
                VideoQuality,
                TransitionType,
            )

            # Initialize video processing service
            video_service = VideoProcessingService()

            # Handle single or multiple prompts
            if isinstance(prompt, str):
                prompts = [prompt]
            else:
                prompts = prompt

            # Get parameters from kwargs
            quality_str = kwargs.pop("quality", "high")
            transition_str = kwargs.pop("transition", "crossfade")
            duration_per_frame = kwargs.pop("duration_per_frame", 3.0)
            use_ai_generation = kwargs.pop("use_ai_generation", True)

            # Convert to enums
            quality = VideoQuality(quality_str)
            transition = TransitionType(transition_str)

            # Pass AI model manager to video service for frame generation
            if use_ai_generation:
                kwargs["ai_model_manager"] = self
                kwargs["use_ai_generation"] = True
                logger.info("Using AI image generation for video frames")
            else:
                kwargs["use_ai_generation"] = False
                logger.info("Using placeholder frames for video")

            # Generate video
            result = await video_service.generate_frame_by_frame_video(
                prompts=prompts,
                duration_per_frame=duration_per_frame,
                quality=quality,
                transition=transition,
                **kwargs,
            )

            logger.info(f"Frame-by-frame video generated: {result['file_path']}")
            return result

        except Exception as e:
            logger.error(f"Frame-by-frame video generation failed: {str(e)}")
            raise

    async def _generate_video_svd(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """
        Generate video using Stable Video Diffusion (SVD).

        This is a placeholder implementation. SVD requires significant resources.

        Prerequisites:
        - 24GB+ VRAM (GPU required)
        - Stable Video Diffusion model from HuggingFace
        - diffusers library with SVD support

        Implementation Steps:
        1. Download SVD model: stabilityai/stable-video-diffusion-img2vid-xt
        2. Ensure sufficient GPU memory (24GB+ VRAM)
        3. Generate initial image from text prompt using Stable Diffusion
        4. Use SVD to animate the image into video
        5. Export to MP4 format

        Supported Output:
        - Resolution: 576x1024 (portrait) or 1024x576 (landscape)
        - Duration: 2-4 seconds (14-25 frames)
        - Format: MP4

        References:
        - HuggingFace Model: https://huggingface.co/stabilityai/stable-video-diffusion-img2vid-xt
        - Documentation: https://huggingface.co/docs/diffusers/api/pipelines/stable_video_diffusion
        """
        try:
            logger.info(
                "Stable Video Diffusion requires 24GB+ VRAM and model download. "
                "Returning placeholder."
            )

            # In production, this would:
            # 1. Load the SVD model from HuggingFace
            # 2. Generate initial image from prompt using Stable Diffusion
            # 3. Use SVD pipeline to create video from image
            # 4. Export video file to storage
            # Example:
            # from diffusers import StableVideoDiffusionPipeline
            # pipe = StableVideoDiffusionPipeline.from_pretrained(
            #     "stabilityai/stable-video-diffusion-img2vid-xt"
            # )
            # video_frames = pipe(image, num_frames=25).frames

            # For now, return placeholder
            return {
                "file_path": "/tmp/svd_placeholder.mp4",
                "duration": 4.0,
                "resolution": "576x1024",
                "format": "MP4",
                "model": "stable-video-diffusion",
                "status": "placeholder",
                "note": "SVD requires model download and 24GB+ VRAM. See implementation guide above.",
            }

        except Exception as e:
            logger.error(f"SVD video generation failed: {str(e)}")
            raise

    async def _generate_video_runway(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """
        Generate video using Runway Gen-2 API.

        Requires RUNWAY_API_KEY environment variable.
        """
        try:
            api_key = os.environ.get("RUNWAY_API_KEY")
            if not api_key:
                raise ValueError("RUNWAY_API_KEY not configured")

            logger.info("Runway Gen-2 API integration (placeholder)")

            # In production, this would:
            # 1. Call Runway API with prompt
            # 2. Poll for completion
            # 3. Download generated video
            # 4. Return video metadata

            # For now, return placeholder
            return {
                "file_path": "/tmp/runway_placeholder.mp4",
                "duration": 4.0,
                "resolution": "1920x1080",
                "format": "MP4",
                "model": "runway-gen2",
                "status": "placeholder",
                "note": "Runway Gen-2 requires API key and active subscription",
            }

        except Exception as e:
            logger.error(f"Runway video generation failed: {str(e)}")
            raise

    async def synchronize_audio_to_video(
        self, video_path: str, audio_path: str, output_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Synchronize audio with video.

        This is a Q2-Q3 2025 feature for audio-visual content.
        """
        try:
            from backend.services.video_processing_service import VideoProcessingService
            from pathlib import Path

            video_service = VideoProcessingService()

            result = await video_service.synchronize_audio_with_video(
                video_path=Path(video_path),
                audio_path=Path(audio_path),
                output_path=Path(output_path) if output_path else None,
            )

            logger.info(f"Audio synchronized with video: {result['file_path']}")
            return result

        except Exception as e:
            logger.error(f"Audio synchronization failed: {str(e)}")
            raise

    async def create_video_storyboard(
        self, scenes: List[Dict[str, Any]], quality: str = "high"
    ) -> Dict[str, Any]:
        """
        Create a storyboarded video from scene descriptions.

        This is a Q2-Q3 2025 feature for complex video composition.

        Args:
            scenes: List of scene dicts with 'prompt', 'duration', 'transition'
            quality: Video quality preset

        Returns:
            Dict with storyboard video metadata
        """
        try:
            from backend.services.video_processing_service import (
                VideoProcessingService,
                VideoQuality,
            )

            video_service = VideoProcessingService()
            video_quality = VideoQuality(quality)

            result = await video_service.create_storyboard(
                scenes=scenes, quality=video_quality
            )

            logger.info(f"Storyboard created: {result['file_path']}")
            return result

        except Exception as e:
            logger.error(f"Storyboard creation failed: {str(e)}")
            raise

    async def close(self) -> None:
        """Clean up resources."""
        await self.http_client.aclose()


# Global instance
ai_models = AIModelManager()
