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
import re
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


# Preferred uncensored model for NSFW content generation
# dolphin-mixtral is uncensored and ideal for private servers
PREFERRED_UNCENSORED_MODEL_PREFIX = "dolphin"


# Compile ANSI escape sequence pattern once at module level for performance
# Pattern matches common ANSI escape sequences:
# \x1b is the ESC character (27 in decimal, 0x1B in hex)
# [ is the CSI (Control Sequence Introducer) start character
_ANSI_ESCAPE_PATTERN = re.compile(
    r"""
    \x1b\[[\x30-\x3f]*[\x20-\x2f]*[\x40-\x7e]  # Standard CSI: ESC [ params final
    |\x1b\].*?(?:\x07|\x1b\\)                   # OSC sequences: ESC ] ... BEL/ST
    |\x1b[PX^_].*?\x1b\\                        # String sequences
    |\x1b[@-_]                                   # Fe sequences: ESC + single char
    |\x9b[\x30-\x3f]*[\x20-\x2f]*[\x40-\x7e]    # Single-byte CSI (rare)
""",
    re.VERBOSE,
)


def strip_ansi_codes(text: str) -> str:
    """
    Remove ANSI escape codes from text.

    This includes:
    - Color codes
    - Cursor positioning
    - Terminal control sequences
    - Spinners and progress indicators

    Args:
        text: Text potentially containing ANSI codes

    Returns:
        Clean text without ANSI codes
    """
    return _ANSI_ESCAPE_PATTERN.sub("", text)


async def download_model_from_huggingface(
    model_id: str,
    model_path: Path,
    model_type: str = "text",
    token: Optional[str] = None,
) -> bool:
    """
    Download a model from Hugging Face Hub.

    Args:
        model_id: Hugging Face model ID (e.g., "meta-llama/Llama-3.1-8B-Instruct")
        model_path: Local path where model should be downloaded
        model_type: Type of model (text, image, voice)
        token: Optional HuggingFace token for accessing gated models

    Returns:
        bool: True if download successful, False otherwise
    """
    try:
        from huggingface_hub import snapshot_download

        logger.info(f"📥 Downloading model {model_id}...")
        logger.info(f"   Target path: {model_path}")
        logger.info(f"   This may take several minutes depending on model size...")

        # Create parent directory
        model_path.parent.mkdir(parents=True, exist_ok=True)

        # Get token from settings if not provided
        if token is None:
            from backend.config.settings import get_settings

            settings = get_settings()
            token = settings.hugging_face_token

        # Download model with authentication token for gated repos
        download_kwargs = {
            "repo_id": model_id,
            "local_dir": str(model_path),
            "resume_download": True,
        }

        if token:
            download_kwargs["token"] = token
            logger.info("   Using HuggingFace authentication token")

        downloaded_path = snapshot_download(**download_kwargs)

        logger.info(f"✅ Model downloaded successfully to {downloaded_path}")
        return True

    except Exception as e:
        logger.error(f"❌ Failed to download model {model_id}: {str(e)}")
        if "gated" in str(e).lower() or "401" in str(e):
            logger.error("   This appears to be a gated model. Make sure to:")
            logger.error("   1. Accept the model license on HuggingFace")
            logger.error("   2. Configure your HuggingFace token in Settings")
        return False


async def download_model_from_civitai(
    model_version_id: int,
    model_path: Path,
    model_type: str = "image",
    api_key: Optional[str] = None,
) -> bool:
    """
    Download a model from CivitAI.

    Args:
        model_version_id: CivitAI model version ID
        model_path: Local path where model should be downloaded
        model_type: Type of model (image, text, etc.)
        api_key: Optional CivitAI API key

    Returns:
        bool: True if download successful, False otherwise
    """
    try:
        from backend.utils.civitai_utils import download_civitai_model

        logger.info(f"📥 Downloading CivitAI model version {model_version_id}...")
        logger.info(f"   Target path: {model_path}")
        logger.info(f"   This may take several minutes depending on model size...")

        # Create parent directory
        model_path.parent.mkdir(parents=True, exist_ok=True)

        # Get API key from settings if not provided
        if api_key is None:
            try:
                from backend.config.settings import get_settings

                settings = get_settings()
                api_key = getattr(settings, "civitai_api_key", None)
            except Exception:
                pass

        # Download the model
        downloaded_file, metadata = await download_civitai_model(
            model_version_id=model_version_id,
            output_dir=model_path,
            api_key=api_key,
        )

        # Save metadata for tracking
        metadata_file = model_path / f"{downloaded_file.stem}_metadata.json"
        with open(metadata_file, "w") as f:
            import json

            json.dump(metadata, f, indent=2)

        logger.info(f"✅ Model downloaded successfully to {downloaded_file}")
        logger.info(f"   Metadata saved to {metadata_file}")

        # Log usage tracking info
        if metadata.get("trained_words"):
            logger.info(f"   Trained words: {', '.join(metadata['trained_words'])}")
        if metadata.get("license"):
            logger.info(f"   License: {metadata['license']}")
        if metadata.get("nsfw"):
            logger.warning("   ⚠️  This model may generate NSFW content")

        return True

    except Exception as e:
        logger.error(
            f"❌ Failed to download CivitAI model {model_version_id}: {str(e)}"
        )
        return False


def verify_model_files_exist(model_path: Path, model_type: str = "text") -> bool:
    """
    Verify that a model directory actually contains required model files.

    Args:
        model_path: Path to the model directory
        model_type: Type of model (text, image, voice)

    Returns:
        bool: True if model files exist, False if directory is empty or incomplete
    """
    if not model_path.exists() or not model_path.is_dir():
        return False

    # Check for required files based on model type
    if model_type == "text":
        # Text models should have at least tokenizer and config files
        required_files = [
            "tokenizer.json",
            "tokenizer_config.json",
        ]
        # At least one of these should exist
        optional_indicators = [
            "chat_template.jinja",  # Common in newer models
            "config.json",  # Model configuration
            "model.safetensors",  # Model weights
            "pytorch_model.bin",  # Legacy model weights
            "model.gguf",  # GGUF format
        ]

        # Check required files
        has_required = all((model_path / f).exists() for f in required_files)
        # Check at least one optional indicator
        has_indicator = any((model_path / f).exists() for f in optional_indicators)

        return has_required or has_indicator

    elif model_type == "image":
        # Image models should have model config and weights
        required_files = [
            "model_index.json",  # Diffusers pipeline config
        ]
        return all((model_path / f).exists() for f in required_files)

    elif model_type == "voice":
        # Voice models should have config
        required_files = [
            "config.json",
        ]
        return all((model_path / f).exists() for f in required_files)

    # Default: check if directory is not empty
    try:
        return any(model_path.iterdir())
    except Exception:
        return False


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
        self.lazy_load_enabled = (
            os.environ.get("AI_MODELS_LAZY_LOAD", "false").lower() == "true"
        )
        self.lazy_load_models = set()  # Models configured for lazy loading

        # Configure which models should use lazy loading
        # Typically large models or those used infrequently
        if self.lazy_load_enabled:
            self.lazy_load_models.update(
                [
                    "llama-3.1-70b",  # 140GB - very large
                    "qwen2.5-72b",  # 144GB - very large
                    "flux.1-dev",  # 12GB - less frequently used
                ]
            )

        # Chat model selection thresholds
        # Used to determine if Ollama should be preferred for simple conversational tasks
        self.CHAT_SIMPLE_PROMPT_MAX_WORDS = int(
            os.environ.get("AI_CHAT_SIMPLE_MAX_WORDS", "100")
        )
        self.CHAT_SIMPLE_MAX_TOKENS = int(
            os.environ.get("AI_CHAT_SIMPLE_MAX_TOKENS", "500")
        )

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
                    "inference_engine": "llama.cpp",  # Changed from vllm to llama.cpp
                    # "fallback_engines": ["vllm", "transformers"],  # DISABLED: No fallbacks during debugging
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

            logger.info("🤖 AI MODEL INITIALIZATION")
            logger.info(
                f"   Hardware: {sys_req['gpu_type'].upper()}, {sys_req['gpu_memory_gb']:.1f}GB VRAM, {sys_req['ram_gb']:.1f}GB RAM"
            )
            logger.info(
                f"   CPU cores: {sys_req['cpu_cores']}, Platform: {sys_req['platform']}"
            )

            if self.lazy_load_enabled:
                logger.info(
                    f"   Lazy loading enabled for: {', '.join(self.lazy_load_models)}"
                )

            # Initialize local models based on hardware
            logger.info("   Detecting local AI models...")
            await self._initialize_local_text_models()
            await self._initialize_local_image_models()
            await self._initialize_local_voice_models()

            # Initialize CivitAI models (downloaded from CivitAI marketplace)
            logger.info("   Detecting CivitAI models...")
            await self._initialize_civitai_models()

            # Initialize cloud API models as fallbacks
            logger.info("   Checking cloud API credentials...")
            await self._initialize_cloud_text_models()
            await self._initialize_cloud_image_models()
            await self._initialize_cloud_voice_models()

            # Initialize video models (still mostly placeholder)
            await self._initialize_video_models()

            self.models_loaded = True

            # Count loaded models
            text_loaded = len(
                [m for m in self.available_models.get("text", []) if m.get("loaded")]
            )
            image_loaded = len(
                [m for m in self.available_models.get("image", []) if m.get("loaded")]
            )
            voice_loaded = len(
                [m for m in self.available_models.get("voice", []) if m.get("loaded")]
            )
            video_loaded = len(
                [m for m in self.available_models.get("video", []) if m.get("loaded")]
            )

            logger.info("✅ AI MODEL INITIALIZATION COMPLETE")
            logger.info(f"   Text models: {text_loaded} loaded")
            logger.info(f"   Image models: {image_loaded} loaded")
            logger.info(f"   Voice models: {voice_loaded} loaded")
            logger.info(f"   Video models: {video_loaded} loaded")

            # Detailed model listing
            for category, models in self.available_models.items():
                for model in models:
                    if model.get("loaded"):
                        provider = model.get("provider", "unknown")
                        logger.info(
                            f"   ✓ {category.upper()}: {model['name']} ({provider})"
                        )

        except Exception as e:
            logger.error(f"❌ Failed to initialize AI models: {str(e)}")
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
                    # IMPORTANT: Verify model files actually exist, not just directory
                    if model_path_with_category.exists() and verify_model_files_exist(
                        model_path_with_category, "text"
                    ):
                        model_path = model_path_with_category
                        is_downloaded = True
                    elif model_path_direct.exists() and verify_model_files_exist(
                        model_path_direct, "text"
                    ):
                        model_path = model_path_direct
                        is_downloaded = True
                    else:
                        model_path = (
                            model_path_with_category  # Default for future downloads
                        )
                        is_downloaded = False
                        # Log warning if directory exists but no files
                        if model_path_with_category.exists():
                            logger.warning(
                                f"   ⚠️  Model directory exists but no model files found: {model_name}"
                            )
                            logger.warning(
                                f"       Directory may be incomplete from failed download"
                            )

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

            # Check for Ollama installation and register available models
            try:
                from backend.utils.model_detection import find_ollama_installation

                ollama_info = find_ollama_installation()

                if ollama_info and ollama_info.get("installed"):
                    logger.info("   🦙 Ollama installation detected")
                    logger.info(
                        f"   Ollama version: {ollama_info.get('version', 'unknown')}"
                    )
                    logger.info(f"   Ollama binary: {ollama_info.get('path')}")
                    logger.info(
                        f"   Server running: {ollama_info.get('server_running', False)}"
                    )

                    available_ollama_models = ollama_info.get("available_models", [])

                    if available_ollama_models:
                        logger.info(
                            f"   Found {len(available_ollama_models)} Ollama model(s):"
                        )

                        for ollama_model in available_ollama_models:
                            logger.info(f"     • {ollama_model}")

                            # Register each Ollama model as available for text generation
                            # Use a simple naming convention: the Ollama model name as-is
                            self.available_models["text"].append(
                                {
                                    "name": ollama_model,
                                    "type": "text-generation",
                                    "model_id": ollama_model,
                                    "provider": "local",
                                    "inference_engine": "ollama",
                                    "loaded": True,  # Ollama models are immediately available
                                    "can_load": True,
                                    "lazy_load": False,
                                    "size_gb": 0,  # Size is managed by Ollama
                                    "description": f"Ollama model: {ollama_model}",
                                    "device": (
                                        "gpu" if sys_req["gpu_memory_gb"] > 0 else "cpu"
                                    ),  # Generic GPU (Ollama handles CUDA/ROCm/Metal)
                                    "ollama_model": ollama_model,  # Store original Ollama model name
                                    "path": ollama_info.get(
                                        "path"
                                    ),  # Ollama binary path
                                }
                            )

                        logger.info(
                            f"   ✓ Registered {len(available_ollama_models)} Ollama model(s) for text generation"
                        )
                    else:
                        logger.info(
                            "   ⚠️  Ollama is installed but no models are pulled"
                        )
                        logger.info(
                            "   To use Ollama, pull a model: ollama pull llama2"
                        )
                else:
                    logger.info("   Ollama not detected (optional)")

            except Exception as ollama_error:
                logger.warning(
                    f"   Failed to check Ollama availability: {str(ollama_error)}"
                )

        except Exception as e:
            logger.error(f"Failed to initialize local text models: {str(e)}")

    async def _initialize_local_image_models(self) -> None:
        """Initialize local image generation models."""
        try:
            sys_req = self._get_system_requirements()

            for model_name, config in self.local_model_configs["image"].items():
                # Check if model can run with available GPU memory
                has_gpu_memory = sys_req["gpu_memory_gb"] >= config.get(
                    "min_gpu_memory_gb", 0
                )
                # Check if model has sufficient RAM (for CPU fallback)
                # Allow 10% tolerance below minimum requirements to account for:
                # - Conservative RAM estimates in model configs
                # - System's ability to use swap space
                # - Real-world usage often works below stated minimums
                min_ram_required = config.get("min_ram_gb", 0)
                ram_tolerance = 0.90  # Allow models with 90% of stated minimum
                has_ram = sys_req["ram_gb"] >= (min_ram_required * ram_tolerance)

                # Image models can run on CPU if inference engine is available,
                # even without GPU (though slower)
                can_run = has_gpu_memory and has_ram
                can_run_cpu = has_ram  # Allow CPU fallback if RAM is sufficient

                # Track if model is running below optimal RAM
                below_optimal_ram = has_ram and sys_req["ram_gb"] < min_ram_required

                if can_run or can_run_cpu:
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

                    # Determine if model can be loaded (either on GPU or CPU)
                    can_load = (can_run or can_run_cpu) and engine_available

                    self.available_models["image"].append(
                        {
                            "name": model_name,
                            "type": "text-to-image",
                            "model_id": config["model_id"],
                            "provider": "local",
                            "inference_engine": inference_engine,
                            "loaded": is_downloaded and engine_available,
                            "can_load": can_load,
                            "size_gb": config["size_gb"],
                            "description": config["description"],
                            "device": "cuda" if sys_req["gpu_memory_gb"] > 0 else "cpu",
                            "path": str(model_path),
                            "below_optimal_ram": below_optimal_ram,
                        }
                    )

                    if is_downloaded and engine_available:
                        device_type = "GPU" if has_gpu_memory else "CPU"
                        ram_warning = (
                            f" (RAM: {sys_req['ram_gb']:.1f}GB < {min_ram_required}GB recommended)"
                            if below_optimal_ram
                            else ""
                        )
                        logger.info(
                            f"Local image model {model_name} ready at {model_path} ({device_type}){ram_warning}"
                        )
                    elif is_downloaded and not engine_available:
                        logger.warning(
                            f"Local image model {model_name} found at {model_path} but inference engine {inference_engine} not available"
                        )
                    elif not is_downloaded and engine_available and can_run_cpu:
                        device_type = "GPU" if has_gpu_memory else "CPU"
                        ram_warning = (
                            f" (RAM: {sys_req['ram_gb']:.1f}GB < {min_ram_required}GB recommended)"
                            if below_optimal_ram
                            else ""
                        )
                        logger.info(
                            f"Image model {model_name} can be downloaded and run on {device_type}{ram_warning}"
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

    async def _initialize_civitai_models(self) -> None:
        """
        Initialize models downloaded from CivitAI.

        Scans the models/civitai/ directory for downloaded models and their
        metadata files. Models are registered with the AIModelManager so they
        can be used for image generation.

        CivitAI models are saved with a _metadata.json file containing:
        - model_name: Display name of the model
        - base_model: Base model architecture (SDXL 1.0, SD 1.5, etc.)
        - model_type: Type of model (Checkpoint, LORA, etc.)
        - trained_words: Trigger words for the model
        - nsfw: Whether the model may generate NSFW content
        """
        try:
            sys_req = self._get_system_requirements()
            civitai_dir = self.models_dir / "civitai"

            if not civitai_dir.exists():
                logger.info("   No CivitAI models directory found")
                return

            # Find all metadata files in the civitai directory
            metadata_files = list(civitai_dir.glob("*_metadata.json"))

            if not metadata_files:
                logger.info("   No CivitAI models found")
                return

            logger.info(f"   Found {len(metadata_files)} CivitAI model(s)")

            # Check diffusers availability for image models
            diffusers_available = await self._check_inference_engine("diffusers")
            # Also check ComfyUI availability as it can load CivitAI checkpoints
            comfyui_available = await self._check_inference_engine("comfyui")

            for metadata_file in metadata_files:
                try:
                    # Read metadata
                    with open(metadata_file, "r") as f:
                        metadata = json.load(f)

                    # Find the corresponding model file
                    model_filename = metadata.get("file_name", "")
                    model_file = civitai_dir / model_filename

                    if not model_file.exists():
                        # Try to find by stem (metadata file stem minus _metadata suffix)
                        stem = metadata_file.stem.replace("_metadata", "")
                        potential_files = list(civitai_dir.glob(f"{stem}.*"))
                        model_files = [
                            f
                            for f in potential_files
                            if f.suffix.lower()
                            in [".safetensors", ".ckpt", ".pt", ".bin"]
                        ]
                        if model_files:
                            model_file = model_files[0]
                        else:
                            logger.warning(
                                f"   ⚠️  Model file not found for: {metadata_file.name}"
                            )
                            continue

                    # Get model details from metadata
                    model_name = metadata.get("model_name", model_file.stem)
                    base_model = metadata.get("base_model", "Unknown")
                    model_type = metadata.get("type", "Checkpoint")
                    trained_words = metadata.get("trained_words", [])
                    is_nsfw = metadata.get("nsfw", False)
                    file_size_kb = metadata.get("file_size_kb", 0)
                    size_gb = (
                        round(file_size_kb / (1024 * 1024), 2) if file_size_kb else 0
                    )

                    # Determine model category based on type
                    # Checkpoints and LoRAs are for image generation
                    if model_type.upper() in [
                        "CHECKPOINT",
                        "LORA",
                        "TEXTUALINVERSION",
                        "HYPERNETWORK",
                        "CONTROLNET",
                    ]:
                        category = "image"
                    else:
                        category = "image"  # Default to image for CivitAI models

                    # Determine inference engine based on base model
                    # SDXL and SD models can use diffusers, ComfyUI for more complex workflows
                    if (
                        "SDXL" in base_model
                        or "SD" in base_model
                        or "Pony" in base_model
                    ):
                        inference_engine = "diffusers"
                        engine_available = diffusers_available
                    elif "Flux" in base_model:
                        inference_engine = "comfyui"
                        engine_available = comfyui_available
                    else:
                        # Default to diffusers for most CivitAI models
                        inference_engine = "diffusers"
                        engine_available = diffusers_available

                    # Create unique identifier for the model
                    # Use CivitAI model/version ID if available
                    model_id = metadata.get("model_id")
                    version_id = metadata.get("version_id")
                    if model_id and version_id:
                        unique_id = f"civitai-{model_id}-{version_id}"
                    else:
                        unique_id = f"civitai-{model_file.stem}"

                    # Check if model is already registered (avoid duplicates)
                    existing_names = [
                        m.get("name") for m in self.available_models.get(category, [])
                    ]
                    if unique_id in existing_names:
                        continue

                    # Determine model capabilities
                    model_capabilities = {
                        "type": "text-to-image",
                        "source": "civitai",
                    }

                    # LoRAs need to be used with a base model
                    if model_type.upper() == "LORA":
                        model_capabilities["type"] = "lora"
                        model_capabilities["requires_base_model"] = True

                    # Add to available models
                    model_entry = {
                        "name": unique_id,
                        "display_name": model_name,
                        "type": model_capabilities["type"],
                        "model_type": model_type,  # CivitAI model type (Checkpoint, LORA, etc.)
                        "model_id": (
                            f"civitai:{model_id}"
                            if model_id
                            else f"civitai:{model_file.stem}"
                        ),
                        "provider": "local",
                        "source": "civitai",
                        "inference_engine": inference_engine,
                        "loaded": engine_available,
                        "can_load": engine_available,
                        "size_gb": size_gb,
                        "description": f"CivitAI {model_type}: {model_name} (Base: {base_model})",
                        "device": "cuda" if sys_req["gpu_memory_gb"] > 0 else "cpu",
                        "path": str(model_file),
                        "metadata_path": str(metadata_file),
                        "base_model": base_model,
                        "trained_words": trained_words,
                        "nsfw": is_nsfw,
                        "civitai_model_id": model_id,
                        "civitai_version_id": version_id,
                    }

                    self.available_models[category].append(model_entry)

                    if engine_available:
                        logger.info(
                            f"   ✓ CivitAI {model_type}: {model_name} ({base_model})"
                        )
                        if trained_words:
                            logger.info(
                                f"     Trigger words: {', '.join(trained_words[:5])}"
                                + ("..." if len(trained_words) > 5 else "")
                            )
                    else:
                        logger.warning(
                            f"   ⚠️  CivitAI model {model_name} found but "
                            f"inference engine {inference_engine} not available"
                        )

                except json.JSONDecodeError as e:
                    logger.warning(
                        f"   ⚠️  Failed to parse metadata file {metadata_file.name}: {e}"
                    )
                except Exception as e:
                    logger.warning(
                        f"   ⚠️  Error loading CivitAI model {metadata_file.name}: {e}"
                    )

            # Count loaded CivitAI models
            civitai_loaded = len(
                [
                    m
                    for m in self.available_models.get("image", [])
                    if m.get("source") == "civitai" and m.get("loaded")
                ]
            )
            if civitai_loaded > 0:
                logger.info(f"   ✓ {civitai_loaded} CivitAI model(s) ready for use")

        except Exception as e:
            logger.error(f"Failed to initialize CivitAI models: {str(e)}")

    async def _check_inference_engine(self, engine: str) -> bool:
        """Check if inference engine is available."""
        # Use comprehensive detection logic from model_detection utility
        return check_inference_engine_available(engine, base_dir=self.models_dir.parent)

    async def _initialize_cloud_text_models(self) -> None:
        """Initialize cloud-based text generation models - DISABLED BY DEFAULT."""
        try:
            # Cloud APIs are disabled by default - only enable if explicitly requested
            enable_cloud_apis = (
                os.environ.get("ENABLE_CLOUD_APIS", "false").lower() == "true"
            )

            if not enable_cloud_apis:
                logger.info(
                    "   Cloud APIs disabled by default (use ENABLE_CLOUD_APIS=true to enable)"
                )
                return

            logger.info("   Cloud APIs explicitly enabled via ENABLE_CLOUD_APIS")

            # Add OpenAI GPT as API option (only if explicitly enabled)
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
                            "description": "OpenAI GPT-4 cloud API (PAID)",
                        },
                        {
                            "name": "gpt-3.5-turbo",
                            "type": "text-generation",
                            "provider": "openai",
                            "loaded": True,
                            "max_tokens": 4096,
                            "description": "OpenAI GPT-3.5 Turbo cloud API (PAID)",
                        },
                    ]
                )
                logger.info("   ⚠️  OpenAI text models available (PAID API)")

            # Add Anthropic Claude as option (only if explicitly enabled)
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
                        "description": "Anthropic Claude-3 Sonnet cloud API (PAID)",
                    }
                )
                logger.info("   ⚠️  Anthropic Claude available (PAID API)")

        except Exception as e:
            logger.error(f"Failed to initialize cloud text models: {str(e)}")

    async def _initialize_cloud_image_models(self) -> None:
        """Initialize cloud-based image generation models - DISABLED BY DEFAULT."""
        try:
            # Cloud APIs are disabled by default - only enable if explicitly requested
            enable_cloud_apis = (
                os.environ.get("ENABLE_CLOUD_APIS", "false").lower() == "true"
            )

            if not enable_cloud_apis:
                return

            # Add OpenAI DALL-E only if explicitly enabled
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
                        "description": "OpenAI DALL-E 3 cloud API (PAID)",
                    }
                )
                logger.info("   ⚠️  OpenAI DALL-E 3 API available (PAID API)")

        except Exception as e:
            logger.error(f"Failed to initialize cloud image models: {str(e)}")

    async def _initialize_cloud_voice_models(self) -> None:
        """Initialize cloud-based voice synthesis models - DISABLED BY DEFAULT."""
        try:
            # Cloud APIs are disabled by default - only enable if explicitly requested
            enable_cloud_apis = (
                os.environ.get("ENABLE_CLOUD_APIS", "false").lower() == "true"
            )

            if not enable_cloud_apis:
                return

            # Check for ElevenLabs API (only if explicitly enabled)
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
                        "description": "ElevenLabs voice synthesis cloud API (PAID)",
                    }
                )
                logger.info("   ⚠️  ElevenLabs voice synthesis available (PAID API)")

            # Check for OpenAI TTS (only if explicitly enabled)
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
                        "description": "OpenAI TTS cloud API (PAID)",
                    }
                )
                logger.info("   ⚠️  OpenAI TTS available (PAID API)")

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
            # Check for explicit NSFW model preference from persona settings
            nsfw_model_pref = kwargs.get("nsfw_model")
            if nsfw_model_pref:
                # Try to find the preferred model in available models
                for model in available_models:
                    # Match by name, display_name, or model_id
                    if (
                        nsfw_model_pref.lower() in model["name"].lower()
                        or nsfw_model_pref.lower() in model.get("model_id", "").lower()
                        or nsfw_model_pref.lower()
                        in model.get("display_name", "").lower()
                    ):
                        logger.info(
                            f"🎯 Model selection: {model.get('display_name', model['name'])} (reason: persona NSFW model preference)"
                        )
                        return model
                logger.warning(
                    f"Preferred NSFW model '{nsfw_model_pref}' not found in available models, "
                    f"falling back to intelligent selection"
                )

            # Check for CivitAI models with matching trigger words in prompt
            prompt_lower = prompt.lower()
            civitai_models = [
                m
                for m in available_models
                if m.get("source") == "civitai" and m.get("trained_words")
            ]
            for model in civitai_models:
                trained_words = model.get("trained_words", [])
                for trigger_word in trained_words:
                    # Use word boundary matching to avoid false positives
                    # e.g., "art" should not match "heart" or "party"
                    pattern = r"\b" + re.escape(trigger_word.lower()) + r"\b"
                    if re.search(pattern, prompt_lower):
                        logger.info(
                            f"🎯 Model selection: {model.get('display_name', model['name'])} "
                            f"(reason: CivitAI trigger word '{trigger_word}' found in prompt)"
                        )
                        return model

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
                            f"🎯 Model selection: {model['name']} (reason: high-quality keywords detected)"
                        )
                        return model

            if needs_speed:
                # Prefer SD 1.5 for speed
                for model in available_models:
                    if "v1-5" in model["name"] or "1.5" in model["name"]:
                        logger.info(
                            f"🎯 Model selection: {model['name']} (reason: speed optimization)"
                        )
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
                    f"🎯 Model selection: {sorted_models[0]['name']} (reason: size-based optimization)"
                )
                return sorted_models[0]

        # For text models
        elif content_type == "text":
            # Calculate prompt characteristics once for all selection logic
            prompt_length = len(prompt.split())
            max_tokens = kwargs.get("max_tokens", 500)

            # Check if user explicitly requested Ollama via inference_engine parameter
            requested_engine = kwargs.get("inference_engine")

            # Prioritize Ollama models for chat/conversational tasks
            # Ollama is optimized for interactive use and provides good performance
            ollama_models = [
                m for m in available_models if m.get("inference_engine") == "ollama"
            ]

            # PRIORITY: Prefer uncensored models (like dolphin-mixtral) for unrestricted content generation
            # These models are uncensored and ideal for NSFW content on private servers
            uncensored_models = [
                m
                for m in ollama_models
                if PREFERRED_UNCENSORED_MODEL_PREFIX in m["name"].lower()
            ]
            if uncensored_models:
                logger.info(
                    f"🎯 Model selection: {uncensored_models[0]['name']} (reason: uncensored model preferred for unrestricted content)"
                )
                return uncensored_models[0]

            if requested_engine == "ollama" and ollama_models:
                # User explicitly requested Ollama - use first available Ollama model
                logger.info(
                    f"🎯 Model selection: {ollama_models[0]['name']} (reason: Ollama explicitly requested)"
                )
                return ollama_models[0]
            elif ollama_models and not requested_engine:
                # Ollama available and no specific engine requested
                # Prefer Ollama for chat/conversational tasks (default use case)
                # Check GPU compatibility to confirm Ollama is recommended
                try:
                    from backend.utils.gpu_detection import should_use_ollama_fallback

                    if should_use_ollama_fallback():
                        logger.info(
                            f"🎯 Model selection: {ollama_models[0]['name']} (reason: Ollama recommended for GPU compatibility)"
                        )
                        return ollama_models[0]
                except Exception:
                    pass

                # Even without GPU issues, prefer Ollama for simple conversational tasks
                # (it's optimized for this use case)
                # Simple conversational task detection using configurable thresholds
                is_simple_chat = (
                    prompt_length < self.CHAT_SIMPLE_PROMPT_MAX_WORDS
                    and max_tokens <= self.CHAT_SIMPLE_MAX_TOKENS
                )

                if is_simple_chat:
                    logger.info(
                        f"🎯 Model selection: {ollama_models[0]['name']} (reason: Ollama optimized for chat)"
                    )
                    return ollama_models[0]

            # Longer prompts or complex tasks need larger models

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
                            f"🎯 Model selection: {model['name']} (reason: complex task detected)"
                        )
                        return model
            else:
                # Prefer smaller, faster models for simple tasks
                for model in available_models:
                    if "8b" in model["name"].lower():
                        logger.info(
                            f"🎯 Model selection: {model['name']} (reason: simple task, optimizing for speed)"
                        )
                        return model

        # Default: return first available model
        logger.info(
            f"🎯 Model selection: {available_models[0]['name']} (reason: default/first available)"
        )
        return available_models[0]

    async def generate_image(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """Generate image from text prompt using intelligently selected optimal model."""
        start_time = time.time()
        benchmark_data = None
        model = None
        had_errors = False
        error_message = None

        # Log generation start
        logger.info(f"🎨 AI IMAGE GENERATION STARTED")
        logger.info(f"   Prompt: {prompt[:100]}{'...' if len(prompt) > 100 else ''}")
        logger.info(f"   Quality: {kwargs.get('quality', 'standard')}")
        logger.info(
            f"   Parameters: {', '.join(f'{k}={v}' for k, v in kwargs.items() if k != 'reference_image_path')}"
        )

        try:
            # Find best available image model - LOCAL ONLY by default
            local_models = [
                m
                for m in self.available_models["image"]
                if m.get("provider") == "local" and m.get("can_load", False)
            ]

            # Check ComfyUI availability for models that require it
            comfyui_url = os.environ.get("COMFYUI_API_URL", "http://127.0.0.1:8188")
            comfyui_available = False
            comfyui_models_count = len(
                [m for m in local_models if m.get("inference_engine") == "comfyui"]
            )

            # Always check ComfyUI availability if any local models exist
            # This allows fallback from diffusers to ComfyUI if needed
            logger.info(f"Checking ComfyUI availability at {comfyui_url}...")
            logger.info(
                f"Found {comfyui_models_count} ComfyUI-specific models in local models"
            )

            try:
                response = await self.http_client.get(
                    f"{comfyui_url}/system_stats", timeout=5.0  # Increased timeout
                )
                comfyui_available = response.status_code == 200
                if comfyui_available:
                    logger.info(
                        f"✓ ComfyUI is available and responding at {comfyui_url}"
                    )
                    if comfyui_models_count > 0:
                        logger.info(
                            f"✓ {comfyui_models_count} ComfyUI models will be available for selection"
                        )
                else:
                    logger.warning(
                        f"ComfyUI responded with unexpected status {response.status_code}"
                    )
            except Exception as e:
                logger.warning(
                    f"ComfyUI not accessible at {comfyui_url}: {type(e).__name__}: {str(e)}"
                )
                logger.info(
                    f"To use ComfyUI, ensure it's running with: python main.py --listen"
                )

            # Filter out ComfyUI models ONLY if ComfyUI is not running
            if not comfyui_available and comfyui_models_count > 0:
                original_count = len(local_models)
                local_models = [
                    m for m in local_models if m.get("inference_engine") != "comfyui"
                ]
                filtered_count = original_count - len(local_models)
                logger.warning(
                    f"⚠️  Filtered out {filtered_count} ComfyUI models (ComfyUI not available)"
                )
                logger.info(f"Remaining local models: {len(local_models)}")

            # Only consider cloud models if explicitly enabled
            enable_cloud_apis = (
                os.environ.get("ENABLE_CLOUD_APIS", "false").lower() == "true"
            )
            if enable_cloud_apis:
                cloud_models = [
                    m
                    for m in self.available_models["image"]
                    if m.get("provider") in ["openai"] and m.get("can_load", False)
                ]
            else:
                cloud_models = []

            # LOCAL FIRST - cloud only as explicit fallback
            available_models = local_models + cloud_models

            logger.info(
                f"🤖 Available models: {len(local_models)} local, {len(cloud_models)} cloud"
            )

            if not available_models:
                logger.error(f"❌ No image generation models available")
                raise ValueError("No image generation models available")

            # Use intelligent model selection
            model = await self._select_optimal_model(
                prompt=prompt,
                content_type="image",
                available_models=available_models,
                **kwargs,
            )

            logger.info(
                f"✓ Model selected: {model['name']} ({model.get('provider', 'unknown')})"
            )

            # Record model selection reasoning for benchmark
            selection_reasoning = (
                model.get("selection_reason")
                or f"Selected based on quality={kwargs.get('quality', 'standard')}"
            )

            # Perform generation
            logger.info(f"⚙️  Generating image with {model['name']}...")
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

            logger.info(f"✅ IMAGE GENERATION COMPLETE in {total_time:.2f}s")
            logger.info(f"   Model: {model['name']}")
            logger.info(f"   Size: {result.get('width', 0)}x{result.get('height', 0)}")
            logger.info(f"   Generation time: {generation_time:.2f}s")

            return result

        except Exception as e:
            had_errors = True
            error_message = str(e)
            total_time = time.time() - start_time
            logger.error(f"❌ IMAGE GENERATION FAILED after {total_time:.2f}s")
            logger.error(f"   Error: {str(e)}")
            if model:
                logger.error(f"   Model attempted: {model.get('name', 'unknown')}")
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
        start_time = time.time()

        # Log generation start
        logger.info(f"📝 AI TEXT GENERATION STARTED")
        logger.info(f"   Prompt: {prompt[:100]}{'...' if len(prompt) > 100 else ''}")
        logger.info(f"   Max tokens: {kwargs.get('max_tokens', 1000)}")
        logger.info(f"   Temperature: {kwargs.get('temperature', 0.7)}")

        # Check GPU compatibility and adjust inference engine preference
        try:
            from backend.utils.gpu_detection import should_use_ollama_fallback

            if should_use_ollama_fallback():
                from backend.utils.model_detection import find_ollama_installation

                ollama_info = find_ollama_installation()

                if ollama_info and ollama_info.get("installed"):
                    logger.info(
                        "   🦙 GPU compatibility: Using Ollama (recommended for this hardware)"
                    )
                    # Force Ollama usage by overriding inference_engine if not explicitly set
                    if "inference_engine" not in kwargs:
                        kwargs["inference_engine"] = "ollama"
        except Exception as e:
            logger.debug(f"   GPU detection skipped: {e}")

        try:
            # Find best available text model - LOCAL ONLY by default
            local_models = [
                m
                for m in self.available_models["text"]
                if m.get("provider") == "local" and m.get("loaded", False)
            ]

            # Only consider cloud models if explicitly enabled
            enable_cloud_apis = (
                os.environ.get("ENABLE_CLOUD_APIS", "false").lower() == "true"
            )
            if enable_cloud_apis:
                cloud_models = [
                    m
                    for m in self.available_models["text"]
                    if m.get("provider") in ["openai", "anthropic"]
                    and m.get("loaded", False)
                ]
            else:
                cloud_models = []

            # LOCAL FIRST - cloud only as explicit fallback
            available_models = local_models + cloud_models

            logger.info(
                f"🤖 Available models: {len(local_models)} local, {len(cloud_models)} cloud"
            )

            if not available_models:
                logger.error(f"❌ No text generation models available")
                raise ValueError("No text generation models available")

            # Use intelligent model selection
            model = await self._select_optimal_model(
                prompt=prompt,
                content_type="text",
                available_models=available_models,
                **kwargs,
            )

            logger.info(
                f"✓ Model selected: {model['name']} ({model.get('provider', 'unknown')})"
            )
            logger.info(f"⚙️  Generating text with {model['name']}...")

            generation_start = time.time()
            if model.get("provider") == "openai":
                result = await self._generate_text_openai(
                    prompt, model["name"], **kwargs
                )
            elif model.get("provider") == "anthropic":
                result = await self._generate_text_anthropic(prompt, **kwargs)
            elif model.get("provider") == "local":
                result = await self._generate_text_local(prompt, model, **kwargs)
            else:
                raise ValueError(f"Unsupported text model: {model['name']}")

            generation_time = time.time() - generation_start
            total_time = time.time() - start_time

            logger.info(f"✅ TEXT GENERATION COMPLETE in {total_time:.2f}s")
            logger.info(f"   Model: {model['name']}")
            logger.info(
                f"   Output length: {len(result)} characters, {len(result.split())} words"
            )
            logger.info(f"   Generation time: {generation_time:.2f}s")

            return result

        except Exception as e:
            total_time = time.time() - start_time
            logger.error(f"❌ TEXT GENERATION FAILED after {total_time:.2f}s")
            logger.error(f"   Error: {str(e)}")
            raise

    async def generate_voice(self, text: str, **kwargs) -> Dict[str, Any]:
        """Generate voice from text using best available model."""
        start_time = time.time()

        # Log generation start
        logger.info(f"🎙️ AI VOICE GENERATION STARTED")
        logger.info(f"   Text: {text[:100]}{'...' if len(text) > 100 else ''}")
        logger.info(
            f"   Voice settings: {kwargs.get('voice', 'default')}, {kwargs.get('voice_id', 'default')}"
        )

        try:
            # Find best available voice model - LOCAL ONLY by default
            local_models = [
                m
                for m in self.available_models["voice"]
                if m.get("provider") == "local" and m.get("loaded", False)
            ]

            # Only consider cloud models if explicitly enabled
            enable_cloud_apis = (
                os.environ.get("ENABLE_CLOUD_APIS", "false").lower() == "true"
            )
            if enable_cloud_apis:
                cloud_models = [
                    m
                    for m in self.available_models["voice"]
                    if m.get("provider") in ["elevenlabs", "openai"]
                    and m.get("loaded", False)
                ]
            else:
                cloud_models = []

            # LOCAL FIRST - cloud only as explicit fallback
            available_models = local_models + cloud_models

            logger.info(
                f"🤖 Available models: {len(local_models)} local, {len(cloud_models)} cloud"
            )

            if not available_models:
                logger.error(f"❌ No voice generation models available")
                raise ValueError("No voice generation models available")

            model = available_models[0]

            logger.info(
                f"✓ Model selected: {model['name']} ({model.get('provider', 'unknown')})"
            )
            logger.info(f"⚙️  Generating voice with {model['name']}...")

            generation_start = time.time()
            if model.get("provider") == "elevenlabs":
                result = await self._generate_voice_elevenlabs(text, **kwargs)
            elif model.get("provider") == "openai":
                result = await self._generate_voice_openai(text, **kwargs)
            elif model.get("provider") == "local":
                result = await self._generate_voice_local(text, model, **kwargs)
            else:
                raise ValueError(f"Unsupported voice model: {model['name']}")

            generation_time = time.time() - generation_start
            total_time = time.time() - start_time

            logger.info(f"✅ VOICE GENERATION COMPLETE in {total_time:.2f}s")
            logger.info(f"   Model: {model['name']}")
            logger.info(f"   Audio size: {len(result.get('audio_data', b''))} bytes")
            logger.info(f"   Format: {result.get('format', 'unknown')}")
            logger.info(f"   Generation time: {generation_time:.2f}s")

            return result

        except Exception as e:
            total_time = time.time() - start_time
            logger.error(f"❌ VOICE GENERATION FAILED after {total_time:.2f}s")
            logger.error(f"   Error: {str(e)}")
            raise

    # Local model generation methods
    async def _generate_text_local(
        self, prompt: str, model: Dict[str, Any], **kwargs
    ) -> str:
        """Generate text using local models with raw engine output streaming and intelligent fallback."""
        model_name = model["name"]
        inference_engine = model.get("inference_engine", "transformers")
        # FALLBACKS DISABLED FOR DEBUGGING
        # We want to see all failures clearly without masking them with fallbacks
        # Once everything works, we can re-enable fallback_engines
        fallback_engines = []  # model.get("fallback_engines", [])

        # Use only the primary engine (no fallbacks during debugging)
        logger.info(f"   Trying inference engine: {inference_engine}")

        # Check if model is downloaded, download if needed
        model_path = Path(model.get("path", ""))
        if not model_path.exists():
            logger.warning(f"   Model not found at {model_path}")

            # Attempt to download model
            model_id = model.get("model_id")
            if model_id:
                logger.info(f"   Attempting to download model {model_id}...")
                success = await download_model_from_huggingface(
                    model_id=model_id, model_path=model_path, model_type="text"
                )
                if not success:
                    error_msg = f"Failed to download model {model_name} from {model_id}"
                    logger.error(f"   ❌ {error_msg}")
                    raise ValueError(error_msg)
            else:
                error_msg = f"Model {model_name} not found and no model_id for download"
                logger.error(f"   ❌ {error_msg}")
                raise ValueError(error_msg)

        # Try generation with the primary engine
        if inference_engine in ["llama.cpp", "llama-cpp"]:  # Support both formats
            try:
                return await self._generate_text_llamacpp(prompt, model, **kwargs)
            except Exception as e:
                # If llama.cpp fails, try Ollama as fallback
                logger.warning(f"   ⚠️  llama.cpp failed: {str(e)}")
                logger.info(f"   🔄 Attempting fallback to Ollama...")

                from backend.utils.model_detection import find_ollama_installation
                from backend.utils.gpu_detection import get_gpu_info

                # Log GPU information for diagnostics
                gpu_info = get_gpu_info()
                if gpu_info.get("architecture"):
                    logger.info(
                        f"   GPU detected: {gpu_info['architecture']} (Ollama recommended: {gpu_info.get('ollama_recommended', False)})"
                    )

                ollama_info = find_ollama_installation()

                if ollama_info and ollama_info.get("installed"):
                    try:
                        # Try Ollama with the same prompt
                        return await self._generate_text_ollama(prompt, model, **kwargs)
                    except Exception as ollama_error:
                        logger.error(
                            f"   ❌ Ollama fallback also failed: {str(ollama_error)}"
                        )
                        # Re-raise the original llama.cpp error since fallback failed too
                        raise e
                else:
                    logger.warning(f"   ⚠️  Ollama not available for fallback")
                    # Re-raise the original error
                    raise e
        elif inference_engine == "ollama":
            return await self._generate_text_ollama(prompt, model, **kwargs)
        elif inference_engine == "vllm":
            return await self._generate_text_vllm(prompt, model, **kwargs)
        elif inference_engine == "transformers":
            return await self._generate_text_transformers(prompt, model, **kwargs)
        else:
            error_msg = f"Unknown inference engine: {inference_engine}"
            logger.error(f"   ❌ {error_msg}")
            raise ValueError(error_msg)

    def _filter_llamacpp_output(self, lines: List[str]) -> str:
        """
        Filter llama.cpp output to extract only generated text.

        Filters out:
        - GGML initialization messages (ggml_cuda_init, ggml_*)
        - llama.cpp system info (llama_*, system_info:)
        - Debug/log messages
        - Empty lines at start/end

        Args:
            lines: List of output lines from llama.cpp

        Returns:
            Cleaned generated text
        """
        # Patterns to filter out (initialization/debug output)
        # Pre-converted to lowercase for efficient case-insensitive matching
        filter_patterns = [
            "ggml_",  # GGML library initialization
            "llama_",  # llama.cpp initialization
            "system_info:",  # System information
            "sampling:",  # Sampling parameters
            "generate:",  # Generation metadata
            "device",  # GPU device info (matches Device, device, DEVICE, etc.)
            "compute:",  # Compute backend info
            "backend:",  # Backend info
        ]

        filtered_lines = []

        for line in lines:
            line_stripped = line.strip()

            # Skip empty lines
            if not line_stripped:
                continue

            # Check if line matches any filter pattern
            should_filter = False
            for pattern in filter_patterns:
                if pattern.lower() in line_stripped.lower():
                    should_filter = True
                    break

            # Skip filtered lines
            if should_filter:
                continue

            # This is likely generated text
            filtered_lines.append(line)

        # Join the filtered lines and clean up
        result = "\n".join(filtered_lines).strip()

        # If we got nothing after filtering, return a message
        if not result:
            logger.warning(
                "   ⚠️  No generated text found after filtering initialization logs"
            )
            return ""

        return result

    async def _generate_text_llamacpp(
        self, prompt: str, model: Dict[str, Any], **kwargs
    ) -> str:
        """Generate text using llama.cpp with filtered output."""
        try:
            logger.info(f"   🦙 Starting llama.cpp engine for {model['name']}...")
            logger.info(f"   Model path: {model.get('path', 'Not specified')}")

            # Check for llama.cpp binary
            llamacpp_binary = shutil.which("llama-cli") or shutil.which("main")
            if not llamacpp_binary:
                logger.warning(f"   ⚠️  llama.cpp not found in PATH")
                logger.warning(
                    f"   Install llama.cpp and ensure 'llama-cli' or 'main' is in PATH"
                )
                raise ValueError("llama.cpp not found")

            model_file = model.get("path")
            if not Path(model_file).exists():
                logger.warning(f"   ⚠️  Model file not found: {model_file}")
                raise ValueError(f"Model file not found: {model_file}")

            logger.info(f"   ✓ Found llama.cpp at: {llamacpp_binary}")
            logger.info(f"   " + "=" * 76)
            logger.info(f"   RAW LLAMA.CPP ENGINE OUTPUT (LIVE):")
            logger.info(f"   " + "=" * 76)

            # Build llama.cpp command
            cmd = [
                llamacpp_binary,
                "-m",
                str(model_file),
                "-p",
                prompt,
                "-n",
                str(kwargs.get("max_tokens", 1000)),
                "--temp",
                str(kwargs.get("temperature", 0.7)),
                "-c",
                "4096",  # context size
                "--log-disable",  # Disable file logging
            ]

            # Run subprocess and stream output in real-time
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                cwd=str(Path(model_file).parent),
            )

            raw_output_lines = []
            async for line in process.stdout:
                line_text = line.decode("utf-8", errors="ignore").rstrip()
                if line_text:
                    # Strip ANSI codes before logging and storing
                    clean_line = strip_ansi_codes(line_text)
                    if clean_line:  # Only log non-empty lines after stripping
                        # Print cleaned llama.cpp output to logs
                        logger.info(f"   {clean_line}")
                        raw_output_lines.append(clean_line)

            await process.wait()

            logger.info(f"   " + "=" * 76)

            # Check if llama.cpp succeeded (exit code 0 = success)
            if process.returncode != 0:
                logger.error(
                    f"   ❌ llama.cpp generation FAILED (exit code: {process.returncode})"
                )
                logger.error(
                    f"   The model may have encountered an error during generation"
                )
                # Show the raw output for debugging
                MAX_ERROR_LINES = 10  # Number of output lines to show for debugging
                error_output = (
                    "\n".join(raw_output_lines[-MAX_ERROR_LINES:])
                    if raw_output_lines
                    else "No output"
                )
                logger.error(f"   Last output lines: {error_output}")
                raise RuntimeError(
                    f"llama.cpp failed with exit code {process.returncode}. "
                    f"The model may be incompatible or corrupted."
                )

            logger.info(
                f"   ✓ llama.cpp generation complete (exit code: {process.returncode})"
            )

            # Filter out initialization logs and extract generated text
            generated_text = self._filter_llamacpp_output(raw_output_lines)

            if not generated_text:
                logger.warning("   ⚠️  No text generated after filtering")
                logger.warning(
                    "   This usually means llama.cpp only output initialization logs"
                )
                logger.warning(
                    "   Check if the model file is compatible with llama.cpp"
                )
                # Don't return raw output - raise an error instead
                raise RuntimeError(
                    "No text generated by llama.cpp. Model may be incompatible or prompt may be invalid."
                )

            return generated_text

        except Exception as e:
            logger.error(f"   ❌ llama.cpp generation failed: {str(e)}")
            raise

    async def _generate_text_ollama(
        self, prompt: str, model: Dict[str, Any], **kwargs
    ) -> str:
        """
        Generate text using Ollama.

        Ollama provides a simple CLI and API for running LLMs locally.
        This method uses the CLI interface for compatibility.

        Args:
            prompt: Text prompt for generation
            model: Model configuration dict (must include 'name' or 'ollama_model')
            **kwargs: Additional generation parameters (temperature, max_tokens, etc.)

        Returns:
            Generated text

        Raises:
            ValueError: If Ollama is not installed or model not available
            RuntimeError: If generation fails
        """
        try:
            logger.info(f"   🦙 Starting Ollama engine for {model['name']}...")

            # Check for Ollama binary
            ollama_binary = shutil.which("ollama")
            if not ollama_binary:
                logger.warning(f"   ⚠️  Ollama not found in PATH")
                logger.warning(f"   Install Ollama from https://ollama.com/download")
                raise ValueError("Ollama not found")

            logger.info(f"   ✓ Found Ollama at: {ollama_binary}")

            # Determine which Ollama model to use
            # Priority: explicit ollama_model config > model name
            ollama_model = model.get("ollama_model", model.get("name", ""))

            if not ollama_model:
                raise ValueError("No Ollama model specified in configuration")

            logger.info(f"   📦 Using Ollama model: {ollama_model}")
            logger.info(f"   " + "=" * 76)
            logger.info(f"   RAW OLLAMA ENGINE OUTPUT (LIVE):")
            logger.info(f"   " + "=" * 76)

            # Build Ollama command
            # Using 'ollama run' with prompt as command argument for simplicity
            cmd = [
                ollama_binary,
                "run",
                ollama_model,
                prompt,
            ]

            # Run subprocess and capture output
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                stdin=asyncio.subprocess.PIPE,
            )

            # Close stdin immediately since we pass prompt as argument
            process.stdin.close()

            raw_output_lines = []
            async for line in process.stdout:
                line_text = line.decode("utf-8", errors="ignore").rstrip()
                if line_text:
                    # Strip ANSI codes before logging and storing
                    clean_line = strip_ansi_codes(line_text)
                    if clean_line:
                        # Filter out lines that are just spinner characters or loading messages
                        # (these are transient and get overwritten in real terminal)
                        stripped = clean_line.strip()
                        # Skip if line is just a single spinner character or "Loading model..."
                        if len(stripped) <= 2 or "Loading model" in stripped:
                            continue
                        # Print cleaned Ollama output to logs
                        logger.info(f"   {clean_line}")
                        raw_output_lines.append(clean_line)

            await process.wait()

            logger.info(f"   " + "=" * 76)

            # Check if Ollama succeeded (exit code 0 = success)
            if process.returncode != 0:
                logger.error(
                    f"   ❌ Ollama generation FAILED (exit code: {process.returncode})"
                )
                logger.error(
                    f"   The model may not be available or Ollama server may not be running"
                )
                # Show the raw output for debugging
                MAX_ERROR_LINES = 10
                error_output = (
                    "\n".join(raw_output_lines[-MAX_ERROR_LINES:])
                    if raw_output_lines
                    else "No output"
                )
                logger.error(f"   Last output lines: {error_output}")
                raise RuntimeError(
                    f"Ollama failed with exit code {process.returncode}. "
                    f"Model may not be pulled or server may not be running. "
                    f"Try: ollama pull {ollama_model}"
                )

            logger.info(
                f"   ✓ Ollama generation complete (exit code: {process.returncode})"
            )

            # Ollama output is cleaner than llama.cpp - just join lines
            generated_text = "\n".join(raw_output_lines).strip()

            if not generated_text:
                logger.warning("   ⚠️  No text generated by Ollama")
                logger.warning("   This may mean the model needs to be pulled first")
                logger.warning(f"   Try: ollama pull {ollama_model}")
                raise RuntimeError(
                    f"No text generated by Ollama. Try pulling the model first: ollama pull {ollama_model}"
                )

            return generated_text

        except Exception as e:
            logger.error(f"   ❌ Ollama generation failed: {str(e)}")
            raise

    async def _generate_image_local(
        self, prompt: str, model: Dict[str, Any], **kwargs
    ) -> Dict[str, Any]:
        """Generate image using local models."""
        model_name = model["name"]
        inference_engine = model.get("inference_engine", "diffusers")

        if inference_engine == "comfyui":
            return await self._generate_image_comfyui(prompt, model, **kwargs)
        elif inference_engine == "diffusers":
            return await self._generate_image_diffusers(prompt, model, **kwargs)
        else:
            raise ValueError(f"Unsupported inference engine: {inference_engine}")

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
        logger.info(f"   🔧 Starting vLLM engine for {model['name']}...")
        logger.info(f"   " + "=" * 76)
        logger.info(f"   RAW vLLM ENGINE OUTPUT:")
        logger.info(f"   " + "=" * 76)

        # Check if vLLM server is running
        vllm_url = os.environ.get("VLLM_API_URL", "http://localhost:8001")

        try:
            # Try to use vLLM API if available
            response = await self.http_client.post(
                f"{vllm_url}/v1/completions",
                json={
                    "model": model.get("model_id", model["name"]),
                    "prompt": prompt,
                    "max_tokens": kwargs.get("max_tokens", 1000),
                    "temperature": kwargs.get("temperature", 0.7),
                    "stream": False,
                },
                timeout=300.0,
            )

            if response.status_code == 200:
                result = response.json()
                generated_text = result["choices"][0]["text"]

                # Log raw vLLM stats
                logger.info(f"   vLLM Stats:")
                logger.info(f"   - Model: {model['name']}")
                logger.info(
                    f"   - Tokens generated: {result.get('usage', {}).get('completion_tokens', 'N/A')}"
                )
                logger.info(
                    f"   - Total tokens: {result.get('usage', {}).get('total_tokens', 'N/A')}"
                )
                logger.info(f"   " + "-" * 76)
                logger.info(f"   GENERATED TEXT:")
                for line in generated_text.split("\n"):
                    logger.info(f"   | {line}")
                logger.info(f"   " + "=" * 76)

                return generated_text
            else:
                error_msg = f"vLLM server returned status {response.status_code}"
                logger.warning(f"   ⚠️  {error_msg}")
                raise ValueError(error_msg)

        except Exception as vllm_error:
            logger.warning(f"   ⚠️  vLLM not available: {str(vllm_error)}")
            logger.info(f"   ℹ️  vLLM engine not configured or not running")
            logger.info(
                f"   To enable: Set VLLM_API_URL or start vLLM server at {vllm_url}"
            )
            logger.info(f"   " + "=" * 76)
            # Raise error to trigger fallback to other engines
            raise ValueError(f"vLLM not available: {str(vllm_error)}")

    async def _generate_text_transformers(
        self, prompt: str, model: Dict[str, Any], **kwargs
    ) -> str:
        """Generate text using Transformers library with raw output streaming."""
        try:
            logger.info(f"   🔧 Starting Transformers engine for {model['name']}...")
            logger.info(f"   Model path: {model.get('path', 'Not specified')}")
            logger.info(f"   " + "=" * 76)
            logger.info(f"   RAW TRANSFORMERS ENGINE OUTPUT:")
            logger.info(f"   " + "=" * 76)

            # Check if model is actually available locally
            model_path = Path(model.get("path", ""))
            if not model_path.exists():
                logger.warning(f"   ⚠️  Model not found at {model_path}")
                logger.warning(f"   To use this model, download it first:")
                logger.warning(
                    f"   huggingface-cli download {model.get('model_id', 'unknown')}"
                )
                raise ValueError(f"Model not found: {model['name']}")

            # Try to load and use transformers
            try:
                from transformers import AutoModelForCausalLM, AutoTokenizer

                logger.info(f"   Loading model from {model_path}...")
                tokenizer = AutoTokenizer.from_pretrained(str(model_path))
                loaded_model = AutoModelForCausalLM.from_pretrained(
                    str(model_path),
                    dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                    device_map="auto" if torch.cuda.is_available() else "cpu",
                )

                logger.info(f"   ✓ Model loaded, tokenizing prompt...")
                inputs = tokenizer(prompt, return_tensors="pt").to(loaded_model.device)

                logger.info(
                    f"   ⚙️  Generating (max_tokens={kwargs.get('max_tokens', 1000)})..."
                )
                logger.info(f"   " + "-" * 76)

                outputs = loaded_model.generate(
                    **inputs,
                    max_new_tokens=kwargs.get("max_tokens", 1000),
                    temperature=kwargs.get("temperature", 0.7),
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id,
                )

                generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

                # Log the raw output
                logger.info(f"   GENERATED TEXT:")
                for line in generated_text.split("\n"):
                    logger.info(f"   | {line}")
                logger.info(f"   " + "=" * 76)

                return generated_text

            except ImportError:
                logger.warning(f"   ⚠️  Transformers library not available")
                logger.warning(f"   Install with: pip install transformers torch")
                raise ValueError("Transformers library not installed")

        except Exception as e:
            logger.error(f"   ❌ Transformers generation failed: {str(e)}")
            raise

    # ComfyUI integration
    async def _generate_image_comfyui(
        self, prompt: str, model: Dict[str, Any], **kwargs
    ) -> Dict[str, Any]:
        """
        Generate image using ComfyUI workflow API.

        ComfyUI provides a REST API for submitting workflows and retrieving generated images.
        This implementation uses a basic text-to-image workflow compatible with FLUX models.

        Args:
            prompt: Text prompt for image generation
            model: Model configuration dict with model_id and path
            **kwargs: Additional parameters (width, height, steps, guidance_scale, seed)

        Returns:
            Dict with image_data, format, width, height, and metadata

        Raises:
            Exception: If ComfyUI is not available or generation fails
        """
        try:
            import uuid as uuid_lib

            # Get ComfyUI API URL from environment or use default
            comfyui_url = os.environ.get("COMFYUI_API_URL", "http://127.0.0.1:8188")

            # Check if ComfyUI is available
            comfyui_path = find_comfyui_installation(self.models_dir.parent)

            if not comfyui_path:
                logger.warning(
                    f"ComfyUI installation not found. "
                    f"Model {model['name']} requires ComfyUI setup. "
                    f"Attempting to use ComfyUI API at {comfyui_url}"
                )

            # Try to connect to ComfyUI API
            try:
                response = await self.http_client.get(
                    f"{comfyui_url}/system_stats", timeout=5.0
                )
                if response.status_code != 200:
                    raise ConnectionError(
                        f"ComfyUI API not responding (status {response.status_code})"
                    )
                logger.info(f"✓ Connected to ComfyUI at {comfyui_url}")
            except Exception as conn_error:
                logger.warning(
                    f"ComfyUI API not accessible at {comfyui_url}: {str(conn_error)}\n"
                    f"Please ensure ComfyUI is running with: python main.py --listen\n"
                    f"Falling back to diffusers-based generation"
                )
                # Fallback to diffusers if available
                return await self._fallback_to_diffusers(prompt, **kwargs)

            # Get generation parameters
            width = kwargs.get("width", 1024)
            height = kwargs.get("height", 1024)
            num_inference_steps = kwargs.get("num_inference_steps", 20)
            guidance_scale = kwargs.get(
                "guidance_scale", 3.5
            )  # FLUX uses lower guidance
            seed = kwargs.get("seed", None)
            if seed is None:
                seed = int(time.time() * 1000) % (2**32)

            # Create a basic FLUX workflow for text-to-image
            # This is a simplified workflow compatible with FLUX.1-dev
            workflow = {
                "3": {  # CLIPTextEncode for positive prompt
                    "inputs": {"text": prompt, "clip": ["11", 0]},
                    "class_type": "CLIPTextEncode",
                },
                "4": {  # Empty latent image
                    "inputs": {"width": width, "height": height, "batch_size": 1},
                    "class_type": "EmptyLatentImage",
                },
                "8": {  # VAEDecode
                    "inputs": {"samples": ["10", 0], "vae": ["11", 2]},
                    "class_type": "VAEDecode",
                },
                "9": {  # SaveImage
                    "inputs": {"filename_prefix": "gator_comfyui", "images": ["8", 0]},
                    "class_type": "SaveImage",
                },
                "10": {  # KSampler
                    "inputs": {
                        "seed": seed,
                        "steps": num_inference_steps,
                        "cfg": guidance_scale,
                        "sampler_name": "euler",
                        "scheduler": "simple",
                        "denoise": 1.0,
                        "model": ["11", 0],
                        "positive": ["3", 0],
                        "negative": ["3", 0],  # FLUX uses same for negative
                        "latent_image": ["4", 0],
                    },
                    "class_type": "KSampler",
                },
                "11": {  # CheckpointLoaderSimple
                    "inputs": {
                        "ckpt_name": model.get("model_id", "flux1-dev.safetensors")
                    },
                    "class_type": "CheckpointLoaderSimple",
                },
            }

            # Generate a unique prompt ID
            prompt_id = str(uuid_lib.uuid4())

            # Submit workflow to ComfyUI
            logger.info(f"Submitting workflow to ComfyUI: {prompt[:100]}...")
            queue_response = await self.http_client.post(
                f"{comfyui_url}/prompt",
                json={"prompt": workflow, "client_id": prompt_id},
                timeout=10.0,
            )

            if queue_response.status_code != 200:
                raise Exception(f"Failed to queue prompt: {queue_response.text}")

            queue_result = queue_response.json()
            prompt_execution_id = queue_result.get("prompt_id")

            logger.info(f"Workflow queued with ID: {prompt_execution_id}")

            # Poll for completion
            max_wait_time = 300  # 5 minutes max
            poll_interval = 2  # Check every 2 seconds
            elapsed_time = 0

            while elapsed_time < max_wait_time:
                await asyncio.sleep(poll_interval)
                elapsed_time += poll_interval

                # Check queue status
                history_response = await self.http_client.get(
                    f"{comfyui_url}/history/{prompt_execution_id}", timeout=5.0
                )

                if history_response.status_code == 200:
                    history_data = history_response.json()
                    if prompt_execution_id in history_data:
                        # Generation completed
                        result_data = history_data[prompt_execution_id]

                        # Extract output image information
                        outputs = result_data.get("outputs", {})
                        for node_id, node_output in outputs.items():
                            if "images" in node_output:
                                images = node_output["images"]
                                if images:
                                    # Get the first image
                                    image_info = images[0]
                                    filename = image_info["filename"]
                                    subfolder = image_info.get("subfolder", "")
                                    image_type = image_info.get("type", "output")

                                    # Download the generated image
                                    image_url = f"{comfyui_url}/view"
                                    params = {
                                        "filename": filename,
                                        "subfolder": subfolder,
                                        "type": image_type,
                                    }

                                    image_response = await self.http_client.get(
                                        image_url, params=params, timeout=30.0
                                    )

                                    if image_response.status_code == 200:
                                        image_data = image_response.content

                                        logger.info(
                                            f"✓ Image generated successfully via ComfyUI: {len(image_data)} bytes"
                                        )

                                        return {
                                            "image_data": image_data,
                                            "format": "PNG",
                                            "width": width,
                                            "height": height,
                                            "model": model["name"],
                                            "workflow": "comfyui",
                                            "provider": "local",
                                            "prompt": prompt,
                                            "num_inference_steps": num_inference_steps,
                                            "guidance_scale": guidance_scale,
                                            "seed": seed,
                                            "status": "success",
                                        }

            # Timeout reached
            raise TimeoutError(
                f"ComfyUI generation timed out after {max_wait_time} seconds"
            )

        except (ConnectionError, TimeoutError) as e:
            logger.warning(
                f"ComfyUI generation failed: {str(e)}, attempting fallback to diffusers"
            )
            # Fallback to diffusers if ComfyUI fails
            return await self._fallback_to_diffusers(prompt, **kwargs)
        except Exception as e:
            logger.error(f"ComfyUI generation failed: {str(e)}")
            # Fallback to diffusers for any other errors
            return await self._fallback_to_diffusers(prompt, **kwargs)

    async def _fallback_to_diffusers(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """
        Fallback to diffusers-based image generation when ComfyUI is unavailable.

        Selects the best available diffusers model (SDXL or SD 1.5) and generates
        the image using the local diffusers pipeline.

        Args:
            prompt: Text prompt for generation
            **kwargs: Generation parameters

        Returns:
            Dict with generated image data
        """
        try:
            # Find available diffusers models
            diffusers_models = [
                m
                for m in self.available_models["image"]
                if m.get("inference_engine") == "diffusers"
                and m.get("provider") == "local"
                and (m.get("loaded") or m.get("can_load"))
            ]

            if not diffusers_models:
                logger.error("No diffusers models available for fallback")
                raise ValueError("No image generation models available")

            # Prefer SDXL for quality, fallback to SD 1.5 for speed
            model = None
            for m in diffusers_models:
                if "xl" in m["name"].lower():
                    model = m
                    break

            if not model:
                model = diffusers_models[0]

            logger.info(f"Using fallback model: {model['name']}")

            # Generate using diffusers
            return await self._generate_image_diffusers(prompt, model, **kwargs)

        except Exception as e:
            logger.error(f"Fallback to diffusers failed: {str(e)}")
            raise

    async def _generate_image_diffusers(
        self, prompt: str, model: Dict[str, Any], **kwargs
    ) -> Dict[str, Any]:
        """Generate image using Diffusers library with optional reference image for img2img consistency."""
        # Initialize variables before try block to prevent UnboundLocalError in return statement
        reference_image_path = kwargs.get("reference_image_path")
        use_controlnet = kwargs.get("use_controlnet", False)
        use_img2img = reference_image_path is not None
        img2img_strength = kwargs.get("img2img_strength", 0.75)

        try:
            from diffusers import (
                StableDiffusionPipeline,
                StableDiffusionImg2ImgPipeline,
                StableDiffusionXLPipeline,
                StableDiffusionXLImg2ImgPipeline,
                StableDiffusionControlNetPipeline,
                StableDiffusionXLControlNetPipeline,
                ControlNetModel,
                DPMSolverMultistepScheduler,
                DiffusionPipeline,
            )

            model_name = model["name"]
            model_id = model["model_id"]

            # Determine if this is an SDXL model
            # Check name, model_id, and base_model field (used by CivitAI models)
            base_model = model.get("base_model", "")
            is_sdxl = (
                "xl" in model_name.lower()
                or "xl" in model_id.lower()
                or "sdxl" in base_model.lower()
                or "pony" in base_model.lower()  # Pony models are SDXL-based
            )
            # Use the path from model info if available (handles both formats)
            model_path = Path(
                model.get("path", str(self.models_dir / "image" / model_name))
            )

            # Check if this is a single-file model (e.g., CivitAI .safetensors or .ckpt)
            is_single_file = model_path.is_file() and model_path.suffix.lower() in [
                ".safetensors",
                ".ckpt",
                ".pt",
                ".bin",
            ]

            # Get device ID if specified for multi-GPU support
            device_id = kwargs.get("device_id", None)

            # Determine device (must be done before caching check to avoid UnboundLocalError)
            if torch.cuda.is_available():
                if device_id is not None:
                    device = f"cuda:{device_id}"
                else:
                    device = "cuda"
            else:
                device = "cpu"

            # Track if we're using ControlNet for this generation (must be initialized before pipeline loading)
            using_controlnet = False
            controlnet = None

            # Load or get cached pipeline (with device-specific caching for multi-GPU)
            # Use different cache key for img2img vs text2img pipelines
            pipeline_mode = "img2img" if use_img2img else "text2img"
            if device_id is not None:
                pipeline_key = f"diffusers_{model_name}_{pipeline_mode}_gpu{device_id}"
            else:
                pipeline_key = f"diffusers_{model_name}_{pipeline_mode}"

            if pipeline_key not in self._loaded_pipelines:
                logger.info(
                    f"Loading diffusion model: {model_name} on device {device_id if device_id is not None else 'default'}"
                )

                # Prepare base loading arguments
                load_args = {
                    "torch_dtype": (
                        torch.float16 if "cuda" in device else torch.float32
                    ),
                }

                # Select appropriate pipeline type based on ControlNet, img2img mode and model type
                # For SDXL models, use standard pipeline with dual text encoders:
                # - CLIP ViT-L encoder: 77 tokens
                # - OpenCLIP ViT-G encoder: 77 tokens
                # Total: 154 tokens without truncation
                # Note: For prompts >154 tokens, consider using the 'compel' library (pip install compel)

                if use_controlnet and reference_image_path:
                    # Use ControlNet pipeline for maximum visual consistency
                    # ControlNet provides structural guidance while allowing prompt control
                    try:
                        logger.info("Loading ControlNet for appearance consistency...")

                        # Select appropriate ControlNet model
                        if is_sdxl:
                            # SDXL ControlNet for Canny edge detection (good for pose/structure)
                            controlnet_model_id = "diffusers/controlnet-canny-sdxl-1.0"
                            pipeline_type = "StableDiffusionXLControlNetPipeline"
                        else:
                            # SD 1.5 ControlNet for Canny
                            controlnet_model_id = "lllyasviel/control_v11p_sd15_canny"
                            pipeline_type = "StableDiffusionControlNetPipeline"

                        # Load ControlNet model
                        controlnet = ControlNetModel.from_pretrained(
                            controlnet_model_id,
                            torch_dtype=(
                                torch.float16 if "cuda" in device else torch.float32
                            ),
                        )
                        load_args["controlnet"] = controlnet
                        using_controlnet = True
                        logger.info(f"✓ ControlNet loaded: {controlnet_model_id}")

                    except Exception as e:
                        logger.warning(f"Failed to load ControlNet: {e}")
                        logger.warning("Falling back to img2img pipeline")
                        use_controlnet = False
                        using_controlnet = False

                if not using_controlnet:
                    # Fall back to img2img or text2img pipelines
                    if use_img2img:
                        # Use img2img pipeline for visual consistency with reference image
                        if is_sdxl:
                            pipeline_type = "StableDiffusionXLImg2ImgPipeline"
                            logger.info(
                                f"Using SDXL img2img pipeline for visual consistency"
                            )
                        else:
                            pipeline_type = "StableDiffusionImg2ImgPipeline"
                            logger.info(
                                f"Using SD img2img pipeline for visual consistency"
                            )
                            # Only add safety_checker params for SD 1.5 models (not SDXL)
                            load_args["safety_checker"] = (
                                None  # Disable for performance
                            )
                            load_args["requires_safety_checker"] = (
                                False  # Suppress warning
                            )
                    else:
                        # Use text2img pipeline for standard generation
                        if is_sdxl:
                            # PREFERRED: Use Long Prompt Weighting (lpw) community pipeline for SDXL
                            # This is the recommended solution for long prompts, replacing the older
                            # compel library approach. The lpw pipeline properly handles prompts > 77 tokens
                            # by chunking and merging embeddings from both CLIP encoders.
                            #
                            # Benefits over compel:
                            # - Handles prompts up to 225+ tokens (vs 154 with compel)
                            # - Better weight distribution for long prompts
                            # - Integrated directly into pipeline (no separate embedding step)
                            # - Community-maintained and actively supported
                            pipeline_type = (
                                "StableDiffusionXLPipeline"  # Base type for reference
                            )
                            load_args["custom_pipeline"] = "lpw_stable_diffusion_xl"
                            logger.info(
                                f"Using SDXL Long Prompt Weighting pipeline (lpw_stable_diffusion_xl)"
                            )
                            logger.info(
                                f"✓ Supports prompts > 77 tokens without truncation via prompt chunking"
                            )
                            logger.info(
                                f"✓ Recommended method for long prompts (replaces compel library)"
                            )
                        else:
                            # For SD 1.5, use standard pipeline
                            pipeline_type = "StableDiffusionPipeline"
                            # Only add safety_checker params for SD 1.5 models (not SDXL)
                            load_args["safety_checker"] = (
                                None  # Disable for performance
                            )
                            load_args["requires_safety_checker"] = (
                                False  # Suppress warning
                            )

                logger.info(f"Using pipeline: {pipeline_type} for model {model_name}")

                # Try to load from local path first, fallback to HuggingFace Hub
                if model_path.exists():
                    logger.info(f"Loading model from local path: {model_path}")

                    # Handle single-file models (CivitAI .safetensors, .ckpt files)
                    # These require from_single_file() instead of from_pretrained()
                    if is_single_file:
                        logger.info(f"Detected single-file model: {model_path.name}")
                        logger.info(
                            "Using from_single_file() for CivitAI/checkpoint model"
                        )

                        # Prepare single-file loading arguments
                        single_file_args = {
                            "torch_dtype": (
                                torch.float16 if "cuda" in device else torch.float32
                            ),
                        }

                        try:
                            if is_sdxl:
                                # Use SDXL pipeline for SDXL-based models
                                pipe = StableDiffusionXLPipeline.from_single_file(
                                    str(model_path), **single_file_args
                                )
                                logger.info(
                                    f"✅ Successfully loaded SDXL model from single file: {model_path.name}"
                                )
                            else:
                                # Use SD 1.5 pipeline for non-SDXL models
                                pipe = StableDiffusionPipeline.from_single_file(
                                    str(model_path), **single_file_args
                                )
                                logger.info(
                                    f"✅ Successfully loaded SD model from single file: {model_path.name}"
                                )
                        except Exception as e:
                            logger.error(f"Failed to load single-file model: {str(e)}")
                            raise

                    # Try to load with fp16 variant first for SDXL models (directory-based)
                    # If variant files are not available, fallback to default loading
                    elif is_sdxl and "cuda" in device:
                        load_args_fp16 = load_args.copy()
                        load_args_fp16["variant"] = "fp16"
                        load_args_fp16["use_safetensors"] = True

                        # Try with custom pipeline first, fallback to standard if it fails
                        custom_pipeline = load_args_fp16.get("custom_pipeline")
                        try:
                            pipe = DiffusionPipeline.from_pretrained(
                                str(model_path), **load_args_fp16
                            )
                            if custom_pipeline:
                                logger.info(
                                    f"✅ Successfully loaded SDXL Long Prompt Weighting pipeline ({custom_pipeline})"
                                )
                                logger.info(
                                    "   Supports prompts > 77 tokens via chunking and embedding merge"
                                )
                            else:
                                logger.info(
                                    "✅ Successfully loaded SDXL pipeline with dual text encoders"
                                )
                        except (ValueError, OSError, Exception) as e:
                            if custom_pipeline and "custom_pipeline" in str(e).lower():
                                # Custom pipeline failed, fallback to standard pipeline
                                logger.warning(
                                    f"Long Prompt Weighting pipeline not available: {e}"
                                )
                                logger.warning(
                                    "Falling back to standard SDXL pipeline with compel support"
                                )
                                load_args_fp16_fallback = load_args_fp16.copy()
                                load_args_fp16_fallback.pop("custom_pipeline", None)
                                pipe = DiffusionPipeline.from_pretrained(
                                    str(model_path), **load_args_fp16_fallback
                                )
                                logger.info(
                                    "✅ Successfully loaded standard SDXL pipeline (fallback)"
                                )
                            elif (
                                "fp16" in str(e).lower() or "variant" in str(e).lower()
                            ):
                                # fp16 variant not available
                                logger.warning(
                                    f"fp16 variant not available, loading without variant: {e}"
                                )
                                load_args_no_variant = load_args.copy()
                                pipe = DiffusionPipeline.from_pretrained(
                                    str(model_path), **load_args_no_variant
                                )
                                if custom_pipeline:
                                    logger.info(
                                        f"✅ Successfully loaded SDXL Long Prompt Weighting pipeline ({custom_pipeline})"
                                    )
                                else:
                                    logger.info("✅ Successfully loaded SDXL pipeline")
                            else:
                                raise
                    else:
                        # CPU or non-CUDA device
                        custom_pipeline = load_args.get("custom_pipeline")
                        try:
                            pipe = DiffusionPipeline.from_pretrained(
                                str(model_path), **load_args
                            )
                            if is_sdxl:
                                if custom_pipeline:
                                    logger.info(
                                        f"✅ Successfully loaded SDXL Long Prompt Weighting pipeline ({custom_pipeline})"
                                    )
                                else:
                                    logger.info(
                                        "✅ Successfully loaded SDXL pipeline with dual text encoders"
                                    )
                        except Exception as e:
                            if custom_pipeline and "custom_pipeline" in str(e).lower():
                                # Custom pipeline failed, fallback to standard
                                logger.warning(
                                    f"Long Prompt Weighting pipeline not available: {e}"
                                )
                                logger.warning("Falling back to standard SDXL pipeline")
                                load_args_fallback = load_args.copy()
                                load_args_fallback.pop("custom_pipeline", None)
                                pipe = DiffusionPipeline.from_pretrained(
                                    str(model_path), **load_args_fallback
                                )
                                logger.info(
                                    "✅ Successfully loaded standard SDXL pipeline (fallback)"
                                )
                            else:
                                raise
                else:
                    logger.info(f"Loading model from HuggingFace Hub: {model_id}")

                    # Try to load with fp16 variant first for SDXL models
                    # If variant files are not available, fallback to default loading
                    if is_sdxl and "cuda" in device:
                        load_args_fp16 = load_args.copy()
                        load_args_fp16["variant"] = "fp16"
                        load_args_fp16["use_safetensors"] = True

                        # Try with custom pipeline first, fallback to standard if it fails
                        custom_pipeline = load_args_fp16.get("custom_pipeline")
                        try:
                            pipe = DiffusionPipeline.from_pretrained(
                                model_id, **load_args_fp16
                            )
                            if custom_pipeline:
                                logger.info(
                                    f"✅ Successfully loaded SDXL Long Prompt Weighting pipeline ({custom_pipeline})"
                                )
                                logger.info(
                                    "   Supports prompts > 77 tokens via chunking and embedding merge"
                                )
                            else:
                                logger.info(
                                    "✅ Successfully loaded SDXL pipeline with dual text encoders"
                                )
                        except (ValueError, OSError, Exception) as e:
                            if custom_pipeline and "custom_pipeline" in str(e).lower():
                                # Custom pipeline failed, fallback to standard pipeline
                                logger.warning(
                                    f"Long Prompt Weighting pipeline not available: {e}"
                                )
                                logger.warning(
                                    "Falling back to standard SDXL pipeline with compel support"
                                )
                                load_args_fp16_fallback = load_args_fp16.copy()
                                load_args_fp16_fallback.pop("custom_pipeline", None)
                                pipe = DiffusionPipeline.from_pretrained(
                                    model_id, **load_args_fp16_fallback
                                )
                                logger.info(
                                    "✅ Successfully loaded standard SDXL pipeline (fallback)"
                                )
                            elif (
                                "fp16" in str(e).lower() or "variant" in str(e).lower()
                            ):
                                # fp16 variant not available
                                logger.warning(
                                    f"fp16 variant not available, loading without variant: {e}"
                                )
                                load_args_no_variant = load_args.copy()
                                pipe = DiffusionPipeline.from_pretrained(
                                    model_id, **load_args_no_variant
                                )
                                if custom_pipeline:
                                    logger.info(
                                        f"✅ Successfully loaded SDXL Long Prompt Weighting pipeline ({custom_pipeline})"
                                    )
                                else:
                                    logger.info("✅ Successfully loaded SDXL pipeline")
                            else:
                                raise
                    else:
                        # CPU or non-CUDA device
                        custom_pipeline = load_args.get("custom_pipeline")
                        try:
                            pipe = DiffusionPipeline.from_pretrained(
                                model_id, **load_args
                            )
                            if is_sdxl:
                                if custom_pipeline:
                                    logger.info(
                                        f"✅ Successfully loaded SDXL Long Prompt Weighting pipeline ({custom_pipeline})"
                                    )
                                else:
                                    logger.info(
                                        "✅ Successfully loaded SDXL pipeline with dual text encoders"
                                    )
                        except Exception as e:
                            if custom_pipeline and "custom_pipeline" in str(e).lower():
                                # Custom pipeline failed, fallback to standard
                                logger.warning(
                                    f"Long Prompt Weighting pipeline not available: {e}"
                                )
                                logger.warning("Falling back to standard SDXL pipeline")
                                load_args_fallback = load_args.copy()
                                load_args_fallback.pop("custom_pipeline", None)
                                pipe = DiffusionPipeline.from_pretrained(
                                    model_id, **load_args_fallback
                                )
                                logger.info(
                                    "✅ Successfully loaded standard SDXL pipeline (fallback)"
                                )
                            else:
                                raise
                    # Save to local path for future use
                    if not model_path.exists():
                        model_path.mkdir(parents=True, exist_ok=True)
                        pipe.save_pretrained(str(model_path))
                        logger.info(f"Model saved to: {model_path}")

                # Use DPM-Solver++ for faster inference
                # Enable Karras sigmas to stabilize step index calculation
                pipe.scheduler = DPMSolverMultistepScheduler.from_config(
                    pipe.scheduler.config,
                    use_karras_sigmas=True,
                )
                pipe = pipe.to(device)

                # Enable memory optimizations
                if "cuda" in device:
                    pipe.enable_attention_slicing()
                    # Try to enable xformers if available
                    try:
                        pipe.enable_xformers_memory_efficient_attention()
                        logger.info("✓ xformers memory efficient attention enabled")
                    except Exception as e:
                        logger.warning(
                            f"xformers not available, using default attention: {e}"
                        )
                        logger.info(
                            "To enable xformers for faster inference: pip install xformers"
                        )
                        # Fallback to PyTorch's scaled_dot_product_attention if available (PyTorch 2.0+)
                        try:
                            # Check PyTorch version
                            pytorch_version = tuple(
                                int(x) for x in torch.__version__.split(".")[:2]
                            )
                            if pytorch_version >= (2, 0):
                                logger.info(
                                    "Using PyTorch 2.0+ scaled_dot_product_attention as fallback"
                                )
                        except Exception:
                            pass

                self._loaded_pipelines[pipeline_key] = pipe
                logger.info(f"Model {model_name} loaded successfully on {device}")

            pipe = self._loaded_pipelines[pipeline_key]

            # Validate pipeline components are properly loaded
            if is_sdxl:
                # SDXL requires both text encoders
                if pipe.text_encoder is None or pipe.text_encoder_2 is None:
                    error_msg = f"SDXL pipeline has None text encoders: text_encoder={pipe.text_encoder is not None}, text_encoder_2={pipe.text_encoder_2 is not None}"
                    logger.error(error_msg)
                    # Remove broken pipeline from cache
                    del self._loaded_pipelines[pipeline_key]
                    raise ValueError(error_msg)
                if pipe.tokenizer is None or pipe.tokenizer_2 is None:
                    error_msg = f"SDXL pipeline has None tokenizers: tokenizer={pipe.tokenizer is not None}, tokenizer_2={pipe.tokenizer_2 is not None}"
                    logger.error(error_msg)
                    # Remove broken pipeline from cache
                    del self._loaded_pipelines[pipeline_key]
                    raise ValueError(error_msg)
            else:
                # SD 1.5/2.x requires single text encoder
                if pipe.text_encoder is None:
                    error_msg = f"SD pipeline has None text_encoder"
                    logger.error(error_msg)
                    # Remove broken pipeline from cache
                    del self._loaded_pipelines[pipeline_key]
                    raise ValueError(error_msg)
                if pipe.tokenizer is None:
                    error_msg = f"SD pipeline has None tokenizer"
                    logger.error(error_msg)
                    # Remove broken pipeline from cache
                    del self._loaded_pipelines[pipeline_key]
                    raise ValueError(error_msg)

            # Get generation parameters
            num_inference_steps = kwargs.get("num_inference_steps", 25)
            guidance_scale = kwargs.get("guidance_scale", 7.5)

            # Parse size parameter if provided (e.g., "1024x1024")
            # Otherwise use explicit width/height or defaults
            size = kwargs.get("size")
            if size and isinstance(size, str) and "x" in size:
                try:
                    width_str, height_str = size.split("x")
                    width = int(width_str)
                    height = int(height_str)
                    logger.info(
                        f"Parsed size parameter: {size} -> width={width}, height={height}"
                    )
                except (ValueError, AttributeError) as e:
                    logger.warning(
                        f"Failed to parse size parameter '{size}': {e}, using defaults"
                    )
                    width = kwargs.get("width", 512)
                    height = kwargs.get("height", 512)
            else:
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

            # Handle ControlNet and img2img mode with reference image
            init_image = None
            control_image = None

            # Process reference image for ControlNet if enabled
            if using_controlnet and reference_image_path:
                try:
                    logger.info(
                        f"🎨 ControlNet mode enabled with reference: {reference_image_path}"
                    )

                    # Load reference image
                    reference_path = Path(reference_image_path)
                    if not reference_path.exists():
                        logger.warning(
                            f"Reference image not found: {reference_image_path}"
                        )
                        logger.warning("Falling back to standard generation")
                        using_controlnet = False
                        use_controlnet = False
                    else:
                        from PIL import Image
                        import cv2
                        import numpy as np

                        # Load and prepare reference image
                        ref_image = Image.open(reference_path).convert("RGB")
                        ref_image = ref_image.resize(
                            (width, height), Image.Resampling.LANCZOS
                        )

                        # Extract Canny edges for structural guidance
                        # Convert to numpy array for OpenCV
                        image_np = np.array(ref_image)

                        # Apply Canny edge detection
                        # Lower threshold = 100, upper threshold = 200
                        edges = cv2.Canny(image_np, 100, 200)

                        # Convert back to PIL Image (3-channel for ControlNet)
                        edges_rgb = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
                        control_image = Image.fromarray(edges_rgb)

                        logger.info(
                            f"   ✓ ControlNet conditioning image prepared (Canny edges)"
                        )
                        logger.info(
                            f"   This will maintain pose/structure while allowing prompt control"
                        )

                except Exception as e:
                    logger.error(f"Failed to prepare ControlNet image: {str(e)}")
                    logger.warning("Falling back to img2img generation")
                    using_controlnet = False
                    use_controlnet = False
                    control_image = None

            if use_img2img and not using_controlnet:
                try:
                    logger.info(
                        f"🖼️  Image-to-image mode enabled with reference: {reference_image_path}"
                    )
                    logger.info(
                        f"   Strength: {img2img_strength} (0.0=exact copy, 1.0=full regeneration)"
                    )

                    # Load reference image
                    reference_path = Path(reference_image_path)
                    if not reference_path.exists():
                        logger.warning(
                            f"Reference image not found: {reference_image_path}"
                        )
                        logger.warning("Falling back to text-to-image generation")
                        use_img2img = False
                    else:
                        init_image = Image.open(reference_path).convert("RGB")
                        # Resize to target dimensions if needed
                        init_image = init_image.resize(
                            (width, height), Image.Resampling.LANCZOS
                        )
                        logger.info(f"   ✓ Reference image loaded: {init_image.size}")
                except Exception as e:
                    logger.error(f"Failed to load reference image: {str(e)}")
                    logger.warning("Falling back to text-to-image generation")
                    use_img2img = False
                    init_image = None

            # Log detailed diagnostics before generation
            logger.info("=" * 60)
            logger.info("DIFFUSERS GENERATION - DIAGNOSTIC INFO")
            logger.info("=" * 60)
            logger.info(f"Model: {model_name} (SDXL={is_sdxl})")
            logger.info(f"Pipeline class: {type(pipe).__name__}")
            logger.info(f"Device: {device}")
            logger.info(f"Pipeline components:")
            logger.info(f"  - vae: {type(pipe.vae).__name__ if pipe.vae else 'None'}")
            logger.info(
                f"  - text_encoder: {type(pipe.text_encoder).__name__ if pipe.text_encoder else 'None'}"
            )
            if is_sdxl:
                logger.info(
                    f"  - text_encoder_2: {type(pipe.text_encoder_2).__name__ if pipe.text_encoder_2 else 'None'}"
                )
            logger.info(
                f"  - tokenizer: {type(pipe.tokenizer).__name__ if pipe.tokenizer else 'None'}"
            )
            if is_sdxl:
                logger.info(
                    f"  - tokenizer_2: {type(pipe.tokenizer_2).__name__ if pipe.tokenizer_2 else 'None'}"
                )
            logger.info(
                f"  - unet: {type(pipe.unet).__name__ if pipe.unet else 'None'}"
            )
            logger.info(
                f"  - scheduler: {type(pipe.scheduler).__name__ if pipe.scheduler else 'None'}"
            )
            logger.info(f"Generation parameters:")
            logger.info(f"  - prompt: {prompt}")
            logger.info(f"  - negative_prompt: {negative_prompt}")
            logger.info(f"  - num_inference_steps: {num_inference_steps}")
            logger.info(f"  - guidance_scale: {guidance_scale}")
            logger.info(f"  - width: {width}")
            logger.info(f"  - height: {height}")
            logger.info(f"  - seed: {seed}")
            logger.info("=" * 60)

            # Helper function to safely extract image from pipeline result
            def safe_extract_image(result):
                """
                Safely extract the first image from a diffusers pipeline result.

                Args:
                    result: The pipeline result object

                Returns:
                    PIL.Image: The first generated image

                Raises:
                    ValueError: If result is None or doesn't contain valid images
                """
                if result is None:
                    raise ValueError(
                        "Pipeline returned None. This may indicate a pipeline configuration error "
                        "or an issue with the model loading."
                    )

                if not hasattr(result, "images"):
                    raise ValueError(
                        f"Pipeline result does not have 'images' attribute. "
                        f"Result type: {type(result).__name__}"
                    )

                if result.images is None:
                    raise ValueError(
                        "Pipeline result.images is None. This may indicate a generation failure."
                    )

                if not result.images:
                    raise ValueError(
                        "Pipeline result.images is empty. No images were generated."
                    )

                return result.images[0]

            # Create a fresh scheduler for this generation to prevent state accumulation
            # Schedulers are stateful and cannot be shared across concurrent runs
            # Without this, step_index accumulates across requests causing index errors
            pipe.scheduler = DPMSolverMultistepScheduler.from_config(
                pipe.scheduler.config,
                use_karras_sigmas=True,
            )

            # Handle long prompts (>77 tokens) with compel library for SDXL
            # This allows prompts exceeding the CLIP token limit without truncation
            prompt_embeds = None
            negative_prompt_embeds = None
            pooled_prompt_embeds = None
            negative_pooled_prompt_embeds = None

            # Preserve original prompts for return value
            original_prompt = prompt
            original_negative_prompt = negative_prompt

            # SDXL supports longer prompts through compel
            # Standard SDXL has 2 CLIP encoders with 77 tokens each
            # Without compel, prompts are truncated at 77 tokens per encoder
            # With compel, we can handle much longer prompts (225+ tokens)
            # NOTE: The lpw_stable_diffusion_xl custom pipeline handles long prompts internally
            # and expects text prompts, not embeddings. Skip compel for lpw pipelines.
            # We use string matching on class name since it's the most reliable way to detect
            # the community pipeline (SDXLLongPromptWeightingPipeline) without dependencies.
            pipeline_class_name = type(pipe).__name__
            is_lpw_pipeline = "LongPromptWeighting" in pipeline_class_name

            if is_sdxl and not is_lpw_pipeline:
                # Fallback to compel for long prompt support when lpw pipeline is not available
                # Note: lpw_stable_diffusion_xl (Long Prompt Weighting) is the preferred method
                # and is automatically used when available. This compel fallback handles cases
                # where lpw pipeline failed to load.
                #
                # Estimate token count (rough approximation: 1.3 tokens per word)
                estimated_tokens = len(prompt.split()) * 1.3

                # Use compel for prompts that would be truncated (>75 tokens allows margin)
                # Note: SDXL dual encoders can handle ~154 tokens without compel,
                # but they still truncate at 77 tokens per encoder. Compel merges them properly.
                if estimated_tokens > 75:
                    try:
                        # Try to use compel for long prompt handling as fallback
                        from compel import Compel, ReturnedEmbeddingsType

                        logger.info(
                            f"Using compel (fallback) for long prompt support (~{int(estimated_tokens)} tokens)"
                        )
                        logger.info(
                            "Note: For better long prompt support, ensure lpw_stable_diffusion_xl pipeline is available"
                        )

                        # Create compel instance for SDXL
                        compel = Compel(
                            tokenizer=[pipe.tokenizer, pipe.tokenizer_2],
                            text_encoder=[pipe.text_encoder, pipe.text_encoder_2],
                            returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED,
                            requires_pooled=[False, True],
                            device=device,
                        )

                        # Generate embeddings for prompt and negative prompt
                        conditioning, pooled = compel(prompt)
                        negative_conditioning, negative_pooled = compel(negative_prompt)

                        # Validate that all embeddings are not None
                        # Note: compel may return None embeddings with certain configurations
                        if (
                            conditioning is None
                            or pooled is None
                            or negative_conditioning is None
                            or negative_pooled is None
                        ):
                            logger.warning(
                                f"Compel returned None embeddings: conditioning={conditioning is not None}, "
                                f"pooled={pooled is not None}, negative_conditioning={negative_conditioning is not None}, "
                                f"negative_pooled={negative_pooled is not None}"
                            )
                            logger.warning(
                                "Falling back to standard prompt encoding (will truncate at 77 tokens)"
                            )
                            # Don't set embeddings if any are None - will use text prompts instead
                        else:
                            # Set the embeddings to use
                            prompt_embeds = conditioning
                            pooled_prompt_embeds = pooled
                            negative_prompt_embeds = negative_conditioning
                            negative_pooled_prompt_embeds = negative_pooled

                            # Clear text prompts since we're using embeddings (but keep originals for return)
                            prompt = None
                            negative_prompt = None

                            logger.info(
                                "✓ Long prompt encoded successfully with compel (fallback method)"
                            )

                    except ImportError:
                        logger.warning(
                            "compel library not available. Install with: pip install compel"
                        )
                        logger.warning(
                            f"Prompt may be truncated to CLIP's token limit (prompt has ~{int(estimated_tokens)} tokens)"
                        )
                        logger.warning(
                            "Recommended: Use lpw_stable_diffusion_xl pipeline for better long prompt support"
                        )
                    except Exception as e:
                        logger.warning(f"Failed to use compel: {e}")
                        logger.warning("Falling back to standard prompt encoding")
            elif is_lpw_pipeline:
                # lpw_stable_diffusion_xl pipeline handles long prompts internally
                # It expects text prompts, not embeddings
                estimated_tokens = len(prompt.split()) * 1.3
                logger.info(
                    f"Using lpw_stable_diffusion_xl pipeline for long prompt support (~{int(estimated_tokens)} tokens)"
                )
                logger.info("✓ lpw pipeline will handle prompt chunking internally")

            # Generate image (run in thread pool to avoid blocking)
            # Use different parameters for ControlNet, img2img vs text2img
            try:
                loop = asyncio.get_event_loop()

                if using_controlnet and control_image:
                    # ControlNet generation with structural conditioning
                    logger.info(f"Generating with ControlNet conditioning")

                    # ControlNet conditioning strength (how much to follow the structure)
                    controlnet_conditioning_scale = kwargs.get(
                        "controlnet_conditioning_scale", 0.8
                    )

                    # Use embeddings if available (from compel for long prompts), otherwise use text prompts
                    # For SDXL, we need all 4 embeddings (including pooled) to be non-None
                    if (
                        prompt_embeds is not None
                        and negative_prompt_embeds is not None
                        and pooled_prompt_embeds is not None
                        and negative_pooled_prompt_embeds is not None
                        and is_sdxl
                    ):
                        # Using compel embeddings for long prompt support with ControlNet
                        result = await loop.run_in_executor(
                            None,
                            lambda: pipe(
                                prompt_embeds=prompt_embeds,
                                negative_prompt_embeds=negative_prompt_embeds,
                                pooled_prompt_embeds=pooled_prompt_embeds,
                                negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
                                image=control_image,  # ControlNet conditioning image (Canny edges)
                                num_inference_steps=num_inference_steps,
                                guidance_scale=guidance_scale,
                                controlnet_conditioning_scale=controlnet_conditioning_scale,
                                width=width,
                                height=height,
                                generator=generator,
                            ),
                        )
                        image = safe_extract_image(result)
                        logger.info(
                            "✓ Image generated successfully via ControlNet with compel embeddings"
                        )
                    else:
                        # Using standard text prompts
                        result = await loop.run_in_executor(
                            None,
                            lambda: pipe(
                                prompt=original_prompt,  # Use original text prompt
                                negative_prompt=original_negative_prompt,
                                image=control_image,  # ControlNet conditioning image (Canny edges)
                                num_inference_steps=num_inference_steps,
                                guidance_scale=guidance_scale,
                                controlnet_conditioning_scale=controlnet_conditioning_scale,
                                width=width,
                                height=height,
                                generator=generator,
                            ),
                        )
                        image = safe_extract_image(result)
                        logger.info("✓ Image generated successfully via ControlNet")
                elif use_img2img and init_image:
                    # img2img generation with reference image
                    logger.info(f"Generating img2img with strength={img2img_strength}")

                    # Use embeddings if available (from compel for long prompts), otherwise use text prompts
                    # For SDXL, we need all 4 embeddings (including pooled) to be non-None
                    if (
                        prompt_embeds is not None
                        and negative_prompt_embeds is not None
                        and pooled_prompt_embeds is not None
                        and negative_pooled_prompt_embeds is not None
                        and is_sdxl
                    ):
                        # Using compel embeddings for long prompt support with img2img
                        result = await loop.run_in_executor(
                            None,
                            lambda: pipe(
                                prompt_embeds=prompt_embeds,
                                negative_prompt_embeds=negative_prompt_embeds,
                                pooled_prompt_embeds=pooled_prompt_embeds,
                                negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
                                image=init_image,
                                strength=img2img_strength,
                                num_inference_steps=num_inference_steps,
                                guidance_scale=guidance_scale,
                                generator=generator,
                            ),
                        )
                        image = safe_extract_image(result)
                        logger.info(
                            "✓ Image generated successfully via img2img with compel embeddings"
                        )
                    else:
                        # Using standard text prompts
                        result = await loop.run_in_executor(
                            None,
                            lambda: pipe(
                                prompt=original_prompt,  # Use original prompt for img2img
                                image=init_image,
                                strength=img2img_strength,
                                negative_prompt=original_negative_prompt,
                                num_inference_steps=num_inference_steps,
                                guidance_scale=guidance_scale,
                                generator=generator,
                            ),
                        )
                        image = safe_extract_image(result)
                        logger.info("✓ Image generated successfully via img2img")
                else:
                    # Standard text2img generation
                    # Use embeddings if available (from compel), otherwise use text prompts
                    # For SDXL models using embeddings, we need all 4 embeddings (including pooled) to be non-None
                    if (
                        prompt_embeds is not None
                        and negative_prompt_embeds is not None
                        and pooled_prompt_embeds is not None
                        and negative_pooled_prompt_embeds is not None
                    ):
                        # Using compel embeddings for long prompt support
                        result = await loop.run_in_executor(
                            None,
                            lambda: pipe(
                                prompt_embeds=prompt_embeds,
                                negative_prompt_embeds=negative_prompt_embeds,
                                pooled_prompt_embeds=pooled_prompt_embeds,
                                negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
                                num_inference_steps=num_inference_steps,
                                guidance_scale=guidance_scale,
                                width=width,
                                height=height,
                                generator=generator,
                            ),
                        )
                        image = safe_extract_image(result)
                        logger.info(
                            "✓ Image generated successfully via text2img with compel embeddings"
                        )
                    else:
                        # Using standard text prompts
                        result = await loop.run_in_executor(
                            None,
                            lambda: pipe(
                                prompt=prompt,
                                negative_prompt=negative_prompt,
                                num_inference_steps=num_inference_steps,
                                guidance_scale=guidance_scale,
                                width=width,
                                height=height,
                                generator=generator,
                            ),
                        )
                        image = safe_extract_image(result)
                        logger.info("✓ Image generated successfully via text2img")
            except Exception as e:
                logger.error(f"Diffusers generation failed: {str(e)}")
                logger.error(f"Error type: {type(e).__name__}")
                import traceback

                logger.error(f"Full traceback:\n{traceback.format_exc()}")
                raise

            # Convert PIL Image to bytes
            img_byte_arr = io.BytesIO()
            image.save(img_byte_arr, format="PNG")

            # Get the buffer position to check if all data was written
            buffer_position = img_byte_arr.tell()
            logger.debug(f"BytesIO buffer position after save: {buffer_position}")

            # Get the actual bytes
            image_data = img_byte_arr.getvalue()

            logger.info(
                f"Image generated successfully: {len(image_data)} bytes "
                f"(buffer position: {buffer_position}, match: {len(image_data) == buffer_position})"
            )

            # Additional sanity check - verify we can load the image from bytes
            try:
                verify_img = Image.open(io.BytesIO(image_data))
                logger.debug(
                    f"Image bytes verification: {verify_img.size} {verify_img.mode}"
                )
            except Exception as e:
                logger.error(f"⚠️  Generated image bytes are corrupted: {e}")

            return {
                "image_data": image_data,
                "format": "PNG",
                "width": width,
                "height": height,
                "model": model_name,
                "library": "diffusers",
                "provider": "local",
                "prompt": original_prompt,  # Use preserved original prompt
                "negative_prompt": original_negative_prompt,  # Use preserved original negative prompt
                "num_inference_steps": num_inference_steps,
                "guidance_scale": guidance_scale,
                "seed": seed,
                "reference_image_used": bool(reference_image_path),
                "controlnet_used": using_controlnet,
                "img2img_mode": use_img2img and not using_controlnet,
                "img2img_strength": (
                    img2img_strength if (use_img2img and not using_controlnet) else None
                ),
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
            logger.info(f"   📡 Calling OpenAI API ({model})...")
            logger.info(
                f"   Request: max_tokens={kwargs.get('max_tokens', 1000)}, temp={kwargs.get('temperature', 0.7)}"
            )

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
                logger.error(f"   ❌ OpenAI API returned error: {response.status_code}")
                logger.error(f"   Response: {response.text[:200]}")
                raise Exception(f"OpenAI API error: {response.text}")

            result = response.json()
            generated_text = result["choices"][0]["message"]["content"].strip()

            # Log the actual AI output
            logger.info(f"   📄 AI OUTPUT (OpenAI {model}):")
            logger.info(f"   " + "-" * 76)
            for line in generated_text.split("\n"):
                logger.info(f"   {line}")
            logger.info(f"   " + "-" * 76)
            logger.info(
                f"   Tokens used: {result.get('usage', {}).get('total_tokens', 'unknown')}"
            )

            return generated_text

        except Exception as e:
            logger.error(f"   ❌ OpenAI text generation failed: {str(e)}")
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

    def _truncate_prompt_for_clip(self, prompt: str, max_tokens: int = 75) -> str:
        """
        Truncate prompt to fit within CLIP's 77 token limit (leaving 2 tokens for special tokens).

        NOTE: This is only used for SD 1.5 models as a fallback. SDXL models use
        StableDiffusionXLLongPromptWeightingPipeline which properly handles prompts
        longer than 77 tokens without truncation.

        CLIP tokenizer has a hard limit of 77 tokens. We use 75 to be safe.
        This function intelligently truncates the prompt while preserving key details.

        Args:
            prompt: Full prompt text
            max_tokens: Maximum number of tokens (default 75 to leave room for special tokens)

        Returns:
            Truncated prompt that fits within token limit
        """
        # Simple word-based approximation (1 token ≈ 0.75 words for English)
        # This is a safe heuristic that slightly underestimates to prevent truncation
        estimated_tokens = len(prompt.split()) * 1.3

        if estimated_tokens <= max_tokens:
            return prompt

        # Need to truncate - keep the most important parts
        # Priority: main subject description > style qualifiers > technical details
        words = prompt.split()
        target_words = int(max_tokens / 1.3)  # Convert tokens back to words

        if len(words) <= target_words:
            return prompt

        # Take the most important words from the beginning and essential style words
        truncated = " ".join(words[:target_words])
        logger.warning(
            f"Prompt truncated from {len(words)} to {target_words} words to fit CLIP's 77 token limit"
        )
        logger.debug(f"Original: {prompt[:100]}...")
        logger.debug(f"Truncated: {truncated[:100]}...")

        return truncated

    def _build_style_specific_prompt(
        self,
        base_prompt: str,
        image_style: str = "photorealistic",
        use_long_prompt: bool = True,
    ) -> tuple[str, str]:
        """
        Build style-specific prompts and negative prompts for image generation.

        For SDXL models, returns full prompts without truncation when use_long_prompt=True,
        as StableDiffusionXLLongPromptWeightingPipeline handles prompts > 77 tokens.
        For SD 1.5 models, truncates to fit CLIP's 77 token limit.

        Args:
            base_prompt: Base appearance/character description
            image_style: Style identifier (photorealistic, anime, cartoon, etc.)
            use_long_prompt: If True, don't truncate (for SDXL Long Prompt Pipeline)

        Returns:
            Tuple of (enhanced_prompt, negative_prompt)
        """
        style_configs = {
            "photorealistic": {
                "prefix": "professional photograph, highly detailed, lifelike, photorealistic,",
                "suffix": "8k uhd, dslr, soft lighting, high quality, film grain, Fujifilm XT3",
                "negative": "cartoon, anime, 3d render, illustration, painting, drawing, art, sketched, ugly, blurry, low quality, distorted, deformed, bad anatomy",
            },
            "anime": {
                "prefix": "anime style, manga art, highly detailed anime character,",
                "suffix": "beautiful anime art, vibrant colors, studio anime, key visual, trending on pixiv",
                "negative": "realistic, photorealistic, 3d, ugly, blurry, low quality, distorted, deformed, bad anatomy, western cartoon",
            },
            "cartoon": {
                "prefix": "cartoon illustration, stylized character design, vector art style,",
                "suffix": "clean lines, vibrant colors, professional cartoon art, high quality illustration",
                "negative": "realistic, photorealistic, anime, 3d render, ugly, blurry, low quality, distorted, deformed, bad anatomy",
            },
            "artistic": {
                "prefix": "artistic painting, oil painting style, fine art portrait,",
                "suffix": "masterpiece, award winning, museum quality, painterly, expressive brushstrokes",
                "negative": "photograph, 3d render, anime, cartoon, ugly, blurry, low quality, distorted, deformed",
            },
            "3d_render": {
                "prefix": "3d render, cgi character, digital art, octane render,",
                "suffix": "highly detailed 3d model, professional 3d render, unreal engine, ray tracing",
                "negative": "photograph, 2d, anime, cartoon, flat, ugly, low quality, distorted, deformed, bad topology",
            },
            "fantasy": {
                "prefix": "fantasy art style, epic fantasy character, digital fantasy painting,",
                "suffix": "trending on artstation, fantasy character concept, dramatic lighting, magical atmosphere",
                "negative": "realistic photograph, modern, mundane, ugly, blurry, low quality, distorted, deformed",
            },
            "cinematic": {
                "prefix": "cinematic portrait, movie still, film photography,",
                "suffix": "dramatic lighting, cinematic composition, film grain, anamorphic lens, movie quality",
                "negative": "cartoon, anime, illustration, amateur, ugly, blurry, low quality, distorted, deformed",
            },
        }

        # Get style config or default to photorealistic
        style_key = image_style.lower() if image_style else "photorealistic"
        config = style_configs.get(style_key, style_configs["photorealistic"])

        # Build enhanced prompt
        enhanced_prompt = f"{config['prefix']} {base_prompt}, {config['suffix']}"

        # For SDXL with Long Prompt Pipeline, don't truncate
        # For SD 1.5 or fallback mode, truncate to fit CLIP's 77 token limit
        if use_long_prompt:
            logger.debug(
                f"Using full prompt for SDXL Long Prompt Pipeline (no truncation)"
            )
            return enhanced_prompt, config["negative"]
        else:
            truncated_prompt = self._truncate_prompt_for_clip(enhanced_prompt)
            truncated_negative = self._truncate_prompt_for_clip(config["negative"])
            return truncated_prompt, truncated_negative

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
        draft images. Automatically selects least loaded GPU if device_id
        is not specified.

        Args:
            appearance_prompt: Detailed appearance description
            personality_context: Optional personality traits
            reference_image_path: Optional draft image for ControlNet refinement
            **kwargs: Additional generation parameters including:
                - device_id (int, optional): Specific GPU to use. If not provided,
                  automatically selects the least loaded GPU.

        Returns:
            Dict with image_data, format, width, height, and metadata

        Raises:
            Exception: If generation fails
        """
        try:
            # Get image style from kwargs
            image_style = kwargs.get("image_style", "photorealistic")

            # Build base prompt with appearance and personality
            base_prompt = appearance_prompt
            if personality_context:
                base_prompt += f", expressing {personality_context}"

            # Check if we're using SDXL to determine if we can use long prompts
            model = self._get_best_local_image_model()
            is_sdxl = "xl" in model.get("name", "").lower()

            # Build style-specific prompt and negative prompt
            # For SDXL, use full prompts (Long Prompt Pipeline handles > 77 tokens)
            # For SD 1.5, truncate to fit CLIP's limit
            full_prompt, style_negative_prompt = self._build_style_specific_prompt(
                base_prompt, image_style, use_long_prompt=is_sdxl
            )

            logger.info(
                f"Generating reference image locally with style '{image_style}': {full_prompt[:100]}..."
            )

            # Auto-select GPU if not specified
            device_id = kwargs.get("device_id")
            if device_id is None:
                from backend.services.gpu_monitoring_service import (
                    get_gpu_monitoring_service,
                )

                gpu_service = get_gpu_monitoring_service()
                device_id = await gpu_service.get_least_loaded_gpu()
                if device_id is not None:
                    logger.info(f"Auto-selected GPU {device_id} based on utilization")

            # Use high-resolution parameters for reference images
            generation_kwargs = {
                "width": kwargs.get("width", 1024),
                "height": kwargs.get("height", 1024),
                "num_inference_steps": kwargs.get(
                    "num_inference_steps", 50
                ),  # Higher quality
                "guidance_scale": kwargs.get("guidance_scale", 8.0),
                "negative_prompt": kwargs.get(
                    "negative_prompt", style_negative_prompt
                ),  # Use style-specific negative prompt
                "seed": kwargs.get("seed"),
                "device_id": device_id,  # Pass through the selected GPU
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
