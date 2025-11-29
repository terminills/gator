"""
Model loading utilities and helpers.

Provides common functionality for downloading, verifying, and loading
AI models from various sources (HuggingFace, CivitAI, local paths).
"""

import asyncio
import os
import re
from pathlib import Path
from typing import Any, Dict, Optional

import httpx

from backend.config.logging import get_logger
from backend.config.settings import get_settings

logger = get_logger(__name__)

# Compile ANSI escape sequence pattern once at module level for performance
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


def is_flux_model(model: Dict[str, Any]) -> bool:
    """
    Check if a model is a FLUX model.

    FLUX models have a completely different architecture from Stable Diffusion
    and cannot be loaded with StableDiffusionPipeline. They require ComfyUI
    or the FluxPipeline class from diffusers (when available).

    Args:
        model: Model dictionary with name, model_id, base_model, path fields

    Returns:
        bool: True if this is a FLUX model
    """
    model_name = model.get("name", "").lower()
    model_id = model.get("model_id", "").lower()
    base_model = model.get("base_model", "").lower()
    model_path = model.get("path", "").lower()
    display_name = model.get("display_name", "").lower()

    all_text = f"{model_name} {model_id} {base_model} {model_path} {display_name}"

    flux_patterns = [
        r"\bflux[.\-_]?\d",
        r"\bflux[.\-_]?dev\b",
        r"\bflux[.\-_]?schnell\b",
        r"\bfluxed",
        r"/flux[.\-_]",
        r"flux[.\-_]?nsfw",
        r"flux[.\-_]?up",
    ]

    for pattern in flux_patterns:
        if re.search(pattern, all_text, re.IGNORECASE):
            return True

    return False


def disable_safety_checker(pipe: Any) -> bool:
    """
    Disable the safety checker on a diffusion pipeline for NSFW content generation.

    Args:
        pipe: A diffusers pipeline instance

    Returns:
        bool: True if safety checker was disabled, False if not present
    """
    disabled = False
    if hasattr(pipe, "safety_checker") and pipe.safety_checker is not None:
        pipe.safety_checker = None
        disabled = True
        logger.info("✓ Safety checker disabled for NSFW content generation")
    if hasattr(pipe, "requires_safety_checker"):
        pipe.requires_safety_checker = False
    return disabled


def filter_scheduler_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Filter out deprecated attributes from scheduler config.

    Args:
        config: Scheduler configuration dictionary

    Returns:
        Filtered config dictionary without deprecated attributes
    """
    # These attributes were deprecated in diffusers >= 0.25.0 (late 2023)
    # and removed in later versions. They cause warnings/errors with
    # DPMSolverMultistepScheduler when loading older model configs.
    # See: https://github.com/huggingface/diffusers/pull/6106
    deprecated_attrs = ["use_beta_sigmas", "use_exponential_sigmas"]
    filtered_config = dict(config)

    for attr in deprecated_attrs:
        if attr in filtered_config:
            logger.debug(f"Removed deprecated scheduler config attribute: {attr}")
            del filtered_config[attr]

    return filtered_config


def verify_model_files_exist(model_path: Path, model_type: str = "text") -> bool:
    """
    Verify that required model files exist at the given path.

    Args:
        model_path: Path to the model directory or file
        model_type: Type of model ("text", "image", "voice", "video")

    Returns:
        bool: True if all required files exist
    """
    if not model_path.exists():
        return False

    if model_type == "text":
        # Check for GGUF files (llama.cpp format)
        if model_path.is_file() and model_path.suffix == ".gguf":
            return True
        # Check for directory with model files
        if model_path.is_dir():
            # Check for transformers format
            if (model_path / "config.json").exists():
                return True
            # Check for GGUF files in directory
            gguf_files = list(model_path.glob("*.gguf"))
            return len(gguf_files) > 0

    elif model_type == "image":
        # Check for safetensors files
        if model_path.is_file():
            return model_path.suffix in [".safetensors", ".ckpt", ".pt"]
        if model_path.is_dir():
            # Check for diffusers format
            if (model_path / "model_index.json").exists():
                return True
            # Check for single-file checkpoints
            checkpoint_files = list(model_path.glob("*.safetensors")) + list(
                model_path.glob("*.ckpt")
            )
            return len(checkpoint_files) > 0

    elif model_type == "voice":
        if model_path.is_dir():
            # Check for XTTS-v2 format
            if (model_path / "model.pth").exists() or (
                model_path / "config.json"
            ).exists():
                return True
            # Check for Piper format
            if any(model_path.glob("*.onnx")):
                return True

    elif model_type == "video":
        if model_path.is_dir():
            # Check for video model format
            return (model_path / "model_index.json").exists()

    return False


async def download_model_from_huggingface(
    model_id: str,
    model_path: Path,
    model_type: str = "text",
    revision: Optional[str] = None,
) -> bool:
    """
    Download a model from HuggingFace Hub.

    Args:
        model_id: HuggingFace model ID (e.g., "meta-llama/Llama-3.1-8B")
        model_path: Local path to save the model
        model_type: Type of model ("text", "image", "voice", "video")
        revision: Specific revision/branch to download

    Returns:
        bool: True if download was successful
    """
    try:
        from huggingface_hub import snapshot_download

        settings = get_settings()
        token = settings.hugging_face_token

        logger.info(f"Downloading model {model_id} from HuggingFace...")

        # Run download in thread pool to not block async loop
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None,
            lambda: snapshot_download(
                repo_id=model_id,
                local_dir=str(model_path),
                token=token,
                revision=revision,
            ),
        )

        logger.info(f"✓ Downloaded model {model_id} to {model_path}")
        return True

    except Exception as e:
        logger.error(f"Failed to download model from HuggingFace: {e}")
        return False


async def download_model_from_civitai(
    model_id: str,
    model_path: Path,
    version_id: Optional[str] = None,
) -> bool:
    """
    Download a model from CivitAI.

    Args:
        model_id: CivitAI model ID
        model_path: Local path to save the model
        version_id: Specific version ID to download

    Returns:
        bool: True if download was successful
    """
    try:
        settings = get_settings()
        api_key = os.environ.get("CIVITAI_API_KEY")

        # Get model info
        model_url = f"https://civitai.com/api/v1/models/{model_id}"
        headers = {}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(model_url, headers=headers)
            response.raise_for_status()
            model_info = response.json()

        # Get download URL
        if version_id:
            version = next(
                (v for v in model_info["modelVersions"] if str(v["id"]) == version_id),
                None,
            )
        else:
            version = model_info["modelVersions"][0]  # Latest version

        if not version:
            logger.error(f"Version not found for CivitAI model {model_id}")
            return False

        download_url = version["files"][0]["downloadUrl"]

        # Download the file
        model_path.parent.mkdir(parents=True, exist_ok=True)

        async with httpx.AsyncClient(timeout=600.0) as client:
            response = await client.get(
                download_url, headers=headers, follow_redirects=True
            )
            response.raise_for_status()

            with open(model_path, "wb") as f:
                f.write(response.content)

        logger.info(f"✓ Downloaded CivitAI model {model_id} to {model_path}")
        return True

    except Exception as e:
        logger.error(f"Failed to download model from CivitAI: {e}")
        return False


def get_model_size_estimate(model_type: str, model_name: str) -> float:
    """
    Estimate the GPU memory requirement for a model in GB.

    Args:
        model_type: Type of model
        model_name: Name or identifier of the model

    Returns:
        Estimated GPU memory requirement in GB
    """
    model_lower = model_name.lower()

    if model_type == "text":
        # LLM size estimates based on parameter count
        if "70b" in model_lower:
            return 40.0
        elif "34b" in model_lower or "32b" in model_lower:
            return 20.0
        elif "13b" in model_lower or "14b" in model_lower:
            return 10.0
        elif "8b" in model_lower or "7b" in model_lower:
            return 6.0
        elif "3b" in model_lower:
            return 3.0
        elif "1b" in model_lower:
            return 1.5
        return 4.0  # Default

    elif model_type == "image":
        # Image model size estimates
        if "flux" in model_lower:
            return 24.0  # FLUX models are large
        elif "sdxl" in model_lower:
            return 8.0
        elif "sd3" in model_lower:
            return 10.0
        return 4.0  # SD 1.5 and similar

    elif model_type == "video":
        return 16.0  # Video models typically need more memory

    elif model_type == "voice":
        return 2.0  # Voice models are relatively small

    return 4.0  # Default estimate
