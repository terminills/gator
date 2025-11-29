"""
GPU Detection Utilities

Provides utilities for detecting GPU hardware and architecture.
Used to determine optimal inference engine selection.
"""

import re
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional

from backend.config.logging import get_logger

logger = get_logger(__name__)


def detect_amd_gpu_architecture() -> Optional[str]:
    """
    Detect AMD GPU architecture (e.g., gfx1030, gfx1100, gfx90a).

    Returns:
        GPU architecture string (e.g., "gfx1030") or None if not detected
    """
    try:
        # Try rocminfo first (most reliable for AMD GPUs)
        result = subprocess.run(
            ["rocminfo"],
            capture_output=True,
            text=True,
            timeout=5,
        )

        if result.returncode == 0:
            # Look for gfx architecture in output
            # Example line: "  Name:                    gfx1030"
            match = re.search(r"gfx\w+", result.stdout)
            if match:
                arch = match.group(0)
                logger.info(f"Detected AMD GPU architecture: {arch}")
                return arch

    except (FileNotFoundError, subprocess.TimeoutExpired):
        logger.debug("rocminfo not available")

    try:
        # Try rocm-smi as fallback
        result = subprocess.run(
            ["rocm-smi", "--showproductname"],
            capture_output=True,
            text=True,
            timeout=5,
        )

        if result.returncode == 0:
            # Map product names to architectures
            product_to_arch = {
                "Radeon RX 6600": "gfx1030",
                "Radeon RX 6700": "gfx1030",
                "Radeon RX 6800": "gfx1030",
                "Radeon RX 6900": "gfx1030",
                "Radeon RX 7600": "gfx1100",
                "Radeon RX 7700": "gfx1100",
                "Radeon RX 7800": "gfx1100",
                "Radeon RX 7900": "gfx1100",
                "Radeon VII": "gfx906",
                "MI25": "gfx900",
                "MI50": "gfx906",
                "MI100": "gfx908",
                "MI210": "gfx90a",
                "MI250": "gfx90a",
                "MI300": "gfx942",
            }

            for product, arch in product_to_arch.items():
                if product in result.stdout:
                    logger.info(f"Detected AMD GPU: {product} ({arch})")
                    return arch

    except (FileNotFoundError, subprocess.TimeoutExpired):
        logger.debug("rocm-smi not available")

    try:
        # Try reading from sysfs as last resort
        gpu_dirs = list(Path("/sys/class/drm").glob("card*"))
        for gpu_dir in gpu_dirs:
            device_file = gpu_dir / "device" / "device"
            if device_file.exists():
                device_id = device_file.read_text().strip()

                # Map PCI device IDs to architectures (common AMD GPUs)
                device_to_arch = {
                    "0x73ff": "gfx1030",  # RX 6600/6700
                    "0x73bf": "gfx1030",  # RX 6800
                    "0x73af": "gfx1030",  # RX 6900
                    "0x744c": "gfx1100",  # RX 7600
                    "0x7448": "gfx1100",  # RX 7700/7800
                    "0x744a": "gfx1100",  # RX 7900
                }

                if device_id in device_to_arch:
                    arch = device_to_arch[device_id]
                    logger.info(f"Detected AMD GPU from device ID {device_id}: {arch}")
                    return arch

    except Exception as e:
        logger.debug(f"Failed to read GPU info from sysfs: {e}")

    logger.debug("Could not detect AMD GPU architecture")
    return None


def is_vllm_compatible_gpu() -> bool:
    """
    Check if the current GPU is compatible with vLLM.

    vLLM has known issues with certain AMD GPU architectures,
    particularly gfx1030 (RDNA2 - RX 6000 series).

    Returns:
        True if vLLM should work, False if Ollama should be preferred
    """
    arch = detect_amd_gpu_architecture()

    if arch is None:
        # Can't detect GPU, assume vLLM might work
        return True

    # Known incompatible architectures
    incompatible = [
        "gfx1030",  # RX 6000 series - vLLM has issues
        "gfx1031",  # RX 6000 series mobile
        "gfx1032",  # RX 6000 series mobile
    ]

    if arch in incompatible:
        logger.info(f"GPU architecture {arch} is not fully compatible with vLLM")
        logger.info("Ollama is recommended for this GPU")
        return False

    # Known compatible architectures
    compatible = [
        "gfx90a",  # MI210/MI250 - excellent vLLM support
        "gfx908",  # MI100
        "gfx942",  # MI300
        "gfx1100",  # RX 7000 series - good vLLM support
        "gfx1101",  # RX 7000 series mobile
    ]

    if arch in compatible:
        logger.info(f"GPU architecture {arch} is compatible with vLLM")
        return True

    # Unknown architecture, might work
    logger.warning(f"Unknown GPU architecture {arch}, vLLM compatibility uncertain")
    return True


def get_gpu_info() -> Dict[str, Any]:
    """
    Get comprehensive GPU information.

    Returns:
        Dictionary with GPU details
    """
    info = {
        "architecture": None,
        "vllm_compatible": True,
        "ollama_recommended": False,
        "vendor": "unknown",
        "compute_units": None,
        "vram_gb": None,
    }

    # Detect AMD GPU
    arch = detect_amd_gpu_architecture()
    if arch:
        info["architecture"] = arch
        info["vendor"] = "amd"
        info["vllm_compatible"] = is_vllm_compatible_gpu()
        info["ollama_recommended"] = not info["vllm_compatible"]

    # Try to get VRAM info
    MB_TO_GB = 1024  # Conversion factor
    try:
        result = subprocess.run(
            ["rocm-smi", "--showmeminfo", "vram"],
            capture_output=True,
            text=True,
            timeout=5,
        )

        if result.returncode == 0:
            # Parse VRAM size from output
            match = re.search(r"(\d+)\s*MB", result.stdout)
            if match:
                vram_mb = int(match.group(1))
                info["vram_gb"] = vram_mb / MB_TO_GB

    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    # Try to get compute units
    try:
        result = subprocess.run(
            ["rocminfo"],
            capture_output=True,
            text=True,
            timeout=5,
        )

        if result.returncode == 0:
            match = re.search(r"Compute Unit:\s*(\d+)", result.stdout)
            if match:
                info["compute_units"] = int(match.group(1))

    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    return info


def should_use_ollama_fallback(force: bool = False) -> bool:
    """
    Determine if Ollama should be used instead of vLLM based on GPU.

    Args:
        force: Force Ollama usage regardless of GPU

    Returns:
        True if Ollama should be used
    """
    if force:
        return True

    # Check if vLLM is incompatible with current GPU
    if not is_vllm_compatible_gpu():
        return True

    # Check user preference from settings (if available)
    try:
        # Try to get setting from environment or config
        # Don't access database here as it may not be initialized yet
        import os

        prefer_ollama_env = os.environ.get("PREFER_OLLAMA_FOR_GFX1030", "").lower()
        if prefer_ollama_env in ("true", "1", "yes"):
            if detect_amd_gpu_architecture() == "gfx1030":
                return True

    except Exception:
        # Settings not available or error accessing them
        pass

    return False
