"""
ROCm Version Detection and PyTorch Installation Utilities

Detects installed ROCm version and provides appropriate PyTorch installation URLs
for ROCm 5.7 (legacy) and ROCm 6.5+ (standard wheels and nightly builds).
"""

import os
import re
import subprocess
from typing import Optional, Tuple, Dict
from pathlib import Path


class ROCmVersionInfo:
    """Container for ROCm version information."""

    def __init__(
        self,
        version: str,
        major: int,
        minor: int,
        patch: int = 0,
        is_installed: bool = True,
    ):
        self.version = version
        self.major = major
        self.minor = minor
        self.patch = patch
        self.is_installed = is_installed

    def __str__(self) -> str:
        return self.version

    def __repr__(self) -> str:
        return f"ROCmVersionInfo(version='{self.version}', major={self.major}, minor={self.minor}, patch={self.patch})"

    @property
    def is_6_5_or_later(self) -> bool:
        """Check if this is ROCm 6.5 or later."""
        return (self.major > 6) or (self.major == 6 and self.minor >= 5)

    @property
    def short_version(self) -> str:
        """Get short version string (e.g., '6.5' for ROCm 6.5.x)."""
        return f"{self.major}.{self.minor}"


def detect_rocm_version() -> Optional[ROCmVersionInfo]:
    """
    Detect installed ROCm version from multiple sources.

    Checks in order:
    1. /opt/rocm/.info/version file
    2. rocminfo command output
    3. ROCm environment variables

    Returns:
        ROCmVersionInfo if ROCm is detected, None otherwise
    """
    # Method 1: Check /opt/rocm/.info/version file
    version_file = Path("/opt/rocm/.info/version")
    if version_file.exists():
        try:
            version_str = version_file.read_text().strip()
            # Strip any suffix like -98
            version_str = version_str.split("-")[0]
            return parse_rocm_version(version_str)
        except Exception:
            pass

    # Method 2: Try rocminfo command
    try:
        result = subprocess.run(["rocminfo"], capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            # Look for ROCm version in output
            for line in result.stdout.split("\n"):
                if "ROCm" in line or "Version" in line:
                    # Try to extract version number
                    match = re.search(r"(\d+)\.(\d+)\.?(\d+)?", line)
                    if match:
                        major = int(match.group(1))
                        minor = int(match.group(2))
                        patch = int(match.group(3)) if match.group(3) else 0
                        version = f"{major}.{minor}.{patch}"
                        return ROCmVersionInfo(version, major, minor, patch)
    except (
        FileNotFoundError,
        subprocess.TimeoutExpired,
        subprocess.CalledProcessError,
    ):
        pass

    # Method 3: Check ROCm environment variables
    rocm_path = os.environ.get("ROCM_PATH") or os.environ.get("ROCM_HOME")
    if rocm_path:
        version_file = Path(rocm_path) / ".info" / "version"
        if version_file.exists():
            try:
                version_str = version_file.read_text().strip()
                version_str = version_str.split("-")[0]
                return parse_rocm_version(version_str)
            except Exception:
                pass

    # Method 4: Check if /opt/rocm exists (unknown version)
    if Path("/opt/rocm").exists():
        # ROCm is installed but version unknown, assume legacy
        return ROCmVersionInfo("5.7.0", 5, 7, 0, True)

    return None


def parse_rocm_version(version_str: str) -> Optional[ROCmVersionInfo]:
    """
    Parse ROCm version string into components.

    Args:
        version_str: Version string like "6.5.0" or "5.7.1"

    Returns:
        ROCmVersionInfo or None if parsing fails
    """
    try:
        # Handle versions like "6.5.0-98" or "6.5.0"
        clean_version = version_str.split("-")[0].strip()
        parts = clean_version.split(".")

        major = int(parts[0])
        minor = int(parts[1]) if len(parts) > 1 else 0
        patch = int(parts[2]) if len(parts) > 2 else 0

        return ROCmVersionInfo(clean_version, major, minor, patch)
    except (ValueError, IndexError):
        return None


def get_pytorch_index_url(
    rocm_version: Optional[ROCmVersionInfo] = None, use_nightly: bool = False
) -> str:
    """
    Get appropriate PyTorch index URL based on ROCm version.

    For ROCm 7.0+:
    - Stable: https://download.pytorch.org/whl/rocm7.0/
    - Nightly: https://download.pytorch.org/whl/nightly/rocm7.0/

    For ROCm 6.5+:
    - Stable: https://download.pytorch.org/whl/rocm6.5/
    - Nightly: https://download.pytorch.org/whl/nightly/rocm6.5/

    For ROCm 6.4 (pre-standard wheels era):
    - https://download.pytorch.org/whl/rocm6.4/

    For ROCm 5.7 (legacy):
    - https://download.pytorch.org/whl/rocm5.7/

    Args:
        rocm_version: ROCm version info. If None, auto-detect.
        use_nightly: Whether to use nightly builds (only for ROCm 6.5+)

    Returns:
        PyTorch wheel index URL
    """
    if rocm_version is None:
        rocm_version = detect_rocm_version()

    if rocm_version is None:
        # No ROCm detected, use CPU-only
        return "https://download.pytorch.org/whl/cpu"

    # ROCm 6.5+ with standard wheels and nightly builds
    if rocm_version.is_6_5_or_later:
        if use_nightly:
            return f"https://download.pytorch.org/whl/nightly/rocm{rocm_version.short_version}/"
        else:
            return f"https://download.pytorch.org/whl/rocm{rocm_version.short_version}/"

    # ROCm 6.4 and earlier
    if rocm_version.major == 6 and rocm_version.minor == 4:
        return "https://download.pytorch.org/whl/rocm6.4/"

    # Legacy ROCm 5.7 (hardcoded for MI-25 compatibility)
    if rocm_version.major == 5 and rocm_version.minor == 7:
        return "https://download.pytorch.org/whl/rocm5.7/"

    # Default fallback to closest version
    if rocm_version.major >= 7:
        # Use ROCm 7.0+ for major version 7+
        return f"https://download.pytorch.org/whl/rocm{rocm_version.short_version}/"
    elif rocm_version.major >= 6:
        # Use latest ROCm 6.x
        return "https://download.pytorch.org/whl/rocm6.5/"
    else:
        # Use ROCm 5.7 for older versions
        return "https://download.pytorch.org/whl/rocm5.7/"


def get_pytorch_install_command(
    rocm_version: Optional[ROCmVersionInfo] = None,
    use_nightly: bool = False,
    include_torchvision: bool = True,
    include_torchaudio: bool = False,
) -> Tuple[str, Dict[str, str]]:
    """
    Generate pip install command for PyTorch with ROCm support.

    Args:
        rocm_version: ROCm version info. If None, auto-detect.
        use_nightly: Whether to use nightly builds
        include_torchvision: Whether to include torchvision
        include_torchaudio: Whether to include torchaudio

    Returns:
        Tuple of (pip command string, metadata dict)
    """
    if rocm_version is None:
        rocm_version = detect_rocm_version()

    index_url = get_pytorch_index_url(rocm_version, use_nightly)

    packages = ["torch"]
    if include_torchvision:
        packages.append("torchvision")
    if include_torchaudio:
        packages.append("torchaudio")

    # For nightly builds, use --pre flag
    pre_flag = "--pre " if use_nightly else ""

    # Build pip command
    packages_str = " ".join(packages)
    command = f"pip3 install {pre_flag}{packages_str} --index-url {index_url}"

    metadata = {
        "rocm_version": str(rocm_version) if rocm_version else "not_detected",
        "index_url": index_url,
        "nightly": use_nightly,
        "packages": packages,
    }

    return command, metadata


def get_recommended_pytorch_version(
    rocm_version: Optional[ROCmVersionInfo] = None,
) -> Dict[str, str]:
    """
    Get recommended PyTorch version for the detected ROCm version.

    Returns:
        Dictionary with package versions
    """
    if rocm_version is None:
        rocm_version = detect_rocm_version()

    if rocm_version is None:
        return {
            "torch": "latest (CPU)",
            "torchvision": "latest (CPU)",
            "note": "No ROCm detected, using CPU-only builds",
        }

    if rocm_version.is_6_5_or_later:
        return {
            "torch": f"latest (ROCm {rocm_version.short_version})",
            "torchvision": f"latest (ROCm {rocm_version.short_version})",
            "torchaudio": f"latest (ROCm {rocm_version.short_version})",
            "note": f"Using standard wheels for ROCm {rocm_version.short_version}+",
            "nightly_available": True,
        }
    elif rocm_version.major == 5 and rocm_version.minor == 7:
        return {
            "torch": "2.3.1+rocm5.7",
            "torchvision": "0.18.1+rocm5.7",
            "note": "Using PyTorch 2.3.1 (latest compatible with ROCm 5.7 and MI-25)",
            "nightly_available": False,
        }
    else:
        return {
            "torch": f"latest (ROCm {rocm_version.short_version})",
            "torchvision": f"latest (ROCm {rocm_version.short_version})",
            "note": f"Using legacy ROCm {rocm_version.short_version} builds",
            "nightly_available": False,
        }


def get_gpu_architecture() -> Dict[str, any]:
    """
    Detect GPU architecture and capabilities.

    Returns:
        Dictionary with GPU architecture details for each device
    """
    gpu_info = {
        "devices": [],
        "architectures": set(),
        "total_memory_gb": 0,
        "multi_gpu": False,
    }

    try:
        import torch

        if not torch.cuda.is_available():
            return gpu_info

        device_count = torch.cuda.device_count()
        gpu_info["multi_gpu"] = device_count > 1

        for i in range(device_count):
            try:
                props = torch.cuda.get_device_properties(i)
                device_name = torch.cuda.get_device_name(i)
                memory_gb = props.total_memory / (1024**3)

                # Detect architecture from compute capability or name
                arch = "unknown"
                if "MI25" in device_name or "Vega 10" in device_name:
                    arch = "gfx900"  # Vega 10
                elif "MI210" in device_name or "MI250" in device_name:
                    arch = "gfx90a"  # CDNA2
                elif "V620" in device_name or "Pro V620" in device_name:
                    arch = "gfx1030"  # RDNA2
                elif "6900 XT" in device_name or "6800" in device_name:
                    arch = "gfx1030"  # RDNA2
                elif "7900 XTX" in device_name or "7900 XT" in device_name:
                    arch = "gfx1100"  # RDNA3
                else:
                    # Try to determine from compute capability
                    arch = f"compute_{props.major}_{props.minor}"

                device_info = {
                    "id": i,
                    "name": device_name,
                    "architecture": arch,
                    "memory_gb": round(memory_gb, 2),
                    "compute_capability": f"{props.major}.{props.minor}",
                    "multi_processor_count": props.multi_processor_count,
                }

                gpu_info["devices"].append(device_info)
                gpu_info["architectures"].add(arch)
                gpu_info["total_memory_gb"] += memory_gb
            except Exception as e:
                gpu_info["devices"].append(
                    {"id": i, "name": "Unknown", "error": str(e)}
                )

        gpu_info["architectures"] = list(gpu_info["architectures"])
        gpu_info["total_memory_gb"] = round(gpu_info["total_memory_gb"], 2)

    except ImportError:
        pass

    return gpu_info


def check_pytorch_installation() -> Dict[str, any]:
    """
    Check if PyTorch is installed and get its ROCm version.

    Returns:
        Dictionary with installation status and version info
    """
    try:
        import torch

        pytorch_version = torch.__version__
        is_rocm_build = hasattr(torch.version, "hip")
        rocm_build_version = None

        if is_rocm_build:
            rocm_build_version = getattr(torch.version, "hip", None)

        gpu_available = torch.cuda.is_available() if hasattr(torch, "cuda") else False
        gpu_count = torch.cuda.device_count() if gpu_available else 0

        # Get detailed GPU architecture info
        gpu_arch_info = get_gpu_architecture()

        # Parse PyTorch version to get major.minor
        pytorch_major_minor = None
        try:
            # Handle versions like "2.10.0+rocm7.0" or "2.3.1"
            version_parts = pytorch_version.split("+")[0].split(".")
            if len(version_parts) >= 2:
                pytorch_major_minor = f"{version_parts[0]}.{version_parts[1]}"
        except Exception:
            pass

        return {
            "installed": True,
            "version": pytorch_version,
            "pytorch_major_minor": pytorch_major_minor,
            "is_rocm_build": is_rocm_build,
            "rocm_build_version": rocm_build_version,
            "gpu_available": gpu_available,
            "gpu_count": gpu_count,
            "gpu_architecture": gpu_arch_info,
        }
    except ImportError:
        return {
            "installed": False,
            "version": None,
            "pytorch_major_minor": None,
            "is_rocm_build": False,
            "rocm_build_version": None,
            "gpu_available": False,
            "gpu_count": 0,
            "gpu_architecture": {
                "devices": [],
                "architectures": [],
                "total_memory_gb": 0,
            },
        }


def get_compatible_dependency_versions(
    pytorch_version: Optional[str] = None,
) -> Dict[str, str]:
    """
    Get compatible dependency versions based on installed PyTorch version.

    This ensures that packages like transformers, diffusers, and accelerate
    are compatible with the installed PyTorch version.

    Args:
        pytorch_version: PyTorch version string (e.g., "2.10.0+rocm7.0", "2.3.1").
                        If None, will attempt to detect from installed PyTorch.

    Returns:
        Dictionary mapping package names to compatible version specifiers
    """
    # Default/latest compatible versions
    default_versions = {
        "transformers": ">=4.41.0",
        "diffusers": ">=0.28.0",
        "accelerate": ">=0.29.0",
        "huggingface_hub": ">=0.23.0",
    }

    if pytorch_version is None:
        # Try to detect from installed PyTorch
        pytorch_info = check_pytorch_installation()
        if pytorch_info["installed"]:
            pytorch_version = pytorch_info["version"]
        else:
            # No PyTorch installed, return defaults
            return default_versions

    # Parse PyTorch version
    try:
        version_base = pytorch_version.split("+")[0]  # Remove build suffix
        version_parts = version_base.split(".")
        major = int(version_parts[0])
        minor = int(version_parts[1]) if len(version_parts) > 1 else 0
    except (ValueError, IndexError):
        # Couldn't parse, return defaults
        return default_versions

    # PyTorch 2.10+ (nightly/future releases)
    if major >= 3 or (major == 2 and minor >= 10):
        return {
            "transformers": ">=4.45.0",  # Latest transformers with PyTorch 2.10+ support
            "diffusers": ">=0.31.0",  # Latest diffusers with PyTorch 2.10+ support
            "accelerate": ">=0.34.0",  # Latest accelerate with PyTorch 2.10+ support
            "huggingface_hub": ">=0.25.0",
        }

    # PyTorch 2.4-2.9
    elif major == 2 and 4 <= minor <= 9:
        return {
            "transformers": ">=4.43.0",
            "diffusers": ">=0.29.0",
            "accelerate": ">=0.30.0",
            "huggingface_hub": ">=0.24.0",
        }

    # PyTorch 2.3.x (ROCm 5.7 legacy)
    elif major == 2 and minor == 3:
        return {
            "transformers": ">=4.41.0,<4.50.0",  # Upper bound for safety
            "diffusers": ">=0.28.0,<0.35.0",
            "accelerate": ">=0.29.0,<0.35.0",
            "huggingface_hub": ">=0.23.0,<0.30.0",
        }

    # PyTorch 2.0-2.2
    elif major == 2 and minor <= 2:
        return {
            "transformers": ">=4.35.0,<4.45.0",
            "diffusers": ">=0.25.0,<0.30.0",
            "accelerate": ">=0.25.0,<0.30.0",
            "huggingface_hub": ">=0.20.0,<0.25.0",
        }

    # Default fallback
    return default_versions


def get_multi_gpu_config(gpu_count: int = None) -> Dict[str, any]:
    """
    Generate optimal multi-GPU configuration.

    Args:
        gpu_count: Number of GPUs to configure. If None, auto-detect.

    Returns:
        Dictionary with multi-GPU configuration recommendations
    """
    if gpu_count is None:
        pytorch_info = check_pytorch_installation()
        gpu_count = pytorch_info.get("gpu_count", 0)

    if gpu_count <= 1:
        return {
            "mode": "single_gpu",
            "gpu_count": gpu_count,
            "note": "Single GPU or CPU mode",
        }

    # Multi-GPU configurations for different workload types
    config = {
        "mode": "multi_gpu",
        "gpu_count": gpu_count,
        "strategies": {
            "data_parallel": {
                "description": "Replicate model across GPUs, split batch",
                "use_for": ["image_generation", "training"],
                "env_vars": {
                    "CUDA_VISIBLE_DEVICES": ",".join(str(i) for i in range(gpu_count)),
                    "ROCR_VISIBLE_DEVICES": ",".join(str(i) for i in range(gpu_count)),
                },
            },
            "pipeline_parallel": {
                "description": "Split model layers across GPUs",
                "use_for": ["large_language_models", "inference"],
                "env_vars": {
                    "CUDA_VISIBLE_DEVICES": ",".join(str(i) for i in range(gpu_count)),
                },
            },
            "task_parallel": {
                "description": "Assign different tasks to different GPUs",
                "use_for": ["mixed_workloads", "multi_persona"],
                "gpu_assignment": {
                    f"gpu_{i}": f"persona_{i % gpu_count}" for i in range(gpu_count)
                },
            },
        },
        "recommendations": [],
    }

    # Add specific recommendations based on GPU count
    if gpu_count == 2:
        config["recommendations"].extend(
            [
                "GPU 0: Primary inference (text/image generation)",
                "GPU 1: Secondary inference (parallel requests)",
                "Use data parallelism for batch processing",
            ]
        )
    elif gpu_count == 3:
        config["recommendations"].extend(
            [
                "GPU 0: Text generation (LLM inference)",
                "GPU 1: Image generation (Stable Diffusion/FLUX)",
                "GPU 2: Video processing or additional capacity",
                "Excellent for running multiple personas simultaneously",
            ]
        )
    elif gpu_count >= 4:
        config["recommendations"].extend(
            [
                "Consider load balancing across all GPUs",
                "Ideal for production multi-tenant setup",
                "Can handle 4+ concurrent personas efficiently",
                "Use Ray or vLLM for distributed inference",
            ]
        )

    return config


def generate_rocm_env_vars(
    rocm_version: Optional[ROCmVersionInfo] = None, gpu_count: int = None
) -> Dict[str, str]:
    """
    Generate recommended ROCm environment variables for optimal performance.

    Args:
        rocm_version: ROCm version info. If None, auto-detect.
        gpu_count: Number of GPUs. If None, auto-detect.

    Returns:
        Dictionary of environment variable key-value pairs
    """
    if rocm_version is None:
        rocm_version = detect_rocm_version()

    if gpu_count is None:
        pytorch_info = check_pytorch_installation()
        gpu_count = pytorch_info.get("gpu_count", 1)

    env_vars = {
        "ROCM_PATH": "/opt/rocm",
        "HIP_PATH": "/opt/rocm/hip",
        "HIP_VISIBLE_DEVICES": ",".join(str(i) for i in range(gpu_count)),
        "ROCR_VISIBLE_DEVICES": ",".join(str(i) for i in range(gpu_count)),
        "HSA_ENABLE_SDMA": "0",  # Recommended for stability
    }

    # Get GPU architecture
    pytorch_info = check_pytorch_installation()
    gpu_arch_info = pytorch_info.get("gpu_architecture", {})
    architectures = gpu_arch_info.get("architectures", [])

    # Architecture-specific settings
    if "gfx900" in architectures:
        # MI25/Vega 10 specific settings
        env_vars["HSA_OVERRIDE_GFX_VERSION"] = "9.0.0"
        env_vars["HCC_AMDGPU_TARGET"] = "gfx900"
        env_vars["PYTORCH_ROCM_ARCH"] = "gfx900"
    elif "gfx1030" in architectures:
        # V620/RDNA2 specific settings
        env_vars["HSA_OVERRIDE_GFX_VERSION"] = "10.3.0"
        env_vars["PYTORCH_ROCM_ARCH"] = "gfx1030"
        # RDNA2 benefits from these optimizations
        env_vars["GPU_MAX_ALLOC_PERCENT"] = "100"
        env_vars["GPU_MAX_HEAP_SIZE"] = "100"
    elif "gfx90a" in architectures:
        # MI210/MI250 (CDNA2) specific settings
        env_vars["PYTORCH_ROCM_ARCH"] = "gfx90a"
        env_vars["GPU_MAX_ALLOC_PERCENT"] = "100"

    # Multi-GPU specific settings
    if gpu_count > 1:
        env_vars["NCCL_DEBUG"] = "INFO"  # For debugging multi-GPU communication
        env_vars["NCCL_IB_DISABLE"] = "1"  # Disable InfiniBand if not available
        # For V620, enable peer access
        env_vars["HIP_FORCE_DEV_KERNARG"] = "1"

    return env_vars


if __name__ == "__main__":
    """CLI for testing ROCm detection."""
    print("ROCm Detection & Multi-GPU Configuration Utility")
    print("=" * 70)

    # Detect ROCm version
    rocm_version = detect_rocm_version()
    if rocm_version:
        print(f"✓ ROCm detected: {rocm_version}")
        print(
            f"  Version: {rocm_version.major}.{rocm_version.minor}.{rocm_version.patch}"
        )
        print(f"  ROCm 6.5+: {rocm_version.is_6_5_or_later}")
    else:
        print("✗ ROCm not detected")

    print()

    # Check current PyTorch installation
    pytorch_info = check_pytorch_installation()
    print("Current PyTorch Installation:")
    if pytorch_info["installed"]:
        print(f"  ✓ Installed: {pytorch_info['version']}")
        print(f"  ROCm build: {pytorch_info['is_rocm_build']}")
        if pytorch_info["is_rocm_build"]:
            print(f"  ROCm version: {pytorch_info['rocm_build_version']}")
        print(f"  GPU available: {pytorch_info['gpu_available']}")
        print(f"  GPU count: {pytorch_info['gpu_count']}")

        # Show GPU architecture details
        gpu_arch = pytorch_info.get("gpu_architecture", {})
        if gpu_arch.get("devices"):
            print("\n  GPU Devices:")
            for device in gpu_arch["devices"]:
                print(f"    [{device['id']}] {device['name']}")
                print(f"        Architecture: {device.get('architecture', 'unknown')}")
                print(f"        Memory: {device.get('memory_gb', 0):.2f} GB")

            print(f"\n  Total GPU Memory: {gpu_arch.get('total_memory_gb', 0):.2f} GB")
            print(f"  Architectures: {', '.join(gpu_arch.get('architectures', []))}")

            if gpu_arch.get("multi_gpu"):
                print("  ✓ Multi-GPU setup detected!")
    else:
        print("  ✗ Not installed")

    print()

    # Multi-GPU configuration
    if pytorch_info.get("gpu_count", 0) > 1:
        print("Multi-GPU Configuration:")
        multi_gpu_config = get_multi_gpu_config(pytorch_info["gpu_count"])
        print(f"  Mode: {multi_gpu_config['mode']}")
        print(f"  GPU Count: {multi_gpu_config['gpu_count']}")
        print("\n  Recommendations:")
        for rec in multi_gpu_config["recommendations"]:
            print(f"    • {rec}")
        print("\n  Available Strategies:")
        for strategy, details in multi_gpu_config["strategies"].items():
            print(f"    {strategy}: {details['description']}")
            print(f"      Best for: {', '.join(details['use_for'])}")
        print()

    # Environment variables
    print("Recommended Environment Variables:")
    env_vars = generate_rocm_env_vars(rocm_version, pytorch_info.get("gpu_count"))
    for key, value in env_vars.items():
        print(f"  export {key}={value}")

    print()

    # Get PyTorch index URLs
    print("PyTorch Index URLs:")
    print(f"  Stable: {get_pytorch_index_url(rocm_version, use_nightly=False)}")
    if rocm_version and rocm_version.is_6_5_or_later:
        print(f"  Nightly: {get_pytorch_index_url(rocm_version, use_nightly=True)}")

    print()

    # Get install command
    command, metadata = get_pytorch_install_command(rocm_version, use_nightly=False)
    print("Recommended Installation Command:")
    print(f"  {command}")

    print()

    # Get recommended versions
    recommended = get_recommended_pytorch_version(rocm_version)
    print("Recommended PyTorch Versions:")
    for key, value in recommended.items():
        if key != "nightly_available":
            print(f"  {key}: {value}")

    print("\n" + "=" * 70)
