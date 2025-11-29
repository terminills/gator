"""
GPU Manager for AI model allocation and load balancing.

Handles GPU detection, memory management, and optimal device selection
for multi-GPU systems with AMD ROCm and NVIDIA CUDA support.
"""

import os
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional

import torch

from backend.config.logging import get_logger

logger = get_logger(__name__)


class GPUType(str, Enum):
    """Supported GPU types."""

    CUDA = "cuda"
    ROCM = "rocm"
    CPU = "cpu"


@dataclass
class GPUInfo:
    """Information about a GPU device."""

    index: int
    name: str
    memory_total: float  # GB
    memory_free: float  # GB
    memory_used: float  # GB
    gpu_type: GPUType
    compute_capability: Optional[str] = None

    @property
    def memory_usage_percent(self) -> float:
        """Get memory usage as a percentage."""
        if self.memory_total == 0:
            return 0.0
        return (self.memory_used / self.memory_total) * 100


class GPUManager:
    """
    Manages GPU resources for AI model operations.

    Provides:
    - GPU detection and capability checking
    - Memory management and monitoring
    - Optimal device selection for model loading
    - Load balancing across multiple GPUs
    """

    _instance: Optional["GPUManager"] = None

    def __new__(cls) -> "GPUManager":
        """Singleton pattern for GPU manager."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        """Initialize GPU manager."""
        if self._initialized:
            return

        self.gpu_type: GPUType = GPUType.CPU
        self.gpus: List[GPUInfo] = []
        self.device_map: Dict[str, int] = {}  # Model name -> GPU index
        self._detect_gpus()
        self._initialized = True

    def _detect_gpus(self) -> None:
        """Detect available GPUs and their capabilities."""
        self.gpus = []

        if torch.cuda.is_available():
            # Check if it's AMD ROCm or NVIDIA CUDA
            rocm_version = os.environ.get("HSA_OVERRIDE_GFX_VERSION", "")
            is_rocm = bool(rocm_version) or "rocm" in torch.__version__.lower()

            self.gpu_type = GPUType.ROCM if is_rocm else GPUType.CUDA

            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                memory_total = props.total_memory / (1024**3)  # Convert to GB

                # Get current memory usage
                torch.cuda.set_device(i)
                memory_free = torch.cuda.memory_reserved(i) / (1024**3)
                memory_used = torch.cuda.memory_allocated(i) / (1024**3)

                compute_capability = None
                if not is_rocm:
                    compute_capability = f"{props.major}.{props.minor}"

                self.gpus.append(
                    GPUInfo(
                        index=i,
                        name=props.name,
                        memory_total=memory_total,
                        memory_free=memory_total - memory_used,
                        memory_used=memory_used,
                        gpu_type=self.gpu_type,
                        compute_capability=compute_capability,
                    )
                )

            logger.info(
                f"Detected {len(self.gpus)} {self.gpu_type.value.upper()} GPU(s)"
            )
            for gpu in self.gpus:
                logger.info(
                    f"  GPU {gpu.index}: {gpu.name} - {gpu.memory_total:.1f}GB "
                    f"({gpu.memory_free:.1f}GB free)"
                )
        else:
            self.gpu_type = GPUType.CPU
            logger.info("No GPU detected, using CPU")

    def get_gpu_count(self) -> int:
        """Get the number of available GPUs."""
        return len(self.gpus)

    def get_gpu_info(self, index: int = 0) -> Optional[GPUInfo]:
        """Get information about a specific GPU."""
        if index < len(self.gpus):
            return self.gpus[index]
        return None

    def get_all_gpu_info(self) -> List[GPUInfo]:
        """Get information about all GPUs."""
        return self.gpus.copy()

    def get_total_memory(self) -> float:
        """Get total GPU memory across all GPUs in GB."""
        return sum(gpu.memory_total for gpu in self.gpus)

    def get_available_memory(self) -> float:
        """Get total available GPU memory across all GPUs in GB."""
        return sum(gpu.memory_free for gpu in self.gpus)

    def refresh_memory_stats(self) -> None:
        """Refresh memory statistics for all GPUs."""
        if self.gpu_type == GPUType.CPU:
            return

        for gpu in self.gpus:
            torch.cuda.set_device(gpu.index)
            memory_used = torch.cuda.memory_allocated(gpu.index) / (1024**3)
            gpu.memory_used = memory_used
            gpu.memory_free = gpu.memory_total - memory_used

    def select_device_for_model(
        self,
        model_name: str,
        required_memory_gb: float = 4.0,
        prefer_dedicated: bool = True,
    ) -> str:
        """
        Select the best device for loading a model.

        Args:
            model_name: Name of the model being loaded
            required_memory_gb: Estimated memory requirement in GB
            prefer_dedicated: Whether to prefer a dedicated GPU over shared

        Returns:
            Device string (e.g., "cuda:0", "cuda:1", "cpu")
        """
        if self.gpu_type == GPUType.CPU or not self.gpus:
            return "cpu"

        self.refresh_memory_stats()

        # Find GPU with most available memory
        best_gpu = None
        best_free_memory = 0.0

        for gpu in self.gpus:
            # Check if model is already assigned to this GPU
            if model_name in self.device_map:
                assigned_idx = self.device_map[model_name]
                if gpu.index == assigned_idx:
                    return f"cuda:{assigned_idx}"

            # Check if GPU has enough free memory
            if gpu.memory_free >= required_memory_gb:
                if gpu.memory_free > best_free_memory:
                    best_free_memory = gpu.memory_free
                    best_gpu = gpu

        if best_gpu:
            self.device_map[model_name] = best_gpu.index
            logger.debug(
                f"Assigned model '{model_name}' to GPU {best_gpu.index} "
                f"({best_free_memory:.1f}GB free)"
            )
            return f"cuda:{best_gpu.index}"

        # Fall back to GPU 0 if no GPU has enough free memory
        logger.warning(
            f"No GPU has {required_memory_gb}GB free for '{model_name}', "
            f"using GPU 0"
        )
        self.device_map[model_name] = 0
        return "cuda:0"

    def release_device(self, model_name: str) -> None:
        """
        Release the device assignment for a model.

        Args:
            model_name: Name of the model to release
        """
        if model_name in self.device_map:
            del self.device_map[model_name]
            logger.debug(f"Released device assignment for '{model_name}'")

    def get_device(self, model_name: str) -> str:
        """
        Get the device assigned to a model.

        Args:
            model_name: Name of the model

        Returns:
            Device string or "cpu" if not assigned
        """
        if model_name in self.device_map:
            return f"cuda:{self.device_map[model_name]}"
        return "cpu"

    def clear_cache(self, device_index: Optional[int] = None) -> None:
        """
        Clear GPU memory cache.

        Args:
            device_index: Specific GPU to clear, or None for all
        """
        if self.gpu_type == GPUType.CPU:
            return

        if device_index is not None:
            torch.cuda.set_device(device_index)
            torch.cuda.empty_cache()
            logger.debug(f"Cleared cache for GPU {device_index}")
        else:
            for gpu in self.gpus:
                torch.cuda.set_device(gpu.index)
                torch.cuda.empty_cache()
            logger.debug("Cleared cache for all GPUs")

    def get_system_info(self) -> Dict[str, Any]:
        """
        Get system GPU information as a dictionary.

        Returns:
            Dictionary with GPU system information
        """
        self.refresh_memory_stats()

        return {
            "gpu_type": self.gpu_type.value,
            "gpu_count": len(self.gpus),
            "total_memory_gb": self.get_total_memory(),
            "available_memory_gb": self.get_available_memory(),
            "gpus": [
                {
                    "index": gpu.index,
                    "name": gpu.name,
                    "memory_total_gb": gpu.memory_total,
                    "memory_free_gb": gpu.memory_free,
                    "memory_used_gb": gpu.memory_used,
                    "memory_usage_percent": gpu.memory_usage_percent,
                    "compute_capability": gpu.compute_capability,
                }
                for gpu in self.gpus
            ],
            "device_assignments": dict(self.device_map),
        }


# Global GPU manager instance
_gpu_manager: Optional[GPUManager] = None


def get_gpu_manager() -> GPUManager:
    """
    Get the global GPU manager instance.

    Returns:
        GPUManager singleton instance
    """
    global _gpu_manager
    if _gpu_manager is None:
        _gpu_manager = GPUManager()
    return _gpu_manager
