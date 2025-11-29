"""
GPU Monitoring Service

Provides GPU temperature monitoring, utilization tracking, and health metrics
for AMD ROCm GPUs. Integrates with PyTorch/ROCm to retrieve real-time GPU data.
"""

import asyncio
import time
from collections import deque
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from backend.config.logging import get_logger

logger = get_logger(__name__)


class GPUMonitoringService:
    """Service for monitoring GPU temperature, utilization, and health."""

    def __init__(self, history_size: int = 100):
        """
        Initialize GPU monitoring service.

        Args:
            history_size: Number of historical readings to maintain per GPU
        """
        self.history_size = history_size
        self.temperature_history: Dict[int, deque] = {}
        self.utilization_history: Dict[int, deque] = {}
        self._torch_available = False
        self._gpu_count = 0
        self._last_update = None

        # Initialize PyTorch if available
        self._initialize_torch()

    def _initialize_torch(self):
        """Initialize PyTorch and check GPU availability."""
        try:
            import torch

            self._torch_available = torch.cuda.is_available()
            if self._torch_available:
                self._gpu_count = torch.cuda.device_count()
                logger.info(
                    f"GPU monitoring initialized: {self._gpu_count} GPU(s) detected"
                )

                # Initialize history deques for each GPU
                for i in range(self._gpu_count):
                    self.temperature_history[i] = deque(maxlen=self.history_size)
                    self.utilization_history[i] = deque(maxlen=self.history_size)
            else:
                logger.warning("No GPUs available for monitoring")
        except ImportError:
            logger.warning("PyTorch not installed - GPU monitoring unavailable")
            self._torch_available = False

    async def get_gpu_temperatures(self) -> Dict[str, Any]:
        """
        Get current GPU temperatures for all devices.

        Returns:
            Dictionary with temperature data for each GPU
        """
        if not self._torch_available:
            return {
                "available": False,
                "error": "GPU monitoring not available - PyTorch not installed or no GPUs detected",
            }

        try:
            import torch

            temperatures = []
            timestamp = datetime.now(timezone.utc).isoformat()

            for i in range(self._gpu_count):
                try:
                    # Get device properties
                    props = torch.cuda.get_device_properties(i)
                    device_name = torch.cuda.get_device_name(i)

                    # Try to get temperature from ROCm SMI
                    temp = await self._get_gpu_temperature_rocm(i)

                    # Store in history
                    if temp is not None:
                        self.temperature_history[i].append(
                            {"timestamp": timestamp, "temperature": temp}
                        )

                    gpu_info = {
                        "device_id": i,
                        "name": device_name,
                        "temperature_c": temp,
                        "memory_total_gb": round(props.total_memory / (1024**3), 2),
                        "memory_used_gb": round(
                            torch.cuda.memory_allocated(i) / (1024**3), 2
                        ),
                        "memory_reserved_gb": round(
                            torch.cuda.memory_reserved(i) / (1024**3), 2
                        ),
                    }

                    temperatures.append(gpu_info)
                except Exception as e:
                    logger.error(f"Error reading GPU {i} temperature: {e}")
                    temperatures.append(
                        {
                            "device_id": i,
                            "name": "Unknown",
                            "temperature_c": None,
                            "error": "Failed to read GPU data",
                        }
                    )

            self._last_update = timestamp

            return {
                "available": True,
                "gpu_count": self._gpu_count,
                "timestamp": timestamp,
                "gpus": temperatures,
            }
        except Exception as e:
            logger.error(f"Error getting GPU temperatures: {e}")
            return {"available": False, "error": "Failed to get GPU temperatures"}

    async def _get_gpu_temperature_rocm(self, device_id: int) -> Optional[float]:
        """
        Get GPU temperature using ROCm SMI.

        Args:
            device_id: GPU device ID

        Returns:
            Temperature in Celsius or None if unavailable
        """
        try:
            # Try rocm-smi command
            proc = await asyncio.create_subprocess_exec(
                "rocm-smi",
                "--showtemp",
                "--device",
                str(device_id),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=5.0)

            if proc.returncode == 0:
                output = stdout.decode("utf-8")
                # Parse temperature from rocm-smi output
                # Format: "Temperature (Sensor edge) (C): 45.0"
                for line in output.split("\n"):
                    if "Temperature" in line and "(C):" in line:
                        temp_str = line.split(":")[-1].strip()
                        return float(temp_str)
        except FileNotFoundError:
            # rocm-smi not available, try alternative methods
            logger.debug("rocm-smi not found, attempting alternative methods")
        except asyncio.TimeoutError:
            logger.warning(f"Timeout reading temperature for GPU {device_id}")
        except Exception as e:
            logger.debug(f"Error using rocm-smi for GPU {device_id}: {e}")

        # Try reading from sysfs (hwmon)
        try:
            import glob

            # Look for hwmon devices associated with AMD GPUs
            hwmon_paths = glob.glob(
                f"/sys/class/drm/card{device_id}/device/hwmon/hwmon*/temp*_input"
            )

            for path in hwmon_paths:
                with open(path, "r") as f:
                    # Temperature is in millidegrees Celsius
                    temp_millic = int(f.read().strip())
                    return temp_millic / 1000.0
        except Exception as e:
            logger.debug(f"Error reading from sysfs for GPU {device_id}: {e}")

        # If all methods fail, return None
        logger.debug(f"Unable to read temperature for GPU {device_id}")
        return None

    async def get_gpu_status(self) -> Dict[str, Any]:
        """
        Get comprehensive GPU status including temperature, utilization, and health.

        Returns:
            Complete GPU status information
        """
        if not self._torch_available:
            return {
                "available": False,
                "error": "GPU monitoring not available",
            }

        try:
            import torch

            temp_data = await self.get_gpu_temperatures()
            if not temp_data.get("available"):
                return temp_data

            status = {
                "available": True,
                "gpu_count": self._gpu_count,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "gpus": [],
            }

            for i in range(self._gpu_count):
                try:
                    props = torch.cuda.get_device_properties(i)
                    device_name = torch.cuda.get_device_name(i)

                    # Get memory info
                    memory_allocated = torch.cuda.memory_allocated(i)
                    memory_reserved = torch.cuda.memory_reserved(i)
                    memory_total = props.total_memory
                    memory_free = memory_total - memory_allocated

                    # Get temperature
                    temp = None
                    for gpu in temp_data.get("gpus", []):
                        if gpu.get("device_id") == i:
                            temp = gpu.get("temperature_c")
                            break

                    # Calculate utilization percentage
                    utilization_pct = (
                        (memory_allocated / memory_total * 100)
                        if memory_total > 0
                        else 0
                    )

                    # Determine health status based on temperature
                    health_status = self._determine_health_status(temp)

                    gpu_status = {
                        "device_id": i,
                        "name": device_name,
                        "temperature_c": temp,
                        "health_status": health_status,
                        "memory": {
                            "total_gb": round(memory_total / (1024**3), 2),
                            "used_gb": round(memory_allocated / (1024**3), 2),
                            "free_gb": round(memory_free / (1024**3), 2),
                            "reserved_gb": round(memory_reserved / (1024**3), 2),
                            "utilization_percent": round(utilization_pct, 2),
                        },
                        "compute_capability": f"{props.major}.{props.minor}",
                        "multi_processor_count": props.multi_processor_count,
                    }

                    status["gpus"].append(gpu_status)
                except Exception as e:
                    logger.error(f"Error getting status for GPU {i}: {e}")
                    status["gpus"].append(
                        {
                            "device_id": i,
                            "name": "Unknown",
                            "error": "Failed to read GPU status",
                        }
                    )

            return status
        except Exception as e:
            logger.error(f"Error getting GPU status: {e}")
            return {"available": False, "error": "Failed to get GPU status"}

    def _determine_health_status(self, temperature: Optional[float]) -> str:
        """
        Determine GPU health status based on temperature.

        Args:
            temperature: GPU temperature in Celsius

        Returns:
            Health status string: 'healthy', 'warm', 'hot', 'critical', or 'unknown'
        """
        if temperature is None:
            return "unknown"

        # Temperature thresholds for AMD GPUs (conservative)
        if temperature < 60:
            return "healthy"
        elif temperature < 75:
            return "warm"
        elif temperature < 85:
            return "hot"
        else:
            return "critical"

    async def get_temperature_history(
        self, device_id: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Get historical temperature data.

        Args:
            device_id: Specific GPU device ID, or None for all GPUs

        Returns:
            Historical temperature data
        """
        if not self._torch_available:
            return {
                "available": False,
                "error": "GPU monitoring not available",
            }

        if device_id is not None:
            if device_id not in self.temperature_history:
                return {
                    "available": False,
                    "error": f"Invalid device_id: {device_id}",
                }
            return {
                "available": True,
                "device_id": device_id,
                "history": list(self.temperature_history[device_id]),
            }
        else:
            history = {}
            for dev_id, temps in self.temperature_history.items():
                history[dev_id] = list(temps)
            return {
                "available": True,
                "history": history,
            }

    async def get_max_temperatures(self) -> Dict[str, Any]:
        """
        Get maximum recorded temperatures for each GPU.

        Returns:
            Maximum temperatures across history
        """
        if not self._torch_available:
            return {
                "available": False,
                "error": "GPU monitoring not available",
            }

        max_temps = {}
        for device_id, history in self.temperature_history.items():
            if history:
                temps = [
                    entry["temperature"]
                    for entry in history
                    if entry["temperature"] is not None
                ]
                if temps:
                    max_temps[device_id] = {
                        "max_temperature_c": max(temps),
                        "avg_temperature_c": sum(temps) / len(temps),
                        "readings_count": len(temps),
                    }

        return {
            "available": True,
            "max_temperatures": max_temps,
        }

    async def get_least_loaded_gpu(self) -> Optional[int]:
        """
        Get the GPU with the lowest memory utilization.

        This method identifies the least loaded GPU by checking memory usage,
        making it ideal for distributing workload across multiple GPUs.

        Returns:
            GPU device ID with lowest utilization, or None if no GPUs available
        """
        if not self._torch_available or self._gpu_count == 0:
            return None

        try:
            import torch

            # Get current GPU status
            gpu_loads = []
            for i in range(self._gpu_count):
                try:
                    memory_allocated = torch.cuda.memory_allocated(i)
                    memory_total = torch.cuda.get_device_properties(i).total_memory
                    utilization = (
                        (memory_allocated / memory_total) if memory_total > 0 else 1.0
                    )

                    gpu_loads.append(
                        {
                            "device_id": i,
                            "utilization": utilization,
                            "memory_allocated": memory_allocated,
                        }
                    )
                except Exception as e:
                    logger.warning(f"Failed to get load for GPU {i}: {e}")
                    # Mark as fully loaded if we can't read it
                    gpu_loads.append(
                        {
                            "device_id": i,
                            "utilization": 1.0,
                            "memory_allocated": float("inf"),
                        }
                    )

            # Sort by utilization (lowest first)
            gpu_loads.sort(key=lambda x: x["utilization"])

            # Return the least loaded GPU
            least_loaded = gpu_loads[0]
            logger.debug(
                f"Selected GPU {least_loaded['device_id']} "
                f"(utilization: {least_loaded['utilization']*100:.1f}%)"
            )
            return least_loaded["device_id"]

        except Exception as e:
            logger.error(f"Error finding least loaded GPU: {e}")
            # Fallback to GPU 0
            return 0

    async def get_available_gpus(self) -> List[int]:
        """
        Get list of all available GPU device IDs sorted by load (least loaded first).

        Returns:
            List of GPU device IDs sorted by utilization (ascending)
        """
        if not self._torch_available or self._gpu_count == 0:
            return []

        try:
            import torch

            # Get current GPU status
            gpu_loads = []
            for i in range(self._gpu_count):
                try:
                    memory_allocated = torch.cuda.memory_allocated(i)
                    memory_total = torch.cuda.get_device_properties(i).total_memory
                    utilization = (
                        (memory_allocated / memory_total) if memory_total > 0 else 1.0
                    )

                    gpu_loads.append(
                        {
                            "device_id": i,
                            "utilization": utilization,
                        }
                    )
                except Exception as e:
                    logger.warning(f"Failed to get load for GPU {i}: {e}")
                    gpu_loads.append(
                        {
                            "device_id": i,
                            "utilization": 1.0,
                        }
                    )

            # Sort by utilization (lowest first)
            gpu_loads.sort(key=lambda x: x["utilization"])

            # Return list of device IDs
            return [gpu["device_id"] for gpu in gpu_loads]

        except Exception as e:
            logger.error(f"Error getting available GPUs: {e}")
            # Fallback to all GPUs in order
            return list(range(self._gpu_count))


# Global instance
_gpu_monitoring_service = None


def get_gpu_monitoring_service() -> GPUMonitoringService:
    """Get or create the global GPU monitoring service instance."""
    global _gpu_monitoring_service
    if _gpu_monitoring_service is None:
        _gpu_monitoring_service = GPUMonitoringService()
    return _gpu_monitoring_service
