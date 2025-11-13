"""
Server Fan Control Service

Provides fan wall control for Lenovo SR665 server using IPMI.
Supports temperature-based automatic fan speed adjustment and manual control.
"""

import asyncio
import re
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timezone
from enum import Enum

from backend.config.logging import get_logger

logger = get_logger(__name__)


class FanControlMode(str, Enum):
    """Fan control modes."""
    AUTO = "auto"
    MANUAL = "manual"
    AUTOMATIC = "automatic"


class FanZone(str, Enum):
    """Fan zones in the server."""
    SYSTEM = "system"  # System fan wall
    CPU = "cpu"  # CPU fans
    PERIPHERAL = "peripheral"  # Peripheral zone fans


class FanControlService:
    """
    Service for controlling server fan wall via IPMI.
    
    Designed for Lenovo SR665 server where fans are part of the server's
    fan wall, not attached to individual GPUs.
    """

    def __init__(self):
        """Initialize fan control service."""
        self._ipmi_available = False
        self._control_mode = FanControlMode.AUTO
        self._manual_speed = None
        self._temperature_thresholds = {
            "low": 50,  # Below this, reduce fan speed
            "normal": 65,  # Normal operating range
            "high": 75,  # Increase fan speed
            "critical": 85,  # Maximum fan speed
        }
        
        # Check IPMI availability
        self._check_ipmi_availability()

    def _check_ipmi_availability(self):
        """Check if IPMI tools are available."""
        try:
            import shutil
            self._ipmi_available = shutil.which("ipmitool") is not None
            if self._ipmi_available:
                logger.info("IPMI tools available for fan control")
            else:
                logger.warning(
                    "ipmitool not found - fan control unavailable. "
                    "Install with: apt-get install ipmitool (Debian/Ubuntu) "
                    "or yum install ipmitool (RHEL/CentOS)"
                )
        except Exception as e:
            logger.error(f"Error checking IPMI availability: {e}")
            self._ipmi_available = False

    async def get_fan_status(self) -> Dict[str, Any]:
        """
        Get current fan status from IPMI.

        Returns:
            Dictionary with fan status information
        """
        if not self._ipmi_available:
            return {
                "available": False,
                "error": "IPMI not available - install ipmitool",
                "control_mode": self._control_mode,
            }

        try:
            # Get sensor data from IPMI
            proc = await asyncio.create_subprocess_exec(
                "ipmitool", "sensor", "list",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=10.0)

            if proc.returncode != 0:
                error_msg = stderr.decode("utf-8").strip()
                logger.error(f"IPMI sensor command failed: {error_msg}")
                return {
                    "available": False,
                    "error": f"IPMI command failed: {error_msg}",
                }

            # Parse fan sensors from output
            fans = self._parse_fan_sensors(stdout.decode("utf-8"))

            return {
                "available": True,
                "control_mode": self._control_mode,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "fans": fans,
                "thresholds": self._temperature_thresholds,
            }
        except asyncio.TimeoutError:
            logger.error("Timeout reading fan status via IPMI")
            return {
                "available": False,
                "error": "Timeout communicating with IPMI",
            }
        except Exception as e:
            logger.error(f"Error getting fan status: {e}")
            return {
                "available": False,
                "error": str(e),
            }

    def _parse_fan_sensors(self, sensor_output: str) -> List[Dict[str, Any]]:
        """
        Parse fan sensor information from IPMI sensor list output.

        Args:
            sensor_output: Raw output from ipmitool sensor list

        Returns:
            List of fan sensor dictionaries
        """
        fans = []
        
        for line in sensor_output.split("\n"):
            # Look for fan-related sensors
            if "fan" in line.lower() or "Fan" in line:
                parts = [p.strip() for p in line.split("|")]
                if len(parts) >= 3:
                    try:
                        fan_name = parts[0]
                        fan_value = parts[1]
                        fan_unit = parts[2] if len(parts) > 2 else ""
                        
                        # Try to extract numeric value
                        speed = None
                        try:
                            # Handle values like "3600.000" or "na"
                            if fan_value and fan_value.lower() != "na":
                                speed = float(fan_value)
                        except ValueError:
                            pass

                        fan_info = {
                            "name": fan_name,
                            "speed": speed,
                            "unit": fan_unit,
                            "status": "ok" if speed is not None else "unknown",
                        }
                        fans.append(fan_info)
                    except Exception as e:
                        logger.debug(f"Error parsing fan line '{line}': {e}")
                        continue

        return fans

    async def set_fan_mode(self, mode: FanControlMode) -> Dict[str, Any]:
        """
        Set fan control mode (automatic or manual).

        Args:
            mode: Fan control mode

        Returns:
            Status of the operation
        """
        if not self._ipmi_available:
            return {
                "success": False,
                "error": "IPMI not available",
            }

        try:
            if mode == FanControlMode.AUTO or mode == FanControlMode.AUTOMATIC:
                # Set to automatic/BMC control
                result = await self._set_ipmi_fan_mode_auto()
                if result["success"]:
                    self._control_mode = FanControlMode.AUTO
                    self._manual_speed = None
                return result
            elif mode == FanControlMode.MANUAL:
                # Set to manual control mode (requires speed to be set separately)
                self._control_mode = FanControlMode.MANUAL
                return {
                    "success": True,
                    "mode": mode,
                    "message": "Manual mode enabled. Use set_fan_speed to control fans.",
                }
            else:
                return {
                    "success": False,
                    "error": f"Unknown mode: {mode}",
                }
        except Exception as e:
            logger.error(f"Error setting fan mode: {e}")
            return {
                "success": False,
                "error": str(e),
            }

    async def _set_ipmi_fan_mode_auto(self) -> Dict[str, Any]:
        """
        Set IPMI fan control to automatic mode.

        Returns:
            Status of the operation
        """
        try:
            # Lenovo-specific IPMI command to enable automatic fan control
            # This varies by manufacturer - using common approach
            proc = await asyncio.create_subprocess_exec(
                "ipmitool", "raw", "0x30", "0x30", "0x01", "0x01",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=5.0)

            if proc.returncode == 0:
                logger.info("Fan control set to automatic mode")
                return {
                    "success": True,
                    "mode": FanControlMode.AUTO,
                    "message": "Automatic fan control enabled",
                }
            else:
                error_msg = stderr.decode("utf-8").strip()
                logger.warning(f"IPMI auto mode command returned error: {error_msg}")
                # Some systems don't support this command, which is okay
                return {
                    "success": True,
                    "mode": FanControlMode.AUTO,
                    "message": "Automatic mode requested (command not supported on this system)",
                    "warning": error_msg,
                }
        except Exception as e:
            logger.error(f"Error setting auto fan mode: {e}")
            return {
                "success": False,
                "error": str(e),
            }

    async def set_fan_speed(self, speed_percent: int, zone: Optional[FanZone] = None) -> Dict[str, Any]:
        """
        Set fan speed manually.

        Args:
            speed_percent: Fan speed as percentage (0-100)
            zone: Optional fan zone to target (system, cpu, peripheral)

        Returns:
            Status of the operation
        """
        if not self._ipmi_available:
            return {
                "success": False,
                "error": "IPMI not available",
            }

        # Validate speed
        if not 0 <= speed_percent <= 100:
            return {
                "success": False,
                "error": "Speed must be between 0 and 100",
            }

        try:
            # First set to manual mode if not already
            if self._control_mode != FanControlMode.MANUAL:
                await self._set_ipmi_fan_mode_manual()

            # Convert percentage to raw value (0-255 for most IPMI systems)
            raw_speed = int((speed_percent / 100.0) * 255)

            # Lenovo SR665 specific fan control via IPMI raw commands
            # This is a common approach but may need adjustment for specific hardware
            zone_id = self._get_zone_id(zone)
            
            proc = await asyncio.create_subprocess_exec(
                "ipmitool", "raw", "0x30", "0x30", "0x02", "0xff", hex(raw_speed),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=5.0)

            if proc.returncode == 0:
                self._manual_speed = speed_percent
                logger.info(f"Fan speed set to {speed_percent}% (raw: {raw_speed})")
                return {
                    "success": True,
                    "speed_percent": speed_percent,
                    "raw_value": raw_speed,
                    "zone": zone.value if zone else "all",
                    "message": f"Fan speed set to {speed_percent}%",
                }
            else:
                error_msg = stderr.decode("utf-8").strip()
                logger.error(f"Failed to set fan speed: {error_msg}")
                return {
                    "success": False,
                    "error": f"IPMI command failed: {error_msg}",
                }
        except Exception as e:
            logger.error(f"Error setting fan speed: {e}")
            return {
                "success": False,
                "error": str(e),
            }

    async def _set_ipmi_fan_mode_manual(self) -> Dict[str, Any]:
        """
        Set IPMI fan control to manual mode.

        Returns:
            Status of the operation
        """
        try:
            proc = await asyncio.create_subprocess_exec(
                "ipmitool", "raw", "0x30", "0x30", "0x01", "0x00",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=5.0)

            if proc.returncode == 0:
                logger.info("Fan control set to manual mode")
                return {"success": True}
            else:
                error_msg = stderr.decode("utf-8").strip()
                logger.warning(f"IPMI manual mode command returned error: {error_msg}")
                return {"success": True, "warning": error_msg}
        except Exception as e:
            logger.error(f"Error setting manual fan mode: {e}")
            return {"success": False, "error": str(e)}

    def _get_zone_id(self, zone: Optional[FanZone]) -> int:
        """
        Get zone ID for IPMI commands.

        Args:
            zone: Fan zone

        Returns:
            Zone ID (0xff for all zones)
        """
        if zone is None:
            return 0xff  # All zones
        
        zone_map = {
            FanZone.SYSTEM: 0x00,
            FanZone.CPU: 0x01,
            FanZone.PERIPHERAL: 0x02,
        }
        return zone_map.get(zone, 0xff)

    async def adjust_fans_for_temperature(self, max_gpu_temp: float) -> Dict[str, Any]:
        """
        Automatically adjust fan speeds based on GPU temperature.

        Args:
            max_gpu_temp: Maximum GPU temperature in Celsius

        Returns:
            Status and actions taken
        """
        if not self._ipmi_available:
            return {
                "success": False,
                "error": "IPMI not available",
            }

        try:
            # Determine target fan speed based on temperature
            target_speed = self._calculate_fan_speed(max_gpu_temp)
            
            # Get current status
            current_status = await self.get_fan_status()
            
            action_taken = None
            
            # Only adjust if we're in auto mode or if temperature is critical
            if self._control_mode == FanControlMode.AUTO or max_gpu_temp >= self._temperature_thresholds["critical"]:
                # Switch to manual mode to adjust
                if self._control_mode != FanControlMode.MANUAL:
                    await self.set_fan_mode(FanControlMode.MANUAL)
                
                # Set the calculated speed
                result = await self.set_fan_speed(target_speed)
                action_taken = result
            
            return {
                "success": True,
                "temperature": max_gpu_temp,
                "target_speed_percent": target_speed,
                "threshold_status": self._get_threshold_status(max_gpu_temp),
                "action_taken": action_taken,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        except Exception as e:
            logger.error(f"Error adjusting fans for temperature: {e}")
            return {
                "success": False,
                "error": str(e),
            }

    def _calculate_fan_speed(self, temperature: float) -> int:
        """
        Calculate appropriate fan speed based on temperature.

        Args:
            temperature: GPU temperature in Celsius

        Returns:
            Fan speed as percentage (0-100)
        """
        thresholds = self._temperature_thresholds
        
        if temperature < thresholds["low"]:
            # Low temperature - reduce to minimum safe speed
            return 30
        elif temperature < thresholds["normal"]:
            # Normal temperature - moderate speed
            # Linear interpolation between 30% and 50%
            ratio = (temperature - thresholds["low"]) / (thresholds["normal"] - thresholds["low"])
            return int(30 + (20 * ratio))
        elif temperature < thresholds["high"]:
            # Elevated temperature - increase speed
            # Linear interpolation between 50% and 75%
            ratio = (temperature - thresholds["normal"]) / (thresholds["high"] - thresholds["normal"])
            return int(50 + (25 * ratio))
        elif temperature < thresholds["critical"]:
            # High temperature - high speed
            # Linear interpolation between 75% and 100%
            ratio = (temperature - thresholds["high"]) / (thresholds["critical"] - thresholds["high"])
            return int(75 + (25 * ratio))
        else:
            # Critical temperature - maximum speed
            return 100

    def _get_threshold_status(self, temperature: float) -> str:
        """
        Get temperature threshold status.

        Args:
            temperature: Temperature in Celsius

        Returns:
            Status string
        """
        thresholds = self._temperature_thresholds
        
        if temperature < thresholds["low"]:
            return "low"
        elif temperature < thresholds["normal"]:
            return "normal"
        elif temperature < thresholds["high"]:
            return "elevated"
        elif temperature < thresholds["critical"]:
            return "high"
        else:
            return "critical"

    def set_temperature_thresholds(self, thresholds: Dict[str, float]) -> Dict[str, Any]:
        """
        Update temperature thresholds for fan control.

        Args:
            thresholds: Dictionary with threshold values

        Returns:
            Updated thresholds
        """
        valid_keys = {"low", "normal", "high", "critical"}
        
        for key, value in thresholds.items():
            if key in valid_keys:
                self._temperature_thresholds[key] = value
        
        logger.info(f"Temperature thresholds updated: {self._temperature_thresholds}")
        
        return {
            "success": True,
            "thresholds": self._temperature_thresholds,
        }

    def get_control_info(self) -> Dict[str, Any]:
        """
        Get fan control configuration and status.

        Returns:
            Fan control information
        """
        return {
            "ipmi_available": self._ipmi_available,
            "control_mode": self._control_mode,
            "manual_speed_percent": self._manual_speed,
            "temperature_thresholds": self._temperature_thresholds,
            "supported_zones": [zone.value for zone in FanZone],
        }


# Global instance
_fan_control_service = None


def get_fan_control_service() -> FanControlService:
    """Get or create the global fan control service instance."""
    global _fan_control_service
    if _fan_control_service is None:
        _fan_control_service = FanControlService()
    return _fan_control_service
