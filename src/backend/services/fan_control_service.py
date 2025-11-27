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
from backend.config.settings import get_settings

logger = get_logger(__name__)


async def get_ipmi_credentials_from_db():
    """
    Attempt to load IPMI credentials from database settings.
    Falls back to None if database is not available.
    
    Returns:
        Dict with ipmi_host, ipmi_username, ipmi_password, ipmi_interface or None
    """
    try:
        from backend.database.connection import database_manager
        from backend.services.settings_service import SettingsService
        
        # Check if database is connected
        if not database_manager.is_connected:
            return None
            
        async with database_manager.get_session() as session:
            service = SettingsService(session)
            
            # Try to fetch each IPMI setting
            ipmi_host = await service.get_setting("ipmi_host")
            ipmi_username = await service.get_setting("ipmi_username")
            ipmi_password = await service.get_setting("ipmi_password")
            ipmi_interface = await service.get_setting("ipmi_interface")
            
            # Return dict if any value exists
            if any([ipmi_host, ipmi_username, ipmi_password, ipmi_interface]):
                return {
                    "ipmi_host": ipmi_host.value if ipmi_host else None,
                    "ipmi_username": ipmi_username.value if ipmi_username else None,
                    "ipmi_password": ipmi_password.value if ipmi_password else None,
                    "ipmi_interface": ipmi_interface.value if ipmi_interface else "lanplus",
                }
            
            return None
            
    except Exception as e:
        logger.debug(f"Could not load IPMI credentials from database: {e}")
        return None


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


class ServerManufacturer(str, Enum):
    """Supported server manufacturers for IPMI fan control."""
    LENOVO = "lenovo"
    DELL = "dell"
    HP = "hp"
    SUPERMICRO = "supermicro"
    GENERIC = "generic"  # Generic/fallback IPMI commands


class FanControlService:
    """
    Service for controlling server fan wall via IPMI.
    
    Designed for Lenovo SR665 server where fans are part of the server's
    fan wall, not attached to individual GPUs.
    """

    def __init__(
        self,
        manufacturer: ServerManufacturer = ServerManufacturer.LENOVO,
        ipmi_host: Optional[str] = None,
        ipmi_username: Optional[str] = None,
        ipmi_password: Optional[str] = None,
        ipmi_interface: str = "lanplus",
    ):
        """
        Initialize fan control service.
        
        Args:
            manufacturer: Server manufacturer for IPMI command selection
            ipmi_host: BMC/XCC IP address or hostname (overrides settings)
            ipmi_username: BMC/XCC username (overrides settings)
            ipmi_password: BMC/XCC password (overrides settings)
            ipmi_interface: IPMI interface type (default: lanplus for remote access)
        """
        self._ipmi_available = False
        self._control_mode = FanControlMode.AUTO
        self._manual_speed = None
        self._manual_control_supported = None  # Will be checked on first use
        self._manufacturer = manufacturer
        self._temperature_thresholds = {
            "low": 50,  # Below this, reduce fan speed
            "normal": 65,  # Normal operating range
            "high": 75,  # Increase fan speed
            "critical": 85,  # Maximum fan speed
        }
        
        # Load credentials from environment settings if not provided
        settings = get_settings()
        self._ipmi_host = ipmi_host or settings.ipmi_host
        self._ipmi_username = ipmi_username or settings.ipmi_username
        self._ipmi_password = ipmi_password or settings.ipmi_password
        self._ipmi_interface = ipmi_interface or settings.ipmi_interface
        
        # Check IPMI availability
        self._check_ipmi_availability()
        
        # Log initialization (don't log credentials)
        auth_status = "with authentication" if self._ipmi_host and self._ipmi_username else "without authentication"
        logger.info(f"Fan control service initialized for manufacturer: {manufacturer.value} {auth_status}")
    
    async def reload_credentials_from_db(self):
        """
        Reload IPMI credentials from database.
        This allows updating credentials without restarting the service.
        """
        db_creds = await get_ipmi_credentials_from_db()
        if db_creds:
            # Update credentials from database (override current values)
            if db_creds.get("ipmi_host"):
                self._ipmi_host = db_creds["ipmi_host"]
            if db_creds.get("ipmi_username"):
                self._ipmi_username = db_creds["ipmi_username"]
            if db_creds.get("ipmi_password"):
                self._ipmi_password = db_creds["ipmi_password"]
            if db_creds.get("ipmi_interface"):
                self._ipmi_interface = db_creds["ipmi_interface"]
            
            # Reset support status since credentials changed
            self._manual_control_supported = None
            
            auth_status = "with authentication" if self._ipmi_host and self._ipmi_username else "without authentication"
            logger.info(f"IPMI credentials reloaded from database {auth_status}")

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

    def _build_ipmi_command(self, raw_command: List[str]) -> List[str]:
        """
        Build complete IPMI command with authentication if configured.
        
        Args:
            raw_command: The raw IPMI command arguments (e.g., ["raw", "0x30", "0x30"])
        
        Returns:
            Complete ipmitool command with authentication parameters
        
        Examples:
            # Local/in-band (no credentials):
            ["ipmitool", "raw", "0x30", "0x30", ...]
            
            # Remote/out-of-band (with credentials):
            ["ipmitool", "-I", "lanplus", "-H", "192.168.1.100", 
             "-U", "USERID", "-P", "PASSW0RD", "raw", "0x30", "0x30", ...]
        """
        cmd = ["ipmitool"]
        
        # Add authentication parameters if configured (out-of-band/remote access)
        if self._ipmi_host and self._ipmi_username and self._ipmi_password:
            cmd.extend([
                "-I", self._ipmi_interface,  # Interface (lanplus for remote)
                "-H", self._ipmi_host,        # BMC/XCC hostname or IP
                "-U", self._ipmi_username,    # Username
                "-P", self._ipmi_password,    # Password
            ])
        # else: in-band/local access (no authentication needed)
        
        # Add the actual command
        cmd.extend(raw_command)
        
        return cmd

    def _is_authentication_error(self, error_msg: str) -> bool:
        """
        Check if IPMI error is related to authentication/authorization.
        
        Args:
            error_msg: Error message from ipmitool
        
        Returns:
            True if error is authentication-related
        """
        auth_error_patterns = [
            "authentication",
            "login",
            "password",
            "username",
            "unauthorized",
            "access denied",
            "insufficient privilege",
            "session",
            "0xd4",  # IPMI: Insufficient privilege level
            "0xcb",  # IPMI: Invalid data field in request
        ]
        error_lower = error_msg.lower()
        return any(pattern in error_lower for pattern in auth_error_patterns)

    def _get_ipmi_commands(self) -> Dict[str, List[str]]:
        """
        Get manufacturer-specific IPMI raw commands.
        
        Returns:
            Dictionary with command lists for auto_mode, manual_mode, and set_speed
        
        Notes:
            - Lenovo: ThinkSystem servers use netfn 0x3A for fan control (cmd 0x07 for mode, 0x32 for speed)
            - Dell: Uses netfn 0x30, cmd 0x30 for fan control
            - HP: Uses netfn 0x30, cmd 0x30 for iLO fan control
            - Supermicro: Uses netfn 0x30, cmd 0x45 for fan control
        """
        commands = {
            ServerManufacturer.LENOVO: {
                # Lenovo ThinkSystem IPMI fan control commands (netfn 0x3A)
                # These commands work on Lenovo ThinkSystem servers (e.g., SR665)
                # Auto mode: ipmitool raw 0x3a 0x07 0xFF 0xFF 0x00
                "auto_mode": ["0x3a", "0x07", "0xFF", "0xFF", "0x00"],
                # Manual mode: ipmitool raw 0x3a 0x07 0xFF 0xFF 0x01
                "manual_mode": ["0x3a", "0x07", "0xFF", "0xFF", "0x01"],
                # Set speed: ipmitool raw 0x3A 0x32 0xff 0x00 0x00 0x03 <speed_hex>
                # where speed_hex is 0x00-0x64 (0-100%)
                "set_speed_prefix": ["0x3a", "0x32", "0xff", "0x00", "0x00", "0x03"],
                "note": "Lenovo ThinkSystem fan control using IPMI OEM commands (netfn 0x3A). "
                        "Tested on Lenovo SR665 servers.",
            },
            ServerManufacturer.DELL: {
                "auto_mode": ["0x30", "0x30", "0x01", "0x01"],
                "manual_mode": ["0x30", "0x30", "0x01", "0x00"],
                "set_speed_prefix": ["0x30", "0x30", "0x02", "0xff"],
                "note": "Dell iDRAC fan control commands",
            },
            ServerManufacturer.HP: {
                "auto_mode": ["0x30", "0x30", "0x01", "0x01"],
                "manual_mode": ["0x30", "0x30", "0x01", "0x00"],
                "set_speed_prefix": ["0x30", "0x30", "0x02", "0xff"],
                "note": "HP iLO fan control commands",
            },
            ServerManufacturer.SUPERMICRO: {
                "auto_mode": ["0x30", "0x45", "0x01", "0x01"],
                "manual_mode": ["0x30", "0x45", "0x01", "0x00"],
                "set_speed_prefix": ["0x30", "0x70", "0x66", "0x01", "0x00"],
                "note": "Supermicro IPMI fan control commands",
            },
            ServerManufacturer.GENERIC: {
                "auto_mode": ["0x30", "0x30", "0x01", "0x01"],
                "manual_mode": ["0x30", "0x30", "0x01", "0x00"],
                "set_speed_prefix": ["0x30", "0x30", "0x02", "0xff"],
                "note": "Generic IPMI fan control commands (may not work on all hardware)",
            },
        }
        
        return commands.get(self._manufacturer, commands[ServerManufacturer.GENERIC])

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
            cmd = self._build_ipmi_command(["sensor", "list"])
            proc = await asyncio.create_subprocess_exec(
                *cmd,
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
                "error": "Operation failed",
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
                "error": "Operation failed",
            }

    async def _set_ipmi_fan_mode_auto(self) -> Dict[str, Any]:
        """
        Set IPMI fan control to automatic mode.

        Returns:
            Status of the operation
        """
        try:
            # Get manufacturer-specific IPMI commands
            commands = self._get_ipmi_commands()
            auto_cmd = commands["auto_mode"]
            
            logger.debug(f"Setting auto mode for {self._manufacturer.value}: {' '.join(auto_cmd)}")
            
            cmd = self._build_ipmi_command(["raw"] + auto_cmd)
            proc = await asyncio.create_subprocess_exec(
                *cmd,
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
                # Check for "Invalid command" error (0xc1) which indicates unsupported hardware
                if "Invalid command" in error_msg or "0xc1" in error_msg.lower():
                    commands = self._get_ipmi_commands()
                    logger.info(
                        f"IPMI raw fan control commands not supported on this {self._manufacturer.value} hardware. "
                        f"Note: {commands.get('note', 'No additional info')}"
                    )
                    return {
                        "success": True,
                        "mode": FanControlMode.AUTO,
                        "message": f"Automatic mode enabled (using BMC default - {self._manufacturer.value} does not support these IPMI fan commands)",
                        "warning": "IPMI raw commands not supported on this system",
                        "manufacturer_note": commands.get('note'),
                    }
                else:
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
                "error": "Operation failed",
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
                "error": "IPMI not available - install ipmitool to enable fan control",
            }

        # Check if we already know manual control is not supported
        if self._manual_control_supported is False:
            return {
                "success": False,
                "error": "Manual fan control not supported on this hardware",
                "details": "This server's BMC does not support the IPMI OEM commands needed for manual fan control. "
                           "The system will use automatic fan control managed by the BMC firmware.",
                "recommendation": "Use automatic fan control mode or consult your server's documentation for supported fan control methods",
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
                mode_result = await self._set_ipmi_fan_mode_manual()
                # Log but don't fail if manual mode setting is not supported
                if not mode_result.get("success"):
                    logger.warning("Could not set manual mode, continuing with speed setting attempt")

            # Convert percentage to raw value
            # Lenovo uses 0x00-0x64 (0-100%), other manufacturers use 0-255
            if self._manufacturer == ServerManufacturer.LENOVO:
                raw_speed = speed_percent  # Direct percentage (0-100)
            else:
                raw_speed = int((speed_percent / 100.0) * 255)  # 0-255 scale

            # Get manufacturer-specific IPMI commands
            commands = self._get_ipmi_commands()
            speed_cmd_prefix = commands["set_speed_prefix"]
            zone_id = self._get_zone_id(zone)
            
            # Build the full command with speed value
            speed_cmd = list(speed_cmd_prefix) + [hex(raw_speed)]
            
            logger.debug(f"Setting fan speed for {self._manufacturer.value}: {' '.join(speed_cmd)}")
            
            cmd = self._build_ipmi_command(["raw"] + speed_cmd)
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=5.0)

            if proc.returncode == 0:
                self._manual_speed = speed_percent
                self._manual_control_supported = True
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
                
                # Check for authentication errors first
                if self._is_authentication_error(error_msg):
                    logger.error(f"IPMI authentication failed: {error_msg}")
                    auth_configured = bool(self._ipmi_host and self._ipmi_username and self._ipmi_password)
                    return {
                        "success": False,
                        "error": "IPMI authentication failed",
                        "details": "Unable to authenticate with BMC/XCC. Check credentials and ensure IPMI over LAN is enabled." if auth_configured else "IPMI credentials not configured. Remote IPMI access requires authentication.",
                        "ipmi_error": error_msg,
                        "recommendation": "Configure IPMI credentials (host, username, password) and ensure IPMI over LAN is enabled in XCC/BMC settings" if not auth_configured else "Verify IPMI credentials are correct and IPMI over LAN is enabled in XCC/BMC settings",
                        "credentials_configured": auth_configured,
                    }
                # Check for "Invalid command" error (0xc1) which may indicate unsupported hardware or missing auth
                elif "Invalid command" in error_msg or "0xc1" in error_msg.lower():
                    # If credentials are not configured, this might be an auth issue
                    if not (self._ipmi_host and self._ipmi_username and self._ipmi_password):
                        logger.warning(
                            f"IPMI command failed with 0xc1 error and no credentials configured. "
                            f"This may require authentication. Error: {error_msg}"
                        )
                        return {
                            "success": False,
                            "error": "IPMI command failed - authentication may be required",
                            "details": "The IPMI command returned 'Invalid command' (0xc1). For Lenovo and most server-class hardware, "
                                       "IPMI over LAN requires authentication. Configure IPMI credentials to enable remote fan control.",
                            "ipmi_error": error_msg,
                            "recommendation": "Configure IPMI credentials (GATOR_IPMI_HOST, GATOR_IPMI_USERNAME, GATOR_IPMI_PASSWORD) and ensure IPMI over LAN is enabled in XCC/BMC",
                            "credentials_configured": False,
                        }
                    else:
                        # Credentials are configured but still got 0xc1 - truly unsupported
                        self._manual_control_supported = False
                        commands = self._get_ipmi_commands()
                        logger.warning(
                            f"IPMI raw fan control commands not supported on {self._manufacturer.value} hardware (authenticated). "
                            f"Error: {error_msg}"
                        )
                        return {
                            "success": False,
                            "error": f"Hardware does not support manual fan control via IPMI raw commands (Manufacturer: {self._manufacturer.value})",
                            "details": f"This {self._manufacturer.value} server's BMC does not support the IPMI OEM commands needed for manual fan control. "
                                       "The system will use automatic fan control managed by the BMC firmware.",
                            "ipmi_error": error_msg,
                            "manufacturer_note": commands.get('note'),
                            "recommendation": "Use automatic fan control mode, change manufacturer setting if incorrect, or consult your server's documentation for supported fan control methods",
                        }
                else:
                    logger.error(f"Failed to set fan speed: {error_msg}")
                    return {
                        "success": False,
                        "error": f"IPMI command failed: {error_msg}",
                    }
        except asyncio.TimeoutError:
            logger.error("Timeout setting fan speed via IPMI")
            return {
                "success": False,
                "error": "Timeout communicating with IPMI",
            }
        except Exception as e:
            logger.error(f"Error setting fan speed: {e}")
            return {
                "success": False,
                "error": f"Operation failed: {str(e)}",
            }

    async def _set_ipmi_fan_mode_manual(self) -> Dict[str, Any]:
        """
        Set IPMI fan control to manual mode.

        Returns:
            Status of the operation
        """
        try:
            # Get manufacturer-specific IPMI commands
            commands = self._get_ipmi_commands()
            manual_cmd = commands["manual_mode"]
            
            logger.debug(f"Setting manual mode for {self._manufacturer.value}: {' '.join(manual_cmd)}")
            
            cmd = self._build_ipmi_command(["raw"] + manual_cmd)
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=5.0)

            if proc.returncode == 0:
                logger.info("Fan control set to manual mode")
                return {"success": True}
            else:
                error_msg = stderr.decode("utf-8").strip()
                # Check for "Invalid command" error which indicates unsupported hardware
                if "Invalid command" in error_msg or "0xc1" in error_msg.lower():
                    logger.info("IPMI raw fan control commands not supported on this hardware")
                    return {
                        "success": False,
                        "warning": "Manual mode not supported on this hardware",
                        "error": error_msg,
                    }
                else:
                    logger.warning(f"IPMI manual mode command returned error: {error_msg}")
                    return {"success": True, "warning": error_msg}
        except asyncio.TimeoutError:
            logger.error("Timeout setting manual fan mode via IPMI")
            return {"success": False, "error": "Timeout communicating with IPMI"}
        except Exception as e:
            logger.error(f"Error setting manual fan mode: {e}")
            return {"success": False, "error": f"Operation failed: {str(e)}"}

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
                
                # If manual control is not supported, log it but don't fail
                if not result.get("success") and self._manual_control_supported is False:
                    logger.info(
                        f"Manual fan control not supported. BMC will manage fans automatically. "
                        f"Temperature: {max_gpu_temp}°C, Recommended speed: {target_speed}%"
                    )
                    return {
                        "success": True,
                        "temperature": max_gpu_temp,
                        "target_speed_percent": target_speed,
                        "threshold_status": self._get_threshold_status(max_gpu_temp),
                        "action_taken": None,
                        "message": "Manual fan control not supported - BMC managing fans automatically",
                        "note": "The server's BMC (Baseboard Management Controller) will automatically manage fan speeds based on its own thermal policies",
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    }
            
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
                "error": f"Operation failed: {str(e)}",
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

    def set_manufacturer(self, manufacturer: ServerManufacturer) -> Dict[str, Any]:
        """
        Update the server manufacturer for IPMI command selection.

        Args:
            manufacturer: Server manufacturer

        Returns:
            Updated configuration
        """
        old_manufacturer = self._manufacturer
        self._manufacturer = manufacturer
        # Reset support status since it may differ with new manufacturer
        self._manual_control_supported = None
        
        logger.info(f"Manufacturer changed from {old_manufacturer.value} to {manufacturer.value}")
        
        commands = self._get_ipmi_commands()
        return {
            "success": True,
            "manufacturer": manufacturer.value,
            "previous_manufacturer": old_manufacturer.value,
            "manufacturer_note": commands.get('note'),
            "message": f"Manufacturer set to {manufacturer.value}. Manual control support will be re-checked on next use.",
        }

    def set_ipmi_credentials(
        self,
        host: str,
        username: str,
        password: str,
        interface: str = "lanplus"
    ) -> Dict[str, Any]:
        """
        Update IPMI credentials for remote BMC/XCC access.

        Args:
            host: BMC/XCC IP address or hostname
            username: BMC/XCC username
            password: BMC/XCC password
            interface: IPMI interface type (default: lanplus)

        Returns:
            Updated configuration status
        """
        self._ipmi_host = host
        self._ipmi_username = username
        self._ipmi_password = password
        self._ipmi_interface = interface
        
        # Reset support status since credentials may now allow access
        self._manual_control_supported = None
        
        logger.info(f"IPMI credentials updated for host: {host} (interface: {interface})")
        
        return {
            "success": True,
            "host": host,
            "username": username,
            "interface": interface,
            "message": "IPMI credentials configured. Remote fan control via IPMI over LAN is now enabled.",
            "note": "Ensure IPMI over LAN is enabled in your BMC/XCC settings. "
                    "For Lenovo XCC, enable this in the XCC web interface under Network > IPMI.",
        }

    def get_control_info(self) -> Dict[str, Any]:
        """
        Get fan control configuration and status.

        Returns:
            Fan control information
        """
        commands = self._get_ipmi_commands()
        credentials_configured = bool(self._ipmi_host and self._ipmi_username and self._ipmi_password)
        
        return {
            "ipmi_available": self._ipmi_available,
            "manufacturer": self._manufacturer.value,
            "control_mode": self._control_mode,
            "manual_speed_percent": self._manual_speed,
            "manual_control_supported": self._manual_control_supported,
            "temperature_thresholds": self._temperature_thresholds,
            "supported_zones": [zone.value for zone in FanZone],
            "supported_manufacturers": [m.value for m in ServerManufacturer],
            "manufacturer_note": commands.get('note'),
            "credentials_configured": credentials_configured,
            "ipmi_host_configured": bool(self._ipmi_host),
            "authentication_mode": "remote (out-of-band)" if credentials_configured else "local (in-band)",
            "note": "Server-class IPMI typically requires authentication for remote access. "
                    "Configure IPMI credentials (GATOR_IPMI_HOST, GATOR_IPMI_USERNAME, GATOR_IPMI_PASSWORD) "
                    "to enable remote fan control. For Lenovo servers, ensure IPMI over LAN is enabled in XCC.",
        }


# Global instance
_fan_control_service = None


def get_fan_control_service(manufacturer: Optional[ServerManufacturer] = None) -> FanControlService:
    """
    Get or create the global fan control service instance.
    
    Args:
        manufacturer: Optional manufacturer override. If not provided, defaults to Lenovo.
                     Only used when creating a new instance.
    
    Returns:
        FanControlService instance
    """
    global _fan_control_service
    if _fan_control_service is None:
        if manufacturer is None:
            manufacturer = ServerManufacturer.LENOVO
        _fan_control_service = FanControlService(manufacturer)
    return _fan_control_service
