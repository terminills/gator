"""
System Monitoring API Routes

Handles GPU temperature monitoring and server fan control for Lenovo SR665.
"""

from typing import Optional
from fastapi import APIRouter, HTTPException, status, Query, Body
from pydantic import BaseModel, Field

from backend.services.gpu_monitoring_service import get_gpu_monitoring_service
from backend.services.fan_control_service import (
    get_fan_control_service,
    FanControlMode,
    FanZone,
    ServerManufacturer,
)
from backend.config.logging import get_logger

logger = get_logger(__name__)

router = APIRouter(
    prefix="/api/v1/system",
    tags=["system-monitoring"],
)


# Request/Response models
class FanSpeedRequest(BaseModel):
    """Request model for setting fan speed."""
    speed_percent: int = Field(..., ge=0, le=100, description="Fan speed as percentage (0-100)")
    zone: Optional[str] = Field(None, description="Fan zone (system, cpu, peripheral)")


class FanModeRequest(BaseModel):
    """Request model for setting fan control mode."""
    mode: str = Field(..., description="Fan control mode (auto, manual)")


class TemperatureThresholdsRequest(BaseModel):
    """Request model for updating temperature thresholds."""
    low: Optional[float] = Field(None, ge=0, le=100, description="Low temperature threshold (C)")
    normal: Optional[float] = Field(None, ge=0, le=100, description="Normal temperature threshold (C)")
    high: Optional[float] = Field(None, ge=0, le=100, description="High temperature threshold (C)")
    critical: Optional[float] = Field(None, ge=0, le=100, description="Critical temperature threshold (C)")


class AutoAdjustRequest(BaseModel):
    """Request model for automatic fan adjustment."""
    target_temperature: Optional[float] = Field(None, description="Target max GPU temperature (C)")


class ManufacturerRequest(BaseModel):
    """Request model for setting server manufacturer."""
    manufacturer: str = Field(..., description="Server manufacturer (lenovo, dell, hp, supermicro, generic)")


class IPMICredentialsRequest(BaseModel):
    """Request model for configuring IPMI credentials."""
    host: str = Field(..., description="BMC/XCC IP address or hostname")
    username: str = Field(..., description="BMC/XCC username")
    password: str = Field(..., description="BMC/XCC password")
    interface: Optional[str] = Field("lanplus", description="IPMI interface (default: lanplus)")


# GPU Monitoring Endpoints

@router.get("/gpu/temperature", status_code=status.HTTP_200_OK)
async def get_gpu_temperatures():
    """
    Get current GPU temperatures for all devices.

    Returns real-time temperature readings from all GPUs in the system.
    """
    try:
        gpu_service = get_gpu_monitoring_service()
        result = await gpu_service.get_gpu_temperatures()
        return result
    except Exception as e:
        logger.error(f"Error getting GPU temperatures: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error retrieving GPU temperatures",
        )


@router.get("/gpu/status", status_code=status.HTTP_200_OK)
async def get_gpu_status():
    """
    Get comprehensive GPU status.

    Returns detailed information about all GPUs including temperature,
    memory usage, utilization, and health status.
    """
    try:
        gpu_service = get_gpu_monitoring_service()
        result = await gpu_service.get_gpu_status()
        return result
    except Exception as e:
        logger.error(f"Error getting GPU status: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error retrieving GPU status",
        )


@router.get("/gpu/temperature/history", status_code=status.HTTP_200_OK)
async def get_temperature_history(
    device_id: Optional[int] = Query(None, description="Specific GPU device ID")
):
    """
    Get historical temperature data.

    Args:
        device_id: Optional GPU device ID. If not provided, returns history for all GPUs.

    Returns:
        Historical temperature readings.
    """
    try:
        gpu_service = get_gpu_monitoring_service()
        result = await gpu_service.get_temperature_history(device_id)
        return result
    except Exception as e:
        logger.error(f"Error getting temperature history: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error retrieving temperature history",
        )


@router.get("/gpu/temperature/max", status_code=status.HTTP_200_OK)
async def get_max_temperatures():
    """
    Get maximum recorded temperatures.

    Returns the maximum, average, and reading count for each GPU
    from the historical data.
    """
    try:
        gpu_service = get_gpu_monitoring_service()
        result = await gpu_service.get_max_temperatures()
        return result
    except Exception as e:
        logger.error(f"Error getting max temperatures: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error retrieving max temperatures",
        )


# Fan Control Endpoints

@router.get("/fans", status_code=status.HTTP_200_OK)
async def get_fan_status():
    """
    Get current fan status from IPMI.

    Returns fan speeds, control mode, and temperature thresholds for the
    Lenovo SR665 server fan wall.
    """
    try:
        fan_service = get_fan_control_service()
        result = await fan_service.get_fan_status()
        return result
    except Exception as e:
        logger.error(f"Error getting fan status: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error retrieving fan status",
        )


@router.get("/fans/info", status_code=status.HTTP_200_OK)
async def get_fan_control_info():
    """
    Get fan control configuration and capabilities.

    Returns information about fan control availability, current mode,
    and supported features.
    """
    try:
        fan_service = get_fan_control_service()
        result = fan_service.get_control_info()
        return result
    except Exception as e:
        logger.error(f"Error getting fan control info: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error retrieving fan control info",
        )


@router.post("/fans/mode", status_code=status.HTTP_200_OK)
async def set_fan_mode(request: FanModeRequest):
    """
    Set fan control mode.

    Args:
        request: Fan mode request with mode value (auto, manual)

    Returns:
        Status of the mode change operation.
    """
    try:
        fan_service = get_fan_control_service()
        
        # Validate mode
        try:
            mode = FanControlMode(request.mode.lower())
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid mode: {request.mode}. Must be 'auto' or 'manual'",
            )
        
        result = await fan_service.set_fan_mode(mode)
        
        if not result.get("success"):
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=result.get("error", "Failed to set fan mode"),
            )
        
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error setting fan mode: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error setting fan mode",
        )


@router.post("/fans/speed", status_code=status.HTTP_200_OK)
async def set_fan_speed(request: FanSpeedRequest):
    """
    Set fan speed manually.

    Args:
        request: Fan speed request with speed percentage and optional zone

    Returns:
        Status of the fan speed change operation.
    """
    try:
        fan_service = get_fan_control_service()
        
        # Validate zone if provided
        zone = None
        if request.zone:
            try:
                zone = FanZone(request.zone.lower())
            except ValueError:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Invalid zone: {request.zone}. Must be 'system', 'cpu', or 'peripheral'",
                )
        
        result = await fan_service.set_fan_speed(request.speed_percent, zone)
        
        if not result.get("success"):
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=result.get("error", "Failed to set fan speed"),
            )
        
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error setting fan speed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error setting fan speed",
        )


@router.post("/fans/auto-adjust", status_code=status.HTTP_200_OK)
async def auto_adjust_fans(request: AutoAdjustRequest = Body(default=None)):
    """
    Automatically adjust fan speeds based on GPU temperature.

    Args:
        request: Optional request with target temperature override

    Returns:
        Status of the automatic adjustment including actions taken.
    """
    try:
        gpu_service = get_gpu_monitoring_service()
        fan_service = get_fan_control_service()
        
        # Get current GPU temperatures
        gpu_temps = await gpu_service.get_gpu_temperatures()
        
        if not gpu_temps.get("available"):
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="GPU temperature monitoring not available",
            )
        
        # Find maximum GPU temperature
        max_temp = 0
        for gpu in gpu_temps.get("gpus", []):
            temp = gpu.get("temperature_c")
            if temp is not None and temp > max_temp:
                max_temp = temp
        
        if max_temp == 0:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="No valid GPU temperature readings available",
            )
        
        # Use target temperature if provided, otherwise use max detected
        target_temp = request.target_temperature if request and request.target_temperature else max_temp
        
        # Adjust fans based on temperature
        result = await fan_service.adjust_fans_for_temperature(target_temp)
        
        if not result.get("success"):
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=result.get("error", "Failed to adjust fans"),
            )
        
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error auto-adjusting fans: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error auto-adjusting fans",
        )


@router.post("/fans/thresholds", status_code=status.HTTP_200_OK)
async def set_temperature_thresholds(request: TemperatureThresholdsRequest):
    """
    Update temperature thresholds for automatic fan control.

    Args:
        request: New threshold values (only provided values will be updated)

    Returns:
        Updated threshold configuration.
    """
    try:
        fan_service = get_fan_control_service()
        
        # Build threshold dict from non-None values
        thresholds = {}
        if request.low is not None:
            thresholds["low"] = request.low
        if request.normal is not None:
            thresholds["normal"] = request.normal
        if request.high is not None:
            thresholds["high"] = request.high
        if request.critical is not None:
            thresholds["critical"] = request.critical
        
        if not thresholds:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="At least one threshold value must be provided",
            )
        
        result = fan_service.set_temperature_thresholds(thresholds)
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error setting temperature thresholds: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error setting temperature thresholds",
        )


@router.post("/fans/manufacturer", status_code=status.HTTP_200_OK)
async def set_manufacturer(request: ManufacturerRequest):
    """
    Set the server manufacturer for IPMI command selection.

    Different server manufacturers use different IPMI OEM commands for fan control.
    Use this endpoint to configure the correct manufacturer to ensure proper IPMI commands are used.

    Args:
        request: Manufacturer selection (lenovo, dell, hp, supermicro, generic)

    Returns:
        Updated manufacturer configuration.
    """
    try:
        fan_service = get_fan_control_service()
        
        # Validate manufacturer
        try:
            manufacturer = ServerManufacturer(request.manufacturer.lower())
        except ValueError:
            valid_manufacturers = [m.value for m in ServerManufacturer]
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid manufacturer: {request.manufacturer}. Must be one of: {', '.join(valid_manufacturers)}",
            )
        
        result = fan_service.set_manufacturer(manufacturer)
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error setting manufacturer: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error setting manufacturer",
        )


@router.post("/fans/credentials", status_code=status.HTTP_200_OK)
async def set_ipmi_credentials(request: IPMICredentialsRequest):
    """
    Configure IPMI credentials for remote BMC/XCC access.

    Most server-class hardware requires authentication for IPMI over LAN access.
    Use this endpoint to configure the BMC/XCC credentials needed for remote fan control.

    For Lenovo servers:
    1. Ensure IPMI over LAN is enabled in XCC (Network > IPMI settings)
    2. Use XCC credentials (default may be USERID/PASSW0RD)
    3. Use the XCC management IP address

    Args:
        request: IPMI credentials (host, username, password, interface)

    Returns:
        Updated credential configuration status.
    """
    try:
        fan_service = get_fan_control_service()
        
        result = fan_service.set_ipmi_credentials(
            host=request.host,
            username=request.username,
            password=request.password,
            interface=request.interface or "lanplus"
        )
        return result
    except Exception as e:
        logger.error(f"Error setting IPMI credentials: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error setting IPMI credentials",
        )
