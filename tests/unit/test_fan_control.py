"""
Tests for Fan Control Service
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch

from backend.services.fan_control_service import (
    FanControlService,
    FanControlMode,
    FanZone,
)


@pytest.fixture
def fan_service():
    """Create a fan control service for testing."""
    return FanControlService()


class TestFanControlService:
    """Test suite for fan control service."""

    def test_initialization(self, fan_service):
        """Test service initializes correctly."""
        assert fan_service._control_mode == FanControlMode.AUTO
        assert fan_service._manual_speed is None
        assert "low" in fan_service._temperature_thresholds
        assert "critical" in fan_service._temperature_thresholds

    @pytest.mark.asyncio
    async def test_get_fan_status_no_ipmi(self):
        """Test getting fan status when IPMI is not available."""
        service = FanControlService()
        service._ipmi_available = False
        
        result = await service.get_fan_status()
        
        assert result["available"] is False
        assert "error" in result
        assert result["control_mode"] == FanControlMode.AUTO

    @pytest.mark.asyncio
    async def test_get_fan_status_with_ipmi(self):
        """Test getting fan status with mocked IPMI."""
        service = FanControlService()
        service._ipmi_available = True
        
        # Mock IPMI command
        with patch("asyncio.create_subprocess_exec") as mock_exec:
            mock_proc = AsyncMock()
            mock_proc.returncode = 0
            mock_proc.communicate = AsyncMock(return_value=(
                b"Fan 1           | 3600.000  | RPM\nFan 2           | 3500.000  | RPM\n",
                b""
            ))
            mock_exec.return_value = mock_proc
            
            result = await service.get_fan_status()
            
            assert result["available"] is True
            assert result["control_mode"] == FanControlMode.AUTO
            assert len(result["fans"]) >= 2
            assert "timestamp" in result

    def test_parse_fan_sensors(self, fan_service):
        """Test parsing fan sensor output."""
        sensor_output = """
Fan 1           | 3600.000  | RPM
Fan 2           | 3500.000  | RPM
Fan 3           | na        | RPM
Temperature     | 45.0      | degrees C
"""
        
        fans = fan_service._parse_fan_sensors(sensor_output)
        
        assert len(fans) == 3
        assert fans[0]["name"] == "Fan 1"
        assert fans[0]["speed"] == 3600.0
        assert fans[0]["unit"] == "RPM"
        assert fans[0]["status"] == "ok"
        assert fans[2]["speed"] is None
        assert fans[2]["status"] == "unknown"

    @pytest.mark.asyncio
    async def test_set_fan_mode_auto(self):
        """Test setting fan mode to automatic."""
        service = FanControlService()
        service._ipmi_available = True
        
        with patch("asyncio.create_subprocess_exec") as mock_exec:
            mock_proc = AsyncMock()
            mock_proc.returncode = 0
            mock_proc.communicate = AsyncMock(return_value=(b"", b""))
            mock_exec.return_value = mock_proc
            
            result = await service.set_fan_mode(FanControlMode.AUTO)
            
            assert result["success"] is True
            assert service._control_mode == FanControlMode.AUTO
            assert service._manual_speed is None

    @pytest.mark.asyncio
    async def test_set_fan_mode_manual(self):
        """Test setting fan mode to manual."""
        service = FanControlService()
        service._ipmi_available = True
        
        result = await service.set_fan_mode(FanControlMode.MANUAL)
        
        assert result["success"] is True
        assert service._control_mode == FanControlMode.MANUAL

    @pytest.mark.asyncio
    async def test_set_fan_speed_invalid_range(self):
        """Test setting fan speed with invalid value."""
        service = FanControlService()
        service._ipmi_available = True
        
        result = await service.set_fan_speed(150)
        assert result["success"] is False
        
        result = await service.set_fan_speed(-10)
        assert result["success"] is False

    @pytest.mark.asyncio
    async def test_set_fan_speed_valid(self):
        """Test setting fan speed with valid value."""
        service = FanControlService()
        service._ipmi_available = True
        service._control_mode = FanControlMode.MANUAL
        
        with patch("asyncio.create_subprocess_exec") as mock_exec:
            mock_proc = AsyncMock()
            mock_proc.returncode = 0
            mock_proc.communicate = AsyncMock(return_value=(b"", b""))
            mock_exec.return_value = mock_proc
            
            result = await service.set_fan_speed(75)
            
            assert result["success"] is True
            assert result["speed_percent"] == 75
            assert service._manual_speed == 75

    def test_calculate_fan_speed(self, fan_service):
        """Test fan speed calculation based on temperature."""
        # Low temperature
        speed = fan_service._calculate_fan_speed(45.0)
        assert speed == 30
        
        # Normal temperature
        speed = fan_service._calculate_fan_speed(60.0)
        assert 30 <= speed <= 50
        
        # Elevated temperature
        speed = fan_service._calculate_fan_speed(70.0)
        assert 50 <= speed <= 75
        
        # High temperature
        speed = fan_service._calculate_fan_speed(80.0)
        assert 75 <= speed <= 100
        
        # Critical temperature
        speed = fan_service._calculate_fan_speed(90.0)
        assert speed == 100

    def test_get_threshold_status(self, fan_service):
        """Test getting temperature threshold status."""
        assert fan_service._get_threshold_status(40.0) == "low"
        assert fan_service._get_threshold_status(60.0) == "normal"
        assert fan_service._get_threshold_status(70.0) == "elevated"
        assert fan_service._get_threshold_status(80.0) == "high"
        assert fan_service._get_threshold_status(90.0) == "critical"

    @pytest.mark.asyncio
    async def test_adjust_fans_for_temperature(self):
        """Test automatic fan adjustment based on temperature."""
        service = FanControlService()
        service._ipmi_available = True
        
        with patch("asyncio.create_subprocess_exec") as mock_exec:
            mock_proc = AsyncMock()
            mock_proc.returncode = 0
            mock_proc.communicate = AsyncMock(return_value=(b"", b""))
            mock_exec.return_value = mock_proc
            
            # Test with high temperature
            result = await service.adjust_fans_for_temperature(80.0)
            
            assert result["success"] is True
            assert result["temperature"] == 80.0
            assert result["threshold_status"] == "high"
            assert result["target_speed_percent"] > 50

    def test_set_temperature_thresholds(self, fan_service):
        """Test updating temperature thresholds."""
        new_thresholds = {
            "low": 55,
            "normal": 70,
            "high": 80,
            "critical": 90,
        }
        
        result = fan_service.set_temperature_thresholds(new_thresholds)
        
        assert result["success"] is True
        assert fan_service._temperature_thresholds["low"] == 55
        assert fan_service._temperature_thresholds["critical"] == 90

    def test_get_control_info(self, fan_service):
        """Test getting fan control information."""
        result = fan_service.get_control_info()
        
        assert "ipmi_available" in result
        assert "control_mode" in result
        assert "temperature_thresholds" in result
        assert "supported_zones" in result
        assert "system" in result["supported_zones"]

    def test_get_zone_id(self, fan_service):
        """Test getting zone ID for IPMI commands."""
        assert fan_service._get_zone_id(None) == 0xff
        assert fan_service._get_zone_id(FanZone.SYSTEM) == 0x00
        assert fan_service._get_zone_id(FanZone.CPU) == 0x01
        assert fan_service._get_zone_id(FanZone.PERIPHERAL) == 0x02

    @pytest.mark.asyncio
    async def test_set_fan_mode_no_ipmi(self):
        """Test setting fan mode when IPMI is not available."""
        service = FanControlService()
        service._ipmi_available = False
        
        result = await service.set_fan_mode(FanControlMode.AUTO)
        
        assert result["success"] is False
        assert "error" in result

    @pytest.mark.asyncio
    async def test_get_fan_status_timeout(self):
        """Test handling of timeout when reading fan status."""
        service = FanControlService()
        service._ipmi_available = True
        
        with patch("asyncio.create_subprocess_exec") as mock_exec:
            mock_proc = AsyncMock()
            mock_proc.communicate = AsyncMock(side_effect=asyncio.TimeoutError())
            mock_exec.return_value = mock_proc
            
            result = await service.get_fan_status()
            
            assert result["available"] is False
            assert "Timeout" in result["error"]


import asyncio  # Add this import at the top if not present
