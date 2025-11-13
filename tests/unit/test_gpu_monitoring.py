"""
Tests for GPU Monitoring Service
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from collections import deque

from backend.services.gpu_monitoring_service import GPUMonitoringService


@pytest.fixture
def gpu_service():
    """Create a GPU monitoring service for testing."""
    return GPUMonitoringService(history_size=10)


class TestGPUMonitoringService:
    """Test suite for GPU monitoring service."""

    def test_initialization_no_torch(self, gpu_service):
        """Test service initializes correctly when PyTorch is not available."""
        # Should not crash even without torch
        assert gpu_service.history_size == 10
        assert isinstance(gpu_service.temperature_history, dict)
        assert isinstance(gpu_service.utilization_history, dict)

    @pytest.mark.asyncio
    async def test_get_gpu_temperatures_no_torch(self):
        """Test getting temperatures when PyTorch is not available."""
        service = GPUMonitoringService()
        service._torch_available = False
        
        result = await service.get_gpu_temperatures()
        
        assert result["available"] is False
        assert "error" in result

    @pytest.mark.asyncio
    async def test_get_gpu_temperatures_with_mock_torch(self):
        """Test getting GPU temperatures with mocked PyTorch."""
        service = GPUMonitoringService()
        service._torch_available = True
        service._gpu_count = 2
        
        # Initialize history for 2 GPUs
        service.temperature_history[0] = deque(maxlen=10)
        service.temperature_history[1] = deque(maxlen=10)
        
        # Mock torch functions
        with patch("torch.cuda") as mock_cuda:
            mock_props_0 = Mock()
            mock_props_0.total_memory = 16 * 1024**3  # 16 GB
            mock_props_1 = Mock()
            mock_props_1.total_memory = 16 * 1024**3
            
            mock_cuda.get_device_properties.side_effect = [mock_props_0, mock_props_1]
            mock_cuda.get_device_name.side_effect = ["GPU 0", "GPU 1"]
            mock_cuda.memory_allocated.return_value = 8 * 1024**3  # 8 GB
            mock_cuda.memory_reserved.return_value = 10 * 1024**3  # 10 GB
            
            # Mock temperature reading
            service._get_gpu_temperature_rocm = AsyncMock(return_value=65.0)
            
            result = await service.get_gpu_temperatures()
            
            assert result["available"] is True
            assert result["gpu_count"] == 2
            assert len(result["gpus"]) == 2
            assert result["gpus"][0]["device_id"] == 0
            assert result["gpus"][0]["name"] == "GPU 0"
            assert result["gpus"][0]["temperature_c"] == 65.0

    @pytest.mark.asyncio
    async def test_get_gpu_status_no_torch(self):
        """Test getting GPU status when PyTorch is not available."""
        service = GPUMonitoringService()
        service._torch_available = False
        
        result = await service.get_gpu_status()
        
        assert result["available"] is False

    @pytest.mark.asyncio
    async def test_get_gpu_status_with_mock_torch(self):
        """Test getting comprehensive GPU status with mocked PyTorch."""
        service = GPUMonitoringService()
        service._torch_available = True
        service._gpu_count = 1
        service.temperature_history[0] = deque(maxlen=10)
        
        with patch("torch.cuda") as mock_cuda:
            mock_props = Mock()
            mock_props.total_memory = 16 * 1024**3
            mock_props.major = 9
            mock_props.minor = 0
            mock_props.multi_processor_count = 64
            
            mock_cuda.get_device_properties.return_value = mock_props
            mock_cuda.get_device_name.return_value = "AMD Radeon Pro V620"
            mock_cuda.memory_allocated.return_value = 4 * 1024**3
            mock_cuda.memory_reserved.return_value = 5 * 1024**3
            
            # Mock get_gpu_temperatures to return temp data
            service.get_gpu_temperatures = AsyncMock(return_value={
                "available": True,
                "gpu_count": 1,
                "gpus": [{"device_id": 0, "temperature_c": 55.0}]
            })
            
            result = await service.get_gpu_status()
            
            assert result["available"] is True
            assert result["gpu_count"] == 1
            assert len(result["gpus"]) == 1
            assert result["gpus"][0]["health_status"] == "healthy"
            assert result["gpus"][0]["memory"]["total_gb"] == 16.0

    def test_determine_health_status(self, gpu_service):
        """Test health status determination based on temperature."""
        assert gpu_service._determine_health_status(None) == "unknown"
        assert gpu_service._determine_health_status(50.0) == "healthy"
        assert gpu_service._determine_health_status(65.0) == "warm"
        assert gpu_service._determine_health_status(80.0) == "hot"
        assert gpu_service._determine_health_status(90.0) == "critical"

    @pytest.mark.asyncio
    async def test_get_temperature_history_empty(self, gpu_service):
        """Test getting temperature history when no data exists."""
        gpu_service._torch_available = True
        gpu_service._gpu_count = 1
        gpu_service.temperature_history[0] = deque(maxlen=10)
        
        result = await gpu_service.get_temperature_history(device_id=0)
        
        # Should return empty list if no history
        assert result["available"] is True
        assert result["device_id"] == 0
        assert result["history"] == []

    @pytest.mark.asyncio
    async def test_get_temperature_history_with_data(self, gpu_service):
        """Test getting temperature history with data."""
        gpu_service._torch_available = True
        gpu_service._gpu_count = 1
        gpu_service.temperature_history[0] = deque(maxlen=10)
        
        # Add some temperature readings
        gpu_service.temperature_history[0].append(
            {"timestamp": "2024-01-01T00:00:00Z", "temperature": 60.0}
        )
        gpu_service.temperature_history[0].append(
            {"timestamp": "2024-01-01T00:01:00Z", "temperature": 65.0}
        )
        
        result = await gpu_service.get_temperature_history(device_id=0)
        
        assert result["available"] is True
        assert result["device_id"] == 0
        assert len(result["history"]) == 2
        assert result["history"][0]["temperature"] == 60.0

    @pytest.mark.asyncio
    async def test_get_temperature_history_all_gpus(self, gpu_service):
        """Test getting temperature history for all GPUs."""
        gpu_service._torch_available = True
        gpu_service._gpu_count = 2
        gpu_service.temperature_history[0] = deque(maxlen=10)
        gpu_service.temperature_history[1] = deque(maxlen=10)
        
        gpu_service.temperature_history[0].append(
            {"timestamp": "2024-01-01T00:00:00Z", "temperature": 60.0}
        )
        gpu_service.temperature_history[1].append(
            {"timestamp": "2024-01-01T00:00:00Z", "temperature": 55.0}
        )
        
        result = await gpu_service.get_temperature_history()
        
        assert result["available"] is True
        assert 0 in result["history"]
        assert 1 in result["history"]
        assert len(result["history"][0]) == 1
        assert len(result["history"][1]) == 1

    @pytest.mark.asyncio
    async def test_get_max_temperatures(self, gpu_service):
        """Test getting maximum recorded temperatures."""
        gpu_service._torch_available = True
        gpu_service._gpu_count = 1
        gpu_service.temperature_history[0] = deque(maxlen=10)
        
        # Add temperature readings
        gpu_service.temperature_history[0].append(
            {"timestamp": "2024-01-01T00:00:00Z", "temperature": 60.0}
        )
        gpu_service.temperature_history[0].append(
            {"timestamp": "2024-01-01T00:01:00Z", "temperature": 75.0}
        )
        gpu_service.temperature_history[0].append(
            {"timestamp": "2024-01-01T00:02:00Z", "temperature": 65.0}
        )
        
        result = await gpu_service.get_max_temperatures()
        
        assert result["available"] is True
        assert 0 in result["max_temperatures"]
        assert result["max_temperatures"][0]["max_temperature_c"] == 75.0
        assert result["max_temperatures"][0]["avg_temperature_c"] == pytest.approx(66.67, rel=0.1)
        assert result["max_temperatures"][0]["readings_count"] == 3

    @pytest.mark.asyncio
    async def test_get_gpu_temperature_rocm_timeout(self, gpu_service):
        """Test handling of timeout when reading from rocm-smi."""
        with patch("asyncio.create_subprocess_exec") as mock_exec:
            mock_proc = AsyncMock()
            mock_proc.communicate = AsyncMock(side_effect=asyncio.TimeoutError())
            mock_exec.return_value = mock_proc
            
            result = await gpu_service._get_gpu_temperature_rocm(0)
            
            # Should return None on timeout
            assert result is None

    @pytest.mark.asyncio
    async def test_get_gpu_temperature_rocm_not_found(self, gpu_service):
        """Test handling when rocm-smi is not found."""
        with patch("asyncio.create_subprocess_exec", side_effect=FileNotFoundError):
            result = await gpu_service._get_gpu_temperature_rocm(0)
            
            # Should return None when command not found
            assert result is None


import asyncio  # Add this import at the top if not present
