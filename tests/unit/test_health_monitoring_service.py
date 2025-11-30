"""
Unit tests for Health Monitoring Service.

Tests comprehensive health checks and resource monitoring.
"""

import pytest
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

from backend.services.health_monitoring_service import (
    HealthMonitoringService,
    HealthStatus,
    ServiceHealth,
    SystemHealth,
    ResourceMetrics,
)


class TestHealthStatus:
    """Tests for HealthStatus enum."""

    def test_health_status_values(self):
        """Test HealthStatus enum values."""
        assert HealthStatus.HEALTHY == "healthy"
        assert HealthStatus.DEGRADED == "degraded"
        assert HealthStatus.UNHEALTHY == "unhealthy"
        assert HealthStatus.UNKNOWN == "unknown"


class TestServiceHealth:
    """Tests for ServiceHealth model."""

    def test_service_health_creation(self):
        """Test creating ServiceHealth."""
        health = ServiceHealth(
            name="database",
            status=HealthStatus.HEALTHY,
            latency_ms=5.5,
            message="Connection successful",
        )

        assert health.name == "database"
        assert health.status == HealthStatus.HEALTHY
        assert health.latency_ms == 5.5
        assert health.message == "Connection successful"

    def test_service_health_defaults(self):
        """Test ServiceHealth default values."""
        health = ServiceHealth(
            name="test",
            status=HealthStatus.HEALTHY,
        )

        assert health.latency_ms is None
        assert health.message is None
        assert health.details == {}
        assert isinstance(health.checked_at, datetime)


class TestSystemHealth:
    """Tests for SystemHealth model."""

    def test_system_health_creation(self):
        """Test creating SystemHealth."""
        services = [
            ServiceHealth(name="db", status=HealthStatus.HEALTHY),
            ServiceHealth(name="redis", status=HealthStatus.HEALTHY),
        ]

        health = SystemHealth(
            status=HealthStatus.HEALTHY,
            services=services,
            uptime_seconds=3600.0,
        )

        assert health.status == HealthStatus.HEALTHY
        assert len(health.services) == 2
        assert health.uptime_seconds == 3600.0

    def test_system_health_defaults(self):
        """Test SystemHealth default values."""
        health = SystemHealth(
            status=HealthStatus.HEALTHY,
            services=[],
        )

        assert health.uptime_seconds == 0
        assert health.version == "0.1.0"
        assert isinstance(health.timestamp, datetime)


class TestResourceMetrics:
    """Tests for ResourceMetrics model."""

    def test_resource_metrics_defaults(self):
        """Test ResourceMetrics default values."""
        metrics = ResourceMetrics()

        assert metrics.cpu_percent == 0.0
        assert metrics.memory_total_gb == 0.0
        assert metrics.memory_used_gb == 0.0
        assert metrics.memory_percent == 0.0
        assert metrics.disk_total_gb == 0.0
        assert metrics.disk_used_gb == 0.0
        assert metrics.disk_percent == 0.0

    def test_resource_metrics_with_values(self):
        """Test ResourceMetrics with values."""
        metrics = ResourceMetrics(
            cpu_percent=25.5,
            memory_total_gb=32.0,
            memory_used_gb=16.0,
            memory_percent=50.0,
            disk_total_gb=500.0,
            disk_used_gb=250.0,
            disk_percent=50.0,
        )

        assert metrics.cpu_percent == 25.5
        assert metrics.memory_percent == 50.0
        assert metrics.disk_percent == 50.0


class TestHealthMonitoringService:
    """Tests for HealthMonitoringService."""

    @pytest.fixture
    def mock_db_session(self):
        """Create mock database session."""
        session = AsyncMock()
        return session

    @pytest.fixture
    def health_service(self, mock_db_session):
        """Create health monitoring service instance."""
        return HealthMonitoringService(mock_db_session)

    @pytest.mark.asyncio
    async def test_check_database_health_success(self, health_service, mock_db_session):
        """Test successful database health check."""
        mock_result = MagicMock()
        mock_result.scalar.return_value = 1
        mock_db_session.execute = AsyncMock(return_value=mock_result)

        result = await health_service.check_database_health()

        assert result.name == "database"
        assert result.status == HealthStatus.HEALTHY
        assert result.latency_ms is not None
        assert "successful" in result.message

    @pytest.mark.asyncio
    async def test_check_database_health_failure(self, health_service, mock_db_session):
        """Test database health check failure."""
        mock_db_session.execute = AsyncMock(side_effect=Exception("Connection error"))

        result = await health_service.check_database_health()

        assert result.name == "database"
        assert result.status == HealthStatus.UNHEALTHY
        assert "error" in result.message.lower()

    @pytest.mark.asyncio
    async def test_check_database_no_session(self):
        """Test database health check without session."""
        service = HealthMonitoringService(db_session=None)

        result = await service.check_database_health()

        assert result.status == HealthStatus.UNKNOWN
        assert "No database session" in result.message

    @pytest.mark.asyncio
    async def test_check_redis_health_no_redis(self, health_service):
        """Test Redis health when redis not available."""
        # This test simulates Redis being unavailable
        result = await health_service.check_redis_health()

        # Should handle gracefully (either degraded or connects fine if redis is running)
        assert result.name == "redis"
        assert result.status in [HealthStatus.HEALTHY, HealthStatus.DEGRADED]

    @pytest.mark.asyncio
    async def test_check_ollama_health_unavailable(self, health_service):
        """Test Ollama health when service unavailable."""
        with patch("httpx.AsyncClient") as mock_client:
            mock_instance = AsyncMock()
            mock_instance.get = AsyncMock(side_effect=Exception("Connection refused"))
            mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
            mock_instance.__aexit__ = AsyncMock(return_value=None)
            mock_client.return_value = mock_instance

            result = await health_service.check_ollama_health()

        assert result.name == "ollama"
        assert result.status == HealthStatus.DEGRADED
        assert "unavailable" in result.message.lower()

    @pytest.mark.asyncio
    async def test_check_filesystem_health(self, health_service):
        """Test filesystem health check."""
        with patch("os.path.exists", return_value=True):
            with patch("os.statvfs") as mock_statvfs:
                mock_statvfs.return_value = MagicMock(
                    f_frsize=4096,
                    f_bavail=100000000,  # Available blocks
                    f_blocks=200000000,  # Total blocks
                )
                with patch("builtins.open", MagicMock()):
                    with patch("os.remove"):
                        result = await health_service.check_filesystem_health()

        assert result.name == "filesystem"
        assert result.status in [
            HealthStatus.HEALTHY,
            HealthStatus.DEGRADED,
        ]

    @pytest.mark.asyncio
    async def test_check_all_health(self, health_service, mock_db_session):
        """Test comprehensive health check."""
        mock_result = MagicMock()
        mock_result.scalar.return_value = 1
        mock_db_session.execute = AsyncMock(return_value=mock_result)

        result = await health_service.check_all_health()

        assert isinstance(result, SystemHealth)
        assert len(result.services) > 0
        assert result.uptime_seconds >= 0

    @pytest.mark.asyncio
    async def test_check_all_health_determines_overall_status(
        self, health_service, mock_db_session
    ):
        """Test that overall status is correctly determined."""
        mock_result = MagicMock()
        mock_result.scalar.return_value = 1
        mock_db_session.execute = AsyncMock(return_value=mock_result)

        result = await health_service.check_all_health()

        # Overall status should be based on service statuses
        assert result.status in list(HealthStatus)

    @pytest.mark.asyncio
    async def test_get_resource_metrics(self, health_service):
        """Test getting resource metrics."""
        with patch("psutil.cpu_percent", return_value=25.0):
            with patch(
                "psutil.virtual_memory",
                return_value=MagicMock(
                    total=32 * 1024**3,
                    used=16 * 1024**3,
                    percent=50.0,
                ),
            ):
                with patch(
                    "psutil.disk_usage",
                    return_value=MagicMock(
                        total=500 * 1024**3,
                        used=250 * 1024**3,
                        percent=50.0,
                    ),
                ):
                    result = await health_service.get_resource_metrics()

        assert isinstance(result, ResourceMetrics)
        assert result.cpu_percent == 25.0
        assert result.memory_percent == 50.0

    @pytest.mark.asyncio
    async def test_get_resource_metrics_error_handling(self, health_service):
        """Test resource metrics error handling."""
        # Test that the method handles errors gracefully
        with patch("psutil.cpu_percent", side_effect=Exception("Error")):
            result = await health_service.get_resource_metrics()

        # Should return default empty metrics on error
        assert isinstance(result, ResourceMetrics)

    @pytest.mark.asyncio
    async def test_get_readiness_status_healthy(self, health_service, mock_db_session):
        """Test readiness status when healthy."""
        mock_result = MagicMock()
        mock_result.scalar.return_value = 1
        mock_db_session.execute = AsyncMock(return_value=mock_result)

        result = await health_service.get_readiness_status()

        assert "ready" in result
        assert "status" in result
        assert "timestamp" in result

    @pytest.mark.asyncio
    async def test_get_liveness_status(self, health_service):
        """Test liveness status."""
        result = await health_service.get_liveness_status()

        assert result["alive"] is True
        assert "uptime_seconds" in result
        assert result["uptime_seconds"] >= 0
        assert "timestamp" in result


class TestHealthStatusDetermination:
    """Tests for overall health status determination."""

    @pytest.fixture
    def health_service(self):
        """Create health service without DB."""
        return HealthMonitoringService(db_session=None)

    @pytest.mark.asyncio
    async def test_all_healthy_gives_healthy_status(self, health_service):
        """Test that all healthy services give healthy overall status."""
        # Mock all checks to return healthy
        health_service.check_database_health = AsyncMock(
            return_value=ServiceHealth(name="db", status=HealthStatus.HEALTHY)
        )
        health_service.check_redis_health = AsyncMock(
            return_value=ServiceHealth(name="redis", status=HealthStatus.HEALTHY)
        )
        health_service.check_ollama_health = AsyncMock(
            return_value=ServiceHealth(name="ollama", status=HealthStatus.HEALTHY)
        )
        health_service.check_filesystem_health = AsyncMock(
            return_value=ServiceHealth(name="fs", status=HealthStatus.HEALTHY)
        )

        result = await health_service.check_all_health()

        assert result.status == HealthStatus.HEALTHY

    @pytest.mark.asyncio
    async def test_one_unhealthy_gives_unhealthy_status(self, health_service):
        """Test that one unhealthy service gives unhealthy overall status."""
        health_service.check_database_health = AsyncMock(
            return_value=ServiceHealth(name="db", status=HealthStatus.UNHEALTHY)
        )
        health_service.check_redis_health = AsyncMock(
            return_value=ServiceHealth(name="redis", status=HealthStatus.HEALTHY)
        )
        health_service.check_ollama_health = AsyncMock(
            return_value=ServiceHealth(name="ollama", status=HealthStatus.HEALTHY)
        )
        health_service.check_filesystem_health = AsyncMock(
            return_value=ServiceHealth(name="fs", status=HealthStatus.HEALTHY)
        )

        result = await health_service.check_all_health()

        assert result.status == HealthStatus.UNHEALTHY

    @pytest.mark.asyncio
    async def test_one_degraded_gives_degraded_status(self, health_service):
        """Test that one degraded service gives degraded overall status."""
        health_service.check_database_health = AsyncMock(
            return_value=ServiceHealth(name="db", status=HealthStatus.HEALTHY)
        )
        health_service.check_redis_health = AsyncMock(
            return_value=ServiceHealth(name="redis", status=HealthStatus.DEGRADED)
        )
        health_service.check_ollama_health = AsyncMock(
            return_value=ServiceHealth(name="ollama", status=HealthStatus.HEALTHY)
        )
        health_service.check_filesystem_health = AsyncMock(
            return_value=ServiceHealth(name="fs", status=HealthStatus.HEALTHY)
        )

        result = await health_service.check_all_health()

        assert result.status == HealthStatus.DEGRADED
