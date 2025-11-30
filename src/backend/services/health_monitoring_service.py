"""
Production Health Monitoring Service

Provides comprehensive health checks for production deployment:
- Database connectivity and pool status
- Redis connectivity
- AI model availability
- External service dependencies
- System resources (memory, disk)
"""

import asyncio
import os
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from backend.config.logging import get_logger
from backend.config.settings import get_settings

logger = get_logger(__name__)
settings = get_settings()


class HealthStatus(str, Enum):
    """Health status levels."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


class ServiceHealth(BaseModel):
    """Health status for a single service."""

    name: str
    status: HealthStatus
    latency_ms: Optional[float] = None
    message: Optional[str] = None
    details: Dict[str, Any] = {}
    checked_at: datetime = datetime.utcnow()


class SystemHealth(BaseModel):
    """Overall system health status."""

    status: HealthStatus
    services: List[ServiceHealth]
    timestamp: datetime = datetime.utcnow()
    uptime_seconds: float = 0
    version: str = "0.1.0"


class ResourceMetrics(BaseModel):
    """System resource metrics."""

    cpu_percent: float = 0.0
    memory_total_gb: float = 0.0
    memory_used_gb: float = 0.0
    memory_percent: float = 0.0
    disk_total_gb: float = 0.0
    disk_used_gb: float = 0.0
    disk_percent: float = 0.0


class HealthMonitoringService:
    """
    Service for monitoring system health and dependencies.

    Performs health checks on all critical system components
    and provides aggregated status for monitoring tools.
    """

    _start_time: float = time.time()

    def __init__(self, db_session: Optional[AsyncSession] = None):
        """
        Initialize health monitoring service.

        Args:
            db_session: Optional database session for DB health checks
        """
        self.db = db_session

    async def check_all_health(self) -> SystemHealth:
        """
        Perform comprehensive health check on all services.

        Returns:
            SystemHealth with aggregated status
        """
        services = []

        # Check database
        db_health = await self.check_database_health()
        services.append(db_health)

        # Check Redis
        redis_health = await self.check_redis_health()
        services.append(redis_health)

        # Check Ollama
        ollama_health = await self.check_ollama_health()
        services.append(ollama_health)

        # Check file system
        fs_health = await self.check_filesystem_health()
        services.append(fs_health)

        # Determine overall status
        statuses = [s.status for s in services]
        if all(s == HealthStatus.HEALTHY for s in statuses):
            overall_status = HealthStatus.HEALTHY
        elif any(s == HealthStatus.UNHEALTHY for s in statuses):
            overall_status = HealthStatus.UNHEALTHY
        elif any(s == HealthStatus.DEGRADED for s in statuses):
            overall_status = HealthStatus.DEGRADED
        else:
            overall_status = HealthStatus.UNKNOWN

        return SystemHealth(
            status=overall_status,
            services=services,
            uptime_seconds=time.time() - self._start_time,
            version=settings.environment,
        )

    async def check_database_health(self) -> ServiceHealth:
        """Check database connectivity and response time."""
        start = time.time()

        try:
            if self.db:
                # Execute simple query
                result = await self.db.execute(text("SELECT 1"))
                result.scalar()

                latency = (time.time() - start) * 1000

                return ServiceHealth(
                    name="database",
                    status=HealthStatus.HEALTHY,
                    latency_ms=round(latency, 2),
                    message="Database connection successful",
                )
            else:
                return ServiceHealth(
                    name="database",
                    status=HealthStatus.UNKNOWN,
                    message="No database session available",
                )

        except Exception as e:
            latency = (time.time() - start) * 1000
            logger.error(f"Database health check failed: {e}")

            return ServiceHealth(
                name="database",
                status=HealthStatus.UNHEALTHY,
                latency_ms=round(latency, 2),
                message=f"Database error: {str(e)}",
            )

    async def check_redis_health(self) -> ServiceHealth:
        """Check Redis connectivity."""
        start = time.time()

        try:
            import redis.asyncio as redis

            client = redis.from_url(settings.redis_url, socket_timeout=5)
            await client.ping()
            await client.close()

            latency = (time.time() - start) * 1000

            return ServiceHealth(
                name="redis",
                status=HealthStatus.HEALTHY,
                latency_ms=round(latency, 2),
                message="Redis connection successful",
            )

        except ImportError:
            return ServiceHealth(
                name="redis",
                status=HealthStatus.DEGRADED,
                message="Redis client not installed",
            )

        except Exception as e:
            latency = (time.time() - start) * 1000
            logger.warning(f"Redis health check failed: {e}")

            return ServiceHealth(
                name="redis",
                status=HealthStatus.DEGRADED,
                latency_ms=round(latency, 2),
                message=f"Redis unavailable: {str(e)}",
                details={"note": "Application can function without Redis using in-memory fallback"},
            )

    async def check_ollama_health(self) -> ServiceHealth:
        """Check Ollama AI service connectivity."""
        start = time.time()

        try:
            import httpx

            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(f"{settings.ollama_base_url}/api/tags")

            latency = (time.time() - start) * 1000

            if response.status_code == 200:
                data = response.json()
                model_count = len(data.get("models", []))

                return ServiceHealth(
                    name="ollama",
                    status=HealthStatus.HEALTHY,
                    latency_ms=round(latency, 2),
                    message=f"Ollama available with {model_count} models",
                    details={"models_loaded": model_count},
                )
            else:
                return ServiceHealth(
                    name="ollama",
                    status=HealthStatus.DEGRADED,
                    latency_ms=round(latency, 2),
                    message=f"Ollama returned status {response.status_code}",
                )

        except Exception as e:
            latency = (time.time() - start) * 1000
            logger.warning(f"Ollama health check failed: {e}")

            return ServiceHealth(
                name="ollama",
                status=HealthStatus.DEGRADED,
                latency_ms=round(latency, 2),
                message="Ollama unavailable",
                details={"note": "Text generation features may be limited"},
            )

    async def check_filesystem_health(self) -> ServiceHealth:
        """Check filesystem access and available space."""
        try:
            content_path = settings.content_storage_path

            # Ensure path exists
            if not os.path.exists(content_path):
                os.makedirs(content_path, exist_ok=True)

            # Check disk space
            statvfs = os.statvfs(content_path)
            free_gb = (statvfs.f_frsize * statvfs.f_bavail) / (1024**3)
            total_gb = (statvfs.f_frsize * statvfs.f_blocks) / (1024**3)
            used_percent = ((total_gb - free_gb) / total_gb) * 100 if total_gb > 0 else 0

            # Test write access
            test_file = os.path.join(content_path, ".health_check")
            with open(test_file, "w") as f:
                f.write("health_check")
            os.remove(test_file)

            if used_percent > 95:
                status = HealthStatus.UNHEALTHY
                message = f"Critical: Disk {used_percent:.1f}% full"
            elif used_percent > 85:
                status = HealthStatus.DEGRADED
                message = f"Warning: Disk {used_percent:.1f}% full"
            else:
                status = HealthStatus.HEALTHY
                message = f"Disk usage {used_percent:.1f}%"

            return ServiceHealth(
                name="filesystem",
                status=status,
                message=message,
                details={
                    "free_gb": round(free_gb, 2),
                    "total_gb": round(total_gb, 2),
                    "used_percent": round(used_percent, 1),
                    "path": content_path,
                },
            )

        except Exception as e:
            logger.error(f"Filesystem health check failed: {e}")

            return ServiceHealth(
                name="filesystem",
                status=HealthStatus.UNHEALTHY,
                message=f"Filesystem error: {str(e)}",
            )

    async def get_resource_metrics(self) -> ResourceMetrics:
        """Get current system resource metrics."""
        try:
            import psutil

            # CPU
            cpu_percent = psutil.cpu_percent(interval=0.1)

            # Memory
            memory = psutil.virtual_memory()
            memory_total_gb = memory.total / (1024**3)
            memory_used_gb = memory.used / (1024**3)

            # Disk
            disk = psutil.disk_usage("/")
            disk_total_gb = disk.total / (1024**3)
            disk_used_gb = disk.used / (1024**3)

            return ResourceMetrics(
                cpu_percent=round(cpu_percent, 1),
                memory_total_gb=round(memory_total_gb, 2),
                memory_used_gb=round(memory_used_gb, 2),
                memory_percent=round(memory.percent, 1),
                disk_total_gb=round(disk_total_gb, 2),
                disk_used_gb=round(disk_used_gb, 2),
                disk_percent=round(disk.percent, 1),
            )

        except ImportError:
            logger.warning("psutil not available for resource metrics")
            return ResourceMetrics()

        except Exception as e:
            logger.error(f"Failed to get resource metrics: {e}")
            return ResourceMetrics()

    async def get_readiness_status(self) -> Dict[str, Any]:
        """
        Check if the application is ready to serve traffic.

        For Kubernetes readiness probe.
        """
        health = await self.check_all_health()

        return {
            "ready": health.status in [HealthStatus.HEALTHY, HealthStatus.DEGRADED],
            "status": health.status.value,
            "timestamp": datetime.utcnow().isoformat(),
        }

    async def get_liveness_status(self) -> Dict[str, Any]:
        """
        Check if the application is alive.

        For Kubernetes liveness probe.
        """
        return {
            "alive": True,
            "uptime_seconds": round(time.time() - self._start_time, 2),
            "timestamp": datetime.utcnow().isoformat(),
        }
