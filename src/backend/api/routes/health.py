"""
Health Monitoring API Routes

Provides production health check endpoints:
- /health - Basic health status
- /health/live - Kubernetes liveness probe
- /health/ready - Kubernetes readiness probe
- /health/detailed - Comprehensive health with all services
- /metrics - System resource metrics
"""

from typing import Optional

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from backend.config.logging import get_logger
from backend.database.connection import get_db_session
from backend.services.health_monitoring_service import (
    HealthMonitoringService,
    HealthStatus,
    ResourceMetrics,
    SystemHealth,
)

logger = get_logger(__name__)

router = APIRouter(
    prefix="/health",
    tags=["health"],
)


def get_health_service(
    db: AsyncSession = Depends(get_db_session),
) -> HealthMonitoringService:
    """Dependency injection for HealthMonitoringService."""
    return HealthMonitoringService(db)


@router.get("")
async def basic_health_check(
    health_service: HealthMonitoringService = Depends(get_health_service),
):
    """
    Basic health check endpoint.

    Returns simple status for load balancers and monitoring systems.
    This is a fast check that verifies the application is running.

    Returns:
        Dict with status and database connectivity
    """
    db_health = await health_service.check_database_health()

    return {
        "status": "healthy" if db_health.status == HealthStatus.HEALTHY else "degraded",
        "database": db_health.status.value,
        "timestamp": db_health.checked_at.isoformat(),
    }


@router.get("/live")
async def liveness_probe(
    health_service: HealthMonitoringService = Depends(get_health_service),
):
    """
    Kubernetes liveness probe.

    Returns 200 if the application is alive.
    Used by Kubernetes to restart unhealthy pods.

    Returns:
        Dict with alive status and uptime
    """
    return await health_service.get_liveness_status()


@router.get("/ready")
async def readiness_probe(
    health_service: HealthMonitoringService = Depends(get_health_service),
):
    """
    Kubernetes readiness probe.

    Returns 200 if the application is ready to receive traffic.
    Used by Kubernetes to control traffic routing.

    Returns:
        Dict with ready status
    """
    return await health_service.get_readiness_status()


@router.get("/detailed", response_model=SystemHealth)
async def detailed_health_check(
    health_service: HealthMonitoringService = Depends(get_health_service),
):
    """
    Detailed health check with all service statuses.

    Performs comprehensive health checks on:
    - Database connection
    - Redis cache
    - Ollama AI service
    - File system access

    Returns:
        SystemHealth with detailed service statuses
    """
    return await health_service.check_all_health()


@router.get("/metrics", response_model=ResourceMetrics)
async def get_system_metrics(
    health_service: HealthMonitoringService = Depends(get_health_service),
):
    """
    Get system resource metrics.

    Returns current CPU, memory, and disk usage.
    Useful for capacity planning and monitoring.

    Returns:
        ResourceMetrics with system resource usage
    """
    return await health_service.get_resource_metrics()


@router.get("/services")
async def get_service_status(
    health_service: HealthMonitoringService = Depends(get_health_service),
):
    """
    Get status of individual services.

    Returns a summary of each service's health status.

    Returns:
        Dict with service statuses
    """
    health = await health_service.check_all_health()

    return {
        "services": {
            s.name: {
                "status": s.status.value,
                "latency_ms": s.latency_ms,
                "message": s.message,
            }
            for s in health.services
        },
        "overall_status": health.status.value,
    }


@router.get("/database")
async def get_database_health(
    health_service: HealthMonitoringService = Depends(get_health_service),
):
    """
    Get detailed database health status.

    Returns:
        ServiceHealth for database
    """
    return await health_service.check_database_health()


@router.get("/redis")
async def get_redis_health(
    health_service: HealthMonitoringService = Depends(get_health_service),
):
    """
    Get Redis cache health status.

    Returns:
        ServiceHealth for Redis
    """
    return await health_service.check_redis_health()


@router.get("/ollama")
async def get_ollama_health(
    health_service: HealthMonitoringService = Depends(get_health_service),
):
    """
    Get Ollama AI service health status.

    Returns:
        ServiceHealth for Ollama
    """
    return await health_service.check_ollama_health()


@router.get("/filesystem")
async def get_filesystem_health(
    health_service: HealthMonitoringService = Depends(get_health_service),
):
    """
    Get filesystem health status.

    Returns:
        ServiceHealth for filesystem
    """
    return await health_service.check_filesystem_health()


@router.get("/circuit-breakers")
async def get_circuit_breaker_status():
    """
    Get status of all circuit breakers.

    Returns the state of each circuit breaker for monitoring external
    service dependencies and identifying failing integrations.

    Returns:
        Dict with circuit breaker statuses
    """
    from backend.utils.circuit_breaker import CircuitBreaker

    return {
        "circuit_breakers": CircuitBreaker.get_all_status(),
        "summary": {
            "total": len(CircuitBreaker._registry),
            "open": sum(
                1 for cb in CircuitBreaker._registry.values() if cb.state.value == "open"
            ),
            "half_open": sum(
                1
                for cb in CircuitBreaker._registry.values()
                if cb.state.value == "half_open"
            ),
            "closed": sum(
                1
                for cb in CircuitBreaker._registry.values()
                if cb.state.value == "closed"
            ),
        },
    }


@router.post("/circuit-breakers/reset")
async def reset_circuit_breakers():
    """
    Reset all circuit breakers to closed state.

    Use this endpoint to recover from transient failures after
    the underlying issues have been resolved.

    Returns:
        Dict with reset confirmation
    """
    from backend.utils.circuit_breaker import CircuitBreaker

    count = len(CircuitBreaker._registry)
    CircuitBreaker.reset_all()

    logger.info(f"Reset {count} circuit breakers")

    return {
        "message": f"Reset {count} circuit breakers",
        "status": "success",
    }
