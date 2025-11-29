"""
Analytics API Routes

Handles performance metrics and analytics data.
"""

import os
import time
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, status
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.config.logging import get_logger
from backend.database.connection import get_db_session
from backend.models.content import ContentModel
from backend.models.persona import PersonaModel

logger = get_logger(__name__)

# Track server start time for uptime calculation
_server_start_time = time.time()

router = APIRouter(
    prefix="/api/v1/analytics",
    tags=["analytics"],
)


@router.get("/metrics", status_code=status.HTTP_200_OK)
async def get_metrics(db: AsyncSession = Depends(get_db_session)):
    """
    Get platform metrics.

    Returns real platform metrics from the database including:
    - Total personas created
    - Total content generated
    - System uptime
    - Operational status

    Args:
        db: Database session

    Returns:
        Dict with platform metrics
    """
    try:
        # Get total personas count
        persona_stmt = select(func.count(PersonaModel.id))
        persona_result = await db.execute(persona_stmt)
        personas_created = persona_result.scalar() or 0

        # Get total content count
        content_stmt = select(func.count(ContentModel.id))
        content_result = await db.execute(content_stmt)
        content_generated = content_result.scalar() or 0

        # Calculate system uptime
        uptime_seconds = int(time.time() - _server_start_time)
        uptime_hours = uptime_seconds // 3600
        uptime_minutes = (uptime_seconds % 3600) // 60
        system_uptime = f"{uptime_hours}h {uptime_minutes}m"

        return {
            "personas_created": personas_created,
            "content_generated": content_generated,
            "api_requests_today": 0,  # Would require request tracking middleware
            "system_uptime": system_uptime,
            "status": "operational",
        }
    except Exception as e:
        logger.error(f"Error fetching metrics: {e}")
        # Return degraded status but don't fail
        return {
            "personas_created": 0,
            "content_generated": 0,
            "api_requests_today": 0,
            "system_uptime": "unknown",
            "status": "degraded",
        }


@router.get("/health", status_code=status.HTTP_200_OK)
async def get_system_health(db: AsyncSession = Depends(get_db_session)):
    """
    Get detailed system health information.

    Returns health status of various system components by actually
    testing database connectivity and checking for AI model configuration.

    Args:
        db: Database session

    Returns:
        Dict with detailed health information for each system component
    """
    health_status = {
        "api": "healthy",
        "database": "unknown",
        "ai_models": "not_loaded",
        "content_generation": "not_configured",
        "gpu_monitoring": "unknown",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    # Test database connectivity
    try:
        await db.execute(select(1))
        health_status["database"] = "healthy"
    except Exception as e:
        logger.error(f"Database health check failed: {e}")
        health_status["database"] = "unhealthy"

    # Check if AI models are configured (environment variables)
    has_openai = bool(os.getenv("OPENAI_API_KEY"))
    has_hf = bool(os.getenv("HUGGING_FACE_TOKEN"))

    if has_openai or has_hf:
        health_status["ai_models"] = "configured"
        health_status["content_generation"] = "configured"

    # Check GPU monitoring status
    try:
        from backend.services.gpu_monitoring_service import get_gpu_monitoring_service

        gpu_service = get_gpu_monitoring_service()
        gpu_temps = await gpu_service.get_gpu_temperatures()

        if gpu_temps.get("available"):
            health_status["gpu_monitoring"] = "healthy"
            health_status["gpu_count"] = gpu_temps.get("gpu_count", 0)

            # Add temperature warnings if any GPU is hot
            max_temp = 0
            for gpu in gpu_temps.get("gpus", []):
                temp = gpu.get("temperature_c")
                if temp is not None and temp > max_temp:
                    max_temp = temp

            if max_temp >= 85:
                health_status["gpu_warning"] = f"Critical GPU temperature: {max_temp}°C"
            elif max_temp >= 75:
                health_status["gpu_warning"] = f"High GPU temperature: {max_temp}°C"
        else:
            health_status["gpu_monitoring"] = "not_available"
    except Exception as e:
        logger.debug(f"GPU monitoring check failed: {e}")
        health_status["gpu_monitoring"] = "unavailable"

    return health_status
