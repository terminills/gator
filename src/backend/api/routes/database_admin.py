"""
Database Administration API Routes

Provides endpoints for database backup and schema synchronization
through the admin panel.
"""

from typing import Dict, Any
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import FileResponse
from pathlib import Path

from backend.services.database_admin_service import (
    DatabaseAdminService,
    get_database_admin_service,
)
from backend.config.logging import get_logger

logger = get_logger(__name__)
router = APIRouter(prefix="/admin/database", tags=["admin", "database"])


@router.get("/info")
async def get_database_info(
    service: DatabaseAdminService = Depends(get_database_admin_service),
) -> Dict[str, Any]:
    """
    Get general database information.

    Returns database type, connection status, size, and table count.
    """
    return await service.get_database_info()


@router.get("/schema/status")
async def get_schema_status(
    service: DatabaseAdminService = Depends(get_database_admin_service),
) -> Dict[str, Any]:
    """
    Check if database schema is in sync with current models.

    Returns information about missing or extra tables and whether
    the schema needs synchronization.
    """
    return await service.check_schema_status()


@router.post("/schema/sync")
async def sync_schema(
    service: DatabaseAdminService = Depends(get_database_admin_service),
) -> Dict[str, Any]:
    """
    Synchronize database schema with current models.

    Creates missing tables to match the current codebase models.
    Does not modify existing tables to prevent data loss.

    Returns:
        Dict with sync results including tables created
    """
    result = await service.sync_schema()

    if not result["success"]:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=result.get("message", "Schema sync failed"),
        )

    return result


@router.post("/backup")
async def create_backup(
    service: DatabaseAdminService = Depends(get_database_admin_service),
) -> Dict[str, Any]:
    """
    Create a backup of the database.

    For SQLite: Creates a copy of the database file
    For PostgreSQL: Runs pg_dump to create a SQL backup

    Returns:
        Dict with backup information (filename, path, size)
    """
    result = await service.create_backup()

    if not result["success"]:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=result.get("message", "Backup creation failed"),
        )

    return result


@router.get("/backups")
async def list_backups(
    service: DatabaseAdminService = Depends(get_database_admin_service),
) -> Dict[str, Any]:
    """
    List all available database backups.

    Returns a list of backup files with their metadata including
    filename, size, and creation timestamp.
    """
    return await service.list_backups()


@router.get("/backups/{filename}")
async def download_backup(
    filename: str,
    service: DatabaseAdminService = Depends(get_database_admin_service),
):
    """
    Download a specific backup file.

    Args:
        filename: Name of the backup file to download

    Returns:
        File download response
    """
    # Validate filename to prevent path traversal
    if ".." in filename or "/" in filename or "\\" in filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid filename",
        )

    backup_path = service.backup_dir / filename

    if not backup_path.exists() or not backup_path.is_file():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Backup file not found",
        )

    return FileResponse(
        path=backup_path,
        filename=filename,
        media_type="application/octet-stream",
    )
