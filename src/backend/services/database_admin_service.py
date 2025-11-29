"""
Database Administration Service

Provides database backup and schema synchronization functionality
for the admin panel.
"""

import os
import shutil
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from sqlalchemy import inspect, text
from sqlalchemy.exc import SQLAlchemyError

from backend.config.logging import get_logger
from backend.config.settings import get_settings
from backend.database.connection import Base, database_manager

logger = get_logger(__name__)


class DatabaseAdminService:
    """Service for database administration tasks."""

    def __init__(self):
        self.settings = get_settings()
        self.db_url = self.settings.database_url
        self.is_sqlite = self.db_url.startswith("sqlite")
        self.backup_dir = Path("./backups")
        self.backup_dir.mkdir(exist_ok=True)

    async def create_backup(self) -> Dict[str, Any]:
        """
        Create a backup of the database.

        Returns:
            Dict with backup information (path, timestamp, size)
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        try:
            if self.is_sqlite:
                backup_info = await self._backup_sqlite(timestamp)
            else:
                backup_info = await self._backup_postgresql(timestamp)

            logger.info(f"Database backup created: {backup_info['filename']}")
            return {
                "success": True,
                "backup": backup_info,
                "message": "Database backup created successfully",
            }
        except Exception as e:
            logger.error(f"Failed to create database backup: {e}")
            return {
                "success": False,
                "error": str(e),
                "message": "Failed to create database backup",
            }

    async def _backup_sqlite(self, timestamp: str) -> Dict[str, Any]:
        """Create SQLite database backup."""
        # Extract database path from URL
        db_path = self.db_url.replace("sqlite:///", "").replace("sqlite:", "")
        if db_path.startswith("./"):
            db_path = db_path[2:]

        source_path = Path(db_path)
        if not source_path.exists():
            raise FileNotFoundError(f"Database file not found: {db_path}")

        # Create backup filename
        filename = f"gator_backup_{timestamp}.db"
        backup_path = self.backup_dir / filename

        # Copy database file
        shutil.copy2(source_path, backup_path)

        # Get file size
        size_bytes = backup_path.stat().st_size
        size_mb = round(size_bytes / (1024 * 1024), 2)

        return {
            "filename": filename,
            "path": str(backup_path.absolute()),
            "timestamp": timestamp,
            "size_bytes": size_bytes,
            "size_mb": size_mb,
            "type": "sqlite",
        }

    async def _backup_postgresql(self, timestamp: str) -> Dict[str, Any]:
        """Create PostgreSQL database backup."""
        filename = f"gator_backup_{timestamp}.sql"
        backup_path = self.backup_dir / filename

        # Parse database URL to extract connection details
        # Format: postgresql://user:password@host:port/database
        url_parts = self.db_url.replace("postgresql://", "").split("@")
        if len(url_parts) != 2:
            raise ValueError("Invalid PostgreSQL URL format")

        user_pass = url_parts[0].split(":")
        host_db = url_parts[1].split("/")

        if len(user_pass) != 2 or len(host_db) != 2:
            raise ValueError("Invalid PostgreSQL URL format")

        username = user_pass[0]
        password = user_pass[1]
        host_port = host_db[0].split(":")
        host = host_port[0]
        port = host_port[1] if len(host_port) > 1 else "5432"
        database = host_db[1]

        # Set password environment variable
        env = os.environ.copy()
        env["PGPASSWORD"] = password

        # Run pg_dump
        cmd = [
            "pg_dump",
            "-h",
            host,
            "-p",
            port,
            "-U",
            username,
            "-d",
            database,
            "-f",
            str(backup_path),
            "--no-owner",
            "--no-privileges",
        ]

        result = subprocess.run(cmd, env=env, capture_output=True, text=True)

        if result.returncode != 0:
            raise RuntimeError(f"pg_dump failed: {result.stderr}")

        # Get file size
        size_bytes = backup_path.stat().st_size
        size_mb = round(size_bytes / (1024 * 1024), 2)

        return {
            "filename": filename,
            "path": str(backup_path.absolute()),
            "timestamp": timestamp,
            "size_bytes": size_bytes,
            "size_mb": size_mb,
            "type": "postgresql",
        }

    async def list_backups(self) -> Dict[str, Any]:
        """
        List all available database backups.

        Returns:
            Dict with list of backups
        """
        try:
            backups = []

            if not self.backup_dir.exists():
                return {"success": True, "backups": [], "count": 0}

            # Find all backup files
            patterns = ["gator_backup_*.db", "gator_backup_*.sql"]
            for pattern in patterns:
                for backup_file in self.backup_dir.glob(pattern):
                    if backup_file.is_file():
                        stat = backup_file.stat()
                        backups.append(
                            {
                                "filename": backup_file.name,
                                "path": str(backup_file.absolute()),
                                "size_bytes": stat.st_size,
                                "size_mb": round(stat.st_size / (1024 * 1024), 2),
                                "created": datetime.fromtimestamp(
                                    stat.st_ctime
                                ).isoformat(),
                                "modified": datetime.fromtimestamp(
                                    stat.st_mtime
                                ).isoformat(),
                                "type": (
                                    "sqlite"
                                    if backup_file.suffix == ".db"
                                    else "postgresql"
                                ),
                            }
                        )

            # Sort by creation time (newest first)
            backups.sort(key=lambda x: x["created"], reverse=True)

            return {
                "success": True,
                "backups": backups,
                "count": len(backups),
                "backup_directory": str(self.backup_dir.absolute()),
            }
        except Exception as e:
            logger.error(f"Failed to list backups: {e}")
            return {
                "success": False,
                "error": str(e),
                "message": "Failed to list backups",
            }

    async def check_schema_status(self) -> Dict[str, Any]:
        """
        Check if database schema is in sync with current models.

        Returns:
            Dict with schema status information
        """
        try:
            if not database_manager.engine:
                return {
                    "success": False,
                    "message": "Database not connected",
                    "in_sync": False,
                }

            # Get current database tables
            async with database_manager.engine.connect() as conn:
                inspector = await conn.run_sync(lambda sync_conn: inspect(sync_conn))
                db_tables = set(inspector.get_table_names())

            # Get expected tables from models
            model_tables = set(Base.metadata.tables.keys())

            # Find differences
            missing_tables = model_tables - db_tables
            extra_tables = db_tables - model_tables

            in_sync = len(missing_tables) == 0 and len(extra_tables) == 0

            result = {
                "success": True,
                "in_sync": in_sync,
                "database_tables": sorted(list(db_tables)),
                "model_tables": sorted(list(model_tables)),
                "missing_tables": sorted(list(missing_tables)),
                "extra_tables": sorted(list(extra_tables)),
                "total_tables_in_db": len(db_tables),
                "total_tables_in_models": len(model_tables),
            }

            if in_sync:
                result["message"] = "Database schema is in sync with current models"
            else:
                result["message"] = "Database schema is out of sync"

            return result

        except Exception as e:
            logger.error(f"Failed to check schema status: {e}")
            return {
                "success": False,
                "error": str(e),
                "message": "Failed to check schema status",
                "in_sync": False,
            }

    async def sync_schema(self) -> Dict[str, Any]:
        """
        Synchronize database schema with current models.
        Creates missing tables but does not modify existing ones.

        Returns:
            Dict with sync operation results
        """
        try:
            if not database_manager.engine:
                return {"success": False, "message": "Database not connected"}

            # Check current status
            status = await self.check_schema_status()
            if not status["success"]:
                return status

            if status["in_sync"]:
                return {
                    "success": True,
                    "message": "Database schema is already in sync",
                    "changes_made": False,
                    "tables_created": [],
                }

            # Create missing tables
            tables_created = []
            if status["missing_tables"]:
                async with database_manager.engine.begin() as conn:
                    # Create only missing tables
                    for table_name in status["missing_tables"]:
                        if table_name in Base.metadata.tables:
                            table = Base.metadata.tables[table_name]
                            await conn.run_sync(table.create)
                            tables_created.append(table_name)
                            logger.info(f"Created table: {table_name}")

            # Verify sync
            final_status = await self.check_schema_status()

            return {
                "success": True,
                "message": f"Schema sync completed. Created {len(tables_created)} table(s).",
                "changes_made": len(tables_created) > 0,
                "tables_created": tables_created,
                "extra_tables_warning": (
                    status["extra_tables"] if status["extra_tables"] else None
                ),
                "final_status": final_status,
            }

        except Exception as e:
            logger.error(f"Failed to sync schema: {e}")
            return {
                "success": False,
                "error": str(e),
                "message": f"Failed to sync schema: {str(e)}",
            }

    async def get_database_info(self) -> Dict[str, Any]:
        """
        Get general database information.

        Returns:
            Dict with database information
        """
        try:
            if not database_manager.engine:
                return {"success": False, "message": "Database not connected"}

            info = {
                "success": True,
                "database_url": self.db_url.split("@")[0].split("//")[0]
                + "//***",  # Hide credentials
                "database_type": "SQLite" if self.is_sqlite else "PostgreSQL",
                "connection_status": "connected",
            }

            # Get database size for SQLite
            if self.is_sqlite:
                db_path = self.db_url.replace("sqlite:///", "").replace("sqlite:", "")
                if db_path.startswith("./"):
                    db_path = db_path[2:]

                db_file = Path(db_path)
                if db_file.exists():
                    size_bytes = db_file.stat().st_size
                    info["size_bytes"] = size_bytes
                    info["size_mb"] = round(size_bytes / (1024 * 1024), 2)
                    info["database_file"] = str(db_file.absolute())

            # Get table count
            async with database_manager.engine.connect() as conn:
                table_names = await conn.run_sync(
                    lambda sync_conn: inspect(sync_conn).get_table_names()
                )
                info["table_count"] = len(table_names)
                info["tables"] = sorted(table_names)

            return info

        except Exception as e:
            logger.error(f"Failed to get database info: {e}")
            return {
                "success": False,
                "error": str(e),
                "message": "Failed to get database info",
            }

    async def optimize_database(self) -> Dict[str, Any]:
        """
        Optimize the database for better performance.

        For SQLite: Runs VACUUM and ANALYZE commands
        For PostgreSQL: Runs VACUUM ANALYZE

        Returns:
            Dict with optimization result
        """
        try:
            async with database_manager.get_session() as session:
                if self.is_sqlite:
                    # SQLite optimization
                    await session.execute(text("VACUUM"))
                    await session.execute(text("ANALYZE"))
                    await session.commit()

                    message = "Database optimized: VACUUM and ANALYZE completed"
                    logger.info(message)

                else:
                    # PostgreSQL optimization
                    # Note: VACUUM cannot run inside a transaction
                    await session.execute(text("VACUUM ANALYZE"))
                    await session.commit()

                    message = "Database optimized: VACUUM ANALYZE completed"
                    logger.info(message)

                return {
                    "success": True,
                    "message": message,
                    "timestamp": datetime.now().isoformat(),
                }

        except SQLAlchemyError as e:
            error_msg = f"Database optimization failed: {str(e)}"
            logger.error(error_msg)
            return {"success": False, "message": error_msg}
        except Exception as e:
            error_msg = f"Unexpected error during optimization: {str(e)}"
            logger.error(error_msg)
            return {"success": False, "message": error_msg}


# Global instance
_database_admin_service: Optional[DatabaseAdminService] = None


def get_database_admin_service() -> DatabaseAdminService:
    """Get or create the global database admin service instance."""
    global _database_admin_service
    if _database_admin_service is None:
        _database_admin_service = DatabaseAdminService()
    return _database_admin_service
