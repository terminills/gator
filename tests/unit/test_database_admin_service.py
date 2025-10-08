"""
Tests for Database Administration Service and API

Tests backup creation, schema synchronization, and database info retrieval.
"""

import pytest
from pathlib import Path
from sqlalchemy import text

from backend.services.database_admin_service import DatabaseAdminService
from backend.database.connection import database_manager, Base


@pytest.fixture
async def db_admin_service(db_session):
    """Create database admin service instance for testing."""
    # Ensure database is connected
    if not database_manager.engine:
        await database_manager.connect()

    service = DatabaseAdminService()
    # Override backup directory for tests
    service.backup_dir = Path("/tmp/gator_test_backups")
    service.backup_dir.mkdir(exist_ok=True)
    return service


@pytest.mark.asyncio
async def test_get_database_info(db_admin_service):
    """Test getting database information."""
    result = await db_admin_service.get_database_info()

    assert result["success"] is True
    assert "database_type" in result
    assert "connection_status" in result
    assert result["connection_status"] == "connected"
    assert "table_count" in result
    assert result["table_count"] > 0


@pytest.mark.asyncio
async def test_check_schema_status(db_admin_service):
    """Test checking database schema status."""
    result = await db_admin_service.check_schema_status()

    assert result["success"] is True
    assert "in_sync" in result
    assert "database_tables" in result
    assert "model_tables" in result
    assert "missing_tables" in result
    assert "extra_tables" in result
    assert isinstance(result["database_tables"], list)
    assert isinstance(result["model_tables"], list)


@pytest.mark.asyncio
async def test_sync_schema_when_in_sync(db_admin_service):
    """Test schema sync when already in sync."""
    result = await db_admin_service.sync_schema()

    assert result["success"] is True
    # Should report no changes needed if already in sync
    if result.get("in_sync"):
        assert result["changes_made"] is False


@pytest.mark.asyncio
async def test_create_sqlite_backup(db_admin_service):
    """Test creating SQLite database backup."""
    # Only test if using SQLite
    if not db_admin_service.is_sqlite:
        pytest.skip("Skipping SQLite backup test for non-SQLite database")

    result = await db_admin_service.create_backup()

    assert result["success"] is True
    assert "backup" in result
    assert "filename" in result["backup"]
    assert "path" in result["backup"]
    assert "size_bytes" in result["backup"]
    assert result["backup"]["type"] == "sqlite"

    # Verify backup file was created
    backup_path = Path(result["backup"]["path"])
    assert backup_path.exists()
    assert backup_path.is_file()


@pytest.mark.asyncio
async def test_list_backups(db_admin_service):
    """Test listing database backups."""
    # Create a backup first
    if db_admin_service.is_sqlite:
        create_result = await db_admin_service.create_backup()
        assert create_result["success"] is True

    # List backups
    result = await db_admin_service.list_backups()

    assert result["success"] is True
    assert "backups" in result
    assert "count" in result
    assert isinstance(result["backups"], list)
    assert result["count"] == len(result["backups"])

    # If we created a backup, verify it's in the list
    if db_admin_service.is_sqlite:
        assert result["count"] > 0
        backup = result["backups"][0]
        assert "filename" in backup
        assert "size_bytes" in backup
        assert "created" in backup


@pytest.mark.asyncio
async def test_backup_contains_data(db_admin_service):
    """Test that backup file contains actual data."""
    if not db_admin_service.is_sqlite:
        pytest.skip("Skipping SQLite backup test for non-SQLite database")

    result = await db_admin_service.create_backup()
    assert result["success"] is True

    # Backup should have reasonable size (at least a few KB)
    assert result["backup"]["size_bytes"] > 1000


@pytest.mark.asyncio
async def test_database_info_includes_tables(db_admin_service):
    """Test that database info includes table list."""
    result = await db_admin_service.get_database_info()

    assert result["success"] is True
    assert "tables" in result
    assert isinstance(result["tables"], list)

    # Should have common tables
    expected_tables = ["personas", "users", "conversations", "messages"]
    for table in expected_tables:
        assert table in result["tables"], f"Expected table '{table}' not found"


@pytest.mark.asyncio
async def test_schema_status_identifies_tables(db_admin_service):
    """Test that schema status correctly identifies tables."""
    result = await db_admin_service.check_schema_status()

    assert result["success"] is True

    # Should have expected model tables
    expected_model_tables = [
        "personas",
        "users",
        "conversations",
        "messages",
        "ppv_offers",
        "content",
        "rss_feeds",
        "feed_items",
    ]

    for table in expected_model_tables:
        assert (
            table in result["model_tables"]
        ), f"Expected model table '{table}' not found"
