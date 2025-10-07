"""
Integration tests for Database Administration API endpoints

Tests the REST API endpoints for database backup and schema sync.
"""

import pytest
from fastapi.testclient import TestClient


@pytest.mark.asyncio
async def test_get_database_info_endpoint(test_client):
    """Test GET /api/v1/admin/database/info endpoint."""
    response = test_client.get("/api/v1/admin/database/info")
    
    assert response.status_code == 200
    data = response.json()
    
    assert data["success"] is True
    assert "database_type" in data
    assert "connection_status" in data
    assert "table_count" in data


@pytest.mark.asyncio
async def test_get_schema_status_endpoint(test_client):
    """Test GET /api/v1/admin/database/schema/status endpoint."""
    response = test_client.get("/api/v1/admin/database/schema/status")
    
    assert response.status_code == 200
    data = response.json()
    
    assert data["success"] is True
    assert "in_sync" in data
    assert "database_tables" in data
    assert "model_tables" in data


@pytest.mark.asyncio
async def test_sync_schema_endpoint(test_client):
    """Test POST /api/v1/admin/database/schema/sync endpoint."""
    response = test_client.post("/api/v1/admin/database/schema/sync")
    
    assert response.status_code == 200
    data = response.json()
    
    assert data["success"] is True
    assert "message" in data


@pytest.mark.asyncio
async def test_list_backups_endpoint(test_client):
    """Test GET /api/v1/admin/database/backups endpoint."""
    response = test_client.get("/api/v1/admin/database/backups")
    
    assert response.status_code == 200
    data = response.json()
    
    assert data["success"] is True
    assert "backups" in data
    assert "count" in data
    assert isinstance(data["backups"], list)


@pytest.mark.asyncio
async def test_create_backup_endpoint(test_client):
    """Test POST /api/v1/admin/database/backup endpoint."""
    response = test_client.post("/api/v1/admin/database/backup")
    
    assert response.status_code == 200
    data = response.json()
    
    assert data["success"] is True
    assert "backup" in data
    assert "filename" in data["backup"]


@pytest.mark.asyncio
async def test_download_backup_invalid_filename(test_client):
    """Test that invalid filenames are rejected."""
    # Test path traversal attempt - FastAPI routing will catch this with 404
    response = test_client.get("/api/v1/admin/database/backups/../etc/passwd")
    # Either 400 (our validation) or 404 (FastAPI routing) is acceptable
    assert response.status_code in [400, 404]
    
    # Test with backslash in filename
    response = test_client.get("/api/v1/admin/database/backups/..\\file.db")
    assert response.status_code in [400, 404]


@pytest.mark.asyncio
async def test_download_nonexistent_backup(test_client):
    """Test downloading a backup that doesn't exist."""
    response = test_client.get("/api/v1/admin/database/backups/nonexistent.db")
    assert response.status_code == 404
