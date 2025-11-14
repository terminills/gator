"""
Test database connection cleanup handling.

Tests that the database manager properly handles errors during connection
disposal, particularly the "Exception terminating connection" error.
"""

import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from backend.database.connection import DatabaseManager


@pytest.mark.asyncio
async def test_disconnect_handles_disposal_errors():
    """Test that disconnect gracefully handles errors during engine disposal."""
    manager = DatabaseManager()
    
    # Create a mock engine that raises an exception during dispose
    mock_engine = AsyncMock()
    mock_engine.dispose = AsyncMock(side_effect=Exception("Terminating connection failed"))
    
    # Set the engine directly
    manager.engine = mock_engine
    manager.session_factory = MagicMock()
    
    # Disconnect should not raise an exception
    await manager.disconnect()
    
    # Verify that dispose was called
    mock_engine.dispose.assert_called_once()
    
    # Verify that engine and session_factory are cleared even with error
    assert manager.engine is None
    assert manager.session_factory is None


@pytest.mark.asyncio
async def test_disconnect_successful_cleanup():
    """Test normal disconnect flow without errors."""
    manager = DatabaseManager()
    
    # Create a mock engine that disposes successfully
    mock_engine = AsyncMock()
    mock_engine.dispose = AsyncMock()
    
    # Set the engine directly
    manager.engine = mock_engine
    manager.session_factory = MagicMock()
    
    # Disconnect should work normally
    await manager.disconnect()
    
    # Verify that dispose was called
    mock_engine.dispose.assert_called_once()
    
    # Verify that engine and session_factory are cleared
    assert manager.engine is None
    assert manager.session_factory is None


@pytest.mark.asyncio
async def test_disconnect_when_already_disconnected():
    """Test that disconnect handles being called when already disconnected."""
    manager = DatabaseManager()
    
    # Engine is already None
    manager.engine = None
    manager.session_factory = None
    
    # Should not raise an exception
    await manager.disconnect()
    
    # Should still be None
    assert manager.engine is None
    assert manager.session_factory is None


@pytest.mark.asyncio
async def test_disconnect_logs_warning_on_error():
    """Test that disconnect logs a warning when disposal fails."""
    manager = DatabaseManager()
    
    # Create a mock engine that raises an exception
    mock_engine = AsyncMock()
    error_msg = "Connection already terminated"
    mock_engine.dispose = AsyncMock(side_effect=Exception(error_msg))
    
    manager.engine = mock_engine
    manager.session_factory = MagicMock()
    
    # Patch the logger to verify warning is logged
    with patch('backend.database.connection.logger') as mock_logger:
        await manager.disconnect()
        
        # Verify warning was logged with error message
        mock_logger.warning.assert_called_once()
        warning_call = mock_logger.warning.call_args[0][0]
        assert "Non-critical error during connection cleanup" in warning_call
        assert error_msg in warning_call
        
        # Verify info log about disconnection
        mock_logger.info.assert_called_once_with("Database disconnected")
