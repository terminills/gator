"""
Pytest configuration for unit tests.
"""

import pytest
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from backend.database.connection import Base

# Import models to register them with Base
from backend.models.generation_feedback import GenerationBenchmarkModel
from backend.models.acd import ACDContextModel, ACDTraceArtifactModel
from backend.models.content import ContentModel
from backend.models.persona import PersonaModel


@pytest.fixture
async def test_db_session():
    """
    Create a test database session.
    
    Uses in-memory SQLite for fast, isolated testing.
    """
    # Create in-memory SQLite database
    engine = create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        echo=False,
    )
    
    # Create all tables
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    
    # Create session factory
    session_factory = async_sessionmaker(
        bind=engine,
        class_=AsyncSession,
        expire_on_commit=False,
    )
    
    # Create session
    async with session_factory() as session:
        yield session
    
    # Cleanup
    await engine.dispose()
