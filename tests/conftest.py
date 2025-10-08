"""
Test configuration and fixtures.

Provides common test setup, fixtures, and utilities.
"""

import pytest
import asyncio
from typing import AsyncGenerator
from uuid import uuid4

from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from fastapi.testclient import TestClient

from backend.database.connection import Base, get_db_session
from backend.api.main import create_app
from backend.models.persona import PersonaModel
from backend.models.user import UserModel
from backend.models.content import ContentModel


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="session")
async def test_db():
    """Create test database engine and session factory."""
    # Use in-memory SQLite for testing
    engine = create_async_engine("sqlite+aiosqlite:///:memory:", echo=False)

    # Create tables
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    session_factory = async_sessionmaker(
        bind=engine, class_=AsyncSession, expire_on_commit=False
    )

    yield session_factory

    await engine.dispose()


@pytest.fixture
async def db_session(test_db):
    """Provide a database session for tests with automatic cleanup."""
    async with test_db() as session:
        try:
            yield session
        finally:
            # Roll back any uncommitted changes to ensure test isolation
            await session.rollback()
            await session.close()


@pytest.fixture
async def test_client(db_session):
    """Create test client with dependency override."""
    app = create_app()

    # Override database dependency with async generator
    async def override_get_db():
        yield db_session

    app.dependency_overrides[get_db_session] = override_get_db

    with TestClient(app) as client:
        yield client

    # Clean up overrides
    app.dependency_overrides.clear()


@pytest.fixture
async def test_persona(db_session):
    """Create a test persona."""
    persona = PersonaModel(
        id=uuid4(),
        name="Test Persona",
        appearance="Test appearance",
        personality="Test personality",
        content_themes=["tech", "fitness"],
        style_preferences={"tone": "casual"},
        default_content_rating="sfw",
        allowed_content_ratings=["sfw", "moderate"],
        platform_restrictions={},
        base_image_status="pending_upload",
        is_active=True,
        generation_count=0,
    )
    db_session.add(persona)
    await db_session.commit()
    await db_session.refresh(persona)
    return persona


@pytest.fixture
async def test_user(db_session):
    """Create a test user."""
    user = UserModel(
        id=uuid4(),
        username="testuser",
        email="test@example.com",
        is_active=True,
    )
    db_session.add(user)
    await db_session.commit()
    await db_session.refresh(user)
    return user


@pytest.fixture
async def test_content(db_session, test_persona):
    """Create test content."""
    content = ContentModel(
        id=uuid4(),
        persona_id=test_persona.id,
        content_type="image",
        title="Test Content",
        description="Test description",
        file_path="/test/path.jpg",
        content_rating="sfw",
        moderation_status="pending",
        is_published=False,
    )
    db_session.add(content)
    await db_session.commit()
    await db_session.refresh(content)
    return content

