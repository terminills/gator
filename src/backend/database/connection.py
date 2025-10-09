"""
Database Connection Management

Handles database connectivity and session management using SQLAlchemy.
Supports async operations for better performance.
"""

from typing import AsyncGenerator, Optional
from contextlib import asynccontextmanager

from sqlalchemy import create_engine, text
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import declarative_base

from backend.config.settings import get_settings
from backend.config.logging import get_logger

logger = get_logger(__name__)

# Base class for all database models
Base = declarative_base()


class DatabaseManager:
    """
    Database connection and session management.

    Provides async database operations with proper connection pooling
    and session lifecycle management.
    """

    def __init__(self):
        self.engine: Optional[AsyncEngine] = None
        self.session_factory: Optional[async_sessionmaker[AsyncSession]] = None
        self._settings = get_settings()

    async def connect(self) -> None:
        """
        Initialize database connection and session factory.

        Creates the async engine and session factory based on the
        configured database URL.
        """
        if self.engine is not None:
            logger.warning("Database already connected")
            return

        # Convert sync SQLite URL to async for development
        database_url = self._settings.database_url
        if database_url.startswith("sqlite:"):
            database_url = database_url.replace("sqlite:", "sqlite+aiosqlite:", 1)

        # Configure engine with SQLite-specific settings for better concurrency
        connect_args = {}
        if "sqlite" in database_url:
            connect_args = {
                "check_same_thread": False,
                "timeout": 30,  # 30 second timeout for database locks
            }

        self.engine = create_async_engine(
            database_url,
            echo=self._settings.debug,
            future=True,
            pool_pre_ping=True,
            connect_args=connect_args,
        )

        self.session_factory = async_sessionmaker(
            bind=self.engine,
            class_=AsyncSession,
            expire_on_commit=False,
        )

        # Create tables if they don't exist (for development)
        if self._settings.debug and "sqlite" in database_url:
            async with self.engine.begin() as conn:
                await conn.run_sync(Base.metadata.create_all)

        # Enable WAL mode for SQLite for better concurrency with multiple workers
        if "sqlite" in database_url:
            async with self.engine.begin() as conn:
                await conn.execute(text("PRAGMA journal_mode=WAL"))
                await conn.execute(text("PRAGMA synchronous=NORMAL"))
                await conn.execute(text("PRAGMA busy_timeout=30000"))  # 30 second busy timeout
                logger.info("SQLite WAL mode enabled for improved concurrency")

        # Run automatic migrations to ensure schema is up to date
        try:
            from backend.database.migrations import run_migrations

            migration_results = await run_migrations(self.engine)
            if migration_results["columns_added"]:
                logger.info(
                    f"Database migrations applied: {migration_results['columns_added']}"
                )
        except Exception as e:
            logger.warning(f"Migration check failed (non-critical): {e}")

        logger.info(f"Database connected database_url={database_url}")

    async def disconnect(self) -> None:
        """Close database connection."""
        if self.engine:
            await self.engine.dispose()
            self.engine = None
            self.session_factory = None
            logger.info("Database disconnected")

    @asynccontextmanager
    async def get_session(self) -> AsyncGenerator[AsyncSession, None]:
        """
        Get async database session with automatic cleanup.

        Yields:
            AsyncSession: Database session
        """
        if not self.session_factory:
            raise RuntimeError("Database not connected")

        async with self.session_factory() as session:
            try:
                yield session
            except Exception:
                await session.rollback()
                raise
            finally:
                await session.close()

    async def health_check(self) -> str:
        """
        Check database connectivity.

        Returns:
            str: Health status ("healthy" or "unhealthy")
        """
        if not self.engine:
            return "unhealthy"

        try:
            async with self.engine.begin() as conn:
                result = await conn.execute(text("SELECT 1"))
                if result.scalar() == 1:
                    return "healthy"
            return "unhealthy"
        except Exception as e:
            logger.error(f"Database health check failed: {str(e)}")
            return "unhealthy"


# Global database manager instance
database_manager = DatabaseManager()


async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    """
    Dependency injection function for FastAPI routes.

    Provides async database session for request handlers.

    Yields:
        AsyncSession: Database session
    """
    async with database_manager.get_session() as session:
        yield session
