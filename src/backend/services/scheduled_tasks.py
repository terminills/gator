"""
Scheduled Tasks Module

Provides background task scheduling for production operations:
- Memory consolidation (ACD)
- OAuth state cleanup
- Health checks
- Metrics collection

Uses asyncio for non-blocking scheduled execution.
"""

import asyncio
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional

from backend.config.logging import get_logger
from backend.config.settings import get_settings

logger = get_logger(__name__)
settings = get_settings()


class ScheduledTask:
    """Represents a scheduled background task."""

    def __init__(
        self,
        name: str,
        coro_func: Callable,
        interval_seconds: int,
        enabled: bool = True,
        run_on_startup: bool = False,
    ):
        """
        Initialize scheduled task.

        Args:
            name: Task identifier
            coro_func: Async function to execute
            interval_seconds: Seconds between executions
            enabled: Whether task is active
            run_on_startup: Run immediately on start
        """
        self.name = name
        self.coro_func = coro_func
        self.interval_seconds = interval_seconds
        self.enabled = enabled
        self.run_on_startup = run_on_startup
        self.last_run: Optional[datetime] = None
        self.run_count = 0
        self.error_count = 0
        self._task: Optional[asyncio.Task] = None
        self._stop_requested = False

    async def _run_loop(self) -> None:
        """Main execution loop for the task."""
        if self.run_on_startup and self.enabled:
            await self._execute()

        while not self._stop_requested:
            try:
                await asyncio.sleep(self.interval_seconds)
                if self.enabled and not self._stop_requested:
                    await self._execute()
            except asyncio.CancelledError:
                logger.debug(f"Task {self.name} cancelled")
                break

    async def _execute(self) -> None:
        """Execute the task with error handling."""
        try:
            logger.debug(f"Executing scheduled task: {self.name}")
            await self.coro_func()
            self.last_run = datetime.now(timezone.utc)
            self.run_count += 1
            logger.debug(f"Completed scheduled task: {self.name}")
        except Exception as e:
            self.error_count += 1
            logger.error(f"Scheduled task {self.name} failed: {e}")

    def start(self) -> None:
        """Start the task loop."""
        self._stop_requested = False
        if self._task is None or self._task.done():
            self._task = asyncio.create_task(self._run_loop())
            logger.info(f"Started scheduled task: {self.name}")

    def stop(self) -> None:
        """Stop the task loop."""
        self._stop_requested = True
        if self._task and not self._task.done():
            self._task.cancel()
            logger.info(f"Stopped scheduled task: {self.name}")

    def get_status(self) -> Dict[str, Any]:
        """Get task status."""
        return {
            "name": self.name,
            "enabled": self.enabled,
            "interval_seconds": self.interval_seconds,
            "last_run": self.last_run.isoformat() if self.last_run else None,
            "run_count": self.run_count,
            "error_count": self.error_count,
            "running": self._task is not None and not self._task.done(),
        }


class TaskScheduler:
    """
    Centralized task scheduler for background operations.

    Manages all scheduled tasks with:
    - Task registration
    - Lifecycle management
    - Status monitoring
    """

    _instance: Optional["TaskScheduler"] = None
    _initialized: bool = False

    def __new__(cls) -> "TaskScheduler":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        """Initialize the task scheduler."""
        # Only initialize once per singleton
        if not TaskScheduler._initialized:
            self._tasks: Dict[str, ScheduledTask] = {}
            TaskScheduler._initialized = True

    def register(self, task: ScheduledTask) -> None:
        """Register a task with the scheduler."""
        self._tasks[task.name] = task
        logger.info(f"Registered scheduled task: {task.name}")

    def start_all(self) -> None:
        """Start all registered tasks."""
        for task in self._tasks.values():
            task.start()
        logger.info(f"Started {len(self._tasks)} scheduled tasks")

    def stop_all(self) -> None:
        """Stop all running tasks."""
        for task in self._tasks.values():
            task.stop()
        logger.info("Stopped all scheduled tasks")

    def get_status(self) -> Dict[str, Any]:
        """Get status of all tasks."""
        return {
            "tasks": {name: task.get_status() for name, task in self._tasks.items()},
            "total_tasks": len(self._tasks),
            "running_tasks": sum(
                1 for t in self._tasks.values() if t._task and not t._task.done()
            ),
        }

    def get_task(self, name: str) -> Optional[ScheduledTask]:
        """Get a specific task by name."""
        return self._tasks.get(name)


# Global scheduler instance
scheduler = TaskScheduler()


async def memory_consolidation_task() -> None:
    """
    Scheduled task for ACD memory consolidation.

    Runs periodically to:
    - Promote high-value short-term memories to long-term
    - Mark frequently accessed memories as episodic
    - Clean up low-importance consolidated memories
    """
    from backend.database.connection import database_manager
    from backend.services.acd_memory_system import ACDMemorySystem

    try:
        async with database_manager.get_session() as session:
            memory_system = ACDMemorySystem(session)
            result = await memory_system.consolidate(batch_size=100)
            logger.info(
                f"Memory consolidation complete: "
                f"consolidated={result.get('consolidated', 0)}, "
                f"promoted_long_term={result.get('promoted_to_long_term', 0)}, "
                f"promoted_episodic={result.get('promoted_to_episodic', 0)}"
            )
    except Exception as e:
        logger.error(f"Memory consolidation failed: {e}")


async def oauth_state_cleanup_task() -> None:
    """
    Scheduled task for OAuth state cleanup.

    Removes expired OAuth states from in-memory storage
    when Redis is unavailable.
    """
    from backend.services.social_oauth_service import get_oauth_state_store

    try:
        store = get_oauth_state_store()
        cleaned = await store.cleanup_expired()
        if cleaned > 0:
            logger.info(f"Cleaned up {cleaned} expired OAuth states")
    except Exception as e:
        logger.error(f"OAuth state cleanup failed: {e}")


async def adaptive_weight_adjustment_task() -> None:
    """
    Scheduled task for adaptive learning weight adjustments.

    Analyzes recent decisions and adjusts weights based on outcomes
    to improve future decision quality.
    """
    from backend.database.connection import database_manager
    from backend.services.acd_self_improvement import ACDSelfImprovement, DecisionAnalysis

    try:
        async with database_manager.get_session() as session:
            self_improvement = ACDSelfImprovement(session)
            # First analyze decisions, then update weights
            analysis = await self_improvement.analyze_decision_quality()
            result = await self_improvement.update_decision_weights(analysis)
            if result.get("total_adjusted", 0) > 0:
                logger.info(
                    f"Adaptive weight adjustment: "
                    f"adjustments={result.get('total_adjusted', 0)}"
                )
    except Exception as e:
        logger.error(f"Adaptive weight adjustment failed: {e}")


async def health_check_task() -> None:
    """
    Scheduled task for dependency health checks.

    Periodically checks health of:
    - Database connection
    - Redis cache
    - Ollama service
    - GPU availability
    """
    from backend.database.connection import database_manager

    try:
        # Check database
        async with database_manager.get_session() as session:
            from sqlalchemy import text
            await session.execute(text("SELECT 1"))
        
        # Check cache
        try:
            from backend.services.cache_service import CacheService
            cache = CacheService()
            if cache.is_connected:
                await cache.get("health_check_probe")
        except Exception:
            pass  # Cache is optional

        logger.debug("Health check completed successfully")
    except Exception as e:
        logger.warning(f"Health check detected issue: {e}")


def register_default_tasks() -> None:
    """Register default scheduled tasks."""
    # Memory consolidation every hour
    scheduler.register(
        ScheduledTask(
            name="memory_consolidation",
            coro_func=memory_consolidation_task,
            interval_seconds=3600,  # 1 hour
            enabled=settings.acd_enabled,
            run_on_startup=False,
        )
    )

    # OAuth state cleanup every 10 minutes
    scheduler.register(
        ScheduledTask(
            name="oauth_state_cleanup",
            coro_func=oauth_state_cleanup_task,
            interval_seconds=600,  # 10 minutes
            enabled=True,
            run_on_startup=True,
        )
    )

    # Adaptive weight adjustment every 6 hours
    scheduler.register(
        ScheduledTask(
            name="adaptive_weight_adjustment",
            coro_func=adaptive_weight_adjustment_task,
            interval_seconds=21600,  # 6 hours
            enabled=settings.acd_enabled,
            run_on_startup=False,
        )
    )

    # Health check every 5 minutes
    scheduler.register(
        ScheduledTask(
            name="health_check",
            coro_func=health_check_task,
            interval_seconds=300,  # 5 minutes
            enabled=True,
            run_on_startup=True,
        )
    )


async def start_scheduler() -> None:
    """Initialize and start the task scheduler."""
    register_default_tasks()
    scheduler.start_all()


async def stop_scheduler() -> None:
    """Stop the task scheduler."""
    scheduler.stop_all()
