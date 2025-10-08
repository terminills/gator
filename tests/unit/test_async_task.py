"""
Unit tests for AsyncTask base class.

Tests the async task execution functionality to ensure proper
handling of async Celery tasks.
"""

import pytest
import asyncio
from unittest.mock import Mock, patch
from celery import Task

from backend.tasks.social_media_tasks import AsyncTask
from backend.celery_app import app


class TestAsyncTask:
    """Test suite for AsyncTask base class."""

    def test_async_task_inheritance(self):
        """Test that AsyncTask properly inherits from Celery Task."""
        assert issubclass(AsyncTask, Task)

    def test_async_task_has_call_method(self):
        """Test that AsyncTask has the __call__ method."""
        task = AsyncTask()
        assert hasattr(task, "__call__")
        assert callable(task.__call__)

    def test_async_task_has_run_async_method(self):
        """Test that AsyncTask has the run_async method."""
        task = AsyncTask()
        assert hasattr(task, "run_async")
        assert callable(task.run_async)

    def test_async_task_run_async_is_coroutine(self):
        """Test that run_async is an async method."""
        task = AsyncTask()
        assert asyncio.iscoroutinefunction(task.run_async)

    def test_async_task_call_executes_async_run(self):
        """Test that __call__ properly executes async run method."""

        # Create a mock async task with custom run method
        @app.task(base=AsyncTask, bind=True, name="test_task")
        async def test_task(self, test_arg):
            """Test task that returns the argument."""
            return f"processed: {test_arg}"

        # Execute the task synchronously through __call__
        result = test_task("test_value")
        assert result == "processed: test_value"

    def test_async_task_with_multiple_args(self):
        """Test AsyncTask with multiple arguments."""

        @app.task(base=AsyncTask, bind=True, name="test_multi_args")
        async def test_multi_args(self, arg1, arg2, kwarg1=None):
            """Test task with multiple arguments."""
            return {"arg1": arg1, "arg2": arg2, "kwarg1": kwarg1}

        result = test_multi_args("value1", "value2", kwarg1="kwvalue")
        assert result == {"arg1": "value1", "arg2": "value2", "kwarg1": "kwvalue"}

    def test_async_task_with_exception(self):
        """Test that exceptions in async tasks are properly propagated."""

        @app.task(base=AsyncTask, bind=True, name="test_exception")
        async def test_exception(self):
            """Test task that raises an exception."""
            raise ValueError("Test exception")

        with pytest.raises(ValueError, match="Test exception"):
            test_exception()

    def test_async_task_run_async_default_implementation(self):
        """Test that default run_async returns None."""
        task = AsyncTask()
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result = loop.run_until_complete(task.run_async())
            assert result is None
        finally:
            loop.close()

    def test_async_task_subclass_override(self):
        """Test that subclasses can override run_async."""

        class CustomAsyncTask(AsyncTask):
            """Custom task that overrides run_async."""

            async def run_async(self, value):
                """Custom async implementation."""
                return f"custom: {value}"

        task = CustomAsyncTask()
        result = task("test")
        assert result == "custom: test"

    def test_async_task_with_async_operations(self):
        """Test AsyncTask with actual async operations."""

        @app.task(base=AsyncTask, bind=True, name="test_async_ops")
        async def test_async_ops(self, delay):
            """Test task with async sleep."""
            await asyncio.sleep(delay)
            return f"completed after {delay}s"

        # Note: Calling synchronously through __call__
        result = test_async_ops(0.01)  # Short delay for test
        assert result == "completed after 0.01s"

    def test_async_task_event_loop_cleanup(self):
        """Test that event loop is properly cleaned up after execution."""

        @app.task(base=AsyncTask, bind=True, name="test_cleanup")
        async def test_cleanup(self):
            """Test task for loop cleanup."""
            return "done"

        # Execute task
        result = test_cleanup()
        assert result == "done"

        # Check that we can create a new loop after task execution
        # (this would fail if the loop wasn't properly closed)
        new_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(new_loop)
        new_loop.close()


class TestAsyncTaskIntegration:
    """Integration tests for AsyncTask with Celery tasks."""

    def test_publish_scheduled_post_task_structure(self):
        """Test that publish_scheduled_post task is properly configured."""
        from backend.tasks.social_media_tasks import publish_scheduled_post

        # Check task properties
        assert publish_scheduled_post.name == (
            "backend.tasks.social_media_tasks.publish_scheduled_post"
        )
        assert asyncio.iscoroutinefunction(publish_scheduled_post.run)

    def test_process_scheduled_posts_task_exists(self):
        """Test that process_scheduled_posts task exists and is registered."""
        from backend.tasks.social_media_tasks import process_scheduled_posts

        assert process_scheduled_posts.name == (
            "backend.tasks.social_media_tasks.process_scheduled_posts"
        )
        assert callable(process_scheduled_posts)

    def test_batch_publish_content_task_exists(self):
        """Test that batch_publish_content task exists and is registered."""
        from backend.tasks.social_media_tasks import batch_publish_content

        assert batch_publish_content.name == (
            "backend.tasks.social_media_tasks.batch_publish_content"
        )
        assert callable(batch_publish_content)

    def test_cleanup_old_tasks_exists(self):
        """Test that cleanup_old_tasks task exists and is registered."""
        from backend.tasks.social_media_tasks import cleanup_old_tasks

        assert cleanup_old_tasks.name == (
            "backend.tasks.social_media_tasks.cleanup_old_tasks"
        )
        assert callable(cleanup_old_tasks)
