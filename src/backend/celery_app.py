"""
Celery Application Configuration

Provides task queue functionality for scheduled content publishing,
background processing, and asynchronous operations.
"""

from celery import Celery
from celery.schedules import crontab
from backend.config.settings import get_settings

settings = get_settings()

# Initialize Celery app
app = Celery(
    "gator",
    broker=(
        settings.REDIS_URL
        if hasattr(settings, "REDIS_URL")
        else "redis://localhost:6379/0"
    ),
    backend=(
        settings.REDIS_URL
        if hasattr(settings, "REDIS_URL")
        else "redis://localhost:6379/0"
    ),
    include=[
        "backend.tasks.social_media_tasks",
        "backend.tasks.backup_tasks",
        "backend.tasks.rss_feed_tasks",
    ],
)

# Celery configuration
app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    task_time_limit=3600,  # 1 hour max per task
    task_soft_time_limit=3000,  # 50 minutes soft limit
    worker_prefetch_multiplier=4,
    worker_max_tasks_per_child=1000,
    result_expires=86400,  # Results expire after 24 hours
    task_acks_late=True,
    task_reject_on_worker_lost=True,
)

# Periodic task schedule
app.conf.beat_schedule = {
    "process-scheduled-posts": {
        "task": "backend.tasks.social_media_tasks.process_scheduled_posts",
        "schedule": 60.0,  # Run every minute
    },
    "fetch-rss-feeds": {
        "task": "backend.tasks.rss_feed_tasks.fetch_all_rss_feeds",
        "schedule": 900.0,  # Run every 15 minutes
    },
    "cleanup-old-tasks": {
        "task": "backend.tasks.social_media_tasks.cleanup_old_tasks",
        "schedule": crontab(hour=2, minute=0),  # Run daily at 2 AM
    },
    "cleanup-old-feed-items": {
        "task": "backend.tasks.rss_feed_tasks.cleanup_old_feed_items",
        "schedule": crontab(hour=3, minute=0),  # Run daily at 3 AM
    },
    "daily-database-backup": {
        "task": "backend.tasks.backup_tasks.create_automated_backup",
        "schedule": crontab(hour=2, minute=30),  # Run daily at 2:30 AM
    },
}

if __name__ == "__main__":
    app.start()
