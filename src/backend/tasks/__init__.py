"""
Background Tasks Module

Contains Celery tasks for asynchronous operations including:
- Scheduled social media publishing
- Automated backups
- Content generation pipelines
- Analytics processing
"""

from backend.tasks.social_media_tasks import (
    publish_scheduled_post,
    process_scheduled_posts,
    cleanup_old_tasks
)

from backend.tasks.backup_tasks import (
    create_automated_backup,
    cleanup_old_backups
)

__all__ = [
    'publish_scheduled_post',
    'process_scheduled_posts',
    'cleanup_old_tasks',
    'create_automated_backup',
    'cleanup_old_backups'
]
