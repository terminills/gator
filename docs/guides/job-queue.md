# Job Queue Implementation with Celery

## Overview

The Gator platform now includes a production-ready job queue system using Celery and Redis. This enables:
- Scheduled content publishing to social media
- Automated database and content backups
- Background task processing
- Distributed task execution across multiple workers

## Architecture

```
┌─────────────────┐      ┌─────────────┐      ┌──────────────────┐
│   FastAPI App   │─────▶│   Redis     │◀─────│  Celery Workers  │
│                 │      │   (Broker)  │      │                  │
└─────────────────┘      └─────────────┘      └──────────────────┘
        │                                              │
        │                                              │
        ▼                                              ▼
┌─────────────────┐                          ┌──────────────────┐
│   Database      │                          │  Task Results    │
│   (SQLite/PG)   │                          │  (Redis)         │
└─────────────────┘                          └──────────────────┘
```

## Components

### 1. Celery App Configuration
**File**: `src/backend/celery_app.py`

Configures the Celery application with:
- Redis broker for task queue
- Redis result backend for task status
- Periodic task scheduling (Celery Beat)
- Task time limits and retry policies

### 2. Social Media Tasks
**File**: `src/backend/tasks/social_media_tasks.py`

Tasks for content publishing:
- `publish_scheduled_post`: Publish a single scheduled post
- `process_scheduled_posts`: Process all posts due for publishing (runs every minute)
- `cleanup_old_tasks`: Clean up completed task results (runs daily)
- `batch_publish_content`: Publish multiple pieces of content in batch

### 3. Backup Tasks
**File**: `src/backend/tasks/backup_tasks.py`

Tasks for automated backups:
- `create_automated_backup`: Create daily backup of database and content (runs at 2:30 AM)
- `cleanup_old_backups`: Remove backups older than retention period
- `restore_backup`: Restore from a backup

### 4. Updated Social Media Service
**File**: `src/backend/services/social_media_service.py`

The `schedule_post` method now uses Celery's `apply_async` with ETA (estimated time of arrival) to schedule tasks for future execution.

## Installation & Setup

### 1. Install Redis

**Ubuntu/Debian**:
```bash
sudo apt-get update
sudo apt-get install redis-server
sudo systemctl start redis
sudo systemctl enable redis
```

**macOS**:
```bash
brew install redis
brew services start redis
```

**Docker**:
```bash
docker run -d -p 6379:6379 redis:latest
```

### 2. Configure Environment

Add to `.env` file:
```bash
REDIS_URL=redis://localhost:6379/0
CELERY_BROKER_URL=redis://localhost:6379/0
CELERY_RESULT_BACKEND=redis://localhost:6379/0
BACKUP_DIR=/backups
BACKUP_RETENTION_DAYS=30
CONTENT_STORAGE_PATH=generated_content
```

### 3. Start Celery Workers

**Development (single worker)**:
```bash
cd src
celery -A backend.celery_app worker --loglevel=info
```

**Production (multiple workers)**:
```bash
cd src
celery -A backend.celery_app worker --loglevel=info --concurrency=4
```

### 4. Start Celery Beat (Scheduler)

For periodic tasks like automated backups:
```bash
cd src
celery -A backend.celery_app beat --loglevel=info
```

### 5. Monitor Tasks (Optional)

**Flower (Web UI)**:
```bash
pip install flower
cd src
celery -A backend.celery_app flower
# Visit http://localhost:5555
```

## Usage Examples

### Schedule a Social Media Post

```python
from backend.services.social_media_service import SocialMediaService, PostRequest, PlatformType
from datetime import datetime, timedelta

# Create request with future schedule time
request = PostRequest(
    content_id="content-uuid-here",
    platforms=[PlatformType.INSTAGRAM, PlatformType.TWITTER],
    caption="Check out this amazing content!",
    hashtags=["AI", "ContentCreation"],
    schedule_time=datetime.utcnow() + timedelta(hours=2)  # 2 hours from now
)

# Schedule the post
async with database_manager.get_session() as session:
    service = SocialMediaService(session)
    schedule_ids = await service.schedule_post(request)
    print(f"Scheduled posts: {schedule_ids}")
```

### Manually Trigger a Backup

```python
from backend.tasks.backup_tasks import create_automated_backup

# Trigger backup task
task = create_automated_backup.delay()
print(f"Backup task started: {task.id}")

# Wait for result
result = task.get(timeout=300)
print(f"Backup completed: {result}")
```

### Batch Publish Content

```python
from backend.tasks.social_media_tasks import batch_publish_content

content_ids = ["content-1", "content-2", "content-3"]
platforms = ["instagram", "twitter"]

task = batch_publish_content.delay(content_ids, platforms)
result = task.get(timeout=600)
print(f"Published {result['published']} pieces of content")
```

## Periodic Tasks Schedule

| Task | Schedule | Description |
|------|----------|-------------|
| `process_scheduled_posts` | Every 60 seconds | Process posts scheduled for publishing |
| `cleanup_old_tasks` | Daily at 2:00 AM | Clean up completed task results |
| `create_automated_backup` | Daily at 2:30 AM | Create database and content backup |

## Task Monitoring

### Check Task Status

```python
from backend.celery_app import app

# Get task result
result = app.AsyncResult(task_id)
print(f"Status: {result.state}")
print(f"Result: {result.result}")
```

### List Active Tasks

```bash
cd src
celery -A backend.celery_app inspect active
```

### List Scheduled Tasks

```bash
cd src
celery -A backend.celery_app inspect scheduled
```

### Cancel a Task

```python
from backend.celery_app import app

app.control.revoke(task_id, terminate=True)
```

## Production Deployment

### Systemd Service Files

**Celery Worker** (`/etc/systemd/system/celery-worker.service`):
```ini
[Unit]
Description=Celery Worker for Gator Platform
After=network.target redis.service

[Service]
Type=forking
User=gator
Group=gator
WorkingDirectory=/opt/gator/src
Environment="PATH=/opt/gator/venv/bin"
ExecStart=/opt/gator/venv/bin/celery -A backend.celery_app worker \
    --loglevel=info \
    --concurrency=4 \
    --pidfile=/var/run/celery/worker.pid \
    --logfile=/var/log/celery/worker.log
ExecStop=/bin/kill -TERM $MAINPID
Restart=always

[Install]
WantedBy=multi-user.target
```

**Celery Beat** (`/etc/systemd/system/celery-beat.service`):
```ini
[Unit]
Description=Celery Beat for Gator Platform
After=network.target redis.service

[Service]
Type=simple
User=gator
Group=gator
WorkingDirectory=/opt/gator/src
Environment="PATH=/opt/gator/venv/bin"
ExecStart=/opt/gator/venv/bin/celery -A backend.celery_app beat \
    --loglevel=info \
    --pidfile=/var/run/celery/beat.pid \
    --logfile=/var/log/celery/beat.log
Restart=always

[Install]
WantedBy=multi-user.target
```

### Enable and Start Services

```bash
sudo systemctl daemon-reload
sudo systemctl enable celery-worker
sudo systemctl enable celery-beat
sudo systemctl start celery-worker
sudo systemctl start celery-beat
```

## Docker Deployment

**docker-compose.yml**:
```yaml
services:
  redis:
    image: redis:latest
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
  
  celery-worker:
    build: .
    command: celery -A backend.celery_app worker --loglevel=info --concurrency=4
    depends_on:
      - redis
    environment:
      - REDIS_URL=redis://redis:6379/0
    volumes:
      - ./generated_content:/app/generated_content
      - ./backups:/backups
  
  celery-beat:
    build: .
    command: celery -A backend.celery_app beat --loglevel=info
    depends_on:
      - redis
    environment:
      - REDIS_URL=redis://redis:6379/0

volumes:
  redis_data:
```

## Kubernetes Deployment

**celery-worker-deployment.yaml**:
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: celery-worker
spec:
  replicas: 3
  selector:
    matchLabels:
      app: celery-worker
  template:
    metadata:
      labels:
        app: celery-worker
    spec:
      containers:
      - name: celery-worker
        image: gator:latest
        command: ["celery", "-A", "backend.celery_app", "worker", "--loglevel=info"]
        env:
        - name: REDIS_URL
          value: redis://redis-service:6379/0
        volumeMounts:
        - name: content-storage
          mountPath: /app/generated_content
        - name: backup-storage
          mountPath: /backups
      volumes:
      - name: content-storage
        persistentVolumeClaim:
          claimName: content-pvc
      - name: backup-storage
        persistentVolumeClaim:
          claimName: backup-pvc
```

## Performance Tuning

### Worker Concurrency

Adjust based on available CPU cores:
```bash
celery -A backend.celery_app worker --concurrency=8
```

### Task Prefetch

Control how many tasks each worker reserves:
```bash
celery -A backend.celery_app worker --prefetch-multiplier=2
```

### Task Time Limits

Set in `celery_app.py`:
```python
app.conf.update(
    task_time_limit=3600,      # 1 hour hard limit
    task_soft_time_limit=3000  # 50 minutes soft limit
)
```

## Troubleshooting

### Redis Connection Issues

```bash
# Test Redis connection
redis-cli ping

# Check Redis logs
sudo tail -f /var/log/redis/redis-server.log
```

### Worker Not Processing Tasks

```bash
# Check worker status
celery -A backend.celery_app inspect stats

# Restart workers
sudo systemctl restart celery-worker
```

### Task Stuck in Pending

```bash
# Purge all tasks
celery -A backend.celery_app purge

# Or specific queue
celery -A backend.celery_app purge -Q celery
```

### High Memory Usage

- Reduce worker concurrency
- Enable task result expiration
- Use `task_acks_late=True` for better reliability

## Security Considerations

1. **Redis Authentication**: Enable Redis password in production
2. **Network Isolation**: Run Redis on internal network only
3. **Task Serialization**: Use JSON (default) not pickle for security
4. **Result Expiration**: Set `result_expires` to prevent memory leaks
5. **Rate Limiting**: Use Celery rate limits for resource-intensive tasks

## Migration from Legacy System

If you have existing scheduled posts:

1. Create migration script to convert existing schedules to Celery tasks
2. Stop old scheduling system
3. Start Celery workers and beat
4. Verify tasks are processing
5. Monitor for 24-48 hours
6. Remove old scheduling code

## Monitoring & Alerts

### Prometheus Metrics

Install `celery-exporter`:
```bash
pip install celery-exporter
celery-exporter --broker-url=redis://localhost:6379/0
```

### Alert Rules

- Worker down for > 5 minutes
- Task failure rate > 10%
- Queue length > 1000 tasks
- Task execution time > threshold

## Best Practices

1. **Task Idempotency**: Design tasks to be safely retried
2. **Small Tasks**: Break large operations into smaller tasks
3. **Error Handling**: Always catch and log exceptions
4. **Timeouts**: Set appropriate time limits
5. **Monitoring**: Use Flower or Prometheus for visibility
6. **Testing**: Test tasks in isolation before deployment
7. **Documentation**: Document task parameters and behavior

## Future Enhancements

- [ ] Add task prioritization (high/medium/low priority queues)
- [ ] Implement task chaining for complex workflows
- [ ] Add webhook notifications for task completion
- [ ] Implement task retry with exponential backoff
- [ ] Add task result webhooks for third-party integrations
- [ ] Implement distributed task locking for singleton tasks

## Support

For questions or issues with the job queue system:
1. Check the logs: `/var/log/celery/`
2. Review Flower dashboard: http://localhost:5555
3. Open GitHub issue with task ID and error details
4. Contact the development team

---

**Last Updated**: January 2025
**Version**: 1.0.0
**Status**: Production Ready
