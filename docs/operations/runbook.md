# Gator AI Platform - Operations Runbook

> Production operations guide for system administrators and operators

**Version**: 1.0
**Last Updated**: November 30, 2024

---

## Table of Contents

1. [System Overview](#system-overview)
2. [Health Monitoring](#health-monitoring)
3. [Scheduled Tasks](#scheduled-tasks)
4. [Circuit Breakers](#circuit-breakers)
5. [Troubleshooting](#troubleshooting)
6. [Common Operations](#common-operations)

---

## System Overview

### Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Gator AI Platform                         │
├─────────────────────────────────────────────────────────────┤
│  FastAPI Application (main.py)                               │
│  ├── API Routes (36 modules)                                 │
│  ├── Services (55+ classes)                                  │
│  ├── Background Tasks (scheduled_tasks.py)                   │
│  └── Middleware (logging, rate limiting, CORS)               │
├─────────────────────────────────────────────────────────────┤
│  External Dependencies                                       │
│  ├── PostgreSQL/SQLite (database)                            │
│  ├── Redis (cache, OAuth state)                              │
│  ├── Ollama (AI model serving)                               │
│  └── GPU (optional, for local inference)                     │
└─────────────────────────────────────────────────────────────┘
```

### Key Endpoints

| Endpoint | Description |
|----------|-------------|
| `/health` | Basic health check |
| `/health/detailed` | Comprehensive health with all services |
| `/health/circuit-breakers` | Circuit breaker status |
| `/health/scheduled-tasks` | Background task status |
| `/admin` | Admin dashboard |
| `/docs` | API documentation (debug mode only) |

---

## Health Monitoring

### Quick Health Check

```bash
# Basic health
curl http://localhost:8000/health

# Expected response:
# {"status": "healthy", "database": "healthy", "timestamp": "..."}
```

### Detailed Health Check

```bash
# All services
curl http://localhost:8000/health/detailed
```

Response includes:
- Database connectivity
- Redis cache status
- Ollama AI service
- Filesystem access

### Kubernetes Probes

| Probe | Endpoint | Purpose |
|-------|----------|---------|
| Liveness | `/health/live` | Restart unhealthy pods |
| Readiness | `/health/ready` | Control traffic routing |

---

## Scheduled Tasks

### Active Tasks

| Task | Interval | Purpose |
|------|----------|---------|
| `memory_consolidation` | 1 hour | Promote memories to long-term storage |
| `oauth_state_cleanup` | 10 minutes | Clean expired OAuth states |
| `adaptive_weight_adjustment` | 6 hours | Adjust learning weights |
| `health_check` | 5 minutes | Check dependency health |

### Monitoring Tasks

```bash
# View task status
curl http://localhost:8000/health/scheduled-tasks
```

### Enable/Disable Tasks

```bash
# Disable a task
curl -X POST "http://localhost:8000/health/scheduled-tasks/memory_consolidation/toggle?enabled=false"

# Enable a task
curl -X POST "http://localhost:8000/health/scheduled-tasks/memory_consolidation/toggle?enabled=true"
```

---

## Circuit Breakers

### Understanding States

| State | Meaning | Action |
|-------|---------|--------|
| `closed` | Normal operation | Requests flow through |
| `open` | Service failing | Requests rejected |
| `half_open` | Testing recovery | Limited requests |

### View Status

```bash
curl http://localhost:8000/health/circuit-breakers
```

### Reset Circuit Breakers

After fixing the underlying issue:

```bash
curl -X POST http://localhost:8000/health/circuit-breakers/reset
```

### Configuration

Environment variables:
```bash
GATOR_CIRCUIT_BREAKER_FAILURE_THRESHOLD=5  # Failures before opening
GATOR_CIRCUIT_BREAKER_RECOVERY_TIMEOUT=60  # Seconds before testing
```

---

## Troubleshooting

### Common Issues

#### Database Connection Failure

**Symptoms**: 500 errors, health check shows database unhealthy

**Resolution**:
1. Check database server is running
2. Verify `DATABASE_URL` environment variable
3. Check network connectivity
4. Review database logs

```bash
# Test database connection
curl http://localhost:8000/health/database
```

#### Redis Connection Failure

**Symptoms**: OAuth issues, cache misses, warnings in logs

**Resolution**:
1. Check Redis server is running
2. Verify `REDIS_URL` environment variable
3. Application falls back to in-memory storage

```bash
# Test Redis connection
curl http://localhost:8000/health/redis
```

#### Ollama Service Unavailable

**Symptoms**: AI generation failures, circuit breaker open

**Resolution**:
1. Check Ollama is running: `ollama list`
2. Verify `OLLAMA_BASE_URL` setting
3. Check GPU availability if using local inference
4. Reset circuit breaker after recovery

```bash
# Test Ollama connection
curl http://localhost:8000/health/ollama

# Reset after fixing
curl -X POST http://localhost:8000/health/circuit-breakers/reset
```

#### High Memory Usage

**Symptoms**: Slow responses, OOM errors

**Resolution**:
1. Check memory consolidation is running
2. Review cache sizes
3. Consider increasing memory limits
4. Trigger manual consolidation

### Log Analysis

#### Finding Correlation IDs

All logs include correlation IDs for tracing:
```
2024-11-30 12:00:00 - service - INFO - [req=abc12345 cor=def67890] Processing request
```

To trace a request:
```bash
grep "cor=<correlation_id>" /var/log/gator/*.log
```

#### Log Levels

| Level | When to Use |
|-------|-------------|
| ERROR | Failures requiring attention |
| WARNING | Degraded operation |
| INFO | Normal operations |
| DEBUG | Detailed debugging |

---

## Common Operations

### Restart Application

```bash
# Systemd
sudo systemctl restart gator

# Docker
docker-compose restart gator
```

### Check Configuration

```bash
# View effective settings
curl http://localhost:8000/api/v1/settings/current
```

### Manual Memory Consolidation

If automatic consolidation is disabled:

```python
# Python script
from backend.services.acd_memory_system import ACDMemorySystem
# ... execute consolidation
```

### Database Maintenance

```sql
-- Analyze query performance
EXPLAIN ANALYZE SELECT * FROM acd_contexts WHERE ai_state = 'PROCESSING';

-- Check index usage
SELECT * FROM pg_stat_user_indexes WHERE relname = 'acd_contexts';
```

### Backup Procedures

```bash
# Database backup
pg_dump gator_db > backup_$(date +%Y%m%d).sql

# Redis backup (if persistent)
redis-cli BGSAVE
```

---

## Environment Variables

### Required

| Variable | Description | Default |
|----------|-------------|---------|
| `DATABASE_URL` | Database connection string | sqlite:///./gator.db |
| `SECRET_KEY` | JWT signing key | - |

### Optional

| Variable | Description | Default |
|----------|-------------|---------|
| `REDIS_URL` | Redis connection string | redis://localhost:6379 |
| `OLLAMA_BASE_URL` | Ollama API URL | http://localhost:11434 |
| `GATOR_DEBUG` | Enable debug mode | false |
| `GATOR_ACD_ENABLED` | Enable ACD system | true |

### Timeouts

| Variable | Description | Default |
|----------|-------------|---------|
| `GATOR_OLLAMA_CONNECT_TIMEOUT` | Ollama connection timeout | 5.0 |
| `GATOR_OLLAMA_GENERATE_TIMEOUT` | Generation timeout | 60.0 |
| `GATOR_HTTP_CLIENT_TIMEOUT` | HTTP client timeout | 30.0 |

---

## Contact & Escalation

For issues not covered by this runbook:

1. Check GitHub Issues
2. Review application logs
3. Contact development team

---

**Remember: Gator don't play no shit. Monitor proactively.**
