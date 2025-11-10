# Gator Platform - Multiple Issues Fixed

This document summarizes the fixes applied to address multiple issues in the Gator AI Influencer Platform.

## Issues Resolved

### 1. Plugin System Errors (500 Internal Server Error) ✅

**Problem**: `/api/v1/plugins/installed` and `/api/v1/plugins/marketplace` endpoints were returning 500 errors.

**Root Cause**: Plugin database tables were not being created during database initialization.

**Fix**:
- Added plugin model imports to `setup_db.py`
- Fixed `plugin.py` to use the shared `Base` from `backend.database.connection` instead of creating its own `declarative_base()`
- Plugin tables (`plugins`, `plugin_installations`, `plugin_reviews`) are now created automatically

**Files Changed**:
- `setup_db.py` - Added plugin model imports
- `src/backend/models/plugin.py` - Fixed Base import

**Test**: Run `python setup_db.py` and verify plugin tables are created.

---

### 2. RSS Feed Updates Not Working ✅

**Problem**: RSS feeds like https://rss.politico.com/politics-news.xml weren't automatically updating.

**Root Cause**: No background task system was configured to fetch RSS feeds periodically.

**Fix**:
- Created `src/backend/tasks/rss_feed_tasks.py` with automated feed fetching
- Added `fetch_all_rss_feeds` task that runs every 15 minutes
- Added `cleanup_old_feed_items` task that runs daily to clean up old items
- Integrated tasks into Celery beat schedule

**Files Changed**:
- `src/backend/tasks/rss_feed_tasks.py` (NEW)
- `src/backend/celery_app.py` - Added RSS feed tasks to schedule

**Usage**:
```bash
# Start Celery worker
celery -A backend.celery_app worker --loglevel=info

# Start Celery beat scheduler
celery -A backend.celery_app beat --loglevel=info
```

**Manual Trigger**:
```bash
curl -X POST http://localhost:8000/api/v1/feeds/fetch
```

---

### 3. Content Generation Model Detection ⚠️

**Problem**: Content generation returns "No image generation models available" error.

**Root Cause**: No local models installed AND no cloud API keys configured.

**Current Behavior**:
- System checks for local models in `./models/` directory
- Falls back to cloud APIs (OpenAI, Anthropic) if local models not found
- Raises error only if BOTH local models and cloud APIs are unavailable

**Solution**:
To fix this issue, you need to either:

**Option A - Use Cloud APIs** (Fastest):
```bash
# Add to .env file
OPENAI_API_KEY=sk-your-key-here
# OR
ANTHROPIC_API_KEY=sk-ant-your-key-here
```

**Option B - Install Local Models**:
```bash
# Use the AI models setup endpoint
curl -X POST http://localhost:8000/api/v1/setup/ai-models/install \
  -H "Content-Type: application/json" \
  -d '{"model_names": ["stable-diffusion-xl-base-1.0"], "model_type": "image"}'
```

---

### 4. Settings Page .env File Generation ✅

**Problem**: Settings page couldn't generate .env file when it didn't exist.

**Fix**:
- Enhanced `setup_service.py` to create .env from template automatically
- Falls back to minimal configuration if .env.template doesn't exist
- Supports updating existing .env or creating from scratch

**Files Changed**:
- `src/backend/services/setup_service.py`

**API Endpoint**:
```bash
POST /api/v1/setup/config
```

---

### 5. AI Models Setup Page Not Detecting Installed Models ✅

**Problem**: `/ai-models-setup` page wasn't showing installed voice models or properly verifying installations.

**Fix**:
- Enhanced model detection in `/api/v1/setup/ai-models/status` endpoint
- Now scans actual model directories and checks for:
  - Model files (.safetensors, .bin, .pt)
  - Configuration files (config.json)
  - Calculates actual disk size
  - Validates model integrity
- Supports all model categories: text, image, voice, video, audio

**Files Changed**:
- `src/backend/api/routes/setup.py`

**Test**:
```bash
curl http://localhost:8000/api/v1/setup/ai-models/status | jq '.installed_models'
```

---

### 6. Install/Uninstall Button Toggle ✅

**Problem**: No way to uninstall models once installed; install button should change to uninstall.

**Fix**:
- Added `POST /api/v1/setup/ai-models/uninstall` endpoint
- Endpoint removes model files from disk and updates configuration
- Frontend can now toggle between install/uninstall based on model status

**Files Changed**:
- `src/backend/api/routes/setup.py` - Added `uninstall_model` endpoint

**Usage**:
```bash
curl -X POST http://localhost:8000/api/v1/setup/ai-models/uninstall \
  -H "Content-Type: application/json" \
  -d '{"model_name": "stable-diffusion-xl-base-1.0", "model_category": "image"}'
```

---

### 7. Gator Agent LLM Connection ✅

**Problem**: "Ask Gator - Your AI Help Agent" wasn't connected to any LLM.

**Fix**:
- Enhanced Gator agent with **prioritized LLM cascade**:
  1. **Local models first** (if available) - Free, private, fast
  2. **Cloud APIs** (OpenAI/Anthropic) - If local models unavailable
  3. **Rule-based fallback** - If no LLMs available at all

- Gator maintains its tough, no-nonsense personality regardless of backend
- Integrated with platform's local model infrastructure

**Files Changed**:
- `src/backend/services/gator_agent_service.py`

**Configuration**:
The agent automatically detects available resources:
```bash
# Option 1: Use local models (preferred)
# Install a text generation model first

# Option 2: Use cloud APIs (fallback)
OPENAI_API_KEY=sk-your-key-here
# OR
ANTHROPIC_API_KEY=sk-ant-your-key-here
```

**Test**:
```bash
curl -X POST http://localhost:8000/api/v1/gator-agent/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "How do I create a persona?"}'
```

---

## Verification Checklist

To verify all fixes are working:

```bash
# 1. Database setup
python setup_db.py
# Verify plugin tables are created

# 2. Start server
cd src && python -m backend.api.main

# 3. Test plugin endpoints
curl http://localhost:8000/api/v1/plugins/marketplace
curl http://localhost:8000/api/v1/plugins/installed

# 4. Test RSS feeds (manual trigger)
curl -X POST http://localhost:8000/api/v1/feeds/fetch

# 5. Test AI models status
curl http://localhost:8000/api/v1/setup/ai-models/status

# 6. Test Gator agent
curl -X POST http://localhost:8000/api/v1/gator-agent/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Hello Gator"}'

# 7. Test .env generation
curl -X POST http://localhost:8000/api/v1/setup/config \
  -H "Content-Type: application/json" \
  -d '{"secret_key": "test-secret-key"}'
```

---

## Starting the Full System

To run the complete system with all features:

```bash
# Terminal 1: Start main API server
cd src && python -m backend.api.main

# Terminal 2: Start Celery worker (for background tasks)
celery -A backend.celery_app worker --loglevel=info

# Terminal 3: Start Celery beat (for scheduled tasks like RSS)
celery -A backend.celery_app beat --loglevel=info

# Optional Terminal 4: Monitor Celery with Flower
celery -A backend.celery_app flower
# Access at http://localhost:5555
```

---

## Configuration Files

### Minimal .env for Testing
```bash
# Database
DATABASE_URL=sqlite:///./gator.db

# Security
SECRET_KEY=your_super_secret_key_change_in_production
JWT_SECRET=your_jwt_secret_key

# Application
DEBUG=true
ENVIRONMENT=development
LOG_LEVEL=DEBUG

# Optional: Cloud AI APIs
# OPENAI_API_KEY=sk-your-key-here
# ANTHROPIC_API_KEY=sk-ant-your-key-here
```

### Production .env
See `.env.template` for full configuration options including:
- Database (PostgreSQL recommended)
- AI Models (local + cloud)
- Social Media APIs
- DNS Management (GoDaddy)
- Monitoring (Sentry, Prometheus)
- Email (SMTP)

---

## Notes

1. **RSS Feed Scheduling**: Feeds update every 15 minutes automatically when Celery beat is running
2. **Model Priority**: System always prefers local models over cloud APIs to reduce costs
3. **Plugin Tables**: Must run `python setup_db.py` after pulling these changes
4. **Celery Requirement**: RSS feed auto-updates require Celery worker + beat running
5. **API Keys**: Most features work without API keys; they're only needed for cloud AI services

---

## Future Improvements

- Add model installation progress tracking
- Implement model version management
- Add RSS feed health monitoring dashboard
- Create model recommendation system based on hardware
- Add batch model installation
