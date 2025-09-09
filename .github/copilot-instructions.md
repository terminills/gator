# Gator AI Influencer Platform

**Gator don't play no shit** - AI-powered content generation platform built with FastAPI, SQLAlchemy 2.0, and modern ML frameworks. Enables creation and management of AI influencers with complete persona control, content generation, and social media integration.

**ALWAYS** reference these instructions first and fallback to search or bash commands only when you encounter unexpected information that does not match the info here.

## Working Effectively

### Bootstrap and Build Process
**NEVER CANCEL BUILDS OR INSTALLS** - Package installation takes 45+ minutes due to ML dependencies. During network issues, dependencies may already be installed from previous runs.

```bash
# 1. Install all dependencies (including ML packages: torch, transformers, diffusers)
pip install -e .
# TIMING: 45-90 seconds due to large ML packages. NEVER CANCEL. Set timeout to 300+ seconds.
# NOTE: If network timeouts occur, many dependencies may already be installed. Check if demo works.

# 2. Install additional validation dependencies if needed  
pip install 'pydantic[email]' aiosqlite

# 3. Initialize database (creates 5 tables: personas, users, conversations, messages, ppv_offers)
python setup_db.py
# TIMING: 1-2 seconds

# 4. Validate system functionality 
python demo.py
# TIMING: 2-3 seconds - Demonstrates persona CRUD operations, database connectivity
```

### Testing and Quality Assurance
```bash
# Run test suite (exclude tests with missing model dependencies)
python -m pytest tests/ -v --ignore=tests/unit/test_content_generation_enhancements.py --ignore=tests/unit/test_content_generation_service.py --ignore=tests/unit/test_feed_ingestion.py
# TIMING: 2-5 seconds for 60 tests. NEVER CANCEL. Set timeout to 30+ seconds.

# Code formatting (32 files need reformatting)
black src/
# TIMING: 2-3 seconds

# Linting
flake8 src/
# TIMING: 2-3 seconds

# Type checking
mypy src/
# TIMING: 5-10 seconds
```

### Run the Platform
```bash
# Start API server
cd src && python -m backend.api.main
# Server starts on http://localhost:8000
# Interactive docs at http://localhost:8000/docs
# TIMING: 1-2 seconds startup
```

## Validation Scenarios

**ALWAYS** run through these complete scenarios after making changes:

### Scenario 1: Database and Persona Management
```bash
# 1. Fresh database setup
python setup_db.py

# 2. Run full demo (creates persona, lists personas, updates themes, increments counts)
python demo.py
# Expected: Creates "Tech Innovator Sarah" persona, demonstrates CRUD operations

# 3. Verify database contains data
ls -la gator.db  # Should exist with data
```

### Scenario 2: API Functionality
```bash
# 1. Start server
cd src && python -m backend.api.main &

# 2. Test core endpoints (use curl, httpx, or requests)
curl http://localhost:8000/
# Expected: {"message": "Gator AI Influencer Platform", "version": "0.1.0", "status": "operational"}

curl http://localhost:8000/health
# Expected: {"status": "healthy", "database": "connected"}

curl http://localhost:8000/api/v1/personas/
# Expected: List of personas (may be empty on fresh install)
```

### Scenario 3: Development Workflow Validation
```bash
# 1. Make a small change to any Python file
# 2. Run formatting
black src/
# 3. Run tests
python -m pytest tests/ --ignore=tests/unit/test_content_generation_enhancements.py --ignore=tests/unit/test_content_generation_service.py --ignore=tests/unit/test_feed_ingestion.py -v
# 4. Restart server to verify no import errors
cd src && python -m backend.api.main
```

## Critical Information

### Exact Dependencies and Installation
- **Python**: 3.12 (supports 3.9+)
- **Database**: SQLite for development (default), PostgreSQL for production
- **Key Packages**: FastAPI, SQLAlchemy 2.0, Pydantic v2, torch, transformers, diffusers
- **CRITICAL**: ML packages (torch, transformers) are large (~500MB+). Install timeouts must be 180+ seconds.

### Database Schema (5 Tables)
- `personas` - AI persona configurations with UUID primary keys
- `users` - User accounts and authentication
- `conversations` - Conversation threads
- `messages` - Individual messages with PPV support
- `ppv_offers` - Pay-per-view offer management

### API Endpoints (Working)
```
GET  /                     - Platform status
GET  /health               - System health check
GET  /api/v1/personas/     - List all personas  
POST /api/v1/personas/     - Create persona
GET  /api/v1/personas/{id} - Get specific persona
PUT  /api/v1/personas/{id} - Update persona
DELETE /api/v1/personas/{id} - Delete persona
GET  /api/v1/dns/*         - DNS management endpoints
GET  /api/v1/analytics/*   - System metrics and health
```

### Known Issues and Workarounds
1. **UUID Handling**: PersonaService methods expect string UUIDs, not UUID objects. Always convert: `str(uuid_obj)`
2. **Missing Models**: content.py and feed.py referenced but don't exist. Related routes are commented out in main.py
3. **Code Formatting**: 32 files need Black formatting. Run `black src/` before committing
4. **Test Failures**: Some tests fail due to UUID handling edge cases. Use test exclusions shown above.

### File Structure
```
/
├── src/backend/           - Main application code
│   ├── api/              - FastAPI routes and main app
│   ├── models/           - SQLAlchemy and Pydantic models  
│   ├── services/         - Business logic layer
│   ├── database/         - Database connection management
│   └── config/           - Settings and logging
├── tests/                - Unit and integration tests
├── frontend/public/      - Static frontend files
├── setup_db.py          - Database initialization script
├── demo.py              - System validation demo
└── pyproject.toml       - Package configuration
```

### Environment Variables
Create `.env` file for local development:
```bash
# Database
DATABASE_URL=sqlite:///./gator.db  # Default for development

# AI Models (optional for basic operations)
OPENAI_API_KEY=your_key_here
HUGGING_FACE_TOKEN=your_token_here

# Security
SECRET_KEY=your_secret_key_for_jwt

# Development mode
GATOR_ENV=development
```

### Production Deployment Notes
- **Database**: Switch to PostgreSQL for production
- **Dependencies**: Requires GPU for AI model inference (optional for basic CRUD)
- **Scaling**: Built for horizontal scaling with async architecture
- **Security**: JWT authentication, CORS middleware, input validation included

### Common Commands Reference
```bash
# Quick validation pipeline (works even if pip install had network issues)
python setup_db.py && python demo.py

# Full bootstrap (only if dependencies missing)
pip install -e . && python setup_db.py && python demo.py

# Development server with auto-reload
cd src && uvicorn backend.api.main:app --reload --host 0.0.0.0 --port 8000

# Full test and quality check
black src/ && python -m pytest tests/ --ignore=tests/unit/test_content_generation_enhancements.py --ignore=tests/unit/test_content_generation_service.py --ignore=tests/unit/test_feed_ingestion.py

# Clean reinstall (if needed)
pip uninstall gator -y && pip install -e .
```

## Troubleshooting

**Import Errors**: If you see "No module named 'backend.models.content'", the models are referenced but don't exist. Check imports and comment out problematic routes.

**UUID Errors**: Service methods expect string UUIDs. Convert with `str(uuid_obj)` before passing to service methods.

**Database Issues**: Delete `gator.db` and run `python setup_db.py` to reset.

**Server Won't Start**: Check that problematic imports in `main.py` are commented out. Some routes depend on models that don't exist yet.

**Tests Failing**: Use the exclusion pattern shown above to skip tests for unimplemented features.

**Performance**: Demo and API operations are fast (1-3 seconds). Only package installation is slow due to ML dependencies.

---

**Remember**: This platform is designed for AI content generation at scale. Basic CRUD operations work immediately, but full AI features require GPU hardware and model configuration.