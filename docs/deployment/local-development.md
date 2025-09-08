# Getting Started with Gator Development

This guide will help you set up your development environment for the Gator AI Influencer Platform.

## Prerequisites

- Python 3.9+ 
- Docker and Docker Compose
- Git
- NVIDIA GPU (recommended for AI model inference)
- 8GB+ RAM
- 20GB+ free disk space

## Quick Start

1. **Clone the repository**
   ```bash
   git clone https://github.com/terminills/gator.git
   cd gator
   ```

2. **Set up environment variables**
   ```bash
   cp .env.template .env
   # Edit .env with your configuration
   nano .env
   ```

3. **Create Python virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

4. **Install dependencies**
   ```bash
   pip install -e .[dev]
   ```

5. **Set up pre-commit hooks**
   ```bash
   pre-commit install
   ```

6. **Start development services**
   ```bash
   docker-compose up -d postgres redis
   ```

7. **Run database migrations**
   ```bash
   # When migration files exist
   alembic upgrade head
   ```

8. **Start the development server**
   ```bash
   uvicorn src.backend.api.main:app --reload --port 8000
   ```

## Development Environment

### Database Setup

The development environment uses PostgreSQL. Connection details:
- Host: localhost
- Port: 5432
- Database: gator_dev
- Username: gator_user
- Password: (set in .env)

### AI Models Setup

Place your AI models in the `models/` directory:
```
models/
├── stable-diffusion/
├── content-moderation/
└── bias-detection/
```

### Testing

Run the test suite:
```bash
# Unit tests
pytest tests/unit/

# Integration tests  
pytest tests/integration/

# All tests with coverage
pytest --cov=src tests/
```

### Code Quality

The project uses several tools for code quality:
```bash
# Format code
black src/ tests/

# Sort imports
isort src/ tests/

# Lint code
flake8 src/ tests/

# Type checking
mypy src/

# Security scan
bandit -r src/
```

## Project Structure

```
gator/
├── src/                    # Source code
│   ├── backend/            # Backend services
│   ├── frontend/           # Frontend application
│   └── ai/                 # AI/ML components
├── tests/                  # Test suites
├── docs/                   # Documentation
├── scripts/                # Utility scripts
└── config/                 # Configuration files
```

## Next Steps

1. Review the [BEST_PRACTICES.md](BEST_PRACTICES.md) for comprehensive development guidelines
2. Check out [DEVELOPMENT_WORKFLOW.md](DEVELOPMENT_WORKFLOW.md) for the development process
3. Read [COPILOT_INTEGRATION.md](COPILOT_INTEGRATION.md) for effective Copilot usage
4. Understand [SECURITY_ETHICS.md](SECURITY_ETHICS.md) for security and ethical considerations

## Getting Help

- Open an issue for bug reports or feature requests
- Check existing documentation in the `docs/` directory
- Review the codebase for examples and patterns

Happy coding! 🚀