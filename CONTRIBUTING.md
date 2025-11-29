# Contributing to Gator AI Platform

Thank you for your interest in contributing to the Gator AI Influencer Platform! This document provides guidelines and information for contributors.

## 🦎 Code of Conduct

By participating in this project, you agree to maintain a respectful and inclusive environment. We welcome contributions from everyone.

## 📋 How to Contribute

### Reporting Issues

1. **Check existing issues** - Search the issue tracker to avoid duplicates
2. **Use issue templates** - Follow the provided templates when available
3. **Provide details** - Include:
   - Clear description of the problem
   - Steps to reproduce
   - Expected vs actual behavior
   - Environment details (OS, Python version, etc.)

### Submitting Pull Requests

1. **Fork the repository** and create your branch from `main`
2. **Follow coding standards** (see below)
3. **Write tests** for new functionality
4. **Update documentation** if needed
5. **Ensure tests pass** before submitting
6. **Keep PRs focused** - One feature/fix per PR

## 🛠️ Development Setup

### Prerequisites

- Python 3.9+ (3.12 recommended)
- pip or uv package manager
- Git

### Quick Start

```bash
# Clone your fork
git clone https://github.com/YOUR_USERNAME/gator.git
cd gator

# Install dependencies
pip install -e .

# Setup database
python setup_db.py

# Verify installation
python demo.py

# Run tests
python -m pytest tests/ -v
```

### Running the Development Server

```bash
cd src && python -m backend.api.main
# Visit http://localhost:8000/docs for API documentation
```

## 📝 Coding Standards

### Python Style

- Follow [PEP 8](https://pep8.org/) style guidelines
- Use [Black](https://github.com/psf/black) for code formatting
- Use [isort](https://pycqa.github.io/isort/) for import sorting
- Use type hints for function signatures

```bash
# Format code
black src/

# Sort imports
isort src/

# Lint code
flake8 src/
```

### Type Hints

Always use type hints for public APIs:

```python
async def generate_content(
    self,
    request: GenerationRequest,
    persona: Optional[PersonaModel] = None,
) -> ContentResponse:
    pass
```

### Documentation

- Document public functions and classes with docstrings
- Follow Google-style docstrings
- Update relevant documentation files when changing functionality

### Testing

- Write tests for new features
- Maintain existing test coverage
- Use pytest for testing

```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test file
python -m pytest tests/unit/test_persona.py -v

# Run with coverage
python -m pytest tests/ --cov=src
```

## 📂 Project Structure

```
gator/
├── src/backend/           # Main application code
│   ├── api/              # FastAPI routes
│   ├── models/           # SQLAlchemy & Pydantic models
│   ├── services/         # Business logic
│   ├── database/         # Database management
│   └── config/           # Configuration
├── tests/                # Test suite
├── docs/                 # Documentation
│   ├── guides/          # Usage guides
│   ├── architecture/    # System design docs
│   ├── api/             # API documentation
│   ├── integrations/    # Third-party integrations
│   └── reference/       # Technical reference
├── frontend/            # Frontend assets
└── plugins/             # Plugin system
```

## 🔀 Branch Naming

Use descriptive branch names:

- `feature/description` - New features
- `fix/description` - Bug fixes
- `docs/description` - Documentation updates
- `refactor/description` - Code refactoring
- `test/description` - Test additions/updates

## 📦 Commit Messages

Follow conventional commit format:

```
type(scope): description

[optional body]

[optional footer]
```

Types:
- `feat` - New feature
- `fix` - Bug fix
- `docs` - Documentation
- `style` - Formatting
- `refactor` - Code restructuring
- `test` - Adding tests
- `chore` - Maintenance

Examples:
```
feat(persona): add appearance locking feature
fix(api): resolve UUID serialization issue
docs(readme): update installation instructions
```

## 🔍 Code Review Process

1. All PRs require at least one review
2. Address review feedback promptly
3. Keep discussions constructive
4. Squash commits before merge if requested

## 📖 Documentation

When contributing documentation:

1. Place files in appropriate `docs/` subdirectory
2. Update `docs/README.md` index if adding new files
3. Use clear, concise language
4. Include code examples where helpful
5. Keep formatting consistent with existing docs

## ❓ Questions?

- Check existing documentation in `docs/`
- Review the [Improvement Guide](IMPROVEMENT_GUIDE.md) for roadmap
- Open an issue for discussion

---

**Remember: Gator don't play no shit.** Let's build something great together! 🦎
