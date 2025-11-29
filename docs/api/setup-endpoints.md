# Setup Endpoint Feature Documentation

## Overview

The Gator platform now includes a `/setup` endpoint and UI wizard that allows administrators to configure environment variables through the admin panel instead of manually editing the `.env` file.

## Features

### 1. Backend API Endpoints

#### GET `/api/v1/setup/status`
Returns the current configuration status including:
- Whether `.env` file exists
- Path to environment file
- Which configuration sections are configured
- Current configuration values (with sensitive values masked)

**Example Response:**
```json
{
  "env_file_exists": true,
  "env_file_path": "/opt/gator/app/.env",
  "configured_sections": {
    "database": true,
    "ai_models": true,
    "social_media": false,
    "dns": true,
    "security": true
  },
  "current_config": {
    "DATABASE_URL": "postgresql://...",
    "SECRET_KEY": "***CONFIGURED***",
    "ENVIRONMENT": "production"
  }
}
```

#### GET `/api/v1/setup/template`
Returns a structured template of all available configuration options organized by section. Used by the UI to generate the setup form dynamically.

#### POST `/api/v1/setup/config`
Updates environment configuration with validation.

**Request Body:**
```json
{
  "database_url": "postgresql://user:pass@localhost:5432/gator",
  "secret_key": "your-secret-key",
  "jwt_secret": "your-jwt-secret",
  "openai_api_key": "sk-...",
  "environment": "production",
  "debug": false
}
```

**Response:**
```json
{
  "success": true,
  "message": "Configuration updated successfully...",
  "validation": {
    "valid": true,
    "errors": [],
    "warnings": ["SECRET_KEY appears to have a placeholder value"]
  },
  "restart_required": true
}
```

### 2. Setup Service

The `SetupService` class handles all environment file operations:

- **Reads `.env` file** and masks sensitive values
- **Writes updates** to `.env` file, preserving comments and structure
- **Creates from template** if `.env` doesn't exist
- **Validates configuration** before saving
- **Preserves existing values** - only updates provided fields

**Validation includes:**
- Database URL format checking
- Port number range validation
- Detection of placeholder values

### 3. Admin Panel UI

Located in the **Settings** tab of the admin panel at `http://localhost:8000/admin`

**Features:**
- **Configuration Status** - Shows which sections are configured
- **Organized Sections** - Database, Security, AI Models, Social Media, DNS, Application
- **Smart Form** - Auto-populates with existing values, masks sensitive data
- **Real-time Feedback** - Success/error messages with validation warnings
- **Restart Reminder** - Clear indication when restart is needed

**Sections:**
1. **Database Configuration** - PostgreSQL, Redis URLs
2. **Security Configuration** - Secret keys, JWT configuration, encryption
3. **AI Model Configuration** - OpenAI, Anthropic, Hugging Face credentials
4. **Social Media APIs** - Facebook, Instagram, Twitter credentials
5. **DNS Management** - GoDaddy API keys and domain settings
6. **Application Settings** - Environment, debug mode, log level

## Usage

### Using the Admin UI

1. Navigate to `http://localhost:8000/admin`
2. Click on the **Settings** tab
3. View current configuration status
4. Fill in desired configuration values
5. Click **Save Configuration**
6. Restart the application for changes to take effect

### Using the API

```bash
# Check current status
curl http://localhost:8000/api/v1/setup/status

# Update configuration
curl -X POST http://localhost:8000/api/v1/setup/config \
  -H "Content-Type: application/json" \
  -d '{
    "database_url": "postgresql://user:pass@localhost/gator",
    "secret_key": "your-secret-key",
    "environment": "production"
  }'
```

## Security

- **Sensitive values are masked** in API responses (shown as `***CONFIGURED***`)
- **Validation** prevents invalid configurations
- **No plaintext secrets** in logs or API responses
- **Existing values preserved** - only specified fields are updated

## Configuration Management

### Supported Environment Variables

All variables from `.env.template` are supported, including:

**Database:**
- `DATABASE_URL`
- `DATABASE_TEST_URL`
- `REDIS_URL`

**AI Models:**
- `AI_MODEL_PATH`
- `OPENAI_API_KEY`
- `ANTHROPIC_API_KEY`
- `ELEVENLABS_API_KEY`
- `HUGGING_FACE_TOKEN`
- `STABLE_DIFFUSION_MODEL`
- `CONTENT_MODERATION_MODEL`

**Security:**
- `SECRET_KEY`
- `JWT_SECRET`
- `JWT_ALGORITHM`
- `JWT_EXPIRATION_HOURS`
- `ENCRYPTION_KEY`

**Social Media:**
- `FACEBOOK_API_KEY`, `FACEBOOK_API_SECRET`
- `INSTAGRAM_API_KEY`, `INSTAGRAM_API_SECRET`
- `TWITTER_API_KEY`, `TWITTER_API_SECRET`

**DNS Management:**
- `GODADDY_API_KEY`
- `GODADDY_API_SECRET`
- `GODADDY_ENVIRONMENT`
- `DEFAULT_DOMAIN`

**Application:**
- `DEBUG`
- `ENVIRONMENT`
- `LOG_LEVEL`
- `MAX_CONTENT_GENERATION_CONCURRENT`
- `CONTENT_CACHE_TTL_SECONDS`

And many more...

## Validation Rules

- **Database URL** must start with `postgresql://` or `sqlite:///`
- **SMTP Port** must be between 1 and 65535
- **Placeholder values** trigger warnings (values starting with `your_`)
- **Empty requests** are rejected

## Testing

Run the setup API tests:
```bash
pytest tests/integration/test_setup_api.py -v
```

All 5 tests should pass:
- ✅ Get setup status
- ✅ Get configuration template
- ✅ Validation with invalid port
- ✅ Successful configuration update
- ✅ Empty configuration rejection

## Files Modified/Created

### Created:
- `src/backend/services/setup_service.py` - Setup service implementation
- `src/backend/api/routes/setup.py` - API routes
- `tests/integration/test_setup_api.py` - Integration tests

### Modified:
- `src/backend/api/main.py` - Added setup router
- `admin.html` - Added setup wizard UI

## Restart Requirement

**Important:** After updating configuration via the setup endpoint, you must **restart the application** for changes to take effect.

The settings are loaded at startup using `pydantic-settings`, which reads from the `.env` file. Runtime changes to the file don't automatically reload settings.

### Restart Methods:

**Using systemd (production):**
```bash
sudo systemctl restart gator
```

**Using the server setup script:**
```bash
sudo systemctl restart gator.service
```

**Development server:**
Stop and restart the uvicorn process.

## Troubleshooting

**Configuration not taking effect?**
- Ensure you restarted the application
- Check file permissions on `.env`
- Verify `.env` is in the correct location (project root)

**Validation errors?**
- Check the validation messages for specific issues
- Ensure database URLs are properly formatted
- Verify port numbers are in valid ranges

**Can't see status updates?**
- Click "Refresh Status" button
- Check browser console for API errors
- Verify the API server is running

## Future Enhancements

Potential improvements for future versions:
- Hot-reload configuration without restart
- Configuration backup/restore
- Import/export configuration as JSON
- Configuration templates for different environments
- Validation for more field types
- Support for environment-specific overrides
