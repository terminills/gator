# AI Model Installation Enhancement - Implementation Summary

## Overview
Added functionality to install and enable/disable AI models directly from the web interface at `http://127.0.0.1:8000/ai-models-setup`.

## Changes Made

### 1. Backend API Endpoints (src/backend/api/routes/setup.py)

#### New Endpoints:

**GET /api/v1/setup/ai-models/recommendations**
- Returns structured model recommendations based on system capabilities
- Groups models into: installable, requires_upgrade, api_only
- Includes system information and hardware capabilities

**POST /api/v1/setup/ai-models/install**
- Installs specified AI models
- Accepts: `{model_names: ["model-name"], model_type: "text|image|voice"}`
- Returns installation progress and status
- Timeout: 5 minutes for model downloads

**POST /api/v1/setup/ai-models/enable**
- Enables or disables installed models
- Accepts: `{model_name: "model-name", enabled: true|false}`
- Updates model_config.json with enabled status

### 2. Frontend UI Changes (ai_models_setup.html)

#### Enhanced "Available Models for Installation" Section:
- Now displays structured recommendations from the backend
- Shows two categories:
  - **✅ Compatible Models (Can Install)** - with Install buttons
  - **⚠️ Models Requiring Hardware Upgrade** - with requirements listed

#### Model Cards Now Include:
- Model name and category badge
- Description
- Size requirements (disk, RAM, GPU memory)
- **"📥 Install Model" button** for compatible models
- Reason why upgrade is needed for incompatible models

#### Installed Models Section Enhanced:
- Added enable/disable checkbox toggle for each installed model
- Checkbox state is persisted to backend via `/api/v1/setup/ai-models/enable`
- Visual feedback with success notifications

#### Installation Flow:
1. User clicks "📥 Install Model" button
2. Button changes to "⏳ Installing..." and disables
3. Backend downloads and configures the model
4. On success: Button shows "✅ Installed!" and page reloads
5. Model appears in "Installed Models" section with enable toggle

### 3. Setup Script Enhancement (setup_ai_models.py)

#### Implemented `--install` Argument:
```bash
python setup_ai_models.py --install model-name-1 model-name-2
```

Features:
- Validates that requested models are installable on current system
- Shows warnings for models that require hardware upgrade
- Groups models by category (text, image, voice)
- Installs dependencies first
- Downloads and configures models
- Creates/updates model_config.json

#### Improved Error Handling:
- Gracefully handles torch installation failures
- Uses mock torch module when GPU libraries unavailable
- Continues operation in CPU-only mode
- Provides clear error messages

## User Experience Flow

### Scenario 1: User with Compatible Hardware

1. User navigates to `http://127.0.0.1:8000/ai-models-setup`
2. System detects: ROCm GPU with 16GB memory, 37.9GB RAM
3. Page shows:
   - **System Information**: GPU status, Python version, platform
   - **Installed Models**: (empty initially)
   - **Available Models**: 
     - ✅ Compatible section shows:
       - llama-3.1-8b (📥 Install Model button)
       - qwen2.5-7b (📥 Install Model button)
       - sdxl-1.0 (📥 Install Model button)
       - etc.
     - ⚠️ Requires Upgrade section shows:
       - llama-3.1-70b (Need 64GB RAM, Need 48GB GPU)
       - mixtral-8x7b (Need 24GB GPU)
       - etc.

4. User clicks "📥 Install Model" on llama-3.1-8b
5. Button shows "⏳ Installing..."
6. Backend downloads model (may take several minutes)
7. Success message: "✅ Model 'llama-3.1-8b' installed successfully!"
8. Page reloads automatically
9. Model now appears in "Installed Models" section with enable toggle

### Scenario 2: Enabling/Disabling Models

1. In "Installed Models" section, user sees checkbox: ☑ Enabled
2. User unchecks the box
3. Backend updates model_config.json: `{"enabled_models": {"model-name": false}}`
4. Success notification appears: "✅ Model model-name disabled successfully"
5. Application can now check this configuration to skip disabled models

## Technical Details

### Model Configuration File
Location: `./models/model_config.json`

Structure:
```json
{
  "system_info": { ... },
  "installed_models": {
    "text": ["llama-3.1-8b"],
    "image": ["sdxl-1.0"],
    "voice": []
  },
  "enabled_models": {
    "llama-3.1-8b": true,
    "sdxl-1.0": false
  },
  "api_services": { ... }
}
```

### Installation Process
1. Backend receives POST to `/api/v1/setup/ai-models/install`
2. Validates model names against system capabilities
3. Runs: `python setup_ai_models.py --install model-name`
4. Script:
   - Checks system requirements
   - Installs Python dependencies (transformers, diffusers, etc.)
   - Downloads model files using Hugging Face transformers
   - Saves models to `./models/{category}/{model-name}/`
   - Updates model_config.json
5. Returns success/failure status

## Benefits

1. **User-Friendly**: No command-line knowledge required
2. **Safe**: Only shows compatible models for installation
3. **Transparent**: Clear feedback on why models can't be installed
4. **Flexible**: Can enable/disable models without uninstalling
5. **Efficient**: Validates system compatibility before attempting install
6. **Informative**: Shows detailed resource requirements

## Testing

To test the implementation:

```bash
# 1. Start the server
cd src && python -m backend.api.main

# 2. Open browser to http://127.0.0.1:8000/ai-models-setup

# 3. Check system analysis
# - Should show GPU status
# - Should show compatible models with install buttons
# - Should show incompatible models with upgrade requirements

# 4. Test installation (pick a small model)
# - Click install button on a compatible model
# - Watch progress indicator
# - Verify model appears in installed section

# 5. Test enable/disable
# - Toggle checkbox on installed model
# - Check model_config.json for updated status

# 6. Command-line installation
python setup_ai_models.py --install llama-3.1-8b
```

## Notes

- Model installation requires significant disk space and internet bandwidth
- Installation time varies by model size (minutes to hours)
- GPU-required models will only show as compatible on systems with sufficient GPU memory
- The enable/disable feature doesn't uninstall models, just marks them as inactive
- API-based models (GPT-4, DALL-E) still require API keys in admin panel
