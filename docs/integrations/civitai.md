# CivitAI Integration and Enhanced Ollama Support

## Overview

This implementation adds comprehensive support for browsing, downloading, and managing AI models from CivitAI, along with enhanced GPU-aware inference engine selection that automatically uses Ollama for incompatible AMD GPUs.

## Features Implemented

### 1. CivitAI Model Integration

#### API Client (`src/backend/utils/civitai_utils.py`)
- **CivitAIClient**: Full-featured client for CivitAI REST API
  - List models with filtering by type, base model, query, NSFW, etc.
  - Get detailed model information including versions and files
  - Download models with progress tracking and integrity verification
  - Automatic API key handling from settings

- **Model Types Supported**:
  - Checkpoint (full Stable Diffusion models)
  - LORA (Low-Rank Adaptation models)
  - TextualInversion (embeddings)
  - Hypernetwork
  - ControlNet
  - And more...

- **Base Model Support**:
  - SD 1.5, SD 2.0, SD 2.1
  - SDXL 0.9, SDXL 1.0
  - SD 3, Flux.1
  - Others

#### API Routes (`src/backend/api/routes/civitai.py`)
- `GET /api/v1/civitai/models` - List models with advanced filtering
- `GET /api/v1/civitai/models/{model_id}` - Get model details
- `GET /api/v1/civitai/model-versions/{version_id}` - Get version details
- `POST /api/v1/civitai/download` - Download a model
- `GET /api/v1/civitai/local-models` - List downloaded CivitAI models
- `GET /api/v1/civitai/types` - Get available model types
- `GET /api/v1/civitai/base-models` - Get available base models

#### Setup Integration (`src/backend/api/routes/setup.py`)
- `GET /api/v1/setup/ai-models/civitai-browse` - Browse CivitAI models in setup UI
  - Returns curated, popular models
  - Formatted for AI models setup page
  - Filters out NSFW content for setup page

#### Settings & Preferences (`src/backend/models/settings.py`)
New settings added:
- `civitai_api_key` - API key for authenticated downloads
- `civitai_allow_nsfw` - Allow/block NSFW models (default: false)
- `civitai_track_usage` - Track model sources for attribution (default: true)
- `prefer_ollama_for_gfx1030` - Use Ollama for AMD RX 6000 GPUs (default: true)

#### Model Download & Tracking
- Downloads save to `{models_path}/civitai/` by default
- Metadata JSON saved alongside each model file
- Tracks licensing, trained words, NSFW status
- SHA256 hash verification for integrity
- Resume-capable downloads

### 2. GPU-Aware Inference Engine Selection

#### GPU Detection (`src/backend/utils/gpu_detection.py`)
- **Architecture Detection**:
  - Detects AMD GPU architecture (gfx1030, gfx1100, gfx90a, etc.)
  - Uses rocminfo, rocm-smi, and sysfs fallbacks
  - Maps product names to architectures

- **Compatibility Checking**:
  - `is_vllm_compatible_gpu()` - Checks if vLLM will work
  - Known incompatible: gfx1030 (RX 6000 series)
  - Known compatible: gfx90a (MI210/250), gfx1100 (RX 7000)
  - Falls back to Ollama for incompatible GPUs

- **System Information**:
  - VRAM detection
  - Compute units count
  - Vendor identification

#### Enhanced Ollama Fallback (`src/backend/services/ai_models.py`)
- **Proactive Selection**: Checks GPU compatibility before starting generation
- **Automatic Fallback**: Falls back to Ollama if llama.cpp fails
- **Logging**: Detailed logs showing why Ollama was chosen
- **Preference Support**: Respects `PREFER_OLLAMA_FOR_GFX1030` env var

### 3. Bug Fixes

#### HuggingFace Hub Deprecation Warning
Fixed deprecated `local_dir_use_symlinks` parameter:
- Removed from `src/backend/services/ai_models.py`
- Removed from `setup_ai_models.py` (4 occurrences)
- Resolves UserWarning about deprecated parameter

#### Database Access Bug
Fixed critical bug in GPU detection:
- **Problem**: Was trying to instantiate `SettingsService` without db session
- **Impact**: Could cause startup crashes, personas/content not showing
- **Solution**: Changed to use environment variable instead
- **File**: `src/backend/utils/gpu_detection.py`

## Usage Examples

### CivitAI Model Download via API

```python
import httpx

# List popular Stable Diffusion models
response = await httpx.get(
    "http://localhost:8000/api/v1/civitai/models",
    params={
        "query": "stable diffusion",
        "model_types": "Checkpoint",
        "limit": 10,
        "sort": "Highest Rated",
    }
)
models = response.json()

# Download a specific model version
response = await httpx.post(
    "http://localhost:8000/api/v1/civitai/download",
    json={
        "model_version_id": 123456,
        "file_type": "Model",
    }
)
result = response.json()
print(f"Downloaded to: {result['file_path']}")
```

### Python SDK Usage

```python
from backend.utils.civitai_utils import CivitAIClient, CivitAIModelType

# Initialize client
client = CivitAIClient(api_key="your_api_key")

# Search for models
models = await client.list_models(
    query="anime",
    model_types=[CivitAIModelType.CHECKPOINT],
    limit=20,
)

# Download a model
file_path, metadata = await client.download_model(
    model_version_id=123456,
    output_path=Path("./models/civitai"),
)

print(f"Model: {metadata['model_name']}")
print(f"License: {metadata['license']}")
print(f"Trained words: {metadata['trained_words']}")
```

### GPU Detection

```python
from backend.utils.gpu_detection import (
    detect_amd_gpu_architecture,
    is_vllm_compatible_gpu,
    should_use_ollama_fallback,
    get_gpu_info,
)

# Detect GPU architecture
arch = detect_amd_gpu_architecture()
print(f"GPU: {arch}")  # e.g., "gfx1030"

# Check vLLM compatibility
if not is_vllm_compatible_gpu():
    print("vLLM not recommended for this GPU")

# Get full GPU info
info = get_gpu_info()
print(f"Architecture: {info['architecture']}")
print(f"VRAM: {info['vram_gb']} GB")
print(f"Ollama recommended: {info['ollama_recommended']}")

# Check if should use Ollama
if should_use_ollama_fallback():
    print("Using Ollama for text generation")
```

### Environment Variables

```bash
# Force Ollama usage for gfx1030 GPUs
export PREFER_OLLAMA_FOR_GFX1030=true

# CivitAI API key (optional, for authenticated downloads)
export CIVITAI_API_KEY=your_key_here
```

## Configuration

### Database Settings
These settings can be configured via the admin panel or API:

```python
# Enable CivitAI with API key
civitai_api_key = "your_api_key_here"

# Allow NSFW models (default: false)
civitai_allow_nsfw = True

# Track model sources for attribution (default: true)
civitai_track_usage = True

# Prefer Ollama for AMD RX 6000 series (default: true)
prefer_ollama_for_gfx1030 = True
```

### AI Models Setup Page
The CivitAI browse endpoint integrates with the existing AI models setup page:
- Endpoint: `GET /api/v1/setup/ai-models/civitai-browse`
- Shows popular, highly-rated models
- Filters by model type (Checkpoint, LORA, etc.)
- Includes download statistics and ratings
- NSFW models are filtered out

## File Structure

```
src/backend/
├── api/routes/
│   ├── civitai.py              # CivitAI API routes
│   └── setup.py                # Enhanced with CivitAI browse
├── models/
│   └── settings.py             # Added CivitAI settings
├── services/
│   └── ai_models.py            # Enhanced with CivitAI & GPU detection
└── utils/
    ├── civitai_utils.py        # CivitAI client & utilities
    └── gpu_detection.py        # AMD GPU detection & compatibility

tests/
├── test_civitai_integration.py # CivitAI tests
└── test_gpu_detection.py       # GPU detection tests
```

## Dependencies

All dependencies are already included in `pyproject.toml`:
- `httpx` - For CivitAI API requests
- `pydantic` - For data validation
- `fastapi` - For API routes

No additional dependencies required!

## GPU Compatibility Matrix

| GPU Architecture | Model           | vLLM Support | Ollama Recommended |
|-----------------|-----------------|--------------|-------------------|
| gfx1030         | RX 6600-6900    | ⚠️ Issues    | ✅ Yes           |
| gfx1031/1032    | RX 6000 Mobile  | ⚠️ Issues    | ✅ Yes           |
| gfx1100         | RX 7600-7900    | ✅ Good      | ⏸️ Optional      |
| gfx90a          | MI210/MI250     | ✅ Excellent | ⏸️ Optional      |
| gfx908          | MI100           | ✅ Good      | ⏸️ Optional      |
| gfx942          | MI300           | ✅ Excellent | ⏸️ Optional      |

## Testing

Run tests with:
```bash
# All tests (CivitAI tests require CIVITAI_TESTS_ENABLED=true)
python -m pytest tests/test_civitai_integration.py tests/test_gpu_detection.py -v

# Run with CivitAI API tests (makes real API calls)
CIVITAI_TESTS_ENABLED=true python -m pytest tests/test_civitai_integration.py -v

# GPU detection tests (works without AMD GPU)
python -m pytest tests/test_gpu_detection.py -v
```

## Security Considerations

1. **API Keys**: Stored as sensitive settings, encrypted in database
2. **NSFW Content**: Filtered by default, requires explicit enable
3. **File Integrity**: SHA256 hash verification for downloads
4. **Content Attribution**: Tracks model sources and licensing
5. **Input Validation**: All inputs validated via Pydantic models

## Future Enhancements

- [ ] Add CivitAI model browser UI section to AI models setup page
- [ ] Implement model update checking
- [ ] Add model version comparison
- [ ] Support for model collections/bundles
- [ ] Automatic model recommendations based on usage
- [ ] Integration with existing model management UI

## Troubleshooting

### CivitAI API Returns Empty Results
- Check if `civitai_allow_nsfw` is disabled (default)
- Try different search queries
- Verify network connectivity

### GPU Detection Not Working
- Install ROCm utilities: `sudo apt install rocm-smi rocminfo`
- Check GPU is recognized: `rocm-smi`
- Verify architecture: `rocminfo | grep gfx`

### Ollama Not Being Used
- Check Ollama is installed: `which ollama`
- Verify models available: `ollama list`
- Set environment variable: `export PREFER_OLLAMA_FOR_GFX1030=true`

### Personas/Content Not Showing Up
- **Fixed**: Database access bug in GPU detection
- Restart server after applying this update
- Check logs for startup errors
- Verify database is accessible

## References

- CivitAI API Documentation: https://developer.civitai.com/docs/api/public-rest
- CivitAI Python SDK: https://developer.civitai.com/docs/api/python-sdk
- Ollama Documentation: https://ollama.com/docs
- AMD ROCm Documentation: https://rocm.docs.amd.com/
