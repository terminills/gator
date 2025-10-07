# AI Model Installation Enhancement - Testing Guide

## Quick Test Commands

### 1. Syntax Validation (No Dependencies Required)
```bash
# Validate Python syntax
python3 -m py_compile src/backend/api/routes/setup.py
python3 -m py_compile setup_ai_models.py
python3 -m py_compile tests/integration/test_setup_api.py

# Validate HTML
python3 -c "from html.parser import HTMLParser; parser = HTMLParser(); parser.feed(open('ai_models_setup.html').read())"
```

### 2. Test Model Analysis (No GPU Required)
```bash
# Test the model setup manager directly
python3 -c "
import sys
sys.path.insert(0, '.')
from setup_ai_models import ModelSetupManager

manager = ModelSetupManager()
print(f'GPU Type: {manager.gpu_type}')
print(f'GPU Memory: {manager.gpu_memory:.1f}GB')

recs = manager.analyze_system_requirements()
print(f'Installable: {len(recs[\"installable\"])}')
print(f'Requires Upgrade: {len(recs[\"requires_upgrade\"])}')
"
```

### 3. Test API Endpoints (Requires Server Running)
```bash
# Start the server
cd src && python -m backend.api.main &

# Wait for server to start
sleep 5

# Test the new endpoints
curl http://localhost:8000/api/v1/setup/ai-models/recommendations | jq .

curl http://localhost:8000/api/v1/setup/ai-models/status | jq .

curl -X POST http://localhost:8000/api/v1/setup/ai-models/enable \
  -H "Content-Type: application/json" \
  -d '{"model_name": "test-model", "enabled": true}' | jq .
```

### 4. Test Web Interface
```bash
# Open browser to the AI models setup page
xdg-open http://localhost:8000/ai-models-setup

# Or manually navigate to:
# http://localhost:8000/ai-models-setup
```

## Expected Behavior

### On Page Load
1. System information displays (GPU status, Python version, platform)
2. "Installed Models" section shows any installed models
3. "Available Models" section loads with two categories:
   - **✅ Compatible Models** with "📥 Install Model" buttons
   - **⚠️ Requires Upgrade** with requirement details

### When Installing a Model
1. Click "📥 Install Model" button
2. Button changes to "⏳ Installing..."
3. Backend downloads model (may take minutes)
4. Success alert: "✅ Model 'model-name' installed successfully!"
5. Page reloads automatically
6. Model appears in "Installed Models" with enable toggle

### When Enabling/Disabling a Model
1. Toggle checkbox on installed model
2. Backend updates model_config.json
3. Success notification appears briefly
4. Model state persists across page reloads

## Testing with Different Hardware Scenarios

### CPU-Only System (No GPU)
```bash
# Expected: All GPU-requiring models show in "Requires Upgrade" section
# Only CPU-compatible models (if any) show as installable
```

### GPU System with 8GB VRAM
```bash
# Expected:
# - Installable: llama-3.1-8b, qwen2.5-7b, sdxl-1.0, stable-diffusion-v1-5
# - Requires Upgrade: llama-3.1-70b, qwen2.5-72b, mixtral-8x7b
```

### GPU System with 24GB VRAM
```bash
# Expected:
# - Installable: All 8GB models + mixtral-8x7b, flux.1-dev
# - Requires Upgrade: llama-3.1-70b, qwen2.5-72b (need 48GB)
```

## Manual Test Checklist

- [ ] Page loads without errors
- [ ] System information displays correctly
- [ ] Compatible models show install buttons
- [ ] Incompatible models show upgrade requirements
- [ ] Install button works (test with small model)
- [ ] Installation progress shows correctly
- [ ] Success notification appears
- [ ] Page reloads after successful install
- [ ] Installed model appears with enable toggle
- [ ] Enable/disable toggle works
- [ ] Enable/disable persists in model_config.json
- [ ] Multiple models can be installed sequentially
- [ ] Error handling works for failed installations

## Known Limitations in Test Environment

1. **Network Issues**: Package downloads may timeout - this is expected in CI environment
2. **No GPU**: Tests will run in CPU-only mode, all models will require upgrade
3. **Dependencies**: Full test requires FastAPI, SQLAlchemy, etc. to be installed
4. **Model Downloads**: Actual model installation requires significant bandwidth and time

## Debugging Tips

### If Installation Fails
```bash
# Check the setup script directly
python setup_ai_models.py --analyze

# Try installing manually
python setup_ai_models.py --install model-name
```

### If Enable/Disable Doesn't Work
```bash
# Check the model config file
cat models/model_config.json | jq .enabled_models
```

### If API Endpoints Return 500 Error
```bash
# Check server logs
# The error will show in the console where you started the server
```

## Success Criteria

✅ **All changes implemented:**
- New API endpoints functional
- UI displays install buttons
- Installation works end-to-end
- Enable/disable persists correctly
- Tests pass
- Documentation complete

✅ **User experience improved:**
- Users can install models via UI
- Clear feedback on compatibility
- Progress indicators work
- Error messages are helpful

✅ **Code quality maintained:**
- Python syntax valid
- HTML valid
- Tests added for new functionality
- Documentation comprehensive
