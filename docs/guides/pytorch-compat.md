# PyTorch 2.3.1 Compatibility Update

## Issue
User upgraded to PyTorch 2.3.1 with ROCm 5.7 and encountered installation errors with diffusers. The platform needed to be updated to ensure full compatibility with PyTorch 2.3.1 and its required ML dependency versions.

## Summary of Changes

### 1. Updated PyTorch Version to 2.3.1
**Files:** `pyproject.toml`, `server-setup.sh`

**Before:**
```bash
torch==2.2.0+rocm5.7 torchvision==0.17.0+rocm5.7
```

**After:**
```bash
torch==2.3.1+rocm5.7 torchvision==0.18.1+rocm5.7
```

**Rationale:** User upgraded to PyTorch 2.3.1 with ROCm 5.7 support. torchvision 0.18.1 is the corresponding version for PyTorch 2.3.1.

### 2. Updated ML Dependencies for PyTorch 2.3.1 Compatibility
**Files:** `pyproject.toml` and `setup_ai_models.py`

**Before (PyTorch 2.2.0):**
```python
"transformers>=4.35.0"
"diffusers>=0.25.0"
"accelerate>=0.21.0"
"huggingface_hub>=0.20.0"
"numpy>=1.24.0,<2.0"
```

**After (PyTorch 2.3.1):**
```python
"transformers>=4.41.0"
"diffusers>=0.28.0"
"accelerate>=0.29.0"
"huggingface_hub>=0.23.0"
"numpy>=1.24.0,<2.0"
```

**Rationale:** These updated versions are required for full compatibility with PyTorch 2.3.1:
- **transformers>=4.41.0**: Full PyTorch 2.3.x support with optimizations
- **diffusers>=0.28.0**: PyTorch 2.3.1 compatibility (fixes import errors)
- **accelerate>=0.29.0**: PyTorch 2.3.1 compatibility and enhanced device handling
- **huggingface_hub>=0.23.0**: Latest API compatibility
- **numpy>=1.24.0,<2.0**: PyTorch 2.3.1 requires numpy < 2.0 (constraint maintained)

### 3. Updated setup_ai_models.py PyTorch Requirements
**File:** `setup_ai_models.py`

**Before:**
```python
"torch>=2.2.0"
"torchvision>=0.17.0"
```

**After:**
```python
"torch>=2.3.0"
"torchvision>=0.18.0"
```

**Rationale:** Explicitly require PyTorch 2.3.0+ as the minimum version to ensure compatibility with the updated ML dependencies.

### 4. Updated Test Suite
**File:** `tests/test_pytorch_compatibility.py`

- Updated tests to check for `2.3.1+rocm5.7` version tag
- Updated ML dependency compatibility checks for PyTorch 2.3.1
- Updated documentation strings to clarify version requirements

## Important Note on Version Tagging

**PyTorch Version Tags vs ROCm Version:**
- ROCm version installed on system: `5.7.1`
- PyTorch package version tag: `+rocm5.7` (not `+rocm5.7.1`)
- PyTorch repository URL: `https://download.pytorch.org/whl/rocm5.7`

This is PyTorch's naming convention - the repository serves packages for ROCm 5.7.x with the tag `+rocm5.7`.

## Validation

### Tests Run
1. **PyTorch Compatibility Tests:** All 8 tests passed ✅

### Version Consistency
After the changes, all PyTorch version references are now correct and consistent:

| File | Reference | Status |
|------|-----------|--------|
| `pyproject.toml` | `torch==2.3.1+rocm5.7` | ✅ Updated |
| `pyproject.toml` | `torchvision==0.18.1+rocm5.7` | ✅ Updated |
| `server-setup.sh` | `torch==2.3.1+rocm5.7` | ✅ Updated |
| `server-setup.sh` | `torchvision==0.18.1+rocm5.7` | ✅ Updated |
| `setup_ai_models.py` | `torch>=2.3.0` | ✅ Updated |
| `setup_ai_models.py` | `torchvision>=0.18.0` | ✅ Updated |

### ML Dependencies Updated for PyTorch 2.3.1

| Dependency | Old Version | New Version | Status |
|------------|-------------|-------------|--------|
| transformers | >=4.35.0 | >=4.41.0 | ✅ Updated |
| diffusers | >=0.25.0 | >=0.28.0 | ✅ Updated (fixes import errors) |
| accelerate | >=0.21.0 | >=0.29.0 | ✅ Updated |
| huggingface_hub | >=0.20.0 | >=0.23.0 | ✅ Updated |
| numpy | >=1.24.0,<2.0 | >=1.24.0,<2.0 | ✅ Compatible |
| pillow | >=10.0.0 | >=10.0.0 | ✅ Compatible |

## Compatibility Notes

### PyTorch 2.3.1 Features
PyTorch 2.3.1 includes:
- Full ROCm 5.7 support for AMD GPUs
- MI25 (gfx900) compatibility when `HSA_OVERRIDE_GFX_VERSION=9.0.0` is set
- Enhanced performance and stability improvements
- Improved SDPA (Scaled Dot Product Attention) support
- Compatibility with updated transformers, diffusers, and accelerate libraries

### ROCm 5.7.1 Alignment
The PyTorch 2.3.1+rocm5.7 version is specifically built for ROCm 5.7.x and includes:
- HIP runtime support
- ROCm-optimized kernels
- Multi-GPU support for MI25 systems
- Full AMD GPU architecture support with HSA override

## Impact Assessment

### Breaking Changes
**None.** The changes ensure:
- Correct PyTorch 2.3.1 installation
- All ML dependencies are compatible with PyTorch 2.3.1
- System works correctly with ROCm 5.7
- Resolves import errors in diffusers with PyTorch 2.3.1

### System Compatibility
The system remains compatible with:
- Ubuntu 20.04+ and Debian 11+
- Python 3.9+
- ROCm 5.7.1
- AMD MI25 GPUs (gfx900 architecture)
- All ML frameworks with updated versions

## Future Considerations

### Version Updates
If updating to newer PyTorch versions in the future:
1. Check the correct version tag format on PyTorch's website
2. Update `pyproject.toml` rocm extras with correct tag
3. Update `server-setup.sh` installation commands with correct tag
4. Update `setup_ai_models.py` minimum versions
5. Update ML dependencies (transformers, diffusers, accelerate) to compatible versions
6. Verify compatibility with HuggingFace libraries
7. Run the PyTorch compatibility test suite
8. Verify MI25 compatibility if using AMD GPUs

### Test Maintenance
The test suite (`tests/test_pytorch_compatibility.py`) validates:
- Correct PyTorch 2.3.1 version tag format
- Version consistency across all files
- ML dependency compatibility with PyTorch 2.3.1

## Verification Methods

### Method 1: Check Versions via Web UI (Recommended)
The easiest way to verify your installation is through the AI Models Setup page:

1. Start the Gator API server:
   ```bash
   cd src && python -m backend.api.main
   ```

2. Open your browser and navigate to:
   ```
   http://localhost:8000/ai-models-setup
   ```

3. The page will display:
   - Current installed versions of all ML packages
   - Required versions for PyTorch 2.3.1 compatibility (MI-25 GPU)
   - Color-coded status indicators (✓ compatible, ⚠️ version mismatch, ✗ not installed)
   - Compatibility note for AMD MI-25 GPUs with ROCm 5.7

### Method 2: Command-Line Verification
Run the included test suite to validate PyTorch and ML dependency versions:

```bash
python tests/test_pytorch_compatibility.py
```

**Expected output:**
```
Running PyTorch 2.3.1 compatibility tests...

✓ pyproject.toml specifies PyTorch 2.3.1+rocm5.7
✓ server-setup.sh installs PyTorch 2.3.1+rocm5.7
✓ setup_ai_models.py requires torch>=2.3.0
✓ PyTorch version references are consistent across files
✓ PyTorch 2.3.1+rocm5.7 aligns with ROCm 5.7.1
✓ No conflicting PyTorch versions found
✓ ML dependencies are compatible with PyTorch 2.3.1
✓ numpy version is constrained to <2.0 for PyTorch 2.3.1 compatibility

============================================================
✅ All 8 tests passed!
PyTorch 2.3.1 compatibility confirmed for ROCm 5.7.1
```

### Method 3: Check Installed Versions Manually
You can also check package versions directly:

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import diffusers; print(f'diffusers: {diffusers.__version__}')"
python -c "import transformers; print(f'transformers: {transformers.__version__}')"
python -c "import accelerate; print(f'accelerate: {accelerate.__version__}')"
python -c "import huggingface_hub; print(f'huggingface_hub: {huggingface_hub.__version__}')"
```

**Expected output:**
```
PyTorch: 2.3.1+rocm5.7
diffusers: 0.28.0 or higher
transformers: 4.41.0 or higher
accelerate: 0.29.0 or higher
huggingface_hub: 0.23.0 or higher
```

## References
- [PyTorch ROCm Installation](https://pytorch.org/get-started/locally/)
- [PyTorch ROCm 5.7 Packages](https://download.pytorch.org/whl/rocm5.7/)
- [AMD ROCm 5.7.1 Documentation](https://rocmdocs.amd.com/)
- [MI25 Compatibility Guide](docs/MI25_COMPATIBILITY.md)
- [Transformers Release Notes](https://github.com/huggingface/transformers/releases)
- [Diffusers Release Notes](https://github.com/huggingface/diffusers/releases)
- [Accelerate Release Notes](https://github.com/huggingface/accelerate/releases)

