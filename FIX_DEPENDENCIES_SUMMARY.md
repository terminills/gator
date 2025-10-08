# Fix Dependencies Implementation - Summary

## Problem
The "Fix Dependencies" button at http://127.0.0.1:8000/ai-models-setup was reinstalling ALL dependencies, including torch and torchvision. This caused issues because:

- AMD MI-25 GPU requires ROCm-specific versions: `torch==2.3.1+rocm5.7` and `torchvision==0.18.1+rocm5.7`
- These versions are only available from PyTorch's ROCm index URL
- Running `pip install -e .` would overwrite them with incompatible standard PyPI versions
- This breaks GPU functionality

## Solution
Modified the `/api/v1/setup/ai-models/fix-dependencies` endpoint to:

1. **Parse pyproject.toml manually** to extract dependencies
2. **Exclude torch and torchvision** from the installation list
3. **Install packages individually** with `--upgrade` flag
4. **Preserve ROCm installations** while updating other ML dependencies

## Changes Made

### File: `src/backend/api/routes/setup.py`

**Before:**
```python
# Install/upgrade all dependencies from pyproject.toml
cmd = [sys.executable, "-m", "pip", "install", "-e", str(project_root)]
result = subprocess.run(cmd, ...)
```

**After:**
```python
# Parse dependencies from pyproject.toml, excluding torch/torchvision
for line in content.split("\n"):
    if dependencies_section and line.startswith('"'):
        dep = match.group(1)
        if not dep.startswith("torch==") and not dep.startswith("torchvision=="):
            packages_to_install.append(dep)

# Install packages one by one
for package in packages_to_install:
    cmd = [sys.executable, "-m", "pip", "install", "--upgrade", package]
    result = subprocess.run(cmd, ...)
```

## Benefits

1. ✅ **Preserves GPU Functionality** - ROCm-specific torch/torchvision remain intact
2. ✅ **Updates Other Dependencies** - All other packages get upgraded to correct versions
3. ✅ **Better Error Handling** - Individual package installation allows isolation of failures
4. ✅ **Detailed Reporting** - Returns exact list of success/failed packages
5. ✅ **No Breaking Changes** - Existing functionality remains intact

## Testing

### Automated Tests
- ✅ `test_fix_dependencies_excludes_torch_torchvision` - Verifies exclusion logic
- ✅ `test_fix_dependencies_response_structure` - Validates response format
- ✅ `test_fix_dependencies_handles_failures` - Tests error handling
- ✅ All existing setup API tests pass

### Manual Verification
```bash
# Test dependency parsing
python /tmp/test_fix_dependencies.py
# Output: ✅ Test PASSED: Dependencies parsed correctly, torch/torchvision excluded

# Packages included: 34 total
# - diffusers>=0.28.0 ✓
# - transformers>=4.41.0 ✓
# - accelerate>=0.29.0 ✓
# - huggingface_hub>=0.23.0 ✓
# - numpy>=1.24.0,<2.0 ✓
# ... (and 29 more)

# Packages excluded:
# - torch (excluded) ✓
# - torchvision (excluded) ✓
```

## Files Modified

1. **src/backend/api/routes/setup.py** (87 lines added, 22 removed)
   - Updated `fix_dependencies()` endpoint
   - Added dependency parsing logic
   - Added individual package installation
   - Enhanced error handling and reporting

2. **tests/integration/test_fix_dependencies.py** (NEW, 135 lines)
   - Comprehensive test suite
   - Tests for torch/torchvision exclusion
   - Tests for response structure
   - Tests for failure handling

3. **FIX_DEPENDENCIES_IMPLEMENTATION.md** (updated)
   - Added explanation of changes
   - Documented why torch/torchvision are excluded
   - Updated testing section

## Impact

This fix ensures that when users click "Fix Dependencies" on the AI Models Setup page:

1. Missing or outdated ML packages (diffusers, transformers, etc.) are updated
2. ROCm-specific torch/torchvision installations remain untouched
3. GPU functionality is preserved
4. Users get detailed feedback on which packages were updated

## Example Response

```json
{
  "success": true,
  "message": "Successfully installed/upgraded all 34 packages (torch/torchvision preserved)",
  "packages_installed": 34,
  "packages_failed": 0,
  "failed_packages": [],
  "stdout": "=== Installing diffusers>=0.28.0 ===\nSuccessfully installed...",
  "stderr": ""
}
```

## Deployment Notes

No special deployment steps required. The change is backward compatible and will work immediately upon deployment.

## Related Issues

- Fixes installation issues where torch/torchvision were being overwritten
- Addresses GPU compatibility problems with AMD MI-25 GPUs
- Maintains proper ROCm 5.7 support
