# Fix Dependencies Button Implementation Summary

## Issue
Add a button to fix ML dependencies shown on the AI Models Setup page (http://127.0.0.1:8000/ai-models-setup)

## Solution Implemented

### 1. Backend API Endpoint
**File**: `src/backend/api/routes/setup.py`

Updated POST endpoint: `/api/v1/setup/ai-models/fix-dependencies`

**Functionality**:
- **Excludes torch and torchvision** to preserve ROCm-specific installations
- Parses pyproject.toml to get list of dependencies
- Installs/upgrades each package individually with `--upgrade` flag
- Skips torch==* and torchvision==* packages during installation
- 5-minute timeout per package for reliable installation
- Returns detailed information including:
  - Success status
  - Number of packages installed/failed
  - List of failed packages
  - Full stdout/stderr logs
- Logs installation progress for each package

**Key Change**:
The endpoint now **excludes torch and torchvision** from reinstallation because:
- These packages are installed with ROCm-specific versions (2.3.1+rocm5.7, 0.18.1+rocm5.7)
- They must be installed from ROCm index URL, not standard PyPI
- Regular `pip install -e .` would overwrite them with incompatible versions
- This preserves GPU compatibility while updating other ML dependencies

**Code**:
```python
@router.post("/ai-models/fix-dependencies")
async def fix_dependencies() -> Dict[str, Any]:
    """
    Install or update missing/outdated ML dependencies.
    
    This excludes torch and torchvision to preserve ROCm-specific installations,
    and installs/upgrades diffusers, transformers, accelerate, huggingface_hub,
    numpy, and other dependencies.
    """
    # Parses pyproject.toml
    # Installs packages individually, skipping torch/torchvision
    # Returns detailed results
```

### 2. Frontend UI Updates
**File**: `ai_models_setup.html`

**Changes**:
1. **Detection Logic**: Added `hasMissingOrOutdated` flag to track package issues
2. **Button Display**: Button only appears when issues detected
3. **Button Handler**: Added `fixDependencies()` JavaScript function
4. **Log Display**: Added `showDependencyInstallLog()` to display results

**Button Location**: 
- Appears in the "ML Package Versions" table section
- Below the package status table
- Only visible when missing/outdated packages exist

**User Flow**:
1. Page loads and checks package versions
2. If issues found, "🔧 Fix Dependencies" button appears
3. User clicks button → confirmation dialog
4. If confirmed → API call starts installation
5. Installation logs displayed in modal
6. Page auto-reloads after successful installation

### 3. Package Detection
The button appears when:
- Any package shows "Not installed" (red ✗)
- Torch/torchvision show version mismatch (yellow ⚠️)

### 4. Testing Performed
✅ Backend endpoint accessible via TestClient
✅ Status endpoint returns correct package info
✅ Button appears/disappears based on package status
✅ Button triggers confirmation dialog correctly
✅ Modal displays installation logs properly
✅ Code formatted with black
✅ **NEW:** Dependency parsing correctly excludes torch/torchvision
✅ **NEW:** Endpoint returns detailed package installation results
✅ **NEW:** Error handling for failed package installations
✅ **NEW:** Comprehensive unit tests added (test_fix_dependencies.py)

## Files Changed
1. `src/backend/api/routes/setup.py` - **Updated** fix_dependencies endpoint to exclude torch/torchvision
2. `ai_models_setup.html` - Added button and JavaScript handlers (previous implementation)
3. `tests/integration/test_fix_dependencies.py` - **NEW:** Added comprehensive tests for the endpoint

## Screenshot
The button is visible in the ML Package Versions section when dependencies need fixing:

![Fix Dependencies Button](https://github.com/user-attachments/assets/95c45ae9-b14c-4d5a-a168-4738f4c80fd2)

## Implementation Notes
- Minimal changes approach - only modified the fix_dependencies endpoint
- **Critical fix**: Excludes torch and torchvision to prevent overwriting ROCm installations
- Individual package installation provides better error isolation and reporting
- Preserves ROCm-specific GPU compatibility while updating other dependencies
- Reused existing modal infrastructure for consistency
- Follows existing code patterns in the file
- No breaking changes to existing functionality
- User-friendly with clear confirmation and progress feedback
- Comprehensive test coverage for the new functionality

## Background: Why Exclude Torch/Torchvision?

The AMD MI-25 GPU requires specific ROCm-optimized versions of PyTorch:
- `torch==2.3.1+rocm5.7` 
- `torchvision==0.18.1+rocm5.7`

These versions are:
1. Only available from PyTorch's ROCm index URL
2. Not part of standard PyPI
3. Critical for GPU functionality

When `pip install -e .` runs, it tries to satisfy torch/torchvision from pyproject.toml dependencies, which would install incompatible versions from standard PyPI. This breaks GPU support.

The fix:
- Parse pyproject.toml manually
- Skip torch==* and torchvision==* packages
- Install all other dependencies normally
- Preserve the ROCm installations
