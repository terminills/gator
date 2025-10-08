# Fix Dependencies - Before and After Comparison

## The Issue (Screenshot from Issue)

When clicking "Fix Dependencies" button on the AI Models Setup page, all dependencies were being reinstalled:

![Before Fix](https://github.com/user-attachments/assets/a8ceed5f-a30c-4272-ba97-4079c95f7efd)

**Problem:** The output shows:
```
Requirement already satisfied: transformers>=4.41.0
Requirement already satisfied: diffusers>=0.28.0
...
```

But torch/torchvision would be reinstalled from standard PyPI, potentially overwriting ROCm versions.

## The Fix

### Code Changes

**Before (src/backend/api/routes/setup.py):**
```python
@router.post("/ai-models/fix-dependencies")
async def fix_dependencies() -> Dict[str, Any]:
    # Install/upgrade all dependencies from pyproject.toml
    cmd = [sys.executable, "-m", "pip", "install", "-e", str(project_root)]
    
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    
    return {
        "success": result.returncode == 0,
        "message": "Dependencies installed/updated successfully" if result.returncode == 0 else "Failed",
        "stdout": result.stdout,
        "stderr": result.stderr,
    }
```

**After (src/backend/api/routes/setup.py):**
```python
@router.post("/ai-models/fix-dependencies")
async def fix_dependencies() -> Dict[str, Any]:
    # Parse pyproject.toml to get dependency list
    content = pyproject_path.read_text()
    packages_to_install = []
    
    for line in content.split("\n"):
        if dependencies_section and line.startswith('"'):
            dep = match.group(1)
            # Skip torch and torchvision to preserve ROCm installations
            if not dep.startswith("torch==") and not dep.startswith("torchvision=="):
                packages_to_install.append(dep)
    
    # Install packages one by one
    for package in packages_to_install:
        cmd = [sys.executable, "-m", "pip", "install", "--upgrade", package]
        result = subprocess.run(cmd, ...)
        # Track success/failures
    
    return {
        "success": success,
        "message": f"Installed {packages_installed}/{total} packages (torch/torchvision preserved)",
        "packages_installed": packages_installed,
        "packages_failed": packages_failed,
        "failed_packages": failed_packages,
        "stdout": "\n".join(all_stdout),
        "stderr": "\n".join(all_stderr),
    }
```

## Behavior Comparison

### Before Fix
| Behavior | Result |
|----------|--------|
| Click "Fix Dependencies" | Runs `pip install -e .` |
| Torch/Torchvision handling | ❌ Reinstalled from standard PyPI |
| ROCm compatibility | ❌ Broken (overwritten with CPU versions) |
| GPU functionality | ❌ Lost |
| Error reporting | ⚠️ Basic (single return code) |
| Package granularity | ⚠️ All-or-nothing installation |

### After Fix
| Behavior | Result |
|----------|--------|
| Click "Fix Dependencies" | Parses and installs packages individually |
| Torch/Torchvision handling | ✅ Explicitly excluded |
| ROCm compatibility | ✅ Preserved (2.3.1+rocm5.7) |
| GPU functionality | ✅ Maintained |
| Error reporting | ✅ Detailed (per-package status) |
| Package granularity | ✅ Individual package success/failure tracking |

## Test Results

### Automated Tests
```bash
$ pytest tests/integration/test_fix_dependencies.py -v
test_fix_dependencies_excludes_torch_torchvision PASSED ✓
test_fix_dependencies_response_structure PASSED ✓
test_fix_dependencies_handles_failures PASSED ✓
```

### Manual Verification
```bash
$ python /tmp/test_fix_dependencies.py
✓ Found pyproject.toml
📦 Total packages to install: 34

🔍 Key ML packages (excluding torch/torchvision):
  ✓ diffusers>=0.28.0
  ✓ transformers>=4.41.0
  ✓ accelerate>=0.29.0
  ✓ huggingface_hub>=0.23.0
  ✓ numpy>=1.24.0,<2.0

🚫 Exclusions:
  ✓ torch excluded
  ✓ torchvision excluded

✅ Test PASSED
```

## Example API Response

**Before:**
```json
{
  "success": true,
  "message": "Dependencies installed/updated successfully",
  "stdout": "...",
  "stderr": "",
  "return_code": 0
}
```

**After:**
```json
{
  "success": true,
  "message": "Successfully installed/upgraded all 34 packages (torch/torchvision preserved)",
  "packages_installed": 34,
  "packages_failed": 0,
  "failed_packages": [],
  "stdout": "=== Installing diffusers>=0.28.0 ===\nSuccessfully installed...\n=== Installing transformers>=4.41.0 ===\n...",
  "stderr": ""
}
```

## User Experience

### Before
1. User clicks "Fix Dependencies"
2. All packages reinstalled (including torch/torchvision)
3. GPU support breaks
4. User must manually reinstall ROCm versions

### After
1. User clicks "Fix Dependencies"
2. Only non-torch/torchvision packages updated
3. GPU support preserved
4. Detailed feedback on which packages were updated

## Impact

✅ **GPU Functionality Preserved**: ROCm-specific torch/torchvision remain intact
✅ **Dependencies Updated**: All other ML packages upgraded to correct versions
✅ **Better Diagnostics**: Detailed per-package installation status
✅ **Error Isolation**: Individual package failures don't block others
✅ **No Breaking Changes**: Existing functionality remains intact

## Files Modified

1. `src/backend/api/routes/setup.py` - Core fix (87 lines added, 22 removed)
2. `tests/integration/test_fix_dependencies.py` - New tests (135 lines)
3. `FIX_DEPENDENCIES_IMPLEMENTATION.md` - Updated documentation
4. `FIX_DEPENDENCIES_SUMMARY.md` - Summary documentation
