# Visual Guide: Fix Dependencies Feature

## The UI (AI Models Setup Page)

The "Fix Dependencies" button appears on the AI Models Setup page at `http://127.0.0.1:8000/ai-models-setup`

### Screenshot from Issue (Before Fix)
![AI Models Setup Page](https://github.com/user-attachments/assets/a8ceed5f-a30c-4272-ba97-4079c95f7efd)

The page shows:
- GPU Status: ✓ GPU Available (Radeon Instinct MI25)
- Python Version: 3.9.5
- PyTorch Version: 2.3.1+rocm5.7
- Platform: linux

**ML Package Versions Table:**
| Package | Installed | Required | Status |
|---------|-----------|----------|--------|
| torch | 2.3.1+rocm5.7 | 2.3.1+rocm5.7 | ✓ |
| torchvision | 0.18.1+rocm5.7 | 0.18.1+rocm5.7 | ✓ |
| diffusers | 0.35.1 | >=0.28.0 | ✓ |
| transformers | 4.57.0 | >=4.41.0 | ✓ |
| accelerate | 1.10.1 | >=0.29.0 | ✓ |
| huggingface_hub | 0.35.3 | >=0.23.0 | ✓ |
| numpy | Not installed | >=1.24.0,<2.0 | ✗ |

**Fix Dependencies Button:**
```
┌──────────────────────┐
│  🔧 Fix Dependencies │
└──────────────────────┘
```
_"This will install/update missing or outdated ML packages. Installation may take several minutes."_

## What Happens When Button is Clicked

### Before This Fix
1. User clicks "Fix Dependencies"
2. Modal shows: "Installing dependencies..."
3. Backend runs: `pip install -e .`
4. **Problem:** This reinstalls torch/torchvision from PyPI
5. ROCm versions get overwritten with standard CPU versions
6. GPU support breaks
7. User must manually run: `pip install torch==2.3.1+rocm5.7 torchvision==0.18.1+rocm5.7 --index-url https://download.pytorch.org/whl/rocm5.7`

### After This Fix
1. User clicks "Fix Dependencies"
2. Confirmation dialog: "Install/update ML dependencies?"
3. Modal shows: "Installing dependencies (excluding torch/torchvision)..."
4. Backend:
   - Parses pyproject.toml
   - Identifies 34 packages to update
   - **Skips torch and torchvision**
   - Installs packages individually:
     ```
     Installing diffusers>=0.28.0... ✓
     Installing transformers>=4.41.0... ✓
     Installing accelerate>=0.29.0... ✓
     Installing huggingface_hub>=0.23.0... ✓
     Installing numpy>=1.24.0,<2.0... ✓
     Installing pandas>=2.0.0... ✓
     ...
     ```
5. Modal shows results:
   ```
   ✅ Successfully installed/upgraded all 34 packages
   (torch/torchvision preserved)
   
   Installed: 34
   Failed: 0
   
   [View detailed logs]
   ```
6. Page reloads automatically
7. Table updates:
   | Package | Installed | Required | Status |
   |---------|-----------|----------|--------|
   | torch | 2.3.1+rocm5.7 | 2.3.1+rocm5.7 | ✓ |
   | torchvision | 0.18.1+rocm5.7 | 0.18.1+rocm5.7 | ✓ |
   | numpy | 1.26.4 | >=1.24.0,<2.0 | ✓ |
   | ... | ... | ... | ✓ |

## Key Differences

### Before Fix
```
API Request: POST /api/v1/setup/ai-models/fix-dependencies

Backend Action:
  pip install -e .
  
Result:
  ❌ torch overwritten: 2.3.1+rocm5.7 → 2.8.0 (CPU)
  ❌ torchvision overwritten: 0.18.1+rocm5.7 → 0.19.0 (CPU)
  ✅ Other packages updated
  
GPU Status: BROKEN ❌
```

### After Fix
```
API Request: POST /api/v1/setup/ai-models/fix-dependencies

Backend Action:
  Parse pyproject.toml
  For each package EXCEPT torch/torchvision:
    pip install --upgrade <package>
  
Result:
  ✅ torch preserved: 2.3.1+rocm5.7 (unchanged)
  ✅ torchvision preserved: 0.18.1+rocm5.7 (unchanged)
  ✅ diffusers updated: 0.35.1 → latest
  ✅ transformers updated: 4.57.0 → latest
  ✅ numpy installed: → 1.26.4
  ... (and 29 more packages)
  
GPU Status: WORKING ✅
```

## Response Structure

### Before
```json
{
  "success": true,
  "message": "Dependencies installed/updated successfully",
  "stdout": "...",
  "stderr": "",
  "return_code": 0
}
```

### After
```json
{
  "success": true,
  "message": "Successfully installed/upgraded all 34 packages (torch/torchvision preserved)",
  "packages_installed": 34,
  "packages_failed": 0,
  "failed_packages": [],
  "stdout": "=== Installing diffusers>=0.28.0 ===\nSuccessfully installed diffusers-0.35.1\n=== Installing transformers>=4.41.0 ===\n...",
  "stderr": ""
}
```

## User Benefits

1. **✅ Preserved GPU Functionality**
   - ROCm-specific torch/torchvision remain intact
   - No need to manually reinstall GPU drivers

2. **✅ Updated Dependencies**
   - All other ML packages get upgraded
   - Missing packages (like numpy) get installed

3. **✅ Better Feedback**
   - See exactly which packages were updated
   - Know if any packages failed to install
   - View detailed installation logs

4. **✅ Safer Operation**
   - Can confidently click "Fix Dependencies"
   - Won't break existing GPU setup
   - Individual package failures don't block others

## Testing the Fix

To test this fix:

1. **Start the server:**
   ```bash
   cd src && python -m backend.api.main
   ```

2. **Open the AI Models Setup page:**
   ```
   http://127.0.0.1:8000/ai-models-setup
   ```

3. **Verify torch/torchvision versions:**
   - Should show: `2.3.1+rocm5.7` and `0.18.1+rocm5.7`

4. **Click "Fix Dependencies"**

5. **Verify after installation:**
   - torch version unchanged: `2.3.1+rocm5.7` ✓
   - torchvision version unchanged: `0.18.1+rocm5.7` ✓
   - Other packages updated to latest versions ✓
   - GPU still works: `rocm-smi` shows GPU ✓

## Implementation Details

See the following documentation files for more details:
- `FIX_DEPENDENCIES_IMPLEMENTATION.md` - Technical implementation
- `FIX_DEPENDENCIES_SUMMARY.md` - High-level summary
- `FIX_DEPENDENCIES_BEFORE_AFTER.md` - Detailed comparison
- `tests/integration/test_fix_dependencies.py` - Test suite
