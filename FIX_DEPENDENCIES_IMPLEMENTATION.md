# Fix Dependencies Button Implementation Summary

## Issue
Add a button to fix ML dependencies shown on the AI Models Setup page (http://127.0.0.1:8000/ai-models-setup)

## Solution Implemented

### 1. Backend API Endpoint
**File**: `src/backend/api/routes/setup.py`

Added new POST endpoint: `/api/v1/setup/ai-models/fix-dependencies`

**Functionality**:
- Runs `pip install -e .` to reinstall all project dependencies
- 10-minute timeout to handle large ML packages
- Returns stdout/stderr for transparency
- Logs installation progress

**Code**:
```python
@router.post("/ai-models/fix-dependencies")
async def fix_dependencies() -> Dict[str, Any]:
    """
    Install or update missing/outdated ML dependencies.
    
    Runs pip install to fix missing or outdated packages required for AI models.
    This includes torch, torchvision, diffusers, transformers, and other ML packages.
    """
    # Implementation details in the file
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

## Files Changed
1. `src/backend/api/routes/setup.py` - Added new endpoint
2. `ai_models_setup.html` - Added button and JavaScript handlers

## Screenshot
The button is visible in the ML Package Versions section when dependencies need fixing:

![Fix Dependencies Button](https://github.com/user-attachments/assets/95c45ae9-b14c-4d5a-a168-4738f4c80fd2)

## Implementation Notes
- Minimal changes approach - only added what was necessary
- Reused existing modal infrastructure for consistency
- Follows existing code patterns in the file
- No breaking changes to existing functionality
- User-friendly with clear confirmation and progress feedback
