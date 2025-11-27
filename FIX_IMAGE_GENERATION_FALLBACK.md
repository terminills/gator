# Image Generation Fallback Mechanism Fix

## Problem Statement

When attempting to generate images, the system failed with the error:
```
SDXL pipeline has None text encoders: text_encoder=True, text_encoder_2=False
```

Even though `stable-diffusion-v1-5` was available as an alternative, the system raised an error immediately instead of trying the alternative model.

## Root Cause

The issue occurred because:

1. **Incomplete SDXL Model**: The SDXL model on disk was marked as "loaded=True" but had incomplete components (missing `text_encoder_2`)
2. **No Fallback Mechanism**: When model loading/validation failed, the error was raised immediately
3. **Model Selection Preference**: The system preferred larger models (SDXL 7GB) over smaller ones (SD1.5 4GB)
4. **Single Attempt**: Only one model was attempted before failing

## Solution

Implemented a robust fallback mechanism in the `generate_image` method that:

1. **Tries Models in Order**: Attempts image generation with available models sequentially
2. **Tracks Failed Models**: Maintains a list of failed models to avoid retrying them
3. **Provides Helpful Context**: Logs informative error messages for incomplete models
4. **Succeeds When Possible**: Only raises an error if ALL available models fail

### Code Changes

**File**: `src/backend/services/ai_models.py`

**Key Changes**:
- Wrapped model generation in a retry loop
- Added `failed_models` list to track unsuccessful attempts
- Enhanced error logging with context about incomplete models
- Added `failed_models` to benchmark data for debugging

### Example Behavior

**Before Fix**:
```
1. Try SDXL → Fails (incomplete model)
2. Raise error immediately
❌ Generation fails
```

**After Fix**:
```
1. Try SDXL → Fails (incomplete model)
   ⚠️  Log warning with helpful context
2. Try SD 1.5 → Success!
   ✅ Return generated image
   📊 Benchmark data includes: failed_models=["sdxl-1.0"]
```

## Testing

### New Tests Created

Created `tests/unit/test_image_generation_fallback.py` with 5 comprehensive tests:

1. **test_fallback_to_alternative_model_on_sdxl_failure**
   - Verifies fallback from SDXL to SD 1.5 when SDXL has incomplete components

2. **test_fallback_records_failed_models**
   - Ensures failed models are properly tracked in benchmark data

3. **test_all_models_fail_raises_last_error**
   - Confirms proper error handling when all models fail

4. **test_incomplete_model_error_logging**
   - Validates helpful error messages for incomplete models

5. **test_single_model_success_no_fallback**
   - Ensures no unnecessary fallback when first model succeeds

### Test Results

✅ **All 47 AI model tests pass**:
- 13 existing image generation tests
- 4 fp16 fallback tests
- 5 new fallback mechanism tests
- 25 video generation tests

### Security Analysis

✅ **CodeQL Scan Results**:
- 1 pre-existing false positive (logging image dimensions)
- No new security vulnerabilities introduced

## Usage

The fix is transparent to users. When generating images:

```python
# If SDXL fails, system automatically tries SD 1.5
result = await manager.generate_image(
    "A serene mountain landscape at sunset, digital art",
    width=512,
    height=512,
)

# Benchmark data includes fallback information
print(result["benchmark_data"]["failed_models"])  # ["sdxl-1.0"]
print(result["benchmark_data"]["model_selected"])  # "stable-diffusion-v1-5"
```

## Benefits

1. **Improved Reliability**: System works even with incomplete model files
2. **Better User Experience**: Users get results instead of cryptic errors
3. **Helpful Diagnostics**: Clear logs indicate which models failed and why
4. **Debugging Support**: Benchmark data tracks fallback attempts
5. **No Breaking Changes**: All existing tests pass

## Recommendations

For users encountering incomplete model errors:

1. **Re-download Models**: Consider re-downloading models with missing components
2. **Check Disk Space**: Ensure sufficient space for complete model downloads
3. **Verify Integrity**: Use model verification tools if available
4. **Monitor Logs**: Check logs for fallback warnings to identify problematic models

## Related Files

- `src/backend/services/ai_models.py` - Main implementation
- `tests/unit/test_image_generation_fallback.py` - Comprehensive test suite
- `tests/unit/test_ai_image_generation.py` - Existing tests (all passing)
- `tests/unit/test_ai_image_generation_fp16_fallback.py` - FP16 tests (all passing)
