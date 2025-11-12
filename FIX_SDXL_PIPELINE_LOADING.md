# Fix: SDXL Pipeline Loading Error

## Issue
Image generation was failing with the error:
```
'NoneType' object has no attribute 'tokenize'
```

This occurred when trying to generate images with SDXL models.

## Root Cause
The code was passing `safety_checker=None` and `requires_safety_checker=False` parameters to **all** Stable Diffusion models when loading from Diffusers:

```python
load_args = {
    "torch_dtype": torch.float16 if "cuda" in device else torch.float32,
    "safety_checker": None,  # ❌ Not compatible with SDXL
    "requires_safety_checker": False,  # ❌ Not compatible with SDXL
}
```

However, these parameters are only valid for **StableDiffusionPipeline** (SD 1.5/2.x), not for **StableDiffusionXLPipeline** (SDXL models).

SDXL models have a different architecture with:
- Two text encoders (`text_encoder` and `text_encoder_2`)
- Two tokenizers (`tokenizer` and `tokenizer_2`)
- No safety_checker component

When these incompatible parameters were passed to SDXL's `from_pretrained`, it could cause the text encoder components to not load properly, resulting in None tokenizers.

## Solution
The fix conditionally adds `safety_checker` parameters **only** for non-SDXL models:

```python
load_args = {
    "torch_dtype": (
        torch.float16 if "cuda" in device else torch.float32
    ),
}

# Only add safety_checker params for SD 1.5 models (not SDXL)
if not is_sdxl:
    load_args["safety_checker"] = None  # Disable for performance
    load_args["requires_safety_checker"] = False  # Suppress warning
```

This ensures:
- ✅ SDXL models load without incompatible parameters
- ✅ SD 1.5 models still get safety_checker disabled for performance
- ✅ Both model types load correctly with their respective pipeline classes

## Files Changed
1. **src/backend/services/ai_models.py**
   - Modified `_generate_image_diffusers` method
   - Applied fix to both local path loading (line 1963-1966)
   - Applied fix to HuggingFace Hub loading (line 1998-2001)

2. **tests/unit/test_sdxl_safety_checker_fix.py**
   - Added comprehensive tests to verify the fix
   - Tests SDXL models don't receive safety_checker params
   - Tests SD 1.5 models do receive safety_checker params

## Testing
All tests pass:
- ✅ 13 existing image generation tests
- ✅ 4 existing fp16 fallback tests  
- ✅ 3 new safety_checker fix tests
- ✅ **Total: 20 tests passed**

## Model Compatibility
| Model Type | Pipeline Class | safety_checker | requires_safety_checker |
|------------|----------------|----------------|-------------------------|
| SD 1.5 | StableDiffusionPipeline | ✅ Supported | ✅ Supported |
| SD 2.x | StableDiffusionPipeline | ✅ Supported | ✅ Supported |
| SDXL | StableDiffusionXLPipeline | ❌ Not supported | ❌ Not supported |

## Detection Logic
Models are identified as SDXL using:
```python
is_sdxl = "xl" in model_name.lower() or "xl" in model_id.lower()
```

This correctly identifies:
- `sdxl-1.0` → SDXL ✅
- `sdxl-turbo` → SDXL ✅
- `stable-diffusion-v1-5` → SD 1.5 ✅
- `stable-diffusion-v2-1` → SD 2.x ✅

## Verification
The fix has been verified to:
1. Correctly detect SDXL vs SD 1.5 models
2. Build appropriate load_args for each model type
3. Pass all existing and new tests
4. Not introduce any security vulnerabilities (CodeQL scan: 0 alerts)
