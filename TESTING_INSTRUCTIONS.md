# Testing Instructions for Image Generation Fix

## What Was Fixed

The image generation was failing with `'NoneType' object has no attribute 'tokenize'` when using SDXL models. The fix adds:

1. **Component Validation**: Checks that all required pipeline components are loaded
2. **Diagnostic Logging**: Provides detailed information about pipeline state and parameters
3. **Error Handling**: Better error messages with full context

## How to Test on Real Hardware

### Step 1: Run the Test Script

```bash
cd /home/terminills/Desktop/gator
python test_local_image_generation.py
```

### Step 2: Review the Diagnostic Output

You should now see detailed diagnostic information like:

```
============================================================
DIFFUSERS GENERATION - DIAGNOSTIC INFO
============================================================
Model: sdxl-1.0 (SDXL=True)
Pipeline class: StableDiffusionXLPipeline
Device: cuda:0
Pipeline components:
  - vae: AutoencoderKL
  - text_encoder: CLIPTextModel
  - text_encoder_2: CLIPTextModelWithProjection
  - tokenizer: CLIPTokenizer
  - tokenizer_2: CLIPTokenizer
  - unet: UNet2DConditionModel
  - scheduler: DPMSolverMultistepScheduler
Generation parameters:
  - prompt: A serene mountain landscape at sunset, digital art
  - negative_prompt: ugly, blurry, low quality, distorted
  - num_inference_steps: 20
  - guidance_scale: 7.5
  - width: 512
  - height: 512
  - seed: None
============================================================
```

### Step 3: Diagnose Any Issues

#### If you see "None" for any component:

**Example:**
```
  - text_encoder_2: None  <-- PROBLEM!
```

This means the SDXL model files are incomplete or corrupted. Solutions:
1. Delete the model directory: `rm -rf ~/.cache/huggingface/hub/models--stabilityai--stable-diffusion-xl-base-1.0/`
2. Re-run the test to download fresh model files

#### If validation fails before generation:

You'll see an error like:
```
ERROR: SDXL pipeline has None text encoders: text_encoder=True, text_encoder_2=False
```

This indicates which specific component failed to load. The broken pipeline has been removed from cache automatically.

#### If you see the full diagnostic output but generation still fails:

The error will now include:
- Error type (e.g., `AttributeError`, `RuntimeError`)
- Full traceback
- All generation parameters that were used

This information will help identify if it's:
- A memory issue (OOM)
- A CUDA issue
- A model compatibility issue
- Something else

## Expected Output on Success

```
🎨 Testing Local Image Generation
============================================================

1. Initializing AI Model Manager...
   - GPU Type: rocm
   - GPU Memory: 59.96875 GB
   - CPU Cores: 128

2. Initializing models...

3. Available image models:
   - stable-diffusion-v1-5: provider=local, loaded=False, can_load=True
   - sdxl-1.0: provider=local, loaded=True, can_load=True
   - flux.1-dev: provider=local, loaded=False, can_load=True

4. Testing image generation with stable-diffusion-v1-5...
   Prompt: A serene mountain landscape at sunset, digital art

============================================================
DIFFUSERS GENERATION - DIAGNOSTIC INFO
============================================================
[... diagnostic output ...]
============================================================

✓ Image generated successfully

✅ Image generated successfully!
   - Size: 245678 bytes
   - Format: PNG
   - Model: sdxl-1.0
   - Dimensions: 512x512
   - Saved to: test_generated_image.png
```

## Common Issues and Solutions

### Issue 1: "fp16 variant not available"
**Symptom:** Warning about fp16 variant
**Impact:** Non-critical warning, model loads without fp16
**Action:** No action needed, this is expected

### Issue 2: "Keyword arguments ... will be ignored"
**Symptom:** Warning about safety_checker parameters
**Impact:** Non-critical warning, parameters are safely ignored
**Action:** No action needed, this is expected for SDXL models

### Issue 3: "None text_encoder" or "None tokenizer"
**Symptom:** Validation error before generation
**Impact:** Generation cannot proceed
**Action:** 
1. Clear model cache
2. Re-download model
3. Check disk space
4. Check file permissions

### Issue 4: CUDA OOM (Out of Memory)
**Symptom:** RuntimeError about CUDA memory
**Impact:** Generation fails
**Action:**
1. Reduce image size (try 256x256 instead of 512x512)
2. Reduce num_inference_steps (try 15 instead of 25)
3. Close other GPU-using applications
4. Try SD 1.5 instead of SDXL (requires less VRAM)

## Providing Feedback

When reporting issues, please include:

1. **Full diagnostic output** (the section between the === lines)
2. **Complete error message** (if any)
3. **Model being used** (SD 1.5, SDXL, etc.)
4. **Available GPU memory** (from the initialization output)
5. **Any previous errors** that might have left the pipeline in a bad state

With this diagnostic information, we can quickly identify and fix the root cause!
