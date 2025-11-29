# Image-to-Image Generation Implementation Guide

## Current State

The codebase currently has **partial support** for image-to-image (img2img) generation:

### What Works Now:
1. **Reference Image Path Support**: When `appearance_locked=True` and `base_image_path` is set, the system:
   - Passes `reference_image_path` to the AI generation pipeline
   - Sets `use_controlnet=True` flag
   - Logs that it's using visual reference for consistency

2. **Current Implementation** (`ai_models.py` lines 2192-2201):
   - Detects when reference image is provided
   - Currently falls back to **prompt-based consistency** 
   - Note in code: "ControlNet support requires additional setup"
   - Enhances prompt with: "maintaining consistent appearance from reference"

### What Doesn't Work Yet:
1. **Actual ControlNet Integration**: Not fully implemented
2. **Image Loading**: Reference image is not loaded into the pipeline
3. **img2img Pipeline**: Not using Stable Diffusion img2img pipeline
4. **ControlNet Models**: No ControlNet model loading

## Recommended Implementation

Based on the issue feedback, here are the recommended approaches:

### Option 1: Direct img2img (Simplest, Good Results)

Use Stable Diffusion's built-in img2img pipeline to start from the base image:

```python
from diffusers import StableDiffusionImg2ImgPipeline

# In _generate_image_diffusers method
if reference_image_path:
    # Load the reference image
    from PIL import Image
    init_image = Image.open(reference_image_path).convert("RGB")
    
    # Use img2img pipeline instead of text2img
    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(...)
    
    # Generate with init_image and strength parameter
    image = pipe(
        prompt=prompt,
        image=init_image,
        strength=0.7,  # 0.0 = no change, 1.0 = full regeneration
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
    ).images[0]
```

**Pros:**
- Simple to implement
- Good for maintaining overall composition and pose
- Works with existing SD models
- Lower denoising strength preserves more details from base image

**Cons:**
- Less precise control over specific features
- Harder to maintain exact appearance details

### Option 2: ControlNet (Most Control, Best Results)

Use ControlNet for precise control over the generation:

```python
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
from controlnet_aux import OpenposeDetector

# Load ControlNet model
controlnet = ControlNetModel.from_pretrained(
    "lllyasviel/sd-controlnet-openpose",  # or canny, depth, etc.
    torch_dtype=torch.float16
)

# Load pipeline with ControlNet
pipe = StableDiffusionControlNetPipeline.from_pretrained(
    model_path,
    controlnet=controlnet,
    torch_dtype=torch.float16
)

# Extract control image (pose, edges, depth, etc.)
if reference_image_path:
    reference_image = Image.open(reference_image_path)
    
    # Extract pose/structure from reference
    processor = OpenposeDetector.from_pretrained('lllyasviel/ControlNet')
    control_image = processor(reference_image)
    
    # Generate with control
    image = pipe(
        prompt=prompt,
        image=control_image,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
    ).images[0]
```

**Available ControlNet Types:**
- **openpose**: Control pose and body structure
- **canny**: Control edges and outlines  
- **depth**: Control depth and spatial layout
- **scribble**: Control with rough sketches
- **seg**: Control with segmentation maps

**Pros:**
- Most precise control over appearance consistency
- Can maintain specific pose, facial features, composition
- Works well for character consistency across images
- Multiple control types for different use cases

**Cons:**
- More complex to implement
- Requires additional model downloads (~1.5GB per ControlNet model)
- Higher computational cost

### Option 3: Hybrid Approach (Recommended)

Combine both methods for best results:

1. Use **img2img** for initial consistency (fast, simple)
2. Add **ControlNet** as optional enhancement for critical features
3. Let user configure per-persona which method to use

```python
# In PersonaModel, add field:
image_consistency_method = Column(
    String(20), 
    default="img2img",  # Options: "img2img", "controlnet", "both"
)

# In generation logic:
if persona.appearance_locked and persona.base_image_path:
    method = persona.image_consistency_method or "img2img"
    
    if method == "img2img":
        # Use direct img2img pipeline
        result = generate_with_img2img(...)
    
    elif method == "controlnet":
        # Use ControlNet pipeline
        result = generate_with_controlnet(...)
    
    elif method == "both":
        # Use img2img as base, then ControlNet for refinement
        result = generate_with_both(...)
```

## Implementation Priority

### Phase 1: Basic img2img (Recommended for immediate fix)
- [ ] Replace prompt-based consistency with actual img2img pipeline
- [ ] Load and use reference image when provided
- [ ] Add strength parameter (configurable per-persona)
- [ ] Test with existing personas that have base_image_path set

### Phase 2: ControlNet Integration (Optional enhancement)
- [ ] Add ControlNet model support
- [ ] Implement pose/structure extraction from reference images
- [ ] Add controlnet type selection to persona config
- [ ] Add UI controls for ControlNet parameters

### Phase 3: Advanced Features
- [ ] Face ID preservation using IP-Adapter
- [ ] Multi-ControlNet support (combine pose + depth)
- [ ] Reference image gallery (multiple reference images)
- [ ] Automatic best-method selection based on use case

## Code Changes Required

### 1. Update ai_models.py

```python
async def _generate_image_diffusers(
    self, prompt: str, model: Dict[str, Any], **kwargs
) -> Dict[str, Any]:
    """Generate image using Diffusers library with optional reference image."""
    
    reference_image_path = kwargs.get("reference_image_path")
    use_img2img = reference_image_path is not None
    strength = kwargs.get("img2img_strength", 0.7)  # Default strength
    
    if use_img2img:
        from diffusers import StableDiffusionImg2ImgPipeline
        from PIL import Image
        
        # Load reference image
        init_image = Image.open(reference_image_path).convert("RGB")
        init_image = init_image.resize((width, height))
        
        # Use img2img pipeline
        pipe = StableDiffusionImg2ImgPipeline.from_pretrained(...)
        
        image = pipe(
            prompt=prompt,
            image=init_image,
            strength=strength,
            negative_prompt=negative_prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator,
        ).images[0]
    else:
        # Standard text2img pipeline
        ...
```

### 2. Update PersonaModel

Add configuration for img2img parameters:

```python
class PersonaModel(Base):
    # ... existing fields ...
    
    # Image consistency configuration
    image_consistency_method = Column(
        String(20), 
        default="img2img",
        nullable=False
    )
    img2img_strength = Column(
        Float, 
        default=0.7,  # 0.0-1.0, lower = more similar to reference
        nullable=False
    )
    use_controlnet = Column(
        Boolean,
        default=False,
        nullable=False
    )
    controlnet_type = Column(
        String(20),
        default="openpose",  # openpose, canny, depth, etc.
        nullable=True
    )
```

### 3. Update Content Generation Service

Already correct - passes reference_image_path when appearance_locked. Just needs to work when ai_models receives it.

## Testing Plan

1. **Test img2img basic functionality:**
   - Create persona with base_image_path and appearance_locked=True
   - Generate new content
   - Verify output maintains visual consistency with base image

2. **Test img2img strength parameter:**
   - Test with strength=0.3 (very similar to reference)
   - Test with strength=0.7 (moderate changes)
   - Test with strength=0.9 (mostly new image)

3. **Test without reference (backward compatibility):**
   - Ensure text2img still works when no reference provided
   - Verify no breakage for personas without base_image_path

## Performance Considerations

- **img2img is faster than text2img**: Fewer denoising steps needed
- **ControlNet adds overhead**: ~20-30% more generation time
- **Cache pipelines**: Don't reload for every generation
- **GPU memory**: img2img uses slightly more VRAM than text2img

## Security Considerations

- **Validate reference image paths**: Prevent directory traversal
- **Check image file types**: Only allow safe formats (PNG, JPG, WebP)
- **Scan for malicious content**: If accepting user-uploaded reference images
- **File size limits**: Prevent memory exhaustion from huge images

## Documentation Updates Needed

1. Update API docs to document reference_image_path parameter
2. Add persona configuration guide for img2img settings
3. Create tutorial for setting up ControlNet models
4. Document best practices for base image selection

## Related Files

- `src/backend/services/ai_models.py` - Main generation logic
- `src/backend/services/content_generation_service.py` - Content generation orchestration
- `src/backend/models/persona.py` - Persona configuration
- `src/backend/models/content.py` - Content metadata

## References

- [Diffusers img2img Documentation](https://huggingface.co/docs/diffusers/using-diffusers/img2img)
- [ControlNet Models](https://huggingface.co/lllyasviel)
- [Stable Diffusion WebUI img2img](https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/Features#img2img)
