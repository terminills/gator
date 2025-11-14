# SDXL Long Prompt Pipeline Implementation

## Overview

This document describes the implementation of SDXL Long Prompt Weighting pipeline support to fix prompt truncation issues when generating images with Stable Diffusion XL.

## Problem Statement

The original issue was that SDXL prompts were being truncated at 77 tokens, even though SDXL has dual CLIP encoders. The logs showed:

```
Token indices sequence length is longer than the specified maximum sequence length for this model (131 > 77)
The following part of your input was truncated because CLIP can only handle sequences up to 77 tokens
```

This was happening because the standard `StableDiffusionXLPipeline` doesn't properly merge embeddings from both encoders for long prompts.

## Solution Implemented

### 1. SDXL Long Prompt Weighting Pipeline

For SDXL text2img generation, we now use the community pipeline `lpw_stable_diffusion_xl`:

```python
load_args["custom_pipeline"] = "lpw_stable_diffusion_xl"
pipe = DiffusionPipeline.from_pretrained(model_path, **load_args)
```

This pipeline:
- Chunks long prompts into segments
- Processes each segment through CLIP encoders
- Merges embeddings properly
- Avoids truncation warnings
- Increases identity stability and persona fidelity

### 2. Compel Library for ControlNet and img2img

Since there's no built-in long prompt pipeline for ControlNet, we use the `compel` library:

```python
from compel import Compel, ReturnedEmbeddingsType

compel = Compel(
    tokenizer=[pipe.tokenizer, pipe.tokenizer_2],
    text_encoder=[pipe.text_encoder, pipe.text_encoder_2],
    returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED,
    requires_pooled=[False, True],
    device=device
)

conditioning, pooled = compel(prompt)
```

### 3. Automatic Fallback

If the long prompt pipeline fails to load, the code automatically falls back to the standard pipeline with compel support.

### 4. Enhanced xformers Logging

Added helpful logging when xformers is not available with installation instructions.

## Expected Log Output

### Successful Long Prompt Pipeline Load

```
Using SDXL Long Prompt Weighting pipeline (lpw_stable_diffusion_xl)
✓ Supports prompts > 77 tokens without truncation via prompt chunking
```

### Fallback to Compel

```
Long Prompt Weighting pipeline not available
Falling back to standard SDXL pipeline with compel support
Using compel for long prompt support (~131 tokens)
```

## Performance Improvements

### With xformers

To improve performance, install xformers:

```bash
pip install xformers
```

Expected speedup: 38s → 10-12s (3-4x faster) for ControlNet generation

## Dependencies

- `diffusers>=0.28.0` - Pipeline support
- `compel>=2.0.0` - Long prompt embeddings  
- `transformers>=4.41.0` - CLIP tokenizers
- `xformers` (optional) - Performance optimization

---

**Gator don't play no shit** - We fixed the prompt truncation! 🐊
