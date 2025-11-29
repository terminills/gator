# AI Content Generation Guide

## Overview

This guide covers the AI-powered content generation features implemented to overcome the 77-token prompt limitation and enable full self-optimization through llama.cpp integration.

## Features Implemented

### 1. Long Prompt Support (>77 Tokens) with Compel

**Problem**: CLIP tokenizer in Stable Diffusion models has a 77-token limit, truncating detailed prompts.

**Solution**: Integrated `compel` library for SDXL models to handle prompts of any length.

**How It Works**:
- Automatically detects prompts >75 tokens
- Uses compel to generate embeddings from dual SDXL text encoders
- Bypasses CLIP's 77-token limitation
- Falls back gracefully if compel unavailable

**Location**: `src/backend/services/ai_models.py`

```python
# Automatic long prompt detection and handling
if is_sdxl and estimated_tokens > 75:
    from compel import Compel, ReturnedEmbeddingsType
    
    compel = Compel(
        tokenizer=[pipe.tokenizer, pipe.tokenizer_2],
        text_encoder=[pipe.text_encoder, pipe.text_encoder_2],
        ...
    )
    
    prompt_embeds, pooled = compel(prompt)
```

### 2. AI-Powered Image Prompt Generation

**Purpose**: Generate detailed, contextual prompts using llama.cpp based on persona characteristics.

**Service**: `src/backend/services/prompt_generation_service.py`

**Features**:
- Uses llama.cpp for AI-powered prompt generation
- Analyzes persona: appearance, personality, interests, preferences
- Considers context and style requirements
- Generates 100-200 word prompts (far exceeding 77 tokens)
- Falls back to templates if llama.cpp unavailable

**Usage**:
```python
from backend.services.prompt_generation_service import get_prompt_service

service = get_prompt_service()
result = await service.generate_image_prompt(
    persona=persona,
    context="beach sunset photo shoot",
    content_rating=ContentRating.SFW,
    image_style="photorealistic",
    use_ai=True
)

# result contains:
# - prompt: detailed image generation prompt
# - negative_prompt: style-appropriate negative prompt
# - style: image style used
# - source: 'ai_generated' or 'template'
# - word_count: number of words in prompt
```

**Integration**: Automatically used in `ContentGenerationService._generate_image()`

### 3. Personality-Based Chat Responses

**Purpose**: Generate chat responses that reflect the persona's defined personality traits.

**Service**: `src/backend/services/persona_chat_service.py`

**Features**:
- Uses llama.cpp for natural language generation
- Based on persona's personality, appearance, interests
- Includes conversation history for context
- Generates character-appropriate responses
- Falls back to template responses if AI unavailable

**Usage**:
```python
from backend.services.persona_chat_service import get_persona_chat_service

service = get_persona_chat_service()
response = await service.generate_response(
    persona=persona,
    user_message="Tell me about your interests",
    conversation_history=message_history,
    use_ai=True
)
```

**Integration**: Used in direct messaging background task at `src/backend/api/routes/direct_messaging.py`

### 4. AI-Powered Persona Generation

**Purpose**: Create coherent, realistic personas using AI instead of random templates.

**Service**: `src/backend/services/ai_persona_generator.py`

**Features**:
- Uses llama.cpp to generate internally consistent personas
- Creates appearance, personality, interests that fit together
- More realistic than random template combinations
- Supports persona type hints (fitness, tech, fashion, etc.)
- Falls back to template generation if AI unavailable

**Usage**:
```python
from backend.services.ai_persona_generator import get_ai_persona_generator

generator = get_ai_persona_generator()
persona_config = await generator.generate_persona(
    name="Custom Name",  # Optional
    persona_type="fitness",  # Optional hint
    use_ai=True
)

# persona_config contains complete persona configuration
```

**Integration**: Used in random persona endpoint `POST /api/v1/personas/random`

## Setup Instructions

### Prerequisites

1. **Python 3.9+** with dependencies installed:
   ```bash
   pip install -e .
   ```

2. **Compel library** for long prompt support:
   ```bash
   pip install compel
   ```

3. **llama.cpp** (optional but recommended for AI features):
   - Download from: https://github.com/ggerganov/llama.cpp
   - Build and install `llama-cli` or `main` binary
   - Ensure binary is in PATH

4. **Language Model** (for llama.cpp):
   - Download a GGUF model (e.g., Llama 3.1 8B, Qwen 2.5, Mixtral)
   - Place in one of these locations:
     - `./models/text/llama-3.1-8b/`
     - `./models/llama-3.1-8b/`
     - `./models/text/qwen2.5-72b/`
     - `./models/qwen2.5-72b/`

### Recommended Models

For optimal performance with different tasks:

| Task | Recommended Model | Size | Reason |
|------|------------------|------|---------|
| Prompt Generation | Llama 3.1 8B | 8GB | Fast, creative, good instruction following |
| Chat Responses | Llama 3.1 8B | 8GB | Natural conversation, personality aware |
| Persona Generation | Qwen 2.5 72B | 72GB | More coherent, better understanding |
| Fast Testing | Llama 3.1 8B | 8GB | Quick responses, good enough quality |

### Model Installation Example

```bash
# Create models directory
mkdir -p models/text/llama-3.1-8b

# Download model (example using huggingface-cli)
huggingface-cli download \
  TheBloke/Llama-2-7B-Chat-GGUF \
  llama-2-7b-chat.Q4_K_M.gguf \
  --local-dir models/text/llama-3.1-8b

# Verify installation
ls models/text/llama-3.1-8b/*.gguf
```

## Testing

Run the test suite to verify all features:

```bash
python test_ai_enhancements.py
```

Expected output:
- ✅ All services initialized
- ✅ Compel library available
- ✓ Prompt generation working
- ✓ Chat service working
- ✓ Persona generator working

## Configuration

### Environment Variables

```bash
# Enable llama.cpp for specific services (optional, auto-detected by default)
export LLAMA_CPP_PATH=/path/to/llama-cli

# Disable AI features (force template fallback)
export USE_AI_GENERATION=false
```

### Model Selection

Models are automatically detected in this priority order:
1. `./models/text/llama-3.1-8b/`
2. `./models/llama-3.1-8b/`
3. `./models/text/qwen2.5-72b/`
4. `./models/qwen2.5-72b/`
5. Other models in `./models/text/` or `./models/`

## API Usage Examples

### Generate Image with Long Prompt

```bash
# The prompt can now exceed 77 tokens thanks to compel
curl -X POST http://localhost:8000/api/v1/content/generate \
  -H "Content-Type: application/json" \
  -d '{
    "persona_id": "uuid-here",
    "content_type": "image",
    "prompt": "beach sunset with dramatic lighting",
    "quality": "hd"
  }'

# Check logs for:
# "Using compel for long prompt support (~N tokens)"
# "Generated prompt (X words, source: ai_generated)"
```

### Test Persona Chat

```bash
# Send message to persona
curl -X POST http://localhost:8000/api/v1/dm/messages \
  -H "Content-Type: application/json" \
  -d '{
    "conversation_id": "uuid-here",
    "sender": "user",
    "content": "Tell me about your hobbies"
  }'

# Persona responds automatically in background
# Response will reflect persona's personality
```

### Generate AI Persona

```bash
# Generate random persona with AI
curl -X POST http://localhost:8000/api/v1/personas/random \
  -H "Content-Type: application/json"

# Returns coherent persona with:
# - Internally consistent appearance
# - Matching personality traits
# - Relevant interests
```

## Architecture

### Service Layer

```
┌─────────────────────────────────────────────────┐
│         Content Generation Service              │
│  (orchestrates image/text/chat generation)      │
└──────────────────┬──────────────────────────────┘
                   │
         ┌─────────┴──────────┐
         │                    │
         ▼                    ▼
┌────────────────┐   ┌─────────────────┐
│ Prompt         │   │ Persona Chat    │
│ Generation     │   │ Service         │
│ Service        │   │                 │
│ (llama.cpp)    │   │ (llama.cpp)     │
└────────┬───────┘   └────────┬────────┘
         │                    │
         ▼                    ▼
┌─────────────────────────────────────┐
│      AI Models Service              │
│  (SDXL + compel for long prompts)   │
└─────────────────────────────────────┘
```

### Data Flow: Image Generation

```
1. User requests image generation
   ↓
2. Content Generation Service receives request
   ↓
3. Prompt Generation Service called
   ├─→ Analyzes persona (appearance, personality, interests)
   ├─→ Considers context and style
   ├─→ Uses llama.cpp to generate detailed prompt (100-200 words)
   └─→ Returns prompt + negative prompt
   ↓
4. AI Models Service generates image
   ├─→ Detects prompt length (e.g., 150 tokens)
   ├─→ Uses compel to encode long prompt
   ├─→ Generates image with SDXL
   └─→ Returns image
   ↓
5. Image saved and returned to user
```

### Fallback Strategy

Every AI component has graceful fallbacks:

```
Try llama.cpp (AI generation)
    ↓
    ✗ (not available/failed)
    ↓
Use template-based generation
    ↓
    ✓ Always succeeds
```

## Performance Considerations

### Generation Times

| Operation | AI Mode | Template Mode |
|-----------|---------|---------------|
| Image Prompt | 2-5s | <0.1s |
| Chat Response | 1-3s | <0.1s |
| Persona Generation | 3-8s | <0.1s |
| Image Generation | 10-30s | 10-30s |

### Optimization Tips

1. **Model Size**: Smaller models (7-8B) are faster but less creative
2. **Quantization**: Q4_K_M provides good speed/quality balance
3. **CPU Threads**: Set `-t` parameter based on available cores
4. **Caching**: Models are cached after first load
5. **Batch Processing**: Generate multiple items concurrently

## Troubleshooting

### Issue: "llama.cpp not found"
**Solution**: Install llama.cpp and ensure binary is in PATH
```bash
which llama-cli  # Should return path
export PATH=$PATH:/path/to/llama-cpp
```

### Issue: "No llama.cpp model found"
**Solution**: Download and place model in correct location
```bash
ls models/text/llama-3.1-8b/*.gguf
# Should show at least one .gguf file
```

### Issue: "compel library not available"
**Solution**: Install compel
```bash
pip install compel
python -c "import compel; print('OK')"
```

### Issue: Prompts still truncated
**Solution**: Verify SDXL model is being used and check logs
```bash
# Look for these log messages:
# "Using compel for long prompt support"
# "Image generated successfully via text2img with compel embeddings"
```

### Issue: AI responses are generic
**Solution**: 
1. Verify persona personality is defined
2. Check llama.cpp is working (run test script)
3. Try larger model (e.g., 70B instead of 8B)

## Future Enhancements

Planned improvements:

1. **RSS Feed Integration**: Include RSS content in prompt generation
2. **Multi-Model Support**: Different models for different tasks
3. **Prompt Refinement**: Learn from generation feedback
4. **Context Caching**: Cache conversation context for faster responses
5. **Model Selection API**: Allow users to choose specific models
6. **Prompt Templates**: Customizable prompt templates per persona
7. **A/B Testing**: Compare AI vs template generation quality

## Security Considerations

1. **Input Validation**: All user inputs are sanitized before AI processing
2. **Content Filtering**: NSFW content rating enforced
3. **Resource Limits**: Timeouts prevent infinite generation
4. **Model Isolation**: Models run in separate processes
5. **API Rate Limiting**: Prevents abuse of AI features

## Support

For issues or questions:
- Check logs: `tail -f logs/gator.log`
- Run test suite: `python test_ai_enhancements.py`
- Review service code: `src/backend/services/`
- File issue on GitHub with log excerpts

## Credits

- **compel**: https://github.com/damian0815/compel
- **llama.cpp**: https://github.com/ggerganov/llama.cpp
- **Stable Diffusion**: https://github.com/Stability-AI/stablediffusion
- **Diffusers**: https://github.com/huggingface/diffusers
