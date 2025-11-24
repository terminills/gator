# Using Ollama for Diagnostics Chat

The `/admin/diagnostics` page includes an AI-powered chat interface with Gator, the platform's help agent. This chat can now use Ollama for local AI text generation, providing a fast and private chat experience.

## What is Ollama?

[Ollama](https://ollama.com/) is a local AI runtime that makes it easy to run large language models on your own machine. It's optimized for:
- **Fast inference** - Optimized for conversational AI
- **Privacy** - All processing happens locally
- **Easy setup** - Simple installation and model management
- **GPU acceleration** - Works with NVIDIA, AMD, and Apple Silicon

## Quick Start

### 1. Install Ollama

Visit [https://ollama.com/download](https://ollama.com/download) and follow the installation instructions for your platform:

- **macOS**: `brew install ollama` or download the app
- **Linux**: `curl -fsSL https://ollama.com/install.sh | sh`
- **Windows**: Download from the website

### 2. Pull a Model

After installing Ollama, pull a language model. We recommend starting with `llama2` or `mistral`:

```bash
# Recommended for chat (fast, good quality)
ollama pull llama2

# Alternative: Mistral (slightly larger, more capable)
ollama pull mistral

# Lightweight option for lower-end hardware
ollama pull phi
```

### 3. Verify Installation

Check that Ollama is working and see available models:

```bash
ollama list
```

You should see your pulled models listed.

### 4. Restart Gator Platform

If the Gator platform was already running, restart it to detect Ollama:

```bash
# Stop the current server (Ctrl+C)
# Then restart:
cd src && python -m backend.api.main
```

## How It Works

When you use the chat on `/admin/diagnostics`:

1. **Automatic Detection**: On startup, Gator detects if Ollama is installed and which models are available
2. **Model Registration**: Each Ollama model is automatically registered as a text generation option
3. **Smart Selection**: For chat messages, Ollama models are prioritized over other local models
4. **Fallback**: If Ollama isn't available, the platform falls back to rule-based responses

## Checking Ollama Status

You can verify Ollama integration in several ways:

### Via Logs

When the platform starts, look for these messages:

```
🦙 Ollama installation detected
Ollama version: 0.12.10
Ollama binary: /usr/local/bin/ollama
Server running: True
Found 2 Ollama model(s):
  • llama2
  • mistral
✓ Registered 2 Ollama model(s) for text generation
```

### Via API

Check the AI models status endpoint:

```bash
curl http://localhost:8000/api/v1/setup/models/status
```

Look for models with `"inference_engine": "ollama"` in the response.

### Via Chat

Try asking a question in the diagnostics chat. If Ollama is working, you'll see:
- Fast response times (< 2 seconds for most questions)
- Natural, contextual answers
- CLI mode output showing "Using Ollama model"

## Recommended Models for Chat

| Model | Size | Speed | Quality | Best For |
|-------|------|-------|---------|----------|
| `phi` | 2.7GB | ⚡⚡⚡ | ⭐⭐ | Quick responses, limited hardware |
| `llama2` | 3.8GB | ⚡⚡ | ⭐⭐⭐ | General chat, balanced |
| `mistral` | 4.1GB | ⚡⚡ | ⭐⭐⭐⭐ | High quality responses |
| `llama3` | 4.7GB | ⚡ | ⭐⭐⭐⭐⭐ | Best quality, slower |

## Troubleshooting

### Ollama Not Detected

If Ollama isn't detected after installation:

1. **Check Installation**:
   ```bash
   which ollama
   ollama --version
   ```

2. **Ensure Server is Running**:
   ```bash
   ollama serve
   ```
   (Usually runs automatically, but manual start may be needed)

3. **Verify Models**:
   ```bash
   ollama list
   ```
   If no models are listed, pull one: `ollama pull llama2`

4. **Restart Gator Platform**: The platform checks for Ollama on startup

### Slow Response Times

If chat responses are slow:

1. **Hardware**: Ollama performs best with:
   - 8GB+ RAM
   - GPU with 4GB+ VRAM (optional but recommended)

2. **Model Size**: Try a smaller model:
   ```bash
   ollama pull phi  # Lightweight 2.7GB model
   ```

3. **Check GPU**: Verify GPU acceleration is working:
   ```bash
   ollama run llama2 "test"  # Should show GPU in logs
   ```

### "No text models available" Error

This means neither Ollama nor other AI models are available:

1. **Install Ollama** (see Quick Start above)
2. **Or** configure other local models (llama.cpp, vLLM)
3. **Or** enable cloud APIs (OpenAI, Anthropic) via environment variables

## Advanced Configuration

### Using Specific Ollama Models

By default, the first available Ollama model is used. To prefer a specific model, you can:

1. **Remove other models** (Ollama will use the only available one):
   ```bash
   ollama rm mistral
   # Now only llama2 will be used
   ```

2. **Model ordering**: Ollama models are registered in the order returned by `ollama list`

### GPU Compatibility

Ollama is automatically preferred for certain GPUs where it provides better compatibility:
- AMD GPUs (ROCm)
- Apple Silicon (M1/M2/M3)
- Older NVIDIA GPUs

The platform's GPU detection automatically recommends Ollama when appropriate.

## Benefits of Using Ollama

1. **Privacy**: All chat happens locally, no data sent to external APIs
2. **Speed**: Optimized for interactive chat, faster than some alternatives
3. **Cost**: Free, no API costs
4. **Offline**: Works without internet connection
5. **Simple**: Easy installation and model management

## Learn More

- [Ollama Documentation](https://github.com/ollama/ollama)
- [Available Models](https://ollama.com/library)
- [Ollama API Reference](https://github.com/ollama/ollama/blob/main/docs/api.md)

## Support

For issues with Ollama integration:
1. Check the platform logs for Ollama detection messages
2. Verify Ollama is working: `ollama run llama2 "test"`
3. Report issues on the Gator repository with log output
