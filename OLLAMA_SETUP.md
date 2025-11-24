# Ollama Setup Guide

This guide explains how to set up Ollama as a fallback for llama.cpp in the Gator AI platform.

## What is Ollama?

Ollama is a local LLM runtime that provides a simple way to run large language models on your machine. It can serve as a reliable fallback when llama.cpp encounters issues like floating point exceptions or model compatibility problems.

## Installation

### Linux

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

### macOS

```bash
brew install ollama
```

### Windows

Download the installer from [ollama.com/download](https://ollama.com/download)

## Verify Installation

Check that Ollama is installed:

```bash
ollama --version
```

Expected output:
```
ollama version is 0.12.10
```

## Start Ollama Server

Ollama runs as a background service. To start it:

```bash
ollama serve
```

Or on systems with systemd:

```bash
systemctl start ollama
```

## Pull Models

Before using Ollama, you need to pull the models you want to use:

```bash
# Pull a small model for testing
ollama pull llama3:8b

# Pull larger models if you have the resources
ollama pull qwen3-vl:32b
ollama pull codellama:34b
```

### List Available Models

To see what models are available:

```bash
ollama list
```

Example output:
```
NAME                                 ID              SIZE     MODIFIED    
qwen3-vl:32b                         ff2e46876908    20 GB    2 weeks ago    
codellama:34b                        685be00e1532    19 GB    3 weeks ago
llama3:8b                            4fa76c321234    4.7 GB   1 day ago
```

## Test Ollama

Test that Ollama can generate text:

```bash
ollama run llama3:8b "Write a short poem about AI"
```

## Integration with Gator

### Automatic Fallback

The Gator platform automatically detects Ollama and uses it as a fallback when llama.cpp fails. No configuration is needed!

When llama.cpp encounters an error (like floating point exceptions), the system will:
1. Log the llama.cpp failure
2. Check if Ollama is available
3. Automatically retry the generation with Ollama
4. Return the Ollama-generated content

### Using Ollama Directly

You can also configure Gator to use Ollama as the primary inference engine:

1. In your model configuration, set `inference_engine` to `"ollama"`
2. Specify the Ollama model name in `ollama_model` field

Example model configuration:
```json
{
  "name": "llama3-8b",
  "inference_engine": "ollama",
  "ollama_model": "llama3:8b",
  "category": "text"
}
```

### Check Ollama Status

The Gator platform includes Ollama in its inference engine status checks:

```python
from backend.utils.model_detection import get_inference_engines_status

engines = get_inference_engines_status()
print(engines["ollama"])
```

Expected output when Ollama is installed:
```python
{
    "category": "text",
    "name": "Ollama",
    "status": "installed",
    "type": "ollama",
    "version": "0.12.10",
    "path": "/usr/local/bin/ollama",
    "available_models": ["llama3:8b", "qwen3-vl:32b"],
    "server_running": True
}
```

## Troubleshooting

### Ollama Not Detected

If Gator can't find Ollama:

1. Verify Ollama is in your PATH:
   ```bash
   which ollama
   ```

2. If not, add Ollama to PATH:
   ```bash
   export PATH="/usr/local/bin:$PATH"
   ```

3. Restart your Gator service

### Server Not Running

If `ollama list` returns an error:

```bash
Error: could not connect to ollama server
```

Start the Ollama server:
```bash
ollama serve
```

Or use systemd:
```bash
systemctl start ollama
systemctl enable ollama  # To start on boot
```

### Model Not Found

If you get "model not found" errors:

1. Check available models:
   ```bash
   ollama list
   ```

2. Pull the missing model:
   ```bash
   ollama pull <model-name>
   ```

3. Retry the generation

### Floating Point Exceptions

If you're experiencing floating point exceptions with llama.cpp, Ollama fallback will automatically handle this. You can verify the fallback is working by checking the logs:

```
⚠️  llama.cpp failed: Floating point exception
🔄 Attempting fallback to Ollama...
🦙 Starting Ollama engine...
✓ Ollama generation complete
```

## Performance Considerations

- **Ollama** is optimized for ease of use and stability
- **llama.cpp** may offer slightly better performance on some systems
- The automatic fallback ensures reliability without sacrificing performance

## Model Compatibility

Ollama supports a wide range of models including:
- Llama 3 (8B, 70B variants)
- Qwen (various sizes)
- CodeLlama
- Mistral
- And many more...

See [ollama.com/library](https://ollama.com/library) for a complete list.

## Resources

- [Ollama Official Website](https://ollama.com)
- [Ollama GitHub Repository](https://github.com/ollama/ollama)
- [Ollama Model Library](https://ollama.com/library)
- [Gator AI Platform Documentation](./README.md)

## Support

If you encounter issues with Ollama integration:
1. Check the Gator logs for detailed error messages
2. Verify Ollama is running: `ollama list`
3. Test Ollama directly: `ollama run llama3:8b "test"`
4. Open an issue on GitHub with logs and system information
