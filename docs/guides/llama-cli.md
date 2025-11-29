# Manual llama-cli Testing Instructions

This guide helps you manually test llama-cli to diagnose why it's failing with exit code 1.

## Quick Test Command

Run this command to test llama-cli manually:

```bash
llama-cli \
  -m "models/text/llama-3.1-8b/*.gguf" \
  -p "You are Gator. User: hello. Gator:" \
  -n 200 \
  --temp 0.8 \
  -c 4096 \
  --log-disable
```

**Replace** `models/text/llama-3.1-8b/*.gguf` with the actual path to your GGUF model file.

## What to Look For

### Success (Exit Code 0)
You should see:
1. **Initialization logs** (ggml_cuda_init, Device info) - NORMAL
2. **Generated text** - THIS IS WHAT MATTERS

Example successful output:
```
ggml_cuda_init: GGML_CUDA_FORCE_MMQ:    no
ggml_cuda_init: found 2 ROCm devices:
  Device 0: AMD Radeon PRO V620...

Listen here, what do you need? I'm a peacock, you gotta let me fly!
```

### Failure (Exit Code 1)
You'll see:
1. **Only initialization logs** (no generated text)
2. **Error messages** at the end

Example failed output:
```
ggml_cuda_init: GGML_CUDA_FORCE_MMQ:    no
ggml_cuda_init: found 2 ROCm devices:
  Device 0: AMD Radeon PRO V620...
error: failed to load model
```

## Automated Test Script

For a more detailed test with diagnostics:

```bash
cd /home/runner/work/gator/gator
./test_llamacpp_manual.sh
```

This script will:
- Check if llama-cli is installed
- Find your model file
- Show the exact command being run
- Display the full output
- Report the exit code
- Suggest fixes if it fails

## Common Issues and Fixes

### Issue 1: Model File Not Found
```
error: failed to load model: llama_model_loader: failed to load model
```
**Fix:** Check the model path exists and contains a .gguf file:
```bash
ls -lh models/text/llama-3.1-8b/
```

### Issue 2: Insufficient Memory
```
error: failed to allocate memory
```
**Fix:** Check available memory:
```bash
free -h
```
You may need a smaller model or quantized version (Q4_K_M instead of FP16).

### Issue 3: GPU/CUDA/ROCm Error
```
ggml_cuda_init: failed to initialize CUDA
```
**Fix:** Check GPU status:
```bash
# For AMD ROCm:
rocm-smi

# For NVIDIA CUDA:
nvidia-smi
```

### Issue 4: Incompatible Model Format
```
error: unknown model architecture
```
**Fix:** The model format may not be supported by your llama.cpp version. Try:
1. Updating llama.cpp: `git pull && make clean && make`
2. Using a different model quantization
3. Downloading a known-compatible model

### Issue 5: Model Corrupted
```
error: failed to load model: invalid magic
```
**Fix:** Re-download the model file. It may be corrupted.

## Detailed Debugging Steps

1. **Verify llama-cli is installed:**
   ```bash
   which llama-cli
   llama-cli --version
   ```

2. **Check model file exists and size:**
   ```bash
   ls -lh models/text/llama-3.1-8b/*.gguf
   ```
   Model should be several GB in size.

3. **Check system resources:**
   ```bash
   free -h          # Check RAM
   df -h .          # Check disk space
   rocm-smi         # Check GPU (if using AMD)
   ```

4. **Test with minimal parameters:**
   ```bash
   llama-cli -m "path/to/model.gguf" -p "Hello" -n 10
   ```

5. **Enable verbose logging (remove --log-disable):**
   ```bash
   llama-cli -m "path/to/model.gguf" -p "Hello" -n 10 -c 2048 --temp 0.7
   ```
   This will show more detailed error messages.

## What to Share for Debugging

If llama-cli fails, please share:

1. **Exit code:** `echo $?` right after running the command
2. **Full output:** Copy the complete terminal output
3. **Model info:**
   ```bash
   ls -lh models/text/llama-3.1-8b/
   file models/text/llama-3.1-8b/*.gguf
   ```
4. **System info:**
   ```bash
   free -h
   rocm-smi --showproduct 2>/dev/null || nvidia-smi 2>/dev/null || echo "No GPU info"
   ```
5. **llama.cpp version:** Output of `llama-cli --version`

## After Testing

Once you've identified the issue:

- **If it's a model problem:** Try a different model or re-download
- **If it's a memory problem:** Use a smaller model (Q4_K_M quantization)
- **If it's a GPU problem:** Check CUDA/ROCm installation
- **If it works manually but fails in code:** There may be a path or parameter issue

Share the output with @copilot for further assistance.
