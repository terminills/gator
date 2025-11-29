#!/bin/bash
# Manual test script for llama-cli debugging
# This replicates what the code does so you can see the actual llama.cpp output

echo "================================================================================================"
echo "LLAMA.CPP MANUAL TEST SCRIPT"
echo "================================================================================================"
echo ""
echo "This script will help debug why llama.cpp is failing (exit code 1)"
echo ""

# Check if llama-cli exists
if ! command -v llama-cli &> /dev/null; then
    echo "ERROR: llama-cli not found in PATH"
    echo ""
    echo "Please ensure llama-cli is installed and in your PATH:"
    echo "  which llama-cli"
    echo ""
    exit 1
fi

LLAMA_CLI=$(which llama-cli)
echo "✓ Found llama-cli at: $LLAMA_CLI"
echo ""

# Check version
echo "Checking llama-cli version:"
"$LLAMA_CLI" --version 2>&1 || echo "(version check failed, but may still work)"
echo ""

# Set model path (adjust based on your setup)
MODEL_PATH="models/text/llama-3.1-8b"
echo "Looking for model in: $MODEL_PATH"
echo ""

# Find GGUF file
if [ -d "$MODEL_PATH" ]; then
    GGUF_FILE=$(find "$MODEL_PATH" -name "*.gguf" -type f | head -1)
    if [ -z "$GGUF_FILE" ]; then
        echo "ERROR: No .gguf file found in $MODEL_PATH"
        echo ""
        echo "Please ensure you have downloaded a GGUF model file."
        echo "Example files to look for:"
        ls -lh "$MODEL_PATH"/ 2>/dev/null || echo "  (directory is empty or doesn't exist)"
        echo ""
        exit 1
    fi
else
    echo "ERROR: Model directory $MODEL_PATH does not exist"
    echo ""
    exit 1
fi

echo "✓ Found model: $GGUF_FILE"
MODEL_SIZE=$(du -h "$GGUF_FILE" | cut -f1)
echo "  Size: $MODEL_SIZE"
echo ""

# Test prompt (matching what Gator agent uses)
PROMPT="You are Gator, a tough, no-nonsense AI help agent. You're direct, confident, and sometimes intimidating, but ultimately helpful. Keep responses concise (2-3 sentences). Use phrases like \"Listen here\", \"Pay attention\".

User: hello
Gator:"

echo "================================================================================================"
echo "RUNNING LLAMA.CPP TEST"
echo "================================================================================================"
echo ""
echo "Command that will be executed:"
echo ""
echo "$LLAMA_CLI \\"
echo "  -m \"$GGUF_FILE\" \\"
echo "  -p \"<prompt>\" \\"
echo "  -n 200 \\"
echo "  --temp 0.8 \\"
echo "  -c 4096 \\"
echo "  --log-disable"
echo ""
echo "Press Enter to continue..."
read

echo ""
echo "================================================================================================"
echo "LLAMA.CPP OUTPUT (RAW):"
echo "================================================================================================"
echo ""

# Run llama-cli with exact same parameters as the code
"$LLAMA_CLI" \
  -m "$GGUF_FILE" \
  -p "$PROMPT" \
  -n 200 \
  --temp 0.8 \
  -c 4096 \
  --log-disable

EXIT_CODE=$?

echo ""
echo "================================================================================================"
echo "TEST COMPLETE"
echo "================================================================================================"
echo ""
echo "Exit code: $EXIT_CODE"
echo ""

if [ $EXIT_CODE -eq 0 ]; then
    echo "✓ SUCCESS: llama.cpp executed successfully"
    echo ""
    echo "The output above should contain:"
    echo "  1. Initialization logs (ggml_cuda_init, etc.) - EXPECTED"
    echo "  2. Generated text response - THIS IS WHAT WE NEED"
    echo ""
    echo "If you only see initialization logs and no generated text,"
    echo "the model may be incompatible or corrupted."
else
    echo "✗ FAILURE: llama.cpp failed with exit code $EXIT_CODE"
    echo ""
    echo "Common causes:"
    echo "  - Model file is corrupted or incompatible"
    echo "  - Insufficient memory (RAM or VRAM)"
    echo "  - GPU/CUDA/ROCm compatibility issues"
    echo "  - Model format not supported by this llama.cpp version"
    echo ""
    echo "To debug further:"
    echo "  1. Check system resources: free -h"
    echo "  2. Check GPU status: rocm-smi (for AMD) or nvidia-smi (for NVIDIA)"
    echo "  3. Try a smaller model or different GGUF quantization"
    echo "  4. Check llama.cpp GitHub issues for similar problems"
fi
echo ""
