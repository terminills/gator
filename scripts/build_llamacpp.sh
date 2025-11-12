#!/bin/bash
# Build llama.cpp for text generation

set -e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LLAMACPP_DIR="$REPO_ROOT/third_party/llama.cpp"

echo "🦙 Building llama.cpp..."
echo "  Repository: $REPO_ROOT"
echo "  llama.cpp directory: $LLAMACPP_DIR"

# Check if llama.cpp exists
if [ ! -d "$LLAMACPP_DIR" ]; then
    echo "  Cloning llama.cpp..."
    mkdir -p "$REPO_ROOT/third_party"
    cd "$REPO_ROOT/third_party"
    git clone https://github.com/ggerganov/llama.cpp.git --depth 1 --branch master
fi

# Build llama.cpp
echo "  Building..."
cd "$LLAMACPP_DIR"
rm -rf build
mkdir -p build
cd build

cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DGGML_NATIVE=OFF \
    -DLLAMA_CURL=OFF

make -j$(nproc) llama-cli

echo "✅ llama.cpp built successfully!"
echo "  Binary: $LLAMACPP_DIR/build/bin/llama-cli"
echo "  Version: $(${LLAMACPP_DIR}/build/bin/llama-cli --version | head -1)"

# Add to PATH
echo ""
echo "To use llama-cli, add to your PATH:"
echo "  export PATH=\"$LLAMACPP_DIR/build/bin:\$PATH\""
