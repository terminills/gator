#!/bin/bash
# Script to build and install vLLM for AMD ROCm systems
# Reference: https://docs.vllm.ai/en/stable/getting_started/installation/gpu.html#unsupported-os-build

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if running in a virtual environment
check_venv() {
    if [[ -z "$VIRTUAL_ENV" ]]; then
        print_warning "Not running in a virtual environment"
        read -p "Continue anyway? (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 1
        fi
    else
        print_info "Virtual environment detected: $VIRTUAL_ENV"
    fi
}

# Detect ROCm version
detect_rocm() {
    if command -v rocminfo &> /dev/null; then
        ROCM_VERSION=$(rocminfo | grep -oP 'ROCm version: \K[0-9.]+' | head -1)
        print_info "Detected ROCm version: $ROCM_VERSION"
        
        # Extract major.minor version
        ROCM_MAJOR=$(echo $ROCM_VERSION | cut -d. -f1)
        ROCM_MINOR=$(echo $ROCM_VERSION | cut -d. -f2)
        export ROCM_VERSION_MAJOR_MINOR="${ROCM_MAJOR}.${ROCM_MINOR}"
    else
        print_error "ROCm not detected. Please install ROCm first."
        print_info "Visit: https://rocm.docs.amd.com/en/latest/deploy/linux/quick_start.html"
        exit 1
    fi
}

# Check for required build dependencies
check_dependencies() {
    print_info "Checking build dependencies..."
    
    local missing_deps=()
    
    # Check for essential build tools
    for cmd in gcc g++ make cmake git ninja-build; do
        if ! command -v $cmd &> /dev/null; then
            missing_deps+=($cmd)
        fi
    done
    
    # Check for Python development headers
    if ! python3 -c "import sysconfig" 2>/dev/null; then
        print_error "Python development headers not found"
        missing_deps+=("python3-dev")
    fi
    
    if [ ${#missing_deps[@]} -ne 0 ]; then
        print_error "Missing dependencies: ${missing_deps[*]}"
        print_info "Install with: sudo apt-get install ${missing_deps[*]}"
        exit 1
    fi
    
    print_info "All required dependencies are installed"
}

# Install Python build dependencies
install_python_deps() {
    print_info "Installing Python build dependencies..."
    
    # Upgrade pip, setuptools, wheel
    python3 -m pip install --upgrade pip setuptools wheel
    
    # Install build dependencies for vLLM
    python3 -m pip install --upgrade \
        cmake>=3.21 \
        ninja \
        packaging \
        wheel \
        setuptools-scm
    
    print_info "Python build dependencies installed"
}

# Install/upgrade PyTorch with ROCm support
install_pytorch_rocm() {
    print_info "Installing PyTorch with ROCm support..."
    
    # Determine PyTorch index URL based on ROCm version
    if (( $(echo "$ROCM_MAJOR >= 6" | bc -l) )); then
        PYTORCH_INDEX="https://download.pytorch.org/whl/rocm6.2"
        print_info "Using PyTorch ROCm 6.2 wheels"
    elif (( $(echo "$ROCM_MAJOR == 5" | bc -l) )); then
        PYTORCH_INDEX="https://download.pytorch.org/whl/rocm5.7"
        print_info "Using PyTorch ROCm 5.7 wheels"
    else
        print_warning "Unsupported ROCm version for standard wheels"
        PYTORCH_INDEX="https://download.pytorch.org/whl/rocm5.7"
    fi
    
    # Install PyTorch and related packages
    python3 -m pip install --upgrade \
        torch torchvision torchaudio \
        --index-url $PYTORCH_INDEX
    
    # Verify PyTorch installation
    if python3 -c "import torch; print(f'PyTorch {torch.__version__} installed successfully')" 2>/dev/null; then
        print_info "PyTorch with ROCm support installed successfully"
    else
        print_error "Failed to install PyTorch with ROCm support"
        exit 1
    fi
}

# Clone vLLM repository if not exists
clone_vllm() {
    local VLLM_DIR="${1:-./vllm-rocm}"
    
    if [ -d "$VLLM_DIR" ]; then
        print_warning "vLLM directory already exists: $VLLM_DIR"
        print_info "Pulling latest changes..."
        cd "$VLLM_DIR"
        git pull
        cd -
    else
        print_info "Cloning vLLM repository..."
        git clone https://github.com/vllm-project/vllm.git "$VLLM_DIR"
    fi
    
    export VLLM_SOURCE_DIR="$VLLM_DIR"
}

# Build and install vLLM from source
build_vllm() {
    print_info "Building vLLM from source (this may take 10-30 minutes)..."
    
    cd "$VLLM_SOURCE_DIR"
    
    # Set environment variables for ROCm build
    export PYTORCH_ROCM_ARCH="${PYTORCH_ROCM_ARCH:-gfx900;gfx906;gfx908;gfx90a;gfx940;gfx941;gfx942;gfx1030;gfx1100}"
    export VLLM_TARGET_DEVICE=rocm
    
    # For AMD build, we need to use ROCm-specific flags
    export ROCM_HOME="${ROCM_HOME:-/opt/rocm}"
    export HIP_PATH="${HIP_PATH:-$ROCM_HOME}"
    
    print_info "Build configuration:"
    print_info "  ROCM_HOME: $ROCM_HOME"
    print_info "  PYTORCH_ROCM_ARCH: $PYTORCH_ROCM_ARCH"
    print_info "  VLLM_TARGET_DEVICE: $VLLM_TARGET_DEVICE"
    
    # Clean previous builds
    if [ -d "build" ]; then
        print_info "Cleaning previous build..."
        rm -rf build dist *.egg-info
    fi
    
    # Build and install vLLM
    python3 -m pip install -e . --verbose
    
    cd -
    
    # Verify installation
    if python3 -c "import vllm; print(f'vLLM {vllm.__version__} installed successfully')" 2>/dev/null; then
        print_info "vLLM built and installed successfully"
    else
        print_error "Failed to verify vLLM installation"
        exit 1
    fi
}

# Main installation flow
main() {
    print_info "=== vLLM ROCm Installation Script ==="
    print_info "This script will build and install vLLM for AMD ROCm systems"
    echo
    
    # Parse command line arguments
    VLLM_DIR="${1:-./vllm-rocm}"
    
    # Run installation steps
    check_venv
    detect_rocm
    check_dependencies
    install_python_deps
    install_pytorch_rocm
    clone_vllm "$VLLM_DIR"
    build_vllm
    
    echo
    print_info "=== Installation Complete ==="
    print_info "vLLM is now installed and ready to use"
    print_info ""
    print_info "Quick test:"
    print_info "  python3 -c 'import vllm; print(vllm.__version__)'"
    print_info ""
    print_info "Usage example:"
    print_info "  from vllm import LLM, SamplingParams"
    print_info "  llm = LLM(model='facebook/opt-125m')"
    print_info "  outputs = llm.generate('Hello, my name is', SamplingParams(max_tokens=50))"
    echo
}

# Run main function
main "$@"
