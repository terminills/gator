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
    for cmd in gcc g++ make cmake git ninja; do
        if ! command -v $cmd &> /dev/null; then
            # Map ninja command to ninja-build package name for apt
            if [ "$cmd" = "ninja" ]; then
                missing_deps+=("ninja-build")
            else
                missing_deps+=($cmd)
            fi
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

# Get PyTorch index URL based on ROCm version
get_pytorch_index_url() {
    # Determine PyTorch index URL based on ROCm version
    if (( $(echo "$ROCM_MAJOR >= 7" | bc -l) )); then
        PYTORCH_INDEX="https://download.pytorch.org/whl/nightly/rocm${ROCM_MAJOR}.${ROCM_MINOR}"
        # Also set AMD repository URL for ROCm 7.0+ as fallback
        AMD_ROCM_REPO="https://repo.radeon.com/rocm/manylinux/rocm-rel-${ROCM_MAJOR}.${ROCM_MINOR}.2/"
        print_info "PyTorch index: ROCm ${ROCM_MAJOR}.${ROCM_MINOR} nightly wheels (PyTorch 2.10+)"
        print_info "Fallback repo: AMD ROCm ${ROCM_MAJOR}.${ROCM_MINOR}.2 (PyTorch 2.8.0)"
    elif (( $(echo "$ROCM_MAJOR == 6" | bc -l) )) && (( $(echo "$ROCM_MINOR >= 5" | bc -l) )); then
        PYTORCH_INDEX="https://download.pytorch.org/whl/rocm${ROCM_MAJOR}.${ROCM_MINOR}"
        print_info "PyTorch index: ROCm ${ROCM_MAJOR}.${ROCM_MINOR} wheels"
    elif (( $(echo "$ROCM_MAJOR == 6" | bc -l) )); then
        PYTORCH_INDEX="https://download.pytorch.org/whl/rocm6.2"
        print_info "PyTorch index: ROCm 6.2 wheels"
    elif (( $(echo "$ROCM_MAJOR == 5" | bc -l) )); then
        PYTORCH_INDEX="https://download.pytorch.org/whl/rocm5.7"
        print_info "PyTorch index: ROCm 5.7 wheels"
    else
        print_warning "Unsupported ROCm version for standard wheels"
        PYTORCH_INDEX="https://download.pytorch.org/whl/rocm5.7"
    fi
}

# Install PyTorch 2.8.0 from AMD ROCm repository (ROCm 7.0+ fallback)
install_pytorch_amd_repo() {
    print_info "Installing PyTorch 2.8.0 from AMD ROCm repository..."
    print_info "This is a stable alternative to nightly builds for ROCm 7.0+"
    
    # Install PyTorch 2.8.0 with triton from AMD repository
    if python3 -m pip install --pre \
        torch==2.8.0 \
        torchvision \
        torchaudio==2.8.0 \
        -f "$AMD_ROCM_REPO"; then
        
        print_info "PyTorch 2.8.0 installed from AMD repository"
        
        # Install triton separately
        if python3 -m pip install triton; then
            print_info "Triton installed successfully"
        else
            print_warning "Failed to install triton, but PyTorch is functional"
        fi
        
        # Verify installation
        if python3 -c "import torch; print(f'PyTorch {torch.__version__} installed successfully')" 2>/dev/null; then
            print_info "PyTorch 2.8.0 verified"
            
            # Check for GPU support
            if python3 -c "import torch; exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
                GPU_COUNT=$(python3 -c "import torch; print(torch.cuda.device_count())")
                print_info "GPU support enabled ($GPU_COUNT device(s) detected)"
            else
                print_warning "GPU support not available"
            fi
            return 0
        else
            print_error "Failed to verify PyTorch installation"
            return 1
        fi
    else
        print_error "Failed to install PyTorch from AMD repository"
        return 1
    fi
}

# Verify PyTorch vision/audio packages are compatible
verify_pytorch_packages() {
    print_info "Verifying PyTorch package versions..."
    
    # Check if torchvision and torchaudio match PyTorch
    if python3 -c "import torch, torchvision; print('torchvision:', torchvision.__version__)" 2>/dev/null; then
        print_info "✓ torchvision is installed"
    else
        print_warning "torchvision not found - will need to install matching version"
        return 1
    fi
    
    if python3 -c "import torch, torchaudio; print('torchaudio:', torchaudio.__version__)" 2>/dev/null; then
        print_info "✓ torchaudio is installed"
    else
        print_warning "torchaudio not found - will need to install matching version"
        return 1
    fi
    
    return 0
}

# Install/upgrade PyTorch with ROCm support
install_pytorch_rocm() {
    # Check if PyTorch is already installed
    if python3 -c "import torch; print(f'PyTorch {torch.__version__} already installed')" 2>/dev/null; then
        local EXISTING_VERSION=$(python3 -c "import torch; print(torch.__version__)" 2>/dev/null)
        print_warning "PyTorch $EXISTING_VERSION is already installed"
        print_info "Skipping PyTorch installation to preserve existing setup"
        
        # Check for GPU support with existing installation
        if python3 -c "import torch; exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
            GPU_COUNT=$(python3 -c "import torch; print(torch.cuda.device_count())")
            print_info "GPU support enabled ($GPU_COUNT device(s) detected)"
        else
            print_warning "GPU support not available with existing PyTorch installation"
        fi
        
        # Verify vision/audio packages are present
        if ! verify_pytorch_packages; then
            print_warning "Missing torchvision or torchaudio - installing matching versions"
            python3 -m pip install torchvision torchaudio --index-url "$PYTORCH_INDEX" --upgrade
        fi
        
        return 0
    fi
    
    # For ROCm 7.0+, offer choice between nightly and AMD repo
    if (( $(echo "$ROCM_MAJOR >= 7" | bc -l) )) && [ "$USE_AMD_REPO" != "1" ]; then
        print_info "Installing PyTorch with ROCm support..."
        print_info "Note: Use --amd-repo flag to install stable PyTorch 2.8.0 instead of nightly"
        
        # Try installing from PyTorch nightly
        if python3 -m pip install \
            torch torchvision torchaudio \
            --index-url "$PYTORCH_INDEX"; then
            print_info "PyTorch installed from nightly wheels"
        else
            print_error "Failed to install PyTorch from nightly wheels"
            print_info "Falling back to AMD ROCm repository (PyTorch 2.8.0)..."
            install_pytorch_amd_repo
            return $?
        fi
    elif (( $(echo "$ROCM_MAJOR >= 7" | bc -l) )) && [ "$USE_AMD_REPO" = "1" ]; then
        # User requested AMD repository explicitly
        print_info "Using AMD ROCm repository as requested..."
        install_pytorch_amd_repo
        return $?
    else
        # ROCm 6.x or 5.7 - use standard method
        print_info "Installing PyTorch with ROCm support..."
        
        python3 -m pip install \
            torch torchvision torchaudio \
            --index-url "$PYTORCH_INDEX"
    fi
    
    # Verify PyTorch installation
    if python3 -c "import torch; print(f'PyTorch {torch.__version__} installed successfully')" 2>/dev/null; then
        print_info "PyTorch with ROCm support installed successfully"
        
        # Check for GPU support
        if python3 -c "import torch; exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
            GPU_COUNT=$(python3 -c "import torch; print(torch.cuda.device_count())")
            print_info "GPU support enabled ($GPU_COUNT device(s) detected)"
        else
            print_warning "GPU support not available, vLLM will run in CPU mode"
        fi
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

# Install vLLM build dependencies
install_vllm_build_deps() {
    print_info "Installing vLLM build dependencies..."
    
    # Install required build dependencies that vLLM needs
    python3 -m pip install --upgrade \
        torch \
        packaging \
        psutil \
        ray
    
    # Ensure we have the correct versions of ninja and cmake
    if ! python3 -c "import ninja" 2>/dev/null; then
        python3 -m pip install ninja
    fi
    
    print_info "Build dependencies installed"
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
    
    # Get installed PyTorch version for compatibility info
    PYTORCH_VERSION=$(python3 -c "import torch; print(torch.__version__)" 2>/dev/null || echo "unknown")
    
    print_info "Build configuration:"
    print_info "  ROCM_HOME: $ROCM_HOME"
    print_info "  PYTORCH_ROCM_ARCH: $PYTORCH_ROCM_ARCH"
    print_info "  VLLM_TARGET_DEVICE: $VLLM_TARGET_DEVICE"
    print_info "  PyTorch version: $PYTORCH_VERSION"
    
    # Clean previous builds
    if [ -d "build" ]; then
        print_info "Cleaning previous build..."
        rm -rf build dist *.egg-info
    fi
    
    # Build and install vLLM
    # Use --no-build-isolation to prevent pip from creating a separate build environment
    # that might install a different PyTorch version, causing conflicts
    print_info "Installing vLLM (using existing PyTorch installation)..."
    python3 -m pip install -e . --no-build-isolation --verbose
    
    cd -
    
    # Verify installation
    if python3 -c "import vllm; print(f'vLLM {vllm.__version__} installed successfully')" 2>/dev/null; then
        print_info "vLLM built and installed successfully"
    else
        print_error "Failed to verify vLLM installation"
        exit 1
    fi
}

# Repair PyTorch installation with AMD repository (ROCm 7.0+)
repair_pytorch() {
    print_info "=== PyTorch Repair Mode ==="
    print_info "This will reinstall PyTorch 2.8.0 from AMD ROCm repository"
    echo
    
    detect_rocm
    
    if ! (( $(echo "$ROCM_MAJOR >= 7" | bc -l) )); then
        print_error "Repair mode is only for ROCm 7.0+"
        print_info "Your ROCm version: $ROCM_VERSION"
        exit 1
    fi
    
    get_pytorch_index_url
    
    print_warning "This will uninstall existing PyTorch and reinstall from AMD repository"
    read -p "Continue? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 0
    fi
    
    # Uninstall existing PyTorch
    print_info "Uninstalling existing PyTorch..."
    python3 -m pip uninstall -y torch torchvision torchaudio triton 2>/dev/null || true
    
    # Install from AMD repository
    install_pytorch_amd_repo
    
    if [ $? -eq 0 ]; then
        print_info "PyTorch repair completed successfully"
        print_info "You can now run the vLLM installation again"
    else
        print_error "PyTorch repair failed"
        exit 1
    fi
}

# Main installation flow
main() {
    print_info "=== vLLM ROCm Installation Script ==="
    print_info "This script will build and install vLLM for AMD ROCm systems"
    echo
    
    # Parse command line arguments
    VLLM_DIR="./vllm-rocm"
    USE_AMD_REPO=0
    REPAIR_MODE=0
    
    while [[ $# -gt 0 ]]; do
        case $1 in
            --amd-repo)
                USE_AMD_REPO=1
                shift
                ;;
            --repair)
                REPAIR_MODE=1
                shift
                ;;
            --help|-h)
                echo "Usage: $0 [OPTIONS] [VLLM_DIR]"
                echo ""
                echo "Options:"
                echo "  --amd-repo    Use AMD ROCm repository for PyTorch 2.8.0 (ROCm 7.0+ only)"
                echo "  --repair      Repair PyTorch installation using AMD repository"
                echo "  --help, -h    Show this help message"
                echo ""
                echo "Arguments:"
                echo "  VLLM_DIR      Directory to clone vLLM (default: ./vllm-rocm)"
                echo ""
                echo "Examples:"
                echo "  $0                           # Standard installation"
                echo "  $0 --amd-repo                # Use AMD repo for PyTorch 2.8.0"
                echo "  $0 --repair                  # Repair PyTorch installation"
                echo "  $0 /custom/path              # Install to custom directory"
                exit 0
                ;;
            --*)
                print_error "Unknown option: $1"
                print_info "Run '$0 --help' to see available options"
                print_info ""
                print_info "Note: The --no-build-isolation flag is used internally by pip"
                print_info "      and should not be passed to this installation script."
                exit 1
                ;;
            *)
                VLLM_DIR="$1"
                shift
                ;;
        esac
    done
    
    # Handle repair mode
    if [ "$REPAIR_MODE" = "1" ]; then
        repair_pytorch
        exit 0
    fi
    
    # Run installation steps
    check_venv
    detect_rocm
    check_dependencies
    get_pytorch_index_url
    install_python_deps
    install_pytorch_rocm
    clone_vllm "$VLLM_DIR"
    install_vllm_build_deps
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
    
    # Show repair option for ROCm 7.0+
    if (( $(echo "$ROCM_MAJOR >= 7" | bc -l) )); then
        echo
        print_info "ROCm 7.0+ Troubleshooting:"
        print_info "  If you encounter PyTorch version conflicts, run repair mode:"
        print_info "  bash $0 --repair"
        print_info "  This will install stable PyTorch 2.8.0 from AMD repository"
    fi
    echo
}

# Run main function
main "$@"
