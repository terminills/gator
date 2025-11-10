#!/bin/bash
# Script to install ComfyUI with AMD ROCm support
# ComfyUI is a powerful node-based UI for Stable Diffusion and other AI models

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
        print_warning "ROCm not detected. ComfyUI will run in CPU mode."
        print_info "For GPU support, install ROCm: https://rocm.docs.amd.com/en/latest/deploy/linux/quick_start.html"
        ROCM_MAJOR=0
        ROCM_MINOR=0
    fi
}

# Check for required dependencies
check_dependencies() {
    print_info "Checking dependencies..."
    
    local missing_deps=()
    
    # Check for git
    if ! command -v git &> /dev/null; then
        missing_deps+=("git")
    fi
    
    if [ ${#missing_deps[@]} -ne 0 ]; then
        print_error "Missing dependencies: ${missing_deps[*]}"
        print_info "Install with: sudo apt-get install ${missing_deps[*]}"
        exit 1
    fi
    
    print_info "All required dependencies are installed"
}

# Install/upgrade PyTorch with ROCm support
install_pytorch_rocm() {
    print_info "Installing PyTorch with ROCm support..."
    
    if [ "$ROCM_MAJOR" -eq 0 ]; then
        print_warning "Installing CPU-only PyTorch"
        python3 -m pip install --upgrade torch torchvision torchaudio
    else
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
    fi
    
    # Verify PyTorch installation
    if python3 -c "import torch; print(f'PyTorch {torch.__version__} installed successfully')" 2>/dev/null; then
        print_info "PyTorch installed successfully"
        
        # Check for GPU support
        if python3 -c "import torch; exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
            GPU_COUNT=$(python3 -c "import torch; print(torch.cuda.device_count())")
            print_info "GPU support enabled ($GPU_COUNT device(s) detected)"
        else
            print_warning "GPU support not available, ComfyUI will run in CPU mode"
        fi
    else
        print_error "Failed to install PyTorch"
        exit 1
    fi
}

# Clone ComfyUI repository
clone_comfyui() {
    local COMFYUI_DIR="${1:-./ComfyUI}"
    
    if [ -d "$COMFYUI_DIR" ]; then
        print_warning "ComfyUI directory already exists: $COMFYUI_DIR"
        print_info "Pulling latest changes..."
        cd "$COMFYUI_DIR"
        git pull
        cd -
    else
        print_info "Cloning ComfyUI repository..."
        git clone https://github.com/comfyanonymous/ComfyUI.git "$COMFYUI_DIR"
    fi
    
    export COMFYUI_DIR
}

# Install ComfyUI dependencies
install_comfyui_deps() {
    print_info "Installing ComfyUI dependencies..."
    
    cd "$COMFYUI_DIR"
    
    # Install requirements
    if [ -f "requirements.txt" ]; then
        python3 -m pip install -r requirements.txt
    else
        print_warning "requirements.txt not found, installing minimal dependencies"
        python3 -m pip install --upgrade \
            transformers>=4.25.1 \
            safetensors>=0.4.0 \
            accelerate \
            einops \
            Pillow \
            scipy \
            tqdm \
            psutil \
            kornia>=0.7.1 \
            spandrel
    fi
    
    cd -
    
    print_info "ComfyUI dependencies installed"
}

# Install popular ComfyUI custom nodes and managers
install_comfyui_manager() {
    print_info "Installing ComfyUI Manager (optional extension manager)..."
    
    local MANAGER_DIR="$COMFYUI_DIR/custom_nodes/ComfyUI-Manager"
    
    if [ -d "$MANAGER_DIR" ]; then
        print_warning "ComfyUI Manager already installed"
        cd "$MANAGER_DIR"
        git pull
        cd -
    else
        mkdir -p "$COMFYUI_DIR/custom_nodes"
        cd "$COMFYUI_DIR/custom_nodes"
        git clone https://github.com/ltdrdata/ComfyUI-Manager.git
        cd -
    fi
    
    print_info "ComfyUI Manager installed"
}

# Create launch script
create_launch_script() {
    print_info "Creating launch script..."
    
    local LAUNCH_SCRIPT="$COMFYUI_DIR/launch_rocm.sh"
    
    cat > "$LAUNCH_SCRIPT" << 'EOF'
#!/bin/bash
# ComfyUI Launch Script for ROCm

# Set ROCm environment variables
export HSA_OVERRIDE_GFX_VERSION="${HSA_OVERRIDE_GFX_VERSION:-}"
export PYTORCH_HIP_ALLOC_CONF=garbage_collection_threshold:0.8,max_split_size_mb:512

# Launch ComfyUI
python3 main.py "$@"
EOF
    
    chmod +x "$LAUNCH_SCRIPT"
    
    print_info "Launch script created: $LAUNCH_SCRIPT"
}

# Create a helper Python script for model downloads
create_model_downloader() {
    print_info "Creating model download helper..."
    
    local DOWNLOADER="$COMFYUI_DIR/download_models.py"
    
    cat > "$DOWNLOADER" << 'EOF'
#!/usr/bin/env python3
"""
Helper script to download recommended models for ComfyUI
"""

import os
import sys
from pathlib import Path
from urllib.request import urlretrieve
from tqdm import tqdm

class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)

def download_url(url, output_path):
    """Download file with progress bar"""
    print(f"Downloading {os.path.basename(output_path)}...")
    with DownloadProgressBar(unit='B', unit_scale=True, miniters=1) as t:
        urlretrieve(url, filename=output_path, reporthook=t.update_to)

# Recommended models (examples - users can customize)
RECOMMENDED_MODELS = {
    "checkpoints": [
        # Stable Diffusion 1.5 base model (popular starting point)
        {
            "name": "v1-5-pruned-emaonly.safetensors",
            "url": "https://huggingface.co/runwayml/stable-diffusion-v1-5/resolve/main/v1-5-pruned-emaonly.safetensors",
            "path": "models/checkpoints"
        }
    ],
    "vae": [
        # VAE for better image quality
        {
            "name": "vae-ft-mse-840000-ema-pruned.safetensors", 
            "url": "https://huggingface.co/stabilityai/sd-vae-ft-mse-original/resolve/main/vae-ft-mse-840000-ema-pruned.safetensors",
            "path": "models/vae"
        }
    ]
}

def main():
    print("ComfyUI Model Downloader")
    print("=" * 50)
    print()
    print("This script downloads recommended models for ComfyUI.")
    print("Downloads can be large (4-8GB). Continue? (y/N): ", end="")
    
    if input().lower() != 'y':
        print("Cancelled")
        return
    
    for category, models in RECOMMENDED_MODELS.items():
        print(f"\n{category.upper()}:")
        for model in models:
            model_dir = Path(model["path"])
            model_dir.mkdir(parents=True, exist_ok=True)
            
            output_path = model_dir / model["name"]
            
            if output_path.exists():
                print(f"  ✓ {model['name']} already exists")
                continue
            
            try:
                download_url(model["url"], str(output_path))
                print(f"  ✓ {model['name']} downloaded")
            except Exception as e:
                print(f"  ✗ Failed to download {model['name']}: {e}")
    
    print("\nDownload complete!")
    print("You can now launch ComfyUI with: ./launch_rocm.sh")

if __name__ == "__main__":
    main()
EOF
    
    chmod +x "$DOWNLOADER"
    
    print_info "Model downloader created: $DOWNLOADER"
}

# Display installation summary
display_summary() {
    echo
    print_info "=== Installation Complete ==="
    print_info "ComfyUI is now installed at: $COMFYUI_DIR"
    echo
    print_info "Next steps:"
    print_info "  1. Download models (optional):"
    print_info "     cd $COMFYUI_DIR && python3 download_models.py"
    print_info ""
    print_info "  2. Launch ComfyUI:"
    print_info "     cd $COMFYUI_DIR && ./launch_rocm.sh"
    print_info "     or: cd $COMFYUI_DIR && python3 main.py"
    print_info ""
    print_info "  3. Access web interface:"
    print_info "     Open browser to: http://localhost:8188"
    print_info ""
    print_info "For AMD GPUs with ROCm, you may need to set:"
    print_info "  export HSA_OVERRIDE_GFX_VERSION=<your_gfx_version>"
    print_info "  (e.g., 9.0.0 for gfx900, 10.3.0 for gfx1030)"
    echo
}

# Main installation flow
main() {
    print_info "=== ComfyUI ROCm Installation Script ==="
    print_info "This script will install ComfyUI with AMD ROCm support"
    echo
    
    # Parse command line arguments
    INSTALL_DIR="${1:-./ComfyUI}"
    
    # Run installation steps
    check_venv
    detect_rocm
    check_dependencies
    install_pytorch_rocm
    clone_comfyui "$INSTALL_DIR"
    install_comfyui_deps
    install_comfyui_manager
    create_launch_script
    create_model_downloader
    display_summary
}

# Run main function
main "$@"
