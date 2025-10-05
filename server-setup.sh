#!/bin/bash

# Gator AI Influencer Platform - Automated Server Setup Script
# Compatible with Ubuntu 20.04+, Debian 11+
# Usage: curl -sSL https://raw.githubusercontent.com/terminills/gator/main/server-setup.sh | sudo bash
# With options: sudo bash server-setup.sh --domain example.com --email admin@example.com

set -euo pipefail

# Script configuration
SCRIPT_VERSION="1.0.4"
GATOR_USER="gator"
GATOR_HOME="/opt/gator"
PYTHON_VERSION="3.9"
NODE_VERSION="18"
ROCM_VERSION="5.7.1"

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default values
DOMAIN=""
EMAIL=""
INSTALL_GPU_SUPPORT=false
INSTALL_NVIDIA_SUPPORT=false
INSTALL_ROCM_SUPPORT=false
INSTALL_NGINX=true
INSTALL_SSL=true
SKIP_FIREWALL=false

# Logging function
log() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')] $1${NC}"
}

warn() {
    echo -e "${YELLOW}[WARNING] $1${NC}"
}

error() {
    echo -e "${RED}[ERROR] $1${NC}"
    exit 1
}

# Help function
show_help() {
    cat << EOF
Gator AI Influencer Platform - Server Setup Script v${SCRIPT_VERSION}

Usage: $0 [OPTIONS]

OPTIONS:
  --domain DOMAIN          Set primary domain name (e.g., example.com)
  --email EMAIL           Admin email for SSL certificates and notifications
  --gpu                   Install GPU support (auto-detect NVIDIA/AMD)
  --nvidia                Install NVIDIA GPU support (drivers, CUDA)
  --rocm                  Install AMD GPU support (ROCm ${ROCM_VERSION})
  --no-nginx             Skip Nginx installation
  --no-ssl               Skip SSL certificate setup
  --skip-firewall        Skip firewall configuration
  --help                 Show this help message

EXAMPLES:
  # Basic installation
  sudo $0

  # Full installation with domain
  sudo $0 --domain myai.com --email admin@myai.com

  # GPU-enabled installation (auto-detect)
  sudo $0 --domain myai.com --email admin@myai.com --gpu

  # NVIDIA-specific installation
  sudo $0 --domain myai.com --email admin@myai.com --nvidia

  # AMD ROCm installation
  sudo $0 --domain myai.com --email admin@myai.com --rocm

EOF
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --domain)
            DOMAIN="$2"
            shift 2
            ;;
        --email)
            EMAIL="$2"
            shift 2
            ;;
        --gpu)
            INSTALL_GPU_SUPPORT=true
            shift
            ;;
        --nvidia)
            INSTALL_GPU_SUPPORT=true
            INSTALL_NVIDIA_SUPPORT=true
            shift
            ;;
        --rocm)
            INSTALL_GPU_SUPPORT=true
            INSTALL_ROCM_SUPPORT=true
            shift
            ;;
        --no-nginx)
            INSTALL_NGINX=false
            shift
            ;;
        --no-ssl)
            INSTALL_SSL=false
            shift
            ;;
        --skip-firewall)
            SKIP_FIREWALL=true
            shift
            ;;
        --help)
            show_help
            exit 0
            ;;
        *)
            error "Unknown option: $1. Use --help for usage information."
            ;;
    esac
done

# Check if running as root
if [[ $EUID -ne 0 ]]; then
   error "This script must be run as root (use sudo)"
fi

# Detect OS
if [[ -f /etc/os-release ]]; then
    . /etc/os-release
    OS=$ID
    OS_VERSION=$VERSION_ID
else
    error "Cannot detect operating system"
fi

# Validate OS compatibility
case $OS in
    ubuntu)
        if [[ $(echo "$OS_VERSION >= 20.04" | bc -l) -eq 0 ]]; then
            error "Ubuntu 20.04 or higher required. Found: $OS_VERSION"
        fi
        PACKAGE_MANAGER="apt"
        ;;
    debian)
        if [[ $(echo "$OS_VERSION >= 11" | bc -l) -eq 0 ]]; then
            error "Debian 11 or higher required. Found: $OS_VERSION"
        fi
        PACKAGE_MANAGER="apt"
        ;;
    *)
        error "Unsupported OS: $OS. This script supports Ubuntu 20.04+ and Debian 11+"
        ;;
esac

log "🦎 Starting Gator AI Influencer Platform setup on $OS $OS_VERSION"

# System update
log "📦 Updating system packages..."
apt update && apt upgrade -y

# Install essential packages
log "🔧 Installing essential packages..."
apt install -y \
    curl \
    wget \
    git \
    build-essential \
    software-properties-common \
    apt-transport-https \
    ca-certificates \
    gnupg \
    lsb-release \
    unzip \
    htop \
    nano \
    vim \
    screen \
    tmux \
    jq \
    bc

# Install Python
log "🐍 Installing Python ${PYTHON_VERSION}..."
if [[ $OS == "ubuntu" ]]; then
    add-apt-repository ppa:deadsnakes/ppa -y
    apt update
    apt install -y python${PYTHON_VERSION} python${PYTHON_VERSION}-venv python${PYTHON_VERSION}-dev python3-pip
else
    apt install -y python3 python3-venv python3-dev python3-pip
fi

# Create Python symlinks
update-alternatives --install /usr/bin/python python /usr/bin/python${PYTHON_VERSION} 1
update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1

# Install Node.js (for frontend development)
log "📦 Installing Node.js ${NODE_VERSION}..."
curl -fsSL https://deb.nodesource.com/setup_${NODE_VERSION}.x | bash -
apt install -y nodejs

# Install PostgreSQL
log "🗄️ Installing PostgreSQL..."
apt install -y postgresql postgresql-contrib postgresql-client
systemctl enable postgresql
systemctl start postgresql

# Clean up existing PostgreSQL user and database for reinstall
log "🔐 Cleaning up existing PostgreSQL user and database..."
sudo -u postgres psql -c "DROP DATABASE IF EXISTS gator_production;" || warn "Failed to drop database, may not exist"
sudo -u postgres psql -c "DROP ROLE IF EXISTS gator;" || warn "Failed to drop role, may not exist"

# Create PostgreSQL database and user
log "🔐 Setting up PostgreSQL database..."
GATOR_DB_PASSWORD=$(openssl rand -base64 12)
sudo -u postgres psql -c "CREATE ROLE gator WITH LOGIN PASSWORD '$GATOR_DB_PASSWORD' CREATEDB;" || error "Failed to create PostgreSQL user"
sudo -u postgres createdb gator_production --owner=gator || error "Failed to create PostgreSQL database"

# Install Redis
log "📮 Installing Redis..."
apt install -y redis-server
systemctl enable redis-server
systemctl start redis-server

# Install NGINX if requested
if [[ $INSTALL_NGINX == true ]]; then
    log "🌐 Installing NGINX..."
    apt install -y nginx
    systemctl enable nginx
    systemctl start nginx
fi

# Install GPU support if requested
if [[ $INSTALL_GPU_SUPPORT == true ]]; then
    log "🎮 Installing GPU support..."
    
    # Auto-detect GPU types if not specifically requested
    if [[ $INSTALL_NVIDIA_SUPPORT == false && $INSTALL_ROCM_SUPPORT == false ]]; then
        if lspci | grep -i nvidia > /dev/null; then
            log "NVIDIA GPU detected, enabling NVIDIA support..."
            INSTALL_NVIDIA_SUPPORT=true
        fi
        if lspci | grep -i amd > /dev/null || lspci | grep -i "advanced micro devices" > /dev/null; then
            log "AMD GPU detected, enabling ROCm support..."
            INSTALL_ROCM_SUPPORT=true
        fi
    fi
    
    # Install NVIDIA support
    if [[ $INSTALL_NVIDIA_SUPPORT == true ]]; then
        log "🟢 Installing NVIDIA GPU support (drivers and CUDA)..."
        
        if lspci | grep -i nvidia > /dev/null; then
            # Install NVIDIA drivers
            apt install -y nvidia-driver-470
            
            # Install CUDA toolkit
            wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin
            mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600
            wget https://developer.download.nvidia.com/compute/cuda/12.0.0/local_installers/cuda-repo-ubuntu2004-12-0-local_12.0.0-525.60.13-1_amd64.deb
            dpkg -i cuda-repo-ubuntu2004-12-0-local_12.0.0-525.60.13-1_amd64.deb
            cp /var/cuda-repo-ubuntu2004-12-0-local/cuda-*-keyring.gpg /usr/share/keyrings/
            apt update
            apt install -y cuda
            
            # Add CUDA to PATH
            echo 'export PATH=/usr/local/cuda/bin:$PATH' >> /etc/environment
            echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> /etc/environment
            
            log "✅ NVIDIA GPU support installed successfully"
        else
            warn "No NVIDIA GPU detected. Skipping NVIDIA driver installation."
        fi
    fi
    
    # Install ROCm support
    if [[ $INSTALL_ROCM_SUPPORT == true ]]; then
        log "🔴 Checking for AMD ROCm support..."

        # Function to check ROCm installation and version
        check_rocm() {
            local desired_version="$1"
            if command -v rocminfo >/dev/null 2>&1; then
                local installed_version=$(cat /opt/rocm/.info/version 2>/dev/null || echo "unknown")
                # Strip any suffix (e.g., -98) for comparison
                local installed_base_version=$(echo "$installed_version" | cut -d'-' -f1)
                if [[ "$installed_base_version" == "$desired_version" ]]; then
                    log "✅ ROCm $installed_version is compatible with $desired_version. Skipping installation."
                    return 0
                else
                    warn "ROCm version $installed_version found, but $desired_version is required. Proceeding with installation."
                    return 1
                fi
            else
                log "No ROCm installation detected. Proceeding with installation."
                return 1
            fi
        }

        # Function to check if amdgpu-install package is installed
        check_amdgpu_install() {
            if dpkg -l | grep -q amdgpu-install; then
                local installed_version=$(dpkg -l | grep amdgpu-install | awk '{print $3}' | cut -d'-' -f1)
                if [[ "$installed_version" == "5.7.50701" ]]; then
                    log "✅ amdgpu-install 5.7.50701 is already installed."
                    return 0
                else
                    warn "amdgpu-install version $installed_version found. Will attempt to install 5.7.50701."
                    return 1
                fi
            else
                log "No amdgpu-install package detected."
                return 1
            fi
        }

        if lspci | grep -i amd > /dev/null || lspci | grep -i "advanced micro devices" > /dev/null; then
            # Check for existing ROCm installation
            if check_rocm "$ROCM_VERSION"; then
                log "Using existing ROCm $ROCM_VERSION installation."
            else
                log "Installing AMD ROCm $ROCM_VERSION..."

                # Detect GPU architecture for version compatibility
                GPU_INFO=$(lspci | grep -i "vga.*amd\|display.*amd\|3d.*amd")
                IS_MI25=false

                # Check for MI25 (Vega10/gfx900) specifically with multiple detection methods
                if lspci -v | grep -i "radeon instinct mi25\|vega.*10" > /dev/null; then
                    IS_MI25=true
                fi

                # Also check via device ID (Vega 10 = 0x6860-0x686f)
                if lspci -n | grep -E "1002:(6860|6861|6862|6863|6864|6865|6866|6867|6868|6869|686a|686b|686c|686d|686e|686f)" > /dev/null; then
                    IS_MI25=true
                fi

                if [[ $IS_MI25 == true ]]; then
                    log "AMD MI25 (gfx900/Vega10) GPU detected - using ROCm $ROCM_VERSION with gfx900 optimizations..."
                    GFX_ARCH="gfx900"
                else
                    log "Using latest ROCm version..."
                    GFX_ARCH="auto"
                fi

                # Install amdgpu-install package if not already installed
                if ! check_amdgpu_install; then
                    log "Downloading AMD GPU installer for ROCm $ROCM_VERSION..."

                    # Determine Ubuntu version for installer package
                    UBUNTU_VERSION=$(lsb_release -rs)
                    if [[ "$UBUNTU_VERSION" == "20.04" ]]; then
                        UBUNTU_CODENAME="focal"
                    elif [[ "$UBUNTU_VERSION" == "22.04" ]]; then
                        UBUNTU_CODENAME="jammy"
                    else
                        warn "Ubuntu version $UBUNTU_VERSION may not be fully supported. Using focal (20.04) installer."
                        UBUNTU_CODENAME="focal"
                    fi

                    # Download and install amdgpu-install package
                    INSTALLER_DEB="amdgpu-install_5.7.50701-1_all.deb"
                    wget https://repo.radeon.com/amdgpu-install/5.7.1/ubuntu/$UBUNTU_CODENAME/$INSTALLER_DEB

                    log "Installing AMD GPU installer package..."
                    dpkg -i ./$INSTALLER_DEB || true
                    apt install -f -y

                    # Clean up installer package
                    rm -f ./$INSTALLER_DEB
                fi

                # Install ROCm runtime and development packages
                log "Installing ROCm runtime and development packages..."
                amdgpu-install --usecase=rocm,hiplibsdk,dkms --rocmrelease=$ROCM_VERSION -y

                # Add users to render and video groups for GPU access
                usermod -aG render,video $GATOR_USER
                usermod -aG render,video root

                # Set ROCm environment variables
                log "Configuring ROCm environment variables..."
                echo 'export PATH=/opt/rocm/bin:/opt/rocm/opencl/bin:$PATH' >> /etc/environment
                echo 'export LD_LIBRARY_PATH=/opt/rocm/lib:/opt/rocm/lib64:$LD_LIBRARY_PATH' >> /etc/environment
                echo 'export ROCM_PATH=/opt/rocm' >> /etc/environment
                echo 'export HIP_PATH=/opt/rocm/hip' >> /etc/environment

                # Configure GPU access for multiple MI25 GPUs
                echo 'export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7' >> /etc/environment
                echo 'export ROCR_VISIBLE_DEVICES=0,1,2,3,4,5,6,7' >> /etc/environment

                # Set gfx900-specific environment variables for MI25
                if [[ $IS_MI25 == true ]]; then
                    log "Setting gfx900-specific environment variables for MI25..."
                    echo 'export HSA_OVERRIDE_GFX_VERSION=9.0.0' >> /etc/environment
                    echo 'export HCC_AMDGPU_TARGET=gfx900' >> /etc/environment
                    echo 'export GPU_DEVICE_ORDINAL=0,1,2,3,4,5,6,7' >> /etc/environment

                    # PyTorch ROCm compatibility
                    echo 'export PYTORCH_ROCM_ARCH=gfx900' >> /etc/environment

                    # TensorFlow ROCm compatibility
                    echo 'export TF_ROCM_AMDGPU_TARGETS=gfx900' >> /etc/environment
                fi

                # Create enhanced ROCm info script for verification
                cat > /opt/gator/check_rocm.sh << 'EOF'
#!/bin/bash
echo "=== ROCm Installation Check ==="
echo ""
echo "ROCm Version:"
cat /opt/rocm/.info/version 2>/dev/null || echo "  Version file not found"
echo ""

echo "Kernel Version:"
uname -r
echo ""

echo "GPU Devices (lspci):"
lspci | grep -i "vga\|display\|3d" | grep -i "amd\|advanced micro"
echo ""

echo "GPU Devices (rocm-smi):"
if command -v /opt/rocm/bin/rocm-smi &> /dev/null; then
    /opt/rocm/bin/rocm-smi --showid --showproductname 2>/dev/null || echo "  rocm-smi available but failed to execute"
else
    echo "  rocm-smi not available"
fi
echo ""

echo "HIP Platform:"
if command -v /opt/rocm/bin/hipconfig &> /dev/null; then
    /opt/rocm/bin/hipconfig --platform 2>/dev/null || echo "  hipconfig available but failed"
else
    echo "  hipconfig not available"
fi
echo ""

echo "GPU Architecture Detection:"
if command -v /opt/rocm/bin/rocminfo &> /dev/null; then
    /opt/rocm/bin/rocminfo | grep -i "name\|gfx" | head -10
else
    echo "  rocminfo not available"
fi
echo ""

echo "Environment Variables:"
echo "  ROCM_PATH: ${ROCM_PATH:-not set}"
echo "  HIP_PATH: ${HIP_PATH:-not set}"
echo "  HIP_VISIBLE_DEVICES: ${HIP_VISIBLE_DEVICES:-not set}"
echo "  HSA_OVERRIDE_GFX_VERSION: ${HSA_OVERRIDE_GFX_VERSION:-not set}"
echo "  HCC_AMDGPU_TARGET: ${HCC_AMDGPU_TARGET:-not set}"
echo "  PYTORCH_ROCM_ARCH: ${PYTORCH_ROCM_ARCH:-not set}"
echo ""

echo "User Groups:"
groups | grep -q "render" && echo "  ✓ User is in 'render' group" || echo "  ✗ User NOT in 'render' group"
groups | grep -q "video" && echo "  ✓ User is in 'video' group" || echo "  ✗ User NOT in 'video' group"
echo ""

echo "ROCm Libraries:"
echo "  HIP: $(ls /opt/rocm/hip/lib/libamdhip64.so* 2>/dev/null | head -1 || echo 'not found')"
echo "  rocBLAS: $(ls /opt/rocm/lib/librocblas.so* 2>/dev/null | head -1 || echo 'not found')"
echo "  rocRAND: $(ls /opt/rocm/lib/librocrand.so* 2>/dev/null | head -1 || echo 'not found')"
echo ""

echo "=== End ROCm Check ==="
EOF
                chmod +x /opt/gator/check_rocm.sh

                log "✅ AMD ROCm support installed successfully"
                log "   • ROCm Version: $ROCM_VERSION"
                if [[ $IS_MI25 == true ]]; then
                    log "   • GPU Architecture: gfx900 (MI25/Vega10)"
                    log "   • HSA_OVERRIDE_GFX_VERSION=9.0.0 configured for compatibility"
                fi
                log "   • Configured for multiple GPU support (up to 8 GPUs)"
                log "   • Use '/opt/gator/check_rocm.sh' to verify installation"
                log ""
                log "⚠️  Important: After reboot, verify ROCm with:"
                log "      sudo -u $GATOR_USER /opt/gator/check_rocm.sh"
                log "      rocm-smi"
            fi
        else
            warn "No AMD GPU detected. Skipping ROCm installation."
        fi
    fi
    
    if [[ $INSTALL_NVIDIA_SUPPORT == false && $INSTALL_ROCM_SUPPORT == false ]]; then
        warn "No compatible GPU detected or GPU support disabled."
        INSTALL_GPU_SUPPORT=false
    else
        warn "GPU drivers installed. System reboot recommended after installation completes."
    fi
fi

# Create gator user
log "👤 Creating gator user..."
if ! id "$GATOR_USER" &>/dev/null; then
    adduser --system --group --home $GATOR_HOME --shell /bin/bash $GATOR_USER
    usermod -aG sudo $GATOR_USER
else
    warn "User $GATOR_USER already exists"
fi

# Create gator directory structure
log "📁 Setting up directory structure..."
mkdir -p $GATOR_HOME/{app,data,logs,backups,ssl}
mkdir -p $GATOR_HOME/data/{generated_content,uploads,models}
chown -R $GATOR_USER:$GATOR_USER $GATOR_HOME

# Clone Gator repository
log "📥 Cloning Gator repository..."
if [[ ! -d "$GATOR_HOME/app/.git" ]]; then
    sudo -u $GATOR_USER git clone https://github.com/terminills/gator.git $GATOR_HOME/app
else
    log "Repository already exists, pulling latest changes..."
    cd $GATOR_HOME/app
    sudo -u $GATOR_USER git pull origin main
fi

cd $GATOR_HOME/app

# Install Python dependencies
log "📦 Installing Python dependencies..."
sudo -u $GATOR_USER python -m venv $GATOR_HOME/venv
sudo -u $GATOR_USER $GATOR_HOME/venv/bin/pip install --upgrade pip
sudo -u $GATOR_USER $GATOR_HOME/venv/bin/pip install wheel setuptools
if [[ $INSTALL_ROCM_SUPPORT == true ]]; then
    sudo -u $GATOR_USER $GATOR_HOME/venv/bin/pip install -e .[rocm] --index-url https://download.pytorch.org/whl/rocm5.7.1 --extra-index-url https://pypi.org/simple
else
    sudo -u $GATOR_USER $GATOR_HOME/venv/bin/pip install -e . --extra-index-url https://pypi.org/simple
fi
sudo -u $GATOR_USER $GATOR_HOME/venv/bin/pip install gunicorn uvicorn[standard] supervisor

# Verify backend.models.content module
log "🔍 Verifying backend.models.content module..."
if [[ ! -f "$GATOR_HOME/app/src/backend/models/content.py" ]]; then
    warn "backend.models.content module not found. Creating placeholder to allow database setup."
    sudo -u $GATOR_USER mkdir -p $GATOR_HOME/app/src/backend/models
    sudo -u $GATOR_USER touch $GATOR_HOME/app/src/backend/models/content.py
    sudo -u $GATOR_USER bash -c "echo '# Placeholder for backend.models.content' > $GATOR_HOME/app/src/backend/models/content.py"
    warn "Placeholder created. Check repository for missing content.py or update __init__.py imports."
fi

# Setup AI models and dependencies
log "🤖 Setting up AI models..."
cd $GATOR_HOME/app
if [[ -f "setup_ai_models.py" ]]; then
    sudo -u $GATOR_USER $GATOR_HOME/venv/bin/python setup_ai_models.py --models-dir $GATOR_HOME/data/models 2>&1 | tee $GATOR_HOME/logs/setup_ai_models.log || {
        warn "AI model setup failed. Check $GATOR_HOME/logs/setup_ai_models.log for details."
        warn "The platform will fall back to API-based models."
        warn "To debug, verify the setup_ai_models.py script and its dependencies."
    }
else
    warn "setup_ai_models.py not found in $GATOR_HOME/app. Skipping AI model setup."
    warn "The platform will fall back to API-based models."
fi

# Create environment file
log "⚙️ Creating environment configuration..."
if [[ ! -f "$GATOR_HOME/app/.env" ]]; then
    sudo -u $GATOR_USER cp $GATOR_HOME/app/.env.template $GATOR_HOME/app/.env
    
    # Generate random secret key
    SECRET_KEY=$(openssl rand -hex 32)
    
    # Update environment file
    sudo -u $GATOR_USER sed -i "s/your_secret_key_here/$SECRET_KEY/g" $GATOR_HOME/app/.env
    sudo -u $GATOR_USER sed -i "s|postgresql://user:password@localhost:5432/gator|postgresql://$GATOR_USER:$GATOR_DB_PASSWORD@localhost:5432/gator_production|g" $GATOR_HOME/app/.env
    
    if [[ -n "$DOMAIN" ]]; then
        sudo -u $GATOR_USER sed -i "s/localhost/$DOMAIN/g" $GATOR_HOME/app/.env
    fi
    
    log "Environment file created. Please update API keys in $GATOR_HOME/app/.env"
else
    log "Environment file already exists. Updating database password..."
    sudo -u $GATOR_USER sed -i "s|postgresql://$GATOR_USER:.*@localhost:5432/gator_production|postgresql://$GATOR_USER:$GATOR_DB_PASSWORD@localhost:5432/gator_production|g" $GATOR_HOME/app/.env
fi

# Setup database
log "🗃️ Initializing database..."
cd $GATOR_HOME/app
if [[ -f "setup_db.py" ]]; then
    sudo -u $GATOR_USER $GATOR_HOME/venv/bin/python setup_db.py 2>&1 | tee $GATOR_HOME/logs/setup_db.log || {
        warn "Database setup failed. Check $GATOR_HOME/logs/setup_db.log for details."
        warn "Ensure all dependencies are installed and check setup_db.py for errors."
    }
else
    warn "setup_db.py not found in $GATOR_HOME/app. Skipping database setup."
    warn "Manually initialize the database using the appropriate setup script."
fi

# Create systemd service
log "🔧 Creating systemd service..."
cat > /etc/systemd/system/gator.service << EOF
[Unit]
Description=Gator AI Influencer Platform
After=network.target postgresql.service redis-server.service

[Service]
User=$GATOR_USER
Group=$GATOR_USER
WorkingDirectory=$GATOR_HOME/app
Environment="PATH=$GATOR_HOME/venv/bin:/opt/rocm/bin:/opt/rocm/opencl/bin:/usr/local/cuda/bin:$PATH"
Environment="LD_LIBRARY_PATH=/opt/rocm/lib:/opt/rocm/lib64:/usr/local/cuda/lib64:$LD_LIBRARY_PATH"
Environment="ROCM_PATH=/opt/rocm"
Environment="HIP_PATH=/opt/rocm/hip"
Environment="HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7"
Environment="ROCR_VISIBLE_DEVICES=0,1,2,3,4,5,6,7"
Environment="HSA_OVERRIDE_GFX_VERSION=9.0.0"
Environment="HCC_AMDGPU_TARGET=gfx900"
Environment="PYTORCH_ROCM_ARCH=gfx900"
Environment="TF_ROCM_AMDGPU_TARGETS=gfx900"
ExecStart=$GATOR_HOME/venv/bin/gunicorn -w 4 -k uvicorn.workers.UvicornWorker main:app -b 0.0.0.0:8000
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload
systemctl enable gator
systemctl start gator

# Configure firewall
if [[ $SKIP_FIREWALL == false ]]; then
    log "🔥 Configuring firewall..."
    apt install -y ufw
    
    # Backup existing UFW rules
    timestamp=$(date +%Y%m%d_%H%M%S)
    for file in /etc/ufw/{user,before,after}{,6}.rules; do
        if [[ -f "$file" ]]; then
            cp "$file" "${file}.${timestamp}"
            log "Backing up '$file' to '${file}.${timestamp}'"
        fi
    done
    
    # Configure UFW
    ufw default deny incoming
    ufw default allow outgoing
    ufw allow 22/tcp    # SSH
    ufw allow 80/tcp    # HTTP
    ufw allow 443/tcp   # HTTPS
    ufw allow 8000/tcp  # Gator app
    ufw allow 5432/tcp  # PostgreSQL
    ufw allow 6379/tcp  # Redis
    ufw --force enable
fi

# Create backup script
log "💾 Creating backup script..."
cat > /opt/gator/backup.sh << 'EOF'
#!/bin/bash
BACKUP_DIR=/opt/gator/backups
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BACKUP_NAME="gator_backup_${TIMESTAMP}"
mkdir -p $BACKUP_DIR
sudo -u postgres pg_dump gator_production | gzip > $BACKUP_DIR/${BACKUP_NAME}.sql.gz
tar -czf $BACKUP_DIR/${BACKUP_NAME}_data.tar.gz -C /opt/gator data
find $BACKUP_DIR -type f -mtime +7 -delete
EOF
chmod +x /opt/gator/backup.sh

# Install SSL if requested
if [[ $INSTALL_SSL == true && -n "$DOMAIN" && -n "$EMAIL" ]]; then
    log "🔒 Installing SSL certificates..."
    apt install -y certbot python3-certbot-nginx
    certbot --nginx -d "$DOMAIN" --email "$EMAIL" --agree-tos --non-interactive || warn "SSL setup may need manual configuration"
fi

# Check system status
log "🔍 Checking system status..."
for service in postgresql redis-server gator; do
    if systemctl is-active --quiet $service; then
        log "✅ $service is running"
    else
        warn "$service is not running"
    fi
done

if [[ $INSTALL_NGINX == true ]]; then
    if systemctl is-active --quiet nginx; then
        log "✅ nginx is running"
    else
        warn "nginx is not running"
    fi
fi

# Installation summary
log ""
log "🎉 Gator AI Influencer Platform installation completed!"
log ""
log "📋 Installation Summary:"
log "   • Platform installed in: $GATOR_HOME"
log "   • Database: PostgreSQL (gator_production)"
log "   • Python environment: $GATOR_HOME/venv"
log "   • System service: gator.service"
log "   • Web server: $(if [[ $INSTALL_NGINX == true ]]; then echo 'NGINX'; else echo 'None'; fi)"
if [[ $INSTALL_GPU_SUPPORT == true ]]; then
    if [[ $INSTALL_ROCM_SUPPORT == true ]]; then
        log "   • GPU support: AMD ROCm $ROCM_VERSION (MI25/gfx900 optimized)"
    elif [[ $INSTALL_NVIDIA_SUPPORT == true ]]; then
        log "   • GPU support: NVIDIA CUDA"
    fi
fi
log ""
log "🚀 Next steps:"
log "   1. Database password set to: $GATOR_DB_PASSWORD"
log "   2. Configure API keys for social media platforms in: $GATOR_HOME/app/.env"
log "   3. Run demo: sudo -u $GATOR_USER /opt/gator/venv/bin/python $GATOR_HOME/app/demo.py"
log "   4. Access your platform: http://YOUR_SERVER_IP:8000"
log "   5. API documentation: http://YOUR_SERVER_IP:8000/docs"
log ""
log "📚 Important files:"
log "   • Main config: $GATOR_HOME/app/.env"
log "   • Service logs: journalctl -u gator -f"
log "   • Application logs: $GATOR_HOME/logs/"
log "   • Backup script: $GATOR_HOME/backup.sh"
if [[ $INSTALL_GPU_SUPPORT == true ]]; then
    log ""
    warn "⚠️ GPU drivers were installed. Please reboot the system:"
    warn "   sudo reboot"
    log ""
    warn "📋 After reboot, verify ROCm installation:"
    warn "   sudo -u $GATOR_USER /opt/gator/check_rocm.sh"
    warn "   rocm-smi    # Check GPU status"
    log ""
    warn "📋 MI25 (gfx900) specific notes:"
    warn "   • ROCm $ROCM_VERSION confirmed working with MI25"
    warn "   • HSA_OVERRIDE_GFX_VERSION=9.0.0 is set for compatibility"
    warn "   • For PyTorch: Ensure ROCm version is installed"
    warn "   • Some ML frameworks may need HSA override for gfx900 support"
fi
log ""
log "🦎 Gator AI Influencer Platform is ready!"
