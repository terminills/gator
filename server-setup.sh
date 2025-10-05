#!/bin/bash

# Gator AI Influencer Platform - Automated Server Setup Script
# Compatible with Ubuntu 20.04+, Debian 11+
# Usage: curl -sSL https://raw.githubusercontent.com/terminills/gator/main/server-setup.sh | sudo bash
# With options: sudo bash server-setup.sh --domain example.com --email admin@example.com

set -euo pipefail

# Script configuration
SCRIPT_VERSION="1.0.0"
GATOR_USER="gator"
GATOR_HOME="/opt/gator"
PYTHON_VERSION="3.9"
NODE_VERSION="18"

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
  --rocm                  Install AMD GPU support (ROCm)
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

# Create PostgreSQL database and user
log "🔐 Setting up PostgreSQL database..."
sudo -u postgres createuser --createdb --login --pwprompt ${GATOR_USER} || warn "User ${GATOR_USER} may already exist"
sudo -u postgres createdb gator_production --owner=${GATOR_USER} || warn "Database may already exist"

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
        log "🔴 Installing AMD ROCm support..."
        
        if lspci | grep -i amd > /dev/null || lspci | grep -i "advanced micro devices" > /dev/null; then
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
                log "AMD MI25 (gfx900/Vega10) GPU detected - using ROCm 4.5.2 for optimal compatibility..."
                ROCM_VERSION="4.5.2"
                GFX_ARCH="gfx900"
                
                # Check kernel version - ROCm 4.5.2 on Ubuntu 20.04 requires kernel 5.4+
                KERNEL_VERSION=$(uname -r | cut -d. -f1-2)
                KERNEL_MAJOR=$(echo $KERNEL_VERSION | cut -d. -f1)
                KERNEL_MINOR=$(echo $KERNEL_VERSION | cut -d. -f2)
                
                if [[ $KERNEL_MAJOR -lt 5 ]] || [[ $KERNEL_MAJOR -eq 5 && $KERNEL_MINOR -lt 4 ]]; then
                    warn "Kernel version $KERNEL_VERSION detected. ROCm 4.5.2 recommends kernel 5.4 or higher."
                    warn "Consider upgrading kernel: sudo apt install linux-generic-hwe-20.04"
                fi
            else
                log "Using latest ROCm version..."
                ROCM_VERSION="5.7.1"
                GFX_ARCH="auto"
            fi
            
            # Install ROCm repository with GPG key
            log "Adding ROCm repository..."
            wget -q -O - https://repo.radeon.com/rocm/rocm.gpg.key | apt-key add -
            
            # Determine Ubuntu version for repository URL
            UBUNTU_VERSION=$(lsb_release -rs)
            if [[ "$UBUNTU_VERSION" == "20.04" ]]; then
                if [[ "$ROCM_VERSION" == "4.5.2" ]]; then
                    # ROCm 4.5.2 for Ubuntu 20.04 uses specific repository structure
                    REPO_URL="deb [arch=amd64] https://repo.radeon.com/rocm/apt/4.5.2 ubuntu main"
                else
                    REPO_URL="deb [arch=amd64] https://repo.radeon.com/rocm/apt/$ROCM_VERSION ubuntu main"
                fi
            elif [[ "$UBUNTU_VERSION" == "22.04" ]]; then
                REPO_URL="deb [arch=amd64] https://repo.radeon.com/rocm/apt/$ROCM_VERSION jammy main"
            else
                warn "Ubuntu version $UBUNTU_VERSION may not be fully supported. Using 20.04 repository."
                REPO_URL="deb [arch=amd64] https://repo.radeon.com/rocm/apt/$ROCM_VERSION ubuntu main"
            fi
            
            echo "$REPO_URL" > /etc/apt/sources.list.d/rocm.list
            apt update
            
            # Install ROCm packages
            log "Installing ROCm runtime and development packages..."
            # For ROCm 4.5.2 on Ubuntu 20.04, install in specific order
            if [[ "$ROCM_VERSION" == "4.5.2" ]]; then
                log "Installing ROCm 4.5.2 packages for MI25 compatibility..."
                apt install -y rocm-dkms rocm-dev rocm-libs rocm-utils || {
                    warn "Some ROCm packages failed to install. Trying alternative package names..."
                    apt install -y rocm-dkms rocm-dev rocm-device-libs rocm-opencl-dev
                }
            else
                apt install -y rocm-dkms rocm-libs rocm-dev rocm-utils
            fi
            
            # Install additional packages for AI/ML workloads
            log "Installing HIP and ROCm math libraries..."
            if [[ "$ROCM_VERSION" == "4.5.2" ]]; then
                # ROCm 4.5.2 package names
                apt install -y hip-runtime-amd hip-dev rocrand rocblas rocsparse rocsolver rocfft || {
                    warn "Some math libraries may not be available for ROCm 4.5.2"
                }
            else
                apt install -y hip-runtime-amd hip-dev rocrand-dev rocblas-dev rocsparse-dev rocsolver-dev rocfft-dev
            fi
            
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
sudo -u $GATOR_USER $GATOR_HOME/venv/bin/pip install -e .

# Install development dependencies for production builds
sudo -u $GATOR_USER $GATOR_HOME/venv/bin/pip install gunicorn uvicorn[standard] supervisor

# Setup AI models and dependencies
log "🤖 Setting up AI models..."
cd $GATOR_HOME/app
if [[ $INSTALL_GPU_SUPPORT == true ]]; then
    # Run AI model setup with GPU support
    sudo -u $GATOR_USER $GATOR_HOME/venv/bin/python setup_ai_models.py --models-dir $GATOR_HOME/data/models --types text image || warn "AI model setup failed - will use API-based models"
else
    # Run AI model setup without GPU-intensive models
    sudo -u $GATOR_USER $GATOR_HOME/venv/bin/python setup_ai_models.py --models-dir $GATOR_HOME/data/models --types text --no-install || warn "AI model setup failed - will use API-based models"
fi

# Create environment file
log "⚙️ Creating environment configuration..."
if [[ ! -f "$GATOR_HOME/app/.env" ]]; then
    sudo -u $GATOR_USER cp $GATOR_HOME/app/.env.template $GATOR_HOME/app/.env
    
    # Generate random secret key
    SECRET_KEY=$(openssl rand -hex 32)
    
    # Update environment file
    sudo -u $GATOR_USER sed -i "s/your_secret_key_here/$SECRET_KEY/g" $GATOR_HOME/app/.env
    sudo -u $GATOR_USER sed -i "s|postgresql://user:password@localhost:5432/gator|postgresql://$GATOR_USER:password@localhost:5432/gator_production|g" $GATOR_HOME/app/.env
    
    if [[ -n "$DOMAIN" ]]; then
        sudo -u $GATOR_USER sed -i "s/localhost/$DOMAIN/g" $GATOR_HOME/app/.env
    fi
    
    log "Environment file created. Please update database password and API keys in $GATOR_HOME/app/.env"
fi

# Setup database
log "🗃️ Initializing database..."
cd $GATOR_HOME/app
sudo -u $GATOR_USER $GATOR_HOME/venv/bin/python setup_db.py || warn "Database setup may have failed"

# Create systemd service
log "🔧 Creating systemd service..."
cat > /etc/systemd/system/gator.service << EOF
[Unit]
Description=Gator AI Influencer Platform
After=network.target postgresql.service redis.service
Requires=postgresql.service redis.service

[Service]
Type=exec
User=$GATOR_USER
Group=$GATOR_USER
WorkingDirectory=$GATOR_HOME/app/src
Environment=PATH=$GATOR_HOME/venv/bin
ExecStart=$GATOR_HOME/venv/bin/uvicorn backend.api.main:app --host 0.0.0.0 --port 8000 --workers 4
ExecReload=/bin/kill -s HUP \$MAINPID
Restart=always
RestartSec=5
StandardOutput=journal
StandardError=journal
SyslogIdentifier=gator

[Install]
WantedBy=multi-user.target
EOF

# Enable and start service
systemctl daemon-reload
systemctl enable gator
systemctl start gator

# Configure NGINX if installed
if [[ $INSTALL_NGINX == true && -n "$DOMAIN" ]]; then
    log "🌐 Configuring NGINX..."
    
    cat > /etc/nginx/sites-available/gator << EOF
server {
    listen 80;
    server_name $DOMAIN www.$DOMAIN;
    
    # Security headers
    add_header X-Content-Type-Options nosniff;
    add_header X-Frame-Options DENY;
    add_header X-XSS-Protection "1; mode=block";
    
    # Gator API
    location /api/ {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
        proxy_read_timeout 300s;
        proxy_connect_timeout 300s;
        proxy_send_timeout 300s;
    }
    
    # Gator docs
    location /docs {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
    }
    
    # Frontend static files
    location / {
        root $GATOR_HOME/app/frontend/public;
        try_files \$uri \$uri/ /index.html;
        expires 1d;
        add_header Cache-Control "public, no-transform";
    }
    
    # Generated content
    location /content/ {
        alias $GATOR_HOME/data/generated_content/;
        expires 7d;
        add_header Cache-Control "public, no-transform";
    }
    
    # File upload size limit
    client_max_body_size 100M;
}
EOF

    ln -sf /etc/nginx/sites-available/gator /etc/nginx/sites-enabled/
    rm -f /etc/nginx/sites-enabled/default
    nginx -t && systemctl reload nginx
fi

# Setup SSL certificates if requested
if [[ $INSTALL_SSL == true && $INSTALL_NGINX == true && -n "$DOMAIN" && -n "$EMAIL" ]]; then
    log "🔒 Setting up SSL certificates..."
    
    # Install certbot
    apt install -y certbot python3-certbot-nginx
    
    # Get SSL certificate
    certbot --nginx -d "$DOMAIN" -d "www.$DOMAIN" --email "$EMAIL" --agree-tos --non-interactive --redirect
    
    # Setup auto-renewal
    systemctl enable certbot.timer
    systemctl start certbot.timer
fi

# Configure firewall if not skipped
if [[ $SKIP_FIREWALL == false ]]; then
    log "🔥 Configuring firewall..."
    
    ufw --force reset
    ufw default deny incoming
    ufw default allow outgoing
    ufw allow ssh
    ufw allow http
    ufw allow https
    ufw allow 8000/tcp  # Gator API (development)
    
    # Enable firewall
    ufw --force enable
fi

# Create backup script
log "💾 Creating backup script..."
cat > $GATOR_HOME/backup.sh << 'EOF'
#!/bin/bash

# Gator backup script
BACKUP_DIR="/opt/gator/backups"
DATE=$(date +%Y%m%d_%H%M%S)
BACKUP_NAME="gator_backup_$DATE"

mkdir -p "$BACKUP_DIR"

# Backup database
sudo -u postgres pg_dump gator_production > "$BACKUP_DIR/${BACKUP_NAME}_db.sql"

# Backup application data
tar -czf "$BACKUP_DIR/${BACKUP_NAME}_data.tar.gz" \
    -C /opt/gator \
    app/.env \
    data/generated_content \
    data/uploads \
    logs

# Backup configuration
tar -czf "$BACKUP_DIR/${BACKUP_NAME}_config.tar.gz" \
    /etc/nginx/sites-available/gator \
    /etc/systemd/system/gator.service

# Clean old backups (keep last 7 days)
find "$BACKUP_DIR" -name "gator_backup_*" -mtime +7 -delete

echo "Backup completed: $BACKUP_NAME"
EOF

chmod +x $GATOR_HOME/backup.sh
chown $GATOR_USER:$GATOR_USER $GATOR_HOME/backup.sh

# Setup daily backups
cat > /etc/cron.d/gator-backup << EOF
# Gator daily backup
0 2 * * * $GATOR_USER $GATOR_HOME/backup.sh >> $GATOR_HOME/logs/backup.log 2>&1
EOF

# Create log rotation config
cat > /etc/logrotate.d/gator << EOF
$GATOR_HOME/logs/*.log {
    daily
    rotate 30
    compress
    delaycompress
    missingok
    notifempty
    create 644 $GATOR_USER $GATOR_USER
    postrotate
        systemctl reload gator
    endscript
}
EOF

# Final system status check
log "🔍 Checking system status..."

# Check services
services=("postgresql" "redis-server" "gator")
if [[ $INSTALL_NGINX == true ]]; then
    services+=("nginx")
fi

for service in "${services[@]}"; do
    if systemctl is-active --quiet "$service"; then
        log "✅ $service is running"
    else
        warn "❌ $service is not running"
    fi
done

# Installation complete
log ""
log "🎉 Gator AI Influencer Platform installation completed!"
log ""
log "📋 Installation Summary:"
log "   • Platform installed in: $GATOR_HOME"
log "   • Database: PostgreSQL (gator_production)"
log "   • Python environment: $GATOR_HOME/venv"
log "   • System service: gator.service"
if [[ -n "$DOMAIN" ]]; then
    log "   • Domain: $DOMAIN"
fi
if [[ $INSTALL_NGINX == true ]]; then
    log "   • Web server: NGINX"
fi
if [[ $INSTALL_SSL == true && -n "$DOMAIN" ]]; then
    log "   • SSL: Enabled with Let's Encrypt"
fi
if [[ $INSTALL_GPU_SUPPORT == true ]]; then
    if [[ $INSTALL_NVIDIA_SUPPORT == true && $INSTALL_ROCM_SUPPORT == true ]]; then
        log "   • GPU support: NVIDIA CUDA + AMD ROCm installed"
    elif [[ $INSTALL_NVIDIA_SUPPORT == true ]]; then
        log "   • GPU support: NVIDIA CUDA installed"
    elif [[ $INSTALL_ROCM_SUPPORT == true ]]; then
        if [[ ${IS_MI25:-false} == true ]]; then
            log "   • GPU support: AMD ROCm 4.5.2 (MI25/gfx900 optimized)"
        else
            log "   • GPU support: AMD ROCm installed"
        fi
    else
        log "   • GPU support: Enabled but no compatible hardware detected"
    fi
fi

log ""
log "🚀 Next steps:"
log "   1. Update database password in: $GATOR_HOME/app/.env"
log "   2. Configure API keys for social media platforms"
log "   3. Run demo: sudo -u $GATOR_USER $GATOR_HOME/venv/bin/python $GATOR_HOME/app/demo.py"
if [[ -n "$DOMAIN" ]]; then
    log "   4. Access your platform: https://$DOMAIN"
    log "   5. API documentation: https://$DOMAIN/docs"
else
    log "   4. Access your platform: http://YOUR_SERVER_IP:8000"
    log "   5. API documentation: http://YOUR_SERVER_IP:8000/docs"
fi

log ""
log "📚 Important files:"
log "   • Main config: $GATOR_HOME/app/.env"
log "   • Service logs: journalctl -u gator -f"
log "   • Application logs: $GATOR_HOME/logs/"
log "   • Backup script: $GATOR_HOME/backup.sh"

if [[ $INSTALL_GPU_SUPPORT == true ]]; then
    warn ""
    warn "⚠️  GPU drivers were installed. Please reboot the system:"
    warn "   sudo reboot"
    if [[ $INSTALL_ROCM_SUPPORT == true ]]; then
        warn ""
        warn "📋 After reboot, verify ROCm installation:"
        warn "   sudo -u $GATOR_USER /opt/gator/check_rocm.sh"
        warn "   rocm-smi    # Check GPU status"
        if [[ ${IS_MI25:-false} == true ]]; then
            warn ""
            warn "📋 MI25 (gfx900) specific notes:"
            warn "   • ROCm 4.5.2 is the recommended version for MI25"
            warn "   • HSA_OVERRIDE_GFX_VERSION=9.0.0 is set for compatibility"
            warn "   • For PyTorch: Install pytorch-rocm from AMD repositories"
            warn "   • TensorFlow ROCm support for gfx900 requires TF 2.7 or earlier"
            warn "   • Some newer ML frameworks may need HSA override to work"
        fi
    fi
    if [[ $INSTALL_NVIDIA_SUPPORT == true ]]; then
        warn ""
        warn "📋 After reboot, verify NVIDIA installation:"
        warn "   nvidia-smi  # Check GPU status"
    fi
fi

log ""
log "🦎 Gator AI Influencer Platform is ready!"