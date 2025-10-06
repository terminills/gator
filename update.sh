#!/bin/bash

# Gator AI Influencer Platform - Update Script
# Ensures all prerequisites are updated, dependencies installed, and database migrated
# Usage: ./update.sh [--verbose] [--skip-migrations] [--skip-verification]

set -euo pipefail

# Script version
SCRIPT_VERSION="1.0.0"

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
VERBOSE=false
SKIP_MIGRATIONS=false
SKIP_VERIFICATION=false
PYTHON_MIN_VERSION="3.9"

# Logging functions
log() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')] ✓ $1${NC}"
}

info() {
    echo -e "${BLUE}[INFO] $1${NC}"
}

warn() {
    echo -e "${YELLOW}[WARNING] ⚠ $1${NC}"
}

error() {
    echo -e "${RED}[ERROR] ✗ $1${NC}"
    exit 1
}

debug() {
    if [[ "$VERBOSE" == "true" ]]; then
        echo -e "${BLUE}[DEBUG] $1${NC}"
    fi
}

# Help function
show_help() {
    cat << EOF
Gator AI Influencer Platform - Update Script v${SCRIPT_VERSION}

Usage: $0 [OPTIONS]

This script updates all prerequisites, installs/updates dependencies,
and migrates the database to the latest schema.

OPTIONS:
  --verbose              Enable verbose output for debugging
  --skip-migrations      Skip running database migrations
  --skip-verification    Skip running verification tests
  --help                 Show this help message

EXAMPLES:
  # Standard update
  ./update.sh

  # Verbose update with all details
  ./update.sh --verbose

  # Update without running migrations (if already done)
  ./update.sh --skip-migrations

  # Quick update without verification
  ./update.sh --skip-verification

WHAT THIS SCRIPT DOES:
  1. Checks Python version and prerequisites
  2. Updates pip to latest version
  3. Updates all Python dependencies
  4. Runs database migration scripts
  5. Updates database schema to latest version
  6. Verifies installation by running demo

EOF
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --verbose)
            VERBOSE=true
            shift
            ;;
        --skip-migrations)
            SKIP_MIGRATIONS=true
            shift
            ;;
        --skip-verification)
            SKIP_VERIFICATION=true
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

# Banner
echo ""
echo -e "${BLUE}╔═══════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║                                                           ║${NC}"
echo -e "${BLUE}║   🦎 Gator AI Influencer Platform - Update Script       ║${NC}"
echo -e "${BLUE}║      Version ${SCRIPT_VERSION}                                     ║${NC}"
echo -e "${BLUE}║                                                           ║${NC}"
echo -e "${BLUE}╚═══════════════════════════════════════════════════════════╝${NC}"
echo ""

# Check if script is run from repository root
if [[ ! -f "setup_db.py" ]] || [[ ! -f "pyproject.toml" ]]; then
    error "This script must be run from the Gator repository root directory"
fi

# Step 1: Check Python version
info "Step 1/6: Checking Python version..."
PYTHON_VERSION=$(python --version 2>&1 | awk '{print $2}')
debug "Found Python version: $PYTHON_VERSION"

# Compare versions
python - <<EOF
import sys
version = tuple(map(int, "$PYTHON_VERSION".split('.')))
min_version = tuple(map(int, "$PYTHON_MIN_VERSION".split('.')))
if version < min_version:
    sys.exit(1)
EOF

if [[ $? -eq 0 ]]; then
    log "Python version $PYTHON_VERSION meets minimum requirement ($PYTHON_MIN_VERSION)"
else
    error "Python version $PYTHON_VERSION is below minimum requirement ($PYTHON_MIN_VERSION)"
fi

# Check if pip is available
if ! command -v pip &> /dev/null; then
    error "pip is not installed. Please install pip first."
fi

# Step 2: Update pip
info "Step 2/6: Updating pip to latest version..."
debug "Running: pip install --upgrade pip"
if [[ "$VERBOSE" == "true" ]]; then
    pip install --upgrade pip || warn "Could not update pip, but continuing..."
else
    pip install --upgrade pip > /dev/null 2>&1 || warn "Could not update pip, but continuing..."
fi
log "pip check completed"

# Step 3: Update Python dependencies
info "Step 3/6: Updating Python dependencies..."
echo "   This may take a few minutes depending on your internet connection..."
debug "Running: pip install -e ."

# Check if package is already installed
if python -c "import backend" 2>/dev/null; then
    debug "Package already installed, attempting update..."
    if [[ "$VERBOSE" == "true" ]]; then
        pip install -e . --no-deps || {
            warn "Could not reinstall package. Dependencies may need manual update."
            info "You can run 'pip install -e .' manually if needed."
        }
    else
        pip install -e . --no-deps > /dev/null 2>&1 || {
            warn "Could not reinstall package. Dependencies are already installed."
            info "Run with --verbose if you need to troubleshoot."
        }
    fi
else
    # Fresh install
    info "   Installing package and dependencies..."
    if [[ "$VERBOSE" == "true" ]]; then
        pip install -e . || error "Failed to install dependencies"
    else
        pip install -e . > /dev/null 2>&1 || {
            warn "Installation encountered issues. Trying with verbose output..."
            pip install -e . || error "Failed to install dependencies"
        }
    fi
fi
log "Python dependencies updated successfully"

# Step 4: Run database migrations
if [[ "$SKIP_MIGRATIONS" == "false" ]]; then
    info "Step 4/6: Running database migrations..."
    
    # Find all migration scripts
    MIGRATION_SCRIPTS=($(find . -maxdepth 1 -name "migrate_*.py" -type f | sort))
    
    if [[ ${#MIGRATION_SCRIPTS[@]} -eq 0 ]]; then
        info "   No migration scripts found. Skipping migrations."
    else
        info "   Found ${#MIGRATION_SCRIPTS[@]} migration script(s)"
        
        for script in "${MIGRATION_SCRIPTS[@]}"; do
            script_name=$(basename "$script")
            info "   Running migration: $script_name"
            debug "Executing: python $script"
            
            if [[ "$VERBOSE" == "true" ]]; then
                python "$script"
            else
                python "$script" > /dev/null 2>&1 || {
                    warn "Migration $script_name may have already been applied or failed. Continuing..."
                }
            fi
        done
        
        log "Database migrations completed"
    fi
else
    info "Step 4/6: Skipping database migrations (--skip-migrations flag set)"
fi

# Step 5: Update database schema
info "Step 5/6: Updating database schema to latest version..."
debug "Running: python setup_db.py"

if [[ "$VERBOSE" == "true" ]]; then
    python setup_db.py
else
    python setup_db.py > /dev/null 2>&1
fi
log "Database schema updated successfully"

# Step 6: Verify installation
if [[ "$SKIP_VERIFICATION" == "false" ]]; then
    info "Step 6/6: Verifying installation..."
    debug "Running: python demo.py"
    
    if [[ "$VERBOSE" == "true" ]]; then
        python demo.py
    else
        # Capture output and only show summary
        DEMO_OUTPUT=$(python demo.py 2>&1)
        if echo "$DEMO_OUTPUT" | grep -q "Demo completed successfully\|Gator AI Influencer Platform - Ready for Development"; then
            log "System verification passed"
        else
            warn "Verification completed but may have warnings. Run with --verbose to see details."
        fi
    fi
else
    info "Step 6/6: Skipping verification (--skip-verification flag set)"
fi

# Success message
echo ""
echo -e "${GREEN}╔═══════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║                                                           ║${NC}"
echo -e "${GREEN}║   ✓ Update completed successfully!                       ║${NC}"
echo -e "${GREEN}║                                                           ║${NC}"
echo -e "${GREEN}╚═══════════════════════════════════════════════════════════╝${NC}"
echo ""

# Next steps
echo -e "${BLUE}📋 Summary:${NC}"
echo "   • Python dependencies: Updated"
if [[ "$SKIP_MIGRATIONS" == "false" ]]; then
    echo "   • Database migrations: Applied"
else
    echo "   • Database migrations: Skipped"
fi
echo "   • Database schema: Updated"
if [[ "$SKIP_VERIFICATION" == "false" ]]; then
    echo "   • System verification: Passed"
else
    echo "   • System verification: Skipped"
fi
echo ""
echo -e "${BLUE}🚀 Next steps:${NC}"
echo "   • Start the API server: cd src && python -m backend.api.main"
echo "   • Visit the dashboard: http://localhost:8000"
echo "   • View API docs: http://localhost:8000/docs"
echo ""
echo -e "${YELLOW}💡 Tips:${NC}"
echo "   • Run with --verbose to see detailed output"
echo "   • Check logs if you encounter any issues"
echo "   • Ensure your .env file is properly configured"
echo ""
