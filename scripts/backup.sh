#!/bin/bash
# Gator Platform Backup Script
# Automates database and content backups for Kubernetes deployments

set -e

# Configuration
NAMESPACE="${NAMESPACE:-gator-prod}"
BACKUP_DIR="${BACKUP_DIR:-/backups}"
RETENTION_DAYS="${RETENTION_DAYS:-30}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
NC='\033[0m'

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Main execution
main() {
    log_info "=== Gator Backup Script Started ==="
    log_info "Namespace: $NAMESPACE"
    log_info "Backup Directory: $BACKUP_DIR"
    log_info "Retention Days: $RETENTION_DAYS"
    
    # Create backup directory
    local timestamp=$(date +%Y%m%d_%H%M%S)
    BACKUP_PATH="$BACKUP_DIR/backup_$timestamp"
    mkdir -p "$BACKUP_PATH"
    
    # Backup database
    log_info "Backing up database..."
    kubectl exec -n "$NAMESPACE" $(kubectl get pod -n "$NAMESPACE" -l component=database -o jsonpath='{.items[0].metadata.name}') -- pg_dump -U gator gator | gzip > "$BACKUP_PATH/database.sql.gz"
    
    # Backup content
    log_info "Backing up content..."
    kubectl exec -n "$NAMESPACE" $(kubectl get pod -n "$NAMESPACE" -l component=api -o jsonpath='{.items[0].metadata.name}') -- tar czf - /app/generated_content > "$BACKUP_PATH/content.tar.gz"
    
    log_info "=== Backup Completed ==="
    log_info "Backup location: $BACKUP_PATH"
}

main "$@"
