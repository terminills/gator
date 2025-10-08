"""
Backup Background Tasks

Celery tasks for automated database and content backups.
"""

import asyncio
import os
import subprocess
from datetime import datetime, timedelta
from typing import Dict, Any
import shutil
from pathlib import Path

from backend.celery_app import app
from backend.config.settings import get_settings
from backend.config.logging import get_logger

logger = get_logger(__name__)
settings = get_settings()


@app.task(name='backend.tasks.backup_tasks.create_automated_backup')
def create_automated_backup() -> Dict[str, Any]:
    """
    Create automated backup of database and content.
    Runs daily via Celery beat at 2:30 AM.
    
    Returns:
        Dictionary with backup details and status
    """
    try:
        timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
        backup_dir = Path(getattr(settings, 'BACKUP_DIR', '/backups'))
        backup_path = backup_dir / f'backup_{timestamp}'
        
        # Create backup directory
        backup_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Starting automated backup: {backup_path}")
        
        # Backup database
        db_backup_file = backup_path / 'database.sql.gz'
        db_result = _backup_database(db_backup_file)
        
        # Backup generated content
        content_backup_file = backup_path / 'content.tar.gz'
        content_result = _backup_content(content_backup_file)
        
        # Create backup metadata
        metadata = {
            'timestamp': timestamp,
            'backup_path': str(backup_path),
            'database_backup': str(db_backup_file) if db_result else None,
            'content_backup': str(content_backup_file) if content_result else None,
            'database_size_mb': _get_file_size_mb(db_backup_file) if db_result else 0,
            'content_size_mb': _get_file_size_mb(content_backup_file) if content_result else 0,
            'status': 'success' if (db_result and content_result) else 'partial',
            'created_at': datetime.utcnow().isoformat()
        }
        
        # Write metadata
        metadata_file = backup_path / 'metadata.json'
        with open(metadata_file, 'w') as f:
            import json
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Backup completed: {backup_path}")
        
        # Trigger cleanup of old backups
        cleanup_old_backups.delay()
        
        return metadata
        
    except Exception as e:
        logger.error(f"Failed to create automated backup: {str(e)}")
        return {
            'status': 'failed',
            'error': str(e),
            'timestamp': datetime.utcnow().isoformat()
        }


@app.task(name='backend.tasks.backup_tasks.cleanup_old_backups')
def cleanup_old_backups() -> Dict[str, Any]:
    """
    Clean up backups older than retention period.
    Default retention: 30 days
    
    Returns:
        Dictionary with cleanup statistics
    """
    try:
        backup_dir = Path(getattr(settings, 'BACKUP_DIR', '/backups'))
        retention_days = getattr(settings, 'BACKUP_RETENTION_DAYS', 30)
        cutoff_date = datetime.utcnow() - timedelta(days=retention_days)
        
        if not backup_dir.exists():
            logger.warning(f"Backup directory does not exist: {backup_dir}")
            return {'removed': 0, 'error': 'Backup directory not found'}
        
        logger.info(f"Cleaning up backups older than {retention_days} days")
        
        removed_count = 0
        freed_space_mb = 0
        
        # Iterate through backup directories
        for backup_path in backup_dir.iterdir():
            if not backup_path.is_dir() or not backup_path.name.startswith('backup_'):
                continue
            
            # Check if backup is older than retention period
            metadata_file = backup_path / 'metadata.json'
            if metadata_file.exists():
                import json
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                    
                backup_date = datetime.fromisoformat(metadata.get('created_at', '2000-01-01'))
                
                if backup_date < cutoff_date:
                    # Calculate size before deletion
                    backup_size = _get_directory_size_mb(backup_path)
                    
                    # Remove the backup directory
                    shutil.rmtree(backup_path)
                    
                    removed_count += 1
                    freed_space_mb += backup_size
                    
                    logger.info(f"Removed old backup: {backup_path.name}")
        
        return {
            'removed': removed_count,
            'freed_space_mb': round(freed_space_mb, 2),
            'retention_days': retention_days,
            'cutoff_date': cutoff_date.isoformat(),
            'timestamp': datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to cleanup old backups: {str(e)}")
        return {
            'removed': 0,
            'error': str(e),
            'timestamp': datetime.utcnow().isoformat()
        }


def _backup_database(output_file: Path) -> bool:
    """
    Backup database to compressed SQL file.
    
    Args:
        output_file: Path to output SQL.gz file
        
    Returns:
        True if backup succeeded, False otherwise
    """
    try:
        database_url = settings.DATABASE_URL
        
        # For SQLite
        if 'sqlite' in database_url:
            db_path = database_url.replace('sqlite:///', '')
            
            # Use sqlite3 to dump database
            with open(output_file.with_suffix('.sql'), 'w') as f:
                subprocess.run(
                    ['sqlite3', db_path, '.dump'],
                    stdout=f,
                    check=True
                )
            
            # Compress the dump
            subprocess.run(
                ['gzip', str(output_file.with_suffix('.sql'))],
                check=True
            )
            
            logger.info(f"SQLite database backed up: {output_file}")
            return True
            
        # For PostgreSQL
        elif 'postgresql' in database_url:
            # Extract connection details from URL
            # Format: postgresql://user:pass@host:port/dbname
            
            subprocess.run(
                ['pg_dump', database_url],
                stdout=subprocess.PIPE,
                check=True
            )
            
            # Compress output
            with open(output_file, 'wb') as f:
                subprocess.run(
                    ['gzip'],
                    input=subprocess.run(['pg_dump', database_url], stdout=subprocess.PIPE).stdout,
                    stdout=f,
                    check=True
                )
            
            logger.info(f"PostgreSQL database backed up: {output_file}")
            return True
        
        else:
            logger.warning(f"Unsupported database type for backup: {database_url}")
            return False
            
    except Exception as e:
        logger.error(f"Failed to backup database: {str(e)}")
        return False


def _backup_content(output_file: Path) -> bool:
    """
    Backup generated content directory to compressed tar file.
    
    Args:
        output_file: Path to output tar.gz file
        
    Returns:
        True if backup succeeded, False otherwise
    """
    try:
        content_dir = Path(getattr(settings, 'CONTENT_STORAGE_PATH', 'generated_content'))
        
        if not content_dir.exists():
            logger.warning(f"Content directory does not exist: {content_dir}")
            return False
        
        # Create tar.gz archive
        subprocess.run(
            ['tar', 'czf', str(output_file), '-C', str(content_dir.parent), content_dir.name],
            check=True
        )
        
        logger.info(f"Content directory backed up: {output_file}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to backup content: {str(e)}")
        return False


def _get_file_size_mb(file_path: Path) -> float:
    """Get file size in megabytes."""
    if file_path.exists():
        return file_path.stat().st_size / (1024 * 1024)
    return 0.0


def _get_directory_size_mb(dir_path: Path) -> float:
    """Get total size of directory in megabytes."""
    total_size = 0
    for path in dir_path.rglob('*'):
        if path.is_file():
            total_size += path.stat().st_size
    return total_size / (1024 * 1024)


@app.task(name='backend.tasks.backup_tasks.restore_backup')
def restore_backup(backup_path: str) -> Dict[str, Any]:
    """
    Restore database and content from a backup.
    
    Args:
        backup_path: Path to backup directory
        
    Returns:
        Dictionary with restore status
    """
    try:
        backup_dir = Path(backup_path)
        
        if not backup_dir.exists():
            raise ValueError(f"Backup not found: {backup_path}")
        
        logger.info(f"Starting backup restore: {backup_path}")
        
        # Restore database
        db_backup_file = backup_dir / 'database.sql.gz'
        db_result = _restore_database(db_backup_file)
        
        # Restore content
        content_backup_file = backup_dir / 'content.tar.gz'
        content_result = _restore_content(content_backup_file)
        
        return {
            'status': 'success' if (db_result and content_result) else 'partial',
            'database_restored': db_result,
            'content_restored': content_result,
            'backup_path': backup_path,
            'restored_at': datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to restore backup: {str(e)}")
        return {
            'status': 'failed',
            'error': str(e),
            'backup_path': backup_path
        }


def _restore_database(backup_file: Path) -> bool:
    """
    Restore database from compressed SQL file.
    
    Args:
        backup_file: Path to SQL.gz file
        
    Returns:
        True if restore succeeded, False otherwise
    """
    try:
        if not backup_file.exists():
            logger.warning(f"Database backup file not found: {backup_file}")
            return False
        
        database_url = settings.DATABASE_URL
        
        # For SQLite
        if 'sqlite' in database_url:
            db_path = database_url.replace('sqlite:///', '')
            
            # Decompress and restore
            with open(backup_file, 'rb') as f:
                subprocess.run(
                    ['gunzip', '-c'],
                    stdin=f,
                    stdout=subprocess.PIPE
                )
            
            subprocess.run(
                ['sqlite3', db_path],
                input=subprocess.run(['gunzip', '-c', str(backup_file)], stdout=subprocess.PIPE).stdout,
                check=True
            )
            
            logger.info(f"SQLite database restored from: {backup_file}")
            return True
            
        # For PostgreSQL
        elif 'postgresql' in database_url:
            subprocess.run(
                ['gunzip', '-c', str(backup_file)],
                stdout=subprocess.PIPE
            )
            
            subprocess.run(
                ['psql', database_url],
                input=subprocess.run(['gunzip', '-c', str(backup_file)], stdout=subprocess.PIPE).stdout,
                check=True
            )
            
            logger.info(f"PostgreSQL database restored from: {backup_file}")
            return True
        
        else:
            logger.warning(f"Unsupported database type for restore: {database_url}")
            return False
            
    except Exception as e:
        logger.error(f"Failed to restore database: {str(e)}")
        return False


def _restore_content(backup_file: Path) -> bool:
    """
    Restore content directory from compressed tar file.
    
    Args:
        backup_file: Path to tar.gz file
        
    Returns:
        True if restore succeeded, False otherwise
    """
    try:
        if not backup_file.exists():
            logger.warning(f"Content backup file not found: {backup_file}")
            return False
        
        content_dir = Path(getattr(settings, 'CONTENT_STORAGE_PATH', 'generated_content'))
        
        # Extract tar.gz archive
        subprocess.run(
            ['tar', 'xzf', str(backup_file), '-C', str(content_dir.parent)],
            check=True
        )
        
        logger.info(f"Content directory restored from: {backup_file}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to restore content: {str(e)}")
        return False
