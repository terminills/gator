# Database Management Features

## Overview

The Gator AI Platform includes comprehensive database management features accessible through the admin panel. These features allow administrators to maintain database integrity, create backups, and ensure schema synchronization with the codebase.

## Features

### 1. Database Information

View real-time information about your database:
- **Database Type**: SQLite (development) or PostgreSQL (production)
- **Connection Status**: Current database connection health
- **Table Count**: Number of tables in the database
- **Database Size**: Storage space used (SQLite only)
- **Table List**: Expandable list of all database tables

### 2. Schema Synchronization

Keep your database schema in sync with your codebase:

#### Check Schema Status
- Compare database tables with model definitions
- Identify missing tables that need to be created
- Detect extra tables that exist in database but not in models
- View table counts for both database and models

#### Sync Schema (One-Click)
- Automatically create missing tables based on current models
- Safe operation that only adds tables, never modifies existing ones
- Prevents data loss by not touching existing table structures
- Provides detailed feedback on changes made

**Use Cases:**
- After pulling code updates that include new models
- When setting up a new environment
- After database migrations
- Regular maintenance checks

### 3. Database Backup

Create and manage database backups:

#### Create Backup
- **SQLite**: Creates a copy of the database file
- **PostgreSQL**: Runs `pg_dump` to create SQL backup
- Automatic timestamped filenames (format: `gator_backup_YYYYMMDD_HHMMSS`)
- Immediate download link after creation
- Size information displayed

#### Backup Management
- List all available backups with metadata
- Sort by creation date (newest first)
- View backup size and creation timestamp
- One-click download for any backup
- Backups stored in `./backups` directory

## API Endpoints

All database management endpoints are under `/api/v1/admin/database/`:

### Database Information
```
GET /api/v1/admin/database/info
```
Returns database type, connection status, size, and table list.

### Schema Management
```
GET /api/v1/admin/database/schema/status
```
Check if database schema matches current models.

```
POST /api/v1/admin/database/schema/sync
```
Synchronize database schema with current models (creates missing tables).

### Backup Management
```
POST /api/v1/admin/database/backup
```
Create a new database backup.

```
GET /api/v1/admin/database/backups
```
List all available backups with metadata.

```
GET /api/v1/admin/database/backups/{filename}
```
Download a specific backup file.

## Usage Examples

### Via Admin Panel

1. **Access Database Management**
   - Navigate to `/admin` in your browser
   - Click on the "Database" tab in the navigation

2. **Check Schema Status**
   - Click "🔍 Check Schema Status" button
   - Review the sync status and any missing/extra tables
   - If tables are missing, click "⚡ Sync Schema Now"

3. **Create a Backup**
   - Click "📦 Create Backup Now" button
   - Wait for confirmation message
   - Click "📥 Download Backup" to save the file locally
   - Backup is also stored in the `./backups` directory

4. **Manage Backups**
   - View the "Available Backups" table
   - Download any previous backup by clicking its download link
   - Check backup sizes and creation dates

### Via API (curl)

```bash
# Check database info
curl http://localhost:8000/api/v1/admin/database/info

# Check schema status
curl http://localhost:8000/api/v1/admin/database/schema/status

# Sync schema
curl -X POST http://localhost:8000/api/v1/admin/database/schema/sync

# Create backup
curl -X POST http://localhost:8000/api/v1/admin/database/backup

# List backups
curl http://localhost:8000/api/v1/admin/database/backups

# Download backup
curl -O http://localhost:8000/api/v1/admin/database/backups/gator_backup_20251007_214314.db
```

### Via Python

```python
import httpx

async with httpx.AsyncClient() as client:
    # Check database info
    response = await client.get("http://localhost:8000/api/v1/admin/database/info")
    info = response.json()
    print(f"Database type: {info['database_type']}")
    print(f"Tables: {info['table_count']}")
    
    # Create backup
    response = await client.post("http://localhost:8000/api/v1/admin/database/backup")
    backup = response.json()
    print(f"Backup created: {backup['backup']['filename']}")
    
    # Sync schema
    response = await client.post("http://localhost:8000/api/v1/admin/database/schema/sync")
    result = response.json()
    print(f"Sync result: {result['message']}")
```

## Best Practices

### Regular Backups
- Create backups before major updates or migrations
- Schedule regular backups (daily/weekly depending on data criticality)
- Store important backups off-site or in cloud storage
- Test backup restoration periodically

### Schema Management
- Check schema status after pulling code updates
- Sync schema immediately after adding new models
- Review missing/extra tables before syncing
- Never manually modify the database schema

### Backup Storage
- Monitor backup directory size
- Archive old backups to prevent disk space issues
- Keep at least 3-5 recent backups
- Document backup retention policies

## Security Considerations

1. **Access Control**: Database management endpoints should be restricted to administrators
2. **Backup Security**: Backup files contain all database data - secure the `./backups` directory
3. **Path Traversal**: The API validates filenames to prevent path traversal attacks
4. **PostgreSQL Credentials**: Database passwords are not exposed in API responses

## Troubleshooting

### Backup Creation Fails
- **SQLite**: Ensure the database file exists and is readable
- **PostgreSQL**: Verify `pg_dump` is installed and credentials are correct
- Check disk space in the `./backups` directory
- Review server logs for detailed error messages

### Schema Sync Issues
- Ensure database connection is active (check `/health` endpoint)
- Verify models are properly imported in your application
- Check for conflicting table names
- Review the schema status before syncing

### Permission Errors
- Ensure write permissions on the `./backups` directory
- For PostgreSQL, verify database user has necessary privileges
- Check file system permissions for database files

## Related Documentation

- [Database Configuration](README.md#database-configuration)
- [API Documentation](README.md#api-documentation)
- [Maintenance & Updates](README.md#maintenance--updates)
