# IPMI Credentials Management Guide

## Overview

The Gator AI platform now supports storing IPMI (Intelligent Platform Management Interface) credentials in the database, allowing you to configure remote server fan control without editing environment variables or restarting the application.

## Features

✅ **Database Storage**: IPMI credentials are stored securely in the database  
✅ **Web UI Management**: Configure credentials via the `/admin/settings` page  
✅ **No Restart Required**: Credentials are dynamically loaded  
✅ **Sensitive Data Protection**: Passwords marked as sensitive in database  
✅ **Fallback Support**: Falls back to environment variables if no database credentials exist

## Setting IPMI Credentials

### Method 1: Admin Settings Page (Recommended)

1. Navigate to `/admin/settings` in your browser
2. Scroll to the "🌡️ IPMI / Server Management" section
3. Fill in the following fields:
   - **IPMI Host**: Your BMC/XCC IP address (e.g., `192.168.1.100`)
   - **IPMI Username**: BMC/XCC username (e.g., `USERID`)
   - **IPMI Password**: BMC/XCC password
   - **IPMI Interface**: Select interface type (default: `lanplus`)
4. Click "💾 Save IPMI Credentials"
5. Credentials are immediately available for use

### Method 2: API Endpoint

Use the bulk-update endpoint to set credentials programmatically:

```bash
curl -X POST http://localhost:8000/api/v1/settings/bulk-update \
  -H "Content-Type: application/json" \
  -d '{
    "ipmi_host": "192.168.1.100",
    "ipmi_username": "admin",
    "ipmi_password": "secretpass",
    "ipmi_interface": "lanplus"
  }'
```

### Method 3: Environment Variables (Fallback)

If no database credentials exist, the system falls back to environment variables:

```bash
export GATOR_IPMI_HOST=192.168.1.100
export GATOR_IPMI_USERNAME=admin
export GATOR_IPMI_PASSWORD=secretpass
export GATOR_IPMI_INTERFACE=lanplus
```

## Retrieving IPMI Credentials

### Get Individual Setting

```bash
curl http://localhost:8000/api/v1/settings/ipmi_host
```

Response:
```json
{
  "id": "uuid",
  "key": "ipmi_host",
  "category": "ipmi",
  "value": "192.168.1.100",
  "description": "BMC/XCC IP address or hostname for IPMI access",
  "is_sensitive": false,
  "is_active": true,
  "created_at": "2025-11-13T17:51:04",
  "updated_at": "2025-11-13T17:51:55"
}
```

### Get All IPMI Settings

```bash
curl http://localhost:8000/api/v1/settings/?category=ipmi
```

## Fan Control Service Integration

The `FanControlService` automatically loads IPMI credentials from the database:

```python
from backend.services.fan_control_service import FanControlService

# Initialize service - loads credentials from database automatically
service = FanControlService()

# Reload credentials dynamically (no restart needed)
await service.reload_credentials_from_db()

# Get control info (shows if credentials are configured)
info = service.get_control_info()
print(f"Credentials configured: {info['credentials_configured']}")
```

## Security Considerations

1. **Sensitive Data Marking**: 
   - `ipmi_username` and `ipmi_password` are marked as sensitive
   - Consider implementing encryption for sensitive values

2. **Access Control**:
   - Ensure `/admin/settings` is protected by authentication
   - Limit API access to authorized users only

3. **Network Security**:
   - Use IPMI over LAN (lanplus interface) for encrypted communication
   - Ensure BMC/XCC interface is on a secure management network

## IPMI Interface Options

| Interface | Description | Use Case |
|-----------|-------------|----------|
| `lanplus` | IPMI v2.0 over LAN with encryption | **Recommended** for remote access |
| `lan` | IPMI v1.5 over LAN (legacy) | Older hardware without v2.0 support |
| `open` | In-band/local access | Direct access on the server itself |

## Supported Server Manufacturers

The fan control service supports manufacturer-specific IPMI commands for:

- **Lenovo** (SR665 and similar servers)
- **Dell** (iDRAC)
- **HP** (iLO)
- **Supermicro**
- **Generic** (fallback for other hardware)

## Database Schema

IPMI settings are stored in the `system_settings` table:

```sql
CREATE TABLE system_settings (
    id UUID PRIMARY KEY,
    key VARCHAR(100) UNIQUE,
    category VARCHAR(50),  -- 'ipmi'
    value JSON,
    description TEXT,
    is_sensitive BOOLEAN,
    is_active BOOLEAN,
    created_at TIMESTAMP,
    updated_at TIMESTAMP
);
```

Default IPMI settings:
- `ipmi_host`: BMC/XCC IP address (non-sensitive)
- `ipmi_username`: BMC/XCC username (sensitive)
- `ipmi_password`: BMC/XCC password (sensitive)
- `ipmi_interface`: Interface type (non-sensitive, default: "lanplus")

## Troubleshooting

### Credentials Not Working

1. **Verify credentials are saved**:
   ```bash
   curl http://localhost:8000/api/v1/settings/ipmi_host
   ```

2. **Check IPMI tool availability**:
   ```bash
   which ipmitool
   # Install if missing: apt-get install ipmitool
   ```

3. **Test IPMI connection manually**:
   ```bash
   ipmitool -I lanplus -H 192.168.1.100 -U USERID -P PASSWORD sensor list
   ```

4. **Check fan control service logs**:
   ```bash
   # Look for: "IPMI credentials reloaded from database"
   # or: "Fan control service initialized ... with authentication"
   ```

### Settings Not Persisting

1. **Verify database connection**:
   ```bash
   sqlite3 gator.db "SELECT key, value FROM system_settings WHERE key LIKE 'ipmi%';"
   ```

2. **Check migration ran**:
   ```bash
   python migrate_add_settings.py
   ```

3. **Verify settings table exists**:
   ```bash
   sqlite3 gator.db ".tables" | grep system_settings
   ```

## Testing

Run the comprehensive test suite:

```bash
python test_ipmi_credentials.py
```

Expected output:
```
✅ All IPMI credentials tests passed!
   • IPMI credentials can be saved to database
   • IPMI credentials can be retrieved from database
   • Category filtering works correctly
   • Sensitive flags are properly set
   • Ready for use in /admin/settings page
```

## API Reference

### Settings Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/v1/settings/` | List all settings |
| GET | `/api/v1/settings/?category=ipmi` | List IPMI settings |
| GET | `/api/v1/settings/{key}` | Get specific setting |
| POST | `/api/v1/settings/` | Create new setting |
| PUT | `/api/v1/settings/{key}` | Update setting |
| POST | `/api/v1/settings/bulk-update` | Update multiple settings |
| DELETE | `/api/v1/settings/{key}` | Soft delete setting |

## Benefits Over Environment Variables

| Feature | Database | Environment Variables |
|---------|----------|----------------------|
| Update without restart | ✅ Yes | ❌ No |
| Web UI management | ✅ Yes | ❌ No |
| API access | ✅ Yes | ⚠️ Limited |
| Multi-tenant support | ✅ Yes | ❌ No |
| Audit trail | ✅ Yes | ❌ No |
| Sensitive data marking | ✅ Yes | ❌ No |

## Migration from Environment Variables

If you're currently using environment variables, credentials will be read from environment variables until you save them to the database via the admin page. Once saved, database credentials take precedence.

**Migration Steps:**
1. Note your current environment variables
2. Navigate to `/admin/settings`
3. Enter credentials in IPMI section
4. Click "Save IPMI Credentials"
5. Credentials are now in database
6. (Optional) Remove environment variables from `.env` file

## Future Enhancements

Planned improvements:
- [ ] Credential encryption at rest
- [ ] Credential rotation/expiry
- [ ] Multiple IPMI endpoint support
- [ ] Credential validation on save
- [ ] Audit logging for credential access
- [ ] Role-based access control for settings

## Support

For issues or questions:
1. Check server logs for IPMI-related messages
2. Verify ipmitool is installed and accessible
3. Test IPMI connection manually
4. Review the IPMI Authentication Guide for hardware-specific setup

## References

- [IPMI Specification](https://www.intel.com/content/www/us/en/servers/ipmi/ipmi-home.html)
- [ipmitool Documentation](https://github.com/ipmitool/ipmitool)
- Gator AI: `IPMI_AUTHENTICATION_GUIDE.md`
- Gator AI: `FAN_CONTROL_MANUFACTURER_GUIDE.md`
