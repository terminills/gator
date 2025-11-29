# IPMI Authentication Guide for Fan Control

## Overview

The fan control service now supports **IPMI authentication** for remote BMC/XCC access. This is required for server-class hardware including Lenovo SR665/SR665 v3 servers.

## Why Authentication is Needed

The original "Invalid command" error (0xc1) was caused by **missing IPMI authentication**, not unsupported commands. Server-class IPMI implementations require login credentials for remote (out-of-band) access:

- **Local/In-Band**: Commands run directly on the server (no authentication needed)
- **Remote/Out-of-Band**: Commands run from a remote system via network (authentication required)

## Configuration

### Method 1: Environment Variables (Recommended for Production)

Set these in your `.env` file or environment:

```bash
GATOR_IPMI_HOST=192.168.1.100         # BMC/XCC IP address or hostname
GATOR_IPMI_USERNAME=USERID            # BMC/XCC username
GATOR_IPMI_PASSWORD=PASSW0RD          # BMC/XCC password
GATOR_IPMI_INTERFACE=lanplus          # IPMI interface (default: lanplus)
```

### Method 2: API Configuration (Runtime)

Configure credentials via the REST API:

```bash
curl -X POST http://localhost:8000/api/v1/system/fans/credentials \
  -H "Content-Type: application/json" \
  -d '{
    "host": "192.168.1.100",
    "username": "USERID",
    "password": "PASSW0RD",
    "interface": "lanplus"
  }'
```

Response:
```json
{
  "success": true,
  "host": "192.168.1.100",
  "username": "USERID",
  "interface": "lanplus",
  "message": "IPMI credentials configured. Remote fan control via IPMI over LAN is now enabled.",
  "note": "Ensure IPMI over LAN is enabled in your BMC/XCC settings..."
}
```

## Lenovo Server Setup

### Step 1: Enable IPMI over LAN in XCC

1. Access XCC web interface (usually https://xcc-ip-address)
2. Navigate to **BMC Configuration > Network**
3. Select **IPMI** tab
4. Enable **IPMI over LAN**
5. Save settings

### Step 2: Get XCC Credentials

Default credentials (may vary):
- **Username**: `USERID`
- **Password**: `PASSW0RD`

⚠️ **Change default credentials** for security!

### Step 3: Configure Gator

Using environment variables:
```bash
GATOR_IPMI_HOST=<your-xcc-ip>
GATOR_IPMI_USERNAME=<your-username>
GATOR_IPMI_PASSWORD=<your-password>
```

Or using API:
```bash
curl -X POST http://localhost:8000/api/v1/system/fans/credentials \
  -H "Content-Type: application/json" \
  -d '{
    "host": "<your-xcc-ip>",
    "username": "<your-username>",
    "password": "<your-password>"
  }'
```

### Step 4: Verify Configuration

```bash
curl http://localhost:8000/api/v1/system/fans/info
```

Check for:
```json
{
  "credentials_configured": true,
  "ipmi_host_configured": true,
  "authentication_mode": "remote (out-of-band)",
  ...
}
```

### Step 5: Test Fan Control

```bash
# Get current status
curl http://localhost:8000/api/v1/system/fans

# Try setting fan speed (requires authentication)
curl -X POST http://localhost:8000/api/v1/system/fans/speed \
  -H "Content-Type: application/json" \
  -d '{"speed_percent": 50}'
```

## Command Examples

### Without Authentication (Local/In-Band)
```bash
# Runs on the server itself
ipmitool sensor list
ipmitool raw 0x30 0x30 0x02 0xff 0xC0
```

### With Authentication (Remote/Out-of-Band)
```bash
# Runs from remote system
ipmitool -I lanplus -H 192.168.1.100 -U USERID -P PASSW0RD sensor list
ipmitool -I lanplus -H 192.168.1.100 -U USERID -P PASSW0RD raw 0x30 0x30 0x02 0xff 0xC0
```

The service automatically chooses the right mode based on whether credentials are configured.

## Error Messages

### Before (Without Authentication)
```
ERROR - Failed to set fan speed: Unable to send RAW command (channel=0x0 netfn=0x30 lun=0x0 cmd=0x30 rsp=0xc1): Invalid command
```

### After (With Authentication)
```
INFO - Fan speed set to 50% (raw: 127)
```

### Authentication Errors
If credentials are wrong:
```json
{
  "success": false,
  "error": "IPMI authentication failed",
  "details": "Unable to authenticate with BMC/XCC. Check credentials and ensure IPMI over LAN is enabled.",
  "recommendation": "Verify IPMI credentials are correct and IPMI over LAN is enabled in XCC/BMC settings",
  "credentials_configured": true
}
```

### Missing Credentials
If credentials not configured:
```json
{
  "success": false,
  "error": "IPMI command failed - authentication may be required",
  "details": "The IPMI command returned 'Invalid command' (0xc1). For Lenovo and most server-class hardware, IPMI over LAN requires authentication.",
  "recommendation": "Configure IPMI credentials (GATOR_IPMI_HOST, GATOR_IPMI_USERNAME, GATOR_IPMI_PASSWORD) and ensure IPMI over LAN is enabled in XCC/BMC",
  "credentials_configured": false
}
```

## Security Considerations

### Credential Storage
- Credentials stored in memory only (not persisted to database)
- Environment variables recommended for production
- Avoid hardcoding credentials in code

### Network Security
- Use secure network for IPMI traffic (management VLAN)
- Consider using VPN for remote IPMI access
- Change default XCC passwords
- Regularly rotate credentials

### Access Control
- Limit IPMI access to authorized systems only
- Use XCC firewall rules to restrict access
- Monitor IPMI access logs

## Troubleshooting

### Issue: "Invalid command" (0xc1) Error

**Possible Causes**:
1. Missing IPMI credentials (most common)
2. IPMI over LAN not enabled in XCC
3. Wrong credentials
4. Network connectivity issues
5. Truly unsupported command (least common)

**Solution Steps**:
1. ✅ Configure IPMI credentials
2. ✅ Verify IPMI over LAN enabled in XCC
3. ✅ Test network connectivity to XCC IP
4. ✅ Verify credentials are correct
5. ✅ Check XCC logs for auth failures

### Issue: "Authentication failed"

**Check**:
- Credentials are correct
- IPMI over LAN is enabled in XCC
- User account has appropriate privileges
- Account is not locked

### Issue: Connection timeout

**Check**:
- XCC IP address is correct and reachable
- Network allows IPMI port (UDP 623)
- Firewall rules allow IPMI traffic
- XCC network interface is up

### Issue: Still getting 0xc1 after configuring auth

If you still get 0xc1 errors **after** configuring authentication correctly, then the command may truly be unsupported on your hardware. The service will detect this and report:

```json
{
  "success": false,
  "error": "Hardware does not support manual fan control via IPMI raw commands",
  "details": "This lenovo server's BMC does not support the IPMI OEM commands needed for manual fan control..."
}
```

## Testing Checklist

- [ ] IPMI credentials configured (environment vars or API)
- [ ] IPMI over LAN enabled in XCC
- [ ] Network connectivity to XCC IP verified
- [ ] Credentials tested with ipmitool command line
- [ ] Control info shows `credentials_configured: true`
- [ ] Fan status endpoint returns data
- [ ] Fan speed can be set successfully

## Additional Resources

### ipmitool Installation
```bash
# Debian/Ubuntu
sudo apt-get install ipmitool

# RHEL/CentOS/Rocky
sudo yum install ipmitool
```

### Lenovo Documentation
- [XCC Users Guide](https://pubs.lenovo.com/xcc/)
- [IPMI Configuration Guide](https://pubs.lenovo.com/xcc/ipmi)

### Testing IPMI Manually
```bash
# Test connectivity and authentication
ipmitool -I lanplus -H <xcc-ip> -U <username> -P <password> chassis status

# List sensors
ipmitool -I lanplus -H <xcc-ip> -U <username> -P <password> sensor list

# Get fan information
ipmitool -I lanplus -H <xcc-ip> -U <username> -P <password> sensor list | grep -i fan
```

## Summary

The fan control service now properly supports IPMI authentication:

✅ **Automatic mode detection** (local vs remote)  
✅ **Flexible configuration** (environment vars or API)  
✅ **Clear error messages** (auth errors vs unsupported commands)  
✅ **Secure credential handling** (not logged or exposed)  
✅ **Lenovo-specific guidance** (XCC setup instructions)  

The original 0xc1 error is resolved by configuring IPMI authentication credentials.
