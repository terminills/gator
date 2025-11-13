# Fan Control - Manufacturer-Specific IPMI Commands Guide

## Overview

The fan control service now supports manufacturer-specific IPMI commands and graceful error handling for unsupported hardware. This resolves the "Invalid command" (0xc1) errors encountered on servers where manual fan control is not supported.

## Problem Background

Different server manufacturers use different IPMI OEM (Original Equipment Manufacturer) commands for fan control. The original implementation used generic commands that are not universally supported.

**Specific Issue:** Lenovo SR665 servers with XCC (eXtreme Cloud Controller) do not support the generic IPMI raw commands (netfn=0x30 cmd=0x30) for manual fan control. According to Lenovo's official OEM IPMI documentation, the documented commands (netfn=0x2E, 0x3A) are for system control and firmware information, but **fan control commands are not publicly documented**.

## Supported Manufacturers

The service now supports the following manufacturers:

| Manufacturer | IPMI Commands | Notes |
|-------------|---------------|-------|
| **Lenovo** | netfn=0x30, cmd=0x30 | Not officially documented; may not work on all Lenovo servers |
| **Dell** | netfn=0x30, cmd=0x30 | iDRAC fan control commands |
| **HP** | netfn=0x30, cmd=0x30 | iLO fan control commands |
| **Supermicro** | netfn=0x30, cmd=0x45/0x70 | Supermicro-specific commands |
| **Generic** | netfn=0x30, cmd=0x30 | Fallback for unknown manufacturers |

## Configuration

### Default Configuration

By default, the service is configured for **Lenovo** servers:

```python
from backend.services.fan_control_service import get_fan_control_service

service = get_fan_control_service()
# Manufacturer: lenovo (default)
```

### Changing Manufacturer

You can change the manufacturer in two ways:

#### 1. At Service Initialization

```python
from backend.services.fan_control_service import (
    FanControlService,
    ServerManufacturer,
)

service = FanControlService(ServerManufacturer.DELL)
```

#### 2. At Runtime via API

```bash
# Set manufacturer to Dell
curl -X POST http://localhost:8000/api/v1/system/fans/manufacturer \
  -H "Content-Type: application/json" \
  -d '{"manufacturer": "dell"}'

# Response:
{
  "success": true,
  "manufacturer": "dell",
  "previous_manufacturer": "lenovo",
  "manufacturer_note": "Dell iDRAC fan control commands",
  "message": "Manufacturer set to dell. Manual control support will be re-checked on next use."
}
```

### Get Current Configuration

```bash
curl http://localhost:8000/api/v1/system/fans/info

# Response includes:
{
  "ipmi_available": true,
  "manufacturer": "lenovo",
  "control_mode": "auto",
  "manual_control_supported": false,
  "supported_manufacturers": ["lenovo", "dell", "hp", "supermicro", "generic"],
  "manufacturer_note": "Lenovo XCC fan control via IPMI may not be supported...",
  ...
}
```

## Error Handling

The service now gracefully handles unsupported IPMI commands:

### Detection of Unsupported Commands

When the service encounters an "Invalid command" error (IPMI response code 0xc1), it:

1. **Detects** the error pattern in IPMI output
2. **Caches** the unsupported status to avoid repeated failed attempts
3. **Returns** clear error messages with recommendations
4. **Allows** the BMC to continue automatic fan management

### Example Error Response

```json
{
  "success": false,
  "error": "Hardware does not support manual fan control via IPMI raw commands (Manufacturer: lenovo)",
  "details": "This lenovo server's BMC does not support the IPMI OEM commands needed for manual fan control. The system will use automatic fan control managed by the BMC firmware.",
  "ipmi_error": "Unable to send RAW command (channel=0x0 netfn=0x30 lun=0x0 cmd=0x30 rsp=0xc1): Invalid command",
  "manufacturer_note": "Lenovo XCC fan control via IPMI may not be supported...",
  "recommendation": "Use automatic fan control mode, change manufacturer setting if incorrect, or consult your server's documentation for supported fan control methods"
}
```

## API Endpoints

### Set Manufacturer

**POST** `/api/v1/system/fans/manufacturer`

Configure the server manufacturer for proper IPMI command selection.

```json
{
  "manufacturer": "dell"
}
```

Valid values: `lenovo`, `dell`, `hp`, `supermicro`, `generic`

### Get Fan Control Info

**GET** `/api/v1/system/fans/info`

Returns current configuration including manufacturer and support status.

### Other Endpoints (Unchanged)

- `GET /api/v1/system/fans` - Get current fan status
- `POST /api/v1/system/fans/mode` - Set fan control mode
- `POST /api/v1/system/fans/speed` - Set fan speed manually
- `POST /api/v1/system/fans/auto-adjust` - Auto-adjust based on temperature
- `POST /api/v1/system/fans/thresholds` - Update temperature thresholds

## Usage Examples

### Python Example

```python
from backend.services.fan_control_service import (
    FanControlService,
    ServerManufacturer,
    FanControlMode,
)

# Create service for Dell server
service = FanControlService(ServerManufacturer.DELL)

# Try to set fan speed
result = await service.set_fan_speed(75)

if result["success"]:
    print(f"Fan speed set to {result['speed_percent']}%")
else:
    # Gracefully handle unsupported hardware
    print(f"Manual control not available: {result['error']}")
    print(f"Recommendation: {result.get('recommendation', 'Use auto mode')}")
```

### cURL Examples

```bash
# 1. Check current manufacturer and support
curl http://localhost:8000/api/v1/system/fans/info

# 2. Change to Supermicro
curl -X POST http://localhost:8000/api/v1/system/fans/manufacturer \
  -H "Content-Type: application/json" \
  -d '{"manufacturer": "supermicro"}'

# 3. Try to set fan speed
curl -X POST http://localhost:8000/api/v1/system/fans/speed \
  -H "Content-Type: application/json" \
  -d '{"speed_percent": 75}'

# 4. Set to automatic mode (always safe)
curl -X POST http://localhost:8000/api/v1/system/fans/mode \
  -H "Content-Type: application/json" \
  -d '{"mode": "auto"}'
```

## Troubleshooting

### Issue: "Invalid command" Error

**Symptoms:**
```
ERROR - Failed to set fan speed: Unable to send RAW command ... rsp=0xc1: Invalid command
```

**Solutions:**
1. **Check manufacturer setting**: Ensure the correct manufacturer is configured
2. **Verify IPMI availability**: Check that `ipmitool` is installed
3. **Use automatic mode**: Let the BMC manage fans automatically
4. **Consult documentation**: Check your server's manual for supported fan control methods

### Issue: Fan Control Not Working on Lenovo

**Expected Behavior:** Lenovo XCC may not support manual fan control via IPMI raw commands. This is not an error - it's a hardware limitation.

**What Happens:**
- Service detects unsupported commands
- Returns clear error message
- BMC continues automatic fan management
- System remains safe and operational

**Action:** Use automatic fan control mode. The BMC will manage fan speeds based on temperature sensors.

## Technical Details

### IPMI Command Structure

Different manufacturers use different network function (netfn) and command codes:

```bash
# Generic/Dell/HP/Lenovo (attempted)
ipmitool raw 0x30 0x30 0x01 0x01  # Auto mode
ipmitool raw 0x30 0x30 0x01 0x00  # Manual mode
ipmitool raw 0x30 0x30 0x02 0xff 0xC0  # Set speed (75% = 0xC0)

# Supermicro
ipmitool raw 0x30 0x45 0x01 0x01  # Auto mode
ipmitool raw 0x30 0x45 0x01 0x00  # Manual mode
ipmitool raw 0x30 0x70 0x66 0x01 0x00 0xC0  # Set speed
```

### Cached Support Status

The service caches whether manual control is supported to avoid repeated IPMI calls:

```python
service._manual_control_supported = None   # Unknown
service._manual_control_supported = True   # Confirmed working
service._manual_control_supported = False  # Confirmed unsupported
```

When `False`, subsequent calls to `set_fan_speed()` return immediately without attempting IPMI commands.

## Lenovo-Specific Information

### Official Lenovo OEM IPMI Commands

According to Lenovo's documentation, the XCC supports these IPMI OEM commands:

- **Network Function 0x2E**: XCC reset and configuration
- **Network Function 0x3A**: Board/firmware info, system control, USB control, etc.

**Important:** Fan control commands are **NOT** documented in the official Lenovo OEM IPMI command list.

### Implications

1. Manual fan control via IPMI may not be available on Lenovo servers
2. The BMC (XCC) manages fans automatically based on internal thermal policies
3. This is expected behavior, not a bug or error
4. System remains safe - the BMC ensures proper cooling

## Best Practices

1. **Always check support status** before attempting manual fan control
2. **Use automatic mode** as the default for production systems
3. **Configure correct manufacturer** for best results
4. **Monitor BMC logs** for thermal events
5. **Trust the BMC** - modern BMCs have sophisticated thermal management

## Demo Script

Run the included demo to see manufacturer configuration in action:

```bash
python3 demo_fan_control_manufacturer.py
```

This demonstrates:
- Different IPMI commands for each manufacturer
- Runtime manufacturer switching
- Error handling for unsupported hardware
- API usage examples

## Summary

The enhanced fan control service provides:

✅ Manufacturer-specific IPMI command support
✅ Graceful handling of unsupported hardware  
✅ Clear error messages with actionable recommendations
✅ Runtime manufacturer configuration
✅ Cached support status to avoid repeated failures
✅ Comprehensive logging for troubleshooting
✅ Full backward compatibility

For Lenovo SR665 servers specifically, the service now correctly identifies that manual fan control may not be supported and provides appropriate guidance while allowing the BMC to manage cooling automatically.
