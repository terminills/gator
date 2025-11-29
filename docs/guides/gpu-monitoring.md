# GPU Temperature Monitoring and Fan Control

This document describes the GPU temperature monitoring and server fan control features for the Lenovo SR665 server.

## Overview

The Gator platform now includes comprehensive GPU monitoring and automated fan control capabilities designed specifically for server environments where fans are managed at the server level (fan wall) rather than per-GPU.

### Key Features

- **Real-time GPU Temperature Monitoring**: Track temperature, memory usage, and health status for all GPUs
- **Historical Temperature Tracking**: Maintain temperature history for trend analysis
- **Server Fan Control**: Control Lenovo SR665 fan wall via IPMI
- **Automatic Fan Adjustment**: Automatically adjust fan speeds based on GPU temperature
- **Multiple Fan Zones**: Support for system, CPU, and peripheral fan zones
- **Health Status Integration**: GPU monitoring integrated into system health endpoints

## GPU Monitoring

### Available Endpoints

#### GET /api/v1/system/gpu/temperature

Get current GPU temperatures for all devices.

**Response Example:**
```json
{
  "available": true,
  "gpu_count": 2,
  "timestamp": "2025-11-13T02:00:00.000Z",
  "gpus": [
    {
      "device_id": 0,
      "name": "AMD Radeon Pro V620",
      "temperature_c": 65.0,
      "memory_total_gb": 32.0,
      "memory_used_gb": 16.5,
      "memory_reserved_gb": 17.0
    },
    {
      "device_id": 1,
      "name": "AMD Radeon Pro V620",
      "temperature_c": 62.0,
      "memory_total_gb": 32.0,
      "memory_used_gb": 14.2,
      "memory_reserved_gb": 15.0
    }
  ]
}
```

#### GET /api/v1/system/gpu/status

Get comprehensive GPU status including health information.

**Response Example:**
```json
{
  "available": true,
  "gpu_count": 2,
  "timestamp": "2025-11-13T02:00:00.000Z",
  "gpus": [
    {
      "device_id": 0,
      "name": "AMD Radeon Pro V620",
      "temperature_c": 65.0,
      "health_status": "warm",
      "memory": {
        "total_gb": 32.0,
        "used_gb": 16.5,
        "free_gb": 15.5,
        "reserved_gb": 17.0,
        "utilization_percent": 51.56
      },
      "compute_capability": "9.0",
      "multi_processor_count": 128
    }
  ]
}
```

**Health Status Values:**
- `healthy`: Temperature < 60°C
- `warm`: Temperature 60-75°C
- `hot`: Temperature 75-85°C
- `critical`: Temperature ≥ 85°C
- `unknown`: Temperature data unavailable

#### GET /api/v1/system/gpu/temperature/history

Get historical temperature data for analysis.

**Query Parameters:**
- `device_id` (optional): Specific GPU device ID. If omitted, returns history for all GPUs.

**Response Example:**
```json
{
  "available": true,
  "device_id": 0,
  "history": [
    {
      "timestamp": "2025-11-13T01:55:00.000Z",
      "temperature": 62.0
    },
    {
      "timestamp": "2025-11-13T01:56:00.000Z",
      "temperature": 65.0
    }
  ]
}
```

#### GET /api/v1/system/gpu/temperature/max

Get maximum recorded temperatures across history.

**Response Example:**
```json
{
  "available": true,
  "max_temperatures": {
    "0": {
      "max_temperature_c": 75.0,
      "avg_temperature_c": 65.5,
      "readings_count": 100
    },
    "1": {
      "max_temperature_c": 72.0,
      "avg_temperature_c": 63.2,
      "readings_count": 100
    }
  }
}
```

## Fan Control

### Prerequisites

Fan control requires IPMI tools to be installed on the server:

```bash
# Ubuntu/Debian
sudo apt-get install ipmitool

# RHEL/CentOS/Rocky Linux
sudo yum install ipmitool
```

### Available Endpoints

#### GET /api/v1/system/fans

Get current fan status from IPMI.

**Response Example:**
```json
{
  "available": true,
  "control_mode": "auto",
  "timestamp": "2025-11-13T02:00:00.000Z",
  "fans": [
    {
      "name": "Fan 1",
      "speed": 3600.0,
      "unit": "RPM",
      "status": "ok"
    },
    {
      "name": "Fan 2",
      "speed": 3500.0,
      "unit": "RPM",
      "status": "ok"
    }
  ],
  "thresholds": {
    "low": 50,
    "normal": 65,
    "high": 75,
    "critical": 85
  }
}
```

#### GET /api/v1/system/fans/info

Get fan control configuration and capabilities.

**Response Example:**
```json
{
  "ipmi_available": true,
  "control_mode": "auto",
  "manual_speed_percent": null,
  "temperature_thresholds": {
    "low": 50,
    "normal": 65,
    "high": 75,
    "critical": 85
  },
  "supported_zones": [
    "system",
    "cpu",
    "peripheral"
  ]
}
```

#### POST /api/v1/system/fans/mode

Set fan control mode (automatic or manual).

**Request Body:**
```json
{
  "mode": "manual"
}
```

**Valid Modes:**
- `auto` / `automatic`: Let the BMC control fan speeds automatically
- `manual`: Enable manual fan speed control

**Response:**
```json
{
  "success": true,
  "mode": "manual",
  "message": "Manual mode enabled. Use set_fan_speed to control fans."
}
```

#### POST /api/v1/system/fans/speed

Set fan speed manually (requires manual mode).

**Request Body:**
```json
{
  "speed_percent": 75,
  "zone": "system"
}
```

**Parameters:**
- `speed_percent` (required): Fan speed as percentage (0-100)
- `zone` (optional): Fan zone - `system`, `cpu`, or `peripheral`

**Response:**
```json
{
  "success": true,
  "speed_percent": 75,
  "raw_value": 191,
  "zone": "system",
  "message": "Fan speed set to 75%"
}
```

#### POST /api/v1/system/fans/auto-adjust

Automatically adjust fan speeds based on GPU temperature.

**Request Body (optional):**
```json
{
  "target_temperature": 80.0
}
```

**Response:**
```json
{
  "success": true,
  "temperature": 75.0,
  "target_speed_percent": 75,
  "threshold_status": "high",
  "action_taken": {
    "success": true,
    "speed_percent": 75,
    "message": "Fan speed set to 75%"
  },
  "timestamp": "2025-11-13T02:00:00.000Z"
}
```

**Threshold Status Values:**
- `low`: < 50°C
- `normal`: 50-65°C
- `elevated`: 65-75°C
- `high`: 75-85°C
- `critical`: ≥ 85°C

#### POST /api/v1/system/fans/thresholds

Update temperature thresholds for automatic fan control.

**Request Body:**
```json
{
  "low": 55,
  "normal": 70,
  "high": 80,
  "critical": 90
}
```

**Response:**
```json
{
  "success": true,
  "thresholds": {
    "low": 55,
    "normal": 70,
    "high": 80,
    "critical": 90
  }
}
```

## Integration with Analytics

GPU monitoring is integrated into the system health endpoint:

```bash
curl http://localhost:8000/api/v1/analytics/health
```

**Response includes GPU status:**
```json
{
  "api": "healthy",
  "database": "healthy",
  "gpu_monitoring": "healthy",
  "gpu_count": 2,
  "gpu_warning": "High GPU temperature: 78°C",
  "timestamp": "2025-11-13T02:00:00.000Z"
}
```

## Automatic Fan Control Strategy

The system uses a temperature-based fan curve:

| Temperature Range | Fan Speed | Status |
|------------------|-----------|---------|
| < 50°C | 30% | Low - Minimum safe speed |
| 50-65°C | 30-50% | Normal - Moderate cooling |
| 65-75°C | 50-75% | Elevated - Increased cooling |
| 75-85°C | 75-100% | High - Aggressive cooling |
| ≥ 85°C | 100% | Critical - Maximum cooling |

The fan speed is calculated using linear interpolation within each range.

## Usage Examples

### Monitor GPU Temperatures

```bash
# Get current temperatures
curl http://localhost:8000/api/v1/system/gpu/temperature

# Get detailed GPU status
curl http://localhost:8000/api/v1/system/gpu/status

# Get temperature history
curl http://localhost:8000/api/v1/system/gpu/temperature/history?device_id=0
```

### Manual Fan Control

```bash
# Set to manual mode
curl -X POST http://localhost:8000/api/v1/system/fans/mode \
  -H "Content-Type: application/json" \
  -d '{"mode": "manual"}'

# Set fan speed to 80%
curl -X POST http://localhost:8000/api/v1/system/fans/speed \
  -H "Content-Type: application/json" \
  -d '{"speed_percent": 80}'

# Return to automatic mode
curl -X POST http://localhost:8000/api/v1/system/fans/mode \
  -H "Content-Type: application/json" \
  -d '{"mode": "auto"}'
```

### Automatic Temperature-Based Control

```bash
# Let the system automatically adjust fans based on GPU temperature
curl -X POST http://localhost:8000/api/v1/system/fans/auto-adjust
```

### Customize Temperature Thresholds

```bash
# Set custom temperature thresholds
curl -X POST http://localhost:8000/api/v1/system/fans/thresholds \
  -H "Content-Type: application/json" \
  -d '{
    "low": 55,
    "normal": 70,
    "high": 80,
    "critical": 90
  }'
```

## Python Integration

```python
import httpx
import asyncio

async def monitor_and_control():
    """Monitor GPU temperature and control fans."""
    async with httpx.AsyncClient() as client:
        # Get GPU temperatures
        response = await client.get("http://localhost:8000/api/v1/system/gpu/temperature")
        gpu_data = response.json()
        
        if gpu_data["available"]:
            max_temp = max(gpu["temperature_c"] for gpu in gpu_data["gpus"] if gpu.get("temperature_c"))
            print(f"Max GPU temperature: {max_temp}°C")
            
            # Auto-adjust fans if temperature is high
            if max_temp > 75:
                response = await client.post(
                    "http://localhost:8000/api/v1/system/fans/auto-adjust"
                )
                result = response.json()
                print(f"Fan speed adjusted to: {result['target_speed_percent']}%")

# Run the monitoring
asyncio.run(monitor_and_control())
```

## Lenovo SR665 Specific Notes

### IPMI Raw Commands

The fan control service uses IPMI raw commands specifically tested for Lenovo SR665:

- **Enable automatic fan control**: `ipmitool raw 0x30 0x30 0x01 0x01`
- **Enable manual fan control**: `ipmitool raw 0x30 0x30 0x01 0x00`
- **Set fan speed**: `ipmitool raw 0x30 0x30 0x02 0xff <speed_hex>`

These commands may need adjustment for different server models.

### Fan Zones

The SR665 supports multiple fan zones:
- **System (0x00)**: Main system fan wall
- **CPU (0x01)**: CPU-specific fans
- **Peripheral (0x02)**: Peripheral zone fans
- **All Zones (0xff)**: Control all fans simultaneously (default)

### Temperature Monitoring

GPU temperature is read using multiple fallback methods:
1. `rocm-smi` command (primary for AMD GPUs)
2. sysfs hwmon interface (`/sys/class/drm/card*/device/hwmon/hwmon*/temp*_input`)
3. PyTorch CUDA API (for temperature awareness)

## Troubleshooting

### IPMI Not Available

If you see "IPMI not available", ensure:
1. `ipmitool` is installed
2. IPMI kernel modules are loaded: `modprobe ipmi_devintf ipmi_si`
3. IPMI device exists: `ls -la /dev/ipmi*`
4. You have proper permissions to access IPMI

### GPU Monitoring Not Available

If GPU monitoring shows "not available":
1. Verify PyTorch is installed: `pip install torch`
2. Check GPU detection: `python -c "import torch; print(torch.cuda.is_available())"`
3. Verify ROCm is properly installed for AMD GPUs
4. Check `rocm-smi` command availability

### Temperature Reading Fails

If temperature readings return `null`:
1. Verify `rocm-smi` works: `rocm-smi --showtemp`
2. Check sysfs access: `cat /sys/class/drm/card0/device/hwmon/hwmon*/temp*_input`
3. Ensure proper ROCm drivers are installed

### Fan Control Commands Fail

If fan control commands fail:
1. Check IPMI sensor access: `ipmitool sensor list | grep -i fan`
2. Verify raw commands work: `ipmitool raw 0x30 0x30 0x01 0x01`
3. Some server BMC versions may use different IPMI commands - consult Lenovo documentation
4. Ensure BMC firmware is up to date

## Security Considerations

- IPMI access typically requires root privileges or membership in the `ipmi` group
- Consider restricting API access to these endpoints using authentication
- Monitor fan control changes to prevent accidental system overheating
- Set up alerts for critical GPU temperatures (≥ 85°C)

## Performance Impact

- GPU temperature monitoring: Negligible (< 1ms per reading)
- Temperature history storage: ~10KB per GPU for 100 readings
- IPMI commands: 50-200ms per command (blocking)
- Automatic fan adjustment: < 5 seconds total execution time

## Best Practices

1. **Use automatic mode for production**: Let the system manage fan speeds based on temperature
2. **Set appropriate thresholds**: Adjust based on your workload and cooling capacity
3. **Monitor temperature history**: Use historical data to optimize cooling strategy
4. **Alert on critical temperatures**: Set up monitoring for temperatures ≥ 85°C
5. **Regular maintenance**: Clean fan filters and check fan operation regularly
6. **Test manual control**: Verify fan control works before relying on automatic mode

## Future Enhancements

Planned improvements include:
- Predictive temperature modeling
- Per-GPU fan zone assignment
- Integration with alert/notification systems
- Dashboard for real-time monitoring
- Historical trend analysis and reporting
- Multi-server fan control coordination
