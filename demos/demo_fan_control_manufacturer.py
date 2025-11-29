#!/usr/bin/env python3
"""
Demo: Fan Control with Manufacturer-Specific IPMI Commands

This demo shows how to configure and use the fan control service with
different server manufacturers (Lenovo, Dell, HP, Supermicro, Generic).

The service now gracefully handles unsupported IPMI commands and provides
clear error messages when manual fan control is not available.
"""

import asyncio
from backend.services.fan_control_service import (
    FanControlService,
    ServerManufacturer,
    FanControlMode,
)


async def demo_manufacturer_config():
    """Demonstrate manufacturer-specific configuration."""
    
    print("=" * 70)
    print("Fan Control Service - Manufacturer Configuration Demo")
    print("=" * 70)
    print()
    
    # 1. Create service with Lenovo (default)
    print("1. Creating fan control service for Lenovo...")
    service = FanControlService(ServerManufacturer.LENOVO)
    
    # Get control info
    info = service.get_control_info()
    print(f"   Manufacturer: {info['manufacturer']}")
    print(f"   IPMI Available: {info['ipmi_available']}")
    print(f"   Control Mode: {info['control_mode']}")
    print(f"   Manual Control Supported: {info['manual_control_supported']}")
    print()
    
    # Show Lenovo commands
    commands = service._get_ipmi_commands()
    print("   Lenovo IPMI Commands:")
    print(f"   - Auto Mode: {' '.join(commands['auto_mode'])}")
    print(f"   - Manual Mode: {' '.join(commands['manual_mode'])}")
    print(f"   - Set Speed: {' '.join(commands['set_speed_prefix'])} <speed>")
    print(f"   Note: {commands['note']}")
    print()
    
    # 2. Show different manufacturer commands
    print("2. Comparing IPMI commands across manufacturers:")
    print()
    
    manufacturers = [
        ServerManufacturer.LENOVO,
        ServerManufacturer.DELL,
        ServerManufacturer.HP,
        ServerManufacturer.SUPERMICRO,
        ServerManufacturer.GENERIC,
    ]
    
    for mfr in manufacturers:
        temp_service = FanControlService(mfr)
        cmds = temp_service._get_ipmi_commands()
        print(f"   {mfr.value.upper()}:")
        print(f"   - Auto: ipmitool raw {' '.join(cmds['auto_mode'])}")
        print(f"   - Manual: ipmitool raw {' '.join(cmds['manual_mode'])}")
        print(f"   - Speed: ipmitool raw {' '.join(cmds['set_speed_prefix'])} <speed>")
        print()
    
    # 3. Demonstrate manufacturer switching
    print("3. Switching manufacturers at runtime:")
    print()
    print(f"   Current manufacturer: {service._manufacturer.value}")
    
    result = service.set_manufacturer(ServerManufacturer.DELL)
    print(f"   Switched to: {result['manufacturer']}")
    print(f"   Previous: {result['previous_manufacturer']}")
    print(f"   Message: {result['message']}")
    print()
    
    # 4. Show error handling for unsupported commands
    print("4. Error Handling for Unsupported IPMI Commands:")
    print()
    print("   When IPMI commands are not supported (error 0xc1):")
    print("   - Service detects 'Invalid command' error")
    print("   - Caches unsupported status to avoid repeated failures")
    print("   - Provides clear error messages with recommendations")
    print("   - BMC continues to manage fans automatically")
    print()
    
    # 5. Show supported manufacturers
    print("5. Supported Manufacturers:")
    print()
    info = service.get_control_info()
    for mfr in info['supported_manufacturers']:
        print(f"   - {mfr}")
    print()
    
    print("=" * 70)
    print("Demo Complete!")
    print()
    print("To use this in your application:")
    print("1. Set manufacturer: POST /api/v1/system/fans/manufacturer")
    print("2. Get fan info: GET /api/v1/system/fans/info")
    print("3. Set fan speed: POST /api/v1/system/fans/speed")
    print("=" * 70)


async def demo_error_handling():
    """Demonstrate graceful error handling."""
    
    print()
    print("=" * 70)
    print("Error Handling Demo")
    print("=" * 70)
    print()
    
    service = FanControlService(ServerManufacturer.LENOVO)
    
    # Simulate unsupported hardware
    service._ipmi_available = True
    service._manual_control_supported = False
    
    print("1. Attempting to set fan speed on unsupported hardware:")
    result = await service.set_fan_speed(75)
    print(f"   Success: {result['success']}")
    print(f"   Error: {result['error']}")
    print(f"   Details: {result['details']}")
    print(f"   Recommendation: {result['recommendation']}")
    print()
    
    print("2. Fan control info with unsupported status:")
    info = service.get_control_info()
    print(f"   Manual Control Supported: {info['manual_control_supported']}")
    print(f"   Note: {info['note']}")
    print()


if __name__ == "__main__":
    asyncio.run(demo_manufacturer_config())
    asyncio.run(demo_error_handling())
