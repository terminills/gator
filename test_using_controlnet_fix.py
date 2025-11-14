#!/usr/bin/env python3
"""
Test script to verify that the using_controlnet variable fix works correctly.

This script validates that the variable is properly initialized before use,
preventing UnboundLocalError when the pipeline is already cached.
"""

import re
import sys


def check_using_controlnet_initialization():
    """Check that using_controlnet is initialized before use."""
    with open('src/backend/services/ai_models.py', 'r') as f:
        content = f.read()
    
    # Find the _generate_image_diffusers method
    method_pattern = r'async def _generate_image_diffusers\(.*?\n(.*?)(?=\n    async def |\n    def |\Z)'
    match = re.search(method_pattern, content, re.DOTALL)
    
    if not match:
        print("❌ Could not find _generate_image_diffusers method")
        return False
    
    method_content = match.group(1)
    lines = method_content.split('\n')
    
    # Find initialization and usage of using_controlnet
    initialization_line = -1
    first_usage_line = -1
    cache_check_line = -1
    
    for i, line in enumerate(lines):
        if 'using_controlnet = False' in line and initialization_line == -1:
            initialization_line = i
            print(f"✅ Found initialization at line {i + 1} (relative)")
        
        if 'if pipeline_key not in self._loaded_pipelines:' in line and cache_check_line == -1:
            cache_check_line = i
            print(f"✅ Found cache check at line {i + 1} (relative)")
        
        if 'using_controlnet' in line and 'using_controlnet =' not in line and first_usage_line == -1 and i > 10:
            first_usage_line = i
            print(f"✅ Found first usage at line {i + 1} (relative): {line.strip()[:80]}")
    
    # Verify initialization happens before cache check
    if initialization_line < 0:
        print("❌ using_controlnet initialization not found")
        return False
    
    if cache_check_line < 0:
        print("❌ Pipeline cache check not found")
        return False
    
    if first_usage_line < 0:
        print("❌ using_controlnet usage not found")
        return False
    
    if initialization_line < cache_check_line:
        print(f"✅ PASS: Initialization (line {initialization_line}) happens BEFORE cache check (line {cache_check_line})")
    else:
        print(f"❌ FAIL: Initialization (line {initialization_line}) happens AFTER cache check (line {cache_check_line})")
        return False
    
    if initialization_line < first_usage_line:
        print(f"✅ PASS: Initialization (line {initialization_line}) happens BEFORE first usage (line {first_usage_line})")
    else:
        print(f"❌ FAIL: Initialization (line {initialization_line}) happens AFTER first usage (line {first_usage_line})")
        return False
    
    return True


if __name__ == '__main__':
    print("=" * 80)
    print("Testing using_controlnet variable initialization fix")
    print("=" * 80)
    
    result = check_using_controlnet_initialization()
    
    print("=" * 80)
    if result:
        print("✅ All checks passed!")
        sys.exit(0)
    else:
        print("❌ Some checks failed")
        sys.exit(1)
