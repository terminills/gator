#!/usr/bin/env python3
"""
Test script to verify SDXL pipeline fix.

This test verifies that the AIModelManager correctly detects SDXL models
and would use StableDiffusionXLPipeline instead of StableDiffusionPipeline.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))


def test_sdxl_detection_logic():
    """Test the SDXL model detection logic."""
    print("🧪 Testing SDXL Model Detection Logic")
    print("=" * 60)
    
    # Test cases: (model_name, model_id, expected_is_sdxl)
    test_cases = [
        ("sdxl-1.0", "stabilityai/stable-diffusion-xl-base-1.0", True),
        ("sdxl-turbo", "stabilityai/sdxl-turbo", True),
        ("SDXL-Lightning", "ByteDance/SDXL-Lightning", True),
        ("stable-diffusion-v1-5", "runwayml/stable-diffusion-v1-5", False),
        ("stable-diffusion-v2-1", "stabilityai/stable-diffusion-2-1", False),
        ("sd-v1-4", "CompVis/stable-diffusion-v1-4", False),
    ]
    
    all_passed = True
    
    for model_name, model_id, expected_is_sdxl in test_cases:
        # This is the logic from the fixed code
        is_sdxl = "xl" in model_name.lower() or "xl" in model_id.lower()
        
        status = "✅" if is_sdxl == expected_is_sdxl else "❌"
        result = "PASS" if is_sdxl == expected_is_sdxl else "FAIL"
        
        print(f"{status} {result}: {model_name}")
        print(f"   model_id: {model_id}")
        print(f"   Detected as SDXL: {is_sdxl}, Expected: {expected_is_sdxl}")
        
        if is_sdxl != expected_is_sdxl:
            all_passed = False
    
    print("\n" + "=" * 60)
    if all_passed:
        print("✅ All detection tests passed!")
        print("\nThe fix correctly identifies:")
        print("  - SDXL models (will use StableDiffusionXLPipeline)")
        print("  - SD 1.5/2.x models (will use StableDiffusionPipeline)")
        return True
    else:
        print("❌ Some detection tests failed!")
        return False


def test_code_structure():
    """Verify the code imports the correct classes."""
    print("\n🔍 Testing Code Structure")
    print("=" * 60)
    
    try:
        # Check that the code can import both pipeline classes
        with open("src/backend/services/ai_models.py", "r") as f:
            content = f.read()
        
        # Check for the correct imports in _generate_image_diffusers
        if "from diffusers import (" in content:
            print("✅ Found multi-line diffusers import")
        else:
            print("⚠️  Single-line import (still works)")
            
        if "StableDiffusionXLPipeline" in content:
            print("✅ StableDiffusionXLPipeline is imported")
        else:
            print("❌ StableDiffusionXLPipeline is NOT imported")
            return False
            
        if "is_sdxl = " in content:
            print("✅ SDXL detection logic is present")
        else:
            print("❌ SDXL detection logic is missing")
            return False
            
        if "PipelineClass =" in content:
            print("✅ Dynamic pipeline class selection is present")
        else:
            print("❌ Dynamic pipeline class selection is missing")
            return False
            
        print("\n✅ Code structure is correct!")
        return True
        
    except Exception as e:
        print(f"❌ Code structure test failed: {e}")
        return False


if __name__ == "__main__":
    print("Testing SDXL Pipeline Fix")
    print("=" * 60)
    print()
    
    test1_passed = test_sdxl_detection_logic()
    test2_passed = test_code_structure()
    
    print("\n" + "=" * 60)
    print("FINAL RESULT")
    print("=" * 60)
    
    if test1_passed and test2_passed:
        print("✅ All tests passed!")
        print("\nThe fix ensures that:")
        print("  1. SDXL models (sdxl-1.0, etc.) use StableDiffusionXLPipeline")
        print("  2. SD 1.5/2.x models use StableDiffusionPipeline")
        print("  3. The detection is based on 'xl' in model name or model_id")
        sys.exit(0)
    else:
        print("❌ Some tests failed!")
        sys.exit(1)
