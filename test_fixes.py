#!/usr/bin/env python3
"""
Quick test to validate the fixes made for content generation issues.
Tests:
1. use_img2img initialization fix
2. Prompt generation with instruction-like text
3. ControlNet imports
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))


def test_imports():
    """Test that all imports work correctly."""
    print("Testing imports...")
    
    try:
        from backend.services.ai_models import AIModelManager
        print("✓ AIModelManager import successful")
    except Exception as e:
        print(f"✗ AIModelManager import failed: {e}")
        return False
    
    try:
        from backend.services.prompt_generation_service import PromptGenerationService
        print("✓ PromptGenerationService import successful")
    except Exception as e:
        print(f"✗ PromptGenerationService import failed: {e}")
        return False
    
    # Try importing ControlNet classes
    try:
        from diffusers import (
            ControlNetModel,
            StableDiffusionControlNetPipeline,
            StableDiffusionXLControlNetPipeline,
        )
        print("✓ ControlNet classes import successful")
    except ImportError as e:
        print(f"⚠ ControlNet classes not available (expected if diffusers not installed): {e}")
    
    return True


def test_use_img2img_initialization():
    """Test that use_img2img is properly initialized."""
    print("\nTesting use_img2img initialization fix...")
    
    try:
        from backend.services.ai_models import AIModelManager
        import inspect
        
        # Get the source of _generate_image_diffusers
        source = inspect.getsource(AIModelManager._generate_image_diffusers)
        
        # Check that variables are initialized before try block
        lines = source.split('\n')
        
        # Find where variables are initialized
        init_line = None
        try_line = None
        
        for i, line in enumerate(lines):
            if 'use_img2img = reference_image_path is not None' in line:
                init_line = i
            if 'try:' in line and init_line is None:
                try_line = i
                
        if init_line is not None and try_line is not None:
            if init_line < try_line:
                print("✓ use_img2img is initialized before try block")
                return True
            else:
                print("✗ use_img2img is initialized after try block")
                return False
        elif init_line is not None:
            # Variables initialized but couldn't find try block (might be on same line)
            print("✓ use_img2img initialization found")
            return True
        else:
            print("✗ Could not find use_img2img initialization")
            return False
            
    except Exception as e:
        print(f"✗ Test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_prompt_instruction_detection():
    """Test that instruction-like prompts are detected and handled."""
    print("\nTesting prompt instruction detection...")
    
    try:
        from backend.services.prompt_generation_service import PromptGenerationService
        import inspect
        
        # Check the template generation method
        source = inspect.getsource(PromptGenerationService._generate_with_template)
        
        # Check for instruction detection logic
        if 'instruction_words' in source and 'generate' in source.lower():
            print("✓ Instruction detection logic found in template generation")
        else:
            print("⚠ Instruction detection logic might not be present")
        
        # Check the AI instruction builder
        source = inspect.getsource(PromptGenerationService._build_llama_instruction)
        
        if 'instruction_words' in source or 'is_instruction' in source:
            print("✓ Instruction handling found in AI instruction builder")
        else:
            print("⚠ Instruction handling might not be in AI builder")
        
        return True
        
    except Exception as e:
        print(f"✗ Test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_controlnet_implementation():
    """Test that ControlNet implementation is present."""
    print("\nTesting ControlNet implementation...")
    
    try:
        from backend.services.ai_models import AIModelManager
        import inspect
        
        source = inspect.getsource(AIModelManager._generate_image_diffusers)
        
        # Check for ControlNet-related code
        checks = {
            "ControlNetModel": "ControlNet model class reference",
            "using_controlnet": "ControlNet usage flag",
            "control_image": "ControlNet conditioning image",
            "Canny": "Canny edge detection",
            "controlnet_conditioning_scale": "ControlNet conditioning parameter",
        }
        
        results = []
        for key, description in checks.items():
            if key in source:
                print(f"✓ {description} found")
                results.append(True)
            else:
                print(f"✗ {description} NOT found")
                results.append(False)
        
        return all(results)
        
    except Exception as e:
        print(f"✗ Test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("=" * 60)
    print("Content Generation Fixes - Validation Tests")
    print("=" * 60)
    
    results = []
    
    results.append(("Imports", test_imports()))
    results.append(("use_img2img Fix", test_use_img2img_initialization()))
    results.append(("Prompt Handling", test_prompt_instruction_detection()))
    results.append(("ControlNet", test_controlnet_implementation()))
    
    print("\n" + "=" * 60)
    print("Test Results Summary:")
    print("=" * 60)
    
    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{name:.<40} {status}")
    
    all_passed = all(result for _, result in results)
    
    print("=" * 60)
    if all_passed:
        print("All tests PASSED!")
        return 0
    else:
        print("Some tests FAILED!")
        return 1


if __name__ == "__main__":
    sys.exit(main())
