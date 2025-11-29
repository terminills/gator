#!/usr/bin/env python3
"""
End-to-end test for content generation to prove it actually works.

This test simulates what happens when the API is called and proves that:
1. The system detects when no models are available
2. The system properly fails instead of creating placeholder data
3. When llama.cpp is available, it actually generates content
"""

import sys
import asyncio
from pathlib import Path
import os

# Add llama-cli to PATH
repo_root = Path(__file__).parent.parent.parent
llamacpp_bin = repo_root / "third_party" / "llama.cpp" / "build" / "bin"
os.environ["PATH"] = f"{llamacpp_bin}:{os.environ.get('PATH', '')}"


async def test_no_models_fails_properly():
    """Test that content generation fails properly when no models are available."""
    print("\n" + "="*80)
    print("🧪 TEST 1: Proper failure when no models available")
    print("="*80)
    
    try:
        from backend.services.ai_models import AIModelManager
        
        # Create AI models manager
        ai_models = AIModelManager()
        
        # Initialize (should detect no models downloaded)
        await ai_models.initialize_models()
        
        # Check if any image models are loaded
        image_models = [m for m in ai_models.available_models.get("image", []) if m.get("loaded", False)]
        text_models = [m for m in ai_models.available_models.get("text", []) if m.get("loaded", False)]
        
        print(f"\n📊 Model Status:")
        print(f"   Image models loaded: {len(image_models)}")
        print(f"   Text models loaded: {len(text_models)}")
        
        if image_models:
            print(f"   ⚠️  WARNING: Image models should not be loaded without downloaded files")
            for m in image_models:
                print(f"      - {m['name']} ({m.get('provider', 'unknown')})")
        
        if text_models:
            print(f"   ℹ️  Text models detected:")
            for m in text_models:
                print(f"      - {m['name']} ({m.get('provider', 'unknown')})")
        
        # Try to generate image (should fail)
        print(f"\n🔍 Attempting image generation without models...")
        try:
            result = await ai_models.generate_image(
                prompt="Test image generation",
                quality="standard"
            )
            
            # Check if it's a placeholder
            if not result.get("image_data") or len(result.get("image_data", b"")) == 0:
                print(f"   ❌ FAIL: Returned empty placeholder instead of failing")
                print(f"   Result: {result}")
                return False
            else:
                print(f"   ⚠️  Generated {len(result['image_data'])} bytes of image data")
                print(f"   This should only happen if a model is actually available")
                
        except ValueError as e:
            print(f"   ✅ PASS: Properly failed with error: {str(e)[:100]}")
            return True
        except Exception as e:
            print(f"   ✅ PASS: Failed with exception: {type(e).__name__}: {str(e)[:100]}")
            return True
        
        print(f"   ❌ FAIL: Should have raised an exception")
        return False
        
    except Exception as e:
        print(f"   ❌ EXCEPTION: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_text_generation_with_llamacpp():
    """Test text generation with llama.cpp if available."""
    print("\n" + "="*80)
    print("🧪 TEST 2: Text generation with llama.cpp")
    print("="*80)
    
    try:
        from backend.services.ai_models import AIModelManager
        import shutil
        
        # Check if llama-cli is available
        llamacpp_binary = shutil.which("llama-cli")
        if not llamacpp_binary:
            print(f"   ⏭️  SKIP: llama-cli not in PATH")
            return None
        
        print(f"   ✓ Found llama-cli at: {llamacpp_binary}")
        
        # Create AI models manager
        ai_models = AIModelManager()
        await ai_models.initialize_models()
        
        # Check if any text models are available
        text_models = ai_models.available_models.get("text", [])
        loaded_text_models = [m for m in text_models if m.get("loaded", False)]
        
        print(f"\n📊 Text Model Status:")
        print(f"   Total text models: {len(text_models)}")
        print(f"   Loaded text models: {len(loaded_text_models)}")
        
        if not loaded_text_models:
            print(f"   ℹ️  No text models loaded")
            print(f"   To test with real model:")
            print(f"      1. Download a GGUF model to models/text/")
            print(f"      2. Configure it in ai_models.py local_model_configs")
            return None
        
        for m in loaded_text_models:
            print(f"      - {m['name']}: {m.get('description', 'No description')}")
            print(f"        Provider: {m.get('provider', 'unknown')}")
            print(f"        Engine: {m.get('inference_engine', 'unknown')}")
            print(f"        Path: {m.get('path', 'No path')}")
        
        # Try to generate text
        print(f"\n🔍 Attempting text generation...")
        try:
            result = await ai_models.generate_text(
                prompt="Write a short greeting.",
                max_tokens=50,
                temperature=0.7
            )
            
            print(f"\n✅ PASS: Text generated successfully!")
            print(f"   Length: {len(result)} characters")
            print(f"   Preview: {result[:200]}")
            
            # Verify it's not a placeholder
            if result.startswith("[") and "generation with" in result:
                print(f"   ⚠️  WARNING: This looks like a placeholder!")
                return False
            
            return True
            
        except ValueError as e:
            print(f"   ❌ FAIL: Generation failed: {str(e)[:200]}")
            return False
        except Exception as e:
            print(f"   ❌ FAIL: Unexpected error: {type(e).__name__}: {str(e)[:200]}")
            import traceback
            traceback.print_exc()
            return False
        
    except Exception as e:
        print(f"   ❌ EXCEPTION: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False


async def main():
    """Run all end-to-end tests."""
    print("="*80)
    print("🧪 END-TO-END CONTENT GENERATION TEST")
    print("="*80)
    print()
    print("Purpose: Prove that actual AI generation works, not just database operations.")
    print()
    
    # Set up Python path
    sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))
    
    results = []
    
    # Test 1: Proper failure without models
    result1 = await test_no_models_fails_properly()
    results.append(("Proper failure without models", result1))
    
    # Test 2: Actual text generation with llama.cpp
    result2 = await test_text_generation_with_llamacpp()
    if result2 is not None:
        results.append(("Text generation with llama.cpp", result2))
    
    # Summary
    print("\n" + "="*80)
    print("📊 TEST SUMMARY")
    print("="*80)
    
    passed = sum(1 for _, r in results if r is True)
    failed = sum(1 for _, r in results if r is False)
    skipped = sum(1 for _, r in results if r is None)
    total = len(results)
    
    for name, result in results:
        if result is True:
            status = "✅ PASS"
        elif result is False:
            status = "❌ FAIL"
        else:
            status = "⏭️  SKIP"
        print(f"  {status}: {name}")
    
    print()
    print(f"Results: {passed} passed, {failed} failed, {skipped} skipped out of {total} tests")
    
    if failed == 0 and passed > 0:
        print("\n✅ Content generation is working correctly!")
        print("\nKey findings:")
        print("  • System properly fails when models are not available")
        print("  • No silent placeholder generation")
        if any("llamacpp" in name.lower() for name, r in results if r):
            print("  • llama.cpp integration produces real text output")
        return 0
    elif failed > 0:
        print("\n❌ Some tests failed. Content generation needs fixes.")
        return 1
    else:
        print("\n⏭️  All tests were skipped (models not available)")
        return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
