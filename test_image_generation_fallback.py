#!/usr/bin/env python3
"""
Test script to verify image generation fallback mechanism.
This test verifies that when a model fails (e.g., due to incomplete components),
the system tries alternative models instead of failing immediately.
"""

import asyncio
import sys
from pathlib import Path
from unittest.mock import Mock, AsyncMock, patch

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from backend.services.ai_models import AIModelManager


async def test_fallback_mechanism():
    """Test that image generation falls back to alternative models on failure."""
    print("🧪 Testing Image Generation Fallback Mechanism")
    print("=" * 60)
    
    # Create a mock AIModelManager
    manager = AIModelManager()
    manager.available_models = {
        "text": [],
        "image": [
            {
                "name": "sdxl-1.0",
                "type": "text-to-image",
                "model_id": "stabilityai/stable-diffusion-xl-base-1.0",
                "provider": "local",
                "inference_engine": "diffusers",
                "loaded": True,
                "can_load": True,
                "size_gb": 7,
            },
            {
                "name": "stable-diffusion-v1-5",
                "type": "text-to-image",
                "model_id": "runwayml/stable-diffusion-v1-5",
                "provider": "local",
                "inference_engine": "diffusers",
                "loaded": True,
                "can_load": True,
                "size_gb": 4,
            },
        ],
        "voice": [],
        "video": [],
        "audio": [],
    }
    manager.models_loaded = True
    
    print("\n1. Available models:")
    for model in manager.available_models["image"]:
        print(f"   - {model['name']} ({model['size_gb']}GB)")
    
    # Mock _select_optimal_model to return models in order
    call_count = {"count": 0}
    original_select = manager._select_optimal_model
    
    async def mock_select_optimal(*args, **kwargs):
        # First call returns SDXL, second call returns SD 1.5
        available = kwargs.get("available_models", [])
        if not available:
            raise ValueError("No models available")
        result = available[0]  # Return first available
        print(f"   Selected: {result['name']}")
        return result
    
    # Mock _generate_image_local to simulate SDXL failure and SD 1.5 success
    async def mock_generate_local(prompt, model, **kwargs):
        if model["name"] == "sdxl-1.0":
            print(f"   Simulating SDXL failure (incomplete model)")
            raise ValueError(
                "SDXL pipeline has None text encoders: text_encoder=True, text_encoder_2=False"
            )
        elif model["name"] == "stable-diffusion-v1-5":
            print(f"   SD 1.5 generation succeeded!")
            return {
                "image_data": b"fake_image_data",
                "format": "PNG",
                "model": model["name"],
                "width": 512,
                "height": 512,
            }
        else:
            raise ValueError(f"Unknown model: {model['name']}")
    
    # Patch the methods
    with patch.object(manager, "_select_optimal_model", new=mock_select_optimal):
        with patch.object(manager, "_generate_image_local", new=mock_generate_local):
            try:
                print("\n2. Testing image generation with fallback:")
                print("   Attempting generation (SDXL should fail, SD 1.5 should succeed)...")
                result = await manager.generate_image(
                    "A test image",
                    width=512,
                    height=512,
                )
                
                print(f"\n✅ SUCCESS!")
                print(f"   Model used: {result.get('benchmark_data', {}).get('model_selected', 'unknown')}")
                print(f"   Failed models: {result.get('benchmark_data', {}).get('failed_models', [])}")
                print(f"   Image size: {result.get('width')}x{result.get('height')}")
                
                # Verify the fallback worked
                benchmark_data = result.get("benchmark_data", {})
                failed_models = benchmark_data.get("failed_models", [])
                
                if "sdxl-1.0" in failed_models:
                    print("\n✓ Fallback mechanism working correctly!")
                    print("  - SDXL failed as expected")
                    print("  - SD 1.5 succeeded as fallback")
                    return True
                else:
                    print("\n❌ Fallback mechanism not working as expected")
                    print("  - SDXL should have failed but didn't appear in failed_models")
                    return False
                    
            except Exception as e:
                print(f"\n❌ Test failed with error: {str(e)}")
                import traceback
                traceback.print_exc()
                return False


async def main():
    """Main test function."""
    success = await test_fallback_mechanism()
    
    print("\n" + "=" * 60)
    if success:
        print("✅ Fallback mechanism test PASSED")
        sys.exit(0)
    else:
        print("❌ Fallback mechanism test FAILED")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
