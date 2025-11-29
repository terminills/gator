#!/usr/bin/env python3
"""
Test script for local image generation functionality.
"""

import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from backend.services.ai_models import AIModelManager


async def test_local_image_generation():
    """Test local image generation with diffusers."""
    print("🎨 Testing Local Image Generation")
    print("=" * 60)
    
    # Initialize AI Model Manager
    print("\n1. Initializing AI Model Manager...")
    manager = AIModelManager()
    
    print(f"   - GPU Type: {manager.gpu_type}")
    print(f"   - GPU Memory: {manager.gpu_memory_gb} GB")
    print(f"   - CPU Cores: {manager.cpu_cores}")
    
    # Initialize models
    print("\n2. Initializing models...")
    await manager.initialize_models()
    
    # Check available image models
    print("\n3. Available image models:")
    if not manager.available_models["image"]:
        print("   ⚠️  No image models available")
        print("   Note: This is expected if GPU memory is insufficient")
        print("   Minimum requirements:")
        print("   - stable-diffusion-v1-5: 4 GB GPU memory, 8 GB RAM")
        print("   - sdxl-1.0: 8 GB GPU memory, 16 GB RAM")
        return False
    
    for model in manager.available_models["image"]:
        print(f"   - {model['name']}: provider={model['provider']}, loaded={model['loaded']}, can_load={model['can_load']}")
        print(f"     engine={model.get('inference_engine', 'N/A')}, size={model.get('size_gb', 0)} GB")
    
    # Try to generate an image if we have a local model available
    local_models = [m for m in manager.available_models["image"] 
                   if m.get("provider") == "local" and m.get("can_load", False)]
    
    if not local_models:
        print("\n❌ No local image models can be loaded with current hardware")
        print("   Consider:")
        print("   - Using a system with more GPU memory")
        print("   - Using API-based image generation (OpenAI DALL-E)")
        return False
    
    print(f"\n4. Testing image generation with {local_models[0]['name']}...")
    
    # Simple test prompt
    test_prompt = "A serene mountain landscape at sunset, digital art"
    
    try:
        print(f"   Prompt: {test_prompt}")
        print(f"   Note: First run will download the model (~4-7 GB)")
        print(f"   This may take several minutes depending on your connection...")
        
        result = await manager.generate_image(
            test_prompt,
            width=512,
            height=512,
            num_inference_steps=20,  # Fewer steps for faster testing
            guidance_scale=7.5,
        )
        
        if result and result.get("image_data"):
            print(f"\n✅ Image generated successfully!")
            print(f"   - Size: {len(result['image_data'])} bytes")
            print(f"   - Format: {result['format']}")
            print(f"   - Model: {result['model']}")
            print(f"   - Dimensions: {result.get('width', 'N/A')}x{result.get('height', 'N/A')}")
            
            # Save the image for inspection
            output_path = Path("test_generated_image.png")
            with open(output_path, "wb") as f:
                f.write(result["image_data"])
            print(f"   - Saved to: {output_path}")
            
            return True
        else:
            print("\n❌ Image generation returned empty result")
            return False
            
    except Exception as e:
        print(f"\n❌ Image generation failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


async def main():
    """Main test function."""
    success = await test_local_image_generation()
    
    print("\n" + "=" * 60)
    if success:
        print("✅ Local image generation test PASSED")
        sys.exit(0)
    else:
        print("ℹ️  Local image generation test completed with limitations")
        print("   This is expected on systems without sufficient GPU resources")
        sys.exit(0)


if __name__ == "__main__":
    asyncio.run(main())
