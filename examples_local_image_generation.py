#!/usr/bin/env python3
"""
Example: Local Image Generation with Gator

This script demonstrates how to use the local image generation feature.
"""

import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from backend.services.ai_models import AIModelManager


async def example_basic_generation():
    """Example 1: Basic image generation."""
    print("\n" + "=" * 60)
    print("Example 1: Basic Image Generation")
    print("=" * 60)
    
    manager = AIModelManager()
    await manager.initialize_models()
    
    # Check if we have models available
    if not manager.available_models["image"]:
        print("⚠️  No image models available")
        print("   Requirements: 4GB GPU VRAM, 8GB RAM minimum")
        return None
    
    local_models = [m for m in manager.available_models["image"] 
                   if m.get("provider") == "local" and m.get("can_load")]
    
    if not local_models:
        print("⚠️  No local image models can be loaded with current hardware")
        return None
    
    print(f"Using model: {local_models[0]['name']}")
    
    # Generate a simple image
    result = await manager.generate_image(
        "A beautiful sunset over the ocean, photorealistic"
    )
    
    output_path = Path("example_basic.png")
    with open(output_path, "wb") as f:
        f.write(result["image_data"])
    
    print(f"✅ Generated: {output_path}")
    print(f"   Size: {len(result['image_data'])} bytes")
    print(f"   Model: {result['model']}")
    
    return result


async def example_custom_parameters():
    """Example 2: Custom generation parameters."""
    print("\n" + "=" * 60)
    print("Example 2: Custom Parameters")
    print("=" * 60)
    
    manager = AIModelManager()
    await manager.initialize_models()
    
    # Check for models
    local_models = [m for m in manager.available_models["image"] 
                   if m.get("provider") == "local" and m.get("can_load")]
    if not local_models:
        print("⚠️  Skipping - no local models available")
        return None
    
    # Generate with custom parameters
    result = await manager.generate_image(
        prompt="A cyberpunk city street at night, neon lights, rain, cinematic",
        width=768,
        height=512,
        num_inference_steps=30,
        guidance_scale=8.0,
        seed=12345,
        negative_prompt="blurry, low quality, distorted, ugly, deformed"
    )
    
    output_path = Path("example_custom.png")
    with open(output_path, "wb") as f:
        f.write(result["image_data"])
    
    print(f"✅ Generated: {output_path}")
    print(f"   Dimensions: {result['width']}x{result['height']}")
    print(f"   Steps: {result['num_inference_steps']}")
    print(f"   Guidance: {result['guidance_scale']}")
    print(f"   Seed: {result['seed']}")
    
    return result


async def example_batch_generation():
    """Example 3: Batch generation with different prompts."""
    print("\n" + "=" * 60)
    print("Example 3: Batch Generation")
    print("=" * 60)
    
    manager = AIModelManager()
    await manager.initialize_models()
    
    # Check for models
    local_models = [m for m in manager.available_models["image"] 
                   if m.get("provider") == "local" and m.get("can_load")]
    if not local_models:
        print("⚠️  Skipping - no local models available")
        return []
    
    prompts = [
        "A serene mountain landscape, morning mist",
        "A modern minimalist living room, natural light",
        "A futuristic spaceship interior, sci-fi",
    ]
    
    results = []
    for i, prompt in enumerate(prompts, 1):
        print(f"\nGenerating {i}/{len(prompts)}: {prompt[:50]}...")
        
        result = await manager.generate_image(
            prompt=prompt,
            width=512,
            height=512,
            num_inference_steps=25,
            seed=i * 1000,  # Different seed for each
        )
        
        output_path = Path(f"example_batch_{i}.png")
        with open(output_path, "wb") as f:
            f.write(result["image_data"])
        
        print(f"   ✅ Saved: {output_path}")
        results.append(result)
    
    print(f"\n✅ Generated {len(results)} images")
    return results


async def example_reproducible_generation():
    """Example 4: Reproducible generation with fixed seed."""
    print("\n" + "=" * 60)
    print("Example 4: Reproducible Generation")
    print("=" * 60)
    
    manager = AIModelManager()
    await manager.initialize_models()
    
    # Check for models
    local_models = [m for m in manager.available_models["image"] 
                   if m.get("provider") == "local" and m.get("can_load")]
    if not local_models:
        print("⚠️  Skipping - no local models available")
        return None
    
    # Generate the same image twice with the same seed
    prompt = "A cozy coffee shop interior, warm lighting"
    seed = 42
    
    print(f"Generating with seed {seed}...")
    result1 = await manager.generate_image(
        prompt=prompt,
        seed=seed,
        width=512,
        height=512,
    )
    
    print(f"Generating again with same seed...")
    result2 = await manager.generate_image(
        prompt=prompt,
        seed=seed,
        width=512,
        height=512,
    )
    
    # Save both
    with open("example_reproducible_1.png", "wb") as f:
        f.write(result1["image_data"])
    with open("example_reproducible_2.png", "wb") as f:
        f.write(result2["image_data"])
    
    # Compare
    if result1["image_data"] == result2["image_data"]:
        print("✅ Images are identical - reproduction successful!")
    else:
        print("⚠️  Images differ slightly (this can happen due to floating point precision)")
    
    return result1


async def main():
    """Run all examples."""
    print("\n" + "=" * 60)
    print("🎨 Gator Local Image Generation Examples")
    print("=" * 60)
    print("\nThese examples demonstrate various image generation capabilities.")
    print("Note: First run will download models (~4-7 GB)")
    print()
    
    try:
        # Run examples
        await example_basic_generation()
        await example_custom_parameters()
        await example_batch_generation()
        await example_reproducible_generation()
        
        print("\n" + "=" * 60)
        print("✅ All examples completed!")
        print("=" * 60)
        print("\nGenerated files:")
        for f in Path(".").glob("example_*.png"):
            print(f"  - {f}")
        print()
        print("Next steps:")
        print("  1. Review the generated images")
        print("  2. Modify prompts and parameters in this script")
        print("  3. Read LOCAL_IMAGE_GENERATION.md for more details")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
