#!/usr/bin/env python3
"""
Demo of AI-Powered Video Generation

Demonstrates the enhanced video generation capabilities with actual
AI image generation for each frame instead of placeholder frames.

This demonstrates both:
1. Placeholder mode (fast, for testing/preview)
2. AI-generated mode (slower, produces actual images)
"""

import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))


async def demo_placeholder_vs_ai_generation():
    """Compare placeholder vs AI-generated video frames."""
    from backend.services.ai_models import AIModelManager
    
    print("\n" + "="*70)
    print("🎬 DEMO: Placeholder vs AI-Generated Video Frames")
    print("="*70)
    
    # Initialize AI Model Manager
    print("\n1. Initializing AI Model Manager...")
    ai_manager = AIModelManager()
    await ai_manager.initialize_models()
    
    print(f"   - GPU Type: {ai_manager.gpu_type}")
    print(f"   - GPU Memory: {ai_manager.gpu_memory_gb} GB")
    print(f"   - Available image models: {len([m for m in ai_manager.available_models['image'] if m.get('loaded') or m.get('can_load')])}")
    
    prompts = [
        "A serene mountain landscape at sunrise",
        "A bustling city skyline at night",
    ]
    
    # Demo 1: Placeholder mode (fast)
    print("\n" + "-"*70)
    print("2. Generating video with PLACEHOLDER frames (fast mode)")
    print("-"*70)
    print(f"   Prompts: {prompts}")
    print(f"   Quality: STANDARD (1280x720)")
    print(f"   Use AI: False")
    
    try:
        result_placeholder = await ai_manager.generate_video(
            prompt=prompts,
            video_type="multi_frame",
            quality="standard",
            transition="crossfade",
            duration_per_frame=2.0,
            use_ai_generation=False,  # Use placeholder frames
        )
        
        print(f"\n✅ Placeholder video generated!")
        print(f"   📁 Path: {result_placeholder['file_path']}")
        print(f"   ⏱️  Duration: {result_placeholder['duration']} seconds")
        print(f"   📊 File size: {result_placeholder.get('file_size', 0) / 1024:.2f} KB")
        print(f"   🎞️  Scenes: {result_placeholder.get('num_scenes', len(prompts))}")
    except Exception as e:
        print(f"\n❌ Placeholder generation failed: {str(e)}")
    
    # Demo 2: AI-generated mode (slower but actual images)
    print("\n" + "-"*70)
    print("3. Generating video with AI-GENERATED frames (quality mode)")
    print("-"*70)
    print(f"   Prompts: {prompts}")
    print(f"   Quality: STANDARD (1280x720)")
    print(f"   Use AI: True")
    
    # Check if we have available image models
    has_image_models = any(
        m.get("loaded") or m.get("can_load")
        for m in ai_manager.available_models["image"]
    )
    
    if not has_image_models:
        print("\n⚠️  No image generation models available")
        print("   This requires:")
        print("   - GPU with sufficient memory (4+ GB for SD 1.5, 8+ GB for SDXL)")
        print("   - Or API keys (OPENAI_API_KEY for DALL-E)")
        print("   Skipping AI generation demo...")
        return
    
    try:
        print(f"   Note: AI generation may take several minutes for first run")
        print(f"         (model download + generation time)")
        
        result_ai = await ai_manager.generate_video(
            prompt=prompts,
            video_type="multi_frame",
            quality="standard",
            transition="crossfade",
            duration_per_frame=2.0,
            use_ai_generation=True,  # Use AI-generated frames
            num_inference_steps=20,  # Fewer steps for faster demo
        )
        
        print(f"\n✅ AI-generated video created!")
        print(f"   📁 Path: {result_ai['file_path']}")
        print(f"   ⏱️  Duration: {result_ai['duration']} seconds")
        print(f"   📊 File size: {result_ai.get('file_size', 0) / 1024:.2f} KB")
        print(f"   🎞️  Scenes: {result_ai.get('num_scenes', len(prompts))}")
        print(f"\n   💡 Compare the two videos to see the difference!")
        print(f"      Placeholder: Simple gradients with text")
        print(f"      AI-generated: Actual images from prompts")
    except Exception as e:
        print(f"\n❌ AI generation failed: {str(e)}")
        print(f"   This is expected if no models are available")
        import traceback
        traceback.print_exc()


async def demo_direct_video_service():
    """Demonstrate using VideoProcessingService directly with AI."""
    from backend.services.video_processing_service import (
        VideoProcessingService,
        VideoQuality,
        TransitionType,
    )
    from backend.services.ai_models import AIModelManager
    
    print("\n" + "="*70)
    print("🎬 DEMO: Direct VideoProcessingService with AI")
    print("="*70)
    
    # Initialize services
    print("\n1. Initializing services...")
    ai_manager = AIModelManager()
    await ai_manager.initialize_models()
    
    video_service = VideoProcessingService(output_dir="/tmp/gator_ai_videos")
    
    has_image_models = any(
        m.get("loaded") or m.get("can_load")
        for m in ai_manager.available_models["image"]
    )
    
    if not has_image_models:
        print("\n⚠️  No image models available, skipping this demo")
        return
    
    prompts = [
        "A beautiful sunset over the ocean",
        "A peaceful forest with rays of sunlight",
        "A starry night sky with the milky way",
    ]
    
    print("\n2. Generating multi-scene video with AI...")
    print(f"   Scenes: {len(prompts)}")
    print(f"   Quality: HIGH (1920x1080)")
    print(f"   Transition: CROSSFADE")
    
    try:
        result = await video_service.generate_frame_by_frame_video(
            prompts=prompts,
            duration_per_frame=2.0,
            quality=VideoQuality.HIGH,
            transition=TransitionType.CROSSFADE,
            use_ai_generation=True,
            ai_model_manager=ai_manager,
            num_inference_steps=20,
        )
        
        print(f"\n✅ Video generated with AI frames!")
        print(f"   📁 Path: {result['file_path']}")
        print(f"   ⏱️  Total duration: {result['duration']} seconds")
        print(f"   📐 Resolution: {result['resolution']}")
        print(f"   🎞️  Scenes: {result['num_scenes']}")
        print(f"   🔀 Transitions: {result['transition_type']}")
        print(f"   📊 File size: {result['file_size'] / 1024:.2f} KB")
    except Exception as e:
        print(f"\n❌ Generation failed: {str(e)}")
        import traceback
        traceback.print_exc()


async def main():
    """Run all demos."""
    print("\n" + "="*70)
    print("🎥 GATOR AI-POWERED VIDEO GENERATION DEMO")
    print("Enhanced Video Features with Real Image Generation")
    print("="*70)
    
    try:
        # Demo 1: Compare placeholder vs AI
        await demo_placeholder_vs_ai_generation()
        
        # Demo 2: Direct service usage
        await demo_direct_video_service()
        
        print("\n" + "="*70)
        print("✅ ALL DEMOS COMPLETED!")
        print("="*70)
        print("\n📁 Generated videos are in:")
        print("   - AI Manager: generated_content/videos/")
        print("   - Direct service: /tmp/gator_ai_videos/")
        print("\n📝 Key Points:")
        print("   - use_ai_generation=False: Fast placeholder frames")
        print("   - use_ai_generation=True: Real AI-generated images")
        print("   - Requires GPU or API keys for AI generation")
        print("   - Fallback to placeholder if AI generation fails")
        print("\n🧪 Run tests:")
        print("   pytest tests/unit/test_ai_video_frame_generation.py -v")
        print("\n")
        
    except Exception as e:
        print(f"\n❌ Demo failed: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
