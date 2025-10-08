#!/usr/bin/env python3
"""
Demo of Advanced Video Features (Q2-Q3 2025)

Demonstrates the new video generation capabilities:
- Frame-by-frame generation
- Multiple transition types
- Video quality presets
- Storyboard creation
"""

import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))


async def demo_frame_by_frame_video():
    """Demonstrate frame-by-frame video generation."""
    from backend.services.video_processing_service import (
        VideoProcessingService,
        VideoQuality,
        TransitionType
    )
    
    print("\n" + "="*70)
    print("🎬 DEMO 1: Frame-by-Frame Video Generation")
    print("="*70)
    
    service = VideoProcessingService(output_dir="/tmp/gator_videos")
    
    prompts = [
        "Scene 1: Morning sunrise over mountains",
        "Scene 2: Afternoon city skyline",
        "Scene 3: Evening sunset at the beach"
    ]
    
    print(f"\n📝 Generating video with {len(prompts)} scenes...")
    print(f"   Quality: HIGH (1920x1080, 30fps)")
    print(f"   Transition: CROSSFADE")
    print(f"   Duration per scene: 2.0 seconds")
    
    result = await service.generate_frame_by_frame_video(
        prompts=prompts,
        duration_per_frame=2.0,
        quality=VideoQuality.HIGH,
        transition=TransitionType.CROSSFADE
    )
    
    print(f"\n✅ Video generated successfully!")
    print(f"   📁 Path: {result['file_path']}")
    print(f"   ⏱️  Duration: {result['duration']} seconds")
    print(f"   📐 Resolution: {result['resolution']}")
    print(f"   🎞️  Scenes: {result['num_scenes']}")
    print(f"   🔀 Transition: {result['transition_type']}")
    print(f"   📊 File size: {result['file_size'] / 1024:.2f} KB")
    
    return result


async def demo_all_transitions():
    """Demonstrate all transition types."""
    from backend.services.video_processing_service import (
        VideoProcessingService,
        VideoQuality,
        TransitionType
    )
    
    print("\n" + "="*70)
    print("🎬 DEMO 2: All Transition Types")
    print("="*70)
    
    service = VideoProcessingService(output_dir="/tmp/gator_videos")
    
    transitions = [
        TransitionType.FADE,
        TransitionType.CROSSFADE,
        TransitionType.WIPE,
        TransitionType.SLIDE,
        TransitionType.ZOOM,
        TransitionType.DISSOLVE
    ]
    
    prompts = ["Scene A", "Scene B"]
    
    print(f"\n📝 Testing {len(transitions)} transition types...")
    print(f"   Scenes: 2")
    print(f"   Quality: STANDARD (1280x720)")
    
    results = []
    for transition in transitions:
        print(f"\n   🔀 Testing {transition.value.upper()}...", end=" ")
        result = await service.generate_frame_by_frame_video(
            prompts=prompts,
            duration_per_frame=1.0,
            quality=VideoQuality.STANDARD,
            transition=transition
        )
        results.append(result)
        print(f"✅ ({result['file_size'] / 1024:.2f} KB)")
    
    print(f"\n✅ All transitions tested successfully!")
    print(f"   Total videos: {len(results)}")
    
    return results


async def demo_storyboard():
    """Demonstrate storyboard creation."""
    from backend.services.video_processing_service import (
        VideoProcessingService,
        VideoQuality
    )
    
    print("\n" + "="*70)
    print("🎬 DEMO 3: Storyboard Creation")
    print("="*70)
    
    service = VideoProcessingService(output_dir="/tmp/gator_videos")
    
    scenes = [
        {
            "prompt": "Opening: Hero walks into frame",
            "duration": 2.0,
            "transition": "fade"
        },
        {
            "prompt": "Act 1: Discovery of the artifact",
            "duration": 3.0,
            "transition": "crossfade"
        },
        {
            "prompt": "Act 2: Chase sequence begins",
            "duration": 2.5,
            "transition": "wipe"
        },
        {
            "prompt": "Climax: Final confrontation",
            "duration": 3.5,
            "transition": "zoom"
        },
        {
            "prompt": "Resolution: Peaceful ending",
            "duration": 2.0,
            "transition": "dissolve"
        }
    ]
    
    print(f"\n📝 Creating storyboard with {len(scenes)} scenes...")
    print(f"   Quality: HIGH (1920x1080)")
    print(f"   Total expected duration: ~{sum(s['duration'] for s in scenes):.1f} seconds")
    
    print("\n   Scene breakdown:")
    for i, scene in enumerate(scenes, 1):
        print(f"     {i}. {scene['prompt'][:40]:<40} ({scene['duration']}s, {scene['transition']})")
    
    result = await service.create_storyboard(
        scenes=scenes,
        quality=VideoQuality.HIGH
    )
    
    print(f"\n✅ Storyboard created successfully!")
    print(f"   📁 Path: {result['file_path']}")
    print(f"   ⏱️  Duration: {result['duration']:.2f} seconds")
    print(f"   📐 Resolution: {result['resolution']}")
    print(f"   🎞️  Scenes: {result['num_scenes']}")
    print(f"   📊 File size: {result['file_size'] / 1024:.2f} KB")
    
    print(f"\n   Scene markers:")
    for marker in result['scene_markers']:
        print(f"     Scene {marker['scene']}: {marker['timestamp']:.2f}s - {marker['prompt'][:40]}")
    
    return result


async def demo_quality_presets():
    """Demonstrate different quality presets."""
    from backend.services.video_processing_service import (
        VideoProcessingService,
        VideoQuality,
        TransitionType
    )
    
    print("\n" + "="*70)
    print("🎬 DEMO 4: Quality Presets Comparison")
    print("="*70)
    
    service = VideoProcessingService(output_dir="/tmp/gator_videos")
    
    qualities = [
        VideoQuality.DRAFT,
        VideoQuality.STANDARD,
        VideoQuality.HIGH,
        VideoQuality.PREMIUM
    ]
    
    prompts = ["Test scene for quality comparison"]
    
    print(f"\n📝 Testing {len(qualities)} quality presets...")
    
    results = []
    for quality in qualities:
        settings = service.quality_settings[quality]
        print(f"\n   📊 {quality.value.upper()}:")
        print(f"      Resolution: {settings['resolution'][0]}x{settings['resolution'][1]}")
        print(f"      FPS: {settings['fps']}")
        print(f"      Bitrate: {settings['bitrate']}")
        print(f"      Generating...", end=" ")
        
        result = await service.generate_frame_by_frame_video(
            prompts=prompts,
            duration_per_frame=1.0,
            quality=quality,
            transition=TransitionType.FADE
        )
        results.append(result)
        print(f"✅ ({result['file_size'] / 1024:.2f} KB)")
    
    print(f"\n✅ Quality comparison complete!")
    print(f"\n   Size comparison:")
    for quality, result in zip(qualities, results):
        size_kb = result['file_size'] / 1024
        print(f"     {quality.value:8s}: {size_kb:6.2f} KB")
    
    return results


async def demo_ai_models_integration():
    """Demonstrate AI models integration."""
    from backend.services.ai_models import ai_models
    
    print("\n" + "="*70)
    print("🎬 DEMO 5: AI Models Integration")
    print("="*70)
    
    # Initialize models
    if not ai_models.models_loaded:
        print("\n📦 Initializing AI models...")
        await ai_models.initialize_models()
        print("   ✅ Models initialized")
    
    # Get available video models
    models = await ai_models.get_available_models()
    video_models = models.get("video", [])
    
    print(f"\n📊 Available Video Models: {len(video_models)}")
    for model in video_models:
        status = "✅" if model.get("loaded") else "⏳"
        print(f"   {status} {model['name']}")
        print(f"      Type: {model.get('type', 'N/A')}")
        print(f"      Provider: {model.get('provider', 'N/A')}")
        if 'features' in model:
            print(f"      Features: {', '.join(model['features'])}")
    
    # Test video generation
    print(f"\n📝 Testing video generation via AI models...")
    print(f"   Using frame-by-frame generator")
    
    try:
        result = await ai_models.generate_video(
            prompt=[
                "Test scene 1",
                "Test scene 2"
            ],
            video_type="multi_frame",
            quality="standard",
            transition="crossfade",
            duration_per_frame=1.0
        )
        
        print(f"\n✅ Video generated via AI models!")
        print(f"   📁 Path: {result['file_path']}")
        print(f"   ⏱️  Duration: {result['duration']} seconds")
        print(f"   🎞️  Scenes: {result.get('num_scenes', 'N/A')}")
        
        return result
    except Exception as e:
        print(f"\n⚠️  Generation test skipped (expected in test environment)")
        print(f"   Reason: {str(e)}")
        return None


async def main():
    """Run all demos."""
    print("\n" + "="*70)
    print("🎥 GATOR ADVANCED VIDEO FEATURES DEMO")
    print("Q2-Q3 2025 Implementation")
    print("="*70)
    
    try:
        # Demo 1: Frame-by-frame generation
        await demo_frame_by_frame_video()
        
        # Demo 2: All transitions
        await demo_all_transitions()
        
        # Demo 3: Storyboard
        await demo_storyboard()
        
        # Demo 4: Quality presets
        await demo_quality_presets()
        
        # Demo 5: AI models integration
        await demo_ai_models_integration()
        
        print("\n" + "="*70)
        print("✅ ALL DEMOS COMPLETED SUCCESSFULLY!")
        print("="*70)
        print("\n📁 Generated videos are in: /tmp/gator_videos/")
        print("📖 Documentation: docs/VIDEO_FEATURES.md")
        print("🧪 Tests: pytest tests/unit/test_video_processing.py -v")
        print("\n")
        
    except Exception as e:
        print(f"\n❌ Demo failed: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
