#!/usr/bin/env python3
"""
ACD Integration Demo

Demonstrates how ACD context tracking integrates with the content generation
feedback loop to enable autonomous learning and improvement.
"""

import asyncio
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from backend.database.connection import database_manager
from backend.services.acd_service import ACDService
from backend.services.generation_feedback_service import GenerationFeedbackService
from backend.models.acd import (
    ACDContextCreate,
    ACDContextUpdate,
    ACDTraceArtifactCreate,
    AIStatus,
    AIState,
    AIComplexity,
    AIConfidence,
    AIQueuePriority,
    AIQueueStatus,
    AIValidation,
)
from backend.models.generation_feedback import (
    GenerationBenchmarkCreate,
    FeedbackSubmission,
    FeedbackRating,
)
from backend.utils.acd_integration import (
    ACDContextManager,
    get_phase_from_content_type,
    get_complexity_from_quality,
)
from backend.config.logging import setup_logging, get_logger

# Setup logging
setup_logging()
logger = get_logger(__name__)


async def demo_basic_acd_tracking():
    """Demo 1: Basic ACD context tracking for content generation."""
    print("\n" + "=" * 70)
    print("DEMO 1: Basic ACD Context Tracking")
    print("=" * 70)

    await database_manager.connect()

    async with database_manager.get_session() as session:
        acd_service = ACDService(session)

        # Simulate content generation with ACD tracking
        print("\n📝 Creating ACD context for image generation...")
        context = await acd_service.create_context(
            ACDContextCreate(
                ai_phase="IMAGE_GENERATION",
                ai_status=AIStatus.IMPLEMENTED,
                ai_complexity=AIComplexity.MEDIUM,
                ai_note="Generating portrait image with Stable Diffusion",
                ai_state=AIState.PROCESSING,
                ai_queue_priority=AIQueuePriority.NORMAL,
                ai_queue_status=AIQueueStatus.IN_PROGRESS,
                ai_context={
                    "prompt": "professional portrait, studio lighting",
                    "model": "stable-diffusion-xl",
                    "steps": 30,
                    "guidance": 7.5,
                },
            )
        )

        print(f"✅ Context created: {context.id}")
        print(f"   Phase: {context.ai_phase}")
        print(f"   Status: {context.ai_status}")
        print(f"   State: {context.ai_state}")

        # Simulate successful generation
        print("\n🎨 Simulating content generation...")
        await asyncio.sleep(0.5)  # Simulate generation time

        print("✅ Generation successful!")
        await acd_service.update_context(
            context.id,
            ACDContextUpdate(
                ai_state=AIState.DONE,
                ai_confidence=AIConfidence.CONFIDENT,
                ai_queue_status=AIQueueStatus.COMPLETED,
            ),
        )

        print(f"📊 Context updated to: {AIState.DONE.value}")

    await database_manager.disconnect()


async def demo_error_tracking():
    """Demo 2: Error tracking with trace artifacts."""
    print("\n" + "=" * 70)
    print("DEMO 2: Error Tracking with Trace Artifacts")
    print("=" * 70)

    await database_manager.connect()

    async with database_manager.get_session() as session:
        acd_service = ACDService(session)

        # Create context for a failing generation
        print("\n📝 Creating ACD context for video generation...")
        context = await acd_service.create_context(
            ACDContextCreate(
                ai_phase="VIDEO_GENERATION",
                ai_status=AIStatus.IMPLEMENTED,
                ai_complexity=AIComplexity.HIGH,
                ai_note="Attempting video generation with limited GPU memory",
                ai_state=AIState.PROCESSING,
            )
        )

        print(f"✅ Context created: {context.id}")

        # Simulate error
        print("\n⚠️  Simulating generation error...")
        try:
            # This would be the actual generation code
            raise RuntimeError("CUDA out of memory: 8GB required, 4GB available")
        except Exception as e:
            print(f"❌ Error occurred: {e}")

            # Create trace artifact
            artifact = await acd_service.create_trace_artifact(
                ACDTraceArtifactCreate(
                    session_id=str(context.id),
                    event_type="runtime_error",
                    error_message=str(e),
                    error_code="CUDA_OOM",
                    error_file="video_generation_service.py",
                    error_line=342,
                    error_function="generate_video",
                    acd_context_id=context.id,
                    stack_trace=[
                        "File video_generation_service.py, line 342, in generate_video",
                        "File model.py, line 156, in forward",
                        "torch.cuda.OutOfMemoryError",
                    ],
                    environment={
                        "gpu_memory_available": "4GB",
                        "gpu_memory_required": "8GB",
                        "model": "video-diffusion-xl",
                    },
                )
            )

            print(f"📋 Trace artifact created: {artifact.id}")

            # Update context to failed
            await acd_service.update_context(
                context.id,
                ACDContextUpdate(
                    ai_state=AIState.FAILED,
                    ai_queue_status=AIQueueStatus.ABANDONED,
                ),
            )

            print(f"📊 Context updated to: {AIState.FAILED.value}")

    await database_manager.disconnect()


async def demo_feedback_loop_integration():
    """Demo 3: Integration with generation feedback for learning."""
    print("\n" + "=" * 70)
    print("DEMO 3: Feedback Loop Integration")
    print("=" * 70)

    await database_manager.connect()

    async with database_manager.get_session() as session:
        acd_service = ACDService(session)
        feedback_service = GenerationFeedbackService(session)

        # Create ACD context
        print("\n📝 Creating ACD context for text generation...")
        acd_context = await acd_service.create_context(
            ACDContextCreate(
                ai_phase="TEXT_GENERATION",
                ai_status=AIStatus.IMPLEMENTED,
                ai_complexity=AIComplexity.LOW,
                ai_note="Generating engaging caption for social media",
                ai_state=AIState.DONE,
                ai_confidence=AIConfidence.CONFIDENT,
                ai_context={
                    "prompt": "Write an engaging caption about sustainable fashion",
                    "model": "gpt-3.5-turbo",
                    "temperature": 0.7,
                },
            )
        )

        print(f"✅ ACD Context: {acd_context.id}")

        # Create generation benchmark with ACD link
        print("\n📊 Recording generation benchmark...")
        benchmark = await feedback_service.record_benchmark(
            GenerationBenchmarkCreate(
                content_type="text",
                prompt="Write an engaging caption about sustainable fashion",
                enhanced_prompt="Write a compelling, authentic caption about sustainable fashion with emotional appeal",
                model_selected="gpt-3.5-turbo",
                model_provider="openai",
                selection_reasoning="Fast, cost-effective for caption generation",
                generation_time_seconds=1.2,
                total_time_seconds=1.5,
                generation_params={"temperature": 0.7, "max_tokens": 150},
                quality_requested="standard",
                quality_score=85.0,
                acd_context_id=acd_context.id,
                acd_phase="TEXT_GENERATION",
            )
        )

        print(f"✅ Benchmark: {benchmark.id}")
        print(f"   Linked to ACD context: {benchmark.acd_context_id}")

        # Simulate human feedback
        print("\n👤 Simulating human feedback (excellent rating)...")
        await feedback_service.submit_feedback(
            FeedbackSubmission(
                benchmark_id=benchmark.id,
                rating=FeedbackRating.EXCELLENT,
                feedback_text="Perfect tone and very engaging!",
            )
        )

        # Update ACD context with validation
        await acd_service.update_context(
            acd_context.id,
            ACDContextUpdate(
                ai_validation=AIValidation.APPROVED,
                ai_suggestions=["Great use of emotional appeal", "Good length"],
            ),
        )

        print("✅ Feedback recorded and ACD context updated")
        print(f"   This data can now be used to improve future generations!")

    await database_manager.disconnect()


async def demo_context_manager():
    """Demo 4: Using ACDContextManager for automatic tracking."""
    print("\n" + "=" * 70)
    print("DEMO 4: ACDContextManager for Automatic Tracking")
    print("=" * 70)

    await database_manager.connect()

    async with database_manager.get_session() as session:
        print("\n🔄 Using context manager for automatic ACD tracking...")

        # Simulate successful generation
        print("\n✅ Scenario 1: Successful generation")
        async with ACDContextManager(
            session,
            phase="IMAGE_GENERATION",
            note="Generating product photo",
            complexity=AIComplexity.MEDIUM,
            initial_context={"product": "sustainable water bottle"},
        ) as acd:
            # Simulate generation work
            print("   🎨 Generating image...")
            await asyncio.sleep(0.3)
            await acd.set_confidence(AIConfidence.CONFIDENT)
            print("   ✅ Image generated successfully")
            # Context automatically updated to DONE

        # Simulate failed generation
        print("\n❌ Scenario 2: Failed generation")
        try:
            async with ACDContextManager(
                session,
                phase="VIDEO_GENERATION",
                note="Generating promotional video",
                complexity=AIComplexity.HIGH,
            ) as acd:
                # Simulate generation work
                print("   🎬 Generating video...")
                await asyncio.sleep(0.2)
                # Simulate error
                raise ValueError("Invalid video format specified")
        except ValueError:
            print("   ❌ Generation failed (trace artifact created automatically)")

    await database_manager.disconnect()


async def demo_statistics_and_reporting():
    """Demo 5: Statistics and validation reporting."""
    print("\n" + "=" * 70)
    print("DEMO 5: Statistics and Validation Reporting")
    print("=" * 70)

    await database_manager.connect()

    async with database_manager.get_session() as session:
        acd_service = ACDService(session)

        # Get statistics
        print("\n📊 Retrieving ACD statistics...")
        stats = await acd_service.get_stats(hours=24)

        print(f"\n📈 Statistics (last 24 hours):")
        print(f"   Total contexts: {stats.total_contexts}")
        print(f"   Active contexts: {stats.active_contexts}")
        print(f"   Completed contexts: {stats.completed_contexts}")
        print(f"   Failed contexts: {stats.failed_contexts}")

        if stats.by_phase:
            print(f"\n   By Phase:")
            for phase, count in stats.by_phase.items():
                print(f"      {phase}: {count}")

        if stats.by_state:
            print(f"\n   By State:")
            for state, count in stats.by_state.items():
                print(f"      {state}: {count}")

        # Generate validation report
        print("\n📋 Generating validation report...")
        report = await acd_service.generate_validation_report()

        print(f"\n✅ Validation Report:")
        print(f"   Schema version: {report.metadata['acd_schema_version']}")
        print(f"   Contexts found: {report.metadata['acd_metadata_found']}")
        print(f"   Errors: {report.metadata['errors']}")
        print(f"   Warnings: {report.metadata['warnings']}")

    await database_manager.disconnect()


async def main():
    """Run all demos."""
    print("\n" + "=" * 70)
    print("🦎 GATOR ACD INTEGRATION DEMO")
    print("=" * 70)
    print("\nDemonstrating how ACD (Autonomous Continuous Development)")
    print("enables context-aware, self-improving AI content generation.")

    try:
        await demo_basic_acd_tracking()
        await demo_error_tracking()
        await demo_feedback_loop_integration()
        await demo_context_manager()
        await demo_statistics_and_reporting()

        print("\n" + "=" * 70)
        print("✅ ALL DEMOS COMPLETED SUCCESSFULLY")
        print("=" * 70)
        print("\nKey Takeaways:")
        print("  1. ACD tracks every generation with rich context")
        print("  2. Errors are captured with full diagnostic information")
        print("  3. Human feedback loops back to improve future generations")
        print("  4. Context managers make tracking automatic and seamless")
        print("  5. Analytics provide insights for continuous improvement")
        print("\n🚀 The system is now learning from every generation!")

    except Exception as e:
        logger.error(f"Demo failed: {e}", exc_info=True)
        print(f"\n❌ Demo failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
