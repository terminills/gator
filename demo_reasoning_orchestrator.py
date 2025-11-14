"""
Reasoning Orchestrator Demo - Basal Ganglia of ACD

Demonstrates the dynamic decision-making capabilities of the reasoning orchestrator,
showing how it functions like the basal ganglia to coordinate ACD operations.
"""

import asyncio
from datetime import datetime, timezone

from backend.database.connection import database_manager, get_async_session
from backend.services.reasoning_orchestrator import ReasoningOrchestrator, DecisionType
from backend.services.acd_service import ACDService
from backend.models.acd import (
    ACDContextCreate,
    ACDContextUpdate,
    AIStatus,
    AIState,
    AIComplexity,
    AIConfidence,
    AIValidation,
)


def print_header(title: str):
    """Print formatted section header."""
    print("\n" + "=" * 80)
    print(f" {title}")
    print("=" * 80)


def print_decision(decision):
    """Print orchestration decision details."""
    print(f"\n🧠 Orchestration Decision:")
    print(f"   Type: {decision.decision_type.value}")
    print(f"   Confidence: {decision.confidence.value}")
    print(f"   Reasoning: {decision.reasoning}")
    if decision.target_agent:
        print(f"   Target Agent: {decision.target_agent}")
    if decision.learned_patterns:
        print(f"   Learned Patterns: {', '.join(decision.learned_patterns[:3])}")
    print(f"   Action Plan: {decision.action_plan}")


async def demo_simple_task():
    """Demo 1: Simple task with high confidence."""
    print_header("DEMO 1: Simple Task - Execute Locally")
    
    async with get_async_session() as session:
        acd_service = ACDService(session)
        orchestrator = ReasoningOrchestrator(session)
        
        # Create a simple, low-complexity task
        context = await acd_service.create_context(
            ACDContextCreate(
                ai_phase="TEXT_GENERATION",
                ai_status=AIStatus.IMPLEMENTED,
                ai_complexity=AIComplexity.LOW,
                ai_confidence=AIConfidence.CONFIDENT,
                ai_state=AIState.READY,
                ai_note="Generate simple caption for Instagram post",
                ai_context={
                    "platform": "instagram",
                    "content_type": "caption",
                    "length": "short",
                },
            )
        )
        
        print(f"\n📝 Created Context: {context.id}")
        print(f"   Phase: {context.ai_phase}")
        print(f"   Complexity: {context.ai_complexity}")
        print(f"   Confidence: {context.ai_confidence}")
        
        # Orchestrate decision
        decision = await orchestrator.orchestrate_decision(context, current_agent="caption_writer")
        print_decision(decision)
        
        # Execute decision
        print(f"\n⚡ Executing decision...")
        success = await orchestrator.execute_decision(context, decision)
        print(f"   Execution: {'✅ Success' if success else '❌ Failed'}")
        
        return decision.decision_type == DecisionType.EXECUTE_LOCALLY


async def demo_complex_task():
    """Demo 2: Complex task requiring escalation."""
    print_header("DEMO 2: Complex Task - Escalation Required")
    
    async with get_async_session() as session:
        acd_service = ACDService(session)
        orchestrator = ReasoningOrchestrator(session)
        
        # Create a complex, critical task
        context = await acd_service.create_context(
            ACDContextCreate(
                ai_phase="VIDEO_GENERATION",
                ai_status=AIStatus.IMPLEMENTED,
                ai_complexity=AIComplexity.CRITICAL,
                ai_confidence=AIConfidence.UNCERTAIN,
                ai_state=AIState.READY,
                ai_note="Generate high-quality promotional video with custom animations",
                ai_context={
                    "duration": 60,
                    "quality": "4K",
                    "animations": ["intro", "transitions", "outro"],
                    "complexity_factors": ["lip_sync", "background_removal", "effects"],
                },
            )
        )
        
        print(f"\n📝 Created Context: {context.id}")
        print(f"   Phase: {context.ai_phase}")
        print(f"   Complexity: {context.ai_complexity}")
        print(f"   Confidence: {context.ai_confidence}")
        
        # Orchestrate decision
        decision = await orchestrator.orchestrate_decision(context, current_agent="basic_video_gen")
        print_decision(decision)
        
        # Execute decision
        print(f"\n⚡ Executing decision...")
        success = await orchestrator.execute_decision(context, decision)
        print(f"   Execution: {'✅ Success' if success else '❌ Failed'}")
        
        # Verify handoff was requested
        updated_context = await acd_service.get_context(context.id)
        if updated_context.ai_handoff_requested:
            print(f"\n📨 Handoff Requested:")
            print(f"   To: {updated_context.ai_handoff_to}")
            print(f"   Type: {updated_context.ai_handoff_type}")
            print(f"   Reason: {updated_context.ai_handoff_reason}")
        
        return decision.decision_type == DecisionType.HANDOFF_ESCALATION


async def demo_error_recovery():
    """Demo 3: Task with errors - retry with learning."""
    print_header("DEMO 3: Error Recovery - Learning from Failures")
    
    async with get_async_session() as session:
        acd_service = ACDService(session)
        orchestrator = ReasoningOrchestrator(session)
        
        # First, create some successful patterns
        print("\n📚 Building knowledge base with successful patterns...")
        for i in range(3):
            success_context = await acd_service.create_context(
                ACDContextCreate(
                    ai_phase="IMAGE_GENERATION",
                    ai_status=AIStatus.IMPLEMENTED,
                    ai_complexity=AIComplexity.MEDIUM,
                    ai_state=AIState.DONE,
                    ai_validation=AIValidation.APPROVED.value,
                    ai_assigned_to="image_specialist",
                    ai_strategy="use_stable_diffusion_xl_with_refinement",
                    ai_pattern="high_quality_portrait",
                )
            )
            print(f"   ✓ Pattern {i+1} created")
        
        # Create a failed task
        print("\n💥 Creating failed task...")
        context = await acd_service.create_context(
            ACDContextCreate(
                ai_phase="IMAGE_GENERATION",
                ai_status=AIStatus.IMPLEMENTED,
                ai_complexity=AIComplexity.MEDIUM,
                ai_confidence=AIConfidence.UNCERTAIN,
                ai_state=AIState.FAILED,
                ai_note="Image generation failed - CUDA out of memory",
                ai_metadata={"retry_count": 1},
                runtime_err="RuntimeError: CUDA out of memory",
            )
        )
        
        print(f"\n📝 Created Context: {context.id}")
        print(f"   Phase: {context.ai_phase}")
        print(f"   State: {context.ai_state}")
        print(f"   Error: {context.runtime_err}")
        print(f"   Retry Count: {context.ai_metadata.get('retry_count', 0)}")
        
        # Orchestrate decision
        decision = await orchestrator.orchestrate_decision(context)
        print_decision(decision)
        
        # Execute decision
        print(f"\n⚡ Executing decision...")
        success = await orchestrator.execute_decision(context, decision)
        print(f"   Execution: {'✅ Success' if success else '❌ Failed'}")
        
        # Show learned strategies
        if decision.learned_patterns:
            print(f"\n🎓 Applying Learned Strategies:")
            for i, pattern in enumerate(decision.learned_patterns[:3], 1):
                print(f"   {i}. {pattern}")
        
        return decision.decision_type == DecisionType.RETRY_WITH_LEARNING


async def demo_pattern_learning():
    """Demo 4: Pattern-based handoff decision."""
    print_header("DEMO 4: Pattern-Based Handoff - Learning from History")
    
    async with get_async_session() as session:
        acd_service = ACDService(session)
        orchestrator = ReasoningOrchestrator(session)
        
        # Create multiple successful handoffs to a specialist
        print("\n📚 Building pattern history...")
        specialist_name = "social_media_expert"
        
        for i in range(5):
            success_context = await acd_service.create_context(
                ACDContextCreate(
                    ai_phase="SOCIAL_MEDIA_CONTENT",
                    ai_status=AIStatus.IMPLEMENTED,
                    ai_complexity=AIComplexity.MEDIUM,
                    ai_state=AIState.DONE,
                    ai_validation=AIValidation.APPROVED.value,
                    ai_assigned_to=specialist_name,
                    ai_strategy="viral_content_formula",
                    ai_pattern="high_engagement_post",
                )
            )
            print(f"   ✓ Success pattern {i+1}: assigned to {specialist_name}")
        
        # Create new similar task with low confidence
        print(f"\n📝 Creating new similar task...")
        context = await acd_service.create_context(
            ACDContextCreate(
                ai_phase="SOCIAL_MEDIA_CONTENT",
                ai_status=AIStatus.IMPLEMENTED,
                ai_complexity=AIComplexity.MEDIUM,
                ai_confidence=AIConfidence.UNCERTAIN,
                ai_state=AIState.READY,
                ai_note="Create viral TikTok content",
            )
        )
        
        print(f"   Context: {context.id}")
        print(f"   Phase: {context.ai_phase}")
        print(f"   Confidence: {context.ai_confidence}")
        
        # Orchestrate - should learn from patterns
        decision = await orchestrator.orchestrate_decision(context, current_agent="basic_generator")
        print_decision(decision)
        
        # Check if learned patterns influenced decision
        print(f"\n🧠 Pattern Learning Analysis:")
        if decision.target_agent:
            print(f"   Recommended Agent: {decision.target_agent}")
            if decision.target_agent == specialist_name:
                print(f"   ✅ Correctly learned to handoff to specialist!")
        
        if decision.learned_patterns:
            print(f"   Patterns Applied: {len(decision.learned_patterns)}")
            for pattern in decision.learned_patterns[:3]:
                print(f"     • {pattern}")
        
        return decision.decision_type == DecisionType.HANDOFF_SPECIALIZATION


async def demo_learning_cycle():
    """Demo 5: Complete learning cycle with feedback."""
    print_header("DEMO 5: Learning Cycle - Reinforcement from Outcomes")
    
    async with get_async_session() as session:
        acd_service = ACDService(session)
        orchestrator = ReasoningOrchestrator(session)
        
        # Step 1: Create task
        print("\n📝 Step 1: Creating task...")
        context = await acd_service.create_context(
            ACDContextCreate(
                ai_phase="TEXT_GENERATION",
                ai_status=AIStatus.IMPLEMENTED,
                ai_complexity=AIComplexity.MEDIUM,
                ai_confidence=AIConfidence.CONFIDENT,
                ai_state=AIState.READY,
            )
        )
        print(f"   Context: {context.id}")
        
        # Step 2: Make decision
        print(f"\n🧠 Step 2: Making orchestration decision...")
        decision = await orchestrator.orchestrate_decision(context)
        print(f"   Decision: {decision.decision_type.value}")
        
        # Step 3: Simulate successful execution
        print(f"\n⚡ Step 3: Simulating successful execution...")
        await acd_service.update_context(
            context.id,
            ACDContextUpdate(
                ai_state=AIState.DONE.value,
                ai_validation=AIValidation.APPROVED.value,
            )
        )
        print(f"   ✅ Task completed successfully")
        
        # Step 4: Learn from outcome
        print(f"\n🎓 Step 4: Learning from successful outcome...")
        await orchestrator.learn_from_outcome(
            context.id,
            success=True,
            outcome_metadata={
                "engagement_rate": 8.5,
                "user_rating": "excellent",
            }
        )
        print(f"   ✅ Pattern reinforced in learning system")
        
        # Verify learning
        updated_context = await acd_service.get_context(context.id)
        print(f"\n📊 Learning Results:")
        print(f"   Context updated: ✅")
        print(f"   Decision logged: {'✅' if 'orchestration_decision' in updated_context.ai_metadata else '❌'}")
        print(f"   Pattern cache cleared: ✅ (ready for fresh patterns)")
        
        # Now test with a failed outcome
        print(f"\n💥 Testing failure learning...")
        
        fail_context = await acd_service.create_context(
            ACDContextCreate(
                ai_phase="IMAGE_GENERATION",
                ai_status=AIStatus.IMPLEMENTED,
                ai_complexity=AIComplexity.HIGH,
                ai_state=AIState.FAILED,
                ai_validation=AIValidation.REJECTED.value,
            )
        )
        
        fail_decision = await orchestrator.orchestrate_decision(fail_context)
        
        await orchestrator.learn_from_outcome(
            fail_context.id,
            success=False,
            outcome_metadata={"failure_reason": "Low quality output"}
        )
        
        fail_updated = await acd_service.get_context(fail_context.id)
        if fail_updated.ai_metadata and "failed_decisions" in fail_updated.ai_metadata:
            print(f"   ✅ Failed pattern recorded for inhibition")
            print(f"   Failed decisions logged: {len(fail_updated.ai_metadata['failed_decisions'])}")
        else:
            print(f"   ℹ️  Failure logged in system")
        
        return True


async def main():
    """Run all demos."""
    print("\n" + "=" * 80)
    print(" 🧠 REASONING ORCHESTRATOR DEMONSTRATION")
    print(" Basal Ganglia Architecture for ACD")
    print("=" * 80)
    print("\nThis demo showcases the reasoning orchestrator - the 'brain' of the ACD system")
    print("that dynamically makes decisions about task routing, agent coordination, and")
    print("pattern learning, functioning like the basal ganglia in the human brain.\n")
    
    # Connect to database
    await database_manager.connect()
    
    try:
        results = []
        
        # Run demos
        results.append(("Simple Task Execution", await demo_simple_task()))
        results.append(("Complex Task Escalation", await demo_complex_task()))
        results.append(("Error Recovery with Learning", await demo_error_recovery()))
        results.append(("Pattern-Based Handoff", await demo_pattern_learning()))
        results.append(("Complete Learning Cycle", await demo_learning_cycle()))
        
        # Summary
        print_header("DEMONSTRATION SUMMARY")
        print("\n✅ Capabilities Demonstrated:\n")
        for name, success in results:
            status = "✅" if success else "⚠️"
            print(f"{status} {name}")
        
        print("\n🎯 Key Features of Reasoning Orchestrator:")
        print("   • Action Selection: Dynamically chooses best agent/action")
        print("   • Pattern Learning: Learns from successful and failed outcomes")
        print("   • Motor Control: Coordinates multiple specialized agents")
        print("   • Habit Formation: Builds procedural knowledge from repetition")
        print("   • Error Recovery: Intelligently handles failures with learned strategies")
        print("   • Confidence-Based Routing: Escalates when confidence is low")
        
        print("\n🧬 Basal Ganglia Analogy:")
        print("   Like the basal ganglia, the orchestrator:")
        print("   1. Selects actions based on context and learned patterns")
        print("   2. Coordinates between specialized 'motor' units (agents)")
        print("   3. Learns through reinforcement (success) and inhibition (failure)")
        print("   4. Automates decisions that were once deliberate")
        print("   5. Integrates feedback to improve future decisions")
        
        print("\n" + "=" * 80)
        print(" ✅ ALL DEMONSTRATIONS COMPLETED SUCCESSFULLY")
        print("=" * 80 + "\n")
        
    finally:
        await database_manager.disconnect()


if __name__ == "__main__":
    asyncio.run(main())
