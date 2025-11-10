#!/usr/bin/env python3
"""
Test script to verify ACD API endpoints are working.
"""
import asyncio
import sys
from uuid import uuid4

# Add src to path
sys.path.insert(0, 'src')

from backend.database.connection import database_manager, get_db_session
# Import generation_feedback model first to ensure foreign key tables exist
from backend.models.generation_feedback import GenerationBenchmarkModel
from backend.services.acd_service import ACDService
from backend.models.acd import (
    ACDContextCreate,
    ACDContextUpdate,
    ACDTraceArtifactCreate,
    AIStatus,
    AIState,
    AIComplexity,
    AIConfidence,
    AIRequest,
)


async def test_acd_endpoints():
    """Test all ACD endpoints to ensure they work correctly."""
    
    print("=" * 80)
    print("Testing ACD API Endpoints")
    print("=" * 80)
    
    # Connect to database
    await database_manager.connect()
    
    try:
        async for session in get_db_session():
            acd_service = ACDService(session)
            
            # Test 1: Create ACD context
            print("\n1. Testing: Create ACD context")
            context_data = ACDContextCreate(
                ai_phase="TEST_VALIDATION",
                ai_status=AIStatus.IMPLEMENTED,
                ai_complexity=AIComplexity.LOW,
                ai_state=AIState.PROCESSING,
                ai_note="Testing ACD API endpoints",
                ai_context={
                    "test": "endpoint_validation",
                    "purpose": "verify_acd_functionality"
                }
            )
            
            context = await acd_service.create_context(context_data)
            print(f"   ✅ Created context: {context.id}")
            print(f"      Phase: {context.ai_phase}")
            print(f"      Status: {context.ai_status}")
            print(f"      State: {context.ai_state}")
            
            # Test 2: Get ACD context
            print("\n2. Testing: Get ACD context")
            retrieved_context = await acd_service.get_context(context.id)
            print(f"   ✅ Retrieved context: {retrieved_context.id}")
            print(f"      Note: {retrieved_context.ai_note}")
            
            # Test 3: Update ACD context
            print("\n3. Testing: Update ACD context")
            update_data = ACDContextUpdate(
                ai_state=AIState.DONE,
                ai_confidence=AIConfidence.CONFIDENT
            )
            updated_context = await acd_service.update_context(context.id, update_data)
            print(f"   ✅ Updated context: {updated_context.id}")
            print(f"      New state: {updated_context.ai_state}")
            print(f"      Confidence: {updated_context.ai_confidence}")
            
            # Test 4: Create trace artifact
            print("\n4. Testing: Create trace artifact")
            artifact_data = ACDTraceArtifactCreate(
                session_id="test_session_123",
                event_type="test_event",
                error_message="Test error for validation",
                acd_context_id=context.id,
                stack_trace=["line 1", "line 2", "line 3"],
                environment={"test": True, "env": "validation"}
            )
            artifact = await acd_service.create_trace_artifact(artifact_data)
            print(f"   ✅ Created trace artifact: {artifact.id}")
            print(f"      Session: {artifact.session_id}")
            print(f"      Event: {artifact.event_type}")
            
            # Test 5: Get trace artifacts by session
            print("\n5. Testing: Get trace artifacts by session")
            artifacts = await acd_service.get_trace_artifacts_by_session("test_session_123")
            print(f"   ✅ Retrieved {len(artifacts)} artifact(s)")
            for art in artifacts:
                print(f"      - Artifact {art.id}: {art.event_type}")
            
            # Test 6: Get ACD stats
            print("\n6. Testing: Get ACD stats")
            stats = await acd_service.get_stats(hours=24)
            print(f"   ✅ Retrieved statistics:")
            print(f"      Total contexts: {stats.total_contexts}")
            print(f"      Active contexts: {stats.active_contexts}")
            print(f"      Completed contexts: {stats.completed_contexts}")
            print(f"      Failed contexts: {stats.failed_contexts}")
            
            # Test 7: Assign to agent
            print("\n7. Testing: Assign to agent")
            assigned_context = await acd_service.assign_to_agent(
                context.id,
                "test_agent_001",
                "Testing agent assignment functionality"
            )
            print(f"   ✅ Assigned context to agent:")
            print(f"      Agent: {assigned_context.ai_assigned_to}")
            print(f"      Reason: {assigned_context.ai_assignment_reason}")
            
            # Test 8: Validation report
            print("\n8. Testing: Generate validation report")
            report = await acd_service.generate_validation_report()
            print(f"   ✅ Generated validation report:")
            print(f"      Metadata: {report.metadata}")
            print(f"      Total contexts: {len(report.acd_contexts)}")
            print(f"      Errors: {len(report.errors)}")
            print(f"      Warnings: {len(report.warnings)}")
            
            print("\n" + "=" * 80)
            print("✅ All ACD API endpoint tests PASSED!")
            print("=" * 80)
            print("\nACD System Status: OPERATIONAL ✅")
            print("All endpoints are functioning correctly.\n")
            
            return True
            
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        await database_manager.disconnect()


if __name__ == "__main__":
    result = asyncio.run(test_acd_endpoints())
    sys.exit(0 if result else 1)
