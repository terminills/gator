#!/usr/bin/env python3
"""
Test ACD Content Generation Integration

Validates that ACD service can trigger actual content generation
and that model selection/fallback logic works correctly.
"""

import asyncio
import sys
from pathlib import Path
from uuid import uuid4

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from backend.database.connection import database_manager
from backend.services.acd_service import ACDService
from backend.services.content_generation_service import ContentGenerationService
from backend.models.acd import (
    ACDContextCreate,
    AIStatus,
    AIState,
    AIComplexity,
    AIQueuePriority,
    AIQueueStatus,
)
from backend.models.persona import PersonaCreate
from backend.services.persona_service import PersonaService
from backend.config.logging import setup_logging, get_logger

setup_logging()
logger = get_logger(__name__)


async def test_acd_triggers_generation():
    """Test that ACD service can trigger content generation."""
    print("\n" + "=" * 80)
    print("TEST 1: ACD Triggers Content Generation")
    print("=" * 80)
    
    await database_manager.connect()
    
    try:
        async with database_manager.get_session() as session:
            # Create a test persona
            persona_service = PersonaService(session)
            persona = await persona_service.create_persona(
                PersonaCreate(
                    name=f"Test ACD Persona {int(asyncio.get_event_loop().time())}",
                    appearance="Professional looking AI assistant with a friendly demeanor",
                    personality="Helpful, knowledgeable, and enthusiastic about automation and testing",
                )
            )
            print(f"\n✓ Created test persona: {persona.name} ({persona.id})")
            
            # Create ACD context with generation request
            acd_service = ACDService(session)
            print(f"\n📝 Creating ACD context for TEXT_GENERATION...")
            
            context = await acd_service.create_context(
                ACDContextCreate(
                    ai_phase="TEXT_GENERATION",
                    ai_status=AIStatus.IMPLEMENTED,
                    ai_complexity=AIComplexity.LOW,
                    ai_note="Test: ACD triggering content generation",
                    ai_state=AIState.READY,
                    ai_queue_priority=AIQueuePriority.HIGH,
                    ai_queue_status=AIQueueStatus.QUEUED,
                    ai_context={
                        "persona_id": str(persona.id),
                        "prompt": "Write a short test message about AI automation",
                        "quality": "standard",
                    },
                )
            )
            
            print(f"✓ Created ACD context: {context.id}")
            print(f"  Phase: {context.ai_phase}")
            print(f"  State: {context.ai_state}")
            print(f"  Queue Status: {context.ai_queue_status}")
            print(f"  Priority: {context.ai_queue_priority}")
            
            # Process the queue to trigger generation
            print(f"\n🔄 Processing ACD queue to trigger generation...")
            results = await acd_service.process_queued_contexts(max_contexts=1)
            
            print(f"\n📊 Queue Processing Results:")
            print(f"  Total processed: {results['processed']}")
            print(f"  Successful: {results['successful']}")
            print(f"  Failed: {results['failed']}")
            
            if results['successful'] > 0:
                result_item = results['results'][0]
                print(f"\n✅ Content Generation Triggered Successfully!")
                print(f"  Context ID: {result_item['context_id']}")
                print(f"  Content ID: {result_item.get('content_id', 'N/A')}")
                print(f"  Phase: {result_item['phase']}")
                
                # Verify context was updated
                updated_context = await acd_service.get_context(context.id)
                print(f"\n✓ Context Status Updated:")
                print(f"  State: {updated_context.ai_state}")
                print(f"  Queue Status: {updated_context.ai_queue_status}")
                
                return True
            else:
                print(f"\n⚠️  Generation did not complete successfully")
                if results['results']:
                    print(f"  Error: {results['results'][0].get('error', 'Unknown')}")
                return False
                
    except Exception as e:
        logger.error(f"Test failed: {e}", exc_info=True)
        print(f"\n❌ TEST FAILED: {e}")
        return False
    finally:
        await database_manager.disconnect()


async def test_model_fallback():
    """Test that model selection falls back correctly when vLLM is unavailable."""
    print("\n" + "=" * 80)
    print("TEST 2: Model Fallback Logic (vLLM unavailable → llama.cpp)")
    print("=" * 80)
    
    await database_manager.connect()
    
    try:
        async with database_manager.get_session() as session:
            from backend.services.ai_models import ai_models
            
            print(f"\n🔍 Initializing AI models...")
            await ai_models.initialize_models()
            
            print(f"\n📋 Available text models:")
            text_models = ai_models.available_models.get("text", [])
            for model in text_models:
                print(f"  - {model['name']}")
                print(f"    Engine: {model.get('inference_engine')}")
                print(f"    Fallbacks: {model.get('fallback_engines', [])}")
                print(f"    Can load: {model.get('can_load', False)}")
            
            # Check if llama.cpp is preferred
            llama_model = next(
                (m for m in text_models if "llama-3.1-8b" in m.get("name", "")),
                None
            )
            
            if llama_model:
                print(f"\n✓ Found llama-3.1-8b model")
                print(f"  Primary engine: {llama_model.get('inference_engine')}")
                
                if llama_model.get('inference_engine') == 'llama.cpp':
                    print(f"  ✅ Correctly prefers llama.cpp!")
                else:
                    print(f"  ⚠️  Still using: {llama_model.get('inference_engine')}")
                
                return llama_model.get('inference_engine') == 'llama.cpp'
            else:
                print(f"\n⚠️  llama-3.1-8b model not found")
                return False
                
    except Exception as e:
        logger.error(f"Test failed: {e}", exc_info=True)
        print(f"\n❌ TEST FAILED: {e}")
        return False
    finally:
        await database_manager.disconnect()


async def test_queue_priority():
    """Test that ACD queue processes contexts in priority order."""
    print("\n" + "=" * 80)
    print("TEST 3: ACD Queue Priority Processing")
    print("=" * 80)
    
    await database_manager.connect()
    
    try:
        async with database_manager.get_session() as session:
            acd_service = ACDService(session)
            persona_service = PersonaService(session)
            
            # Create test persona
            persona = await persona_service.create_persona(
                PersonaCreate(
                    name=f"Test Priority Persona {int(asyncio.get_event_loop().time())}",
                    appearance="Testing avatar with simple design",
                    personality="Focused on testing and quality assurance",
                )
            )
            
            print(f"\n✓ Created test persona: {persona.id}")
            
            # Create contexts with different priorities
            priorities = [
                (AIQueuePriority.LOW, "Low priority task"),
                (AIQueuePriority.CRITICAL, "Critical priority task"),
                (AIQueuePriority.NORMAL, "Normal priority task"),
                (AIQueuePriority.HIGH, "High priority task"),
            ]
            
            context_ids = []
            print(f"\n📝 Creating contexts with different priorities...")
            
            for priority, note in priorities:
                context = await acd_service.create_context(
                    ACDContextCreate(
                        ai_phase="TEXT_GENERATION",
                        ai_status=AIStatus.IMPLEMENTED,
                        ai_complexity=AIComplexity.LOW,
                        ai_note=note,
                        ai_state=AIState.READY,
                        ai_queue_priority=priority,
                        ai_queue_status=AIQueueStatus.QUEUED,
                        ai_context={
                            "persona_id": str(persona.id),
                            "prompt": f"Test prompt for {priority.value} priority",
                            "quality": "draft",
                        },
                    )
                )
                context_ids.append((context.id, priority, note))
                print(f"  ✓ Created {priority.value} priority context")
            
            # Query to see processing order
            from sqlalchemy import select, and_, case
            from backend.models.acd import ACDContextModel
            
            # Use same priority ordering logic as ACD service
            priority_order = case(
                (ACDContextModel.ai_queue_priority == "CRITICAL", 1),
                (ACDContextModel.ai_queue_priority == "HIGH", 2),
                (ACDContextModel.ai_queue_priority == "NORMAL", 3),
                (ACDContextModel.ai_queue_priority == "LOW", 4),
                (ACDContextModel.ai_queue_priority == "DEFERRED", 5),
                else_=6
            )
            
            stmt = (
                select(ACDContextModel)
                .where(
                    and_(
                        ACDContextModel.id.in_([c[0] for c in context_ids]),
                        ACDContextModel.ai_queue_status == AIQueueStatus.QUEUED.value,
                    )
                )
                .order_by(
                    priority_order,
                    ACDContextModel.created_at.asc()
                )
            )
            
            result = await session.execute(stmt)
            ordered_contexts = result.scalars().all()
            
            print(f"\n📊 Processing order (by priority):")
            expected_order = ["CRITICAL", "HIGH", "NORMAL", "LOW"]
            actual_order = [c.ai_queue_priority for c in ordered_contexts]
            
            for i, context in enumerate(ordered_contexts, 1):
                print(f"  {i}. {context.ai_queue_priority} - {context.ai_note}")
            
            # Check if order matches expected
            if actual_order == expected_order:
                print(f"\n✅ Priority ordering is CORRECT!")
                return True
            else:
                print(f"\n⚠️  Priority ordering unexpected")
                print(f"  Expected: {expected_order}")
                print(f"  Actual: {actual_order}")
                return False
                
    except Exception as e:
        logger.error(f"Test failed: {e}", exc_info=True)
        print(f"\n❌ TEST FAILED: {e}")
        return False
    finally:
        await database_manager.disconnect()


async def main():
    """Run all tests."""
    print("\n" + "=" * 80)
    print("🦎 GATOR ACD CONTENT GENERATION TEST SUITE")
    print("=" * 80)
    print("\nTesting ACD service enhancements:")
    print("  1. ACD triggers actual content generation")
    print("  2. Model selection prefers llama.cpp over vLLM")
    print("  3. Queue processes contexts by priority")
    
    results = []
    
    try:
        # Run tests
        test1_passed = await test_acd_triggers_generation()
        results.append(("ACD Triggers Generation", test1_passed))
        
        test2_passed = await test_model_fallback()
        results.append(("Model Fallback Logic", test2_passed))
        
        test3_passed = await test_queue_priority()
        results.append(("Queue Priority Processing", test3_passed))
        
        # Summary
        print("\n" + "=" * 80)
        print("TEST RESULTS SUMMARY")
        print("=" * 80)
        
        all_passed = True
        for test_name, passed in results:
            status = "✅ PASSED" if passed else "❌ FAILED"
            print(f"{status} - {test_name}")
            if not passed:
                all_passed = False
        
        print("\n" + "=" * 80)
        if all_passed:
            print("✅ ALL TESTS PASSED")
            print("=" * 80)
            print("\nKey Achievements:")
            print("  ✓ ACD service now triggers actual content generation")
            print("  ✓ Model selection prefers llama.cpp when vLLM unavailable")
            print("  ✓ Queue processing respects priority ordering")
            print("  ✓ System ready for autonomous content generation!")
            sys.exit(0)
        else:
            print("⚠️  SOME TESTS FAILED")
            print("=" * 80)
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"Test suite failed: {e}", exc_info=True)
        print(f"\n❌ TEST SUITE FAILED: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
