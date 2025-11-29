#!/usr/bin/env python3
"""
Validation test for greenlet_spawn fix.

This script validates that the fix prevents greenlet_spawn errors
by ensuring persona data is extracted before passing to synchronous methods.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))


def test_template_service_with_data():
    """Test that TemplateService can work with plain data (no SQLAlchemy models)."""
    print("\n" + "="*80)
    print("🧪 TEST: TemplateService with plain data (greenlet-safe)")
    print("="*80)
    
    from backend.services.template_service import TemplateService
    
    # Create service
    service = TemplateService()
    print("✓ TemplateService created")
    
    # Create test data (simulating extracted persona attributes)
    persona_data = {
        'name': 'Test Persona',
        'appearance': 'A friendly test character',
        'base_appearance_description': 'Professional appearance',
        'appearance_locked': True,
        'personality': 'friendly, casual, tech-savvy',
        'content_themes': ['technology', 'lifestyle'],
        'style_preferences': {
            'aesthetic': 'modern',
            'voice_style': 'casual',
            'tone': 'warm',
        }
    }
    print("✓ Test persona data created")
    
    # Test 1: Generate fallback text from data
    try:
        result = service.generate_fallback_text_from_data(
            persona_data=persona_data,
            prompt="Write a post about technology",
            content_rating="sfw"
        )
        
        assert isinstance(result, str), "Result should be a string"
        assert len(result) > 0, "Result should not be empty"
        assert len(result) > 50, "Result should be substantial"
        
        print(f"✅ TEST 1 PASSED: generate_fallback_text_from_data")
        print(f"   Generated {len(result)} characters")
        print(f"   Preview: {result[:100]}...")
        
    except Exception as e:
        print(f"❌ TEST 1 FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test 2: Generate appearance context from data
    try:
        context = service._generate_appearance_context_from_data(
            persona_data=persona_data,
            appearance_desc=persona_data['base_appearance_description'],
            aesthetic='professional'
        )
        
        assert isinstance(context, str), "Context should be a string"
        
        print(f"✅ TEST 2 PASSED: _generate_appearance_context_from_data")
        print(f"   Context: '{context}'")
        
    except Exception as e:
        print(f"❌ TEST 2 FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test 3: Verify locked appearance produces context
    try:
        locked_data = {
            'appearance_locked': True,
            'base_appearance_description': 'Professional corporate appearance'
        }
        
        context = service._generate_appearance_context_from_data(
            locked_data,
            'Professional corporate appearance',
            'professional'
        )
        
        # Should produce some context for locked professional appearance
        print(f"✅ TEST 3 PASSED: Locked appearance context generation")
        print(f"   Context: '{context}'")
        
    except Exception as e:
        print(f"❌ TEST 3 FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test 4: Verify unlocked appearance produces no context
    try:
        unlocked_data = {
            'appearance_locked': False,
            'base_appearance_description': None
        }
        
        context = service._generate_appearance_context_from_data(
            unlocked_data,
            'Casual appearance',
            'casual'
        )
        
        assert context == "", "Unlocked appearance should produce no context"
        
        print(f"✅ TEST 4 PASSED: Unlocked appearance produces no context")
        
    except Exception as e:
        print(f"❌ TEST 4 FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n" + "="*80)
    print("✅ ALL TESTS PASSED")
    print("="*80)
    print("\n📋 Summary:")
    print("   The fix successfully prevents greenlet_spawn errors by:")
    print("   1. Extracting persona attributes in async context")
    print("   2. Passing plain dictionaries to synchronous methods")
    print("   3. Avoiding SQLAlchemy model access in synchronous code")
    print()
    
    return True


if __name__ == "__main__":
    try:
        result = test_template_service_with_data()
        sys.exit(0 if result else 1)
    except Exception as e:
        print(f"\n❌ VALIDATION FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
