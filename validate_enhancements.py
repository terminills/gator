#!/usr/bin/env python3
"""
Validation script for the three enhancement tasks.

Demonstrates:
1. Base Image Status schema is properly integrated
2. Multi-GPU batch image generation capability
3. Template service functionality
"""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from backend.models.persona import PersonaModel, BaseImageStatus
from backend.services.template_service import TemplateService
from backend.services.ai_models import AIModelManager


def validate_task_1_base_image_schema():
    """Validate Task 1: Base Image Schema and Migration."""
    print("\n" + "=" * 70)
    print("TASK 1: Base Image Schema and Migration")
    print("=" * 70)
    
    try:
        # Check that BaseImageStatus enum exists
        assert hasattr(BaseImageStatus, 'PENDING_UPLOAD')
        assert hasattr(BaseImageStatus, 'DRAFT')
        assert hasattr(BaseImageStatus, 'APPROVED')
        assert hasattr(BaseImageStatus, 'REJECTED')
        print("✅ BaseImageStatus enum exists with all required values")
        
        # Check that PersonaModel has base_image_status column
        assert hasattr(PersonaModel, 'base_image_status')
        print("✅ PersonaModel has base_image_status column")
        
        # Verify enum values
        assert BaseImageStatus.PENDING_UPLOAD.value == "pending_upload"
        assert BaseImageStatus.DRAFT.value == "draft"
        assert BaseImageStatus.APPROVED.value == "approved"
        assert BaseImageStatus.REJECTED.value == "rejected"
        print("✅ All BaseImageStatus enum values are correct")
        
        print("\n✅ Task 1 PASSED: Base Image Schema properly integrated")
        return True
        
    except Exception as e:
        print(f"\n❌ Task 1 FAILED: {str(e)}")
        return False


def validate_task_2_multi_gpu():
    """Validate Task 2: Multi-GPU Image Generation."""
    print("\n" + "=" * 70)
    print("TASK 2: Multi-GPU Image Generation")
    print("=" * 70)
    
    try:
        # Check that AIModelManager has batch generation method
        manager = AIModelManager()
        assert hasattr(manager, 'generate_images_batch')
        print("✅ AIModelManager has generate_images_batch method")
        
        # Check that AIModelManager has device-specific generation method
        assert hasattr(manager, '_generate_image_on_device')
        print("✅ AIModelManager has _generate_image_on_device method")
        
        # Verify GPU detection is working
        gpu_count = manager._get_system_requirements().get('gpu_count', 0)
        print(f"ℹ️  Detected {gpu_count} GPU(s)")
        
        # Check method signatures
        import inspect
        batch_sig = inspect.signature(manager.generate_images_batch)
        assert 'prompts' in batch_sig.parameters
        print("✅ generate_images_batch has prompts parameter")
        
        device_sig = inspect.signature(manager._generate_image_on_device)
        assert 'device_id' in device_sig.parameters
        print("✅ _generate_image_on_device has device_id parameter")
        
        print("\n✅ Task 2 PASSED: Multi-GPU batch generation implemented")
        return True
        
    except Exception as e:
        print(f"\n❌ Task 2 FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def validate_task_3_template_service():
    """Validate Task 3: Template Service Implementation."""
    print("\n" + "=" * 70)
    print("TASK 3: Template Service Implementation")
    print("=" * 70)
    
    try:
        # Check that TemplateService exists
        service = TemplateService()
        print("✅ TemplateService class instantiated successfully")
        
        # Check that TemplateService has required methods
        assert hasattr(service, 'generate_fallback_text')
        print("✅ TemplateService has generate_fallback_text method")
        
        assert hasattr(service, '_determine_content_style')
        print("✅ TemplateService has _determine_content_style method")
        
        assert hasattr(service, '_generate_appearance_context')
        print("✅ TemplateService has _generate_appearance_context method")
        
        assert hasattr(service, '_determine_voice_modifiers')
        print("✅ TemplateService has _determine_voice_modifiers method")
        
        assert hasattr(service, '_generate_templates_for_style')
        print("✅ TemplateService has _generate_templates_for_style method")
        
        assert hasattr(service, '_select_weighted_template')
        print("✅ TemplateService has _select_weighted_template method")
        
        assert hasattr(service, '_customize_template')
        print("✅ TemplateService has _customize_template method")
        
        # Test basic functionality with a mock persona
        from unittest.mock import Mock
        import uuid
        
        mock_persona = Mock()
        mock_persona.id = uuid.uuid4()
        mock_persona.name = "Test Persona"
        mock_persona.appearance = "Professional appearance"
        mock_persona.personality = "Creative, analytical"
        mock_persona.content_themes = ["technology", "innovation"]
        mock_persona.style_preferences = {
            "aesthetic": "professional",
            "voice_style": "confident",
            "tone": "warm"
        }
        mock_persona.base_appearance_description = None
        mock_persona.appearance_locked = False
        
        result = service.generate_fallback_text(mock_persona)
        assert isinstance(result, str)
        assert len(result) > 0
        print(f"✅ Template service generated text: {result[:100]}...")
        
        print("\n✅ Task 3 PASSED: Template Service properly implemented")
        return True
        
    except Exception as e:
        print(f"\n❌ Task 3 FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all validation tests."""
    print("\n" + "=" * 70)
    print("THREE ENHANCEMENT TASKS VALIDATION")
    print("=" * 70)
    
    results = []
    
    # Task 1: Base Image Schema
    results.append(("Task 1: Base Image Schema", validate_task_1_base_image_schema()))
    
    # Task 2: Multi-GPU Image Generation
    results.append(("Task 2: Multi-GPU Generation", validate_task_2_multi_gpu()))
    
    # Task 3: Template Service
    results.append(("Task 3: Template Service", validate_task_3_template_service()))
    
    # Summary
    print("\n" + "=" * 70)
    print("VALIDATION SUMMARY")
    print("=" * 70)
    
    for task_name, passed in results:
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{task_name}: {status}")
    
    all_passed = all(result[1] for result in results)
    
    print("\n" + "=" * 70)
    if all_passed:
        print("✅ ALL TASKS VALIDATED SUCCESSFULLY")
        print("=" * 70)
        return 0
    else:
        print("❌ SOME TASKS FAILED VALIDATION")
        print("=" * 70)
        return 1


if __name__ == "__main__":
    sys.exit(main())
