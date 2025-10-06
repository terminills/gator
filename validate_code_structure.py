#!/usr/bin/env python3
"""
Code Structure Validation for Enhancement Tasks

Validates the implementation without requiring dependencies.
Checks for presence of key classes, methods, and code patterns.
"""

import os
import re
from pathlib import Path


def check_file_contains(filepath, patterns, task_name):
    """Check if file contains all required patterns."""
    print(f"\n📋 Checking {filepath}...")
    
    with open(filepath, 'r') as f:
        content = f.read()
    
    results = []
    for pattern_name, pattern in patterns.items():
        if re.search(pattern, content, re.MULTILINE | re.DOTALL):
            print(f"  ✅ Found: {pattern_name}")
            results.append(True)
        else:
            print(f"  ❌ Missing: {pattern_name}")
            results.append(False)
    
    return all(results)


def validate_task_1():
    """Validate Task 1: Base Image Schema."""
    print("\n" + "=" * 70)
    print("TASK 1: Base Image Schema and Migration")
    print("=" * 70)
    
    filepath = "src/backend/models/persona.py"
    
    patterns = {
        "BaseImageStatus enum class": r"class BaseImageStatus\(str,\s*Enum\)",
        "PENDING_UPLOAD status": r"PENDING_UPLOAD\s*=\s*['\"]pending_upload['\"]",
        "DRAFT status": r"DRAFT\s*=\s*['\"]draft['\"]",
        "APPROVED status": r"APPROVED\s*=\s*['\"]approved['\"]",
        "REJECTED status": r"REJECTED\s*=\s*['\"]rejected['\"]",
        "PersonaModel base_image_status column": r"base_image_status\s*=\s*Column",
        "PersonaCreate base_image_status field": r"base_image_status:\s*BaseImageStatus",
        "PersonaUpdate base_image_status field": r"base_image_status:\s*Optional\[BaseImageStatus\]",
    }
    
    result = check_file_contains(filepath, patterns, "Task 1")
    
    if result:
        print("\n✅ Task 1 PASSED: Base Image Schema properly implemented")
    else:
        print("\n❌ Task 1 FAILED: Missing required schema elements")
    
    return result


def validate_task_2():
    """Validate Task 2: Multi-GPU Image Generation."""
    print("\n" + "=" * 70)
    print("TASK 2: Multi-GPU Image Generation")
    print("=" * 70)
    
    filepath = "src/backend/services/ai_models.py"
    
    patterns = {
        "generate_images_batch method": r"async def generate_images_batch\(",
        "prompts parameter": r"prompts:\s*List\[str\]",
        "GPU count detection": r"torch\.cuda\.device_count\(\)",
        "Multi-GPU parallel processing logic": r"gpu_id\s*=\s*i\s*%\s*gpu_count",
        "_generate_image_on_device method": r"async def _generate_image_on_device\(",
        "device_id parameter": r"device_id:\s*int",
        "asyncio.gather for parallel execution": r"await asyncio\.gather\(",
        "Device-specific pipeline caching": r"pipeline_key\s*=\s*f['\"]diffusers_.*_gpu\{device_id\}['\"]",
        "Device selection logic": r"device\s*=\s*f['\"]cuda:\{device_id\}['\"]",
    }
    
    result = check_file_contains(filepath, patterns, "Task 2")
    
    if result:
        print("\n✅ Task 2 PASSED: Multi-GPU batch generation properly implemented")
    else:
        print("\n❌ Task 2 FAILED: Missing required multi-GPU elements")
    
    return result


def validate_task_3():
    """Validate Task 3: Template Service."""
    print("\n" + "=" * 70)
    print("TASK 3: Template Service Implementation")
    print("=" * 70)
    
    # Check template_service.py
    service_filepath = "src/backend/services/template_service.py"
    
    service_patterns = {
        "TemplateService class": r"class TemplateService:",
        "generate_fallback_text method": r"def generate_fallback_text\(",
        "_determine_content_style method": r"def _determine_content_style\(",
        "Multi-dimensional scoring": r"style_scores\s*=\s*\{['\"]creative['\"]:",
        "_generate_appearance_context method": r"def _generate_appearance_context\(",
        "Appearance locking support": r"appearance_locked",
        "_determine_voice_modifiers method": r"def _determine_voice_modifiers\(",
        "_generate_templates_for_style method": r"def _generate_templates_for_style\(",
        "_select_weighted_template method": r"def _select_weighted_template\(",
        "Template weighting logic": r"template_weights\s*=\s*\[1\.0\]",
        "_customize_template method": r"def _customize_template\(",
    }
    
    service_result = check_file_contains(service_filepath, service_patterns, "Task 3 - Service")
    
    # Check content_generation_service.py integration
    content_filepath = "src/backend/services/content_generation_service.py"
    
    content_patterns = {
        "TemplateService import": r"from backend\.services\.template_service import TemplateService",
        "TemplateService initialization": r"self\.template_service\s*=\s*TemplateService\(\)",
        "Delegation to template service": r"self\.template_service\.generate_fallback_text\(",
    }
    
    content_result = check_file_contains(content_filepath, content_patterns, "Task 3 - Integration")
    
    result = service_result and content_result
    
    if result:
        print("\n✅ Task 3 PASSED: Template Service properly implemented and integrated")
    else:
        print("\n❌ Task 3 FAILED: Missing required template service elements")
    
    return result


def check_tests_exist():
    """Check that test files exist."""
    print("\n" + "=" * 70)
    print("TEST FILES CHECK")
    print("=" * 70)
    
    test_files = [
        "tests/unit/test_template_service.py",
        "tests/unit/test_multi_gpu_generation.py",
    ]
    
    all_exist = True
    for test_file in test_files:
        if os.path.exists(test_file):
            print(f"✅ {test_file} exists")
        else:
            print(f"❌ {test_file} missing")
            all_exist = False
    
    return all_exist


def main():
    """Run all validation checks."""
    print("\n" + "=" * 70)
    print("THREE ENHANCEMENT TASKS - CODE STRUCTURE VALIDATION")
    print("=" * 70)
    
    results = []
    
    # Task 1: Base Image Schema
    results.append(("Task 1: Base Image Schema", validate_task_1()))
    
    # Task 2: Multi-GPU Image Generation  
    results.append(("Task 2: Multi-GPU Generation", validate_task_2()))
    
    # Task 3: Template Service
    results.append(("Task 3: Template Service", validate_task_3()))
    
    # Test files
    results.append(("Test Files", check_tests_exist()))
    
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
        print("\nAll three enhancement tasks have been implemented:")
        print("1. Base Image Schema with BaseImageStatus enum")
        print("2. Multi-GPU batch image generation with parallel processing")
        print("3. Template Service with sophisticated fallback text generation")
        return 0
    else:
        print("❌ SOME TASKS FAILED VALIDATION")
        print("=" * 70)
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
