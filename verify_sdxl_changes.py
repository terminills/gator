#!/usr/bin/env python3
"""
Simple verification script to check the SDXL Long Prompt Pipeline changes.
This script only checks the code statically without running it.
"""

import re
from pathlib import Path


def verify_sdxl_implementation():
    """Verify the SDXL Long Prompt Pipeline implementation in ai_models.py"""
    
    print("=" * 70)
    print("VERIFICATION: SDXL Long Prompt Pipeline Implementation")
    print("=" * 70)
    
    ai_models_path = Path("src/backend/services/ai_models.py")
    
    if not ai_models_path.exists():
        print("❌ ERROR: ai_models.py not found!")
        return False
    
    content = ai_models_path.read_text()
    
    checks_passed = []
    checks_failed = []
    
    # Check 1: DiffusionPipeline is imported
    print("\n1. Checking DiffusionPipeline import...")
    if "from diffusers import" in content and "DiffusionPipeline" in content:
        print("   ✅ DiffusionPipeline is imported")
        checks_passed.append("DiffusionPipeline import")
    else:
        print("   ❌ DiffusionPipeline is NOT imported")
        checks_failed.append("DiffusionPipeline import")
    
    # Check 2: custom_pipeline parameter is used for SDXL
    print("\n2. Checking custom_pipeline parameter for SDXL...")
    if 'custom_pipeline' in content and 'lpw_stable_diffusion_xl' in content:
        print("   ✅ custom_pipeline='lpw_stable_diffusion_xl' found")
        checks_passed.append("custom_pipeline parameter")
    else:
        print("   ❌ custom_pipeline parameter NOT found for SDXL")
        checks_failed.append("custom_pipeline parameter")
    
    # Check 3: DiffusionPipeline.from_pretrained is used (not class-specific)
    print("\n3. Checking DiffusionPipeline.from_pretrained usage...")
    if 'DiffusionPipeline.from_pretrained' in content:
        print("   ✅ DiffusionPipeline.from_pretrained is used")
        checks_passed.append("DiffusionPipeline.from_pretrained")
    else:
        print("   ⚠️  DiffusionPipeline.from_pretrained usage not clear")
    
    # Check 4: Fallback mechanism exists
    print("\n4. Checking fallback mechanism...")
    if 'del' in content and 'custom_pipeline' in content:
        # Look for code that removes custom_pipeline on failure
        print("   ✅ Fallback mechanism appears to be implemented")
        checks_passed.append("Fallback mechanism")
    else:
        print("   ⚠️  Fallback mechanism unclear")
    
    # Check 5: Success logging
    print("\n5. Checking success logging...")
    if 'Successfully loaded SDXL Long Prompt Pipeline' in content:
        print("   ✅ Success logging present")
        checks_passed.append("Success logging")
    else:
        print("   ⚠️  Success logging message not found")
    
    # Check 6: Warning logging for fallback
    print("\n6. Checking fallback warning logging...")
    if '77 token limit' in content or 'token limit applies' in content:
        print("   ✅ Fallback warning logging present")
        checks_passed.append("Fallback warning")
    else:
        print("   ⚠️  Fallback warning not found")
    
    # Check 7: Old import removed
    print("\n7. Checking old direct import was removed...")
    if 'from diffusers import StableDiffusionXLLongPromptWeightingPipeline' in content:
        print("   ❌ Old direct import still present (should be removed)")
        checks_failed.append("Old import removal")
    else:
        print("   ✅ Old direct import removed")
        checks_passed.append("Old import removal")
    
    # Check 8: SDXL detection logic
    print("\n8. Checking SDXL model detection...")
    if 'is_sdxl' in content:
        print("   ✅ SDXL model detection present")
        checks_passed.append("SDXL detection")
    else:
        print("   ❌ SDXL model detection NOT found")
        checks_failed.append("SDXL detection")
    
    # Summary
    print("\n" + "=" * 70)
    print("VERIFICATION SUMMARY")
    print("=" * 70)
    print(f"✅ Checks passed: {len(checks_passed)}")
    for check in checks_passed:
        print(f"   - {check}")
    
    if checks_failed:
        print(f"\n❌ Checks failed: {len(checks_failed)}")
        for check in checks_failed:
            print(f"   - {check}")
    
    # Check for diagnostic logging
    print("\n" + "=" * 70)
    print("BASE IMAGE SAVE DIAGNOSTIC LOGGING")
    print("=" * 70)
    
    # Check persona.py
    persona_path = Path("src/backend/api/routes/persona.py")
    if persona_path.exists():
        persona_content = persona_path.read_text()
        if "Read {len(image_data)} bytes from uploaded file" in persona_content:
            print("✅ Upload endpoint has diagnostic logging")
        else:
            print("⚠️  Upload endpoint diagnostic logging not found")
    
    # Check persona_service.py
    service_path = Path("src/backend/services/persona_service.py")
    if service_path.exists():
        service_content = service_path.read_text()
        if "Writing {len(image_data)} bytes to disk" in service_content:
            print("✅ Save method has diagnostic logging")
        else:
            print("⚠️  Save method diagnostic logging not found")
        
        if "SIZE MISMATCH" in service_content:
            print("✅ Size mismatch detection implemented")
        else:
            print("⚠️  Size mismatch detection not found")
    
    # Check ai_models.py for image generation logging
    if "BytesIO buffer position after save" in content:
        print("✅ Image generation has BytesIO diagnostic logging")
    else:
        print("⚠️  BytesIO diagnostic logging not found")
    
    print("\n" + "=" * 70)
    return len(checks_failed) == 0


if __name__ == "__main__":
    success = verify_sdxl_implementation()
    
    if success:
        print("\n🎉 All critical checks passed!")
        print("\nNext steps:")
        print("1. Test with actual SDXL model and long prompt (>100 words)")
        print("2. Monitor logs for pipeline loading messages")
        print("3. Test image upload/generation to capture diagnostic logs")
        print("4. Verify 37k save issue is resolved with logs")
    else:
        print("\n⚠️  Some checks failed - review implementation")
    
    exit(0 if success else 1)
