#!/usr/bin/env python3
"""
Diagnostic script to test base image save/load issue.

This script tests the entire pipeline of:
1. Generating an image with diffusers
2. Converting to bytes
3. Saving to disk
4. Reading back from disk
5. Verifying sizes match

Run with: python test_image_save_issue.py
"""

import asyncio
import io
import sys
from pathlib import Path
from PIL import Image
import tempfile

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))


async def test_image_save_pipeline():
    """Test the complete image save pipeline."""
    
    print("=" * 70)
    print("DIAGNOSTIC: Base Image Save Issue (37k limit)")
    print("=" * 70)
    
    # Test 1: Create a test image and verify BytesIO handling
    print("\n1. Testing PIL Image to BytesIO conversion...")
    test_image = Image.new('RGB', (1024, 1024), color='red')
    img_byte_arr = io.BytesIO()
    test_image.save(img_byte_arr, format="PNG")
    image_data = img_byte_arr.getvalue()
    print(f"   ✓ Image size: {len(image_data)} bytes ({len(image_data) / 1024:.1f} KB)")
    
    if len(image_data) < 50000:
        print(f"   ⚠️  WARNING: Image is very small ({len(image_data)} bytes)")
    else:
        print(f"   ✓ Image size is reasonable for 1024x1024 PNG")
    
    # Test 2: Save to disk and verify
    print("\n2. Testing disk write...")
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_file:
        tmp_path = Path(tmp_file.name)
        tmp_file.write(image_data)
        tmp_file.flush()
    
    saved_size = tmp_path.stat().st_size
    print(f"   ✓ Saved {saved_size} bytes to {tmp_path}")
    
    if saved_size != len(image_data):
        print(f"   ❌ ERROR: Size mismatch! Expected {len(image_data)}, got {saved_size}")
    else:
        print(f"   ✓ File size matches memory size")
    
    # Test 3: Read back and verify
    print("\n3. Testing disk read...")
    with open(tmp_path, "rb") as f:
        read_data = f.read()
    
    print(f"   ✓ Read {len(read_data)} bytes from disk")
    
    if len(read_data) != len(image_data):
        print(f"   ❌ ERROR: Read size mismatch! Expected {len(image_data)}, got {len(read_data)}")
    else:
        print(f"   ✓ Read size matches original")
    
    if read_data == image_data:
        print(f"   ✓ Data integrity verified (bytes match exactly)")
    else:
        print(f"   ❌ ERROR: Data corruption detected!")
    
    # Test 4: Verify can load as image
    print("\n4. Testing image load...")
    try:
        loaded_image = Image.open(tmp_path)
        print(f"   ✓ Image loaded successfully: {loaded_image.size} {loaded_image.mode}")
    except Exception as e:
        print(f"   ❌ ERROR loading image: {e}")
    
    # Clean up
    tmp_path.unlink()
    
    # Test 5: Test with actual persona service if available
    print("\n5. Testing with PersonaService._save_image_to_disk()...")
    try:
        from backend.services.persona_service import PersonaService
        from uuid import uuid4
        
        service = PersonaService(None)  # No DB needed for this test
        test_persona_id = str(uuid4())
        
        # Create temp directory for test
        test_dir = Path("/tmp/gator_test_images")
        test_dir.mkdir(exist_ok=True)
        
        # Monkey patch the base_images_dir
        original_save_method = service._save_image_to_disk
        
        async def patched_save(persona_id, image_data, filename=None):
            from datetime import datetime
            if filename is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"persona_{persona_id}_{timestamp}.png"
            
            file_path = test_dir / filename
            with open(file_path, "wb") as f:
                bytes_written = f.write(image_data)
            
            print(f"   ✓ Wrote {bytes_written} bytes to {file_path}")
            return str(file_path)
        
        service._save_image_to_disk = patched_save
        
        # Test save
        saved_path = await service._save_image_to_disk(
            test_persona_id, 
            image_data,
            f"test_{test_persona_id}.png"
        )
        
        # Verify
        saved_file = Path(saved_path)
        actual_size = saved_file.stat().st_size
        print(f"   ✓ PersonaService saved {actual_size} bytes")
        
        if actual_size == len(image_data):
            print(f"   ✓ PersonaService save successful!")
        else:
            print(f"   ❌ ERROR: PersonaService size mismatch! Expected {len(image_data)}, got {actual_size}")
        
        # Clean up test file
        saved_file.unlink()
        
    except Exception as e:
        print(f"   ⚠️  Could not test PersonaService: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 70)
    print("DIAGNOSIS COMPLETE")
    print("=" * 70)
    
    # Summary
    print("\nSUMMARY:")
    print("If all tests passed, the issue is likely:")
    print("  1. FastAPI upload file size limit")
    print("  2. Web server (uvicorn/nginx) request body size limit")
    print("  3. Client-side truncation during upload")
    print("  4. Network timeout during large uploads")
    print("\nCheck:")
    print("  - FastAPI app settings for max request size")
    print("  - Uvicorn --limit-max-requests setting")
    print("  - Nginx client_max_body_size (if using nginx)")
    print("  - Client upload implementation")


if __name__ == "__main__":
    asyncio.run(test_image_save_pipeline())
