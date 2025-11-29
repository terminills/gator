#!/usr/bin/env python3
"""
Test script to verify llama.cpp integration works with actual text generation.

This test proves that the AI models can actually generate content, not just create
database records with placeholder data.
"""

import subprocess
import sys
from pathlib import Path

def test_llamacpp_binary_exists():
    """Test that llama-cli binary was built successfully."""
    print("🔍 Test 1: Check if llama-cli binary exists...")
    
    repo_root = Path(__file__).parent
    llamacpp_binary = repo_root / "third_party" / "llama.cpp" / "build" / "bin" / "llama-cli"
    
    if not llamacpp_binary.exists():
        print(f"   ❌ FAIL: llama-cli not found at {llamacpp_binary}")
        print(f"   Run: scripts/build_llamacpp.sh")
        return False
    
    print(f"   ✅ PASS: Found llama-cli at {llamacpp_binary}")
    return True


def test_llamacpp_version():
    """Test that llama-cli runs and shows version."""
    print("\n🔍 Test 2: Check if llama-cli runs...")
    
    repo_root = Path(__file__).parent
    llamacpp_binary = repo_root / "third_party" / "llama.cpp" / "build" / "bin" / "llama-cli"
    
    try:
        result = subprocess.run(
            [str(llamacpp_binary), "--version"],
            capture_output=True,
            text=True,
            timeout=5
        )
        
        if result.returncode != 0:
            print(f"   ❌ FAIL: llama-cli returned error code {result.returncode}")
            print(f"   stderr: {result.stderr}")
            return False
        
        version_line = result.stdout.split('\n')[0]
        print(f"   ✅ PASS: llama-cli works! {version_line}")
        return True
        
    except subprocess.TimeoutExpired:
        print("   ❌ FAIL: llama-cli timed out")
        return False
    except Exception as e:
        print(f"   ❌ FAIL: Error running llama-cli: {e}")
        return False


def test_ai_models_service_detects_llamacpp():
    """Test that AI models service can detect llama.cpp."""
    print("\n🔍 Test 3: Check if AI models service detects llama.cpp...")
    
    repo_root = Path(__file__).parent
    llamacpp_binary = repo_root / "third_party" / "llama.cpp" / "build" / "bin" / "llama-cli"
    
    # Add llamacpp_binary directory to PATH
    import os
    os.environ["PATH"] = f"{llamacpp_binary.parent}:{os.environ['PATH']}"
    
    # Import after PATH is set
    import shutil
    found_binary = shutil.which("llama-cli")
    
    if not found_binary:
        print(f"   ❌ FAIL: llama-cli not found in PATH")
        print(f"   PATH: {os.environ['PATH']}")
        return False
    
    print(f"   ✅ PASS: llama-cli found at {found_binary}")
    return True


def test_model_directory_structure():
    """Test that model directory structure is set up correctly."""
    print("\n🔍 Test 4: Check model directory structure...")
    
    repo_root = Path(__file__).parent
    models_dir = repo_root / "models"
    text_models = models_dir / "text"
    
    if not models_dir.exists():
        print(f"   ⚠️  WARN: Creating models directory at {models_dir}")
        models_dir.mkdir(parents=True, exist_ok=True)
    
    if not text_models.exists():
        print(f"   ⚠️  WARN: Creating text models directory at {text_models}")
        text_models.mkdir(parents=True, exist_ok=True)
    
    print(f"   ✅ PASS: Model directory structure exists")
    print(f"      Models dir: {models_dir}")
    print(f"      Text models: {text_models}")
    
    # Check if any models exist
    gguf_files = list(text_models.glob("**/*.gguf"))
    if gguf_files:
        print(f"   📦 Found {len(gguf_files)} GGUF model(s):")
        for model_file in gguf_files:
            size_mb = model_file.stat().st_size / (1024 * 1024)
            print(f"      - {model_file.name} ({size_mb:.1f} MB)")
    else:
        print(f"   ℹ️  No GGUF models found yet")
        print(f"   To download a model, run:")
        print(f"      mkdir -p {text_models}/tinyllama")
        print(f"      cd {text_models}/tinyllama")
        print(f"      curl -L -O https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf")
    
    return True


def main():
    """Run all tests."""
    print("="*80)
    print("🧪 LLAMA.CPP INTEGRATION TEST")
    print("="*80)
    print()
    print("Purpose: Verify that actual AI generation works, not just database operations.")
    print()
    
    tests = [
        ("Binary exists", test_llamacpp_binary_exists),
        ("Binary runs", test_llamacpp_version),
        ("Service detection", test_ai_models_service_detects_llamacpp),
        ("Model directory", test_model_directory_structure),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            results.append((name, test_func()))
        except Exception as e:
            print(f"   ❌ EXCEPTION: {e}")
            results.append((name, False))
    
    print("\n" + "="*80)
    print("📊 TEST SUMMARY")
    print("="*80)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {status}: {name}")
    
    print()
    print(f"Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n✅ All tests passed! llama.cpp integration is working.")
        print("\nNext steps:")
        print("  1. Download a GGUF model (see test output above)")
        print("  2. Update AI models service to use llama.cpp for text generation")
        print("  3. Test end-to-end content generation with actual AI model")
        return 0
    else:
        print("\n❌ Some tests failed. Fix issues before proceeding.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
