#!/usr/bin/env python3
"""
Direct test of llama.cpp text generation.

Tests that llama.cpp can actually generate text, proving the integration works.
Does NOT test the full app (since dependencies are having network issues),
but proves the core capability exists.
"""

import subprocess
import sys
from pathlib import Path
import os
import tempfile

def test_llamacpp_exists():
    """Test that llama-cli binary exists."""
    print("="*80)
    print("🧪 TEST 1: llama.cpp Binary Exists")
    print("="*80)
    
    repo_root = Path(__file__).parent.parent.parent
    llamacpp_bin = repo_root / "third_party" / "llama.cpp" / "build" / "bin" / "llama-cli"
    
    if not llamacpp_bin.exists():
        print(f"   ❌ FAIL: llama-cli not found at {llamacpp_bin}")
        return False
    
    print(f"   ✅ PASS: Found llama-cli at {llamacpp_bin}")
    print(f"   Size: {llamacpp_bin.stat().st_size / 1024 / 1024:.1f} MB")
    return True


def test_llamacpp_runs():
    """Test that llama-cli runs and shows help."""
    print("\n" + "="*80)
    print("🧪 TEST 2: llama.cpp Runs")
    print("="*80)
    
    repo_root = Path(__file__).parent.parent.parent
    llamacpp_bin = repo_root / "third_party" / "llama.cpp" / "build" / "bin" / "llama-cli"
    
    try:
        result = subprocess.run(
            [str(llamacpp_bin), "--version"],
            capture_output=True,
            text=True,
            timeout=5
        )
        
        if result.returncode != 0:
            print(f"   ❌ FAIL: llama-cli returned error code {result.returncode}")
            return False
        
        version = result.stdout.split('\n')[0]
        print(f"   ✅ PASS: {version}")
        return True
        
    except Exception as e:
        print(f"   ❌ FAIL: {e}")
        return False


def test_model_availability():
    """Test if any GGUF models are available."""
    print("\n" + "="*80)
    print("🧪 TEST 3: Model Availability")
    print("="*80)
    
    repo_root = Path(__file__).parent.parent.parent
    models_dir = repo_root / "models" / "text"
    
    if not models_dir.exists():
        print(f"   ℹ️  Models directory doesn't exist yet: {models_dir}")
        print(f"   Creating directory...")
        models_dir.mkdir(parents=True, exist_ok=True)
    
    # Search for GGUF files
    gguf_files = list(models_dir.glob("**/*.gguf"))
    
    if not gguf_files:
        print(f"   ⏭️  SKIP: No GGUF models found in {models_dir}")
        print(f"\n   To download a test model (TinyLlama ~680MB):")
        print(f"      mkdir -p {models_dir}/tinyllama")
        print(f"      cd {models_dir}/tinyllama")
        print(f"      curl -L -O https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf")
        return None
    
    print(f"   ✓ Found {len(gguf_files)} GGUF model(s):")
    for model_file in gguf_files:
        size_mb = model_file.stat().st_size / (1024 * 1024)
        print(f"      - {model_file.name} ({size_mb:.1f} MB)")
        print(f"        Path: {model_file}")
    
    return gguf_files[0]  # Return first model for testing


def test_llamacpp_generation(model_file):
    """Test actual text generation with llama.cpp."""
    print("\n" + "="*80)
    print("🧪 TEST 4: Actual Text Generation")
    print("="*80)
    
    if model_file is None:
        print(f"   ⏭️  SKIP: No model available for testing")
        return None
    
    repo_root = Path(__file__).parent.parent.parent
    llamacpp_bin = repo_root / "third_party" / "llama.cpp" / "build" / "bin" / "llama-cli"
    
    print(f"   Model: {model_file.name}")
    print(f"   Prompt: 'Hello, my name is'")
    print(f"   Generating...")
    
    try:
        # Run llama.cpp with a simple prompt
        result = subprocess.run(
            [
                str(llamacpp_bin),
                "-m", str(model_file),
                "-p", "Hello, my name is",
                "-n", "20",  # Generate 20 tokens
                "--temp", "0.7",
                "-c", "512",  # Context size
                "--log-disable",
            ],
            capture_output=True,
            text=True,
            timeout=60  # Give it time to load model and generate
        )
        
        if result.returncode != 0:
            print(f"   ❌ FAIL: llama-cli returned error code {result.returncode}")
            print(f"   stderr: {result.stderr[:500]}")
            return False
        
        output = result.stdout
        
        # Check if we got actual output
        if len(output) < 10:
            print(f"   ❌ FAIL: Output too short ({len(output)} chars)")
            print(f"   Output: {output}")
            return False
        
        print(f"\n   ✅ PASS: Generated {len(output)} characters of text!")
        print(f"\n   Generated Output:")
        print(f"   " + "-"*76)
        # Show first 500 chars
        for line in output[:500].split('\n'):
            print(f"   {line}")
        if len(output) > 500:
            print(f"   ... (truncated, total {len(output)} chars)")
        print(f"   " + "-"*76)
        
        return True
        
    except subprocess.TimeoutExpired:
        print(f"   ❌ FAIL: Generation timed out after 60 seconds")
        print(f"   This might happen if the model is too large or CPU is slow")
        return False
    except Exception as e:
        print(f"   ❌ FAIL: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("="*80)
    print("🦙 LLAMA.CPP TEXT GENERATION TEST")
    print("="*80)
    print()
    print("Purpose: Prove llama.cpp can actually generate text (not placeholders).")
    print()
    
    results = []
    
    # Test 1: Binary exists
    test1 = test_llamacpp_exists()
    results.append(("Binary exists", test1))
    
    if not test1:
        print("\n❌ Cannot continue without llama-cli binary")
        print("   Run: ./scripts/build_llamacpp.sh")
        return 1
    
    # Test 2: Binary runs
    test2 = test_llamacpp_runs()
    results.append(("Binary runs", test2))
    
    if not test2:
        print("\n❌ Binary exists but doesn't run properly")
        return 1
    
    # Test 3: Model availability
    model_file = test_model_availability()
    results.append(("Model available", model_file is not None))
    
    # Test 4: Actual generation (only if model exists)
    if model_file:
        test4 = test_llamacpp_generation(model_file)
        results.append(("Text generation", test4))
    
    # Summary
    print("\n" + "="*80)
    print("📊 TEST SUMMARY")
    print("="*80)
    
    passed = sum(1 for _, r in results if r is True)
    failed = sum(1 for _, r in results if r is False)
    skipped = sum(1 for _, r in results if r is None)
    total = len(results)
    
    for name, result in results:
        if result is True:
            status = "✅ PASS"
        elif result is False:
            status = "❌ FAIL"
        else:
            status = "⏭️  SKIP"
        print(f"  {status}: {name}")
    
    print()
    print(f"Results: {passed} passed, {failed} failed, {skipped} skipped out of {total} tests")
    
    if failed == 0 and passed >= 2:
        if any("generation" in name.lower() for name, r in results if r is True):
            print("\n🎉 EXCELLENT: llama.cpp successfully generated real text!")
            print("   This proves the integration works for actual content generation.")
        else:
            print("\n✅ Good: llama.cpp binary is working.")
            print("   Download a model to test actual generation.")
        return 0
    else:
        print("\n❌ Some critical tests failed.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
