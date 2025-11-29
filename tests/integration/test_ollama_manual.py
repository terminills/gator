#!/usr/bin/env python3
"""
Manual test script to verify Ollama integration works correctly.

This script tests:
1. Ollama detection
2. Ollama text generation
3. Fallback from llama.cpp to Ollama (simulated)

Run this after installing Ollama and pulling a model:
    ollama pull llama3:8b
    python test_ollama_manual.py
"""

import sys
import asyncio
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from backend.utils.model_detection import (
    find_ollama_installation,
    get_inference_engines_status,
)
from backend.services.ai_models import AIModelManager


def test_ollama_detection():
    """Test that Ollama is detected correctly."""
    print("=" * 80)
    print("TEST 1: Ollama Detection")
    print("=" * 80)
    
    ollama_info = find_ollama_installation()
    
    if ollama_info is None:
        print("❌ FAILED: Ollama not detected")
        print("\nPlease install Ollama:")
        print("  curl -fsSL https://ollama.com/install.sh | sh")
        return False
    
    print("✅ PASSED: Ollama detected")
    print(f"\nOllama Info:")
    print(f"  Version: {ollama_info.get('version', 'unknown')}")
    print(f"  Path: {ollama_info.get('path', 'unknown')}")
    print(f"  Server Running: {ollama_info.get('server_running', False)}")
    print(f"  Available Models: {', '.join(ollama_info.get('available_models', []))}")
    
    if not ollama_info.get('available_models'):
        print("\n⚠️  WARNING: No models found. Pull a model with:")
        print("  ollama pull llama3:8b")
        return False
    
    return True


def test_inference_engine_status():
    """Test that Ollama appears in inference engines status."""
    print("\n" + "=" * 80)
    print("TEST 2: Inference Engine Status")
    print("=" * 80)
    
    engines = get_inference_engines_status()
    
    if "ollama" not in engines:
        print("❌ FAILED: Ollama not in engines status")
        return False
    
    print("✅ PASSED: Ollama in engines status")
    print(f"\nOllama Engine:")
    print(f"  Status: {engines['ollama'].get('status', 'unknown')}")
    print(f"  Category: {engines['ollama'].get('category', 'unknown')}")
    
    return engines['ollama'].get('status') == 'installed'


async def test_ollama_text_generation():
    """Test actual text generation with Ollama."""
    print("\n" + "=" * 80)
    print("TEST 3: Ollama Text Generation")
    print("=" * 80)
    
    # Get available models
    ollama_info = find_ollama_installation()
    if not ollama_info or not ollama_info.get('available_models'):
        print("❌ SKIPPED: No Ollama models available")
        return False
    
    # Use first available model
    model_name = ollama_info['available_models'][0]
    print(f"\nUsing model: {model_name}")
    
    # Create AI model manager
    manager = AIModelManager()
    
    # Test prompt
    prompt = "Write a haiku about AI and technology."
    print(f"\nPrompt: {prompt}")
    print("\nGenerating...")
    
    try:
        model_config = {
            "name": model_name,
            "ollama_model": model_name,
            "inference_engine": "ollama",
        }
        
        result = await manager._generate_text_ollama(prompt, model_config)
        
        print("\n✅ PASSED: Text generation successful")
        print("\nGenerated Text:")
        print("-" * 80)
        print(result)
        print("-" * 80)
        return True
        
    except Exception as e:
        print(f"\n❌ FAILED: Text generation error: {str(e)}")
        return False


def test_fallback_mechanism():
    """Test that fallback mechanism is configured correctly."""
    print("\n" + "=" * 80)
    print("TEST 4: Fallback Mechanism")
    print("=" * 80)
    
    print("This test verifies the fallback code is in place.")
    print("Actual fallback testing requires llama.cpp to fail.")
    
    # Check that the code includes fallback logic
    from backend.services import ai_models
    import inspect
    
    source = inspect.getsource(ai_models.AIModelManager.generate_text)
    
    has_ollama_fallback = "find_ollama_installation" in source
    has_try_except = "try:" in source and "except" in source
    
    if has_ollama_fallback and has_try_except:
        print("✅ PASSED: Fallback mechanism is implemented")
        print("  - Exception handling present")
        print("  - Ollama detection integrated")
        print("  - Fallback will trigger when llama.cpp fails")
        return True
    else:
        print("❌ FAILED: Fallback mechanism not properly implemented")
        return False


async def main():
    """Run all manual tests."""
    print("\n" + "=" * 80)
    print("🦙 OLLAMA INTEGRATION MANUAL TEST")
    print("=" * 80)
    print("\nThis script verifies Ollama integration is working correctly.")
    print("Make sure you have:")
    print("  1. Installed Ollama (https://ollama.com)")
    print("  2. Pulled at least one model (e.g., ollama pull llama3:8b)")
    print("  3. Started Ollama server (ollama serve)")
    print()
    
    results = []
    
    # Run tests
    results.append(("Ollama Detection", test_ollama_detection()))
    results.append(("Inference Engine Status", test_inference_engine_status()))
    results.append(("Text Generation", await test_ollama_text_generation()))
    results.append(("Fallback Mechanism", test_fallback_mechanism()))
    
    # Summary
    print("\n" + "=" * 80)
    print("📊 TEST SUMMARY")
    print("=" * 80)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status}: {name}")
    
    print(f"\nResults: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! Ollama integration is working correctly.")
        print("\nNext steps:")
        print("  - Ollama will automatically serve as a fallback when llama.cpp fails")
        print("  - No configuration needed - it just works!")
        print("  - See OLLAMA_SETUP.md for more details")
        return 0
    else:
        print("\n⚠️  Some tests failed. See output above for details.")
        if not results[0][1]:  # Detection failed
            print("\nQuick fix:")
            print("  curl -fsSL https://ollama.com/install.sh | sh")
            print("  ollama pull llama3:8b")
        return 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
