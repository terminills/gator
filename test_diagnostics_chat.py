#!/usr/bin/env python3
"""
Test script for diagnostics chat functionality.
Demonstrates local model usage with verbose CLI output.
"""

import sys
import asyncio

sys.path.insert(0, 'src')

from backend.services.gator_agent_service import gator_agent
from backend.services.ai_models import ai_models


async def test_chat():
    """Test chat functionality with verbose output."""
    
    # Initialize AI models
    print("=== INITIALIZING AI MODELS ===")
    await ai_models.initialize_models()
    
    # Check available models
    local_text_models = [m for m in ai_models.available_models.get('text', []) 
                         if m.get('provider') == 'local' and m.get('loaded')]
    
    print(f"\nLocal text models loaded: {len(local_text_models)}")
    for model in local_text_models:
        print(f"  - {model['name']} ({model['inference_engine']})")
    
    print("\n" + "="*70)
    print("TESTING NORMAL MODE (Non-verbose)")
    print("="*70)
    
    try:
        response = await gator_agent.process_message(
            "Hello Gator, how do I create a persona?",
            verbose=False
        )
        print("\nResponse:")
        print(response)
    except Exception as e:
        print(f"\nERROR: {e}")
    
    print("\n" + "="*70)
    print("TESTING CLI MODE (Verbose)")
    print("="*70)
    
    try:
        response = await gator_agent.process_message(
            "Help me generate content",
            verbose=True
        )
        print("\nVerbose Response:")
        print(response)
    except Exception as e:
        print(f"\nERROR: {e}")
    
    print("\n" + "="*70)
    print("TESTING DIRECT AI MODEL CALL")
    print("="*70)
    
    try:
        response = await ai_models.generate_text(
            "Generate a short greeting from Gator"
        )
        print("\nDirect Model Response:")
        print(response)
    except Exception as e:
        print(f"\nERROR: {e}")


if __name__ == "__main__":
    print("🐊 GATOR DIAGNOSTICS CHAT TEST")
    print("Testing local AI model integration\n")
    asyncio.run(test_chat())
    print("\n✅ Test complete!")
