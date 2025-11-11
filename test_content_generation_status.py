#!/usr/bin/env python3
"""
Test script to generate content and view its status through the API.
"""

import asyncio
import httpx
from uuid import uuid4

BASE_URL = "http://localhost:8000"


async def main():
    print("🧪 Testing Content Generation Status Feature")
    print("=" * 60)
    
    async with httpx.AsyncClient(timeout=30.0) as client:
        # First, create a persona
        print("\n1. Creating test persona...")
        persona_data = {
            "name": "Test AI Agent",
            "appearance": "futuristic AI assistant with glowing blue circuits",
            "personality": "helpful, technical, and friendly",
            "content_themes": ["technology", "AI", "programming"],
            "style_preferences": {
                "visual_style": "cyberpunk",
                "lighting": "neon"
            },
            "default_content_rating": "sfw",
            "allowed_content_ratings": ["sfw"],
            "is_active": True
        }
        
        response = await client.post(f"{BASE_URL}/api/v1/personas/", json=persona_data)
        if response.status_code != 201:
            print(f"❌ Failed to create persona: {response.text}")
            return
        
        persona = response.json()
        persona_id = persona["id"]
        print(f"✅ Persona created: {persona['name']} (ID: {persona_id})")
        
        # Generate image content
        print("\n2. Generating image content...")
        gen_request = {
            "persona_id": persona_id,
            "content_type": "image",
            "content_rating": "sfw",
            "prompt": "A beautiful landscape with mountains and a sunset",
            "quality": "high"
        }
        
        response = await client.post(f"{BASE_URL}/api/v1/content/generate", json=gen_request)
        if response.status_code != 202:
            print(f"❌ Failed to generate content: {response.text}")
            return
        
        gen_result = response.json()
        content_id = gen_result.get("content_id")
        print(f"✅ Content generation started: {gen_result['status']}")
        print(f"   Content ID: {content_id}")
        print(f"   Type: {gen_result['generation_type']}")
        print(f"   Prompt: {gen_result['prompt']}")
        
        # Wait a moment for generation to process
        print("\n3. Waiting for generation to process...")
        await asyncio.sleep(2)
        
        # Get content details
        print("\n4. Fetching content details...")
        response = await client.get(f"{BASE_URL}/api/v1/content/{content_id}")
        if response.status_code != 200:
            print(f"❌ Failed to get content: {response.text}")
            return
        
        content = response.json()
        print(f"✅ Content retrieved:")
        print(f"   Title: {content['title']}")
        print(f"   Type: {content['content_type']}")
        print(f"   Rating: {content['content_rating']}")
        print(f"   Moderation Status: {content['moderation_status']}")
        print(f"   File Path: {content.get('file_path', 'N/A')}")
        
        # Get generation status with ACD context
        print("\n5. Fetching AI agent generation status...")
        response = await client.get(f"{BASE_URL}/api/v1/content/{content_id}/status")
        if response.status_code != 200:
            print(f"❌ Failed to get status: {response.text}")
            return
        
        status = response.json()
        print(f"✅ Generation status retrieved:")
        print(f"   Status: {status['status']}")
        print(f"   Has ACD Context: {status['has_acd_context']}")
        
        if status['has_acd_context']:
            acd = status['acd_context']
            print(f"\n   🤖 AI Agent Details:")
            print(f"      Phase: {acd['phase']}")
            print(f"      State: {acd['state']}")
            print(f"      Status: {acd['status']}")
            print(f"      Confidence: {acd.get('confidence', 'N/A')}")
            print(f"      Queue Status: {acd.get('queue_status', 'N/A')}")
            print(f"      Note: {acd.get('note', 'N/A')}")
            
            if acd.get('context'):
                print(f"\n   📊 Context Details:")
                ctx = acd['context']
                print(f"      Prompt: {ctx.get('prompt', 'N/A')[:50]}...")
                print(f"      Quality: {ctx.get('quality', 'N/A')}")
                print(f"      Using Fallback: {ctx.get('using_fallback', False)}")
                
                if ctx.get('error'):
                    print(f"\n   ❌ Error Details:")
                    print(f"      {ctx['error']}")
        
        # Generate text content to test fallback
        print("\n\n6. Generating text content (may use fallback)...")
        gen_request = {
            "persona_id": persona_id,
            "content_type": "text",
            "content_rating": "sfw",
            "prompt": "Write a social media post about AI technology",
            "quality": "high"
        }
        
        response = await client.post(f"{BASE_URL}/api/v1/content/generate", json=gen_request)
        if response.status_code == 202:
            gen_result = response.json()
            content_id2 = gen_result.get("content_id")
            print(f"✅ Text content generation started: {content_id2}")
            
            await asyncio.sleep(2)
            
            # Get status for text content
            response = await client.get(f"{BASE_URL}/api/v1/content/{content_id2}/status")
            if response.status_code == 200:
                status2 = response.json()
                print(f"\n   Text Generation Status:")
                print(f"   Has ACD Context: {status2['has_acd_context']}")
                
                if status2['has_acd_context']:
                    acd2 = status2['acd_context']
                    print(f"   State: {acd2['state']}")
                    print(f"   Confidence: {acd2.get('confidence', 'N/A')}")
                    
                    if acd2.get('context', {}).get('using_fallback'):
                        print(f"   ⚠️  Using fallback generation method")
        
        print("\n\n✅ Test completed successfully!")
        print(f"\n📱 View content in browser:")
        print(f"   http://localhost:8000/admin/content/view?id={content_id}")


if __name__ == "__main__":
    asyncio.run(main())
