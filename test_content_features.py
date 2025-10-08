#!/usr/bin/env python3
"""
Test script to validate content view and delete features.

This script tests:
1. Content generation (text and image)
2. Content listing
3. Content viewing (GET endpoint)
4. Content deletion (DELETE endpoint)
"""

import asyncio
import sys
from uuid import UUID
import httpx


BASE_URL = "http://localhost:8000"


async def test_content_features():
    """Test all content features."""
    async with httpx.AsyncClient() as client:
        print("🧪 Testing Content Features")
        print("=" * 50)
        
        # Step 1: Create a test persona
        print("\n1️⃣ Creating test persona...")
        persona_data = {
            "name": "Test Bot",
            "appearance": "Friendly AI assistant",
            "personality": "Helpful and engaging",
            "content_themes": ["technology", "AI"],
            "style_preferences": {"tone": "casual"},
            "default_content_rating": "sfw",
            "allowed_content_ratings": ["sfw"]
        }
        
        response = await client.post(
            f"{BASE_URL}/api/v1/personas/",
            json=persona_data
        )
        
        if response.status_code not in (200, 201):
            print(f"❌ Failed to create persona: {response.status_code}")
            return False
        
        persona = response.json()
        persona_id = persona["id"]
        print(f"✅ Persona created: {persona['name']} (ID: {persona_id})")
        
        # Step 2: Generate text content
        print("\n2️⃣ Generating text content...")
        gen_request = {
            "persona_id": persona_id,
            "content_type": "text",
            "prompt": "Write about AI technology"
        }
        
        response = await client.post(
            f"{BASE_URL}/api/v1/content/generate",
            json=gen_request
        )
        
        if response.status_code != 202:
            print(f"❌ Failed to generate content: {response.status_code}")
            return False
        
        result = response.json()
        content_id = result["content_id"]
        print(f"✅ Content generated: {content_id}")
        
        # Step 3: List all content
        print("\n3️⃣ Listing all content...")
        response = await client.get(f"{BASE_URL}/api/v1/content/")
        
        if response.status_code != 200:
            print(f"❌ Failed to list content: {response.status_code}")
            return False
        
        data = response.json()
        print(f"✅ Found {data['count']} content items")
        
        # Step 4: View specific content (test GET endpoint)
        print("\n4️⃣ Viewing specific content...")
        response = await client.get(f"{BASE_URL}/api/v1/content/{content_id}")
        
        if response.status_code != 200:
            print(f"❌ Failed to get content: {response.status_code}")
            return False
        
        content = response.json()
        print(f"✅ Content details retrieved:")
        print(f"   - Title: {content['title']}")
        print(f"   - Type: {content['content_type']}")
        print(f"   - Rating: {content['content_rating']}")
        print(f"   - Status: {content['moderation_status']}")
        print(f"   - Is Deleted: {content['is_deleted']}")
        
        # Step 5: Delete content (test DELETE endpoint)
        print("\n5️⃣ Deleting content...")
        response = await client.delete(f"{BASE_URL}/api/v1/content/{content_id}")
        
        if response.status_code != 200:
            print(f"❌ Failed to delete content: {response.status_code}")
            return False
        
        result = response.json()
        print(f"✅ Content deleted: {result['message']}")
        
        # Step 6: Verify soft delete
        print("\n6️⃣ Verifying soft delete...")
        response = await client.get(f"{BASE_URL}/api/v1/content/{content_id}")
        
        if response.status_code != 200:
            print(f"❌ Failed to verify delete: {response.status_code}")
            return False
        
        content = response.json()
        if content['is_deleted']:
            print(f"✅ Soft delete confirmed: is_deleted = True")
        else:
            print(f"❌ Soft delete failed: is_deleted = False")
            return False
        
        # Step 7: Verify deleted content excluded from list
        print("\n7️⃣ Verifying deleted content excluded from list...")
        response = await client.get(f"{BASE_URL}/api/v1/content/")
        
        if response.status_code != 200:
            print(f"❌ Failed to list content: {response.status_code}")
            return False
        
        data = response.json()
        deleted_found = any(c['id'] == content_id for c in data['content'])
        
        if not deleted_found:
            print(f"✅ Deleted content excluded from list (count: {data['count']})")
        else:
            print(f"❌ Deleted content still in list!")
            return False
        
        print("\n" + "=" * 50)
        print("✅ All tests passed!")
        print("\n📋 Summary:")
        print("   ✓ Content generation works")
        print("   ✓ Content listing works")
        print("   ✓ Content viewing (GET) works")
        print("   ✓ Content deletion (DELETE) works")
        print("   ✓ Soft delete implementation verified")
        print("   ✓ Deleted content excluded from listings")
        
        return True


if __name__ == "__main__":
    try:
        success = asyncio.run(test_content_features())
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n⚠️  Test interrupted")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
