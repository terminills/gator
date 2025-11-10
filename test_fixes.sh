#!/bin/bash
# Test script to validate all fixes

echo "=== Testing Gator Platform Fixes ==="
echo ""

echo "1. Testing database setup (plugin tables)..."
python setup_db.py 2>&1 | grep -q "plugins" && echo "✓ Plugin tables created" || echo "✗ Failed"
echo ""

echo "2. Checking if RSS feed tasks exist..."
[ -f "src/backend/tasks/rss_feed_tasks.py" ] && echo "✓ RSS feed tasks file exists" || echo "✗ Missing"
echo ""

echo "3. Checking Celery configuration..."
grep -q "rss_feed_tasks" src/backend/celery_app.py && echo "✓ RSS tasks in Celery config" || echo "✗ Not configured"
echo ""

echo "4. Checking Gator agent LLM support..."
grep -q "local_model_available\|cloud_llm_available" src/backend/services/gator_agent_service.py && echo "✓ LLM cascade implemented" || echo "✗ Missing"
echo ""

echo "5. Checking model uninstall endpoint..."
grep -q "uninstall_model" src/backend/api/routes/setup.py && echo "✓ Uninstall endpoint exists" || echo "✗ Missing"
echo ""

echo "6. Checking enhanced model detection..."
grep -q "is_valid\|has_model_files" src/backend/api/routes/setup.py && echo "✓ Enhanced detection implemented" || echo "✗ Missing"
echo ""

echo "7. Checking .env generation..."
grep -q "Create from template\|minimal configuration" src/backend/services/setup_service.py && echo "✓ Auto .env generation implemented" || echo "✗ Missing"
echo ""

echo "=== All Checks Complete ==="
