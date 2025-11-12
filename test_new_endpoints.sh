#!/bin/bash
# Test script for new AI model management endpoints

echo "=================================================="
echo "Testing New AI Model Management Endpoints"
echo "=================================================="
echo ""

BASE_URL="${BASE_URL:-http://localhost:8000}"

echo "1. Testing Health Check Endpoint"
echo "   GET /api/v1/setup/dependencies/health"
echo "--------------------------------------------------"
curl -s "$BASE_URL/api/v1/setup/dependencies/health" | jq -r '
  "Overall Status: \(.overall_status)",
  "Core Dependencies: \(.dependencies | to_entries | map(select(.value.category == "core")) | length) installed",
  "ML Dependencies: \(.dependencies | to_entries | map(select(.value.category == "ml")) | length) installed",
  "Issues: \(.issues | length)",
  "Warnings: \(.warnings | length)"
' 2>/dev/null || echo "Endpoint not available (expected if server not running)"
echo ""

echo "2. Testing Model Warm-Up Endpoint"
echo "   POST /api/v1/setup/ai-models/warm-up"
echo "--------------------------------------------------"
curl -s -X POST "$BASE_URL/api/v1/setup/ai-models/warm-up" | jq -r '
  "Status: \(.status)",
  "Models Loaded: \(.models_loaded)",
  "Elapsed Time: \(.elapsed_time_seconds)s",
  "Total Loaded: \(.total_loaded)"
' 2>/dev/null || echo "Endpoint not available (expected if server not running)"
echo ""

echo "3. Testing Telemetry Endpoint"
echo "   GET /api/v1/setup/ai-models/telemetry"
echo "--------------------------------------------------"
curl -s "$BASE_URL/api/v1/setup/ai-models/telemetry" | jq -r '
  "Total Models: \(.summary.total_models)",
  "Loaded Models: \(.summary.loaded_models)",
  "Used Models: \(.summary.used_models)",
  "Unused Models: \(.summary.unused_models)",
  "Recommendations: \(.recommendations | length)"
' 2>/dev/null || echo "Endpoint not available (expected if server not running)"
echo ""

echo "4. Testing Lazy Loading Configuration"
echo "   Environment: AI_MODELS_LAZY_LOAD"
echo "--------------------------------------------------"
if [ "$AI_MODELS_LAZY_LOAD" = "true" ]; then
    echo "✓ Lazy loading ENABLED"
    echo "  Large models will load on first use"
else
    echo "○ Lazy loading DISABLED (default)"
    echo "  To enable: export AI_MODELS_LAZY_LOAD=true"
fi
echo ""

echo "=================================================="
echo "Test Complete"
echo "=================================================="
echo ""
echo "Usage Examples:"
echo ""
echo "# Check system health"
echo "curl $BASE_URL/api/v1/setup/dependencies/health | jq"
echo ""
echo "# Warm up models"
echo "curl -X POST $BASE_URL/api/v1/setup/ai-models/warm-up | jq"
echo ""
echo "# Get telemetry"
echo "curl $BASE_URL/api/v1/setup/ai-models/telemetry | jq"
echo ""
echo "# Enable lazy loading"
echo "export AI_MODELS_LAZY_LOAD=true"
echo ""
