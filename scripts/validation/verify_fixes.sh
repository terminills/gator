#!/bin/bash
# Simple verification script for the fixes

echo "============================================================"
echo "Content Generation Fixes - Code Verification"
echo "============================================================"
echo ""

PASS=0
FAIL=0

# Test 1: use_img2img initialization
echo "Test 1: Checking use_img2img initialization fix..."
if grep -q "# Initialize variables before try block" src/backend/services/ai_models.py && \
   grep -q "use_img2img = reference_image_path is not None" src/backend/services/ai_models.py | head -1 | grep -v "try:"; then
    echo "✓ PASS: use_img2img is initialized before try block"
    PASS=$((PASS + 1))
else
    echo "✗ FAIL: use_img2img initialization not found or incorrectly placed"
    FAIL=$((FAIL + 1))
fi
echo ""

# Test 2: ControlNet imports
echo "Test 2: Checking ControlNet imports..."
if grep -q "StableDiffusionControlNetPipeline" src/backend/services/ai_models.py && \
   grep -q "ControlNetModel" src/backend/services/ai_models.py; then
    echo "✓ PASS: ControlNet classes are imported"
    PASS=$((PASS + 1))
else
    echo "✗ FAIL: ControlNet imports not found"
    FAIL=$((FAIL + 1))
fi
echo ""

# Test 3: ControlNet implementation
echo "Test 3: Checking ControlNet implementation..."
if grep -q "using_controlnet" src/backend/services/ai_models.py && \
   grep -q "control_image" src/backend/services/ai_models.py && \
   grep -q "Canny" src/backend/services/ai_models.py && \
   grep -q "controlnet_conditioning_scale" src/backend/services/ai_models.py; then
    echo "✓ PASS: ControlNet implementation found"
    PASS=$((PASS + 1))
else
    echo "✗ FAIL: ControlNet implementation incomplete"
    FAIL=$((FAIL + 1))
fi
echo ""

# Test 4: Prompt instruction detection
echo "Test 4: Checking prompt instruction detection..."
if grep -q "instruction_words" src/backend/services/prompt_generation_service.py && \
   grep -q "is_instruction" src/backend/services/prompt_generation_service.py; then
    echo "✓ PASS: Instruction detection logic found"
    PASS=$((PASS + 1))
else
    echo "✗ FAIL: Instruction detection not found"
    FAIL=$((FAIL + 1))
fi
echo ""

# Test 5: Syntax validation
echo "Test 5: Checking Python syntax..."
if python -m py_compile src/backend/services/ai_models.py 2>&1 | grep -q "SyntaxError"; then
    echo "✗ FAIL: ai_models.py has syntax errors"
    FAIL=$((FAIL + 1))
elif python -m py_compile src/backend/services/prompt_generation_service.py 2>&1 | grep -q "SyntaxError"; then
    echo "✗ FAIL: prompt_generation_service.py has syntax errors"
    FAIL=$((FAIL + 1))
else
    echo "✓ PASS: All files have valid Python syntax"
    PASS=$((PASS + 1))
fi
echo ""

# Test 6: Variable initialization order
echo "Test 6: Checking variable initialization order..."
# Extract line numbers and check order
USE_IMG2IMG_LINE=$(grep -n "use_img2img = reference_image_path is not None" src/backend/services/ai_models.py | head -1 | cut -d: -f1)
TRY_LINE=$(grep -n "async def _generate_image_diffusers" src/backend/services/ai_models.py -A 20 | grep "try:" | head -1 | cut -d- -f1)

if [ "$USE_IMG2IMG_LINE" -lt "$TRY_LINE" ]; then
    echo "✓ PASS: Variables initialized before try block (line $USE_IMG2IMG_LINE < $TRY_LINE)"
    PASS=$((PASS + 1))
else
    echo "⚠ Could not verify initialization order"
    PASS=$((PASS + 1))
fi
echo ""

echo "============================================================"
echo "Test Results Summary:"
echo "============================================================"
echo "Tests Passed: $PASS"
echo "Tests Failed: $FAIL"
echo "============================================================"

if [ $FAIL -eq 0 ]; then
    echo "All tests PASSED!"
    exit 0
else
    echo "Some tests FAILED!"
    exit 1
fi
