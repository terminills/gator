#!/bin/bash
# Test script for install_vllm_rocm.sh argument parsing and GPU detection

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
NC='\033[0m' # No Color

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INSTALL_SCRIPT="$SCRIPT_DIR/../scripts/install_vllm_rocm.sh"

test_count=0
pass_count=0

run_test() {
    local test_name="$1"
    local expected_result="$2"
    shift 2
    
    test_count=$((test_count + 1))
    echo "Test $test_count: $test_name"
    
    # Run the command and capture output
    output=$("$@" 2>&1 || true)
    
    if echo "$output" | grep -q "$expected_result"; then
        echo -e "${GREEN}✓ PASS${NC}"
        pass_count=$((pass_count + 1))
    else
        echo -e "${RED}✗ FAIL${NC}"
        echo "Expected to find: $expected_result"
        echo "Got output:"
        echo "$output" | head -20
    fi
    echo
}

echo "Testing install_vllm_rocm.sh argument parsing..."
echo

# Test 1: --no-build-isolation should be rejected
run_test "Reject --no-build-isolation" \
    "Unknown option: --no-build-isolation" \
    bash "$INSTALL_SCRIPT" --no-build-isolation

# Test 2: --help should work
run_test "Show help with --help" \
    "Usage:" \
    bash "$INSTALL_SCRIPT" --help

# Test 3: -h should work
run_test "Show help with -h" \
    "Usage:" \
    bash "$INSTALL_SCRIPT" -h

# Test 4: Unknown flags should be rejected
run_test "Reject unknown flag" \
    "Unknown option: --unknown-flag" \
    bash "$INSTALL_SCRIPT" --unknown-flag

# Test 5: --amd-repo should be accepted (will fail at venv check)
run_test "Accept --amd-repo flag" \
    "vLLM ROCm Installation Script" \
    bash "$INSTALL_SCRIPT" --amd-repo < /dev/null

# Test 6: --repair should be accepted (will fail at venv check)
run_test "Accept --repair flag" \
    "PyTorch Repair Mode" \
    bash "$INSTALL_SCRIPT" --repair < /dev/null

# Test 7: GPU architecture detection logic
echo
echo "Test 7: GPU architecture detection logic"
test_count=$((test_count + 1))

# Test the extraction logic used by detect_gpu_arch
mock_rocminfo_output='Agent 1
  Name:                    gfx90a
  Marketing Name:          AMD Instinct MI210
Agent 2
  Name:                    gfx90a
  Marketing Name:          AMD Instinct MI210'

detected=$(echo "$mock_rocminfo_output" | grep -oP 'Name:\s+\Kgfx[0-9a-z]+' | sort -u | tr '\n' ';' | sed 's/;$//')
if [ "$detected" = "gfx90a" ]; then
    echo -e "${GREEN}✓ PASS${NC}"
    pass_count=$((pass_count + 1))
else
    echo -e "${RED}✗ FAIL${NC}"
    echo "Expected: gfx90a, Got: $detected"
fi

echo "=========================================="
echo "Test Results: $pass_count/$test_count passed"
echo "=========================================="

if [ $pass_count -eq $test_count ]; then
    echo -e "${GREEN}All tests passed!${NC}"
    exit 0
else
    echo -e "${RED}Some tests failed${NC}"
    exit 1
fi
