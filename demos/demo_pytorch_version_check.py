#!/usr/bin/env python3
"""
Demonstration of PyTorch version detection and compatible dependency checking.

This script shows how to:
1. Detect installed ROCm version
2. Get appropriate PyTorch installation URLs
3. Check currently installed PyTorch version
4. Determine compatible dependency versions based on PyTorch version
"""

from backend.utils.rocm_utils import (
    detect_rocm_version,
    get_pytorch_index_url,
    get_pytorch_install_command,
    check_pytorch_installation,
    get_compatible_dependency_versions,
    get_recommended_pytorch_version,
)


def main():
    print("=" * 80)
    print("PyTorch Version Detection and Dependency Compatibility Check")
    print("=" * 80)
    print()

    # Step 1: Detect ROCm version
    print("Step 1: Detecting ROCm version...")
    rocm_version = detect_rocm_version()
    if rocm_version:
        print(f"✓ ROCm detected: {rocm_version.version}")
        print(f"  - Major: {rocm_version.major}")
        print(f"  - Minor: {rocm_version.minor}")
        print(f"  - Is ROCm 6.5+: {rocm_version.is_6_5_or_later}")
    else:
        print("✗ ROCm not detected (will use CPU-only PyTorch)")
    print()

    # Step 2: Get PyTorch installation URLs
    print("Step 2: PyTorch installation URLs...")
    stable_url = get_pytorch_index_url(rocm_version, use_nightly=False)
    print(f"  Stable builds: {stable_url}")

    if rocm_version and rocm_version.is_6_5_or_later:
        nightly_url = get_pytorch_index_url(rocm_version, use_nightly=True)
        print(f"  Nightly builds: {nightly_url}")
    print()

    # Step 3: Check currently installed PyTorch
    print("Step 3: Checking installed PyTorch...")
    pytorch_info = check_pytorch_installation()
    if pytorch_info["installed"]:
        print(f"✓ PyTorch is installed")
        print(f"  - Version: {pytorch_info['version']}")
        print(f"  - Major.Minor: {pytorch_info['pytorch_major_minor']}")
        print(f"  - ROCm build: {pytorch_info['is_rocm_build']}")
        if pytorch_info["is_rocm_build"]:
            print(f"  - ROCm version: {pytorch_info['rocm_build_version']}")
        print(f"  - GPU available: {pytorch_info['gpu_available']}")
        print(f"  - GPU count: {pytorch_info['gpu_count']}")
    else:
        print("✗ PyTorch is not installed")
    print()

    # Step 4: Get compatible dependency versions
    print("Step 4: Compatible dependency versions for installed PyTorch...")
    if pytorch_info["installed"]:
        deps = get_compatible_dependency_versions(pytorch_info["version"])
        print(f"  Based on PyTorch {pytorch_info['version']}:")
        for package, version_spec in deps.items():
            print(f"    - {package}: {version_spec}")
    else:
        print("  No PyTorch installed, showing default recommendations:")
        deps = get_compatible_dependency_versions(None)
        for package, version_spec in deps.items():
            print(f"    - {package}: {version_spec}")
    print()

    # Step 5: Get recommended PyTorch version for this system
    print("Step 5: Recommended PyTorch version for this system...")
    recommended = get_recommended_pytorch_version(rocm_version)
    print(f"  PyTorch: {recommended.get('torch', 'N/A')}")
    print(f"  torchvision: {recommended.get('torchvision', 'N/A')}")
    if "torchaudio" in recommended:
        print(f"  torchaudio: {recommended.get('torchaudio', 'N/A')}")
    print(f"  Note: {recommended.get('note', 'N/A')}")
    if recommended.get("nightly_available"):
        print(f"  ✓ Nightly builds available")
    print()

    # Step 6: Generate installation command
    print("Step 6: PyTorch installation command...")
    if rocm_version:
        command, metadata = get_pytorch_install_command(
            rocm_version,
            use_nightly=False,
            include_torchvision=True,
            include_torchaudio=True,
        )
        print(f"  {command}")
        print(f"  Index URL: {metadata['index_url']}")
    else:
        print("  pip3 install torch torchvision torchaudio")
    print()

    # Summary
    print("=" * 80)
    print("Summary")
    print("=" * 80)

    if rocm_version and rocm_version.major >= 7:
        print("✓ Your system supports PyTorch 2.10+ with ROCm 7.0+")
        print("  To install PyTorch 2.10 nightly:")
        nightly_command, _ = get_pytorch_install_command(
            rocm_version,
            use_nightly=True,
            include_torchvision=True,
            include_torchaudio=True,
        )
        print(f"  {nightly_command}")
        print()
        print("  Compatible ML libraries for PyTorch 2.10+:")
        deps_2_10 = get_compatible_dependency_versions("2.10.0+rocm7.0")
        for package, version_spec in deps_2_10.items():
            print(f"    pip install '{package}{version_spec}'")
    elif rocm_version and rocm_version.is_6_5_or_later:
        print("✓ Your system supports PyTorch 2.4+ with ROCm 6.5+")
        print(f"  Recommended: {recommended.get('torch', 'latest')}")
    elif rocm_version:
        print("✓ Your system uses legacy ROCm (5.7 or 6.4)")
        print(f"  Recommended: {recommended.get('torch', 'N/A')}")
    else:
        print("ℹ No ROCm detected - will use CPU-only PyTorch")

    print()
    print("For more information, see:")
    print("  - ROCm documentation: https://rocm.docs.amd.com/")
    print("  - PyTorch ROCm builds: https://pytorch.org/get-started/locally/")
    print("=" * 80)


if __name__ == "__main__":
    main()
