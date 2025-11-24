"""
Model Detection Utilities

Provides common functions for detecting AI model inference engines
and installations across the platform.
"""

import os
import shutil
import subprocess
from pathlib import Path
from typing import Optional, Dict, Any, List


def find_comfyui_installation(base_dir: Optional[Path] = None) -> Optional[Path]:
    """
    Find ComfyUI installation in common locations.
    
    This function checks multiple common installation locations to detect
    a valid ComfyUI installation. It validates the installation by checking
    for the presence of main.py.
    
    Args:
        base_dir: Optional base directory to use for relative paths (e.g., models_dir.parent)
    
    Returns:
        Path to ComfyUI directory if found, None otherwise
    """
    # Check environment variable first (set by install script)
    if "COMFYUI_DIR" in os.environ:
        comfyui_path = Path(os.environ["COMFYUI_DIR"])
        if comfyui_path.exists() and (comfyui_path / "main.py").exists():
            return comfyui_path
    
    # Build list of possible locations
    possible_locations = []
    
    # If base_dir provided (e.g., models directory parent), check next to it
    if base_dir:
        possible_locations.append(base_dir / "ComfyUI")
    
    # Check common installation locations
    possible_locations.extend([
        Path("./ComfyUI"),  # Current directory
        Path.cwd() / "ComfyUI",  # Current working directory
        Path(__file__).parent.parent.parent.parent / "ComfyUI",  # Repository root
        Path.home() / "ComfyUI",  # Home directory
        Path("/opt/ComfyUI"),  # System-wide installation
        Path("/usr/local/ComfyUI"),  # Alternative system location
    ])
    
    for location in possible_locations:
        # Check if directory exists and has main.py file (not directory)
        if location.exists() and location.is_dir():
            main_py = location / "main.py"
            if main_py.exists() and main_py.is_file():
                # Return absolute path for consistency
                return location.resolve()
    
    return None


def check_comfyui_api_available(api_url: str = "http://127.0.0.1:8188", timeout: float = 3.0) -> bool:
    """
    Check if ComfyUI API is running and accessible.
    
    Args:
        api_url: ComfyUI API URL (default: http://127.0.0.1:8188)
        timeout: Request timeout in seconds (default: 3.0)
    
    Returns:
        True if ComfyUI API is accessible, False otherwise
    """
    try:
        import httpx
        
        # Try multiple endpoints to verify ComfyUI is running
        endpoints_to_check = [
            f"{api_url}/system_stats",
            f"{api_url}/queue",
            f"{api_url}/object_info",
        ]
        
        for endpoint in endpoints_to_check:
            try:
                response = httpx.get(endpoint, timeout=timeout)
                if response.status_code == 200:
                    return True
            except Exception:
                continue
        
        return False
        
    except ImportError:
        # httpx not available, cannot check API
        return False
    except Exception:
        return False


def find_automatic1111_installation(base_dir: Optional[Path] = None) -> Optional[Path]:
    """
    Find Automatic1111 (Stable Diffusion WebUI) installation.
    
    Args:
        base_dir: Optional base directory to use for relative paths
    
    Returns:
        Path to Automatic1111 directory if found, None otherwise
    """
    # Check environment variable
    if "A1111_DIR" in os.environ or "SD_WEBUI_DIR" in os.environ:
        a1111_path = Path(os.environ.get("A1111_DIR", os.environ.get("SD_WEBUI_DIR", "")))
        if a1111_path.exists() and (a1111_path / "launch.py").exists():
            return a1111_path
    
    # Build list of possible locations
    possible_locations = []
    
    if base_dir:
        possible_locations.extend([
            base_dir / "stable-diffusion-webui",
            base_dir / "automatic1111",
        ])
    
    possible_locations.extend([
        Path("./stable-diffusion-webui"),
        Path.cwd() / "stable-diffusion-webui",
        Path(__file__).parent.parent.parent.parent / "stable-diffusion-webui",
        Path.home() / "stable-diffusion-webui",
        Path("./automatic1111"),
        Path.cwd() / "automatic1111",
    ])
    
    for location in possible_locations:
        if location.exists() and location.is_dir():
            launch_py = location / "launch.py"
            if launch_py.exists() and launch_py.is_file():
                return location.resolve()
    
    return None


def find_vllm_installation() -> Optional[Dict[str, Any]]:
    """
    Detect vLLM installation and get version info.
    
    Returns:
        Dictionary with vLLM info if installed, None otherwise
    """
    try:
        import vllm
        version = getattr(vllm, "__version__", "unknown")
        
        # Check if it's a ROCm build by looking for hip/rocm in the installation
        import sys
        vllm_path = Path(vllm.__file__).parent
        is_rocm = any([
            (vllm_path / "rocm").exists(),
            "rocm" in str(vllm_path).lower(),
            "hip" in str(vllm_path).lower(),
        ])
        
        return {
            "installed": True,
            "version": version,
            "path": str(vllm_path),
            "is_rocm_build": is_rocm,
            "python_package": True,
        }
    except ImportError:
        # Check if vllm source installation exists
        possible_locations = [
            Path("./vllm-rocm"),
            Path.cwd() / "vllm-rocm",
            Path(__file__).parent.parent.parent.parent / "vllm-rocm",
            Path.home() / "vllm-rocm",
            Path("./vllm"),
            Path.cwd() / "vllm",
        ]
        
        for location in possible_locations:
            if location.exists() and location.is_dir():
                setup_py = location / "setup.py"
                if setup_py.exists():
                    return {
                        "installed": True,
                        "version": "source",
                        "path": str(location.resolve()),
                        "is_rocm_build": "rocm" in str(location).lower(),
                        "python_package": False,
                        "source_dir": str(location.resolve()),
                    }
        
        return None


def find_llama_cpp_installation() -> Optional[Dict[str, Any]]:
    """
    Detect llama.cpp or llama-cpp-python installation.
    
    Returns:
        Dictionary with llama.cpp info if installed, None otherwise
    """
    # Check for Python bindings first
    try:
        import llama_cpp
        version = getattr(llama_cpp, "__version__", "unknown")
        
        # Check if it's a HIP/ROCm build
        llama_path = Path(llama_cpp.__file__).parent
        is_hip = any([
            "hip" in str(llama_path).lower(),
            "rocm" in str(llama_path).lower(),
        ])
        
        return {
            "installed": True,
            "type": "python-bindings",
            "version": version,
            "path": str(llama_path),
            "is_hip_build": is_hip,
        }
    except ImportError:
        pass
    
    # Check for standalone llama.cpp binary (system-wide installation)
    # Check common binary names
    binary_names = [
        "llama-server",
        "llama-cli", 
        "llama",
        "llama.cpp",
        "main",  # Legacy llama.cpp binary name
    ]
    
    llama_server = None
    for binary_name in binary_names:
        found_binary = shutil.which(binary_name)
        if found_binary:
            llama_server = found_binary
            break
    
    if llama_server:
        try:
            # Try to get version info
            result = subprocess.run(
                [llama_server, "--version"],
                capture_output=True,
                text=True,
                timeout=5
            )
            version_output = result.stdout.strip() if result.returncode == 0 else "unknown"
            
            # Check if HIP/ROCm support is compiled in
            is_hip = "hip" in version_output.lower() or "rocm" in version_output.lower()
            
            return {
                "installed": True,
                "type": "binary",
                "version": version_output,
                "path": llama_server,
                "is_hip_build": is_hip,
            }
        except (subprocess.TimeoutExpired, Exception):
            # Even if version check fails, binary exists so mark as installed
            return {
                "installed": True,
                "type": "binary",
                "version": "unknown",
                "path": llama_server,
                "is_hip_build": False,
            }
    
    # Check for source installation in common locations
    possible_locations = [
        Path("./llama.cpp"),
        Path.cwd() / "llama.cpp",
        Path(__file__).parent.parent.parent.parent / "llama.cpp",
        Path.home() / "llama.cpp",
        Path("/usr/local/llama.cpp"),  # System-wide installation
        Path("/opt/llama.cpp"),  # Alternative system location
    ]
    
    for location in possible_locations:
        if location.exists() and location.is_dir():
            makefile = location / "Makefile"
            if makefile.exists():
                # Check if there's a built server binary
                for binary_name in ["llama-server", "llama-cli", "main", "llama"]:
                    server_binary = location / binary_name
                    if server_binary.exists():
                        return {
                            "installed": True,
                            "type": "source",
                            "version": "source-build",
                            "path": str(location.resolve()),
                            "is_hip_build": False,  # Would need to check build flags
                            "source_dir": str(location.resolve()),
                            "binary": str(server_binary.resolve()),
                        }
    
    # Check system-wide shared library installations
    # These are common on Linux when llama.cpp is installed via package manager
    for lib_path in ["/usr/lib", "/usr/local/lib", "/opt/lib"]:
        lib_dir = Path(lib_path)
        if lib_dir.exists():
            # Look for llama.cpp shared libraries
            for lib_file in ["libllama.so", "libllama.dylib", "libllama.dll"]:
                if (lib_dir / lib_file).exists():
                    return {
                        "installed": True,
                        "type": "system-library",
                        "version": "system-installed",
                        "path": str(lib_dir / lib_file),
                        "is_hip_build": False,
                    }
    
    return None


def find_ollama_installation() -> Optional[Dict[str, Any]]:
    """
    Detect Ollama installation and get version info.
    
    Ollama is a local LLM runtime that provides a simple API for running
    various language models. It can serve as a fallback when llama.cpp fails.
    
    Returns:
        Dictionary with Ollama info if installed, None otherwise
    """
    # Check for Ollama binary in PATH
    ollama_binary = shutil.which("ollama")
    
    if not ollama_binary:
        return None
    
    try:
        # Try to get version info
        result = subprocess.run(
            [ollama_binary, "--version"],
            capture_output=True,
            text=True,
            timeout=5
        )
        
        if result.returncode == 0:
            version_output = result.stdout.strip()
            # Parse version from output like "ollama version is 0.12.10"
            version = "unknown"
            if "version" in version_output.lower():
                parts = version_output.split()
                if len(parts) >= 3:
                    version = parts[-1]
            
            # Check if Ollama server is running by trying to list models
            try:
                list_result = subprocess.run(
                    [ollama_binary, "list"],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                
                # Parse available models from output
                available_models = []
                if list_result.returncode == 0:
                    lines = list_result.stdout.strip().split('\n')
                    # Skip header line and parse model names
                    for line in lines[1:]:  # Skip first line (header)
                        if line.strip():
                            # Model name is the first column
                            model_name = line.split()[0]
                            available_models.append(model_name)
                
                return {
                    "installed": True,
                    "type": "ollama",
                    "version": version,
                    "path": ollama_binary,
                    "available_models": available_models,
                    "server_running": list_result.returncode == 0,
                }
            except (subprocess.TimeoutExpired, subprocess.CalledProcessError, OSError, FileNotFoundError) as e:
                # Ollama is installed but server might not be running
                return {
                    "installed": True,
                    "type": "ollama",
                    "version": version,
                    "path": ollama_binary,
                    "available_models": [],
                    "server_running": False,
                }
        else:
            # Version check failed but binary exists
            return {
                "installed": True,
                "type": "ollama",
                "version": "unknown",
                "path": ollama_binary,
                "available_models": [],
                "server_running": False,
            }
    except (subprocess.TimeoutExpired, subprocess.CalledProcessError, OSError, FileNotFoundError):
        # Binary exists but failed to execute
        return {
            "installed": True,
            "type": "ollama",
            "version": "unknown",
            "path": ollama_binary,
            "available_models": [],
            "server_running": False,
        }


def get_inference_engines_status(base_dir: Optional[Path] = None) -> Dict[str, Dict[str, Any]]:
    """
    Get status of all inference engines.
    
    Args:
        base_dir: Optional base directory for relative path checks
    
    Returns:
        Dictionary mapping engine names to their status information
    """
    engines = {}
    
    # Text generation engines
    vllm_info = find_vllm_installation()
    if vllm_info:
        engines["vllm"] = {
            "category": "text",
            "name": "vLLM",
            "status": "installed",
            **vllm_info
        }
    else:
        engines["vllm"] = {
            "category": "text",
            "name": "vLLM",
            "status": "not_installed",
            "install_script": "scripts/install_vllm_rocm.sh"
        }
    
    llama_cpp_info = find_llama_cpp_installation()
    if llama_cpp_info:
        engines["llama-cpp"] = {
            "category": "text",
            "name": "llama.cpp",
            "status": "installed",
            **llama_cpp_info
        }
    else:
        engines["llama-cpp"] = {
            "category": "text",
            "name": "llama.cpp",
            "status": "not_installed",
            "install_url": "https://github.com/ggerganov/llama.cpp"
        }
    
    # Ollama (alternative text generation engine)
    ollama_info = find_ollama_installation()
    if ollama_info:
        engines["ollama"] = {
            "category": "text",
            "name": "Ollama",
            "status": "installed",
            **ollama_info
        }
    else:
        engines["ollama"] = {
            "category": "text",
            "name": "Ollama",
            "status": "not_installed",
            "install_url": "https://ollama.com/download"
        }
    
    # Image generation engines
    comfyui_path = find_comfyui_installation(base_dir)
    if comfyui_path:
        engines["comfyui"] = {
            "category": "image",
            "name": "ComfyUI",
            "status": "installed",
            "path": str(comfyui_path),
            "web_ui": "http://localhost:8188"
        }
    else:
        engines["comfyui"] = {
            "category": "image",
            "name": "ComfyUI",
            "status": "not_installed",
            "install_script": "scripts/install_comfyui_rocm.sh"
        }
    
    a1111_path = find_automatic1111_installation(base_dir)
    if a1111_path:
        engines["automatic1111"] = {
            "category": "image",
            "name": "Automatic1111 WebUI",
            "status": "installed",
            "path": str(a1111_path),
            "web_ui": "http://localhost:7860"
        }
    else:
        engines["automatic1111"] = {
            "category": "image",
            "name": "Automatic1111 WebUI",
            "status": "not_installed",
            "install_url": "https://github.com/AUTOMATIC1111/stable-diffusion-webui"
        }
    
    # Python library inference engines
    for lib_name, display_name, category in [
        ("transformers", "Transformers", "text"),
        ("diffusers", "Diffusers", "image"),
    ]:
        try:
            lib = __import__(lib_name)
            version = getattr(lib, "__version__", "unknown")
            engines[lib_name] = {
                "category": category,
                "name": display_name,
                "status": "installed",
                "version": version,
                "python_package": True,
            }
        except ImportError:
            engines[lib_name] = {
                "category": category,
                "name": display_name,
                "status": "not_installed",
                "python_package": True,
            }
    
    return engines


def check_inference_engine_available(engine: str, base_dir: Optional[Path] = None, check_api: bool = False) -> bool:
    """
    Check if an inference engine is available.
    
    Args:
        engine: Name of the inference engine (vllm, comfyui, diffusers, transformers, llama-cpp, llama.cpp, ollama)
        base_dir: Optional base directory for relative path checks
        check_api: For ComfyUI, also check if API is running (default: False)
    
    Returns:
        True if the engine is available, False otherwise
    """
    if engine == "vllm":
        return find_vllm_installation() is not None
    elif engine == "comfyui":
        # Check installation first
        comfyui_path = find_comfyui_installation(base_dir)
        if comfyui_path is None:
            return False
        
        # Optionally check if API is running
        if check_api:
            api_url = os.environ.get("COMFYUI_API_URL", "http://127.0.0.1:8188")
            return check_comfyui_api_available(api_url)
        
        return True
    elif engine == "automatic1111":
        return find_automatic1111_installation(base_dir) is not None
    elif engine in ["llama-cpp", "llama.cpp"]:  # Support both formats
        return find_llama_cpp_installation() is not None
    elif engine == "ollama":
        return find_ollama_installation() is not None
    elif engine in ["diffusers", "transformers"]:
        try:
            __import__(engine)
            return True
        except ImportError:
            return False
    else:
        return False
