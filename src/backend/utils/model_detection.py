"""
Model Detection Utilities

Provides common functions for detecting AI model inference engines
and installations across the platform.
"""

import os
from pathlib import Path
from typing import Optional


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
    ])
    
    for location in possible_locations:
        # Check if directory exists and has main.py file (not directory)
        if location.exists() and location.is_dir():
            main_py = location / "main.py"
            if main_py.exists() and main_py.is_file():
                # Return absolute path for consistency
                return location.resolve()
    
    return None


def check_inference_engine_available(engine: str, base_dir: Optional[Path] = None) -> bool:
    """
    Check if an inference engine is available.
    
    Args:
        engine: Name of the inference engine (vllm, comfyui, diffusers, transformers)
        base_dir: Optional base directory for relative path checks
    
    Returns:
        True if the engine is available, False otherwise
    """
    try:
        if engine == "vllm":
            import vllm
            return True
        elif engine == "comfyui":
            # Use comprehensive ComfyUI detection
            comfyui_path = find_comfyui_installation(base_dir)
            return comfyui_path is not None
        elif engine == "diffusers":
            import diffusers
            return True
        elif engine == "transformers":
            import transformers
            return True
        else:
            return False
    except ImportError:
        return False
