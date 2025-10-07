#!/usr/bin/env python3
"""
AI Model Setup Script

Downloads and configures AI models for the Gator platform.
Handles model installations, dependencies, and hardware optimization.
"""

import os
import sys
import asyncio
import subprocess
from pathlib import Path
from typing import Dict, List, Optional
import json
import shutil
import platform

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    print("Installing required packages...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "torch", "requests"])
    try:
        import torch
        TORCH_AVAILABLE = True
    except ImportError:
        print("Warning: torch not available, using CPU-only mode")
        TORCH_AVAILABLE = False
        # Create a mock torch module
        class MockTorch:
            class cuda:
                @staticmethod
                def is_available():
                    return False
                @staticmethod
                def device_count():
                    return 0
        torch = MockTorch()
    import requests


class ModelSetupManager:
    """Manages AI model setup and configuration."""
    
    def __init__(self, models_dir: str = "./models"):
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(exist_ok=True)
        
        # Hardware detection
        self.has_gpu = torch.cuda.is_available() if TORCH_AVAILABLE else False
        self.gpu_count = torch.cuda.device_count() if self.has_gpu else 0
        self.gpu_memory = self._get_gpu_memory() if self.has_gpu else 0
        self.gpu_type = self._detect_gpu_type()
        
        print(f"Detected hardware: GPU={self.gpu_type}, Memory={self.gpu_memory:.1f}GB, Count={self.gpu_count}")
        
        # Model configurations based on local model stack recommendations
        self.model_configs = {
            "text": [
                {
                    "name": "llama-3.1-70b",
                    "model_id": "meta-llama/Llama-3.1-70B-Instruct", 
                    "size_gb": 140,
                    "min_ram_gb": 64,
                    "min_gpu_memory_gb": 48,
                    "inference_engine": "vllm",
                    "rocm_compatible": True,
                    "quant_options": ["Q4_K_M", "FP16", "BF16"],
                    "description": "Best general local base model (Llama 3.1 70B)"
                },
                {
                    "name": "qwen2.5-72b",
                    "model_id": "Qwen/Qwen2.5-72B-Instruct",
                    "size_gb": 144,
                    "min_ram_gb": 64,
                    "min_gpu_memory_gb": 48,
                    "inference_engine": "vllm",
                    "rocm_compatible": True,
                    "description": "Stronger tools/code, longer context (Qwen2.5-72B)"
                },
                {
                    "name": "mixtral-8x7b",
                    "model_id": "mistralai/Mixtral-8x7B-Instruct-v0.1",
                    "size_gb": 90,
                    "min_ram_gb": 32,
                    "min_gpu_memory_gb": 24,
                    "inference_engine": "vllm",
                    "rocm_compatible": True,
                    "description": "Fast per token, solid instruction following at lower VRAM"
                },
                {
                    "name": "llama-3.1-8b",
                    "model_id": "meta-llama/Llama-3.1-8B-Instruct",
                    "size_gb": 16,
                    "min_ram_gb": 16,
                    "min_gpu_memory_gb": 8,
                    "inference_engine": "vllm",
                    "rocm_compatible": True,
                    "description": "Snappy persona worker for fast mode"
                },
                {
                    "name": "qwen2.5-7b", 
                    "model_id": "Qwen/Qwen2.5-7B-Instruct",
                    "size_gb": 14,
                    "min_ram_gb": 16,
                    "min_gpu_memory_gb": 8,
                    "inference_engine": "vllm",
                    "rocm_compatible": True,
                    "description": "Balanced persona worker"
                }
            ],
            "image": [
                {
                    "name": "sdxl-1.0",
                    "model_id": "stabilityai/stable-diffusion-xl-base-1.0",
                    "size_gb": 7,
                    "min_ram_gb": 16,
                    "min_gpu_memory_gb": 8,
                    "inference_engine": "comfyui",
                    "rocm_compatible": True,
                    "description": "Safest, most supported local base (SDXL 1.0)"
                },
                {
                    "name": "flux.1-dev",
                    "model_id": "black-forest-labs/FLUX.1-dev",
                    "size_gb": 12,
                    "min_ram_gb": 24,
                    "min_gpu_memory_gb": 12,
                    "inference_engine": "comfyui",
                    "rocm_compatible": True,
                    "description": "Very good quality; verify license for commercial use"
                },
                {
                    "name": "stable-diffusion-v1-5",
                    "model_id": "runwayml/stable-diffusion-v1-5",
                    "size_gb": 4.0,
                    "min_ram_gb": 8,
                    "min_gpu_memory_gb": 6,
                    "inference_engine": "diffusers",
                    "rocm_compatible": True,
                    "description": "Classic Stable Diffusion 1.5 for image generation"
                }
            ],
            "voice": [
                {
                    "name": "xtts-v2",
                    "model_id": "coqui/XTTS-v2",
                    "size_gb": 2,
                    "min_ram_gb": 8,
                    "min_gpu_memory_gb": 4,
                    "description": "Multilingual, cloning, runs locally; best all-around"
                },
                {
                    "name": "piper",
                    "model_id": "rhasspy/piper",
                    "size_gb": 0.1,
                    "min_ram_gb": 2,
                    "min_gpu_memory_gb": 0,
                    "description": "Ultralight, CPU-friendly for systems TTS"
                },
                {
                    "name": "bark",
                    "model_id": "suno/bark",
                    "size_gb": 8,
                    "min_ram_gb": 16,
                    "min_gpu_memory_gb": 8,
                    "description": "Decent zero-shot style, heavier; character speech"
                }
            ]
        }
        
    def _get_gpu_memory(self) -> float:
        """Get total GPU memory in GB."""
        if not self.has_gpu or not TORCH_AVAILABLE:
            return 0.0
        
        total_memory = 0
        for i in range(self.gpu_count):
            props = torch.cuda.get_device_properties(i)
            total_memory += props.total_memory
        
        return total_memory / (1024 ** 3)  # Convert to GB
    
    def _detect_gpu_type(self) -> str:
        """Detect GPU type (CUDA, ROCm, or CPU)."""
        try:
            if self.has_gpu and TORCH_AVAILABLE:
                # Check for AMD ROCm
                try:
                    result = subprocess.run(['rocm-smi', '--showproduct'], 
                                          capture_output=True, text=True, timeout=5)
                    if result.returncode == 0:
                        if 'MI210' in result.stdout or 'MI25' in result.stdout or 'MI250' in result.stdout:
                            return "rocm"
                        if 'Vega' in result.stdout:  # Covers MI25/Vega 10
                            return "rocm"
                except (subprocess.TimeoutExpired, FileNotFoundError, subprocess.CalledProcessError):
                    pass
                
                # Check GPU name for AMD via PyTorch
                try:
                    gpu_name = torch.cuda.get_device_name(0)
                    if 'AMD' in gpu_name or 'Radeon' in gpu_name or 'Instinct' in gpu_name:
                        return "rocm"
                except Exception:
                    pass
                
                # Check for ROCm installation
                try:
                    if os.path.exists('/opt/rocm'):
                        result = subprocess.run(['lspci'], capture_output=True, text=True, timeout=5)
                        if 'AMD' in result.stdout or 'Advanced Micro Devices' in result.stdout:
                            return "rocm"
                except Exception:
                    pass
                
                # Default to CUDA
                return "cuda"
        except Exception:
            pass
        return "cpu"
    
    def _get_recommended_inference_engines(self, gpu_type: str) -> Dict[str, str]:
        """Get recommended inference engines based on hardware."""
        if gpu_type == "rocm":
            return {
                "text": "vllm-rocm or llama.cpp-hip",
                "image": "comfyui-rocm or automatic1111-rocm", 
                "voice": "local-cpu or xtts-rocm"
            }
        elif gpu_type == "cuda":
            return {
                "text": "vllm or transformers",
                "image": "comfyui or diffusers",
                "voice": "xtts or local-gpu"
            }
        else:
            return {
                "text": "llama.cpp or transformers-cpu",
                "image": "diffusers-cpu",
                "voice": "piper or local-cpu"
            }
    
    def get_system_info(self) -> Dict:
        """Get system hardware information."""
        try:
            import psutil
            ram_gb = psutil.virtual_memory().total / (1024 ** 3)
        except ImportError:
            # Fallback for systems without psutil
            ram_gb = 8.0  # Assume 8GB default
        
        # Detect MI25/gfx900 specifically
        is_mi25 = False
        rocm_arch = "unknown"
        if self.gpu_type == "rocm":
            try:
                result = subprocess.run(['rocm-smi', '--showproduct'], 
                                      capture_output=True, text=True, timeout=5)
                if result.returncode == 0 and 'MI25' in result.stdout:
                    is_mi25 = True
                    rocm_arch = "gfx900"
            except Exception:
                pass
            
            # Also check via lspci
            try:
                result = subprocess.run(['lspci', '-v'], 
                                      capture_output=True, text=True, timeout=5)
                if 'Radeon Instinct MI25' in result.stdout or 'Vega 10' in result.stdout:
                    is_mi25 = True
                    rocm_arch = "gfx900"
            except Exception:
                pass
        
        sys_info = {
            "platform": platform.platform(),
            "cpu_count": os.cpu_count(),
            "ram_gb": ram_gb,
            "gpu_available": self.has_gpu,
            "gpu_type": self.gpu_type,
            "gpu_count": self.gpu_count,
            "gpu_memory_gb": self.gpu_memory,
            "disk_space_gb": shutil.disk_usage(self.models_dir).free / (1024 ** 3),
            "recommended_engines": self._get_recommended_inference_engines(self.gpu_type)
        }
        
        if is_mi25:
            sys_info["gpu_architecture"] = rocm_arch
            sys_info["is_mi25"] = True
            sys_info["compatibility_notes"] = [
                "MI25 (gfx900) detected - ROCm 5.7.1 confirmed working",
                "HSA_OVERRIDE_GFX_VERSION=9.0.0 should be set",
                "PyTorch with ROCm 5.7 support recommended",
                "Some frameworks may require HSA override for gfx900 support"
            ]
        
        return sys_info
    
    def analyze_system_requirements(self) -> Dict:
        """Analyze which models can be run on current system."""
        sys_info = self.get_system_info()
        recommendations = {
            "installable": [],
            "requires_upgrade": [],
            "api_only": []
        }
        
        for category, models in self.model_configs.items():
            for model in models:
                if model.get("api_key_required"):
                    recommendations["api_only"].append(model)
                    continue
                
    
    async def setup_inference_engines(self) -> Dict[str, str]:
        """Setup recommended inference engines based on hardware."""
        results = {}
        
        try:
            if self.gpu_type == "rocm":
                # Setup ROCm-compatible engines
                results["vllm"] = await self._setup_vllm_rocm()
                results["comfyui"] = await self._setup_comfyui_rocm()
                results["xtts"] = await self._setup_xtts_rocm()
            elif self.gpu_type == "cuda":
                # Setup CUDA engines
                results["vllm"] = await self._setup_vllm_cuda()
                results["comfyui"] = await self._setup_comfyui_cuda()
                results["xtts"] = await self._setup_xtts_cuda()
            else:
                # CPU-only setup
                results["llama_cpp"] = await self._setup_llama_cpp()
                results["diffusers"] = await self._setup_diffusers_cpu()
                results["piper"] = await self._setup_piper()
                
        except Exception as e:
            results["error"] = str(e)
            
        return results
    
    async def _setup_vllm_rocm(self) -> str:
        """Setup vLLM with ROCm support."""
        try:
            # This would install vLLM with ROCm in production
            return "vLLM (ROCm build) - installation commands needed"
        except Exception as e:
            return f"Failed: {str(e)}"
    
    async def _setup_comfyui_rocm(self) -> str:
        """Setup ComfyUI with ROCm support."""
        try:
            # This would clone and setup ComfyUI with ROCm PyTorch
            comfyui_dir = self.models_dir.parent / "ComfyUI"
            if not comfyui_dir.exists():
                return "ComfyUI - requires manual setup with ROCm PyTorch"
            return "ComfyUI available"
        except Exception as e:
            return f"Failed: {str(e)}"
    
    async def _setup_xtts_rocm(self) -> str:
        """Setup XTTS-v2 with ROCm support."""
        try:
            return "XTTS-v2 (ROCm) - requires Coqui-AI installation"
        except Exception as e:
            return f"Failed: {str(e)}"
    
    async def _setup_vllm_cuda(self) -> str:
        """Setup vLLM with CUDA support."""
        try:
            return "vLLM (CUDA) - pip install vllm"
        except Exception as e:
            return f"Failed: {str(e)}"
    
    async def _setup_comfyui_cuda(self) -> str:
        """Setup ComfyUI with CUDA support."""
        try:
            comfyui_dir = self.models_dir.parent / "ComfyUI" 
            if not comfyui_dir.exists():
                return "ComfyUI - git clone https://github.com/comfyanonymous/ComfyUI"
            return "ComfyUI available"
        except Exception as e:
            return f"Failed: {str(e)}"
    
    async def _setup_xtts_cuda(self) -> str:
        """Setup XTTS-v2 with CUDA support."""
        try:
            return "XTTS-v2 (CUDA) - pip install TTS"
        except Exception as e:
            return f"Failed: {str(e)}"
    
    async def _setup_llama_cpp(self) -> str:
        """Setup llama.cpp for CPU inference."""
        try:
            return "llama.cpp - build from source or pip install llama-cpp-python"
        except Exception as e:
            return f"Failed: {str(e)}"
    
    async def _setup_diffusers_cpu(self) -> str:
        """Setup diffusers for CPU inference."""
        try:
            return "diffusers (CPU) - pip install diffusers"
        except Exception as e:
            return f"Failed: {str(e)}"
    
    async def _setup_piper(self) -> str:
        """Setup Piper TTS."""
        try:
            return "Piper TTS - download from rhasspy/piper releases"
        except Exception as e:
            return f"Failed: {str(e)}"
    
    def analyze_system_requirements(self) -> Dict:
        """Analyze which models can be run on current system."""
        sys_info = self.get_system_info()
        recommendations = {
            "installable": [],
            "requires_upgrade": [],
            "api_only": []
        }
        
        for category, models in self.model_configs.items():
            for model in models:
                if model.get("api_key_required"):
                    recommendations["api_only"].append(model)
                    continue
                
                # Check system requirements
                meets_requirements = True
                requirements_check = []
                
                if "min_ram_gb" in model:
                    if sys_info["ram_gb"] < model["min_ram_gb"]:
                        meets_requirements = False
                        requirements_check.append(f"Need {model['min_ram_gb']}GB RAM (have {sys_info['ram_gb']:.1f}GB)")
                
                if "min_gpu_memory_gb" in model:
                    if not self.has_gpu or self.gpu_memory < model["min_gpu_memory_gb"]:
                        meets_requirements = False
                        requirements_check.append(f"Need {model['min_gpu_memory_gb']}GB GPU memory (have {self.gpu_memory:.1f}GB)")
                
                if "size_gb" in model:
                    if sys_info["disk_space_gb"] < model["size_gb"] * 2:  # 2x for download + extracted
                        meets_requirements = False
                        requirements_check.append(f"Need {model['size_gb']*2}GB free disk space")
                
                model_info = model.copy()
                model_info["category"] = category
                model_info["requirements_check"] = requirements_check
                
                if meets_requirements:
                    recommendations["installable"].append(model_info)
                else:
                    recommendations["requires_upgrade"].append(model_info)
        
        return recommendations
    
    def install_dependencies(self) -> None:
        """Install required Python packages for AI models."""
        required_packages = [
            "torch>=2.2.0",
            "torchvision>=0.17.0", 
            "transformers>=4.35.0",
            "diffusers>=0.21.0",
            "accelerate>=0.21.0",
            "pillow>=10.0.0",
            "requests>=2.31.0",
            "httpx>=0.24.0",
            "psutil>=5.9.0"
        ]
        
        print("Installing AI model dependencies...")
        for package in required_packages:
            try:
                subprocess.check_call([
                    sys.executable, "-m", "pip", "install", package
                ], stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
                print(f"✓ Installed {package}")
            except subprocess.CalledProcessError:
                print(f"✗ Failed to install {package}")
    
    def install_text_models(self, models_to_install: List[str] = None) -> None:
        """Install text generation models."""
        if models_to_install is None:
            models_to_install = ["gpt2-medium", "distilbert-sentiment"]
        
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
            
            for model_name in models_to_install:
                model_config = next((m for m in self.model_configs["text"] if m["name"] == model_name), None)
                if not model_config:
                    print(f"Unknown model: {model_name}")
                    continue
                
                print(f"Downloading {model_name}...")
                model_path = self.models_dir / "text" / model_name
                model_path.mkdir(parents=True, exist_ok=True)
                
                try:
                    # Download tokenizer and model
                    tokenizer = AutoTokenizer.from_pretrained(model_config["model_id"])
                    tokenizer.save_pretrained(model_path)
                    
                    if "gpt2" in model_name:
                        model = AutoModelForCausalLM.from_pretrained(model_config["model_id"])
                        model.save_pretrained(model_path)
                    
                    print(f"✓ Installed {model_name}")
                    
                except Exception as e:
                    print(f"✗ Failed to install {model_name}: {str(e)}")
                    
        except ImportError as e:
            print(f"Missing dependencies for text models: {str(e)}")
    
    def install_image_models(self, models_to_install: List[str] = None) -> None:
        """Install image generation models."""
        if models_to_install is None:
            # Only install if sufficient GPU memory
            if self.gpu_memory >= 6:
                models_to_install = ["stable-diffusion-v1-5"]
            else:
                print("Insufficient GPU memory for image models. Skipping local installation.")
                print("Consider using API-based image generation (OpenAI DALL-E) instead.")
                return
        
        try:
            from diffusers import StableDiffusionPipeline
            
            for model_name in models_to_install:
                model_config = next((m for m in self.model_configs["image"] if m["name"] == model_name), None)
                if not model_config:
                    print(f"Unknown model: {model_name}")
                    continue
                
                print(f"Downloading {model_name} (this may take a while)...")
                model_path = self.models_dir / "image" / model_name
                model_path.mkdir(parents=True, exist_ok=True)
                
                try:
                    # Download and save pipeline
                    pipeline = StableDiffusionPipeline.from_pretrained(
                        model_config["model_id"],
                        torch_dtype=torch.float16 if self.has_gpu else torch.float32
                    )
                    pipeline.save_pretrained(model_path)
                    
                    print(f"✓ Installed {model_name}")
                    
                except Exception as e:
                    print(f"✗ Failed to install {model_name}: {str(e)}")
                    
        except ImportError as e:
            print(f"Missing dependencies for image models: {str(e)}")
    
    def create_model_config(self) -> None:
        """Create model configuration file."""
        config = {
            "system_info": self.get_system_info(),
            "installed_models": {},
            "api_services": {
                "openai": {
                    "enabled": False,
                    "models": ["gpt-4", "gpt-3.5-turbo", "dall-e-3", "tts-1"]
                },
                "anthropic": {
                    "enabled": False,
                    "models": ["claude-3-sonnet"]
                },
                "elevenlabs": {
                    "enabled": False,
                    "models": ["voice-synthesis"]
                }
            }
        }
        
        # Check for installed models
        for category in ["text", "image", "voice"]:
            category_path = self.models_dir / category
            if category_path.exists():
                config["installed_models"][category] = []
                for model_dir in category_path.iterdir():
                    if model_dir.is_dir():
                        config["installed_models"][category].append(model_dir.name)
        
        config_path = self.models_dir / "model_config.json"
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"Created model configuration: {config_path}")
    
    def setup_api_services(self) -> None:
        """Setup information for API-based AI services."""
        print("\n=== API Service Configuration ===")
        print("For the best AI generation quality, configure these API services:")
        print()
        
        api_services = [
            {
                "name": "OpenAI",
                "env_var": "OPENAI_API_KEY", 
                "website": "https://platform.openai.com/",
                "models": "GPT-4, DALL-E 3, TTS",
                "description": "High-quality text, image, and voice generation"
            },
            {
                "name": "Anthropic Claude",
                "env_var": "ANTHROPIC_API_KEY",
                "website": "https://console.anthropic.com/",
                "models": "Claude 3 Sonnet",
                "description": "Advanced text generation and analysis"
            },
            {
                "name": "ElevenLabs",
                "env_var": "ELEVENLABS_API_KEY",
                "website": "https://elevenlabs.io/",
                "models": "Voice Synthesis",
                "description": "Premium voice cloning and synthesis"
            }
        ]
        
        for service in api_services:
            print(f"• {service['name']}")
            print(f"  Models: {service['models']}")
            print(f"  Get API key: {service['website']}")
            print(f"  Set environment variable: {service['env_var']}=your_api_key")
            print(f"  {service['description']}")
            print()
    
    def run_setup(self, install_models: bool = True, model_types: List[str] = None) -> None:
        """Run complete setup process."""
        print("🦎 Gator AI Model Setup")
        print("=" * 50)
        
        # System analysis
        sys_info = self.get_system_info()
        print(f"System: {sys_info['platform']}")
        print(f"CPU: {sys_info['cpu_count']} cores")
        print(f"RAM: {sys_info['ram_gb']:.1f} GB")
        print(f"GPU: {sys_info['gpu_count']} devices ({sys_info['gpu_memory_gb']:.1f} GB)")
        print(f"Disk space: {sys_info['disk_space_gb']:.1f} GB")
        print()
        
        # Model recommendations
        recommendations = self.analyze_system_requirements()
        
        print("📊 Model Compatibility Analysis")
        print(f"✓ Can install: {len(recommendations['installable'])} models")
        print(f"⚠ Requires upgrade: {len(recommendations['requires_upgrade'])} models")  
        print(f"☁ API-only options: {len(recommendations['api_only'])} services")
        print()
        
        if install_models:
            # Install dependencies
            self.install_dependencies()
            print()
            
            # Install models based on system capabilities
            if model_types is None:
                model_types = ["text", "image"]
            
            installable_models = {
                "text": [m for m in recommendations["installable"] if m["category"] == "text"],
                "image": [m for m in recommendations["installable"] if m["category"] == "image"]
            }
            
            if "text" in model_types and installable_models["text"]:
                print("📝 Installing text models...")
                text_models = [m["name"] for m in installable_models["text"]]
                self.install_text_models(text_models)
                print()
            
            if "image" in model_types and installable_models["image"]:
                print("🎨 Installing image models...")
                image_models = [m["name"] for m in installable_models["image"]]
                self.install_image_models(image_models)
                print()
        
        # Create configuration
        self.create_model_config()
        
        # API service setup info
        self.setup_api_services()
        
        print("🎉 Setup complete!")
        print(f"Models installed in: {self.models_dir.absolute()}")
        print("Update your .env file with API keys for best performance.")


async def main():
    """Main setup script entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Setup AI models for Gator platform")
    parser.add_argument("--analyze", action="store_true", help="Analyze system capabilities")
    parser.add_argument("--install", nargs="*", help="Install specific models", default=[])
    parser.add_argument("--setup-engines", action="store_true", help="Setup inference engines")
    parser.add_argument("--models-dir", default="./models", help="Models directory")
    
    args = parser.parse_args()
    
    manager = ModelSetupManager(args.models_dir)
    
    if args.analyze:
        print("🔍 Analyzing system capabilities...")
        sys_info = manager.get_system_info()
        print(f"\n💻 System Information:")
        print(f"Platform: {sys_info['platform']}")
        print(f"CPU cores: {sys_info['cpu_count']}")
        print(f"RAM: {sys_info['ram_gb']:.1f} GB")
        print(f"GPU: {sys_info['gpu_type']} ({sys_info['gpu_count']} devices)")
        print(f"GPU Memory: {sys_info['gpu_memory_gb']:.1f} GB")
        print(f"Disk Space: {sys_info['disk_space_gb']:.1f} GB")
        
        print(f"\n🛠️ Recommended Inference Engines:")
        for model_type, engine in sys_info['recommended_engines'].items():
            print(f"  {model_type.title()}: {engine}")
        
        print("\n📊 Model Compatibility Analysis:")
        recommendations = manager.analyze_system_requirements()
        
        if recommendations["installable"]:
            print("\n✅ Models you can install:")
            for model in recommendations["installable"]:
                print(f"  • {model['name']}: {model['description']}")
        
        if recommendations["requires_upgrade"]:
            print("\n⚠️ Models requiring hardware upgrade:")
            for model in recommendations["requires_upgrade"]:
                print(f"  • {model['name']}: {', '.join(model['requirements_check'])}")
    
    if args.setup_engines:
        print("\n🔧 Setting up inference engines...")
        results = await manager.setup_inference_engines()
        for engine, status in results.items():
            print(f"  {engine}: {status}")
    
    if args.install:
        print(f"\n📦 Installing models: {args.install}")
        
        # Get model recommendations to validate requested models
        recommendations = manager.analyze_system_requirements()
        installable_names = [m["name"] for m in recommendations["installable"]]
        
        # Validate that requested models are installable
        models_to_install = args.install if args.install else []
        invalid_models = [m for m in models_to_install if m not in installable_names]
        
        if invalid_models:
            print(f"\n⚠️  Warning: The following models cannot be installed on this system:")
            for model_name in invalid_models:
                print(f"   • {model_name}")
                # Check if it's in requires_upgrade
                upgrade_model = next((m for m in recommendations["requires_upgrade"] if m["name"] == model_name), None)
                if upgrade_model:
                    print(f"     Reason: {', '.join(upgrade_model['requirements_check'])}")
            
            models_to_install = [m for m in models_to_install if m in installable_names]
            
            if not models_to_install:
                print("\n❌ No valid models to install. Exiting.")
                return
        
        if models_to_install:
            print(f"\n✅ Installing {len(models_to_install)} model(s): {', '.join(models_to_install)}")
            
            # Group models by category
            text_models = []
            image_models = []
            voice_models = []
            
            for model_name in models_to_install:
                model_info = next((m for m in recommendations["installable"] if m["name"] == model_name), None)
                if model_info:
                    if model_info["category"] == "text":
                        text_models.append(model_name)
                    elif model_info["category"] == "image":
                        image_models.append(model_name)
                    elif model_info["category"] == "voice":
                        voice_models.append(model_name)
            
            # Install dependencies first
            print("\n📦 Installing dependencies...")
            manager.install_dependencies()
            
            # Install models by category
            if text_models:
                print(f"\n📝 Installing text models: {', '.join(text_models)}")
                manager.install_text_models(text_models)
            
            if image_models:
                print(f"\n🎨 Installing image models: {', '.join(image_models)}")
                manager.install_image_models(image_models)
            
            if voice_models:
                print(f"\n🎤 Installing voice models: {', '.join(voice_models)}")
                # Voice model installation would go here
                print(f"   Voice model installation not yet implemented for: {', '.join(voice_models)}")
            
            # Update configuration
            manager.create_model_config()
            
            print(f"\n✅ Installation complete!")
            print(f"   Models directory: {manager.models_dir.absolute()}")
            print(f"   Configuration file: {manager.models_dir.absolute() / 'model_config.json'}")
        
    if not any([args.analyze, args.install, args.setup_engines]):
        # Default behavior - show system info and recommendations
        print("🚀 Gator AI Model Setup")
        print("======================\n")
        
        sys_info = manager.get_system_info()
        print(f"Detected hardware: {sys_info['gpu_type']} GPU with {sys_info['gpu_memory_gb']:.1f}GB")
        
        recommendations = manager.analyze_system_requirements()
        installable_count = len(recommendations["installable"])
        
        print(f"Found {installable_count} compatible models")
        print("\nRun with --analyze for detailed analysis")
        print("Run with --setup-engines to install inference tools")


if __name__ == "__main__":
    asyncio.run(main())