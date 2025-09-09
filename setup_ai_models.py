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
    import requests
except ImportError:
    print("Installing required packages...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "torch", "requests"])
    import torch
    import requests


class ModelSetupManager:
    """Manages AI model setup and configuration."""
    
    def __init__(self, models_dir: str = "./models"):
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(exist_ok=True)
        
        # Hardware detection
        self.has_gpu = torch.cuda.is_available()
        self.gpu_count = torch.cuda.device_count() if self.has_gpu else 0
        self.gpu_memory = self._get_gpu_memory() if self.has_gpu else 0
        
        # Model configurations
        self.model_configs = {
            "text": [
                {
                    "name": "gpt2-medium",
                    "model_id": "gpt2-medium", 
                    "size_gb": 1.5,
                    "min_ram_gb": 4,
                    "type": "transformers",
                    "description": "Medium-sized GPT-2 model for text generation"
                },
                {
                    "name": "distilbert-sentiment",
                    "model_id": "distilbert-base-uncased-finetuned-sst-2-english",
                    "size_gb": 0.3,
                    "min_ram_gb": 2,
                    "type": "transformers",
                    "description": "Sentiment analysis model"
                }
            ],
            "image": [
                {
                    "name": "stable-diffusion-v1-5",
                    "model_id": "runwayml/stable-diffusion-v1-5",
                    "size_gb": 4.0,
                    "min_ram_gb": 8,
                    "min_gpu_memory_gb": 6,
                    "type": "diffusers",
                    "description": "Stable Diffusion 1.5 for image generation"
                },
                {
                    "name": "stable-diffusion-xl-base",
                    "model_id": "stabilityai/stable-diffusion-xl-base-1.0",
                    "size_gb": 6.9,
                    "min_ram_gb": 16,
                    "min_gpu_memory_gb": 10,
                    "type": "diffusers", 
                    "description": "Stable Diffusion XL for higher quality images"
                }
            ],
            "voice": [
                {
                    "name": "tts-1",
                    "provider": "openai",
                    "api_key_required": True,
                    "description": "OpenAI Text-to-Speech API"
                },
                {
                    "name": "elevenlabs-tts",
                    "provider": "elevenlabs",
                    "api_key_required": True,
                    "description": "ElevenLabs Voice Synthesis API"
                }
            ]
        }
        
    def _get_gpu_memory(self) -> float:
        """Get total GPU memory in GB."""
        if not self.has_gpu:
            return 0.0
        
        total_memory = 0
        for i in range(self.gpu_count):
            props = torch.cuda.get_device_properties(i)
            total_memory += props.total_memory
        
        return total_memory / (1024 ** 3)  # Convert to GB
    
    def get_system_info(self) -> Dict:
        """Get system hardware information."""
        import psutil
        
        return {
            "platform": platform.platform(),
            "cpu_count": os.cpu_count(),
            "ram_gb": psutil.virtual_memory().total / (1024 ** 3),
            "gpu_available": self.has_gpu,
            "gpu_count": self.gpu_count,
            "gpu_memory_gb": self.gpu_memory,
            "disk_space_gb": shutil.disk_usage(self.models_dir).free / (1024 ** 3)
        }
    
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
            "torch>=2.0.0",
            "torchvision>=0.15.0", 
            "transformers>=4.30.0",
            "diffusers>=0.18.0",
            "accelerate>=0.20.0",
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


def main():
    """Main setup script entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Setup AI models for Gator")
    parser.add_argument("--models-dir", default="./models", help="Directory to store models")
    parser.add_argument("--no-install", action="store_true", help="Skip model installation")
    parser.add_argument("--types", nargs="+", choices=["text", "image", "voice"], 
                       default=["text", "image"], help="Model types to install")
    
    args = parser.parse_args()
    
    setup_manager = ModelSetupManager(args.models_dir)
    setup_manager.run_setup(
        install_models=not args.no_install,
        model_types=args.types
    )


if __name__ == "__main__":
    main()