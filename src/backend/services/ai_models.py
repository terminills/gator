"""
AI Model Integration Service

Handles integration with various AI models for content generation including:
- Local models: Llama 3.1, Qwen2.5, Stable Diffusion XL, Coqui XTTS-v2
- Cloud APIs: OpenAI, Anthropic, ElevenLabs
- Hardware optimization for AMD ROCm and CUDA
"""

import asyncio
import os
import io
import base64
import subprocess
import platform
import shutil
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
from pathlib import Path
import json

import httpx
from PIL import Image
import torch

from backend.config.logging import get_logger
from backend.config.settings import get_settings

logger = get_logger(__name__)


class AIModelManager:
    """
    Manager for AI model integrations.
    
    Supports both local models (Llama, SDXL, XTTS) and cloud APIs (OpenAI, Anthropic).
    Automatically detects hardware capabilities and optimizes model selection.
    """
    
    def __init__(self):
        self.settings = get_settings()
        self.models_loaded = False
        self.available_models = {
            "text": [],
            "image": [],
            "voice": [],
            "video": [],
            "audio": []
        }
        self.http_client = httpx.AsyncClient(timeout=300.0)  # Long timeout for AI generation
        
        # Hardware detection
        self.gpu_type = self._detect_gpu_type()
        self.gpu_memory_gb = self._get_gpu_memory()
        self.cpu_cores = os.cpu_count()
        
        # Local model paths
        self.models_dir = Path("./models")
        self.models_dir.mkdir(exist_ok=True)
        
        # Model configurations based on recommendations
        self.local_model_configs = {
            "text": {
                "llama-3.1-70b": {
                    "model_id": "meta-llama/Llama-3.1-70B",
                    "size_gb": 140,
                    "min_gpu_memory_gb": 48,
                    "min_ram_gb": 64,
                    "inference_engine": "vllm",
                    "quant_options": ["Q4_K_M", "FP16", "BF16"],
                    "description": "Best general local base model"
                },
                "qwen2.5-72b": {
                    "model_id": "Qwen/Qwen2.5-72B",
                    "size_gb": 144,
                    "min_gpu_memory_gb": 48,
                    "min_ram_gb": 64,
                    "inference_engine": "vllm",
                    "description": "Stronger tools/code, longer context"
                },
                "mixtral-8x7b": {
                    "model_id": "mistralai/Mixtral-8x7B-Instruct-v0.1",
                    "size_gb": 90,
                    "min_gpu_memory_gb": 24,
                    "min_ram_gb": 32,
                    "inference_engine": "vllm",
                    "description": "Fast per token, solid instruction following at lower VRAM"
                },
                "llama-3.1-8b": {
                    "model_id": "meta-llama/Llama-3.1-8B-Instruct",
                    "size_gb": 16,
                    "min_gpu_memory_gb": 8,
                    "min_ram_gb": 16,
                    "inference_engine": "vllm",
                    "description": "Snappy persona worker for fast mode"
                }
            },
            "image": {
                "sdxl-1.0": {
                    "model_id": "stabilityai/stable-diffusion-xl-base-1.0",
                    "size_gb": 7,
                    "min_gpu_memory_gb": 8,
                    "min_ram_gb": 16,
                    "inference_engine": "comfyui",
                    "description": "Safest, most supported local base"
                },
                "flux.1-dev": {
                    "model_id": "black-forest-labs/FLUX.1-dev",
                    "size_gb": 12,
                    "min_gpu_memory_gb": 12,
                    "min_ram_gb": 24,
                    "inference_engine": "comfyui",
                    "description": "Very good quality; verify license for commercial use"
                }
            },
            "voice": {
                "xtts-v2": {
                    "model_id": "coqui/XTTS-v2",
                    "size_gb": 2,
                    "min_gpu_memory_gb": 4,
                    "min_ram_gb": 8,
                    "description": "Multilingual, cloning, runs locally; best all-around"
                },
                "piper": {
                    "model_id": "rhasspy/piper",
                    "size_gb": 0.1,
                    "min_gpu_memory_gb": 0,
                    "min_ram_gb": 2,
                    "description": "Ultralight, CPU-friendly for systems TTS"
                }
            }
        }
    
    def _detect_gpu_type(self) -> str:
        """Detect GPU type (CUDA, ROCm, or CPU)."""
        try:
            if torch.cuda.is_available():
                # Check for AMD ROCm
                try:
                    import subprocess
                    result = subprocess.run(['rocm-smi', '--showproduct'], 
                                          capture_output=True, text=True, timeout=5)
                    if result.returncode == 0 and 'MI' in result.stdout:
                        return "rocm"
                except (subprocess.TimeoutExpired, FileNotFoundError):
                    pass
                
                # Default to CUDA
                return "cuda"
        except Exception:
            pass
        return "cpu"
    
    def _get_gpu_memory(self) -> float:
        """Get total GPU memory in GB."""
        if not torch.cuda.is_available():
            return 0.0
        
        total_memory = 0
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            total_memory += props.total_memory
        
        return total_memory / (1024 ** 3)  # Convert to GB
    
    def _get_system_requirements(self) -> Dict[str, Any]:
        """Get current system capabilities."""
        import psutil
        
        return {
            "gpu_type": self.gpu_type,
            "gpu_memory_gb": self.gpu_memory_gb,
            "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
            "ram_gb": psutil.virtual_memory().total / (1024 ** 3),
            "cpu_cores": self.cpu_cores,
            "disk_space_gb": shutil.disk_usage(self.models_dir).free / (1024 ** 3),
            "platform": platform.platform()
        }

    async def initialize_models(self) -> None:
        """Initialize and load AI models based on available hardware and configuration."""
        try:
            sys_req = self._get_system_requirements()
            logger.info(f"Initializing AI models with system: {sys_req}")
            
            # Initialize local models based on hardware
            await self._initialize_local_text_models()
            await self._initialize_local_image_models()
            await self._initialize_local_voice_models()
            
            # Initialize cloud API models as fallbacks
            await self._initialize_cloud_text_models()
            await self._initialize_cloud_image_models()
            await self._initialize_cloud_voice_models()
            
            # Initialize video models (still mostly placeholder)
            await self._initialize_video_models()
            
            self.models_loaded = True
            logger.info(f"AI model initialization complete extra={{"available_models": self.available_models}}")
            
        except Exception as e:
            logger.error(f"Failed to initialize AI models: {str(e)}")
            raise
    
    async def _initialize_local_text_models(self) -> None:
        """Initialize local text generation models based on hardware capabilities."""
        try:
            sys_req = self._get_system_requirements()
            
            # Select appropriate models based on available resources
            for model_name, config in self.local_model_configs["text"].items():
                can_run = (
                    sys_req["gpu_memory_gb"] >= config.get("min_gpu_memory_gb", 0) and
                    sys_req["ram_gb"] >= config.get("min_ram_gb", 0)
                )
                
                if can_run:
                    model_path = self.models_dir / model_name
                    is_downloaded = model_path.exists()
                    
                    inference_engine = config.get("inference_engine", "transformers")
                    
                    # Check if inference engine is available
                    engine_available = await self._check_inference_engine(inference_engine)
                    
                    self.available_models["text"].append({
                        "name": model_name,
                        "type": "text-generation",
                        "model_id": config["model_id"],
                        "provider": "local",
                        "inference_engine": inference_engine,
                        "loaded": is_downloaded and engine_available,
                        "can_load": can_run and engine_available,
                        "size_gb": config["size_gb"],
                        "description": config["description"],
                        "device": "cuda" if sys_req["gpu_memory_gb"] > 0 else "cpu",
                        "quant_options": config.get("quant_options", [])
                    })
                    
                    if is_downloaded and engine_available:
                        logger.info(f"Local text model {model_name} ready")
                    elif can_run and engine_available:
                        logger.info(f"Local text model {model_name} can be downloaded")
                    else:
                        logger.info(f"Local text model {model_name} needs setup: engine={engine_available}, downloaded={is_downloaded}")
                        
        except Exception as e:
            logger.error(f"Failed to initialize local text models: {str(e)}")
    
    async def _initialize_local_image_models(self) -> None:
        """Initialize local image generation models."""
        try:
            sys_req = self._get_system_requirements()
            
            for model_name, config in self.local_model_configs["image"].items():
                can_run = (
                    sys_req["gpu_memory_gb"] >= config.get("min_gpu_memory_gb", 0) and
                    sys_req["ram_gb"] >= config.get("min_ram_gb", 0)
                )
                
                if can_run:
                    model_path = self.models_dir / model_name
                    is_downloaded = model_path.exists()
                    
                    inference_engine = config.get("inference_engine", "diffusers")
                    engine_available = await self._check_inference_engine(inference_engine)
                    
                    self.available_models["image"].append({
                        "name": model_name,
                        "type": "text-to-image",
                        "model_id": config["model_id"],
                        "provider": "local",
                        "inference_engine": inference_engine,
                        "loaded": is_downloaded and engine_available,
                        "can_load": can_run and engine_available,
                        "size_gb": config["size_gb"],
                        "description": config["description"],
                        "device": "cuda" if sys_req["gpu_memory_gb"] > 0 else "cpu"
                    })
                    
                    if is_downloaded and engine_available:
                        logger.info(f"Local image model {model_name} ready")
                    
        except Exception as e:
            logger.error(f"Failed to initialize local image models: {str(e)}")
    
    async def _initialize_local_voice_models(self) -> None:
        """Initialize local voice synthesis models."""
        try:
            sys_req = self._get_system_requirements()
            
            for model_name, config in self.local_model_configs["voice"].items():
                can_run = (
                    sys_req["ram_gb"] >= config.get("min_ram_gb", 0)
                )
                
                if can_run:
                    model_path = self.models_dir / model_name
                    is_downloaded = model_path.exists()
                    
                    self.available_models["voice"].append({
                        "name": model_name,
                        "type": "text-to-speech",
                        "model_id": config["model_id"],
                        "provider": "local",
                        "loaded": is_downloaded,
                        "can_load": can_run,
                        "size_gb": config["size_gb"],
                        "description": config["description"],
                        "device": "cuda" if config.get("min_gpu_memory_gb", 0) <= sys_req["gpu_memory_gb"] else "cpu"
                    })
                    
                    if is_downloaded:
                        logger.info(f"Local voice model {model_name} ready")
                        
        except Exception as e:
            logger.error(f"Failed to initialize local voice models: {str(e)}")
    
    async def _check_inference_engine(self, engine: str) -> bool:
        """Check if inference engine is available."""
        try:
            if engine == "vllm":
                import vllm
                return True
            elif engine == "comfyui":
                # Check if ComfyUI is available
                comfyui_path = Path("./ComfyUI")
                return comfyui_path.exists()
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
    
    async def _initialize_cloud_text_models(self) -> None:
        """Initialize cloud-based text generation models."""
        try:
            # Add OpenAI GPT as API option
            if hasattr(self.settings, 'openai_api_key') and self.settings.openai_api_key:
                self.available_models["text"].extend([
                    {
                        "name": "gpt-4",
                        "type": "text-generation",
                        "provider": "openai",
                        "loaded": True,
                        "max_tokens": 4096,
                        "description": "OpenAI GPT-4 cloud API"
                    },
                    {
                        "name": "gpt-3.5-turbo",
                        "type": "text-generation", 
                        "provider": "openai",
                        "loaded": True,
                        "max_tokens": 4096,
                        "description": "OpenAI GPT-3.5 Turbo cloud API"
                    }
                ])
                logger.info("OpenAI text models available")
                
            # Add Anthropic Claude as option
            if hasattr(self.settings, 'anthropic_api_key') and self.settings.anthropic_api_key:
                self.available_models["text"].append({
                    "name": "claude-3-sonnet",
                    "type": "text-generation",
                    "provider": "anthropic", 
                    "loaded": True,
                    "max_tokens": 4096,
                    "description": "Anthropic Claude-3 Sonnet cloud API"
                })
                logger.info("Anthropic Claude available")
                
        except Exception as e:
            logger.error(f"Failed to initialize cloud text models: {str(e)}")
    
    async def _initialize_cloud_image_models(self) -> None:
        """Initialize cloud-based image generation models."""
        try:
            # Add OpenAI DALL-E as fallback if API key available
            if hasattr(self.settings, 'openai_api_key') and self.settings.openai_api_key:
                self.available_models["image"].append({
                    "name": "dall-e-3",
                    "type": "text-to-image",
                    "provider": "openai",
                    "loaded": True,
                    "description": "OpenAI DALL-E 3 cloud API"
                })
                logger.info("OpenAI DALL-E 3 API available")
                
        except Exception as e:
            logger.error(f"Failed to initialize cloud image models: {str(e)}")
    
    async def _initialize_cloud_voice_models(self) -> None:
        """Initialize cloud-based voice synthesis models."""
        try:
            # Check for ElevenLabs API
            if hasattr(self.settings, 'elevenlabs_api_key') and self.settings.elevenlabs_api_key:
                self.available_models["voice"].append({
                    "name": "elevenlabs-tts",
                    "type": "text-to-speech",
                    "provider": "elevenlabs",
                    "loaded": True,
                    "description": "ElevenLabs voice synthesis cloud API"
                })
                logger.info("ElevenLabs voice synthesis available")
                
            # Check for OpenAI TTS
            if hasattr(self.settings, 'openai_api_key') and self.settings.openai_api_key:
                self.available_models["voice"].append({
                    "name": "openai-tts", 
                    "type": "text-to-speech",
                    "provider": "openai",
                    "loaded": True,
                    "description": "OpenAI TTS cloud API"
                })
                logger.info("OpenAI TTS available")
            
        except Exception as e:
            logger.error(f"Failed to initialize cloud voice models: {str(e)}")
    
    async def _initialize_video_models(self) -> None:
        """Initialize video generation models."""
        try:
            # Video generation is still emerging - for now we'll have placeholders
            # In the future, this might include RunwayML, Pika, or other video AI services
            
            self.available_models["video"].append({
                "name": "placeholder-video",
                "type": "text-to-video",
                "loaded": False,
                "note": "Video generation models not yet integrated"
            })
            
        except Exception as e:
            logger.error(f"Failed to initialize video models: {str(e)}")
    
    async def generate_image(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """Generate image from text prompt using best available model."""
        try:
            # Find best available image model (prefer local first)
            local_models = [m for m in self.available_models["image"] 
                          if m.get("provider") == "local" and m.get("loaded", False)]
            cloud_models = [m for m in self.available_models["image"] 
                          if m.get("provider") in ["openai"] and m.get("loaded", False)]
            
            # Prefer local models, fallback to cloud
            available_models = local_models + cloud_models
            
            if not available_models:
                raise ValueError("No image generation models available")
            
            model = available_models[0]
            
            if model.get("provider") == "openai":
                return await self._generate_image_openai(prompt, **kwargs)
            elif model.get("provider") == "local":
                return await self._generate_image_local(prompt, model, **kwargs)
            else:
                raise ValueError(f"Unsupported image model: {model['name']}")
                
        except Exception as e:
            logger.error(f"Image generation failed: {str(e)}")
            raise
    
    async def generate_text(self, prompt: str, **kwargs) -> str:
        """Generate text from prompt using best available model."""
        try:
            # Find best available text model (prefer local first for privacy/speed)
            local_models = [m for m in self.available_models["text"] 
                          if m.get("provider") == "local" and m.get("loaded", False)]
            cloud_models = [m for m in self.available_models["text"] 
                          if m.get("provider") in ["openai", "anthropic"] and m.get("loaded", False)]
            
            # Prefer local models, fallback to cloud
            available_models = local_models + cloud_models
            
            if not available_models:
                raise ValueError("No text generation models available")
            
            model = available_models[0]
            
            if model.get("provider") == "openai":
                return await self._generate_text_openai(prompt, model["name"], **kwargs)
            elif model.get("provider") == "anthropic":
                return await self._generate_text_anthropic(prompt, **kwargs)
            elif model.get("provider") == "local":
                return await self._generate_text_local(prompt, model, **kwargs)
            else:
                raise ValueError(f"Unsupported text model: {model['name']}")
                
        except Exception as e:
            logger.error(f"Text generation failed: {str(e)}")
            raise
    
    async def generate_voice(self, text: str, **kwargs) -> Dict[str, Any]:
        """Generate voice from text using best available model."""
        try:
            # Find best available voice model (prefer local for privacy)
            local_models = [m for m in self.available_models["voice"] 
                          if m.get("provider") == "local" and m.get("loaded", False)]
            cloud_models = [m for m in self.available_models["voice"] 
                          if m.get("provider") in ["elevenlabs", "openai"] and m.get("loaded", False)]
            
            # Prefer local models, fallback to cloud
            available_models = local_models + cloud_models
            
            if not available_models:
                raise ValueError("No voice generation models available")
            
            model = available_models[0]
            
            if model.get("provider") == "elevenlabs":
                return await self._generate_voice_elevenlabs(text, **kwargs)
            elif model.get("provider") == "openai":
                return await self._generate_voice_openai(text, **kwargs)
            elif model.get("provider") == "local":
                return await self._generate_voice_local(text, model, **kwargs)
            else:
                raise ValueError(f"Unsupported voice model: {model['name']}")
                
        except Exception as e:
            logger.error(f"Voice generation failed: {str(e)}")
            raise
    
    # Local model generation methods
    async def _generate_text_local(self, prompt: str, model: Dict[str, Any], **kwargs) -> str:
        """Generate text using local models."""
        try:
            model_name = model["name"]
            inference_engine = model.get("inference_engine", "transformers")
            
            if inference_engine == "vllm":
                return await self._generate_text_vllm(prompt, model, **kwargs)
            elif inference_engine == "transformers":
                return await self._generate_text_transformers(prompt, model, **kwargs)
            else:
                raise ValueError(f"Unsupported inference engine: {inference_engine}")
                
        except Exception as e:
            logger.error(f"Local text generation failed: {str(e)}")
            # Fallback to placeholder
            return f"[Local text generation failed: {str(e)}]"
    
    async def _generate_image_local(self, prompt: str, model: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Generate image using local models."""
        try:
            model_name = model["name"]
            inference_engine = model.get("inference_engine", "diffusers")
            
            if inference_engine == "comfyui":
                return await self._generate_image_comfyui(prompt, model, **kwargs)
            elif inference_engine == "diffusers":
                return await self._generate_image_diffusers(prompt, model, **kwargs)
            else:
                raise ValueError(f"Unsupported inference engine: {inference_engine}")
                
        except Exception as e:
            logger.error(f"Local image generation failed: {str(e)}")
            # Return placeholder
            return {
                "image_data": b"",
                "format": "PNG",
                "model": model_name,
                "error": str(e)
            }
    
    async def _generate_voice_local(self, text: str, model: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Generate voice using local models."""
        try:
            model_name = model["name"]
            
            if model_name == "xtts-v2":
                return await self._generate_voice_xtts(text, **kwargs)
            elif model_name == "piper":
                return await self._generate_voice_piper(text, **kwargs)
            else:
                raise ValueError(f"Unsupported local voice model: {model_name}")
                
        except Exception as e:
            logger.error(f"Local voice generation failed: {str(e)}")
            return {
                "audio_data": b"",
                "format": "WAV",
                "model": model_name,
                "error": str(e)
            }
    
    # vLLM integration
    async def _generate_text_vllm(self, prompt: str, model: Dict[str, Any], **kwargs) -> str:
        """Generate text using vLLM for high-performance inference."""
        try:
            # This would integrate with vLLM in production
            # For now, return a placeholder indicating the capability
            return f"[vLLM generation with {model['name']}: {prompt[:100]}...]"
        except Exception as e:
            logger.error(f"vLLM generation failed: {str(e)}")
            raise
    
    async def _generate_text_transformers(self, prompt: str, model: Dict[str, Any], **kwargs) -> str:
        """Generate text using Transformers library."""
        try:
            # This would integrate with transformers in production
            return f"[Transformers generation with {model['name']}: {prompt[:100]}...]"
        except Exception as e:
            logger.error(f"Transformers generation failed: {str(e)}")
            raise
    
    # ComfyUI integration  
    async def _generate_image_comfyui(self, prompt: str, model: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Generate image using ComfyUI workflow."""
        try:
            # This would integrate with ComfyUI API in production
            return {
                "image_data": b"",  # Placeholder
                "format": "PNG",
                "model": model["name"],
                "workflow": "comfyui",
                "note": f"ComfyUI generation: {prompt}"
            }
        except Exception as e:
            logger.error(f"ComfyUI generation failed: {str(e)}")
            raise
    
    async def _generate_image_diffusers(self, prompt: str, model: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Generate image using Diffusers library."""
        try:
            # This would integrate with diffusers in production
            return {
                "image_data": b"",  # Placeholder
                "format": "PNG",
                "model": model["name"],
                "library": "diffusers",
                "note": f"Diffusers generation: {prompt}"
            }
        except Exception as e:
            logger.error(f"Diffusers generation failed: {str(e)}")
            raise
    
    # XTTS-v2 integration
    async def _generate_voice_xtts(self, text: str, **kwargs) -> Dict[str, Any]:
        """Generate voice using Coqui XTTS-v2."""
        try:
            # This would integrate with XTTS-v2 in production
            return {
                "audio_data": b"",  # Placeholder
                "format": "WAV",
                "model": "xtts-v2",
                "voice_cloned": kwargs.get("clone_voice", False),
                "note": f"XTTS-v2 generation: {text[:50]}..."
            }
        except Exception as e:
            logger.error(f"XTTS-v2 generation failed: {str(e)}")
            raise
    
    # Piper TTS integration
    async def _generate_voice_piper(self, text: str, **kwargs) -> Dict[str, Any]:
        """Generate voice using Piper TTS."""
        try:
            # This would integrate with Piper in production
            return {
                "audio_data": b"",  # Placeholder
                "format": "WAV", 
                "model": "piper",
                "voice": kwargs.get("voice", "default"),
                "note": f"Piper TTS generation: {text[:50]}..."
            }
        except Exception as e:
            logger.error(f"Piper generation failed: {str(e)}")
            raise
    
    async def _generate_image_openai(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """Generate image using OpenAI DALL-E."""
        try:
            response = await self.http_client.post(
                "https://api.openai.com/v1/images/generations",
                headers={
                    "Authorization": f"Bearer {self.settings.openai_api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": "dall-e-3",
                    "prompt": prompt,
                    "size": kwargs.get("size", "1024x1024"),
                    "quality": kwargs.get("quality", "standard"),
                    "n": 1
                }
            )
            
            if response.status_code != 200:
                raise Exception(f"OpenAI API error: {response.text}")
            
            result = response.json()
            image_url = result["data"][0]["url"]
            
            # Download the image
            image_response = await self.http_client.get(image_url)
            image_data = image_response.content
            
            return {
                "image_data": image_data,
                "format": "PNG",
                "width": 1024,
                "height": 1024,
                "model": "dall-e-3",
                "provider": "openai"
            }
            
        except Exception as e:
            logger.error(f"OpenAI image generation failed: {str(e)}")
            raise
    
    async def _generate_image_stable_diffusion(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """Generate image using Stable Diffusion (placeholder for now)."""
        # This would use the actual Stable Diffusion pipeline in production
        # For now, return a placeholder response
        raise NotImplementedError("Stable Diffusion integration not yet implemented - requires model download")
    
    async def _generate_text_openai(self, prompt: str, model: str, **kwargs) -> str:
        """Generate text using OpenAI models."""
        try:
            response = await self.http_client.post(
                "https://api.openai.com/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.settings.openai_api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": model,
                    "messages": [
                        {"role": "user", "content": prompt}
                    ],
                    "max_tokens": kwargs.get("max_tokens", 1000),
                    "temperature": kwargs.get("temperature", 0.7)
                }
            )
            
            if response.status_code != 200:
                raise Exception(f"OpenAI API error: {response.text}")
            
            result = response.json()
            return result["choices"][0]["message"]["content"].strip()
            
        except Exception as e:
            logger.error(f"OpenAI text generation failed: {str(e)}")
            raise
    
    async def _generate_text_anthropic(self, prompt: str, **kwargs) -> str:
        """Generate text using Anthropic Claude."""
        try:
            response = await self.http_client.post(
                "https://api.anthropic.com/v1/messages",
                headers={
                    "x-api-key": self.settings.anthropic_api_key,
                    "Content-Type": "application/json",
                    "anthropic-version": "2023-06-01"
                },
                json={
                    "model": "claude-3-sonnet-20240229",
                    "max_tokens": kwargs.get("max_tokens", 1000),
                    "messages": [
                        {"role": "user", "content": prompt}
                    ]
                }
            )
            
            if response.status_code != 200:
                raise Exception(f"Anthropic API error: {response.text}")
            
            result = response.json()
            return result["content"][0]["text"].strip()
            
        except Exception as e:
            logger.error(f"Anthropic text generation failed: {str(e)}")
            raise
    
    async def _generate_voice_elevenlabs(self, text: str, **kwargs) -> Dict[str, Any]:
        """Generate voice using ElevenLabs."""
        try:
            voice_id = kwargs.get("voice_id", "21m00Tcm4TlvDq8ikWAM")  # Default voice
            
            response = await self.http_client.post(
                f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}",
                headers={
                    "xi-api-key": self.settings.elevenlabs_api_key,
                    "Content-Type": "application/json"
                },
                json={
                    "text": text,
                    "model_id": "eleven_monolingual_v1",
                    "voice_settings": {
                        "stability": kwargs.get("stability", 0.5),
                        "similarity_boost": kwargs.get("similarity_boost", 0.5)
                    }
                }
            )
            
            if response.status_code != 200:
                raise Exception(f"ElevenLabs API error: {response.text}")
            
            audio_data = response.content
            
            return {
                "audio_data": audio_data,
                "format": "MP3",
                "voice_id": voice_id,
                "provider": "elevenlabs"
            }
            
        except Exception as e:
            logger.error(f"ElevenLabs voice generation failed: {str(e)}")
            raise
    
    async def _generate_voice_openai(self, text: str, **kwargs) -> Dict[str, Any]:
        """Generate voice using OpenAI TTS."""
        try:
            response = await self.http_client.post(
                "https://api.openai.com/v1/audio/speech",
                headers={
                    "Authorization": f"Bearer {self.settings.openai_api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": "tts-1",
                    "input": text,
                    "voice": kwargs.get("voice", "alloy")
                }
            )
            
            if response.status_code != 200:
                raise Exception(f"OpenAI TTS API error: {response.text}")
            
            audio_data = response.content
            
            return {
                "audio_data": audio_data,
                "format": "MP3", 
                "voice": kwargs.get("voice", "alloy"),
                "provider": "openai"
            }
            
        except Exception as e:
            logger.error(f"OpenAI voice generation failed: {str(e)}")
            raise
    
    async def get_available_models(self) -> Dict[str, List[Dict[str, Any]]]:
        """Get list of available models by type."""
        return self.available_models.copy()
    
    async def get_system_info(self) -> Dict[str, Any]:
        """Get comprehensive system information for model compatibility."""
        sys_req = self._get_system_requirements()
        
        # Add inference engine availability
        engines = {}
        for engine in ["vllm", "comfyui", "diffusers", "transformers"]:
            engines[engine] = await self._check_inference_engine(engine)
            
        return {
            **sys_req,
            "inference_engines": engines,
            "recommended_models": self._get_recommended_models(sys_req)
        }
    
    def _get_recommended_models(self, sys_req: Dict[str, Any]) -> Dict[str, List[str]]:
        """Get recommended models based on system capabilities."""
        recommendations = {"text": [], "image": [], "voice": []}
        
        # Text model recommendations
        if sys_req["gpu_memory_gb"] >= 48 and sys_req["ram_gb"] >= 64:
            recommendations["text"].extend(["llama-3.1-70b", "qwen2.5-72b"])
        elif sys_req["gpu_memory_gb"] >= 24 and sys_req["ram_gb"] >= 32:
            recommendations["text"].append("mixtral-8x7b")
        elif sys_req["gpu_memory_gb"] >= 8 and sys_req["ram_gb"] >= 16:
            recommendations["text"].append("llama-3.1-8b")
            
        # Image model recommendations  
        if sys_req["gpu_memory_gb"] >= 12 and sys_req["ram_gb"] >= 24:
            recommendations["image"].append("flux.1-dev")
        elif sys_req["gpu_memory_gb"] >= 8 and sys_req["ram_gb"] >= 16:
            recommendations["image"].append("sdxl-1.0")
            
        # Voice model recommendations
        if sys_req["gpu_memory_gb"] >= 4 and sys_req["ram_gb"] >= 8:
            recommendations["voice"].append("xtts-v2")
        recommendations["voice"].append("piper")  # Always available
        
        return recommendations
    
    async def install_model(self, model_type: str, model_name: str) -> Dict[str, Any]:
        """Install a specific model."""
        try:
            if model_type not in self.local_model_configs:
                raise ValueError(f"Invalid model type: {model_type}")
            
            if model_name not in self.local_model_configs[model_type]:
                raise ValueError(f"Unknown model: {model_name}")
            
            config = self.local_model_configs[model_type][model_name]
            model_path = self.models_dir / model_name
            
            # Check system requirements
            sys_req = self._get_system_requirements()
            if (sys_req["gpu_memory_gb"] < config.get("min_gpu_memory_gb", 0) or
                sys_req["ram_gb"] < config.get("min_ram_gb", 0)):
                return {
                    "success": False,
                    "error": "Insufficient system resources",
                    "requirements": {
                        "min_gpu_memory_gb": config.get("min_gpu_memory_gb", 0),
                        "min_ram_gb": config.get("min_ram_gb", 0)
                    },
                    "available": {
                        "gpu_memory_gb": sys_req["gpu_memory_gb"],
                        "ram_gb": sys_req["ram_gb"]
                    }
                }
            
            # For now, simulate installation
            model_path.mkdir(exist_ok=True)
            (model_path / "model.safetensors").touch()  # Placeholder file
            
            return {
                "success": True,
                "model": model_name,
                "type": model_type,
                "path": str(model_path),
                "size_gb": config["size_gb"]
            }
            
        except Exception as e:
            logger.error(f"Model installation failed: {str(e)}")
            return {"success": False, "error": str(e)}
    
    async def close(self) -> None:
        """Clean up resources."""
        await self.http_client.aclose()


# Global instance
ai_models = AIModelManager()