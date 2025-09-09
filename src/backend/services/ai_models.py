"""
AI Model Integration Service

Handles integration with various AI models for content generation including
Stable Diffusion for images, OpenAI/Anthropic for text, and other specialized models.
"""

import asyncio
import os
import io
import base64
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
    
    Handles loading, configuration, and execution of various AI models
    for content generation.
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
    
    async def initialize_models(self) -> None:
        """Initialize and load AI models based on available hardware and configuration."""
        try:
            # Check for GPU availability
            gpu_available = torch.cuda.is_available()
            gpu_count = torch.cuda.device_count() if gpu_available else 0
            
            logger.info(f"Initializing AI models - GPU available: {gpu_available}, GPU count: {gpu_count}")
            
            # Initialize image generation models
            await self._initialize_image_models()
            
            # Initialize text generation models
            await self._initialize_text_models()
            
            # Initialize voice/audio models
            await self._initialize_voice_models()
            
            # Initialize video models
            await self._initialize_video_models()
            
            self.models_loaded = True
            logger.info("AI model initialization complete", extra={"available_models": self.available_models})
            
        except Exception as e:
            logger.error(f"Failed to initialize AI models: {str(e)}")
            raise
    
    async def _initialize_image_models(self) -> None:
        """Initialize image generation models."""
        try:
            # Check for Stable Diffusion model availability
            if torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 7:
                # Check if diffusers models are available
                try:
                    from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
                    
                    # Try to load a lightweight model first
                    model_id = "runwayml/stable-diffusion-v1-5"  # Popular stable model
                    
                    # For now, we'll use a simulated model loading to avoid downloading large models
                    # In production, this would actually load the model:
                    # self.sd_pipeline = StableDiffusionPipeline.from_pretrained(
                    #     model_id, torch_dtype=torch.float16
                    # ).to("cuda")
                    
                    self.available_models["image"].append({
                        "name": "stable-diffusion-v1-5",
                        "type": "text-to-image", 
                        "model_id": model_id,
                        "loaded": False,  # Set to True when actually loaded
                        "device": "cuda" if torch.cuda.is_available() else "cpu"
                    })
                    
                    logger.info("Stable Diffusion model configured (not loaded to save memory)")
                    
                except ImportError:
                    logger.warning("Diffusers library not fully available for image generation")
                    
            # Add OpenAI DALL-E as fallback if API key available
            if hasattr(self.settings, 'openai_api_key') and self.settings.openai_api_key:
                self.available_models["image"].append({
                    "name": "dall-e-3",
                    "type": "text-to-image",
                    "provider": "openai",
                    "loaded": True
                })
                logger.info("OpenAI DALL-E 3 API available")
                
        except Exception as e:
            logger.error(f"Failed to initialize image models: {str(e)}")
    
    async def _initialize_text_models(self) -> None:
        """Initialize text generation models."""
        try:
            # Check for local text models
            if torch.cuda.is_available():
                # For production, would load actual models like:
                # from transformers import AutoTokenizer, AutoModelForCausalLM
                
                self.available_models["text"].append({
                    "name": "local-llm",
                    "type": "text-generation",
                    "loaded": False,  # Would be True when loaded
                    "max_length": 2048
                })
                
            # Add OpenAI GPT as API option
            if hasattr(self.settings, 'openai_api_key') and self.settings.openai_api_key:
                self.available_models["text"].extend([
                    {
                        "name": "gpt-4",
                        "type": "text-generation",
                        "provider": "openai",
                        "loaded": True,
                        "max_tokens": 4096
                    },
                    {
                        "name": "gpt-3.5-turbo",
                        "type": "text-generation", 
                        "provider": "openai",
                        "loaded": True,
                        "max_tokens": 4096
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
                    "max_tokens": 4096
                })
                logger.info("Anthropic Claude available")
                
        except Exception as e:
            logger.error(f"Failed to initialize text models: {str(e)}")
    
    async def _initialize_voice_models(self) -> None:
        """Initialize voice synthesis models."""
        try:
            # Check for ElevenLabs API
            if hasattr(self.settings, 'elevenlabs_api_key') and self.settings.elevenlabs_api_key:
                self.available_models["voice"].append({
                    "name": "elevenlabs-tts",
                    "type": "text-to-speech",
                    "provider": "elevenlabs",
                    "loaded": True
                })
                logger.info("ElevenLabs voice synthesis available")
                
            # Check for OpenAI TTS
            if hasattr(self.settings, 'openai_api_key') and self.settings.openai_api_key:
                self.available_models["voice"].append({
                    "name": "openai-tts", 
                    "type": "text-to-speech",
                    "provider": "openai",
                    "loaded": True
                })
                logger.info("OpenAI TTS available")
                
        except Exception as e:
            logger.error(f"Failed to initialize voice models: {str(e)}")
    
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
        """Generate image from text prompt."""
        try:
            # Find best available image model
            image_models = [m for m in self.available_models["image"] if m.get("loaded", False)]
            
            if not image_models:
                raise ValueError("No image generation models available")
            
            # Use first available model (priority could be added later)
            model = image_models[0]
            
            if model.get("provider") == "openai":
                return await self._generate_image_openai(prompt, **kwargs)
            elif model.get("name") == "stable-diffusion-v1-5":
                return await self._generate_image_stable_diffusion(prompt, **kwargs)
            else:
                raise ValueError(f"Unsupported image model: {model['name']}")
                
        except Exception as e:
            logger.error(f"Image generation failed: {str(e)}")
            raise
    
    async def generate_text(self, prompt: str, **kwargs) -> str:
        """Generate text from prompt."""
        try:
            # Find best available text model
            text_models = [m for m in self.available_models["text"] if m.get("loaded", False)]
            
            if not text_models:
                raise ValueError("No text generation models available")
            
            model = text_models[0]  # Use first available
            
            if model.get("provider") == "openai":
                return await self._generate_text_openai(prompt, model["name"], **kwargs)
            elif model.get("provider") == "anthropic":
                return await self._generate_text_anthropic(prompt, **kwargs)
            else:
                raise ValueError(f"Unsupported text model: {model['name']}")
                
        except Exception as e:
            logger.error(f"Text generation failed: {str(e)}")
            raise
    
    async def generate_voice(self, text: str, **kwargs) -> Dict[str, Any]:
        """Generate voice audio from text."""
        try:
            voice_models = [m for m in self.available_models["voice"] if m.get("loaded", False)]
            
            if not voice_models:
                raise ValueError("No voice generation models available")
            
            model = voice_models[0]
            
            if model.get("provider") == "elevenlabs":
                return await self._generate_voice_elevenlabs(text, **kwargs)
            elif model.get("provider") == "openai":
                return await self._generate_voice_openai(text, **kwargs)
            else:
                raise ValueError(f"Unsupported voice model: {model['name']}")
                
        except Exception as e:
            logger.error(f"Voice generation failed: {str(e)}")
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
    
    async def close(self) -> None:
        """Clean up resources."""
        await self.http_client.aclose()


# Global instance
ai_models = AIModelManager()