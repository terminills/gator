"""
Text model handlers for LLM operations.

Handles text generation using various backends:
- Local: llama.cpp, vLLM, Ollama, Transformers
- Cloud: OpenAI, Anthropic
"""

import asyncio
import os
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx

from backend.config.logging import get_logger
from backend.config.settings import get_settings

from .base import BaseModelHandler, ModelCapabilities, ModelType
from .gpu_manager import get_gpu_manager
from .model_cache import get_model_cache
from .model_loader import strip_ansi_codes

logger = get_logger(__name__)

# Preferred uncensored model for NSFW content generation
PREFERRED_UNCENSORED_MODEL_PREFIX = "dolphin"


class TextModelHandler(BaseModelHandler):
    """
    Handler for text generation models.

    Supports multiple backends:
    - llama.cpp (local GGUF models)
    - Ollama (managed local models)
    - vLLM (high-performance inference)
    - Transformers (HuggingFace)
    - OpenAI API
    - Anthropic API
    """

    def __init__(self):
        """Initialize the text model handler."""
        super().__init__(ModelType.TEXT)
        self.settings = get_settings()
        self.gpu_manager = get_gpu_manager()
        self.model_cache = get_model_cache()

        # Backend availability
        self.llamacpp_binary: Optional[str] = None
        self.ollama_available = False
        self.vllm_available = False
        self.openai_available = False
        self.anthropic_available = False

        # Model paths
        self.model_dir = Path(
            os.environ.get("AI_MODEL_PATH", "/opt/gator/data/models/text")
        )

    async def initialize(self) -> None:
        """Initialize the text model handler and discover available models."""
        if self._initialized:
            return

        logger.info("Initializing text model handler...")

        # Detect available backends
        await self._detect_backends()

        # Discover local models
        await self._discover_local_models()

        # Register cloud models
        await self._register_cloud_models()

        self._initialized = True
        logger.info(
            f"Text model handler initialized with {len(self.available_models)} models"
        )

    async def _detect_backends(self) -> None:
        """Detect available inference backends."""
        # Check llama.cpp
        self.llamacpp_binary = self._find_llamacpp_binary()
        if self.llamacpp_binary:
            logger.info(f"✓ llama.cpp found: {self.llamacpp_binary}")

        # Check Ollama
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get("http://localhost:11434/api/version")
                if response.status_code == 200:
                    self.ollama_available = True
                    logger.info("✓ Ollama is available")
        except Exception:
            logger.debug("Ollama not available")

        # Check vLLM
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get("http://localhost:8000/v1/models")
                if response.status_code == 200:
                    self.vllm_available = True
                    logger.info("✓ vLLM is available")
        except Exception:
            logger.debug("vLLM not available")

        # Check cloud APIs
        if self.settings.openai_api_key:
            self.openai_available = True
            logger.info("✓ OpenAI API key configured")

        if self.settings.anthropic_api_key:
            self.anthropic_available = True
            logger.info("✓ Anthropic API key configured")

    def _find_llamacpp_binary(self) -> Optional[str]:
        """Find the llama.cpp inference binary."""
        possible_names = [
            "llama-cli",
            "main",
            "llama.cpp",
            "llama-server",
        ]

        possible_paths = [
            "/opt/gator/llama.cpp/build/bin",
            "/usr/local/bin",
            "/usr/bin",
            str(Path.home() / "llama.cpp" / "build" / "bin"),
        ]

        for path in possible_paths:
            path_obj = Path(path)
            if path_obj.exists():
                for name in possible_names:
                    binary = path_obj / name
                    if binary.exists() and binary.is_file():
                        return str(binary)

        return None

    async def _discover_local_models(self) -> None:
        """Discover local text models."""
        if not self.model_dir.exists():
            logger.debug(f"Model directory does not exist: {self.model_dir}")
            return

        # Find GGUF models
        gguf_files = list(self.model_dir.glob("**/*.gguf"))
        for gguf_path in gguf_files:
            model_name = gguf_path.stem
            self.available_models[model_name] = {
                "name": model_name,
                "path": str(gguf_path),
                "format": "gguf",
                "source": "local",
                "backend": "llamacpp",
                "capabilities": ModelCapabilities(
                    supports_nsfw="dolphin" in model_name.lower()
                    or "uncensored" in model_name.lower(),
                    supports_streaming=True,
                ).to_dict(),
            }
            logger.debug(f"Found local model: {model_name}")

        # Discover Ollama models
        if self.ollama_available:
            await self._discover_ollama_models()

    async def _discover_ollama_models(self) -> None:
        """Discover Ollama models."""
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get("http://localhost:11434/api/tags")
                if response.status_code == 200:
                    data = response.json()
                    for model in data.get("models", []):
                        model_name = model.get("name", "")
                        self.available_models[f"ollama:{model_name}"] = {
                            "name": model_name,
                            "source": "ollama",
                            "backend": "ollama",
                            "size": model.get("size", 0),
                            "capabilities": ModelCapabilities(
                                supports_nsfw="dolphin" in model_name.lower()
                                or "uncensored" in model_name.lower(),
                                supports_streaming=True,
                            ).to_dict(),
                        }
                        logger.debug(f"Found Ollama model: {model_name}")
        except Exception as e:
            logger.warning(f"Failed to discover Ollama models: {e}")

    async def _register_cloud_models(self) -> None:
        """Register cloud API models."""
        if self.openai_available:
            openai_models = [
                "gpt-4o",
                "gpt-4o-mini",
                "gpt-4-turbo",
                "gpt-3.5-turbo",
            ]
            for model_name in openai_models:
                self.available_models[f"openai:{model_name}"] = {
                    "name": model_name,
                    "source": "openai",
                    "backend": "openai",
                    "capabilities": ModelCapabilities(
                        supports_streaming=True,
                        max_tokens=128000 if "gpt-4" in model_name else 16384,
                    ).to_dict(),
                }

        if self.anthropic_available:
            anthropic_models = [
                "claude-3-5-sonnet-latest",
                "claude-3-opus-latest",
                "claude-3-haiku-latest",
            ]
            for model_name in anthropic_models:
                self.available_models[f"anthropic:{model_name}"] = {
                    "name": model_name,
                    "source": "anthropic",
                    "backend": "anthropic",
                    "capabilities": ModelCapabilities(
                        supports_streaming=True,
                        max_tokens=200000,
                    ).to_dict(),
                }

    async def generate(
        self,
        prompt: str,
        model: Optional[str] = None,
        max_tokens: int = 2048,
        temperature: float = 0.7,
        system_prompt: Optional[str] = None,
        **kwargs,
    ) -> str:
        """
        Generate text using the selected model.

        Args:
            prompt: Input prompt
            model: Model name (optional, will select best available)
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            system_prompt: System prompt for instruction-following models
            **kwargs: Additional generation parameters

        Returns:
            Generated text
        """
        if not self._initialized:
            await self.initialize()

        # Select model if not specified
        if not model:
            model = self._select_best_model(**kwargs)

        if not model:
            raise ValueError("No text model available for generation")

        model_info = self.available_models.get(model)
        if not model_info:
            raise ValueError(f"Model not found: {model}")

        backend = model_info.get("backend", "llamacpp")

        logger.info(f"Generating text with model '{model}' using {backend}")

        if backend == "llamacpp":
            return await self._generate_llamacpp(
                prompt=prompt,
                model_info=model_info,
                max_tokens=max_tokens,
                temperature=temperature,
                system_prompt=system_prompt,
                **kwargs,
            )
        elif backend == "ollama":
            return await self._generate_ollama(
                prompt=prompt,
                model_info=model_info,
                max_tokens=max_tokens,
                temperature=temperature,
                system_prompt=system_prompt,
                **kwargs,
            )
        elif backend == "openai":
            return await self._generate_openai(
                prompt=prompt,
                model_info=model_info,
                max_tokens=max_tokens,
                temperature=temperature,
                system_prompt=system_prompt,
                **kwargs,
            )
        elif backend == "anthropic":
            return await self._generate_anthropic(
                prompt=prompt,
                model_info=model_info,
                max_tokens=max_tokens,
                temperature=temperature,
                system_prompt=system_prompt,
                **kwargs,
            )
        else:
            raise ValueError(f"Unsupported backend: {backend}")

    def _select_best_model(self, prefer_nsfw: bool = False, **kwargs) -> Optional[str]:
        """Select the best available model for generation."""
        if not self.available_models:
            return None

        # Prefer local models over cloud
        local_models = [
            name
            for name, info in self.available_models.items()
            if info.get("source") in ["local", "ollama"]
        ]

        if prefer_nsfw:
            # Prefer uncensored models
            nsfw_models = [
                name
                for name in local_models
                if self.available_models[name]
                .get("capabilities", {})
                .get("supports_nsfw", False)
            ]
            if nsfw_models:
                return nsfw_models[0]

        if local_models:
            return local_models[0]

        # Fall back to cloud models
        if self.openai_available:
            return "openai:gpt-4o-mini"
        if self.anthropic_available:
            return "anthropic:claude-3-haiku-latest"

        return None

    async def _generate_llamacpp(
        self,
        prompt: str,
        model_info: Dict[str, Any],
        max_tokens: int,
        temperature: float,
        system_prompt: Optional[str],
        **kwargs,
    ) -> str:
        """Generate text using llama.cpp."""
        if not self.llamacpp_binary:
            raise RuntimeError("llama.cpp binary not found")

        model_path = model_info.get("path")
        if not model_path:
            raise ValueError("Model path not specified")

        # Build command
        cmd = [
            self.llamacpp_binary,
            "-m",
            model_path,
            "-n",
            str(max_tokens),
            "--temp",
            str(temperature),
            "-p",
            prompt,
            "--no-display-prompt",
        ]

        if system_prompt:
            cmd.extend(["--system-prompt", system_prompt])

        # Add GPU layers if available
        if self.gpu_manager.get_gpu_count() > 0:
            cmd.extend(["-ngl", "999"])  # Offload all layers to GPU

        try:
            # Run in thread pool
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                lambda: subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=300,
                ),
            )

            if result.returncode != 0:
                logger.error(f"llama.cpp failed: {result.stderr}")
                raise RuntimeError(f"Generation failed: {result.stderr}")

            # Clean output
            output = strip_ansi_codes(result.stdout)
            return self._filter_llamacpp_output(output)

        except subprocess.TimeoutExpired:
            raise RuntimeError("Generation timed out")

    def _filter_llamacpp_output(self, output: str) -> str:
        """Filter llama.cpp output to extract generated text."""
        lines = output.strip().split("\n")
        filtered_lines = []

        for line in lines:
            # Skip llama.cpp info lines
            if any(
                skip in line.lower()
                for skip in [
                    "llama_",
                    "ggml_",
                    "sampling:",
                    "generate:",
                    "ctx size:",
                    "system_info:",
                    "main:",
                    "loaded",
                    "model size",
                    "layers:",
                ]
            ):
                continue
            filtered_lines.append(line)

        return "\n".join(filtered_lines).strip()

    async def _generate_ollama(
        self,
        prompt: str,
        model_info: Dict[str, Any],
        max_tokens: int,
        temperature: float,
        system_prompt: Optional[str],
        **kwargs,
    ) -> str:
        """Generate text using Ollama."""
        model_name = model_info.get("name")

        payload = {
            "model": model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "num_predict": max_tokens,
                "temperature": temperature,
            },
        }

        if system_prompt:
            payload["system"] = system_prompt

        try:
            async with httpx.AsyncClient(timeout=300.0) as client:
                response = await client.post(
                    "http://localhost:11434/api/generate",
                    json=payload,
                )
                response.raise_for_status()
                data = response.json()
                return data.get("response", "")

        except httpx.TimeoutException:
            raise RuntimeError("Ollama generation timed out")
        except Exception as e:
            raise RuntimeError(f"Ollama generation failed: {e}")

    async def _generate_openai(
        self,
        prompt: str,
        model_info: Dict[str, Any],
        max_tokens: int,
        temperature: float,
        system_prompt: Optional[str],
        **kwargs,
    ) -> str:
        """Generate text using OpenAI API."""
        model_name = model_info.get("name")

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": model_name,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }

        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                response = await client.post(
                    "https://api.openai.com/v1/chat/completions",
                    json=payload,
                    headers={
                        "Authorization": f"Bearer {self.settings.openai_api_key}",
                        "Content-Type": "application/json",
                    },
                )
                response.raise_for_status()
                data = response.json()
                return data["choices"][0]["message"]["content"]

        except Exception as e:
            raise RuntimeError(f"OpenAI generation failed: {e}")

    async def _generate_anthropic(
        self,
        prompt: str,
        model_info: Dict[str, Any],
        max_tokens: int,
        temperature: float,
        system_prompt: Optional[str],
        **kwargs,
    ) -> str:
        """Generate text using Anthropic API."""
        model_name = model_info.get("name")

        payload = {
            "model": model_name,
            "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": prompt}],
        }

        if system_prompt:
            payload["system"] = system_prompt

        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                response = await client.post(
                    "https://api.anthropic.com/v1/messages",
                    json=payload,
                    headers={
                        "x-api-key": self.settings.anthropic_api_key,
                        "anthropic-version": "2023-06-01",
                        "Content-Type": "application/json",
                    },
                )
                response.raise_for_status()
                data = response.json()
                return data["content"][0]["text"]

        except Exception as e:
            raise RuntimeError(f"Anthropic generation failed: {e}")

    async def get_available_models(self) -> List[Dict[str, Any]]:
        """Get list of available text models."""
        if not self._initialized:
            await self.initialize()

        return [
            {
                "name": name,
                **info,
            }
            for name, info in self.available_models.items()
        ]


# Global text model handler instance
_text_handler: Optional[TextModelHandler] = None


def get_text_handler() -> TextModelHandler:
    """Get the global text model handler instance."""
    global _text_handler
    if _text_handler is None:
        _text_handler = TextModelHandler()
    return _text_handler
