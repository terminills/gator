"""
Voice model handlers for speech synthesis and recognition.

Handles voice generation using various backends:
- Local: XTTS-v2, Piper, Coqui TTS
- Cloud: ElevenLabs, OpenAI TTS
"""

import asyncio
import base64
import os
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx

from backend.config.logging import get_logger
from backend.config.settings import get_settings

from .base import BaseModelHandler, ModelCapabilities, ModelType
from .gpu_manager import get_gpu_manager
from .model_cache import get_model_cache

logger = get_logger(__name__)


class VoiceModelHandler(BaseModelHandler):
    """
    Handler for voice synthesis models.

    Supports multiple backends:
    - XTTS-v2 (local)
    - Piper (local, fast)
    - ElevenLabs API
    - OpenAI TTS API
    """

    def __init__(self):
        """Initialize the voice model handler."""
        super().__init__(ModelType.VOICE)
        self.settings = get_settings()
        self.gpu_manager = get_gpu_manager()
        self.model_cache = get_model_cache()

        # Backend availability
        self.xtts_available = False
        self.piper_available = False
        self.elevenlabs_available = False
        self.openai_tts_available = False

        # Model paths
        self.model_dir = Path(
            os.environ.get("AI_MODEL_PATH", "/opt/gator/data/models/voice")
        )
        self.output_dir = Path(
            os.environ.get("CONTENT_OUTPUT_PATH", "generated_content/audio")
        )

    async def initialize(self) -> None:
        """Initialize the voice model handler and discover available models."""
        if self._initialized:
            return

        logger.info("Initializing voice model handler...")

        # Detect available backends
        await self._detect_backends()

        # Discover local models
        await self._discover_local_models()

        # Register cloud models
        await self._register_cloud_models()

        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self._initialized = True
        logger.info(
            f"Voice model handler initialized with {len(self.available_models)} models"
        )

    async def _detect_backends(self) -> None:
        """Detect available inference backends."""
        # Check XTTS-v2
        try:
            from TTS.api import TTS  # noqa: F401

            self.xtts_available = True
            logger.info("✓ XTTS-v2 (Coqui TTS) is available")
        except ImportError:
            logger.debug("XTTS-v2 not available")

        # Check Piper
        piper_binary = self._find_piper_binary()
        if piper_binary:
            self.piper_available = True
            self.piper_binary = piper_binary
            logger.info(f"✓ Piper is available at {piper_binary}")

        # Check cloud APIs
        if self.settings.elevenlabs_api_key:
            self.elevenlabs_available = True
            logger.info("✓ ElevenLabs API key configured")

        if self.settings.openai_api_key:
            self.openai_tts_available = True
            logger.info("✓ OpenAI TTS API available")

    def _find_piper_binary(self) -> Optional[str]:
        """Find the Piper TTS binary."""
        possible_paths = [
            "/usr/local/bin/piper",
            "/usr/bin/piper",
            str(Path.home() / ".local" / "bin" / "piper"),
        ]

        for path in possible_paths:
            if Path(path).exists():
                return path

        return None

    async def _discover_local_models(self) -> None:
        """Discover local voice models."""
        if not self.model_dir.exists():
            logger.debug(f"Model directory does not exist: {self.model_dir}")
            return

        # Find XTTS-v2 models
        if self.xtts_available:
            for model_dir in self.model_dir.iterdir():
                if model_dir.is_dir() and (model_dir / "config.json").exists():
                    model_name = model_dir.name
                    self.available_models[f"xtts:{model_name}"] = {
                        "name": model_name,
                        "path": str(model_dir),
                        "source": "local",
                        "backend": "xtts",
                        "capabilities": ModelCapabilities(
                            supports_streaming=True,
                            supported_formats=["wav", "mp3"],
                        ).to_dict(),
                    }
                    logger.debug(f"Found XTTS model: {model_name}")

        # Find Piper models
        if self.piper_available:
            onnx_files = list(self.model_dir.glob("**/*.onnx"))
            for onnx_path in onnx_files:
                model_name = onnx_path.stem
                self.available_models[f"piper:{model_name}"] = {
                    "name": model_name,
                    "path": str(onnx_path),
                    "source": "local",
                    "backend": "piper",
                    "capabilities": ModelCapabilities(
                        supported_formats=["wav"],
                    ).to_dict(),
                }
                logger.debug(f"Found Piper model: {model_name}")

    async def _register_cloud_models(self) -> None:
        """Register cloud API models."""
        if self.elevenlabs_available:
            # Default ElevenLabs voices
            voices = [
                "Rachel",
                "Drew",
                "Clyde",
                "Paul",
                "Domi",
                "Dave",
                "Fin",
                "Sarah",
                "Antoni",
                "Thomas",
            ]
            for voice in voices:
                self.available_models[f"elevenlabs:{voice.lower()}"] = {
                    "name": voice,
                    "source": "elevenlabs",
                    "backend": "elevenlabs",
                    "capabilities": ModelCapabilities(
                        supports_streaming=True,
                        supported_formats=["mp3", "wav"],
                    ).to_dict(),
                }

        if self.openai_tts_available:
            openai_voices = [
                "alloy",
                "echo",
                "fable",
                "onyx",
                "nova",
                "shimmer",
            ]
            for voice in openai_voices:
                self.available_models[f"openai:{voice}"] = {
                    "name": voice,
                    "source": "openai",
                    "backend": "openai",
                    "capabilities": ModelCapabilities(
                        supported_formats=["mp3", "opus", "aac", "flac"],
                    ).to_dict(),
                }

    async def generate(
        self,
        text: str,
        model: Optional[str] = None,
        voice: Optional[str] = None,
        language: str = "en",
        speed: float = 1.0,
        output_format: str = "mp3",
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Generate speech from text.

        Args:
            text: Text to synthesize
            model: Model name (optional, will select best available)
            voice: Voice/speaker name (for multi-speaker models)
            language: Language code
            speed: Speech speed multiplier
            output_format: Output audio format (mp3, wav, etc.)
            **kwargs: Additional generation parameters

        Returns:
            Dictionary with audio path, base64 data, and metadata
        """
        if not self._initialized:
            await self.initialize()

        # Select model if not specified
        if not model:
            model = self._select_best_model(**kwargs)

        if not model:
            raise ValueError("No voice model available for generation")

        model_info = self.available_models.get(model)
        if not model_info:
            raise ValueError(f"Model not found: {model}")

        backend = model_info.get("backend", "xtts")

        logger.info(f"Generating voice with model '{model}' using {backend}")

        if backend == "xtts":
            return await self._generate_xtts(
                text=text,
                model_info=model_info,
                language=language,
                speed=speed,
                **kwargs,
            )
        elif backend == "piper":
            return await self._generate_piper(
                text=text,
                model_info=model_info,
                **kwargs,
            )
        elif backend == "elevenlabs":
            return await self._generate_elevenlabs(
                text=text,
                model_info=model_info,
                output_format=output_format,
                **kwargs,
            )
        elif backend == "openai":
            return await self._generate_openai_tts(
                text=text,
                model_info=model_info,
                speed=speed,
                output_format=output_format,
                **kwargs,
            )
        else:
            raise ValueError(f"Unsupported backend: {backend}")

    def _select_best_model(self, **kwargs) -> Optional[str]:
        """Select the best available model for generation."""
        if not self.available_models:
            return None

        # Prefer local models over cloud for privacy
        local_models = [
            name
            for name, info in self.available_models.items()
            if info.get("source") == "local"
        ]

        if local_models:
            # Prefer XTTS for quality
            xtts_models = [m for m in local_models if m.startswith("xtts:")]
            if xtts_models:
                return xtts_models[0]
            return local_models[0]

        # Fall back to cloud models
        if self.elevenlabs_available:
            return "elevenlabs:rachel"
        if self.openai_tts_available:
            return "openai:alloy"

        return None

    async def _generate_xtts(
        self,
        text: str,
        model_info: Dict[str, Any],
        language: str,
        speed: float,
        speaker_wav: Optional[str] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Generate speech using XTTS-v2."""
        from TTS.api import TTS

        model_path = model_info.get("path")

        # Check cache
        cache_key = f"xtts:{model_path}"
        tts = self.model_cache.get(cache_key)

        if tts is None:
            device = self.gpu_manager.select_device_for_model(
                cache_key, required_memory_gb=2.0
            )

            logger.info(f"Loading XTTS model from {model_path}")

            # Load model
            loop = asyncio.get_event_loop()
            tts = await loop.run_in_executor(
                None,
                lambda: TTS(model_path=model_path).to(device),
            )

            self.model_cache.put(
                cache_key,
                tts,
                model_type="voice",
                device=device,
                memory_mb=2000,
            )

        # Generate audio
        output_filename = f"{uuid.uuid4()}.wav"
        output_path = self.output_dir / output_filename

        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None,
            lambda: tts.tts_to_file(
                text=text,
                file_path=str(output_path),
                speaker_wav=speaker_wav,
                language=language,
                speed=speed,
            ),
        )

        # Read and encode
        with open(output_path, "rb") as f:
            audio_data = f.read()
        base64_data = base64.b64encode(audio_data).decode("utf-8")

        return {
            "path": str(output_path),
            "base64": base64_data,
            "format": "wav",
            "model": model_info.get("name"),
            "duration": len(audio_data) / (16000 * 2),  # Rough estimate
        }

    async def _generate_piper(
        self,
        text: str,
        model_info: Dict[str, Any],
        **kwargs,
    ) -> Dict[str, Any]:
        """Generate speech using Piper."""
        import subprocess

        model_path = model_info.get("path")
        output_filename = f"{uuid.uuid4()}.wav"
        output_path = self.output_dir / output_filename

        cmd = [
            self.piper_binary,
            "--model",
            model_path,
            "--output_file",
            str(output_path),
        ]

        loop = asyncio.get_event_loop()
        process = await loop.run_in_executor(
            None,
            lambda: subprocess.run(
                cmd,
                input=text.encode(),
                capture_output=True,
            ),
        )

        if process.returncode != 0:
            raise RuntimeError(f"Piper failed: {process.stderr.decode()}")

        with open(output_path, "rb") as f:
            audio_data = f.read()
        base64_data = base64.b64encode(audio_data).decode("utf-8")

        return {
            "path": str(output_path),
            "base64": base64_data,
            "format": "wav",
            "model": model_info.get("name"),
        }

    async def _generate_elevenlabs(
        self,
        text: str,
        model_info: Dict[str, Any],
        output_format: str = "mp3",
        **kwargs,
    ) -> Dict[str, Any]:
        """Generate speech using ElevenLabs API."""
        voice_name = model_info.get("name", "Rachel")

        # Get voice ID (simplified - in production, would look up from API)
        voice_ids = {
            "rachel": "21m00Tcm4TlvDq8ikWAM",
            "drew": "29vD33N1CtxCmqQRPOHJ",
            "clyde": "2EiwWnXFnvU5JabPnv8n",
            "paul": "5Q0t7uMcjvnagumLfvZi",
            "domi": "AZnzlk1XvdvUeBnXmlld",
            "dave": "CYw3kZ02Hs0563khs1Fj",
        }
        voice_id = voice_ids.get(voice_name.lower(), voice_ids["rachel"])

        payload = {
            "text": text,
            "model_id": "eleven_monolingual_v1",
            "voice_settings": {
                "stability": 0.5,
                "similarity_boost": 0.75,
            },
        }

        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.post(
                    f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}",
                    json=payload,
                    headers={
                        "xi-api-key": self.settings.elevenlabs_api_key,
                        "Content-Type": "application/json",
                        "Accept": f"audio/{output_format}",
                    },
                )
                response.raise_for_status()

                audio_data = response.content

                output_filename = f"{uuid.uuid4()}.{output_format}"
                output_path = self.output_dir / output_filename
                with open(output_path, "wb") as f:
                    f.write(audio_data)

                return {
                    "path": str(output_path),
                    "base64": base64.b64encode(audio_data).decode("utf-8"),
                    "format": output_format,
                    "model": voice_name,
                }

        except Exception as e:
            raise RuntimeError(f"ElevenLabs generation failed: {e}")

    async def _generate_openai_tts(
        self,
        text: str,
        model_info: Dict[str, Any],
        speed: float = 1.0,
        output_format: str = "mp3",
        **kwargs,
    ) -> Dict[str, Any]:
        """Generate speech using OpenAI TTS API."""
        voice = model_info.get("name", "alloy")

        payload = {
            "model": "tts-1",
            "voice": voice,
            "input": text,
            "speed": speed,
            "response_format": output_format,
        }

        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.post(
                    "https://api.openai.com/v1/audio/speech",
                    json=payload,
                    headers={
                        "Authorization": f"Bearer {self.settings.openai_api_key}",
                        "Content-Type": "application/json",
                    },
                )
                response.raise_for_status()

                audio_data = response.content

                output_filename = f"{uuid.uuid4()}.{output_format}"
                output_path = self.output_dir / output_filename
                with open(output_path, "wb") as f:
                    f.write(audio_data)

                return {
                    "path": str(output_path),
                    "base64": base64.b64encode(audio_data).decode("utf-8"),
                    "format": output_format,
                    "model": voice,
                }

        except Exception as e:
            raise RuntimeError(f"OpenAI TTS generation failed: {e}")

    async def get_available_models(self) -> List[Dict[str, Any]]:
        """Get list of available voice models."""
        if not self._initialized:
            await self.initialize()

        return [
            {
                "name": name,
                **info,
            }
            for name, info in self.available_models.items()
        ]


# Global voice model handler instance
_voice_handler: Optional[VoiceModelHandler] = None


def get_voice_handler() -> VoiceModelHandler:
    """Get the global voice model handler instance."""
    global _voice_handler
    if _voice_handler is None:
        _voice_handler = VoiceModelHandler()
    return _voice_handler
