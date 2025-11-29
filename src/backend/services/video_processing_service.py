"""
Video Processing Service

Handles advanced video generation features including:
- Frame-by-frame generation for longer videos
- Audio synchronization with voice synthesis
- Video editing and transitions
- Scene composition and storyboarding

This service integrates with AI models for video generation and provides
utilities for video manipulation using opencv and ffmpeg.
"""

import asyncio
import os
import subprocess
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from backend.config.logging import get_logger

logger = get_logger(__name__)


class TransitionType(Enum):
    """Video transition types."""

    FADE = "fade"
    CROSSFADE = "crossfade"
    WIPE = "wipe"
    SLIDE = "slide"
    ZOOM = "zoom"
    DISSOLVE = "dissolve"


class VideoQuality(Enum):
    """Video quality presets."""

    DRAFT = "draft"  # 480p, lower bitrate
    STANDARD = "standard"  # 720p, standard bitrate
    HIGH = "high"  # 1080p, high bitrate
    PREMIUM = "premium"  # 4K, highest bitrate


class VideoProcessingService:
    """Service for advanced video processing and generation."""

    def __init__(self, output_dir: str = "generated_content/videos"):
        """
        Initialize video processing service.

        Args:
            output_dir: Directory to store generated videos
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Video quality settings
        self.quality_settings = {
            VideoQuality.DRAFT: {
                "resolution": (854, 480),
                "fps": 24,
                "bitrate": "1500k",
                "crf": 28,
            },
            VideoQuality.STANDARD: {
                "resolution": (1280, 720),
                "fps": 30,
                "bitrate": "3000k",
                "crf": 23,
            },
            VideoQuality.HIGH: {
                "resolution": (1920, 1080),
                "fps": 30,
                "bitrate": "6000k",
                "crf": 20,
            },
            VideoQuality.PREMIUM: {
                "resolution": (3840, 2160),
                "fps": 60,
                "bitrate": "15000k",
                "crf": 18,
            },
        }

    def _check_ffmpeg_available(self) -> bool:
        """Check if ffmpeg is available."""
        try:
            subprocess.run(
                ["ffmpeg", "-version"], capture_output=True, check=True, timeout=5
            )
            return True
        except (
            subprocess.CalledProcessError,
            FileNotFoundError,
            subprocess.TimeoutExpired,
        ):
            logger.warning("ffmpeg not available")
            return False

    async def generate_frame_by_frame_video(
        self,
        prompts: List[str],
        duration_per_frame: float = 2.0,
        quality: VideoQuality = VideoQuality.HIGH,
        transition: TransitionType = TransitionType.CROSSFADE,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Generate longer video by creating multiple frames and stitching them together.

        This is the core feature for Q2-Q3 2025 advanced video generation.
        Each prompt generates a frame/scene, then frames are combined with transitions.

        Args:
            prompts: List of prompts for each frame/scene
            duration_per_frame: Duration each frame should last in seconds
            quality: Video quality preset
            transition: Transition type between frames
            **kwargs: Additional parameters for video generation

        Returns:
            Dict with video metadata including path, duration, resolution
        """
        logger.info(f"Generating frame-by-frame video with {len(prompts)} scenes")

        try:
            # Generate frames for each prompt
            frames = []
            for i, prompt in enumerate(prompts):
                logger.info(f"Generating frame {i+1}/{len(prompts)}: {prompt[:50]}...")
                frame = await self._generate_single_frame(
                    prompt, quality, frame_index=i, **kwargs
                )
                frames.append(frame)

            # Apply transitions between frames
            logger.info(f"Applying {transition.value} transitions between frames")
            video_with_transitions = await self._apply_transitions(
                frames, transition, duration_per_frame
            )

            # Export final video
            output_filename = (
                f"video_multi_frame_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
            )
            output_path = self.output_dir / output_filename

            success = await self._export_video(
                video_with_transitions, output_path, quality
            )

            if not success:
                raise RuntimeError("Failed to export video")

            total_duration = len(prompts) * duration_per_frame
            settings = self.quality_settings[quality]

            return {
                "file_path": str(output_path),
                "file_size": output_path.stat().st_size if output_path.exists() else 0,
                "duration": total_duration,
                "resolution": f"{settings['resolution'][0]}x{settings['resolution'][1]}",
                "fps": settings["fps"],
                "format": "MP4",
                "quality": quality.value,
                "num_scenes": len(prompts),
                "transition_type": transition.value,
                "frames_generated": len(frames),
            }

        except Exception as e:
            logger.error(f"Frame-by-frame video generation failed: {str(e)}")
            raise

    async def _generate_single_frame(
        self, prompt: str, quality: VideoQuality, frame_index: int = 0, **kwargs
    ) -> np.ndarray:
        """
        Generate a single video frame.

        Uses AI image generation to create frames from prompts.
        Falls back to placeholder frames if AI generation fails or is disabled.

        Args:
            prompt: Text prompt for frame generation
            quality: Video quality preset
            frame_index: Index of the frame in sequence
            **kwargs: Additional generation parameters
                - use_ai_generation (bool): Whether to use AI for frame generation (default: True)
                - ai_model_manager: AIModelManager instance to use for generation

        Returns:
            Numpy array representing the frame
        """
        settings = self.quality_settings[quality]
        width, height = settings["resolution"]

        # Check if AI generation should be used
        use_ai = kwargs.get("use_ai_generation", True)
        ai_manager = kwargs.get("ai_model_manager")

        if use_ai and ai_manager:
            try:
                logger.info(
                    f"Generating frame {frame_index + 1} with AI: {prompt[:50]}..."
                )

                # Generate image using AI
                result = await ai_manager.generate_image(
                    prompt=prompt,
                    width=width,
                    height=height,
                    num_inference_steps=kwargs.get("num_inference_steps", 20),
                    guidance_scale=kwargs.get("guidance_scale", 7.5),
                )

                if result and result.get("image_data"):
                    # Convert image data to numpy array
                    import io

                    from PIL import Image

                    image = Image.open(io.BytesIO(result["image_data"]))
                    # Convert to RGB if needed
                    if image.mode != "RGB":
                        image = image.convert("RGB")
                    # Resize to exact dimensions if needed
                    if image.size != (width, height):
                        image = image.resize((width, height), Image.Resampling.LANCZOS)
                    # Convert to numpy array (OpenCV format: BGR)
                    frame = np.array(image)
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

                    logger.info(
                        f"Frame {frame_index + 1} generated successfully with AI"
                    )
                    return frame
                else:
                    logger.warning(
                        f"AI generation returned empty result, using placeholder"
                    )

            except Exception as e:
                logger.warning(
                    f"AI frame generation failed: {str(e)}, using placeholder"
                )

        # Fallback to placeholder frame
        logger.debug(f"Generating placeholder frame {frame_index + 1}")

        # Create placeholder frame (solid color with text)
        frame = np.zeros((height, width, 3), dtype=np.uint8)

        # Add gradient background
        for i in range(height):
            color_value = int(50 + (i / height) * 150)
            frame[i, :] = [color_value, color_value // 2, color_value // 3]

        # Add text to frame (prompt summary)
        text = f"Scene {frame_index + 1}: {prompt[:30]}..."
        font = cv2.FONT_HERSHEY_SIMPLEX
        text_size = cv2.getTextSize(text, font, 1, 2)[0]
        text_x = (width - text_size[0]) // 2
        text_y = (height + text_size[1]) // 2

        cv2.putText(
            frame, text, (text_x, text_y), font, 1, (255, 255, 255), 2, cv2.LINE_AA
        )

        return frame

    async def _apply_transitions(
        self,
        frames: List[np.ndarray],
        transition: TransitionType,
        duration_per_frame: float,
    ) -> List[np.ndarray]:
        """
        Apply transitions between video frames.

        Args:
            frames: List of frames to transition between
            transition: Type of transition to apply
            duration_per_frame: Duration each frame should last

        Returns:
            List of frames with transitions applied
        """
        if len(frames) < 2:
            return frames

        result_frames = []
        fps = 30  # Standard fps for transitions
        transition_duration = 0.5  # 0.5 seconds transition
        transition_frames = int(transition_duration * fps)
        hold_frames = int((duration_per_frame - transition_duration) * fps)

        for i in range(len(frames)):
            # Add frames for holding current scene
            result_frames.extend([frames[i]] * hold_frames)

            # Add transition to next frame (if not last frame)
            if i < len(frames) - 1:
                transition_sequence = self._create_transition(
                    frames[i], frames[i + 1], transition, transition_frames
                )
                result_frames.extend(transition_sequence)

        return result_frames

    def _create_transition(
        self,
        frame1: np.ndarray,
        frame2: np.ndarray,
        transition: TransitionType,
        num_frames: int,
    ) -> List[np.ndarray]:
        """
        Create transition frames between two frames.

        Args:
            frame1: Starting frame
            frame2: Ending frame
            transition: Type of transition
            num_frames: Number of transition frames to generate

        Returns:
            List of transition frames
        """
        transitions = []

        for i in range(num_frames):
            alpha = i / num_frames

            if transition == TransitionType.FADE:
                # Fade through black
                if alpha < 0.5:
                    # Fade out first frame
                    factor = 1 - (alpha * 2)
                    blended = (frame1 * factor).astype(np.uint8)
                else:
                    # Fade in second frame
                    factor = (alpha - 0.5) * 2
                    blended = (frame2 * factor).astype(np.uint8)

            elif transition == TransitionType.CROSSFADE:
                # Direct crossfade
                blended = cv2.addWeighted(frame1, 1 - alpha, frame2, alpha, 0)

            elif transition == TransitionType.DISSOLVE:
                # Similar to crossfade but with slight blur
                crossfade = cv2.addWeighted(frame1, 1 - alpha, frame2, alpha, 0)
                # Ensure kernel_size is always positive and odd
                kernel_size = max(3, 3 + int(alpha * 4) * 2)
                if kernel_size % 2 == 0:
                    kernel_size += 1
                blended = cv2.GaussianBlur(crossfade, (kernel_size, kernel_size), 0)

            elif transition == TransitionType.WIPE:
                # Left-to-right wipe
                blended = frame1.copy()
                wipe_position = int(alpha * frame1.shape[1])
                blended[:, :wipe_position] = frame2[:, :wipe_position]

            elif transition == TransitionType.SLIDE:
                # Slide in from right
                blended = frame1.copy()
                slide_offset = int((1 - alpha) * frame1.shape[1])
                if slide_offset < frame1.shape[1]:
                    blended[:, :-slide_offset] = frame2[:, slide_offset:]

            else:
                # Default to crossfade
                blended = cv2.addWeighted(frame1, 1 - alpha, frame2, alpha, 0)

            transitions.append(blended)

        return transitions

    async def _export_video(
        self, frames: List[np.ndarray], output_path: Path, quality: VideoQuality
    ) -> bool:
        """
        Export frames to video file using opencv or ffmpeg.

        Args:
            frames: List of frames to export
            output_path: Output video file path
            quality: Video quality preset

        Returns:
            True if export successful, False otherwise
        """
        if not frames:
            logger.error("No frames to export")
            return False

        try:
            settings = self.quality_settings[quality]
            height, width = frames[0].shape[:2]
            fps = settings["fps"]

            # Use opencv VideoWriter
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

            for frame in frames:
                writer.write(frame)

            writer.release()

            # If ffmpeg is available, re-encode for better compression
            if self._check_ffmpeg_available():
                await self._reencode_with_ffmpeg(output_path, settings)

            logger.info(f"Video exported successfully to {output_path}")
            return True

        except Exception as e:
            logger.error(f"Video export failed: {str(e)}")
            return False

    async def _reencode_with_ffmpeg(
        self, video_path: Path, settings: Dict[str, Any]
    ) -> bool:
        """
        Re-encode video with ffmpeg for better compression.

        Args:
            video_path: Path to video file
            settings: Quality settings dictionary

        Returns:
            True if re-encoding successful
        """
        try:
            temp_path = video_path.with_suffix(".temp.mp4")

            cmd = [
                "ffmpeg",
                "-i",
                str(video_path),
                "-c:v",
                "libx264",
                "-crf",
                str(settings["crf"]),
                "-preset",
                "medium",
                "-b:v",
                settings["bitrate"],
                "-movflags",
                "+faststart",
                "-y",
                str(temp_path),
            ]

            process = await asyncio.create_subprocess_exec(
                *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
            )

            stdout, stderr = await process.communicate()

            if process.returncode == 0:
                # Replace original with re-encoded version
                temp_path.replace(video_path)
                logger.info("Video re-encoded successfully with ffmpeg")
                return True
            else:
                logger.warning(f"ffmpeg re-encoding failed: {stderr.decode()}")
                if temp_path.exists():
                    temp_path.unlink()
                return False

        except Exception as e:
            logger.error(f"ffmpeg re-encoding error: {str(e)}")
            return False

    async def synchronize_audio_with_video(
        self, video_path: Path, audio_path: Path, output_path: Optional[Path] = None
    ) -> Dict[str, Any]:
        """
        Synchronize audio track with video.

        This is a key Q2-Q3 2025 feature for audio-visual content generation.

        Args:
            video_path: Path to video file
            audio_path: Path to audio file (voice/music)
            output_path: Optional output path for synchronized video

        Returns:
            Dict with synchronized video metadata
        """
        logger.info(f"Synchronizing audio {audio_path} with video {video_path}")

        if not self._check_ffmpeg_available():
            logger.error("ffmpeg required for audio synchronization")
            raise RuntimeError("ffmpeg not available")

        if not output_path:
            output_path = video_path.parent / f"{video_path.stem}_with_audio.mp4"

        try:
            # Merge audio and video with ffmpeg
            cmd = [
                "ffmpeg",
                "-i",
                str(video_path),
                "-i",
                str(audio_path),
                "-c:v",
                "copy",
                "-c:a",
                "aac",
                "-b:a",
                "192k",
                "-map",
                "0:v:0",
                "-map",
                "1:a:0",
                "-shortest",  # Cut to shortest stream
                "-y",
                str(output_path),
            ]

            process = await asyncio.create_subprocess_exec(
                *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
            )

            stdout, stderr = await process.communicate()

            if process.returncode != 0:
                raise RuntimeError(f"Audio sync failed: {stderr.decode()}")

            # Get video duration
            duration = await self._get_video_duration(output_path)

            return {
                "file_path": str(output_path),
                "file_size": output_path.stat().st_size,
                "duration": duration,
                "has_audio": True,
                "audio_source": str(audio_path),
                "format": "MP4",
            }

        except Exception as e:
            logger.error(f"Audio synchronization failed: {str(e)}")
            raise

    async def _get_video_duration(self, video_path: Path) -> float:
        """
        Get video duration in seconds.

        Args:
            video_path: Path to video file

        Returns:
            Duration in seconds
        """
        try:
            cap = cv2.VideoCapture(str(video_path))
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            cap.release()

            if fps > 0:
                return frame_count / fps
            return 0.0

        except Exception as e:
            logger.error(f"Failed to get video duration: {str(e)}")
            return 0.0

    async def create_storyboard(
        self, scenes: List[Dict[str, Any]], quality: VideoQuality = VideoQuality.HIGH
    ) -> Dict[str, Any]:
        """
        Create a storyboarded video from scene descriptions.

        This is a key Q2-Q3 2025 feature for complex video composition.

        Args:
            scenes: List of scene dicts with 'prompt', 'duration', 'transition'
            quality: Video quality preset

        Returns:
            Dict with storyboard video metadata
        """
        logger.info(f"Creating storyboard with {len(scenes)} scenes")

        try:
            all_frames = []
            scene_markers = []
            current_time = 0.0

            for i, scene in enumerate(scenes):
                prompt = scene.get("prompt", f"Scene {i+1}")
                duration = scene.get("duration", 3.0)
                transition = TransitionType(scene.get("transition", "crossfade"))

                # Generate scene
                scene_frames = await self._generate_scene_frames(
                    prompt, duration, quality
                )

                # Apply transition if not first scene
                if i > 0 and all_frames:
                    transition_frames = self._create_transition(
                        all_frames[-1],
                        scene_frames[0],
                        transition,
                        num_frames=15,  # 0.5s transition at 30fps
                    )
                    all_frames.extend(transition_frames)
                    current_time += 0.5

                all_frames.extend(scene_frames)
                scene_markers.append(
                    {"scene": i + 1, "timestamp": current_time, "prompt": prompt}
                )
                current_time += duration

            # Export final storyboard
            output_filename = (
                f"storyboard_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
            )
            output_path = self.output_dir / output_filename

            success = await self._export_video(all_frames, output_path, quality)

            if not success:
                raise RuntimeError("Failed to export storyboard video")

            settings = self.quality_settings[quality]

            return {
                "file_path": str(output_path),
                "file_size": output_path.stat().st_size,
                "duration": current_time,
                "resolution": f"{settings['resolution'][0]}x{settings['resolution'][1]}",
                "format": "MP4",
                "num_scenes": len(scenes),
                "scene_markers": scene_markers,
                "quality": quality.value,
            }

        except Exception as e:
            logger.error(f"Storyboard creation failed: {str(e)}")
            raise

    async def _generate_scene_frames(
        self, prompt: str, duration: float, quality: VideoQuality
    ) -> List[np.ndarray]:
        """
        Generate frames for a single scene.

        Args:
            prompt: Scene prompt
            duration: Scene duration in seconds
            quality: Video quality preset

        Returns:
            List of frames for the scene
        """
        settings = self.quality_settings[quality]
        fps = settings["fps"]
        num_frames = int(duration * fps)

        # Generate base frame
        base_frame = await self._generate_single_frame(prompt, quality)

        # Create smooth variations for the scene
        frames = []
        for i in range(num_frames):
            # Add subtle animation (slight zoom or pan)
            scale = 1.0 + (i / num_frames) * 0.05  # Slight zoom in
            h, w = base_frame.shape[:2]
            center = (w // 2, h // 2)

            # Create zoom matrix
            M = cv2.getRotationMatrix2D(center, 0, scale)
            frame = cv2.warpAffine(base_frame, M, (w, h))

            frames.append(frame)

        return frames

    def get_supported_transitions(self) -> List[str]:
        """Get list of supported transition types."""
        return [t.value for t in TransitionType]

    def get_supported_qualities(self) -> List[str]:
        """Get list of supported quality presets."""
        return [q.value for q in VideoQuality]
