"""
Reel Generation Service

Handles generation of short-form video reels including:
- Single persona reels
- Duet/reaction reels with split-screen layouts
- Multi-persona collaborative content
- Integration with friend groups for social interactions
"""

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.config.logging import get_logger
from backend.models.content import ContentModel
from backend.models.friend_groups import (
    DuetRequestModel,
)
from backend.models.persona import PersonaModel
from backend.services.video_processing_service import (
    VideoProcessingService,
    VideoQuality,
)

logger = get_logger(__name__)


class ReelLayout:
    """Reel layout configurations for different duet/reaction styles."""

    # Standard vertical short-form video
    STANDARD = {"width": 1080, "height": 1920, "fps": 30}

    # Side-by-side duet layout
    SIDE_BY_SIDE = {
        "width": 1080,
        "height": 1920,
        "layout": "horizontal_split",
        "panels": [
            {"x": 0, "y": 0, "width": 540, "height": 1920},  # Left panel
            {"x": 540, "y": 0, "width": 540, "height": 1920},  # Right panel
        ],
    }

    # Reaction layout (original large, reaction small overlay)
    REACTION = {
        "width": 1080,
        "height": 1920,
        "layout": "overlay",
        "panels": [
            {"x": 0, "y": 0, "width": 1080, "height": 1920},  # Main video
            {
                "x": 720,
                "y": 100,
                "width": 300,
                "height": 400,
                "border": True,
            },  # Reaction overlay
        ],
    }

    # Grid layout for multiple reactions (up to 4)
    GRID_2X2 = {
        "width": 1080,
        "height": 1920,
        "layout": "grid",
        "panels": [
            {"x": 0, "y": 0, "width": 540, "height": 960},  # Top-left
            {"x": 540, "y": 0, "width": 540, "height": 960},  # Top-right
            {"x": 0, "y": 960, "width": 540, "height": 960},  # Bottom-left
            {"x": 540, "y": 960, "width": 540, "height": 960},  # Bottom-right
        ],
    }


class ReelGenerationService:
    """
    Service for generating short-form video reels with AI personas.

    Supports single-persona reels, duets, reactions, and multi-persona content.
    """

    def __init__(
        self, db_session: AsyncSession, output_dir: str = "generated_content/reels"
    ):
        """
        Initialize reel generation service.

        Args:
            db_session: Database session
            output_dir: Directory to store generated reels
        """
        self.db = db_session
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.video_service = VideoProcessingService(str(self.output_dir))

    async def generate_single_reel(
        self,
        persona_id: UUID,
        prompt: str,
        duration: float = 15.0,
        quality: VideoQuality = VideoQuality.HIGH,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Generate a single-persona reel.

        Args:
            persona_id: Persona to feature
            prompt: Content prompt for reel
            duration: Duration in seconds (default 15s for reels)
            quality: Video quality
            **kwargs: Additional generation parameters

        Returns:
            Dict with reel metadata and file path
        """
        try:
            # Get persona
            persona = await self._get_persona(persona_id)
            if not persona:
                raise ValueError(f"Persona {persona_id} not found")

            logger.info(
                f"Generating single reel for persona {persona.name}: {prompt[:50]}..."
            )

            # Generate reel content using AI models
            # For now, create a placeholder structure
            filename = (
                f"reel_{persona.name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
            )
            file_path = self.output_dir / filename

            # In production, this would:
            # 1. Generate video frames using AI image generation
            # 2. Add motion/transitions
            # 3. Sync with generated audio/voice
            # 4. Add text overlays, effects

            # Create placeholder for now
            await self._create_placeholder_video(
                file_path,
                width=1080,
                height=1920,
                duration=duration,
                text=f"{persona.name}: {prompt[:30]}...",
            )

            return {
                "file_path": str(file_path),
                "persona_id": str(persona_id),
                "duration": duration,
                "resolution": "1080x1920",
                "format": "MP4",
                "type": "single_reel",
            }

        except Exception as e:
            logger.error(f"Failed to generate single reel: {str(e)}")
            raise

    async def generate_duet_reel(
        self,
        original_content_id: UUID,
        participant_persona_ids: List[UUID],
        duet_type: str = "side_by_side",
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Generate a duet/reaction reel with multiple personas.

        Args:
            original_content_id: Original reel to duet with
            participant_persona_ids: Personas creating reactions
            duet_type: Layout type (side_by_side, reaction, grid)
            **kwargs: Additional parameters

        Returns:
            Dict with duet reel metadata and file path
        """
        try:
            # Get original content
            original_content = await self._get_content(original_content_id)
            if not original_content:
                raise ValueError(f"Original content {original_content_id} not found")

            # Get participating personas
            personas = []
            for pid in participant_persona_ids:
                persona = await self._get_persona(pid)
                if persona:
                    personas.append(persona)

            if not personas:
                raise ValueError("No valid participant personas found")

            logger.info(
                f"Generating duet reel: {len(personas)} reactions to {original_content.id}"
            )

            # Select layout based on number of participants
            layout = self._select_layout(duet_type, len(personas))

            # Generate reaction videos for each persona
            reaction_videos = []
            for persona in personas:
                reaction = await self._generate_reaction_video(
                    persona, original_content, layout, **kwargs
                )
                reaction_videos.append(reaction)

            # Composite all videos into final duet reel
            output_filename = f"duet_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
            output_path = self.output_dir / output_filename

            await self._composite_duet_video(
                original_video_path=original_content.file_path,
                reaction_videos=reaction_videos,
                output_path=output_path,
                layout=layout,
            )

            return {
                "file_path": str(output_path),
                "original_content_id": str(original_content_id),
                "participant_personas": [str(p.id) for p in personas],
                "duet_type": duet_type,
                "layout": layout,
                "resolution": f"{layout['width']}x{layout['height']}",
                "format": "MP4",
                "type": "duet_reel",
            }

        except Exception as e:
            logger.error(f"Failed to generate duet reel: {str(e)}")
            raise

    async def create_duet_request(
        self,
        original_content_id: UUID,
        participant_personas: List[UUID],
        duet_type: str = "side_by_side",
        layout_config: Optional[Dict[str, Any]] = None,
    ) -> DuetRequestModel:
        """
        Create a duet request in the database.

        Args:
            original_content_id: Original content to duet with
            participant_personas: List of persona IDs to participate
            duet_type: Type of duet layout
            layout_config: Custom layout configuration

        Returns:
            Created duet request model
        """
        try:
            # Get original content to find original persona
            original_content = await self._get_content(original_content_id)
            if not original_content:
                raise ValueError(f"Content {original_content_id} not found")

            # Create duet request
            duet_request = DuetRequestModel(
                original_content_id=original_content_id,
                original_persona_id=original_content.persona_id,
                participant_personas=[str(pid) for pid in participant_personas],
                duet_type=duet_type,
                layout_config=layout_config or {},
                status="pending",
            )

            self.db.add(duet_request)
            await self.db.commit()
            await self.db.refresh(duet_request)

            logger.info(f"Created duet request {duet_request.id}")
            return duet_request

        except Exception as e:
            await self.db.rollback()
            logger.error(f"Failed to create duet request: {str(e)}")
            raise

    async def process_duet_request(self, request_id: UUID) -> Dict[str, Any]:
        """
        Process a pending duet request and generate the duet video.

        Args:
            request_id: Duet request ID

        Returns:
            Generated duet reel metadata
        """
        try:
            # Get duet request
            stmt = select(DuetRequestModel).where(DuetRequestModel.id == request_id)
            result = await self.db.execute(stmt)
            duet_request = result.scalar_one_or_none()

            if not duet_request:
                raise ValueError(f"Duet request {request_id} not found")

            # Update status
            duet_request.status = "in_progress"
            await self.db.commit()

            # Parse participant personas
            participant_ids = [UUID(pid) for pid in duet_request.participant_personas]

            # Generate duet reel
            result = await self.generate_duet_reel(
                original_content_id=duet_request.original_content_id,
                participant_persona_ids=participant_ids,
                duet_type=duet_request.duet_type,
                layout_config=duet_request.layout_config,
            )

            # Save result as content
            # TODO: Create ContentModel entry for the duet

            # Update duet request
            duet_request.status = "completed"
            duet_request.completed_at = datetime.now()
            # duet_request.result_content_id = result_content.id
            await self.db.commit()

            logger.info(f"Completed duet request {request_id}")
            return result

        except Exception as e:
            # Update status to failed
            try:
                stmt = select(DuetRequestModel).where(DuetRequestModel.id == request_id)
                result = await self.db.execute(stmt)
                duet_request = result.scalar_one_or_none()
                if duet_request:
                    duet_request.status = "failed"
                    await self.db.commit()
            except Exception:
                pass

            logger.error(f"Failed to process duet request {request_id}: {str(e)}")
            raise

    def _select_layout(self, duet_type: str, num_participants: int) -> Dict[str, Any]:
        """
        Select appropriate layout based on duet type and participant count.

        Args:
            duet_type: Requested layout type
            num_participants: Number of participating personas

        Returns:
            Layout configuration dict
        """
        if duet_type == "side_by_side":
            return ReelLayout.SIDE_BY_SIDE
        elif duet_type == "reaction":
            return ReelLayout.REACTION
        elif duet_type == "grid" or num_participants > 2:
            return ReelLayout.GRID_2X2
        else:
            return ReelLayout.SIDE_BY_SIDE

    async def _generate_reaction_video(
        self,
        persona: PersonaModel,
        original_content: ContentModel,
        layout: Dict[str, Any],
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Generate a reaction video for a persona.

        Args:
            persona: Persona creating reaction
            original_content: Original content being reacted to
            layout: Layout configuration
            **kwargs: Additional parameters

        Returns:
            Dict with reaction video path and metadata
        """
        # Generate reaction prompt
        _reaction_prompt = (
            f"{persona.name} reacts to content with {persona.personality} personality"
        )

        # For now, create placeholder
        filename = (
            f"reaction_{persona.name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
        )
        file_path = self.output_dir / "reactions" / filename
        file_path.parent.mkdir(exist_ok=True)

        # In production: generate AI reaction video
        await self._create_placeholder_video(
            file_path,
            width=540,
            height=1920,
            duration=15.0,
            text=f"{persona.name} reacts",
        )

        return {
            "persona_id": str(persona.id),
            "file_path": str(file_path),
            "persona_name": persona.name,
        }

    async def _composite_duet_video(
        self,
        original_video_path: str,
        reaction_videos: List[Dict[str, Any]],
        output_path: Path,
        layout: Dict[str, Any],
    ) -> None:
        """
        Composite original and reaction videos into final duet reel.

        Uses ffmpeg to create split-screen or overlay layouts.

        Args:
            original_video_path: Path to original video
            reaction_videos: List of reaction video dicts
            output_path: Output path for composited video
            layout: Layout configuration
        """
        try:
            # For now, just copy the original as placeholder
            # In production, this would use ffmpeg to composite videos

            if Path(original_video_path).exists():
                import shutil

                shutil.copy2(original_video_path, output_path)
            else:
                # Create placeholder
                await self._create_placeholder_video(
                    output_path,
                    width=layout["width"],
                    height=layout["height"],
                    duration=15.0,
                    text="Duet Reel (placeholder)",
                )

            logger.info(f"Composited duet video: {output_path}")

        except Exception as e:
            logger.error(f"Failed to composite duet video: {str(e)}")
            raise

    async def _create_placeholder_video(
        self, output_path: Path, width: int, height: int, duration: float, text: str
    ) -> None:
        """Create a placeholder video file."""
        # Create a simple text file placeholder for now
        # In production, would use ffmpeg or opencv to create actual video
        output_path.write_text(
            f"# Reel Placeholder\n"
            f"# Resolution: {width}x{height}\n"
            f"# Duration: {duration}s\n"
            f"# Content: {text}\n"
            f"# Note: Reel generation requires AI video models\n"
        )
        logger.info(f"Created placeholder reel: {output_path}")

    async def _get_persona(self, persona_id: UUID) -> Optional[PersonaModel]:
        """Get persona by ID."""
        stmt = select(PersonaModel).where(PersonaModel.id == persona_id)
        result = await self.db.execute(stmt)
        return result.scalar_one_or_none()

    async def _get_content(self, content_id: UUID) -> Optional[ContentModel]:
        """Get content by ID."""
        stmt = select(ContentModel).where(ContentModel.id == content_id)
        result = await self.db.execute(stmt)
        return result.scalar_one_or_none()
