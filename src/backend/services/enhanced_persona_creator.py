"""
Enhanced Persona Creator Service

Advanced persona creation wizard with:
- Preset templates (fitness, fashion, gaming, tech, etc.)
- Dropdown selections for physical features
- Personality trait selectors
- 4-image preview generation on creation
- User selects favorite face
- Locks selected face as base_image for consistency
"""

import asyncio
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from uuid import UUID

from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from backend.config.logging import get_logger
from backend.models.persona import BaseImageStatus, PersonaCreate, PersonaModel
from backend.services.ai_models import ai_models

logger = get_logger(__name__)


# Persona Presets

PERSONA_PRESETS = {
    "fitness_influencer": {
        "name_prefix": "Fit",
        "personality": "Energetic, motivational, health-focused. Inspires followers to live their best life through fitness and wellness.",
        "content_themes": [
            "fitness",
            "health",
            "wellness",
            "nutrition",
            "workout routines",
            "motivation",
        ],
        "style_preferences": {
            "tone": "upbeat and encouraging",
            "content_style": "action-oriented, before-after transformations",
            "platform_focus": ["instagram", "tiktok", "youtube"],
        },
        "physical_defaults": {
            "body_type": "athletic",
            "age_range": "25-35",
            "style": "athletic wear, gym fashion",
        },
    },
    "fashion_influencer": {
        "name_prefix": "Style",
        "personality": "Trendy, creative, fashion-forward. Has impeccable taste and knows how to put together stunning outfits.",
        "content_themes": [
            "fashion",
            "style",
            "beauty",
            "trends",
            "outfit ideas",
            "designer reviews",
        ],
        "style_preferences": {
            "tone": "chic and sophisticated",
            "content_style": "aesthetic, curated, magazine-quality",
            "platform_focus": ["instagram", "pinterest", "tiktok"],
        },
        "physical_defaults": {
            "body_type": "model",
            "age_range": "22-30",
            "style": "high fashion, trendy",
        },
    },
    "gaming_streamer": {
        "name_prefix": "Game",
        "personality": "Enthusiastic, competitive, funny. Loves gaming and entertaining audiences with gameplay and commentary.",
        "content_themes": [
            "gaming",
            "esports",
            "game reviews",
            "streaming",
            "tech",
            "gameplay tips",
        ],
        "style_preferences": {
            "tone": "casual and entertaining",
            "content_style": "energetic, reaction-heavy, meme-friendly",
            "platform_focus": ["twitch", "youtube", "tiktok"],
        },
        "physical_defaults": {
            "body_type": "average",
            "age_range": "20-28",
            "style": "casual, gaming gear, hoodies",
        },
    },
    "tech_reviewer": {
        "name_prefix": "Tech",
        "personality": "Knowledgeable, analytical, passionate about technology. Provides in-depth reviews and tech insights.",
        "content_themes": [
            "technology",
            "gadgets",
            "reviews",
            "tutorials",
            "tech news",
            "productivity",
        ],
        "style_preferences": {
            "tone": "professional and informative",
            "content_style": "detailed, well-researched, comparison-based",
            "platform_focus": ["youtube", "twitter", "linkedin"],
        },
        "physical_defaults": {
            "body_type": "average",
            "age_range": "28-40",
            "style": "business casual, modern",
        },
    },
    "lifestyle_blogger": {
        "name_prefix": "Life",
        "personality": "Relatable, authentic, positive. Shares daily life, experiences, and personal growth journey.",
        "content_themes": [
            "lifestyle",
            "daily life",
            "travel",
            "food",
            "personal growth",
            "relationships",
        ],
        "style_preferences": {
            "tone": "friendly and conversational",
            "content_style": "candid, storytelling, behind-the-scenes",
            "platform_focus": ["instagram", "youtube", "blog"],
        },
        "physical_defaults": {
            "body_type": "average",
            "age_range": "25-35",
            "style": "casual chic, comfortable",
        },
    },
    "food_creator": {
        "name_prefix": "Chef",
        "personality": "Creative, passionate about food, enjoys sharing recipes and culinary experiences.",
        "content_themes": [
            "cooking",
            "recipes",
            "food",
            "restaurants",
            "baking",
            "food photography",
        ],
        "style_preferences": {
            "tone": "warm and inviting",
            "content_style": "visual, step-by-step, appetizing",
            "platform_focus": ["instagram", "tiktok", "youtube"],
        },
        "physical_defaults": {
            "body_type": "average",
            "age_range": "28-45",
            "style": "apron, chef's attire, casual",
        },
    },
}


# Physical Feature Options

PHYSICAL_FEATURES = {
    "body_type": [
        "slim",
        "athletic",
        "average",
        "curvy",
        "muscular",
        "model",
        "plus-size",
    ],
    "hair_color": [
        "black",
        "brown",
        "blonde",
        "red",
        "auburn",
        "gray",
        "white",
        "blue",
        "pink",
        "purple",
        "multicolored",
    ],
    "hair_style": [
        "long straight",
        "long wavy",
        "long curly",
        "shoulder-length",
        "short",
        "bob",
        "pixie cut",
        "buzz cut",
        "bald",
        "braided",
        "dreadlocks",
        "bun",
        "ponytail",
    ],
    "eye_color": [
        "brown",
        "blue",
        "green",
        "hazel",
        "gray",
        "amber",
        "black",
    ],
    "skin_tone": [
        "fair",
        "light",
        "medium",
        "tan",
        "olive",
        "brown",
        "dark brown",
        "deep",
    ],
    "age_range": [
        "18-24",
        "25-30",
        "31-40",
        "41-50",
        "50+",
    ],
    "gender": [
        "female",
        "male",
        "non-binary",
        "androgynous",
    ],
    "ethnicity": [
        "african",
        "asian",
        "caucasian",
        "hispanic/latino",
        "middle eastern",
        "mixed",
        "pacific islander",
        "south asian",
    ],
}


# Personality Traits

PERSONALITY_TRAITS = {
    "energy_level": [
        "calm and relaxed",
        "moderate energy",
        "high energy",
        "very energetic",
    ],
    "communication_style": [
        "formal",
        "professional",
        "casual",
        "friendly",
        "humorous",
        "witty",
    ],
    "authenticity": [
        "highly polished",
        "professional",
        "relatable",
        "very authentic",
        "raw and real",
    ],
    "expertise_level": ["beginner", "knowledgeable", "expert", "authority"],
    "engagement_style": [
        "reserved",
        "conversational",
        "very interactive",
        "community-focused",
    ],
}


class PhysicalFeaturesSelection(BaseModel):
    """Model for physical feature selections."""

    body_type: str
    hair_color: str
    hair_style: str
    eye_color: str
    skin_tone: str
    age_range: str
    gender: str
    ethnicity: str
    additional_details: Optional[str] = None


class PersonalitySelection(BaseModel):
    """Model for personality trait selections."""

    energy_level: str
    communication_style: str
    authenticity: str
    expertise_level: str
    engagement_style: str
    custom_traits: Optional[str] = None


class EnhancedPersonaCreate(BaseModel):
    """Enhanced persona creation with preset and selections."""

    name: str
    preset_id: Optional[str] = Field(None, description="Preset template to use")
    physical_features: PhysicalFeaturesSelection
    personality_selection: PersonalitySelection
    content_themes: List[str] = Field(default=[])
    platform_focus: List[str] = Field(default=[])
    custom_appearance_details: Optional[str] = None


class FacePreview(BaseModel):
    """Model for face preview options."""

    preview_id: int
    image_path: str
    image_data: Optional[str] = None  # Base64 encoded
    prompt_used: str
    seed: Optional[int] = None


class EnhancedPersonaCreatorService:
    """
    Service for enhanced persona creation with wizard interface.

    Provides presets, feature selection, and 4-image preview generation.
    """

    def __init__(
        self,
        db_session: AsyncSession,
        output_dir: str = "generated_content/persona_previews",
    ):
        """
        Initialize enhanced persona creator.

        Args:
            db_session: Database session
            output_dir: Directory for preview images
        """
        self.db = db_session
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def get_available_presets(self) -> Dict[str, Any]:
        """Get all available persona presets."""
        return {
            preset_id: {
                "id": preset_id,
                "name": preset_id.replace("_", " ").title(),
                "description": data["personality"],
                "themes": data["content_themes"],
                "platforms": data["style_preferences"]["platform_focus"],
            }
            for preset_id, data in PERSONA_PRESETS.items()
        }

    def get_feature_options(self) -> Dict[str, List[str]]:
        """Get all available physical feature options."""
        return PHYSICAL_FEATURES.copy()

    def get_personality_options(self) -> Dict[str, List[str]]:
        """Get all available personality trait options."""
        return PERSONALITY_TRAITS.copy()

    def build_appearance_description(
        self,
        physical_features: PhysicalFeaturesSelection,
        preset_id: Optional[str] = None,
    ) -> str:
        """
        Build detailed appearance description from selections.

        Args:
            physical_features: Selected physical features
            preset_id: Optional preset for context

        Returns:
            Detailed appearance description string
        """
        parts = []

        # Start with gender, ethnicity, and age
        parts.append(f"A {physical_features.gender} person")
        parts.append(f"of {physical_features.ethnicity} ethnicity")
        parts.append(f"aged {physical_features.age_range}")

        # Add body type
        parts.append(f"with a {physical_features.body_type} build")

        # Add hair description
        parts.append(
            f"with {physical_features.hair_color} {physical_features.hair_style} hair"
        )

        # Add eyes
        parts.append(f"and {physical_features.eye_color} eyes")

        # Add skin tone
        parts.append(f"with {physical_features.skin_tone} skin")

        # Add preset context if provided
        if preset_id and preset_id in PERSONA_PRESETS:
            preset_data = PERSONA_PRESETS[preset_id]
            if "style" in preset_data["physical_defaults"]:
                parts.append(f"wearing {preset_data['physical_defaults']['style']}")

        # Add additional details if provided
        if physical_features.additional_details:
            parts.append(physical_features.additional_details)

        # Join into coherent description
        description = ", ".join(parts)

        # Add portrait framing
        description = f"Professional portrait photo: {description}. High quality, studio lighting, clear facial features, photorealistic."

        return description

    def build_personality_description(
        self,
        personality_selection: PersonalitySelection,
        preset_id: Optional[str] = None,
    ) -> str:
        """
        Build personality description from selections.

        Args:
            personality_selection: Selected personality traits
            preset_id: Optional preset for base personality

        Returns:
            Personality description string
        """
        parts = []

        # Start with preset if provided
        if preset_id and preset_id in PERSONA_PRESETS:
            parts.append(PERSONA_PRESETS[preset_id]["personality"])

        # Add selected traits
        parts.append(f"Has {personality_selection.energy_level} energy")
        parts.append(
            f"communicates in a {personality_selection.communication_style} manner"
        )
        parts.append(f"presents as {personality_selection.authenticity}")
        parts.append(
            f"demonstrates {personality_selection.expertise_level} level knowledge"
        )
        parts.append(
            f"engages with audience in a {personality_selection.engagement_style} way"
        )

        # Add custom traits
        if personality_selection.custom_traits:
            parts.append(personality_selection.custom_traits)

        return ". ".join(parts)

    async def generate_face_previews(
        self,
        appearance_description: str,
        count: int = 4,
        quality: str = "high",
    ) -> List[FacePreview]:
        """
        Generate multiple face preview options for selection.

        Args:
            appearance_description: Detailed appearance description
            count: Number of previews to generate (default 4)
            quality: Quality preset for generation

        Returns:
            List of face preview options
        """
        try:
            logger.info(f"Generating {count} face previews...")

            # Ensure AI models are initialized
            if not ai_models.models_loaded:
                await ai_models.initialize_models()

            previews = []
            tasks = []

            # Generate multiple variations with different seeds
            for i in range(count):
                seed = 1000 + i  # Different seed for each preview
                task = self._generate_single_preview(
                    appearance_description,
                    preview_id=i,
                    seed=seed,
                    quality=quality,
                )
                tasks.append(task)

            # Generate all previews in parallel
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Process results
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.error(f"Preview {i} generation failed: {str(result)}")
                    # Create placeholder
                    previews.append(
                        self._create_placeholder_preview(i, appearance_description)
                    )
                else:
                    previews.append(result)

            logger.info(f"Generated {len(previews)} face previews")
            return previews

        except Exception as e:
            logger.error(f"Failed to generate face previews: {str(e)}")
            # Return placeholder previews
            return [
                self._create_placeholder_preview(i, appearance_description)
                for i in range(count)
            ]

    async def _generate_single_preview(
        self,
        appearance_description: str,
        preview_id: int,
        seed: int,
        quality: str,
    ) -> FacePreview:
        """Generate a single face preview."""
        try:
            # Generate image using AI models
            result = await ai_models.generate_image(
                prompt=appearance_description,
                quality=quality,
                seed=seed,
                width=512,
                height=512,
            )

            # Save preview image
            filename = (
                f"preview_{preview_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            )
            file_path = self.output_dir / filename

            if result.get("image_data"):
                with open(file_path, "wb") as f:
                    f.write(result["image_data"])

            return FacePreview(
                preview_id=preview_id,
                image_path=str(file_path),
                prompt_used=appearance_description,
                seed=seed,
            )

        except Exception as e:
            logger.error(f"Failed to generate preview {preview_id}: {str(e)}")
            return self._create_placeholder_preview(preview_id, appearance_description)

    def _create_placeholder_preview(self, preview_id: int, prompt: str) -> FacePreview:
        """Create a placeholder preview when generation fails."""
        filename = f"preview_{preview_id}_placeholder.txt"
        file_path = self.output_dir / filename

        file_path.write_text(
            f"# Preview {preview_id} Placeholder\n"
            f"# Prompt: {prompt}\n"
            f"# Note: Face generation requires AI image models\n"
        )

        return FacePreview(
            preview_id=preview_id,
            image_path=str(file_path),
            prompt_used=prompt,
        )

    async def create_persona_with_preview(
        self,
        creation_data: EnhancedPersonaCreate,
        selected_preview_id: int,
        preview_image_path: str,
    ) -> PersonaModel:
        """
        Create persona with selected face preview as base image.

        Args:
            creation_data: Enhanced persona creation data
            selected_preview_id: ID of selected preview
            preview_image_path: Path to selected preview image

        Returns:
            Created persona model
        """
        try:
            # Build appearance description
            appearance_description = self.build_appearance_description(
                creation_data.physical_features,
                creation_data.preset_id,
            )

            # Build personality description
            personality_description = self.build_personality_description(
                creation_data.personality_selection,
                creation_data.preset_id,
            )

            # Get content themes
            content_themes = creation_data.content_themes
            if creation_data.preset_id and creation_data.preset_id in PERSONA_PRESETS:
                preset_themes = PERSONA_PRESETS[creation_data.preset_id][
                    "content_themes"
                ]
                content_themes = list(set(content_themes + preset_themes))

            # Create persona
            persona = PersonaModel(
                name=creation_data.name,
                appearance=appearance_description,
                base_appearance_description=appearance_description,
                personality=personality_description,
                content_themes=content_themes,
                style_preferences={
                    "platform_focus": creation_data.platform_focus,
                    "physical_features": creation_data.physical_features.model_dump(),
                    "personality_selection": creation_data.personality_selection.model_dump(),
                },
                base_image_path=preview_image_path,
                appearance_locked=True,  # Lock appearance for consistency
                base_image_status=BaseImageStatus.APPROVED.value,
            )

            self.db.add(persona)
            await self.db.commit()
            await self.db.refresh(persona)

            logger.info(
                f"Created persona {persona.id} ({persona.name}) with locked appearance"
            )

            return persona

        except Exception as e:
            await self.db.rollback()
            logger.error(f"Failed to create persona: {str(e)}")
            raise
