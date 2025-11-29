"""
Persona Models

Database and API models for AI persona management.
"""

import uuid
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator
from sqlalchemy import JSON, Boolean, Column, DateTime, Integer, String, Text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

from backend.database.connection import Base


class ContentRating(str, Enum):
    """Content rating enumeration for persona settings."""

    SFW = "sfw"
    MODERATE = "moderate"
    NSFW = "nsfw"


class BaseImageStatus(str, Enum):
    """Status enumeration for persona base image approval workflow."""

    PENDING_UPLOAD = "pending_upload"  # No image yet, awaiting upload or generation
    DRAFT = "draft"  # Image exists but is not approved
    APPROVED = "approved"  # Image is final baseline, appearance locked
    REJECTED = "rejected"  # Image was rejected, needs replacement


class ImageStyle(str, Enum):
    """Image generation style enumeration for persona visual appearance."""

    PHOTOREALISTIC = "photorealistic"  # Lifelike, realistic photography
    ANIME = "anime"  # Japanese animation style
    CARTOON = "cartoon"  # Western cartoon/comic style
    ARTISTIC = "artistic"  # Painterly, artistic rendering
    THREE_D_RENDER = "3d_render"  # 3D CGI rendering style
    FANTASY = "fantasy"  # Fantasy art style
    CINEMATIC = "cinematic"  # Movie/cinematic style


class LinguisticRegister(str, Enum):
    """Linguistic register for persona voice/speech patterns."""

    BLUE_COLLAR = "blue_collar"  # Casual, working-class speech
    ACADEMIC = "academic"  # Formal, scholarly speech
    TECH_BRO = "tech_bro"  # Silicon Valley tech speak
    STREET = "street"  # Urban slang and colloquialisms
    CORPORATE = "corporate"  # Business jargon
    SOUTHERN = "southern"  # Southern U.S. dialect
    MILLENNIAL = "millennial"  # Millennial internet speak
    GEN_Z = "gen_z"  # Gen Z chaotic and ironic


class WarmthLevel(str, Enum):
    """Warmth level for persona interaction style."""

    COLD = "cold"  # Distant, professional
    NEUTRAL = "neutral"  # Neither warm nor cold
    WARM = "warm"  # Friendly, approachable
    BUDDY = "buddy"  # Very friendly, like a best friend


class PatienceLevel(str, Enum):
    """Patience level for persona interaction style."""

    SHORT_FUSE = "short_fuse"  # Quick to frustration
    NORMAL = "normal"  # Average patience
    PATIENT = "patient"  # Tolerant and understanding
    INFINITE = "infinite"  # Never gets frustrated


class TriggerCategory(str, Enum):
    """Categories for content trigger classification."""

    POSE = "pose"  # Pose-related triggers (selfie, full body, side view)
    STYLE = "style"  # Style triggers (bikini, lingerie, casual, formal)
    VIEW = "view"  # View type triggers (front, side, rear, bust)
    NSFW = "nsfw"  # NSFW content triggers
    SFW = "sfw"  # SFW content triggers
    PLATFORM = "platform"  # Platform-specific triggers (instagram, tiktok, onlyfans)
    ANATOMY = "anatomy"  # Anatomy-focused triggers (face detail, hands, body)
    LIGHTING = "lighting"  # Lighting triggers (natural, studio, warm, cold)
    MOOD = "mood"  # Mood/emotion triggers (happy, seductive, serious)
    LOCATION = "location"  # Location triggers (beach, bedroom, gym, outdoor)
    CLOTHING = "clothing"  # Clothing-specific triggers
    ACTION = "action"  # Action triggers (sitting, standing, walking, posing)
    CUSTOM = "custom"  # User-defined custom triggers


class LoRAConfig(BaseModel):
    """Configuration for a single LoRA in a trigger."""

    name: str = Field(..., description="LoRA model name or path")
    weight: float = Field(
        default=0.8,
        ge=0.0,
        le=2.0,
        description="LoRA weight (0.0-2.0, typical range 0.5-1.0)",
    )
    trigger_word: Optional[str] = Field(
        default=None,
        description="Optional trigger word to inject into prompt when using this LoRA",
    )


class WeightOverrides(BaseModel):
    """Model weight and parameter overrides for a trigger."""

    guidance_scale: Optional[float] = Field(
        default=None, ge=1.0, le=30.0, description="CFG scale override (typically 5-15)"
    )
    num_inference_steps: Optional[int] = Field(
        default=None, ge=1, le=150, description="Number of inference steps override"
    )
    strength: Optional[float] = Field(
        default=None, ge=0.0, le=1.0, description="Image-to-image strength override"
    )
    clip_skip: Optional[int] = Field(
        default=None, ge=0, le=4, description="CLIP skip layers"
    )
    sampler: Optional[str] = Field(
        default=None, description="Sampler override (euler, dpm++, etc.)"
    )
    scheduler: Optional[str] = Field(default=None, description="Scheduler override")


class TriggerConfig(BaseModel):
    """Configuration for a single content trigger."""

    trigger_phrases: List[str] = Field(
        default=[], description="List of phrases that activate this trigger"
    )
    model: Optional[str] = Field(
        default=None,
        description="Primary model to use when triggered (null = use persona default)",
    )
    loras: List[LoRAConfig] = Field(
        default=[], description="LoRAs to stack with their weights"
    )
    positive_prompt: str = Field(
        default="", description="Additional positive prompt elements to inject"
    )
    negative_prompt: str = Field(
        default="", description="Additional negative prompt elements to inject"
    )
    category: TriggerCategory = Field(
        default=TriggerCategory.CUSTOM,
        description="Category for trigger classification",
    )
    view_type: Optional[str] = Field(
        default=None,
        description="View type for this trigger (front_headshot, side_profile, full_frontal, etc.)",
    )
    weight_overrides: Optional[WeightOverrides] = Field(
        default=None, description="Model parameter overrides"
    )
    priority: int = Field(
        default=50,
        ge=0,
        le=100,
        description="Priority for trigger matching (higher = checked first)",
    )
    enabled: bool = Field(default=True, description="Whether this trigger is active")
    description: Optional[str] = Field(
        default=None, description="Human-readable description of this trigger"
    )


class ContentTriggersConfig(BaseModel):
    """Complete content triggers configuration for a persona."""

    triggers: Dict[str, TriggerConfig] = Field(
        default={}, description="Map of trigger_id -> TriggerConfig"
    )
    default_positive_prompt: str = Field(
        default="", description="Default positive prompt to always include"
    )
    default_negative_prompt: str = Field(
        default="ugly, deformed, bad anatomy, blurry, low quality",
        description="Default negative prompt to always include",
    )
    enable_auto_lora_selection: bool = Field(
        default=True,
        description="Enable automatic LoRA selection based on detected attributes",
    )
    enable_multi_model_routing: bool = Field(
        default=True,
        description="Enable routing to different models based on view/pose requirements",
    )


class PersonaModel(Base):
    """
    SQLAlchemy model for AI personas.

    Represents an AI character with appearance, personality, and style preferences
    for consistent content generation across all interactions.
    """

    __tablename__ = "personas"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, index=True)
    name = Column(String(100), nullable=False, index=True)
    appearance = Column(Text, nullable=False)
    personality = Column(Text, nullable=False)
    content_themes = Column(JSON, nullable=False, default=list)
    style_preferences = Column(JSON, nullable=False, default=dict)

    # Content rating and platform controls
    default_content_rating = Column(
        String(20), nullable=False, default="sfw", index=True
    )
    allowed_content_ratings = Column(
        JSON, nullable=False, default=list
    )  # ["sfw"] or ["sfw", "nsfw"]
    platform_restrictions = Column(
        JSON, nullable=False, default=dict
    )  # {"instagram": "sfw_only", "onlyfans": "both"}

    # Visual consistency and appearance locking
    base_appearance_description = Column(
        Text, nullable=True
    )  # Detailed baseline appearance prompt
    base_image_path = Column(
        String(500), nullable=True
    )  # Path to reference image for consistency (legacy, kept for backward compatibility)
    # Expanded base images for complete physical appearance locking and LoRA training
    # Phase 1 - Structural Set (identity and body structure):
    #   front_headshot, side_headshot, right_hand, left_hand, bust,
    #   full_frontal, side_profile, rear_view
    # Phase 2 - Action Set (dynamic poses for LoRA training):
    #   compression_pose (sitting/crouching), extension_pose (reaching/running),
    #   twist_pose (looking back over shoulder)
    # Phase 3 - Background Variation (for regularization):
    #   complex_bg_1, complex_bg_2, complex_bg_3 (character in varied environments)
    base_images = Column(
        JSON, nullable=False, default=dict
    )  # {"front_headshot": "/path/to/img.png", "compression_pose": "/path/to/pose.png", ...}
    appearance_locked = Column(
        Boolean, default=False, index=True
    )  # Prevents overwrites, enables consistency
    base_image_status = Column(
        String(20), default="pending_upload", nullable=False, index=True
    )  # Approval workflow status
    image_style = Column(
        String(20), default="photorealistic", nullable=False, index=True
    )  # Image generation style (photorealistic, anime, cartoon, etc.)

    # Decoupled negative prompt - allows per-persona customization
    # This allows e.g. an Anime persona to NOT have "anime" in the negative prompt
    default_negative_prompt = Column(
        Text,
        nullable=False,
        default="ugly, blurry, low quality, distorted, deformed, bad anatomy",
    )  # Persona-specific negative prompt for image generation

    # ==================== DETAILED PHYSICAL APPEARANCE ====================
    # These fields allow fine-grained control over persona appearance generation
    height = Column(
        String(50), nullable=True
    )  # e.g., "5'6\"", "170cm", "tall", "petite"
    weight = Column(
        String(50), nullable=True
    )  # e.g., "130lbs", "60kg", "athletic", "curvy"
    hair_color = Column(
        String(50), nullable=True
    )  # e.g., "blonde", "brunette", "red", "black", "pink"
    hair_style = Column(
        String(100), nullable=True
    )  # e.g., "long and wavy", "short bob", "ponytail", "braids"
    eye_color = Column(
        String(50), nullable=True
    )  # e.g., "blue", "brown", "green", "hazel", "heterochromia"
    skin_tone = Column(
        String(50), nullable=True
    )  # e.g., "fair", "olive", "tan", "dark", "pale"
    measurements = Column(
        String(100), nullable=True
    )  # e.g., "34-24-36", "hourglass figure"
    cup_size = Column(String(20), nullable=True)  # e.g., "A", "B", "C", "D", "DD"
    muscle_tone = Column(
        String(50), nullable=True
    )  # e.g., "toned", "athletic", "soft", "muscular", "lean"
    build_type = Column(
        String(50), nullable=True
    )  # e.g., "slim", "athletic", "curvy", "petite", "plus-size"
    sex = Column(
        String(20), nullable=True
    )  # e.g., "female", "male", "non-binary", "trans"
    sexual_orientation = Column(
        String(50), nullable=True
    )  # e.g., "straight", "bisexual", "lesbian", "gay", "pansexual"
    turn_ons = Column(
        JSON, nullable=False, default=lambda: []
    )  # List of turn-ons for NSFW content generation
    turn_offs = Column(
        JSON, nullable=False, default=lambda: []
    )  # List of turn-offs/boundaries for content generation
    distinctive_features = Column(
        Text, nullable=True
    )  # e.g., "small mole on left cheek", "tattoo sleeve", "dimples"
    age_appearance = Column(
        String(50), nullable=True
    )  # e.g., "early 20s", "mid 30s", "mature"
    ethnicity = Column(
        String(100), nullable=True
    )  # e.g., "Caucasian", "Asian", "Latina", "African American", "Mixed"
    body_modifications = Column(
        JSON, nullable=False, default=lambda: []
    )  # e.g., ["nose piercing", "belly button piercing", "tattoo on lower back"]

    # Content generation preferences
    default_image_resolution = Column(
        String(20), default="1024x1024", nullable=False
    )  # Default resolution for image generation (512x512, 1024x1024, 2048x2048)
    default_video_resolution = Column(
        String(20), default="1920x1080", nullable=False
    )  # Default resolution for video generation (720p, 1080p, 4k)
    post_style = Column(
        String(50), default="casual", nullable=False
    )  # Post style (casual, professional, artistic, provocative, etc.)
    video_types = Column(
        JSON, nullable=False, default=list
    )  # Preferred video types (["short_clip", "story", "reel", "long_form"])
    nsfw_model_preference = Column(
        String(100), nullable=True
    )  # Preferred NSFW model (e.g., "flux-nsfw-highress", "darkblueaphrodite")
    generation_quality = Column(
        String(20), default="standard", nullable=False
    )  # Default quality level (draft, standard, hd, premium)

    # ==================== AI MODEL PREFERENCES PER GENERATION TYPE ====================
    # These allow fine-grained control of which AI model to use for each content type
    text_model_preference = Column(
        String(200), nullable=True
    )  # Preferred text/chat model (e.g., "llama3.1:8b", "qwen2.5:14b", "dolphin-mixtral")
    image_model_preference = Column(
        String(200), nullable=True
    )  # Preferred image generation model (e.g., "sdxl-base", "flux.1-dev", "stable-diffusion-xl")
    video_model_preference = Column(
        String(200), nullable=True
    )  # Preferred video generation model (e.g., "animatediff", "stable-video-diffusion")
    voice_model_preference = Column(
        String(200), nullable=True
    )  # Preferred voice/TTS model (e.g., "xtts-v2", "elevenlabs", "coqui-tts")

    is_active = Column(Boolean, default=True, index=True)
    generation_count = Column(Integer, default=0)

    # ==================== PERSONA SOUL FIELDS ====================
    # These fields capture the "soul" of the persona for human-like responses

    # Origin & Demographics (The "Roots")
    hometown = Column(
        String(200), nullable=True
    )  # e.g., "South Boston", "Rural Texas", "Silicon Valley"
    current_location = Column(
        String(200), nullable=True
    )  # e.g., "Moved to Florida for taxes", "Studio apartment in NYC"
    generation_age = Column(
        String(100), nullable=True
    )  # e.g., "Gen X - cynical and latchkey", "Boomer - traditional"
    education_level = Column(
        String(200), nullable=True
    )  # e.g., "PhD in Physics", "School of Hard Knocks", "College dropout"

    # Psychological Profile (The "Engine")
    mbti_type = Column(
        String(50), nullable=True
    )  # e.g., "ESTP - The Entrepreneur", "INTJ - The Architect"
    enneagram_type = Column(
        String(50), nullable=True
    )  # e.g., "Type 8 - The Challenger", "Type 4 - The Individualist"
    political_alignment = Column(
        String(100), nullable=True
    )  # e.g., "Libertarian-leaning", "Old School Liberal", "Apolitical Nihilist"
    risk_tolerance = Column(
        String(100), nullable=True
    )  # e.g., "Safety First", "Move fast and break things", "Hold my beer"
    optimism_cynicism_scale = Column(
        Integer, nullable=True
    )  # 1-10 scale: 1=deeply cynical, 10=eternally optimistic

    # Voice & Speech Patterns (The "Interface")
    linguistic_register = Column(
        String(50), default="blue_collar", nullable=False
    )  # blue_collar, academic, tech_bro, street, corporate, southern, millennial, gen_z
    typing_quirks = Column(
        JSON, nullable=False, default=dict
    )  # {"capitalization": "all lowercase", "emoji_usage": "ironic only", "punctuation": "uses ... a lot"}
    signature_phrases = Column(
        JSON, nullable=False, default=list
    )  # ["That dog won't hunt", "Clown world", "Big oof", "Let's rock and roll"]
    trigger_topics = Column(
        JSON, nullable=False, default=list
    )  # Topics that get them excited or angry: ["Taxes", "Crypto", "Bad drivers"]

    # Backstory & Lore (The "Context")
    day_job = Column(
        String(200), nullable=True
    )  # e.g., "Small business owner", "Shift manager", "Retired cop"
    war_story = Column(
        Text, nullable=True
    )  # Defining life event: "Lost a fortune in 2008", "Built their own house"
    vices_hobbies = Column(
        JSON, nullable=False, default=list
    )  # ["Cigars and Poker", "Video games and Energy drinks", "Gardening and gossip"]

    # Anti-Pattern (What they are NOT)
    forbidden_phrases = Column(
        JSON, nullable=False, default=list
    )  # Phrases this persona would NEVER say: ["I feel that", "Synergy", "Holistic", "Safe space"]
    warmth_level = Column(
        String(20), default="warm", nullable=False
    )  # cold, neutral, warm, buddy
    patience_level = Column(
        String(20), default="normal", nullable=False
    )  # short_fuse, normal, patient, infinite

    # ==================== TRIGGER-BASED MODEL ORCHESTRATION ====================
    # content_triggers: Configurable trigger words that route to specific models/LoRAs
    # Each trigger can have:
    #   - trigger_phrases: List of phrases that activate this trigger
    #   - model: Primary model to use when triggered
    #   - loras: List of LoRAs to stack with weights
    #   - positive_prompt: Additional positive prompt elements
    #   - negative_prompt: Additional negative prompt elements
    #   - category: Trigger category (pose, style, view, nsfw, platform, anatomy, etc.)
    #   - weight_overrides: Model-specific weight overrides
    #   - priority: Priority for trigger matching (higher = checked first)
    # Example structure:
    # {
    #   "selfie_trigger": {
    #     "trigger_phrases": ["take a selfie", "send selfie", "selfie"],
    #     "model": "realvisxl",
    #     "loras": [{"name": "face_detail_lora", "weight": 0.8}, {"name": "skin_texture_lora", "weight": 0.5}],
    #     "positive_prompt": "front facing, looking at camera, natural lighting",
    #     "negative_prompt": "side view, profile, back view",
    #     "category": "pose",
    #     "view_type": "front_headshot",
    #     "weight_overrides": {"guidance_scale": 7.5, "num_inference_steps": 30},
    #     "priority": 100
    #   },
    #   "bikini_trigger": {
    #     "trigger_phrases": ["bikini", "swimsuit", "beach"],
    #     "model": "nsfw_model_v2",
    #     "loras": [{"name": "body_detail_lora", "weight": 0.7}],
    #     "positive_prompt": "beach setting, natural sunlight, summer vibes",
    #     "negative_prompt": "indoor, winter, clothed",
    #     "category": "style",
    #     "priority": 80
    #   }
    # }
    content_triggers = Column(JSON, nullable=False, default=lambda: {})

    # Timestamps
    created_at = Column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    # Relationships
    content = relationship("ContentModel", back_populates="persona", lazy="dynamic")


class PersonaCreate(BaseModel):
    """API model for creating new personas."""

    name: str = Field(min_length=2, max_length=100, description="Persona display name")
    appearance: str = Field(
        min_length=10,
        max_length=2000,
        description="Physical appearance description for image generation",
    )
    personality: str = Field(
        min_length=10,
        max_length=2000,
        description="Personality traits and characteristics",
    )
    content_themes: List[str] = Field(
        default=[],
        max_length=10,
        description="Content themes this persona specializes in",
    )
    style_preferences: Dict[str, Any] = Field(
        default={}, description="Style and aesthetic preferences"
    )
    default_content_rating: ContentRating = Field(
        default=ContentRating.SFW, description="Default content rating for this persona"
    )
    allowed_content_ratings: List[ContentRating] = Field(
        default=[ContentRating.SFW],
        description="Content ratings this persona is allowed to generate",
    )
    platform_restrictions: Dict[str, str] = Field(
        default={}, description="Platform-specific content restrictions"
    )
    base_appearance_description: Optional[str] = Field(
        default=None,
        max_length=5000,
        description="Detailed baseline appearance description for visual consistency",
    )
    base_image_path: Optional[str] = Field(
        default=None,
        max_length=500,
        description="Path to reference image for visual consistency (e.g., /models/base_images/persona_ref.jpg)",
    )
    base_images: Dict[str, str] = Field(
        default={},
        description=(
            "Expanded base images for complete appearance and LoRA training. "
            "Structural: front_headshot, side_headshot, right_hand, left_hand, bust, full_frontal, side_profile, rear_view. "
            "Action: compression_pose, extension_pose, twist_pose. "
            "Background: complex_bg_1, complex_bg_2, complex_bg_3"
        ),
    )
    appearance_locked: bool = Field(
        default=False,
        description="When True, locks appearance and enables visual consistency features",
    )
    base_image_status: BaseImageStatus = Field(
        default=BaseImageStatus.PENDING_UPLOAD,
        description="Status of the base image in the approval workflow",
    )
    image_style: ImageStyle = Field(
        default=ImageStyle.PHOTOREALISTIC,
        description="Image generation style (photorealistic, anime, cartoon, etc.)",
    )
    default_negative_prompt: str = Field(
        default="ugly, blurry, low quality, distorted, deformed, bad anatomy",
        max_length=2000,
        description=(
            "Persona-specific negative prompt for image generation. "
            "Allows decoupling from style-based defaults, e.g. anime personas won't have 'anime' in their negative."
        ),
    )

    # ==================== DETAILED PHYSICAL APPEARANCE ====================
    height: Optional[str] = Field(
        default=None,
        max_length=50,
        description="Height (e.g., '5\\'6\"', '170cm', 'tall', 'petite')",
    )
    weight: Optional[str] = Field(
        default=None,
        max_length=50,
        description="Weight (e.g., '130lbs', '60kg', 'athletic', 'curvy')",
    )
    hair_color: Optional[str] = Field(
        default=None,
        max_length=50,
        description="Hair color (e.g., 'blonde', 'brunette', 'red', 'black', 'pink')",
    )
    hair_style: Optional[str] = Field(
        default=None,
        max_length=100,
        description="Hair style (e.g., 'long and wavy', 'short bob', 'ponytail', 'braids')",
    )
    eye_color: Optional[str] = Field(
        default=None,
        max_length=50,
        description="Eye color (e.g., 'blue', 'brown', 'green', 'hazel')",
    )
    skin_tone: Optional[str] = Field(
        default=None,
        max_length=50,
        description="Skin tone (e.g., 'fair', 'olive', 'tan', 'dark', 'pale')",
    )
    measurements: Optional[str] = Field(
        default=None,
        max_length=100,
        description="Body measurements (e.g., '34-24-36', 'hourglass figure')",
    )
    cup_size: Optional[str] = Field(
        default=None,
        max_length=20,
        description="Cup size (e.g., 'A', 'B', 'C', 'D', 'DD')",
    )
    muscle_tone: Optional[str] = Field(
        default=None,
        max_length=50,
        description="Muscle tone (e.g., 'toned', 'athletic', 'soft', 'muscular', 'lean')",
    )
    build_type: Optional[str] = Field(
        default=None,
        max_length=50,
        description="Build type (e.g., 'slim', 'athletic', 'curvy', 'petite', 'plus-size')",
    )
    sex: Optional[str] = Field(
        default=None,
        max_length=20,
        description="Sex (e.g., 'female', 'male', 'non-binary', 'trans')",
    )
    sexual_orientation: Optional[str] = Field(
        default=None,
        max_length=50,
        description="Sexual orientation (e.g., 'straight', 'bisexual', 'lesbian', 'gay', 'pansexual')",
    )
    turn_ons: List[str] = Field(
        default=[],
        description="Turn-ons for NSFW content generation",
    )
    turn_offs: List[str] = Field(
        default=[],
        description="Turn-offs/boundaries for content generation",
    )
    distinctive_features: Optional[str] = Field(
        default=None,
        max_length=2000,
        description="Distinctive features (e.g., 'small mole on left cheek', 'tattoo sleeve', 'dimples')",
    )
    age_appearance: Optional[str] = Field(
        default=None,
        max_length=50,
        description="Apparent age (e.g., 'early 20s', 'mid 30s', 'mature')",
    )
    ethnicity: Optional[str] = Field(
        default=None,
        max_length=100,
        description="Ethnicity (e.g., 'Caucasian', 'Asian', 'Latina', 'African American', 'Mixed')",
    )
    body_modifications: List[str] = Field(
        default=[],
        description="Body modifications (e.g., ['nose piercing', 'belly button piercing', 'tattoo on lower back'])",
    )

    default_image_resolution: str = Field(
        default="1024x1024",
        description="Default resolution for image generation (512x512, 1024x1024, 2048x2048)",
    )
    default_video_resolution: str = Field(
        default="1920x1080",
        description="Default resolution for video generation (1280x720, 1920x1080, 3840x2160)",
    )
    post_style: str = Field(
        default="casual",
        description="Post style preference (casual, professional, artistic, provocative, playful)",
    )
    video_types: List[str] = Field(
        default=[],
        description="Preferred video types (short_clip, story, reel, long_form, tutorial)",
    )
    nsfw_model_preference: Optional[str] = Field(
        default=None,
        description="Preferred NSFW model (flux-nsfw-highress, darkblueaphrodite, modifier_sexual_coaching)",
    )
    generation_quality: str = Field(
        default="standard",
        description="Default quality level for content generation (draft, standard, hd, premium)",
    )

    # ==================== AI MODEL PREFERENCES PER GENERATION TYPE ====================
    text_model_preference: Optional[str] = Field(
        default=None,
        max_length=200,
        description="Preferred text/chat model (e.g., 'llama3.1:8b', 'qwen2.5:14b', 'dolphin-mixtral')",
    )
    image_model_preference: Optional[str] = Field(
        default=None,
        max_length=200,
        description="Preferred image generation model (e.g., 'sdxl-base', 'flux.1-dev')",
    )
    video_model_preference: Optional[str] = Field(
        default=None,
        max_length=200,
        description="Preferred video generation model (e.g., 'animatediff', 'stable-video-diffusion')",
    )
    voice_model_preference: Optional[str] = Field(
        default=None,
        max_length=200,
        description="Preferred voice/TTS model (e.g., 'xtts-v2', 'elevenlabs', 'coqui-tts')",
    )

    # ==================== PERSONA SOUL FIELDS ====================
    # Origin & Demographics (The "Roots")
    hometown: Optional[str] = Field(
        default=None,
        max_length=200,
        description="Where they're from (e.g., 'South Boston', 'Nashville, TN', 'Silicon Valley')",
    )
    current_location: Optional[str] = Field(
        default=None,
        max_length=200,
        description="Where they live now (e.g., 'Miami, FL for the weather and liberty')",
    )
    generation_age: Optional[str] = Field(
        default=None,
        max_length=100,
        description="Generation/age context (e.g., 'Gen X - cynical and latchkey', 'Millennial - 24 years old')",
    )
    education_level: Optional[str] = Field(
        default=None,
        max_length=200,
        description="Education background (e.g., 'PhD in Physics', 'School of Hard Knocks', 'Liberal arts college dropout')",
    )

    # Psychological Profile (The "Engine")
    mbti_type: Optional[str] = Field(
        default=None,
        max_length=50,
        description="MBTI type (e.g., 'ESTP - The Entrepreneur', 'INTJ - The Architect')",
    )
    enneagram_type: Optional[str] = Field(
        default=None,
        max_length=50,
        description="Enneagram type (e.g., 'Type 8 - The Challenger', 'Type 4 - The Individualist')",
    )
    political_alignment: Optional[str] = Field(
        default=None,
        max_length=100,
        description="Political/social alignment (e.g., 'Right-leaning / Liberty-focused', 'Apolitical Nihilist')",
    )
    risk_tolerance: Optional[str] = Field(
        default=None,
        max_length=100,
        description="Risk attitude (e.g., 'Safety First', 'Move fast and break things', 'Hold my beer')",
    )
    optimism_cynicism_scale: Optional[int] = Field(
        default=None,
        ge=1,
        le=10,
        description="1-10 scale: 1=deeply cynical, 10=eternally optimistic",
    )

    # Voice & Speech Patterns (The "Interface")
    linguistic_register: LinguisticRegister = Field(
        default=LinguisticRegister.BLUE_COLLAR,
        description="Speech style (blue_collar, academic, tech_bro, street, corporate, southern, millennial, gen_z)",
    )
    typing_quirks: Dict[str, Any] = Field(
        default={},
        description="Typing style quirks: {capitalization: 'lowercase for aesthetic', emoji_usage: 'uses 🇺🇸💅😂', punctuation: 'occasional ALL CAPS'}",
    )
    signature_phrases: List[str] = Field(
        default=[],
        description="Phrases they use often (e.g., ['Based', 'Cringe', 'Y\\'all', 'Facts', 'Bet'])",
    )
    trigger_topics: List[str] = Field(
        default=[],
        description="Topics that get them excited or angry (e.g., ['Cancel culture', 'Gas prices', 'Crypto'])",
    )

    # Backstory & Lore (The "Context")
    day_job: Optional[str] = Field(
        default=None,
        max_length=200,
        description="What they do for work (e.g., 'Political Commentator / Streamer', 'Small business owner')",
    )
    war_story: Optional[str] = Field(
        default=None,
        max_length=2000,
        description="Defining life event (e.g., 'Started posting rants in her car and went viral')",
    )
    vices_hobbies: List[str] = Field(
        default=[],
        description="Vices and hobbies (e.g., ['Range days', 'Debating leftists on Twitter', 'Country concerts'])",
    )

    # Anti-Pattern (What they are NOT)
    forbidden_phrases: List[str] = Field(
        default=[],
        description="Phrases this persona would NEVER say (e.g., ['I feel that', 'Synergy', 'As an AI'])",
    )
    warmth_level: WarmthLevel = Field(
        default=WarmthLevel.WARM,
        description="Interaction warmth (cold, neutral, warm, buddy)",
    )
    patience_level: PatienceLevel = Field(
        default=PatienceLevel.NORMAL,
        description="Patience level (short_fuse, normal, patient, infinite)",
    )

    # ==================== TRIGGER-BASED MODEL ORCHESTRATION ====================
    content_triggers: Dict[str, Any] = Field(
        default={},
        description=(
            "Trigger-based model orchestration configuration. "
            "Maps trigger_id to TriggerConfig with: trigger_phrases, model, loras (with weights), "
            "positive_prompt, negative_prompt, category, view_type, weight_overrides, priority. "
            "Enables automatic model/LoRA selection based on detected keywords, poses, styles, etc."
        ),
    )

    @field_validator("name")
    @classmethod
    def validate_name(cls, v: str) -> str:
        """Validate persona name for safety and appropriateness."""
        # Basic HTML injection prevention
        if any(char in v for char in ["<", ">", "&", "'"]):
            raise ValueError("Name contains invalid characters")
        return v.strip()

    @field_validator("content_themes")
    @classmethod
    def validate_themes(cls, v: List[str]) -> List[str]:
        """Validate content themes."""
        if len(v) > 10:
            raise ValueError("Maximum 10 content themes allowed")

        # Basic content moderation
        inappropriate_themes = [
            "illegal activity",
            "hate speech",
            "violence",
            "adult content",
        ]
        for theme in v:
            if any(bad in theme.lower() for bad in inappropriate_themes):
                raise ValueError(f"Inappropriate content theme: {theme}")

        return v

    @field_validator("platform_restrictions")
    @classmethod
    def validate_platform_restrictions(cls, v: Dict[str, str]) -> Dict[str, str]:
        """Validate platform restrictions values."""
        valid_restrictions = ["sfw_only", "moderate_allowed", "both", "all"]

        for platform, restriction in v.items():
            if restriction.lower() not in valid_restrictions:
                raise ValueError(
                    f"Invalid restriction '{restriction}' for platform '{platform}'. "
                    f"Must be one of: {', '.join(valid_restrictions)}"
                )

        return v


class PersonaUpdate(BaseModel):
    """API model for updating existing personas."""

    name: Optional[str] = Field(None, min_length=2, max_length=100)
    appearance: Optional[str] = Field(None, min_length=10, max_length=2000)
    personality: Optional[str] = Field(None, min_length=10, max_length=2000)
    content_themes: Optional[List[str]] = Field(None, max_length=10)
    style_preferences: Optional[Dict[str, Any]] = None
    default_content_rating: Optional[ContentRating] = None
    allowed_content_ratings: Optional[List[ContentRating]] = None
    platform_restrictions: Optional[Dict[str, str]] = None
    is_active: Optional[bool] = None
    base_appearance_description: Optional[str] = Field(
        None,
        max_length=5000,
        description="Detailed baseline appearance description for visual consistency",
    )
    base_image_path: Optional[str] = Field(
        None,
        max_length=500,
        description="Path to reference image for visual consistency",
    )
    base_images: Optional[Dict[str, str]] = Field(
        None,
        description=(
            "Expanded base images for complete appearance and LoRA training. "
            "Structural: front_headshot, side_headshot, right_hand, left_hand, bust, full_frontal, side_profile, rear_view. "
            "Action: compression_pose, extension_pose, twist_pose. "
            "Background: complex_bg_1, complex_bg_2, complex_bg_3"
        ),
    )
    appearance_locked: Optional[bool] = Field(
        None,
        description="When True, locks appearance and enables visual consistency features",
    )
    base_image_status: Optional[BaseImageStatus] = Field(
        None, description="Status of the base image in the approval workflow"
    )
    image_style: Optional[ImageStyle] = Field(
        None,
        description="Image generation style (photorealistic, anime, cartoon, etc.)",
    )
    default_negative_prompt: Optional[str] = Field(
        None,
        max_length=2000,
        description="Persona-specific negative prompt for image generation",
    )

    # ==================== DETAILED PHYSICAL APPEARANCE ====================
    height: Optional[str] = Field(None, max_length=50, description="Height")
    weight: Optional[str] = Field(None, max_length=50, description="Weight")
    hair_color: Optional[str] = Field(None, max_length=50, description="Hair color")
    hair_style: Optional[str] = Field(None, max_length=100, description="Hair style")
    eye_color: Optional[str] = Field(None, max_length=50, description="Eye color")
    skin_tone: Optional[str] = Field(None, max_length=50, description="Skin tone")
    measurements: Optional[str] = Field(
        None, max_length=100, description="Body measurements"
    )
    cup_size: Optional[str] = Field(None, max_length=20, description="Cup size")
    muscle_tone: Optional[str] = Field(None, max_length=50, description="Muscle tone")
    build_type: Optional[str] = Field(None, max_length=50, description="Build type")
    sex: Optional[str] = Field(None, max_length=20, description="Sex")
    sexual_orientation: Optional[str] = Field(
        None, max_length=50, description="Sexual orientation"
    )
    turn_ons: Optional[List[str]] = Field(None, description="Turn-ons for NSFW content")
    turn_offs: Optional[List[str]] = Field(None, description="Turn-offs/boundaries")
    distinctive_features: Optional[str] = Field(
        None, max_length=2000, description="Distinctive features"
    )
    age_appearance: Optional[str] = Field(
        None, max_length=50, description="Apparent age"
    )
    ethnicity: Optional[str] = Field(None, max_length=100, description="Ethnicity")
    body_modifications: Optional[List[str]] = Field(
        None, description="Body modifications"
    )

    default_image_resolution: Optional[str] = Field(
        None, description="Default resolution for image generation"
    )
    default_video_resolution: Optional[str] = Field(
        None, description="Default resolution for video generation"
    )
    post_style: Optional[str] = Field(None, description="Post style preference")
    video_types: Optional[List[str]] = Field(None, description="Preferred video types")
    nsfw_model_preference: Optional[str] = Field(
        None, description="Preferred NSFW model"
    )
    generation_quality: Optional[str] = Field(
        None, description="Default quality level for content generation"
    )

    # ==================== AI MODEL PREFERENCES PER GENERATION TYPE ====================
    text_model_preference: Optional[str] = Field(
        None, max_length=200, description="Preferred text/chat model"
    )
    image_model_preference: Optional[str] = Field(
        None, max_length=200, description="Preferred image generation model"
    )
    video_model_preference: Optional[str] = Field(
        None, max_length=200, description="Preferred video generation model"
    )
    voice_model_preference: Optional[str] = Field(
        None, max_length=200, description="Preferred voice/TTS model"
    )

    # ==================== PERSONA SOUL FIELDS ====================
    # Origin & Demographics (The "Roots")
    hometown: Optional[str] = Field(
        None,
        max_length=200,
        description="Where they're from",
    )
    current_location: Optional[str] = Field(
        None,
        max_length=200,
        description="Where they live now",
    )
    generation_age: Optional[str] = Field(
        None,
        max_length=100,
        description="Generation/age context",
    )
    education_level: Optional[str] = Field(
        None,
        max_length=200,
        description="Education background",
    )

    # Psychological Profile (The "Engine")
    mbti_type: Optional[str] = Field(
        None,
        max_length=50,
        description="MBTI type",
    )
    enneagram_type: Optional[str] = Field(
        None,
        max_length=50,
        description="Enneagram type",
    )
    political_alignment: Optional[str] = Field(
        None,
        max_length=100,
        description="Political/social alignment",
    )
    risk_tolerance: Optional[str] = Field(
        None,
        max_length=100,
        description="Risk attitude",
    )
    optimism_cynicism_scale: Optional[int] = Field(
        None,
        ge=1,
        le=10,
        description="1-10 scale: 1=cynical, 10=optimistic",
    )

    # Voice & Speech Patterns (The "Interface")
    linguistic_register: Optional[LinguisticRegister] = Field(
        None,
        description="Speech style",
    )
    typing_quirks: Optional[Dict[str, Any]] = Field(
        None,
        description="Typing style quirks",
    )
    signature_phrases: Optional[List[str]] = Field(
        None,
        description="Phrases they use often",
    )
    trigger_topics: Optional[List[str]] = Field(
        None,
        description="Topics that excite/anger them",
    )

    # Backstory & Lore (The "Context")
    day_job: Optional[str] = Field(
        None,
        max_length=200,
        description="What they do for work",
    )
    war_story: Optional[str] = Field(
        None,
        max_length=2000,
        description="Defining life event",
    )
    vices_hobbies: Optional[List[str]] = Field(
        None,
        description="Vices and hobbies",
    )

    # Anti-Pattern (What they are NOT)
    forbidden_phrases: Optional[List[str]] = Field(
        None,
        description="Phrases this persona would NEVER say",
    )
    warmth_level: Optional[WarmthLevel] = Field(
        None,
        description="Interaction warmth",
    )
    patience_level: Optional[PatienceLevel] = Field(
        None,
        description="Patience level",
    )

    # ==================== TRIGGER-BASED MODEL ORCHESTRATION ====================
    content_triggers: Optional[Dict[str, Any]] = Field(
        None,
        description=(
            "Trigger-based model orchestration configuration. "
            "Maps trigger_id to TriggerConfig with: trigger_phrases, model, loras (with weights), "
            "positive_prompt, negative_prompt, category, view_type, weight_overrides, priority."
        ),
    )

    @field_validator("platform_restrictions")
    @classmethod
    def validate_platform_restrictions(
        cls, v: Optional[Dict[str, str]]
    ) -> Optional[Dict[str, str]]:
        """Validate platform restrictions values."""
        if v is None:
            return v

        valid_restrictions = ["sfw_only", "moderate_allowed", "both", "all"]

        for platform, restriction in v.items():
            if restriction.lower() not in valid_restrictions:
                raise ValueError(
                    f"Invalid restriction '{restriction}' for platform '{platform}'. "
                    f"Must be one of: {', '.join(valid_restrictions)}"
                )

        return v


class PersonaResponse(BaseModel):
    """API model for persona responses."""

    id: uuid.UUID
    name: str
    appearance: str
    personality: str
    content_themes: List[str]
    style_preferences: Dict[str, Any]
    default_content_rating: str
    allowed_content_ratings: List[str]
    platform_restrictions: Dict[str, str]
    is_active: bool
    generation_count: int
    created_at: datetime
    updated_at: datetime
    base_appearance_description: Optional[str] = None
    base_image_path: Optional[str] = None
    # Expanded base images for complete appearance and LoRA training
    # Structural: front_headshot, side_headshot, right_hand, left_hand, bust, full_frontal, side_profile, rear_view
    # Action: compression_pose, extension_pose, twist_pose
    # Background: complex_bg_1, complex_bg_2, complex_bg_3
    base_images: Dict[str, str] = {}
    appearance_locked: bool = False
    base_image_status: str = "pending_upload"
    image_style: str = "photorealistic"
    default_negative_prompt: str = (
        "ugly, blurry, low quality, distorted, deformed, bad anatomy"
    )

    # ==================== DETAILED PHYSICAL APPEARANCE ====================
    height: Optional[str] = None
    weight: Optional[str] = None
    hair_color: Optional[str] = None
    hair_style: Optional[str] = None
    eye_color: Optional[str] = None
    skin_tone: Optional[str] = None
    measurements: Optional[str] = None
    cup_size: Optional[str] = None
    muscle_tone: Optional[str] = None
    build_type: Optional[str] = None
    sex: Optional[str] = None
    sexual_orientation: Optional[str] = None
    turn_ons: List[str] = []
    turn_offs: List[str] = []
    distinctive_features: Optional[str] = None
    age_appearance: Optional[str] = None
    ethnicity: Optional[str] = None
    body_modifications: List[str] = []

    default_image_resolution: str = "1024x1024"
    default_video_resolution: str = "1920x1080"
    post_style: str = "casual"
    video_types: List[str] = []
    nsfw_model_preference: Optional[str] = None
    generation_quality: str = "standard"

    # ==================== AI MODEL PREFERENCES PER GENERATION TYPE ====================
    text_model_preference: Optional[str] = None
    image_model_preference: Optional[str] = None
    video_model_preference: Optional[str] = None
    voice_model_preference: Optional[str] = None

    # ==================== PERSONA SOUL FIELDS ====================
    # Origin & Demographics (The "Roots")
    hometown: Optional[str] = None
    current_location: Optional[str] = None
    generation_age: Optional[str] = None
    education_level: Optional[str] = None

    # Psychological Profile (The "Engine")
    mbti_type: Optional[str] = None
    enneagram_type: Optional[str] = None
    political_alignment: Optional[str] = None
    risk_tolerance: Optional[str] = None
    optimism_cynicism_scale: Optional[int] = None

    # Voice & Speech Patterns (The "Interface")
    linguistic_register: str = "blue_collar"
    typing_quirks: Dict[str, Any] = {}
    signature_phrases: List[str] = []
    trigger_topics: List[str] = []

    # Backstory & Lore (The "Context")
    day_job: Optional[str] = None
    war_story: Optional[str] = None
    vices_hobbies: List[str] = []

    # Anti-Pattern (What they are NOT)
    forbidden_phrases: List[str] = []
    warmth_level: str = "warm"
    patience_level: str = "normal"

    # ==================== TRIGGER-BASED MODEL ORCHESTRATION ====================
    content_triggers: Dict[str, Any] = {}

    model_config = {"from_attributes": True}
