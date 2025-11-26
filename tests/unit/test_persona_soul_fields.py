"""
Tests for Persona Soul Fields

Validates the persona soul fields for human-like response generation:
- Origin & Demographics
- Psychological Profile
- Voice & Speech Patterns
- Backstory & Lore
- Anti-Pattern
"""

import pytest
from pydantic import ValidationError

from backend.models.persona import (
    PersonaCreate,
    PersonaUpdate,
    PersonaResponse,
    LinguisticRegister,
    WarmthLevel,
    PatienceLevel,
)


class TestPersonaSoulFieldsValidation:
    """Test persona soul field validation."""

    def test_linguistic_register_enum_values(self):
        """Test that linguistic register has expected enum values."""
        assert LinguisticRegister.BLUE_COLLAR.value == "blue_collar"
        assert LinguisticRegister.ACADEMIC.value == "academic"
        assert LinguisticRegister.TECH_BRO.value == "tech_bro"
        assert LinguisticRegister.STREET.value == "street"
        assert LinguisticRegister.CORPORATE.value == "corporate"
        assert LinguisticRegister.SOUTHERN.value == "southern"
        assert LinguisticRegister.MILLENNIAL.value == "millennial"
        assert LinguisticRegister.GEN_Z.value == "gen_z"

    def test_warmth_level_enum_values(self):
        """Test that warmth level has expected enum values."""
        assert WarmthLevel.COLD.value == "cold"
        assert WarmthLevel.NEUTRAL.value == "neutral"
        assert WarmthLevel.WARM.value == "warm"
        assert WarmthLevel.BUDDY.value == "buddy"

    def test_patience_level_enum_values(self):
        """Test that patience level has expected enum values."""
        assert PatienceLevel.SHORT_FUSE.value == "short_fuse"
        assert PatienceLevel.NORMAL.value == "normal"
        assert PatienceLevel.PATIENT.value == "patient"
        assert PatienceLevel.INFINITE.value == "infinite"

    def test_optimism_cynicism_scale_validation(self):
        """Test that optimism/cynicism scale is validated (1-10)."""
        # Valid values
        persona = PersonaCreate(
            name="Test",
            appearance="Test appearance description",
            personality="Test personality description",
            optimism_cynicism_scale=5,
        )
        assert persona.optimism_cynicism_scale == 5

        # Test boundary values
        persona_low = PersonaCreate(
            name="Test",
            appearance="Test appearance description",
            personality="Test personality description",
            optimism_cynicism_scale=1,
        )
        assert persona_low.optimism_cynicism_scale == 1

        persona_high = PersonaCreate(
            name="Test",
            appearance="Test appearance description",
            personality="Test personality description",
            optimism_cynicism_scale=10,
        )
        assert persona_high.optimism_cynicism_scale == 10

    def test_optimism_cynicism_scale_out_of_range(self):
        """Test that invalid optimism/cynicism scale values are rejected."""
        with pytest.raises(ValidationError):
            PersonaCreate(
                name="Test",
                appearance="Test appearance description",
                personality="Test personality description",
                optimism_cynicism_scale=0,  # Below minimum
            )

        with pytest.raises(ValidationError):
            PersonaCreate(
                name="Test",
                appearance="Test appearance description",
                personality="Test personality description",
                optimism_cynicism_scale=11,  # Above maximum
            )


class TestPersonaCreateWithSoulFields:
    """Test PersonaCreate with soul fields."""

    def test_create_minimal_persona(self):
        """Test creating a persona with minimal required fields."""
        persona = PersonaCreate(
            name="Test Persona",
            appearance="Test appearance description",
            personality="Test personality description",
        )
        
        # Soul fields should have defaults
        assert persona.linguistic_register == LinguisticRegister.BLUE_COLLAR
        assert persona.warmth_level == WarmthLevel.WARM
        assert persona.patience_level == PatienceLevel.NORMAL
        assert persona.typing_quirks == {}
        assert persona.signature_phrases == []
        assert persona.forbidden_phrases == []

    def test_create_full_persona_like_sydney(self):
        """Test creating a full persona like Sydney with all soul fields."""
        persona = PersonaCreate(
            name="Sydney",
            appearance="24-year-old woman, blonde hair, confident smile",
            personality="Flirty, mischievous, confident, high-energy",
            content_themes=["politics", "liberty", "lifestyle"],
            # Origin & Demographics
            hometown="Nashville, TN",
            current_location="Miami, FL",
            generation_age="Gen Z - 24 years old",
            education_level="Liberal arts college",
            # Psychological Profile
            mbti_type="ESTP - The Entrepreneur",
            enneagram_type="Type 7 - The Enthusiast",
            political_alignment="Right-leaning / Liberty-focused",
            risk_tolerance="Move fast and break things",
            optimism_cynicism_scale=7,
            # Voice & Speech Patterns
            linguistic_register=LinguisticRegister.GEN_Z,
            typing_quirks={
                "capitalization": "lowercase for aesthetic",
                "emoji_usage": "heavy",
            },
            signature_phrases=["Based", "Facts", "Y'all"],
            trigger_topics=["Cancel culture", "Gas prices"],
            # Backstory & Lore
            day_job="Political Commentator / Streamer",
            war_story="Started posting rants in her car and went viral",
            vices_hobbies=["Range days", "Country concerts"],
            # Anti-Pattern
            forbidden_phrases=["I feel that", "Synergy", "As an AI"],
            warmth_level=WarmthLevel.BUDDY,
            patience_level=PatienceLevel.SHORT_FUSE,
        )
        
        assert persona.name == "Sydney"
        assert persona.hometown == "Nashville, TN"
        assert persona.mbti_type == "ESTP - The Entrepreneur"
        assert persona.linguistic_register == LinguisticRegister.GEN_Z
        assert persona.warmth_level == WarmthLevel.BUDDY
        assert "Based" in persona.signature_phrases
        assert "As an AI" in persona.forbidden_phrases


class TestPersonaUpdateWithSoulFields:
    """Test PersonaUpdate with soul fields."""

    def test_update_single_soul_field(self):
        """Test updating a single soul field."""
        update = PersonaUpdate(warmth_level=WarmthLevel.COLD)
        
        assert update.warmth_level == WarmthLevel.COLD
        assert update.hometown is None  # Not being updated

    def test_update_multiple_soul_fields(self):
        """Test updating multiple soul fields at once."""
        update = PersonaUpdate(
            hometown="New York, NY",
            day_job="Software Engineer",
            linguistic_register=LinguisticRegister.TECH_BRO,
            signature_phrases=["Ship it", "Move fast"],
        )
        
        assert update.hometown == "New York, NY"
        assert update.day_job == "Software Engineer"
        assert update.linguistic_register == LinguisticRegister.TECH_BRO
        assert "Ship it" in update.signature_phrases


class TestPersonaResponseWithSoulFields:
    """Test PersonaResponse with soul fields."""

    def test_response_has_soul_field_defaults(self):
        """Test that PersonaResponse has proper defaults for soul fields."""
        from datetime import datetime
        from uuid import uuid4
        
        response = PersonaResponse(
            id=uuid4(),
            name="Test",
            appearance="Test appearance",
            personality="Test personality",
            content_themes=[],
            style_preferences={},
            default_content_rating="sfw",
            allowed_content_ratings=["sfw"],
            platform_restrictions={},
            is_active=True,
            generation_count=0,
            created_at=datetime.now(),
            updated_at=datetime.now(),
        )
        
        # Check soul field defaults
        assert response.hometown is None
        assert response.current_location is None
        assert response.linguistic_register == "blue_collar"
        assert response.typing_quirks == {}
        assert response.signature_phrases == []
        assert response.warmth_level == "warm"
        assert response.patience_level == "normal"


class TestTypingQuirksStructure:
    """Test typing quirks JSON structure."""

    def test_typing_quirks_with_all_fields(self):
        """Test creating typing quirks with all expected fields."""
        persona = PersonaCreate(
            name="Test",
            appearance="Test appearance",
            personality="Test personality",
            typing_quirks={
                "capitalization": "all lowercase",
                "emoji_usage": "heavy - uses 🇺🇸💅😂",
                "punctuation": "uses ... a lot",
                "custom_quirk": "says 'lol' instead of laughing",
            },
        )
        
        assert persona.typing_quirks["capitalization"] == "all lowercase"
        assert "heavy" in persona.typing_quirks["emoji_usage"]
        assert "..." in persona.typing_quirks["punctuation"]

    def test_typing_quirks_empty(self):
        """Test that empty typing quirks works."""
        persona = PersonaCreate(
            name="Test",
            appearance="Test appearance",
            personality="Test personality",
            typing_quirks={},
        )
        
        assert persona.typing_quirks == {}


class TestForbiddenPhrasesValidation:
    """Test forbidden phrases list validation."""

    def test_forbidden_phrases_list(self):
        """Test creating forbidden phrases list."""
        persona = PersonaCreate(
            name="Test",
            appearance="Test appearance",
            personality="Test personality",
            forbidden_phrases=[
                "As an AI",
                "I don't have opinions",
                "Is there anything else",
                "I hope this helps",
                "In summary",
            ],
        )
        
        assert len(persona.forbidden_phrases) == 5
        assert "As an AI" in persona.forbidden_phrases

    def test_empty_forbidden_phrases(self):
        """Test that empty forbidden phrases works."""
        persona = PersonaCreate(
            name="Test",
            appearance="Test appearance",
            personality="Test personality",
            forbidden_phrases=[],
        )
        
        assert persona.forbidden_phrases == []
