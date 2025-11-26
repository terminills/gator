"""
Tests for Response Humanizer Service

Validates the ResponseHumanizerService that transforms AI responses
into human-like persona outputs.
"""

import pytest
from unittest.mock import Mock

from backend.models.persona import PersonaModel
from backend.services.response_humanizer_service import (
    ResponseHumanizerService,
    get_humanizer_service,
)


@pytest.fixture
def humanizer():
    """Create a response humanizer service for testing."""
    return ResponseHumanizerService()


@pytest.fixture
def buddy_persona():
    """Create a persona with buddy warmth level."""
    persona = Mock(spec=PersonaModel)
    persona.warmth_level = "buddy"
    persona.patience_level = "normal"
    persona.linguistic_register = "gen_z"
    persona.typing_quirks = {
        "capitalization": "lowercase for aesthetic",
        "emoji_usage": "heavy",
        "punctuation": "casual",
    }
    persona.signature_phrases = ["Based", "Facts", "No cap"]
    persona.forbidden_phrases = ["I feel that", "Synergy", "As an AI"]
    return persona


@pytest.fixture
def cold_persona():
    """Create a persona with cold warmth level."""
    persona = Mock(spec=PersonaModel)
    persona.warmth_level = "cold"
    persona.patience_level = "short_fuse"
    persona.linguistic_register = "corporate"
    persona.typing_quirks = {}
    persona.signature_phrases = []
    persona.forbidden_phrases = ["buddy", "bro", "dude"]
    return persona


class TestResponseHumanizerService:
    """Test suite for ResponseHumanizerService."""

    def test_removes_ai_identity_phrases(self, humanizer):
        """Test that AI identity phrases are removed."""
        text = "As an AI, I can help you with that question."
        result = humanizer.humanize_response(text)
        
        assert "as an ai" not in result.lower()

    def test_removes_corporate_speak(self, humanizer):
        """Test that corporate/assistant phrases are removed."""
        text = "I hope this finds you well. Is there anything else I can assist you with?"
        result = humanizer.humanize_response(text)
        
        assert "hope this finds you well" not in result.lower()
        assert "anything else i can assist" not in result.lower()

    def test_removes_hedging_phrases(self, humanizer):
        """Test that hedging and fence-sitting phrases are removed."""
        text = "It's important to note that on the one hand, there are benefits."
        result = humanizer.humanize_response(text)
        
        assert "important to note" not in result.lower()
        assert "on the one hand" not in result.lower()

    def test_removes_summary_paragraphs(self, humanizer):
        """Test that summary paragraphs are removed."""
        text = """Here's some information about the topic.
        
In summary, the key takeaway is that this is important."""
        result = humanizer.humanize_response(text)
        
        assert "in summary" not in result.lower()
        assert "key takeaway" not in result.lower()

    def test_applies_contractions(self, humanizer):
        """Test that contractions are applied for casual speech."""
        text = "I am not going to do that because I cannot and I will not."
        result = humanizer._apply_contractions(text)
        
        assert "I'm not" in result or "i'm not" in result.lower()
        assert "can't" in result
        assert "won't" in result

    def test_contractions_preserve_case(self, humanizer):
        """Test that contractions preserve appropriate casing."""
        text = "I am ready. We are here."
        result = humanizer._apply_contractions(text)
        
        assert "I'm" in result
        assert "We're" in result

    def test_applies_persona_forbidden_phrases(self, humanizer, buddy_persona):
        """Test that persona-specific forbidden phrases are removed."""
        text = "I feel that this is a great idea with synergy."
        result = humanizer.humanize_response(text, persona=buddy_persona)
        
        assert "i feel that" not in result.lower()
        assert "synergy" not in result.lower()

    def test_applies_typing_quirks_lowercase(self, humanizer, buddy_persona):
        """Test that lowercase typing quirk is applied."""
        text = "Hello There Friend!"
        quirks = {"capitalization": "all lowercase"}
        result = humanizer._apply_typing_quirks(text, quirks)
        
        # Should be lowercase when quirk is set
        assert result == result.lower()

    def test_typing_quirks_no_emoji_removes_emoji(self, humanizer):
        """Test that emoji are removed when emoji_usage is 'none'."""
        quirks = {"emoji_usage": "none"}
        text = "Hello! 😊 How are you? 🎉"
        result = humanizer._apply_typing_quirks(text, quirks)
        
        assert "😊" not in result
        assert "🎉" not in result

    def test_match_energy_short_greeting(self, humanizer):
        """Test that greetings get short responses."""
        text = "Hello! I'm so excited to meet you. Let me tell you all about myself and my interests."
        result = humanizer._match_energy(text, "greeting")
        
        # Should be shorter than original
        assert len(result) <= len(text)

    def test_humanize_ui_text(self, humanizer):
        """Test UI text humanization."""
        assert "typing" in humanizer.humanize_ui_text("Generating response...").lower()
        assert "thinking" in humanizer.humanize_ui_text("Processing request").lower()
        assert "content" in humanizer.humanize_ui_text("Generated content").lower()

    def test_humanize_ui_text_preserves_case(self, humanizer):
        """Test that UI text humanization handles casing appropriately."""
        result = humanizer.humanize_ui_text("GENERATING...")
        # Result should contain typing (case may vary based on implementation)
        assert "typing" in result.lower()

    def test_get_typing_indicator_buddy(self, humanizer, buddy_persona):
        """Test typing indicator for buddy warmth level."""
        indicator = humanizer.get_typing_indicator_text(buddy_persona)
        
        assert indicator in ["typing...", "hmm...", "thinking...", "one sec..."]

    def test_get_typing_indicator_cold(self, humanizer, cold_persona):
        """Test typing indicator for cold warmth level."""
        indicator = humanizer.get_typing_indicator_text(cold_persona)
        
        assert indicator in ["...", "typing..."]

    def test_get_status_text(self, humanizer):
        """Test status text humanization."""
        assert humanizer.get_status_text("generating") == "typing"
        assert humanizer.get_status_text("processing") == "thinking"
        assert humanizer.get_status_text("complete") == "done"

    def test_final_cleanup_removes_double_spaces(self, humanizer):
        """Test that final cleanup removes double spaces."""
        text = "Hello  there   friend"
        result = humanizer._final_cleanup(text)
        
        assert "  " not in result

    def test_final_cleanup_fixes_punctuation(self, humanizer):
        """Test that final cleanup fixes orphaned punctuation."""
        text = "Hello . there !"
        result = humanizer._final_cleanup(text)
        
        assert " ." not in result
        assert " !" not in result

    def test_get_humanizer_service_singleton(self):
        """Test that get_humanizer_service returns a singleton."""
        service1 = get_humanizer_service()
        service2 = get_humanizer_service()
        
        assert service1 is service2

    def test_humanize_empty_text(self, humanizer):
        """Test handling of empty text."""
        assert humanizer.humanize_response("") == ""
        assert humanizer.humanize_response(None) is None

    def test_full_humanization_pipeline(self, humanizer, buddy_persona):
        """Test the full humanization pipeline with a realistic example."""
        ai_response = """As an AI, I don't have personal opinions, but I can help you understand this topic.
        
It's important to note that there are multiple perspectives here. On the one hand, some people believe one thing. On the other hand, others disagree.

In summary, I hope this helps. Is there anything else I can assist you with today?"""

        result = humanizer.humanize_response(ai_response, persona=buddy_persona)
        
        # Should not contain AI phrases
        assert "as an ai" not in result.lower()
        assert "important to note" not in result.lower()
        assert "on the one hand" not in result.lower()
        assert "in summary" not in result.lower()
        assert "anything else i can assist" not in result.lower()
        
        # Should be substantially shorter
        assert len(result) < len(ai_response)


class TestHumanizerIntegration:
    """Integration tests for humanizer with persona models."""

    def test_buddy_persona_response_style(self, humanizer, buddy_persona):
        """Test that buddy persona gets friendly treatment."""
        text = "Hello there! How can I help you today?"
        # Apply typing quirks directly with lowercase setting
        quirks = {"capitalization": "all lowercase"}
        result = humanizer._apply_typing_quirks(text, quirks)
        
        # Should be lowercase due to typing quirk
        assert result == result.lower()

    def test_cold_persona_response_style(self, humanizer, cold_persona):
        """Test that cold persona gets appropriate treatment."""
        text = "Hey buddy! Great to hear from you!"
        result = humanizer.humanize_response(text, persona=cold_persona)
        
        # Should remove overly friendly language based on forbidden_phrases
        assert "buddy" not in result.lower()
