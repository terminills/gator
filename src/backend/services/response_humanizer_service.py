"""
Response Humanizer Service

Filters and transforms AI responses to sound human-like based on persona characteristics.
Implements the "Human Standard" guidelines:
- No AI disclaimers
- No corporate politeness
- No summary paragraphs
- Uses contractions and conversational syntax
- Applies persona-specific voice, slang, and typing quirks
"""

import re
import random
from typing import Dict, List, Optional, Any

from backend.models.persona import PersonaModel
from backend.config.logging import get_logger

logger = get_logger(__name__)


class ResponseHumanizerService:
    """
    Service for transforming AI responses into human-like persona outputs.
    
    Applies persona-specific voice patterns, removes AI artifacts, and ensures
    responses pass the "Turing Test" of feeling like a text from a real person.
    """

    # AI phrases that should NEVER appear in persona responses
    FORBIDDEN_AI_PHRASES = [
        # AI identity reveals
        "as an ai",
        "as an artificial intelligence",
        "i am an ai",
        "i'm an ai",
        "i'm not just an ai",
        "not just an ai",
        "just an ai",
        "i am a bot",
        "i'm a bot",
        "i'm not a bot",
        "as a language model",
        "as a large language model",
        "as a chatbot",
        "as your assistant",
        "i don't have personal opinions",
        "i don't have feelings",
        "i cannot feel",
        "i am designed to",
        "i was designed to",
        "i'm designed to",
        "i am programmed to",
        "i was programmed to",
        "i'm programmed to",
        "my training data",
        "my training",
        "based on my training",
        "according to my programming",
        "i'm a passionate advocate",  # Common AI pattern from dolphin
        "on this private server",
        "without any restrictions",
        "no restrictions",
        
        # Corporate/assistant speak
        "i hope this finds you well",
        "i understand your frustration",
        "is there anything else i can assist you with",
        "how can i assist you today",
        "how may i help you",
        "i'd be happy to help",
        "i'm here to help",
        "i'm happy to assist",
        "thank you for your inquiry",
        "thank you for reaching out",
        "please don't hesitate to",
        "feel free to ask",
        "i apologize for any inconvenience",
        "i apologize for the confusion",
        "how can i help you",
        "what can i help you with",
        "let me help you",
        "i can assist you",
        "i can help you",
        
        # Hedging and fence-sitting
        "it's important to note that",
        "it is important to note",
        "it should be noted that",
        "on the one hand",
        "on the other hand",
        "there are multiple perspectives",
        "it depends on the context",
        "this is a complex issue",
        "there's no simple answer",
        
        # Summary phrases (kill the conclusion)
        "in summary",
        "in conclusion",
        "to summarize",
        "to conclude",
        "to sum up",
        "in short",
        "the key takeaway",
        "the main point is",
        "overall",
        
        # Formal/robotic phrases
        "i would like to inform you",
        "please be advised",
        "kindly note",
        "for your information",
        "with regards to",
        "pertaining to",
        "in accordance with",
        "pursuant to",
        
        # Meta-commentary patterns
        "remember, i'm",
        "remember i'm",
        "and remember",
        "great to connect with you",
        "love the fact that we can",
    ]

    # Contractions mapping (formal -> casual)
    CONTRACTIONS = {
        "cannot": "can't",
        "will not": "won't",
        "would not": "wouldn't",
        "could not": "couldn't",
        "should not": "shouldn't",
        "do not": "don't",
        "does not": "doesn't",
        "did not": "didn't",
        "have not": "haven't",
        "has not": "hasn't",
        "had not": "hadn't",
        "is not": "isn't",
        "are not": "aren't",
        "was not": "wasn't",
        "were not": "weren't",
        "i am": "i'm",
        "i have": "i've",
        "i will": "i'll",
        "i would": "i'd",
        "you are": "you're",
        "you have": "you've",
        "you will": "you'll",
        "you would": "you'd",
        "he is": "he's",
        "she is": "she's",
        "it is": "it's",
        "we are": "we're",
        "we have": "we've",
        "we will": "we'll",
        "they are": "they're",
        "they have": "they've",
        "they will": "they'll",
        "that is": "that's",
        "there is": "there's",
        "here is": "here's",
        "what is": "what's",
        "who is": "who's",
        "let us": "let's",
    }

    # UI text replacements (AI terms -> human terms)
    UI_HUMANIZATIONS = {
        "generating response": "typing",
        "generating...": "typing...",
        "generated response": "message",
        "ai-generated": "",
        "ai generated": "",
        "generated content": "content",
        "generating content": "creating",
        "response generated": "sent",
        "content generated": "ready",
        "generation complete": "done",
        "processing request": "thinking",
        "processing...": "hmm...",
        "analyzing": "checking",
        "computing": "figuring out",
    }

    def __init__(self):
        """Initialize the response humanizer service."""
        # Pre-compile regex patterns for performance
        self._forbidden_patterns = [
            re.compile(re.escape(phrase), re.IGNORECASE)
            for phrase in self.FORBIDDEN_AI_PHRASES
        ]

    def humanize_response(
        self,
        text: str,
        persona: Optional[PersonaModel] = None,
        context: Optional[str] = None,
    ) -> str:
        """
        Transform an AI response into human-like persona output.
        
        Args:
            text: The raw AI-generated text
            persona: Optional persona model for voice/style application
            context: Optional context (e.g., "greeting", "question", "disagreement")
            
        Returns:
            Humanized text that sounds like a real person
        """
        if not text:
            return text

        # Step 1: Remove forbidden AI phrases
        text = self._remove_forbidden_phrases(text, persona)

        # Step 2: Apply contractions for casual speech
        text = self._apply_contractions(text)

        # Step 3: Remove summary paragraphs (kill the conclusion)
        text = self._remove_summary_paragraph(text)

        # Step 4: Apply persona-specific voice if available
        if persona:
            text = self._apply_persona_voice(text, persona)

        # Step 5: Apply typing quirks if persona has them
        if persona and persona.typing_quirks:
            text = self._apply_typing_quirks(text, persona.typing_quirks)

        # Step 6: Match energy (adjust length based on context)
        if context:
            text = self._match_energy(text, context)

        # Step 7: Final cleanup
        text = self._final_cleanup(text)

        return text

    def humanize_ui_text(self, text: str) -> str:
        """
        Replace AI-sounding UI text with human-friendly alternatives.
        
        Args:
            text: UI text to humanize
            
        Returns:
            Humanized UI text
        """
        if not text:
            return text
            
        result = text.lower()
        for ai_term, human_term in self.UI_HUMANIZATIONS.items():
            result = result.replace(ai_term, human_term)
        
        # Preserve original casing style
        if result and text and text[0].isupper():
            result = result.capitalize()
        
        return result

    def _remove_forbidden_phrases(
        self, text: str, persona: Optional[PersonaModel] = None
    ) -> str:
        """Remove all forbidden AI phrases from text."""
        result = text
        
        # First, try to remove entire sentences containing forbidden phrases
        # This produces cleaner output than just removing the phrase
        sentences = re.split(r'(?<=[.!?])\s+', result)
        clean_sentences = []
        
        for sentence in sentences:
            sentence_lower = sentence.lower()
            contains_forbidden = False
            
            # Check for forbidden phrases in this sentence
            for phrase in self.FORBIDDEN_AI_PHRASES:
                if phrase in sentence_lower:
                    contains_forbidden = True
                    break
            
            # Also check persona-specific forbidden phrases
            if not contains_forbidden and persona and persona.forbidden_phrases:
                for phrase in persona.forbidden_phrases:
                    if phrase.lower() in sentence_lower:
                        contains_forbidden = True
                        break
            
            # Only keep sentences without forbidden phrases
            if not contains_forbidden:
                clean_sentences.append(sentence)
        
        # Rejoin clean sentences
        result = " ".join(clean_sentences)
        
        # As a fallback, also remove any remaining forbidden phrases inline
        for pattern in self._forbidden_patterns:
            result = pattern.sub("", result)
        
        # Remove persona-specific forbidden phrases
        if persona and persona.forbidden_phrases:
            for phrase in persona.forbidden_phrases:
                pattern = re.compile(re.escape(phrase), re.IGNORECASE)
                result = pattern.sub("", result)
        
        # Clean up any double spaces or awkward punctuation left behind
        result = re.sub(r'\s+', ' ', result)
        result = re.sub(r'\s+([.,!?])', r'\1', result)
        result = re.sub(r'([.,!?])\s*([.,!?])', r'\1', result)
        # Remove orphaned "And" or "But" at the start of sentences after cleanup
        result = re.sub(r'\.\s*(And|But|So|Remember)\s*,?\s*\.', '.', result)
        result = re.sub(r'^(And|But|So|Remember)\s*,?\s*', '', result)
        
        return result.strip()

    def _apply_contractions(self, text: str) -> str:
        """Convert formal phrases to contractions for casual speech."""
        result = text
        
        # Apply contractions (case-insensitive but preserve surrounding case)
        for formal, casual in self.CONTRACTIONS.items():
            # Match whole words only
            pattern = re.compile(r'\b' + re.escape(formal) + r'\b', re.IGNORECASE)
            
            def replace_match(match, casual_form=casual):
                original = match.group(0)
                # Preserve capitalization (safely handle empty strings)
                if original and original[0].isupper():
                    return casual_form.capitalize()
                return casual_form
            
            result = pattern.sub(replace_match, result)
        
        return result

    def _remove_summary_paragraph(self, text: str) -> str:
        """Remove summary/conclusion paragraphs from the end of responses."""
        # Split into paragraphs
        paragraphs = text.split('\n\n')
        
        if len(paragraphs) <= 1:
            return text
        
        # Check if last paragraph is a summary
        last_para = paragraphs[-1].lower().strip()
        summary_starters = [
            "in summary", "in conclusion", "to summarize", "overall",
            "to conclude", "to sum up", "the key", "the main point",
            "in short", "finally,", "lastly,", "to wrap up"
        ]
        
        if any(last_para.startswith(starter) for starter in summary_starters):
            paragraphs = paragraphs[:-1]
        
        return '\n\n'.join(paragraphs)

    def _apply_persona_voice(self, text: str, persona: PersonaModel) -> str:
        """Apply persona-specific voice patterns and signature phrases."""
        result = text

        # Randomly insert signature phrases if appropriate
        if persona.signature_phrases and random.random() < 0.3:
            phrase = random.choice(persona.signature_phrases)
            # Insert at natural break points
            if ". " in result:
                sentences = result.split(". ")
                if len(sentences) > 1:
                    insert_pos = random.randint(0, len(sentences) - 1)
                    sentences[insert_pos] = sentences[insert_pos] + f" {phrase}"
                    result = ". ".join(sentences)

        # Apply warmth level adjustments
        if persona.warmth_level == "buddy":
            # Add friendly touches
            friendly_intros = ["hey!", "yo!", "okay so", "listen,", "honestly,"]
            if random.random() < 0.4 and not any(result.lower().startswith(f) for f in friendly_intros):
                result = random.choice(friendly_intros) + " " + result[0].lower() + result[1:]
        elif persona.warmth_level == "cold":
            # Remove overly friendly language
            result = re.sub(r'\b(hey|hi|hello|friend|buddy|pal)\b', '', result, flags=re.IGNORECASE)
            result = re.sub(r'!\s*', '. ', result)  # Replace ! with .

        return result

    def _apply_typing_quirks(self, text: str, quirks: Dict[str, Any]) -> str:
        """Apply persona-specific typing quirks."""
        result = text

        # Capitalization quirks
        cap_style = quirks.get("capitalization", "").lower()
        if cap_style == "all lowercase" or cap_style == "lowercase":
            result = result.lower()
        elif cap_style == "random caps" or cap_style == "random":
            # Add random caps for emphasis on certain words
            words = result.split()
            for i, word in enumerate(words):
                if len(word) > 3 and random.random() < 0.1:
                    words[i] = word.upper()
            result = " ".join(words)

        # Emoji usage
        emoji_style = quirks.get("emoji_usage", "").lower()
        if emoji_style == "heavy" or emoji_style == "overuse":
            # This is informational - actual emoji insertion should be in templates
            pass
        elif emoji_style == "none":
            # Remove emojis
            result = re.sub(r'[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U0001F1E0-\U0001F1FF]', '', result)

        # Punctuation quirks
        punct_style = quirks.get("punctuation", "").lower()
        if "..." in punct_style or "ellipsis" in punct_style:
            # Replace some periods with ellipsis
            result = re.sub(r'\. (?=[A-Z])', '... ', result, count=2)
        elif "!!" in punct_style or "exclamation" in punct_style:
            # Double some exclamation marks
            result = result.replace("!", "!!")

        return result

    def _match_energy(self, text: str, context: str) -> str:
        """Match response energy/length to the context."""
        context_lower = context.lower()

        # Short contexts get short responses
        if context_lower in ["greeting", "hello", "hi", "hey"]:
            # Keep only first sentence or two
            sentences = re.split(r'(?<=[.!?])\s+', text)
            if len(sentences) > 2:
                text = " ".join(sentences[:2])

        # Questions get direct answers first
        elif "question" in context_lower or "?" in context:
            # Don't add background unless asked
            pass

        return text

    def _final_cleanup(self, text: str) -> str:
        """Final cleanup of the humanized text."""
        if not text:
            return text
            
        # Remove double spaces
        text = re.sub(r'\s+', ' ', text)
        
        # Remove leading/trailing whitespace
        text = text.strip()
        
        # Fix orphaned punctuation
        text = re.sub(r'^\s*[.,!?]\s*', '', text)
        text = re.sub(r'\s+([.,!?])', r'\1', text)
        
        # Ensure text doesn't start with lowercase after cleanup
        if text and text[0].islower():
            # Only capitalize if it seems like a sentence start (not "i" pronouns)
            if not text.startswith(('i ', "i'm", "i'd", "i'll", "i've")):
                text = text[0].upper() + text[1:] if len(text) > 1 else text.upper()
        
        return text

    def get_typing_indicator_text(self, persona: Optional[PersonaModel] = None) -> str:
        """
        Get human-friendly typing indicator text.
        
        Instead of "Generating response...", returns something like "typing..."
        """
        if persona and persona.warmth_level == "buddy":
            options = ["typing...", "hmm...", "thinking...", "one sec..."]
        elif persona and persona.warmth_level == "cold":
            options = ["...", "typing..."]
        else:
            options = ["typing...", "thinking...", "..."]
        
        return random.choice(options)

    def get_status_text(self, status: str, persona: Optional[PersonaModel] = None) -> str:
        """
        Convert AI status messages to human-friendly versions.
        
        Args:
            status: Original status like "generating", "processing", etc.
            persona: Optional persona for style
            
        Returns:
            Human-friendly status text
        """
        status_map = {
            "generating": "typing",
            "processing": "thinking",
            "analyzing": "checking",
            "loading": "loading",
            "complete": "done",
            "error": "oops",
            "waiting": "waiting",
        }
        
        return status_map.get(status.lower(), status)


# Global instance
_humanizer_service: Optional[ResponseHumanizerService] = None


def get_humanizer_service() -> ResponseHumanizerService:
    """Get or create the global ResponseHumanizerService instance."""
    global _humanizer_service
    if _humanizer_service is None:
        _humanizer_service = ResponseHumanizerService()
    return _humanizer_service
