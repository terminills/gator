"""
AI Persona Generator Service

Generates realistic, coherent personas using Ollama for full AI-powered creation.
Uses dolphin-mixtral as the default model for uncensored, creative persona generation.

Produces more realistic and internally consistent personas compared to random templates.

Now includes full "soul" fields for human-like response patterns:
- Origin & Demographics (hometown, location, generation, education)
- Psychological Profile (MBTI, enneagram, political alignment, risk tolerance)
- Voice & Speech Patterns (linguistic register, typing quirks, signature phrases)
- Backstory & Lore (day job, war story, vices & hobbies)
- Anti-Pattern (forbidden phrases, warmth, patience)
"""

import asyncio
import json
from typing import Any, Dict, List, Optional

import httpx

from backend.config.logging import get_logger
from backend.models.persona import ContentRating

logger = get_logger(__name__)

# Default Ollama endpoint
OLLAMA_BASE_URL = "http://localhost:11434"

# Ollama models to try (in order of preference) - dolphin-mixtral is our default
OLLAMA_MODELS = [
    "dolphin-mixtral",
    "dolphin-mixtral:8x7b",
    "dolphin-mixtral:latest",
    "dolphin-llama3",
    "dolphin-mistral",
    "llama3.1:8b",
    "llama3:8b",
    "llama3.1",
    "llama3",
    "mistral",
    "mixtral",
    "qwen2.5:7b",
    "gemma2:9b",
]


class AIPersonaGenerator:
    """
    AI-powered persona generator using Ollama (with dolphin-mixtral as default).

    Creates coherent, realistic personas with internally consistent:
    - Appearance descriptions
    - Personality traits (soul fields)
    - Content themes and interests
    - Voice and speech patterns
    - Backstory and lore
    - Anti-patterns (what they would NEVER say)

    Falls back to template-based generation if Ollama is unavailable.
    """

    def __init__(self, ollama_url: str = OLLAMA_BASE_URL):
        self.ollama_url = ollama_url
        self._available_model: Optional[str] = None
        self._model_checked = False

    async def _check_ollama_available(self) -> bool:
        """Check if Ollama is running and get available model."""
        if self._model_checked:
            return self._available_model is not None

        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(f"{self.ollama_url}/api/tags")
                if response.status_code == 200:
                    data = response.json()
                    available_models = [m["name"] for m in data.get("models", [])]

                    # Find first available model from our preference list
                    for preferred in OLLAMA_MODELS:
                        for available in available_models:
                            if preferred in available.lower():
                                self._available_model = available
                                logger.info(
                                    f"Ollama available with model: {self._available_model}"
                                )
                                self._model_checked = True
                                return True

                    # If none of our preferred models, use any available
                    if available_models:
                        self._available_model = available_models[0]
                        logger.info(f"Using Ollama model: {self._available_model}")
                        self._model_checked = True
                        return True

        except Exception as e:
            logger.debug(f"Ollama not available: {e}")

        self._model_checked = True
        self._available_model = None
        return False

    async def _generate_with_ollama(self, prompt: str) -> Optional[str]:
        """Generate text using Ollama API."""
        if not await self._check_ollama_available():
            return None

        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                response = await client.post(
                    f"{self.ollama_url}/api/generate",
                    json={
                        "model": self._available_model,
                        "prompt": prompt,
                        "stream": False,
                        "options": {
                            "temperature": 0.9,
                            "top_p": 0.95,
                            "num_predict": 2000,
                        },
                    },
                )

                if response.status_code == 200:
                    data = response.json()
                    return data.get("response", "")
                else:
                    logger.error(f"Ollama API error: {response.status_code}")
                    return None

        except Exception as e:
            logger.error(f"Ollama generation failed: {e}")
            return None

    async def generate_persona(
        self,
        name: Optional[str] = None,
        persona_type: Optional[str] = None,
        use_ai: bool = True,
    ) -> Dict[str, Any]:
        """
        Generate a complete persona configuration with soul fields.

        Args:
            name: Optional custom name (generated if not provided)
            persona_type: Optional persona type hint (e.g., "fitness", "tech", "political")
            use_ai: Whether to use AI (Ollama) or fall back to templates

        Returns:
            Dict with complete persona configuration including soul fields
        """
        try:
            # Check if we should use AI generation
            if use_ai and await self._check_ollama_available():
                logger.info(
                    f"Generating AI-powered persona with Ollama (type: {persona_type or 'general'})"
                )
                return await self._generate_with_ai(
                    name=name, persona_type=persona_type
                )
            else:
                logger.info(
                    "Using template-based persona generation (Ollama unavailable)"
                )
                return self._generate_with_template(name, persona_type)

        except Exception as e:
            logger.error(f"AI persona generation failed: {e}")
            # Fall back to template
            return self._generate_with_template(name, persona_type)

    async def generate_batch(
        self,
        count: int = 5,
        persona_types: Optional[List[str]] = None,
        use_ai: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Generate multiple personas.

        Args:
            count: Number of personas to generate
            persona_types: Optional list of persona types for variety
            use_ai: Whether to use AI generation

        Returns:
            List of persona configurations
        """
        personas = []

        for i in range(count):
            # Pick a persona type if provided
            persona_type = None
            if persona_types:
                persona_type = persona_types[i % len(persona_types)]

            persona = await self.generate_persona(
                persona_type=persona_type, use_ai=use_ai
            )
            personas.append(persona)

        return personas

    async def _generate_with_ai(
        self, name: Optional[str], persona_type: Optional[str]
    ) -> Dict[str, Any]:
        """
        Generate persona using Ollama AI model.

        Creates a coherent, realistic persona with internally consistent attributes
        including all soul fields for human-like response patterns.
        """
        # Generate a name if not provided
        if not name:
            name = self._generate_random_name()

        # Build the instruction for the language model
        prompt = self._build_persona_prompt(name, persona_type)

        # Generate with Ollama
        output = await self._generate_with_ollama(prompt)

        if output:
            # Parse the output
            persona_data = self._parse_persona_output(output, name, persona_type)
            if persona_data:
                logger.info(f"Generated AI persona with soul: {persona_data['name']}")
                return persona_data

        # Fall back to template if AI generation failed
        logger.warning("AI generation failed, using template fallback")
        return self._generate_with_template(name, persona_type)

    def _build_persona_prompt(self, name: str, persona_type: Optional[str]) -> str:
        """
        Build the instruction prompt for Ollama persona generation.

        Includes all soul fields for comprehensive persona creation.
        """
        type_guidance = ""
        if persona_type:
            type_guidance = f"\nThis persona should be focused on: {persona_type}. Make their personality, interests, and content align with this focus.\n"

        prompt = f"""You are an expert at creating realistic AI influencer personas that feel like real people, not bots.

Create a detailed, coherent persona profile for "{name}".{type_guidance}

The persona must feel like a REAL PERSON who would text like a human, not an AI assistant. They should have:
- Strong opinions and a distinct personality
- Quirks and flaws that make them relatable  
- A unique voice and way of speaking
- Things they love and things that annoy them

# CRITICAL RULES FOR THE PERSONA
1. They should NEVER sound like an AI or customer service bot
2. They should have strong opinions and not sit on the fence
3. They should use casual language, slang, and contractions
4. They should have specific quirks and personality traits
5. They should feel like someone you'd actually DM with

# OUTPUT FORMAT
Respond with ONLY a valid JSON object (no markdown, no explanation, just JSON):

{{
  "name": "{name}",
  "age": 24,
  "appearance": "detailed physical description - age, ethnicity, build, hair, eyes, style",
  "personality": "2-3 sentences describing their character traits and vibe",
  
  "hometown": "specific city/region they're from",
  "current_location": "where they live now and why",
  "generation_age": "generation label with context (e.g., 'Gen Z - 24, grew up on the internet')",
  "education_level": "their education background",
  
  "mbti_type": "MBTI type with description (e.g., 'ESTP - The Entrepreneur')",
  "enneagram_type": "Enneagram type (e.g., 'Type 7 - The Enthusiast')",
  "political_alignment": "their political/social worldview",
  "risk_tolerance": "their attitude toward risk (e.g., 'Move fast and break things')",
  "optimism_cynicism_scale": 7,
  
  "linguistic_register": "blue_collar|academic|tech_bro|street|corporate|southern|millennial|gen_z",
  "typing_quirks": {{
    "capitalization": "how they capitalize (all lowercase, normal, RANDOM CAPS)",
    "emoji_usage": "none|minimal|moderate|heavy",
    "punctuation": "their punctuation style"
  }},
  "signature_phrases": ["phrase1", "phrase2", "phrase3", "phrase4", "phrase5"],
  "trigger_topics": ["topic that excites/angers them 1", "topic 2", "topic 3"],
  
  "day_job": "what they do for money",
  "war_story": "one defining life event that shaped them",
  "vices_hobbies": ["hobby1", "hobby2", "hobby3"],
  
  "forbidden_phrases": ["phrase they would NEVER say 1", "phrase 2", "phrase 3", "phrase 4", "phrase 5"],
  "warmth_level": "cold|neutral|warm|buddy",
  "patience_level": "short_fuse|normal|patient|infinite",
  
  "content_themes": ["theme1", "theme2", "theme3", "theme4"],
  "image_style": "photorealistic",
  "post_style": "casual|professional|artistic"
}}

Remember: Make them feel REAL and HUMAN, not like a bot. Give them personality, opinions, and quirks.

JSON:"""

        return prompt

    def _parse_persona_output(
        self, output: str, name: str, persona_type: Optional[str]
    ) -> Optional[Dict[str, Any]]:
        """
        Parse Ollama output and extract persona data with soul fields.
        """
        try:
            # Find JSON in the output (look for { ... })
            start_idx = output.find("{")
            end_idx = output.rfind("}")

            if start_idx == -1 or end_idx == -1:
                logger.warning("No JSON found in Ollama output")
                return None

            json_str = output[start_idx : end_idx + 1]
            persona_data = json.loads(json_str)

            # Validate required fields
            required_fields = ["appearance", "personality"]
            if not all(field in persona_data for field in required_fields):
                logger.warning("Missing required fields in AI output")
                return None

            # Build complete persona configuration with soul fields
            return self._build_persona_config(persona_data, name)

        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse JSON from Ollama output: {e}")
            return None
        except Exception as e:
            logger.warning(f"Error parsing persona data: {e}")
            return None

    def _build_persona_config(
        self, ai_data: Dict[str, Any], name: str
    ) -> Dict[str, Any]:
        """
        Build a complete persona configuration from AI-generated data.

        Includes all soul fields for human-like responses.
        """
        # Generate content rating
        default_rating, allowed_ratings = self._generate_random_content_rating()

        # Map linguistic register string to enum value
        register_str = ai_data.get("linguistic_register", "blue_collar").lower()
        register_map = {
            "blue_collar": "blue_collar",
            "academic": "academic",
            "tech_bro": "tech_bro",
            "street": "street",
            "corporate": "corporate",
            "southern": "southern",
            "millennial": "millennial",
            "gen_z": "gen_z",
        }
        linguistic_register = register_map.get(register_str, "blue_collar")

        # Map warmth level
        warmth_str = ai_data.get("warmth_level", "warm").lower()
        warmth_map = {
            "cold": "cold",
            "neutral": "neutral",
            "warm": "warm",
            "buddy": "buddy",
        }
        warmth_level = warmth_map.get(warmth_str, "warm")

        # Map patience level
        patience_str = ai_data.get("patience_level", "normal").lower()
        patience_map = {
            "short_fuse": "short_fuse",
            "normal": "normal",
            "patient": "patient",
            "infinite": "infinite",
        }
        patience_level = patience_map.get(patience_str, "normal")

        # Ensure optimism scale is within bounds
        optimism_scale = ai_data.get("optimism_cynicism_scale", 5)
        if isinstance(optimism_scale, (int, float)):
            optimism_scale = max(1, min(10, int(optimism_scale)))
        else:
            optimism_scale = 5

        # Ensure lists have proper formats
        content_themes = ai_data.get("content_themes", ["lifestyle", "entertainment"])
        if isinstance(content_themes, list):
            content_themes = content_themes[:10]  # Max 10 themes
        else:
            content_themes = ["lifestyle", "entertainment"]

        signature_phrases = ai_data.get("signature_phrases", [])
        if not isinstance(signature_phrases, list):
            signature_phrases = []

        forbidden_phrases = ai_data.get("forbidden_phrases", [])
        if not isinstance(forbidden_phrases, list):
            forbidden_phrases = []

        trigger_topics = ai_data.get("trigger_topics", [])
        if not isinstance(trigger_topics, list):
            trigger_topics = []

        vices_hobbies = ai_data.get("vices_hobbies", [])
        if not isinstance(vices_hobbies, list):
            vices_hobbies = []

        # Handle typing quirks
        typing_quirks = ai_data.get("typing_quirks", {})
        if not isinstance(typing_quirks, dict):
            typing_quirks = {}

        return {
            "name": ai_data.get("name", name),
            "appearance": ai_data.get("appearance", ""),
            "personality": ai_data.get("personality", ""),
            "content_themes": content_themes,
            "style_preferences": {
                "image_style": ai_data.get("image_style", "photorealistic"),
                "post_style": ai_data.get("post_style", "casual"),
            },
            "default_content_rating": default_rating,
            "allowed_content_ratings": allowed_ratings,
            "platform_restrictions": self._generate_platform_restrictions(
                default_rating
            ),
            "image_style": ai_data.get("image_style", "photorealistic"),
            "post_style": ai_data.get("post_style", "casual"),
            "is_active": True,
            # Soul Fields - Origin & Demographics
            "hometown": ai_data.get("hometown"),
            "current_location": ai_data.get("current_location"),
            "generation_age": ai_data.get("generation_age"),
            "education_level": ai_data.get("education_level"),
            # Soul Fields - Psychological Profile
            "mbti_type": ai_data.get("mbti_type"),
            "enneagram_type": ai_data.get("enneagram_type"),
            "political_alignment": ai_data.get("political_alignment"),
            "risk_tolerance": ai_data.get("risk_tolerance"),
            "optimism_cynicism_scale": optimism_scale,
            # Soul Fields - Voice & Speech Patterns
            "linguistic_register": linguistic_register,
            "typing_quirks": typing_quirks,
            "signature_phrases": signature_phrases,
            "trigger_topics": trigger_topics,
            # Soul Fields - Backstory & Lore
            "day_job": ai_data.get("day_job"),
            "war_story": ai_data.get("war_story"),
            "vices_hobbies": vices_hobbies,
            # Soul Fields - Anti-Pattern
            "forbidden_phrases": forbidden_phrases,
            "warmth_level": warmth_level,
            "patience_level": patience_level,
            # Metadata
            "generation_method": "ollama_ai",
        }

    def _generate_with_template(
        self, name: Optional[str], persona_type: Optional[str]
    ) -> Dict[str, Any]:
        """
        Generate persona using template-based approach (fallback).

        Uses the PersonaRandomizer with soul fields.
        """
        from backend.services.persona_randomizer import PersonaRandomizer

        persona = PersonaRandomizer.generate_complete_random_persona(
            name=name, detailed=True
        )

        # Add source indicator
        persona["generation_method"] = "template"

        return persona

    def _generate_random_name(self) -> str:
        """Generate a random name."""
        from backend.services.persona_randomizer import PersonaRandomizer

        return PersonaRandomizer.generate_random_name()

    def _generate_random_content_rating(self) -> tuple:
        """Generate random content rating."""
        from backend.services.persona_randomizer import PersonaRandomizer

        return PersonaRandomizer.generate_random_content_rating()

    def _generate_platform_restrictions(
        self, default_rating: ContentRating
    ) -> Dict[str, str]:
        """Generate platform restrictions based on content rating."""
        from backend.services.persona_randomizer import PersonaRandomizer

        return PersonaRandomizer.generate_random_platform_restrictions()


# Global instance
_ai_persona_generator: Optional[AIPersonaGenerator] = None


def get_ai_persona_generator() -> AIPersonaGenerator:
    """Get or create the global AIPersonaGenerator instance."""
    global _ai_persona_generator
    if _ai_persona_generator is None:
        _ai_persona_generator = AIPersonaGenerator()
    return _ai_persona_generator
