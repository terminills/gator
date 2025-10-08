"""
Template Service

Handles enhanced fallback text generation using persona characteristics and prompt analysis.
This service provides sophisticated template-based content generation with multi-dimensional scoring.
"""

from typing import Dict, Any, List
import random

from backend.models.persona import PersonaModel
from backend.config.logging import get_logger

logger = get_logger(__name__)


class TemplateService:
    """
    Service for sophisticated template-based content generation.

    Provides enhanced fallback text generation using persona characteristics,
    style preferences, and prompt analysis with multi-dimensional scoring.
    """

    def __init__(self):
        """Initialize the template service."""
        pass

    def generate_fallback_text(
        self, persona: PersonaModel, prompt: str = None, content_rating: str = "sfw"
    ) -> str:
        """
        Create enhanced fallback text using persona characteristics and prompt analysis.

        Uses base_appearance_description when appearance_locked is True for consistency.
        Leverages style_preferences for sophisticated content styling and tone.

        Args:
            persona: PersonaModel with all persona characteristics
            prompt: Optional prompt for context-aware generation
            content_rating: Content rating for the generated text

        Returns:
            str: Generated fallback text with persona-appropriate styling
        """
        # Extract key elements - use locked appearance if available
        appearance_desc = (
            persona.base_appearance_description
            if persona.appearance_locked and persona.base_appearance_description
            else persona.appearance
        )

        # Parse personality traits more comprehensively
        personality_full = persona.personality.lower()
        personality_traits = [t.strip() for t in persona.personality.split(",")]

        themes = (
            persona.content_themes[:3]
            if persona.content_themes
            else ["lifestyle", "thoughts"]
        )
        prompt_keywords = prompt.lower().split() if prompt else ["content"]

        # Extract style preferences for sophisticated content styling
        style_prefs = persona.style_preferences or {}
        aesthetic = style_prefs.get("aesthetic", "").lower()
        voice_style = style_prefs.get("voice_style", "").lower()
        tone_pref = style_prefs.get("tone", "").lower()

        # Determine content style using multi-attribute scoring
        style = self._determine_content_style(
            personality_traits, aesthetic, voice_style
        )

        # Generate appearance context
        appearance_context = self._generate_appearance_context(
            persona, appearance_desc, aesthetic
        )

        # Determine voice modifiers
        voice_modifiers = self._determine_voice_modifiers(tone_pref, personality_full)

        # Generate templates based on style
        templates = self._generate_templates_for_style(
            style, themes, appearance_context, voice_modifiers
        )

        # Select template with sophisticated logic
        selected_template = self._select_weighted_template(templates, prompt_keywords)

        # Apply dynamic customization
        customized_template = self._customize_template(
            selected_template, prompt_keywords
        )

        return customized_template

    def _determine_content_style(
        self, personality_traits: List[str], aesthetic: str, voice_style: str
    ) -> str:
        """
        Determine content style using multi-attribute scoring.

        This replaces simple keyword matching with weighted analysis.

        Args:
            personality_traits: List of personality trait strings
            aesthetic: Aesthetic preference from style_preferences
            voice_style: Voice style from style_preferences

        Returns:
            str: Determined style ("creative", "professional", "tech", or "casual")
        """
        style_scores = {"creative": 0, "professional": 0, "tech": 0, "casual": 0}

        # Score based on personality traits (primary weight)
        for trait in personality_traits:
            trait_lower = trait.lower()
            if any(
                kw in trait_lower
                for kw in [
                    "creative",
                    "artistic",
                    "innovative",
                    "imaginative",
                    "original",
                ]
            ):
                style_scores["creative"] += 3
            if any(
                kw in trait_lower
                for kw in [
                    "professional",
                    "business",
                    "corporate",
                    "executive",
                    "strategic",
                ]
            ):
                style_scores["professional"] += 3
            if any(
                kw in trait_lower
                for kw in ["tech", "technology", "analytical", "data", "engineer"]
            ):
                style_scores["tech"] += 3
            if any(
                kw in trait_lower
                for kw in ["casual", "friendly", "approachable", "warm", "relaxed"]
            ):
                style_scores["casual"] += 3

        # Score based on style_preferences (secondary weight)
        if aesthetic in ["professional", "corporate", "executive"]:
            style_scores["professional"] += 2
        elif aesthetic in ["creative", "artistic", "vibrant"]:
            style_scores["creative"] += 2
        elif aesthetic in ["tech", "modern", "futuristic"]:
            style_scores["tech"] += 2
        elif aesthetic in ["casual", "relaxed", "friendly"]:
            style_scores["casual"] += 2

        if voice_style in ["professional", "formal"]:
            style_scores["professional"] += 1
        elif voice_style in ["creative", "expressive"]:
            style_scores["creative"] += 1
        elif voice_style in ["technical", "precise"]:
            style_scores["tech"] += 1
        elif voice_style in ["casual", "conversational"]:
            style_scores["casual"] += 1

        # Select style with highest score, default to casual if tied
        style = (
            max(style_scores, key=style_scores.get)
            if max(style_scores.values()) > 0
            else "casual"
        )

        return style

    def _generate_appearance_context(
        self, persona: PersonaModel, appearance_desc: str, aesthetic: str
    ) -> str:
        """
        Generate dynamic appearance context based on multiple factors.

        Args:
            persona: PersonaModel instance
            appearance_desc: Appearance description text
            aesthetic: Aesthetic preference

        Returns:
            str: Appearance context string to append to templates
        """
        appearance_keywords = appearance_desc.lower() if appearance_desc else ""
        is_visual_locked = (
            persona.appearance_locked and persona.base_appearance_description
        )

        appearance_context = ""
        if is_visual_locked:
            # Use style preferences and appearance keywords together
            if "professional" in appearance_keywords or aesthetic == "professional":
                appearance_context = " (staying true to my professional image)"
            elif (
                "creative" in appearance_keywords
                or "artistic" in appearance_keywords
                or aesthetic == "creative"
            ):
                appearance_context = " (expressing my creative side)"
            elif (
                "casual" in appearance_keywords
                or "relaxed" in appearance_keywords
                or aesthetic == "casual"
            ):
                appearance_context = " (keeping it authentic and real)"
            elif "tech" in appearance_keywords or aesthetic in ["modern", "futuristic"]:
                appearance_context = " (maintaining my tech-forward presence)"

        return appearance_context

    def _determine_voice_modifiers(
        self, tone_pref: str, personality_full: str
    ) -> List[str]:
        """
        Determine voice modifiers based on style_preferences and personality.

        Args:
            tone_pref: Tone preference from style_preferences
            personality_full: Full personality text

        Returns:
            List[str]: List of voice modifiers
        """
        voice_modifiers = []
        if tone_pref in ["warm", "friendly", "approachable"]:
            voice_modifiers.append("warm")
        if tone_pref in ["confident", "assertive", "bold"]:
            voice_modifiers.append("confident")
        if "passionate" in personality_full:
            voice_modifiers.append("passionate")
        if "analytical" in personality_full or "data" in personality_full:
            voice_modifiers.append("analytical")

        return voice_modifiers

    def _generate_templates_for_style(
        self,
        style: str,
        themes: List[str],
        appearance_context: str,
        voice_modifiers: List[str],
    ) -> List[str]:
        """
        Generate content templates based on style and themes with enhanced variation.

        Args:
            style: Content style ("creative", "professional", "tech", or "casual")
            themes: List of content themes
            appearance_context: Appearance context string
            voice_modifiers: List of voice modifiers

        Returns:
            List[str]: List of template strings
        """
        templates = []

        if style == "creative":
            templates = [
                f"🎨 Exploring the intersection of {themes[0]} and creativity today{appearance_context}. There's something magical about how innovation sparks when we blend different perspectives. What inspires your creative process? #creativity #{themes[0].replace(' ', '')} #inspiration",
                f"✨ Just had a breakthrough moment thinking about {themes[0]}{appearance_context}. Sometimes the best ideas come when we least expect them. The creative journey is all about embracing those unexpected connections. Share your latest 'aha' moment! 💡",
                f"🚀 Passionate about {themes[0]} and the endless possibilities it brings{appearance_context}. Every challenge is just a canvas waiting for the right creative solution. What problem are you solving creatively today? #innovation #{themes[0].replace(' ', '')}",
            ]
            # Add variation based on voice modifiers
            if "passionate" in voice_modifiers:
                templates.append(
                    f"🌟 Can't stop thinking about the incredible potential in {themes[0]}{appearance_context}! The creative energy around this is absolutely electric. When passion meets purpose, magic happens. What's fueling your creative fire? 🔥 #passion #{themes[0].replace(' ', '')}"
                )
            if "warm" in voice_modifiers:
                templates.append(
                    f"💫 Hey friends! Been diving deep into {themes[0]} lately{appearance_context}, and I'm so excited to share what I've discovered. The creative community around this is amazing. Would love to hear your thoughts and experiences! ✨ #{themes[0].replace(' ', '')}"
                )
        elif style == "professional":
            templates = [
                f"Reflecting on the latest developments in {themes[0]}{appearance_context}. The landscape continues to evolve rapidly, and staying ahead requires continuous learning and adaptation. Key insights from today's analysis: strategic thinking remains paramount. Thoughts? #leadership #{themes[0].replace(' ', '')}",
                f"Professional insight{appearance_context}: {themes[0]} is reshaping how we approach business strategy. Organizations that embrace this transformation will gain significant competitive advantages. What trends are you monitoring in your industry? #business #strategy",
                f"Executive perspective on {themes[0]}{appearance_context}: Success in today's market requires both vision and execution. The companies thriving are those that balance innovation with operational excellence. How is your organization adapting? #leadership #{themes[0].replace(' ', '')}",
            ]
            # Add variation based on voice modifiers
            if "confident" in voice_modifiers:
                templates.append(
                    f"Leadership insight on {themes[0]}{appearance_context}: The data is clear - organizations that invest in this area see measurable ROI. Strategic execution is non-negotiable. What's your organization's approach? #leadership #strategy #{themes[0].replace(' ', '')}"
                )
            if "analytical" in voice_modifiers:
                templates.append(
                    f"Analysis: {themes[0]} trends reveal three critical factors{appearance_context}: 1) Market dynamics are shifting, 2) Consumer expectations are evolving, 3) Technology enables new capabilities. The intersection of these creates opportunity. Your thoughts? #{themes[0].replace(' ', '')}"
                )
        elif style == "tech":
            templates = [
                f"🔧 Diving deep into {themes[0]} today{appearance_context}. The technical implications are fascinating - we're seeing unprecedented innovation in this space. For developers and tech enthusiasts: the future is being built now. What's on your tech radar? #technology #{themes[0].replace(' ', '')} #innovation",
                f"💻 Just analyzed the latest {themes[0]} developments{appearance_context}. The algorithmic approaches being implemented are genuinely impressive. Technical breakdown: efficiency gains are substantial. Fellow engineers - what are your thoughts on the current implementation patterns?",
                f"⚡ {themes[0]} technology stack evolution{appearance_context}: From proof-of-concept to production-ready solutions, the journey has been remarkable. System architecture considerations continue to be crucial. What technical challenges are you solving? #engineering #tech",
            ]
            # Add variation based on voice modifiers
            if "analytical" in voice_modifiers:
                templates.append(
                    f"🔍 Technical analysis of {themes[0]}{appearance_context}: Performance metrics show 3x improvement over baseline. Key optimization: algorithmic efficiency at scale. Open-source contributors: what patterns are you seeing? #tech #{themes[0].replace(' ', '')} #engineering"
                )
            if "passionate" in voice_modifiers:
                templates.append(
                    f"⚙️ Absolutely loving the innovation happening in {themes[0]} right now{appearance_context}! The technical solutions being developed are game-changing. This is what drives me as a technologist - solving hard problems at scale. Who else is excited about this? 🚀 #{themes[0].replace(' ', '')}"
                )
        else:  # casual
            templates = [
                f"💭 Had some interesting thoughts about {themes[0]} today{appearance_context}. It's amazing how much this topic touches our daily lives without us even realizing it. What's your take on this? Would love to hear different perspectives! #{themes[0].replace(' ', '')} #thoughts",
                f"🌟 Something about {themes[0]} just clicked for me today{appearance_context}. Sometimes the simplest insights are the most powerful. Life's full of these little learning moments. What did you discover today? #learning #growth",
                f"✌️ Quick reflection on {themes[0]}{appearance_context} - there's so much depth here that we often overlook. Taking time to really think about these things makes such a difference. Anyone else find themselves going down these thought rabbit holes? 😄",
            ]
            # Add variation based on voice modifiers
            if "warm" in voice_modifiers:
                templates.append(
                    f"☕ Good morning friends! Sitting here thinking about {themes[0]}{appearance_context} and how it connects to our everyday experiences. Love having these conversations with you all. What's on your mind today? #{themes[0].replace(' ', '')} #community"
                )
            if "passionate" in voice_modifiers:
                templates.append(
                    f"🔥 Can we talk about {themes[0]} for a sec{appearance_context}? This stuff really matters and I'm genuinely excited to dive deeper. The more I learn, the more fascinated I become. Who else is on this journey with me? #{themes[0].replace(' ', '')}"
                )

        return templates

    def _select_weighted_template(
        self, templates: List[str], prompt_keywords: List[str]
    ) -> str:
        """
        Select template with sophisticated logic considering multiple factors.

        Args:
            templates: List of template strings
            prompt_keywords: Keywords from the prompt

        Returns:
            str: Selected template
        """
        # Weight template selection based on prompt keywords and persona attributes
        template_weights = [1.0] * len(templates)  # Start with equal weights

        # Boost certain templates based on prompt keywords
        for i, template in enumerate(templates):
            template_lower = template.lower()

            # If prompt mentions analysis/research, prefer analytical templates
            if any(
                kw in prompt_keywords
                for kw in ["analysis", "study", "research", "data"]
            ):
                if any(
                    word in template_lower
                    for word in ["analysis", "breakdown", "metrics", "data"]
                ):
                    template_weights[i] *= 2.0

            # If prompt mentions future/trends, prefer forward-looking templates
            if any(
                kw in prompt_keywords for kw in ["future", "trends", "upcoming", "next"]
            ):
                if any(
                    word in template_lower
                    for word in ["future", "evolution", "innovation", "potential"]
                ):
                    template_weights[i] *= 2.0

            # If prompt mentions community/social, prefer engagement templates
            if any(
                kw in prompt_keywords
                for kw in ["community", "social", "together", "share"]
            ):
                if any(
                    word in template_lower
                    for word in [
                        "thoughts",
                        "friends",
                        "community",
                        "share",
                        "conversation",
                    ]
                ):
                    template_weights[i] *= 1.5

        # Use weighted random selection for more contextual results
        selected_template = random.choices(templates, weights=template_weights, k=1)[0]

        return selected_template

    def _customize_template(self, template: str, prompt_keywords: List[str]) -> str:
        """
        Apply dynamic customization based on prompt keywords.

        Args:
            template: Template string to customize
            prompt_keywords: Keywords from the prompt

        Returns:
            str: Customized template
        """
        customized = template

        # Apply dynamic customization based on prompt keywords
        if any(
            keyword in ["trends", "future", "upcoming"] for keyword in prompt_keywords
        ):
            customized = customized.replace("today", "for the future").replace(
                "Today's", "Upcoming"
            )
        elif any(
            keyword in ["analysis", "study", "research"] for keyword in prompt_keywords
        ):
            customized = customized.replace("thoughts", "analysis").replace(
                "thinking", "researching"
            )
        elif any(
            keyword in ["community", "together", "social"]
            for keyword in prompt_keywords
        ):
            # Add community engagement elements
            if not any(
                word in customized.lower()
                for word in ["share", "thoughts", "perspective"]
            ):
                customized += " 🤝"

        return customized
