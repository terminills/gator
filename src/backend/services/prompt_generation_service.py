"""
Prompt Generation Service

Generates optimized image generation prompts using llama.cpp based on:
- Persona description (appearance, personality, preferences)
- RSS feed content
- Base image characteristics (if locked)
- Content rating and style preferences

This service creates detailed, contextually-aware prompts that exceed the 77-token
limit when using SDXL with compel library support.
"""

import asyncio
import shutil
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.config.logging import get_logger
from backend.models.content import ContentRating
from backend.models.persona import PersonaModel

logger = get_logger(__name__)


class PromptGenerationService:
    """
    Service for generating detailed image generation prompts using llama.cpp.

    Analyzes persona characteristics, RSS feed content, and style preferences
    to create comprehensive prompts optimized for SDXL image generation.
    """

    def __init__(self, db_session: Optional[AsyncSession] = None):
        self.llamacpp_binary = shutil.which("llama-cli") or shutil.which("main")
        if not self.llamacpp_binary:
            logger.warning(
                "llama.cpp not found in PATH. Prompt generation will use templates."
            )
        self.db = db_session

    async def generate_image_prompt(
        self,
        persona: PersonaModel,
        context: Optional[str] = None,
        content_rating: ContentRating = ContentRating.SFW,
        rss_content: Optional[Dict[str, Any]] = None,
        image_style: Optional[str] = None,
        use_ai: bool = True,
    ) -> Dict[str, Any]:
        """
        Generate a detailed image prompt for the given persona and context.

        Args:
            persona: Persona model with appearance, personality, preferences
            context: Optional context/situation for the image
            content_rating: Content rating (SFW/NSFW)
            rss_content: Optional RSS feed item for inspiration (if None, will fetch from DB)
            image_style: Optional style override (photorealistic, anime, etc.)
            use_ai: Whether to use AI (llama.cpp) or fall back to templates

        Returns:
            Dict with 'prompt', 'negative_prompt', 'style', and metadata
        """
        try:
            # Determine the style to use
            style = image_style or persona.image_style or "photorealistic"

            # Fetch RSS content from database if not provided and database is available
            if rss_content is None and self.db is not None:
                rss_content = await self._fetch_rss_content_for_persona(persona)

            # Check if we should use AI generation
            if use_ai and self.llamacpp_binary and self._get_llama_model():
                logger.info(f"Generating AI-powered prompt for persona {persona.name}")
                return await self._generate_with_ai(
                    persona=persona,
                    context=context,
                    content_rating=content_rating,
                    rss_content=rss_content,
                    style=style,
                )
            else:
                logger.info(f"Using template-based prompt for persona {persona.name}")
                return self._generate_with_template(
                    persona=persona,
                    context=context,
                    content_rating=content_rating,
                    rss_content=rss_content,
                    style=style,
                )

        except Exception as e:
            logger.error(f"Prompt generation failed: {e}")
            # Fall back to basic template
            return self._generate_fallback_prompt(persona, style)

    def _get_llama_model(self) -> Optional[str]:
        """
        Find a suitable llama.cpp model for prompt generation.

        Returns:
            Path to model file or None if not found
        """
        # Look for models in standard locations
        model_dirs = [
            Path("./models/text/llama-3.1-8b"),
            Path("./models/llama-3.1-8b"),
            Path("./models/text/qwen2.5-72b"),
            Path("./models/qwen2.5-72b"),
        ]

        for model_dir in model_dirs:
            if model_dir.exists():
                # Look for GGUF model files
                for model_file in model_dir.glob("*.gguf"):
                    logger.debug(f"Found llama model: {model_file}")
                    return str(model_file)
                # Look for other model formats
                for model_file in model_dir.glob("*.bin"):
                    logger.debug(f"Found llama model: {model_file}")
                    return str(model_file)

        logger.debug("No llama.cpp model found")
        return None

    async def _fetch_rss_content_for_persona(
        self, persona: PersonaModel
    ) -> Optional[Dict[str, Any]]:
        """
        Fetch relevant RSS feed content from the database for a persona.

        Queries recent feed items for the persona's assigned feeds and matches them
        with the persona's content themes and interests.

        Args:
            persona: Persona to fetch RSS content for

        Returns:
            Dict with RSS content including title, summary, categories, keywords
            or None if no relevant content found
        """
        if self.db is None:
            logger.debug("No database session available for RSS content fetching")
            return None

        try:
            from backend.models.feed import FeedItemModel, PersonaFeedModel

            # Get RSS feeds assigned to this persona
            stmt = (
                select(PersonaFeedModel)
                .where(PersonaFeedModel.persona_id == persona.id)
                .where(PersonaFeedModel.is_active.is_(True))
                .order_by(PersonaFeedModel.priority.desc())
                .limit(5)
            )
            result = await self.db.execute(stmt)
            persona_feeds = result.scalars().all()

            if not persona_feeds:
                logger.debug(f"No RSS feeds assigned to persona {persona.name}")
                return None

            # Extract feed IDs and topics for filtering
            feed_ids = [pf.feed_id for pf in persona_feeds]
            persona_topics = []
            for pf in persona_feeds:
                if pf.topics:
                    persona_topics.extend(pf.topics)

            # Get recent, high-relevance feed items from the last 48 hours
            cutoff_time = datetime.now(timezone.utc) - timedelta(hours=48)

            stmt = (
                select(FeedItemModel)
                .where(FeedItemModel.feed_id.in_(feed_ids))
                .where(FeedItemModel.created_at >= cutoff_time)
                .where(FeedItemModel.processed.is_(True))
                .order_by(FeedItemModel.relevance_score.desc())
                .limit(10)
            )
            result = await self.db.execute(stmt)
            feed_items = result.scalars().all()

            if not feed_items:
                logger.debug(f"No recent feed items found for persona {persona.name}")
                return None

            # Score and select the best matching item based on persona themes
            best_item = self._select_best_feed_item(
                feed_items, persona_topics, persona.content_themes
            )

            if best_item:
                logger.info(
                    f"Found relevant RSS content for persona {persona.name}: {best_item.title[:50]}"
                )
                return {
                    "title": best_item.title,
                    "summary": best_item.content_summary or best_item.description or "",
                    "categories": best_item.categories or [],
                    "keywords": best_item.keywords or [],
                    "topics": best_item.topics or [],
                    "relevance_score": best_item.relevance_score,
                }

            return None

        except Exception as e:
            logger.warning(
                f"Failed to fetch RSS content for persona {persona.name}: {e}"
            )
            return None

    def _select_best_feed_item(
        self,
        feed_items: List[Any],
        persona_topics: List[str],
        persona_themes: List[str],
    ) -> Optional[Any]:
        """
        Select the most relevant feed item based on persona topics and themes.

        Args:
            feed_items: List of FeedItemModel objects
            persona_topics: Topics assigned to persona's feeds
            persona_themes: Persona's content themes

        Returns:
            Best matching FeedItemModel or None
        """
        if not feed_items:
            return None

        # If no filtering criteria, return the highest relevance item
        if not persona_topics and not persona_themes:
            return feed_items[0]

        # Score each item based on topic and theme matches
        scored_items = []
        for item in feed_items:
            score = item.relevance_score or 0.5

            # Combine item text for matching
            item_text = f"{item.title} {item.description or ''} {' '.join(item.categories or [])} {' '.join(item.keywords or [])}".lower()

            # Boost score for persona topic matches
            for topic in persona_topics:
                if topic.lower() in item_text:
                    score += 0.2

            # Boost score for persona theme matches
            for theme in persona_themes:
                if theme.lower() in item_text:
                    score += 0.15

            # Boost score for item topics matching persona themes
            if item.topics:
                for item_topic in item.topics:
                    if item_topic.lower() in [t.lower() for t in persona_themes]:
                        score += 0.25

            scored_items.append((score, item))

        # Sort by score and return the best match
        scored_items.sort(key=lambda x: x[0], reverse=True)
        return scored_items[0][1] if scored_items else None

    async def _generate_with_ai(
        self,
        persona: PersonaModel,
        context: Optional[str],
        content_rating: ContentRating,
        rss_content: Optional[Dict[str, Any]],
        style: str,
    ) -> Dict[str, Any]:
        """
        Generate prompt using llama.cpp AI model.

        Creates a detailed, contextually-aware prompt by analyzing persona
        characteristics and providing specific guidance to the language model.
        """
        model_file = self._get_llama_model()
        if not model_file:
            logger.warning("No llama.cpp model available, falling back to template")
            return self._generate_with_template(
                persona, context, content_rating, rss_content, style
            )

        # Build the instruction for the language model
        instruction = self._build_llama_instruction(
            persona=persona,
            context=context,
            content_rating=content_rating,
            rss_content=rss_content,
            style=style,
        )

        try:
            # Run llama.cpp to generate the prompt
            cmd = [
                self.llamacpp_binary,
                "-m",
                model_file,
                "-p",
                instruction,
                "-n",
                "300",  # Max tokens to generate
                "-t",
                "4",  # CPU threads
                "--temp",
                "0.7",  # Temperature
                "--top-p",
                "0.9",
                "-c",
                "2048",  # Context size
                "--silent-prompt",  # Don't echo the prompt
            ]

            logger.debug(f"Running llama.cpp: {' '.join(cmd[:4])}...")

            # Run async to avoid blocking
            process = await asyncio.create_subprocess_exec(
                *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
            )

            stdout, stderr = await asyncio.wait_for(
                process.communicate(), timeout=30.0  # 30 second timeout
            )

            if process.returncode != 0:
                logger.error(
                    f"llama.cpp failed with code {process.returncode}: {stderr.decode()}"
                )
                return self._generate_with_template(
                    persona, context, content_rating, rss_content, style
                )

            # Parse the output
            output = stdout.decode().strip()
            logger.debug(f"llama.cpp output: {output[:200]}...")

            # Extract the prompt from the output
            parsed = self._parse_llama_output(output, style)

            logger.info(f"Generated AI prompt ({len(parsed['prompt'].split())} words)")
            return parsed

        except asyncio.TimeoutError:
            logger.error("llama.cpp timed out, falling back to template")
            return self._generate_with_template(
                persona, context, content_rating, rss_content, style
            )
        except Exception as e:
            logger.error(f"llama.cpp execution failed: {e}")
            return self._generate_with_template(
                persona, context, content_rating, rss_content, style
            )

    def _build_llama_instruction(
        self,
        persona: PersonaModel,
        context: Optional[str],
        content_rating: ContentRating,
        rss_content: Optional[Dict[str, Any]],
        style: str,
    ) -> str:
        """
        Build the instruction prompt for llama.cpp.

        Provides detailed guidance to the language model about what kind
        of image prompt to generate.
        """
        # Start with the system instruction
        instruction_parts = [
            "You are an expert at creating detailed image generation prompts for Stable Diffusion XL.",
            "Generate a detailed, specific prompt for an image based on the following information.",
            "",
            "# Persona Information",
        ]

        # Add appearance details - use base_appearance_description if appearance is locked
        if persona.appearance_locked and persona.base_appearance_description:
            instruction_parts.append(
                f"Appearance (LOCKED - maintain consistency): {persona.base_appearance_description}"
            )
            instruction_parts.append(
                "IMPORTANT: The appearance MUST match the locked description exactly for visual consistency"
            )
        elif persona.appearance:
            instruction_parts.append(f"Appearance: {persona.appearance}")

        # Add personality
        if persona.personality:
            instruction_parts.append(f"Personality: {persona.personality}")

        # Add post style for engagement context
        if hasattr(persona, "post_style") and persona.post_style:
            instruction_parts.append(f"Post Style: {persona.post_style}")

        # Add interests/preferences from content themes
        if persona.content_themes:
            instruction_parts.append(f"Interests: {', '.join(persona.content_themes)}")

        # Add style preference
        instruction_parts.append(f"Image Style: {style}")

        # Add content rating
        if content_rating == ContentRating.NSFW:
            instruction_parts.append("Content Rating: NSFW allowed")
        else:
            instruction_parts.append("Content Rating: SFW only (family-friendly)")

        # Add persona's AI model preferences if available
        # These inform the prompt generator about the persona's preferred generation style
        if (
            hasattr(persona, "image_model_preference")
            and persona.image_model_preference
        ):
            instruction_parts.append(
                f"Preferred Image Model: {persona.image_model_preference}"
            )
        if hasattr(persona, "nsfw_model_preference") and persona.nsfw_model_preference:
            instruction_parts.append(
                f"Preferred NSFW Model: {persona.nsfw_model_preference}"
            )

        # Add context if provided
        # Interpret instruction-like text as hints rather than literal context
        if context:
            instruction_words = ["generate", "create", "make", "produce", "based on"]
            context_lower = context.lower()
            is_instruction = any(word in context_lower for word in instruction_words)

            if is_instruction:
                # If context looks like an instruction, interpret it as guidance
                instruction_parts.append(f"User Request: {context}")
                instruction_parts.append(
                    "Note: Interpret the above as guidance for content direction, not literal text."
                )
            else:
                # Standard situational context
                instruction_parts.append(f"Context/Situation: {context}")

        # Add RSS content if available - generate a reaction/engagement prompt
        if rss_content:
            instruction_parts.append("")
            instruction_parts.append("# RSS Feed Content - Generate a Reaction")
            instruction_parts.append(f"Title: {rss_content.get('title', '')}")

            summary = rss_content.get("summary", "")
            if summary:
                instruction_parts.append(f"Summary: {summary[:200]}")

            # Add categories and keywords for context
            categories = rss_content.get("categories", [])
            if categories:
                instruction_parts.append(f"Categories: {', '.join(categories[:5])}")

            keywords = rss_content.get("keywords", [])
            if keywords:
                instruction_parts.append(f"Keywords: {', '.join(keywords[:5])}")

            instruction_parts.append("")
            instruction_parts.append(
                "IMPORTANT: Generate an image prompt showing the persona reacting to or engaging with this content."
            )
            instruction_parts.append("Examples:")
            instruction_parts.append(
                "- Reading/viewing the content on a device (phone, tablet, laptop)"
            )
            instruction_parts.append(
                "- Expressing emotion about the topic (excited, thoughtful, curious)"
            )
            instruction_parts.append("- In a setting related to the content theme")
            instruction_parts.append(
                "- Their facial expression and body language should reflect engagement with the topic"
            )

        # Add generation guidelines
        instruction_parts.extend(
            [
                "",
                "# Guidelines",
                f"- Create a detailed {style} image prompt",
                "- Include specific visual details about appearance, pose, setting, lighting",
                "- Focus on the persona's unique characteristics",
                "- Make the prompt vivid and specific (aim for 100-200 words)",
                "- Use professional photography/art terminology",
                "- Ensure the prompt is appropriate for the content rating",
                "",
                "# Output Format",
                "Generate ONLY the image prompt text, nothing else. Do not include explanations or metadata.",
                "",
                "Image Prompt:",
            ]
        )

        return "\n".join(instruction_parts)

    def _parse_llama_output(self, output: str, style: str) -> Dict[str, Any]:
        """
        Parse llama.cpp output and extract/clean the prompt.
        """
        # Clean up the output
        lines = output.strip().split("\n")

        # Find the actual prompt (skip any meta-text)
        prompt_lines = []
        for line in lines:
            # Skip empty lines and common meta patterns
            if not line.strip():
                continue
            if line.startswith("#") or line.startswith("Image Prompt:"):
                continue
            if "generate" in line.lower() and "prompt" in line.lower():
                continue
            prompt_lines.append(line.strip())

        prompt = " ".join(prompt_lines)

        # Ensure minimum quality
        if len(prompt.split()) < 20:
            logger.warning(
                f"AI prompt too short ({len(prompt.split())} words), enriching"
            )
            # Add style-specific qualifiers
            prompt = self._enrich_short_prompt(prompt, style)

        # Generate appropriate negative prompt
        negative_prompt = self._generate_negative_prompt(style)

        return {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "style": style,
            "source": "ai_generated",
            "word_count": len(prompt.split()),
        }

    def _enrich_short_prompt(self, prompt: str, style: str) -> str:
        """Enrich a short prompt with style-specific details."""
        style_additions = {
            "photorealistic": "professional photograph, highly detailed, lifelike, 8k quality, sharp focus, professional lighting",
            "anime": "beautiful anime art, highly detailed, vibrant colors, studio quality animation",
            "cartoon": "cartoon illustration, stylized, clean lines, vibrant colors",
            "artistic": "artistic painting, fine art, museum quality, expressive",
            "3d_render": "3d render, cgi, octane render, highly detailed 3d model",
            "fantasy": "fantasy art, epic, magical atmosphere, detailed fantasy illustration",
            "cinematic": "cinematic shot, movie quality, dramatic lighting, film grain",
        }

        addition = style_additions.get(style, style_additions["photorealistic"])
        return f"{prompt}, {addition}"

    def _generate_with_template(
        self,
        persona: PersonaModel,
        context: Optional[str],
        content_rating: ContentRating,
        rss_content: Optional[Dict[str, Any]],
        style: str,
    ) -> Dict[str, Any]:
        """
        Generate prompt using template-based approach (fallback).

        Creates a structured prompt from persona attributes without AI.
        """
        prompt_parts = []

        # Start with style prefix
        style_prefixes = {
            "photorealistic": "Professional high-resolution portrait photograph of",
            "anime": "Beautiful anime art depicting",
            "cartoon": "Cartoon illustration of",
            "artistic": "Artistic painting of",
            "3d_render": "3D rendered character of",
            "fantasy": "Fantasy art portrait of",
            "cinematic": "Cinematic portrait of",
        }

        prefix = style_prefixes.get(style, style_prefixes["photorealistic"])
        prompt_parts.append(prefix)

        # Add appearance - use base_appearance_description if appearance is locked
        if persona.appearance_locked and persona.base_appearance_description:
            prompt_parts.append(persona.base_appearance_description)
        elif persona.appearance:
            prompt_parts.append(persona.appearance)

        # Add personality context
        if persona.personality:
            # Use first 100 chars of personality description for prompt
            personality_desc = (
                persona.personality[:100]
                if len(persona.personality) > 100
                else persona.personality
            )
            prompt_parts.append(f"expressing {personality_desc} personality")

        # Add context if provided
        # Skip contexts that look like instructions (contain words like "generate", "create", "make")
        if context:
            instruction_words = ["generate", "create", "make", "produce", "based on"]
            context_lower = context.lower()
            # Only add context if it doesn't look like an instruction to the system
            is_instruction = any(word in context_lower for word in instruction_words)
            if not is_instruction:
                prompt_parts.append(f"in {context}")

        # Add RSS-inspired reaction elements
        if rss_content:
            rss_title = rss_content.get("title", "")
            rss_keywords = rss_content.get("keywords", [])
            rss_topics = rss_content.get("topics", [])

            # Generate a reaction/engagement scenario
            reaction_elements = []

            # Add device/reading context
            reaction_elements.append("holding smartphone")

            # Add emotional reaction based on content
            if rss_keywords:
                # Positive/exciting keywords
                exciting_words = [
                    "breakthrough",
                    "innovation",
                    "success",
                    "amazing",
                    "win",
                    "achievement",
                ]
                if any(
                    word in " ".join(rss_keywords).lower() for word in exciting_words
                ):
                    reaction_elements.append("excited expression")
                else:
                    reaction_elements.append("thoughtfully engaged")
            else:
                reaction_elements.append("reading attentively")

            # Add thematic setting based on RSS content
            if rss_topics:
                topic_settings = {
                    "technology": "modern tech workspace",
                    "business": "professional office setting",
                    "science": "contemporary study",
                    "health": "wellness environment",
                    "entertainment": "casual lifestyle setting",
                    "sports": "active lifestyle environment",
                }
                for topic in rss_topics:
                    if topic.lower() in topic_settings:
                        reaction_elements.append(f"in {topic_settings[topic.lower()]}")
                        break

            # Add the reaction context
            if reaction_elements:
                prompt_parts.append(", ".join(reaction_elements))

            # Add content inspiration
            if rss_title:
                prompt_parts.append(f"reacting to content about {rss_title[:40]}")

        # Add style-specific qualities
        style_qualities = {
            "photorealistic": "realistic style, natural lighting, photorealistic, ultra detailed, 8k quality, sharp focus, professional photography",
            "anime": "anime style, vibrant colors, detailed anime art, studio quality",
            "cartoon": "cartoon style, clean lines, vibrant colors, stylized",
            "artistic": "artistic style, painterly, expressive, museum quality",
            "3d_render": "3d style, cgi, octane render, highly detailed",
            "fantasy": "fantasy style, magical, epic, detailed",
            "cinematic": "cinematic style, dramatic lighting, film quality",
        }

        prompt_parts.append(
            style_qualities.get(style, style_qualities["photorealistic"])
        )

        # Add safety for SFW content
        if content_rating == ContentRating.SFW:
            prompt_parts.append(
                "safe for work, family-friendly, appropriate for all audiences"
            )

        prompt = ", ".join(prompt_parts)
        negative_prompt = self._generate_negative_prompt(style)

        return {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "style": style,
            "source": "template",
            "word_count": len(prompt.split()),
        }

    def _generate_fallback_prompt(
        self, persona: PersonaModel, style: str
    ) -> Dict[str, Any]:
        """
        Generate a minimal fallback prompt.
        """
        prompt = f"Professional portrait of {persona.name}, {style} style, high quality"
        if persona.appearance:
            prompt = f"Professional portrait of {persona.appearance}, {style} style, high quality"

        return {
            "prompt": prompt,
            "negative_prompt": "ugly, blurry, low quality, distorted",
            "style": style,
            "source": "fallback",
            "word_count": len(prompt.split()),
        }

    def _generate_negative_prompt(self, style: str) -> str:
        """
        Generate style-appropriate negative prompt.
        """
        base_negative = "ugly, blurry, low quality, distorted, deformed, bad anatomy"

        style_specific = {
            "photorealistic": "cartoon, anime, 3d render, illustration, painting, drawing, art, sketched",
            "anime": "realistic, photorealistic, 3d, western cartoon",
            "cartoon": "realistic, photorealistic, anime, 3d render",
            "artistic": "photograph, 3d render, anime, cartoon",
            "3d_render": "photograph, 2d, anime, cartoon, flat",
            "fantasy": "realistic photograph, modern, mundane",
            "cinematic": "cartoon, anime, illustration, amateur",
        }

        specific = style_specific.get(style, "")
        if specific:
            return f"{base_negative}, {specific}"
        return base_negative


# Global instance (deprecated - use with database session instead)
_prompt_service: Optional[PromptGenerationService] = None


def get_prompt_service(
    db_session: Optional[AsyncSession] = None,
) -> PromptGenerationService:
    """
    Get or create the PromptGenerationService instance.

    Args:
        db_session: Optional database session for RSS content fetching.
                    If provided, creates a new instance with database access.
                    If None, returns the global instance without database access.

    Returns:
        PromptGenerationService instance
    """
    # If database session is provided, create a new instance with it
    if db_session is not None:
        return PromptGenerationService(db_session)

    # Otherwise, return the global instance
    global _prompt_service
    if _prompt_service is None:
        _prompt_service = PromptGenerationService()
    return _prompt_service
