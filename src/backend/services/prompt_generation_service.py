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
import json
import subprocess
import shutil
from typing import Dict, List, Optional, Any
from pathlib import Path

from backend.config.logging import get_logger
from backend.models.persona import PersonaModel
from backend.models.content import ContentRating

logger = get_logger(__name__)


class PromptGenerationService:
    """
    Service for generating detailed image generation prompts using llama.cpp.
    
    Analyzes persona characteristics, RSS feed content, and style preferences
    to create comprehensive prompts optimized for SDXL image generation.
    """
    
    def __init__(self):
        self.llamacpp_binary = shutil.which("llama-cli") or shutil.which("main")
        if not self.llamacpp_binary:
            logger.warning("llama.cpp not found in PATH. Prompt generation will use templates.")
        
    async def generate_image_prompt(
        self,
        persona: PersonaModel,
        context: Optional[str] = None,
        content_rating: ContentRating = ContentRating.SFW,
        rss_content: Optional[Dict[str, Any]] = None,
        image_style: Optional[str] = None,
        use_ai: bool = True
    ) -> Dict[str, Any]:
        """
        Generate a detailed image prompt for the given persona and context.
        
        Args:
            persona: Persona model with appearance, personality, preferences
            context: Optional context/situation for the image
            content_rating: Content rating (SFW/NSFW)
            rss_content: Optional RSS feed item for inspiration
            image_style: Optional style override (photorealistic, anime, etc.)
            use_ai: Whether to use AI (llama.cpp) or fall back to templates
            
        Returns:
            Dict with 'prompt', 'negative_prompt', 'style', and metadata
        """
        try:
            # Determine the style to use
            style = image_style or persona.image_style or "photorealistic"
            
            # Check if we should use AI generation
            if use_ai and self.llamacpp_binary and self._get_llama_model():
                logger.info(f"Generating AI-powered prompt for persona {persona.name}")
                return await self._generate_with_ai(
                    persona=persona,
                    context=context,
                    content_rating=content_rating,
                    rss_content=rss_content,
                    style=style
                )
            else:
                logger.info(f"Using template-based prompt for persona {persona.name}")
                return self._generate_with_template(
                    persona=persona,
                    context=context,
                    content_rating=content_rating,
                    rss_content=rss_content,
                    style=style
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
    
    async def _generate_with_ai(
        self,
        persona: PersonaModel,
        context: Optional[str],
        content_rating: ContentRating,
        rss_content: Optional[Dict[str, Any]],
        style: str
    ) -> Dict[str, Any]:
        """
        Generate prompt using llama.cpp AI model.
        
        Creates a detailed, contextually-aware prompt by analyzing persona
        characteristics and providing specific guidance to the language model.
        """
        model_file = self._get_llama_model()
        if not model_file:
            logger.warning("No llama.cpp model available, falling back to template")
            return self._generate_with_template(persona, context, content_rating, rss_content, style)
        
        # Build the instruction for the language model
        instruction = self._build_llama_instruction(
            persona=persona,
            context=context,
            content_rating=content_rating,
            rss_content=rss_content,
            style=style
        )
        
        try:
            # Run llama.cpp to generate the prompt
            cmd = [
                self.llamacpp_binary,
                "-m", model_file,
                "-p", instruction,
                "-n", "300",  # Max tokens to generate
                "-t", "4",    # CPU threads
                "--temp", "0.7",  # Temperature
                "--top-p", "0.9",
                "-c", "2048",  # Context size
                "--silent-prompt",  # Don't echo the prompt
            ]
            
            logger.debug(f"Running llama.cpp: {' '.join(cmd[:4])}...")
            
            # Run async to avoid blocking
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=30.0  # 30 second timeout
            )
            
            if process.returncode != 0:
                logger.error(f"llama.cpp failed with code {process.returncode}: {stderr.decode()}")
                return self._generate_with_template(persona, context, content_rating, rss_content, style)
            
            # Parse the output
            output = stdout.decode().strip()
            logger.debug(f"llama.cpp output: {output[:200]}...")
            
            # Extract the prompt from the output
            parsed = self._parse_llama_output(output, style)
            
            logger.info(f"Generated AI prompt ({len(parsed['prompt'].split())} words)")
            return parsed
            
        except asyncio.TimeoutError:
            logger.error("llama.cpp timed out, falling back to template")
            return self._generate_with_template(persona, context, content_rating, rss_content, style)
        except Exception as e:
            logger.error(f"llama.cpp execution failed: {e}")
            return self._generate_with_template(persona, context, content_rating, rss_content, style)
    
    def _build_llama_instruction(
        self,
        persona: PersonaModel,
        context: Optional[str],
        content_rating: ContentRating,
        rss_content: Optional[Dict[str, Any]],
        style: str
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
        
        # Add appearance details
        if persona.appearance:
            instruction_parts.append(f"Appearance: {persona.appearance}")
        
        # Add personality
        if persona.personality_traits:
            instruction_parts.append(f"Personality: {', '.join(persona.personality_traits)}")
        
        # Add interests/preferences
        if persona.interests:
            instruction_parts.append(f"Interests: {', '.join(persona.interests)}")
        
        # Add style preference
        instruction_parts.append(f"Image Style: {style}")
        
        # Add content rating
        if content_rating == ContentRating.NSFW:
            instruction_parts.append("Content Rating: NSFW allowed")
        else:
            instruction_parts.append("Content Rating: SFW only (family-friendly)")
        
        # Add context if provided
        if context:
            instruction_parts.append(f"Context/Situation: {context}")
        
        # Add RSS content if available
        if rss_content:
            instruction_parts.append(f"Inspiration: {rss_content.get('title', '')} - {rss_content.get('summary', '')[:200]}")
        
        # Add generation guidelines
        instruction_parts.extend([
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
        ])
        
        return "\n".join(instruction_parts)
    
    def _parse_llama_output(self, output: str, style: str) -> Dict[str, Any]:
        """
        Parse llama.cpp output and extract/clean the prompt.
        """
        # Clean up the output
        lines = output.strip().split('\n')
        
        # Find the actual prompt (skip any meta-text)
        prompt_lines = []
        for line in lines:
            # Skip empty lines and common meta patterns
            if not line.strip():
                continue
            if line.startswith('#') or line.startswith('Image Prompt:'):
                continue
            if 'generate' in line.lower() and 'prompt' in line.lower():
                continue
            prompt_lines.append(line.strip())
        
        prompt = ' '.join(prompt_lines)
        
        # Ensure minimum quality
        if len(prompt.split()) < 20:
            logger.warning(f"AI prompt too short ({len(prompt.split())} words), enriching")
            # Add style-specific qualifiers
            prompt = self._enrich_short_prompt(prompt, style)
        
        # Generate appropriate negative prompt
        negative_prompt = self._generate_negative_prompt(style)
        
        return {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "style": style,
            "source": "ai_generated",
            "word_count": len(prompt.split())
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
        style: str
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
        
        # Add appearance
        if persona.appearance:
            prompt_parts.append(persona.appearance)
        
        # Add personality context
        if persona.personality_traits:
            traits = ', '.join(persona.personality_traits[:3])  # Top 3 traits
            prompt_parts.append(f"expressing {traits} personality")
        
        # Add context if provided
        if context:
            prompt_parts.append(f"in {context}")
        
        # Add RSS-inspired elements
        if rss_content and rss_content.get('title'):
            # Extract key themes from RSS title
            prompt_parts.append(f"themed around {rss_content['title'][:50]}")
        
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
        
        prompt_parts.append(style_qualities.get(style, style_qualities["photorealistic"]))
        
        # Add safety for SFW content
        if content_rating == ContentRating.SFW:
            prompt_parts.append("safe for work, family-friendly, appropriate for all audiences")
        
        prompt = ", ".join(prompt_parts)
        negative_prompt = self._generate_negative_prompt(style)
        
        return {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "style": style,
            "source": "template",
            "word_count": len(prompt.split())
        }
    
    def _generate_fallback_prompt(
        self,
        persona: PersonaModel,
        style: str
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
            "word_count": len(prompt.split())
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


# Global instance
_prompt_service: Optional[PromptGenerationService] = None


def get_prompt_service() -> PromptGenerationService:
    """Get or create the global PromptGenerationService instance."""
    global _prompt_service
    if _prompt_service is None:
        _prompt_service = PromptGenerationService()
    return _prompt_service
