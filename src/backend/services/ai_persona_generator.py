"""
AI Persona Generator Service

Generates realistic, coherent personas using llama.cpp for full AI-powered creation.
Produces more realistic and internally consistent personas compared to random templates.
"""

import asyncio
import json
import shutil
import random
from typing import Dict, List, Optional, Any
from pathlib import Path

from backend.config.logging import get_logger
from backend.models.persona import ContentRating

logger = get_logger(__name__)


class AIPersonaGenerator:
    """
    AI-powered persona generator using llama.cpp.
    
    Creates coherent, realistic personas with internally consistent:
    - Appearance descriptions
    - Personality traits
    - Content themes and interests
    - Style preferences
    
    Falls back to template-based generation if AI is unavailable.
    """
    
    def __init__(self):
        self.llamacpp_binary = shutil.which("llama-cli") or shutil.which("main")
        if not self.llamacpp_binary:
            logger.warning("llama.cpp not found in PATH. Will use template-based generation.")
        
        # Cache for loaded models
        self._model_cache: Dict[str, str] = {}
    
    async def generate_persona(
        self,
        name: Optional[str] = None,
        persona_type: Optional[str] = None,
        use_ai: bool = True
    ) -> Dict[str, Any]:
        """
        Generate a complete persona configuration.
        
        Args:
            name: Optional custom name (generated if not provided)
            persona_type: Optional persona type hint (e.g., "fitness", "tech", "fashion")
            use_ai: Whether to use AI (llama.cpp) or fall back to templates
            
        Returns:
            Dict with complete persona configuration
        """
        try:
            # Check if we should use AI generation
            if use_ai and self.llamacpp_binary and self._get_llama_model():
                logger.info(f"Generating AI-powered persona (type: {persona_type or 'general'})")
                return await self._generate_with_ai(
                    name=name,
                    persona_type=persona_type
                )
            else:
                logger.info("Using template-based persona generation")
                return self._generate_with_template(name, persona_type)
                
        except Exception as e:
            logger.error(f"AI persona generation failed: {e}")
            # Fall back to template
            return self._generate_with_template(name, persona_type)
    
    async def generate_batch(
        self,
        count: int = 5,
        persona_types: Optional[List[str]] = None,
        use_ai: bool = True
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
                persona_type=persona_type,
                use_ai=use_ai
            )
            personas.append(persona)
        
        return personas
    
    def _get_llama_model(self) -> Optional[str]:
        """
        Find a suitable llama.cpp model for persona generation.
        
        Returns:
            Path to model file or None if not found
        """
        # Check cache first
        if "persona_model" in self._model_cache:
            return self._model_cache["persona_model"]
        
        # Look for models in standard locations
        model_dirs = [
            Path("./models/text/llama-3.1-8b"),
            Path("./models/llama-3.1-8b"),
            Path("./models/text/qwen2.5-72b"),
            Path("./models/qwen2.5-72b"),
            Path("./models/text/mixtral-8x7b"),
            Path("./models/mixtral-8x7b"),
        ]
        
        for model_dir in model_dirs:
            if model_dir.exists():
                # Look for GGUF model files
                for model_file in model_dir.glob("*.gguf"):
                    logger.debug(f"Found llama persona model: {model_file}")
                    self._model_cache["persona_model"] = str(model_file)
                    return str(model_file)
                # Look for other model formats
                for model_file in model_dir.glob("*.bin"):
                    logger.debug(f"Found llama persona model: {model_file}")
                    self._model_cache["persona_model"] = str(model_file)
                    return str(model_file)
        
        logger.debug("No llama.cpp persona model found")
        return None
    
    async def _generate_with_ai(
        self,
        name: Optional[str],
        persona_type: Optional[str]
    ) -> Dict[str, Any]:
        """
        Generate persona using llama.cpp AI model.
        
        Creates a coherent, realistic persona with internally consistent attributes.
        """
        model_file = self._get_llama_model()
        if not model_file:
            logger.warning("No llama.cpp model available, falling back to template")
            return self._generate_with_template(name, persona_type)
        
        # Generate a name if not provided
        if not name:
            name = self._generate_random_name()
        
        # Build the instruction for the language model
        instruction = self._build_persona_instruction(name, persona_type)
        
        try:
            # Run llama.cpp to generate the persona
            cmd = [
                self.llamacpp_binary,
                "-m", model_file,
                "-p", instruction,
                "-n", "500",  # Max tokens to generate
                "-t", "4",    # CPU threads
                "--temp", "0.9",  # Higher temperature for creativity
                "--top-p", "0.95",
                "-c", "2048",  # Context size
                "--silent-prompt",  # Don't echo the prompt
            ]
            
            logger.debug(f"Running llama.cpp for persona generation: {' '.join(cmd[:4])}...")
            
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
                return self._generate_with_template(name, persona_type)
            
            # Parse the output
            output = stdout.decode().strip()
            logger.debug(f"llama.cpp output: {output[:300]}...")
            
            # Extract and parse the persona JSON
            persona_data = self._parse_persona_output(output, name, persona_type)
            
            logger.info(f"Generated AI persona: {persona_data['name']}")
            return persona_data
            
        except asyncio.TimeoutError:
            logger.error("llama.cpp timed out, falling back to template")
            return self._generate_with_template(name, persona_type)
        except Exception as e:
            logger.error(f"llama.cpp execution failed: {e}")
            return self._generate_with_template(name, persona_type)
    
    def _build_persona_instruction(
        self,
        name: str,
        persona_type: Optional[str]
    ) -> str:
        """
        Build the instruction prompt for llama.cpp persona generation.
        """
        # Start with the system instruction
        instruction_parts = [
            "You are an expert at creating realistic AI influencer personas.",
            f"Create a detailed, coherent persona profile for {name}.",
            "",
        ]
        
        # Add persona type guidance if provided
        if persona_type:
            instruction_parts.append(f"This persona should be focused on: {persona_type}")
            instruction_parts.append("")
        
        instruction_parts.extend([
            "# Requirements",
            "Create a persona with:",
            "1. Appearance: Detailed physical description (age, build, ethnicity, hair, eyes, clothing style)",
            "2. Personality: Character traits, communication style, values (2-3 sentences)",
            "3. Interests: 3-5 specific interests or topics they're passionate about",
            "4. Content Themes: 4-6 content themes they would create content about",
            "5. Style Preferences: Image style, post style, video types",
            "",
            "# Important Guidelines",
            "- Make the persona realistic and internally consistent",
            "- The appearance, personality, interests, and content should all fit together naturally",
            "- Use specific, concrete details (not generic descriptions)",
            "- Think about what makes this persona unique and interesting",
            "- Consider their target audience and content niche",
            "",
            "# Output Format",
            "Provide ONLY a JSON object with this exact structure (no additional text):",
            "{",
            '  "appearance": "detailed appearance description",',
            '  "personality": "personality traits and communication style",',
            '  "interests": ["interest1", "interest2", "interest3"],',
            '  "content_themes": ["theme1", "theme2", "theme3", "theme4"],',
            '  "image_style": "photorealistic or anime or cartoon",',
            '  "post_style": "casual or professional or artistic",',
            '  "video_types": ["short_clip", "story"]',
            "}",
            "",
            "JSON:",
        ])
        
        return "\n".join(instruction_parts)
    
    def _parse_persona_output(
        self,
        output: str,
        name: str,
        persona_type: Optional[str]
    ) -> Dict[str, Any]:
        """
        Parse llama.cpp output and extract persona data.
        """
        # Try to extract JSON from the output
        try:
            # Find JSON in the output (look for { ... })
            start_idx = output.find('{')
            end_idx = output.rfind('}')
            
            if start_idx != -1 and end_idx != -1:
                json_str = output[start_idx:end_idx + 1]
                persona_data = json.loads(json_str)
                
                # Validate required fields
                required_fields = ['appearance', 'personality', 'interests', 'content_themes']
                if all(field in persona_data for field in required_fields):
                    # Build complete persona configuration
                    return self._build_persona_config(
                        name=name,
                        appearance=persona_data.get('appearance', ''),
                        personality=persona_data.get('personality', ''),
                        interests=persona_data.get('interests', []),
                        content_themes=persona_data.get('content_themes', []),
                        image_style=persona_data.get('image_style', 'photorealistic'),
                        post_style=persona_data.get('post_style', 'casual'),
                        video_types=persona_data.get('video_types', ['short_clip', 'story'])
                    )
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse JSON from llama output: {e}")
        except Exception as e:
            logger.warning(f"Error parsing persona data: {e}")
        
        # If parsing failed, fall back to template
        logger.warning("Could not parse AI output, using template fallback")
        return self._generate_with_template(name, persona_type)
    
    def _build_persona_config(
        self,
        name: str,
        appearance: str,
        personality: str,
        interests: List[str],
        content_themes: List[str],
        image_style: str,
        post_style: str,
        video_types: List[str]
    ) -> Dict[str, Any]:
        """
        Build a complete persona configuration from components.
        """
        # Generate content rating (random for now, could be AI-powered too)
        default_rating, allowed_ratings = self._generate_random_content_rating()
        
        return {
            "name": name,
            "appearance": appearance,
            "personality": personality,
            "content_themes": content_themes[:6],  # Max 6 themes
            "style_preferences": {
                "image_style": image_style,
                "post_style": post_style,
                "video_types": video_types[:3]  # Max 3 video types
            },
            "default_content_rating": default_rating,
            "allowed_content_ratings": allowed_ratings,
            "platform_restrictions": self._generate_platform_restrictions(default_rating),
            "image_style": image_style,
            "post_style": post_style,
            "video_types": video_types[:3],
            "is_active": True
        }
    
    def _generate_with_template(
        self,
        name: Optional[str],
        persona_type: Optional[str]
    ) -> Dict[str, Any]:
        """
        Generate persona using template-based approach (fallback).
        
        Uses the original PersonaRandomizer logic as fallback.
        """
        from backend.services.persona_randomizer import PersonaRandomizer
        
        persona = PersonaRandomizer.generate_complete_random_persona(
            name=name,
            detailed=True
        )
        
        # Add source indicator
        persona['generation_method'] = 'template'
        
        return persona
    
    def _generate_random_name(self) -> str:
        """Generate a random name."""
        from backend.services.persona_randomizer import PersonaRandomizer
        return PersonaRandomizer.generate_random_name()
    
    def _generate_random_content_rating(self) -> tuple:
        """Generate random content rating."""
        from backend.services.persona_randomizer import PersonaRandomizer
        return PersonaRandomizer.generate_random_content_rating()
    
    def _generate_platform_restrictions(self, default_rating: ContentRating) -> Dict[str, str]:
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
