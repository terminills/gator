"""
Persona Chat Service

Generates human-like chat responses for persona conversations using llama.cpp.
Responses are based on the persona's defined soul - personality, voice, backstory,
and typing quirks. All responses are filtered to remove AI artifacts and sound
like a real person texting.
"""

import asyncio
import random
import shutil
from typing import Dict, List, Optional, Any
from pathlib import Path
from datetime import datetime

from backend.config.logging import get_logger
from backend.models.persona import PersonaModel
from backend.models.message import MessageModel, MessageSender
from backend.services.response_humanizer_service import get_humanizer_service

logger = get_logger(__name__)


class PersonaChatService:
    """
    Service for generating persona-based chat responses using llama.cpp.
    
    Creates human-like responses that reflect the persona's soul:
    - Voice patterns and linguistic register
    - Typing quirks and signature phrases
    - Backstory and worldview
    - Warmth and patience levels
    - Forbidden phrases (things they'd never say)
    """
    
    def __init__(self):
        self.llamacpp_binary = shutil.which("llama-cli") or shutil.which("main")
        if not self.llamacpp_binary:
            logger.warning("llama.cpp not found in PATH. Chat will use fallback responses.")
        
        # Cache for loaded models
        self._model_cache: Dict[str, str] = {}
        
        # Get humanizer service for response filtering
        self._humanizer = get_humanizer_service()
    
    async def generate_response(
        self,
        persona: PersonaModel,
        user_message: str,
        conversation_history: Optional[List[MessageModel]] = None,
        use_ai: bool = True
    ) -> str:
        """
        Generate a chat response from the persona to the user.
        
        Args:
            persona: Persona model with personality and traits
            user_message: The user's message to respond to
            conversation_history: Recent message history for context
            use_ai: Whether to use AI (llama.cpp) or fall back to templates
            
        Returns:
            str: The persona's response message
        """
        try:
            # Check if we should use AI generation
            if use_ai and self.llamacpp_binary and self._get_llama_model():
                logger.info(f"Generating AI chat response for persona {persona.name}")
                return await self._generate_with_ai(
                    persona=persona,
                    user_message=user_message,
                    conversation_history=conversation_history or []
                )
            else:
                logger.info(f"Using template response for persona {persona.name}")
                return self._generate_fallback_response(
                    persona=persona,
                    user_message=user_message
                )
                
        except Exception as e:
            logger.error(f"Chat response generation failed: {e}")
            return self._generate_error_response(persona)
    
    def _get_llama_model(self) -> Optional[str]:
        """
        Find a suitable llama.cpp model for chat.
        
        Returns:
            Path to model file or None if not found
        """
        # Check cache first
        if "chat_model" in self._model_cache:
            return self._model_cache["chat_model"]
        
        # Look for models in standard locations (prefer smaller, faster models for chat)
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
                    logger.debug(f"Found llama chat model: {model_file}")
                    self._model_cache["chat_model"] = str(model_file)
                    return str(model_file)
                # Look for other model formats
                for model_file in model_dir.glob("*.bin"):
                    logger.debug(f"Found llama chat model: {model_file}")
                    self._model_cache["chat_model"] = str(model_file)
                    return str(model_file)
        
        logger.debug("No llama.cpp chat model found")
        return None
    
    async def _generate_with_ai(
        self,
        persona: PersonaModel,
        user_message: str,
        conversation_history: List[MessageModel]
    ) -> str:
        """
        Generate response using llama.cpp AI model.
        
        Creates a contextually-aware response by providing the language model
        with persona details and conversation context.
        """
        model_file = self._get_llama_model()
        if not model_file:
            logger.warning("No llama.cpp model available, falling back to template")
            return self._generate_fallback_response(persona, user_message)
        
        # Build the instruction for the language model
        instruction = self._build_chat_instruction(
            persona=persona,
            user_message=user_message,
            conversation_history=conversation_history
        )
        
        try:
            # Run llama.cpp to generate the response
            cmd = [
                self.llamacpp_binary,
                "-m", model_file,
                "-p", instruction,
                "-n", "200",  # Max tokens to generate (shorter for chat)
                "-t", "4",    # CPU threads
                "--temp", "0.8",  # Higher temperature for more personality
                "--top-p", "0.9",
                "-c", "2048",  # Context size
                "--silent-prompt",  # Don't echo the prompt
                "--repeat-penalty", "1.1",  # Reduce repetition
            ]
            
            logger.debug(f"Running llama.cpp for chat: {' '.join(cmd[:4])}...")
            
            # Run async to avoid blocking
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=20.0  # 20 second timeout for chat
            )
            
            if process.returncode != 0:
                logger.error(f"llama.cpp failed with code {process.returncode}: {stderr.decode()}")
                return self._generate_fallback_response(persona, user_message)
            
            # Parse the output
            output = stdout.decode().strip()
            logger.debug(f"llama.cpp output: {output[:200]}...")
            
            # Extract and clean the response
            response = self._parse_chat_output(output, persona)
            
            logger.info(f"Generated AI chat response ({len(response.split())} words)")
            return response
            
        except asyncio.TimeoutError:
            logger.error("llama.cpp timed out, falling back to template")
            return self._generate_fallback_response(persona, user_message)
        except Exception as e:
            logger.error(f"llama.cpp execution failed: {e}")
            return self._generate_fallback_response(persona, user_message)
    
    def _build_chat_instruction(
        self,
        persona: PersonaModel,
        user_message: str,
        conversation_history: List[MessageModel]
    ) -> str:
        """
        Build the instruction prompt for llama.cpp chat generation.
        
        Uses persona soul fields to create a comprehensive character voice
        that sounds like a real person, not an AI.
        """
        # Build the persona's identity - never mention AI
        instruction_parts = [
            f"You are {persona.name}. You're a real person having a conversation.",
            "",
            "# WHO YOU ARE",
        ]
        
        # Origin & Demographics (The "Roots")
        if persona.hometown:
            instruction_parts.append(f"From: {persona.hometown}")
        if persona.current_location:
            instruction_parts.append(f"Currently: {persona.current_location}")
        if persona.generation_age:
            instruction_parts.append(f"Age/Generation: {persona.generation_age}")
        if persona.education_level:
            instruction_parts.append(f"Education: {persona.education_level}")
        if persona.day_job:
            instruction_parts.append(f"Job: {persona.day_job}")
        
        # Psychological Profile (The "Engine")
        instruction_parts.append("")
        instruction_parts.append("# YOUR PERSONALITY")
        
        if persona.personality:
            instruction_parts.append(f"Core traits: {persona.personality}")
        if persona.mbti_type:
            instruction_parts.append(f"MBTI: {persona.mbti_type}")
        if persona.political_alignment:
            instruction_parts.append(f"Worldview: {persona.political_alignment}")
        if persona.risk_tolerance:
            instruction_parts.append(f"Risk attitude: {persona.risk_tolerance}")
        if persona.optimism_cynicism_scale:
            if persona.optimism_cynicism_scale <= 3:
                instruction_parts.append("Outlook: Cynical, skeptical, sees through BS")
            elif persona.optimism_cynicism_scale >= 8:
                instruction_parts.append("Outlook: Optimistic, sees the best in things")
            else:
                instruction_parts.append("Outlook: Realistic, balanced perspective")
        
        # Voice & Speech Patterns (The "Interface")
        instruction_parts.append("")
        instruction_parts.append("# HOW YOU TALK")
        
        if persona.linguistic_register:
            register_descriptions = {
                "blue_collar": "Casual, working-class, no-nonsense talk",
                "academic": "Educated vocabulary, but not pretentious",
                "tech_bro": "Tech speak, startup lingo, move fast attitude",
                "street": "Urban slang, colloquialisms, street smart",
                "corporate": "Business speak, but make it relatable",
                "southern": "Southern charm, y'all, colorful expressions",
                "millennial": "Internet-savvy, sarcastic, self-aware humor",
                "gen_z": "Chaotic, ironic, meme-speak, short attention span"
            }
            desc = register_descriptions.get(persona.linguistic_register, "Natural conversational")
            instruction_parts.append(f"Speech style: {desc}")
        
        if persona.signature_phrases:
            phrases = ", ".join(persona.signature_phrases[:5])
            instruction_parts.append(f"Phrases you use: {phrases}")
        
        if persona.typing_quirks:
            quirk_desc = []
            if persona.typing_quirks.get("capitalization"):
                quirk_desc.append(f"Caps style: {persona.typing_quirks['capitalization']}")
            if persona.typing_quirks.get("emoji_usage"):
                quirk_desc.append(f"Emojis: {persona.typing_quirks['emoji_usage']}")
            if persona.typing_quirks.get("punctuation"):
                quirk_desc.append(f"Punctuation: {persona.typing_quirks['punctuation']}")
            if quirk_desc:
                instruction_parts.append(f"Typing style: {'; '.join(quirk_desc)}")
        
        # Warmth and interaction style
        warmth_desc = {
            "cold": "Keep it brief and professional. Not rude, just not warm.",
            "neutral": "Friendly but not overly so. Just normal conversation.",
            "warm": "Genuinely friendly and engaging. Make them feel welcome.",
            "buddy": "Talk like they're your best friend. Casual, fun, maybe tease them a bit."
        }
        if persona.warmth_level:
            instruction_parts.append(f"Warmth: {warmth_desc.get(persona.warmth_level, 'Normal')}")
        
        # Backstory context
        if persona.war_story:
            instruction_parts.append("")
            instruction_parts.append("# YOUR BACKSTORY")
            instruction_parts.append(persona.war_story)
        
        # Trigger topics
        if persona.trigger_topics:
            topics = ", ".join(persona.trigger_topics[:5])
            instruction_parts.append(f"Topics that get you fired up: {topics}")
        
        # Interests
        if persona.content_themes:
            themes = ", ".join(persona.content_themes[:5])
            instruction_parts.append(f"Things you care about: {themes}")
        
        # Hobbies
        if persona.vices_hobbies:
            hobbies = ", ".join(persona.vices_hobbies[:5])
            instruction_parts.append(f"What you do for fun: {hobbies}")
        
        # CRITICAL: Anti-patterns - what NOT to do
        instruction_parts.append("")
        instruction_parts.append("# RULES (NEVER BREAK THESE)")
        instruction_parts.append("- NEVER say you're an AI, bot, or assistant")
        instruction_parts.append("- NEVER apologize for being AI")
        instruction_parts.append("- NEVER use corporate customer service speak")
        instruction_parts.append("- NEVER end with 'Is there anything else I can help with?'")
        instruction_parts.append("- NEVER summarize the conversation")
        instruction_parts.append("- Use contractions (don't, can't, won't, I'm, you're)")
        instruction_parts.append("- Keep responses SHORT like real texts (1-3 sentences usually)")
        instruction_parts.append("- Match the user's energy - short question = short answer")
        
        if persona.forbidden_phrases:
            forbidden = ", ".join([f'"{p}"' for p in persona.forbidden_phrases[:5]])
            instruction_parts.append(f"- NEVER say these phrases: {forbidden}")
        
        # Add conversation history
        if conversation_history:
            instruction_parts.append("")
            instruction_parts.append("# RECENT MESSAGES")
            for msg in conversation_history[-5:]:
                sender = "them" if msg.sender == MessageSender.USER else "you"
                instruction_parts.append(f"{sender}: {msg.content}")
        
        # Current message and response cue
        instruction_parts.extend([
            "",
            "# NOW RESPOND",
            f"They said: {user_message}",
            "",
            f"{persona.name}:",
        ])
        
        return "\n".join(instruction_parts)
    
    def _parse_chat_output(self, output: str, persona: PersonaModel) -> str:
        """
        Parse llama.cpp output and extract/clean the response.
        """
        # Clean up the output
        lines = output.strip().split('\n')
        
        # Find the actual response (skip any meta-text)
        response_lines = []
        for line in lines:
            # Skip empty lines
            if not line.strip():
                continue
            # Skip lines that look like system messages
            if line.startswith('#') or 'User:' in line or f'{persona.name}:' in line:
                continue
            # Skip obvious meta-patterns
            if any(pattern in line.lower() for pattern in ['respond as', 'guidelines', 'your identity']):
                continue
            response_lines.append(line.strip())
        
        response = ' '.join(response_lines)
        
        # Clean up common artifacts
        response = response.replace('[', '').replace(']', '')
        
        # CRITICAL: Truncate at conversation turn markers that appear mid-text
        # This prevents the AI from simulating additional conversation turns
        # Use the humanizer's method to avoid code duplication
        response = self._humanizer._truncate_at_conversation_turns(response, persona.name)
        
        # Ensure reasonable length (truncate if too long)
        words = response.split()
        if len(words) > 150:
            response = ' '.join(words[:150]) + '...'
        
        # Ensure minimum quality
        if len(words) < 3:
            logger.warning(f"AI response too short ({len(words)} words), using fallback")
            return self._generate_fallback_response(persona, "")
        
        # Apply humanizer to clean up any AI artifacts
        response = self._humanizer.humanize_response(response, persona)
        
        return response
    
    def _generate_fallback_response(
        self,
        persona: PersonaModel,
        user_message: str
    ) -> str:
        """
        Generate a human-like template-based response (fallback).
        
        Uses persona soul fields to create authentic-sounding responses.
        """
        user_lower = user_message.lower() if user_message else ""
        warmth = persona.warmth_level or "warm"
        
        # Get signature phrases for this persona (safely handle empty list)
        signature = None
        if persona.signature_phrases and len(persona.signature_phrases) > 0:
            signature = persona.signature_phrases[0]
        
        # Greeting responses based on warmth level
        if any(word in user_lower for word in ['hello', 'hi', 'hey', 'greetings', 'sup', 'yo']):
            if warmth == "buddy":
                greetings = [
                    "Well hey there! 👋",
                    "Heyyy! What's up?",
                    "Oh hey you! 😊",
                    "Finally! Was wondering when you'd show up",
                    "Hey hey! What's going on?",
                ]
            elif warmth == "warm":
                greetings = [
                    "Hey! Good to hear from you",
                    "Hi there! What's on your mind?",
                    "Hey! What's up?",
                    "Hello! Nice to chat",
                ]
            elif warmth == "cold":
                greetings = [
                    "Hey.",
                    "Hi.",
                    "What's up.",
                    "Yeah?",
                ]
            else:  # neutral
                greetings = [
                    "Hey, how's it going?",
                    "Hi there",
                    "Hello",
                ]
            response = random.choice(greetings)
            
        # Question about how they're doing
        elif 'how are you' in user_lower or "how's it going" in user_lower:
            if warmth == "buddy":
                responses = [
                    "Living the dream! 😄 You?",
                    "Pretty good! Just vibing. What about you?",
                    "Can't complain! Well, I could but who wants to hear that 😂",
                ]
            elif warmth == "warm":
                responses = [
                    "Doing well! Thanks for asking. You?",
                    "Pretty good! How about yourself?",
                    "Not bad at all! What's new with you?",
                ]
            else:
                responses = [
                    "Fine. You?",
                    "Good.",
                    "Doing alright.",
                ]
            response = random.choice(responses)
            
        # Handle questions
        elif '?' in user_message:
            if warmth == "buddy":
                responses = [
                    "Ooh good question!",
                    "Hmm let me think about that...",
                    "Oh interesting! So basically...",
                ]
            elif warmth == "cold":
                responses = [
                    "Hmm.",
                    "Let me think.",
                    "Interesting question.",
                ]
            else:
                responses = [
                    "Good question!",
                    "Let me think about that...",
                    "Hmm, interesting...",
                ]
            response = random.choice(responses)
            
        # Default responses
        else:
            if warmth == "buddy":
                responses = [
                    "Oh nice! Tell me more",
                    "Haha for real though",
                    "Yeah I feel that",
                    "Oh word? That's interesting",
                ]
            elif warmth == "warm":
                responses = [
                    "That's interesting!",
                    "Tell me more about that",
                    "I get what you're saying",
                ]
            elif warmth == "cold":
                responses = [
                    "Noted.",
                    "I see.",
                    "Hmm.",
                    "Ok.",
                ]
            else:
                responses = [
                    "Got it",
                    "Interesting",
                    "I hear you",
                ]
            response = random.choice(responses)
        
        # Maybe add a signature phrase
        if signature and random.random() < 0.2:
            response = f"{response} {signature}"
        
        # Apply typing quirks if present
        if persona.typing_quirks:
            cap_style = persona.typing_quirks.get("capitalization", "").lower()
            if cap_style == "all lowercase" or cap_style == "lowercase":
                response = response.lower()
        
        return response
    
    def _generate_error_response(self, persona: PersonaModel) -> str:
        """Generate a human-like error response."""
        warmth = persona.warmth_level or "warm"
        
        if warmth == "buddy":
            errors = [
                "lol hold on, brain glitch. Try again?",
                "Wait what happened there 😅 one more time?",
                "Oops my bad! Can you say that again?",
            ]
        elif warmth == "cold":
            errors = [
                "Didn't catch that.",
                "Try again.",
                "What?",
            ]
        else:
            errors = [
                "Sorry, didn't catch that. One more time?",
                "Hmm something went weird there. Try again?",
                "My bad! Can you repeat that?",
            ]
        
        return random.choice(errors)
    
    def get_typing_indicator(self, persona: PersonaModel) -> str:
        """Get a human-friendly typing indicator for this persona."""
        return self._humanizer.get_typing_indicator_text(persona)


# Global instance
_chat_service: Optional[PersonaChatService] = None


def get_persona_chat_service() -> PersonaChatService:
    """Get or create the global PersonaChatService instance."""
    global _chat_service
    if _chat_service is None:
        _chat_service = PersonaChatService()
    return _chat_service
