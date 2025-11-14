"""
Persona Chat Service

Generates AI-powered chat responses for persona conversations using llama.cpp.
Responses are based on the persona's defined personality, appearance, and preferences.
"""

import asyncio
import shutil
from typing import Dict, List, Optional, Any
from pathlib import Path
from datetime import datetime

from backend.config.logging import get_logger
from backend.models.persona import PersonaModel
from backend.models.message import MessageModel, MessageSender

logger = get_logger(__name__)


class PersonaChatService:
    """
    Service for generating persona-based chat responses using llama.cpp.
    
    Creates contextually-aware responses that reflect the persona's:
    - Personality traits and communication style
    - Appearance and self-awareness
    - Interests and preferences
    - Conversation history
    """
    
    def __init__(self):
        self.llamacpp_binary = shutil.which("llama-cli") or shutil.which("main")
        if not self.llamacpp_binary:
            logger.warning("llama.cpp not found in PATH. Chat will use fallback responses.")
        
        # Cache for loaded models
        self._model_cache: Dict[str, str] = {}
    
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
        
        Provides the language model with persona details and conversation context.
        """
        # Start with the system instruction
        instruction_parts = [
            f"You are {persona.name}, an AI persona with the following characteristics:",
            "",
            "# Your Identity",
        ]
        
        # Add personality
        if persona.personality:
            instruction_parts.append(f"Personality: {persona.personality}")
        
        # Add appearance (for self-awareness)
        if persona.appearance:
            instruction_parts.append(f"Your appearance: {persona.appearance}")
        
        # Add content themes/interests
        if persona.content_themes:
            themes = ", ".join(persona.content_themes) if isinstance(persona.content_themes, list) else str(persona.content_themes)
            instruction_parts.append(f"Your interests: {themes}")
        
        # Add communication style hints from post_style
        if persona.post_style:
            instruction_parts.append(f"Communication style: {persona.post_style}")
        
        # Add conversation history for context
        if conversation_history:
            instruction_parts.append("")
            instruction_parts.append("# Recent Conversation")
            
            # Include last 5 messages for context
            for msg in conversation_history[-5:]:
                sender = "User" if msg.sender == MessageSender.USER else persona.name
                instruction_parts.append(f"{sender}: {msg.content}")
        
        # Add current user message
        instruction_parts.extend([
            "",
            "# Current Message",
            f"User: {user_message}",
            "",
            "# Guidelines",
            f"- Respond as {persona.name} with your unique personality",
            "- Keep responses conversational and engaging (1-3 sentences)",
            "- Stay in character based on your personality traits",
            "- Be natural and authentic",
            "- Don't mention that you're an AI unless directly asked",
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
        
        # Ensure reasonable length (truncate if too long)
        words = response.split()
        if len(words) > 150:
            response = ' '.join(words[:150]) + '...'
        
        # Ensure minimum quality
        if len(words) < 3:
            logger.warning(f"AI response too short ({len(words)} words), using fallback")
            return self._generate_fallback_response(persona, "")
        
        return response
    
    def _generate_fallback_response(
        self,
        persona: PersonaModel,
        user_message: str
    ) -> str:
        """
        Generate a template-based response (fallback).
        
        Creates a basic response when AI generation is not available.
        """
        # Extract personality hints
        personality_lower = persona.personality.lower() if persona.personality else ""
        
        # Simple keyword-based responses
        user_lower = user_message.lower()
        
        # Greeting responses
        if any(word in user_lower for word in ['hello', 'hi', 'hey', 'greetings']):
            if 'professional' in personality_lower or 'formal' in personality_lower:
                return f"Hello! How can I assist you today?"
            elif 'friendly' in personality_lower or 'warm' in personality_lower:
                return f"Hey there! Great to hear from you! What's on your mind?"
            else:
                return f"Hi! What can I do for you?"
        
        # Question responses
        if '?' in user_message:
            if 'how are you' in user_lower or 'how do you' in user_lower:
                return f"I'm doing well, thanks for asking! What about you?"
            else:
                return f"That's an interesting question. Let me think about that..."
        
        # Default responses based on personality
        if 'professional' in personality_lower:
            return f"Thank you for your message. I appreciate you reaching out."
        elif 'friendly' in personality_lower or 'warm' in personality_lower:
            return f"Thanks for chatting with me! I love hearing from you."
        elif 'creative' in personality_lower or 'artistic' in personality_lower:
            return f"That's fascinating! I'd love to explore this idea more with you."
        else:
            return f"Got your message! Let's continue this conversation."
    
    def _generate_error_response(self, persona: PersonaModel) -> str:
        """Generate a friendly error response."""
        return f"Sorry, I'm having a moment here. Can you try that again?"


# Global instance
_chat_service: Optional[PersonaChatService] = None


def get_persona_chat_service() -> PersonaChatService:
    """Get or create the global PersonaChatService instance."""
    global _chat_service
    if _chat_service is None:
        _chat_service = PersonaChatService()
    return _chat_service
