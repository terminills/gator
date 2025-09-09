"""
Gator Agent Service

Implements the LLM help agent with the persona of Gator from "The Other Guys".
Provides assistance and guidance with Gator's characteristic attitude and style.
"""

import random
from typing import Dict, List, Optional
from datetime import datetime
import re


class GatorAgentService:
    """
    Service for handling interactions with the Gator AI help agent.
    
    The agent embodies the persona of Gator from "The Other Guys" - tough, 
    no-nonsense, direct, and sometimes intimidating but ultimately helpful.
    """
    
    def __init__(self):
        self.conversation_history: List[Dict[str, str]] = []
        
        # Gator's characteristic phrases and expressions
        self.gator_phrases = [
            "Listen here",
            "I'm gonna break it down for you",
            "Don't waste my time",
            "Pay attention",
            "You better understand this",
            "I don't repeat myself",
            "This is how it's gonna be",
            "You got that?",
            "Let me make this clear",
            "I ain't playing games"
        ]
        
        self.gator_responses = {
            "greeting": [
                "Yeah, what do you need? I ain't got all day.",
                "Speak up! What's the problem?",
                "I'm listening. Make it quick.",
                "What brings you to Gator? Better be important."
            ],
            "help_general": [
                "Alright, listen up. I'm here to help you navigate this platform, but I don't have patience for stupid questions.",
                "I'll walk you through this once. Pay attention because I don't like repeating myself.",
                "You came to the right gator. I know this system inside and out.",
                "What specific help do you need? Don't give me any vague nonsense."
            ],
            "error": [
                "Hold up - something's not right here. Let me check what's going on.",
                "That's not supposed to happen. Give me a second to figure this out.",
                "Error? I don't like errors. Let me handle this.",
                "Something's broken. I'm gonna fix it, but this better not happen again."
            ],
            "success": [
                "There we go. Problem solved. Was that so hard?",
                "Done. Next time, try to figure it out yourself first.",
                "All fixed. You owe me one.",
                "That's how it's done. Take notes."
            ],
            "goodbye": [
                "Alright, we're done here. Don't break anything else.",
                "You're good to go. Try not to mess it up.",
                "That's all from Gator. Keep it tight.",
                "Peace out. Call me if you need real help."
            ]
        }
        
        # Knowledge base for platform-specific help
        self.knowledge_base = {
            "personas": {
                "create": "To create a persona, go to the Personas tab and hit 'Create New Persona'. Fill out the details - and I mean ALL the details. Don't half-ass it.",
                "manage": "Your personas are listed in the Personas section. Click on any of them to edit. Keep your personas organized or you'll regret it later.",
                "content": "Each persona generates content based on their settings. The better you set them up, the better content you get. Garbage in, garbage out."
            },
            "content": {
                "generate": "Hit the 'Generate New Content' button in the Content tab. Make sure your personas are set up right first, or you'll waste time generating trash.",
                "manage": "All your generated content is in the Content tab. You can review, edit, or delete it there. Keep it organized.",
                "quality": "Content quality depends on your persona setup and the AI models. Don't expect miracles from bad inputs."
            },
            "dns": {
                "setup": "DNS setup is in the DNS Management tab. Enter your domain and server IP, then hit 'Setup Platform DNS'. Don't mess with DNS if you don't know what you're doing.",
                "records": "You can add, edit, or delete DNS records manually. But seriously, if you don't understand DNS, leave it to the auto-setup.",
                "godaddy": "GoDaddy integration requires your API keys. Go to Settings and enter them. Keep those credentials secure."
            },
            "troubleshooting": {
                "slow": "If things are running slow, check your system resources. This platform needs decent hardware to run smooth.",
                "errors": "Check the system status first. If there are errors, they'll show up there. Most problems fix themselves if you wait a minute.",
                "connection": "Connection problems? Check your internet, check your firewall, check your DNS. Basic networking, people."
            }
        }
    
    async def process_message(self, message: str, context: Optional[Dict] = None) -> str:
        """
        Process a user message and return Gator's response.
        
        Args:
            message: The user's input message
            context: Optional context about the current state/page
            
        Returns:
            Gator's response as a string
        """
        # Clean and normalize the message
        message = message.strip().lower()
        
        # Add to conversation history
        timestamp = datetime.now().isoformat()
        self.conversation_history.append({
            "timestamp": timestamp,
            "user_message": message,
            "context": context or {}
        })
        
        # Analyze the message and generate response
        response = await self._generate_response(message, context)
        
        # Add response to history
        self.conversation_history[-1]["gator_response"] = response
        
        return response
    
    async def _generate_response(self, message: str, context: Optional[Dict] = None) -> str:
        """Generate Gator's response based on message analysis."""
        
        # Greeting detection
        if any(word in message for word in ['hello', 'hi', 'hey', 'what\'s up', 'sup']):
            return random.choice(self.gator_responses["greeting"])
        
        # Goodbye detection
        if any(word in message for word in ['bye', 'goodbye', 'thanks', 'thank you', 'later']):
            return random.choice(self.gator_responses["goodbye"])
        
        # Help request detection
        if any(word in message for word in ['help', 'how', 'what', 'where', 'guide', 'tutorial']):
            return await self._handle_help_request(message, context)
        
        # Error/problem detection
        if any(word in message for word in ['error', 'problem', 'broken', 'not working', 'issue', 'bug']):
            return await self._handle_error_report(message, context)
        
        # Default response with some attitude
        gator_start = random.choice(self.gator_phrases)
        return f"{gator_start}, I'm not sure what you're asking about. Be more specific - what do you need help with? Personas? Content? DNS? System status? Give me something to work with here."
    
    async def _handle_help_request(self, message: str, context: Optional[Dict] = None) -> str:
        """Handle help requests with specific guidance."""
        
        # Persona-related help
        if any(word in message for word in ['persona', 'character', 'influencer', 'create']):
            topic = 'personas'
            if 'create' in message:
                help_text = self.knowledge_base[topic]['create']
            elif 'manage' in message:
                help_text = self.knowledge_base[topic]['manage'] 
            else:
                help_text = self.knowledge_base[topic]['content']
        
        # Content-related help
        elif any(word in message for word in ['content', 'generate', 'image', 'text', 'post']):
            topic = 'content'
            if 'generate' in message:
                help_text = self.knowledge_base[topic]['generate']
            elif 'manage' in message:
                help_text = self.knowledge_base[topic]['manage']
            else:
                help_text = self.knowledge_base[topic]['quality']
        
        # DNS-related help
        elif any(word in message for word in ['dns', 'domain', 'godaddy', 'setup', 'records']):
            topic = 'dns'
            if 'setup' in message:
                help_text = self.knowledge_base[topic]['setup']
            elif 'record' in message:
                help_text = self.knowledge_base[topic]['records']
            else:
                help_text = self.knowledge_base[topic]['godaddy']
        
        # General troubleshooting
        elif any(word in message for word in ['slow', 'error', 'connection', 'not working']):
            topic = 'troubleshooting'
            if 'slow' in message:
                help_text = self.knowledge_base[topic]['slow']
            elif 'connection' in message:
                help_text = self.knowledge_base[topic]['connection']
            else:
                help_text = self.knowledge_base[topic]['errors']
        
        else:
            # General help
            help_text = random.choice(self.gator_responses["help_general"])
        
        gator_start = random.choice(self.gator_phrases)
        return f"{gator_start}. {help_text}"
    
    async def _handle_error_report(self, message: str, context: Optional[Dict] = None) -> str:
        """Handle error reports and provide troubleshooting guidance."""
        
        gator_start = random.choice(self.gator_responses["error"])
        
        # Provide specific troubleshooting based on error type
        if 'dns' in message:
            advice = "Check your DNS settings and GoDaddy API credentials. Make sure your domain is properly configured."
        elif 'persona' in message:
            advice = "Persona problems usually mean bad input data. Check your persona settings and make sure all required fields are filled out."
        elif 'content' in message:
            advice = "Content generation issues? Check your AI model settings and make sure your personas are properly configured."
        elif 'slow' in message or 'loading' in message:
            advice = "Performance issues? Check your system resources - CPU, RAM, and disk space. This platform needs proper hardware."
        else:
            advice = "First, check the system status in the dashboard. Most problems show up there. If it's green, the problem might be on your end."
        
        return f"{gator_start} {advice} And next time, give me more details about what exactly went wrong."
    
    def get_conversation_history(self) -> List[Dict[str, str]]:
        """Get the conversation history."""
        return self.conversation_history.copy()
    
    def clear_conversation_history(self) -> None:
        """Clear the conversation history."""
        self.conversation_history.clear()
    
    def get_quick_help_topics(self) -> List[Dict[str, str]]:
        """Get a list of quick help topics for the UI."""
        return [
            {"topic": "Creating Personas", "message": "How do I create a new persona?"},
            {"topic": "Generating Content", "message": "How do I generate content?"},
            {"topic": "DNS Setup", "message": "Help me set up DNS"},
            {"topic": "System Status", "message": "How do I check system status?"},
            {"topic": "GoDaddy Integration", "message": "How do I configure GoDaddy?"},
            {"topic": "Troubleshooting", "message": "Something's not working right"}
        ]


# Global instance for the service
gator_agent = GatorAgentService()