"""
Proactive Topics Service

Generates proactive content topics and opinions for personas based on:
1. RSS feeds assigned to the persona
2. Persona's content themes and interests (fallback when no feeds assigned)

This service enables personas to proactively share opinions on topics
instead of only responding when users initiate conversations.
"""

import random
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from backend.config.logging import get_logger
from backend.models.persona import PersonaModel
from backend.models.feed import FeedItemModel, PersonaFeedModel, RSSFeedModel
from backend.services.rss_ingestion_service import RSSIngestionService
from backend.services.persona_chat_service import get_persona_chat_service

logger = get_logger(__name__)


class ProactiveTopicsService:
    """
    Service for generating proactive content topics and persona opinions.
    
    Features:
    - Auto-pulls topics from RSS feeds assigned to personas
    - Falls back to content_themes when no RSS feeds are assigned
    - Generates AI-powered persona opinions on topics
    - Maintains topic freshness with recency scoring
    """
    
    def __init__(self, db_session: AsyncSession):
        """
        Initialize the proactive topics service.
        
        Args:
            db_session: Database session for data access
        """
        self.db = db_session
        self.rss_service = RSSIngestionService(db_session)
    
    async def get_proactive_topics(
        self,
        persona_id: UUID,
        limit: int = 5,
        hours_window: int = 48,
    ) -> List[Dict[str, Any]]:
        """
        Get proactive topics for a persona to discuss.
        
        First tries to get topics from assigned RSS feeds. If no feeds are
        assigned or no recent items found, generates topics based on the
        persona's content_themes.
        
        Args:
            persona_id: UUID of the persona
            limit: Maximum number of topics to return
            hours_window: Time window for RSS items in hours
            
        Returns:
            List of topic dictionaries with title, summary, source, relevance_score
        """
        try:
            # Get the persona
            persona = await self._get_persona(persona_id)
            if not persona:
                logger.warning(f"Persona not found: {persona_id}")
                return []
            
            # Try to get topics from RSS feeds first
            rss_topics = await self._get_topics_from_rss(persona_id, limit, hours_window)
            
            if rss_topics:
                logger.info(f"Found {len(rss_topics)} RSS topics for persona {persona_id}")
                return rss_topics
            
            # Fallback: Generate topics based on persona's content themes
            logger.info(f"No RSS topics found, using content themes for persona {persona_id}")
            theme_topics = await self._generate_topics_from_themes(persona, limit)
            
            return theme_topics
            
        except Exception as e:
            logger.error(f"Error getting proactive topics for persona {persona_id}: {e}")
            return []
    
    async def generate_opinion(
        self,
        persona_id: UUID,
        topic: str,
        topic_summary: Optional[str] = None,
        use_ai: bool = True,
    ) -> Dict[str, Any]:
        """
        Generate a persona's opinion on a given topic.
        
        Uses the persona's personality, worldview, and content themes to
        generate an authentic, character-consistent opinion.
        
        Args:
            persona_id: UUID of the persona
            topic: The topic title or headline
            topic_summary: Optional additional context about the topic
            use_ai: Whether to use AI generation (True) or templates (False)
            
        Returns:
            Dict with opinion text, sentiment, and metadata
        """
        try:
            # Get the persona
            persona = await self._get_persona(persona_id)
            if not persona:
                return {
                    "success": False,
                    "error": f"Persona not found: {persona_id}"
                }
            
            # Generate opinion using chat service
            chat_service = get_persona_chat_service()
            
            # Build a prompt that asks for the persona's opinion
            topic_context = f"Topic: {topic}"
            if topic_summary:
                topic_context += f"\nContext: {topic_summary}"
            
            # Create a message asking for the persona's opinion
            prompt_variations = [
                f"What do you think about this? {topic_context}",
                f"I'd love to hear your thoughts on: {topic_context}",
                f"Have you seen this? {topic_context} What's your take?",
                f"This caught my attention: {topic_context} - thoughts?",
            ]
            
            user_message = random.choice(prompt_variations)
            
            # Generate the opinion using persona chat service
            opinion_text = await chat_service.generate_response(
                persona=persona,
                user_message=user_message,
                conversation_history=None,
                use_ai=use_ai,
            )
            
            # Analyze sentiment of the opinion
            sentiment = self._analyze_opinion_sentiment(opinion_text)
            
            return {
                "success": True,
                "opinion": opinion_text,
                "topic": topic,
                "topic_summary": topic_summary,
                "sentiment": sentiment,
                "persona_name": persona.name,
                "generated_at": datetime.now(timezone.utc).isoformat(),
            }
            
        except Exception as e:
            logger.error(f"Error generating opinion for persona {persona_id}: {e}")
            return {
                "success": False,
                "error": str(e),
                "topic": topic,
            }
    
    async def get_proactive_post_content(
        self,
        persona_id: UUID,
        post_style: Optional[str] = None,
        use_ai: bool = True,
    ) -> Dict[str, Any]:
        """
        Generate complete proactive post content including topic and opinion.
        
        This is a convenience method that combines topic selection and
        opinion generation into a single call, ready for social media posting.
        
        Args:
            persona_id: UUID of the persona
            post_style: Optional style override (casual, professional, etc.)
            use_ai: Whether to use AI generation
            
        Returns:
            Dict with topic, opinion, hashtags, and metadata
        """
        try:
            # Get topics
            topics = await self.get_proactive_topics(persona_id, limit=3)
            
            if not topics:
                return {
                    "success": False,
                    "error": "No topics available for this persona",
                }
            
            # Select the most relevant/recent topic
            selected_topic = topics[0]
            
            # Generate opinion on the topic
            opinion_result = await self.generate_opinion(
                persona_id=persona_id,
                topic=selected_topic.get("title", ""),
                topic_summary=selected_topic.get("summary"),
                use_ai=use_ai,
            )
            
            if not opinion_result.get("success"):
                return opinion_result
            
            # Get the persona for additional context
            persona = await self._get_persona(persona_id)
            
            # Generate relevant hashtags
            hashtags = self._generate_hashtags(
                topic=selected_topic,
                persona=persona,
            )
            
            return {
                "success": True,
                "topic": selected_topic,
                "opinion": opinion_result.get("opinion"),
                "sentiment": opinion_result.get("sentiment"),
                "hashtags": hashtags,
                "persona_name": persona.name if persona else "Unknown",
                "source": selected_topic.get("source"),
                "generated_at": datetime.now(timezone.utc).isoformat(),
            }
            
        except Exception as e:
            logger.error(f"Error generating proactive post for persona {persona_id}: {e}")
            return {
                "success": False,
                "error": str(e),
            }
    
    async def _get_persona(self, persona_id: UUID) -> Optional[PersonaModel]:
        """Get persona by ID."""
        try:
            stmt = select(PersonaModel).where(PersonaModel.id == persona_id)
            result = await self.db.execute(stmt)
            return result.scalar_one_or_none()
        except Exception as e:
            logger.error(f"Error fetching persona {persona_id}: {e}")
            return None
    
    async def _get_topics_from_rss(
        self,
        persona_id: UUID,
        limit: int,
        hours_window: int,
    ) -> List[Dict[str, Any]]:
        """
        Get topics from RSS feeds assigned to the persona.
        
        Args:
            persona_id: UUID of the persona
            limit: Maximum topics to return
            hours_window: Time window in hours for recent items
            
        Returns:
            List of topic dictionaries
        """
        try:
            # Get persona's assigned feeds
            stmt = (
                select(PersonaFeedModel, RSSFeedModel)
                .join(RSSFeedModel, PersonaFeedModel.feed_id == RSSFeedModel.id)
                .where(PersonaFeedModel.persona_id == persona_id)
                .where(PersonaFeedModel.is_active.is_(True))
            )
            result = await self.db.execute(stmt)
            persona_feeds = result.all()
            
            if not persona_feeds:
                logger.debug(f"No RSS feeds assigned to persona {persona_id}")
                return []
            
            # Collect feed IDs and topics
            feed_ids = [pf.feed_id for pf, _ in persona_feeds]
            filter_topics = []
            for pf, _ in persona_feeds:
                if pf.topics:
                    filter_topics.extend(pf.topics)
            
            # Get recent items from assigned feeds
            cutoff_time = datetime.now(timezone.utc) - timedelta(hours=hours_window)
            
            stmt = (
                select(FeedItemModel)
                .where(FeedItemModel.feed_id.in_(feed_ids))
                .where(FeedItemModel.created_at >= cutoff_time)
                .order_by(FeedItemModel.relevance_score.desc())
                .limit(limit * 3)  # Get more to filter
            )
            result = await self.db.execute(stmt)
            items = result.scalars().all()
            
            if not items:
                return []
            
            # Filter by topics if specified
            if filter_topics:
                filtered_items = []
                for item in items:
                    text = f"{item.title} {item.description or ''} {' '.join(item.categories or [])}".lower()
                    for topic in filter_topics:
                        if topic.lower() in text:
                            filtered_items.append(item)
                            break
                items = filtered_items or items  # Fall back to all items if no matches
            
            # Convert to topic dictionaries
            topics = []
            for item in items[:limit]:
                topics.append({
                    "title": item.title,
                    "summary": item.content_summary or item.description or "",
                    "source": "rss",
                    "source_url": item.link,
                    "published_date": item.published_date.isoformat() if item.published_date else None,
                    "relevance_score": item.relevance_score or 50,
                    "sentiment_score": item.sentiment_score or 0,
                    "categories": item.categories or [],
                    "keywords": item.keywords or [],
                })
            
            return topics
            
        except Exception as e:
            logger.error(f"Error getting RSS topics for persona {persona_id}: {e}")
            return []
    
    async def _generate_topics_from_themes(
        self,
        persona: PersonaModel,
        limit: int,
    ) -> List[Dict[str, Any]]:
        """
        Generate topics based on persona's content themes when no RSS feeds available.
        
        Args:
            persona: The persona model
            limit: Maximum topics to generate
            
        Returns:
            List of generated topic dictionaries
        """
        topics = []
        
        # Get content themes
        themes = persona.content_themes or []
        if not themes:
            # Fall back to extracting themes from personality
            themes = self._extract_themes_from_personality(persona.personality or "")
        
        if not themes:
            # Ultimate fallback: generic themes
            themes = ["daily life", "thoughts", "observations"]
        
        # Generate topic prompts for each theme
        topic_templates = [
            "{theme} - What's happening lately",
            "My thoughts on {theme}",
            "Something interesting about {theme}",
            "Can we talk about {theme}?",
            "{theme} update",
        ]
        
        # Create topics from themes
        selected_themes = random.sample(themes, min(len(themes), limit))
        
        for theme in selected_themes:
            template = random.choice(topic_templates)
            topics.append({
                "title": template.format(theme=theme.capitalize()),
                "summary": f"Discussion about {theme} based on personal interests",
                "source": "interests",
                "source_url": None,
                "published_date": datetime.now(timezone.utc).isoformat(),
                "relevance_score": 75,  # Default relevance for interest-based topics
                "sentiment_score": 0,
                "categories": [theme],
                "keywords": [theme],
            })
        
        return topics
    
    def _extract_themes_from_personality(self, personality: str) -> List[str]:
        """Extract potential themes from personality description."""
        # Common interest-related keywords
        interest_indicators = [
            "passionate about",
            "loves",
            "interested in",
            "enjoys",
            "enthusiastic about",
            "focuses on",
            "expert in",
            "known for",
        ]
        
        themes = []
        personality_lower = personality.lower()
        
        for indicator in interest_indicators:
            if indicator in personality_lower:
                # Try to extract what follows the indicator
                idx = personality_lower.find(indicator)
                end_idx = personality_lower.find(".", idx)
                if end_idx == -1:
                    end_idx = min(idx + 50, len(personality_lower))
                
                extract = personality[idx + len(indicator):end_idx].strip()
                if extract and len(extract) < 50:
                    themes.append(extract.strip(" ,."))
        
        return themes[:5]
    
    def _analyze_opinion_sentiment(self, opinion: str) -> str:
        """Analyze the sentiment of an opinion text."""
        opinion_lower = opinion.lower()
        
        positive_indicators = [
            "love", "great", "amazing", "awesome", "exciting",
            "wonderful", "fantastic", "brilliant", "excellent",
            "happy", "glad", "thrilled", "impressed",
        ]
        
        negative_indicators = [
            "hate", "terrible", "awful", "bad", "disappointing",
            "sad", "annoyed", "frustrated", "angry", "worried",
            "concerned", "skeptical", "doubtful",
        ]
        
        positive_count = sum(1 for word in positive_indicators if word in opinion_lower)
        negative_count = sum(1 for word in negative_indicators if word in opinion_lower)
        
        if positive_count > negative_count:
            return "positive"
        elif negative_count > positive_count:
            return "negative"
        else:
            return "neutral"
    
    def _generate_hashtags(
        self,
        topic: Dict[str, Any],
        persona: Optional[PersonaModel],
    ) -> List[str]:
        """Generate relevant hashtags for a topic."""
        hashtags = []
        
        # Add hashtags from topic categories
        for category in topic.get("categories", [])[:3]:
            clean_cat = category.replace(" ", "").replace("-", "")
            if clean_cat:
                hashtags.append(f"#{clean_cat}")
        
        # Add hashtags from keywords
        for keyword in topic.get("keywords", [])[:2]:
            clean_kw = keyword.replace(" ", "").replace("-", "")
            if clean_kw and f"#{clean_kw}" not in hashtags:
                hashtags.append(f"#{clean_kw}")
        
        # Add persona-specific hashtags from content themes
        if persona and persona.content_themes:
            for theme in persona.content_themes[:2]:
                clean_theme = theme.replace(" ", "").replace("-", "")
                tag = f"#{clean_theme}"
                if tag not in hashtags:
                    hashtags.append(tag)
        
        return hashtags[:5]  # Limit to 5 hashtags


# Global instance
_proactive_topics_service: Optional[ProactiveTopicsService] = None


def get_proactive_topics_service(db_session: AsyncSession) -> ProactiveTopicsService:
    """Get or create ProactiveTopicsService instance."""
    return ProactiveTopicsService(db_session)
