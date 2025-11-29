"""
Proactive Topics Service

Generates proactive content topics and opinions for personas based on:
1. RSS feeds assigned to the persona
2. Persona's content themes and interests (fallback when no feeds assigned)

This service enables personas to proactively share opinions on topics
instead of only responding when users initiate conversations.
"""

import random
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional
from uuid import UUID

from sqlalchemy import func as sql_func
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.config.logging import get_logger
from backend.models.feed import FeedItemModel, PersonaFeedModel, RSSFeedModel
from backend.models.persona import PersonaModel
from backend.models.social_media_post import SocialMediaPostModel
from backend.services.persona_chat_service import get_persona_chat_service
from backend.services.rss_ingestion_service import RSSIngestionService

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
        cooldown_hours: int = 24,
    ) -> List[Dict[str, Any]]:
        """
        Get proactive topics for a persona to discuss.

        First tries to get topics from assigned RSS feeds. If no feeds are
        assigned or no recent items found, generates topics based on the
        persona's content_themes.

        Includes cooldown filtering to prevent repeating recently discussed topics
        and diversity scoring to provide varied content.

        Args:
            persona_id: UUID of the persona
            limit: Maximum number of topics to return
            hours_window: Time window for RSS items in hours
            cooldown_hours: Hours to wait before reusing a topic

        Returns:
            List of topic dictionaries with title, summary, source, relevance_score
        """
        try:
            # Get the persona
            persona = await self._get_persona(persona_id)
            if not persona:
                logger.warning(f"Persona not found: {persona_id}")
                return []

            # Get recently used topics for cooldown filtering
            recent_topics = await self._get_recent_topic_keywords(
                persona_id, cooldown_hours
            )

            # Try to get topics from RSS feeds first
            rss_topics = await self._get_topics_from_rss(
                persona_id, limit * 2, hours_window
            )

            if rss_topics:
                # Filter out topics in cooldown and apply diversity scoring
                filtered_topics = await self._filter_and_diversify_topics(
                    rss_topics, recent_topics, limit
                )
                if filtered_topics:
                    logger.info(
                        f"Found {len(filtered_topics)} RSS topics for persona {persona_id}"
                    )
                    return filtered_topics

            # Fallback: Generate topics based on persona's content themes
            logger.info(
                f"No RSS topics found, using content themes for persona {persona_id}"
            )
            theme_topics = await self._generate_topics_from_themes(persona, limit * 2)

            # Apply diversity filtering to theme topics too
            filtered_themes = await self._filter_and_diversify_topics(
                theme_topics, recent_topics, limit
            )

            return filtered_themes

        except Exception as e:
            logger.error(
                f"Error getting proactive topics for persona {persona_id}: {e}"
            )
            return []

    async def check_topic_cooldown(
        self,
        persona_id: UUID,
        topic_title: str,
        cooldown_hours: int = 24,
    ) -> bool:
        """
        Check if a topic is in cooldown period (recently posted about).

        Args:
            persona_id: UUID of the persona
            topic_title: The topic title to check
            cooldown_hours: Hours to look back for recent posts

        Returns:
            True if topic can be used (not in cooldown), False otherwise
        """
        try:
            cutoff = datetime.now(timezone.utc) - timedelta(hours=cooldown_hours)

            # Escape SQL wildcards in topic_title to prevent injection
            safe_topic = topic_title[:50].replace("%", r"\%").replace("_", r"\_")

            # Check for recent posts with similar topic in caption
            stmt = (
                select(sql_func.count(SocialMediaPostModel.id))
                .where(SocialMediaPostModel.persona_id == persona_id)
                .where(SocialMediaPostModel.created_at >= cutoff)
                .where(SocialMediaPostModel.caption.ilike(f"%{safe_topic}%"))
            )
            result = await self.db.execute(stmt)
            recent_count = result.scalar() or 0

            return recent_count == 0  # True if no recent posts about this topic

        except Exception as e:
            logger.warning(f"Error checking topic cooldown: {e}")
            return True  # Default to allowing the topic if check fails

    async def _get_recent_topic_keywords(
        self,
        persona_id: UUID,
        hours_lookback: int = 24,
    ) -> set:
        """
        Get keywords from recent posts for diversity tracking.

        Args:
            persona_id: UUID of the persona
            hours_lookback: Hours to look back

        Returns:
            Set of keywords from recent posts
        """
        try:
            cutoff = datetime.now(timezone.utc) - timedelta(hours=hours_lookback)

            stmt = (
                select(SocialMediaPostModel.caption)
                .where(SocialMediaPostModel.persona_id == persona_id)
                .where(SocialMediaPostModel.created_at >= cutoff)
                .order_by(SocialMediaPostModel.created_at.desc())
                .limit(10)
            )
            result = await self.db.execute(stmt)
            recent_captions = result.scalars().all()

            # Extract keywords from captions
            keywords = set()
            for caption in recent_captions:
                if caption:
                    # Simple keyword extraction - strip punctuation first, then check length
                    words = caption.lower().split()
                    for word in words:
                        clean_word = word.strip(".,!?\";:-'()[]{}<>")
                        if len(clean_word) > 3:  # Check length after stripping
                            keywords.add(clean_word)

            return keywords

        except Exception as e:
            logger.warning(f"Error getting recent topic keywords: {e}")
            return set()

    async def _filter_and_diversify_topics(
        self,
        topics: List[Dict[str, Any]],
        recent_keywords: set,
        limit: int,
    ) -> List[Dict[str, Any]]:
        """
        Filter topics by cooldown and score by diversity.

        Topics that overlap heavily with recent content are deprioritized.

        Args:
            topics: List of candidate topics
            recent_keywords: Keywords from recent posts
            limit: Maximum topics to return

        Returns:
            Filtered and diversified topic list
        """
        if not topics:
            return []

        scored_topics = []
        for topic in topics:
            title = topic.get("title", "").lower()
            summary = topic.get("summary", "").lower()
            topic_text = f"{title} {summary}"
            topic_words = set(word for word in topic_text.split() if len(word) > 3)

            # Calculate overlap with recent content
            overlap = len(topic_words & recent_keywords)
            total_words = len(topic_words) or 1
            overlap_ratio = overlap / total_words

            # Diversity score: higher = more novel
            diversity_score = 1.0 - min(overlap_ratio, 1.0)

            # Combine with relevance score
            relevance = topic.get("relevance_score", 50) / 100
            combined_score = (diversity_score * 0.4) + (relevance * 0.6)

            # Skip if too much overlap (>70% overlap with recent content)
            if overlap_ratio < 0.7:
                scored_topics.append((combined_score, topic))

        # Sort by combined score
        scored_topics.sort(key=lambda x: x[0], reverse=True)

        return [topic for _, topic in scored_topics[:limit]]

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
                return {"success": False, "error": f"Persona not found: {persona_id}"}

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
            logger.error(
                f"Error generating proactive post for persona {persona_id}: {e}"
            )
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
                topics.append(
                    {
                        "title": item.title,
                        "summary": item.content_summary or item.description or "",
                        "source": "rss",
                        "source_url": item.link,
                        "published_date": (
                            item.published_date.isoformat()
                            if item.published_date
                            else None
                        ),
                        "relevance_score": item.relevance_score or 50,
                        "sentiment_score": item.sentiment_score or 0,
                        "categories": item.categories or [],
                        "keywords": item.keywords or [],
                    }
                )

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
            topics.append(
                {
                    "title": template.format(theme=theme.capitalize()),
                    "summary": f"Discussion about {theme} based on personal interests",
                    "source": "interests",
                    "source_url": None,
                    "published_date": datetime.now(timezone.utc).isoformat(),
                    "relevance_score": 75,  # Default relevance for interest-based topics
                    "sentiment_score": 0,
                    "categories": [theme],
                    "keywords": [theme],
                }
            )

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

    def _analyze_opinion_sentiment(self, opinion: str) -> Dict[str, Any]:
        """
        Analyze the sentiment of an opinion text using VADER sentiment analysis.

        Returns detailed sentiment information including compound score,
        confidence, and categorical sentiment.

        Args:
            opinion: The opinion text to analyze

        Returns:
            Dict with sentiment category, confidence, and raw scores
        """
        try:
            from nltk.sentiment.vader import SentimentIntensityAnalyzer

            analyzer = SentimentIntensityAnalyzer()
            scores = analyzer.polarity_scores(opinion)

            # Get compound score (-1 to 1)
            compound = scores["compound"]

            # Determine categorical sentiment with thresholds
            if compound >= 0.05:
                sentiment = "positive"
            elif compound <= -0.05:
                sentiment = "negative"
            else:
                sentiment = "neutral"

            # Calculate confidence based on how decisive the score is
            confidence = abs(compound)

            return {
                "sentiment": sentiment,
                "confidence": round(confidence, 3),
                "compound_score": round(compound, 3),
                "positive_score": round(scores["pos"], 3),
                "negative_score": round(scores["neg"], 3),
                "neutral_score": round(scores["neu"], 3),
            }

        except ImportError:
            logger.warning(
                "NLTK VADER not available, using fallback sentiment analysis"
            )
            return self._fallback_sentiment_analysis(opinion)
        except Exception as e:
            logger.warning(f"VADER sentiment analysis failed: {e}, using fallback")
            return self._fallback_sentiment_analysis(opinion)

    def _fallback_sentiment_analysis(self, opinion: str) -> Dict[str, Any]:
        """Fallback keyword-based sentiment analysis."""
        opinion_lower = opinion.lower()

        positive_indicators = [
            "love",
            "great",
            "amazing",
            "awesome",
            "exciting",
            "wonderful",
            "fantastic",
            "brilliant",
            "excellent",
            "happy",
            "glad",
            "thrilled",
            "impressed",
            "best",
            "incredible",
            "outstanding",
            "perfect",
            "beautiful",
        ]

        negative_indicators = [
            "hate",
            "terrible",
            "awful",
            "bad",
            "disappointing",
            "sad",
            "annoyed",
            "frustrated",
            "angry",
            "worried",
            "concerned",
            "skeptical",
            "doubtful",
            "worst",
            "horrible",
            "disgusting",
            "pathetic",
            "useless",
        ]

        positive_count = sum(1 for word in positive_indicators if word in opinion_lower)
        negative_count = sum(1 for word in negative_indicators if word in opinion_lower)

        total = positive_count + negative_count
        if total == 0:
            return {
                "sentiment": "neutral",
                "confidence": 0.0,
                "compound_score": 0.0,
            }

        compound = (positive_count - negative_count) / total

        if positive_count > negative_count:
            sentiment = "positive"
        elif negative_count > positive_count:
            sentiment = "negative"
        else:
            sentiment = "neutral"

        return {
            "sentiment": sentiment,
            "confidence": round(abs(compound), 3),
            "compound_score": round(compound, 3),
        }

    def _generate_hashtags(
        self,
        topic: Dict[str, Any],
        persona: Optional[PersonaModel],
        platform: str = "instagram",
    ) -> List[str]:
        """
        Generate relevant, validated hashtags for a topic.

        Platform-aware hashtag generation with validation.

        Args:
            topic: Topic dictionary with categories and keywords
            persona: Optional persona model for personalized hashtags
            platform: Target platform (instagram, twitter, tiktok, etc.)

        Returns:
            List of validated hashtags appropriate for the platform
        """
        hashtags = []

        # Platform-specific limits
        max_hashtags = {
            "instagram": 15,  # Instagram allows 30, but 15-20 is optimal
            "twitter": 3,  # Twitter recommends 2-3
            "tiktok": 5,  # TikTok works well with fewer
            "linkedin": 5,
            "facebook": 5,
        }.get(platform.lower(), 5)

        def normalize_hashtag_text(text: str) -> str:
            """Normalize text for hashtag use (shared logic)."""
            clean = text.strip().replace(" ", "").replace("-", "")
            return "".join(c for c in clean if c.isalnum() or c == "_")

        def is_valid_hashtag(tag: str) -> bool:
            """Validate hashtag format and length."""
            clean = normalize_hashtag_text(tag.strip("#"))
            return (
                len(clean) >= 3 and len(clean) <= 30  # Minimum 3 chars  # Max 30 chars
            )

        def clean_hashtag(text: str) -> str:
            """Clean text into valid hashtag format."""
            clean = normalize_hashtag_text(text)
            return f"#{clean}" if clean else ""

        # Add hashtags from topic categories
        for category in topic.get("categories", [])[:5]:
            tag = clean_hashtag(category)
            if tag and is_valid_hashtag(tag) and tag not in hashtags:
                hashtags.append(tag)

        # Add hashtags from keywords
        for keyword in topic.get("keywords", [])[:3]:
            tag = clean_hashtag(keyword)
            if tag and is_valid_hashtag(tag) and tag not in hashtags:
                hashtags.append(tag)

        # Add persona-specific hashtags from content themes
        if persona and persona.content_themes:
            for theme in persona.content_themes[:3]:
                tag = clean_hashtag(theme)
                if tag and is_valid_hashtag(tag) and tag not in hashtags:
                    hashtags.append(tag)

        # Validate all hashtags before returning
        validated = [tag for tag in hashtags if is_valid_hashtag(tag)]

        return validated[:max_hashtags]


# Global instance
_proactive_topics_service: Optional[ProactiveTopicsService] = None


def get_proactive_topics_service(db_session: AsyncSession) -> ProactiveTopicsService:
    """Get or create ProactiveTopicsService instance."""
    return ProactiveTopicsService(db_session)
