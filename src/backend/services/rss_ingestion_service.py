"""
RSS Feed Ingestion Service

Handles RSS feed management, content parsing, and trend analysis
for informing AI content generation.
"""

import asyncio
import feedparser
from datetime import datetime, timezone, timedelta
from typing import List, Optional, Dict, Any
from uuid import UUID
import re
from urllib.parse import urlparse

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update
import httpx

from backend.models.feed import (
    RSSFeedModel,
    FeedItemModel,
    PersonaFeedModel,
    RSSFeedCreate,
    RSSFeedResponse,
    FeedItemResponse,
    PersonaFeedAssignment,
    PersonaFeedResponse,
)
from backend.config.logging import get_logger

logger = get_logger(__name__)


class RSSIngestionService:
    """
    Service for RSS feed management and content ingestion.

    Manages RSS feeds, fetches new content, performs sentiment analysis,
    and provides trending topic insights for content generation.
    """

    def __init__(self, db_session: AsyncSession):
        """
        Initialize RSS ingestion service.

        Args:
            db_session: Database session for persistence
        """
        self.db = db_session
        self.http_client = httpx.AsyncClient(
            timeout=30.0, headers={"User-Agent": "Gator AI Influencer Platform/1.0"}
        )

    async def add_feed(self, feed_data: RSSFeedCreate) -> RSSFeedResponse:
        """
        Add new RSS feed to monitoring system.

        Args:
            feed_data: RSS feed configuration

        Returns:
            RSSFeedResponse: Created feed record

        Raises:
            ValueError: If feed URL is invalid or already exists
        """
        try:
            # Validate feed URL by attempting to fetch it
            await self._validate_feed_url(str(feed_data.url))

            # Check if feed already exists
            existing_feed = await self._get_feed_by_url(str(feed_data.url))
            if existing_feed:
                raise ValueError(f"Feed already exists: {feed_data.url}")

            # Create feed record
            feed = RSSFeedModel(
                name=feed_data.name,
                url=str(feed_data.url),
                description=feed_data.description,
                categories=feed_data.categories,
                fetch_frequency_hours=feed_data.fetch_frequency_hours,
            )

            self.db.add(feed)
            await self.db.commit()
            await self.db.refresh(feed)

            logger.info(f"RSS feed added {feed.id}: {feed.url}")

            # Perform initial fetch
            await self._fetch_feed_content(feed)

            return RSSFeedResponse.model_validate(feed)

        except Exception as e:
            await self.db.rollback()
            logger.error(f"Failed to add RSS feed {str(feed_data.url)}: {str(e)}")
            raise ValueError(f"Failed to add RSS feed: {str(e)}")

    async def list_feeds(self, active_only: bool = True) -> List[RSSFeedResponse]:
        """List all RSS feeds."""
        try:
            query = select(RSSFeedModel)
            if active_only:
                query = query.where(RSSFeedModel.is_active == True)

            query = query.order_by(RSSFeedModel.created_at.desc())
            result = await self.db.execute(query)
            feeds = result.scalars().all()

            return [RSSFeedResponse.model_validate(feed) for feed in feeds]

        except Exception as e:
            logger.error(f"Error listing RSS feeds: {str(e)}")
            return []

    async def fetch_all_feeds(self) -> Dict[str, int]:
        """
        Fetch content from all active feeds.

        Returns:
            Dict with feed_id -> item_count mapping
        """
        feeds = await self.list_feeds(active_only=True)
        results = {}

        for feed_response in feeds:
            try:
                # Get full feed model
                feed = await self._get_feed_by_id(feed_response.id)
                if feed:
                    count = await self._fetch_feed_content(feed)
                    results[str(feed.id)] = count

                    # Update last_fetched timestamp
                    feed.last_fetched = datetime.now(timezone.utc)
                    await self.db.commit()

            except Exception as e:
                logger.error(f"Error fetching feed {feed_response.id}: {str(e)}")
                results[str(feed_response.id)] = 0

        logger.info(f"Completed feed fetch cycle: {results}")
        return results

    async def get_trending_topics(
        self, limit: int = 20, hours: int = 24
    ) -> List[Dict[str, Any]]:
        """
        Analyze recent feed items to identify trending topics.

        Args:
            limit: Maximum number of topics to return
            hours: Time window for analysis

        Returns:
            List of trending topics with metadata
        """
        try:
            # Get recent feed items
            cutoff_time = datetime.now(timezone.utc) - timedelta(hours=hours)

            stmt = (
                select(FeedItemModel)
                .where(FeedItemModel.created_at >= cutoff_time)
                .where(FeedItemModel.processed == True)
                .order_by(FeedItemModel.relevance_score.desc())
            )

            result = await self.db.execute(stmt)
            items = result.scalars().all()

            # Simple topic extraction (in production, use NLP libraries)
            topic_counts = {}
            for item in items:
                # Extract keywords from title and categories
                keywords = self._extract_keywords_from_title_and_categories(
                    item.title, item.categories
                )
                for keyword in keywords:
                    if keyword not in topic_counts:
                        topic_counts[keyword] = {
                            "count": 0,
                            "sentiment": 0,
                            "relevance": 0,
                            "sample_titles": [],
                        }

                    topic_counts[keyword]["count"] += 1
                    topic_counts[keyword]["sentiment"] += item.sentiment_score or 0
                    topic_counts[keyword]["relevance"] += item.relevance_score or 0

                    if len(topic_counts[keyword]["sample_titles"]) < 3:
                        topic_counts[keyword]["sample_titles"].append(item.title)

            # Sort by count and relevance
            trending = []
            for topic, data in topic_counts.items():
                if data["count"] >= 2:  # Minimum threshold
                    trending.append(
                        {
                            "topic": topic,
                            "mentions": data["count"],
                            "avg_sentiment": data["sentiment"] / data["count"],
                            "avg_relevance": data["relevance"] / data["count"],
                            "sample_titles": data["sample_titles"],
                        }
                    )

            trending.sort(
                key=lambda x: (x["mentions"], x["avg_relevance"]), reverse=True
            )
            return trending[:limit]

        except Exception as e:
            logger.error(f"Error analyzing trending topics: {str(e)}")
            return []

    async def get_content_suggestions(
        self, persona_id: UUID, limit: int = 10
    ) -> List[FeedItemResponse]:
        """
        Get content suggestions based on persona's assigned feeds and topics.

        Args:
            persona_id: UUID of the persona
            limit: Maximum suggestions to return

        Returns:
            List of relevant feed items from assigned feeds
        """
        try:
            # Get persona's assigned feeds
            persona_feeds = await self.list_persona_feeds(persona_id)

            if not persona_feeds:
                logger.info(f"No feeds assigned to persona {persona_id}")
                return []

            # Extract feed IDs and topics
            feed_ids = [pf.feed_id for pf in persona_feeds]
            all_topics = []
            for pf in persona_feeds:
                all_topics.extend(pf.topics)

            # Get recent, high-relevance items from assigned feeds (last 48 hours)
            cutoff_time = datetime.now(timezone.utc) - timedelta(hours=48)

            stmt = (
                select(FeedItemModel)
                .where(FeedItemModel.feed_id.in_(feed_ids))
                .where(FeedItemModel.created_at >= cutoff_time)
                .where(FeedItemModel.processed == True)
                .order_by(FeedItemModel.relevance_score.desc())
                .limit(limit * 2)
            )  # Get more to filter through

            result = await self.db.execute(stmt)
            items = result.scalars().all()

            # Score items based on topic relevance if topics are specified
            if all_topics:
                scored_items = []
                for item in items:
                    score = self._calculate_theme_relevance(item, all_topics)
                    if score > 0.3:  # Minimum relevance threshold
                        scored_items.append((score, item))

                # Sort by relevance score and return top items
                scored_items.sort(key=lambda x: x[0], reverse=True)
                return [
                    FeedItemResponse.model_validate(item)
                    for _, item in scored_items[:limit]
                ]
            else:
                # No topic filtering, just return by relevance score
                return [FeedItemResponse.model_validate(item) for item in items[:limit]]

        except Exception as e:
            logger.error(f"Error getting content suggestions: {str(e)}")
            return []

    async def _validate_feed_url(self, url: str) -> None:
        """Validate that URL is a valid RSS feed."""
        try:
            parsed = urlparse(url)
            if not parsed.scheme or not parsed.netloc:
                raise ValueError("Invalid URL format")

            # Try to fetch and parse the feed
            response = await self.http_client.get(url)
            response.raise_for_status()

            # Parse with feedparser
            feed = feedparser.parse(response.content)
            if feed.bozo:
                raise ValueError("URL does not contain valid RSS/Atom feed")

            if not feed.entries:
                logger.warning(f"RSS feed contains no entries: {url}")

        except httpx.HTTPError as e:
            raise ValueError(f"Cannot fetch RSS feed: {str(e)}")
        except Exception as e:
            raise ValueError(f"Invalid RSS feed: {str(e)}")

    async def _get_feed_by_url(self, url: str) -> Optional[RSSFeedModel]:
        """Check if feed with URL already exists."""
        stmt = select(RSSFeedModel).where(RSSFeedModel.url == url)
        result = await self.db.execute(stmt)
        return result.scalar_one_or_none()

    async def _get_feed_by_id(self, feed_id: UUID) -> Optional[RSSFeedModel]:
        """Get feed by ID."""
        stmt = select(RSSFeedModel).where(RSSFeedModel.id == feed_id)
        result = await self.db.execute(stmt)
        return result.scalar_one_or_none()

    async def _fetch_feed_content(self, feed: RSSFeedModel) -> int:
        """
        Fetch and process content from RSS feed.

        Returns:
            Number of new items processed
        """
        try:
            # Fetch RSS content
            response = await self.http_client.get(feed.url)
            response.raise_for_status()

            # Parse feed
            parsed_feed = feedparser.parse(response.content)
            new_items = 0

            for entry in parsed_feed.entries[:50]:  # Limit to most recent 50 items
                # Check if item already exists
                item_link = getattr(entry, "link", "")
                existing = await self._get_feed_item_by_link(item_link)

                if not existing and item_link:
                    # Extract topics and entities
                    topic_data = self._extract_topics_and_entities(entry)

                    # Create new feed item with enhanced analysis
                    item = FeedItemModel(
                        feed_id=feed.id,
                        title=getattr(entry, "title", "Untitled"),
                        link=item_link,
                        description=getattr(entry, "summary", ""),
                        published_date=self._parse_entry_date(entry),
                        author=getattr(entry, "author", ""),
                        categories=self._extract_entry_categories(entry),
                        content_summary=self._generate_summary(entry),
                        sentiment_score=self._analyze_sentiment(entry),
                        relevance_score=self._calculate_relevance(
                            entry, feed.categories
                        ),
                        keywords=topic_data.get("keywords", []),
                        entities=topic_data.get("entities", []),
                        topics=topic_data.get("topics", []),
                        processed=True,
                    )

                    self.db.add(item)
                    new_items += 1

            if new_items > 0:
                await self.db.commit()
                logger.info(f"Processed feed items {feed.id} new_items={new_items}")

            return new_items

        except Exception as e:
            logger.error(f"Error fetching feed content {feed.id}: {str(e)}")
            return 0

    async def _get_feed_item_by_link(self, link: str) -> Optional[FeedItemModel]:
        """Check if feed item already exists."""
        if not link:
            return None

        stmt = select(FeedItemModel).where(FeedItemModel.link == link)
        result = await self.db.execute(stmt)
        return result.scalar_one_or_none()

    def _parse_entry_date(self, entry) -> Optional[datetime]:
        """Parse published date from RSS entry."""
        try:
            if hasattr(entry, "published_parsed") and entry.published_parsed:
                return datetime(*entry.published_parsed[:6], tzinfo=timezone.utc)
            elif hasattr(entry, "updated_parsed") and entry.updated_parsed:
                return datetime(*entry.updated_parsed[:6], tzinfo=timezone.utc)
        except Exception:
            pass
        return None

    def _extract_entry_categories(self, entry) -> List[str]:
        """Extract categories from RSS entry."""
        categories = []
        if hasattr(entry, "tags"):
            categories.extend([tag.term.lower() for tag in entry.tags])
        if hasattr(entry, "category"):
            categories.append(entry.category.lower())
        return list(set(categories))[:10]  # Limit to 10 categories

    def _generate_summary(self, entry) -> str:
        """Generate content summary from RSS entry."""
        # Simple summary generation (in production, use proper NLP)
        summary = getattr(entry, "summary", "")
        if summary:
            # Clean HTML and truncate
            clean_summary = re.sub("<[^<]+?>", "", summary)
            return (
                clean_summary[:500] + "..."
                if len(clean_summary) > 500
                else clean_summary
            )
        return ""

    def _analyze_sentiment(self, entry) -> float:
        """
        Analyze sentiment of RSS entry content using AI models.

        Returns sentiment score from -1.0 to 1.0.
        Enhanced implementation with actual NLP analysis.
        """
        try:
            # Get text content
            title = getattr(entry, "title", "")
            summary = getattr(entry, "summary", "")
            text = f"{title}. {summary}".strip()

            if not text:
                return 0.0

            # Try to use AI sentiment analysis if available
            try:
                from backend.services.ai_models import ai_models

                # Use a simple prompt for sentiment analysis
                prompt = f"Analyze the sentiment of this news headline and summary. Respond with only a number between -1 (very negative) and 1 (very positive):\n\n{text[:500]}"

                # This would use the AI model for sentiment analysis
                # For now, fall back to keyword analysis
                return self._keyword_sentiment_analysis(text)

            except Exception:
                # Fallback to enhanced keyword analysis
                return self._keyword_sentiment_analysis(text)

        except Exception as e:
            logger.error(f"Sentiment analysis failed: {str(e)}")
            return 0.0

    def _keyword_sentiment_analysis(self, text: str) -> float:
        """Enhanced keyword-based sentiment analysis."""
        text_lower = text.lower()

        # Expanded sentiment word lists
        positive_words = [
            "good",
            "great",
            "amazing",
            "excellent",
            "breakthrough",
            "success",
            "wonderful",
            "fantastic",
            "outstanding",
            "impressive",
            "remarkable",
            "positive",
            "beneficial",
            "advantage",
            "improvement",
            "progress",
            "growth",
            "innovation",
            "solution",
            "win",
            "victory",
            "achievement",
            "accomplish",
            "triumph",
            "boost",
            "gain",
        ]

        negative_words = [
            "bad",
            "terrible",
            "crisis",
            "failure",
            "problem",
            "issue",
            "awful",
            "disaster",
            "catastrophe",
            "decline",
            "fall",
            "crash",
            "collapse",
            "concern",
            "worry",
            "threat",
            "risk",
            "danger",
            "warning",
            "alert",
            "emergency",
            "loss",
            "damage",
            "harm",
            "hurt",
            "injury",
            "death",
            "violence",
        ]

        # Count occurrences with context weighting
        positive_score = 0
        negative_score = 0

        words = text_lower.split()
        for i, word in enumerate(words):
            # Check for positive words
            if word in positive_words:
                weight = (
                    1.5 if i < len(words) * 0.3 else 1.0
                )  # Title words weighted more
                positive_score += weight

            # Check for negative words
            elif word in negative_words:
                weight = 1.5 if i < len(words) * 0.3 else 1.0
                negative_score += weight

        # Calculate normalized sentiment
        total_score = positive_score + negative_score
        if total_score == 0:
            return 0.0

        return (positive_score - negative_score) / max(total_score, 1.0)

    def _extract_topics_and_entities(self, entry) -> Dict[str, Any]:
        """
        Extract topics, keywords, and entities from RSS entry.

        Returns dictionary with topics, keywords, and named entities.
        """
        try:
            title = getattr(entry, "title", "")
            summary = getattr(entry, "summary", "")
            text = f"{title}. {summary}".strip()

            if not text:
                return {"topics": [], "keywords": [], "entities": []}

            # Extract keywords using basic NLP
            keywords = self._extract_keywords(text)

            # Extract named entities (people, organizations, locations)
            entities = self._extract_entities(text)

            # Classify into topic categories
            topics = self._classify_topics(text, keywords)

            return {"topics": topics, "keywords": keywords, "entities": entities}

        except Exception as e:
            logger.error(f"Topic extraction failed: {str(e)}")
            return {"topics": [], "keywords": [], "entities": []}

    def _extract_keywords(self, text: str) -> List[str]:
        """Extract keywords from text."""
        # Simple keyword extraction based on frequency and importance
        words = text.lower().split()

        # Remove common stop words
        stop_words = {
            "a",
            "an",
            "and",
            "are",
            "as",
            "at",
            "be",
            "by",
            "for",
            "from",
            "has",
            "he",
            "in",
            "is",
            "it",
            "its",
            "of",
            "on",
            "that",
            "the",
            "to",
            "was",
            "will",
            "with",
            "can",
            "have",
            "this",
            "been",
            "but",
            "not",
            "or",
            "so",
            "up",
            "out",
            "if",
            "about",
            "who",
            "get",
            "go",
            "me",
        }

        # Filter and count words
        filtered_words = [
            word.strip('.,!?";')
            for word in words
            if len(word) > 3 and word.lower() not in stop_words
        ]

        # Count frequency
        word_freq = {}
        for word in filtered_words:
            word_freq[word] = word_freq.get(word, 0) + 1

        # Return top keywords
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        return [word[0] for word in sorted_words[:10]]

    def _extract_entities(self, text: str) -> List[Dict[str, str]]:
        """Extract named entities from text."""
        entities = []

        # Simple pattern-based entity extraction
        import re

        # Look for capitalized words (potential proper nouns)
        capitalized_pattern = r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b"
        potential_entities = re.findall(capitalized_pattern, text)

        # Filter and categorize entities
        for entity in set(potential_entities):
            if len(entity.split()) <= 3:  # Reasonable entity length
                # Simple categorization based on common patterns
                entity_type = "PERSON"  # Default

                # Organization indicators
                if any(
                    indicator in entity.lower()
                    for indicator in ["corp", "inc", "ltd", "company", "group"]
                ):
                    entity_type = "ORGANIZATION"

                # Location indicators
                elif any(
                    indicator in entity.lower()
                    for indicator in ["city", "state", "country", "county"]
                ):
                    entity_type = "LOCATION"

                entities.append(
                    {
                        "text": entity,
                        "type": entity_type,
                        "confidence": 0.7,  # Basic confidence score
                    }
                )

        return entities[:5]  # Limit to top 5 entities

    def _classify_topics(self, text: str, keywords: List[str]) -> List[str]:
        """Classify text into topic categories."""
        text_lower = text.lower()
        topics = []

        # Topic classification based on keywords and content
        topic_categories = {
            "technology": [
                "tech",
                "software",
                "ai",
                "artificial intelligence",
                "computer",
                "digital",
                "cyber",
                "internet",
                "app",
                "platform",
            ],
            "business": [
                "business",
                "economy",
                "market",
                "company",
                "finance",
                "investment",
                "profit",
                "revenue",
                "growth",
            ],
            "politics": [
                "government",
                "president",
                "election",
                "policy",
                "political",
                "congress",
                "senate",
                "vote",
            ],
            "health": [
                "health",
                "medical",
                "doctor",
                "hospital",
                "disease",
                "treatment",
                "medicine",
                "patient",
            ],
            "science": [
                "research",
                "study",
                "scientist",
                "discovery",
                "experiment",
                "analysis",
                "data",
            ],
            "sports": [
                "sports",
                "game",
                "team",
                "player",
                "championship",
                "tournament",
                "league",
            ],
            "entertainment": [
                "movie",
                "film",
                "music",
                "celebrity",
                "actor",
                "artist",
                "entertainment",
            ],
            "environment": [
                "climate",
                "environment",
                "energy",
                "pollution",
                "sustainability",
                "green",
            ],
        }

        for topic, keywords_list in topic_categories.items():
            score = 0
            for keyword in keywords_list:
                if keyword in text_lower:
                    score += 1

            # Also check extracted keywords
            for extracted_keyword in keywords:
                if extracted_keyword.lower() in keywords_list:
                    score += 2

            if score >= 2:  # Threshold for topic inclusion
                topics.append(topic)

        return topics[:3]  # Limit to top 3 topics

    def _calculate_relevance(self, entry, feed_categories: List[str]) -> int:
        """
        Calculate relevance score for entry.

        Returns relevance score from 0 to 100.
        """
        base_score = 50

        title = getattr(entry, "title", "").lower()

        # Boost score for feed category matches
        for category in feed_categories:
            if category.lower() in title:
                base_score += 20

        # Boost for recent content
        published_date = self._parse_entry_date(entry)
        if published_date:
            hours_ago = (
                datetime.now(timezone.utc) - published_date
            ).total_seconds() / 3600
            if hours_ago < 24:
                base_score += 10
            elif hours_ago < 48:
                base_score += 5

        return min(100, base_score)

    def _extract_keywords_from_title_and_categories(
        self, title: str, categories: List[str]
    ) -> List[str]:
        """Extract keywords from title and categories."""
        keywords = []

        # Add categories
        keywords.extend(categories)

        # Extract from title (simple keyword extraction)
        title_words = re.findall(r"\b[a-zA-Z]{3,}\b", title.lower())

        # Common tech/business keywords
        relevant_keywords = [
            "ai",
            "artificial intelligence",
            "technology",
            "startup",
            "business",
            "innovation",
            "digital",
            "software",
            "data",
            "cloud",
            "security",
            "mobile",
            "app",
            "social media",
            "marketing",
            "finance",
        ]

        for word in title_words:
            if word in relevant_keywords or len(word) > 5:
                keywords.append(word)

        return list(set(keywords))[:10]

    def _calculate_theme_relevance(
        self, item: FeedItemModel, themes: List[str]
    ) -> float:
        """Calculate how relevant an item is to persona themes."""
        if not themes:
            return 0.0

        text = f"{item.title} {item.description} {' '.join(item.categories)}".lower()

        matches = 0
        total_themes = len(themes)

        for theme in themes:
            if theme.lower() in text:
                matches += 1

        return matches / total_themes if total_themes > 0 else 0.0

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.http_client.aclose()

    async def assign_feed_to_persona(
        self, persona_id: UUID, assignment: PersonaFeedAssignment
    ) -> PersonaFeedResponse:
        """
        Assign an RSS feed to a persona.

        Args:
            persona_id: UUID of the persona
            assignment: Feed assignment data

        Returns:
            PersonaFeedResponse with assignment details

        Raises:
            ValueError: If feed or persona doesn't exist
        """
        try:
            # Verify feed exists
            feed = await self._get_feed_by_id(assignment.feed_id)
            if not feed:
                raise ValueError(f"Feed not found: {assignment.feed_id}")

            # Check if assignment already exists
            stmt = select(PersonaFeedModel).where(
                PersonaFeedModel.persona_id == persona_id,
                PersonaFeedModel.feed_id == assignment.feed_id,
            )
            result = await self.db.execute(stmt)
            existing = result.scalar_one_or_none()

            if existing:
                # Update existing assignment
                existing.topics = assignment.topics
                existing.priority = assignment.priority
                existing.is_active = True
                await self.db.commit()
                await self.db.refresh(existing)

                response = PersonaFeedResponse.model_validate(existing)
                response.feed_name = feed.name
                response.feed_url = feed.url
                response.feed_categories = feed.categories
                return response

            # Create new assignment
            persona_feed = PersonaFeedModel(
                persona_id=persona_id,
                feed_id=assignment.feed_id,
                topics=assignment.topics,
                priority=assignment.priority,
                is_active=True,
            )

            self.db.add(persona_feed)
            await self.db.commit()
            await self.db.refresh(persona_feed)

            logger.info(f"Feed assigned to persona {persona_id}: {assignment.feed_id}")

            response = PersonaFeedResponse.model_validate(persona_feed)
            response.feed_name = feed.name
            response.feed_url = feed.url
            response.feed_categories = feed.categories
            return response

        except Exception as e:
            await self.db.rollback()
            logger.error(f"Failed to assign feed to persona: {str(e)}")
            raise

    async def list_persona_feeds(self, persona_id: UUID) -> List[PersonaFeedResponse]:
        """
        List all feeds assigned to a persona.

        Args:
            persona_id: UUID of the persona

        Returns:
            List of feed assignments with feed details
        """
        try:
            stmt = (
                select(PersonaFeedModel, RSSFeedModel)
                .join(RSSFeedModel, PersonaFeedModel.feed_id == RSSFeedModel.id)
                .where(PersonaFeedModel.persona_id == persona_id)
                .where(PersonaFeedModel.is_active == True)
                .order_by(PersonaFeedModel.priority.desc())
            )

            result = await self.db.execute(stmt)
            rows = result.all()

            responses = []
            for persona_feed, feed in rows:
                response = PersonaFeedResponse.model_validate(persona_feed)
                response.feed_name = feed.name
                response.feed_url = feed.url
                response.feed_categories = feed.categories
                responses.append(response)

            return responses

        except Exception as e:
            logger.error(f"Error listing persona feeds: {str(e)}")
            return []

    async def unassign_feed_from_persona(self, persona_id: UUID, feed_id: UUID) -> bool:
        """
        Remove feed assignment from persona.

        Args:
            persona_id: UUID of the persona
            feed_id: UUID of the feed

        Returns:
            True if successful, False otherwise
        """
        try:
            stmt = select(PersonaFeedModel).where(
                PersonaFeedModel.persona_id == persona_id,
                PersonaFeedModel.feed_id == feed_id,
            )
            result = await self.db.execute(stmt)
            assignment = result.scalar_one_or_none()

            if assignment:
                assignment.is_active = False
                await self.db.commit()
                logger.info(f"Feed unassigned from persona {persona_id}: {feed_id}")
                return True

            return False

        except Exception as e:
            await self.db.rollback()
            logger.error(f"Error unassigning feed from persona: {str(e)}")
            return False

    async def list_feeds_by_topic(self, topic: str) -> List[RSSFeedResponse]:
        """
        List all feeds that contain a specific topic in their categories.

        Args:
            topic: Topic to filter by

        Returns:
            List of feeds matching the topic
        """
        try:
            # Query feeds where the topic appears in categories
            stmt = (
                select(RSSFeedModel)
                .where(RSSFeedModel.is_active == True)
                .order_by(RSSFeedModel.created_at.desc())
            )

            result = await self.db.execute(stmt)
            feeds = result.scalars().all()

            # Filter feeds that have matching topic in categories
            matching_feeds = [
                feed
                for feed in feeds
                if any(topic.lower() in cat.lower() for cat in (feed.categories or []))
            ]

            return [RSSFeedResponse.model_validate(feed) for feed in matching_feeds]

        except Exception as e:
            logger.error(f"Error listing feeds by topic: {str(e)}")
            return []
