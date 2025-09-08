"""
RSS Feed Ingestion Service

Handles RSS feed management, content parsing, and trend analysis
for informing AI content generation.
"""

import asyncio
import feedparser
from datetime import datetime, timezone
from typing import List, Optional, Dict, Any
from uuid import UUID
import re
from urllib.parse import urlparse

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update
import httpx

from backend.models.feed import (
    RSSFeedModel, FeedItemModel, 
    RSSFeedCreate, RSSFeedResponse, FeedItemResponse
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
            timeout=30.0,
            headers={"User-Agent": "Gator AI Influencer Platform/1.0"}
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
                fetch_frequency_hours=feed_data.fetch_frequency_hours
            )
            
            self.db.add(feed)
            await self.db.commit()
            await self.db.refresh(feed)
            
            logger.info("RSS feed added", feed_id=feed.id, url=feed.url)
            
            # Perform initial fetch
            await self._fetch_feed_content(feed)
            
            return RSSFeedResponse.model_validate(feed)
            
        except Exception as e:
            await self.db.rollback()
            logger.error("Failed to add RSS feed", error=str(e), url=str(feed_data.url))
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
            logger.error("Error listing RSS feeds", error=str(e))
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
                logger.error("Error fetching feed", 
                           feed_id=feed_response.id, 
                           error=str(e))
                results[str(feed_response.id)] = 0
        
        logger.info("Completed feed fetch cycle", results=results)
        return results
    
    async def get_trending_topics(self, limit: int = 20, hours: int = 24) -> List[Dict[str, Any]]:
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
            cutoff_time = datetime.now(timezone.utc).replace(
                hour=datetime.now(timezone.utc).hour - hours
            )
            
            stmt = (select(FeedItemModel)
                   .where(FeedItemModel.created_at >= cutoff_time)
                   .where(FeedItemModel.processed == True)
                   .order_by(FeedItemModel.relevance_score.desc()))
            
            result = await self.db.execute(stmt)
            items = result.scalars().all()
            
            # Simple topic extraction (in production, use NLP libraries)
            topic_counts = {}
            for item in items:
                # Extract keywords from title and categories
                keywords = self._extract_keywords(item.title, item.categories)
                for keyword in keywords:
                    if keyword not in topic_counts:
                        topic_counts[keyword] = {
                            'count': 0,
                            'sentiment': 0,
                            'relevance': 0,
                            'sample_titles': []
                        }
                    
                    topic_counts[keyword]['count'] += 1
                    topic_counts[keyword]['sentiment'] += (item.sentiment_score or 0)
                    topic_counts[keyword]['relevance'] += (item.relevance_score or 0)
                    
                    if len(topic_counts[keyword]['sample_titles']) < 3:
                        topic_counts[keyword]['sample_titles'].append(item.title)
            
            # Sort by count and relevance
            trending = []
            for topic, data in topic_counts.items():
                if data['count'] >= 2:  # Minimum threshold
                    trending.append({
                        'topic': topic,
                        'mentions': data['count'],
                        'avg_sentiment': data['sentiment'] / data['count'],
                        'avg_relevance': data['relevance'] / data['count'],
                        'sample_titles': data['sample_titles']
                    })
            
            trending.sort(key=lambda x: (x['mentions'], x['avg_relevance']), reverse=True)
            return trending[:limit]
            
        except Exception as e:
            logger.error("Error analyzing trending topics", error=str(e))
            return []
    
    async def get_content_suggestions(self, persona_themes: List[str], limit: int = 10) -> List[FeedItemResponse]:
        """
        Get content suggestions based on persona themes.
        
        Args:
            persona_themes: List of themes from persona
            limit: Maximum suggestions to return
            
        Returns:
            List of relevant feed items
        """
        try:
            if not persona_themes:
                return []
            
            # Build query to find items matching persona themes
            theme_filters = []
            for theme in persona_themes:
                # Check categories and content summary
                theme_filters.extend([
                    FeedItemModel.categories.contains([theme.lower()]),
                    FeedItemModel.content_summary.ilike(f'%{theme}%'),
                    FeedItemModel.title.ilike(f'%{theme}%')
                ])
            
            # Get recent, high-relevance items
            cutoff_time = datetime.now(timezone.utc).replace(
                hour=datetime.now(timezone.utc).hour - 48
            )
            
            stmt = (select(FeedItemModel)
                   .where(FeedItemModel.created_at >= cutoff_time)
                   .where(FeedItemModel.processed == True)
                   .order_by(FeedItemModel.relevance_score.desc())
                   .limit(limit * 2))  # Get more to filter through
            
            result = await self.db.execute(stmt)
            items = result.scalars().all()
            
            # Score items based on theme relevance
            scored_items = []
            for item in items:
                score = self._calculate_theme_relevance(item, persona_themes)
                if score > 0.3:  # Minimum relevance threshold
                    scored_items.append((score, item))
            
            # Sort by relevance score and return top items
            scored_items.sort(key=lambda x: x[0], reverse=True)
            return [FeedItemResponse.model_validate(item) for _, item in scored_items[:limit]]
            
        except Exception as e:
            logger.error("Error getting content suggestions", error=str(e))
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
                logger.warning("RSS feed contains no entries", url=url)
                
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
                item_link = getattr(entry, 'link', '')
                existing = await self._get_feed_item_by_link(item_link)
                
                if not existing and item_link:
                    # Create new feed item
                    item = FeedItemModel(
                        feed_id=feed.id,
                        title=getattr(entry, 'title', 'Untitled'),
                        link=item_link,
                        description=getattr(entry, 'summary', ''),
                        published_date=self._parse_entry_date(entry),
                        author=getattr(entry, 'author', ''),
                        categories=self._extract_entry_categories(entry),
                        content_summary=self._generate_summary(entry),
                        sentiment_score=self._analyze_sentiment(entry),
                        relevance_score=self._calculate_relevance(entry, feed.categories),
                        processed=True
                    )
                    
                    self.db.add(item)
                    new_items += 1
            
            if new_items > 0:
                await self.db.commit()
                logger.info("Processed feed items", 
                           feed_id=feed.id, 
                           new_items=new_items)
            
            return new_items
            
        except Exception as e:
            logger.error("Error fetching feed content", 
                        feed_id=feed.id, 
                        error=str(e))
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
            if hasattr(entry, 'published_parsed') and entry.published_parsed:
                return datetime(*entry.published_parsed[:6], tzinfo=timezone.utc)
            elif hasattr(entry, 'updated_parsed') and entry.updated_parsed:
                return datetime(*entry.updated_parsed[:6], tzinfo=timezone.utc)
        except Exception:
            pass
        return None
    
    def _extract_entry_categories(self, entry) -> List[str]:
        """Extract categories from RSS entry."""
        categories = []
        if hasattr(entry, 'tags'):
            categories.extend([tag.term.lower() for tag in entry.tags])
        if hasattr(entry, 'category'):
            categories.append(entry.category.lower())
        return list(set(categories))[:10]  # Limit to 10 categories
    
    def _generate_summary(self, entry) -> str:
        """Generate content summary from RSS entry."""
        # Simple summary generation (in production, use proper NLP)
        summary = getattr(entry, 'summary', '')
        if summary:
            # Clean HTML and truncate
            clean_summary = re.sub('<[^<]+?>', '', summary)
            return clean_summary[:500] + '...' if len(clean_summary) > 500 else clean_summary
        return ''
    
    def _analyze_sentiment(self, entry) -> int:
        """
        Analyze sentiment of RSS entry content.
        
        Returns sentiment score from -100 to 100.
        This is a placeholder implementation.
        """
        # Simple keyword-based sentiment analysis
        title = getattr(entry, 'title', '').lower()
        summary = getattr(entry, 'summary', '').lower()
        text = f"{title} {summary}"
        
        positive_words = ['good', 'great', 'amazing', 'excellent', 'breakthrough', 'success']
        negative_words = ['bad', 'terrible', 'crisis', 'failure', 'problem', 'issue']
        
        positive_count = sum(1 for word in positive_words if word in text)
        negative_count = sum(1 for word in negative_words if word in text)
        
        if positive_count + negative_count == 0:
            return 0
        
        return int(((positive_count - negative_count) / (positive_count + negative_count)) * 100)
    
    def _calculate_relevance(self, entry, feed_categories: List[str]) -> int:
        """
        Calculate relevance score for entry.
        
        Returns relevance score from 0 to 100.
        """
        base_score = 50
        
        title = getattr(entry, 'title', '').lower()
        
        # Boost score for feed category matches
        for category in feed_categories:
            if category.lower() in title:
                base_score += 20
        
        # Boost for recent content
        published_date = self._parse_entry_date(entry)
        if published_date:
            hours_ago = (datetime.now(timezone.utc) - published_date).total_seconds() / 3600
            if hours_ago < 24:
                base_score += 10
            elif hours_ago < 48:
                base_score += 5
        
        return min(100, base_score)
    
    def _extract_keywords(self, title: str, categories: List[str]) -> List[str]:
        """Extract keywords from title and categories."""
        keywords = []
        
        # Add categories
        keywords.extend(categories)
        
        # Extract from title (simple keyword extraction)
        title_words = re.findall(r'\b[a-zA-Z]{3,}\b', title.lower())
        
        # Common tech/business keywords
        relevant_keywords = [
            'ai', 'artificial intelligence', 'technology', 'startup', 'business',
            'innovation', 'digital', 'software', 'data', 'cloud', 'security',
            'mobile', 'app', 'social media', 'marketing', 'finance'
        ]
        
        for word in title_words:
            if word in relevant_keywords or len(word) > 5:
                keywords.append(word)
        
        return list(set(keywords))[:10]
    
    def _calculate_theme_relevance(self, item: FeedItemModel, themes: List[str]) -> float:
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