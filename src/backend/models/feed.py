"""
RSS Feed Models

Database and API models for RSS feed management and content ingestion.
"""

import uuid
from datetime import datetime
from typing import Dict, List, Optional, Any
from enum import Enum

from pydantic import BaseModel, Field, HttpUrl
from sqlalchemy import Column, String, DateTime, Boolean, ForeignKey, Text, JSON, Integer, Float, ARRAY
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship

from backend.database.connection import Base


class RSSFeedModel(Base):
    """
    SQLAlchemy model for RSS feeds.
    
    Represents RSS feed sources that are monitored for content inspiration
    and trending topics.
    """
    
    __tablename__ = "rss_feeds"
    
    id = Column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        index=True
    )
    
    # Feed metadata
    name = Column(String(255), nullable=False)
    url = Column(String(500), nullable=False, unique=True, index=True)
    description = Column(Text, nullable=True)
    categories = Column(JSON, nullable=True, default=list)
    
    # Fetch configuration
    fetch_frequency_hours = Column(Integer, default=6, nullable=False)
    last_fetched = Column(DateTime(timezone=True), nullable=True)
    
    # Status flags
    is_active = Column(Boolean, default=True, index=True)
    is_deleted = Column(Boolean, default=False, index=True)
    
    # Timestamps
    created_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
        index=True
    )
    updated_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False
    )
    
    # Relationships
    items = relationship("FeedItemModel", back_populates="feed", cascade="all, delete-orphan")


class FeedItemModel(Base):
    """
    SQLAlchemy model for RSS feed items.
    
    Represents individual content items fetched from RSS feeds,
    with sentiment analysis and topic extraction.
    """
    
    __tablename__ = "feed_items"
    
    id = Column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        index=True
    )
    
    # Foreign key
    feed_id = Column(
        UUID(as_uuid=True),
        ForeignKey("rss_feeds.id", ondelete="CASCADE"),
        nullable=False,
        index=True
    )
    
    # Content metadata
    title = Column(String(500), nullable=False)
    link = Column(String(1000), nullable=False, unique=True, index=True)
    description = Column(Text, nullable=True)
    published_date = Column(DateTime(timezone=True), nullable=True, index=True)
    author = Column(String(255), nullable=True)
    categories = Column(JSON, nullable=True, default=list)
    
    # Content analysis
    content_summary = Column(Text, nullable=True)
    sentiment_score = Column(Float, nullable=True)  # -1.0 to 1.0
    relevance_score = Column(Float, nullable=True)  # 0.0 to 1.0
    keywords = Column(JSON, nullable=True, default=list)
    entities = Column(JSON, nullable=True, default=list)  # Named entities
    topics = Column(JSON, nullable=True, default=list)  # Extracted topics
    
    # Processing status
    processed = Column(Boolean, default=False, index=True)
    
    # Timestamps
    created_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
        index=True
    )
    
    # Relationships
    feed = relationship("RSSFeedModel", back_populates="items")


class RSSFeedCreate(BaseModel):
    """API model for creating new RSS feed."""
    
    name: str = Field(min_length=1, max_length=255, description="Feed name")
    url: HttpUrl = Field(description="RSS feed URL")
    description: Optional[str] = Field(default=None, description="Feed description")
    categories: Optional[List[str]] = Field(default=[], description="Feed categories")
    fetch_frequency_hours: int = Field(default=6, ge=1, le=168, description="Fetch frequency in hours")


class RSSFeedUpdate(BaseModel):
    """API model for updating RSS feed."""
    
    name: Optional[str] = Field(default=None, min_length=1, max_length=255)
    description: Optional[str] = Field(default=None)
    categories: Optional[List[str]] = Field(default=None)
    fetch_frequency_hours: Optional[int] = Field(default=None, ge=1, le=168)
    is_active: Optional[bool] = Field(default=None)


class RSSFeedResponse(BaseModel):
    """API model for RSS feed responses."""
    
    id: uuid.UUID
    name: str
    url: str
    description: Optional[str]
    categories: List[str]
    fetch_frequency_hours: int
    last_fetched: Optional[datetime]
    is_active: bool
    created_at: datetime
    updated_at: datetime
    
    model_config = {"from_attributes": True}


class FeedItemResponse(BaseModel):
    """API model for feed item responses."""
    
    id: uuid.UUID
    feed_id: uuid.UUID
    title: str
    link: str
    description: Optional[str]
    published_date: Optional[datetime]
    author: Optional[str]
    categories: List[str]
    content_summary: Optional[str]
    sentiment_score: Optional[float]
    relevance_score: Optional[float]
    keywords: List[str]
    entities: List[Any]
    topics: List[str]
    processed: bool
    created_at: datetime
    
    model_config = {"from_attributes": True}


class FeedItemListResponse(BaseModel):
    """Response for feed item list with pagination."""
    
    items: List[FeedItemResponse]
    total: int
    page: int = 1
    page_size: int = 50
