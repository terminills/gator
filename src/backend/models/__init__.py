"""
Database Models Package

Contains all Pydantic and SQLAlchemy models for the Gator AI platform.
"""

from .persona import PersonaModel, PersonaCreate, PersonaResponse, PersonaUpdate
from .user import UserModel, UserCreate, UserResponse, UserUpdate
from .conversation import ConversationModel, ConversationCreate, ConversationResponse
from .message import MessageModel, MessageCreate, MessageResponse
from .ppv_offer import PPVOfferModel, PPVOfferCreate, PPVOfferResponse
from .content import (
    ContentModel, ContentCreate, ContentResponse, ContentUpdate, 
    GenerationRequest, ContentType, ContentRating, ModerationStatus
)
from .feed import (
    RSSFeedModel, FeedItemModel, RSSFeedCreate, RSSFeedUpdate, 
    RSSFeedResponse, FeedItemCreate, FeedItemResponse,
    FeedStatus, ItemStatus, TrendingTopics, ContentInspiration
)

__all__ = [
    "PersonaModel",
    "PersonaCreate", 
    "PersonaResponse",
    "PersonaUpdate",
    "UserModel",
    "UserCreate",
    "UserResponse", 
    "UserUpdate",
    "ConversationModel",
    "ConversationCreate",
    "ConversationResponse",
    "MessageModel",
    "MessageCreate",
    "MessageResponse",
    "PPVOfferModel",
    "PPVOfferCreate",
    "PPVOfferResponse",
    "ContentModel",
    "ContentCreate",
    "ContentResponse",
    "ContentUpdate",
    "GenerationRequest",
    "ContentType",
    "ContentRating",
    "ModerationStatus",
    "RSSFeedModel",
    "FeedItemModel", 
    "RSSFeedCreate",
    "RSSFeedUpdate",
    "RSSFeedResponse",
    "FeedItemCreate",
    "FeedItemResponse",
    "FeedStatus",
    "ItemStatus",
    "TrendingTopics",
    "ContentInspiration",
]