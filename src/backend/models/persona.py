"""
Persona Models

Database and API models for AI persona management.
"""

import uuid
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any

from pydantic import BaseModel, Field, field_validator
from sqlalchemy import Column, String, DateTime, Boolean, Integer, Text, JSON
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.sql import func

from backend.database.connection import Base


class PersonaModel(Base):
    """
    SQLAlchemy model for AI personas.
    
    Represents an AI character with appearance, personality, and style preferences
    for consistent content generation across all interactions.
    """
    
    __tablename__ = "personas"
    
    id = Column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        index=True
    )
    name = Column(String(100), nullable=False, index=True)
    appearance = Column(Text, nullable=False)
    personality = Column(Text, nullable=False)
    content_themes = Column(JSON, nullable=False, default=list)
    style_preferences = Column(JSON, nullable=False, default=dict)
    
    is_active = Column(Boolean, default=True, index=True)
    generation_count = Column(Integer, default=0)
    
    # Timestamps
    created_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False
    )
    updated_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False
    )


class PersonaCreate(BaseModel):
    """API model for creating new personas."""
    
    name: str = Field(
        min_length=2,
        max_length=100,
        description="Persona display name"
    )
    appearance: str = Field(
        min_length=10,
        max_length=2000,
        description="Physical appearance description for image generation"
    )
    personality: str = Field(
        min_length=10,
        max_length=2000,
        description="Personality traits and characteristics"
    )
    content_themes: List[str] = Field(
        default=[],
        max_length=10,
        description="Content themes this persona specializes in"
    )
    style_preferences: Dict[str, Any] = Field(
        default={},
        description="Style and aesthetic preferences"
    )
    
    @field_validator("name")
    @classmethod
    def validate_name(cls, v: str) -> str:
        """Validate persona name for safety and appropriateness."""
        # Basic HTML injection prevention
        if any(char in v for char in ["<", ">", "&", "'"]):
            raise ValueError("Name contains invalid characters")
        return v.strip()
    
    @field_validator("content_themes")
    @classmethod
    def validate_themes(cls, v: List[str]) -> List[str]:
        """Validate content themes."""
        if len(v) > 10:
            raise ValueError("Maximum 10 content themes allowed")
        
        # Basic content moderation
        inappropriate_themes = [
            "illegal activity", "hate speech", "violence", "adult content"
        ]
        for theme in v:
            if any(bad in theme.lower() for bad in inappropriate_themes):
                raise ValueError(f"Inappropriate content theme: {theme}")
        
        return v


class PersonaUpdate(BaseModel):
    """API model for updating existing personas."""
    
    name: Optional[str] = Field(None, min_length=2, max_length=100)
    appearance: Optional[str] = Field(None, min_length=10, max_length=2000)
    personality: Optional[str] = Field(None, min_length=10, max_length=2000)
    content_themes: Optional[List[str]] = Field(None, max_length=10)
    style_preferences: Optional[Dict[str, Any]] = None
    is_active: Optional[bool] = None


class PersonaResponse(BaseModel):
    """API model for persona responses."""
    
    id: uuid.UUID
    name: str
    appearance: str
    personality: str
    content_themes: List[str]
    style_preferences: Dict[str, Any]
    is_active: bool
    generation_count: int
    created_at: datetime
    updated_at: datetime
    
    model_config = {"from_attributes": True}