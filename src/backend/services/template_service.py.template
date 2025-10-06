"""
Template: Persona Management Service

This template demonstrates the recommended code structure and patterns
for implementing services in the Gator AI Influencer Platform.

Use this template as a starting point for new services and components.
"""

from typing import List, Optional, Dict, Any
from datetime import datetime, timezone
from uuid import uuid4
import logging

from pydantic import BaseModel, Field, validator
from sqlalchemy.orm import Session
from fastapi import HTTPException, status

# Configure logging
logger = logging.getLogger(__name__)


# Data Models
class PersonaCreate(BaseModel):
    """Request model for creating a new persona."""
    
    name: str = Field(..., min_length=1, max_length=100, description="Persona display name")
    appearance: str = Field(..., min_length=10, description="Detailed physical description")
    personality: str = Field(..., min_length=10, description="Personality traits and characteristics")
    content_themes: List[str] = Field(default=[], description="Preferred content themes")
    style_preferences: List[str] = Field(default=[], description="Visual style preferences")
    
    @validator('content_themes')
    def validate_content_themes(cls, v):
        """Validate content themes for appropriateness."""
        if len(v) > 10:
            raise ValueError("Maximum 10 content themes allowed")
        return v


class PersonaData(PersonaCreate):
    """Full persona data model with metadata."""
    
    id: str = Field(default_factory=lambda: str(uuid4()))
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: Optional[datetime] = None
    is_active: bool = Field(default=True)
    generation_count: int = Field(default=0, description="Number of content items generated")
    
    class Config:
        from_attributes = True


# Exceptions
class PersonaError(Exception):
    """Base exception for persona-related errors."""
    pass


class PersonaNotFoundError(PersonaError):
    """Raised when a persona is not found."""
    pass


class PersonaValidationError(PersonaError):
    """Raised when persona data validation fails."""
    pass


# Service Implementation
class PersonaService:
    """
    Service for managing AI personas with comprehensive validation and monitoring.
    
    This service handles the core business logic for persona management including
    creation, validation, consistency checking, and lifecycle management.
    """
    
    def __init__(self, db_session: Session, content_moderator: 'ContentModerator'):
        self.db = db_session
        self.content_moderator = content_moderator
        
    async def create_persona(self, persona_data: PersonaCreate) -> PersonaData:
        """
        Create a new AI persona with comprehensive validation.
        
        Args:
            persona_data: The persona configuration data
            
        Returns:
            PersonaData: The created persona with generated ID and metadata
            
        Raises:
            PersonaValidationError: If persona data fails validation
            ContentModerationError: If persona content fails moderation
        """
        try:
            # Validate persona content for appropriateness
            moderation_result = await self.content_moderator.moderate_persona(persona_data)
            if not moderation_result.approved:
                raise PersonaValidationError(
                    f"Persona content failed moderation: {moderation_result.reason}"
                )
            
            # Create persona instance
            persona = PersonaData(**persona_data.dict())
            
            # Save to database
            # For now, return the persona without database persistence
            # In production, this would save to PersonaModel table
            logger.info(f"Created new persona (placeholder mode): {persona.id}")
            return persona
            
        except Exception as e:
            logger.error(f"Failed to create persona: {str(e)}")
            raise PersonaValidationError(f"Persona creation failed: {str(e)}"
    )
    async def get_persona(self, persona_id: str) -> PersonaData:
        """
        Retrieve a persona by ID.
        
        Args:
            persona_id: The unique persona identifier
            
        Returns:
            PersonaData: The requested persona
            
        Raises:
            PersonaNotFoundError: If persona doesn't exist
        """
        # For now, return None indicating persona not found
        # In production, this would query PersonaModel table
        logger.warning(f"Get persona {persona_id} not implemented - returning None")
        return None
    
    async def update_persona(self, persona_id: str, updates: Dict[str, Any]) -> PersonaData:
        """
        Update an existing persona with validation.
        
        Args:
            persona_id: The persona to update
            updates: Dictionary of fields to update
            
        Returns:
            PersonaData: The updated persona
            
        Raises:
            PersonaNotFoundError: If persona doesn't exist
            PersonaValidationError: If updates fail validation
        """
        # Get existing persona
        persona = await self.get_persona(persona_id)
        
        # Apply updates
        for field, value in updates.items():
            if hasattr(persona, field):
                setattr(persona, field, value)
        
        # Update timestamp
        persona.updated_at = datetime.now(timezone.utc)
        
        # Validate updated persona
        moderation_result = await self.content_moderator.moderate_persona(persona)
        if not moderation_result.approved:
            raise PersonaValidationError(
                f"Updated persona failed moderation: {moderation_result.reason}"
            )
        
        # Save changes
        # For now, just log the update without database persistence
        # In production, this would update PersonaModel table
        logger.info(f"Updated persona (placeholder mode): {persona_id}")
        return persona
    
    async def list_personas(
        self, 
        skip: int = 0, 
        limit: int = 10,
        active_only: bool = True
    ) -> List[PersonaData]:
        """
        List personas with pagination and filtering.
        
        Args:
            skip: Number of records to skip
            limit: Maximum number of records to return
            active_only: Whether to return only active personas
            
        Returns:
            List[PersonaData]: List of personas
        """
        # For now, return empty list
        # In production, this would query PersonaModel table with filters
        logger.warning(f"List personas not implemented - returning empty list")
        return []
    
    async def delete_persona(self, persona_id: str) -> bool:
        """
        Soft delete a persona by marking it inactive.
        
        Args:
            persona_id: The persona to delete
            
        Returns:
            bool: True if deletion successful
            
        Raises:
            PersonaNotFoundError: If persona doesn't exist
        """
        persona = await self.get_persona(persona_id)
        persona.is_active = False
        persona.updated_at = datetime.now(timezone.utc)
        
        # Save changes
        # For now, just log the deletion without database persistence
        # In production, this would update PersonaModel table
        logger.info(f"Deleted persona (placeholder mode): {persona_id}")
        return True


# API Router (FastAPI)
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import JSONResponse

router = APIRouter(prefix="/api/v1/personas", tags=["personas"])


def get_persona_service() -> PersonaService:
    """Dependency injection for PersonaService."""
    # For now, return a basic service instance without database
    # In production, this would inject proper database session
    logger.warning("Using placeholder PersonaService without database session")
    return PersonaService()


@router.post("/", response_model=PersonaData, status_code=status.HTTP_201_CREATED)
async def create_persona(
    persona_data: PersonaCreate,
    persona_service: PersonaService = Depends(get_persona_service)
):
    """
    Create a new AI persona.
    
    This endpoint creates a new persona configuration that will be used
    for consistent character generation across all content.
    """
    try:
        persona = await persona_service.create_persona(persona_data)
        return persona
    except PersonaValidationError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Unexpected error creating persona: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )


@router.get("/{persona_id}", response_model=PersonaData)
async def get_persona(
    persona_id: str,
    persona_service: PersonaService = Depends(get_persona_service)
):
    """Get a persona by ID."""
    try:
        persona = await persona_service.get_persona(persona_id)
        return persona
    except PersonaNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Persona {persona_id} not found"
        )


@router.get("/", response_model=List[PersonaData])
async def list_personas(
    skip: int = 0,
    limit: int = 10,
    active_only: bool = True,
    persona_service: PersonaService = Depends(get_persona_service)
):
    """List personas with pagination."""
    personas = await persona_service.list_personas(skip, limit, active_only)
    return personas


# Tests
import pytest
from unittest.mock import AsyncMock, patch


class TestPersonaService:
    """Comprehensive test suite for PersonaService."""
    
    @pytest.fixture
    def persona_service(self):
        """Create PersonaService instance for testing."""
        mock_db = AsyncMock()
        mock_moderator = AsyncMock()
        return PersonaService(mock_db, mock_moderator)
    
    @pytest.fixture
    def sample_persona_data(self):
        """Create sample persona data for testing."""
        return PersonaCreate(
            name="Test Persona",
            appearance="Young woman with blonde hair and blue eyes, professional attire",
            personality="Friendly, outgoing, tech-savvy, professional",
            content_themes=["technology", "business", "lifestyle"],
            style_preferences=["professional", "modern", "clean"]
        )
    
    async def test_create_persona_success(self, persona_service, sample_persona_data):
        """Test successful persona creation."""
        # Mock moderation approval
        persona_service.content_moderator.moderate_persona.return_value = AsyncMock(
            approved=True,
            reason=None
        )
        
        result = await persona_service.create_persona(sample_persona_data)
        
        assert result.name == sample_persona_data.name
        assert result.appearance == sample_persona_data.appearance
        assert result.id is not None
        assert result.created_at is not None
        assert result.is_active is True
    
    async def test_create_persona_moderation_failure(self, persona_service, sample_persona_data):
        """Test persona creation with moderation failure."""
        # Mock moderation rejection
        persona_service.content_moderator.moderate_persona.return_value = AsyncMock(
            approved=False,
            reason="Inappropriate content detected"
        )
        
        with pytest.raises(PersonaValidationError, match="failed moderation"):
            await persona_service.create_persona(sample_persona_data)
    
    async def test_persona_data_validation(self):
        """Test persona data validation rules."""
        # Test name length validation
        with pytest.raises(ValueError):
            PersonaCreate(name="", appearance="test", personality="test")
        
        # Test content themes limit
        with pytest.raises(ValueError):
            PersonaCreate(
                name="Test",
                appearance="test appearance",
                personality="test personality",
                content_themes=["theme" + str(i) for i in range(15)]  # Too many themes
            )


if __name__ == "__main__":
    # Example usage
    print("This is a template file demonstrating Gator platform patterns")
    print("Use this as a starting point for implementing new services")