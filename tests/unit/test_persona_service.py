"""
Test PersonaService functionality.

Tests the core persona management business logic.
"""

import pytest
from unittest.mock import AsyncMock

from backend.services.persona_service import PersonaService
from backend.models.persona import PersonaCreate, PersonaUpdate


class TestPersonaService:
    """Test suite for PersonaService."""
    
    @pytest.fixture
    def sample_persona_data(self):
        """Create sample persona data for testing."""
        return PersonaCreate(
            name="Test Persona",
            appearance="Young woman with blonde hair and blue eyes, professional attire",
            personality="Friendly, outgoing, tech-savvy, professional",
            content_themes=["technology", "business", "lifestyle"],
            style_preferences={"style": "professional", "tone": "modern", "aesthetic": "clean"}
        )
    
    async def test_create_persona_success(self, db_session, sample_persona_data):
        """Test successful persona creation."""
        service = PersonaService(db_session)
        
        result = await service.create_persona(sample_persona_data)
        
        assert result.name == sample_persona_data.name
        assert result.appearance == sample_persona_data.appearance
        assert result.personality == sample_persona_data.personality
        assert result.content_themes == sample_persona_data.content_themes
        assert result.style_preferences == sample_persona_data.style_preferences
        assert result.id is not None
        assert result.created_at is not None
        assert result.is_active is True
        assert result.generation_count == 0
    
    async def test_get_persona_success(self, db_session, sample_persona_data):
        """Test successful persona retrieval."""
        service = PersonaService(db_session)
        
        # Create persona first
        created = await service.create_persona(sample_persona_data)
        
        # Retrieve it
        result = await service.get_persona(created.id)
        
        assert result is not None
        assert result.id == created.id
        assert result.name == created.name
    
    async def test_get_persona_not_found(self, db_session):
        """Test persona retrieval when persona doesn't exist."""
        service = PersonaService(db_session)
        
        from uuid import uuid4
        result = await service.get_persona(uuid4())
        
        assert result is None
    
    async def test_list_personas_empty(self, db_session):
        """Test listing personas when none exist."""
        service = PersonaService(db_session)
        
        result = await service.list_personas()
        
        assert isinstance(result, list)
        assert len(result) == 0
    
    async def test_list_personas_with_data(self, db_session, sample_persona_data):
        """Test listing personas with existing data."""
        service = PersonaService(db_session)
        
        # Create a persona
        await service.create_persona(sample_persona_data)
        
        result = await service.list_personas()
        
        assert len(result) == 1
        assert result[0].name == sample_persona_data.name
    
    async def test_update_persona_success(self, db_session, sample_persona_data):
        """Test successful persona update."""
        service = PersonaService(db_session)
        
        # Create persona first
        created = await service.create_persona(sample_persona_data)
        
        # Update it
        updates = PersonaUpdate(name="Updated Name")
        result = await service.update_persona(created.id, updates)
        
        assert result is not None
        assert result.name == "Updated Name"
        assert result.appearance == sample_persona_data.appearance  # Unchanged
        assert result.updated_at is not None
    
    async def test_update_persona_not_found(self, db_session):
        """Test updating a non-existent persona."""
        service = PersonaService(db_session)
        
        from uuid import uuid4
        updates = PersonaUpdate(name="Test")
        result = await service.update_persona(uuid4(), updates)
        
        assert result is None
    
    async def test_delete_persona_success(self, db_session, sample_persona_data):
        """Test successful persona deletion (soft delete)."""
        service = PersonaService(db_session)
        
        # Create persona first
        created = await service.create_persona(sample_persona_data)
        
        # Delete it
        result = await service.delete_persona(created.id)
        
        assert result is True
        
        # Verify it's marked inactive
        persona = await service.get_persona(created.id)
        assert persona is not None
        assert persona.is_active is False
    
    async def test_delete_persona_not_found(self, db_session):
        """Test deleting a non-existent persona."""
        service = PersonaService(db_session)
        
        from uuid import uuid4
        result = await service.delete_persona(uuid4())
        
        assert result is False
    
    async def test_increment_generation_count(self, db_session, sample_persona_data):
        """Test incrementing generation count."""
        service = PersonaService(db_session)
        
        # Create persona first
        created = await service.create_persona(sample_persona_data)
        assert created.generation_count == 0
        
        # Increment count
        result = await service.increment_generation_count(created.id)
        assert result is True
        
        # Verify count increased
        updated = await service.get_persona(created.id)
        assert updated.generation_count == 1


class TestPersonaValidation:
    """Test persona data validation."""
    
    def test_persona_create_valid(self):
        """Test creating valid persona data."""
        persona = PersonaCreate(
            name="Test",
            appearance="Test appearance description",
            personality="Test personality description",
            content_themes=["theme1", "theme2"],
            style_preferences={"style1": "value1", "style2": "value2"}
        )
        
        assert persona.name == "Test"
        assert len(persona.content_themes) == 2
    
    def test_persona_create_too_many_themes(self):
        """Test validation with too many content themes."""
        with pytest.raises(ValueError, match="Maximum 10 content themes"):
            PersonaCreate(
                name="Test",
                appearance="Test appearance",
                personality="Test personality",
                content_themes=[f"theme{i}" for i in range(15)]
            )
    
    def test_persona_create_invalid_name(self):
        """Test validation with invalid name."""
        with pytest.raises(ValueError, match="Name contains invalid characters"):
            PersonaCreate(
                name="Test<script>",
                appearance="Test appearance",
                personality="Test personality"
            )
    
    def test_persona_create_inappropriate_theme(self):
        """Test validation with inappropriate content theme."""
        with pytest.raises(ValueError, match="Inappropriate content theme"):
            PersonaCreate(
                name="Test",
                appearance="Test appearance", 
                personality="Test personality",
                content_themes=["illegal activity"]
            )