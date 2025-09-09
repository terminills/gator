"""
Persona Management Service

Core business logic for AI persona management including validation,
moderation, and lifecycle management.
"""

from typing import List, Optional
from datetime import datetime, timezone

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update
from sqlalchemy.exc import IntegrityError

from backend.models.persona import PersonaModel, PersonaCreate, PersonaResponse, PersonaUpdate
from backend.config.logging import get_logger

logger = get_logger(__name__)


class PersonaService:
    """
    Service for managing AI personas.
    
    Handles the core business logic for persona management including
    creation, validation, and lifecycle management. Follows the
    patterns established in the template service.
    """
    
    def __init__(self, db_session: AsyncSession):
        """
        Initialize the service with a database session.
        
        Args:
            db_session: Async SQLAlchemy session
        """
        self.db = db_session
    
    async def create_persona(self, persona_data: PersonaCreate) -> PersonaResponse:
        """
        Create a new AI persona with validation.
        
        Args:
            persona_data: The persona configuration data
        
        Returns:
            PersonaResponse: The created persona with metadata
        
        Raises:
            ValueError: If persona data fails validation
        """
        try:
            # Create database model instance
            db_persona = PersonaModel(
                name=persona_data.name,
                appearance=persona_data.appearance,
                personality=persona_data.personality,
                content_themes=persona_data.content_themes,
                style_preferences=persona_data.style_preferences,
            )
            
            # Add to session and commit
            self.db.add(db_persona)
            await self.db.commit()
            await self.db.refresh(db_persona)
            
            logger.info(f"Created new persona {db_persona.id}: {db_persona.name}")
            
            # Convert to response model
            return PersonaResponse.model_validate(db_persona)
        
        except IntegrityError as e:
            await self.db.rollback()
            logger.error(f"Database integrity error creating persona: {str(e)}")
            raise ValueError("Failed to create persona due to data constraints")
        
        except Exception as e:
            await self.db.rollback()
            logger.error(f"Unexpected error creating persona: {str(e)}")
            raise ValueError(f"Persona creation failed: {str(e)}"
    )
    async def get_persona(self, persona_id: str) -> Optional[PersonaResponse]:
        """
        Retrieve a persona by ID.
        
        Args:
            persona_id: The unique persona identifier
        
        Returns:
            PersonaResponse: The requested persona, or None if not found
        """
        try:
            # Convert string to UUID 
            import uuid
            try:
                uuid_id = uuid.UUID(persona_id)
            except ValueError:
                # Invalid UUID format
                logger.debug(f"Invalid UUID format for persona_id: {persona_id}")
                return None
                
            stmt = select(PersonaModel).where(PersonaModel.id == uuid_id)
            result = await self.db.execute(stmt)
            db_persona = result.scalar_one_or_none()
            
            if not db_persona:
                logger.debug(f"Persona not found: {persona_id}")
                return None
            
            return PersonaResponse.model_validate(db_persona)
        
        except Exception as e:
            logger.error(f"Failed to get persona {persona_id}: {str(e)}")
            raise
    
    async def list_personas(
        self, 
        skip: int = 0, 
        limit: int = 10, 
        active_only: bool = True
    ) -> List[PersonaResponse]:
        """
        List personas with pagination and filtering.
        
        Args:
            skip: Number of records to skip
            limit: Maximum number of records to return
            active_only: Whether to return only active personas
        
        Returns:
            List[PersonaResponse]: List of personas
        """
        try:
            stmt = select(PersonaModel)
            
            if active_only:
                stmt = stmt.where(PersonaModel.is_active == True)
            
            stmt = stmt.offset(skip).limit(limit).order_by(PersonaModel.created_at.desc())
            
            result = await self.db.execute(stmt)
            db_personas = result.scalars().all()
            
            return [PersonaResponse.model_validate(persona) for persona in db_personas]
        
        except Exception as e:
            logger.error(f"Failed to list personas: {str(e)}")
            raise
    
    async def update_persona(
        self, 
        persona_id: str, 
        updates: PersonaUpdate
    ) -> Optional[PersonaResponse]:
        """
        Update an existing persona.
        
        Args:
            persona_id: The persona to update
            updates: Fields to update
        
        Returns:
            PersonaResponse: Updated persona, or None if not found
        
        Raises:
            ValueError: If updates fail validation
        """
        try:
            # First check if persona exists
            existing = await self.get_persona(persona_id)
            if not existing:
                return None
            
            # Build update dictionary with only provided fields
            update_data = {}
            if updates.name is not None:
                update_data['name'] = updates.name
            if updates.appearance is not None:
                update_data['appearance'] = updates.appearance
            if updates.personality is not None:
                update_data['personality'] = updates.personality
            if updates.content_themes is not None:
                update_data['content_themes'] = updates.content_themes
            if updates.style_preferences is not None:
                update_data['style_preferences'] = updates.style_preferences
            if updates.is_active is not None:
                update_data['is_active'] = updates.is_active
            
            if not update_data:
                # No updates provided, return existing persona
                return existing
            
            # Add updated timestamp
            update_data['updated_at'] = datetime.now(timezone.utc)
            
            # Perform update
            import uuid
            uuid_id = uuid.UUID(persona_id)
            stmt = (
                update(PersonaModel)
                .where(PersonaModel.id == uuid_id)
                .values(**update_data)
            )
            
            await self.db.execute(stmt)
            await self.db.commit()
            
            logger.info(f"Updated persona {persona_id}: {list(update_data.keys())}")
            
            # Return updated persona
            return await self.get_persona(persona_id)
        
        except IntegrityError as e:
            await self.db.rollback()
            logger.error(f"Database integrity error updating persona {persona_id}: {str(e)}")
            raise ValueError("Failed to update persona due to data constraints")
        
        except Exception as e:
            await self.db.rollback()
            logger.error(f"Failed to update persona {persona_id}: {str(e)}")
            raise
    
    async def delete_persona(self, persona_id: str) -> bool:
        """
        Soft delete a persona by marking it inactive.
        
        Args:
            persona_id: The persona to delete
        
        Returns:
            bool: True if deletion successful, False if persona not found
        """
        try:
            # Check if persona exists
            existing = await self.get_persona(persona_id)
            if not existing:
                return False
            
            # Soft delete by marking inactive
            import uuid
            uuid_id = uuid.UUID(persona_id)
            stmt = (
                update(PersonaModel)
                .where(PersonaModel.id == uuid_id)
                .values(
                    is_active=False,
                    updated_at=datetime.now(timezone.utc)
                )
            )
            
            result = await self.db.execute(stmt)
            await self.db.commit()
            
            if result.rowcount > 0:
                logger.info(f"Soft deleted persona: {persona_id}")
                return True
            
            return False
        
        except Exception as e:
            await self.db.rollback()
            logger.error(f"Failed to delete persona {persona_id}: {str(e)}")
            raise
    
    async def increment_generation_count(self, persona_id: str) -> bool:
        """
        Increment the generation count for a persona.
        
        This method is called when content is generated using this persona.
        
        Args:
            persona_id: The persona to update
        
        Returns:
            bool: True if successful, False if persona not found
        """
        try:
            stmt = (
                update(PersonaModel)
                .where(PersonaModel.id == persona_id)
                .values(generation_count=PersonaModel.generation_count + 1)
            )
            
            result = await self.db.execute(stmt)
            await self.db.commit()
            
            if result.rowcount > 0:
                logger.debug(f"Incremented generation count for persona: {persona_id}")
                return True
            
            return False
        
        except Exception as e:
            await self.db.rollback()
            logger.error(f"Failed to increment generation count {persona_id}: {str(e)}")
            return False