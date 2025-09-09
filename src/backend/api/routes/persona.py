"""
Persona Management API Routes

Handles AI persona creation, management, and configuration.
"""

from typing import List, Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.ext.asyncio import AsyncSession

from backend.database.connection import get_db_session
from backend.models.persona import PersonaCreate, PersonaResponse, PersonaUpdate
from backend.services.persona_service import PersonaService
from backend.config.logging import get_logger

logger = get_logger(__name__)

router = APIRouter(
    prefix="/api/v1/personas",
    tags=["personas"],
    responses={404: {"description": "Persona not found"}},
)


def get_persona_service(
    db: AsyncSession = Depends(get_db_session)
) -> PersonaService:
    """Dependency injection for PersonaService."""
    return PersonaService(db)


@router.post("/", response_model=PersonaResponse, status_code=status.HTTP_201_CREATED)
async def create_persona(
    persona_data: PersonaCreate,
    persona_service: PersonaService = Depends(get_persona_service),
):
    """
    Create a new AI persona.
    
    Creates a new persona configuration that will be used for consistent
    character generation across all content. The persona includes appearance,
    personality traits, content themes, and style preferences.
    
    Args:
        persona_data: Persona configuration data
        persona_service: Injected persona service
    
    Returns:
        PersonaResponse: Created persona with metadata
    
    Raises:
        400: Validation error or content moderation failure
        500: Internal server error
    """
    try:
        persona = await persona_service.create_persona(persona_data)
        logger.info(f"Persona created: {persona.id}")
        return persona
    except ValueError as e:
        logger.warning(f"Persona validation failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Failed to create persona: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )


@router.get("/", response_model=List[PersonaResponse])
async def list_personas(
    skip: int = Query(0, ge=0, description="Number of personas to skip"),
    limit: int = Query(10, ge=1, le=100, description="Number of personas to return"),
    active_only: bool = Query(True, description="Return only active personas"),
    persona_service: PersonaService = Depends(get_persona_service),
):
    """
    List personas with pagination.
    
    Retrieves a paginated list of personas with optional filtering
    by active status.
    
    Args:
        skip: Number of records to skip for pagination
        limit: Maximum number of records to return
        active_only: Whether to return only active personas
        persona_service: Injected persona service
    
    Returns:
        List[PersonaResponse]: List of personas
    """
    try:
        personas = await persona_service.list_personas(
            skip=skip, 
            limit=limit, 
            active_only=active_only
        )
        return personas
    except Exception as e:
        logger.error(f"Failed to list personas: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )


@router.get("/{persona_id}", response_model=PersonaResponse)
async def get_persona(
    persona_id: str,
    persona_service: PersonaService = Depends(get_persona_service),
):
    """
    Get a specific persona by ID.
    
    Args:
        persona_id: The persona identifier
        persona_service: Injected persona service
    
    Returns:
        PersonaResponse: The requested persona
    
    Raises:
        404: Persona not found
        500: Internal server error
    """
    try:
        persona = await persona_service.get_persona(persona_id)
        if not persona:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Persona {persona_id} not found"
            )
        return persona
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get persona {persona_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )


@router.put("/{persona_id}", response_model=PersonaResponse)
async def update_persona(
    persona_id: str,
    updates: PersonaUpdate,
    persona_service: PersonaService = Depends(get_persona_service),
):
    """
    Update an existing persona.
    
    Updates the specified persona with new data. All updates go through
    validation and content moderation.
    
    Args:
        persona_id: The persona to update
        updates: Fields to update
        persona_service: Injected persona service
    
    Returns:
        PersonaResponse: Updated persona
    
    Raises:
        404: Persona not found
        400: Validation error
        500: Internal server error
    """
    try:
        persona = await persona_service.update_persona(persona_id, updates)
        if not persona:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Persona {persona_id} not found"
            )
        logger.info(f"Persona updated {persona_id}")
        return persona
    except HTTPException:
        raise
    except ValueError as e:
        logger.warning(f"Persona update validation failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Failed to update persona {persona_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )


@router.delete("/{persona_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_persona(
    persona_id: str,
    persona_service: PersonaService = Depends(get_persona_service),
):
    """
    Delete a persona (soft delete).
    
    Marks the persona as inactive rather than permanently deleting it.
    This preserves referential integrity and audit trails.
    
    Args:
        persona_id: The persona to delete
        persona_service: Injected persona service
    
    Raises:
        404: Persona not found
        500: Internal server error
    """
    try:
        success = await persona_service.delete_persona(persona_id)
        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Persona {persona_id} not found"
            )
        logger.info(f"Persona deleted {persona_id}")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete persona {persona_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )