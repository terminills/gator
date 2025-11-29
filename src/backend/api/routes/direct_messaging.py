"""
Direct Messaging API Routes

Handles persona-based direct messaging with round-robin queue management
and PPV upselling functionality.
"""

import uuid
from typing import List, Optional

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query, status
from sqlalchemy.ext.asyncio import AsyncSession

from backend.config.logging import get_logger
from backend.database.connection import get_db_session
from backend.models.conversation import (
    ConversationCreate,
    ConversationResponse,
    ConversationStatus,
)
from backend.models.message import MessageCreate, MessageResponse
from backend.models.ppv_offer import PPVOfferCreate, PPVOfferResponse
from backend.services.direct_messaging_service import DirectMessagingService

logger = get_logger(__name__)

router = APIRouter(
    prefix="/api/v1/dm",
    tags=["direct-messaging"],
    responses={404: {"description": "Resource not found"}},
)


def get_dm_service(
    db: AsyncSession = Depends(get_db_session),
) -> DirectMessagingService:
    """Dependency injection for DirectMessagingService."""
    return DirectMessagingService(db)


# ==================== Conversation Endpoints ====================


@router.post(
    "/conversations",
    response_model=ConversationResponse,
    status_code=status.HTTP_201_CREATED,
)
async def create_conversation(
    conversation_data: ConversationCreate,
    dm_service: DirectMessagingService = Depends(get_dm_service),
):
    """
    Create a new conversation between a user and persona.

    Initializes a direct messaging conversation that will be managed by the
    round-robin queue system for persona responses.
    """
    try:
        conversation = await dm_service.create_conversation(conversation_data)
        logger.info(f"Conversation created via API {conversation.id}")
        return conversation
    except ValueError as e:
        logger.warning(f"Conversation creation failed: {str(e)}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to create conversation: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error",
        )


@router.get("/conversations/{conversation_id}", response_model=ConversationResponse)
async def get_conversation(
    conversation_id: uuid.UUID,
    dm_service: DirectMessagingService = Depends(get_dm_service),
):
    """Get a specific conversation by ID."""
    try:
        conversation = await dm_service.get_conversation(conversation_id)
        if not conversation:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Conversation {conversation_id} not found",
            )
        return conversation
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get conversation {conversation_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error",
        )


@router.get("/users/{user_id}/conversations", response_model=List[ConversationResponse])
async def get_user_conversations(
    user_id: uuid.UUID,
    status_filter: Optional[ConversationStatus] = Query(
        None, alias="status", description="Filter by conversation status"
    ),
    skip: int = Query(0, ge=0, description="Number of conversations to skip"),
    limit: int = Query(
        20, ge=1, le=100, description="Number of conversations to return"
    ),
    dm_service: DirectMessagingService = Depends(get_dm_service),
):
    """Get conversations for a specific user with optional filtering."""
    try:
        conversations = await dm_service.get_user_conversations(
            user_id=user_id, status=status_filter, skip=skip, limit=limit
        )
        return conversations
    except Exception as e:
        logger.error(f"Failed to get user conversations {user_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error",
        )


# ==================== Message Endpoints ====================


@router.post(
    "/messages", response_model=MessageResponse, status_code=status.HTTP_201_CREATED
)
async def send_message(
    message_data: MessageCreate,
    background_tasks: BackgroundTasks,
    dm_service: DirectMessagingService = Depends(get_dm_service),
):
    """
    Send a message in a conversation.

    Automatically triggers the round-robin queue processing for persona responses
    when a user sends a message.
    """
    try:
        message = await dm_service.send_message(message_data)

        # If this was a user message, trigger persona response processing in background
        if message.sender == "user":
            background_tasks.add_task(
                _process_persona_response_queue, dm_service, message.conversation_id
            )

        logger.info(
            f"Message sent via API message_id={message.id} sender={message.sender}"
        )
        return message
    except ValueError as e:
        logger.warning(f"Message sending failed: {str(e)}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to send message: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error",
        )


@router.get(
    "/conversations/{conversation_id}/messages", response_model=List[MessageResponse]
)
async def get_conversation_messages(
    conversation_id: uuid.UUID,
    skip: int = Query(0, ge=0, description="Number of messages to skip"),
    limit: int = Query(50, ge=1, le=100, description="Number of messages to return"),
    include_deleted: bool = Query(False, description="Include deleted messages"),
    dm_service: DirectMessagingService = Depends(get_dm_service),
):
    """Get messages from a conversation."""
    try:
        messages = await dm_service.get_conversation_messages(
            conversation_id=conversation_id,
            skip=skip,
            limit=limit,
            include_deleted=include_deleted,
        )
        return messages
    except Exception as e:
        logger.error(f"Failed to get conversation messages {conversation_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error",
        )


# ==================== Queue Management Endpoints ====================


@router.get("/queue/next", response_model=Optional[ConversationResponse])
async def get_next_conversation_for_response(
    persona_id: Optional[uuid.UUID] = Query(
        None, description="Specific persona ID, or None for any"
    ),
    max_hours: int = Query(
        24, ge=1, le=168, description="Max hours since last persona response"
    ),
    dm_service: DirectMessagingService = Depends(get_dm_service),
):
    """
    Get the next conversation needing a persona response from the round-robin queue.

    This endpoint is used by the AI system to determine which conversation
    should receive the next automated persona response.
    """
    try:
        conversation = await dm_service.get_next_conversation_for_persona_response(
            persona_id=persona_id, max_hours_since_last_response=max_hours
        )
        return conversation
    except Exception as e:
        logger.error(f"Failed to get next conversation for response: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error",
        )


@router.get("/queue/status")
async def get_queue_status(
    dm_service: DirectMessagingService = Depends(get_dm_service),
):
    """
    Get current queue status and statistics.

    Returns information about active conversations, pending responses,
    and overall queue health.
    """
    try:
        status_info = await dm_service.get_queue_status()
        return status_info
    except Exception as e:
        logger.error(f"Failed to get queue status: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error",
        )


# ==================== PPV Offer Endpoints ====================


@router.post(
    "/ppv-offers", response_model=PPVOfferResponse, status_code=status.HTTP_201_CREATED
)
async def create_ppv_offer(
    offer_data: PPVOfferCreate,
    dm_service: DirectMessagingService = Depends(get_dm_service),
):
    """
    Create a new PPV (Pay-Per-View) offer for a user.

    PPV offers are upselling opportunities where personas can offer
    premium content for a fee during conversations.
    """
    try:
        offer = await dm_service.create_ppv_offer(offer_data)
        logger.info(
            f"PPV offer created via API {offer.id} offer_type={offer.offer_type}"
        )
        return offer
    except ValueError as e:
        logger.warning(f"PPV offer creation failed: {str(e)}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to create PPV offer: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error",
        )


@router.get("/ppv-offers", response_model=List[PPVOfferResponse])
async def get_ppv_offers(
    persona_id: Optional[uuid.UUID] = Query(None, description="Filter by persona ID"),
    user_id: Optional[uuid.UUID] = Query(None, description="Filter by user ID"),
    status_filter: Optional[str] = Query(
        None, alias="status", description="Filter by offer status"
    ),
    skip: int = Query(0, ge=0, description="Number of offers to skip"),
    limit: int = Query(50, ge=1, le=100, description="Number of offers to return"),
    dm_service: DirectMessagingService = Depends(get_dm_service),
):
    """Get PPV offers with optional filtering."""
    try:
        from backend.models.ppv_offer import PPVOfferStatus

        status_enum = None
        if status_filter:
            try:
                status_enum = PPVOfferStatus(status_filter)
            except ValueError:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Invalid status: {status_filter}",
                )

        offers = await dm_service.get_ppv_offers(
            persona_id=persona_id,
            user_id=user_id,
            status=status_enum,
            skip=skip,
            limit=limit,
        )
        return offers
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get PPV offers: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error",
        )


@router.post("/ppv-offers/{offer_id}/accept", response_model=PPVOfferResponse)
async def accept_ppv_offer(
    offer_id: uuid.UUID,
    dm_service: DirectMessagingService = Depends(get_dm_service),
):
    """
    Accept a PPV offer.

    This would typically integrate with a payment processor.
    For now, it simulates accepting the offer and updating the status.
    """
    try:
        offer = await dm_service.accept_ppv_offer(offer_id)
        logger.info(f"PPV offer accepted via API {offer_id}")
        return offer
    except ValueError as e:
        logger.warning(f"PPV offer acceptance failed: {str(e)}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to accept PPV offer {offer_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error",
        )


# ==================== Background Tasks ====================


async def _process_persona_response_queue(
    dm_service: DirectMessagingService, triggering_conversation_id: uuid.UUID
) -> None:
    """
    Background task to process persona responses using the round-robin queue.

    This is called after a user sends a message to ensure timely persona responses
    across all active conversations.
    """
    try:
        # Get the next conversation from the queue
        conversation = await dm_service.get_next_conversation_for_persona_response()
        if conversation:
            logger.info(
                f"Processing persona response for conversation {conversation.id} triggered_by={triggering_conversation_id}"
            )

            # Load the persona's characteristics
            from backend.services.persona_service import PersonaService

            persona_service = PersonaService(dm_service.db)
            persona = await persona_service.get_persona(conversation.persona_id)

            if not persona:
                logger.warning(
                    f"Persona {conversation.persona_id} not found for conversation {conversation.id}"
                )
                return

            # Get conversation context (recent messages)
            messages = await dm_service.get_conversation_messages(
                conversation_id=conversation.id, limit=10
            )

            # Get the last user message to respond to
            last_user_message = "Hi"  # Default if no messages
            for msg in reversed(messages):
                if msg.sender == "user":
                    last_user_message = msg.content
                    break

            # Generate response using persona chat service (uses llama.cpp with personality)
            from backend.models.message import MessageModel
            from backend.services.persona_chat_service import get_persona_chat_service

            chat_service = get_persona_chat_service()

            # Convert message responses to message models for history
            message_models = []
            for msg_response in messages:
                # Create a mock MessageModel with the necessary attributes
                msg_model = type(
                    "MessageModel",
                    (),
                    {
                        "sender": msg_response.sender,
                        "content": msg_response.content,
                        "created_at": msg_response.created_at,
                    },
                )()
                message_models.append(msg_model)

            try:
                # Use persona chat service with llama.cpp for personality-based responses
                response_text = await chat_service.generate_response(
                    persona=persona,
                    user_message=last_user_message,
                    conversation_history=message_models,
                    use_ai=True,  # Enable llama.cpp AI generation
                )
                logger.info(
                    f"Generated persona response using llama.cpp for {persona.name}"
                )
            except Exception as e:
                logger.warning(f"AI generation failed, using fallback: {str(e)}")
                # Fallback to a simple response
                response_text = f"Thanks for your message! I appreciate you reaching out. What would you like to know more about?"

            # Determine if PPV offer should be included (10% chance for example)
            import random

            include_ppv = random.random() < 0.1 and conversation.ppv_enabled

            ppv_offer_id = None
            if include_ppv:
                # Create a PPV offer if applicable
                from decimal import Decimal

                from backend.models.ppv_offer import PPVOfferCreate, PPVOfferType

                ppv_data = PPVOfferCreate(
                    message_id=None,  # Will be set after message creation
                    content_type=PPVOfferType.PHOTO,
                    preview_text="Exclusive content available",
                    price=Decimal("5.99"),
                    is_active=True,
                )
                try:
                    ppv_offer = await dm_service.create_ppv_offer(ppv_data)
                    ppv_offer_id = ppv_offer.id
                    response_text += f"\n\n💎 I have some exclusive content you might enjoy! Check out my offer above."
                except Exception as e:
                    logger.warning(f"Failed to create PPV offer: {str(e)}")

            # Send the generated response
            from backend.models.message import MessageCreate, MessageSender, MessageType

            message_data = MessageCreate(
                conversation_id=conversation.id,
                sender=MessageSender.PERSONA,
                content=response_text,
                message_type=MessageType.TEXT,
                ppv_offer_id=ppv_offer_id,
            )

            await dm_service.send_message(message_data)
            logger.info(f"Sent persona response for conversation {conversation.id}")

        else:
            logger.debug("No conversations need persona response at this time")

    except Exception as e:
        logger.error(
            f"Error processing persona response queue triggering_conversation_id={triggering_conversation_id}: {str(e)}"
        )
