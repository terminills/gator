"""
Direct Messaging Service

Core business logic for persona-based direct messaging with round-robin FIFO queue
system and PPV upselling functionality.
"""

import asyncio
import uuid
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import Any, Dict, List, Optional, Tuple

from sqlalchemy import and_, asc, desc, func, or_, select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from backend.config.logging import get_logger
from backend.models.conversation import (
    ConversationCreate,
    ConversationModel,
    ConversationResponse,
    ConversationStatus,
)
from backend.models.message import (
    MessageCreate,
    MessageModel,
    MessageResponse,
    MessageSender,
    MessageType,
)
from backend.models.persona import PersonaModel
from backend.models.ppv_offer import (
    PPVOfferCreate,
    PPVOfferModel,
    PPVOfferResponse,
    PPVOfferStatus,
    PPVOfferType,
)
from backend.models.user import UserModel

logger = get_logger(__name__)


class DirectMessagingService:
    """
    Service for managing direct messages between users and AI personas.

    Implements a round-robin FIFO queue system to ensure all users get
    attention from AI personas, along with PPV upselling capabilities.
    """

    def __init__(self, db_session: AsyncSession):
        """Initialize the service with a database session."""
        self.db = db_session

    # ==================== Conversation Management ====================

    async def create_conversation(
        self, conversation_data: ConversationCreate
    ) -> ConversationResponse:
        """
        Create a new conversation between a user and persona.

        Args:
            conversation_data: The conversation configuration data

        Returns:
            ConversationResponse: The created conversation

        Raises:
            ValueError: If conversation data fails validation
        """
        try:
            # Verify user and persona exist and are active
            await self._verify_conversation_participants(
                conversation_data.user_id, conversation_data.persona_id
            )

            # Check if conversation already exists
            existing = await self._get_existing_conversation(
                conversation_data.user_id, conversation_data.persona_id
            )
            if existing:
                logger.info(
                    f"Returning existing conversation {conversation_data.user_id} {conversation_data.persona_id}"
                )
                return ConversationResponse.model_validate(existing)

            # Create new conversation
            db_conversation = ConversationModel(
                user_id=conversation_data.user_id,
                persona_id=conversation_data.persona_id,
                title=conversation_data.title,
                queue_priority=0,  # Start with normal priority
                status=ConversationStatus.ACTIVE,
            )

            self.db.add(db_conversation)
            await self.db.commit()
            await self.db.refresh(db_conversation)

            logger.info(
                f"Created new conversation {db_conversation.id} {conversation_data.user_id} {conversation_data.persona_id}"
            )

            return ConversationResponse.model_validate(db_conversation)

        except IntegrityError as e:
            await self.db.rollback()
            logger.error(f"Database integrity error creating conversation: {str(e)}")
            raise ValueError("Failed to create conversation due to data constraints")
        except Exception as e:
            await self.db.rollback()
            logger.error(f"Unexpected error creating conversation: {str(e)}")
            raise ValueError(f"Conversation creation failed: {str(e)}")

    async def get_conversation(
        self, conversation_id: uuid.UUID
    ) -> Optional[ConversationResponse]:
        """Get a conversation by ID."""
        try:
            stmt = select(ConversationModel).where(
                ConversationModel.id == conversation_id
            )
            result = await self.db.execute(stmt)
            db_conversation = result.scalar_one_or_none()

            if not db_conversation:
                return None

            return ConversationResponse.model_validate(db_conversation)
        except Exception as e:
            logger.error(f"Failed to get conversation {conversation_id}: {str(e)}")
            raise

    async def get_user_conversations(
        self,
        user_id: uuid.UUID,
        status: Optional[ConversationStatus] = None,
        skip: int = 0,
        limit: int = 20,
    ) -> List[ConversationResponse]:
        """Get conversations for a specific user."""
        try:
            stmt = select(ConversationModel).where(ConversationModel.user_id == user_id)

            if status:
                stmt = stmt.where(ConversationModel.status == status)

            stmt = (
                stmt.order_by(desc(ConversationModel.last_message_at))
                .offset(skip)
                .limit(limit)
            )

            result = await self.db.execute(stmt)
            db_conversations = result.scalars().all()

            return [
                ConversationResponse.model_validate(conv) for conv in db_conversations
            ]
        except Exception as e:
            logger.error(f"Failed to get user conversations for {user_id}: {str(e)}")
            raise

    # ==================== Message Management ====================

    async def send_message(self, message_data: MessageCreate) -> MessageResponse:
        """
        Send a message in a conversation.

        Updates conversation metadata and queue positioning for round-robin system.
        """
        try:
            # Verify conversation exists and is active
            conversation = await self.get_conversation(message_data.conversation_id)
            if not conversation:
                raise ValueError("Conversation not found")
            if conversation.status != ConversationStatus.ACTIVE:
                raise ValueError("Cannot send message to inactive conversation")

            # Create message
            db_message = MessageModel(
                conversation_id=message_data.conversation_id,
                sender=message_data.sender,
                message_type=message_data.message_type,
                content=message_data.content,
                media_urls=message_data.media_urls or [],
                message_metadata=message_data.message_metadata or {},
            )

            self.db.add(db_message)

            # Update conversation metadata
            now = datetime.now(timezone.utc)
            update_data = {
                "last_message_at": now,
                "message_count": ConversationModel.message_count + 1,
                "updated_at": now,
            }

            # If this is a persona message, update persona message timestamp for queue management
            if message_data.sender == MessageSender.PERSONA:
                update_data["last_persona_message_at"] = now

            await self.db.execute(
                update(ConversationModel)
                .where(ConversationModel.id == message_data.conversation_id)
                .values(**update_data)
            )

            await self.db.commit()
            await self.db.refresh(db_message)

            logger.info(
                f"Message sent message_id={db_message.id} {message_data.conversation_id} sender={message_data.sender}"
            )

            return MessageResponse.model_validate(db_message)

        except Exception as e:
            await self.db.rollback()
            logger.error(f"Failed to send message: {str(e)}")
            raise

    async def get_conversation_messages(
        self,
        conversation_id: uuid.UUID,
        skip: int = 0,
        limit: int = 50,
        include_deleted: bool = False,
    ) -> List[MessageResponse]:
        """Get messages from a conversation."""
        try:
            stmt = select(MessageModel).where(
                MessageModel.conversation_id == conversation_id
            )

            if not include_deleted:
                stmt = stmt.where(MessageModel.is_deleted == False)

            stmt = stmt.order_by(asc(MessageModel.created_at)).offset(skip).limit(limit)

            result = await self.db.execute(stmt)
            db_messages = result.scalars().all()

            return [MessageResponse.model_validate(msg) for msg in db_messages]
        except Exception as e:
            logger.error(
                f"Failed to get conversation messages {conversation_id}: {str(e)}"
            )
            raise

    # ==================== Round-Robin Queue System ====================

    async def get_next_conversation_for_persona_response(
        self,
        persona_id: Optional[uuid.UUID] = None,
        max_hours_since_last_response: int = 24,
    ) -> Optional[ConversationResponse]:
        """
        Get the next conversation that needs a persona response using round-robin FIFO logic.

        Prioritizes conversations that:
        1. Haven't received a persona response recently
        2. Have pending user messages
        3. Are active and not blocked
        4. Follow FIFO ordering for fairness

        Args:
            persona_id: Specific persona to find conversations for, or None for any
            max_hours_since_last_response: Maximum hours since last persona message

        Returns:
            ConversationResponse: Next conversation needing response, or None
        """
        try:
            cutoff_time = datetime.now(timezone.utc) - timedelta(
                hours=max_hours_since_last_response
            )

            # Build query for conversations needing persona responses
            stmt = select(ConversationModel).where(
                and_(
                    ConversationModel.status == ConversationStatus.ACTIVE,
                    ConversationModel.auto_responses_enabled == True,
                    # Either never had a persona message, or it's been long enough
                    or_(
                        ConversationModel.last_persona_message_at.is_(None),
                        ConversationModel.last_persona_message_at < cutoff_time,
                    ),
                    # Must have recent user activity
                    ConversationModel.last_message_at.is_not(None),
                    ConversationModel.last_message_at > cutoff_time,
                )
            )

            if persona_id:
                stmt = stmt.where(ConversationModel.persona_id == persona_id)

            # Order by priority (higher first), then by FIFO (oldest first)
            stmt = stmt.order_by(
                desc(ConversationModel.queue_priority),
                asc(ConversationModel.last_persona_message_at.nulls_first()),
                asc(ConversationModel.created_at),
            ).limit(1)

            result = await self.db.execute(stmt)
            db_conversation = result.scalar_one_or_none()

            if db_conversation:
                logger.info(
                    f"Found next conversation for persona response {db_conversation.id} {db_conversation.persona_id}"
                )
                return ConversationResponse.model_validate(db_conversation)

            logger.debug(
                f"No conversations need persona response for persona {persona_id}"
            )
            return None

        except Exception as e:
            logger.error(f"Failed to get next conversation for response: {str(e)}")
            raise

    async def get_queue_status(self) -> Dict[str, Any]:
        """Get current queue status and statistics."""
        try:
            # Count conversations by status
            status_counts = await self.db.execute(
                select(
                    ConversationModel.status, func.count(ConversationModel.id)
                ).group_by(ConversationModel.status)
            )

            # Count conversations awaiting responses
            cutoff_time = datetime.now(timezone.utc) - timedelta(hours=24)
            awaiting_response = await self.db.execute(
                select(func.count(ConversationModel.id)).where(
                    and_(
                        ConversationModel.status == ConversationStatus.ACTIVE,
                        or_(
                            ConversationModel.last_persona_message_at.is_(None),
                            ConversationModel.last_persona_message_at < cutoff_time,
                        ),
                        ConversationModel.last_message_at > cutoff_time,
                    )
                )
            )

            return {
                "conversation_counts": dict(status_counts.fetchall()),
                "conversations_awaiting_response": awaiting_response.scalar() or 0,
                "queue_updated_at": datetime.now(timezone.utc).isoformat(),
            }

        except Exception as e:
            logger.error(f"Failed to get queue status: {str(e)}")
            raise

    # ==================== PPV Offer System ====================

    async def create_ppv_offer(self, offer_data: PPVOfferCreate) -> PPVOfferResponse:
        """Create a new PPV offer for a user."""
        try:
            # Verify conversation exists and participants allow PPV offers
            await self._verify_ppv_offer_eligibility(
                offer_data.conversation_id, offer_data.user_id
            )

            # Set default expiration if not provided (24 hours)
            expires_at = offer_data.expires_at
            if not expires_at:
                expires_at = datetime.now(timezone.utc) + timedelta(hours=24)

            # Create PPV offer
            db_offer = PPVOfferModel(
                conversation_id=offer_data.conversation_id,
                user_id=offer_data.user_id,
                persona_id=offer_data.persona_id,
                title=offer_data.title,
                description=offer_data.description,
                offer_type=offer_data.offer_type,
                price=offer_data.price,
                currency=offer_data.currency,
                preview_url=offer_data.preview_url,
                content_metadata=offer_data.content_metadata or {},
                estimated_delivery_hours=offer_data.estimated_delivery_hours,
                delivery_instructions=offer_data.delivery_instructions,
                expires_at=expires_at,
            )

            self.db.add(db_offer)

            # Update conversation PPV stats
            await self.db.execute(
                update(ConversationModel)
                .where(ConversationModel.id == offer_data.conversation_id)
                .values(ppv_offers_sent=ConversationModel.ppv_offers_sent + 1)
            )

            await self.db.commit()
            await self.db.refresh(db_offer)

            logger.info(
                f"Created PPV offer {db_offer.id} {offer_data.conversation_id} offer_type={offer_data.offer_type} price={offer_data.price}"
            )

            return PPVOfferResponse.model_validate(db_offer)

        except Exception as e:
            await self.db.rollback()
            logger.error(f"Failed to create PPV offer: {str(e)}")
            raise

    async def get_ppv_offers(
        self,
        persona_id: Optional[uuid.UUID] = None,
        user_id: Optional[uuid.UUID] = None,
        status: Optional[PPVOfferStatus] = None,
        skip: int = 0,
        limit: int = 50,
    ) -> List[PPVOfferResponse]:
        """Get PPV offers with optional filtering."""
        try:
            stmt = select(PPVOfferModel).order_by(PPVOfferModel.created_at.desc())

            if persona_id:
                stmt = stmt.where(PPVOfferModel.persona_id == persona_id)
            if user_id:
                stmt = stmt.where(PPVOfferModel.user_id == user_id)
            if status:
                stmt = stmt.where(PPVOfferModel.status == status)

            stmt = stmt.offset(skip).limit(limit)

            result = await self.db.execute(stmt)
            db_offers = result.scalars().all()

            return [PPVOfferResponse.model_validate(offer) for offer in db_offers]

        except Exception as e:
            logger.error(f"Failed to get PPV offers: {str(e)}")
            raise

    async def accept_ppv_offer(self, offer_id: uuid.UUID) -> PPVOfferResponse:
        """Accept a PPV offer (simulate payment processing)."""
        try:
            # Get the offer
            stmt = select(PPVOfferModel).where(PPVOfferModel.id == offer_id)
            result = await self.db.execute(stmt)
            db_offer = result.scalar_one_or_none()

            if not db_offer:
                raise ValueError("PPV offer not found")

            if db_offer.status != PPVOfferStatus.PENDING:
                raise ValueError(f"Cannot accept offer with status: {db_offer.status}")

            # Check if expired
            now = datetime.now(timezone.utc)
            if db_offer.expires_at and db_offer.expires_at < now:
                # Mark as expired
                await self.db.execute(
                    update(PPVOfferModel)
                    .where(PPVOfferModel.id == offer_id)
                    .values(status=PPVOfferStatus.EXPIRED, updated_at=now)
                )
                await self.db.commit()
                raise ValueError("PPV offer has expired")

            # Accept the offer
            await self.db.execute(
                update(PPVOfferModel)
                .where(PPVOfferModel.id == offer_id)
                .values(status=PPVOfferStatus.ACCEPTED, accepted_at=now, updated_at=now)
            )

            # Update conversation stats
            await self.db.execute(
                update(ConversationModel)
                .where(ConversationModel.id == db_offer.conversation_id)
                .values(ppv_offers_accepted=ConversationModel.ppv_offers_accepted + 1)
            )

            await self.db.commit()

            # Refresh to get updated data
            await self.db.refresh(db_offer)

            logger.info(
                f"PPV offer accepted {offer_id} {db_offer.conversation_id} price={db_offer.price}"
            )

            return PPVOfferResponse.model_validate(db_offer)

        except Exception as e:
            await self.db.rollback()
            logger.error(f"Failed to accept PPV offer {offer_id}: {str(e)}")
            raise

    # ==================== Helper Methods ====================

    async def _verify_conversation_participants(
        self, user_id: uuid.UUID, persona_id: uuid.UUID
    ) -> None:
        """Verify that user and persona exist and are active."""
        # Check user exists and is active
        user_stmt = select(UserModel).where(
            and_(UserModel.id == user_id, UserModel.is_active == True)
        )
        user_result = await self.db.execute(user_stmt)
        user = user_result.scalar_one_or_none()
        if not user:
            raise ValueError("User not found or inactive")

        # Check persona exists and is active
        persona_stmt = select(PersonaModel).where(
            and_(PersonaModel.id == persona_id, PersonaModel.is_active == True)
        )
        persona_result = await self.db.execute(persona_stmt)
        persona = persona_result.scalar_one_or_none()
        if not persona:
            raise ValueError("Persona not found or inactive")

    async def _get_existing_conversation(
        self, user_id: uuid.UUID, persona_id: uuid.UUID
    ) -> Optional[ConversationModel]:
        """Check if a conversation already exists between user and persona."""
        stmt = select(ConversationModel).where(
            and_(
                ConversationModel.user_id == user_id,
                ConversationModel.persona_id == persona_id,
                ConversationModel.status != ConversationStatus.ARCHIVED,
            )
        )
        result = await self.db.execute(stmt)
        return result.scalar_one_or_none()

    async def _verify_ppv_offer_eligibility(
        self, conversation_id: uuid.UUID, user_id: uuid.UUID
    ) -> None:
        """Verify that a PPV offer can be sent to this user/conversation."""
        # Get conversation with user data
        stmt = (
            select(ConversationModel, UserModel)
            .join(UserModel, UserModel.id == ConversationModel.user_id)
            .where(ConversationModel.id == conversation_id)
        )
        result = await self.db.execute(stmt)
        row = result.first()

        if not row:
            raise ValueError("Conversation not found")

        conversation, user = row

        if conversation.user_id != user_id:
            raise ValueError("User ID does not match conversation")

        if conversation.status != ConversationStatus.ACTIVE:
            raise ValueError("Cannot send PPV offer to inactive conversation")

        if not user.allow_ppv_offers:
            raise ValueError("User has disabled PPV offers")
