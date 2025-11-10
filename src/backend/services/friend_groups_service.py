"""
Friend Groups Service

Manages persona friend groups and social interactions between personas.
Handles friend group CRUD, member management, and interaction generation.
"""

import asyncio
from typing import Dict, List, Optional, Any
from uuid import UUID
from datetime import datetime, timezone, timedelta
import random

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_, or_, func, text
from sqlalchemy.orm import selectinload

from backend.models.persona import PersonaModel
from backend.models.content import ContentModel
from backend.models.friend_groups import (
    FriendGroupModel,
    PersonaInteractionModel,
    DuetRequestModel,
    FriendGroupCreate,
    FriendGroupUpdate,
    FriendGroupResponse,
    PersonaInteractionCreate,
    PersonaInteractionResponse,
    InteractionType,
    persona_group_members,
)
from backend.config.logging import get_logger

logger = get_logger(__name__)


class FriendGroupsService:
    """
    Service for managing persona friend groups and interactions.

    Enables personas to form social networks, interact with each other's content,
    and participate in collaborative content creation.
    """

    def __init__(self, db_session: AsyncSession):
        """
        Initialize friend groups service.

        Args:
            db_session: Database session
        """
        self.db = db_session

    # Friend Group Management

    async def create_friend_group(
        self, group_data: FriendGroupCreate
    ) -> FriendGroupResponse:
        """
        Create a new friend group.

        Args:
            group_data: Friend group creation data

        Returns:
            Created friend group response
        """
        try:
            # Create friend group
            group = FriendGroupModel(
                name=group_data.name,
                description=group_data.description,
                shared_platforms=group_data.shared_platforms,
                allow_auto_interactions=group_data.allow_auto_interactions,
                interaction_frequency=group_data.interaction_frequency,
            )

            self.db.add(group)
            await self.db.flush()  # Get the ID

            # Add personas to group
            if group_data.persona_ids:
                for persona_id in group_data.persona_ids:
                    # Verify persona exists
                    persona = await self._get_persona(persona_id)
                    if persona:
                        # Insert into association table
                        stmt = persona_group_members.insert().values(
                            group_id=group.id,
                            persona_id=persona_id,
                            role="member",
                        )
                        await self.db.execute(stmt)

            await self.db.commit()
            await self.db.refresh(group)

            logger.info(
                f"Created friend group {group.id}: {group.name} "
                f"with {len(group_data.persona_ids)} members"
            )

            # Get member count
            member_count = await self._get_group_member_count(group.id)

            response = FriendGroupResponse.model_validate(group)
            response.member_count = member_count
            return response

        except Exception as e:
            await self.db.rollback()
            logger.error(f"Failed to create friend group: {str(e)}")
            raise

    async def get_friend_group(self, group_id: UUID) -> Optional[FriendGroupResponse]:
        """
        Get friend group by ID.

        Args:
            group_id: Friend group ID

        Returns:
            Friend group response or None
        """
        try:
            stmt = select(FriendGroupModel).where(FriendGroupModel.id == group_id)
            result = await self.db.execute(stmt)
            group = result.scalar_one_or_none()

            if not group:
                return None

            member_count = await self._get_group_member_count(group.id)

            response = FriendGroupResponse.model_validate(group)
            response.member_count = member_count
            return response

        except Exception as e:
            logger.error(f"Failed to get friend group: {str(e)}")
            return None

    async def list_friend_groups(
        self,
        active_only: bool = True,
        persona_id: Optional[UUID] = None,
        limit: int = 50,
    ) -> List[FriendGroupResponse]:
        """
        List friend groups.

        Args:
            active_only: Only return active groups
            persona_id: Filter by persona membership
            limit: Maximum groups to return

        Returns:
            List of friend groups
        """
        try:
            stmt = select(FriendGroupModel)

            if active_only:
                stmt = stmt.where(FriendGroupModel.is_active == True)

            if persona_id:
                # Join with members table to filter by persona
                stmt = stmt.join(
                    persona_group_members,
                    FriendGroupModel.id == persona_group_members.c.group_id,
                ).where(persona_group_members.c.persona_id == persona_id)

            stmt = stmt.order_by(FriendGroupModel.created_at.desc()).limit(limit)

            result = await self.db.execute(stmt)
            groups = result.scalars().all()

            # Build responses with member counts
            responses = []
            for group in groups:
                member_count = await self._get_group_member_count(group.id)
                response = FriendGroupResponse.model_validate(group)
                response.member_count = member_count
                responses.append(response)

            return responses

        except Exception as e:
            logger.error(f"Failed to list friend groups: {str(e)}")
            return []

    async def update_friend_group(
        self, group_id: UUID, update_data: FriendGroupUpdate
    ) -> Optional[FriendGroupResponse]:
        """
        Update friend group.

        Args:
            group_id: Friend group ID
            update_data: Update data

        Returns:
            Updated friend group or None
        """
        try:
            stmt = select(FriendGroupModel).where(FriendGroupModel.id == group_id)
            result = await self.db.execute(stmt)
            group = result.scalar_one_or_none()

            if not group:
                return None

            # Update fields
            if update_data.name is not None:
                group.name = update_data.name
            if update_data.description is not None:
                group.description = update_data.description
            if update_data.is_active is not None:
                group.is_active = update_data.is_active
            if update_data.allow_auto_interactions is not None:
                group.allow_auto_interactions = update_data.allow_auto_interactions
            if update_data.interaction_frequency is not None:
                group.interaction_frequency = update_data.interaction_frequency
            if update_data.shared_platforms is not None:
                group.shared_platforms = update_data.shared_platforms

            await self.db.commit()
            await self.db.refresh(group)

            member_count = await self._get_group_member_count(group.id)
            response = FriendGroupResponse.model_validate(group)
            response.member_count = member_count
            return response

        except Exception as e:
            await self.db.rollback()
            logger.error(f"Failed to update friend group: {str(e)}")
            return None

    async def add_persona_to_group(
        self, group_id: UUID, persona_id: UUID, role: str = "member"
    ) -> bool:
        """
        Add persona to friend group.

        Args:
            group_id: Friend group ID
            persona_id: Persona ID
            role: Role in group (member, admin, creator)

        Returns:
            True if successful
        """
        try:
            # Verify group and persona exist
            group = await self._get_group(group_id)
            persona = await self._get_persona(persona_id)

            if not group or not persona:
                return False

            # Check if already a member
            stmt = select(persona_group_members).where(
                and_(
                    persona_group_members.c.group_id == group_id,
                    persona_group_members.c.persona_id == persona_id,
                )
            )
            result = await self.db.execute(stmt)
            if result.first():
                logger.info(f"Persona {persona_id} already in group {group_id}")
                return True

            # Add to group
            stmt = persona_group_members.insert().values(
                group_id=group_id,
                persona_id=persona_id,
                role=role,
            )
            await self.db.execute(stmt)
            await self.db.commit()

            logger.info(f"Added persona {persona_id} to group {group_id}")
            return True

        except Exception as e:
            await self.db.rollback()
            logger.error(f"Failed to add persona to group: {str(e)}")
            return False

    async def remove_persona_from_group(self, group_id: UUID, persona_id: UUID) -> bool:
        """
        Remove persona from friend group.

        Args:
            group_id: Friend group ID
            persona_id: Persona ID

        Returns:
            True if successful
        """
        try:
            stmt = persona_group_members.delete().where(
                and_(
                    persona_group_members.c.group_id == group_id,
                    persona_group_members.c.persona_id == persona_id,
                )
            )
            result = await self.db.execute(stmt)
            await self.db.commit()

            if result.rowcount > 0:
                logger.info(f"Removed persona {persona_id} from group {group_id}")
                return True
            return False

        except Exception as e:
            await self.db.rollback()
            logger.error(f"Failed to remove persona from group: {str(e)}")
            return False

    async def get_group_members(self, group_id: UUID) -> List[PersonaModel]:
        """
        Get all personas in a friend group.

        Args:
            group_id: Friend group ID

        Returns:
            List of personas in group
        """
        try:
            stmt = (
                select(PersonaModel)
                .join(
                    persona_group_members,
                    PersonaModel.id == persona_group_members.c.persona_id,
                )
                .where(persona_group_members.c.group_id == group_id)
            )

            result = await self.db.execute(stmt)
            return list(result.scalars().all())

        except Exception as e:
            logger.error(f"Failed to get group members: {str(e)}")
            return []

    # Persona Interactions

    async def create_interaction(
        self, interaction_data: PersonaInteractionCreate
    ) -> PersonaInteractionResponse:
        """
        Create a persona interaction.

        Args:
            interaction_data: Interaction creation data

        Returns:
            Created interaction response
        """
        try:
            # Get target content to find target persona
            stmt = select(ContentModel).where(
                ContentModel.id == interaction_data.target_content_id
            )
            result = await self.db.execute(stmt)
            target_content = result.scalar_one_or_none()

            if not target_content:
                raise ValueError(
                    f"Content {interaction_data.target_content_id} not found"
                )

            # Create interaction
            interaction = PersonaInteractionModel(
                source_persona_id=interaction_data.source_persona_id,
                target_content_id=interaction_data.target_content_id,
                target_persona_id=target_content.persona_id,
                interaction_type=interaction_data.interaction_type.value,
                comment_text=interaction_data.comment_text,
                platform=interaction_data.platform,
            )

            self.db.add(interaction)
            await self.db.commit()
            await self.db.refresh(interaction)

            logger.info(
                f"Created {interaction.interaction_type} interaction: "
                f"{interaction.source_persona_id} -> {interaction.target_content_id}"
            )

            return PersonaInteractionResponse.model_validate(interaction)

        except Exception as e:
            await self.db.rollback()
            logger.error(f"Failed to create interaction: {str(e)}")
            raise

    async def get_content_interactions(
        self, content_id: UUID, limit: int = 50
    ) -> List[PersonaInteractionResponse]:
        """
        Get all interactions for a piece of content.

        Args:
            content_id: Content ID
            limit: Maximum interactions to return

        Returns:
            List of interactions
        """
        try:
            stmt = (
                select(PersonaInteractionModel)
                .where(PersonaInteractionModel.target_content_id == content_id)
                .order_by(PersonaInteractionModel.created_at.desc())
                .limit(limit)
            )

            result = await self.db.execute(stmt)
            interactions = result.scalars().all()

            return [PersonaInteractionResponse.model_validate(i) for i in interactions]

        except Exception as e:
            logger.error(f"Failed to get content interactions: {str(e)}")
            return []

    async def get_persona_interactions(
        self, persona_id: UUID, as_source: bool = True, limit: int = 50
    ) -> List[PersonaInteractionResponse]:
        """
        Get interactions by or targeting a persona.

        Args:
            persona_id: Persona ID
            as_source: If True, get interactions made by persona; if False, get interactions targeting persona
            limit: Maximum interactions to return

        Returns:
            List of interactions
        """
        try:
            if as_source:
                stmt = (
                    select(PersonaInteractionModel)
                    .where(PersonaInteractionModel.source_persona_id == persona_id)
                    .order_by(PersonaInteractionModel.created_at.desc())
                    .limit(limit)
                )
            else:
                stmt = (
                    select(PersonaInteractionModel)
                    .where(PersonaInteractionModel.target_persona_id == persona_id)
                    .order_by(PersonaInteractionModel.created_at.desc())
                    .limit(limit)
                )

            result = await self.db.execute(stmt)
            interactions = result.scalars().all()

            return [PersonaInteractionResponse.model_validate(i) for i in interactions]

        except Exception as e:
            logger.error(f"Failed to get persona interactions: {str(e)}")
            return []

    async def generate_auto_interactions(
        self, group_id: UUID, content_id: UUID
    ) -> List[PersonaInteractionResponse]:
        """
        Auto-generate interactions from group members for a piece of content.

        Args:
            group_id: Friend group ID
            content_id: Content to interact with

        Returns:
            List of generated interactions
        """
        try:
            # Get group
            group = await self._get_group(group_id)
            if not group or not group.allow_auto_interactions:
                return []

            # Get content
            stmt = select(ContentModel).where(ContentModel.id == content_id)
            result = await self.db.execute(stmt)
            content = result.scalar_one_or_none()
            if not content:
                return []

            # Get group members (excluding content creator)
            members = await self.get_group_members(group_id)
            members = [m for m in members if m.id != content.persona_id]

            if not members:
                return []

            # Determine interaction frequency
            interaction_count = self._calculate_interaction_count(
                group.interaction_frequency, len(members)
            )

            # Randomly select members to interact
            interacting_members = random.sample(
                members, min(interaction_count, len(members))
            )

            # Generate interactions
            interactions = []
            for persona in interacting_members:
                interaction_type = self._select_random_interaction_type()

                interaction_data = PersonaInteractionCreate(
                    source_persona_id=persona.id,
                    target_content_id=content_id,
                    interaction_type=interaction_type,
                    comment_text=(
                        self._generate_comment(persona, content)
                        if interaction_type == InteractionType.COMMENT
                        else None
                    ),
                )

                interaction = await self.create_interaction(interaction_data)
                interactions.append(interaction)

            logger.info(
                f"Auto-generated {len(interactions)} interactions for content {content_id}"
            )

            return interactions

        except Exception as e:
            logger.error(f"Failed to generate auto interactions: {str(e)}")
            return []

    # Helper methods

    async def _get_group(self, group_id: UUID) -> Optional[FriendGroupModel]:
        """Get friend group by ID."""
        stmt = select(FriendGroupModel).where(FriendGroupModel.id == group_id)
        result = await self.db.execute(stmt)
        return result.scalar_one_or_none()

    async def _get_persona(self, persona_id: UUID) -> Optional[PersonaModel]:
        """Get persona by ID."""
        stmt = select(PersonaModel).where(PersonaModel.id == persona_id)
        result = await self.db.execute(stmt)
        return result.scalar_one_or_none()

    async def _get_group_member_count(self, group_id: UUID) -> int:
        """Get count of members in a group."""
        stmt = (
            select(func.count())
            .select_from(persona_group_members)
            .where(persona_group_members.c.group_id == group_id)
        )
        result = await self.db.execute(stmt)
        return result.scalar() or 0

    def _calculate_interaction_count(self, frequency: str, member_count: int) -> int:
        """Calculate number of interactions to generate based on frequency."""
        if frequency == "low":
            return max(1, member_count // 4)
        elif frequency == "high":
            return max(3, (member_count * 3) // 4)
        else:  # normal
            return max(2, member_count // 2)

    def _select_random_interaction_type(self) -> InteractionType:
        """Randomly select an interaction type with weighted probabilities."""
        choices = [
            (InteractionType.LIKE, 0.5),  # 50% likes
            (InteractionType.COMMENT, 0.25),  # 25% comments
            (InteractionType.SHARE, 0.15),  # 15% shares
            (InteractionType.REACTION, 0.08),  # 8% reactions
            (InteractionType.DUET, 0.02),  # 2% duets
        ]

        rand = random.random()
        cumulative = 0
        for interaction_type, probability in choices:
            cumulative += probability
            if rand <= cumulative:
                return interaction_type

        return InteractionType.LIKE

    def _generate_comment(self, persona: PersonaModel, content: ContentModel) -> str:
        """Generate a comment from persona based on their personality."""
        # Simple comment generation based on personality
        comments = [
            f"Love this! 💕",
            f"This is amazing! 🔥",
            f"So good! 👏",
            f"Can't wait for more!",
            f"This speaks to me 💯",
            f"Absolutely fantastic!",
            f"You're killing it! 💪",
        ]

        return random.choice(comments)
