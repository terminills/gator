"""
User Management Service

Core business logic for platform user management.
"""

from datetime import datetime, timezone
from typing import List, Optional

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update, and_
from sqlalchemy.exc import IntegrityError

from backend.models.user import UserModel, UserCreate, UserResponse, UserUpdate
from backend.config.logging import get_logger

logger = get_logger(__name__)


class UserService:
    """
    Service for managing platform users.

    Handles user registration, profile management, and preferences
    for direct messaging and PPV offers.
    """

    def __init__(self, db_session: AsyncSession):
        """Initialize the service with a database session."""
        self.db = db_session

    async def create_user(self, user_data: UserCreate) -> UserResponse:
        """Create a new user account."""
        try:
            # Create database model instance
            db_user = UserModel(
                username=user_data.username,
                email=user_data.email,
                display_name=user_data.display_name,
                profile_picture_url=user_data.profile_picture_url,
                bio=user_data.bio,
                is_active=True,
                receive_dm_notifications=True,
                allow_ppv_offers=True,
                last_active_at=datetime.now(timezone.utc),
            )

            self.db.add(db_user)
            await self.db.commit()
            await self.db.refresh(db_user)

            logger.info(f"Created new user {db_user.id} {db_user.username}")
            return UserResponse.model_validate(db_user)

        except IntegrityError as e:
            await self.db.rollback()
            if "username" in str(e):
                raise ValueError("Username already exists")
            elif "email" in str(e):
                raise ValueError("Email already exists")
            else:
                raise ValueError("User creation failed due to data constraints")
        except Exception as e:
            await self.db.rollback()
            logger.error(f"Unexpected error creating user: {str(e)}")
            raise ValueError(f"User creation failed: {str(e)}")

    async def get_user(self, user_id: str) -> Optional[UserResponse]:
        """Get a user by ID."""
        try:
            stmt = select(UserModel).where(UserModel.id == user_id)
            result = await self.db.execute(stmt)
            db_user = result.scalar_one_or_none()

            if not db_user:
                return None

            return UserResponse.model_validate(db_user)
        except Exception as e:
            logger.error(f"Failed to get user {user_id}: {str(e)}")
            raise

    async def get_user_by_username(self, username: str) -> Optional[UserResponse]:
        """Get a user by username."""
        try:
            stmt = select(UserModel).where(UserModel.username == username.lower())
            result = await self.db.execute(stmt)
            db_user = result.scalar_one_or_none()

            if not db_user:
                return None

            return UserResponse.model_validate(db_user)
        except Exception as e:
            logger.error(f"Failed to get user by username {username}: {str(e)}")
            raise

    async def update_user(
        self, user_id: str, updates: UserUpdate
    ) -> Optional[UserResponse]:
        """Update an existing user."""
        try:
            # Check if user exists
            existing = await self.get_user(user_id)
            if not existing:
                return None

            # Build update dictionary
            update_data = {}
            if updates.display_name is not None:
                update_data["display_name"] = updates.display_name
            if updates.profile_picture_url is not None:
                update_data["profile_picture_url"] = updates.profile_picture_url
            if updates.bio is not None:
                update_data["bio"] = updates.bio
            if updates.receive_dm_notifications is not None:
                update_data["receive_dm_notifications"] = (
                    updates.receive_dm_notifications
                )
            if updates.allow_ppv_offers is not None:
                update_data["allow_ppv_offers"] = updates.allow_ppv_offers
            if updates.is_active is not None:
                update_data["is_active"] = updates.is_active

            if not update_data:
                return existing

            update_data["updated_at"] = datetime.now(timezone.utc)

            # Perform update
            stmt = (
                update(UserModel).where(UserModel.id == user_id).values(**update_data)
            )

            await self.db.execute(stmt)
            await self.db.commit()
            
            # Expire session cache to ensure fresh data on next query
            # This is necessary because Core update() doesn't update the session identity map
            self.db.expire_all()

            logger.info(f"Updated user {user_id} fields={list(update_data.keys())}")
            return await self.get_user(user_id)

        except Exception as e:
            await self.db.rollback()
            logger.error(f"Failed to update user {user_id}: {str(e)}")
            raise

    async def update_last_active(self, user_id: str) -> bool:
        """Update user's last active timestamp."""
        try:
            stmt = (
                update(UserModel)
                .where(UserModel.id == user_id)
                .values(last_active_at=datetime.now(timezone.utc))
            )

            result = await self.db.execute(stmt)
            await self.db.commit()

            return result.rowcount > 0
        except Exception as e:
            await self.db.rollback()
            logger.error(f"Failed to update last active {user_id}: {str(e)}")
            return False

    async def list_users(
        self, skip: int = 0, limit: int = 50, active_only: bool = True
    ) -> List[UserResponse]:
        """List users with pagination."""
        try:
            stmt = select(UserModel)

            if active_only:
                stmt = stmt.where(UserModel.is_active == True)

            stmt = stmt.offset(skip).limit(limit).order_by(UserModel.created_at.desc())

            result = await self.db.execute(stmt)
            db_users = result.scalars().all()

            return [UserResponse.model_validate(user) for user in db_users]
        except Exception as e:
            logger.error(f"Failed to list users: {str(e)}")
            raise
