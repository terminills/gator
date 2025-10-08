"""
User Management API Routes

Handles user registration, profile management, and preferences.
"""

from typing import List

from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.ext.asyncio import AsyncSession

from backend.database.connection import get_db_session
from backend.models.user import UserCreate, UserResponse, UserUpdate
from backend.services.user_service import UserService
from backend.config.logging import get_logger

logger = get_logger(__name__)

router = APIRouter(
    prefix="/api/v1/users",
    tags=["users"],
    responses={404: {"description": "User not found"}},
)


def get_user_service(db: AsyncSession = Depends(get_db_session)) -> UserService:
    """Dependency injection for UserService."""
    return UserService(db)


@router.post("/", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
async def create_user(
    user_data: UserCreate,
    user_service: UserService = Depends(get_user_service),
):
    """
    Create a new user account.

    Registers a new user who can interact with AI personas through
    direct messaging and receive PPV offers.
    """
    try:
        user = await user_service.create_user(user_data)
        logger.info(f"User created via API {user.id} {user.username}")
        return user
    except ValueError as e:
        logger.warning(f"User creation failed: {str(e)}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to create user: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error",
        )


@router.get("/{user_id}", response_model=UserResponse)
async def get_user(
    user_id: str,
    user_service: UserService = Depends(get_user_service),
):
    """Get a user by ID."""
    try:
        user = await user_service.get_user(user_id)
        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"User {user_id} not found",
            )
        return user
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get user {user_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error",
        )


@router.get("/username/{username}", response_model=UserResponse)
async def get_user_by_username(
    username: str,
    user_service: UserService = Depends(get_user_service),
):
    """Get a user by username."""
    try:
        user = await user_service.get_user_by_username(username)
        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"User with username '{username}' not found",
            )
        return user
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get user by username {username}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error",
        )


@router.put("/{user_id}", response_model=UserResponse)
async def update_user(
    user_id: str,
    updates: UserUpdate,
    user_service: UserService = Depends(get_user_service),
):
    """Update an existing user."""
    try:
        user = await user_service.update_user(user_id, updates)
        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"User {user_id} not found",
            )
        logger.info(f"User updated via API {user_id}")
        return user
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to update user {user_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error",
        )


@router.post("/{user_id}/activity", status_code=status.HTTP_200_OK)
async def update_user_activity(
    user_id: str,
    user_service: UserService = Depends(get_user_service),
):
    """Update user's last active timestamp."""
    try:
        success = await user_service.update_last_active(user_id)
        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"User {user_id} not found",
            )
        return {"message": "Activity updated"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to update user activity {user_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error",
        )


@router.get("/", response_model=List[UserResponse])
async def list_users(
    skip: int = Query(0, ge=0, description="Number of users to skip"),
    limit: int = Query(50, ge=1, le=100, description="Number of users to return"),
    active_only: bool = Query(True, description="Return only active users"),
    user_service: UserService = Depends(get_user_service),
):
    """List users with pagination."""
    try:
        users = await user_service.list_users(
            skip=skip, limit=limit, active_only=active_only
        )
        return users
    except Exception as e:
        logger.error(f"Failed to list users: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error",
        )
