"""
Content Moderation API Routes

Provides endpoints for the content moderation pipeline including:
- Content analysis
- Review queue management
- Moderation statistics
"""

import uuid
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from backend.api.routes.auth import get_current_user
from backend.config.logging import get_logger
from backend.database.connection import get_db_session
from backend.models.user import UserResponse
from backend.services.content_moderation_service import (
    ContentModerationPipeline,
    ContentType,
    ModerationAction,
    ModerationCategory,
    ModerationConfig,
    ModerationResult,
    ModerationSeverity,
)

logger = get_logger(__name__)

router = APIRouter(
    prefix="/api/v1/moderation",
    tags=["moderation"],
    responses={
        401: {"description": "Unauthorized"},
        404: {"description": "Resource not found"},
    },
)


class ModerateContentRequest(BaseModel):
    """Request model for content moderation."""

    content_id: str
    content_type: ContentType
    content_data: str
    persona_id: Optional[str] = None
    metadata: Optional[dict] = None


class ReviewRequest(BaseModel):
    """Request model for submitting a review."""

    action: ModerationAction
    notes: Optional[str] = None


class ModerationConfigUpdate(BaseModel):
    """Request model for updating moderation config."""

    auto_approve_threshold: Optional[float] = None
    auto_reject_threshold: Optional[float] = None
    human_review_threshold: Optional[float] = None
    max_queue_size: Optional[int] = None


def get_moderation_service(
    db: AsyncSession = Depends(get_db_session),
) -> ContentModerationPipeline:
    """Dependency injection for ContentModerationPipeline."""
    return ContentModerationPipeline(db)


# =============================================================================
# Content Analysis Endpoints
# =============================================================================


@router.post("/analyze", response_model=ModerationResult)
async def analyze_content(
    request: ModerateContentRequest,
    current_user: UserResponse = Depends(get_current_user),
    moderation_service: ContentModerationPipeline = Depends(get_moderation_service),
):
    """
    Analyze content through the moderation pipeline.

    Runs the content through automated moderation checks and returns
    the analysis result with recommended action.

    Args:
        request: Content moderation request

    Returns:
        ModerationResult with analysis and recommendation
    """
    try:
        result = await moderation_service.moderate_content(
            content_id=request.content_id,
            content_type=request.content_type,
            content_data=request.content_data,
            persona_id=request.persona_id,
            metadata=request.metadata,
        )

        logger.info(
            f"Content analyzed: {request.content_id} by user {current_user.username}"
        )

        return result

    except Exception as e:
        logger.error(f"Content moderation error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Content moderation failed",
        )


@router.post("/analyze/text")
async def analyze_text_content(
    text: str = Query(..., description="Text content to analyze"),
    content_id: Optional[str] = Query(None, description="Optional content ID"),
    current_user: UserResponse = Depends(get_current_user),
    moderation_service: ContentModerationPipeline = Depends(get_moderation_service),
):
    """
    Quick analysis endpoint for text content.

    Simplified endpoint for text-only analysis.

    Args:
        text: Text to analyze
        content_id: Optional content identifier

    Returns:
        ModerationResult for the text
    """
    result = await moderation_service.moderate_content(
        content_id=content_id or str(uuid.uuid4()),
        content_type=ContentType.TEXT,
        content_data=text,
    )

    return result


# =============================================================================
# Review Queue Endpoints
# =============================================================================


@router.get("/queue")
async def get_review_queue(
    status: str = Query("pending", description="Queue status filter"),
    limit: int = Query(50, ge=1, le=100, description="Max items to return"),
    offset: int = Query(0, ge=0, description="Offset for pagination"),
    current_user: UserResponse = Depends(get_current_user),
    moderation_service: ContentModerationPipeline = Depends(get_moderation_service),
):
    """
    Get items from the moderation review queue.

    Returns items pending human review with their analysis details.

    Args:
        status: Filter by queue status (pending, reviewed)
        limit: Maximum items to return
        offset: Pagination offset

    Returns:
        List of queue items
    """
    items = await moderation_service.get_review_queue(
        status=status,
        limit=limit,
        offset=offset,
    )

    return {
        "items": items,
        "count": len(items),
        "status": status,
        "offset": offset,
        "limit": limit,
    }


@router.post("/queue/{item_id}/review")
async def submit_review(
    item_id: str,
    review: ReviewRequest,
    current_user: UserResponse = Depends(get_current_user),
    moderation_service: ContentModerationPipeline = Depends(get_moderation_service),
):
    """
    Submit a human review for a queued item.

    Marks the item as reviewed with the specified action.

    Args:
        item_id: Queue item ID
        review: Review action and notes

    Returns:
        Success status
    """
    success = await moderation_service.review_content(
        queue_item_id=item_id,
        action=review.action,
        reviewer_id=str(current_user.id),
        notes=review.notes,
    )

    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Queue item not found",
        )

    logger.info(
        f"Review submitted for item {item_id} by {current_user.username}: {review.action}"
    )

    return {
        "success": True,
        "item_id": item_id,
        "action": review.action,
        "reviewer": current_user.username,
    }


@router.post("/queue/{item_id}/approve")
async def approve_content(
    item_id: str,
    notes: Optional[str] = Query(None, description="Optional review notes"),
    current_user: UserResponse = Depends(get_current_user),
    moderation_service: ContentModerationPipeline = Depends(get_moderation_service),
):
    """Quick approve endpoint for reviewed content."""
    success = await moderation_service.review_content(
        queue_item_id=item_id,
        action=ModerationAction.APPROVE,
        reviewer_id=str(current_user.id),
        notes=notes,
    )

    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Queue item not found",
        )

    return {"success": True, "action": "approved"}


@router.post("/queue/{item_id}/reject")
async def reject_content(
    item_id: str,
    notes: Optional[str] = Query(None, description="Reason for rejection"),
    current_user: UserResponse = Depends(get_current_user),
    moderation_service: ContentModerationPipeline = Depends(get_moderation_service),
):
    """Quick reject endpoint for reviewed content."""
    success = await moderation_service.review_content(
        queue_item_id=item_id,
        action=ModerationAction.REJECT,
        reviewer_id=str(current_user.id),
        notes=notes,
    )

    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Queue item not found",
        )

    return {"success": True, "action": "rejected"}


# =============================================================================
# Statistics and Configuration Endpoints
# =============================================================================


@router.get("/stats")
async def get_moderation_stats(
    current_user: UserResponse = Depends(get_current_user),
    moderation_service: ContentModerationPipeline = Depends(get_moderation_service),
):
    """
    Get moderation statistics.

    Returns overview of moderation queue status and metrics.

    Returns:
        Dict with moderation statistics
    """
    stats = await moderation_service.get_moderation_stats()
    return stats


@router.get("/categories")
async def list_moderation_categories():
    """
    List all moderation categories.

    Returns information about each moderation category
    that content can be flagged for.
    """
    return {
        "categories": [
            {
                "id": cat.value,
                "name": cat.name.replace("_", " ").title(),
                "description": _get_category_description(cat),
            }
            for cat in ModerationCategory
        ]
    }


@router.get("/actions")
async def list_moderation_actions():
    """
    List all available moderation actions.

    Returns information about each action that can be taken
    on flagged content.
    """
    return {
        "actions": [
            {
                "id": action.value,
                "name": action.name.replace("_", " ").title(),
                "description": _get_action_description(action),
            }
            for action in ModerationAction
        ]
    }


@router.get("/severity-levels")
async def list_severity_levels():
    """
    List all moderation severity levels.

    Returns information about severity classifications.
    """
    return {
        "levels": [
            {
                "id": level.value,
                "name": level.name.title(),
                "description": _get_severity_description(level),
            }
            for level in ModerationSeverity
        ]
    }


# =============================================================================
# Helper Functions
# =============================================================================


def _get_category_description(category: ModerationCategory) -> str:
    """Get description for a moderation category."""
    descriptions = {
        ModerationCategory.ADULT: "Adult or sexually explicit content",
        ModerationCategory.VIOLENCE: "Violent or graphic content",
        ModerationCategory.HATE_SPEECH: "Content promoting hate or discrimination",
        ModerationCategory.HARASSMENT: "Harassing or bullying content",
        ModerationCategory.SELF_HARM: "Content promoting self-harm",
        ModerationCategory.SPAM: "Spam or misleading content",
        ModerationCategory.MISINFORMATION: "False or misleading information",
        ModerationCategory.COPYRIGHT: "Potential copyright infringement",
        ModerationCategory.PERSONAL_INFO: "Exposed personal information",
        ModerationCategory.SAFE: "Content cleared by moderation",
    }
    return descriptions.get(category, "No description available")


def _get_action_description(action: ModerationAction) -> str:
    """Get description for a moderation action."""
    descriptions = {
        ModerationAction.APPROVE: "Allow content to be published",
        ModerationAction.REJECT: "Block content from being published",
        ModerationAction.FLAG_FOR_REVIEW: "Queue for human review",
        ModerationAction.WARN: "Allow with warning to user",
        ModerationAction.REMOVE: "Remove existing published content",
        ModerationAction.RESTRICT: "Limit content visibility",
    }
    return descriptions.get(action, "No description available")


def _get_severity_description(severity: ModerationSeverity) -> str:
    """Get description for a severity level."""
    descriptions = {
        ModerationSeverity.LOW: "Minor issue, may not require action",
        ModerationSeverity.MEDIUM: "Moderate concern, review recommended",
        ModerationSeverity.HIGH: "Serious violation, action required",
        ModerationSeverity.CRITICAL: "Severe violation, immediate action required",
    }
    return descriptions.get(severity, "No description available")
