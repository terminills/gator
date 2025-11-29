"""
Business Intelligence Models

Database and API models for tracking traffic, user retention, and revenue insights.
These models enable LLM reasoning about business performance and optimization.
"""

import uuid
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import (
    DECIMAL,
    JSON,
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.sql import func

from backend.database.connection import Base

# ============================================================
# Traffic Metrics Model
# ============================================================


class TrafficSource(str, Enum):
    """Source platforms for traffic."""

    INSTAGRAM = "instagram"
    TWITTER = "twitter"
    TIKTOK = "tiktok"
    ONLYFANS = "onlyfans"
    DIRECT = "direct"
    REFERRAL = "referral"
    ORGANIC = "organic"
    PAID = "paid"


class TrafficMetricsModel(Base):
    """
    Track user traffic patterns for conversion analysis.

    Enables LLM reasoning about:
    - Where users come from
    - User journey through the platform
    - Conversion funnels (traffic → DM → PPV)
    - Revenue attribution by source
    """

    __tablename__ = "traffic_metrics"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, index=True)

    # User and Persona
    user_id = Column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=True,
        index=True,
    )
    persona_id = Column(
        UUID(as_uuid=True),
        ForeignKey("personas.id", ondelete="CASCADE"),
        nullable=True,
        index=True,
    )

    # Session tracking
    session_id = Column(String(100), nullable=False, index=True)

    # Traffic sources
    source_platform = Column(
        String(50), nullable=False, index=True
    )  # TrafficSource enum
    referrer_url = Column(String(1000), nullable=True)
    utm_source = Column(String(100), nullable=True)
    utm_medium = Column(String(100), nullable=True)
    utm_campaign = Column(String(100), nullable=True)

    # User journey
    entry_page = Column(String(500), nullable=True)
    exit_page = Column(String(500), nullable=True)
    pages_visited = Column(JSON, nullable=True)  # List of pages with timestamps
    session_duration = Column(Integer, nullable=True)  # Seconds
    page_views = Column(Integer, default=1)

    # Device info
    device_type = Column(String(50), nullable=True)  # mobile, desktop, tablet
    browser = Column(String(100), nullable=True)
    os = Column(String(100), nullable=True)

    # Conversion tracking
    converted_to_dm = Column(Boolean, default=False)
    converted_to_ppv = Column(Boolean, default=False)
    converted_to_subscription = Column(Boolean, default=False)
    revenue_generated = Column(DECIMAL(10, 2), default=0)

    # Engagement during session
    messages_sent = Column(Integer, default=0)
    content_viewed = Column(Integer, default=0)
    ppv_offers_seen = Column(Integer, default=0)
    ppv_offers_accepted = Column(Integer, default=0)

    # Timestamps
    session_start = Column(DateTime(timezone=True), nullable=False)
    session_end = Column(DateTime(timezone=True), nullable=True)
    created_at = Column(
        DateTime(timezone=True), server_default=func.now(), nullable=False, index=True
    )


# ============================================================
# User Retention Model
# ============================================================


class ChurnRiskLevel(str, Enum):
    """Churn risk levels for user retention."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"
    CHURNED = "churned"


class UserRetentionModel(Base):
    """
    Track user retention and churn indicators.

    Enables LLM reasoning about:
    - User engagement trends
    - Churn prediction
    - Lifetime value optimization
    - Re-engagement opportunities
    """

    __tablename__ = "user_retention"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, index=True)

    user_id = Column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        unique=True,
        index=True,
    )

    # Engagement tracking
    first_interaction = Column(DateTime(timezone=True), nullable=False)
    last_interaction = Column(DateTime(timezone=True), nullable=False, index=True)
    total_sessions = Column(Integer, default=1)
    avg_session_duration = Column(Integer, nullable=True)  # Seconds
    total_time_spent = Column(Integer, default=0)  # Seconds

    # Activity patterns
    active_days_last_7 = Column(Integer, default=0)
    active_days_last_30 = Column(Integer, default=0)
    peak_activity_hour = Column(Integer, nullable=True)  # 0-23
    preferred_platform = Column(String(50), nullable=True)

    # Churn indicators
    days_since_last_active = Column(Integer, default=0, index=True)
    churn_risk_score = Column(Float, default=0.0)  # 0-1, calculated
    churn_risk_level = Column(String(20), default="low")  # ChurnRiskLevel enum
    churn_predicted = Column(Boolean, default=False, index=True)
    churn_predicted_date = Column(DateTime(timezone=True), nullable=True)

    # Value tracking
    lifetime_value = Column(DECIMAL(10, 2), default=0)
    ppv_purchases = Column(Integer, default=0)
    total_spent = Column(DECIMAL(10, 2), default=0)
    avg_purchase_value = Column(DECIMAL(10, 2), nullable=True)

    # Engagement depth
    total_messages_sent = Column(Integer, default=0)
    total_messages_received = Column(Integer, default=0)
    response_rate = Column(Float, nullable=True)  # % of messages responded to
    avg_response_time = Column(Integer, nullable=True)  # Seconds

    # Persona preferences
    favorite_personas = Column(
        JSON, nullable=True
    )  # List of persona IDs with interaction counts
    preferred_content_type = Column(String(50), nullable=True)

    # Re-engagement tracking
    re_engagement_attempts = Column(Integer, default=0)
    last_re_engagement_attempt = Column(DateTime(timezone=True), nullable=True)
    re_engagement_success = Column(Boolean, nullable=True)

    # Timestamps
    created_at = Column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )


# ============================================================
# Revenue Insight Model
# ============================================================


class RevenuePeriod(str, Enum):
    """Time periods for revenue aggregation."""

    HOURLY = "hourly"
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"


class RevenueInsightModel(Base):
    """
    Aggregate revenue insights for business intelligence.

    Enables LLM reasoning about:
    - Revenue trends over time
    - Per-persona performance
    - PPV optimization opportunities
    - Revenue forecasting
    """

    __tablename__ = "revenue_insights"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, index=True)

    persona_id = Column(
        UUID(as_uuid=True),
        ForeignKey("personas.id", ondelete="CASCADE"),
        nullable=True,
        index=True,
    )

    # Time period
    period_type = Column(String(20), nullable=False, index=True)  # RevenuePeriod enum
    period_start = Column(DateTime(timezone=True), nullable=False, index=True)
    period_end = Column(DateTime(timezone=True), nullable=False)

    # Revenue breakdown
    ppv_revenue = Column(DECIMAL(10, 2), default=0)
    subscription_revenue = Column(DECIMAL(10, 2), default=0)
    tips_revenue = Column(DECIMAL(10, 2), default=0)
    total_revenue = Column(DECIMAL(10, 2), default=0)

    # Transaction counts
    ppv_transactions = Column(Integer, default=0)
    subscription_transactions = Column(Integer, default=0)
    tip_transactions = Column(Integer, default=0)
    total_transactions = Column(Integer, default=0)

    # Performance metrics
    ppv_conversion_rate = Column(Float, nullable=True)  # % of offers accepted
    avg_ppv_price = Column(DECIMAL(10, 2), nullable=True)
    revenue_per_user = Column(DECIMAL(10, 2), nullable=True)
    revenue_per_message = Column(DECIMAL(10, 2), nullable=True)

    # Comparison to previous period
    revenue_change_percent = Column(Float, nullable=True)
    conversion_change_percent = Column(Float, nullable=True)

    # User metrics for period
    active_users = Column(Integer, default=0)
    new_users = Column(Integer, default=0)
    paying_users = Column(Integer, default=0)

    # Content performance
    top_performing_content_types = Column(JSON, nullable=True)
    top_performing_ppv_types = Column(JSON, nullable=True)

    # AI-generated recommendations
    optimization_suggestions = Column(JSON, nullable=True)
    predicted_next_period_revenue = Column(DECIMAL(10, 2), nullable=True)

    # Timestamps
    created_at = Column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )


# ============================================================
# Content Schedule Model
# ============================================================


class ContentScheduleModel(Base):
    """
    Scheduled content for automatic posting.

    Enables LLM-driven scheduling with:
    - Optimal time prediction
    - Multi-platform coordination
    - Performance feedback loop
    - Decision transparency
    """

    __tablename__ = "content_schedules"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, index=True)

    persona_id = Column(
        UUID(as_uuid=True),
        ForeignKey("personas.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    # ACD context link
    acd_context_id = Column(
        UUID(as_uuid=True),
        ForeignKey("acd_contexts.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )

    # Schedule details
    schedule_type = Column(String(50), nullable=False, index=True)  # ScheduleType enum
    scheduled_at = Column(DateTime(timezone=True), nullable=False, index=True)

    # Content
    content_type = Column(String(50), nullable=False)  # image, text, video, etc.
    content_data = Column(JSON, nullable=False)  # Topic, opinion, media, prompts, etc.
    platform = Column(
        String(50), nullable=False, index=True
    )  # instagram, twitter, etc.

    # Optimization
    optimization_goal = Column(String(50), nullable=True)  # OptimizationGoal enum
    decision_source = Column(String(50), nullable=False)  # ScheduleDecisionSource enum
    decision_reasoning = Column(Text, nullable=True)  # LLM's explanation

    # Predicted performance
    predicted_engagement = Column(Float, nullable=True)
    predicted_reach = Column(Integer, nullable=True)
    predicted_conversions = Column(Integer, nullable=True)
    confidence_score = Column(Float, nullable=True)  # 0-1

    # Constraints considered
    constraints = Column(JSON, nullable=True)  # Time windows, frequency limits, etc.
    competing_schedules = Column(JSON, nullable=True)  # Other posts scheduled nearby

    # Status
    status = Column(
        String(20), default="scheduled", index=True
    )  # scheduled, processing, posted, failed, cancelled
    error_message = Column(Text, nullable=True)
    retry_count = Column(Integer, default=0)

    # Execution results
    posted_at = Column(DateTime(timezone=True), nullable=True)
    # Note: result_post_id and result_content_id are UUIDs but without FK constraints
    # to avoid circular dependencies with tables that may not exist yet
    result_post_id = Column(UUID(as_uuid=True), nullable=True)
    result_content_id = Column(UUID(as_uuid=True), nullable=True)

    # Feedback data (filled after posting)
    actual_engagement = Column(Float, nullable=True)
    actual_reach = Column(Integer, nullable=True)
    actual_conversions = Column(Integer, nullable=True)
    performance_vs_predicted = Column(Float, nullable=True)  # ratio

    # Timestamps
    created_at = Column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )


# ============================================================
# Scheduling Feedback Model
# ============================================================


class SchedulingFeedbackModel(Base):
    """
    Learning data for improving scheduling decisions.

    Enables continuous improvement by tracking:
    - Prediction accuracy
    - What worked and what didn't
    - Context at scheduling time
    - AI-generated insights
    """

    __tablename__ = "scheduling_feedback"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, index=True)

    schedule_id = Column(
        UUID(as_uuid=True),
        ForeignKey("content_schedules.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    persona_id = Column(
        UUID(as_uuid=True),
        ForeignKey("personas.id", ondelete="CASCADE"),
        nullable=True,
        index=True,
    )

    # What was scheduled
    schedule_type = Column(String(50), nullable=False)
    scheduled_time = Column(DateTime(timezone=True), nullable=False)
    actual_post_time = Column(DateTime(timezone=True), nullable=True)
    time_drift_seconds = Column(Integer, nullable=True)  # Difference from scheduled

    # Platform and content
    platform = Column(String(50), nullable=False)
    content_type = Column(String(50), nullable=False)

    # Predicted vs actual performance
    predicted_engagement = Column(Float, nullable=True)
    actual_engagement = Column(Float, nullable=True)
    engagement_prediction_error = Column(Float, nullable=True)  # % error

    predicted_reach = Column(Integer, nullable=True)
    actual_reach = Column(Integer, nullable=True)
    reach_prediction_error = Column(Float, nullable=True)

    prediction_accuracy = Column(Float, nullable=True)  # Overall accuracy score

    # Context at scheduling time (for learning)
    system_state_snapshot = Column(JSON, nullable=True)  # Queue depth, GPU load, etc.
    audience_state_snapshot = Column(
        JSON, nullable=True
    )  # Active users, recent engagement
    competing_posts = Column(JSON, nullable=True)  # Other posts scheduled nearby
    time_of_day = Column(Integer, nullable=True)  # 0-23
    day_of_week = Column(Integer, nullable=True)  # 0-6

    # Learning insights (AI-generated)
    what_worked = Column(JSON, nullable=True)  # Factors that contributed to success
    what_failed = Column(JSON, nullable=True)  # Factors that hurt performance
    recommendations = Column(JSON, nullable=True)  # Suggestions for future

    # Decision source evaluation
    decision_source = Column(
        String(50), nullable=True
    )  # Was it LLM, rule-based, manual?
    decision_source_score = Column(
        Float, nullable=True
    )  # How well did the source perform?

    # Timestamps
    created_at = Column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )


# ============================================================
# Pydantic Models for API
# ============================================================


class TrafficMetricsResponse(BaseModel):
    """API response for traffic metrics."""

    id: uuid.UUID
    user_id: Optional[uuid.UUID]
    persona_id: Optional[uuid.UUID]
    session_id: str
    source_platform: str
    session_duration: Optional[int]
    page_views: int
    converted_to_dm: bool
    converted_to_ppv: bool
    revenue_generated: Decimal
    session_start: datetime

    model_config = ConfigDict(from_attributes=True)


class UserRetentionResponse(BaseModel):
    """API response for user retention."""

    id: uuid.UUID
    user_id: uuid.UUID
    first_interaction: datetime
    last_interaction: datetime
    total_sessions: int
    days_since_last_active: int
    churn_risk_score: float
    churn_risk_level: str
    lifetime_value: Decimal
    ppv_purchases: int

    model_config = ConfigDict(from_attributes=True)


class RevenueInsightResponse(BaseModel):
    """API response for revenue insights."""

    id: uuid.UUID
    persona_id: Optional[uuid.UUID]
    period_type: str
    period_start: datetime
    period_end: datetime
    total_revenue: Decimal
    ppv_revenue: Decimal
    ppv_conversion_rate: Optional[float]
    revenue_per_user: Optional[Decimal]
    optimization_suggestions: Optional[Dict[str, Any]]

    model_config = ConfigDict(from_attributes=True)


class ContentScheduleCreate(BaseModel):
    """API model for creating content schedules."""

    persona_id: uuid.UUID
    schedule_type: str
    scheduled_at: datetime
    content_type: str
    content_data: Dict[str, Any]
    platform: str
    optimization_goal: Optional[str] = None
    decision_source: str = "manual"
    decision_reasoning: Optional[str] = None
    constraints: Optional[Dict[str, Any]] = None


class ContentScheduleResponse(BaseModel):
    """API response for content schedules."""

    id: uuid.UUID
    persona_id: uuid.UUID
    schedule_type: str
    scheduled_at: datetime
    content_type: str
    platform: str
    status: str
    optimization_goal: Optional[str]
    decision_source: str
    predicted_engagement: Optional[float]
    actual_engagement: Optional[float]
    posted_at: Optional[datetime]

    model_config = ConfigDict(from_attributes=True)


class SchedulingFeedbackResponse(BaseModel):
    """API response for scheduling feedback."""

    id: uuid.UUID
    schedule_id: uuid.UUID
    scheduled_time: datetime
    actual_post_time: Optional[datetime]
    prediction_accuracy: Optional[float]
    what_worked: Optional[Dict[str, Any]]
    what_failed: Optional[Dict[str, Any]]
    recommendations: Optional[Dict[str, Any]]

    model_config = ConfigDict(from_attributes=True)
