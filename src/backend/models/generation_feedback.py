"""
Generation Feedback Models

Database and API models for tracking AI generation benchmarks and human feedback
to enable learning and improvement of prompt enhancement and model selection.
"""

import uuid
from datetime import datetime
from enum import Enum
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field
from sqlalchemy import (
    JSON,
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    String,
    Text,
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.sql import func

from backend.database.connection import Base


class FeedbackRating(str, Enum):
    """Human feedback rating for generated content."""

    EXCELLENT = "excellent"  # 5 stars
    GOOD = "good"  # 4 stars
    ACCEPTABLE = "acceptable"  # 3 stars
    POOR = "poor"  # 2 stars
    UNACCEPTABLE = "unacceptable"  # 1 star


class GenerationBenchmarkModel(Base):
    """
    SQLAlchemy model for AI generation benchmarks and feedback.

    Tracks performance metrics, model selection decisions, and human feedback
    to enable continuous improvement of the system.
    """

    __tablename__ = "generation_benchmarks"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, index=True)

    # Content reference
    content_id = Column(
        UUID(as_uuid=True),
        ForeignKey("content.id", ondelete="CASCADE"),
        nullable=True,
        index=True,
    )

    # Generation parameters
    content_type = Column(
        String(20), nullable=False, index=True
    )  # image, text, video, etc.
    prompt = Column(Text, nullable=False)
    enhanced_prompt = Column(Text, nullable=True)  # If prompt was enhanced

    # Model selection
    model_selected = Column(String(100), nullable=False, index=True)
    model_provider = Column(String(50), nullable=False)  # local, openai, etc.
    selection_reasoning = Column(Text, nullable=True)  # Why this model was chosen
    available_models = Column(JSON, nullable=True)  # List of models that were available

    # Performance metrics
    generation_time_seconds = Column(Float, nullable=False)
    queue_time_seconds = Column(Float, nullable=True)
    total_time_seconds = Column(Float, nullable=False)

    # Resource usage
    gpu_memory_used_gb = Column(Float, nullable=True)
    peak_memory_gb = Column(Float, nullable=True)

    # Generation parameters used
    generation_params = Column(JSON, nullable=False)  # All kwargs passed to generation

    # Quality metrics
    quality_requested = Column(
        String(20), nullable=False
    )  # draft, standard, high, premium
    quality_score = Column(Float, nullable=True)  # Automated quality assessment (0-100)

    # Human feedback
    human_rating = Column(String(20), nullable=True, index=True)  # FeedbackRating enum
    human_feedback_text = Column(Text, nullable=True)
    feedback_timestamp = Column(DateTime(timezone=True), nullable=True)

    # Issues/errors
    had_errors = Column(Boolean, default=False, index=True)
    error_message = Column(Text, nullable=True)
    fallback_used = Column(Boolean, default=False, index=True)

    # Learning data
    prompt_keywords = Column(JSON, nullable=True)  # Extracted keywords for analysis
    content_features = Column(JSON, nullable=True)  # Extracted features for learning

    # ACD Integration - Context metadata for autonomous operation
    acd_context_id = Column(
        UUID(as_uuid=True),
        ForeignKey("acd_contexts.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    acd_phase = Column(
        String(100), nullable=True, index=True
    )  # Content generation phase
    acd_metadata = Column(JSON, nullable=True)  # Additional ACD metadata

    # Timestamps
    created_at = Column(
        DateTime(timezone=True), server_default=func.now(), nullable=False, index=True
    )
    updated_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )


class GenerationBenchmarkCreate(BaseModel):
    """API model for creating generation benchmark records."""

    content_id: Optional[uuid.UUID] = None
    content_type: str
    prompt: str
    enhanced_prompt: Optional[str] = None
    model_selected: str
    model_provider: str
    selection_reasoning: Optional[str] = None
    available_models: Optional[list] = None
    generation_time_seconds: float
    queue_time_seconds: Optional[float] = None
    total_time_seconds: float
    gpu_memory_used_gb: Optional[float] = None
    peak_memory_gb: Optional[float] = None
    generation_params: Dict[str, Any]
    quality_requested: str
    quality_score: Optional[float] = None
    had_errors: bool = False
    error_message: Optional[str] = None
    fallback_used: bool = False
    prompt_keywords: Optional[list] = None
    content_features: Optional[Dict[str, Any]] = None
    acd_context_id: Optional[uuid.UUID] = None
    acd_phase: Optional[str] = None
    acd_metadata: Optional[Dict[str, Any]] = None


class FeedbackSubmission(BaseModel):
    """API model for submitting human feedback on generated content."""

    benchmark_id: uuid.UUID = Field(description="ID of the benchmark record")
    rating: FeedbackRating = Field(description="Quality rating from human reviewer")
    feedback_text: Optional[str] = Field(
        default=None, description="Optional detailed feedback"
    )
    issues: Optional[list] = Field(
        default=None, description="List of specific issues identified"
    )


class GenerationBenchmarkResponse(BaseModel):
    """API model for benchmark responses."""

    id: uuid.UUID
    content_id: Optional[uuid.UUID]
    content_type: str
    prompt: str
    enhanced_prompt: Optional[str]
    model_selected: str
    model_provider: str
    selection_reasoning: Optional[str]
    generation_time_seconds: float
    total_time_seconds: float
    quality_requested: str
    quality_score: Optional[float]
    human_rating: Optional[str]
    human_feedback_text: Optional[str]
    had_errors: bool
    fallback_used: bool
    acd_context_id: Optional[uuid.UUID] = None
    acd_phase: Optional[str] = None
    created_at: datetime

    model_config = {"from_attributes": True}


class BenchmarkStats(BaseModel):
    """Statistics about generation performance and feedback."""

    total_generations: int = 0
    by_model: Dict[str, int] = {}
    by_rating: Dict[str, int] = {}
    avg_generation_time: float = 0.0
    avg_quality_score: Optional[float] = None
    success_rate: float = 0.0
    fallback_rate: float = 0.0
    feedback_count: int = 0
