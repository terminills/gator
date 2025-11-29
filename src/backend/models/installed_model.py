"""
Installed AI Model Models

Database models for tracking installed AI models (checkpoints, LoRAs, etc.)
with detailed metadata from CivitAI API including trigger words, descriptions,
and model parameters.
"""

import uuid
from datetime import datetime
from typing import Optional, List, Dict, Any
from enum import Enum

from pydantic import BaseModel, Field
from sqlalchemy import Column, String, DateTime, Boolean, Integer, Float, Text, JSON
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.sql import func

from backend.database.connection import Base


class ModelType(str, Enum):
    """Types of AI models."""
    CHECKPOINT = "Checkpoint"
    LORA = "LORA"
    TEXTUAL_INVERSION = "TextualInversion"
    HYPERNETWORK = "Hypernetwork"
    CONTROLNET = "Controlnet"
    VAE = "VAE"
    AESTHETIC_GRADIENT = "AestheticGradient"
    POSES = "Poses"
    OTHER = "Other"


class ModelSource(str, Enum):
    """Source of the model."""
    CIVITAI = "civitai"
    HUGGINGFACE = "huggingface"
    LOCAL = "local"
    CUSTOM = "custom"


class InstalledModelModel(Base):
    """
    SQLAlchemy model for installed AI models.
    
    Stores detailed information about installed models including
    CivitAI metadata, trigger words, and usage tracking.
    """

    __tablename__ = "installed_models"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, index=True)
    
    # Model identification
    name = Column(String(255), nullable=False, index=True)
    display_name = Column(String(255), nullable=True)
    model_type = Column(String(50), nullable=False, index=True)  # Checkpoint, LORA, etc.
    source = Column(String(50), nullable=False, default="local", index=True)  # civitai, huggingface, local
    
    # Local file info
    file_path = Column(String(1000), nullable=False, unique=True)
    file_name = Column(String(255), nullable=False)
    file_size_mb = Column(Float, nullable=True)
    file_hash = Column(String(128), nullable=True)  # SHA256 hash
    
    # CivitAI specific info
    civitai_model_id = Column(Integer, nullable=True, index=True)
    civitai_version_id = Column(Integer, nullable=True, index=True)
    civitai_version_name = Column(String(255), nullable=True)
    civitai_url = Column(String(500), nullable=True)
    
    # HuggingFace specific info
    huggingface_repo_id = Column(String(255), nullable=True, index=True)  # e.g., "stabilityai/stable-diffusion-xl-base-1.0"
    huggingface_revision = Column(String(100), nullable=True)  # Git revision/branch
    huggingface_filename = Column(String(255), nullable=True)  # Specific file in the repo
    huggingface_url = Column(String(500), nullable=True)
    
    # Model metadata
    description = Column(Text, nullable=True)
    base_model = Column(String(100), nullable=True, index=True)  # SD 1.5, SDXL 1.0, etc.
    
    # Trigger words (stored as JSON array)
    trigger_words = Column(JSON, nullable=False, default=list)  # ["keyword1", "keyword2"]
    trained_words = Column(JSON, nullable=True, default=list)  # Original trained words from CivitAI
    
    # Recommended settings
    recommended_weight = Column(Float, nullable=True, default=1.0)
    recommended_steps = Column(Integer, nullable=True)
    recommended_sampler = Column(String(100), nullable=True)
    recommended_cfg_scale = Column(Float, nullable=True)
    
    # Default prompts
    default_positive_prompt = Column(Text, nullable=True)
    default_negative_prompt = Column(Text, nullable=True)
    
    # Usage settings
    is_nsfw = Column(Boolean, default=False, nullable=False)
    is_active = Column(Boolean, default=True, nullable=False, index=True)
    usage_count = Column(Integer, default=0, nullable=False)
    last_used_at = Column(DateTime(timezone=True), nullable=True)
    
    # Additional metadata (flexible JSON for any extra data)
    extra_metadata = Column(JSON, nullable=True, default=dict)  # For any additional CivitAI data
    
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


class InstalledModelCreate(BaseModel):
    """API model for creating installed model entries."""
    
    name: str = Field(min_length=1, max_length=255, description="Model name")
    display_name: Optional[str] = Field(None, max_length=255, description="Display name")
    model_type: str = Field(description="Model type (Checkpoint, LORA, etc.)")
    source: str = Field(default="local", description="Source (civitai, huggingface, local)")
    
    file_path: str = Field(min_length=1, max_length=1000, description="Full path to model file")
    file_name: str = Field(min_length=1, max_length=255, description="File name")
    file_size_mb: Optional[float] = None
    file_hash: Optional[str] = Field(None, max_length=128)
    
    civitai_model_id: Optional[int] = None
    civitai_version_id: Optional[int] = None
    civitai_version_name: Optional[str] = None
    civitai_url: Optional[str] = None
    
    huggingface_repo_id: Optional[str] = None
    huggingface_revision: Optional[str] = None
    huggingface_filename: Optional[str] = None
    huggingface_url: Optional[str] = None
    
    description: Optional[str] = None
    base_model: Optional[str] = None
    
    trigger_words: List[str] = Field(default_factory=list)
    trained_words: Optional[List[str]] = Field(default_factory=list)
    
    recommended_weight: Optional[float] = 1.0
    recommended_steps: Optional[int] = None
    recommended_sampler: Optional[str] = None
    recommended_cfg_scale: Optional[float] = None
    
    default_positive_prompt: Optional[str] = None
    default_negative_prompt: Optional[str] = None
    
    is_nsfw: bool = False
    is_active: bool = True
    
    extra_metadata: Optional[Dict[str, Any]] = None


class InstalledModelUpdate(BaseModel):
    """API model for updating installed model entries."""
    
    display_name: Optional[str] = None
    model_type: Optional[str] = None
    
    description: Optional[str] = None
    base_model: Optional[str] = None
    
    trigger_words: Optional[List[str]] = None
    trained_words: Optional[List[str]] = None
    
    recommended_weight: Optional[float] = None
    recommended_steps: Optional[int] = None
    recommended_sampler: Optional[str] = None
    recommended_cfg_scale: Optional[float] = None
    
    default_positive_prompt: Optional[str] = None
    default_negative_prompt: Optional[str] = None
    
    is_nsfw: Optional[bool] = None
    is_active: Optional[bool] = None
    
    extra_metadata: Optional[Dict[str, Any]] = None


class InstalledModelResponse(BaseModel):
    """API response model for installed models."""
    
    id: str
    name: str
    display_name: Optional[str]
    model_type: str
    source: str
    
    file_path: str
    file_name: str
    file_size_mb: Optional[float]
    file_hash: Optional[str]
    
    civitai_model_id: Optional[int]
    civitai_version_id: Optional[int]
    civitai_version_name: Optional[str]
    civitai_url: Optional[str]
    
    huggingface_repo_id: Optional[str]
    huggingface_revision: Optional[str]
    huggingface_filename: Optional[str]
    huggingface_url: Optional[str]
    
    description: Optional[str]
    base_model: Optional[str]
    
    trigger_words: List[str]
    trained_words: Optional[List[str]]
    
    recommended_weight: Optional[float]
    recommended_steps: Optional[int]
    recommended_sampler: Optional[str]
    recommended_cfg_scale: Optional[float]
    
    default_positive_prompt: Optional[str]
    default_negative_prompt: Optional[str]
    
    is_nsfw: bool
    is_active: bool
    usage_count: int
    last_used_at: Optional[datetime]
    
    extra_metadata: Optional[Dict[str, Any]]
    
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class TriggerWordResponse(BaseModel):
    """Response model for aggregated trigger words."""
    
    trigger_word: str
    source: str  # "model" or "persona" or "custom"
    models: List[str]  # List of model names that use this trigger
    usage_count: int = 0


class ModelsByTriggerResponse(BaseModel):
    """Response model for models matching a trigger word."""
    
    trigger_word: str
    models: List[InstalledModelResponse]
    total_count: int
