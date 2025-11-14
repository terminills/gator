"""
ACD (Autonomous Continuous Development) Models

Database and API models for ACD context management system.
Enables autonomous content generation with contextual intelligence.
"""

import uuid
from datetime import datetime
from typing import Optional, Dict, Any, List
from enum import Enum

from pydantic import BaseModel, Field
from sqlalchemy import (
    Column,
    String,
    DateTime,
    Integer,
    Text,
    JSON,
    Boolean,
    ForeignKey,
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.sql import func

from backend.database.connection import Base


# Enums for ACD metadata
class AIStatus(str, Enum):
    """Implementation status of content generation phase."""

    IMPLEMENTED = "IMPLEMENTED"
    PARTIAL = "PARTIAL"
    NOT_STARTED = "NOT_STARTED"
    FIXED = "FIXED"
    DEPRECATED = "DEPRECATED"
    EXPERIMENTAL = "EXPERIMENTAL"


class AIComplexity(str, Enum):
    """Complexity level of content generation task."""

    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class AIConfidence(str, Enum):
    """Agent's confidence level in generated content."""

    CONFIDENT = "CONFIDENT"
    UNCERTAIN = "UNCERTAIN"
    HYPOTHESIS = "HYPOTHESIS"
    VALIDATED = "VALIDATED"
    EXPERIMENTAL = "EXPERIMENTAL"


class AIRequest(str, Enum):
    """Communication flag for agent requests."""

    REQUEST_FEEDBACK = "REQUEST_FEEDBACK"
    REQUEST_REVIEW = "REQUEST_REVIEW"
    WAITING_FOR_INPUT = "WAITING_FOR_INPUT"
    NEEDS_VALIDATION = "NEEDS_VALIDATION"
    NEEDS_APPROVAL = "NEEDS_APPROVAL"
    REQUEST_ASSISTANCE = "REQUEST_ASSISTANCE"


class AIState(str, Enum):
    """Current processing state of content generation."""

    PROCESSING = "PROCESSING"
    READY = "READY"
    DONE = "DONE"
    BLOCKED = "BLOCKED"
    PAUSED = "PAUSED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


class AIQueuePriority(str, Enum):
    """Priority level for content generation queue."""

    CRITICAL = "CRITICAL"
    HIGH = "HIGH"
    NORMAL = "NORMAL"
    LOW = "LOW"
    DEFERRED = "DEFERRED"


class AIQueueStatus(str, Enum):
    """Queue status for content generation task."""

    QUEUED = "QUEUED"
    ASSIGNED = "ASSIGNED"
    IN_PROGRESS = "IN_PROGRESS"
    REVIEW_PENDING = "REVIEW_PENDING"
    REVIEW_IN_PROGRESS = "REVIEW_IN_PROGRESS"
    APPROVED = "APPROVED"
    REJECTED = "REJECTED"
    COMPLETED = "COMPLETED"
    ABANDONED = "ABANDONED"


class AIValidation(str, Enum):
    """Validation status from reviewer."""

    ANALYZED = "ANALYZED"
    APPROVED = "APPROVED"
    CONDITIONALLY_APPROVED = "CONDITIONALLY_APPROVED"
    REJECTED = "REJECTED"
    PENDING = "PENDING"


class HandoffType(str, Enum):
    """Type of agent handoff."""

    ESCALATION = "ESCALATION"
    SPECIALIZATION = "SPECIALIZATION"
    LOAD_BALANCE = "LOAD_BALANCE"
    FAILURE = "FAILURE"
    COMPLETION = "COMPLETION"
    COLLABORATION = "COLLABORATION"


class HandoffStatus(str, Enum):
    """Handoff workflow state."""

    REQUESTED = "REQUESTED"
    ACCEPTED = "ACCEPTED"
    REJECTED = "REJECTED"
    COMPLETED = "COMPLETED"


class SkillLevel(str, Enum):
    """Required skill level for agent."""

    NOVICE = "NOVICE"
    INTERMEDIATE = "INTERMEDIATE"
    EXPERT = "EXPERT"
    SPECIALIST = "SPECIALIST"


class AIDomain(str, Enum):
    """
    Top-level domain classification for ACD contexts.
    
    Represents cognitive "cortical regions" - separates different types of work
    to enable clean cross-domain orchestration and prevent noisy correlations.
    """
    
    # Meta-level reasoning and orchestration
    METAREASONING = "METAREASONING"
    
    # Content generation and transformation
    CODE_GENERATION = "CODE_GENERATION"
    TEXT_GENERATION = "TEXT_GENERATION"
    IMAGE_GENERATION = "IMAGE_GENERATION"
    VIDEO_GENERATION = "VIDEO_GENERATION"
    AUDIO_GENERATION = "AUDIO_GENERATION"
    GRAPH_GENERATION = "GRAPH_GENERATION"
    DATA_TRANSFORMATION = "DATA_TRANSFORMATION"
    MODEL_GENERATION = "MODEL_GENERATION"
    MODEL_REFINEMENT = "MODEL_REFINEMENT"
    
    # Analysis and understanding
    COMPRESSION = "COMPRESSION"
    SUMMARIZATION = "SUMMARIZATION"
    PLANNING = "PLANNING"
    ANALYSIS = "ANALYSIS"
    
    # Multimodal semantics
    MULTIMODAL_SEMANTICS = "MULTIMODAL_SEMANTICS"
    
    # System operations
    SYSTEM_OPERATIONS = "SYSTEM_OPERATIONS"
    
    # Human interface
    HUMAN_INTERFACE = "HUMAN_INTERFACE"


class AISubdomain(str, Enum):
    """
    Sub-domain classification for fine-grained categorization.
    
    Provides detailed specialization within each domain for precise agent matching.
    """
    
    # METAREASONING subdomains
    ORCHESTRATION = "ORCHESTRATION"
    REASONING_ENGINE = "REASONING_ENGINE"
    CORRELATION_INDEX = "CORRELATION_INDEX"
    AGENT_SELECTION = "AGENT_SELECTION"
    COMPLEXITY_HANDLING = "COMPLEXITY_HANDLING"
    ERROR_RESOLUTION = "ERROR_RESOLUTION"
    RESCHEDULING = "RESCHEDULING"
    
    # CODE_GENERATION subdomains
    PYTHON = "PYTHON"
    JAVASCRIPT = "JAVASCRIPT"
    TYPESCRIPT = "TYPESCRIPT"
    SQL = "SQL"
    BACKEND = "BACKEND"
    FRONTEND = "FRONTEND"
    API = "API"
    TESTING = "TESTING"
    
    # TEXT_GENERATION subdomains
    CREATIVE_WRITING = "CREATIVE_WRITING"
    TECHNICAL_WRITING = "TECHNICAL_WRITING"
    SOCIAL_MEDIA = "SOCIAL_MEDIA"
    MARKETING = "MARKETING"
    DOCUMENTATION = "DOCUMENTATION"
    DIALOGUE = "DIALOGUE"
    
    # IMAGE_GENERATION subdomains
    PORTRAITS = "PORTRAITS"
    LANDSCAPES = "LANDSCAPES"
    ABSTRACT = "ABSTRACT"
    PHOTOREALISTIC = "PHOTOREALISTIC"
    STYLIZED = "STYLIZED"
    LOGO_DESIGN = "LOGO_DESIGN"
    IMAGE_EDITING = "IMAGE_EDITING"
    UPSCALING = "UPSCALING"
    
    # VIDEO_GENERATION subdomains
    SHORT_FORM = "SHORT_FORM"
    LONG_FORM = "LONG_FORM"
    ANIMATION = "ANIMATION"
    VIDEO_EDITING = "VIDEO_EDITING"
    EFFECTS = "EFFECTS"
    TRANSITIONS = "TRANSITIONS"
    
    # AUDIO_GENERATION subdomains
    VOICE = "VOICE"
    MUSIC = "MUSIC"
    SOUND_EFFECTS = "SOUND_EFFECTS"
    AUDIO_MIXING = "AUDIO_MIXING"
    
    # MULTIMODAL_SEMANTICS subdomains
    EMBEDDINGS = "EMBEDDINGS"
    ALIGNMENT = "ALIGNMENT"
    CAPTIONING = "CAPTIONING"
    SCENE_PARSING = "SCENE_PARSING"
    OCR = "OCR"
    CROSS_MODAL_INFERENCE = "CROSS_MODAL_INFERENCE"
    
    # SYSTEM_OPERATIONS subdomains
    FILE_OPERATIONS = "FILE_OPERATIONS"
    SCHEDULING = "SCHEDULING"
    PIPELINE_CONTROL = "PIPELINE_CONTROL"
    ENVIRONMENT = "ENVIRONMENT"
    MICROSERVICE = "MICROSERVICE"
    PLUGIN = "PLUGIN"
    MIDDLEWARE = "MIDDLEWARE"
    DATABASE = "DATABASE"
    MIGRATION = "MIGRATION"
    
    # HUMAN_INTERFACE subdomains
    UI_GENERATION = "UI_GENERATION"
    UX_DESIGN = "UX_DESIGN"
    FORM_VALIDATION = "FORM_VALIDATION"
    DASHBOARD = "DASHBOARD"
    DOCUMENTATION_UI = "DOCUMENTATION_UI"
    TUTORIAL = "TUTORIAL"
    EXPLANATION = "EXPLANATION"
    
    # PLANNING subdomains
    STRATEGIC = "STRATEGIC"
    TACTICAL = "TACTICAL"
    OPERATIONAL = "OPERATIONAL"
    
    # ANALYSIS subdomains
    DATA_ANALYSIS = "DATA_ANALYSIS"
    PERFORMANCE = "PERFORMANCE"
    QUALITY = "QUALITY"
    SECURITY = "SECURITY"


# Domain compatibility matrix - defines which domains can correlate
DOMAIN_COMPATIBILITY = {
    AIDomain.CODE_GENERATION: [
        AIDomain.TEXT_GENERATION,
        AIDomain.PLANNING,
        AIDomain.SYSTEM_OPERATIONS,
    ],
    AIDomain.TEXT_GENERATION: [
        AIDomain.CODE_GENERATION,
        AIDomain.HUMAN_INTERFACE,
        AIDomain.PLANNING,
        AIDomain.MULTIMODAL_SEMANTICS,
    ],
    AIDomain.IMAGE_GENERATION: [
        AIDomain.MULTIMODAL_SEMANTICS,
        AIDomain.VIDEO_GENERATION,
        AIDomain.HUMAN_INTERFACE,
    ],
    AIDomain.VIDEO_GENERATION: [
        AIDomain.IMAGE_GENERATION,
        AIDomain.AUDIO_GENERATION,
        AIDomain.MULTIMODAL_SEMANTICS,
    ],
    AIDomain.AUDIO_GENERATION: [
        AIDomain.VIDEO_GENERATION,
        AIDomain.MULTIMODAL_SEMANTICS,
    ],
    AIDomain.MULTIMODAL_SEMANTICS: [
        AIDomain.IMAGE_GENERATION,
        AIDomain.VIDEO_GENERATION,
        AIDomain.AUDIO_GENERATION,
        AIDomain.TEXT_GENERATION,
    ],
    AIDomain.SYSTEM_OPERATIONS: [
        AIDomain.PLANNING,
        AIDomain.CODE_GENERATION,
    ],
    AIDomain.HUMAN_INTERFACE: [
        AIDomain.TEXT_GENERATION,
        AIDomain.CODE_GENERATION,
        AIDomain.IMAGE_GENERATION,
    ],
    AIDomain.PLANNING: [
        AIDomain.CODE_GENERATION,
        AIDomain.TEXT_GENERATION,
        AIDomain.SYSTEM_OPERATIONS,
        AIDomain.ANALYSIS,
    ],
    AIDomain.ANALYSIS: [
        AIDomain.PLANNING,
        AIDomain.DATA_TRANSFORMATION,
        AIDomain.MULTIMODAL_SEMANTICS,
    ],
    AIDomain.METAREASONING: [
        # Meta-reasoning can work with all domains
        AIDomain.CODE_GENERATION,
        AIDomain.TEXT_GENERATION,
        AIDomain.IMAGE_GENERATION,
        AIDomain.VIDEO_GENERATION,
        AIDomain.AUDIO_GENERATION,
        AIDomain.MULTIMODAL_SEMANTICS,
        AIDomain.SYSTEM_OPERATIONS,
        AIDomain.HUMAN_INTERFACE,
        AIDomain.PLANNING,
        AIDomain.ANALYSIS,
        AIDomain.DATA_TRANSFORMATION,
        AIDomain.MODEL_GENERATION,
        AIDomain.MODEL_REFINEMENT,
    ],
}


# Database Models
class ACDContextModel(Base):
    """
    SQLAlchemy model for ACD context metadata.

    Stores comprehensive context for content generation tasks,
    enabling autonomous decision-making and continuous improvement.
    """

    __tablename__ = "acd_contexts"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, index=True)

    # Link to generation benchmark
    benchmark_id = Column(
        UUID(as_uuid=True),
        ForeignKey("generation_benchmarks.id", ondelete="CASCADE"),
        nullable=True,
        index=True,
    )

    # Link to content
    content_id = Column(
        UUID(as_uuid=True),
        ForeignKey("content.id", ondelete="CASCADE"),
        nullable=True,
        index=True,
    )

    # SCIS Metadata - Core fields
    ai_phase = Column(String(100), nullable=False, index=True)
    ai_status = Column(String(20), nullable=False, index=True)
    ai_complexity = Column(String(20), nullable=True)
    ai_note = Column(Text, nullable=True)
    ai_dependencies = Column(JSON, nullable=True)
    
    # Domain classification - cortical region classification
    ai_domain = Column(String(50), nullable=True, index=True)
    ai_subdomain = Column(String(50), nullable=True, index=True)

    # Version tracking
    ai_commit = Column(String(40), nullable=True)
    ai_commit_history = Column(JSON, nullable=True)
    ai_version = Column(String(50), nullable=True)
    ai_change = Column(Text, nullable=True)

    # Implementation details
    ai_pattern = Column(String(100), nullable=True)
    ai_strategy = Column(Text, nullable=True)
    ai_train_hash = Column(String(64), nullable=True)

    # Extended context
    ai_context = Column(JSON, nullable=True)
    ai_metadata = Column(JSON, nullable=True)

    # Error tracking
    compiler_err = Column(Text, nullable=True)
    runtime_err = Column(Text, nullable=True)
    fix_reason = Column(Text, nullable=True)
    human_override = Column(Text, nullable=True)

    # Agent assignment
    ai_assigned_to = Column(String(100), nullable=True, index=True)
    ai_assigned_by = Column(String(100), nullable=True)
    ai_assigned_at = Column(DateTime(timezone=True), nullable=True)
    ai_assignment_reason = Column(Text, nullable=True)
    ai_previous_assignee = Column(String(100), nullable=True)
    ai_assignment_history = Column(JSON, nullable=True)

    # Agent handoff
    ai_handoff_requested = Column(Boolean, default=False)
    ai_handoff_reason = Column(Text, nullable=True)
    ai_handoff_to = Column(String(100), nullable=True)
    ai_handoff_type = Column(String(20), nullable=True)
    ai_handoff_at = Column(DateTime(timezone=True), nullable=True)
    ai_handoff_notes = Column(Text, nullable=True)
    ai_handoff_status = Column(String(20), nullable=True)

    # Capability matching
    ai_required_capabilities = Column(JSON, nullable=True)
    ai_preferred_agent_type = Column(String(100), nullable=True)
    ai_agent_pool = Column(JSON, nullable=True)
    ai_skill_level_required = Column(String(20), nullable=True)

    # Distributed coordination
    ai_timeout = Column(Integer, nullable=True)
    ai_max_retries = Column(Integer, nullable=True)

    # Communication flags
    ai_confidence = Column(String(20), nullable=True, index=True)
    ai_request = Column(String(30), nullable=True)
    ai_state = Column(String(20), nullable=False, default="READY", index=True)
    ai_note_confidence = Column(Text, nullable=True)
    ai_request_from = Column(String(100), nullable=True)
    ai_note_request = Column(Text, nullable=True)

    # Queuing flags
    ai_queue_priority = Column(String(20), nullable=True, default="NORMAL", index=True)
    ai_queue_status = Column(String(30), nullable=True, default="QUEUED", index=True)
    ai_queue_reason = Column(Text, nullable=True)
    ai_started = Column(DateTime(timezone=True), nullable=True)
    ai_estimated_completion = Column(DateTime(timezone=True), nullable=True)

    # Validation (dual-agent)
    ai_validation = Column(String(20), nullable=True)
    ai_issues = Column(JSON, nullable=True)
    ai_suggestions = Column(JSON, nullable=True)
    ai_refinement = Column(String(20), nullable=True)
    ai_changes = Column(Text, nullable=True)
    ai_rationale = Column(Text, nullable=True)
    ai_validation_result = Column(String(20), nullable=True)
    ai_approval = Column(String(20), nullable=True)

    # Collaboration tracking
    ai_exchange_id = Column(String(50), nullable=True, index=True)
    ai_round = Column(Integer, nullable=True)

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


class ACDTraceArtifactModel(Base):
    """
    SQLAlchemy model for ACD trace artifacts.

    Stores diagnostic context for failures in content generation.
    """

    __tablename__ = "acd_trace_artifacts"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, index=True)

    # Session tracking
    session_id = Column(String(100), nullable=False, index=True)
    event_type = Column(String(30), nullable=False, index=True)

    # Error information
    error_code = Column(String(50), nullable=True)
    error_message = Column(Text, nullable=False)
    error_file = Column(String(500), nullable=True)
    error_line = Column(Integer, nullable=True)
    error_function = Column(String(200), nullable=True)

    # Context
    acd_context_id = Column(
        UUID(as_uuid=True),
        ForeignKey("acd_contexts.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )

    # Additional data
    stack_trace = Column(JSON, nullable=True)
    environment = Column(JSON, nullable=True)
    related_fixes = Column(JSON, nullable=True)

    # Timestamps
    timestamp = Column(
        DateTime(timezone=True), server_default=func.now(), nullable=False, index=True
    )


# Pydantic Models for API
class ACDContextCreate(BaseModel):
    """API model for creating ACD context."""

    benchmark_id: Optional[uuid.UUID] = None
    content_id: Optional[uuid.UUID] = None
    ai_phase: str = Field(description="Content generation phase")
    ai_status: AIStatus = Field(description="Implementation status")
    ai_complexity: Optional[AIComplexity] = None
    ai_note: Optional[str] = None
    ai_dependencies: Optional[List[str]] = None
    ai_domain: Optional[AIDomain] = None
    ai_subdomain: Optional[AISubdomain] = None
    ai_pattern: Optional[str] = None
    ai_strategy: Optional[str] = None
    ai_context: Optional[Dict[str, Any]] = None
    ai_metadata: Optional[Dict[str, Any]] = None
    ai_assigned_to: Optional[str] = None
    ai_confidence: Optional[AIConfidence] = None
    ai_state: AIState = Field(default=AIState.READY)
    ai_queue_priority: AIQueuePriority = Field(default=AIQueuePriority.NORMAL)
    ai_queue_status: AIQueueStatus = Field(default=AIQueueStatus.QUEUED)


class ACDContextUpdate(BaseModel):
    """API model for updating ACD context."""

    ai_status: Optional[AIStatus] = None
    ai_note: Optional[str] = None
    ai_domain: Optional[AIDomain] = None
    ai_subdomain: Optional[AISubdomain] = None
    ai_state: Optional[AIState] = None
    ai_confidence: Optional[AIConfidence] = None
    ai_validation: Optional[AIValidation] = None
    ai_issues: Optional[List[str]] = None
    ai_suggestions: Optional[List[str]] = None
    ai_changes: Optional[str] = None
    ai_rationale: Optional[str] = None
    ai_queue_status: Optional[AIQueueStatus] = None
    human_override: Optional[str] = None


class ACDContextResponse(BaseModel):
    """API model for ACD context responses."""

    id: uuid.UUID
    benchmark_id: Optional[uuid.UUID]
    content_id: Optional[uuid.UUID]
    ai_phase: str
    ai_status: str
    ai_complexity: Optional[str]
    ai_note: Optional[str]
    ai_dependencies: Optional[List[str]] = None
    ai_domain: Optional[str] = None
    ai_subdomain: Optional[str] = None
    ai_state: str
    ai_confidence: Optional[str]
    ai_queue_priority: Optional[str]
    ai_queue_status: Optional[str]
    ai_validation: Optional[str]
    ai_assigned_to: Optional[str]
    ai_assigned_at: Optional[datetime] = None
    ai_assignment_reason: Optional[str] = None
    ai_previous_assignee: Optional[str] = None
    ai_assignment_history: Optional[List[Dict[str, Any]]] = None
    ai_handoff_requested: Optional[bool] = None
    ai_handoff_to: Optional[str] = None
    ai_handoff_type: Optional[str] = None
    ai_handoff_reason: Optional[str] = None
    ai_handoff_status: Optional[str] = None
    ai_started: Optional[datetime] = None
    runtime_err: Optional[str] = None
    compiler_err: Optional[str] = None
    ai_context: Optional[Dict[str, Any]] = None
    ai_metadata: Optional[Dict[str, Any]] = None
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


class ACDTraceArtifactCreate(BaseModel):
    """API model for creating trace artifacts."""

    session_id: str
    event_type: str = Field(
        description="Type of event: build_error, runtime_error, test_failure, validation_error"
    )
    error_code: Optional[str] = None
    error_message: str
    error_file: Optional[str] = None
    error_line: Optional[int] = None
    error_function: Optional[str] = None
    acd_context_id: Optional[uuid.UUID] = None
    stack_trace: Optional[List[str]] = None
    environment: Optional[Dict[str, Any]] = None
    related_fixes: Optional[List[str]] = None


class ACDTraceArtifactResponse(BaseModel):
    """API model for trace artifact responses."""

    id: uuid.UUID
    session_id: str
    event_type: str
    error_message: str
    error_file: Optional[str]
    error_line: Optional[int]
    acd_context_id: Optional[uuid.UUID]
    timestamp: datetime

    model_config = {"from_attributes": True}


class ACDValidationReport(BaseModel):
    """API model for ACD validation reports."""

    metadata: Dict[str, Any] = Field(
        description="Report metadata including schema version and counts"
    )
    acd_contexts: List[ACDContextResponse] = Field(
        description="List of ACD contexts in the system"
    )
    errors: List[Dict[str, Any]] = Field(description="Validation errors")
    warnings: List[Dict[str, Any]] = Field(description="Validation warnings")


class ACDStats(BaseModel):
    """Statistics about ACD context usage."""

    total_contexts: int = 0
    by_phase: Dict[str, int] = {}
    by_status: Dict[str, int] = {}
    by_state: Dict[str, int] = {}
    by_confidence: Dict[str, int] = {}
    by_validation: Dict[str, int] = {}
    avg_completion_time: Optional[float] = None
    active_contexts: int = 0
    completed_contexts: int = 0
    failed_contexts: int = 0
