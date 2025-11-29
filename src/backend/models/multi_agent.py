"""
Multi-Agent Ecosystem Models

Database and API models for the multi-agent system including specialized
agent types, capabilities, and coordination.
"""

import uuid
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field
from sqlalchemy import (
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


class AgentType(str, Enum):
    """Type of specialized agent."""

    GENERATOR = "generator"
    REVIEWER = "reviewer"
    OPTIMIZER = "optimizer"
    COORDINATOR = "coordinator"
    ANALYZER = "analyzer"
    CUSTOM = "custom"


class AgentStatus(str, Enum):
    """Agent availability status."""

    ACTIVE = "active"
    IDLE = "idle"
    BUSY = "busy"
    OFFLINE = "offline"
    MAINTENANCE = "maintenance"


class TaskPriority(str, Enum):
    """Priority level for tasks."""

    CRITICAL = "critical"
    HIGH = "high"
    NORMAL = "normal"
    LOW = "low"


class TaskStatus(str, Enum):
    """Status of agent tasks."""

    PENDING = "pending"
    ASSIGNED = "assigned"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


# Database Models
class AgentModel(Base):
    """
    SQLAlchemy model for agent registry.

    Stores agent configurations, capabilities, and status.
    """

    __tablename__ = "agents"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, index=True)

    # Basic info
    agent_name = Column(String(100), nullable=False, unique=True, index=True)
    agent_type = Column(String(20), nullable=False, index=True)
    version = Column(String(20), nullable=False, default="1.0.0")
    description = Column(Text, nullable=True)

    # Status
    status = Column(String(20), nullable=False, default="idle", index=True)
    last_heartbeat = Column(DateTime(timezone=True), nullable=True)

    # Capabilities
    capabilities = Column(JSON, nullable=True)  # List of capability strings
    specializations = Column(JSON, nullable=True)  # Specific specialization areas
    supported_tasks = Column(JSON, nullable=True)  # Task types agent can handle

    # Performance metrics
    tasks_completed = Column(Integer, nullable=False, default=0)
    tasks_failed = Column(Integer, nullable=False, default=0)
    success_rate = Column(Float, nullable=False, default=0.0)
    average_completion_time = Column(Float, nullable=True)  # In seconds

    # Load balancing
    current_load = Column(Integer, nullable=False, default=0)
    max_concurrent_tasks = Column(Integer, nullable=False, default=5)

    # Configuration
    config = Column(JSON, nullable=True)

    # Plugin info (if agent is a plugin)
    is_plugin = Column(Boolean, default=False)
    plugin_source = Column(String(500), nullable=True)
    plugin_author = Column(String(100), nullable=True)

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


class AgentTaskModel(Base):
    """
    SQLAlchemy model for agent tasks.

    Tracks work items assigned to agents.
    """

    __tablename__ = "agent_tasks"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, index=True)

    # Task info
    task_type = Column(String(50), nullable=False, index=True)
    task_name = Column(String(200), nullable=False)
    description = Column(Text, nullable=True)
    priority = Column(String(20), nullable=False, default="normal", index=True)

    # Assignment
    agent_id = Column(
        UUID(as_uuid=True),
        ForeignKey("agents.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    assigned_by = Column(String(100), nullable=True)  # Coordinator or system

    # Status
    status = Column(String(20), nullable=False, default="pending", index=True)

    # Task data
    input_data = Column(JSON, nullable=True)
    output_data = Column(JSON, nullable=True)
    error_message = Column(Text, nullable=True)

    # ACD context linkage
    acd_context_id = Column(
        UUID(as_uuid=True),
        ForeignKey("acd_contexts.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )

    # Timing
    started_at = Column(DateTime(timezone=True), nullable=True)
    completed_at = Column(DateTime(timezone=True), nullable=True)
    deadline = Column(DateTime(timezone=True), nullable=True)

    # Retry management
    retry_count = Column(Integer, nullable=False, default=0)
    max_retries = Column(Integer, nullable=False, default=3)

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


class AgentCommunicationModel(Base):
    """
    SQLAlchemy model for agent-to-agent communication.

    Tracks messages and coordination between agents.
    """

    __tablename__ = "agent_communications"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, index=True)

    # Communication parties
    from_agent_id = Column(
        UUID(as_uuid=True),
        ForeignKey("agents.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    to_agent_id = Column(
        UUID(as_uuid=True),
        ForeignKey("agents.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    # Message info
    message_type = Column(String(50), nullable=False, index=True)
    subject = Column(String(200), nullable=True)
    message_body = Column(JSON, nullable=True)

    # Related task
    task_id = Column(
        UUID(as_uuid=True),
        ForeignKey("agent_tasks.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )

    # Status
    delivered = Column(Boolean, default=False)
    read = Column(Boolean, default=False)
    replied = Column(Boolean, default=False)

    # Timestamps
    created_at = Column(
        DateTime(timezone=True), server_default=func.now(), nullable=False, index=True
    )
    delivered_at = Column(DateTime(timezone=True), nullable=True)
    read_at = Column(DateTime(timezone=True), nullable=True)


# Pydantic Models for API
class AgentCapability(BaseModel):
    """Agent capability specification."""

    capability_name: str
    description: str
    parameters: Optional[Dict[str, Any]] = None


class AgentCreate(BaseModel):
    """API model for creating an agent."""

    agent_name: str = Field(description="Unique agent name")
    agent_type: AgentType
    version: str = Field(default="1.0.0")
    description: Optional[str] = None
    capabilities: List[str] = Field(default_factory=list)
    specializations: List[str] = Field(default_factory=list)
    supported_tasks: List[str] = Field(default_factory=list)
    max_concurrent_tasks: int = Field(default=5, ge=1, le=100)
    config: Optional[Dict[str, Any]] = None
    is_plugin: bool = False
    plugin_source: Optional[str] = None
    plugin_author: Optional[str] = None


class AgentUpdate(BaseModel):
    """API model for updating an agent."""

    status: Optional[AgentStatus] = None
    capabilities: Optional[List[str]] = None
    config: Optional[Dict[str, Any]] = None
    current_load: Optional[int] = None


class AgentResponse(BaseModel):
    """API model for agent responses."""

    id: uuid.UUID
    agent_name: str
    agent_type: str
    version: str
    description: Optional[str]
    status: str
    capabilities: Optional[List[str]]
    specializations: Optional[List[str]]
    tasks_completed: int
    tasks_failed: int
    success_rate: float
    current_load: int
    max_concurrent_tasks: int
    is_plugin: bool
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


class AgentTaskCreate(BaseModel):
    """API model for creating a task."""

    task_type: str
    task_name: str
    description: Optional[str] = None
    priority: TaskPriority = TaskPriority.NORMAL
    input_data: Optional[Dict[str, Any]] = None
    acd_context_id: Optional[uuid.UUID] = None
    deadline: Optional[datetime] = None
    max_retries: int = Field(default=3, ge=0, le=10)


class AgentTaskResponse(BaseModel):
    """API model for task responses."""

    id: uuid.UUID
    task_type: str
    task_name: str
    priority: str
    status: str
    agent_id: Optional[uuid.UUID]
    input_data: Optional[Dict[str, Any]]
    output_data: Optional[Dict[str, Any]]
    error_message: Optional[str]
    started_at: Optional[datetime]
    completed_at: Optional[datetime]
    created_at: datetime

    model_config = {"from_attributes": True}


class AgentCommunicationCreate(BaseModel):
    """API model for creating agent communication."""

    from_agent_id: uuid.UUID
    to_agent_id: uuid.UUID
    message_type: str
    subject: Optional[str] = None
    message_body: Optional[Dict[str, Any]] = None
    task_id: Optional[uuid.UUID] = None


class AgentCommunicationResponse(BaseModel):
    """API model for communication responses."""

    id: uuid.UUID
    from_agent_id: uuid.UUID
    to_agent_id: uuid.UUID
    message_type: str
    subject: Optional[str]
    delivered: bool
    read: bool
    created_at: datetime

    model_config = {"from_attributes": True}


class AgentRoutingRequest(BaseModel):
    """Request for automatic agent routing."""

    task_type: str
    required_capabilities: List[str]
    priority: TaskPriority = TaskPriority.NORMAL
    preferred_agent_type: Optional[AgentType] = None
    min_success_rate: float = Field(default=0.7, ge=0.0, le=1.0)


class AgentRoutingResponse(BaseModel):
    """Response from agent routing."""

    selected_agent_id: Optional[uuid.UUID]
    selected_agent_name: Optional[str]
    routing_reason: str
    confidence: float
    alternative_agents: List[Dict[str, Any]] = Field(default_factory=list)


class AgentMarketplaceEntry(BaseModel):
    """Entry in the agent marketplace."""

    agent_id: uuid.UUID
    agent_name: str
    agent_type: str
    version: str
    author: str
    description: str
    capabilities: List[str]
    rating: float = Field(ge=0.0, le=5.0)
    download_count: int = 0
    price: float = Field(default=0.0, ge=0.0)  # 0 for free
    source_url: str
    documentation_url: Optional[str] = None
    published_at: datetime
