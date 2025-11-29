"""
Database Models Package

Contains all Pydantic and SQLAlchemy models for the Gator AI platform.
"""

from .persona import PersonaModel, PersonaCreate, PersonaResponse, PersonaUpdate
from .user import UserModel, UserCreate, UserResponse, UserUpdate
from .conversation import ConversationModel, ConversationCreate, ConversationResponse
from .message import MessageModel, MessageCreate, MessageResponse
from .ppv_offer import PPVOfferModel, PPVOfferCreate, PPVOfferResponse
from .content import (
    ContentModel,
    ContentCreate,
    ContentResponse,
    ContentUpdate,
    GenerationRequest,
    ContentListResponse,
    ContentStats,
    ContentType,
    ContentRating,
    ModerationStatus,
)
from .feed import (
    RSSFeedModel,
    FeedItemModel,
    RSSFeedCreate,
    RSSFeedResponse,
    RSSFeedUpdate,
    FeedItemResponse,
    FeedItemListResponse,
)
from .generation_feedback import (
    GenerationBenchmarkModel,
    GenerationBenchmarkCreate,
    GenerationBenchmarkResponse,
    FeedbackSubmission,
    FeedbackRating,
    BenchmarkStats,
)
from .acd import (
    ACDContextModel,
    ACDTraceArtifactModel,
    ACDContextCreate,
    ACDContextUpdate,
    ACDContextResponse,
    ACDTraceArtifactCreate,
    ACDTraceArtifactResponse,
    ACDValidationReport,
    ACDStats,
    AIStatus,
    AIComplexity,
    AIConfidence,
    AIRequest,
    AIState,
    AIQueuePriority,
    AIQueueStatus,
    AIValidation,
    HandoffType,
    HandoffStatus,
    SkillLevel,
    # Scheduling enums
    ScheduleType,
    ScheduleOptimizationGoal,
    ScheduleDecisionSource,
    ScheduleFeedbackType,
)

# Business Intelligence models
from .business_intelligence import (
    TrafficMetricsModel,
    UserRetentionModel,
    RevenueInsightModel,
    ContentScheduleModel,
    SchedulingFeedbackModel,
    TrafficSource,
    ChurnRiskLevel,
    RevenuePeriod,
    TrafficMetricsResponse,
    UserRetentionResponse,
    RevenueInsightResponse,
    ContentScheduleCreate,
    ContentScheduleResponse,
    SchedulingFeedbackResponse,
)

# Installed Model models
from .installed_model import (
    InstalledModelModel,
    InstalledModelCreate,
    InstalledModelUpdate,
    InstalledModelResponse,
    TriggerWordResponse,
    ModelsByTriggerResponse,
    ModelType,
    ModelSource,
)

__all__ = [
    "PersonaModel",
    "PersonaCreate",
    "PersonaResponse",
    "PersonaUpdate",
    "UserModel",
    "UserCreate",
    "UserResponse",
    "UserUpdate",
    "ConversationModel",
    "ConversationCreate",
    "ConversationResponse",
    "MessageModel",
    "MessageCreate",
    "MessageResponse",
    "PPVOfferModel",
    "PPVOfferCreate",
    "PPVOfferResponse",
    # Content models
    "ContentModel",
    "ContentCreate",
    "ContentResponse",
    "ContentUpdate",
    "GenerationRequest",
    "ContentListResponse",
    "ContentStats",
    "ContentType",
    "ContentRating",
    "ModerationStatus",
    # Feed models
    "RSSFeedModel",
    "FeedItemModel",
    "RSSFeedCreate",
    "RSSFeedResponse",
    "RSSFeedUpdate",
    "FeedItemResponse",
    "FeedItemListResponse",
    # Generation feedback models
    "GenerationBenchmarkModel",
    "GenerationBenchmarkCreate",
    "GenerationBenchmarkResponse",
    "FeedbackSubmission",
    "FeedbackRating",
    "BenchmarkStats",
    # ACD models
    "ACDContextModel",
    "ACDTraceArtifactModel",
    "ACDContextCreate",
    "ACDContextUpdate",
    "ACDContextResponse",
    "ACDTraceArtifactCreate",
    "ACDTraceArtifactResponse",
    "ACDValidationReport",
    "ACDStats",
    "AIStatus",
    "AIComplexity",
    "AIConfidence",
    "AIRequest",
    "AIState",
    "AIQueuePriority",
    "AIQueueStatus",
    "AIValidation",
    "HandoffType",
    "HandoffStatus",
    "SkillLevel",
    # Scheduling enums
    "ScheduleType",
    "ScheduleOptimizationGoal",
    "ScheduleDecisionSource",
    "ScheduleFeedbackType",
    # Business Intelligence models
    "TrafficMetricsModel",
    "UserRetentionModel",
    "RevenueInsightModel",
    "ContentScheduleModel",
    "SchedulingFeedbackModel",
    "TrafficSource",
    "ChurnRiskLevel",
    "RevenuePeriod",
    "TrafficMetricsResponse",
    "UserRetentionResponse",
    "RevenueInsightResponse",
    "ContentScheduleCreate",
    "ContentScheduleResponse",
    "SchedulingFeedbackResponse",
    # Installed Model models
    "InstalledModelModel",
    "InstalledModelCreate",
    "InstalledModelUpdate",
    "InstalledModelResponse",
    "TriggerWordResponse",
    "ModelsByTriggerResponse",
    "ModelType",
    "ModelSource",
]
