"""
ACD Schema Exchange System

AI_PHASE: IMPLEMENTING
AI_STATUS: IN_PROGRESS
AI_PATTERN: ACD_SCHEMA_EXCHANGE
AI_STRATEGY: Enable cross-system ACD schema import/export and correlation
AI_NOTE: Allows Johnny to pull ACD specs from other systems and understand them
AI_CONTEXT: {
    "scope": "acd_schema_exchange",
    "features": ["import", "export", "validation", "transformation", "correlation"],
    "cross_system": true
}

This module provides the ACD Schema Exchange system that enables:
- Importing ACD specifications from external systems
- Exporting Johnny's ACD contexts in standard formats
- Validating external schemas against ACD standards
- Transforming between different ACD schema versions
- Cross-correlating ACD data across multiple systems/domains
"""

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Union
from uuid import UUID, uuid4

from sqlalchemy import and_, select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.config.logging import get_logger
from backend.models.acd import (
    ACDContextCreate,
    ACDContextModel,
    ACDContextResponse,
    AIDomain,
    AIState,
    AIStatus,
    GenerationRating,
    MemoryType,
    MisgenerationTag,
)

logger = get_logger(__name__)


# ============================================================
# Schema Version and Format Definitions
# ============================================================


class ACDSchemaVersion(str, Enum):
    """Supported ACD schema versions."""

    V1_0 = "1.0"
    V1_1 = "1.1"
    V2_0 = "2.0"
    LATEST = "2.0"


class ACDExportFormat(str, Enum):
    """Supported export formats."""

    JSON = "json"
    YAML = "yaml"
    JSONLD = "jsonld"  # JSON-LD for linked data
    NDJSON = "ndjson"  # Newline-delimited JSON for streaming


class ACDSystemType(str, Enum):
    """Types of external systems that can exchange ACD data."""

    JOHNNY = "johnny"
    CONTENT_GENERATION = "content_generation"
    IMAGE_GENERATION = "image_generation"
    VIDEO_GENERATION = "video_generation"
    TEXT_GENERATION = "text_generation"
    AUDIO_GENERATION = "audio_generation"
    MULTIMODAL = "multimodal"
    ANALYTICS = "analytics"
    ORCHESTRATION = "orchestration"
    EXTERNAL = "external"


class ValidationSeverity(str, Enum):
    """Severity levels for validation issues."""

    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


# ============================================================
# Data Classes for Schema Exchange
# ============================================================


@dataclass
class ACDSchemaMetadata:
    """Metadata about an ACD schema."""

    version: str
    system_type: str
    system_id: str
    created_at: str
    schema_hash: str
    capabilities: List[str] = field(default_factory=list)
    domain_support: List[str] = field(default_factory=list)
    extensions: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ValidationIssue:
    """A validation issue found during schema validation."""

    field: str
    message: str
    severity: str
    suggested_fix: Optional[str] = None
    context: Optional[Dict[str, Any]] = None


@dataclass
class ValidationResult:
    """Result of schema validation."""

    is_valid: bool
    issues: List[ValidationIssue] = field(default_factory=list)
    schema_version: Optional[str] = None
    compatibility_score: float = 0.0
    transformations_needed: List[str] = field(default_factory=list)


@dataclass
class SchemaTransformation:
    """A transformation to apply to a schema."""

    source_version: str
    target_version: str
    field_mappings: Dict[str, str] = field(default_factory=dict)
    value_transformations: Dict[str, Any] = field(default_factory=dict)
    additions: Dict[str, Any] = field(default_factory=dict)
    removals: List[str] = field(default_factory=list)


@dataclass
class ExportedSchema:
    """An exported ACD schema package."""

    metadata: ACDSchemaMetadata
    contexts: List[Dict[str, Any]]
    relationships: List[Dict[str, Any]] = field(default_factory=list)
    correlation_hints: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ImportResult:
    """Result of importing an external schema."""

    success: bool
    contexts_imported: int = 0
    contexts_skipped: int = 0
    contexts_failed: int = 0
    validation_result: Optional[ValidationResult] = None
    transformations_applied: List[str] = field(default_factory=list)
    imported_ids: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)


@dataclass
class CrossCorrelation:
    """Cross-system correlation result."""

    source_system: str
    target_system: str
    source_context_id: str
    target_context_id: str
    correlation_type: str
    correlation_score: float
    shared_attributes: Dict[str, Any] = field(default_factory=dict)
    insights: List[str] = field(default_factory=list)


# ============================================================
# Schema Transformation Maps
# ============================================================

# Field mappings between schema versions
SCHEMA_TRANSFORMATIONS: Dict[str, SchemaTransformation] = {
    "1.0_to_1.1": SchemaTransformation(
        source_version="1.0",
        target_version="1.1",
        field_mappings={
            "ai_status": "ai_status",
            "ai_phase": "ai_phase",
        },
        additions={
            "ai_domain": None,
            "ai_subdomain": None,
        },
    ),
    "1.1_to_2.0": SchemaTransformation(
        source_version="1.1",
        target_version="2.0",
        field_mappings={
            "ai_status": "ai_status",
            "ai_phase": "ai_phase",
            "ai_domain": "ai_domain",
            "ai_subdomain": "ai_subdomain",
        },
        additions={
            "hil_rating": None,
            "memory_type": None,
            "learning_weight": 1.0,
            "outcome_score": None,
            "ai_state": "READY",  # Default state for v2.0
        },
    ),
}

# Required fields by schema version
REQUIRED_FIELDS: Dict[str, List[str]] = {
    "1.0": ["ai_phase", "ai_status"],
    "1.1": ["ai_phase", "ai_status"],
    "2.0": ["ai_phase", "ai_status", "ai_state"],
}

# Domain mappings for external systems
EXTERNAL_DOMAIN_MAPPINGS: Dict[str, Dict[str, str]] = {
    "stable_diffusion": {
        "txt2img": AIDomain.IMAGE_GENERATION.value,
        "img2img": AIDomain.IMAGE_GENERATION.value,
        "inpainting": AIDomain.IMAGE_GENERATION.value,
    },
    "llm": {
        "completion": AIDomain.TEXT_GENERATION.value,
        "chat": AIDomain.TEXT_GENERATION.value,
        "code": AIDomain.CODE_GENERATION.value,
    },
    "video": {
        "generation": AIDomain.VIDEO_GENERATION.value,
        "editing": AIDomain.VIDEO_GENERATION.value,
    },
}


# ============================================================
# ACD Schema Exchange Service
# ============================================================


class ACDSchemaExchange:
    """
    Service for importing, exporting, validating, and transforming ACD schemas.

    Enables cross-system ACD data exchange with:
    - Schema version detection and transformation
    - Validation against ACD standards
    - Export in multiple formats
    - Cross-correlation of ACD contexts
    """

    # Current schema version
    CURRENT_VERSION = ACDSchemaVersion.V2_0.value

    # System identifier
    SYSTEM_ID = "gator_acd"
    SYSTEM_TYPE = ACDSystemType.CONTENT_GENERATION.value

    def __init__(self, db_session: AsyncSession):
        """
        Initialize the schema exchange service.

        Args:
            db_session: Database session for persistence
        """
        self.db = db_session

    # ============================================================
    # Schema Validation
    # ============================================================

    def validate_schema(
        self,
        data: Dict[str, Any],
        expected_version: Optional[str] = None,
    ) -> ValidationResult:
        """
        Validate an ACD schema against standards.

        Args:
            data: Schema data to validate
            expected_version: Expected schema version (auto-detect if None)

        Returns:
            ValidationResult with validation details
        """
        issues: List[ValidationIssue] = []
        detected_version = self._detect_schema_version(data)

        if expected_version and detected_version != expected_version:
            issues.append(
                ValidationIssue(
                    field="version",
                    message=f"Expected version {expected_version}, detected {detected_version}",
                    severity=ValidationSeverity.WARNING.value,
                )
            )

        # Check required fields
        version_to_check = expected_version or detected_version or "2.0"
        required = REQUIRED_FIELDS.get(version_to_check, REQUIRED_FIELDS["2.0"])

        for field_name in required:
            if field_name not in data or data[field_name] is None:
                issues.append(
                    ValidationIssue(
                        field=field_name,
                        message=f"Required field '{field_name}' is missing",
                        severity=ValidationSeverity.ERROR.value,
                        suggested_fix=f"Add '{field_name}' field with appropriate value",
                    )
                )

        # Validate enum fields
        enum_validations = [
            ("ai_status", [s.value for s in AIStatus]),
            ("ai_state", [s.value for s in AIState]),
            ("ai_domain", [d.value for d in AIDomain] + [None]),
            ("memory_type", [m.value for m in MemoryType] + [None]),
        ]

        for field_name, valid_values in enum_validations:
            if field_name in data and data[field_name] is not None:
                if data[field_name] not in valid_values:
                    issues.append(
                        ValidationIssue(
                            field=field_name,
                            message=f"Invalid value '{data[field_name]}' for {field_name}",
                            severity=ValidationSeverity.ERROR.value,
                            suggested_fix=f"Use one of: {valid_values[:5]}...",
                        )
                    )

        # Calculate compatibility score
        error_count = sum(1 for i in issues if i.severity == ValidationSeverity.ERROR.value)
        warning_count = sum(1 for i in issues if i.severity == ValidationSeverity.WARNING.value)
        compatibility_score = max(0, 1.0 - (error_count * 0.2) - (warning_count * 0.05))

        # Determine transformations needed
        transformations_needed = []
        if detected_version and detected_version != self.CURRENT_VERSION:
            transformations_needed.append(
                f"Transform from v{detected_version} to v{self.CURRENT_VERSION}"
            )

        is_valid = error_count == 0

        return ValidationResult(
            is_valid=is_valid,
            issues=issues,
            schema_version=detected_version,
            compatibility_score=compatibility_score,
            transformations_needed=transformations_needed,
        )

    def _detect_schema_version(self, data: Dict[str, Any]) -> Optional[str]:
        """Detect the schema version from data structure."""
        # Check for v2.0 specific fields
        if any(
            f in data
            for f in ["hil_rating", "memory_type", "learning_weight", "ai_handoff_requested"]
        ):
            return "2.0"

        # Check for v1.1 specific fields
        if any(f in data for f in ["ai_domain", "ai_subdomain"]):
            return "1.1"

        # Check for v1.0 minimum fields
        if "ai_phase" in data and "ai_status" in data:
            return "1.0"

        return None

    # ============================================================
    # Schema Transformation
    # ============================================================

    def transform_schema(
        self,
        data: Dict[str, Any],
        source_version: str,
        target_version: str,
    ) -> Dict[str, Any]:
        """
        Transform a schema from one version to another.

        Args:
            data: Schema data to transform
            source_version: Source schema version
            target_version: Target schema version

        Returns:
            Transformed schema data
        """
        result = data.copy()

        # Get transformation path
        path = self._get_transformation_path(source_version, target_version)

        for step in path:
            transformation = SCHEMA_TRANSFORMATIONS.get(step)
            if transformation:
                result = self._apply_transformation(result, transformation)

        return result

    def _get_transformation_path(
        self, source: str, target: str
    ) -> List[str]:
        """Get the transformation steps needed."""
        version_order = ["1.0", "1.1", "2.0"]

        try:
            source_idx = version_order.index(source)
            target_idx = version_order.index(target)
        except ValueError:
            return []

        path = []
        if source_idx < target_idx:
            # Upgrade path
            for i in range(source_idx, target_idx):
                path.append(f"{version_order[i]}_to_{version_order[i + 1]}")
        elif source_idx > target_idx:
            # Downgrade path (reverse transformations)
            for i in range(source_idx, target_idx, -1):
                path.append(f"{version_order[i]}_to_{version_order[i - 1]}")

        return path

    def _apply_transformation(
        self,
        data: Dict[str, Any],
        transformation: SchemaTransformation,
    ) -> Dict[str, Any]:
        """Apply a single transformation step."""
        result = {}

        # Apply field mappings
        for source_field, target_field in transformation.field_mappings.items():
            if source_field in data:
                result[target_field] = data[source_field]

        # Copy unmapped fields
        for key, value in data.items():
            if key not in transformation.field_mappings:
                if key not in transformation.removals:
                    result[key] = value

        # Apply additions
        for key, default_value in transformation.additions.items():
            if key not in result:
                result[key] = default_value

        # Apply value transformations
        for field_name, transform_spec in transformation.value_transformations.items():
            if field_name in result and isinstance(transform_spec, dict):
                old_value = result[field_name]
                if old_value in transform_spec:
                    result[field_name] = transform_spec[old_value]

        return result

    # ============================================================
    # Export Functions
    # ============================================================

    async def export_contexts(
        self,
        context_ids: Optional[List[UUID]] = None,
        domain: Optional[AIDomain] = None,
        format_type: ACDExportFormat = ACDExportFormat.JSON,
        include_relationships: bool = True,
        schema_version: str = "2.0",
    ) -> ExportedSchema:
        """
        Export ACD contexts to a portable format.

        Args:
            context_ids: Specific context IDs to export (None for all)
            domain: Filter by domain
            format_type: Export format
            include_relationships: Include relationship data
            schema_version: Target schema version

        Returns:
            ExportedSchema with contexts and metadata
        """
        try:
            # Build query
            conditions = []
            if domain:
                conditions.append(ACDContextModel.ai_domain == domain.value)

            stmt = select(ACDContextModel)
            if conditions:
                stmt = stmt.where(and_(*conditions))

            if context_ids:
                stmt = stmt.where(ACDContextModel.id.in_(context_ids))

            result = await self.db.execute(stmt)
            contexts = result.scalars().all()

            # Convert to export format
            exported_contexts = []
            relationships = []

            for ctx in contexts:
                ctx_dict = self._context_to_dict(ctx)

                # Transform to target version if needed
                if schema_version != self.CURRENT_VERSION:
                    ctx_dict = self.transform_schema(
                        ctx_dict, self.CURRENT_VERSION, schema_version
                    )

                exported_contexts.append(ctx_dict)

                # Extract relationships
                if include_relationships and ctx.related_contexts:
                    for related_id in ctx.related_contexts:
                        relationships.append({
                            "source_id": str(ctx.id),
                            "target_id": related_id,
                            "relationship_type": "related",
                        })

            # Build metadata
            schema_hash = self._calculate_schema_hash(exported_contexts)
            metadata = ACDSchemaMetadata(
                version=schema_version,
                system_type=self.SYSTEM_TYPE,
                system_id=self.SYSTEM_ID,
                created_at=datetime.now(timezone.utc).isoformat(),
                schema_hash=schema_hash,
                capabilities=["import", "export", "validation", "transformation"],
                domain_support=[d.value for d in AIDomain],
            )

            # Build correlation hints
            correlation_hints = {
                "domains_present": list(set(
                    c.get("ai_domain") for c in exported_contexts if c.get("ai_domain")
                )),
                "total_contexts": len(exported_contexts),
                "has_hil_ratings": any(
                    c.get("hil_rating") is not None for c in exported_contexts
                ),
            }

            logger.info(
                f"Exported {len(exported_contexts)} contexts in format {format_type.value}"
            )

            return ExportedSchema(
                metadata=metadata,
                contexts=exported_contexts,
                relationships=relationships,
                correlation_hints=correlation_hints,
            )

        except Exception as e:
            logger.error(f"Failed to export contexts: {e}")
            raise

    def _context_to_dict(self, ctx: ACDContextModel) -> Dict[str, Any]:
        """Convert a context model to a dictionary."""
        return {
            "id": str(ctx.id),
            "benchmark_id": str(ctx.benchmark_id) if ctx.benchmark_id else None,
            "content_id": str(ctx.content_id) if ctx.content_id else None,
            "ai_phase": ctx.ai_phase,
            "ai_status": ctx.ai_status,
            "ai_complexity": ctx.ai_complexity,
            "ai_note": ctx.ai_note,
            "ai_dependencies": ctx.ai_dependencies,
            "ai_domain": ctx.ai_domain,
            "ai_subdomain": ctx.ai_subdomain,
            "ai_state": ctx.ai_state,
            "ai_confidence": ctx.ai_confidence,
            "ai_queue_priority": ctx.ai_queue_priority,
            "ai_queue_status": ctx.ai_queue_status,
            "ai_validation": ctx.ai_validation,
            "ai_assigned_to": ctx.ai_assigned_to,
            "ai_context": ctx.ai_context,
            "ai_metadata": ctx.ai_metadata,
            "hil_rating": ctx.hil_rating,
            "hil_rating_tags": ctx.hil_rating_tags,
            "workflow_id": ctx.workflow_id,
            "model_id": ctx.model_id,
            "lora_ids": ctx.lora_ids,
            "learning_weight": ctx.learning_weight,
            "outcome_score": ctx.outcome_score,
            "memory_type": ctx.memory_type,
            "memory_importance": ctx.memory_importance,
            "related_contexts": ctx.related_contexts,
            "correlation_scores": ctx.correlation_scores,
            "created_at": ctx.created_at.isoformat() if ctx.created_at else None,
            "updated_at": ctx.updated_at.isoformat() if ctx.updated_at else None,
        }

    def _calculate_schema_hash(self, contexts: List[Dict[str, Any]]) -> str:
        """Calculate a hash of the schema structure."""
        # Use first context structure if available
        if contexts:
            structure = sorted(contexts[0].keys())
        else:
            structure = []

        hash_input = json.dumps(structure, sort_keys=True)
        return hashlib.sha256(hash_input.encode()).hexdigest()[:16]

    def format_export(
        self,
        exported: ExportedSchema,
        format_type: ACDExportFormat,
    ) -> str:
        """
        Format exported schema to string representation.

        Args:
            exported: Exported schema
            format_type: Target format

        Returns:
            Formatted string
        """
        data = {
            "metadata": asdict(exported.metadata),
            "contexts": exported.contexts,
            "relationships": exported.relationships,
            "correlation_hints": exported.correlation_hints,
        }

        if format_type == ACDExportFormat.JSON:
            return json.dumps(data, indent=2, default=str)

        elif format_type == ACDExportFormat.NDJSON:
            lines = [json.dumps(asdict(exported.metadata), default=str)]
            for ctx in exported.contexts:
                lines.append(json.dumps(ctx, default=str))
            return "\n".join(lines)

        elif format_type == ACDExportFormat.JSONLD:
            # JSON-LD with context
            data["@context"] = {
                "@vocab": "https://gator.ai/acd/schema/",
                "ai_domain": "https://gator.ai/acd/domain",
                "ai_phase": "https://gator.ai/acd/phase",
            }
            data["@type"] = "ACDExport"
            return json.dumps(data, indent=2, default=str)

        elif format_type == ACDExportFormat.YAML:
            # Simple YAML-like format (avoiding yaml dependency)
            lines = ["# ACD Schema Export"]
            lines.append(f"version: {exported.metadata.version}")
            lines.append(f"system_id: {exported.metadata.system_id}")
            lines.append(f"created_at: {exported.metadata.created_at}")
            lines.append(f"contexts_count: {len(exported.contexts)}")
            lines.append("contexts:")
            for ctx in exported.contexts[:5]:  # Limit preview
                lines.append(f"  - id: {ctx.get('id')}")
                lines.append(f"    phase: {ctx.get('ai_phase')}")
                lines.append(f"    domain: {ctx.get('ai_domain')}")
            if len(exported.contexts) > 5:
                lines.append(f"  # ... and {len(exported.contexts) - 5} more")
            return "\n".join(lines)

        return json.dumps(data, indent=2, default=str)

    # ============================================================
    # Import Functions
    # ============================================================

    async def import_schema(
        self,
        data: Union[str, Dict[str, Any]],
        source_system: str,
        validate: bool = True,
        transform: bool = True,
        merge_strategy: str = "skip_existing",
    ) -> ImportResult:
        """
        Import ACD schema from external system.

        Args:
            data: Schema data (JSON string or dict)
            source_system: Identifier of source system
            validate: Validate before importing
            transform: Transform to current version
            merge_strategy: How to handle existing contexts
                - "skip_existing": Skip if ID exists
                - "update_existing": Update existing contexts
                - "create_new": Always create new with new ID

        Returns:
            ImportResult with import details
        """
        try:
            # Parse data if string
            if isinstance(data, str):
                try:
                    parsed_data = json.loads(data)
                except json.JSONDecodeError as e:
                    return ImportResult(
                        success=False,
                        errors=[f"Invalid JSON: {e}"],
                    )
            else:
                parsed_data = data

            # Handle both full export format and raw contexts
            if "contexts" in parsed_data:
                contexts = parsed_data["contexts"]
                metadata = parsed_data.get("metadata", {})
            elif isinstance(parsed_data, list):
                contexts = parsed_data
                metadata = {}
            else:
                contexts = [parsed_data]
                metadata = {}

            source_version = metadata.get("version") or self._detect_schema_version(
                contexts[0] if contexts else {}
            )

            validation_result = None
            transformations_applied = []
            imported_ids = []
            errors = []
            imported_count = 0
            skipped_count = 0
            failed_count = 0

            for ctx_data in contexts:
                try:
                    # Validate if requested
                    if validate:
                        validation_result = self.validate_schema(ctx_data)
                        if not validation_result.is_valid:
                            failed_count += 1
                            errors.append(
                                f"Validation failed for context: "
                                f"{[asdict(i) for i in validation_result.issues]}"
                            )
                            continue

                    # Transform if needed
                    if transform and source_version and source_version != self.CURRENT_VERSION:
                        ctx_data = self.transform_schema(
                            ctx_data, source_version, self.CURRENT_VERSION
                        )
                        if f"v{source_version}_to_v{self.CURRENT_VERSION}" not in transformations_applied:
                            transformations_applied.append(
                                f"v{source_version}_to_v{self.CURRENT_VERSION}"
                            )

                    # Map external domains if needed
                    ctx_data = self._map_external_domain(ctx_data, source_system)

                    # Add source tracking
                    ctx_data["ai_metadata"] = ctx_data.get("ai_metadata") or {}
                    ctx_data["ai_metadata"]["import_source"] = source_system
                    ctx_data["ai_metadata"]["import_timestamp"] = datetime.now(
                        timezone.utc
                    ).isoformat()

                    # Handle merge strategy
                    existing_id = ctx_data.get("id")
                    if existing_id and merge_strategy == "skip_existing":
                        # Check if exists
                        stmt = select(ACDContextModel).where(
                            ACDContextModel.id == existing_id
                        )
                        result = await self.db.execute(stmt)
                        if result.scalar_one_or_none():
                            skipped_count += 1
                            continue

                    # Create context
                    if merge_strategy == "create_new" or not existing_id:
                        ctx_data.pop("id", None)

                    # Create the context
                    context_create = self._dict_to_context_create(ctx_data)
                    context = ACDContextModel(**context_create.model_dump())

                    self.db.add(context)
                    await self.db.flush()

                    imported_ids.append(str(context.id))
                    imported_count += 1

                except Exception as ctx_error:
                    failed_count += 1
                    errors.append(str(ctx_error))

            await self.db.commit()

            logger.info(
                f"Import complete: {imported_count} imported, "
                f"{skipped_count} skipped, {failed_count} failed"
            )

            return ImportResult(
                success=failed_count == 0,
                contexts_imported=imported_count,
                contexts_skipped=skipped_count,
                contexts_failed=failed_count,
                validation_result=validation_result,
                transformations_applied=transformations_applied,
                imported_ids=imported_ids,
                errors=errors,
            )

        except Exception as e:
            logger.error(f"Import failed: {e}")
            await self.db.rollback()
            return ImportResult(
                success=False,
                errors=[str(e)],
            )

    def _map_external_domain(
        self,
        data: Dict[str, Any],
        source_system: str,
    ) -> Dict[str, Any]:
        """Map external system domain to ACD domain."""
        result = data.copy()

        if source_system in EXTERNAL_DOMAIN_MAPPINGS:
            mappings = EXTERNAL_DOMAIN_MAPPINGS[source_system]
            external_domain = data.get("ai_domain") or data.get("domain")

            if external_domain and external_domain in mappings:
                result["ai_domain"] = mappings[external_domain]

        return result

    def _dict_to_context_create(self, data: Dict[str, Any]) -> ACDContextCreate:
        """Convert a dictionary to ACDContextCreate."""
        # Map common fields with proper enum defaults
        return ACDContextCreate(
            ai_phase=data.get("ai_phase", "SCHEMA_IMPORT"),
            ai_status=data.get("ai_status", AIStatus.IMPLEMENTED.value),
            ai_complexity=data.get("ai_complexity"),
            ai_note=data.get("ai_note"),
            ai_dependencies=data.get("ai_dependencies"),
            ai_domain=data.get("ai_domain"),
            ai_subdomain=data.get("ai_subdomain"),
            ai_context=data.get("ai_context"),
            ai_metadata=data.get("ai_metadata"),
        )

    # ============================================================
    # Cross-Correlation Functions
    # ============================================================

    async def correlate_with_external(
        self,
        external_contexts: List[Dict[str, Any]],
        source_system: str,
        correlation_threshold: float = 0.5,
    ) -> List[CrossCorrelation]:
        """
        Find correlations between external contexts and local ACD contexts.

        Args:
            external_contexts: Contexts from external system
            source_system: Source system identifier
            correlation_threshold: Minimum score for correlation

        Returns:
            List of cross-correlations found
        """
        try:
            correlations: List[CrossCorrelation] = []

            # Get local contexts
            stmt = select(ACDContextModel)
            result = await self.db.execute(stmt)
            local_contexts = result.scalars().all()

            for ext_ctx in external_contexts:
                ext_domain = ext_ctx.get("ai_domain")
                ext_phase = ext_ctx.get("ai_phase")
                ext_model = ext_ctx.get("model_id")

                for local_ctx in local_contexts:
                    score = 0.0
                    shared_attrs: Dict[str, Any] = {}
                    insights: List[str] = []

                    # Domain match
                    if ext_domain and ext_domain == local_ctx.ai_domain:
                        score += 0.3
                        shared_attrs["domain"] = ext_domain
                        insights.append(f"Same domain: {ext_domain}")

                    # Phase match
                    if ext_phase and ext_phase == local_ctx.ai_phase:
                        score += 0.2
                        shared_attrs["phase"] = ext_phase

                    # Model match
                    if ext_model and ext_model == local_ctx.model_id:
                        score += 0.3
                        shared_attrs["model"] = ext_model
                        insights.append(f"Same model: {ext_model}")

                    # Workflow match
                    ext_workflow = ext_ctx.get("workflow_id")
                    if ext_workflow and ext_workflow == local_ctx.workflow_id:
                        score += 0.2
                        shared_attrs["workflow"] = ext_workflow

                    if score >= correlation_threshold:
                        correlations.append(
                            CrossCorrelation(
                                source_system=source_system,
                                target_system=self.SYSTEM_ID,
                                source_context_id=ext_ctx.get("id", "unknown"),
                                target_context_id=str(local_ctx.id),
                                correlation_type="attribute_match",
                                correlation_score=score,
                                shared_attributes=shared_attrs,
                                insights=insights,
                            )
                        )

            logger.info(
                f"Found {len(correlations)} correlations with {source_system}"
            )

            return correlations

        except Exception as e:
            logger.error(f"Correlation failed: {e}")
            return []

    async def get_correlation_insights(
        self,
        correlations: List[CrossCorrelation],
    ) -> Dict[str, Any]:
        """
        Generate insights from cross-correlations.

        Args:
            correlations: List of correlations to analyze

        Returns:
            Dictionary of insights
        """
        if not correlations:
            return {"total_correlations": 0, "insights": []}

        # Analyze correlations
        domains = {}
        models = {}
        scores = []

        for corr in correlations:
            scores.append(corr.correlation_score)

            if "domain" in corr.shared_attributes:
                domain = corr.shared_attributes["domain"]
                domains[domain] = domains.get(domain, 0) + 1

            if "model" in corr.shared_attributes:
                model = corr.shared_attributes["model"]
                models[model] = models.get(model, 0) + 1

        insights = []

        # Domain insights
        if domains:
            top_domain = max(domains.items(), key=lambda x: x[1])
            insights.append(
                f"Most correlated domain: {top_domain[0]} ({top_domain[1]} correlations)"
            )

        # Model insights
        if models:
            top_model = max(models.items(), key=lambda x: x[1])
            insights.append(
                f"Most correlated model: {top_model[0]} ({top_model[1]} correlations)"
            )

        # Score insights
        avg_score = sum(scores) / len(scores) if scores else 0
        high_score_count = sum(1 for s in scores if s >= 0.7)

        if high_score_count > 0:
            insights.append(
                f"{high_score_count} high-confidence correlations found (score >= 0.7)"
            )

        return {
            "total_correlations": len(correlations),
            "average_score": avg_score,
            "high_confidence_count": high_score_count,
            "domains_correlated": domains,
            "models_correlated": models,
            "insights": insights,
        }
