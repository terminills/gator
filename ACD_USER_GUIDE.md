# ACD (Autonomous Continuous Development) User Guide

## Table of Contents
1. [Introduction](#introduction)
2. [Quick Start](#quick-start)
3. [Core Concepts](#core-concepts)
4. [Basic Usage](#basic-usage)
5. [Advanced Features](#advanced-features)
6. [API Reference](#api-reference)
7. [Best Practices](#best-practices)
8. [Troubleshooting](#troubleshooting)

---

## Introduction

The ACD (Autonomous Continuous Development) system is a revolutionary feature in Gator that enables AI-to-AI communication and autonomous learning. Unlike traditional logging systems, ACD creates a **machine-readable context protocol** that allows AI agents to coordinate, learn, and improve autonomously.

### What Makes ACD Different?

**Traditional AI System:**
```
User → AI → Output → Human evaluates → Human adjusts
```

**ACD-Enabled System:**
```
AI Agent 1 → [ACD Context] → AI Agent 2 → [ACD Context] → AI Agent 3
                ↓                            ↓                    ↓
            Learning                     Validation          Refinement
```

### Key Benefits

1. **Institutional Memory**: Knowledge persists across generations and improves over time
2. **Self-Improving System**: Automatically learns from every piece of content generated
3. **Multi-Agent Coordination**: AI agents communicate and coordinate without human intervention
4. **Comprehensive Error Tracking**: Full diagnostics for debugging and pattern detection
5. **Pattern Learning**: Extracts successful strategies and applies them to future content

---

## Quick Start

### 1. Verify ACD is Enabled

ACD is automatically enabled in Gator. Verify it's working:

```bash
# Initialize database (includes ACD tables)
python setup_db.py

# Run ACD demo to verify functionality
python demo_acd_integration.py
```

### 2. Your First ACD Context

```python
from backend.services.acd_service import ACDService
from backend.models.acd import ACDContextCreate, AIStatus, AIState, AIComplexity

# Create an ACD context for content generation
async def generate_with_acd(db_session):
    acd_service = ACDService(db_session)
    
    context = await acd_service.create_context(
        ACDContextCreate(
            ai_phase="IMAGE_GENERATION",
            ai_status=AIStatus.IMPLEMENTED,
            ai_complexity=AIComplexity.MEDIUM,
            ai_note="Generating profile photo for tech influencer",
            ai_state=AIState.PROCESSING,
            ai_context={
                "prompt": "professional headshot, tech entrepreneur, modern office",
                "model": "stable-diffusion-xl",
                "resolution": "1024x1024"
            }
        )
    )
    
    print(f"Created ACD context: {context.id}")
    return context
```

### 3. Automatic Tracking with Context Manager

The easiest way to use ACD is with the `ACDContextManager`:

```python
from backend.utils.acd_integration import ACDContextManager
from backend.models.acd import AIComplexity, AIConfidence

async def generate_content_automatically(db_session):
    async with ACDContextManager(
        db_session,
        phase="CONTENT_GENERATION",
        note="Creating social media post",
        complexity=AIComplexity.LOW,
        initial_context={"platform": "instagram", "style": "casual"}
    ) as acd:
        try:
            # Your content generation code here
            content = await generate_instagram_post()
            
            # Update confidence
            await acd.set_confidence(AIConfidence.CONFIDENT)
            
            return content
        except Exception as e:
            # Error is automatically logged as trace artifact
            raise
        # Context automatically updated to DONE on exit
```

---

## Core Concepts

### ACD Context

An **ACD Context** is a structured record that captures:

- **Phase**: What type of work is being done (IMAGE_GENERATION, TEXT_GENERATION, etc.)
- **Status**: Implementation status (IMPLEMENTED, PARTIAL, etc.)
- **State**: Current execution state (PROCESSING, DONE, FAILED, etc.)
- **Complexity**: Task difficulty level (LOW, MEDIUM, HIGH, CRITICAL)
- **Confidence**: Agent's confidence level (CONFIDENT, UNCERTAIN, VALIDATED, etc.)
- **Metadata**: Arbitrary JSON data specific to the task

### Trace Artifacts

**Trace Artifacts** capture errors and diagnostic information:

- Full stack traces
- Environment information
- Error codes and messages
- File/line/function location
- Related context links

### Agent Coordination

AI agents use ACD to coordinate:

- **Handoffs**: Transfer work to specialized agents
- **Reviews**: Request validation from other agents
- **Assistance**: Request help when uncertain
- **Validation**: Provide feedback on other agents' work

---

## Basic Usage

### Creating a Context

```python
from backend.services.acd_service import ACDService
from backend.models.acd import ACDContextCreate, AIStatus, AIState

context = await acd_service.create_context(
    ACDContextCreate(
        ai_phase="TEXT_GENERATION",
        ai_status=AIStatus.IMPLEMENTED,
        ai_state=AIState.PROCESSING,
        ai_note="Generating Instagram caption"
    )
)
```

### Updating a Context

```python
from backend.models.acd import ACDContextUpdate, AIState, AIConfidence

updated_context = await acd_service.update_context(
    context.id,
    ACDContextUpdate(
        ai_state=AIState.DONE,
        ai_confidence=AIConfidence.VALIDATED
    )
)
```

### Logging Errors

```python
from backend.models.acd import ACDTraceArtifactCreate

try:
    result = await risky_operation()
except Exception as e:
    await acd_service.create_trace_artifact(
        ACDTraceArtifactCreate(
            session_id=str(context.id),
            event_type="runtime_error",
            error_message=str(e),
            acd_context_id=context.id,
            stack_trace=[...],
            environment={"gpu_memory": "8GB", "cuda_available": True}
        )
    )
    raise
```

### Querying Statistics

```python
# Get statistics for the last 24 hours
stats = await acd_service.get_stats(hours=24)

print(f"Total contexts: {stats.total_contexts}")
print(f"Success rate: {stats.completed_contexts / stats.total_contexts * 100:.1f}%")
print(f"By phase: {stats.by_phase}")
print(f"Average completion time: {stats.avg_completion_time}s")
```

---

## Advanced Features

### Multi-Agent Coordination

```python
# Agent 1: Generator creates content with uncertainty
generator_context = await acd_service.create_context(
    ACDContextCreate(
        ai_phase="CONTENT_GENERATION",
        ai_confidence=AIConfidence.UNCERTAIN,
        ai_request=AIRequest.REQUEST_REVIEW,
        ai_note="Generated content but unsure about tone"
    )
)

# Agent 2: Reviewer finds contexts needing review
review_contexts = await acd_service.get_contexts(
    request="REQUEST_REVIEW"
)

for ctx in review_contexts:
    # Review the content
    review = await review_content(ctx.content_id)
    
    # Update context with validation
    await acd_service.update_context(
        ctx.id,
        ACDContextUpdate(
            ai_validation=AIValidation.APPROVED,
            ai_issues=review.issues,
            ai_suggestions=review.suggestions
        )
    )
```

### Agent Handoffs

```python
# Create context that needs specialized handling
context = await acd_service.create_context(
    ACDContextCreate(
        ai_phase="IMAGE_ENHANCEMENT",
        ai_handoff_requested=True,
        ai_handoff_to="high_resolution_specialist",
        ai_handoff_type=AIHandoffType.SPECIALIZATION,
        ai_required_capabilities=["4k_upscaling", "face_enhancement"],
        ai_skill_level_required=AISkillLevel.EXPERT
    )
)

# System routes to appropriate agent based on capabilities
```

### Pattern Learning

```python
from backend.utils.pattern_analysis import PatternAnalyzer

analyzer = PatternAnalyzer(db_session)

# Get successful patterns from last 30 days
patterns = await analyzer.get_successful_patterns(
    persona_id=persona_id,
    platform="instagram",
    min_engagement_rate=5.0,  # Only high performers
    days=30
)

# Get optimal posting times
optimal_times = await analyzer.get_optimal_posting_times(
    persona_id=persona_id,
    platform="instagram"
)

# Get effective hashtags
hashtags = await analyzer.get_effective_hashtags(
    persona_id=persona_id,
    platform="instagram",
    min_posts=5  # Require statistical significance
)
```

### Validation Reports

```python
# Generate system health report
report = await acd_service.generate_validation_report(hours=24)

print(f"System Status: {report.status}")
print(f"Total Operations: {report.total_operations}")
print(f"Success Rate: {report.success_rate}%")
print(f"Warnings: {report.warnings}")
print(f"Critical Issues: {report.critical_issues}")
```

---

## API Reference

### REST Endpoints

#### Create Context
```http
POST /api/v1/acd/contexts/
Content-Type: application/json

{
  "ai_phase": "IMAGE_GENERATION",
  "ai_status": "IMPLEMENTED",
  "ai_complexity": "MEDIUM",
  "ai_note": "Generating portrait",
  "ai_context": {"model": "sdxl", "prompt": "..."}
}
```

#### Get Context
```http
GET /api/v1/acd/contexts/{context_id}
```

#### Update Context
```http
PUT /api/v1/acd/contexts/{context_id}
Content-Type: application/json

{
  "ai_state": "DONE",
  "ai_confidence": "VALIDATED"
}
```

#### Delete Context
```http
DELETE /api/v1/acd/contexts/{context_id}
```

#### List Contexts
```http
GET /api/v1/acd/contexts/?phase=IMAGE_GENERATION&state=DONE&limit=50
```

#### Create Trace Artifact
```http
POST /api/v1/acd/trace-artifacts/
Content-Type: application/json

{
  "session_id": "gen_123",
  "event_type": "runtime_error",
  "error_message": "CUDA out of memory",
  "acd_context_id": "uuid...",
  "stack_trace": ["..."]
}
```

#### Get Statistics
```http
GET /api/v1/acd/stats/?hours=24&phase=IMAGE_GENERATION
```

#### Get Validation Report
```http
GET /api/v1/acd/validation-report/?hours=24
```

#### Assign to Agent
```http
POST /api/v1/acd/contexts/{context_id}/assign
Content-Type: application/json

{
  "agent_name": "content_reviewer",
  "reason": "Requires expert review"
}
```

---

## Best Practices

### 1. Always Use Context Managers for Simple Cases

```python
# Good - automatic cleanup and error handling
async with ACDContextManager(db, phase="GENERATION", note="Creating post") as acd:
    result = await generate_content()
    return result

# Avoid - manual context management is error-prone
context = await acd_service.create_context(...)
try:
    result = await generate_content()
    await acd_service.update_context(context.id, ...)
finally:
    # Easy to forget cleanup
```

### 2. Include Meaningful Context Data

```python
# Good - detailed context for debugging and learning
ai_context = {
    "model": "stable-diffusion-xl",
    "prompt": "professional headshot...",
    "negative_prompt": "blurry, low quality...",
    "resolution": "1024x1024",
    "steps": 50,
    "cfg_scale": 7.5,
    "seed": 12345,
    "persona_id": str(persona.id)
}

# Avoid - minimal context limits debugging
ai_context = {"model": "sdxl"}
```

### 3. Set Appropriate Complexity Levels

```python
# Use complexity to prioritize and allocate resources
AIComplexity.LOW       # Simple text generation
AIComplexity.MEDIUM    # Standard image generation
AIComplexity.HIGH      # Complex multi-step workflows
AIComplexity.CRITICAL  # Mission-critical operations
```

### 4. Use Confidence Levels Honestly

```python
# Be honest about confidence for better coordination
AIConfidence.CONFIDENT    # High quality, validated output
AIConfidence.UNCERTAIN    # May need review
AIConfidence.HYPOTHESIS   # Experimental approach
AIConfidence.VALIDATED    # Confirmed by validation agent
```

### 5. Log All Errors as Trace Artifacts

```python
# Always capture errors for pattern analysis
try:
    result = await operation()
except Exception as e:
    await acd_service.create_trace_artifact(
        ACDTraceArtifactCreate(
            session_id=session_id,
            event_type="runtime_error",
            error_message=str(e),
            acd_context_id=context.id,
            stack_trace=traceback.format_exc().split('\n')
        )
    )
    raise
```

### 6. Query Patterns for Learning

```python
# Regularly analyze successful patterns
if context.ai_validation == AIValidation.APPROVED:
    patterns = await analyzer.get_successful_patterns(
        persona_id=persona.id,
        days=30
    )
    # Apply learned patterns to future generations
```

### 7. Clean Up Old Contexts

```python
# Archive contexts older than 90 days
async def archive_old_contexts():
    cutoff = datetime.now() - timedelta(days=90)
    old_contexts = await acd_service.get_contexts(
        created_before=cutoff
    )
    # Export and delete
```

---

## Troubleshooting

### Context Not Created

**Problem**: `create_context()` fails or returns None

**Solutions**:
1. Check database connection
2. Verify all required fields are provided
3. Check logs for validation errors
4. Ensure enums are valid values

```python
# Verify database is accessible
from backend.database.connection import database_manager
await database_manager.connect()
```

### Trace Artifacts Not Logged

**Problem**: Errors occur but no trace artifacts are created

**Solutions**:
1. Ensure `acd_context_id` is valid
2. Check that error handling includes trace artifact creation
3. Verify database write permissions

```python
# Always wrap operations with error logging
try:
    result = await operation()
except Exception as e:
    logger.error(f"Operation failed: {e}")
    await acd_service.create_trace_artifact(...)
    raise
```

### High Memory Usage

**Problem**: ACD metadata consuming excessive memory

**Solutions**:
1. Limit size of `ai_context` and `ai_metadata` JSON fields
2. Implement archiving for old contexts
3. Use sampling for high-volume operations

```python
# Limit context data size
if len(json.dumps(ai_context)) > 10000:
    ai_context = {"summary": "Large context, see linked artifacts"}
```

### Slow Queries

**Problem**: ACD queries taking too long

**Solutions**:
1. Use indexed fields for filtering (ai_phase, ai_status, ai_state)
2. Add date range filters to limit results
3. Use pagination for large result sets

```python
# Good - uses indexes
contexts = await acd_service.get_contexts(
    ai_phase="IMAGE_GENERATION",
    ai_state=AIState.DONE,
    limit=50
)

# Avoid - full table scan on JSON field
contexts = await acd_service.get_contexts(
    ai_context={"model": "sdxl"}  # No index on JSON fields
)
```

### Pattern Learning Not Working

**Problem**: Pattern analyzer returns no patterns

**Solutions**:
1. Ensure sufficient data exists (minimum thresholds)
2. Verify contexts have validation status set
3. Check that social engagement data is being tracked

```python
# Need minimum data for patterns
patterns = await analyzer.get_successful_patterns(
    persona_id=persona_id,
    min_engagement_rate=5.0,
    days=30  # Need sufficient time range
)

if not patterns:
    print("Insufficient data - need more successful generations")
```

---

## Examples

### Complete Content Generation with ACD

```python
async def generate_social_post_with_acd(
    db_session,
    persona_id: str,
    prompt: str,
    platform: str
):
    """Generate social media post with full ACD tracking."""
    
    async with ACDContextManager(
        db_session,
        phase="SOCIAL_MEDIA_CONTENT",
        note=f"Generating {platform} post for persona {persona_id}",
        complexity=AIComplexity.MEDIUM,
        initial_context={
            "persona_id": persona_id,
            "platform": platform,
            "prompt": prompt
        }
    ) as acd:
        try:
            # Generate image
            image = await generate_image(prompt)
            await acd.update_context({"image_generated": True})
            
            # Generate caption
            caption = await generate_caption(image, platform)
            await acd.update_context({"caption_generated": True})
            
            # Generate hashtags
            hashtags = await generate_hashtags(caption, platform)
            await acd.update_context({"hashtags": hashtags})
            
            # Mark as confident
            await acd.set_confidence(AIConfidence.CONFIDENT)
            
            return {
                "image": image,
                "caption": caption,
                "hashtags": hashtags,
                "acd_context_id": acd.context_id
            }
            
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            await acd.set_confidence(AIConfidence.UNCERTAIN)
            raise
```

### Multi-Agent Content Review

```python
async def review_content_pipeline(db_session):
    """Multi-agent content review using ACD coordination."""
    
    acd_service = ACDService(db_session)
    
    # Find content needing review
    contexts = await acd_service.get_contexts(
        ai_request=AIRequest.REQUEST_REVIEW,
        ai_state=AIState.READY
    )
    
    for context in contexts:
        # Review content
        content = await get_content(context.content_id)
        review_result = await ai_review_content(content)
        
        # Update context with review
        await acd_service.update_context(
            context.id,
            ACDContextUpdate(
                ai_validation=review_result.validation,
                ai_confidence=review_result.confidence,
                ai_issues=review_result.issues,
                ai_suggestions=review_result.suggestions,
                ai_state=AIState.DONE
            )
        )
        
        # If approved, extract patterns
        if review_result.validation == AIValidation.APPROVED:
            await extract_and_store_patterns(context)
```

---

## Further Reading

- [ACD Implementation Summary](ACD_IMPLEMENTATION_SUMMARY.md) - Technical architecture and database schema
- [ACD AI-First Perspective](ACD_AI_FIRST_PERSPECTIVE.md) - Understanding ACD as AI-to-AI communication
- [ACD Phase 2 Implementation](ACD_PHASE2_IMPLEMENTATION.md) - Social engagement tracking and learning loop
- [Pattern Analysis Guide](docs/PATTERN_ANALYSIS.md) - Deep dive into learning from ACD data

---

## Support

For issues or questions about ACD:

1. Check the [troubleshooting section](#troubleshooting) above
2. Review the [implementation summary](ACD_IMPLEMENTATION_SUMMARY.md)
3. Run the demo script: `python demo_acd_integration.py`
4. Check API docs at `http://localhost:8000/docs`
5. Open an issue on GitHub with "[ACD]" prefix

---

**Last Updated**: November 2024  
**ACD Version**: 1.1.0  
**Status**: Production Ready ✅
