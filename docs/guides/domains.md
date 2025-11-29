# AI_DOMAIN and AI_SUBDOMAIN - Cortical Region Classification

## Overview

Added `AI_DOMAIN` and `AI_SUBDOMAIN` fields to transform ACD from "a giant pile of contexts" into a **multi-domain cognitive system** with cortical region separation.

## Why Domain Classification Matters

Without domain classification:
- ❌ Correlations get polluted (image errors mix with Python errors)
- ❌ Agent selection becomes noisy
- ❌ Pattern learning is unreliable
- ❌ Cross-domain routing causes failures

With domain classification:
- ✅ Clean correlation boundaries
- ✅ Intelligent agent routing
- ✅ Domain-aware pattern learning
- ✅ Safe cross-domain orchestration

## Domain Hierarchy

### Top-Level Domains (AI_DOMAIN)

Like cortical regions in the brain, these separate major categories of work:

1. **METAREASONING** - Meta-level reasoning and orchestration
   - Orchestration decisions
   - Reasoning engine operations
   - Agent selection
   - Complexity handling
   - Error resolution planning

2. **Content Generation Domains**
   - `CODE_GENERATION` - Software development
   - `TEXT_GENERATION` - Natural language content
   - `IMAGE_GENERATION` - Visual content
   - `VIDEO_GENERATION` - Video content
   - `AUDIO_GENERATION` - Audio/voice content
   - `GRAPH_GENERATION` - Data visualizations
   - `DATA_TRANSFORMATION` - Data processing
   - `MODEL_GENERATION` - AI model creation
   - `MODEL_REFINEMENT` - Model improvement

3. **Analysis Domains**
   - `COMPRESSION` - Data compression
   - `SUMMARIZATION` - Content summarization
   - `PLANNING` - Strategic planning
   - `ANALYSIS` - Data analysis

4. **MULTIMODAL_SEMANTICS** - Cross-modal understanding
   - Embeddings, alignment, captioning
   - Scene parsing, OCR
   - Cross-modal inference

5. **SYSTEM_OPERATIONS** - Infrastructure
   - File operations, scheduling
   - Pipeline control, environment
   - Microservices, plugins
   - Database, migrations

6. **HUMAN_INTERFACE** - User interaction
   - UI/UX generation
   - Forms, dashboards
   - Documentation, tutorials
   - Explanations

### Subdomains (AI_SUBDOMAIN)

Fine-grained specialization within each domain:

**CODE_GENERATION**:
- PYTHON, JAVASCRIPT, TYPESCRIPT, SQL
- BACKEND, FRONTEND, API, TESTING

**TEXT_GENERATION**:
- CREATIVE_WRITING, TECHNICAL_WRITING
- SOCIAL_MEDIA, MARKETING
- DOCUMENTATION, DIALOGUE

**IMAGE_GENERATION**:
- PORTRAITS, LANDSCAPES, ABSTRACT
- PHOTOREALISTIC, STYLIZED
- LOGO_DESIGN, IMAGE_EDITING, UPSCALING

**VIDEO_GENERATION**:
- SHORT_FORM, LONG_FORM
- ANIMATION, VIDEO_EDITING
- EFFECTS, TRANSITIONS

And many more... (see `src/backend/models/acd.py` for complete list)

## Domain Compatibility Matrix

Defines which domains can correlate safely:

```python
DOMAIN_COMPATIBILITY = {
    AIDomain.CODE_GENERATION: [
        AIDomain.TEXT_GENERATION,
        AIDomain.PLANNING,
        AIDomain.SYSTEM_OPERATIONS,
    ],
    AIDomain.IMAGE_GENERATION: [
        AIDomain.MULTIMODAL_SEMANTICS,
        AIDomain.VIDEO_GENERATION,
        AIDomain.HUMAN_INTERFACE,
    ],
    AIDomain.METAREASONING: [
        # Can work with all domains
    ],
    ...
}
```

## Database Schema

### New Fields in `acd_contexts` Table

```sql
ALTER TABLE acd_contexts 
ADD COLUMN ai_domain VARCHAR(50);

ALTER TABLE acd_contexts 
ADD COLUMN ai_subdomain VARCHAR(50);

CREATE INDEX ix_acd_contexts_ai_domain ON acd_contexts (ai_domain);
CREATE INDEX ix_acd_contexts_ai_subdomain ON acd_contexts (ai_subdomain);
```

### Migration

Run the migration script to add these fields:

```bash
python add_domain_fields_migration.py
```

Or run setup_db.py to recreate tables with new schema:

```bash
python setup_db.py
```

## Usage Examples

### Creating Context with Domain

```python
from backend.models.acd import AIDomain, AISubdomain

context = await acd_service.create_context(
    ACDContextCreate(
        ai_phase="IMAGE_GENERATION",
        ai_domain=AIDomain.IMAGE_GENERATION,
        ai_subdomain=AISubdomain.PORTRAITS,
        ai_complexity=AIComplexity.MEDIUM,
        ai_note="Generate portrait for Instagram",
    )
)
```

### Reasoning Engine Uses Domain

The reasoning engine now includes domain in prompts:

```
## Current Task Context

**Domain**: IMAGE_GENERATION
**Subdomain**: PORTRAITS
**Phase**: IMAGE_GENERATION
**Complexity**: MEDIUM
```

### Pattern Queries Filter by Domain

```python
# Old: queries all patterns for phase
patterns = await orchestrator._query_relevant_patterns(context)

# New: filters by domain compatibility
# Only includes same-domain + compatible domains
# Prevents noise from incompatible domains
```

### Domain Compatibility Checking

```python
from backend.services.reasoning_engine import ReasoningEngine

engine = ReasoningEngine(db)

# Check if two domains are compatible
weight = engine.check_domain_compatibility(
    "IMAGE_GENERATION",
    "VIDEO_GENERATION"
)
# Returns: 0.6 (compatible but different)

weight = engine.check_domain_compatibility(
    "IMAGE_GENERATION",
    "IMAGE_GENERATION"
)
# Returns: 1.0 (same domain)

weight = engine.check_domain_compatibility(
    "IMAGE_GENERATION",
    "CODE_GENERATION"
)
# Returns: 0.1 (incompatible)
```

### Agent Capability Matching with Domains

```python
# Agent evaluation now considers domain compatibility
match_score = await engine.evaluate_capability_match(
    agent_name="image_specialist",
    task_requirements={"capabilities": ["image_generation", "sdxl"]},
    task_domain="IMAGE_GENERATION"
)
# Blends capability match with domain compatibility
```

## Integration Points

### 1. ACDContextManager

Automatically sets domain when creating contexts:

```python
async with ACDContextManager(
    db,
    phase="IMAGE_GENERATION",
    ai_domain=AIDomain.IMAGE_GENERATION,
    ai_subdomain=AISubdomain.PORTRAITS,
) as acd:
    # Domain is tracked and used for orchestration
    pass
```

### 2. ReasoningOrchestrator

Pattern queries now filter by domain:

```python
# Cache key includes domain
cache_key = f"{context.ai_domain}_{context.ai_phase}_{context.ai_complexity}"

# Query filters by compatible domains
if context.ai_domain:
    compatible_domains = DOMAIN_COMPATIBILITY.get(domain, [])
    stmt = stmt.where(ACDContextModel.ai_domain.in_(domain_filter))
```

### 3. ReasoningEngine

Prompts include domain information:

```python
prompt = f"""
## Current Task Context

**Domain**: {context.ai_domain or 'UNKNOWN'}
**Subdomain**: {context.ai_subdomain or 'UNKNOWN'}
...
```

### 4. Multi-Agent System

Agents can register with domain specializations:

```python
agent = AgentModel(
    agent_name="image_specialist",
    specializations=[
        {"domain": "IMAGE_GENERATION", "subdomain": "PORTRAITS"},
        {"domain": "IMAGE_GENERATION", "subdomain": "PHOTOREALISTIC"},
    ]
)
```

## Automatic Domain Detection

When domain is not explicitly provided, the system can infer it:

### From Phase Name

```python
def infer_domain_from_phase(phase: str) -> Optional[AIDomain]:
    phase_lower = phase.lower()
    
    if "image" in phase_lower or "photo" in phase_lower:
        return AIDomain.IMAGE_GENERATION
    elif "video" in phase_lower:
        return AIDomain.VIDEO_GENERATION
    elif "text" in phase_lower or "caption" in phase_lower:
        return AIDomain.TEXT_GENERATION
    elif "code" in phase_lower or "python" in phase_lower:
        return AIDomain.CODE_GENERATION
    # ... more rules
    
    return None
```

### From AI_NOTE

```python
def infer_domain_from_note(note: str) -> Optional[AIDomain]:
    note_lower = note.lower()
    
    keywords = {
        AIDomain.IMAGE_GENERATION: ["image", "photo", "picture", "portrait"],
        AIDomain.TEXT_GENERATION: ["text", "caption", "write", "post"],
        AIDomain.CODE_GENERATION: ["code", "function", "class", "api"],
        # ... more mappings
    }
    
    for domain, keywords_list in keywords.items():
        if any(kw in note_lower for kw in keywords_list):
            return domain
    
    return None
```

## Benefits for Orchestration

### 1. Clean Correlations

```python
# Without domain: noisy correlations
patterns = [
    {"phase": "IMAGE_GEN", "error": "CUDA OOM"},
    {"phase": "CODE_GEN", "error": "SyntaxError"},  # ❌ Wrong domain
    {"phase": "IMAGE_GEN", "error": "Invalid model"},
]

# With domain: clean correlations
patterns = [
    {"domain": "IMAGE_GENERATION", "error": "CUDA OOM"},
    {"domain": "IMAGE_GENERATION", "error": "Invalid model"},
    # CODE_GEN patterns filtered out ✅
]
```

### 2. Intelligent Routing

```python
# Reasoning engine knows specialist domains
if task.ai_domain == AIDomain.IMAGE_GENERATION:
    # Only consider agents with IMAGE_GENERATION specialization
    candidates = [a for a in agents if AIDomain.IMAGE_GENERATION in a.domains]
```

### 3. Safe Cross-Domain Orchestration

```python
# Check compatibility before handoff
source_domain = context.ai_domain
target_agent_domain = agent.primary_domain

weight = engine.check_domain_compatibility(source_domain, target_agent_domain)

if weight < 0.3:
    # Domains incompatible, reject handoff
    decision = DecisionType.DEFER_TO_HUMAN
```

### 4. Domain-Specific Strategies

```python
# Different strategies for different domains
if context.ai_domain == AIDomain.IMAGE_GENERATION:
    strategies = ["use_sdxl", "4k_resolution", "face_enhancement"]
elif context.ai_domain == AIDomain.TEXT_GENERATION:
    strategies = ["creative_tone", "engagement_optimized"]
```

## Migration Checklist

- [x] Add `AIDomain` and `AISubdomain` enums to models
- [x] Define `DOMAIN_COMPATIBILITY` matrix
- [x] Add `ai_domain` and `ai_subdomain` columns to database
- [x] Update `ACDContextCreate` Pydantic model
- [x] Update `ACDContextUpdate` Pydantic model
- [x] Update `ACDContextResponse` Pydantic model
- [x] Integrate domain filtering in `ReasoningOrchestrator`
- [x] Add domain compatibility checking in `ReasoningEngine`
- [x] Update prompts to include domain information
- [x] Enhance agent capability matching with domain awareness
- [ ] Run database migration script
- [ ] Update existing contexts with inferred domains (optional)
- [ ] Update agent registrations with domain specializations

## Next Steps

1. **Run Migration**
   ```bash
   python add_domain_fields_migration.py
   ```

2. **Update Agent Registrations**
   Add domain specializations to existing agents

3. **Backfill Existing Contexts** (Optional)
   Infer and set domains for historical contexts

4. **Monitor Improvements**
   Track pattern quality and routing accuracy

## Conclusion

**AI_DOMAIN and AI_SUBDOMAIN** transform the ACD system from a flat context pile into a **hierarchical, cortical-region-separated cognitive architecture**. This enables:

- 🧠 **Clean pattern learning** - No cross-domain noise
- 🎯 **Intelligent routing** - Domain-aware agent selection
- 🔄 **Safe handoffs** - Compatibility-checked orchestration
- 📊 **Better correlations** - Domain-filtered historical data
- 🚀 **Scalability** - Clean separation supports growth

**The brain now knows which cortical region it's working in.**
