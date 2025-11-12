# ACD Service Content Generation Fix

## Summary

Fixed three critical issues with the ACD (Autonomous Continuous Development) service that prevented it from actually triggering content generation.

## Issues Addressed

### Issue 1: ACD Service Only Logged to Database ✅ FIXED
**Problem**: The ACD service created context records but never triggered actual content generation.

**Solution**: 
- Added `process_queued_contexts()` method to `ACDService`
- Queries database for QUEUED/ASSIGNED contexts with READY/PROCESSING state
- Triggers `ContentGenerationService.generate_content()` for each context
- Updates context status through the generation lifecycle (IN_PROGRESS → DONE/FAILED)
- Creates trace artifacts for failures
- Returns detailed processing summary

### Issue 2: System Preferred vLLM Over llama.cpp ✅ FIXED
**Problem**: System defaulted to vLLM which wasn't available, then returned placeholder text instead of falling back.

**Solution**:
- Changed `llama-3.1-8b` default inference engine from `vllm` to `llama.cpp`
- Added `fallback_engines` list to model configuration: `['vllm', 'transformers']`
- Implemented intelligent fallback in `_generate_text_local()`:
  - Tries primary engine (llama.cpp)
  - On failure, tries each fallback engine in order
  - Raises error only if all engines fail
- Fixed `_generate_text_vllm()` to raise errors instead of returning placeholder text

### Issue 3: System Didn't Auto-Download Models ✅ FIXED
**Problem**: Missing models caused generation failures with no automatic recovery.

**Solution**:
- Added `download_model_from_huggingface()` function
- Uses `huggingface_hub.snapshot_download` for model acquisition
- Integrated into `_generate_text_local()` fallback chain
- Attempts download before trying each inference engine
- Progress logging for download tracking

## Implementation Details

### New Functions

#### `download_model_from_huggingface()`
```python
async def download_model_from_huggingface(
    model_id: str,
    model_path: Path,
    model_type: str = "text"
) -> bool:
    """Download a model from Hugging Face Hub."""
```

Located in: `src/backend/services/ai_models.py`

Downloads models from Hugging Face when needed, with:
- Resume support for interrupted downloads
- Progress logging
- Error handling and reporting

### Modified Methods

#### `ACDService.process_queued_contexts()`
```python
async def process_queued_contexts(
    self,
    max_contexts: int = 10,
    phase_filter: Optional[str] = None
) -> Dict[str, Any]:
    """Process queued ACD contexts and trigger content generation."""
```

Located in: `src/backend/services/acd_service.py`

Features:
- SQL CASE-based priority ordering: CRITICAL > HIGH > NORMAL > LOW > DEFERRED
- Batch processing with configurable limit
- Optional phase filtering
- Comprehensive error handling
- Trace artifact creation for failures
- Detailed result reporting

#### `AIModelManager._generate_text_local()`
Enhanced with:
- Multi-engine fallback support
- Auto-download integration
- Proper error propagation
- Per-engine error logging

### API Endpoints

#### POST `/api/v1/acd/process-queue/`
Process queued ACD contexts and trigger content generation.

**Query Parameters:**
- `max_contexts` (int, 1-100): Maximum contexts to process (default: 10)
- `phase` (str, optional): Filter by phase (e.g., "TEXT_GENERATION")

**Response:**
```json
{
  "status": "processing_complete",
  "summary": {
    "processed": 5,
    "successful": 4,
    "failed": 1,
    "results": [
      {
        "context_id": "uuid",
        "status": "success",
        "content_id": "uuid",
        "phase": "TEXT_GENERATION"
      }
    ]
  }
}
```

## Testing

### Test Suite: `test_acd_content_generation.py`

**Test 1: ACD Triggers Content Generation** ✅ PASSED
- Creates persona and ACD context
- Calls `process_queued_contexts()`
- Verifies content generation was triggered
- Validates context state transitions

**Test 2: Model Configuration** ✅ PASSED
- Checks llama-3.1-8b uses llama.cpp as primary engine
- Verifies fallback engines are configured
- Validates configuration without requiring model availability

**Test 3: Queue Priority Processing** ✅ PASSED
- Creates contexts with different priorities
- Validates SQL CASE ordering works correctly
- Confirms CRITICAL > HIGH > NORMAL > LOW ordering

### Test Results
```
✅ PASSED - ACD Triggers Generation
✅ PASSED - Model Fallback Logic
✅ PASSED - Queue Priority Processing

✅ ALL TESTS PASSED
```

## Usage Examples

### Create and Process ACD Context

```python
from backend.services.acd_service import ACDService
from backend.models.acd import (
    ACDContextCreate,
    AIStatus,
    AIState,
    AIQueuePriority,
    AIQueueStatus,
)

# Create ACD context
context = await acd_service.create_context(
    ACDContextCreate(
        ai_phase="TEXT_GENERATION",
        ai_status=AIStatus.IMPLEMENTED,
        ai_state=AIState.READY,
        ai_queue_priority=AIQueuePriority.HIGH,
        ai_queue_status=AIQueueStatus.QUEUED,
        ai_context={
            "persona_id": "...",
            "prompt": "Generate engaging content about AI",
            "quality": "high",
        },
    )
)

# Process queue to trigger generation
results = await acd_service.process_queued_contexts(max_contexts=10)

print(f"Processed: {results['processed']}")
print(f"Successful: {results['successful']}")
print(f"Failed: {results['failed']}")
```

### API Usage

```bash
# Process all queued contexts
curl -X POST "http://localhost:8000/api/v1/acd/process-queue/?max_contexts=10"

# Process only TEXT_GENERATION contexts
curl -X POST "http://localhost:8000/api/v1/acd/process-queue/?phase=TEXT_GENERATION&max_contexts=5"
```

### Scheduled Processing

For autonomous operation, set up a scheduler (cron, celery, etc.) to call the endpoint:

```python
# Example with AsyncIO periodic task
async def process_queue_periodically():
    while True:
        await acd_service.process_queued_contexts(max_contexts=20)
        await asyncio.sleep(300)  # Every 5 minutes
```

## Files Modified

1. **src/backend/services/ai_models.py**
   - Added `download_model_from_huggingface()` function
   - Changed llama-3.1-8b inference engine to llama.cpp
   - Added fallback_engines to model configuration
   - Enhanced `_generate_text_local()` with fallback logic
   - Fixed `_generate_text_vllm()` to raise errors

2. **src/backend/services/acd_service.py**
   - Added `process_queued_contexts()` method
   - Implemented SQL CASE priority ordering
   - Added batch processing with error handling
   - Integrated with ContentGenerationService

3. **src/backend/api/routes/acd.py**
   - Added POST `/api/v1/acd/process-queue/` endpoint
   - Query parameters for max_contexts and phase filtering
   - Returns processing summary

4. **test_acd_content_generation.py** (NEW)
   - Comprehensive test suite
   - Tests ACD generation trigger
   - Tests model configuration
   - Tests priority ordering

## Security

✅ **CodeQL Analysis**: No security vulnerabilities detected

Key security considerations:
- Input validation on API endpoints
- Proper error handling prevents information leakage
- No hardcoded credentials
- Safe SQL queries using SQLAlchemy ORM
- Proper exception handling in download operations

## Performance Impact

- Minimal overhead for queue processing
- Batch processing limits prevent resource exhaustion
- Priority-based ordering ensures critical tasks execute first
- Auto-download only occurs when models are missing
- Asynchronous operations prevent blocking

## Migration Guide

No database migrations required. The existing ACD tables support all new functionality.

To enable autonomous processing:

1. Create ACD contexts with appropriate queue status:
   ```python
   ai_queue_status=AIQueueStatus.QUEUED,
   ai_state=AIState.READY,
   ```

2. Call process queue periodically:
   ```python
   await acd_service.process_queued_contexts()
   ```

3. Or use the API endpoint:
   ```bash
   curl -X POST "http://localhost:8000/api/v1/acd/process-queue/"
   ```

## Future Enhancements

Potential improvements for future releases:

1. **Automatic Scheduling**: Built-in scheduler for queue processing
2. **Distributed Processing**: Support for multiple workers
3. **Advanced Routing**: Route contexts to specific agents/workers
4. **Performance Metrics**: Track generation times and success rates
5. **Resource Management**: Dynamic batch sizing based on system load
6. **Retry Logic**: Automatic retry with exponential backoff for failures

## Conclusion

The ACD service is now fully functional and can autonomously:
- ✅ Trigger content generation from queued contexts
- ✅ Fall back intelligently between inference engines
- ✅ Auto-download missing models
- ✅ Process contexts in priority order
- ✅ Track and report processing results

This enables true autonomous content generation capability for the Gator platform.
