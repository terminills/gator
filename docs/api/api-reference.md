# Gator Platform API Reference

> **Complete API documentation for all endpoints**

This document provides a comprehensive reference for all API endpoints available in the Gator AI Platform. Access the interactive API documentation at `http://localhost:8000/docs`.

## Table of Contents

1. [Core API Endpoints](#core-api-endpoints)
2. [Diagnostics API](#diagnostics-api)
3. [Reasoning Orchestrator API](#reasoning-orchestrator-api)
4. [System Monitoring API](#system-monitoring-api)
5. [Content Moderation API](#content-moderation-api)
6. [Scheduled Posts API](#scheduled-posts-api)
7. [Cache Management API](#cache-management-api)
8. [ACD (Autonomous Continuous Development) API](#acd-api)

---

## Core API Endpoints

### Health Check

**GET** `/health`

Check system health status.

```json
{
  "status": "healthy",
  "database": "connected",
  "timestamp": "2024-11-30T12:00:00Z"
}
```

### Root

**GET** `/`

Get platform information.

```json
{
  "message": "Gator AI Influencer Platform",
  "version": "0.1.0",
  "status": "operational"
}
```

---

## Diagnostics API

Base URL: `/api/v1/diagnostics`

The Diagnostics API provides comprehensive insights into AI activity, content generation attempts, and system performance.

### Get AI Activity Summary

**GET** `/api/v1/diagnostics/ai-activity`

Returns a comprehensive summary of AI generation activity.

**Query Parameters:**
- `hours` (int, default: 24): Time window in hours (1-168)

**Response:**
```json
{
  "total_generations": 100,
  "successful_generations": 85,
  "failed_generations": 10,
  "fallback_generations": 5,
  "success_rate": 85.0,
  "by_content_type": {
    "image": 60,
    "text": 30,
    "video": 10
  },
  "by_persona": {
    "persona-uuid-1": 50,
    "persona-uuid-2": 50
  },
  "active_contexts": 5,
  "completed_contexts": 90,
  "failed_contexts": 5,
  "recent_errors": [
    {
      "timestamp": "2024-11-30T12:00:00Z",
      "event_type": "GENERATION_ERROR",
      "error_message": "Model not available",
      "error_file": "generation_service.py",
      "error_line": 150,
      "acd_context_id": "context-uuid"
    }
  ]
}
```

### Get Generation Attempts

**GET** `/api/v1/diagnostics/generation-attempts`

Returns detailed list of content generation attempts.

**Query Parameters:**
- `hours` (int, default: 24): Time window in hours
- `content_type` (string, optional): Filter by content type
- `persona_id` (UUID, optional): Filter by persona
- `status` (string, optional): Filter by status (success, failed, fallback)
- `limit` (int, default: 50): Maximum results (1-200)

**Response:**
```json
[
  {
    "content_id": "content-uuid",
    "persona_id": "persona-uuid",
    "persona_name": "Tech Influencer",
    "content_type": "image",
    "status": "success",
    "created_at": "2024-11-30T12:00:00Z",
    "has_acd_context": true,
    "acd_context_id": "context-uuid",
    "ai_phase": "GENERATION",
    "ai_state": "DONE",
    "ai_confidence": "HIGH",
    "ai_note": "Generated successfully",
    "prompt": "A professional workspace photo",
    "quality": "high",
    "using_fallback": false,
    "error_message": null,
    "file_path": "/content/image.jpg"
  }
]
```

### Get AI Model Activity

**GET** `/api/v1/diagnostics/ai-models`

Returns AI model usage statistics.

**Query Parameters:**
- `hours` (int, default: 24): Time window in hours

**Response:**
```json
[
  {
    "model_name": "sdxl-1.0",
    "provider": "local",
    "total_calls": 100,
    "successful_calls": 95,
    "failed_calls": 5,
    "avg_generation_time": 8.5
  }
]
```

### Get ACD Contexts

**GET** `/api/v1/diagnostics/acd-contexts`

Returns ACD contexts with full details.

**Query Parameters:**
- `hours` (int, default: 24): Time window in hours
- `phase` (string, optional): Filter by phase
- `state` (string, optional): Filter by state
- `limit` (int, default: 50): Maximum results

**Response:**
```json
[
  {
    "id": "context-uuid",
    "ai_phase": "GENERATION",
    "ai_state": "DONE",
    "ai_confidence": "HIGH",
    "ai_note": "Completed successfully",
    "created_at": "2024-11-30T12:00:00Z"
  }
]
```

---

## Reasoning Orchestrator API

Base URL: `/api/v1/reasoning`

The Reasoning Orchestrator provides decision-making capabilities for the ACD system, acting as the "basal ganglia" of the platform.

### Request Orchestration Decision

**POST** `/api/v1/reasoning/orchestrate`

Request an orchestration decision for an ACD context.

**Request Body:**
```json
{
  "context_id": "acd-context-uuid",
  "current_agent": "content_generator",
  "additional_context": {
    "priority": "high"
  }
}
```

**Response:**
```json
{
  "context_id": "acd-context-uuid",
  "decision_type": "PROCEED",
  "confidence": "HIGH",
  "reasoning": "Content generation is on track with high quality indicators",
  "target_agent": null,
  "action_plan": {
    "steps": ["continue", "validate"],
    "estimated_time": 30
  },
  "learned_patterns": ["high_quality_prompt", "optimal_timing"],
  "risk_assessment": "low",
  "timestamp": "2024-11-30T12:00:00Z"
}
```

### Orchestrate and Execute

**POST** `/api/v1/reasoning/orchestrate-and-execute`

Request orchestration decision and immediately execute it.

**Request Body:**
```json
{
  "context_id": "acd-context-uuid",
  "current_agent": "content_generator",
  "additional_context": {}
}
```

**Response:**
```json
{
  "context_id": "acd-context-uuid",
  "executed": true,
  "decision_type": "PROCEED",
  "message": "Decision executed: PROCEED"
}
```

### Execute Previous Decision

**POST** `/api/v1/reasoning/execute`

Execute a previously made orchestration decision.

**Request Body:**
```json
{
  "context_id": "acd-context-uuid",
  "execute_immediately": true
}
```

### Learn from Outcome

**POST** `/api/v1/reasoning/learn`

Record learning from an orchestration outcome.

**Request Body:**
```json
{
  "context_id": "acd-context-uuid",
  "success": true,
  "outcome_metadata": {
    "engagement_rate": 0.15,
    "quality_score": 0.92
  }
}
```

**Response:**
```json
{
  "context_id": "acd-context-uuid",
  "learned": true,
  "message": "Pattern reinforced for context acd-context-uuid"
}
```

### Get Orchestration Statistics

**GET** `/api/v1/reasoning/stats`

Get orchestration statistics.

**Query Parameters:**
- `hours` (int, default: 24): Time window for stats

**Response:**
```json
{
  "time_window_hours": 24,
  "total_decisions": 150,
  "decision_types": {
    "PROCEED": 100,
    "HANDOFF": 30,
    "PAUSE": 15,
    "RETRY": 5
  },
  "confidence_levels": {
    "HIGH": 80,
    "MEDIUM": 50,
    "LOW": 20
  },
  "outcomes": {
    "successful": 130,
    "failed": 10,
    "pending": 10
  },
  "success_rate": 86.67,
  "handoffs": {
    "quality_reviewer": 20,
    "content_moderator": 10
  },
  "learning_enabled": true,
  "basal_ganglia_active": true
}
```

---

## System Monitoring API

Base URL: `/api/v1/system`

### Get GPU Temperatures

**GET** `/api/v1/system/gpu/temperature`

Returns current GPU temperatures for all devices.

**Response:**
```json
{
  "gpus": [
    {
      "index": 0,
      "name": "AMD Radeon RX 7900 XTX",
      "temperature_c": 65,
      "status": "normal"
    }
  ],
  "max_temperature": 65,
  "avg_temperature": 65
}
```

### Get GPU Status

**GET** `/api/v1/system/gpu/status`

Returns detailed GPU status including memory usage.

### Get Temperature History

**GET** `/api/v1/system/gpu/temperature/history`

Returns GPU temperature history.

**Query Parameters:**
- `hours` (int, default: 24): Time window for history

### Get Fan Status

**GET** `/api/v1/system/fans`

Returns current fan status.

### Set Fan Mode

**POST** `/api/v1/system/fans/mode`

Set fan control mode.

**Request Body:**
```json
{
  "mode": "auto"
}
```

### Set Fan Speed

**POST** `/api/v1/system/fans/speed`

Set fan speed manually.

**Request Body:**
```json
{
  "speed_percent": 50,
  "zone": "system"
}
```

### Auto-adjust Fans

**POST** `/api/v1/system/fans/auto-adjust`

Auto-adjust fan speed based on GPU temperature.

**Request Body:**
```json
{
  "target_temperature": 70
}
```

---

## Content Moderation API

Base URL: `/api/v1/moderation`

### Analyze Content

**POST** `/api/v1/moderation/analyze`

Analyze content for moderation.

**Request Body:**
```json
{
  "content_type": "text",
  "content": "Content to analyze",
  "persona_id": "persona-uuid"
}
```

### Get Moderation Queue

**GET** `/api/v1/moderation/queue`

Returns content pending moderation.

### Approve Content

**POST** `/api/v1/moderation/queue/{item_id}/approve`

Approve content in the moderation queue.

### Reject Content

**POST** `/api/v1/moderation/queue/{item_id}/reject`

Reject content in the moderation queue.

**Request Body:**
```json
{
  "reason": "Content violates guidelines"
}
```

### Get Moderation Stats

**GET** `/api/v1/moderation/stats`

Returns moderation statistics.

### Get Categories

**GET** `/api/v1/moderation/categories`

Returns available moderation categories.

---

## Scheduled Posts API

Base URL: `/api/v1/scheduled-posts`

### Create Scheduled Post

**POST** `/api/v1/scheduled-posts/`

Create a new scheduled post.

**Request Body:**
```json
{
  "content_id": "content-uuid",
  "platform": "instagram",
  "scheduled_at": "2024-11-30T18:00:00Z",
  "caption": "Check out my new post!",
  "hashtags": ["ai", "tech"]
}
```

### List Scheduled Posts

**GET** `/api/v1/scheduled-posts/`

Returns list of scheduled posts.

**Query Parameters:**
- `page` (int, default: 1): Page number
- `page_size` (int, default: 20): Items per page
- `status` (string, optional): Filter by status
- `platform` (string, optional): Filter by platform
- `persona_id` (UUID, optional): Filter by persona

### Get Scheduling Stats

**GET** `/api/v1/scheduled-posts/stats`

Returns scheduling statistics.

### Get Upcoming Posts

**GET** `/api/v1/scheduled-posts/upcoming`

Returns upcoming scheduled posts.

---

## Cache Management API

Base URL: `/api/v1/cache`

### Get Cache Status

**GET** `/api/v1/cache/status`

Returns cache connection status and statistics.

**Response:**
```json
{
  "status": "connected",
  "memory_used": "50MB",
  "total_keys": 1500,
  "hit_rate": 0.85
}
```

### Cache Health Check

**GET** `/api/v1/cache/health`

Check if cache is healthy and responsive.

**Response:**
```json
{
  "status": "healthy",
  "message": "Cache is operational"
}
```

### Invalidate Cache by Prefix

**DELETE** `/api/v1/cache/invalidate/{prefix}`

Invalidate all cache entries matching a prefix.

**Supported Prefixes:**
- `persona` - Persona-related cache
- `user` - User-related cache
- `session` - Session data
- `generation` - Content generation cache
- `acd` - ACD context cache
- `api` - API response cache

**Response:**
```json
{
  "status": "success",
  "prefix": "persona",
  "keys_deleted": 150
}
```

---

## ACD API

Base URL: `/api/v1/acd`

The ACD (Autonomous Continuous Development) API provides comprehensive context management for AI operations.

### List Contexts

**GET** `/api/v1/acd/contexts`

Returns list of ACD contexts.

### Create Context

**POST** `/api/v1/acd/contexts/`

Create a new ACD context.

### Get Context

**GET** `/api/v1/acd/contexts/{context_id}`

Get a specific ACD context.

### Update Context

**PUT** `/api/v1/acd/contexts/{context_id}`

Update an ACD context.

### Get ACD Stats

**GET** `/api/v1/acd/stats/`

Returns ACD system statistics.

### Rate Content (HIL)

**POST** `/api/v1/acd/rate/{context_id}`

Submit a human-in-the-loop rating for content.

**Request Body:**
```json
{
  "rating": 4,
  "feedback": "Great content, very engaging"
}
```

### Get HIL Ratings

**GET** `/api/v1/acd/ratings`

Returns all HIL ratings.

### Get Rating Stats

**GET** `/api/v1/acd/ratings/stats`

Returns HIL rating statistics.

---

## Error Handling

All endpoints return standard error responses:

### 400 Bad Request
```json
{
  "detail": "Invalid request data"
}
```

### 401 Unauthorized
```json
{
  "detail": "Authentication required"
}
```

### 404 Not Found
```json
{
  "detail": "Resource not found"
}
```

### 422 Validation Error
```json
{
  "detail": [
    {
      "loc": ["body", "field_name"],
      "msg": "field required",
      "type": "value_error.missing"
    }
  ]
}
```

### 500 Internal Server Error
```json
{
  "detail": "An internal error occurred"
}
```

---

## Authentication

Most endpoints require authentication. Include the JWT token in the Authorization header:

```
Authorization: Bearer <token>
```

To obtain a token, use the authentication endpoints:

### Login
**POST** `/api/v1/auth/login`

### Register
**POST** `/api/v1/auth/register`

---

## Rate Limiting

API requests are rate-limited to prevent abuse:

- **Default**: 100 requests/minute per IP
- **Authenticated**: 1000 requests/minute per user

Rate limit headers are included in responses:
```
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 95
X-RateLimit-Reset: 1704067200
```

---

## Interactive Documentation

Access the full interactive API documentation at:
- **Swagger UI**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`
