# Interactive Content Feature Implementation

## Overview

The Interactive Content feature enables AI personas to create and manage engaging interactive content including polls, stories, Q&A sessions, and quizzes. This feature was outlined in the Q2 2025 roadmap in `docs/ENHANCEMENTS_ROADMAP.md`.

## Features Implemented

### 1. Interactive Content Types

**Supported Types:**
- **Polls**: Multi-choice questions with vote tracking and real-time results
- **Stories**: Ephemeral content with automatic 24-hour expiration
- **Q&A**: Question and answer sessions with AI personas
- **Quizzes**: Educational content with answer validation

### 2. Database Models

**Tables Created:**
- `interactive_content` - Stores interactive content items
- `interactive_content_responses` - Tracks user responses

**Key Fields:**
- Content type, title, question, description
- Options (for polls/quizzes)
- Expiration timestamps (for stories)
- View count, response count, share count
- Status tracking (draft, active, expired, archived)

### 3. Service Layer

**InteractiveContentService** provides:
- Create, read, update, delete operations
- Publishing and status management
- Response submission and tracking
- Statistics and analytics
- Automatic expiration handling

### 4. API Endpoints

**Base URL:** `/api/v1/interactive`

#### Core Operations
- `POST /` - Create new interactive content
- `GET /{content_id}` - Get content by ID (increments view count)
- `GET /` - List content with filters (persona, type, status)
- `PUT /{content_id}` - Update content
- `DELETE /{content_id}` - Delete content

#### Actions
- `POST /{content_id}/publish` - Publish content (activate)
- `POST /{content_id}/respond` - Submit a response
- `POST /{content_id}/share` - Increment share count
- `GET /{content_id}/stats` - Get content statistics
- `GET /health` - Service health check

### 5. Usage Examples

#### Creating a Poll

```python
import httpx

response = httpx.post(
    "http://localhost:8000/api/v1/interactive/",
    json={
        "persona_id": "123e4567-e89b-12d3-a456-426614174000",
        "content_type": "poll",
        "title": "Favorite Workout?",
        "question": "What's your favorite type of workout?",
        "options": [
            {"text": "Running"},
            {"text": "Yoga"},
            {"text": "Weight training"},
            {"text": "Swimming"}
        ]
    }
)
poll = response.json()
```

#### Creating a Story with Expiration

```python
response = httpx.post(
    "http://localhost:8000/api/v1/interactive/",
    json={
        "persona_id": "123e4567-e89b-12d3-a456-426614174000",
        "content_type": "story",
        "title": "My morning routine",
        "media_url": "https://example.com/story.jpg"
    }
)
# Stories automatically expire in 24 hours
```

#### Publishing Content

```python
response = httpx.post(
    f"http://localhost:8000/api/v1/interactive/{poll['id']}/publish"
)
published_poll = response.json()
```

#### Submitting a Response

```python
response = httpx.post(
    f"http://localhost:8000/api/v1/interactive/{poll['id']}/respond",
    json={
        "content_id": poll['id'],
        "response_data": {"option_id": 1}
    }
)
```

#### Getting Statistics

```python
response = httpx.get(
    f"http://localhost:8000/api/v1/interactive/{poll['id']}/stats"
)
stats = response.json()
print(f"Response rate: {stats['response_rate']}%")
print(f"Top options: {stats['top_options']}")
```

### 6. Response Data Formats

**For Polls:**
```json
{
  "option_id": 1
}
```

**For Q&A:**
```json
{
  "answer": "Your answer text here"
}
```

**For Quizzes:**
```json
{
  "answers": [1, 3, 4]
}
```

### 7. Filtering and Listing

```python
# Get all active polls for a persona
response = httpx.get(
    "http://localhost:8000/api/v1/interactive/",
    params={
        "persona_id": "123e4567-e89b-12d3-a456-426614174000",
        "content_type": "poll",
        "status": "active"
    }
)

# Get all content including expired
response = httpx.get(
    "http://localhost:8000/api/v1/interactive/",
    params={
        "persona_id": "123e4567-e89b-12d3-a456-426614174000",
        "include_expired": True
    }
)
```

## Testing

**Test Coverage:**
- 11 comprehensive unit tests
- All tests passing
- Tests cover:
  - Content creation (polls, stories, Q&A)
  - Publishing and status changes
  - Response submission and vote counting
  - Statistics calculation
  - Expiration handling
  - Filtering and listing

**Run Tests:**
```bash
python -m pytest tests/unit/test_interactive_content.py -v
```

## Database Schema

### interactive_content Table
```sql
CREATE TABLE interactive_content (
    id UUID PRIMARY KEY,
    persona_id UUID REFERENCES personas(id),
    content_type VARCHAR(20) NOT NULL,
    title VARCHAR(200),
    question TEXT,
    description TEXT,
    options JSON,
    responses JSON,
    media_url VARCHAR(500),
    status VARCHAR(20) DEFAULT 'draft',
    view_count INTEGER DEFAULT 0,
    response_count INTEGER DEFAULT 0,
    share_count INTEGER DEFAULT 0,
    published_at TIMESTAMP,
    expires_at TIMESTAMP,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);
```

### interactive_content_responses Table
```sql
CREATE TABLE interactive_content_responses (
    id UUID PRIMARY KEY,
    content_id UUID REFERENCES interactive_content(id),
    user_id UUID REFERENCES users(id),
    response_data JSON NOT NULL,
    user_identifier VARCHAR(100),
    created_at TIMESTAMP DEFAULT NOW()
);
```

## Future Enhancements

1. **Real-time Updates**: WebSocket support for live poll results
2. **Advanced Analytics**: Detailed engagement metrics and trends
3. **Social Sharing**: Integration with social media platforms
4. **Notifications**: Alert users when content expires or new polls are available
5. **Leaderboards**: For quizzes and competitive content
6. **Rich Media**: Support for video/audio in stories
7. **Scheduled Publishing**: Automatic publishing at specified times

## Performance Considerations

- Indexes on `persona_id`, `content_type`, `status`, `expires_at`
- Efficient querying with filters
- Automatic expiration handling
- View count increment is non-blocking

## Security

- User authentication for responses (optional anonymous)
- Content validation
- Rate limiting on response submissions
- Input sanitization

## Integration with Existing Features

- Works seamlessly with persona management
- Integrates with content generation pipeline
- Complements audience segmentation for targeted polls
- Analytics data feeds into performance metrics

## Conclusion

The Interactive Content feature provides a robust foundation for engaging users with AI personas through polls, stories, Q&A, and quizzes. The implementation follows best practices with comprehensive testing, clean API design, and scalable database architecture.
