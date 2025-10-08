# RSS Feed Enhancement Documentation

## Overview

The RSS Feed Enhancement feature enables AI personas to receive curated content suggestions from assigned RSS feeds, filtered by topics and prioritized for relevance. This allows personas to stay informed about specific subjects and use trending content as inspiration for generated posts.

## Key Features

### 1. **RSS Feed Management**
- Add and manage RSS feeds from any source
- Automatic feed validation and content fetching
- Topic extraction and categorization
- Sentiment analysis on feed items

### 2. **Persona-Feed Assignment**
- Link specific RSS feeds to personas
- Filter content by topics of interest
- Set priority levels for content selection (0-100)
- Enable/disable assignments without deletion

### 3. **Content Suggestions**
- Get relevant feed items for each persona
- Automatic relevance scoring
- Topic-based filtering
- Priority-weighted selection

### 4. **Topic Organization**
- Discover feeds by topic category
- Automatic topic extraction from content
- Trending topic analysis

## Database Schema

### New Table: `persona_feeds`

```sql
CREATE TABLE persona_feeds (
    id UUID PRIMARY KEY,
    persona_id UUID NOT NULL REFERENCES personas(id) ON DELETE CASCADE,
    feed_id UUID NOT NULL REFERENCES rss_feeds(id) ON DELETE CASCADE,
    topics JSON,  -- Array of topic filters
    priority INTEGER NOT NULL DEFAULT 50,  -- 0-100
    is_active BOOLEAN NOT NULL DEFAULT TRUE,
    created_at TIMESTAMP WITH TIME ZONE NOT NULL
);

CREATE INDEX idx_persona_feeds_persona_id ON persona_feeds(persona_id);
CREATE INDEX idx_persona_feeds_feed_id ON persona_feeds(feed_id);
CREATE INDEX idx_persona_feeds_active ON persona_feeds(is_active);
```

## API Endpoints

### 1. Assign Feed to Persona

**Endpoint:** `POST /api/v1/feeds/personas/{persona_id}/feeds`

**Description:** Links an RSS feed to a persona with optional topic filtering and priority.

**Request Body:**
```json
{
  "feed_id": "uuid-of-feed",
  "topics": ["ai", "machine learning", "technology"],
  "priority": 80
}
```

**Response:**
```json
{
  "id": "uuid",
  "persona_id": "uuid",
  "feed_id": "uuid",
  "topics": ["ai", "machine learning", "technology"],
  "priority": 80,
  "is_active": true,
  "created_at": "2025-10-08T00:00:00Z",
  "feed_name": "TechCrunch",
  "feed_url": "https://techcrunch.com/feed/",
  "feed_categories": ["technology", "startups"]
}
```

**Use Cases:**
- Assign tech news feeds to tech-focused personas
- Assign business feeds to business personas
- Filter specific topics from general news feeds

### 2. List Persona's Feeds

**Endpoint:** `GET /api/v1/feeds/personas/{persona_id}/feeds`

**Description:** Returns all RSS feeds assigned to a specific persona.

**Response:**
```json
[
  {
    "id": "uuid",
    "persona_id": "uuid",
    "feed_id": "uuid",
    "topics": ["ai", "machine learning"],
    "priority": 80,
    "is_active": true,
    "created_at": "2025-10-08T00:00:00Z",
    "feed_name": "TechCrunch",
    "feed_url": "https://techcrunch.com/feed/",
    "feed_categories": ["technology", "startups"]
  }
]
```

### 3. Unassign Feed from Persona

**Endpoint:** `DELETE /api/v1/feeds/personas/{persona_id}/feeds/{feed_id}`

**Description:** Removes the assignment between a persona and a feed (soft delete - marks as inactive).

**Response:** `204 No Content`

### 4. List Feeds by Topic

**Endpoint:** `GET /api/v1/feeds/by-topic/{topic}`

**Description:** Returns all RSS feeds that contain the specified topic in their categories.

**Example:** `GET /api/v1/feeds/by-topic/technology`

**Response:**
```json
[
  {
    "id": "uuid",
    "name": "TechCrunch",
    "url": "https://techcrunch.com/feed/",
    "description": "Tech news and analysis",
    "categories": ["technology", "startups", "innovation"],
    "fetch_frequency_hours": 6,
    "last_fetched": "2025-10-08T00:00:00Z",
    "is_active": true,
    "created_at": "2025-10-08T00:00:00Z",
    "updated_at": "2025-10-08T00:00:00Z"
  }
]
```

### 5. Get Content Suggestions (Enhanced)

**Endpoint:** `GET /api/v1/feeds/suggestions/{persona_id}?limit=10`

**Description:** Returns relevant feed items from the persona's assigned feeds, filtered by their topic preferences.

**Query Parameters:**
- `limit` (optional): Maximum number of suggestions (1-20, default 10)

**Response:**
```json
[
  {
    "id": "uuid",
    "feed_id": "uuid",
    "title": "New AI Breakthrough in Machine Learning",
    "link": "https://example.com/article",
    "description": "Researchers announce...",
    "published_date": "2025-10-08T00:00:00Z",
    "author": "John Doe",
    "categories": ["ai", "research"],
    "content_summary": "...",
    "sentiment_score": 0.8,
    "relevance_score": 0.95,
    "keywords": ["ai", "machine learning", "research"],
    "entities": [
      {"text": "OpenAI", "type": "ORGANIZATION", "confidence": 0.9}
    ],
    "topics": ["artificial intelligence", "machine learning"],
    "processed": true,
    "created_at": "2025-10-08T00:00:00Z"
  }
]
```

## Workflow Example

### Setting Up a Tech Persona with News Feeds

```python
import httpx

base_url = "http://localhost:8000"

# 1. Create a persona
persona_response = httpx.post(f"{base_url}/api/v1/personas/", json={
    "name": "Tech Sarah",
    "appearance": "Professional tech entrepreneur",
    "personality": "Innovative and forward-thinking",
    "content_themes": ["technology", "ai", "startups"],
    "style_preferences": {"tone": "professional"}
})
persona_id = persona_response.json()["id"]

# 2. Add RSS feeds
feeds_to_add = [
    {
        "name": "TechCrunch",
        "url": "https://techcrunch.com/feed/",
        "categories": ["technology", "startups"],
        "fetch_frequency_hours": 6
    },
    {
        "name": "Wired AI",
        "url": "https://www.wired.com/feed/tag/ai/latest/rss",
        "categories": ["technology", "ai"],
        "fetch_frequency_hours": 12
    }
]

feed_ids = []
for feed_data in feeds_to_add:
    response = httpx.post(f"{base_url}/api/v1/feeds/", json=feed_data)
    feed_ids.append(response.json()["id"])

# 3. Assign feeds to persona with topic filtering
httpx.post(
    f"{base_url}/api/v1/feeds/personas/{persona_id}/feeds",
    json={
        "feed_id": feed_ids[0],
        "topics": ["ai", "machine learning", "startups"],
        "priority": 90
    }
)

httpx.post(
    f"{base_url}/api/v1/feeds/personas/{persona_id}/feeds",
    json={
        "feed_id": feed_ids[1],
        "topics": ["artificial intelligence", "robotics"],
        "priority": 85
    }
)

# 4. Fetch feed content
httpx.post(f"{base_url}/api/v1/feeds/fetch")

# 5. Get content suggestions for the persona
suggestions = httpx.get(
    f"{base_url}/api/v1/feeds/suggestions/{persona_id}?limit=10"
).json()

# Use suggestions to inform content generation
for item in suggestions:
    print(f"Topic: {item['title']}")
    print(f"Sentiment: {item['sentiment_score']}")
    print(f"Topics: {', '.join(item['topics'])}")
```

## Content Processing Pipeline

### 1. Feed Ingestion
- RSS feed is fetched via HTTP
- Content is parsed using `feedparser`
- Items are deduplicated by link URL

### 2. Content Analysis
Each feed item undergoes:

- **Topic Extraction**: Identifies main topics using keyword analysis
- **Sentiment Analysis**: Scores content from -1 (negative) to 1 (positive)
- **Entity Extraction**: Identifies people, organizations, locations
- **Keyword Extraction**: Extracts relevant terms
- **Relevance Scoring**: Calculates relevance (0-1) based on recency and feed category matches

### 3. Content Categorization
Topics are classified into categories:
- Technology
- Business
- Politics
- Health
- Science
- Sports
- Entertainment
- Environment

### 4. Persona Matching
When requesting content suggestions:
1. Retrieve persona's assigned feeds
2. Filter by assigned topics (if specified)
3. Apply relevance scoring
4. Sort by priority and relevance
5. Return top N items

## Configuration

### Feed Fetch Frequency
Set via `fetch_frequency_hours` when adding a feed (default: 6 hours).

### Content Retention
Feed items are retained indefinitely but queries typically look at the last 48 hours for suggestions.

### Priority Levels
- **90-100**: Critical feeds (always prioritized)
- **70-89**: High priority
- **50-69**: Medium priority (default)
- **1-49**: Low priority
- **0**: Archived (not considered)

## Monitoring & Maintenance

### Check Feed Health
```bash
GET /api/v1/feeds/
```
Look for:
- `last_fetched` timestamp
- `is_active` status

### Manual Fetch
Trigger immediate fetch of all active feeds:
```bash
POST /api/v1/feeds/fetch
```

### Trending Topics
Monitor what's trending across all feeds:
```bash
GET /api/v1/feeds/trending?limit=20&hours=24
```

## Best Practices

### 1. Feed Selection
- Choose high-quality, reliable RSS feeds
- Prefer feeds with good metadata (categories, authors)
- Balance general and niche feeds

### 2. Topic Filtering
- Use specific topics for focused content
- Empty topic list = accept all content from feed
- Topics are case-insensitive

### 3. Priority Management
- Reserve 90+ for most important feeds
- Use priority to balance diverse content sources
- Adjust based on content quality over time

### 4. Persona Configuration
- Align feed topics with persona's `content_themes`
- Assign 3-5 feeds per persona for diversity
- Review suggestions regularly to ensure quality

## Troubleshooting

### No Suggestions Returned

**Possible Causes:**
1. No feeds assigned to persona
2. No recent content in assigned feeds
3. Topic filters too restrictive
4. Feeds haven't been fetched yet

**Solutions:**
```bash
# Check assigned feeds
GET /api/v1/feeds/personas/{persona_id}/feeds

# Manually fetch feeds
POST /api/v1/feeds/fetch

# Check feed items exist
GET /api/v1/feeds/
```

### Feed Validation Fails

**Possible Causes:**
1. Invalid URL format
2. URL doesn't point to RSS/Atom feed
3. Network connectivity issues
4. Feed requires authentication

**Solutions:**
- Verify URL in browser
- Check feed format (RSS 2.0, Atom)
- Test with curl: `curl -L {feed_url}`

### Low Relevance Scores

**Possible Causes:**
1. Content doesn't match persona themes
2. Feed categories misaligned
3. Old content

**Solutions:**
- Adjust topic filters
- Review feed selection
- Increase fetch frequency

## Performance Considerations

### Database Indexing
The implementation includes indexes on:
- `persona_feeds.persona_id`
- `persona_feeds.feed_id`
- `persona_feeds.is_active`
- `feed_items.feed_id`
- `feed_items.created_at`
- `feed_items.processed`

### Query Optimization
- Content suggestions query is limited to 2x requested items
- Only processed items are considered
- Recent content (48 hours) is prioritized

### Scalability
- Feed fetching runs asynchronously
- Can handle 100+ feeds per instance
- Consider cron job for automated fetching

## Future Enhancements

Potential improvements:
- [ ] Advanced NLP for better topic extraction
- [ ] ML-based relevance scoring
- [ ] Multi-language support
- [ ] Feed deduplication across sources
- [ ] Content summarization
- [ ] Image extraction from feeds
- [ ] Webhook notifications for new content
- [ ] Feed health monitoring
- [ ] A/B testing for content selection

## Testing

Run the test suite:
```bash
pytest tests/unit/test_rss_persona_assignment.py -v
```

Test coverage includes:
- Feed assignment and unassignment
- Topic filtering
- Content suggestions
- Priority management
- Edge cases (non-existent feeds, empty results)

## Support

For issues or questions:
1. Check this documentation
2. Review test cases for usage examples
3. Run `python demo_rss_enhancements.py` for interactive demo
4. Check API logs for errors
