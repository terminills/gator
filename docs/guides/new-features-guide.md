# Gator Platform User Guide - New Features

> **Guide to using the new admin panel features and tabs**

This guide covers the new features added to the Gator AI Platform admin panel, including scheduling, monitoring, moderation, authentication, caching, and more.

## Table of Contents

1. [Accessing the Admin Panel](#accessing-the-admin-panel)
2. [Scheduling Tab](#scheduling-tab)
3. [System Monitoring Tab](#system-monitoring-tab)
4. [Content Moderation Tab](#content-moderation-tab)
5. [Authentication & OAuth Tab](#authentication--oauth-tab)
6. [Cache Management Tab](#cache-management-tab)
7. [Groups & Reels Tab](#groups--reels-tab)
8. [ACD (AI Development) Tab](#acd-ai-development-tab)
9. [Agents Tab](#agents-tab)
10. [ML Learning Tab](#ml-learning-tab)

---

## Accessing the Admin Panel

The admin panel can be accessed at: `http://localhost:8000/admin`

The panel provides a unified interface for managing all aspects of your Gator AI platform.

---

## Scheduling Tab

📅 **Purpose**: Schedule content to be published to social media platforms at specific times.

### Features

1. **Create Scheduled Posts**
   - Select content to schedule
   - Choose target platform (Instagram, TikTok, Twitter, etc.)
   - Set date and time for publication
   - Add custom captions and hashtags

2. **View Scheduled Queue**
   - See all upcoming scheduled posts
   - Filter by status (pending, completed, failed)
   - Filter by platform
   - Filter by persona

3. **Manage Posts**
   - Pause/resume scheduled posts
   - Retry failed posts
   - Cancel scheduled posts
   - Edit post details before publication

4. **Scheduling Statistics**
   - View success rates
   - Track posts by platform
   - Monitor optimal posting times

### API Endpoints

- `GET /api/v1/scheduled-posts/` - List scheduled posts
- `POST /api/v1/scheduled-posts/` - Create scheduled post
- `GET /api/v1/scheduled-posts/stats` - Get statistics
- `GET /api/v1/scheduled-posts/upcoming` - Get upcoming posts

---

## System Monitoring Tab

📊 **Purpose**: Monitor GPU temperatures, fan control, and hardware status.

### Features

1. **GPU Temperature Monitoring**
   - Real-time temperature readings
   - Temperature history graphs
   - Maximum temperature alerts
   - Per-GPU status breakdown

2. **Fan Control**
   - Set fan control mode (auto/manual)
   - Adjust fan speed percentage
   - Set temperature thresholds
   - Zone-specific control (system, CPU, peripheral)

3. **IPMI Integration**
   - Configure IPMI credentials
   - Remote hardware management
   - Server manufacturer profiles (Lenovo, Dell, HP, Supermicro)

4. **Health Alerts**
   - Critical temperature warnings
   - Fan failure notifications
   - Hardware status indicators

### API Endpoints

- `GET /api/v1/system/gpu/temperature` - Get GPU temperatures
- `GET /api/v1/system/gpu/status` - Get GPU status
- `GET /api/v1/system/fans` - Get fan status
- `POST /api/v1/system/fans/mode` - Set fan mode
- `POST /api/v1/system/fans/speed` - Set fan speed

---

## Content Moderation Tab

🛡️ **Purpose**: Review and moderate AI-generated content before publication.

### Features

1. **Moderation Queue**
   - View content pending review
   - See content previews
   - Check AI analysis results
   - Priority sorting

2. **Content Analysis**
   - Automated content scoring
   - Category classification
   - Risk assessment
   - Suggested actions

3. **Review Actions**
   - Approve content
   - Reject with reason
   - Request modifications
   - Set review priority

4. **Moderation Statistics**
   - Approval rates
   - Common rejection reasons
   - Review times
   - Moderator performance

### API Endpoints

- `GET /api/v1/moderation/queue` - Get moderation queue
- `POST /api/v1/moderation/queue/{id}/approve` - Approve content
- `POST /api/v1/moderation/queue/{id}/reject` - Reject content
- `GET /api/v1/moderation/stats` - Get statistics

---

## Authentication & OAuth Tab

🔐 **Purpose**: Manage user authentication and social media platform connections.

### Features

1. **User Authentication**
   - User registration
   - Login management
   - JWT token handling
   - Password management

2. **OAuth Platform Connections**
   - Connect Instagram accounts
   - Connect TikTok accounts
   - Connect Twitter accounts
   - Token status monitoring

3. **Security Settings**
   - Session management
   - API key management
   - Access control

### API Endpoints

- `POST /api/v1/auth/register` - Register user
- `POST /api/v1/auth/login` - Login
- `GET /api/v1/oauth/platforms` - List OAuth platforms
- `GET /api/v1/oauth/status` - Check OAuth status

---

## Cache Management Tab

💾 **Purpose**: Monitor and manage the Redis cache for optimal performance.

### Features

1. **Cache Status**
   - Connection status
   - Memory usage
   - Key counts
   - Hit/miss rates

2. **Cache Operations**
   - View cached keys by prefix
   - Invalidate specific cache entries
   - Clear cache patterns
   - Warm up cache

3. **Health Monitoring**
   - Cache responsiveness
   - Latency metrics
   - Error tracking

### Cache Prefixes

| Prefix | Description |
|--------|-------------|
| `persona` | Persona data cache |
| `user` | User session data |
| `generation` | Content generation cache |
| `acd` | ACD context cache |
| `api` | API response cache |

### API Endpoints

- `GET /api/v1/cache/status` - Get cache status
- `GET /api/v1/cache/health` - Health check
- `DELETE /api/v1/cache/invalidate/{prefix}` - Invalidate by prefix

---

## Groups & Reels Tab

👥 **Purpose**: Manage persona groups and generate collaborative content like reels and duets.

### Features

1. **Friend Groups**
   - Create persona groups
   - Add/remove group members
   - Set group interaction rules
   - Define collaboration themes

2. **Auto-Interaction**
   - Enable automatic interactions between personas
   - Set interaction frequency
   - Define interaction types
   - Monitor interaction quality

3. **Reel Generation**
   - Generate single persona reels
   - Create duet content
   - Set video parameters
   - Schedule reel publication

### API Endpoints

- `GET /api/v1/friend-groups/` - List groups
- `POST /api/v1/friend-groups/` - Create group
- `POST /api/v1/friend-groups/{id}/members/{persona_id}` - Add member

---

## ACD (AI Development) Tab

🧬 **Purpose**: Monitor and manage the Autonomous Continuous Development system.

### Features

1. **Context Management**
   - View active ACD contexts
   - Monitor context states
   - Track context transitions
   - View context history

2. **Human-in-the-Loop (HIL) Ratings**
   - Rate generated content
   - Provide feedback
   - View rating statistics
   - Track rating trends

3. **Correlation Engine**
   - View pattern correlations
   - Track learning progress
   - Analyze decision trees
   - Monitor optimization

4. **Self-Improvement Metrics**
   - Quality score trends
   - Engagement improvements
   - Error rate tracking
   - Model performance

5. **ACD Statistics**
   - Total contexts processed
   - Success rates by phase
   - Average processing times
   - System health indicators

### ACD States

| State | Description |
|-------|-------------|
| `READY` | Context is ready for processing |
| `PROCESSING` | Currently being processed |
| `DONE` | Successfully completed |
| `FAILED` | Processing failed |
| `PAUSED` | Temporarily paused |
| `BLOCKED` | Waiting for external input |

### API Endpoints

- `GET /api/v1/acd/contexts` - List contexts
- `GET /api/v1/acd/stats/` - Get statistics
- `POST /api/v1/acd/rate/{context_id}` - Submit rating
- `GET /api/v1/acd/ratings` - Get ratings

---

## Agents Tab

🤖 **Purpose**: Manage multi-agent system for distributed AI operations.

> **Note**: This feature may require additional configuration.

### Features

1. **Agent Management**
   - Create new agents
   - Configure agent capabilities
   - Set agent parameters
   - Monitor agent status

2. **Task Assignment**
   - Assign tasks to agents
   - Monitor task progress
   - Track task completion
   - Handle task failures

3. **Workload Balancing**
   - View agent workloads
   - Automatic load distribution
   - Manual reassignment
   - Performance metrics

4. **Agent Marketplace**
   - Browse available agents
   - Install marketplace agents
   - Rate and review agents
   - Publish custom agents

### API Endpoints

- `GET /api/v1/multi-agent/agents/` - List agents
- `POST /api/v1/multi-agent/agents/` - Create agent
- `GET /api/v1/multi-agent/workload/` - Get workload
- `GET /api/v1/multi-agent/marketplace/search` - Search marketplace

---

## ML Learning Tab

🧠 **Purpose**: Manage machine learning model training and optimization.

> **Note**: This feature may require additional configuration.

### Features

1. **Model Training**
   - Train engagement prediction models
   - Train success classifiers
   - Configure training parameters
   - Monitor training progress

2. **A/B Testing**
   - Create A/B tests
   - Configure test variants
   - Monitor test results
   - Statistical analysis

3. **Feature Importance**
   - View feature rankings
   - Analyze feature contributions
   - Optimize feature selection
   - Track feature changes

4. **Cross-Persona Learning**
   - Aggregate learning across personas
   - Benchmark persona performance
   - Privacy-safe data sharing
   - Pattern transfer

### API Endpoints

- `POST /api/v1/ml-learning/models/train-engagement` - Train engagement model
- `GET /api/v1/ml-learning/models/feature-importance` - Get feature importance
- `POST /api/v1/ml-learning/ab-tests/create` - Create A/B test
- `GET /api/v1/ml-learning/cross-persona/aggregate` - Get cross-persona data

---

## Best Practices

### Performance Optimization

1. **Caching**: Use the cache tab to monitor and optimize caching
2. **Scheduling**: Schedule posts during off-peak hours for better performance
3. **Monitoring**: Set up temperature alerts to prevent hardware issues

### Content Quality

1. **Moderation**: Always review AI-generated content before publication
2. **HIL Ratings**: Provide consistent ratings to improve AI quality
3. **A/B Testing**: Use ML learning features to optimize content performance

### Security

1. **Authentication**: Regularly rotate API keys and tokens
2. **OAuth**: Review connected accounts periodically
3. **Access Control**: Limit admin access to authorized users

---

## Troubleshooting

### Common Issues

**Problem**: Scheduled posts not publishing
- Check OAuth token validity
- Verify platform connection status
- Review moderation queue for blocked content

**Problem**: High GPU temperatures
- Check fan control settings
- Verify IPMI credentials
- Review cooling system health

**Problem**: Cache connection failed
- Verify Redis is running
- Check connection settings
- Review firewall rules

**Problem**: Content failing moderation
- Review rejection reasons
- Adjust generation parameters
- Check persona guidelines

---

## Getting Help

- **API Documentation**: `/docs` (Swagger UI)
- **System Health**: `/health` endpoint
- **Diagnostics**: Use the Diagnostics API for detailed system information
- **Support**: Check the repository issues for known problems

---

**Remember**: *"Gator don't play no shit"* - Use these features wisely to build your AI influencer empire!
