# WebSocket Real-Time Chat Implementation

## Overview

The Gator platform now supports real-time communication via WebSocket connections. This enables:
- Instant direct messaging between users and AI personas
- Real-time typing indicators
- Online presence tracking
- Live notifications
- Simultaneous multi-device support

## Architecture

```
┌─────────────┐         WebSocket          ┌──────────────────┐
│   Client    │◀──────────────────────────▶│  FastAPI Server  │
│  (Browser)  │         ws://...           │  (WebSocket)     │
└─────────────┘                            └──────────────────┘
                                                    │
                                                    │
                                           ┌────────┴─────────┐
                                           │                  │
                                    ┌──────▼──────┐    ┌─────▼──────┐
                                    │  Connection │    │  Database  │
                                    │   Manager   │    │  (SQLite)  │
                                    └─────────────┘    └────────────┘
```

## Key Features

### 1. Connection Management
- Multiple connections per user (multi-device support)
- Automatic reconnection handling
- Connection pooling and cleanup
- User presence tracking

### 2. Real-Time Messaging
- Instant message delivery
- Message persistence to database
- AI response generation
- Read receipts (future)
- Message reactions (future)

### 3. Typing Indicators
- Show when other users are typing
- Automatic timeout after inactivity
- Per-conversation indicators

### 4. Presence System
- Online/offline status
- Last seen timestamps (future)
- Activity status (active, away, do not disturb) (future)

## WebSocket Endpoint

### Connection URL
```
ws://localhost:8000/ws/{user_id}
```

### Authentication
Currently uses user_id in URL. In production, use JWT token authentication:
```
ws://localhost:8000/ws/auth?token={jwt_token}
```

## Message Protocol

All messages are JSON formatted with a `type` field indicating the message type.

### Client → Server Messages

#### 1. Join Conversation
```json
{
  "type": "join_conversation",
  "conversation_id": "conversation-uuid"
}
```

**Response:**
```json
{
  "type": "conversation_joined",
  "conversation_id": "conversation-uuid",
  "timestamp": "2025-01-15T10:30:00Z"
}
```

#### 2. Leave Conversation
```json
{
  "type": "leave_conversation",
  "conversation_id": "conversation-uuid"
}
```

#### 3. Send Message
```json
{
  "type": "send_message",
  "conversation_id": "conversation-uuid",
  "content": "Hello! This is my message.",
  "persona_id": "persona-uuid"  // Optional, for AI responses
}
```

#### 4. Typing Indicator
```json
{
  "type": "typing_start",
  "conversation_id": "conversation-uuid"
}
```

```json
{
  "type": "typing_stop",
  "conversation_id": "conversation-uuid"
}
```

#### 5. Get Online Status
```json
{
  "type": "get_online_status",
  "user_ids": ["user1-uuid", "user2-uuid"]
}
```

### Server → Client Messages

#### 1. New Message
```json
{
  "type": "new_message",
  "conversation_id": "conversation-uuid",
  "message": {
    "id": "message-uuid",
    "sender_id": "user-uuid",
    "content": "Message text",
    "timestamp": "2025-01-15T10:30:00Z",
    "is_ai": false
  },
  "timestamp": "2025-01-15T10:30:00Z"
}
```

#### 2. Typing Indicator
```json
{
  "type": "typing_indicator",
  "conversation_id": "conversation-uuid",
  "user_id": "user-uuid",
  "is_typing": true,
  "timestamp": "2025-01-15T10:30:00Z"
}
```

#### 3. Presence Update
```json
{
  "type": "presence_update",
  "user_id": "user-uuid",
  "online": true,
  "timestamp": "2025-01-15T10:30:00Z"
}
```

#### 4. Online Users List
```json
{
  "type": "online_users",
  "users": ["user1-uuid", "user2-uuid", "user3-uuid"],
  "timestamp": "2025-01-15T10:30:00Z"
}
```

#### 5. Error Message
```json
{
  "type": "error",
  "message": "Error description",
  "timestamp": "2025-01-15T10:30:00Z"
}
```

## Client Implementation Examples

### JavaScript/Browser

```javascript
// Connect to WebSocket
const userId = 'user-uuid-here';
const ws = new WebSocket(`ws://localhost:8000/ws/${userId}`);

// Connection opened
ws.onopen = function(event) {
  console.log('Connected to WebSocket');
  
  // Join a conversation
  ws.send(JSON.stringify({
    type: 'join_conversation',
    conversation_id: 'conversation-uuid'
  }));
};

// Receive messages
ws.onmessage = function(event) {
  const data = JSON.parse(event.data);
  
  switch(data.type) {
    case 'new_message':
      displayMessage(data.message);
      break;
    
    case 'typing_indicator':
      showTypingIndicator(data.user_id, data.is_typing);
      break;
    
    case 'presence_update':
      updateUserPresence(data.user_id, data.online);
      break;
    
    case 'online_users':
      updateOnlineUsersList(data.users);
      break;
    
    case 'error':
      console.error('WebSocket error:', data.message);
      break;
  }
};

// Send a message
function sendMessage(conversationId, content, personaId = null) {
  ws.send(JSON.stringify({
    type: 'send_message',
    conversation_id: conversationId,
    content: content,
    persona_id: personaId
  }));
}

// Typing indicators
let typingTimeout;
function handleTyping(conversationId) {
  // Send typing start
  ws.send(JSON.stringify({
    type: 'typing_start',
    conversation_id: conversationId
  }));
  
  // Clear previous timeout
  clearTimeout(typingTimeout);
  
  // Auto-stop after 3 seconds of inactivity
  typingTimeout = setTimeout(() => {
    ws.send(JSON.stringify({
      type: 'typing_stop',
      conversation_id: conversationId
    }));
  }, 3000);
}

// Connection closed
ws.onclose = function(event) {
  console.log('WebSocket connection closed');
  // Implement reconnection logic here
  setTimeout(() => {
    // Reconnect
  }, 3000);
};

// Error handling
ws.onerror = function(error) {
  console.error('WebSocket error:', error);
};
```

### Python Client

```python
import asyncio
import websockets
import json

async def chat_client(user_id: str, conversation_id: str):
    uri = f"ws://localhost:8000/ws/{user_id}"
    
    async with websockets.connect(uri) as websocket:
        # Join conversation
        await websocket.send(json.dumps({
            'type': 'join_conversation',
            'conversation_id': conversation_id
        }))
        
        # Start listening for messages
        async def receive_messages():
            async for message in websocket:
                data = json.loads(message)
                print(f"Received: {data}")
        
        # Send messages
        async def send_messages():
            while True:
                content = input("Enter message: ")
                await websocket.send(json.dumps({
                    'type': 'send_message',
                    'conversation_id': conversation_id,
                    'content': content
                }))
        
        # Run both tasks
        await asyncio.gather(
            receive_messages(),
            send_messages()
        )

# Run the client
asyncio.run(chat_client('user-uuid', 'conversation-uuid'))
```

### React Component Example

```jsx
import React, { useState, useEffect, useRef } from 'react';

function ChatComponent({ userId, conversationId }) {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [isTyping, setIsTyping] = useState(false);
  const [onlineUsers, setOnlineUsers] = useState([]);
  const ws = useRef(null);

  useEffect(() => {
    // Connect to WebSocket
    ws.current = new WebSocket(`ws://localhost:8000/ws/${userId}`);

    ws.current.onopen = () => {
      // Join conversation
      ws.current.send(JSON.stringify({
        type: 'join_conversation',
        conversation_id: conversationId
      }));
    };

    ws.current.onmessage = (event) => {
      const data = JSON.parse(event.data);

      switch(data.type) {
        case 'new_message':
          setMessages(prev => [...prev, data.message]);
          break;
        
        case 'typing_indicator':
          setIsTyping(data.is_typing);
          break;
        
        case 'online_users':
          setOnlineUsers(data.users);
          break;
      }
    };

    // Cleanup on unmount
    return () => {
      if (ws.current) {
        ws.current.close();
      }
    };
  }, [userId, conversationId]);

  const sendMessage = () => {
    if (input.trim() && ws.current) {
      ws.current.send(JSON.stringify({
        type: 'send_message',
        conversation_id: conversationId,
        content: input
      }));
      setInput('');
    }
  };

  const handleTyping = () => {
    if (ws.current) {
      ws.current.send(JSON.stringify({
        type: 'typing_start',
        conversation_id: conversationId
      }));
      
      // Auto-stop after 3 seconds
      setTimeout(() => {
        ws.current.send(JSON.stringify({
          type: 'typing_stop',
          conversation_id: conversationId
        }));
      }, 3000);
    }
  };

  return (
    <div className="chat-container">
      <div className="online-users">
        Online: {onlineUsers.length}
      </div>
      
      <div className="messages">
        {messages.map((msg, idx) => (
          <div key={idx} className={msg.is_ai ? 'ai-message' : 'user-message'}>
            <span className="content">{msg.content}</span>
            <span className="timestamp">{msg.timestamp}</span>
          </div>
        ))}
        {isTyping && <div className="typing-indicator">Someone is typing...</div>}
      </div>
      
      <div className="input-area">
        <input
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyUp={handleTyping}
          onKeyPress={(e) => e.key === 'Enter' && sendMessage()}
          placeholder="Type a message..."
        />
        <button onClick={sendMessage}>Send</button>
      </div>
    </div>
  );
}

export default ChatComponent;
```

## Testing

### Manual Testing with wscat

Install wscat:
```bash
npm install -g wscat
```

Connect and test:
```bash
# Connect
wscat -c ws://localhost:8000/ws/test-user-id

# Join conversation
> {"type": "join_conversation", "conversation_id": "test-conv-id"}

# Send message
> {"type": "send_message", "conversation_id": "test-conv-id", "content": "Hello!"}

# Start typing
> {"type": "typing_start", "conversation_id": "test-conv-id"}

# Stop typing
> {"type": "typing_stop", "conversation_id": "test-conv-id"}
```

### Automated Testing with pytest

```python
import pytest
from fastapi.testclient import TestClient
from backend.api.main import app

@pytest.mark.asyncio
async def test_websocket_connection():
    """Test basic WebSocket connection."""
    client = TestClient(app)
    
    with client.websocket_connect("/ws/test-user") as websocket:
        # Should receive online users list on connection
        data = websocket.receive_json()
        assert data['type'] == 'online_users'
        assert isinstance(data['users'], list)

@pytest.mark.asyncio
async def test_join_conversation():
    """Test joining a conversation."""
    client = TestClient(app)
    
    with client.websocket_connect("/ws/test-user") as websocket:
        # Skip initial online users message
        websocket.receive_json()
        
        # Join conversation
        websocket.send_json({
            'type': 'join_conversation',
            'conversation_id': 'test-conv'
        })
        
        # Should receive confirmation
        data = websocket.receive_json()
        assert data['type'] == 'conversation_joined'
        assert data['conversation_id'] == 'test-conv'
```

## Production Deployment

### NGINX Configuration

For WebSocket support with NGINX:

```nginx
upstream gator_websocket {
    server 127.0.0.1:8000;
}

server {
    listen 443 ssl;
    server_name your-domain.com;

    # WebSocket location
    location /ws {
        proxy_pass http://gator_websocket;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # Timeout settings
        proxy_connect_timeout 7d;
        proxy_send_timeout 7d;
        proxy_read_timeout 7d;
    }
}
```

### Docker Compose

```yaml
services:
  gator-api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - REDIS_URL=redis://redis:6379/0
    depends_on:
      - redis
  
  redis:
    image: redis:latest
    ports:
      - "6379:6379"
```

### Kubernetes Deployment

```yaml
apiVersion: v1
kind: Service
metadata:
  name: gator-websocket
spec:
  selector:
    app: gator
  ports:
  - port: 8000
    targetPort: 8000
  sessionAffinity: ClientIP  # Important for WebSocket
```

## Scaling Considerations

### Horizontal Scaling

When running multiple instances, use Redis for pub/sub:

```python
# backend/api/websocket.py
import redis.asyncio as redis

redis_client = redis.from_url(settings.REDIS_URL)

async def broadcast_via_redis(channel: str, message: dict):
    """Broadcast message to all instances via Redis."""
    await redis_client.publish(channel, json.dumps(message))

# Subscribe to Redis channels for cross-instance messages
```

### Load Balancing

- Use sticky sessions (session affinity) at load balancer
- Or implement Redis-based connection tracking
- Monitor connection count per instance

## Performance Optimization

### Connection Limits

```python
# Limit connections per user
MAX_CONNECTIONS_PER_USER = 5

async def connect(self, websocket: WebSocket, user_id: str):
    if len(self.active_connections.get(user_id, [])) >= MAX_CONNECTIONS_PER_USER:
        await websocket.close(code=1008, reason="Too many connections")
        return
    # ... continue connection
```

### Message Rate Limiting

```python
from datetime import datetime, timedelta

class RateLimiter:
    def __init__(self, max_messages: int, time_window: int):
        self.max_messages = max_messages
        self.time_window = timedelta(seconds=time_window)
        self.user_messages = {}
    
    def check_rate_limit(self, user_id: str) -> bool:
        now = datetime.utcnow()
        if user_id not in self.user_messages:
            self.user_messages[user_id] = []
        
        # Remove old messages
        self.user_messages[user_id] = [
            msg_time for msg_time in self.user_messages[user_id]
            if now - msg_time < self.time_window
        ]
        
        if len(self.user_messages[user_id]) >= self.max_messages:
            return False
        
        self.user_messages[user_id].append(now)
        return True
```

## Security

### Authentication

Implement JWT token authentication:

```python
from backend.services.user_service import verify_jwt_token

async def websocket_endpoint(
    websocket: WebSocket,
    token: str,
    session: AsyncSession = Depends(get_db_session)
):
    # Verify token
    user = await verify_jwt_token(token)
    if not user:
        await websocket.close(code=1008, reason="Invalid token")
        return
    
    await manager.connect(websocket, str(user.id))
    # ... rest of logic
```

### Message Validation

Always validate and sanitize user input:

```python
from pydantic import BaseModel, validator

class SendMessageRequest(BaseModel):
    type: str
    conversation_id: str
    content: str
    
    @validator('content')
    def validate_content(cls, v):
        if len(v) > 5000:
            raise ValueError('Message too long')
        if not v.strip():
            raise ValueError('Message cannot be empty')
        return v
```

## Monitoring

### Metrics to Track

- Active connections count
- Messages per second
- Connection duration
- Error rate
- Latency (message delivery time)

### Logging

```python
logger.info(f"WebSocket metrics: "
           f"active_connections={len(manager.active_connections)} "
           f"active_conversations={len(manager.conversation_connections)}")
```

## Troubleshooting

### Connection Drops

- Check firewall rules
- Verify WebSocket support in reverse proxy
- Review timeout settings
- Check network stability

### High Memory Usage

- Implement connection limits
- Clean up stale connections
- Monitor message queue sizes
- Use message expiration

### Performance Issues

- Enable Redis pub/sub for multi-instance
- Optimize database queries
- Use connection pooling
- Implement message batching

## Future Enhancements

- [ ] Read receipts
- [ ] Message reactions (emoji)
- [ ] File uploads via WebSocket
- [ ] Voice/video call signaling
- [ ] Screen sharing support
- [ ] End-to-end encryption
- [ ] Message search
- [ ] Conversation archiving

## Support

For WebSocket-related issues:
1. Check browser console for errors
2. Verify WebSocket connection in Network tab
3. Review server logs
4. Test with wscat for isolation
5. Open GitHub issue with details

---

**Last Updated**: January 2025
**Version**: 1.0.0
**Status**: Production Ready
