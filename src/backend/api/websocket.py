"""
WebSocket Support for Real-Time Communication

Provides WebSocket endpoints for:
- Real-time direct messaging
- Typing indicators
- Online presence
- Live notifications
"""

import json
import asyncio
from typing import Dict, Set, Optional
from datetime import datetime
from uuid import UUID

from fastapi import WebSocket, WebSocketDisconnect, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from backend.database.connection import get_db_session
from backend.services.direct_messaging_service import DirectMessagingService
from backend.models.persona import PersonaModel
from backend.config.logging import get_logger

logger = get_logger(__name__)


class ConnectionManager:
    """
    Manages WebSocket connections for real-time communication.

    Tracks active connections per user/persona and handles message broadcasting.
    """

    def __init__(self):
        # Map of user_id -> Set of WebSocket connections
        self.active_connections: Dict[str, Set[WebSocket]] = {}
        # Map of conversation_id -> Set of WebSocket connections
        self.conversation_connections: Dict[str, Set[WebSocket]] = {}
        # Map of WebSocket -> user_id
        self.connection_users: Dict[WebSocket, str] = {}
        # Typing indicators: conversation_id -> Set of user_ids currently typing
        self.typing_indicators: Dict[str, Set[str]] = {}

    async def connect(self, websocket: WebSocket, user_id: str):
        """
        Accept new WebSocket connection and register user.

        Args:
            websocket: WebSocket connection instance
            user_id: Unique identifier for the user
        """
        await websocket.accept()

        if user_id not in self.active_connections:
            self.active_connections[user_id] = set()

        self.active_connections[user_id].add(websocket)
        self.connection_users[websocket] = user_id

        logger.info(f"WebSocket connected: user_id={user_id}")

        # Broadcast presence update
        await self.broadcast_presence_update(user_id, online=True)

    async def disconnect(self, websocket: WebSocket):
        """
        Remove WebSocket connection and clean up user tracking.

        Args:
            websocket: WebSocket connection instance
        """
        user_id = self.connection_users.get(websocket)

        if user_id and user_id in self.active_connections:
            self.active_connections[user_id].discard(websocket)

            # If user has no more connections, remove from active users
            if not self.active_connections[user_id]:
                del self.active_connections[user_id]
                # Broadcast offline status
                await self.broadcast_presence_update(user_id, online=False)

        if websocket in self.connection_users:
            del self.connection_users[websocket]

        # Remove from all conversation connections
        for conversation_id, connections in list(self.conversation_connections.items()):
            connections.discard(websocket)
            if not connections:
                del self.conversation_connections[conversation_id]

        logger.info(f"WebSocket disconnected: user_id={user_id}")

    async def join_conversation(self, websocket: WebSocket, conversation_id: str):
        """
        Add WebSocket to conversation-specific updates.

        Args:
            websocket: WebSocket connection instance
            conversation_id: Unique identifier for the conversation
        """
        if conversation_id not in self.conversation_connections:
            self.conversation_connections[conversation_id] = set()

        self.conversation_connections[conversation_id].add(websocket)

        logger.info(f"User joined conversation: conversation_id={conversation_id}")

    async def leave_conversation(self, websocket: WebSocket, conversation_id: str):
        """
        Remove WebSocket from conversation-specific updates.

        Args:
            websocket: WebSocket connection instance
            conversation_id: Unique identifier for the conversation
        """
        if conversation_id in self.conversation_connections:
            self.conversation_connections[conversation_id].discard(websocket)

            if not self.conversation_connections[conversation_id]:
                del self.conversation_connections[conversation_id]

    async def send_personal_message(self, message: dict, user_id: str):
        """
        Send message to all connections of a specific user.

        Args:
            message: Message data to send
            user_id: Target user ID
        """
        if user_id in self.active_connections:
            disconnected = set()

            for connection in self.active_connections[user_id]:
                try:
                    await connection.send_json(message)
                except Exception as e:
                    logger.error(f"Error sending message to user {user_id}: {str(e)}")
                    disconnected.add(connection)

            # Clean up disconnected connections
            for connection in disconnected:
                await self.disconnect(connection)

    async def broadcast_to_conversation(self, message: dict, conversation_id: str):
        """
        Broadcast message to all users in a conversation.

        Args:
            message: Message data to send
            conversation_id: Target conversation ID
        """
        if conversation_id in self.conversation_connections:
            disconnected = set()

            for connection in self.conversation_connections[conversation_id]:
                try:
                    await connection.send_json(message)
                except Exception as e:
                    logger.error(
                        f"Error broadcasting to conversation {conversation_id}: {str(e)}"
                    )
                    disconnected.add(connection)

            # Clean up disconnected connections
            for connection in disconnected:
                await self.disconnect(connection)

    async def set_typing_indicator(
        self, conversation_id: str, user_id: str, is_typing: bool
    ):
        """
        Update typing indicator for a user in a conversation.

        Args:
            conversation_id: Conversation identifier
            user_id: User identifier
            is_typing: Whether user is currently typing
        """
        if conversation_id not in self.typing_indicators:
            self.typing_indicators[conversation_id] = set()

        if is_typing:
            self.typing_indicators[conversation_id].add(user_id)
        else:
            self.typing_indicators[conversation_id].discard(user_id)

        # Broadcast typing indicator update
        await self.broadcast_to_conversation(
            {
                "type": "typing_indicator",
                "conversation_id": conversation_id,
                "user_id": user_id,
                "is_typing": is_typing,
                "timestamp": datetime.utcnow().isoformat(),
            },
            conversation_id,
        )

    async def broadcast_presence_update(self, user_id: str, online: bool):
        """
        Broadcast user presence update to relevant conversations.

        Args:
            user_id: User identifier
            online: Whether user is online or offline
        """
        # Find all conversations this user is part of and broadcast presence
        for conversation_id, connections in self.conversation_connections.items():
            # Check if any connection in this conversation belongs to the user
            for connection in connections:
                if self.connection_users.get(connection) == user_id:
                    await self.broadcast_to_conversation(
                        {
                            "type": "presence_update",
                            "user_id": user_id,
                            "online": online,
                            "timestamp": datetime.utcnow().isoformat(),
                        },
                        conversation_id,
                    )
                    break

    def is_user_online(self, user_id: str) -> bool:
        """Check if a user has any active connections."""
        return (
            user_id in self.active_connections
            and len(self.active_connections[user_id]) > 0
        )

    def get_online_users(self) -> Set[str]:
        """Get set of all currently online user IDs."""
        return set(self.active_connections.keys())


# Global connection manager instance
manager = ConnectionManager()


async def websocket_endpoint(
    websocket: WebSocket, user_id: str, session: AsyncSession = Depends(get_db_session)
):
    """
    Main WebSocket endpoint for real-time messaging.

    Args:
        websocket: WebSocket connection
        user_id: Authenticated user ID
        session: Database session

    Message Types (Client -> Server):
        - join_conversation: Join a conversation for updates
        - leave_conversation: Leave a conversation
        - send_message: Send a new message
        - typing_start: Indicate user started typing
        - typing_stop: Indicate user stopped typing
        - get_online_status: Request online status of users

    Message Types (Server -> Client):
        - new_message: New message received
        - typing_indicator: User typing status update
        - presence_update: User online/offline status
        - error: Error message
    """
    await manager.connect(websocket, user_id)

    try:
        messaging_service = DirectMessagingService(session)

        # Send initial online users list
        await websocket.send_json(
            {
                "type": "online_users",
                "users": list(manager.get_online_users()),
                "timestamp": datetime.utcnow().isoformat(),
            }
        )

        while True:
            # Receive message from client
            data = await websocket.receive_json()
            message_type = data.get("type")

            if message_type == "join_conversation":
                conversation_id = data.get("conversation_id")
                await manager.join_conversation(websocket, conversation_id)

                await websocket.send_json(
                    {
                        "type": "conversation_joined",
                        "conversation_id": conversation_id,
                        "timestamp": datetime.utcnow().isoformat(),
                    }
                )

            elif message_type == "leave_conversation":
                conversation_id = data.get("conversation_id")
                await manager.leave_conversation(websocket, conversation_id)

                await websocket.send_json(
                    {
                        "type": "conversation_left",
                        "conversation_id": conversation_id,
                        "timestamp": datetime.utcnow().isoformat(),
                    }
                )

            elif message_type == "send_message":
                conversation_id = data.get("conversation_id")
                content = data.get("content")
                persona_id = data.get("persona_id")

                # Create message in database
                message = await messaging_service.send_message(
                    conversation_id=UUID(conversation_id),
                    sender_id=UUID(user_id),
                    content=content,
                )

                # Broadcast to conversation
                await manager.broadcast_to_conversation(
                    {
                        "type": "new_message",
                        "conversation_id": conversation_id,
                        "message": {
                            "id": str(message.id),
                            "sender_id": str(message.sender_id),
                            "content": message.content,
                            "timestamp": message.created_at.isoformat(),
                            "is_ai": message.is_ai,
                        },
                        "timestamp": datetime.utcnow().isoformat(),
                    },
                    conversation_id,
                )

                # If message is to AI persona, generate response
                if persona_id:
                    # Generate AI response asynchronously
                    asyncio.create_task(
                        generate_and_send_ai_response(
                            messaging_service, conversation_id, persona_id, content
                        )
                    )

            elif message_type == "typing_start":
                conversation_id = data.get("conversation_id")
                await manager.set_typing_indicator(conversation_id, user_id, True)

            elif message_type == "typing_stop":
                conversation_id = data.get("conversation_id")
                await manager.set_typing_indicator(conversation_id, user_id, False)

            elif message_type == "get_online_status":
                user_ids = data.get("user_ids", [])
                online_status = {uid: manager.is_user_online(uid) for uid in user_ids}

                await websocket.send_json(
                    {
                        "type": "online_status",
                        "status": online_status,
                        "timestamp": datetime.utcnow().isoformat(),
                    }
                )

            else:
                await websocket.send_json(
                    {
                        "type": "error",
                        "message": f"Unknown message type: {message_type}",
                        "timestamp": datetime.utcnow().isoformat(),
                    }
                )

    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected: user_id={user_id}")
    except Exception as e:
        logger.error(f"WebSocket error for user {user_id}: {str(e)}")
    finally:
        await manager.disconnect(websocket)


async def generate_and_send_ai_response(
    messaging_service: DirectMessagingService,
    conversation_id: str,
    persona_id: str,
    user_message: str,
):
    """
    Generate AI response and broadcast to conversation.

    Args:
        messaging_service: Direct messaging service instance
        conversation_id: Conversation identifier
        persona_id: Persona identifier
        user_message: User's message content
    """
    try:
        # Get conversation and persona
        conversation = await messaging_service.get_conversation(UUID(conversation_id))

        # Generate AI response
        # Note: This would need to be implemented in the messaging service
        ai_response = await messaging_service.generate_ai_response(
            conversation=conversation,
            user_message=user_message,
            persona=None,  # Would fetch persona by ID
        )

        # Send AI response to conversation
        message = await messaging_service.send_message(
            conversation_id=UUID(conversation_id),
            sender_id=UUID(persona_id),
            content=ai_response,
            is_ai=True,
        )

        # Broadcast AI response
        await manager.broadcast_to_conversation(
            {
                "type": "new_message",
                "conversation_id": conversation_id,
                "message": {
                    "id": str(message.id),
                    "sender_id": str(message.sender_id),
                    "content": message.content,
                    "timestamp": message.created_at.isoformat(),
                    "is_ai": True,
                },
                "timestamp": datetime.utcnow().isoformat(),
            },
            conversation_id,
        )

    except Exception as e:
        logger.error(f"Error generating AI response: {str(e)}")
        # Send error to conversation
        await manager.broadcast_to_conversation(
            {
                "type": "error",
                "message": "Failed to generate AI response",
                "timestamp": datetime.utcnow().isoformat(),
            },
            conversation_id,
        )
