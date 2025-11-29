"""
Gator Agent API Routes

API endpoints for the Gator help agent functionality.
"""

from typing import Dict, List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from backend.services.gator_agent_service import gator_agent

router = APIRouter(tags=["gator-agent"])


class ChatMessage(BaseModel):
    """Chat message model for API requests."""

    message: str = Field(..., min_length=1, max_length=1000, description="User message")
    context: Optional[Dict] = Field(
        None, description="Optional context about current page/state"
    )
    verbose: bool = Field(
        False,
        description="Enable verbose command-line style output with execution details",
    )


class ChatResponse(BaseModel):
    """Chat response model for API responses."""

    response: str = Field(..., description="Gator's response")
    timestamp: str = Field(..., description="Response timestamp")


class QuickHelpTopic(BaseModel):
    """Quick help topic model."""

    topic: str = Field(..., description="Help topic title")
    message: str = Field(..., description="Pre-filled message for this topic")


@router.post("/chat", response_model=ChatResponse)
async def chat_with_gator(message_data: ChatMessage):
    """
    Send a message to Gator and get a response.

    Gator is the tough, no-nonsense help agent who will guide you through
    the platform with his characteristic attitude and expertise.

    Set verbose=True for command-line style output with detailed execution logs.
    """
    try:
        from datetime import datetime

        response = await gator_agent.process_message(
            message=message_data.message,
            context=message_data.context,
            verbose=message_data.verbose,
        )

        return ChatResponse(response=response, timestamp=datetime.now().isoformat())

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error processing message: {str(e)}"
        )


@router.get("/quick-help", response_model=List[QuickHelpTopic])
async def get_quick_help_topics():
    """
    Get quick help topics that users can click for common questions.
    """
    try:
        topics = gator_agent.get_quick_help_topics()
        return [QuickHelpTopic(**topic) for topic in topics]

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error getting help topics: {str(e)}"
        )


@router.get("/conversation-history")
async def get_conversation_history():
    """
    Get the conversation history with Gator.
    """
    try:
        history = gator_agent.get_conversation_history()
        return {"history": history}

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error getting conversation history: {str(e)}"
        )


@router.delete("/conversation-history")
async def clear_conversation_history():
    """
    Clear the conversation history with Gator.
    """
    try:
        gator_agent.clear_conversation_history()
        return {"message": "Conversation history cleared"}

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error clearing conversation history: {str(e)}"
        )


@router.get("/status")
async def get_agent_status():
    """
    Get the status of the Gator agent.
    """
    try:
        history = gator_agent.get_conversation_history()

        return {
            "status": "operational",
            "agent": "Gator from The Other Guys",
            "attitude": "No-nonsense, direct, helpful but tough",
            "conversation_count": len(history),
            "last_interaction": history[-1]["timestamp"] if history else None,
            "available_topics": [
                "Personas",
                "Content Generation",
                "DNS Management",
                "System Status",
                "GoDaddy Integration",
                "Troubleshooting",
            ],
        }

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error getting agent status: {str(e)}"
        )
