"""
API Routes Package

Exports all API route modules for the Gator AI Influencer Platform.
"""

from . import (
    public,
    gator_agent,
    interactive,
    segments,
    friend_groups,
)

__all__ = [
    "public",
    "gator_agent",
    "interactive",
    "segments",
    "friend_groups",
]
