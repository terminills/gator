"""
FastAPI Application Configuration

Main entry point for the Gator AI Influencer Platform backend API.
Configured following best practices from BEST_PRACTICES.md.
"""

from contextlib import asynccontextmanager
from typing import AsyncGenerator

import uvicorn
from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse

from backend.config.settings import get_settings
from backend.config.logging import setup_logging
from backend.api.routes import persona, content, analytics
from backend.database.connection import database_manager

# Configure logging
setup_logging()

# Get application settings
settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Application lifespan management.
    
    Handles startup and shutdown tasks including database connections,
    AI model loading, and resource cleanup.
    """
    # Startup
    await database_manager.connect()
    
    # TODO: Initialize AI models and services
    # await ai_model_manager.load_models()
    # await content_generation_service.initialize()
    
    yield
    
    # Shutdown
    await database_manager.disconnect()


def create_app() -> FastAPI:
    """
    Create and configure FastAPI application instance.
    
    Returns:
        FastAPI: Configured application instance
    """
    app = FastAPI(
        title="Gator AI Influencer Platform",
        description="Private hosting solution for AI-driven content generation",
        version="0.1.0",
        docs_url="/docs" if settings.debug else None,
        redoc_url="/redoc" if settings.debug else None,
        openapi_url="/openapi.json" if settings.debug else None,
        lifespan=lifespan,
    )
    
    # Security middleware
    app.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=settings.allowed_hosts,
    )
    
    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.allowed_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        allow_headers=["*"],
    )
    
    # Include API routers
    app.include_router(persona.router)
    app.include_router(content.router)
    app.include_router(analytics.router)
    
    @app.get("/", tags=["system"])
    async def root():
        """Root endpoint - system status."""
        return {
            "message": "Gator AI Influencer Platform",
            "version": "0.1.0",
            "status": "operational"
        }
    
    @app.get("/health", tags=["system"])
    async def health_check():
        """Health check endpoint for monitoring."""
        return {
            "status": "healthy",
            "database": await database_manager.health_check(),
            "timestamp": "2024-01-01T00:00:00Z"  # TODO: Use actual timestamp
        }
    
    @app.exception_handler(404)
    async def not_found_handler(request: Request, exc) -> Response:
        """Custom 404 handler."""
        return JSONResponse(
            status_code=404,
            content={"detail": f"Path {request.url.path} not found"}
        )
    
    return app


# Create application instance
app = create_app()


if __name__ == "__main__":
    # Development server
    uvicorn.run(
        "backend.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_config=None,  # Use our custom logging
    )