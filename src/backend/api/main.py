"""
FastAPI Application Configuration

Main entry point for the Gator AI Influencer Platform backend API.
Configured following best practices from BEST_PRACTICES.md.
"""

import os
from contextlib import asynccontextmanager
from typing import AsyncGenerator

import uvicorn
from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles

from backend.config.settings import get_settings
from backend.config.logging import setup_logging
from backend.api.routes import public, dns

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
    # Startup - for now just log startup
    print("Starting up Gator AI Platform...")
    
    yield
    
    # Shutdown
    print("Shutting down Gator AI Platform...")


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
    
    # Mount static files (frontend directory is at repo root level)
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
    frontend_path = os.path.join(project_root, "frontend", "public")
    if os.path.exists(frontend_path):
        app.mount("/static", StaticFiles(directory=frontend_path), name="static")
    
    # Include API routers
    app.include_router(public.router)
    app.include_router(dns.router, prefix="/api/v1")
    
    @app.get("/", tags=["system"])
    async def root():
        """Root endpoint - serve admin dashboard."""
        index_path = os.path.join(frontend_path, "index.html")
        if os.path.exists(index_path):
            return FileResponse(index_path)
        return {
            "message": "Gator AI Influencer Platform",
            "version": "0.1.0", 
            "status": "operational"
        }
    
    @app.get("/gallery", tags=["public"])
    async def public_gallery():
        """Serve public gallery page."""
        gallery_path = os.path.join(frontend_path, "gallery.html")
        if os.path.exists(gallery_path):
            return FileResponse(gallery_path)
        return {"error": "Gallery page not found"}
    
    @app.get("/gallery/persona/{persona_id}", tags=["public"])
    async def persona_detail(persona_id: str):
        """Serve persona detail page."""
        persona_path = os.path.join(frontend_path, "persona.html") 
        if os.path.exists(persona_path):
            return FileResponse(persona_path)
        return {"error": "Persona page not found"}
    
    @app.get("/health", tags=["system"])
    async def health_check():
        """Health check endpoint for monitoring."""
        return {
            "status": "healthy",
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