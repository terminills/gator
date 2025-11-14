"""
FastAPI Application Configuration

Main entry point for the Gator AI Influencer Platform backend API.
Configured following best practices from BEST_PRACTICES.md.
"""

import os
from contextlib import asynccontextmanager
from typing import AsyncGenerator

import uvicorn
from fastapi import FastAPI, Request, Response, Depends, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from backend.database.connection import get_db_session

from backend.config.settings import get_settings
from backend.config.logging import setup_logging
from backend.api.routes import (
    public,
    dns,
    persona,
    users,
    direct_messaging,
    gator_agent,
    analytics,
    content,
    setup,
    creator,
    feeds,
    social,
    database_admin,
    sentiment,
    interactive,
    segments,
    plugins,
    friend_groups,
    enhanced_persona,
    branding,
    acd,
    diagnostics,
    settings as settings_routes,
    system_monitoring,
    reasoning_orchestrator,
)
from backend.api.websocket import websocket_endpoint

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
    print("Starting up Gator AI Platform...")

    # Initialize database connection
    from backend.database.connection import database_manager
    from backend.database.migrations import run_migrations

    await database_manager.connect()
    print("Database connection established.")

    # Run database migrations automatically (if AUTO_MIGRATE is enabled)
    migration_results = await run_migrations(database_manager.engine)
    if migration_results.get("columns_added"):
        print(
            f"  ✓ Applied database migrations: {', '.join(migration_results['columns_added'])}"
        )
    else:
        print("  ✓ Database schema is up to date")

    # Initialize AI models for content generation
    try:
        from backend.services.ai_models import ai_models

        print("\n" + "=" * 80)
        print("🤖 Initializing AI models...")
        print("=" * 80)
        await ai_models.initialize_models()

        # Log available models
        available_counts = {
            "text": len(
                [
                    m
                    for m in ai_models.available_models.get("text", [])
                    if m.get("loaded")
                ]
            ),
            "image": len(
                [
                    m
                    for m in ai_models.available_models.get("image", [])
                    if m.get("loaded")
                ]
            ),
            "voice": len(
                [
                    m
                    for m in ai_models.available_models.get("voice", [])
                    if m.get("loaded")
                ]
            ),
            "video": len(
                [
                    m
                    for m in ai_models.available_models.get("video", [])
                    if m.get("loaded")
                ]
            ),
        }

        print("=" * 80)
        print(f"✅ AI models initialized:")
        print(f"  - Text models loaded: {available_counts['text']}")
        print(f"  - Image models loaded: {available_counts['image']}")
        print(f"  - Voice models loaded: {available_counts['voice']}")
        print(f"  - Video models loaded: {available_counts['video']}")
        print("=" * 80)

        if sum(available_counts.values()) == 0:
            print("  ⚠️  No local models found. Using cloud APIs if configured.")

        print("\n")

    except Exception as e:
        print(f"⚠️  Warning: Failed to initialize AI models: {str(e)}")
        print("  Content generation may use fallback mechanisms.")

    yield

    # Shutdown
    print("Shutting down Gator AI Platform...")

    # Clean up AI models
    try:
        from backend.services.ai_models import ai_models

        await ai_models.close()
        print("AI models cleaned up.")
    except Exception as e:
        print(f"Warning: Error cleaning up AI models: {str(e)}")

    # Disconnect from database
    await database_manager.disconnect()
    print("Database connection closed.")


def create_app() -> FastAPI:
    """
    Create and configure FastAPI application instance.

    Returns:
        FastAPI: Configured application instance
    """
    app = FastAPI(
        title="Gator AI Influencer Platform",
        description="Gator don't play no shit - AI-powered content generation platform",
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
    project_root = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    )
    frontend_path = os.path.join(project_root, "frontend", "public")
    if os.path.exists(frontend_path):
        app.mount("/static", StaticFiles(directory=frontend_path), name="static")

    # Mount content directory for generated content (images, videos, etc.)
    content_path = os.path.join(project_root, "data", "content")
    if os.path.exists(content_path):
        app.mount("/content", StaticFiles(directory=content_path), name="content")
        print(f"Mounted content directory: {content_path}")
    else:
        print(f"Warning: Content directory not found: {content_path}")
    
    # Mount base_images directory for persona base images
    base_images_path = "/opt/gator/data/models/base_images"
    if os.path.exists(base_images_path):
        app.mount("/base_images", StaticFiles(directory=base_images_path), name="base_images")
        print(f"Mounted base_images directory: {base_images_path}")
    else:
        print(f"Warning: Base images directory not found: {base_images_path}")

    # Include API routers
    app.include_router(public.router, prefix="/api/v1")
    app.include_router(branding.router)
    app.include_router(dns.router, prefix="/api/v1")
    app.include_router(setup.router, prefix="/api/v1")
    app.include_router(database_admin.router, prefix="/api/v1")
    app.include_router(persona.router)
    app.include_router(users.router)
    app.include_router(direct_messaging.router)
    app.include_router(gator_agent.router, prefix="/api/v1/gator-agent")
    app.include_router(gator_agent.router, prefix="/gator-agent")  # Backward compatibility
    app.include_router(analytics.router)
    app.include_router(diagnostics.router)
    app.include_router(content.router)
    app.include_router(creator.router, prefix="/api/v1")
    app.include_router(feeds.router)
    app.include_router(social.router)
    app.include_router(sentiment.router, prefix="/api/v1")
    app.include_router(interactive.router)
    app.include_router(segments.router)
    app.include_router(plugins.router, prefix="/api/v1")
    app.include_router(friend_groups.router, prefix="/api/v1")
    app.include_router(enhanced_persona.router, prefix="/api/v1")
    app.include_router(acd.router)
    app.include_router(reasoning_orchestrator.router)
    app.include_router(settings_routes.router, prefix="/api/v1")
    app.include_router(system_monitoring.router)

    # WebSocket endpoint for real-time communication
    @app.websocket("/ws/{user_id}")
    async def websocket_route(websocket: WebSocket, user_id: str):
        """WebSocket endpoint for real-time messaging."""
        await websocket_endpoint(websocket, user_id)

    @app.get("/", tags=["public"])
    async def root(request: Request):
        """Root endpoint - serve public gallery (end user landing page) or API info."""
        # For API requests or test environments, return JSON
        accept_header = request.headers.get("accept", "")
        user_agent = request.headers.get("user-agent", "")

        # Return JSON for API clients (including test clients)
        if (
            "application/json" in accept_header
            or "testclient" in user_agent.lower()
            or "httpx" in user_agent.lower()
        ):
            return {
                "message": "Gator AI Influencer Platform",
                "version": "0.1.0",
                "status": "operational",
            }

        # For browser requests, serve the public gallery as the landing page
        gallery_path = os.path.join(frontend_path, "gallery.html")
        if os.path.exists(gallery_path):
            return FileResponse(gallery_path)
        return {
            "message": "Gator AI Influencer Platform",
            "version": "0.1.0",
            "status": "operational",
        }

    @app.get("/admin", tags=["system"])
    async def admin_dashboard():
        """Serve main admin dashboard hub."""
        # Serve the new modern admin dashboard
        dashboard_path = os.path.join(project_root, "admin_panel", "dashboard.html")
        if os.path.exists(dashboard_path):
            return FileResponse(dashboard_path)
        # Fallback to simple admin panel
        admin_panel_path = os.path.join(project_root, "admin_panel", "index.html")
        if os.path.exists(admin_panel_path):
            return FileResponse(admin_panel_path)
        # Last resort: legacy admin.html
        admin_path = os.path.join(project_root, "admin.html")
        if os.path.exists(admin_path):
            return FileResponse(admin_path)
        return {"error": "Admin dashboard not found"}

    @app.get("/admin/personas", tags=["system"])
    async def admin_personas(request: Request):
        """Serve persona management page or persona editor based on query params."""
        # Check if action parameter is present (create or edit)
        action = request.query_params.get("action")

        if action in ["create", "edit"]:
            # Serve the persona editor
            editor_path = os.path.join(
                project_root, "admin_panel", "persona-editor.html"
            )
            if os.path.exists(editor_path):
                return FileResponse(editor_path)

        # Default: serve the personas list page
        personas_path = os.path.join(project_root, "admin_panel", "personas.html")
        if os.path.exists(personas_path):
            return FileResponse(personas_path)
        # Fallback to main admin
        admin_panel_path = os.path.join(project_root, "admin_panel", "index.html")
        if os.path.exists(admin_panel_path):
            return FileResponse(admin_panel_path)
        return {"error": "Persona management page not found"}

    @app.get("/admin/content", tags=["system"])
    async def admin_content():
        """Serve content management page."""
        content_path = os.path.join(project_root, "admin_panel", "content.html")
        if os.path.exists(content_path):
            return FileResponse(content_path)
        return {"error": "Content management page not found"}

    @app.get("/admin/content/view", tags=["system"])
    async def admin_content_view():
        """Serve individual content view page."""
        content_view_path = os.path.join(
            project_root, "admin_panel", "content-view.html"
        )
        if os.path.exists(content_view_path):
            return FileResponse(content_view_path)
        return {"error": "Content view page not found"}

    @app.get("/admin/rss", tags=["system"])
    async def admin_rss():
        """Serve RSS feed management page."""
        rss_path = os.path.join(project_root, "admin_panel", "rss.html")
        if os.path.exists(rss_path):
            return FileResponse(rss_path)
        return {"error": "RSS management page not found"}

    @app.get("/admin/analytics", tags=["system"])
    async def admin_analytics():
        """Serve analytics dashboard page."""
        analytics_path = os.path.join(project_root, "admin_panel", "analytics.html")
        if os.path.exists(analytics_path):
            return FileResponse(analytics_path)
        return {"error": "Analytics page not found"}

    @app.get("/admin/settings", tags=["system"])
    async def admin_settings():
        """Serve system settings page."""
        settings_path = os.path.join(project_root, "admin_panel", "settings.html")
        if os.path.exists(settings_path):
            return FileResponse(settings_path)
        return {"error": "Settings page not found"}

    @app.get("/admin/diagnostics", tags=["system"])
    async def admin_diagnostics():
        """Serve AI diagnostics page."""
        diagnostics_path = os.path.join(project_root, "admin_panel", "diagnostics.html")
        if os.path.exists(diagnostics_path):
            return FileResponse(diagnostics_path)
        return {"error": "AI diagnostics page not found"}

    @app.get("/admin/system-monitoring", tags=["system"])
    async def admin_system_monitoring():
        """Serve system monitoring page for GPU temperature and fan control."""
        monitoring_path = os.path.join(project_root, "admin_panel", "system-monitoring.html")
        if os.path.exists(monitoring_path):
            return FileResponse(monitoring_path)
        return {"error": "System monitoring page not found"}

    @app.get("/ai-models-setup", tags=["system"])
    async def ai_models_setup():
        """Serve AI models setup page."""
        setup_path = os.path.join(project_root, "ai_models_setup.html")
        if os.path.exists(setup_path):
            return FileResponse(setup_path)
        return {"error": "AI models setup page not found"}

    @app.get("/gallery", tags=["public"])
    async def public_gallery():
        """Serve public gallery page (same as root for backward compatibility)."""
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
    async def health_check(db: AsyncSession = Depends(get_db_session)):
        """Health check endpoint for monitoring."""
        from datetime import datetime, timezone

        # Basic database connectivity check
        database_status = "unknown"
        try:
            # Simple query to check database connectivity
            result = await db.execute(text("SELECT 1"))
            if result.scalar() == 1:
                database_status = "healthy"
            else:
                database_status = "unhealthy"
        except Exception:
            database_status = "unhealthy"

        return {
            "status": "healthy",
            "database": database_status,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    @app.get("/gator-agent/status", tags=["gator-agent"])
    async def gator_agent_status_alias():
        """Alias for Gator agent status endpoint (backward compatibility)."""
        from backend.services.gator_agent_service import gator_agent

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

    @app.exception_handler(404)
    async def not_found_handler(request: Request, exc) -> Response:
        """Custom 404 handler."""
        return JSONResponse(
            status_code=404, content={"detail": f"Path {request.url.path} not found"}
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
