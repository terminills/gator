"""
FastAPI Application Configuration

Main entry point for the Gator AI Influencer Platform backend API.
Configured following best practices from BEST_PRACTICES.md.
"""

from contextlib import asynccontextmanager
from typing import AsyncGenerator

import uvicorn
from fastapi import Depends, FastAPI, Request, Response, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from backend.api.routes import (
    acd,
    analytics,
    branding,
    civitai,
    content,
    creator,
    database_admin,
    diagnostics,
    direct_messaging,
    dns,
    enhanced_persona,
    feeds,
    friend_groups,
    gator_agent,
    huggingface,
    installed_models,
    interactive,
    persona,
    plugins,
    public,
    reasoning_orchestrator,
    segments,
    sentiment,
)
from backend.api.routes import settings as settings_routes
from backend.api.routes import (
    setup,
    social,
    system_monitoring,
    users,
)
from backend.api.websocket import websocket_endpoint
from backend.config.logging import setup_logging
from backend.config.settings import get_settings
from backend.database.connection import get_db_session
from backend.utils.paths import get_paths

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
    # Get centralized paths
    paths = get_paths()

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

    # Mount static files using centralized paths
    frontend_path = paths.frontend_dir
    if frontend_path.exists():
        app.mount("/static", StaticFiles(directory=str(frontend_path)), name="static")

    # Mount content directory for generated content (images, videos, etc.)
    content_path = paths.generated_content_dir
    if content_path.exists():
        app.mount("/content", StaticFiles(directory=str(content_path)), name="content")
        print(f"Mounted content directory: {content_path}")
    else:
        # Create the directory if it doesn't exist
        content_path.mkdir(parents=True, exist_ok=True)
        app.mount("/content", StaticFiles(directory=str(content_path)), name="content")
        print(f"Created and mounted content directory: {content_path}")

    # Mount base_images directory for persona base images
    base_images_path = paths.base_images_dir
    if base_images_path.exists():
        app.mount(
            "/base_images",
            StaticFiles(directory=str(base_images_path)),
            name="base_images",
        )
        print(f"Mounted base_images directory: {base_images_path}")
    else:
        # Create the directory if it doesn't exist
        base_images_path.mkdir(parents=True, exist_ok=True)
        app.mount(
            "/base_images",
            StaticFiles(directory=str(base_images_path)),
            name="base_images",
        )
        print(f"Created and mounted base_images directory: {base_images_path}")

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
    app.include_router(civitai.router, prefix="/api/v1")
    app.include_router(installed_models.router, prefix="/api/v1")
    app.include_router(huggingface.router, prefix="/api/v1")

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
        gallery_path = frontend_path / "gallery.html"
        if gallery_path.exists():
            return FileResponse(str(gallery_path))
        return {
            "message": "Gator AI Influencer Platform",
            "version": "0.1.0",
            "status": "operational",
        }

    @app.get("/admin", tags=["system"])
    async def admin_dashboard():
        """Serve main admin dashboard hub."""
        admin_panel = paths.admin_panel_dir
        # Serve the new modern admin dashboard
        dashboard_path = admin_panel / "dashboard.html"
        if dashboard_path.exists():
            return FileResponse(str(dashboard_path))
        # Fallback to simple admin panel
        admin_panel_path = admin_panel / "index.html"
        if admin_panel_path.exists():
            return FileResponse(str(admin_panel_path))
        return {"error": "Admin dashboard not found"}

    @app.get("/admin/personas", tags=["system"])
    async def admin_personas(request: Request):
        """Serve persona management page or persona editor based on query params."""
        admin_panel = paths.admin_panel_dir
        # Check if action parameter is present (create or edit)
        action = request.query_params.get("action")

        if action in ["create", "edit"]:
            # Serve the persona editor
            editor_path = admin_panel / "persona-editor.html"
            if editor_path.exists():
                return FileResponse(str(editor_path))

        # Default: serve the personas list page
        personas_path = admin_panel / "personas.html"
        if personas_path.exists():
            return FileResponse(str(personas_path))
        # Fallback to main admin
        admin_panel_path = admin_panel / "index.html"
        if admin_panel_path.exists():
            return FileResponse(str(admin_panel_path))
        return {"error": "Persona management page not found"}

    @app.get("/admin/content", tags=["system"])
    async def admin_content():
        """Serve content management page."""
        content_page = paths.admin_panel_dir / "content.html"
        if content_page.exists():
            return FileResponse(str(content_page))
        return {"error": "Content management page not found"}

    @app.get("/admin/content/view", tags=["system"])
    async def admin_content_view():
        """Serve individual content view page."""
        content_view_path = paths.admin_panel_dir / "content-view.html"
        if content_view_path.exists():
            return FileResponse(str(content_view_path))
        return {"error": "Content view page not found"}

    @app.get("/admin/rss", tags=["system"])
    async def admin_rss():
        """Serve RSS feed management page."""
        rss_path = paths.admin_panel_dir / "rss.html"
        if rss_path.exists():
            return FileResponse(str(rss_path))
        return {"error": "RSS management page not found"}

    @app.get("/admin/analytics", tags=["system"])
    async def admin_analytics():
        """Serve analytics dashboard page."""
        analytics_path = paths.admin_panel_dir / "analytics.html"
        if analytics_path.exists():
            return FileResponse(str(analytics_path))
        return {"error": "Analytics page not found"}

    @app.get("/admin/settings", tags=["system"])
    async def admin_settings():
        """Serve system settings page."""
        settings_path = paths.admin_panel_dir / "settings.html"
        if settings_path.exists():
            return FileResponse(str(settings_path))
        return {"error": "Settings page not found"}

    @app.get("/admin/diagnostics", tags=["system"])
    async def admin_diagnostics():
        """Serve AI diagnostics page."""
        diagnostics_path = paths.admin_panel_dir / "diagnostics.html"
        if diagnostics_path.exists():
            return FileResponse(str(diagnostics_path))
        return {"error": "AI diagnostics page not found"}

    @app.get("/admin/system-monitoring", tags=["system"])
    async def admin_system_monitoring():
        """Serve system monitoring page for GPU temperature and fan control."""
        monitoring_path = paths.admin_panel_dir / "system-monitoring.html"
        if monitoring_path.exists():
            return FileResponse(str(monitoring_path))
        return {"error": "System monitoring page not found"}

    @app.get("/ai-models-setup", tags=["system"])
    async def ai_models_setup():
        """Serve AI models setup page."""
        setup_path = paths.project_root / "ai_models_setup.html"
        if setup_path.exists():
            return FileResponse(str(setup_path))
        return {"error": "AI models setup page not found"}

    @app.get("/gallery", tags=["public"])
    async def public_gallery():
        """Serve public gallery page (same as root for backward compatibility)."""
        gallery_path = frontend_path / "gallery.html"
        if gallery_path.exists():
            return FileResponse(str(gallery_path))
        return {"error": "Gallery page not found"}

    @app.get("/gallery/persona/{persona_id}", tags=["public"])
    async def persona_detail(persona_id: str):
        """Serve persona detail page."""
        persona_page = frontend_path / "persona.html"
        if persona_page.exists():
            return FileResponse(str(persona_page))
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
        """Alias for Gator agent status endpoint."""
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
