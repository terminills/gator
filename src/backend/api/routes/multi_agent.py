"""
Multi-Agent Ecosystem API Routes

Endpoints for Phase 4: Multi-Agent system features.
"""

from typing import Optional, List
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession

from backend.database.connection import get_db_session
from backend.services.multi_agent_service import MultiAgentService
from backend.services.agent_marketplace_service import AgentMarketplaceService
from backend.models.multi_agent import (
    AgentCreate,
    AgentUpdate,
    AgentResponse,
    AgentTaskCreate,
    AgentTaskResponse,
    AgentRoutingRequest,
    AgentRoutingResponse,
    AgentType,
    AgentStatus,
    AgentMarketplaceEntry,
)
from backend.config.logging import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/api/v1/multi-agent", tags=["multi-agent"])


# Agent Management Endpoints
@router.post("/agents/", response_model=AgentResponse, status_code=201)
async def register_agent(
    agent_data: AgentCreate,
    db: AsyncSession = Depends(get_db_session),
):
    """
    Register a new agent in the system.
    
    Args:
        agent_data: Agent configuration
    """
    try:
        service = MultiAgentService(db)
        agent = await service.register_agent(agent_data)
        return agent
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Agent registration failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/agents/{agent_id}", response_model=AgentResponse)
async def get_agent(
    agent_id: UUID,
    db: AsyncSession = Depends(get_db_session),
):
    """
    Get agent by ID.
    
    Args:
        agent_id: Agent identifier
    """
    try:
        service = MultiAgentService(db)
        agent = await service.get_agent(agent_id)
        
        if not agent:
            raise HTTPException(status_code=404, detail="Agent not found")
        
        return agent
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Get agent failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/agents/{agent_id}", response_model=AgentResponse)
async def update_agent(
    agent_id: UUID,
    update_data: AgentUpdate,
    db: AsyncSession = Depends(get_db_session),
):
    """
    Update agent configuration or status.
    
    Args:
        agent_id: Agent identifier
        update_data: Fields to update
    """
    try:
        service = MultiAgentService(db)
        agent = await service.update_agent(agent_id, update_data)
        
        if not agent:
            raise HTTPException(status_code=404, detail="Agent not found")
        
        return agent
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Agent update failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/agents/", response_model=List[AgentResponse])
async def list_agents(
    agent_type: Optional[AgentType] = Query(None),
    status: Optional[AgentStatus] = Query(None),
    is_plugin: Optional[bool] = Query(None),
    db: AsyncSession = Depends(get_db_session),
):
    """
    List all agents with optional filters.
    
    Args:
        agent_type: Filter by agent type
        status: Filter by status
        is_plugin: Filter by plugin flag
    """
    try:
        service = MultiAgentService(db)
        agents = await service.list_agents(agent_type, status, is_plugin)
        return agents
    except Exception as e:
        logger.error(f"List agents failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Task Management Endpoints
@router.post("/tasks/", response_model=AgentTaskResponse, status_code=201)
async def create_task(
    task_data: AgentTaskCreate,
    auto_assign: bool = Query(True, description="Automatically assign to best agent"),
    db: AsyncSession = Depends(get_db_session),
):
    """
    Create a new task.
    
    Args:
        task_data: Task configuration
        auto_assign: Automatically assign to best agent
    """
    try:
        service = MultiAgentService(db)
        task = await service.create_task(task_data, auto_assign)
        return task
    except Exception as e:
        logger.error(f"Task creation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/tasks/{task_id}/assign/{agent_id}")
async def assign_task(
    task_id: UUID,
    agent_id: UUID,
    db: AsyncSession = Depends(get_db_session),
):
    """
    Assign task to specific agent.
    
    Args:
        task_id: Task identifier
        agent_id: Agent identifier
    """
    try:
        service = MultiAgentService(db)
        success = await service.assign_task(task_id, agent_id)
        
        if not success:
            raise HTTPException(
                status_code=400,
                detail="Task assignment failed - check agent capacity"
            )
        
        return {"success": True, "message": "Task assigned"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Task assignment failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/tasks/{task_id}/complete")
async def complete_task(
    task_id: UUID,
    output_data: Optional[dict] = None,
    success: bool = True,
    db: AsyncSession = Depends(get_db_session),
):
    """
    Mark task as completed or failed.
    
    Args:
        task_id: Task identifier
        output_data: Task results
        success: Whether task succeeded
    """
    try:
        service = MultiAgentService(db)
        result = await service.complete_task(task_id, output_data, success)
        
        if not result:
            raise HTTPException(status_code=404, detail="Task not found")
        
        return {"success": True, "message": "Task completed"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Task completion failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Routing Endpoints
@router.post("/routing/", response_model=AgentRoutingResponse)
async def route_task(
    routing_request: AgentRoutingRequest,
    db: AsyncSession = Depends(get_db_session),
):
    """
    Find best agent for a task using automatic routing.
    
    Args:
        routing_request: Task routing requirements
    """
    try:
        service = MultiAgentService(db)
        routing_result = await service.route_task_to_agent(routing_request)
        return routing_result
    except Exception as e:
        logger.error(f"Task routing failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/workload/")
async def get_workload(
    db: AsyncSession = Depends(get_db_session),
):
    """Get system-wide agent workload statistics."""
    try:
        service = MultiAgentService(db)
        workload = await service.get_agent_workload()
        return workload
    except Exception as e:
        logger.error(f"Workload stats failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Marketplace Endpoints
@router.get("/marketplace/search")
async def search_marketplace(
    query: Optional[str] = Query(None),
    agent_type: Optional[AgentType] = Query(None),
    capabilities: Optional[List[str]] = Query(None),
    min_rating: float = Query(0.0, ge=0.0, le=5.0),
    free_only: bool = Query(False),
    sort_by: str = Query("rating", regex="^(rating|download_count|published_at)$"),
    limit: int = Query(20, ge=1, le=100),
    db: AsyncSession = Depends(get_db_session),
):
    """
    Search agent marketplace.
    
    Args:
        query: Text search query
        agent_type: Filter by agent type
        capabilities: Required capabilities
        min_rating: Minimum rating threshold
        free_only: Only show free agents
        sort_by: Sort field
        limit: Maximum results
    """
    try:
        service = AgentMarketplaceService(db)
        results = await service.search_marketplace(
            query=query,
            agent_type=agent_type,
            capabilities=capabilities,
            min_rating=min_rating,
            free_only=free_only,
            sort_by=sort_by,
            limit=limit
        )
        return results
    except Exception as e:
        logger.error(f"Marketplace search failed: {e}")
        raise HTTPException(status_code=500, detail="Marketplace search failed")


@router.get("/marketplace/agents/{agent_id}")
async def get_marketplace_agent(
    agent_id: UUID,
    db: AsyncSession = Depends(get_db_session),
):
    """
    Get detailed information about a marketplace agent.
    
    Args:
        agent_id: Agent identifier
    """
    try:
        service = AgentMarketplaceService(db)
        agent = await service.get_agent_details(agent_id)
        
        if not agent:
            raise HTTPException(status_code=404, detail="Agent not found")
        
        return agent
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Get marketplace agent failed: {e}")
        raise HTTPException(status_code=500, detail="Get marketplace agent failed")


@router.post("/marketplace/install/{agent_id}")
async def install_agent(
    agent_id: UUID,
    custom_name: Optional[str] = Query(None),
    db: AsyncSession = Depends(get_db_session),
):
    """
    Install an agent from the marketplace.
    
    Args:
        agent_id: Marketplace agent ID
        custom_name: Optional custom name for installed instance
    """
    try:
        service = AgentMarketplaceService(db)
        result = await service.install_agent(agent_id, custom_name)
        
        if not result.get("success"):
            raise HTTPException(status_code=400, detail=result.get("error"))
        
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Agent installation failed: {e}")
        raise HTTPException(status_code=500, detail="Agent installation failed")


@router.delete("/marketplace/uninstall/{agent_id}")
async def uninstall_agent(
    agent_id: UUID,
    db: AsyncSession = Depends(get_db_session),
):
    """
    Uninstall an agent.
    
    Args:
        agent_id: Installed agent ID
    """
    try:
        service = AgentMarketplaceService(db)
        result = await service.uninstall_agent(agent_id)
        
        if not result.get("success"):
            raise HTTPException(status_code=400, detail=result.get("error"))
        
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Agent uninstallation failed: {e}")
        raise HTTPException(status_code=500, detail="Agent uninstallation failed")


@router.get("/marketplace/updates/{agent_id}")
async def check_agent_updates(
    agent_id: UUID,
    db: AsyncSession = Depends(get_db_session),
):
    """
    Check if updates are available for an installed agent.
    
    Args:
        agent_id: Installed agent ID
    """
    try:
        service = AgentMarketplaceService(db)
        update_info = await service.check_updates(agent_id)
        return update_info
    except Exception as e:
        logger.error(f"Update check failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/marketplace/installed")
async def get_installed_agents(
    db: AsyncSession = Depends(get_db_session),
):
    """Get list of all installed agents."""
    try:
        service = AgentMarketplaceService(db)
        installed = await service.get_installed_agents()
        return installed
    except Exception as e:
        logger.error(f"Get installed agents failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/marketplace/stats")
async def get_marketplace_stats(
    db: AsyncSession = Depends(get_db_session),
):
    """Get marketplace statistics."""
    try:
        service = AgentMarketplaceService(db)
        stats = await service.get_marketplace_stats()
        return stats
    except Exception as e:
        logger.error(f"Marketplace stats failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/marketplace/publish")
async def publish_agent(
    agent_entry: AgentMarketplaceEntry,
    db: AsyncSession = Depends(get_db_session),
):
    """
    Publish a new agent to the marketplace.
    
    Args:
        agent_entry: Marketplace entry for the agent
    """
    try:
        service = AgentMarketplaceService(db)
        result = await service.publish_agent(agent_entry)
        
        if not result.get("success"):
            raise HTTPException(status_code=400, detail=result.get("error"))
        
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Agent publication failed: {e}")
        raise HTTPException(status_code=500, detail="Agent publication failed")
