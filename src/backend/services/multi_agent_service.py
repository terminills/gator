"""
Multi-Agent Coordination Service

Manages specialized agents, automatic routing, and distributed coordination.
"""

from typing import Dict, List, Optional, Any
from datetime import datetime, timezone, timedelta
from uuid import UUID
import asyncio

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_, func, or_

from backend.models.multi_agent import (
    AgentModel,
    AgentTaskModel,
    AgentCommunicationModel,
    AgentCreate,
    AgentUpdate,
    AgentResponse,
    AgentTaskCreate,
    AgentTaskResponse,
    AgentRoutingRequest,
    AgentRoutingResponse,
    AgentType,
    AgentStatus,
    TaskStatus,
)
from backend.config.logging import get_logger

logger = get_logger(__name__)


class MultiAgentService:
    """
    Service for multi-agent ecosystem management.
    
    Features:
    - Agent registration and management
    - Automatic task routing
    - Load balancing
    - Agent coordination
    """
    
    def __init__(self, db_session: AsyncSession):
        """
        Initialize multi-agent service.
        
        Args:
            db_session: Database session
        """
        self.db = db_session
    
    async def register_agent(
        self,
        agent_data: AgentCreate
    ) -> AgentResponse:
        """
        Register a new agent in the system.
        
        Args:
            agent_data: Agent configuration
            
        Returns:
            Registered agent record
        """
        try:
            # Check if agent name already exists
            stmt = select(AgentModel).where(
                AgentModel.agent_name == agent_data.agent_name
            )
            result = await self.db.execute(stmt)
            existing = result.scalar_one_or_none()
            
            if existing:
                raise ValueError(f"Agent {agent_data.agent_name} already exists")
            
            # Create agent
            agent_dict = agent_data.model_dump()
            if "agent_type" in agent_dict:
                agent_dict["agent_type"] = agent_dict["agent_type"].value
            
            agent = AgentModel(**agent_dict)
            agent.status = AgentStatus.IDLE.value
            agent.last_heartbeat = datetime.now(timezone.utc)
            
            self.db.add(agent)
            await self.db.commit()
            await self.db.refresh(agent)
            
            logger.info(
                f"Registered agent {agent.agent_name} "
                f"(type={agent.agent_type}, id={agent.id})"
            )
            
            return AgentResponse.model_validate(agent)
            
        except Exception as e:
            logger.error(f"Agent registration failed: {e}")
            await self.db.rollback()
            raise
    
    async def update_agent(
        self,
        agent_id: UUID,
        update_data: AgentUpdate
    ) -> Optional[AgentResponse]:
        """
        Update agent configuration or status.
        
        Args:
            agent_id: Agent identifier
            update_data: Fields to update
            
        Returns:
            Updated agent record
        """
        try:
            stmt = select(AgentModel).where(AgentModel.id == agent_id)
            result = await self.db.execute(stmt)
            agent = result.scalar_one_or_none()
            
            if not agent:
                return None
            
            # Update fields
            update_dict = update_data.model_dump(exclude_unset=True)
            for key, value in update_dict.items():
                if value is not None:
                    if hasattr(value, "value"):
                        value = value.value
                    setattr(agent, key, value)
            
            # Update heartbeat
            agent.last_heartbeat = datetime.now(timezone.utc)
            
            await self.db.commit()
            await self.db.refresh(agent)
            
            return AgentResponse.model_validate(agent)
            
        except Exception as e:
            logger.error(f"Agent update failed: {e}")
            await self.db.rollback()
            raise
    
    async def get_agent(self, agent_id: UUID) -> Optional[AgentResponse]:
        """Get agent by ID."""
        try:
            stmt = select(AgentModel).where(AgentModel.id == agent_id)
            result = await self.db.execute(stmt)
            agent = result.scalar_one_or_none()
            
            if agent:
                return AgentResponse.model_validate(agent)
            return None
            
        except Exception as e:
            logger.error(f"Get agent failed: {e}")
            return None
    
    async def list_agents(
        self,
        agent_type: Optional[AgentType] = None,
        status: Optional[AgentStatus] = None,
        is_plugin: Optional[bool] = None
    ) -> List[AgentResponse]:
        """
        List all agents with optional filters.
        
        Args:
            agent_type: Filter by agent type
            status: Filter by status
            is_plugin: Filter by plugin flag
            
        Returns:
            List of agents
        """
        try:
            stmt = select(AgentModel)
            
            conditions = []
            if agent_type:
                conditions.append(AgentModel.agent_type == agent_type.value)
            if status:
                conditions.append(AgentModel.status == status.value)
            if is_plugin is not None:
                conditions.append(AgentModel.is_plugin == is_plugin)
            
            if conditions:
                stmt = stmt.where(and_(*conditions))
            
            stmt = stmt.order_by(AgentModel.created_at.desc())
            
            result = await self.db.execute(stmt)
            agents = result.scalars().all()
            
            return [AgentResponse.model_validate(a) for a in agents]
            
        except Exception as e:
            logger.error(f"List agents failed: {e}")
            return []
    
    async def route_task_to_agent(
        self,
        routing_request: AgentRoutingRequest
    ) -> AgentRoutingResponse:
        """
        Automatically route task to best available agent.
        
        Args:
            routing_request: Task routing requirements
            
        Returns:
            Selected agent and routing decision
        """
        try:
            # Find candidate agents
            stmt = select(AgentModel).where(
                and_(
                    AgentModel.status.in_([AgentStatus.IDLE.value, AgentStatus.ACTIVE.value]),
                    AgentModel.success_rate >= routing_request.min_success_rate
                )
            )
            
            # Filter by agent type if specified
            if routing_request.preferred_agent_type:
                stmt = stmt.where(
                    AgentModel.agent_type == routing_request.preferred_agent_type.value
                )
            
            result = await self.db.execute(stmt)
            candidates = result.scalars().all()
            
            if not candidates:
                return AgentRoutingResponse(
                    selected_agent_id=None,
                    selected_agent_name=None,
                    routing_reason="No available agents matching criteria",
                    confidence=0.0
                )
            
            # Score candidates based on multiple factors
            scored_candidates = []
            
            for agent in candidates:
                score = 0.0
                reasons = []
                
                # Check required capabilities
                agent_capabilities = set(agent.capabilities or [])
                required_capabilities = set(routing_request.required_capabilities)
                
                capabilities_match = required_capabilities.issubset(agent_capabilities)
                if not capabilities_match:
                    continue  # Skip if missing required capabilities
                
                # Capability match (30%)
                score += 30
                reasons.append("has required capabilities")
                
                # Success rate (25%)
                score += agent.success_rate * 25
                reasons.append(f"{agent.success_rate:.1%} success rate")
                
                # Load balancing (20%)
                load_ratio = agent.current_load / agent.max_concurrent_tasks
                load_score = (1 - load_ratio) * 20
                score += load_score
                reasons.append(f"{agent.current_load}/{agent.max_concurrent_tasks} tasks")
                
                # Task completion history (15%)
                total_tasks = agent.tasks_completed + agent.tasks_failed
                if total_tasks > 0:
                    history_score = (agent.tasks_completed / total_tasks) * 15
                    score += history_score
                    reasons.append(f"{agent.tasks_completed} tasks completed")
                
                # Responsiveness (10%) - based on heartbeat recency
                if agent.last_heartbeat:
                    time_since_heartbeat = (
                        datetime.now(timezone.utc) - agent.last_heartbeat
                    ).total_seconds()
                    
                    if time_since_heartbeat < 60:  # Less than 1 minute
                        score += 10
                        reasons.append("highly responsive")
                    elif time_since_heartbeat < 300:  # Less than 5 minutes
                        score += 5
                        reasons.append("responsive")
                
                scored_candidates.append({
                    "agent": agent,
                    "score": score,
                    "reasons": reasons
                })
            
            if not scored_candidates:
                return AgentRoutingResponse(
                    selected_agent_id=None,
                    selected_agent_name=None,
                    routing_reason="No agents with required capabilities available",
                    confidence=0.0
                )
            
            # Sort by score
            scored_candidates.sort(key=lambda x: x["score"], reverse=True)
            
            # Select best agent
            best = scored_candidates[0]
            best_agent = best["agent"]
            
            # Calculate confidence (0-1)
            confidence = min(1.0, best["score"] / 100)
            
            # Prepare alternatives (top 3)
            alternatives = [
                {
                    "agent_id": str(c["agent"].id),
                    "agent_name": c["agent"].agent_name,
                    "score": c["score"],
                    "reasons": c["reasons"]
                }
                for c in scored_candidates[1:4]
            ]
            
            logger.info(
                f"Routed task to agent {best_agent.agent_name} "
                f"(score={best['score']:.1f}, confidence={confidence:.2f})"
            )
            
            return AgentRoutingResponse(
                selected_agent_id=best_agent.id,
                selected_agent_name=best_agent.agent_name,
                routing_reason=", ".join(best["reasons"]),
                confidence=confidence,
                alternative_agents=alternatives
            )
            
        except Exception as e:
            logger.error(f"Task routing failed: {e}")
            return AgentRoutingResponse(
                selected_agent_id=None,
                selected_agent_name=None,
                routing_reason=f"Routing error: {str(e)}",
                confidence=0.0
            )
    
    async def create_task(
        self,
        task_data: AgentTaskCreate,
        auto_assign: bool = True
    ) -> AgentTaskResponse:
        """
        Create a new task and optionally assign to agent.
        
        Args:
            task_data: Task configuration
            auto_assign: Automatically assign to best agent
            
        Returns:
            Created task record
        """
        try:
            # Create task
            task_dict = task_data.model_dump()
            if "priority" in task_dict:
                task_dict["priority"] = task_dict["priority"].value
            
            task = AgentTaskModel(**task_dict)
            task.status = TaskStatus.PENDING.value
            
            self.db.add(task)
            await self.db.commit()
            await self.db.refresh(task)
            
            # Auto-assign if requested
            if auto_assign:
                # Determine required capabilities from task type
                required_capabilities = self._get_capabilities_for_task_type(
                    task.task_type
                )
                
                routing_request = AgentRoutingRequest(
                    task_type=task.task_type,
                    required_capabilities=required_capabilities,
                    priority=getattr(task_data, "priority", "normal")
                )
                
                routing_result = await self.route_task_to_agent(routing_request)
                
                if routing_result.selected_agent_id:
                    await self.assign_task(task.id, routing_result.selected_agent_id)
                    # Refresh task after assignment
                    await self.db.refresh(task)
            
            logger.info(f"Created task {task.id}: {task.task_name}")
            
            return AgentTaskResponse.model_validate(task)
            
        except Exception as e:
            logger.error(f"Task creation failed: {e}")
            await self.db.rollback()
            raise
    
    def _get_capabilities_for_task_type(self, task_type: str) -> List[str]:
        """Map task types to required capabilities."""
        capability_map = {
            "content_generation": ["generate", "create", "text"],
            "content_review": ["review", "analyze", "quality_check"],
            "content_optimization": ["optimize", "improve", "analyze"],
            "image_generation": ["generate", "image", "visual"],
            "video_generation": ["generate", "video", "multimedia"],
            "social_posting": ["publish", "social_media", "scheduling"],
            "analytics": ["analyze", "metrics", "reporting"],
        }
        
        return capability_map.get(task_type, ["general"])
    
    async def assign_task(
        self,
        task_id: UUID,
        agent_id: UUID
    ) -> bool:
        """
        Assign task to specific agent.
        
        Args:
            task_id: Task identifier
            agent_id: Agent identifier
            
        Returns:
            Success status
        """
        try:
            # Get task
            task_stmt = select(AgentTaskModel).where(AgentTaskModel.id == task_id)
            task_result = await self.db.execute(task_stmt)
            task = task_result.scalar_one_or_none()
            
            if not task:
                return False
            
            # Get agent
            agent_stmt = select(AgentModel).where(AgentModel.id == agent_id)
            agent_result = await self.db.execute(agent_stmt)
            agent = agent_result.scalar_one_or_none()
            
            if not agent:
                return False
            
            # Check agent capacity
            if agent.current_load >= agent.max_concurrent_tasks:
                logger.warning(f"Agent {agent.agent_name} at capacity")
                return False
            
            # Assign task
            task.agent_id = agent_id
            task.status = TaskStatus.ASSIGNED.value
            
            # Update agent load
            agent.current_load += 1
            if agent.status == AgentStatus.IDLE.value:
                agent.status = AgentStatus.ACTIVE.value
            
            await self.db.commit()
            
            logger.info(f"Assigned task {task_id} to agent {agent.agent_name}")
            
            return True
            
        except Exception as e:
            logger.error(f"Task assignment failed: {e}")
            await self.db.rollback()
            return False
    
    async def complete_task(
        self,
        task_id: UUID,
        output_data: Optional[Dict[str, Any]] = None,
        success: bool = True
    ) -> bool:
        """
        Mark task as completed or failed.
        
        Args:
            task_id: Task identifier
            output_data: Task results
            success: Whether task succeeded
            
        Returns:
            Success status
        """
        try:
            # Get task
            task_stmt = select(AgentTaskModel).where(AgentTaskModel.id == task_id)
            task_result = await self.db.execute(task_stmt)
            task = task_result.scalar_one_or_none()
            
            if not task or not task.agent_id:
                return False
            
            # Get agent
            agent_stmt = select(AgentModel).where(AgentModel.id == task.agent_id)
            agent_result = await self.db.execute(agent_stmt)
            agent = agent_result.scalar_one_or_none()
            
            if not agent:
                return False
            
            # Update task
            task.status = TaskStatus.COMPLETED.value if success else TaskStatus.FAILED.value
            task.completed_at = datetime.now(timezone.utc)
            task.output_data = output_data
            
            # Calculate completion time
            if task.started_at:
                completion_time = (
                    task.completed_at - task.started_at
                ).total_seconds()
            else:
                completion_time = 0
            
            # Update agent metrics
            if success:
                agent.tasks_completed += 1
            else:
                agent.tasks_failed += 1
            
            total_tasks = agent.tasks_completed + agent.tasks_failed
            agent.success_rate = agent.tasks_completed / total_tasks
            
            # Update average completion time
            if agent.average_completion_time is None:
                agent.average_completion_time = completion_time
            else:
                # Exponential moving average
                alpha = 0.2
                agent.average_completion_time = (
                    alpha * completion_time +
                    (1 - alpha) * agent.average_completion_time
                )
            
            # Update agent load
            agent.current_load = max(0, agent.current_load - 1)
            if agent.current_load == 0:
                agent.status = AgentStatus.IDLE.value
            
            await self.db.commit()
            
            logger.info(
                f"Completed task {task_id} "
                f"(success={success}, time={completion_time:.1f}s)"
            )
            
            return True
            
        except Exception as e:
            logger.error(f"Task completion failed: {e}")
            await self.db.rollback()
            return False
    
    async def get_agent_workload(self) -> Dict[str, Any]:
        """
        Get system-wide agent workload statistics.
        
        Returns:
            Workload metrics
        """
        try:
            # Get all agents
            agents_stmt = select(AgentModel)
            agents_result = await self.db.execute(agents_stmt)
            agents = agents_result.scalars().all()
            
            # Get pending tasks
            pending_stmt = select(func.count(AgentTaskModel.id)).where(
                AgentTaskModel.status == TaskStatus.PENDING.value
            )
            pending_result = await self.db.execute(pending_stmt)
            pending_count = pending_result.scalar()
            
            # Calculate metrics
            total_agents = len(agents)
            active_agents = sum(1 for a in agents if a.status == AgentStatus.ACTIVE.value)
            idle_agents = sum(1 for a in agents if a.status == AgentStatus.IDLE.value)
            offline_agents = sum(1 for a in agents if a.status == AgentStatus.OFFLINE.value)
            
            total_capacity = sum(a.max_concurrent_tasks for a in agents)
            current_load = sum(a.current_load for a in agents)
            
            utilization = (current_load / total_capacity * 100) if total_capacity > 0 else 0
            
            # Agent type distribution
            type_distribution = {}
            for agent in agents:
                agent_type = agent.agent_type
                type_distribution[agent_type] = type_distribution.get(agent_type, 0) + 1
            
            return {
                "total_agents": total_agents,
                "active_agents": active_agents,
                "idle_agents": idle_agents,
                "offline_agents": offline_agents,
                "total_capacity": total_capacity,
                "current_load": current_load,
                "utilization_percent": float(utilization),
                "pending_tasks": pending_count,
                "agent_type_distribution": type_distribution,
                "avg_success_rate": float(
                    sum(a.success_rate for a in agents) / total_agents
                ) if total_agents > 0 else 0.0
            }
            
        except Exception as e:
            logger.error(f"Workload stats failed: {e}")
            return {}
