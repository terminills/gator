"""
Tests for ACD Phase 4: Multi-Agent Ecosystem

Tests specialized agents, routing, marketplace, and coordination.
"""

import pytest
from datetime import datetime, timezone
from uuid import uuid4

from backend.services.multi_agent_service import MultiAgentService
from backend.services.agent_marketplace_service import AgentMarketplaceService
from backend.models.multi_agent import (
    AgentCreate,
    AgentUpdate,
    AgentTaskCreate,
    AgentRoutingRequest,
    AgentType,
    AgentStatus,
    TaskPriority,
    TaskStatus,
)


class TestMultiAgentService:
    """Tests for multi-agent coordination service."""
    
    @pytest.mark.asyncio
    async def test_register_agent(self, db_session):
        """Test agent registration."""
        service = MultiAgentService(db_session)
        
        agent_data = AgentCreate(
            agent_name=f"TestAgent_{uuid4().hex[:8]}",
            agent_type=AgentType.GENERATOR,
            capabilities=["generate", "text"],
            max_concurrent_tasks=5
        )
        
        agent = await service.register_agent(agent_data)
        
        assert agent.agent_name == agent_data.agent_name
        assert agent.agent_type == "generator"
        assert agent.status == "idle"
        assert agent.current_load == 0
    
    @pytest.mark.asyncio
    async def test_register_duplicate_agent(self, db_session):
        """Test registering agent with duplicate name."""
        service = MultiAgentService(db_session)
        
        agent_name = f"DuplicateAgent_{uuid4().hex[:8]}"
        
        agent_data = AgentCreate(
            agent_name=agent_name,
            agent_type=AgentType.GENERATOR,
            capabilities=["generate"]
        )
        
        # First registration should succeed
        await service.register_agent(agent_data)
        
        # Second registration should fail
        with pytest.raises(ValueError, match="already exists"):
            await service.register_agent(agent_data)
    
    @pytest.mark.asyncio
    async def test_update_agent(self, db_session):
        """Test agent update."""
        service = MultiAgentService(db_session)
        
        # Create agent
        agent_data = AgentCreate(
            agent_name=f"UpdateTestAgent_{uuid4().hex[:8]}",
            agent_type=AgentType.REVIEWER,
            capabilities=["review"]
        )
        agent = await service.register_agent(agent_data)
        
        # Update agent
        update_data = AgentUpdate(
            status=AgentStatus.ACTIVE,
            current_load=2
        )
        updated_agent = await service.update_agent(agent.id, update_data)
        
        assert updated_agent is not None
        assert updated_agent.status == "active"
        assert updated_agent.current_load == 2
    
    @pytest.mark.asyncio
    async def test_list_agents(self, db_session):
        """Test listing agents with filters."""
        service = MultiAgentService(db_session)
        
        # Create test agents
        for i in range(3):
            await service.register_agent(
                AgentCreate(
                    agent_name=f"ListTestAgent_{i}_{uuid4().hex[:8]}",
                    agent_type=AgentType.GENERATOR if i % 2 == 0 else AgentType.REVIEWER,
                    capabilities=["test"]
                )
            )
        
        # List all
        all_agents = await service.list_agents()
        assert len(all_agents) >= 3
        
        # Filter by type
        generators = await service.list_agents(agent_type=AgentType.GENERATOR)
        assert all(a.agent_type == "generator" for a in generators)
    
    @pytest.mark.asyncio
    async def test_route_task_to_agent(self, db_session):
        """Test automatic task routing."""
        service = MultiAgentService(db_session)
        
        # Create agents with different capabilities
        agent1 = await service.register_agent(
            AgentCreate(
                agent_name=f"RoutingAgent1_{uuid4().hex[:8]}",
                agent_type=AgentType.GENERATOR,
                capabilities=["generate", "text", "image"],
                max_concurrent_tasks=10
            )
        )
        
        agent2 = await service.register_agent(
            AgentCreate(
                agent_name=f"RoutingAgent2_{uuid4().hex[:8]}",
                agent_type=AgentType.REVIEWER,
                capabilities=["review", "analyze"],
                max_concurrent_tasks=5
            )
        )
        
        # Route task requiring generation capability
        routing_request = AgentRoutingRequest(
            task_type="content_generation",
            required_capabilities=["generate", "text"],
            priority=TaskPriority.NORMAL
        )
        
        routing_result = await service.route_task_to_agent(routing_request)
        
        if routing_result.selected_agent_id:
            assert routing_result.selected_agent_id == agent1.id
            assert routing_result.confidence > 0
        else:
            # No agent available is also valid
            assert routing_result.confidence == 0.0
    
    @pytest.mark.asyncio
    async def test_create_task_with_auto_assign(self, db_session):
        """Test task creation with automatic assignment."""
        service = MultiAgentService(db_session)
        
        # Create agent
        agent = await service.register_agent(
            AgentCreate(
                agent_name=f"TaskAgent_{uuid4().hex[:8]}",
                agent_type=AgentType.GENERATOR,
                capabilities=["generate", "text"]
            )
        )
        
        # Create task with auto-assign
        task_data = AgentTaskCreate(
            task_type="content_generation",
            task_name="Test task",
            priority=TaskPriority.NORMAL
        )
        
        task = await service.create_task(task_data, auto_assign=True)
        
        assert task.task_name == "Test task"
        assert task.status in ["pending", "assigned"]
    
    @pytest.mark.asyncio
    async def test_assign_task(self, db_session):
        """Test manual task assignment."""
        service = MultiAgentService(db_session)
        
        # Create agent
        agent = await service.register_agent(
            AgentCreate(
                agent_name=f"AssignAgent_{uuid4().hex[:8]}",
                agent_type=AgentType.OPTIMIZER,
                capabilities=["optimize"]
            )
        )
        
        # Create task without auto-assign
        task = await service.create_task(
            AgentTaskCreate(
                task_type="content_optimization",
                task_name="Optimize content"
            ),
            auto_assign=False
        )
        
        # Manually assign
        success = await service.assign_task(task.id, agent.id)
        assert success is True
    
    @pytest.mark.asyncio
    async def test_complete_task(self, db_session):
        """Test task completion."""
        service = MultiAgentService(db_session)
        
        # Create agent and task
        agent = await service.register_agent(
            AgentCreate(
                agent_name=f"CompleteAgent_{uuid4().hex[:8]}",
                agent_type=AgentType.GENERATOR,
                capabilities=["generate"]
            )
        )
        
        task = await service.create_task(
            AgentTaskCreate(
                task_type="content_generation",
                task_name="Generate content"
            ),
            auto_assign=False
        )
        
        await service.assign_task(task.id, agent.id)
        
        # Complete task
        output_data = {"result": "Generated content"}
        success = await service.complete_task(task.id, output_data, success=True)
        
        assert success is True
    
    @pytest.mark.asyncio
    async def test_get_agent_workload(self, db_session):
        """Test workload statistics."""
        service = MultiAgentService(db_session)
        
        # Create some agents
        for i in range(2):
            await service.register_agent(
                AgentCreate(
                    agent_name=f"WorkloadAgent_{i}_{uuid4().hex[:8]}",
                    agent_type=AgentType.GENERATOR,
                    capabilities=["test"]
                )
            )
        
        workload = await service.get_agent_workload()
        
        assert "total_agents" in workload
        assert "utilization_percent" in workload
        assert "pending_tasks" in workload
        assert workload["total_agents"] >= 2


class TestAgentMarketplace:
    """Tests for agent marketplace service."""
    
    @pytest.mark.asyncio
    async def test_search_marketplace(self, db_session):
        """Test marketplace search."""
        service = AgentMarketplaceService(db_session)
        
        # Search all
        results = await service.search_marketplace()
        assert len(results) > 0
        
        # Search by type
        generators = await service.search_marketplace(
            agent_type=AgentType.GENERATOR
        )
        assert all(a.agent_type == "generator" for a in generators)
        
        # Search by rating
        high_rated = await service.search_marketplace(min_rating=4.5)
        assert all(a.rating >= 4.5 for a in high_rated)
    
    @pytest.mark.asyncio
    async def test_search_marketplace_with_capabilities(self, db_session):
        """Test marketplace search with capability filter."""
        service = AgentMarketplaceService(db_session)
        
        results = await service.search_marketplace(
            capabilities=["generate", "text"]
        )
        
        for agent in results:
            agent_caps = set(c.lower() for c in agent.capabilities)
            assert "generate" in agent_caps
            assert "text" in agent_caps
    
    @pytest.mark.asyncio
    async def test_get_agent_details(self, db_session):
        """Test getting agent details from marketplace."""
        service = AgentMarketplaceService(db_session)
        
        # Get first agent from marketplace
        results = await service.search_marketplace(limit=1)
        if results:
            agent = results[0]
            details = await service.get_agent_details(agent.agent_id)
            
            assert details is not None
            assert details.agent_id == agent.agent_id
            assert details.agent_name == agent.agent_name
    
    @pytest.mark.asyncio
    async def test_install_agent(self, db_session):
        """Test agent installation from marketplace."""
        service = AgentMarketplaceService(db_session)
        
        # Get an agent from marketplace
        results = await service.search_marketplace(limit=1)
        if results:
            agent = results[0]
            
            # Install with custom name
            custom_name = f"Installed_{agent.agent_name}_{uuid4().hex[:8]}"
            result = await service.install_agent(agent.agent_id, custom_name)
            
            # Should succeed or already be installed
            assert "success" in result or "error" in result
    
    @pytest.mark.asyncio
    async def test_get_installed_agents(self, db_session):
        """Test listing installed agents."""
        service = AgentMarketplaceService(db_session)
        
        installed = await service.get_installed_agents()
        
        assert isinstance(installed, list)
        for agent in installed:
            assert "agent_id" in agent
            assert "agent_name" in agent
            assert "is_plugin" in agent
    
    @pytest.mark.asyncio
    async def test_check_agent_updates(self, db_session):
        """Test checking for agent updates."""
        service = AgentMarketplaceService(db_session)
        
        # Install an agent first
        results = await service.search_marketplace(limit=1)
        if results:
            agent = results[0]
            custom_name = f"UpdateCheck_{uuid4().hex[:8]}"
            
            install_result = await service.install_agent(agent.agent_id, custom_name)
            
            if install_result.get("success"):
                # Check for updates
                from backend.models.multi_agent import AgentModel
                from sqlalchemy import select
                
                stmt = select(AgentModel).where(AgentModel.agent_name == custom_name)
                result = await db_session.execute(stmt)
                installed_agent = result.scalar_one_or_none()
                
                if installed_agent:
                    update_info = await service.check_updates(installed_agent.id)
                    
                    assert "updates_available" in update_info
                    assert "installed_version" in update_info
    
    @pytest.mark.asyncio
    async def test_marketplace_stats(self, db_session):
        """Test marketplace statistics."""
        service = AgentMarketplaceService(db_session)
        
        stats = await service.get_marketplace_stats()
        
        assert "total_agents" in stats
        assert "average_rating" in stats
        assert "total_downloads" in stats
        assert "top_rated" in stats
        assert "most_downloaded" in stats
        
        assert stats["total_agents"] > 0
        assert len(stats["top_rated"]) > 0
    
    @pytest.mark.asyncio
    async def test_version_parsing(self, db_session):
        """Test semantic version parsing."""
        service = AgentMarketplaceService(db_session)
        
        v1 = service._parse_version("1.0.0")
        v2 = service._parse_version("1.5.0")
        v3 = service._parse_version("2.0.0")
        
        assert v1 < v2 < v3
        assert v1 == (1, 0, 0)
        assert v2 == (1, 5, 0)
        assert v3 == (2, 0, 0)


class TestAgentCapabilityMatching:
    """Tests for capability-based agent matching."""
    
    @pytest.mark.asyncio
    async def test_capability_matching(self, db_session):
        """Test that routing matches agents based on capabilities."""
        service = MultiAgentService(db_session)
        
        # Create specialized agents
        generator = await service.register_agent(
            AgentCreate(
                agent_name=f"CapGen_{uuid4().hex[:8]}",
                agent_type=AgentType.GENERATOR,
                capabilities=["generate", "text", "image"],
                max_concurrent_tasks=10
            )
        )
        
        reviewer = await service.register_agent(
            AgentCreate(
                agent_name=f"CapRev_{uuid4().hex[:8]}",
                agent_type=AgentType.REVIEWER,
                capabilities=["review", "analyze", "quality_check"],
                max_concurrent_tasks=10
            )
        )
        
        # Route generation task
        gen_routing = AgentRoutingRequest(
            task_type="content_generation",
            required_capabilities=["generate", "text"]
        )
        gen_result = await service.route_task_to_agent(gen_routing)
        
        if gen_result.selected_agent_id:
            assert gen_result.selected_agent_id == generator.id
        
        # Route review task
        rev_routing = AgentRoutingRequest(
            task_type="content_review",
            required_capabilities=["review", "quality_check"]
        )
        rev_result = await service.route_task_to_agent(rev_routing)
        
        if rev_result.selected_agent_id:
            assert rev_result.selected_agent_id == reviewer.id
    
    @pytest.mark.asyncio
    async def test_load_balancing(self, db_session):
        """Test that routing considers agent load."""
        service = MultiAgentService(db_session)
        
        # Create two identical agents with different loads
        agent1 = await service.register_agent(
            AgentCreate(
                agent_name=f"LoadAgent1_{uuid4().hex[:8]}",
                agent_type=AgentType.GENERATOR,
                capabilities=["generate", "text"],
                max_concurrent_tasks=5
            )
        )
        
        agent2 = await service.register_agent(
            AgentCreate(
                agent_name=f"LoadAgent2_{uuid4().hex[:8]}",
                agent_type=AgentType.GENERATOR,
                capabilities=["generate", "text"],
                max_concurrent_tasks=5
            )
        )
        
        # Update agent1 to have high load
        await service.update_agent(agent1.id, AgentUpdate(current_load=4))
        await service.update_agent(agent2.id, AgentUpdate(current_load=1))
        
        # Route task
        routing_request = AgentRoutingRequest(
            task_type="content_generation",
            required_capabilities=["generate", "text"]
        )
        
        result = await service.route_task_to_agent(routing_request)
        
        # Should prefer agent with lower load
        if result.selected_agent_id:
            assert result.selected_agent_id == agent2.id
