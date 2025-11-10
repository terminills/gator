"""
Agent Marketplace and Plugin System Service

Manages agent discovery, installation, versioning, and marketplace operations.
"""

from typing import Dict, List, Optional, Any
from datetime import datetime, timezone
from uuid import UUID
import json
import hashlib

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_, func, or_

from backend.models.multi_agent import (
    AgentModel,
    AgentCreate,
    AgentMarketplaceEntry,
    AgentType,
)
from backend.services.multi_agent_service import MultiAgentService
from backend.config.logging import get_logger

logger = get_logger(__name__)


class AgentMarketplaceService:
    """
    Service for agent marketplace and plugin management.
    
    Features:
    - Agent discovery and search
    - Plugin installation
    - Version management
    - Ratings and reviews
    """
    
    def __init__(self, db_session: AsyncSession):
        """
        Initialize agent marketplace service.
        
        Args:
            db_session: Database session
        """
        self.db = db_session
        self.multi_agent_service = MultiAgentService(db_session)
        
        # In-memory marketplace catalog (in production, this would be a separate DB/service)
        self.marketplace_catalog: Dict[str, AgentMarketplaceEntry] = {}
        self._init_default_agents()
    
    def _init_default_agents(self):
        """Initialize marketplace with default agent plugins."""
        default_agents = [
            AgentMarketplaceEntry(
                agent_id=UUID("00000000-0000-0000-0000-000000000001"),
                agent_name="ContentGeneratorPro",
                agent_type="generator",
                version="1.0.0",
                author="Gator Team",
                description="Advanced content generator with multi-format support. "
                           "Generates text, images, and video content with high quality.",
                capabilities=["generate", "text", "image", "video", "creative"],
                rating=4.8,
                download_count=1250,
                price=0.0,
                source_url="https://github.com/gator/agents/content-generator-pro",
                documentation_url="https://docs.gator.ai/agents/content-generator-pro",
                published_at=datetime.now(timezone.utc)
            ),
            AgentMarketplaceEntry(
                agent_id=UUID("00000000-0000-0000-0000-000000000002"),
                agent_name="QualityReviewerAI",
                agent_type="reviewer",
                version="2.1.0",
                author="Gator Team",
                description="Intelligent quality reviewer with comprehensive checks. "
                           "Reviews content for quality, appropriateness, and brand consistency.",
                capabilities=["review", "analyze", "quality_check", "brand_safety"],
                rating=4.9,
                download_count=980,
                price=0.0,
                source_url="https://github.com/gator/agents/quality-reviewer",
                documentation_url="https://docs.gator.ai/agents/quality-reviewer",
                published_at=datetime.now(timezone.utc)
            ),
            AgentMarketplaceEntry(
                agent_id=UUID("00000000-0000-0000-0000-000000000003"),
                agent_name="EngagementOptimizer",
                agent_type="optimizer",
                version="1.5.0",
                author="Gator Team",
                description="Data-driven engagement optimizer. Analyzes performance "
                           "and provides actionable recommendations.",
                capabilities=["optimize", "analyze", "metrics", "recommendations"],
                rating=4.7,
                download_count=1100,
                price=0.0,
                source_url="https://github.com/gator/agents/engagement-optimizer",
                documentation_url="https://docs.gator.ai/agents/engagement-optimizer",
                published_at=datetime.now(timezone.utc)
            ),
            AgentMarketplaceEntry(
                agent_id=UUID("00000000-0000-0000-0000-000000000004"),
                agent_name="SocialMediaScheduler",
                agent_type="generator",
                version="1.2.0",
                author="Community",
                description="Smart social media scheduler with optimal timing predictions. "
                           "Posts content at the best times for maximum engagement.",
                capabilities=["schedule", "publish", "social_media", "timing"],
                rating=4.6,
                download_count=850,
                price=0.0,
                source_url="https://github.com/community/social-scheduler",
                documentation_url="https://github.com/community/social-scheduler/wiki",
                published_at=datetime.now(timezone.utc)
            ),
            AgentMarketplaceEntry(
                agent_id=UUID("00000000-0000-0000-0000-000000000005"),
                agent_name="TrendAnalyzer",
                agent_type="analyzer",
                version="1.0.0",
                author="Community",
                description="Real-time trend analyzer for social media. "
                           "Identifies trending topics and suggests timely content.",
                capabilities=["analyze", "trends", "research", "recommendations"],
                rating=4.5,
                download_count=720,
                price=0.0,
                source_url="https://github.com/community/trend-analyzer",
                published_at=datetime.now(timezone.utc)
            ),
        ]
        
        for agent in default_agents:
            self.marketplace_catalog[str(agent.agent_id)] = agent
    
    async def search_marketplace(
        self,
        query: Optional[str] = None,
        agent_type: Optional[AgentType] = None,
        capabilities: Optional[List[str]] = None,
        min_rating: float = 0.0,
        free_only: bool = False,
        sort_by: str = "rating",
        limit: int = 20
    ) -> List[AgentMarketplaceEntry]:
        """
        Search agent marketplace.
        
        Args:
            query: Text search query
            agent_type: Filter by agent type
            capabilities: Required capabilities
            min_rating: Minimum rating threshold
            free_only: Only show free agents
            sort_by: Sort field (rating, download_count, published_at)
            limit: Maximum results
            
        Returns:
            Matching marketplace entries
        """
        try:
            results = list(self.marketplace_catalog.values())
            
            # Apply filters
            if query:
                query_lower = query.lower()
                results = [
                    agent for agent in results
                    if (query_lower in agent.agent_name.lower() or
                        query_lower in agent.description.lower() or
                        any(query_lower in cap.lower() for cap in agent.capabilities))
                ]
            
            if agent_type:
                results = [
                    agent for agent in results
                    if agent.agent_type == agent_type.value
                ]
            
            if capabilities:
                capability_set = set(c.lower() for c in capabilities)
                results = [
                    agent for agent in results
                    if capability_set.issubset(set(c.lower() for c in agent.capabilities))
                ]
            
            if min_rating > 0:
                results = [
                    agent for agent in results
                    if agent.rating >= min_rating
                ]
            
            if free_only:
                results = [
                    agent for agent in results
                    if agent.price == 0.0
                ]
            
            # Sort results
            if sort_by == "rating":
                results.sort(key=lambda x: x.rating, reverse=True)
            elif sort_by == "download_count":
                results.sort(key=lambda x: x.download_count, reverse=True)
            elif sort_by == "published_at":
                results.sort(key=lambda x: x.published_at, reverse=True)
            
            return results[:limit]
            
        except Exception as e:
            logger.error(f"Marketplace search failed: {e}")
            return []
    
    async def get_agent_details(
        self,
        agent_id: UUID
    ) -> Optional[AgentMarketplaceEntry]:
        """
        Get detailed information about a marketplace agent.
        
        Args:
            agent_id: Agent identifier
            
        Returns:
            Agent details or None
        """
        return self.marketplace_catalog.get(str(agent_id))
    
    async def install_agent(
        self,
        agent_id: UUID,
        custom_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Install an agent from the marketplace.
        
        Args:
            agent_id: Marketplace agent ID
            custom_name: Optional custom name for installed instance
            
        Returns:
            Installation result
        """
        try:
            # Get agent from marketplace
            marketplace_entry = self.marketplace_catalog.get(str(agent_id))
            if not marketplace_entry:
                return {
                    "success": False,
                    "error": "Agent not found in marketplace"
                }
            
            # Check if already installed
            agent_name = custom_name or marketplace_entry.agent_name
            stmt = select(AgentModel).where(AgentModel.agent_name == agent_name)
            result = await self.db.execute(stmt)
            existing = result.scalar_one_or_none()
            
            if existing:
                return {
                    "success": False,
                    "error": f"Agent {agent_name} already installed"
                }
            
            # Create agent from marketplace entry
            agent_data = AgentCreate(
                agent_name=agent_name,
                agent_type=AgentType(marketplace_entry.agent_type),
                version=marketplace_entry.version,
                description=marketplace_entry.description,
                capabilities=marketplace_entry.capabilities,
                is_plugin=True,
                plugin_source=marketplace_entry.source_url,
                plugin_author=marketplace_entry.author
            )
            
            # Register agent
            installed_agent = await self.multi_agent_service.register_agent(agent_data)
            
            # Update download count
            marketplace_entry.download_count += 1
            
            logger.info(f"Installed agent {agent_name} from marketplace")
            
            return {
                "success": True,
                "agent_id": str(installed_agent.id),
                "agent_name": installed_agent.agent_name,
                "version": installed_agent.version,
                "message": f"Successfully installed {agent_name}"
            }
            
        except Exception as e:
            logger.error(f"Agent installation failed: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def uninstall_agent(
        self,
        agent_id: UUID
    ) -> Dict[str, Any]:
        """
        Uninstall an agent.
        
        Args:
            agent_id: Installed agent ID
            
        Returns:
            Uninstallation result
        """
        try:
            # Get agent
            stmt = select(AgentModel).where(AgentModel.id == agent_id)
            result = await self.db.execute(stmt)
            agent = result.scalar_one_or_none()
            
            if not agent:
                return {
                    "success": False,
                    "error": "Agent not found"
                }
            
            if not agent.is_plugin:
                return {
                    "success": False,
                    "error": "Cannot uninstall built-in agent"
                }
            
            # Delete agent
            await self.db.delete(agent)
            await self.db.commit()
            
            logger.info(f"Uninstalled agent {agent.agent_name}")
            
            return {
                "success": True,
                "message": f"Successfully uninstalled {agent.agent_name}"
            }
            
        except Exception as e:
            logger.error(f"Agent uninstallation failed: {e}")
            await self.db.rollback()
            return {
                "success": False,
                "error": str(e)
            }
    
    async def check_updates(
        self,
        agent_id: UUID
    ) -> Dict[str, Any]:
        """
        Check if updates are available for an installed agent.
        
        Args:
            agent_id: Installed agent ID
            
        Returns:
            Update information
        """
        try:
            # Get installed agent
            stmt = select(AgentModel).where(AgentModel.id == agent_id)
            result = await self.db.execute(stmt)
            agent = result.scalar_one_or_none()
            
            if not agent or not agent.is_plugin:
                return {
                    "updates_available": False,
                    "message": "Agent not found or not a plugin"
                }
            
            # Find matching marketplace entry
            marketplace_agent = None
            for entry in self.marketplace_catalog.values():
                if (entry.agent_name == agent.agent_name and
                    entry.agent_type == agent.agent_type):
                    marketplace_agent = entry
                    break
            
            if not marketplace_agent:
                return {
                    "updates_available": False,
                    "message": "Agent not found in marketplace"
                }
            
            # Compare versions
            installed_version = self._parse_version(agent.version)
            latest_version = self._parse_version(marketplace_agent.version)
            
            updates_available = latest_version > installed_version
            
            return {
                "updates_available": updates_available,
                "installed_version": agent.version,
                "latest_version": marketplace_agent.version,
                "changelog": (
                    f"Update to {marketplace_agent.version} available"
                    if updates_available else "Up to date"
                )
            }
            
        except Exception as e:
            logger.error(f"Update check failed: {e}")
            return {
                "updates_available": False,
                "error": str(e)
            }
    
    def _parse_version(self, version_str: str) -> tuple:
        """Parse semantic version string."""
        try:
            parts = version_str.split(".")
            return tuple(int(p) for p in parts)
        except:
            return (0, 0, 0)
    
    async def get_installed_agents(self) -> List[Dict[str, Any]]:
        """
        Get list of all installed agents.
        
        Returns:
            List of installed agents with plugin status
        """
        try:
            agents = await self.multi_agent_service.list_agents()
            
            installed = []
            for agent in agents:
                # Check for updates
                update_info = await self.check_updates(agent.id)
                
                installed.append({
                    "agent_id": str(agent.id),
                    "agent_name": agent.agent_name,
                    "agent_type": agent.agent_type,
                    "version": agent.version,
                    "is_plugin": agent.is_plugin,
                    "author": agent.plugin_author if agent.is_plugin else "Built-in",
                    "status": agent.status,
                    "updates_available": update_info.get("updates_available", False),
                    "latest_version": update_info.get("latest_version"),
                    "success_rate": agent.success_rate,
                    "tasks_completed": agent.tasks_completed
                })
            
            return installed
            
        except Exception as e:
            logger.error(f"Get installed agents failed: {e}")
            return []
    
    async def publish_agent(
        self,
        agent_entry: AgentMarketplaceEntry
    ) -> Dict[str, Any]:
        """
        Publish a new agent to the marketplace.
        
        Args:
            agent_entry: Marketplace entry for the agent
            
        Returns:
            Publication result
        """
        try:
            # Validate agent entry
            if str(agent_entry.agent_id) in self.marketplace_catalog:
                return {
                    "success": False,
                    "error": "Agent ID already exists in marketplace"
                }
            
            # Add to catalog
            self.marketplace_catalog[str(agent_entry.agent_id)] = agent_entry
            
            logger.info(f"Published agent {agent_entry.agent_name} to marketplace")
            
            return {
                "success": True,
                "agent_id": str(agent_entry.agent_id),
                "message": f"Successfully published {agent_entry.agent_name}"
            }
            
        except Exception as e:
            logger.error(f"Agent publication failed: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def get_marketplace_stats(self) -> Dict[str, Any]:
        """
        Get marketplace statistics.
        
        Returns:
            Marketplace metrics
        """
        try:
            total_agents = len(self.marketplace_catalog)
            
            # Calculate stats
            type_distribution = {}
            avg_rating = 0.0
            total_downloads = 0
            free_agents = 0
            
            for agent in self.marketplace_catalog.values():
                agent_type = agent.agent_type
                type_distribution[agent_type] = type_distribution.get(agent_type, 0) + 1
                avg_rating += agent.rating
                total_downloads += agent.download_count
                if agent.price == 0.0:
                    free_agents += 1
            
            avg_rating = avg_rating / total_agents if total_agents > 0 else 0.0
            
            # Get top agents
            top_rated = sorted(
                self.marketplace_catalog.values(),
                key=lambda x: x.rating,
                reverse=True
            )[:5]
            
            top_downloaded = sorted(
                self.marketplace_catalog.values(),
                key=lambda x: x.download_count,
                reverse=True
            )[:5]
            
            return {
                "total_agents": total_agents,
                "type_distribution": type_distribution,
                "average_rating": float(avg_rating),
                "total_downloads": total_downloads,
                "free_agents": free_agents,
                "paid_agents": total_agents - free_agents,
                "top_rated": [
                    {
                        "name": agent.agent_name,
                        "rating": agent.rating,
                        "downloads": agent.download_count
                    }
                    for agent in top_rated
                ],
                "most_downloaded": [
                    {
                        "name": agent.agent_name,
                        "downloads": agent.download_count,
                        "rating": agent.rating
                    }
                    for agent in top_downloaded
                ]
            }
            
        except Exception as e:
            logger.error(f"Marketplace stats failed: {e}")
            return {}
