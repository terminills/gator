"""
Platform Policy Service

Service for managing platform-specific content policies.
"""

from typing import Dict, List, Optional
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from backend.models.platform_policy import (
    PlatformPolicyModel,
    PlatformPolicyCreate,
    PlatformPolicyUpdate,
    PlatformPolicyResponse,
    DEFAULT_PLATFORM_POLICIES,
)
from backend.models.content import ContentRating
from backend.config.logging import get_logger

logger = get_logger(__name__)


class PlatformPolicyService:
    """Service for managing platform content policies."""
    
    def __init__(self, db_session: AsyncSession):
        self.db = db_session
        self._policy_cache: Dict[str, PlatformPolicyModel] = {}
        self._cache_loaded = False
    
    async def _load_cache(self) -> None:
        """Load all active platform policies into cache."""
        if self._cache_loaded:
            return
        
        stmt = select(PlatformPolicyModel).where(PlatformPolicyModel.is_active == True)
        result = await self.db.execute(stmt)
        policies = result.scalars().all()
        
        self._policy_cache = {
            policy.platform_name.lower(): policy
            for policy in policies
        }
        self._cache_loaded = True
        logger.info(f"Loaded {len(self._policy_cache)} platform policies into cache")
    
    async def get_platform_policy(
        self, platform_name: str
    ) -> Optional[PlatformPolicyResponse]:
        """Get policy for a specific platform."""
        await self._load_cache()
        
        policy = self._policy_cache.get(platform_name.lower())
        if policy:
            return PlatformPolicyResponse.model_validate(policy)
        return None
    
    async def list_all_policies(
        self, active_only: bool = True
    ) -> List[PlatformPolicyResponse]:
        """List all platform policies."""
        stmt = select(PlatformPolicyModel)
        if active_only:
            stmt = stmt.where(PlatformPolicyModel.is_active == True)
        
        result = await self.db.execute(stmt)
        policies = result.scalars().all()
        
        return [PlatformPolicyResponse.model_validate(p) for p in policies]
    
    async def create_platform_policy(
        self, policy_data: PlatformPolicyCreate
    ) -> PlatformPolicyResponse:
        """Create a new platform policy."""
        # Check if platform already exists
        stmt = select(PlatformPolicyModel).where(
            PlatformPolicyModel.platform_name == policy_data.platform_name.lower()
        )
        result = await self.db.execute(stmt)
        existing = result.scalar_one_or_none()
        
        if existing:
            raise ValueError(f"Platform policy for '{policy_data.platform_name}' already exists")
        
        # Create new policy
        policy = PlatformPolicyModel(
            platform_name=policy_data.platform_name.lower(),
            platform_display_name=policy_data.platform_display_name,
            platform_url=policy_data.platform_url,
            allowed_content_ratings=policy_data.allowed_content_ratings,
            requires_content_warning=policy_data.requires_content_warning,
            requires_age_verification=policy_data.requires_age_verification,
            min_age_requirement=policy_data.min_age_requirement,
            policy_description=policy_data.policy_description,
            policy_url=policy_data.policy_url,
        )
        
        self.db.add(policy)
        await self.db.commit()
        await self.db.refresh(policy)
        
        # Clear cache to force reload
        self._cache_loaded = False
        
        logger.info(f"Created platform policy for {policy.platform_name}")
        return PlatformPolicyResponse.model_validate(policy)
    
    async def update_platform_policy(
        self, platform_name: str, updates: PlatformPolicyUpdate
    ) -> Optional[PlatformPolicyResponse]:
        """Update an existing platform policy."""
        stmt = select(PlatformPolicyModel).where(
            PlatformPolicyModel.platform_name == platform_name.lower()
        )
        result = await self.db.execute(stmt)
        policy = result.scalar_one_or_none()
        
        if not policy:
            return None
        
        # Apply updates
        update_data = updates.model_dump(exclude_unset=True)
        for field, value in update_data.items():
            setattr(policy, field, value)
        
        await self.db.commit()
        await self.db.refresh(policy)
        
        # Clear cache to force reload
        self._cache_loaded = False
        
        logger.info(f"Updated platform policy for {policy.platform_name}")
        return PlatformPolicyResponse.model_validate(policy)
    
    async def delete_platform_policy(self, platform_name: str) -> bool:
        """Soft delete a platform policy."""
        stmt = select(PlatformPolicyModel).where(
            PlatformPolicyModel.platform_name == platform_name.lower()
        )
        result = await self.db.execute(stmt)
        policy = result.scalar_one_or_none()
        
        if not policy:
            return False
        
        policy.is_active = False
        await self.db.commit()
        
        # Clear cache to force reload
        self._cache_loaded = False
        
        logger.info(f"Deactivated platform policy for {policy.platform_name}")
        return True
    
    async def check_content_allowed(
        self, platform_name: str, content_rating: ContentRating
    ) -> bool:
        """
        Check if content rating is allowed on platform.
        
        Args:
            platform_name: Name of the platform
            content_rating: Content rating to check
        
        Returns:
            True if allowed, False otherwise
        """
        await self._load_cache()
        
        policy = self._policy_cache.get(platform_name.lower())
        if not policy:
            # Default to safe-for-work only if platform unknown
            logger.warning(
                f"No policy found for platform '{platform_name}', "
                f"defaulting to SFW-only"
            )
            return content_rating == ContentRating.SFW
        
        return content_rating.value in [
            r.lower() for r in policy.allowed_content_ratings
        ]
    
    async def initialize_default_policies(self) -> int:
        """
        Initialize database with default platform policies.
        Only adds policies that don't already exist.
        
        Returns:
            Number of policies created
        """
        created_count = 0
        
        for policy_data in DEFAULT_PLATFORM_POLICIES:
            # Check if policy already exists
            stmt = select(PlatformPolicyModel).where(
                PlatformPolicyModel.platform_name == policy_data["platform_name"]
            )
            result = await self.db.execute(stmt)
            existing = result.scalar_one_or_none()
            
            if existing:
                logger.debug(
                    f"Platform policy for '{policy_data['platform_name']}' already exists, skipping"
                )
                continue
            
            # Create policy
            policy = PlatformPolicyModel(**policy_data)
            self.db.add(policy)
            created_count += 1
            logger.info(f"Created default policy for {policy_data['platform_name']}")
        
        if created_count > 0:
            await self.db.commit()
            # Clear cache to force reload
            self._cache_loaded = False
        
        logger.info(f"Initialized {created_count} default platform policies")
        return created_count
