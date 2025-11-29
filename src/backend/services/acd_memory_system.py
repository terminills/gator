"""
ACD Memory System

Hierarchical memory system for ACD with:
- Working Memory (current task context)
- Short-Term Memory (recent generations, 24h)
- Long-Term Memory (persistent patterns)
- Episodic Memory (specific memorable outcomes)
"""

from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional
from uuid import UUID, uuid4

from sqlalchemy import and_, func, or_, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from backend.config.logging import get_logger
from backend.models.acd import (
    ACDContextModel,
    MemoryCreate,
    MemoryRecallRequest,
    MemoryResponse,
    MemoryType,
)

logger = get_logger(__name__)


class ACDMemorySystem:
    """
    Hierarchical memory system for ACD with:
    - Working Memory (current task context)
    - Short-Term Memory (recent generations, 24h)
    - Long-Term Memory (persistent patterns)
    - Episodic Memory (specific memorable outcomes)

    Key capabilities:
    - Store memories with importance weighting
    - Recall relevant memories using semantic search
    - Consolidate short-term to long-term memory
    - Track memory access patterns
    """

    # Memory type time boundaries
    SHORT_TERM_HOURS = 24
    EPISODIC_MIN_IMPORTANCE = 0.7
    CONSOLIDATION_THRESHOLD = 5  # Access count before considering consolidation

    def __init__(self, db_session: AsyncSession):
        """
        Initialize the memory system.

        Args:
            db_session: Database session for persistence
        """
        self.db = db_session

    async def store_memory(
        self,
        memory_data: MemoryCreate,
    ) -> MemoryResponse:
        """
        Store a memory with importance weighting.

        Args:
            memory_data: Memory data to store

        Returns:
            Stored memory response
        """
        try:
            # Create or update context for memory storage
            if memory_data.context_id:
                # Update existing context with memory
                stmt = select(ACDContextModel).where(
                    ACDContextModel.id == memory_data.context_id
                )
                result = await self.db.execute(stmt)
                context = result.scalar_one_or_none()

                if context:
                    context.memory_type = memory_data.memory_type.value
                    context.memory_importance = memory_data.importance
                    context.ai_context = memory_data.content
                    context.memory_consolidated = False

                    await self.db.commit()
                    await self.db.refresh(context)

                    return MemoryResponse(
                        id=context.id,
                        memory_type=context.memory_type,
                        content=context.ai_context or {},
                        importance=context.memory_importance or 0.5,
                        access_count=context.memory_access_count or 0,
                        created_at=context.created_at,
                        last_recalled_at=context.last_recalled_at,
                    )

            # Create new memory context
            context = ACDContextModel(
                ai_phase="MEMORY_STORAGE",
                ai_status="IMPLEMENTED",
                ai_state="DONE",
                memory_type=memory_data.memory_type.value,
                memory_importance=memory_data.importance,
                ai_context=memory_data.content,
                memory_consolidated=False,
                memory_access_count=0,
            )

            self.db.add(context)
            await self.db.commit()
            await self.db.refresh(context)

            logger.info(
                f"Stored memory {context.id}: type={memory_data.memory_type.value}, "
                f"importance={memory_data.importance:.2f}"
            )

            return MemoryResponse(
                id=context.id,
                memory_type=context.memory_type,
                content=context.ai_context or {},
                importance=context.memory_importance or 0.5,
                access_count=context.memory_access_count or 0,
                created_at=context.created_at,
                last_recalled_at=context.last_recalled_at,
            )

        except Exception as e:
            logger.error(f"Failed to store memory: {e}")
            await self.db.rollback()
            raise

    async def recall(
        self,
        recall_request: MemoryRecallRequest,
    ) -> List[MemoryResponse]:
        """
        Recall relevant memories using semantic search.

        Prioritizes:
        - Recency (recent memories weighted higher)
        - Importance (marked important memories)
        - Relevance (semantic similarity to query)

        Args:
            recall_request: Recall request with query and filters

        Returns:
            List of relevant memories
        """
        try:
            # Build query conditions
            conditions = []

            # Filter by memory types if specified
            if recall_request.memory_types:
                type_conditions = [
                    ACDContextModel.memory_type == mt.value
                    for mt in recall_request.memory_types
                ]
                conditions.append(or_(*type_conditions))

            # Filter by minimum importance if specified
            if recall_request.min_importance is not None:
                conditions.append(
                    ACDContextModel.memory_importance >= recall_request.min_importance
                )

            # Only include contexts with memory content
            conditions.append(ACDContextModel.ai_context.isnot(None))

            stmt = select(ACDContextModel)
            if conditions:
                stmt = stmt.where(and_(*conditions))

            # Order by importance and recency
            stmt = stmt.order_by(
                ACDContextModel.memory_importance.desc(),
                ACDContextModel.created_at.desc(),
            ).limit(recall_request.max_results * 3)  # Get more for filtering

            result = await self.db.execute(stmt)
            candidates = result.scalars().all()

            # Calculate relevance scores
            scored_memories = []
            query_lower = recall_request.query.lower()
            query_words = set(query_lower.split())

            for ctx in candidates:
                if not ctx.ai_context:
                    continue

                # Simple keyword-based relevance scoring
                relevance = self._calculate_relevance(
                    query_words, ctx.ai_context, ctx
                )

                if relevance > 0:
                    scored_memories.append((ctx, relevance))

            # Sort by combined score (relevance + importance + recency)
            now = datetime.now(timezone.utc)
            scored_memories.sort(
                key=lambda x: self._combined_score(x[0], x[1], now),
                reverse=True,
            )

            # Take top results and update access counts
            results = []
            for ctx, relevance in scored_memories[:recall_request.max_results]:
                # Update access tracking
                ctx.memory_access_count = (ctx.memory_access_count or 0) + 1
                ctx.last_recalled_at = now

                results.append(
                    MemoryResponse(
                        id=ctx.id,
                        memory_type=ctx.memory_type or "WORKING",
                        content=ctx.ai_context or {},
                        importance=ctx.memory_importance or 0.5,
                        access_count=ctx.memory_access_count,
                        created_at=ctx.created_at,
                        last_recalled_at=ctx.last_recalled_at,
                        relevance_score=relevance,
                    )
                )

            await self.db.commit()

            logger.info(
                f"Recalled {len(results)} memories for query: {recall_request.query[:50]}..."
            )

            return results

        except Exception as e:
            logger.error(f"Failed to recall memories: {e}")
            return []

    def _calculate_relevance(
        self,
        query_words: set,
        context: Dict[str, Any],
        ctx: ACDContextModel,
    ) -> float:
        """
        Calculate relevance score based on keyword matching.

        Args:
            query_words: Set of query words
            context: Memory content dictionary
            ctx: The context model

        Returns:
            Relevance score between 0 and 1
        """
        score = 0.0

        # Convert context to searchable text
        searchable_text = ""

        # Add context content
        if isinstance(context, dict):
            for key, value in context.items():
                if isinstance(value, str):
                    searchable_text += f" {value}"
                elif isinstance(value, list):
                    searchable_text += f" {' '.join(str(v) for v in value)}"

        # Add other relevant fields
        if ctx.ai_phase:
            searchable_text += f" {ctx.ai_phase}"
        if ctx.ai_domain:
            searchable_text += f" {ctx.ai_domain}"
        if ctx.ai_subdomain:
            searchable_text += f" {ctx.ai_subdomain}"
        if ctx.ai_note:
            searchable_text += f" {ctx.ai_note}"

        searchable_lower = searchable_text.lower()
        searchable_words = set(searchable_lower.split())

        # Calculate word overlap
        if query_words and searchable_words:
            overlap = len(query_words & searchable_words)
            score = overlap / len(query_words)

        return score

    def _combined_score(
        self,
        ctx: ACDContextModel,
        relevance: float,
        now: datetime,
    ) -> float:
        """
        Calculate combined score for ranking.

        Args:
            ctx: Context model
            relevance: Relevance score
            now: Current timestamp

        Returns:
            Combined ranking score
        """
        # Weights
        relevance_weight = 0.4
        importance_weight = 0.3
        recency_weight = 0.2
        access_weight = 0.1

        # Relevance component
        score = relevance_weight * relevance

        # Importance component
        importance = ctx.memory_importance or 0.5
        score += importance_weight * importance

        # Recency component (decay over time)
        if ctx.created_at:
            age_hours = (now - ctx.created_at).total_seconds() / 3600
            recency = max(0, 1 - (age_hours / (24 * 7)))  # Decay over 1 week
            score += recency_weight * recency

        # Access frequency component (frequently accessed = more relevant)
        access_count = ctx.memory_access_count or 0
        access_score = min(1.0, access_count / 10)  # Cap at 10 accesses
        score += access_weight * access_score

        return score

    async def consolidate(
        self,
        batch_size: int = 100,
    ) -> Dict[str, Any]:
        """
        Periodically consolidate short-term to long-term memory.

        Identifies:
        - Repeated patterns → Generalized knowledge
        - High-importance events → Preserved episodes
        - Outdated memories → Pruned

        Args:
            batch_size: Maximum memories to process per consolidation

        Returns:
            Consolidation results
        """
        try:
            now = datetime.now(timezone.utc)
            short_term_cutoff = now - timedelta(hours=self.SHORT_TERM_HOURS)

            # Find candidates for consolidation
            # 1. Short-term memories older than threshold
            # 2. Memories with high access count
            # 3. Memories with high importance
            stmt = (
                select(ACDContextModel)
                .where(
                    and_(
                        ACDContextModel.memory_consolidated.is_(False),
                        or_(
                            # Old short-term memories
                            and_(
                                ACDContextModel.memory_type == MemoryType.SHORT_TERM.value,
                                ACDContextModel.created_at < short_term_cutoff,
                            ),
                            # Frequently accessed memories
                            ACDContextModel.memory_access_count >= self.CONSOLIDATION_THRESHOLD,
                            # High importance memories
                            ACDContextModel.memory_importance >= self.EPISODIC_MIN_IMPORTANCE,
                        ),
                    )
                )
                .order_by(ACDContextModel.memory_importance.desc())
                .limit(batch_size)
            )

            result = await self.db.execute(stmt)
            candidates = result.scalars().all()

            consolidated = 0
            promoted_to_long_term = 0
            promoted_to_episodic = 0
            pruned = 0

            for ctx in candidates:
                # Determine target memory type
                if ctx.memory_importance >= self.EPISODIC_MIN_IMPORTANCE:
                    # High importance → Episodic memory
                    ctx.memory_type = MemoryType.EPISODIC.value
                    promoted_to_episodic += 1
                elif ctx.memory_access_count >= self.CONSOLIDATION_THRESHOLD:
                    # Frequently accessed → Long-term memory
                    ctx.memory_type = MemoryType.LONG_TERM.value
                    promoted_to_long_term += 1
                elif ctx.memory_importance < 0.3 and ctx.outcome_score is not None:
                    if ctx.outcome_score < 0.3:
                        # Low importance + low outcome → Consider pruning
                        # Don't actually delete, just mark as consolidated
                        pruned += 1

                ctx.memory_consolidated = True
                consolidated += 1

            await self.db.commit()

            results = {
                "total_processed": consolidated,
                "promoted_to_long_term": promoted_to_long_term,
                "promoted_to_episodic": promoted_to_episodic,
                "pruned": pruned,
                "timestamp": now.isoformat(),
            }

            logger.info(
                f"Memory consolidation complete: processed={consolidated}, "
                f"long_term={promoted_to_long_term}, episodic={promoted_to_episodic}"
            )

            return results

        except Exception as e:
            logger.error(f"Failed to consolidate memories: {e}")
            await self.db.rollback()
            return {"error": str(e)}

    async def get_memory_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the memory system.

        Returns:
            Memory system statistics
        """
        try:
            # Count by memory type
            type_counts = {}
            for mt in MemoryType:
                stmt = select(func.count()).where(
                    ACDContextModel.memory_type == mt.value
                )
                result = await self.db.execute(stmt)
                type_counts[mt.value] = result.scalar() or 0

            # Total memories
            total_stmt = select(func.count()).where(
                ACDContextModel.ai_context.isnot(None)
            )
            total_result = await self.db.execute(total_stmt)
            total = total_result.scalar() or 0

            # Average importance
            avg_stmt = select(func.avg(ACDContextModel.memory_importance)).where(
                ACDContextModel.memory_importance.isnot(None)
            )
            avg_result = await self.db.execute(avg_stmt)
            avg_importance = avg_result.scalar()

            # Consolidated count
            consolidated_stmt = select(func.count()).where(
                ACDContextModel.memory_consolidated.is_(True)
            )
            consolidated_result = await self.db.execute(consolidated_stmt)
            consolidated = consolidated_result.scalar() or 0

            # Frequently accessed (access_count >= 5)
            frequent_stmt = select(func.count()).where(
                ACDContextModel.memory_access_count >= 5
            )
            frequent_result = await self.db.execute(frequent_stmt)
            frequently_accessed = frequent_result.scalar() or 0

            return {
                "total_memories": total,
                "by_type": type_counts,
                "consolidated": consolidated,
                "pending_consolidation": total - consolidated,
                "frequently_accessed": frequently_accessed,
                "average_importance": round(avg_importance, 3) if avg_importance else None,
            }

        except Exception as e:
            logger.error(f"Failed to get memory stats: {e}")
            return {"error": str(e)}

    async def forget(
        self,
        memory_id: UUID,
    ) -> bool:
        """
        Mark a memory as forgotten (reduce importance to 0).

        Args:
            memory_id: UUID of the memory to forget

        Returns:
            True if successful
        """
        try:
            stmt = (
                update(ACDContextModel)
                .where(ACDContextModel.id == memory_id)
                .values(
                    memory_importance=0.0,
                    memory_consolidated=True,
                )
            )
            await self.db.execute(stmt)
            await self.db.commit()

            logger.info(f"Forgot memory {memory_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to forget memory: {e}")
            await self.db.rollback()
            return False

    async def reinforce(
        self,
        memory_id: UUID,
        boost: float = 0.1,
    ) -> Optional[MemoryResponse]:
        """
        Reinforce a memory by increasing its importance.

        Args:
            memory_id: UUID of the memory to reinforce
            boost: Amount to increase importance (default 0.1)

        Returns:
            Updated memory if successful
        """
        try:
            stmt = select(ACDContextModel).where(ACDContextModel.id == memory_id)
            result = await self.db.execute(stmt)
            ctx = result.scalar_one_or_none()

            if not ctx:
                return None

            # Increase importance (cap at 1.0)
            current = ctx.memory_importance or 0.5
            ctx.memory_importance = min(1.0, current + boost)
            ctx.memory_access_count = (ctx.memory_access_count or 0) + 1
            ctx.last_recalled_at = datetime.now(timezone.utc)

            await self.db.commit()
            await self.db.refresh(ctx)

            logger.info(
                f"Reinforced memory {memory_id}: importance={ctx.memory_importance:.2f}"
            )

            return MemoryResponse(
                id=ctx.id,
                memory_type=ctx.memory_type or "WORKING",
                content=ctx.ai_context or {},
                importance=ctx.memory_importance or 0.5,
                access_count=ctx.memory_access_count or 0,
                created_at=ctx.created_at,
                last_recalled_at=ctx.last_recalled_at,
            )

        except Exception as e:
            logger.error(f"Failed to reinforce memory: {e}")
            await self.db.rollback()
            return None
