"""
Reasoning Orchestrator Service - The Basal Ganglia of ACD

This service implements the central decision-making and coordination system for ACD,
functioning like the basal ganglia in the human brain:

1. Action Selection: Dynamically chooses which agent/action to execute
2. Pattern Learning: Learns from successful and failed handoffs
3. Motor Control Analogy: Coordinates multiple specialized agents
4. Habit Formation: Builds procedural knowledge from repeated successes

The orchestrator transforms ACD from a static system into a dynamic, self-organizing
system that improves with every decision.
"""

import asyncio
from typing import Dict, List, Optional, Any, Tuple
from uuid import UUID
from datetime import datetime, timezone, timedelta
from enum import Enum

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_, func, or_

from backend.models.acd import (
    ACDContextModel,
    ACDContextCreate,
    ACDContextUpdate,
    ACDContextResponse,
    AIStatus,
    AIState,
    AIComplexity,
    AIConfidence,
    AIRequest,
    AIValidation,
    HandoffType,
    HandoffStatus,
    SkillLevel,
)
from backend.services.acd_service import ACDService
from backend.services.multi_agent_service import MultiAgentService
from backend.config.logging import get_logger

logger = get_logger(__name__)


class DecisionType(str, Enum):
    """Types of orchestration decisions."""
    
    EXECUTE_LOCALLY = "EXECUTE_LOCALLY"  # Handle in current agent
    HANDOFF_SPECIALIZATION = "HANDOFF_SPECIALIZATION"  # Route to specialist
    HANDOFF_ESCALATION = "HANDOFF_ESCALATION"  # Escalate due to complexity
    REQUEST_REVIEW = "REQUEST_REVIEW"  # Ask for validation
    REQUEST_COLLABORATION = "REQUEST_COLLABORATION"  # Multi-agent task
    RETRY_WITH_LEARNING = "RETRY_WITH_LEARNING"  # Retry with learned patterns
    DEFER_TO_HUMAN = "DEFER_TO_HUMAN"  # Human intervention needed


class DecisionConfidence(str, Enum):
    """Confidence levels for orchestration decisions."""
    
    VERY_HIGH = "VERY_HIGH"  # >90% confidence
    HIGH = "HIGH"  # 70-90%
    MEDIUM = "MEDIUM"  # 50-70%
    LOW = "LOW"  # 30-50%
    VERY_LOW = "VERY_LOW"  # <30%


class OrchestrationDecision:
    """
    Represents a reasoning decision made by the orchestrator.
    """
    
    def __init__(
        self,
        decision_type: DecisionType,
        confidence: DecisionConfidence,
        reasoning: str,
        target_agent: Optional[str] = None,
        action_plan: Optional[Dict[str, Any]] = None,
        learned_patterns: Optional[List[str]] = None,
        risk_assessment: Optional[str] = None,
    ):
        self.decision_type = decision_type
        self.confidence = confidence
        self.reasoning = reasoning
        self.target_agent = target_agent
        self.action_plan = action_plan or {}
        self.learned_patterns = learned_patterns or []
        self.risk_assessment = risk_assessment
        self.timestamp = datetime.now(timezone.utc)


class ReasoningOrchestrator:
    """
    Central reasoning and coordination service for ACD system.
    
    Implements basal ganglia-like functionality:
    - Action selection based on context
    - Pattern-based decision making
    - Dynamic agent coordination
    - Learning from outcomes
    """
    
    def __init__(self, db_session: AsyncSession):
        """
        Initialize reasoning orchestrator.
        
        Args:
            db_session: Database session for persistence
        """
        self.db = db_session
        self.acd_service = ACDService(db_session)
        self.multi_agent_service = MultiAgentService(db_session)
        
        # Decision-making thresholds
        self.complexity_escalation_threshold = AIComplexity.HIGH
        self.confidence_handoff_threshold = AIConfidence.UNCERTAIN
        self.retry_limit = 3
        
        # Pattern cache for quick lookups
        self._pattern_cache: Dict[str, List[Dict[str, Any]]] = {}
        self._cache_timeout = timedelta(hours=1)
        self._last_cache_update = datetime.now(timezone.utc)
    
    async def orchestrate_decision(
        self,
        context: ACDContextResponse,
        current_agent: Optional[str] = None,
        additional_context: Optional[Dict[str, Any]] = None,
    ) -> OrchestrationDecision:
        """
        Make a reasoning decision about how to handle an ACD context.
        
        This is the core "basal ganglia" function that evaluates context and
        decides on action selection.
        
        NOW USES THE REASONING ENGINE (THE BRAIN) FOR ALL DECISIONS.
        
        Args:
            context: Current ACD context
            current_agent: Name of agent requesting decision
            additional_context: Extra context for decision making
            
        Returns:
            OrchestrationDecision with recommended action
        """
        logger.info(
            f"🧠 Orchestrating decision for context {context.id}, "
            f"phase={context.ai_phase}, state={context.ai_state}"
        )
        
        try:
            # Import reasoning engine
            from backend.services.reasoning_engine import ReasoningEngine
            
            # Step 1: Assess current situation
            situation = await self._assess_situation(context, current_agent)
            
            # Step 2: Query learned patterns
            patterns = await self._query_relevant_patterns(context)
            
            # Step 3: USE THE REASONING ENGINE (THE BRAIN)
            logger.info("🧠 Calling reasoning engine for intelligent decision making...")
            engine = ReasoningEngine(self.db)
            
            decision_dict = await engine.reason_about_context(
                context=context,
                current_agent=current_agent,
                situation=situation,
                patterns=patterns,
            )
            
            # Step 4: Convert dict to OrchestrationDecision
            decision = self._create_decision_from_dict(decision_dict)
            
            # Step 5: Decompose into subtasks if needed
            if decision.decision_type in [DecisionType.REQUEST_COLLABORATION, DecisionType.HANDOFF_ESCALATION]:
                logger.info(f"🔧 Decomposing task for {decision.decision_type.value}...")
                subtasks = await engine.decompose_task(context, decision_dict)
                if subtasks:
                    decision.action_plan["subtasks"] = subtasks
                    logger.info(f"✅ Task decomposed into {len(subtasks)} subtasks")
            
            # Step 6: Log decision for learning
            await self._log_decision(context.id, decision)
            
            logger.info(
                f"✅ Decision made by reasoning engine: {decision.decision_type.value} "
                f"(confidence={decision.confidence.value})"
            )
            
            return decision
            
        except Exception as e:
            logger.error(f"❌ Reasoning engine failed: {e}")
            logger.info("🔧 Using fallback reasoning...")
            
            # Fallback to simple rule-based reasoning
            return await self._fallback_orchestration(context, current_agent)
    
    def _create_decision_from_dict(self, decision_dict: Dict[str, Any]) -> OrchestrationDecision:
        """
        Create OrchestrationDecision from reasoning engine output.
        
        Args:
            decision_dict: Decision dictionary from reasoning engine
            
        Returns:
            OrchestrationDecision object
        """
        return OrchestrationDecision(
            decision_type=DecisionType(decision_dict["decision_type"]),
            confidence=DecisionConfidence(decision_dict["confidence"]),
            reasoning=decision_dict["reasoning"],
            target_agent=decision_dict.get("target_agent"),
            action_plan=decision_dict.get("action_plan", {}),
            learned_patterns=decision_dict.get("learned_patterns", []),
            risk_assessment=decision_dict.get("risk_assessment"),
        )
    
    async def _fallback_orchestration(
        self,
        context: ACDContextResponse,
        current_agent: Optional[str] = None,
    ) -> OrchestrationDecision:
        """
        Fallback orchestration when reasoning engine is unavailable.
        
        This is the safety net - uses simple rules.
        """
        logger.info("🔧 Using fallback orchestration (simple rules)")
        
        # Simple rule: if FAILED, request review; otherwise execute locally
        if context.ai_state == AIState.FAILED.value:
            return OrchestrationDecision(
                decision_type=DecisionType.REQUEST_REVIEW,
                confidence=DecisionConfidence.MEDIUM,
                reasoning="Task failed - requesting review (fallback reasoning)",
                target_agent=None,
                action_plan={"request_type": "REQUEST_REVIEW"},
                learned_patterns=[],
                risk_assessment="Medium - fallback reasoning used",
            )
        else:
            return OrchestrationDecision(
                decision_type=DecisionType.EXECUTE_LOCALLY,
                confidence=DecisionConfidence.MEDIUM,
                reasoning="Task appears manageable - executing locally (fallback reasoning)",
                target_agent=current_agent,
                action_plan={"execute_agent": current_agent or "default"},
                learned_patterns=[],
                risk_assessment="Medium - fallback reasoning used",
            )
    
    async def _assess_situation(
        self,
        context: ACDContextResponse,
        current_agent: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Assess the current situation from multiple angles.
        
        Args:
            context: ACD context to assess
            current_agent: Current agent handling the task
            
        Returns:
            Situation assessment dictionary
        """
        # Check for errors or failures
        has_errors = bool(context.runtime_err or context.compiler_err)
        
        # Check handoff history
        handoff_count = 0
        if context.ai_assignment_history:
            handoff_count = len(context.ai_assignment_history)
        
        # Check retry attempts
        retry_count = 0
        if context.ai_metadata and "retry_count" in context.ai_metadata:
            retry_count = context.ai_metadata["retry_count"]
        
        # Check for blocked state
        is_blocked = context.ai_state == AIState.BLOCKED.value
        
        # Check time spent
        time_spent = None
        if context.ai_started:
            time_spent = (datetime.now(timezone.utc) - context.ai_started).total_seconds()
        
        return {
            "has_errors": has_errors,
            "handoff_count": handoff_count,
            "retry_count": retry_count,
            "is_blocked": is_blocked,
            "time_spent_seconds": time_spent,
            "current_agent": current_agent,
            "state": context.ai_state,
            "complexity": context.ai_complexity,
            "confidence": context.ai_confidence,
            "validation": context.ai_validation,
        }
    
    async def _query_relevant_patterns(
        self,
        context: ACDContextResponse,
    ) -> List[Dict[str, Any]]:
        """
        Query database for relevant successful patterns.
        
        Args:
            context: ACD context to find patterns for
            
        Returns:
            List of relevant successful patterns
        """
        try:
            # Check cache first
            cache_key = f"{context.ai_phase}_{context.ai_complexity}"
            if (
                cache_key in self._pattern_cache
                and (datetime.now(timezone.utc) - self._last_cache_update) < self._cache_timeout
            ):
                return self._pattern_cache[cache_key]
            
            # Query successful contexts with similar characteristics
            stmt = select(ACDContextModel).where(
                and_(
                    ACDContextModel.ai_phase == context.ai_phase,
                    ACDContextModel.ai_state == AIState.DONE.value,
                    ACDContextModel.ai_validation.in_([
                        AIValidation.APPROVED.value,
                        AIValidation.CONDITIONALLY_APPROVED.value,
                    ]),
                )
            )
            
            # Filter by similar complexity if available
            if context.ai_complexity:
                stmt = stmt.where(
                    ACDContextModel.ai_complexity == context.ai_complexity
                )
            
            # Limit to recent patterns (last 90 days)
            cutoff = datetime.now(timezone.utc) - timedelta(days=90)
            stmt = stmt.where(ACDContextModel.created_at >= cutoff)
            
            # Order by most recent first
            stmt = stmt.order_by(ACDContextModel.created_at.desc()).limit(20)
            
            result = await self.db.execute(stmt)
            contexts = result.scalars().all()
            
            # Extract patterns
            patterns = []
            for ctx in contexts:
                pattern = {
                    "phase": ctx.ai_phase,
                    "complexity": ctx.ai_complexity,
                    "assigned_to": ctx.ai_assigned_to,
                    "strategy": ctx.ai_strategy,
                    "pattern": ctx.ai_pattern,
                    "handoff_type": ctx.ai_handoff_type,
                    "success_rate": 1.0,  # All are successful
                }
                patterns.append(pattern)
            
            # Cache results
            self._pattern_cache[cache_key] = patterns
            self._last_cache_update = datetime.now(timezone.utc)
            
            return patterns
            
        except Exception as e:
            logger.error(f"Failed to query patterns: {e}")
            return []
    
    def _evaluate_complexity(
        self,
        context: ACDContextResponse,
        situation: Dict[str, Any],
    ) -> float:
        """
        Evaluate task complexity on a 0-1 scale.
        
        Args:
            context: ACD context
            situation: Current situation assessment
            
        Returns:
            Complexity score (0.0 = simple, 1.0 = very complex)
        """
        score = 0.0
        
        # Base complexity from context
        complexity_map = {
            "LOW": 0.2,
            "MEDIUM": 0.5,
            "HIGH": 0.8,
            "CRITICAL": 1.0,
        }
        if context.ai_complexity:
            score = complexity_map.get(context.ai_complexity, 0.5)
        
        # Increase if errors present
        if situation["has_errors"]:
            score += 0.15
        
        # Increase if blocked
        if situation["is_blocked"]:
            score += 0.1
        
        # Increase with handoff count (indicates difficulty)
        score += min(situation["handoff_count"] * 0.1, 0.2)
        
        # Increase with retry count
        score += min(situation["retry_count"] * 0.1, 0.2)
        
        return min(score, 1.0)
    
    def _evaluate_confidence(
        self,
        context: ACDContextResponse,
        patterns: List[Dict[str, Any]],
    ) -> float:
        """
        Evaluate confidence in handling task locally.
        
        Args:
            context: ACD context
            patterns: Learned patterns from similar tasks
            
        Returns:
            Confidence score (0.0 = no confidence, 1.0 = very confident)
        """
        # Start with base confidence from context
        confidence_map = {
            "VALIDATED": 1.0,
            "CONFIDENT": 0.8,
            "HYPOTHESIS": 0.5,
            "UNCERTAIN": 0.3,
            "EXPERIMENTAL": 0.4,
        }
        
        score = 0.5  # Default medium confidence
        if context.ai_confidence:
            score = confidence_map.get(context.ai_confidence, 0.5)
        
        # Increase confidence if we have successful patterns
        if patterns:
            pattern_boost = min(len(patterns) * 0.05, 0.3)
            score += pattern_boost
        
        # Decrease if validation shows issues
        if context.ai_validation == AIValidation.REJECTED.value:
            score *= 0.5
        
        if context.ai_validation == AIValidation.CONDITIONALLY_APPROVED.value:
            score *= 0.8
        
        return min(score, 1.0)
    
    async def _select_action(
        self,
        context: ACDContextResponse,
        situation: Dict[str, Any],
        patterns: List[Dict[str, Any]],
        complexity_score: float,
        confidence_score: float,
        current_agent: Optional[str] = None,
    ) -> OrchestrationDecision:
        """
        Core decision-making logic - selects the best action.
        
        This implements the "action selection" function of the basal ganglia.
        NOW USES THE ACTUAL REASONING ENGINE (THE BRAIN).
        
        Args:
            context: ACD context
            situation: Situation assessment
            patterns: Learned patterns
            complexity_score: Task complexity (0-1)
            confidence_score: Confidence in local handling (0-1)
            current_agent: Current agent
            
        Returns:
            OrchestrationDecision with selected action
        """
        # 🧠 USE THE REASONING ENGINE (THE BRAIN)
        try:
            from backend.services.reasoning_engine import ReasoningEngine
            
            engine = ReasoningEngine(self.db)
            
            # Call the actual reasoning brain
            decision_dict = await engine.reason_about_context(
                context=context,
                current_agent=current_agent,
                situation=situation,
                patterns=patterns,
            )
            
            # Convert dict to OrchestrationDecision
            decision = OrchestrationDecision(
                decision_type=DecisionType(decision_dict["decision_type"]),
                confidence=DecisionConfidence(decision_dict["confidence"]),
                reasoning=decision_dict["reasoning"],
                target_agent=decision_dict.get("target_agent"),
                action_plan=decision_dict.get("action_plan", {}),
                learned_patterns=decision_dict.get("learned_patterns", []),
                risk_assessment=decision_dict.get("risk_assessment"),
            )
            
            logger.info("🧠 Decision made by reasoning engine (THE BRAIN)")
            return decision
            
        except Exception as e:
            logger.error(f"Reasoning engine failed, using fallback: {e}")
            # Fallback to rule-based logic below
        
        # FALLBACK: Decision logic tree based on basal ganglia principles
        # This is the safety net when the reasoning engine is not available
        
        # Rule 1: If critical complexity and low confidence -> escalate
        if complexity_score >= 0.8 and confidence_score < 0.5:
            return OrchestrationDecision(
                decision_type=DecisionType.HANDOFF_ESCALATION,
                confidence=DecisionConfidence.HIGH,
                reasoning=(
                    f"High complexity ({complexity_score:.2f}) with low confidence "
                    f"({confidence_score:.2f}) requires escalation to expert agent"
                ),
                target_agent="expert_coordinator",
                action_plan={
                    "handoff_type": HandoffType.ESCALATION.value,
                    "skill_level": SkillLevel.EXPERT.value,
                },
            )
        
        # Rule 2: If retry limit exceeded -> defer to human
        if situation["retry_count"] >= self.retry_limit:
            return OrchestrationDecision(
                decision_type=DecisionType.DEFER_TO_HUMAN,
                confidence=DecisionConfidence.VERY_HIGH,
                reasoning=(
                    f"Retry limit ({self.retry_limit}) exceeded. "
                    "Task requires human intervention"
                ),
                action_plan={
                    "create_ticket": True,
                    "priority": "high",
                },
            )
        
        # Rule 3: If blocked and no progress -> request collaboration
        if situation["is_blocked"] and situation["time_spent_seconds"] and situation["time_spent_seconds"] > 300:
            return OrchestrationDecision(
                decision_type=DecisionType.REQUEST_COLLABORATION,
                confidence=DecisionConfidence.MEDIUM,
                reasoning=(
                    f"Task blocked for {situation['time_spent_seconds']:.0f}s. "
                    "Requesting collaborative problem-solving"
                ),
                action_plan={
                    "collaboration_type": "parallel_processing",
                    "num_agents": 2,
                },
            )
        
        # Rule 4: If patterns suggest specialist and confidence low -> handoff
        if patterns and confidence_score < 0.6:
            # Find most common successful agent from patterns
            agent_counts = {}
            for pattern in patterns:
                agent = pattern.get("assigned_to")
                if agent and agent != current_agent:
                    agent_counts[agent] = agent_counts.get(agent, 0) + 1
            
            if agent_counts:
                best_agent = max(agent_counts, key=agent_counts.get)
                pattern_confidence = agent_counts[best_agent] / len(patterns)
                
                return OrchestrationDecision(
                    decision_type=DecisionType.HANDOFF_SPECIALIZATION,
                    confidence=DecisionConfidence.HIGH if pattern_confidence > 0.6 else DecisionConfidence.MEDIUM,
                    reasoning=(
                        f"Learned patterns indicate agent '{best_agent}' has "
                        f"{pattern_confidence:.0%} success rate for similar tasks"
                    ),
                    target_agent=best_agent,
                    learned_patterns=[p.get("strategy", "") for p in patterns[:3]],
                    action_plan={
                        "handoff_type": HandoffType.SPECIALIZATION.value,
                        "pattern_confidence": pattern_confidence,
                    },
                )
        
        # Rule 5: If uncertain but manageable -> request review
        if context.ai_confidence == AIConfidence.UNCERTAIN.value and complexity_score < 0.7:
            return OrchestrationDecision(
                decision_type=DecisionType.REQUEST_REVIEW,
                confidence=DecisionConfidence.MEDIUM,
                reasoning=(
                    "Moderate complexity with uncertain confidence. "
                    "Requesting validation before proceeding"
                ),
                action_plan={
                    "request_type": AIRequest.REQUEST_REVIEW.value,
                    "reviewer_type": "quality_assurance",
                },
            )
        
        # Rule 6: If errors but patterns available -> retry with learning
        if situation["has_errors"] and patterns and situation["retry_count"] < self.retry_limit:
            return OrchestrationDecision(
                decision_type=DecisionType.RETRY_WITH_LEARNING,
                confidence=DecisionConfidence.MEDIUM,
                reasoning=(
                    f"Errors detected but {len(patterns)} successful patterns available. "
                    "Retrying with learned strategies"
                ),
                learned_patterns=[p.get("strategy", "") for p in patterns[:5]],
                action_plan={
                    "retry_count": situation["retry_count"] + 1,
                    "apply_patterns": True,
                    "strategies": [p.get("strategy") for p in patterns[:3] if p.get("strategy")],
                },
            )
        
        # Rule 7: Default - execute locally if confidence acceptable
        if confidence_score >= 0.5:
            conf_level = DecisionConfidence.HIGH if confidence_score >= 0.7 else DecisionConfidence.MEDIUM
            return OrchestrationDecision(
                decision_type=DecisionType.EXECUTE_LOCALLY,
                confidence=conf_level,
                reasoning=(
                    f"Confidence ({confidence_score:.2f}) and complexity "
                    f"({complexity_score:.2f}) within acceptable ranges for local execution"
                ),
                action_plan={
                    "execute_agent": current_agent or "default",
                    "apply_patterns": bool(patterns),
                },
                learned_patterns=[p.get("strategy", "") for p in patterns[:3]],
            )
        
        # Fallback: Request assistance if no clear path
        return OrchestrationDecision(
            decision_type=DecisionType.REQUEST_REVIEW,
            confidence=DecisionConfidence.LOW,
            reasoning=(
                f"No clear action path. Confidence={confidence_score:.2f}, "
                f"Complexity={complexity_score:.2f}. Requesting assistance"
            ),
            action_plan={
                "request_type": AIRequest.REQUEST_ASSISTANCE.value,
            },
        )
    
    async def _log_decision(
        self,
        context_id: UUID,
        decision: OrchestrationDecision,
    ) -> None:
        """
        Log orchestration decision for learning and audit.
        
        Args:
            context_id: ACD context ID
            decision: Decision made
        """
        try:
            # Update context metadata with decision
            update_data = ACDContextUpdate(
                ai_metadata={
                    "orchestration_decision": {
                        "type": decision.decision_type.value,
                        "confidence": decision.confidence.value,
                        "reasoning": decision.reasoning,
                        "timestamp": decision.timestamp.isoformat(),
                        "target_agent": decision.target_agent,
                        "action_plan": decision.action_plan,
                        "learned_patterns": decision.learned_patterns,
                    }
                }
            )
            
            await self.acd_service.update_context(context_id, update_data)
            
            logger.debug(f"Logged decision for context {context_id}")
            
        except Exception as e:
            logger.error(f"Failed to log decision: {e}")
    
    async def execute_decision(
        self,
        context: ACDContextResponse,
        decision: OrchestrationDecision,
    ) -> bool:
        """
        Execute the orchestration decision.
        
        Args:
            context: ACD context
            decision: Decision to execute
            
        Returns:
            True if execution initiated successfully
        """
        try:
            logger.info(
                f"Executing decision {decision.decision_type.value} "
                f"for context {context.id}"
            )
            
            if decision.decision_type == DecisionType.HANDOFF_SPECIALIZATION:
                return await self._execute_handoff(context, decision)
            
            elif decision.decision_type == DecisionType.HANDOFF_ESCALATION:
                return await self._execute_escalation(context, decision)
            
            elif decision.decision_type == DecisionType.REQUEST_REVIEW:
                return await self._execute_review_request(context, decision)
            
            elif decision.decision_type == DecisionType.REQUEST_COLLABORATION:
                return await self._execute_collaboration_request(context, decision)
            
            elif decision.decision_type == DecisionType.RETRY_WITH_LEARNING:
                return await self._execute_retry(context, decision)
            
            elif decision.decision_type == DecisionType.DEFER_TO_HUMAN:
                return await self._execute_human_deferral(context, decision)
            
            elif decision.decision_type == DecisionType.EXECUTE_LOCALLY:
                # Already being executed locally, just update metadata
                await self.acd_service.update_context(
                    context.id,
                    ACDContextUpdate(
                        ai_note="Executing locally based on orchestrator decision",
                    )
                )
                return True
            
            else:
                logger.warning(f"Unknown decision type: {decision.decision_type}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to execute decision: {e}")
            return False
    
    async def _execute_handoff(
        self,
        context: ACDContextResponse,
        decision: OrchestrationDecision,
    ) -> bool:
        """Execute handoff to specialized agent."""
        try:
            update = ACDContextUpdate(
                ai_handoff_requested=True,
                ai_handoff_to=decision.target_agent,
                ai_handoff_type=HandoffType.SPECIALIZATION.value,
                ai_handoff_reason=decision.reasoning,
                ai_handoff_status=HandoffStatus.REQUESTED.value,
                ai_handoff_notes=f"Patterns: {', '.join(decision.learned_patterns[:3])}",
            )
            
            await self.acd_service.update_context(context.id, update)
            logger.info(f"Handoff requested to {decision.target_agent}")
            return True
            
        except Exception as e:
            logger.error(f"Handoff execution failed: {e}")
            return False
    
    async def _execute_escalation(
        self,
        context: ACDContextResponse,
        decision: OrchestrationDecision,
    ) -> bool:
        """Execute escalation to expert."""
        try:
            update = ACDContextUpdate(
                ai_handoff_requested=True,
                ai_handoff_to=decision.target_agent or "expert_coordinator",
                ai_handoff_type=HandoffType.ESCALATION.value,
                ai_handoff_reason=decision.reasoning,
                ai_skill_level_required=SkillLevel.EXPERT.value,
                ai_queue_priority="HIGH",
            )
            
            await self.acd_service.update_context(context.id, update)
            logger.info("Escalation initiated")
            return True
            
        except Exception as e:
            logger.error(f"Escalation execution failed: {e}")
            return False
    
    async def _execute_review_request(
        self,
        context: ACDContextResponse,
        decision: OrchestrationDecision,
    ) -> bool:
        """Execute review request."""
        try:
            update = ACDContextUpdate(
                ai_request=AIRequest.REQUEST_REVIEW.value,
                ai_note_request=decision.reasoning,
                ai_state=AIState.READY.value,
            )
            
            await self.acd_service.update_context(context.id, update)
            logger.info("Review requested")
            return True
            
        except Exception as e:
            logger.error(f"Review request execution failed: {e}")
            return False
    
    async def _execute_collaboration_request(
        self,
        context: ACDContextResponse,
        decision: OrchestrationDecision,
    ) -> bool:
        """Execute collaboration request."""
        try:
            update = ACDContextUpdate(
                ai_request=AIRequest.REQUEST_ASSISTANCE.value,
                ai_note_request=f"Collaboration: {decision.reasoning}",
                ai_metadata={
                    "collaboration_type": decision.action_plan.get("collaboration_type"),
                    "num_agents": decision.action_plan.get("num_agents", 2),
                },
            )
            
            await self.acd_service.update_context(context.id, update)
            logger.info("Collaboration requested")
            return True
            
        except Exception as e:
            logger.error(f"Collaboration request execution failed: {e}")
            return False
    
    async def _execute_retry(
        self,
        context: ACDContextResponse,
        decision: OrchestrationDecision,
    ) -> bool:
        """Execute retry with learned patterns."""
        try:
            current_metadata = context.ai_metadata or {}
            current_metadata["retry_count"] = decision.action_plan.get("retry_count", 1)
            current_metadata["learned_strategies"] = decision.action_plan.get("strategies", [])
            
            update = ACDContextUpdate(
                ai_state=AIState.READY.value,
                ai_note=f"Retry #{current_metadata['retry_count']} with learned patterns",
                ai_metadata=current_metadata,
                ai_strategy=", ".join(decision.learned_patterns[:3]),
            )
            
            await self.acd_service.update_context(context.id, update)
            logger.info(f"Retry initiated (attempt {current_metadata['retry_count']})")
            return True
            
        except Exception as e:
            logger.error(f"Retry execution failed: {e}")
            return False
    
    async def _execute_human_deferral(
        self,
        context: ACDContextResponse,
        decision: OrchestrationDecision,
    ) -> bool:
        """Execute deferral to human."""
        try:
            update = ACDContextUpdate(
                ai_state=AIState.BLOCKED.value,
                ai_request=AIRequest.WAITING_FOR_INPUT.value,
                ai_note_request=f"Human intervention required: {decision.reasoning}",
                human_override="REQUESTED",
            )
            
            await self.acd_service.update_context(context.id, update)
            logger.info("Deferred to human")
            return True
            
        except Exception as e:
            logger.error(f"Human deferral execution failed: {e}")
            return False
    
    async def learn_from_outcome(
        self,
        context_id: UUID,
        success: bool,
        outcome_metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Learn from the outcome of an orchestration decision.
        
        This implements the "learning" function of the basal ganglia,
        reinforcing successful patterns and inhibiting failed patterns.
        
        Args:
            context_id: ACD context ID
            success: Whether the outcome was successful
            outcome_metadata: Additional outcome information
        """
        try:
            context = await self.acd_service.get_context(context_id)
            if not context:
                return
            
            # Extract decision from metadata
            if not context.ai_metadata or "orchestration_decision" not in context.ai_metadata:
                return
            
            decision_data = context.ai_metadata["orchestration_decision"]
            
            # Update pattern cache based on outcome
            cache_key = f"{context.ai_phase}_{context.ai_complexity}"
            
            if success:
                # Reinforce successful pattern
                logger.info(
                    f"Learning: Reinforcing successful pattern "
                    f"({decision_data['type']}) for {cache_key}"
                )
                
                # Clear cache to force refresh with new successful pattern
                if cache_key in self._pattern_cache:
                    del self._pattern_cache[cache_key]
                
            else:
                # Inhibit failed pattern
                logger.info(
                    f"Learning: Inhibiting failed pattern "
                    f"({decision_data['type']}) for {cache_key}"
                )
                
                # Update metadata to mark this decision as unsuccessful
                current_metadata = context.ai_metadata or {}
                if "failed_decisions" not in current_metadata:
                    current_metadata["failed_decisions"] = []
                
                current_metadata["failed_decisions"].append({
                    "decision_type": decision_data["type"],
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "reason": outcome_metadata.get("failure_reason") if outcome_metadata else None,
                })
                
                await self.acd_service.update_context(
                    context_id,
                    ACDContextUpdate(ai_metadata=current_metadata)
                )
            
            logger.info(f"Learning complete for context {context_id}")
            
        except Exception as e:
            logger.error(f"Learning from outcome failed: {e}")
