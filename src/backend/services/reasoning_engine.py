"""
Reasoning Engine - The Actual Brain

This is the REAL reasoning model that makes orchestration decisions.
It uses an LLM to evaluate context and decide what to do.

This is what was missing - the CPU, the brain, the intelligence.
"""

import json
import asyncio
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timezone
import traceback

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from backend.models.acd import (
    ACDContextResponse,
    ACDTraceArtifactModel,
    AIComplexity,
    AIConfidence,
    AIState,
)
from backend.services.multi_agent_service import MultiAgentService
from backend.models.multi_agent import AgentModel, AgentStatus
from backend.config.logging import get_logger

logger = get_logger(__name__)


class ReasoningEngine:
    """
    The actual reasoning brain that makes orchestration decisions.
    
    This is NOT a scheduler. This is NOT a task queue.
    This IS a meta-model that reasons over context and decides actions.
    """
    
    def __init__(self, db_session: AsyncSession):
        """
        Initialize reasoning engine.
        
        Args:
            db_session: Database session
        """
        self.db = db_session
        self.multi_agent_service = MultiAgentService(db_session)
        
        # Model configuration
        self.reasoning_model = None
        self.model_available = False
        self._initialize_reasoning_model()
    
    def _initialize_reasoning_model(self):
        """Initialize the reasoning model (LLM)."""
        try:
            # Try to import AI models manager
            from backend.services.ai_models import ai_models
            self.reasoning_model = ai_models
            self.model_available = True
            logger.info("🧠 Reasoning engine initialized with AI models")
        except Exception as e:
            logger.warning(f"⚠️ AI models not available for reasoning: {e}")
            logger.warning("Reasoning engine will use rule-based fallback")
            self.model_available = False
    
    async def reason_about_context(
        self,
        context: ACDContextResponse,
        current_agent: Optional[str] = None,
        situation: Optional[Dict[str, Any]] = None,
        patterns: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """
        THE BRAIN - Reason about an ACD context and decide what to do.
        
        This is where actual LLM-based reasoning happens.
        
        Args:
            context: ACD context to reason about
            current_agent: Current agent handling the task
            situation: Situation assessment
            patterns: Learned patterns
            
        Returns:
            Decision dictionary with:
            - decision_type: What action to take
            - reasoning: Why this decision was made
            - target_agent: Which agent should handle it
            - confidence: How confident we are
            - action_plan: Detailed execution plan
            - learned_patterns: Patterns to apply
            - risk_assessment: Risk evaluation
        """
        logger.info(f"🧠 Reasoning about context {context.id} (phase={context.ai_phase})")
        
        # Step 1: Build reasoning prompt
        prompt = await self._build_reasoning_prompt(
            context, current_agent, situation, patterns
        )
        
        # Step 2: Call reasoning model
        if self.model_available:
            decision = await self._call_reasoning_model(prompt, context)
        else:
            # Fallback to rule-based reasoning
            decision = await self._fallback_reasoning(context, situation, patterns, current_agent)
        
        logger.info(
            f"🧠 Reasoning complete: {decision['decision_type']} "
            f"(confidence={decision['confidence']})"
        )
        
        return decision
    
    async def _build_reasoning_prompt(
        self,
        context: ACDContextResponse,
        current_agent: Optional[str],
        situation: Optional[Dict[str, Any]],
        patterns: Optional[List[Dict[str, Any]]],
    ) -> str:
        """
        Build a comprehensive prompt for the reasoning model.
        
        This prompt contains everything the model needs to make a decision.
        """
        # Get available agents
        agents = await self._get_available_agents()
        
        # Get trace artifacts (errors)
        trace_artifacts = await self._get_trace_artifacts(context.id)
        
        # Build prompt
        prompt = f"""You are an intelligent orchestration system for AI content generation. Your role is to analyze the current task context and decide the best course of action.

## Current Task Context

**Phase**: {context.ai_phase}
**Current State**: {context.ai_state}
**Complexity**: {context.ai_complexity or 'UNKNOWN'}
**Confidence**: {context.ai_confidence or 'UNKNOWN'}
**Current Agent**: {current_agent or 'NONE'}
**Assigned To**: {context.ai_assigned_to or 'NONE'}

**Task Description**: {context.ai_note or 'No description'}

**Context Metadata**:
```json
{json.dumps(context.ai_context or {}, indent=2)}
```

## Situation Assessment

"""
        
        if situation:
            prompt += f"""
**Has Errors**: {situation.get('has_errors', False)}
**Handoff Count**: {situation.get('handoff_count', 0)}
**Retry Count**: {situation.get('retry_count', 0)}
**Is Blocked**: {situation.get('is_blocked', False)}
**Time Spent**: {situation.get('time_spent_seconds', 0)} seconds
"""
        
        if trace_artifacts:
            prompt += f"\n## Recent Errors\n"
            for artifact in trace_artifacts[:3]:  # Last 3 errors
                prompt += f"- {artifact.event_type}: {artifact.error_message}\n"
        
        if patterns:
            prompt += f"\n## Learned Patterns from Similar Tasks\n"
            prompt += f"Found {len(patterns)} successful patterns:\n"
            for i, pattern in enumerate(patterns[:5], 1):
                prompt += f"{i}. Agent: {pattern.get('assigned_to', 'unknown')}, "
                prompt += f"Strategy: {pattern.get('strategy', 'none')}, "
                prompt += f"Pattern: {pattern.get('pattern', 'none')}\n"
        
        if agents:
            prompt += f"\n## Available Agents\n"
            for agent in agents[:10]:  # Top 10 agents
                prompt += f"- **{agent.agent_name}** ({agent.agent_type})\n"
                prompt += f"  Specializations: {agent.specializations or []}\n"
                prompt += f"  Success Rate: {agent.success_rate * 100:.1f}%\n"
                prompt += f"  Load: {agent.current_load}/{agent.max_concurrent_tasks}\n"
        
        prompt += """

## Your Decision

Analyze the above context and provide a structured decision. Consider:

1. **Current Capability**: Can the current agent handle this task?
2. **Complexity**: Is the task too complex for local execution?
3. **Errors**: Are there errors that suggest a different approach?
4. **Patterns**: Do learned patterns suggest a better agent?
5. **State**: Is the task blocked, failed, or needs review?
6. **Risk**: What are the risks of each decision?

## Required Decision Types

Choose ONE of these decision types:

- **EXECUTE_LOCALLY**: Current agent can handle it
- **HANDOFF_SPECIALIZATION**: Route to a specialist based on capabilities
- **HANDOFF_ESCALATION**: Escalate to expert due to complexity
- **REQUEST_REVIEW**: Ask for validation before proceeding
- **REQUEST_COLLABORATION**: Multi-agent collaboration needed
- **RETRY_WITH_LEARNING**: Retry applying learned strategies
- **DEFER_TO_HUMAN**: Human intervention required

## Response Format

Provide your decision in JSON format:

```json
{
  "decision_type": "EXECUTE_LOCALLY",
  "reasoning": "Detailed explanation of why this decision was made...",
  "target_agent": "agent_name or null",
  "confidence": "HIGH, MEDIUM, or LOW",
  "action_plan": {
    "key": "value",
    "next_steps": ["step1", "step2"]
  },
  "learned_patterns": ["pattern1", "pattern2"],
  "risk_assessment": "Low/Medium/High risk explanation",
  "metadata_updates": {
    "ai_queue_priority": "NORMAL",
    "ai_skill_level_required": "INTERMEDIATE"
  }
}
```

Provide ONLY the JSON response, no other text.
"""
        
        return prompt
    
    async def _call_reasoning_model(
        self,
        prompt: str,
        context: ACDContextResponse,
    ) -> Dict[str, Any]:
        """
        Call the reasoning model (LLM) to make a decision.
        
        This is THE BRAIN - where actual AI reasoning happens.
        """
        try:
            logger.info("🧠 Calling reasoning model for decision...")
            
            # Check if we have text generation available
            if not self.reasoning_model:
                logger.warning("No reasoning model available, using fallback")
                return await self._fallback_reasoning(context, None, None, None)
            
            # Try to use available text models
            try:
                # Generate reasoning using text model
                response = await self.reasoning_model.generate_text(
                    prompt=prompt,
                    max_tokens=1000,
                    temperature=0.3,  # Lower temperature for more deterministic reasoning
                    system_message="You are an intelligent orchestration system that makes decisions about task routing and agent coordination.",
                )
                
                logger.info(f"🧠 Reasoning model response received ({len(response)} chars)")
                
                # Parse JSON from response
                decision = self._parse_reasoning_response(response)
                
                # Validate and enhance decision
                decision = self._validate_decision(decision, context)
                
                return decision
                
            except Exception as e:
                logger.error(f"Error calling reasoning model: {e}")
                logger.info("Falling back to rule-based reasoning")
                return await self._fallback_reasoning(context, None, None, None)
        
        except Exception as e:
            logger.error(f"Error in reasoning model call: {e}")
            return await self._fallback_reasoning(context, None, None, None)
    
    def _parse_reasoning_response(self, response: str) -> Dict[str, Any]:
        """
        Parse the reasoning model's JSON response.
        
        Args:
            response: Raw text response from model
            
        Returns:
            Parsed decision dictionary
        """
        try:
            # Try to find JSON in the response
            start_idx = response.find('{')
            end_idx = response.rfind('}') + 1
            
            if start_idx != -1 and end_idx > start_idx:
                json_str = response[start_idx:end_idx]
                decision = json.loads(json_str)
                
                logger.info("✅ Successfully parsed reasoning model JSON")
                return decision
            else:
                raise ValueError("No JSON found in response")
        
        except Exception as e:
            logger.error(f"Failed to parse reasoning response: {e}")
            logger.error(f"Response was: {response[:500]}")
            
            # Return a safe default
            return {
                "decision_type": "REQUEST_REVIEW",
                "reasoning": f"Failed to parse reasoning model output: {str(e)}",
                "target_agent": None,
                "confidence": "LOW",
                "action_plan": {},
                "learned_patterns": [],
                "risk_assessment": "Unable to assess - parsing failed",
            }
    
    def _validate_decision(
        self,
        decision: Dict[str, Any],
        context: ACDContextResponse,
    ) -> Dict[str, Any]:
        """
        Validate and enhance the reasoning model's decision.
        
        Args:
            decision: Raw decision from model
            context: ACD context
            
        Returns:
            Validated and enhanced decision
        """
        # Validate decision_type
        valid_types = [
            "EXECUTE_LOCALLY",
            "HANDOFF_SPECIALIZATION",
            "HANDOFF_ESCALATION",
            "REQUEST_REVIEW",
            "REQUEST_COLLABORATION",
            "RETRY_WITH_LEARNING",
            "DEFER_TO_HUMAN",
        ]
        
        if decision.get("decision_type") not in valid_types:
            logger.warning(f"Invalid decision type: {decision.get('decision_type')}, defaulting to REQUEST_REVIEW")
            decision["decision_type"] = "REQUEST_REVIEW"
        
        # Validate confidence
        valid_confidence = ["VERY_HIGH", "HIGH", "MEDIUM", "LOW", "VERY_LOW"]
        if decision.get("confidence") not in valid_confidence:
            decision["confidence"] = "MEDIUM"
        
        # Ensure required fields exist
        decision.setdefault("reasoning", "No reasoning provided")
        decision.setdefault("target_agent", None)
        decision.setdefault("action_plan", {})
        decision.setdefault("learned_patterns", [])
        decision.setdefault("risk_assessment", "Not assessed")
        decision.setdefault("metadata_updates", {})
        
        # Add timestamp
        decision["timestamp"] = datetime.now(timezone.utc).isoformat()
        
        return decision
    
    async def _fallback_reasoning(
        self,
        context: ACDContextResponse,
        situation: Optional[Dict[str, Any]],
        patterns: Optional[List[Dict[str, Any]]],
        current_agent: Optional[str],
    ) -> Dict[str, Any]:
        """
        Fallback rule-based reasoning when LLM is not available.
        
        This is the safety net - simpler but still functional.
        """
        logger.info("🔧 Using rule-based fallback reasoning")
        
        # Extract info
        complexity = context.ai_complexity or "MEDIUM"
        confidence = context.ai_confidence or "UNCERTAIN"
        state = context.ai_state
        has_errors = bool(context.runtime_err or context.compiler_err)
        
        # Calculate scores
        complexity_scores = {"LOW": 0.2, "MEDIUM": 0.5, "HIGH": 0.8, "CRITICAL": 1.0}
        confidence_scores = {
            "VALIDATED": 1.0,
            "CONFIDENT": 0.8,
            "HYPOTHESIS": 0.5,
            "UNCERTAIN": 0.3,
            "EXPERIMENTAL": 0.4,
        }
        
        complexity_score = complexity_scores.get(complexity, 0.5)
        confidence_score = confidence_scores.get(confidence, 0.5)
        
        # Add error penalty
        if has_errors:
            complexity_score = min(complexity_score + 0.2, 1.0)
            confidence_score = max(confidence_score - 0.2, 0.0)
        
        # Decision logic
        if complexity_score >= 0.8 and confidence_score < 0.5:
            return {
                "decision_type": "HANDOFF_ESCALATION",
                "reasoning": f"High complexity ({complexity_score:.2f}) with low confidence ({confidence_score:.2f}) requires expert",
                "target_agent": "expert_coordinator",
                "confidence": "HIGH",
                "action_plan": {"handoff_type": "ESCALATION", "skill_level": "EXPERT"},
                "learned_patterns": [],
                "risk_assessment": "Medium - escalation needed to prevent failure",
            }
        
        elif state == "FAILED" and patterns:
            return {
                "decision_type": "RETRY_WITH_LEARNING",
                "reasoning": f"Task failed but {len(patterns)} successful patterns available for retry",
                "target_agent": current_agent,
                "confidence": "MEDIUM",
                "action_plan": {"retry": True, "apply_patterns": True, "strategies": [p.get("strategy") for p in patterns[:3]]},
                "learned_patterns": [p.get("strategy", "") for p in patterns[:3]],
                "risk_assessment": "Low - patterns suggest retry will succeed",
            }
        
        elif confidence_score < 0.5:
            return {
                "decision_type": "REQUEST_REVIEW",
                "reasoning": f"Low confidence ({confidence_score:.2f}) requires validation before proceeding",
                "target_agent": None,
                "confidence": "MEDIUM",
                "action_plan": {"request_type": "REQUEST_REVIEW", "reviewer_type": "quality_assurance"},
                "learned_patterns": [],
                "risk_assessment": "Medium - validation needed to ensure quality",
            }
        
        elif patterns and confidence_score < 0.7:
            # Find best agent from patterns
            agent_counts = {}
            for pattern in patterns:
                agent = pattern.get("assigned_to")
                if agent and agent != current_agent:
                    agent_counts[agent] = agent_counts.get(agent, 0) + 1
            
            if agent_counts:
                best_agent = max(agent_counts, key=agent_counts.get)
                pattern_confidence = agent_counts[best_agent] / len(patterns)
                
                return {
                    "decision_type": "HANDOFF_SPECIALIZATION",
                    "reasoning": f"Patterns indicate agent '{best_agent}' has {pattern_confidence:.0%} success rate for similar tasks",
                    "target_agent": best_agent,
                    "confidence": "HIGH" if pattern_confidence > 0.6 else "MEDIUM",
                    "action_plan": {"handoff_type": "SPECIALIZATION", "pattern_confidence": pattern_confidence},
                    "learned_patterns": [p.get("strategy", "") for p in patterns[:3]],
                    "risk_assessment": f"Low - {pattern_confidence:.0%} success rate with this agent",
                }
        
        # Default: execute locally
        return {
            "decision_type": "EXECUTE_LOCALLY",
            "reasoning": f"Confidence ({confidence_score:.2f}) and complexity ({complexity_score:.2f}) acceptable for local execution",
            "target_agent": current_agent,
            "confidence": "HIGH" if confidence_score >= 0.7 else "MEDIUM",
            "action_plan": {"execute_agent": current_agent or "default", "apply_patterns": bool(patterns)},
            "learned_patterns": [p.get("strategy", "") for p in patterns[:3]] if patterns else [],
            "risk_assessment": "Low - task within capability",
        }
    
    async def _get_available_agents(self) -> List[Any]:
        """Get list of available agents for routing decisions."""
        try:
            agents = await self.multi_agent_service.list_agents(
                status=AgentStatus.IDLE,
            )
            return agents if agents else []
        except Exception as e:
            logger.error(f"Error fetching agents: {e}")
            return []
    
    async def _get_trace_artifacts(self, context_id) -> List[Any]:
        """Get trace artifacts (errors) for a context."""
        try:
            stmt = (
                select(ACDTraceArtifactModel)
                .where(ACDTraceArtifactModel.acd_context_id == context_id)
                .order_by(ACDTraceArtifactModel.timestamp.desc())
                .limit(5)
            )
            result = await self.db.execute(stmt)
            artifacts = result.scalars().all()
            return list(artifacts)
        except Exception as e:
            logger.error(f"Error fetching trace artifacts: {e}")
            return []
    
    async def decompose_task(
        self,
        context: ACDContextResponse,
        decision: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """
        Decompose a complex task into sub-tasks.
        
        Args:
            context: ACD context
            decision: Orchestration decision
            
        Returns:
            List of sub-task specifications
        """
        # This would use the reasoning model to break down complex tasks
        # For now, return empty list (not implemented yet)
        return []
    
    async def evaluate_capability_match(
        self,
        agent_name: str,
        task_requirements: Dict[str, Any],
    ) -> float:
        """
        Evaluate how well an agent's capabilities match task requirements.
        
        Args:
            agent_name: Name of agent to evaluate
            task_requirements: Required capabilities
            
        Returns:
            Match score (0.0-1.0)
        """
        try:
            # Get agent from database
            stmt = select(AgentModel).where(AgentModel.agent_name == agent_name)
            result = await self.db.execute(stmt)
            agent = result.scalar_one_or_none()
            
            if not agent:
                return 0.0
            
            # Simple capability matching
            agent_caps = set(agent.capabilities or [])
            required_caps = set(task_requirements.get("capabilities", []))
            
            if not required_caps:
                return 0.5  # No requirements, neutral match
            
            overlap = len(agent_caps & required_caps)
            match_score = overlap / len(required_caps)
            
            # Boost for success rate
            match_score = (match_score + agent.success_rate) / 2
            
            return min(match_score, 1.0)
        
        except Exception as e:
            logger.error(f"Error evaluating capability match: {e}")
            return 0.5  # Default neutral
