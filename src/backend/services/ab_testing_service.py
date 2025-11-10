"""
Automated A/B Testing Service

Manages A/B test creation, execution, analysis, and recommendations.
"""

import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timezone, timedelta
from uuid import UUID, uuid4
from enum import Enum

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_, func
from pydantic import BaseModel, Field

from backend.models.acd import ACDContextModel
from backend.config.logging import get_logger

logger = get_logger(__name__)


class TestStatus(str, Enum):
    """A/B test status."""
    DRAFT = "draft"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


class VariantPerformance(BaseModel):
    """Performance metrics for a test variant."""
    variant_id: str
    name: str
    impressions: int = 0
    clicks: int = 0
    conversions: int = 0
    engagement_count: int = 0
    engagement_rate: float = 0.0
    click_through_rate: float = 0.0
    conversion_rate: float = 0.0
    confidence_score: float = 0.0


class ABTestConfig(BaseModel):
    """Configuration for an A/B test."""
    test_id: UUID = Field(default_factory=uuid4)
    test_name: str
    description: Optional[str] = None
    variants: List[Dict[str, Any]]
    control_variant_id: str = "control"
    success_metric: str = "engagement_rate"
    minimum_sample_size: int = 100
    minimum_runtime_hours: int = 24
    confidence_level: float = 0.95
    significance_threshold: float = 0.05
    auto_winner_selection: bool = True
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    status: TestStatus = TestStatus.DRAFT


class ABTestResult(BaseModel):
    """Results of an A/B test."""
    test_id: UUID
    test_name: str
    status: TestStatus
    variants_performance: List[VariantPerformance]
    winner: Optional[str] = None
    confidence_level: float
    statistical_significance: bool
    p_value: Optional[float] = None
    effect_size: Optional[float] = None
    recommendation: str
    insights: List[str]
    completed_at: Optional[datetime] = None


class ABTestingService:
    """
    Service for automated A/B testing.
    
    Features:
    - Test configuration and management
    - Statistical significance testing
    - Automatic winner selection
    - Performance tracking
    """
    
    def __init__(self, db_session: AsyncSession):
        """
        Initialize A/B testing service.
        
        Args:
            db_session: Database session
        """
        self.db = db_session
        self.active_tests: Dict[UUID, ABTestConfig] = {}
        self.test_results: Dict[UUID, Dict[str, VariantPerformance]] = {}
    
    async def create_test(
        self,
        test_name: str,
        variants: List[Dict[str, Any]],
        success_metric: str = "engagement_rate",
        description: Optional[str] = None,
        minimum_sample_size: int = 100,
        minimum_runtime_hours: int = 24
    ) -> ABTestConfig:
        """
        Create a new A/B test configuration.
        
        Args:
            test_name: Name of the test
            variants: List of variant configurations
            success_metric: Metric to optimize
            description: Optional description
            minimum_sample_size: Minimum samples per variant
            minimum_runtime_hours: Minimum test duration
            
        Returns:
            Test configuration
        """
        try:
            # Validate variants
            if len(variants) < 2:
                raise ValueError("At least 2 variants required (including control)")
            
            # Ensure control variant exists
            control_exists = any(v.get("variant_id") == "control" for v in variants)
            if not control_exists:
                variants.insert(0, {
                    "variant_id": "control",
                    "name": "Control",
                    "description": "Baseline version",
                    "changes": {}
                })
            
            config = ABTestConfig(
                test_name=test_name,
                description=description,
                variants=variants,
                success_metric=success_metric,
                minimum_sample_size=minimum_sample_size,
                minimum_runtime_hours=minimum_runtime_hours
            )
            
            self.active_tests[config.test_id] = config
            
            # Initialize results tracking
            self.test_results[config.test_id] = {
                v["variant_id"]: VariantPerformance(
                    variant_id=v["variant_id"],
                    name=v.get("name", v["variant_id"])
                )
                for v in variants
            }
            
            logger.info(f"Created A/B test {config.test_id}: {test_name}")
            
            return config
            
        except Exception as e:
            logger.error(f"Test creation failed: {e}")
            raise
    
    async def start_test(self, test_id: UUID) -> bool:
        """
        Start an A/B test.
        
        Args:
            test_id: Test identifier
            
        Returns:
            Success status
        """
        try:
            if test_id not in self.active_tests:
                raise ValueError(f"Test {test_id} not found")
            
            config = self.active_tests[test_id]
            config.status = TestStatus.RUNNING
            
            logger.info(f"Started A/B test {test_id}")
            return True
            
        except Exception as e:
            logger.error(f"Test start failed: {e}")
            return False
    
    async def record_variant_event(
        self,
        test_id: UUID,
        variant_id: str,
        event_type: str,
        value: float = 1.0
    ) -> bool:
        """
        Record an event for a test variant.
        
        Args:
            test_id: Test identifier
            variant_id: Variant identifier
            event_type: Type of event (impression, click, conversion, engagement)
            value: Event value
            
        Returns:
            Success status
        """
        try:
            if test_id not in self.test_results:
                logger.warning(f"Test {test_id} not found")
                return False
            
            variant_perf = self.test_results[test_id].get(variant_id)
            if not variant_perf:
                logger.warning(f"Variant {variant_id} not found in test {test_id}")
                return False
            
            # Update metrics based on event type
            if event_type == "impression":
                variant_perf.impressions += int(value)
            elif event_type == "click":
                variant_perf.clicks += int(value)
            elif event_type == "conversion":
                variant_perf.conversions += int(value)
            elif event_type == "engagement":
                variant_perf.engagement_count += int(value)
            
            # Recalculate rates
            if variant_perf.impressions > 0:
                variant_perf.click_through_rate = (
                    variant_perf.clicks / variant_perf.impressions
                ) * 100
                variant_perf.engagement_rate = (
                    variant_perf.engagement_count / variant_perf.impressions
                ) * 100
            
            if variant_perf.clicks > 0:
                variant_perf.conversion_rate = (
                    variant_perf.conversions / variant_perf.clicks
                ) * 100
            
            return True
            
        except Exception as e:
            logger.error(f"Event recording failed: {e}")
            return False
    
    def _calculate_statistical_significance(
        self,
        control_performance: VariantPerformance,
        variant_performance: VariantPerformance,
        metric: str = "engagement_rate"
    ) -> Tuple[float, bool]:
        """
        Calculate statistical significance using two-proportion z-test.
        
        Args:
            control_performance: Control variant metrics
            variant_performance: Test variant metrics
            metric: Metric to compare
            
        Returns:
            (p_value, is_significant)
        """
        try:
            # Get success counts and sample sizes based on metric
            if metric == "engagement_rate":
                x1 = control_performance.engagement_count
                n1 = control_performance.impressions
                x2 = variant_performance.engagement_count
                n2 = variant_performance.impressions
            elif metric == "click_through_rate":
                x1 = control_performance.clicks
                n1 = control_performance.impressions
                x2 = variant_performance.clicks
                n2 = variant_performance.impressions
            elif metric == "conversion_rate":
                x1 = control_performance.conversions
                n1 = control_performance.clicks
                x2 = variant_performance.conversions
                n2 = variant_performance.clicks
            else:
                return 1.0, False
            
            # Check minimum sample sizes
            if n1 < 30 or n2 < 30:
                return 1.0, False
            
            # Calculate proportions
            p1 = x1 / n1 if n1 > 0 else 0
            p2 = x2 / n2 if n2 > 0 else 0
            
            # Pooled proportion
            p_pool = (x1 + x2) / (n1 + n2) if (n1 + n2) > 0 else 0
            
            # Standard error
            se = np.sqrt(p_pool * (1 - p_pool) * (1/n1 + 1/n2))
            
            if se == 0:
                return 1.0, False
            
            # Z-score
            z = (p2 - p1) / se
            
            # Two-tailed p-value
            from scipy.stats import norm
            p_value = 2 * (1 - norm.cdf(abs(z)))
            
            is_significant = p_value < 0.05
            
            return float(p_value), is_significant
            
        except Exception as e:
            logger.error(f"Statistical test failed: {e}")
            return 1.0, False
    
    def _calculate_effect_size(
        self,
        control_performance: VariantPerformance,
        variant_performance: VariantPerformance,
        metric: str = "engagement_rate"
    ) -> float:
        """
        Calculate effect size (Cohen's h for proportions).
        
        Args:
            control_performance: Control variant metrics
            variant_performance: Test variant metrics
            metric: Metric to compare
            
        Returns:
            Effect size
        """
        try:
            # Get rates based on metric
            if metric == "engagement_rate":
                p1 = control_performance.engagement_rate / 100
                p2 = variant_performance.engagement_rate / 100
            elif metric == "click_through_rate":
                p1 = control_performance.click_through_rate / 100
                p2 = variant_performance.click_through_rate / 100
            elif metric == "conversion_rate":
                p1 = control_performance.conversion_rate / 100
                p2 = variant_performance.conversion_rate / 100
            else:
                return 0.0
            
            # Cohen's h = 2 * (arcsin(sqrt(p1)) - arcsin(sqrt(p2)))
            h = 2 * (np.arcsin(np.sqrt(p2)) - np.arcsin(np.sqrt(p1)))
            
            return float(h)
            
        except Exception as e:
            logger.error(f"Effect size calculation failed: {e}")
            return 0.0
    
    async def analyze_test(
        self,
        test_id: UUID,
        auto_select_winner: bool = True
    ) -> ABTestResult:
        """
        Analyze A/B test results and determine winner.
        
        Args:
            test_id: Test identifier
            auto_select_winner: Automatically select winning variant
            
        Returns:
            Test results with winner and insights
        """
        try:
            if test_id not in self.active_tests:
                raise ValueError(f"Test {test_id} not found")
            
            config = self.active_tests[test_id]
            variant_results = self.test_results[test_id]
            
            # Get control performance
            control_perf = variant_results.get("control")
            if not control_perf:
                raise ValueError("Control variant not found")
            
            # Check if test meets minimum requirements
            min_samples_met = all(
                perf.impressions >= config.minimum_sample_size
                for perf in variant_results.values()
            )
            
            if not min_samples_met:
                return ABTestResult(
                    test_id=test_id,
                    test_name=config.test_name,
                    status=config.status,
                    variants_performance=list(variant_results.values()),
                    confidence_level=config.confidence_level,
                    statistical_significance=False,
                    recommendation="Continue test - minimum sample size not yet reached",
                    insights=["Insufficient data for statistical analysis"]
                )
            
            # Analyze each variant against control
            variant_analyses = []
            best_variant = None
            best_performance = getattr(control_perf, config.success_metric)
            
            for variant_id, variant_perf in variant_results.items():
                if variant_id == "control":
                    continue
                
                p_value, is_significant = self._calculate_statistical_significance(
                    control_perf, variant_perf, config.success_metric
                )
                
                effect_size = self._calculate_effect_size(
                    control_perf, variant_perf, config.success_metric
                )
                
                variant_metric_value = getattr(variant_perf, config.success_metric)
                
                variant_analyses.append({
                    "variant_id": variant_id,
                    "variant_name": variant_perf.name,
                    "metric_value": variant_metric_value,
                    "vs_control": f"{((variant_metric_value / best_performance - 1) * 100):.1f}%"
                        if best_performance > 0 else "N/A",
                    "p_value": p_value,
                    "is_significant": is_significant,
                    "effect_size": effect_size
                })
                
                # Track best performing variant
                if variant_metric_value > best_performance and is_significant:
                    best_variant = variant_id
                    best_performance = variant_metric_value
            
            # Determine winner
            winner = best_variant if auto_select_winner and best_variant else None
            if not winner and auto_select_winner:
                winner = "control"
            
            # Generate insights
            insights = []
            
            if winner and winner != "control":
                winner_data = next(v for v in variant_analyses if v["variant_id"] == winner)
                insights.append(
                    f"Variant '{winner}' wins with {winner_data['vs_control']} "
                    f"improvement over control (p={winner_data['p_value']:.4f})"
                )
            elif winner == "control":
                insights.append("Control performs best - no variants show significant improvement")
            
            # Add effect size interpretation
            for analysis in variant_analyses:
                effect = abs(analysis["effect_size"])
                if effect < 0.2:
                    size_desc = "small"
                elif effect < 0.5:
                    size_desc = "medium"
                else:
                    size_desc = "large"
                
                insights.append(
                    f"Variant '{analysis['variant_name']}' shows {size_desc} effect "
                    f"({analysis['vs_control']} vs control)"
                )
            
            # Generate recommendation
            if winner and winner != "control":
                recommendation = f"Deploy variant '{winner}' - shows significant improvement"
            elif winner == "control":
                recommendation = "Maintain control - test variants did not improve performance"
            else:
                recommendation = "Continue testing or design new variants"
            
            # Select overall statistical significance
            any_significant = any(v["is_significant"] for v in variant_analyses)
            
            # Get p-value and effect size for winner comparison
            winner_analysis = next(
                (v for v in variant_analyses if v["variant_id"] == winner),
                None
            ) if winner and winner != "control" else None
            
            result = ABTestResult(
                test_id=test_id,
                test_name=config.test_name,
                status=TestStatus.COMPLETED,
                variants_performance=list(variant_results.values()),
                winner=winner,
                confidence_level=config.confidence_level,
                statistical_significance=any_significant,
                p_value=winner_analysis["p_value"] if winner_analysis else None,
                effect_size=winner_analysis["effect_size"] if winner_analysis else None,
                recommendation=recommendation,
                insights=insights,
                completed_at=datetime.now(timezone.utc)
            )
            
            # Update test status
            config.status = TestStatus.COMPLETED
            
            logger.info(f"Analyzed test {test_id}: winner={winner}")
            
            return result
            
        except Exception as e:
            logger.error(f"Test analysis failed: {e}")
            raise
    
    async def get_test_status(self, test_id: UUID) -> Dict[str, Any]:
        """
        Get current status of an A/B test.
        
        Args:
            test_id: Test identifier
            
        Returns:
            Test status and metrics
        """
        try:
            if test_id not in self.active_tests:
                return {"error": "Test not found"}
            
            config = self.active_tests[test_id]
            results = self.test_results.get(test_id, {})
            
            # Calculate progress
            total_impressions = sum(v.impressions for v in results.values())
            required_impressions = (
                len(results) * config.minimum_sample_size
            )
            progress_pct = min(100, (total_impressions / required_impressions * 100)
                if required_impressions > 0 else 0)
            
            # Check runtime
            runtime_hours = (
                datetime.now(timezone.utc) - config.created_at
            ).total_seconds() / 3600
            
            runtime_met = runtime_hours >= config.minimum_runtime_hours
            samples_met = total_impressions >= required_impressions
            
            return {
                "test_id": str(test_id),
                "test_name": config.test_name,
                "status": config.status.value,
                "created_at": config.created_at.isoformat(),
                "runtime_hours": float(runtime_hours),
                "minimum_runtime_hours": config.minimum_runtime_hours,
                "runtime_requirement_met": runtime_met,
                "total_impressions": total_impressions,
                "required_impressions": required_impressions,
                "sample_requirement_met": samples_met,
                "progress_percent": float(progress_pct),
                "ready_for_analysis": runtime_met and samples_met,
                "variants": [
                    {
                        "variant_id": v.variant_id,
                        "name": v.name,
                        "impressions": v.impressions,
                        "engagement_rate": v.engagement_rate,
                        "click_through_rate": v.click_through_rate,
                        "conversion_rate": v.conversion_rate
                    }
                    for v in results.values()
                ]
            }
            
        except Exception as e:
            logger.error(f"Status check failed: {e}")
            return {"error": str(e)}
    
    async def get_all_tests(self) -> List[Dict[str, Any]]:
        """
        Get all A/B tests.
        
        Returns:
            List of test summaries
        """
        return [
            {
                "test_id": str(test_id),
                "test_name": config.test_name,
                "status": config.status.value,
                "created_at": config.created_at.isoformat(),
                "num_variants": len(config.variants)
            }
            for test_id, config in self.active_tests.items()
        ]
