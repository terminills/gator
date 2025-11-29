"""
ML Learning and Predictive Scoring API Routes

Endpoints for Phase 3: Advanced Learning features.
"""

from typing import Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession

from backend.config.logging import get_logger
from backend.database.connection import get_db_session
from backend.services.ab_testing_service import (
    ABTestConfig,
    ABTestingService,
    ABTestResult,
)
from backend.services.ab_testing_service import AgentTaskCreate as ABTestCreate
from backend.services.acd_service import ACDService
from backend.services.cross_persona_learning import CrossPersonaLearningService
from backend.services.ml_pattern_recognition import MLPatternRecognitionService
from backend.services.predictive_engagement_scoring import (
    PredictiveEngagementScoringService,
)

logger = get_logger(__name__)

router = APIRouter(prefix="/api/v1/ml-learning", tags=["ml-learning"])


# ML Pattern Recognition Endpoints
@router.post("/models/train-engagement", status_code=200)
async def train_engagement_model(
    min_samples: int = Query(50, ge=10, le=1000),
    lookback_days: int = Query(90, ge=7, le=365),
    db: AsyncSession = Depends(get_db_session),
):
    """
    Train engagement prediction model from historical data.

    Args:
        min_samples: Minimum training samples
        lookback_days: Days of historical data
    """
    try:
        service = MLPatternRecognitionService(db)
        result = await service.train_engagement_model(min_samples, lookback_days)
        return result
    except Exception as e:
        logger.error(f"Model training failed: {e}")
        raise HTTPException(status_code=500, detail="Model training failed")


@router.post("/models/train-success-classifier", status_code=200)
async def train_success_classifier(
    success_threshold: float = Query(5.0, ge=0.0, le=100.0),
    min_samples: int = Query(50, ge=10, le=1000),
    lookback_days: int = Query(90, ge=7, le=365),
    db: AsyncSession = Depends(get_db_session),
):
    """
    Train success classification model.

    Args:
        success_threshold: Engagement rate threshold for success
        min_samples: Minimum training samples
        lookback_days: Days of historical data
    """
    try:
        service = MLPatternRecognitionService(db)
        result = await service.train_success_classifier(
            success_threshold, min_samples, lookback_days
        )
        return result
    except Exception as e:
        logger.error(f"Classifier training failed: {e}")
        raise HTTPException(status_code=500, detail="Classifier training failed")


@router.get("/models/feature-importance")
async def get_feature_importance(
    db: AsyncSession = Depends(get_db_session),
):
    """Get feature importance analysis from trained models."""
    try:
        service = MLPatternRecognitionService(db)
        analysis = await service.analyze_feature_importance()

        if not analysis:
            raise HTTPException(status_code=404, detail="No trained models found")

        return analysis
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Feature importance analysis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Predictive Scoring Endpoints
@router.get("/score/{context_id}")
async def score_content(
    context_id: UUID,
    db: AsyncSession = Depends(get_db_session),
):
    """
    Get engagement score for content.

    Args:
        context_id: ACD context identifier
    """
    try:
        acd_service = ACDService(db)
        context = await acd_service.get_context(context_id)

        if not context:
            raise HTTPException(status_code=404, detail="Context not found")

        # Convert to model
        from backend.models.acd import ACDContextModel

        stmt = "SELECT * FROM acd_contexts WHERE id = :context_id"
        from sqlalchemy import select

        stmt = select(ACDContextModel).where(ACDContextModel.id == context_id)
        result = await db.execute(stmt)
        context_model = result.scalar_one_or_none()

        if not context_model:
            raise HTTPException(status_code=404, detail="Context not found")

        scoring_service = PredictiveEngagementScoringService(db)
        score = await scoring_service.score_content(context_model)

        return score
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Content scoring failed: {e}")
        raise HTTPException(status_code=500, detail="Content scoring failed")


@router.get("/optimize/{context_id}")
async def optimize_content(
    context_id: UUID,
    target_score: float = Query(70.0, ge=0.0, le=100.0),
    db: AsyncSession = Depends(get_db_session),
):
    """
    Get optimization recommendations for content.

    Args:
        context_id: ACD context identifier
        target_score: Target engagement score
    """
    try:
        from sqlalchemy import select

        from backend.models.acd import ACDContextModel

        stmt = select(ACDContextModel).where(ACDContextModel.id == context_id)
        result = await db.execute(stmt)
        context_model = result.scalar_one_or_none()

        if not context_model:
            raise HTTPException(status_code=404, detail="Context not found")

        scoring_service = PredictiveEngagementScoringService(db)
        recommendations = await scoring_service.optimize_content(
            context_model, target_score
        )

        return recommendations
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Content optimization failed: {e}")
        raise HTTPException(status_code=500, detail="Content optimization failed")


@router.get("/predict-timing")
async def predict_optimal_timing(
    persona_id: Optional[UUID] = Query(None),
    platform: str = Query("instagram"),
    lookback_days: int = Query(30, ge=7, le=90),
    db: AsyncSession = Depends(get_db_session),
):
    """
    Predict optimal posting times.

    Args:
        persona_id: Optional persona filter
        platform: Social platform
        lookback_days: Days of historical data
    """
    try:
        scoring_service = PredictiveEngagementScoringService(db)
        prediction = await scoring_service.predict_optimal_posting_time(
            persona_id, platform, lookback_days
        )

        return prediction
    except Exception as e:
        logger.error(f"Timing prediction failed: {e}")
        raise HTTPException(status_code=500, detail="Timing prediction failed")


# Cross-Persona Learning Endpoints
@router.get("/cross-persona/aggregate")
async def aggregate_cross_persona_patterns(
    platform: str = Query("instagram"),
    min_personas: int = Query(5, ge=3, le=50),
    lookback_days: int = Query(30, ge=7, le=90),
    db: AsyncSession = Depends(get_db_session),
):
    """
    Get aggregated patterns across personas with privacy preservation.

    Args:
        platform: Social platform
        min_personas: Minimum personas required
        lookback_days: Days of historical data
    """
    try:
        service = CrossPersonaLearningService(db)
        patterns = await service.aggregate_engagement_patterns(
            platform, min_personas, lookback_days
        )

        return patterns
    except Exception as e:
        logger.error(f"Cross-persona aggregation failed: {e}")
        raise HTTPException(status_code=500, detail="Cross-persona aggregation failed")


@router.get("/cross-persona/benchmark/{persona_id}")
async def benchmark_persona_performance(
    persona_id: UUID,
    platform: str = Query("instagram"),
    metric: str = Query("engagement_rate"),
    db: AsyncSession = Depends(get_db_session),
):
    """
    Benchmark persona performance against aggregated data.

    Args:
        persona_id: Persona identifier
        platform: Social platform
        metric: Performance metric
    """
    try:
        service = CrossPersonaLearningService(db)
        benchmark = await service.get_benchmarked_performance(
            persona_id, platform, metric
        )

        return benchmark
    except Exception as e:
        logger.error(f"Benchmarking failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/cross-persona/privacy-report")
async def get_privacy_report(
    db: AsyncSession = Depends(get_db_session),
):
    """Get privacy compliance report for cross-persona learning."""
    try:
        service = CrossPersonaLearningService(db)
        report = await service.get_privacy_report()

        return report
    except Exception as e:
        logger.error(f"Privacy report generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# A/B Testing Endpoints
@router.post("/ab-tests/create")
async def create_ab_test(
    test_name: str,
    variants: list,
    success_metric: str = "engagement_rate",
    description: Optional[str] = None,
    minimum_sample_size: int = 100,
    minimum_runtime_hours: int = 24,
    db: AsyncSession = Depends(get_db_session),
):
    """
    Create a new A/B test.

    Args:
        test_name: Name of the test
        variants: List of variant configurations
        success_metric: Metric to optimize
        description: Optional description
        minimum_sample_size: Minimum samples per variant
        minimum_runtime_hours: Minimum test duration
    """
    try:
        service = ABTestingService(db)
        config = await service.create_test(
            test_name=test_name,
            variants=variants,
            success_metric=success_metric,
            description=description,
            minimum_sample_size=minimum_sample_size,
            minimum_runtime_hours=minimum_runtime_hours,
        )

        return config
    except Exception as e:
        logger.error(f"A/B test creation failed: {e}")
        raise HTTPException(status_code=500, detail="A/B test creation failed")


@router.post("/ab-tests/{test_id}/start")
async def start_ab_test(
    test_id: UUID,
    db: AsyncSession = Depends(get_db_session),
):
    """Start an A/B test."""
    try:
        service = ABTestingService(db)
        success = await service.start_test(test_id)

        if not success:
            raise HTTPException(status_code=404, detail="Test not found")

        return {"success": True, "message": "Test started"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Test start failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/ab-tests/{test_id}/status")
async def get_ab_test_status(
    test_id: UUID,
    db: AsyncSession = Depends(get_db_session),
):
    """Get A/B test status."""
    try:
        service = ABTestingService(db)
        status = await service.get_test_status(test_id)

        if "error" in status:
            raise HTTPException(status_code=404, detail=status["error"])

        return status
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Status check failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/ab-tests/{test_id}/analyze")
async def analyze_ab_test(
    test_id: UUID,
    auto_select_winner: bool = True,
    db: AsyncSession = Depends(get_db_session),
):
    """
    Analyze A/B test results.

    Args:
        test_id: Test identifier
        auto_select_winner: Automatically select winning variant
    """
    try:
        service = ABTestingService(db)
        result = await service.analyze_test(test_id, auto_select_winner)

        return result
    except Exception as e:
        logger.error(f"Test analysis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/ab-tests/")
async def list_ab_tests(
    db: AsyncSession = Depends(get_db_session),
):
    """List all A/B tests."""
    try:
        service = ABTestingService(db)
        tests = await service.get_all_tests()

        return tests
    except Exception as e:
        logger.error(f"List tests failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
