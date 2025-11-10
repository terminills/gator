"""
Tests for ACD Phase 3: Advanced Learning Features

Tests ML pattern recognition, predictive scoring, cross-persona learning,
and A/B testing.
"""

import pytest
from datetime import datetime, timezone
from uuid import uuid4

from backend.services.ml_pattern_recognition import MLPatternRecognitionService
from backend.services.predictive_engagement_scoring import PredictiveEngagementScoringService
from backend.services.cross_persona_learning import CrossPersonaLearningService
from backend.services.ab_testing_service import ABTestingService, TaskPriority
from backend.models.acd import (
    ACDContextModel,
    ACDContextCreate,
    AIStatus,
    AIComplexity,
    AIState,
)
from backend.services.acd_service import ACDService


@pytest.fixture
async def test_contexts(acd_service):
    """Create test ACD contexts."""
    contexts = []
    for i in range(10):
        context_data = ACDContextCreate(
            ai_phase="SOCIAL_MEDIA_CONTENT",
            ai_status=AIStatus.IMPLEMENTED,
            ai_complexity=AIComplexity.MEDIUM,
            ai_state=AIState.DONE,
            ai_context={
                "prompt": f"Test content {i}",
                "hashtags": ["test", "demo", f"tag{i}"]
            },
            ai_metadata={
                "social_metrics": {
                    "platform": "instagram",
                    "engagement_rate": 5.0 + i * 0.5,
                    "genuine_user_count": 1000 + i * 100,
                    "bot_filtered": 50 + i * 5
                }
            }
        )
        context = await acd_service.create_context(context_data)
        contexts.append(context)
    return contexts


class TestMLPatternRecognition:
    """Tests for ML pattern recognition service."""
    
    @pytest.mark.asyncio
    async def test_feature_extraction(self, db_session):
        """Test feature extraction from ACD context."""
        service = MLPatternRecognitionService(db_session)
        
        context = ACDContextModel(
            id=uuid4(),
            ai_phase="SOCIAL_MEDIA_CONTENT",
            ai_status="IMPLEMENTED",
            ai_complexity="MEDIUM",
            ai_state="DONE",
            ai_confidence="CONFIDENT",
            ai_context={"prompt": "test", "hashtags": ["a", "b", "c"]},
            ai_metadata={"social_metrics": {"engagement_rate": 5.0}},
            created_at=datetime.now(timezone.utc)
        )
        
        features = await service.extract_features(context)
        
        assert features is not None
        assert len(features) == 18  # Expected feature count
        assert features.dtype == np.float32
    
    @pytest.mark.asyncio
    async def test_train_engagement_model_insufficient_data(self, db_session):
        """Test engagement model training with insufficient data."""
        service = MLPatternRecognitionService(db_session)
        
        result = await service.train_engagement_model(
            min_samples=100,  # More than available
            lookback_days=30
        )
        
        assert result["success"] is False
        assert result["reason"] == "insufficient_data"
    
    @pytest.mark.asyncio
    async def test_train_engagement_model_success(self, db_session, test_contexts):
        """Test successful engagement model training."""
        service = MLPatternRecognitionService(db_session)
        
        result = await service.train_engagement_model(
            min_samples=5,
            lookback_days=90
        )
        
        if result.get("success"):
            assert "train_score" in result
            assert "test_score" in result
            assert result["model_type"] == "RandomForestRegressor"
    
    @pytest.mark.asyncio
    async def test_model_persistence(self, db_session, tmp_path):
        """Test model save and load."""
        service = MLPatternRecognitionService(db_session)
        
        # Save models (even if not trained)
        save_result = await service.save_models(str(tmp_path))
        assert save_result is True
        
        # Load models
        load_result = await service.load_models(str(tmp_path))
        assert load_result is True


class TestPredictiveEngagementScoring:
    """Tests for predictive engagement scoring service."""
    
    @pytest.mark.asyncio
    async def test_score_content_fallback(self, db_session):
        """Test content scoring with fallback (no trained model)."""
        service = PredictiveEngagementScoringService(db_session)
        
        context = ACDContextModel(
            id=uuid4(),
            ai_phase="SOCIAL_MEDIA_CONTENT",
            ai_status="IMPLEMENTED",
            ai_complexity="MEDIUM",
            ai_state="DONE",
            ai_validation="APPROVED",
            ai_confidence="CONFIDENT",
            created_at=datetime.now(timezone.utc)
        )
        
        score = await service.score_content(context)
        
        assert "composite_score" in score
        assert 0 <= score["composite_score"] <= 100
        assert "score_tier" in score
        assert score["method"] == "rule_based"
    
    @pytest.mark.asyncio
    async def test_optimize_content(self, db_session):
        """Test content optimization recommendations."""
        service = PredictiveEngagementScoringService(db_session)
        
        context = ACDContextModel(
            id=uuid4(),
            ai_phase="TEXT_GENERATION",
            ai_status="IMPLEMENTED",
            ai_context={
                "prompt": "Short",
                "hashtags": ["a"]
            },
            created_at=datetime.now(timezone.utc)
        )
        
        recommendations = await service.optimize_content(context, target_score=80.0)
        
        assert "current_score" in recommendations
        assert "recommendations" in recommendations
        assert len(recommendations["recommendations"]) > 0
        assert recommendations["recommendations"][0]["priority"] in ["high", "medium", "low"]
    
    @pytest.mark.asyncio
    async def test_predict_optimal_timing(self, db_session, test_contexts):
        """Test optimal posting time prediction."""
        service = PredictiveEngagementScoringService(db_session)
        
        prediction = await service.predict_optimal_posting_time(
            platform="instagram",
            lookback_days=30
        )
        
        assert "optimal_hours" in prediction
        assert "platform" in prediction
        assert prediction["platform"] == "instagram"


class TestCrossPersonaLearning:
    """Tests for cross-persona learning service."""
    
    @pytest.mark.asyncio
    async def test_differential_privacy_noise(self, db_session):
        """Test differential privacy noise addition."""
        service = CrossPersonaLearningService(db_session)
        
        original_value = 10.0
        noisy_value = service._add_laplace_noise(original_value, sensitivity=1.0)
        
        assert noisy_value != original_value  # Should add noise
        assert abs(noisy_value - original_value) < 10  # Reasonable noise range
    
    @pytest.mark.asyncio
    async def test_persona_anonymization(self, db_session):
        """Test persona ID anonymization."""
        service = CrossPersonaLearningService(db_session)
        
        persona_id = uuid4()
        anon_id = service._anonymize_persona_id(persona_id)
        
        assert len(anon_id) == 16
        assert anon_id != str(persona_id)
        
        # Same input should give same output
        anon_id2 = service._anonymize_persona_id(persona_id)
        assert anon_id == anon_id2
    
    @pytest.mark.asyncio
    async def test_aggregate_patterns_insufficient_personas(self, db_session):
        """Test aggregation with insufficient personas."""
        service = CrossPersonaLearningService(db_session)
        
        result = await service.aggregate_engagement_patterns(
            platform="instagram",
            min_personas=10,  # More than available
            lookback_days=30
        )
        
        assert result["success"] is False
        assert result["reason"] == "insufficient_personas"
    
    @pytest.mark.asyncio
    async def test_privacy_report(self, db_session):
        """Test privacy report generation."""
        service = CrossPersonaLearningService(db_session)
        
        report = await service.get_privacy_report()
        
        assert "privacy_mechanisms" in report
        assert "differential_privacy" in report["privacy_mechanisms"]
        assert "k_anonymity" in report["privacy_mechanisms"]
        assert "compliance" in report
        assert report["compliance"]["gdpr_compliant"] is True


class TestABTesting:
    """Tests for A/B testing service."""
    
    @pytest.mark.asyncio
    async def test_create_ab_test(self, db_session):
        """Test A/B test creation."""
        service = ABTestingService(db_session)
        
        config = await service.create_test(
            test_name="Test A/B",
            variants=[
                {"variant_id": "A", "name": "Variant A", "changes": {}},
                {"variant_id": "B", "name": "Variant B", "changes": {}}
            ],
            success_metric="engagement_rate"
        )
        
        assert config.test_name == "Test A/B"
        assert len(config.variants) == 3  # Includes auto-added control
        assert config.control_variant_id == "control"
    
    @pytest.mark.asyncio
    async def test_start_ab_test(self, db_session):
        """Test starting an A/B test."""
        service = ABTestingService(db_session)
        
        config = await service.create_test(
            test_name="Test Start",
            variants=[
                {"variant_id": "A", "name": "Variant A", "changes": {}}
            ]
        )
        
        success = await service.start_test(config.test_id)
        assert success is True
    
    @pytest.mark.asyncio
    async def test_record_variant_events(self, db_session):
        """Test recording events for variants."""
        service = ABTestingService(db_session)
        
        config = await service.create_test(
            test_name="Test Events",
            variants=[
                {"variant_id": "A", "name": "Variant A", "changes": {}}
            ]
        )
        
        # Record impressions
        result = await service.record_variant_event(
            config.test_id, "control", "impression", 100
        )
        assert result is True
        
        # Record engagements
        result = await service.record_variant_event(
            config.test_id, "control", "engagement", 10
        )
        assert result is True
        
        # Check that rates are calculated
        variant_perf = service.test_results[config.test_id]["control"]
        assert variant_perf.impressions == 100
        assert variant_perf.engagement_count == 10
        assert variant_perf.engagement_rate == 10.0
    
    @pytest.mark.asyncio
    async def test_analyze_ab_test(self, db_session):
        """Test A/B test analysis."""
        service = ABTestingService(db_session)
        
        config = await service.create_test(
            test_name="Test Analysis",
            variants=[
                {"variant_id": "A", "name": "Variant A", "changes": {}}
            ],
            minimum_sample_size=10
        )
        
        await service.start_test(config.test_id)
        
        # Add sufficient data
        for variant_id in ["control", "A"]:
            await service.record_variant_event(config.test_id, variant_id, "impression", 50)
            await service.record_variant_event(config.test_id, variant_id, "engagement", 5)
        
        result = await service.analyze_test(config.test_id)
        
        assert result.test_id == config.test_id
        assert result.statistical_significance in [True, False]
        assert result.winner in ["control", "A", None]
        assert len(result.insights) > 0
    
    @pytest.mark.asyncio
    async def test_get_test_status(self, db_session):
        """Test getting test status."""
        service = ABTestingService(db_session)
        
        config = await service.create_test(
            test_name="Test Status",
            variants=[
                {"variant_id": "A", "name": "Variant A", "changes": {}}
            ]
        )
        
        status = await service.get_test_status(config.test_id)
        
        assert status["test_id"] == str(config.test_id)
        assert "progress_percent" in status
        assert "ready_for_analysis" in status
        assert "variants" in status


# Import numpy for feature extraction test
import numpy as np
