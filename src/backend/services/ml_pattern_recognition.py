"""
ML-Based Pattern Recognition Service

Implements machine learning models for content performance prediction and pattern extraction.
Uses scikit-learn for lightweight models with minimal dependencies.
"""

from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional

import joblib
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sqlalchemy import and_, select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.config.logging import get_logger
from backend.models.acd import ACDContextModel

logger = get_logger(__name__)


class MLPatternRecognitionService:
    """
    Service for ML-based pattern recognition in content generation.

    Features:
    - Content performance prediction
    - Success pattern extraction
    - Feature importance analysis
    - Model training and persistence
    """

    def __init__(self, db_session: AsyncSession):
        """
        Initialize ML pattern recognition service.

        Args:
            db_session: Database session for data access
        """
        self.db = db_session
        self.engagement_model = None
        self.success_classifier = None
        self.scaler = StandardScaler()
        self.feature_names = []

    async def extract_features(self, context: ACDContextModel) -> Optional[np.ndarray]:
        """
        Extract feature vector from ACD context for ML models.

        Args:
            context: ACD context record

        Returns:
            Feature vector or None if insufficient data
        """
        try:
            features = []

            # Temporal features
            hour_of_day = context.created_at.hour if context.created_at else 12
            day_of_week = context.created_at.weekday() if context.created_at else 0
            features.extend([hour_of_day, day_of_week])

            # Complexity encoding
            complexity_map = {"LOW": 1, "MEDIUM": 2, "HIGH": 3, "CRITICAL": 4}
            complexity = complexity_map.get(context.ai_complexity or "MEDIUM", 2)
            features.append(complexity)

            # State encoding
            state_map = {
                "READY": 1,
                "PROCESSING": 2,
                "DONE": 3,
                "BLOCKED": 4,
                "FAILED": 5,
                "CANCELLED": 6,
            }
            state = state_map.get(context.ai_state or "READY", 1)
            features.append(state)

            # Confidence encoding
            confidence_map = {
                "CONFIDENT": 4,
                "VALIDATED": 5,
                "UNCERTAIN": 2,
                "HYPOTHESIS": 1,
                "EXPERIMENTAL": 1,
            }
            confidence = confidence_map.get(context.ai_confidence or "UNCERTAIN", 2)
            features.append(confidence)

            # Metadata features
            metadata = context.ai_metadata or {}

            # Social engagement metrics (if available)
            social_metrics = metadata.get("social_metrics", {})
            engagement_rate = social_metrics.get("engagement_rate", 0.0)
            genuine_user_count = social_metrics.get("genuine_user_count", 0)
            bot_filtered = social_metrics.get("bot_filtered", 0)

            features.extend(
                [
                    engagement_rate,
                    np.log1p(genuine_user_count),  # Log transform for count features
                    np.log1p(bot_filtered),
                ]
            )

            # Content features
            context_data = context.ai_context or {}
            prompt_length = len(str(context_data.get("prompt", "")))
            has_hashtags = int(bool(context_data.get("hashtags", [])))
            num_hashtags = len(context_data.get("hashtags", []))

            features.extend([np.log1p(prompt_length), has_hashtags, num_hashtags])

            # Assignment features
            has_assignment = int(bool(context.ai_assigned_to))
            has_handoff = int(context.ai_handoff_requested or False)

            features.extend([has_assignment, has_handoff])

            # Validation features
            validation_map = {
                "APPROVED": 5,
                "CONDITIONALLY_APPROVED": 3,
                "REJECTED": 1,
                "PENDING": 2,
                "ANALYZED": 4,
            }
            validation = validation_map.get(context.ai_validation or "PENDING", 2)
            features.append(validation)

            # Phase encoding (one-hot for major phases)
            phase = context.ai_phase or ""
            is_image_gen = int("IMAGE" in phase.upper())
            is_text_gen = int("TEXT" in phase.upper())
            is_social = int("SOCIAL" in phase.upper())
            is_video = int("VIDEO" in phase.upper())

            features.extend([is_image_gen, is_text_gen, is_social, is_video])

            return np.array(features, dtype=np.float32)

        except Exception as e:
            logger.error(f"Feature extraction failed: {e}")
            return None

    async def train_engagement_model(
        self, min_samples: int = 50, lookback_days: int = 90
    ) -> Dict[str, Any]:
        """
        Train engagement prediction model from historical data.

        Args:
            min_samples: Minimum samples required for training
            lookback_days: Days of historical data to use

        Returns:
            Training results with metrics
        """
        try:
            cutoff_time = datetime.now(timezone.utc) - timedelta(days=lookback_days)

            # Fetch training data
            stmt = select(ACDContextModel).where(
                and_(
                    ACDContextModel.created_at >= cutoff_time,
                    ACDContextModel.ai_state == "DONE",
                    ACDContextModel.ai_metadata.isnot(None),
                )
            )

            result = await self.db.execute(stmt)
            contexts = result.scalars().all()

            if len(contexts) < min_samples:
                logger.warning(
                    f"Insufficient training data: {len(contexts)} < {min_samples}"
                )
                return {
                    "success": False,
                    "reason": "insufficient_data",
                    "samples": len(contexts),
                }

            # Extract features and labels
            X_list = []
            y_list = []

            for context in contexts:
                features = await self.extract_features(context)
                if features is None:
                    continue

                # Target: engagement rate
                metadata = context.ai_metadata or {}
                social_metrics = metadata.get("social_metrics", {})
                engagement_rate = social_metrics.get("engagement_rate", 0.0)

                # Only use samples with engagement data
                if engagement_rate > 0:
                    X_list.append(features)
                    y_list.append(engagement_rate)

            if len(X_list) < min_samples:
                return {
                    "success": False,
                    "reason": "insufficient_engagement_data",
                    "samples_with_engagement": len(X_list),
                }

            X = np.array(X_list)
            y = np.array(y_list)

            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )

            # Scale features
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)

            # Train model
            self.engagement_model = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                random_state=42,
                n_jobs=-1,
            )

            self.engagement_model.fit(X_train_scaled, y_train)

            # Evaluate
            train_score = self.engagement_model.score(X_train_scaled, y_train)
            test_score = self.engagement_model.score(X_test_scaled, y_test)

            # Feature importance
            feature_importance = self.engagement_model.feature_importances_

            logger.info(
                f"Trained engagement model: "
                f"train_R2={train_score:.3f}, test_R2={test_score:.3f}"
            )

            return {
                "success": True,
                "model_type": "RandomForestRegressor",
                "train_samples": len(X_train),
                "test_samples": len(X_test),
                "train_score": float(train_score),
                "test_score": float(test_score),
                "feature_importance": feature_importance.tolist(),
                "trained_at": datetime.now(timezone.utc).isoformat(),
            }

        except Exception as e:
            logger.error(f"Model training failed: {e}")
            return {"success": False, "error": str(e)}

    async def predict_engagement(
        self, context: ACDContextModel
    ) -> Optional[Dict[str, Any]]:
        """
        Predict engagement rate for a content generation context.

        Args:
            context: ACD context to predict for

        Returns:
            Prediction with confidence intervals
        """
        try:
            if self.engagement_model is None:
                logger.warning("Engagement model not trained")
                return None

            features = await self.extract_features(context)
            if features is None:
                return None

            # Scale and predict
            features_scaled = self.scaler.transform(features.reshape(1, -1))

            # Get predictions from all trees for confidence interval
            predictions = np.array(
                [
                    tree.predict(features_scaled)[0]
                    for tree in self.engagement_model.estimators_
                ]
            )

            mean_prediction = predictions.mean()
            std_prediction = predictions.std()

            # 95% confidence interval
            ci_lower = max(0, mean_prediction - 1.96 * std_prediction)
            ci_upper = mean_prediction + 1.96 * std_prediction

            return {
                "predicted_engagement_rate": float(mean_prediction),
                "confidence_interval_lower": float(ci_lower),
                "confidence_interval_upper": float(ci_upper),
                "confidence_std": float(std_prediction),
                "model_confidence": "high" if std_prediction < 1.0 else "medium",
            }

        except Exception as e:
            logger.error(f"Engagement prediction failed: {e}")
            return None

    async def train_success_classifier(
        self,
        success_threshold: float = 5.0,
        min_samples: int = 50,
        lookback_days: int = 90,
    ) -> Dict[str, Any]:
        """
        Train binary classifier for content success prediction.

        Args:
            success_threshold: Engagement rate threshold for success
            min_samples: Minimum samples required
            lookback_days: Days of historical data

        Returns:
            Training results
        """
        try:
            cutoff_time = datetime.now(timezone.utc) - timedelta(days=lookback_days)

            # Fetch training data
            stmt = select(ACDContextModel).where(
                and_(
                    ACDContextModel.created_at >= cutoff_time,
                    ACDContextModel.ai_state == "DONE",
                    ACDContextModel.ai_metadata.isnot(None),
                )
            )

            result = await self.db.execute(stmt)
            contexts = result.scalars().all()

            if len(contexts) < min_samples:
                return {
                    "success": False,
                    "reason": "insufficient_data",
                    "samples": len(contexts),
                }

            # Extract features and labels
            X_list = []
            y_list = []

            for context in contexts:
                features = await self.extract_features(context)
                if features is None:
                    continue

                metadata = context.ai_metadata or {}
                social_metrics = metadata.get("social_metrics", {})
                engagement_rate = social_metrics.get("engagement_rate", 0.0)

                if engagement_rate > 0:
                    X_list.append(features)
                    # Binary label: success if above threshold
                    y_list.append(int(engagement_rate >= success_threshold))

            if len(X_list) < min_samples:
                return {"success": False, "reason": "insufficient_engagement_data"}

            X = np.array(X_list)
            y = np.array(y_list)

            # Check class balance
            success_rate = y.mean()
            if success_rate < 0.1 or success_rate > 0.9:
                logger.warning(f"Imbalanced classes: success_rate={success_rate:.2f}")

            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )

            # Scale features
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)

            # Train classifier
            self.success_classifier = GradientBoostingClassifier(
                n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42
            )

            self.success_classifier.fit(X_train_scaled, y_train)

            # Evaluate
            train_accuracy = self.success_classifier.score(X_train_scaled, y_train)
            test_accuracy = self.success_classifier.score(X_test_scaled, y_test)

            logger.info(
                f"Trained success classifier: "
                f"train_acc={train_accuracy:.3f}, test_acc={test_accuracy:.3f}"
            )

            return {
                "success": True,
                "model_type": "GradientBoostingClassifier",
                "success_threshold": success_threshold,
                "train_samples": len(X_train),
                "test_samples": len(X_test),
                "train_accuracy": float(train_accuracy),
                "test_accuracy": float(test_accuracy),
                "class_balance": float(success_rate),
                "trained_at": datetime.now(timezone.utc).isoformat(),
            }

        except Exception as e:
            logger.error(f"Classifier training failed: {e}")
            return {"success": False, "error": str(e)}

    async def predict_success_probability(
        self, context: ACDContextModel
    ) -> Optional[Dict[str, Any]]:
        """
        Predict probability of content success.

        Args:
            context: ACD context to predict for

        Returns:
            Success probability and prediction
        """
        try:
            if self.success_classifier is None:
                logger.warning("Success classifier not trained")
                return None

            features = await self.extract_features(context)
            if features is None:
                return None

            features_scaled = self.scaler.transform(features.reshape(1, -1))

            # Get probability predictions
            proba = self.success_classifier.predict_proba(features_scaled)[0]
            prediction = self.success_classifier.predict(features_scaled)[0]

            return {
                "success_probability": float(proba[1]),
                "failure_probability": float(proba[0]),
                "predicted_success": bool(prediction),
                "confidence": "high" if abs(proba[1] - 0.5) > 0.3 else "medium",
            }

        except Exception as e:
            logger.error(f"Success prediction failed: {e}")
            return None

    async def save_models(self, path_prefix: str = "ml_models") -> bool:
        """
        Save trained models to disk.

        Args:
            path_prefix: Directory prefix for model files

        Returns:
            Success status
        """
        try:
            import os

            os.makedirs(path_prefix, exist_ok=True)

            if self.engagement_model:
                joblib.dump(
                    self.engagement_model, f"{path_prefix}/engagement_model.joblib"
                )

            if self.success_classifier:
                joblib.dump(
                    self.success_classifier, f"{path_prefix}/success_classifier.joblib"
                )

            joblib.dump(self.scaler, f"{path_prefix}/scaler.joblib")

            logger.info(f"Models saved to {path_prefix}")
            return True

        except Exception as e:
            logger.error(f"Model save failed: {e}")
            return False

    async def load_models(self, path_prefix: str = "ml_models") -> bool:
        """
        Load trained models from disk.

        Args:
            path_prefix: Directory prefix for model files

        Returns:
            Success status
        """
        try:
            import os

            engagement_path = f"{path_prefix}/engagement_model.joblib"
            classifier_path = f"{path_prefix}/success_classifier.joblib"
            scaler_path = f"{path_prefix}/scaler.joblib"

            if os.path.exists(engagement_path):
                self.engagement_model = joblib.load(engagement_path)

            if os.path.exists(classifier_path):
                self.success_classifier = joblib.load(classifier_path)

            if os.path.exists(scaler_path):
                self.scaler = joblib.load(scaler_path)

            logger.info(f"Models loaded from {path_prefix}")
            return True

        except Exception as e:
            logger.error(f"Model load failed: {e}")
            return False

    async def analyze_feature_importance(self) -> Optional[Dict[str, Any]]:
        """
        Analyze which features are most important for predictions.

        Returns:
            Feature importance analysis
        """
        try:
            feature_names = [
                "hour_of_day",
                "day_of_week",
                "complexity",
                "state",
                "confidence",
                "engagement_rate",
                "genuine_users_log",
                "bots_filtered_log",
                "prompt_length_log",
                "has_hashtags",
                "num_hashtags",
                "has_assignment",
                "has_handoff",
                "validation",
                "is_image",
                "is_text",
                "is_social",
                "is_video",
            ]

            analysis = {}

            if self.engagement_model:
                importance = self.engagement_model.feature_importances_
                engagement_features = list(zip(feature_names, importance))
                engagement_features.sort(key=lambda x: x[1], reverse=True)

                analysis["engagement_model"] = {
                    "top_features": [
                        {"feature": name, "importance": float(imp)}
                        for name, imp in engagement_features[:10]
                    ],
                    "all_features": [
                        {"feature": name, "importance": float(imp)}
                        for name, imp in engagement_features
                    ],
                }

            if self.success_classifier:
                importance = self.success_classifier.feature_importances_
                classifier_features = list(zip(feature_names, importance))
                classifier_features.sort(key=lambda x: x[1], reverse=True)

                analysis["success_classifier"] = {
                    "top_features": [
                        {"feature": name, "importance": float(imp)}
                        for name, imp in classifier_features[:10]
                    ],
                    "all_features": [
                        {"feature": name, "importance": float(imp)}
                        for name, imp in classifier_features
                    ],
                }

            return analysis

        except Exception as e:
            logger.error(f"Feature importance analysis failed: {e}")
            return None
