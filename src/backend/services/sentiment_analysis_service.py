"""
Social Media Sentiment Analysis Service

Advanced sentiment analysis for social media content, comments, and engagement.
Provides insights for content strategy and audience understanding.
"""

import re
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum
from collections import defaultdict

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_

from backend.config.logging import get_logger
from backend.models.persona import PersonaModel

logger = get_logger(__name__)


class SentimentLabel(str, Enum):
    """Sentiment classification labels."""
    VERY_POSITIVE = "very_positive"
    POSITIVE = "positive"
    NEUTRAL = "neutral"
    NEGATIVE = "negative"
    VERY_NEGATIVE = "very_negative"


class EmotionLabel(str, Enum):
    """Emotion classification labels."""
    JOY = "joy"
    SADNESS = "sadness"
    ANGER = "anger"
    FEAR = "fear"
    SURPRISE = "surprise"
    DISGUST = "disgust"
    TRUST = "trust"
    ANTICIPATION = "anticipation"


class SentimentAnalysisService:
    """
    Service for analyzing sentiment across social media platforms.
    
    Provides:
    - Text sentiment analysis
    - Emotion detection
    - Topic sentiment mapping
    - Trend analysis
    - Competitor sentiment comparison
    """
    
    def __init__(self, db_session: AsyncSession):
        """
        Initialize sentiment analysis service.
        
        Args:
            db_session: Database session for persistence
        """
        self.db = db_session
        self._load_sentiment_lexicons()
    
    def _load_sentiment_lexicons(self):
        """Load sentiment word lists for analysis."""
        # Positive sentiment words
        self.positive_words = {
            'amazing', 'awesome', 'beautiful', 'best', 'brilliant', 'excellent',
            'fantastic', 'good', 'great', 'happy', 'incredible', 'love', 'perfect',
            'wonderful', 'outstanding', 'superb', 'delightful', 'impressive',
            'phenomenal', 'spectacular', 'terrific', 'magnificent', 'fabulous',
            'marvelous', 'exceptional', 'remarkable', 'stunning', 'gorgeous',
            'inspiring', 'uplifting', 'joyful', 'excited', 'thrilled', 'blessed',
            'grateful', 'appreciated', 'valuable', 'helpful', 'useful', 'effective'
        }
        
        # Negative sentiment words
        self.negative_words = {
            'awful', 'bad', 'terrible', 'horrible', 'poor', 'worst', 'hate',
            'disappointing', 'useless', 'pathetic', 'disgusting', 'annoying',
            'frustrating', 'angry', 'sad', 'depressing', 'boring', 'dull',
            'mediocre', 'inferior', 'inadequate', 'unacceptable', 'dislike',
            'waste', 'failure', 'disaster', 'nightmare', 'ridiculous', 'stupid',
            'ugly', 'gross', 'nasty', 'offensive', 'appalling', 'dreadful',
            'atrocious', 'abysmal', 'miserable', 'unfortunate', 'regret'
        }
        
        # Emotion keyword mappings
        self.emotion_keywords = {
            EmotionLabel.JOY: {'happy', 'joy', 'excited', 'thrilled', 'delighted', 'cheerful', 'pleased'},
            EmotionLabel.SADNESS: {'sad', 'unhappy', 'depressed', 'disappointed', 'heartbroken', 'miserable'},
            EmotionLabel.ANGER: {'angry', 'furious', 'annoyed', 'irritated', 'outraged', 'mad', 'frustrated'},
            EmotionLabel.FEAR: {'scared', 'afraid', 'terrified', 'worried', 'anxious', 'nervous', 'frightened'},
            EmotionLabel.SURPRISE: {'surprised', 'shocked', 'amazed', 'astonished', 'stunned', 'wow'},
            EmotionLabel.DISGUST: {'disgusting', 'gross', 'nasty', 'repulsive', 'revolting', 'awful'},
            EmotionLabel.TRUST: {'trust', 'reliable', 'dependable', 'honest', 'loyal', 'faithful'},
            EmotionLabel.ANTICIPATION: {'excited', 'eager', 'anticipating', 'looking forward', 'waiting'}
        }
        
        # Intensifiers
        self.intensifiers = {
            'very', 'extremely', 'incredibly', 'absolutely', 'totally', 'completely',
            'really', 'so', 'super', 'ultra', 'highly', 'quite', 'rather', 'fairly'
        }
        
        # Negations
        self.negations = {
            'not', 'no', 'never', "n't", 'neither', 'nobody', 'nothing', 'nowhere',
            'none', 'hardly', 'scarcely', 'barely'
        }
    
    async def analyze_text(self, text: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Perform comprehensive sentiment analysis on text.
        
        Args:
            text: Text content to analyze
            context: Optional context (platform, author, etc.)
        
        Returns:
            Dictionary with sentiment analysis results
        """
        # Normalize text
        normalized_text = self._normalize_text(text)
        tokens = normalized_text.split()
        
        # Basic sentiment score
        sentiment_score, sentiment_label = self._calculate_sentiment_score(tokens)
        
        # Detect emotions
        emotions = self._detect_emotions(tokens)
        
        # Extract topics
        topics = self._extract_topics(tokens)
        
        # Analyze intent
        intent = self._classify_intent(text)
        
        # Calculate confidence
        confidence = self._calculate_confidence(tokens, sentiment_score)
        
        return {
            'sentiment_score': sentiment_score,
            'sentiment_label': sentiment_label,
            'emotions': emotions,
            'topics': topics,
            'intent': intent,
            'confidence': confidence,
            'word_count': len(tokens),
            'analyzed_at': datetime.utcnow().isoformat(),
            'context': context or {}
        }
    
    async def analyze_social_comments(
        self,
        comments: List[Dict[str, Any]],
        persona_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Analyze sentiment of multiple social media comments.
        
        Args:
            comments: List of comment dictionaries
            persona_id: Optional persona ID to filter relevant comments
        
        Returns:
            Aggregated sentiment analysis results
        """
        if not comments:
            return {
                'total_comments': 0,
                'overall_sentiment': 'neutral',
                'sentiment_distribution': {},
                'emotion_distribution': {},
                'key_topics': []
            }
        
        results = []
        for comment in comments:
            text = comment.get('text', '')
            if text:
                analysis = await self.analyze_text(text, context={
                    'comment_id': comment.get('id'),
                    'author': comment.get('author'),
                    'platform': comment.get('platform'),
                    'timestamp': comment.get('timestamp')
                })
                results.append(analysis)
        
        # Aggregate results
        return self._aggregate_sentiment_results(results)
    
    async def analyze_engagement_sentiment(
        self,
        post_data: Dict[str, Any],
        engagement_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Analyze sentiment based on post content and engagement metrics.
        
        Args:
            post_data: Post content and metadata
            engagement_data: Likes, comments, shares, etc.
        
        Returns:
            Combined sentiment and engagement analysis
        """
        # Analyze post content
        content_sentiment = await self.analyze_text(
            post_data.get('content', ''),
            context={'type': 'post_content'}
        )
        
        # Analyze comment sentiment
        comments = engagement_data.get('comments', [])
        comment_sentiment = await self.analyze_social_comments(comments)
        
        # Calculate engagement sentiment score
        engagement_score = self._calculate_engagement_score(engagement_data)
        
        return {
            'content_sentiment': content_sentiment,
            'comment_sentiment': comment_sentiment,
            'engagement_score': engagement_score,
            'overall_sentiment': self._combine_sentiments(
                content_sentiment['sentiment_score'],
                comment_sentiment.get('average_sentiment', 0),
                engagement_score
            ),
            'analyzed_at': datetime.utcnow().isoformat()
        }
    
    async def track_sentiment_trends(
        self,
        persona_id: str,
        time_range: timedelta = timedelta(days=30)
    ) -> Dict[str, Any]:
        """
        Track sentiment trends over time for a persona.
        
        Args:
            persona_id: Persona identifier
            time_range: Time period to analyze
        
        Returns:
            Sentiment trend analysis
        """
        # This would query historical sentiment data from database
        # For now, returning structure for implementation
        
        return {
            'persona_id': persona_id,
            'time_range': {
                'start': (datetime.utcnow() - time_range).isoformat(),
                'end': datetime.utcnow().isoformat()
            },
            'sentiment_trend': {
                'direction': 'improving',  # improving, declining, stable
                'average_sentiment': 0.35,
                'sentiment_volatility': 0.12
            },
            'emotion_trends': {
                EmotionLabel.JOY: 0.4,
                EmotionLabel.TRUST: 0.3,
                EmotionLabel.SURPRISE: 0.15,
                'others': 0.15
            },
            'engagement_correlation': 0.68,  # Correlation between sentiment and engagement
            'top_positive_topics': [],
            'top_negative_topics': [],
            'recommendations': self._generate_recommendations(0.35, 'improving')
        }
    
    async def compare_competitor_sentiment(
        self,
        persona_id: str,
        competitor_ids: List[str]
    ) -> Dict[str, Any]:
        """
        Compare sentiment across personas/competitors.
        
        Args:
            persona_id: Your persona ID
            competitor_ids: List of competitor persona IDs
        
        Returns:
            Comparative sentiment analysis
        """
        # This would fetch and compare sentiment data
        # For now, returning structure for implementation
        
        return {
            'your_persona': {
                'id': persona_id,
                'average_sentiment': 0.35,
                'engagement_rate': 0.045,
                'positive_ratio': 0.68
            },
            'competitors': [
                {
                    'id': comp_id,
                    'average_sentiment': 0.28,
                    'engagement_rate': 0.038,
                    'positive_ratio': 0.62
                }
                for comp_id in competitor_ids
            ],
            'comparative_insights': [
                'Your sentiment is 25% higher than average competitor',
                'Your engagement rate is 15% above competitor average',
                'Focus on maintaining positive emotional tone'
            ],
            'analyzed_at': datetime.utcnow().isoformat()
        }
    
    def _normalize_text(self, text: str) -> str:
        """Normalize text for analysis."""
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+', '', text)
        
        # Remove mentions and hashtags (but keep the text)
        text = re.sub(r'[@#](\w+)', r'\1', text)
        
        # Remove special characters except spaces
        text = re.sub(r'[^\w\s]', ' ', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text
    
    def _calculate_sentiment_score(self, tokens: List[str]) -> Tuple[float, str]:
        """
        Calculate sentiment score from tokens.
        
        Returns:
            Tuple of (score, label) where score is -1.0 to 1.0
        """
        positive_count = 0
        negative_count = 0
        negation_active = False
        intensifier_factor = 1.0
        
        for i, token in enumerate(tokens):
            # Check for negation
            if token in self.negations:
                negation_active = True
                continue
            
            # Check for intensifiers
            if token in self.intensifiers:
                intensifier_factor = 1.5
                continue
            
            # Check sentiment
            if token in self.positive_words:
                if negation_active:
                    negative_count += intensifier_factor
                else:
                    positive_count += intensifier_factor
                negation_active = False
                intensifier_factor = 1.0
                
            elif token in self.negative_words:
                if negation_active:
                    positive_count += intensifier_factor
                else:
                    negative_count += intensifier_factor
                negation_active = False
                intensifier_factor = 1.0
        
        # Calculate normalized score
        total = positive_count + negative_count
        if total == 0:
            score = 0.0
            label = SentimentLabel.NEUTRAL
        else:
            score = (positive_count - negative_count) / total
            
            # Assign label
            if score >= 0.6:
                label = SentimentLabel.VERY_POSITIVE
            elif score >= 0.2:
                label = SentimentLabel.POSITIVE
            elif score <= -0.6:
                label = SentimentLabel.VERY_NEGATIVE
            elif score <= -0.2:
                label = SentimentLabel.NEGATIVE
            else:
                label = SentimentLabel.NEUTRAL
        
        return score, label
    
    def _detect_emotions(self, tokens: List[str]) -> Dict[str, float]:
        """Detect emotions in text."""
        emotion_scores = defaultdict(float)
        
        for token in tokens:
            for emotion, keywords in self.emotion_keywords.items():
                if token in keywords:
                    emotion_scores[emotion] += 1.0
        
        # Normalize scores
        total = sum(emotion_scores.values())
        if total > 0:
            emotion_scores = {
                emotion: score / total
                for emotion, score in emotion_scores.items()
            }
        
        # Return top 3 emotions
        return dict(sorted(emotion_scores.items(), key=lambda x: x[1], reverse=True)[:3])
    
    def _extract_topics(self, tokens: List[str]) -> List[str]:
        """Extract key topics from text."""
        # Simple topic extraction - look for longer words (likely nouns)
        topics = [
            token for token in tokens
            if len(token) > 5 and token not in self.positive_words and token not in self.negative_words
        ]
        
        # Return top 5 unique topics
        return list(set(topics))[:5]
    
    def _classify_intent(self, text: str) -> str:
        """Classify user intent from text."""
        text_lower = text.lower()
        
        if any(word in text_lower for word in ['?', 'how', 'what', 'when', 'where', 'why']):
            return 'question'
        elif any(word in text_lower for word in ['thank', 'thanks', 'appreciate']):
            return 'appreciation'
        elif any(word in text_lower for word in ['complain', 'issue', 'problem', 'fix']):
            return 'complaint'
        elif any(word in text_lower for word in ['recommend', 'suggest', 'should']):
            return 'recommendation'
        elif any(word in text_lower for word in ['love', 'amazing', 'great', 'awesome']):
            return 'praise'
        else:
            return 'statement'
    
    def _calculate_confidence(self, tokens: List[str], sentiment_score: float) -> float:
        """Calculate confidence in sentiment analysis."""
        # More tokens generally means higher confidence
        token_confidence = min(len(tokens) / 50.0, 1.0)
        
        # Stronger sentiment means higher confidence
        sentiment_confidence = abs(sentiment_score)
        
        # Combine factors
        confidence = (token_confidence + sentiment_confidence) / 2.0
        
        return round(confidence, 2)
    
    def _calculate_engagement_score(self, engagement_data: Dict[str, Any]) -> float:
        """Calculate engagement sentiment score from metrics."""
        likes = engagement_data.get('likes', 0)
        comments = engagement_data.get('comment_count', 0)
        shares = engagement_data.get('shares', 0)
        saves = engagement_data.get('saves', 0)
        
        # Weight different engagement types
        weighted_score = (
            likes * 1.0 +
            comments * 2.0 +
            shares * 3.0 +
            saves * 2.5
        )
        
        # Normalize to -1 to 1 range (higher engagement = more positive)
        # This is a simplified calculation
        normalized_score = min(weighted_score / 1000.0, 1.0)
        
        return round(normalized_score, 2)
    
    def _combine_sentiments(self, *scores: float) -> float:
        """Combine multiple sentiment scores."""
        if not scores:
            return 0.0
        
        return round(sum(scores) / len(scores), 2)
    
    def _aggregate_sentiment_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate multiple sentiment analysis results."""
        if not results:
            return {}
        
        # Calculate averages
        avg_sentiment = sum(r['sentiment_score'] for r in results) / len(results)
        
        # Count sentiment labels
        sentiment_distribution = defaultdict(int)
        for r in results:
            sentiment_distribution[r['sentiment_label']] += 1
        
        # Aggregate emotions
        emotion_distribution = defaultdict(float)
        for r in results:
            for emotion, score in r.get('emotions', {}).items():
                emotion_distribution[emotion] += score
        
        # Normalize emotion scores
        total_emotions = sum(emotion_distribution.values())
        if total_emotions > 0:
            emotion_distribution = {
                emotion: score / total_emotions
                for emotion, score in emotion_distribution.items()
            }
        
        # Extract all topics
        all_topics = []
        for r in results:
            all_topics.extend(r.get('topics', []))
        
        # Count topic frequency
        topic_counts = defaultdict(int)
        for topic in all_topics:
            topic_counts[topic] += 1
        
        key_topics = sorted(topic_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        
        return {
            'total_comments': len(results),
            'average_sentiment': round(avg_sentiment, 2),
            'overall_sentiment': self._score_to_label(avg_sentiment),
            'sentiment_distribution': dict(sentiment_distribution),
            'emotion_distribution': dict(emotion_distribution),
            'key_topics': [topic for topic, count in key_topics],
            'positive_ratio': sum(1 for r in results if r['sentiment_score'] > 0.2) / len(results),
            'negative_ratio': sum(1 for r in results if r['sentiment_score'] < -0.2) / len(results)
        }
    
    def _score_to_label(self, score: float) -> str:
        """Convert sentiment score to label."""
        if score >= 0.6:
            return SentimentLabel.VERY_POSITIVE
        elif score >= 0.2:
            return SentimentLabel.POSITIVE
        elif score <= -0.6:
            return SentimentLabel.VERY_NEGATIVE
        elif score <= -0.2:
            return SentimentLabel.NEGATIVE
        else:
            return SentimentLabel.NEUTRAL
    
    def _generate_recommendations(self, sentiment_score: float, trend: str) -> List[str]:
        """Generate content strategy recommendations based on sentiment."""
        recommendations = []
        
        if sentiment_score < 0:
            recommendations.extend([
                'Focus on addressing negative feedback',
                'Create more positive, uplifting content',
                'Engage directly with dissatisfied audience members'
            ])
        elif sentiment_score > 0.5:
            recommendations.extend([
                'Maintain current content strategy',
                'Amplify successful content types',
                'Encourage more user-generated content'
            ])
        
        if trend == 'declining':
            recommendations.append('Review recent content changes that may have impacted sentiment')
        elif trend == 'improving':
            recommendations.append('Continue current engagement strategies')
        
        return recommendations
