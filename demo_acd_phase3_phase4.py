"""
Demo: ACD Phase 3 & 4 Implementation

Demonstrates advanced learning and multi-agent ecosystem features.
"""

import asyncio
from datetime import datetime, timezone
from uuid import uuid4

from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker

from backend.services.ml_pattern_recognition import MLPatternRecognitionService
from backend.services.predictive_engagement_scoring import PredictiveEngagementScoringService
from backend.services.cross_persona_learning import CrossPersonaLearningService
from backend.services.ab_testing_service import ABTestingService, TaskPriority
from backend.services.multi_agent_service import MultiAgentService
from backend.services.agent_marketplace_service import AgentMarketplaceService
from backend.models.multi_agent import AgentCreate, AgentType, AgentTaskCreate, AgentRoutingRequest
from backend.models.acd import ACDContextCreate, AIStatus, AIComplexity, AIState
from backend.services.acd_service import ACDService


# Database setup
DATABASE_URL = "sqlite+aiosqlite:///./gator.db"


async def init_db():
    """Initialize database session."""
    engine = create_async_engine(DATABASE_URL, echo=False)
    async_session = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    return async_session()


async def demo_phase3_ml_pattern_recognition():
    """Demo ML-based pattern recognition."""
    print("=" * 80)
    print("PHASE 3 DEMO 1: ML-Based Pattern Recognition")
    print("=" * 80)
    print()
    
    db = await init_db()
    
    try:
        # Create some sample ACD contexts for training
        acd_service = ACDService(db)
        
        print("📊 Creating sample training data...")
        for i in range(10):
            context = await acd_service.create_context(
                ACDContextCreate(
                    ai_phase="SOCIAL_MEDIA_CONTENT",
                    ai_status=AIStatus.IMPLEMENTED,
                    ai_complexity=AIComplexity.MEDIUM,
                    ai_state=AIState.DONE,
                    ai_context={
                        "prompt": f"Sample content {i}",
                        "hashtags": ["trending", "viral", "lifestyle"]
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
            )
            print(f"  ✓ Created context {i+1}/10 (engagement: {5.0 + i * 0.5}%)")
        
        print()
        print("🤖 Training ML models...")
        
        # Train engagement model
        ml_service = MLPatternRecognitionService(db)
        engagement_result = await ml_service.train_engagement_model(
            min_samples=5,
            lookback_days=90
        )
        
        if engagement_result.get("success"):
            print("  ✓ Engagement prediction model trained")
            print(f"    - Train R²: {engagement_result['train_score']:.3f}")
            print(f"    - Test R²: {engagement_result['test_score']:.3f}")
        else:
            print(f"  ✗ Model training: {engagement_result.get('reason')}")
        
        # Train success classifier
        classifier_result = await ml_service.train_success_classifier(
            success_threshold=7.0,
            min_samples=5,
            lookback_days=90
        )
        
        if classifier_result.get("success"):
            print("  ✓ Success classifier trained")
            print(f"    - Train accuracy: {classifier_result['train_accuracy']:.3f}")
            print(f"    - Test accuracy: {classifier_result['test_accuracy']:.3f}")
        
        print()
        print("✅ ML Pattern Recognition Demo Complete!")
        print()
        
    finally:
        await db.close()


async def demo_phase3_predictive_scoring():
    """Demo predictive engagement scoring."""
    print("=" * 80)
    print("PHASE 3 DEMO 2: Predictive Engagement Scoring")
    print("=" * 80)
    print()
    
    db = await init_db()
    
    try:
        # Create a test context
        acd_service = ACDService(db)
        context_data = ACDContextCreate(
            ai_phase="SOCIAL_MEDIA_CONTENT",
            ai_status=AIStatus.IMPLEMENTED,
            ai_complexity=AIComplexity.MEDIUM,
            ai_context={
                "prompt": "Check out this amazing sunset! 🌅",
                "hashtags": ["sunset", "nature", "photography"]
            }
        )
        
        context_response = await acd_service.create_context(context_data)
        print(f"📝 Created test context: {context_response.id}")
        print()
        
        # Get the actual model
        from backend.models.acd import ACDContextModel
        from sqlalchemy import select
        
        stmt = select(ACDContextModel).where(ACDContextModel.id == context_response.id)
        result = await db.execute(stmt)
        context_model = result.scalar_one()
        
        # Score content
        scoring_service = PredictiveEngagementScoringService(db)
        score = await scoring_service.score_content(context_model)
        
        print("🎯 Engagement Score:")
        print(f"  Composite Score: {score['composite_score']:.1f}/100")
        print(f"  Score Tier: {score['score_tier']}")
        print(f"  Confidence: {score['overall_confidence']}")
        print()
        
        # Get optimization recommendations
        recommendations = await scoring_service.optimize_content(
            context_model,
            target_score=80.0
        )
        
        print("💡 Optimization Recommendations:")
        print(f"  Current Score: {recommendations['current_score']:.1f}")
        print(f"  Potential Score: {recommendations['potential_score']:.1f}")
        print(f"  Achievable: {recommendations['achievable']}")
        print()
        print(f"  Top Recommendations ({recommendations['total_recommendations']}):")
        for i, rec in enumerate(recommendations['recommendations'][:3], 1):
            print(f"    {i}. [{rec['priority'].upper()}] {rec['recommendation']}")
            print(f"       Expected: {rec['expected_improvement']}")
        
        print()
        print("✅ Predictive Scoring Demo Complete!")
        print()
        
    finally:
        await db.close()


async def demo_phase3_cross_persona_learning():
    """Demo cross-persona learning with privacy preservation."""
    print("=" * 80)
    print("PHASE 3 DEMO 3: Cross-Persona Learning with Privacy")
    print("=" * 80)
    print()
    
    db = await init_db()
    
    try:
        cross_persona_service = CrossPersonaLearningService(db)
        
        # Aggregate patterns
        print("🔒 Aggregating patterns across personas (privacy-preserved)...")
        patterns = await cross_persona_service.aggregate_engagement_patterns(
            platform="instagram",
            min_personas=3,
            lookback_days=30
        )
        
        if patterns.get("success"):
            print("  ✓ Aggregation successful")
            print()
            print("  Privacy Guarantees:")
            for mechanism, details in patterns["privacy_guarantees"].items():
                print(f"    • {mechanism}: {details}")
            print()
            print("  Aggregate Patterns:")
            agg = patterns["aggregate_patterns"]
            print(f"    • Mean Engagement: {agg['mean_engagement_rate']:.2f}%")
            print(f"    • Optimal Hours: {agg['optimal_posting_hours']}")
            print(f"    • Effective Hashtags: {len(agg['effective_hashtags'])} (k-anonymous)")
        else:
            print(f"  ℹ️  {patterns.get('reason')}")
        
        print()
        
        # Get privacy report
        print("📋 Privacy Compliance Report:")
        privacy_report = await cross_persona_service.get_privacy_report()
        
        print(f"  Differential Privacy: ε={privacy_report['privacy_mechanisms']['differential_privacy']['epsilon']}")
        print(f"  K-Anonymity: k={privacy_report['privacy_mechanisms']['k_anonymity']['k_value']}")
        print(f"  GDPR Compliant: {privacy_report['compliance']['gdpr_compliant']}")
        print(f"  CCPA Compliant: {privacy_report['compliance']['ccpa_compliant']}")
        
        print()
        print("✅ Cross-Persona Learning Demo Complete!")
        print()
        
    finally:
        await db.close()


async def demo_phase3_ab_testing():
    """Demo automated A/B testing."""
    print("=" * 80)
    print("PHASE 3 DEMO 4: Automated A/B Testing")
    print("=" * 80)
    print()
    
    db = await init_db()
    
    try:
        ab_service = ABTestingService(db)
        
        # Create A/B test
        print("🧪 Creating A/B test...")
        test_config = await ab_service.create_test(
            test_name="Hashtag Optimization Test",
            variants=[
                {
                    "variant_id": "control",
                    "name": "Control",
                    "changes": {}
                },
                {
                    "variant_id": "A",
                    "name": "Trending Hashtags",
                    "changes": {"hashtags": ["trending", "viral", "popular"]}
                },
                {
                    "variant_id": "B",
                    "name": "Niche Hashtags",
                    "changes": {"hashtags": ["photography", "sunset", "nature"]}
                }
            ],
            success_metric="engagement_rate",
            minimum_sample_size=50,
            minimum_runtime_hours=24
        )
        
        print(f"  ✓ Test created: {test_config.test_id}")
        print(f"  Variants: {len(test_config.variants)}")
        print()
        
        # Start test
        await ab_service.start_test(test_config.test_id)
        print("  ✓ Test started")
        print()
        
        # Simulate some events
        print("📊 Simulating test data...")
        import random
        for variant_id in ["control", "A", "B"]:
            impressions = random.randint(60, 80)
            engagement = random.randint(3, 8)
            
            await ab_service.record_variant_event(
                test_config.test_id, variant_id, "impression", impressions
            )
            await ab_service.record_variant_event(
                test_config.test_id, variant_id, "engagement", engagement
            )
            
            print(f"  • Variant {variant_id}: {impressions} impressions, {engagement} engagements")
        
        print()
        
        # Analyze results
        print("🔬 Analyzing test results...")
        analysis = await ab_service.analyze_test(test_config.test_id)
        
        print(f"  Status: {analysis.status.value}")
        print(f"  Winner: {analysis.winner or 'TBD'}")
        print(f"  Statistical Significance: {analysis.statistical_significance}")
        print(f"  Recommendation: {analysis.recommendation}")
        print()
        print("  Insights:")
        for insight in analysis.insights:
            print(f"    • {insight}")
        
        print()
        print("✅ A/B Testing Demo Complete!")
        print()
        
    finally:
        await db.close()


async def demo_phase4_multi_agent():
    """Demo multi-agent ecosystem."""
    print("=" * 80)
    print("PHASE 4 DEMO 1: Multi-Agent Ecosystem")
    print("=" * 80)
    print()
    
    db = await init_db()
    
    try:
        multi_agent_service = MultiAgentService(db)
        
        # Register agents
        print("🤖 Registering specialized agents...")
        
        agents = []
        agent_configs = [
            ("ContentGeneratorAlpha", AgentType.GENERATOR, ["generate", "text", "image"]),
            ("QualityReviewerBeta", AgentType.REVIEWER, ["review", "analyze", "quality_check"]),
            ("EngagementOptimizerGamma", AgentType.OPTIMIZER, ["optimize", "analyze", "recommendations"]),
        ]
        
        for name, agent_type, capabilities in agent_configs:
            try:
                agent = await multi_agent_service.register_agent(
                    AgentCreate(
                        agent_name=name,
                        agent_type=agent_type,
                        capabilities=capabilities,
                        max_concurrent_tasks=10
                    )
                )
                agents.append(agent)
                print(f"  ✓ Registered: {name} ({agent_type.value})")
            except Exception as e:
                print(f"  ℹ️  {name} may already exist")
        
        print()
        
        # Create tasks with auto-routing
        print("📋 Creating tasks with automatic routing...")
        
        task_configs = [
            ("Generate social post", "content_generation", ["generate", "text"]),
            ("Review generated content", "content_review", ["review", "quality_check"]),
            ("Optimize engagement", "content_optimization", ["optimize", "analyze"]),
        ]
        
        for task_name, task_type, required_caps in task_configs:
            task = await multi_agent_service.create_task(
                AgentTaskCreate(
                    task_type=task_type,
                    task_name=task_name,
                    priority=TaskPriority.NORMAL
                ),
                auto_assign=True
            )
            print(f"  ✓ Task created: {task_name}")
            print(f"    Assigned to: agent {task.agent_id if task.agent_id else 'pending'}")
        
        print()
        
        # Get workload stats
        print("📊 System Workload:")
        workload = await multi_agent_service.get_agent_workload()
        print(f"  Total Agents: {workload['total_agents']}")
        print(f"  Active: {workload['active_agents']} | Idle: {workload['idle_agents']}")
        print(f"  Utilization: {workload['utilization_percent']:.1f}%")
        print(f"  Pending Tasks: {workload['pending_tasks']}")
        print(f"  Avg Success Rate: {workload['avg_success_rate']:.1%}")
        
        print()
        print("✅ Multi-Agent Ecosystem Demo Complete!")
        print()
        
    finally:
        await db.close()


async def demo_phase4_marketplace():
    """Demo agent marketplace and plugin system."""
    print("=" * 80)
    print("PHASE 4 DEMO 2: Agent Marketplace & Plugin System")
    print("=" * 80)
    print()
    
    db = await init_db()
    
    try:
        marketplace_service = AgentMarketplaceService(db)
        
        # Search marketplace
        print("🔍 Searching agent marketplace...")
        results = await marketplace_service.search_marketplace(
            agent_type=AgentType.GENERATOR,
            min_rating=4.0,
            limit=5
        )
        
        print(f"  Found {len(results)} agents:")
        for agent in results[:3]:
            print(f"    • {agent.agent_name} v{agent.version}")
            print(f"      Rating: {'⭐' * int(agent.rating)} {agent.rating:.1f}")
            print(f"      Downloads: {agent.download_count}")
            print(f"      Author: {agent.author}")
            print()
        
        # Get marketplace stats
        print("📊 Marketplace Statistics:")
        stats = await marketplace_service.get_marketplace_stats()
        print(f"  Total Agents: {stats['total_agents']}")
        print(f"  Average Rating: {stats['average_rating']:.2f} ⭐")
        print(f"  Total Downloads: {stats['total_downloads']}")
        print(f"  Free Agents: {stats['free_agents']}")
        print()
        
        print("  Top Rated:")
        for i, agent in enumerate(stats['top_rated'][:3], 1):
            print(f"    {i}. {agent['name']} ({agent['rating']:.1f}⭐)")
        
        print()
        
        # Install an agent
        if results:
            agent_to_install = results[0]
            print(f"📦 Installing agent: {agent_to_install.agent_name}...")
            
            install_result = await marketplace_service.install_agent(
                agent_to_install.agent_id,
                custom_name=f"{agent_to_install.agent_name}_Instance1"
            )
            
            if install_result.get("success"):
                print(f"  ✓ {install_result['message']}")
            else:
                print(f"  ℹ️  {install_result.get('error', 'Already installed')}")
        
        print()
        
        # List installed agents
        print("📦 Installed Agents:")
        installed = await marketplace_service.get_installed_agents()
        for agent in installed[:5]:
            print(f"  • {agent['agent_name']} v{agent['version']}")
            print(f"    Type: {agent['agent_type']} | Status: {agent['status']}")
            print(f"    Tasks: {agent['tasks_completed']} | Success: {agent['success_rate']:.1%}")
            if agent['updates_available']:
                print(f"    🔄 Update available: {agent['latest_version']}")
        
        print()
        print("✅ Agent Marketplace Demo Complete!")
        print()
        
    finally:
        await db.close()


async def main():
    """Run all demos."""
    print()
    print("╔════════════════════════════════════════════════════════════════════════════╗")
    print("║                  ACD PHASE 3 & 4 IMPLEMENTATION DEMO                       ║")
    print("║                    Gator AI Influencer Platform                            ║")
    print("╚════════════════════════════════════════════════════════════════════════════╝")
    print()
    
    demos = [
        ("Phase 3: ML Pattern Recognition", demo_phase3_ml_pattern_recognition),
        ("Phase 3: Predictive Scoring", demo_phase3_predictive_scoring),
        ("Phase 3: Cross-Persona Learning", demo_phase3_cross_persona_learning),
        ("Phase 3: A/B Testing", demo_phase3_ab_testing),
        ("Phase 4: Multi-Agent System", demo_phase4_multi_agent),
        ("Phase 4: Agent Marketplace", demo_phase4_marketplace),
    ]
    
    for title, demo_func in demos:
        try:
            await demo_func()
        except Exception as e:
            print(f"❌ Demo failed: {e}")
            import traceback
            traceback.print_exc()
            print()
    
    print("=" * 80)
    print("🎉 All Demos Complete!")
    print("=" * 80)
    print()
    print("Summary of Implemented Features:")
    print("  ✓ ML-based pattern recognition with scikit-learn")
    print("  ✓ Predictive engagement scoring")
    print("  ✓ Cross-persona learning with differential privacy")
    print("  ✓ Automated A/B testing with statistical analysis")
    print("  ✓ Multi-agent ecosystem with specialized agent types")
    print("  ✓ Automatic agent routing and load balancing")
    print("  ✓ Agent marketplace with plugin system")
    print("  ✓ Comprehensive API endpoints for all features")
    print()


if __name__ == "__main__":
    asyncio.run(main())
