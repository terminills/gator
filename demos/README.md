# Gator Demo Scripts

This directory contains demonstration scripts that showcase various Gator platform features.

## Available Demos

| Demo | Description |
|------|-------------|
| `demo.py` | Main system validation demo - creates persona, demonstrates CRUD operations |
| `demo_acd_integration.py` | ACD (Autonomous Continuous Development) system integration |
| `demo_acd_phase3_phase4.py` | Advanced ACD phases demonstration |
| `demo_ai_video_generation.py` | AI video generation capabilities |
| `demo_enhanced_fallback.py` | Enhanced fallback text generation |
| `demo_error_intelligence.py` | Error intelligence and diagnostics |
| `demo_fan_control_manufacturer.py` | Fan control manufacturer integration |
| `demo_plugin_system.py` | Plugin system capabilities |
| `demo_pytorch_version_check.py` | PyTorch version compatibility checking |
| `demo_q2_features.py` | Q2 2025 features demonstration |
| `demo_reasoning_orchestrator.py` | Reasoning orchestrator capabilities |
| `demo_rss_enhancements.py` | RSS feed enhancements |
| `demo_social_engagement_tracking.py` | Social engagement tracking features |
| `demo_video_features.py` | Video processing features |

## Running Demos

Run demos from the repository root:

```bash
# Main validation demo
python demos/demo.py

# Specific feature demos
python demos/demo_acd_integration.py
python demos/demo_plugin_system.py
```

## Prerequisites

Ensure you have installed dependencies and set up the database:

```bash
pip install -e .
python setup_db.py
```
