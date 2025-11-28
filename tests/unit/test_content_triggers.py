"""
Tests for content triggers integration in persona chat.

Tests the trigger matching system that routes specific keywords to 
different models and LoRAs for image generation.
"""

import pytest
from unittest.mock import MagicMock, AsyncMock
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from backend.api.routes.persona import (
    _check_image_trigger,
    _build_generation_params_from_trigger,
    IMAGE_TRIGGER_PHRASES,
)


class MockPersona:
    """Mock persona object for testing."""
    
    def __init__(self, **kwargs):
        self.name = kwargs.get('name', 'Test Persona')
        self.content_triggers = kwargs.get('content_triggers', {})
        self.image_model_preference = kwargs.get('image_model_preference', None)
        self.nsfw_model_preference = kwargs.get('nsfw_model_preference', None)
        self.default_image_resolution = kwargs.get('default_image_resolution', '1024x1024')
        self.base_images = kwargs.get('base_images', {})
        self.base_image_path = kwargs.get('base_image_path', None)


class TestCheckImageTrigger:
    """Tests for the _check_image_trigger function."""
    
    def test_default_trigger_match(self):
        """Test that default IMAGE_TRIGGER_PHRASES are matched."""
        result = _check_image_trigger("take a selfie")
        assert result['should_generate'] is True
        assert result['matched_trigger'] is None  # No persona trigger, default match
        assert result['trigger_phrase'] == "take a selfie"
    
    def test_default_trigger_case_insensitive(self):
        """Test that trigger matching is case insensitive."""
        result = _check_image_trigger("SEND ME A SELFIE")
        assert result['should_generate'] is True
        assert result['trigger_phrase'] == "send me a selfie"
    
    def test_no_trigger_match(self):
        """Test that non-trigger messages don't match."""
        result = _check_image_trigger("Hello, how are you today?")
        assert result['should_generate'] is False
        assert result['matched_trigger'] is None
        assert result['trigger_phrase'] is None
    
    def test_persona_trigger_match(self):
        """Test that persona-specific triggers are matched."""
        persona = MockPersona(content_triggers={
            'bikini_trigger': {
                'trigger_phrases': ['bikini', 'swimsuit', 'beach pic'],
                'model': 'nsfw_model_v2',
                'loras': [{'name': 'body_lora', 'weight': 0.7}],
                'positive_prompt': 'beach setting, sunny',
                'enabled': True,
                'priority': 80
            }
        })
        
        result = _check_image_trigger("send me a bikini pic", persona)
        assert result['should_generate'] is True
        assert result['matched_trigger'] is not None
        assert result['matched_trigger']['model'] == 'nsfw_model_v2'
        assert result['trigger_id'] == 'bikini_trigger'
    
    def test_persona_trigger_disabled(self):
        """Test that disabled persona triggers are skipped."""
        persona = MockPersona(content_triggers={
            'bikini_trigger': {
                'trigger_phrases': ['bikini', 'swimsuit'],
                'model': 'nsfw_model_v2',
                'enabled': False,  # Disabled
                'priority': 80
            }
        })
        
        result = _check_image_trigger("send me a bikini pic", persona)
        # Should not generate because the persona trigger is disabled
        # and 'bikini pic' is not in default IMAGE_TRIGGER_PHRASES
        assert result['should_generate'] is False
        assert result['matched_trigger'] is None
    
    def test_persona_trigger_priority(self):
        """Test that higher priority triggers are checked first."""
        persona = MockPersona(content_triggers={
            'low_priority': {
                'trigger_phrases': ['selfie'],
                'model': 'low_priority_model',
                'enabled': True,
                'priority': 20
            },
            'high_priority': {
                'trigger_phrases': ['selfie', 'send me a selfie'],
                'model': 'high_priority_model',
                'enabled': True,
                'priority': 90
            }
        })
        
        result = _check_image_trigger("send me a selfie", persona)
        assert result['should_generate'] is True
        assert result['matched_trigger']['model'] == 'high_priority_model'
    
    def test_internal_config_skipped(self):
        """Test that _config entries are skipped during trigger matching."""
        persona = MockPersona(content_triggers={
            '_config': {
                'default_positive_prompt': 'high quality',
                'default_negative_prompt': 'ugly, blurry'
            },
            'real_trigger': {
                'trigger_phrases': ['selfie'],
                'model': 'test_model',
                'enabled': True
            }
        })
        
        result = _check_image_trigger("take a selfie", persona)
        # Should match the real trigger, not crash on _config
        assert result['should_generate'] is True


class TestBuildGenerationParams:
    """Tests for the _build_generation_params_from_trigger function."""
    
    def test_default_params(self):
        """Test that default parameters are returned when no persona preferences."""
        persona = MockPersona()
        params = _build_generation_params_from_trigger(persona)
        
        assert params['width'] == 1024
        assert params['height'] == 1024
        assert params['guidance_scale'] == 7.5
        assert params['num_inference_steps'] == 30
        assert 'ugly' in params['negative_prompt']
    
    def test_persona_model_preferences(self):
        """Test that persona model preferences are extracted."""
        persona = MockPersona(
            image_model_preference='my_custom_model',
            nsfw_model_preference='nsfw_special_model'
        )
        params = _build_generation_params_from_trigger(persona)
        
        assert params['image_model_pref'] == 'my_custom_model'
        assert params['nsfw_model_pref'] == 'nsfw_special_model'
    
    def test_persona_resolution(self):
        """Test that persona resolution preferences are parsed."""
        persona = MockPersona(default_image_resolution='768x1024')
        params = _build_generation_params_from_trigger(persona)
        
        assert params['width'] == 768
        assert params['height'] == 1024
    
    def test_trigger_model_override(self):
        """Test that trigger model overrides persona default."""
        persona = MockPersona(image_model_preference='default_model')
        matched_trigger = {
            'model': 'trigger_specific_model',
            'loras': []
        }
        params = _build_generation_params_from_trigger(persona, matched_trigger)
        
        assert params['image_model_pref'] == 'trigger_specific_model'
    
    def test_trigger_weight_overrides(self):
        """Test that trigger weight overrides are applied."""
        persona = MockPersona()
        matched_trigger = {
            'weight_overrides': {
                'guidance_scale': 12.0,
                'num_inference_steps': 50,
                'strength': 0.6
            }
        }
        params = _build_generation_params_from_trigger(persona, matched_trigger)
        
        assert params['guidance_scale'] == 12.0
        assert params['num_inference_steps'] == 50
        assert params['img2img_strength'] == 0.6
    
    def test_trigger_negative_prompt_append(self):
        """Test that trigger negative prompt is appended to default."""
        persona = MockPersona()
        matched_trigger = {
            'negative_prompt': 'extra_bad_stuff'
        }
        params = _build_generation_params_from_trigger(persona, matched_trigger)
        
        assert 'ugly' in params['negative_prompt']  # Default
        assert 'extra_bad_stuff' in params['negative_prompt']  # Trigger addition
    
    def test_base_images_view_type(self):
        """Test that base images are selected based on view_type."""
        persona = MockPersona(base_images={
            'front_headshot': '/path/to/front.png',
            'side_profile': '/path/to/side.png'
        })
        matched_trigger = {
            'view_type': 'side_profile'
        }
        params = _build_generation_params_from_trigger(persona, matched_trigger)
        
        assert params['reference_image_path'] == '/path/to/side.png'
    
    def test_base_images_fallback_to_front(self):
        """Test that front_headshot is used as fallback when view_type not available."""
        persona = MockPersona(base_images={
            'front_headshot': '/path/to/front.png'
        })
        matched_trigger = {
            'view_type': 'rear_view'  # Not available
        }
        params = _build_generation_params_from_trigger(persona, matched_trigger)
        
        assert params['reference_image_path'] == '/path/to/front.png'
    
    def test_lora_info_passed(self):
        """Test that LoRA information is passed through."""
        persona = MockPersona()
        matched_trigger = {
            'loras': [
                {'name': 'face_detail_lora', 'weight': 0.8},
                {'name': 'skin_texture_lora', 'weight': 0.5}
            ]
        }
        params = _build_generation_params_from_trigger(persona, matched_trigger)
        
        assert 'loras' in params
        assert len(params['loras']) == 2
        assert params['loras'][0]['name'] == 'face_detail_lora'


class TestImageTriggerPhrases:
    """Tests for the default IMAGE_TRIGGER_PHRASES list."""
    
    def test_common_phrases_included(self):
        """Test that common image request phrases are in the list."""
        assert "take a selfie" in IMAGE_TRIGGER_PHRASES
        assert "send me a picture" in IMAGE_TRIGGER_PHRASES
        assert "show me a photo" in IMAGE_TRIGGER_PHRASES
    
    def test_no_empty_phrases(self):
        """Test that there are no empty trigger phrases."""
        for phrase in IMAGE_TRIGGER_PHRASES:
            assert phrase.strip() != ""
    
    def test_all_lowercase(self):
        """Test that all phrases are lowercase for matching."""
        for phrase in IMAGE_TRIGGER_PHRASES:
            assert phrase == phrase.lower()
