"""
Test that verifies the RSS Feed Error fix.

This test specifically verifies that the method name collision fix
resolves the error: "Topic extraction failed: _extract_keywords()
missing 1 required positional argument: 'categories'"
"""

import pytest
from unittest.mock import MagicMock
from backend.services.rss_ingestion_service import RSSIngestionService


class TestRSSFeedErrorFix:
    """Test suite to verify the RSS feed error fix."""

    @pytest.fixture
    def rss_service(self):
        """Create RSS ingestion service with mock database."""
        mock_db = MagicMock()
        return RSSIngestionService(mock_db)

    @pytest.fixture
    def mock_rss_entry(self):
        """Create a mock RSS entry like the one from NBC News in the logs."""

        class MockEntry:
            def __init__(self):
                self.title = "Waves hit Puerto Vallarta as Hurricane Priscilla looms"
                self.summary = (
                    "Large waves are hitting the coast of Puerto Vallarta "
                    "as Hurricane Priscilla approaches the Mexican coast."
                )

        return MockEntry()

    def test_extract_keywords_from_text(self, rss_service):
        """Test that _extract_keywords works with just text parameter."""
        # This call was failing before the fix
        text = "artificial intelligence machine learning technology innovation"
        result = rss_service._extract_keywords(text)

        # Should return keywords extracted from text
        assert isinstance(result, list)
        assert len(result) > 0
        # Check some expected keywords are present
        assert any(
            keyword in result for keyword in ["artificial", "intelligence", "machine"]
        )

    def test_extract_keywords_from_title_and_categories(self, rss_service):
        """Test that _extract_keywords_from_title_and_categories works with title and categories."""
        # This is the renamed method that takes two parameters
        title = "AI Startup Raises $100M Funding"
        categories = ["technology", "startups", "venture capital"]

        result = rss_service._extract_keywords_from_title_and_categories(
            title, categories
        )

        # Should return keywords from both title and categories
        assert isinstance(result, list)
        assert len(result) > 0
        # Categories should be included
        assert "technology" in result
        assert "startups" in result

    def test_extract_topics_and_entities_no_error(self, rss_service, mock_rss_entry):
        """Test that _extract_topics_and_entities works without error."""
        # This method was failing in the logs with:
        # "Topic extraction failed: _extract_keywords() missing 1 required positional argument: 'categories'"

        # This should not raise an exception anymore
        result = rss_service._extract_topics_and_entities(mock_rss_entry)

        # Verify the result has expected structure
        assert isinstance(result, dict)
        assert "keywords" in result
        assert "entities" in result
        assert "topics" in result

        # All should be lists
        assert isinstance(result["keywords"], list)
        assert isinstance(result["entities"], list)
        assert isinstance(result["topics"], list)

        # Keywords should be extracted from the entry
        assert len(result["keywords"]) > 0

    def test_both_methods_exist_independently(self, rss_service):
        """Verify both keyword extraction methods exist and work independently."""
        # Test 1-parameter method
        text_result = rss_service._extract_keywords("test text with keywords")
        assert isinstance(text_result, list)

        # Test 2-parameter method
        title_result = rss_service._extract_keywords_from_title_and_categories(
            "Test Title", ["category1"]
        )
        assert isinstance(title_result, list)

        # Both should work without interfering with each other
        assert text_result is not None
        assert title_result is not None

    def test_real_world_scenario_from_logs(self, rss_service):
        """Test the exact scenario from the error logs."""

        class NBCNewsEntry:
            def __init__(self):
                self.title = "Crowd attacks Ecuadorian president's car with rocks"
                self.summary = "Crowd attacks Ecuadorian president's car with rocks"

        entry = NBCNewsEntry()

        # This should work without raising the error:
        # "_extract_keywords() missing 1 required positional argument: 'categories'"
        result = rss_service._extract_topics_and_entities(entry)

        # Verify no exception was raised and we got a valid result
        assert result is not None
        assert "keywords" in result
        assert "entities" in result
        assert "topics" in result
