#!/usr/bin/env python
"""
Verification script for RSS Feed Enhancement implementation.

Checks that all components are properly implemented and working.
"""

import asyncio
import sys
from uuid import uuid4

from backend.database.connection import database_manager
from backend.services.rss_ingestion_service import RSSIngestionService
from backend.services.persona_service import PersonaService
from backend.models.feed import (
    RSSFeedCreate,
    PersonaFeedAssignment,
    RSSFeedModel,
    PersonaFeedModel,
)
from backend.models.persona import PersonaCreate


class Colors:
    """ANSI color codes for terminal output."""

    GREEN = "\033[92m"
    RED = "\033[91m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    RESET = "\033[0m"
    BOLD = "\033[1m"


def print_header(text):
    """Print a formatted header."""
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'=' * 70}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.BLUE}{text}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.BLUE}{'=' * 70}{Colors.RESET}\n")


def print_success(text):
    """Print success message."""
    print(f"{Colors.GREEN}✓{Colors.RESET} {text}")


def print_error(text):
    """Print error message."""
    print(f"{Colors.RED}✗{Colors.RESET} {text}")


def print_info(text):
    """Print info message."""
    print(f"{Colors.YELLOW}ℹ{Colors.RESET} {text}")


async def verify_database_schema():
    """Verify database schema includes new table."""
    print_header("1. Database Schema Verification")

    await database_manager.connect()

    try:
        async with database_manager.get_session() as db:
            # Check if persona_feeds table exists by attempting a query
            from sqlalchemy import select, text

            result = await db.execute(
                text(
                    "SELECT name FROM sqlite_master WHERE type='table' AND name='persona_feeds'"
                )
            )
            table_exists = result.fetchone() is not None

            if table_exists:
                print_success("persona_feeds table exists")

                # Check columns
                result = await db.execute(text("PRAGMA table_info(persona_feeds)"))
                columns = {row[1] for row in result.fetchall()}

                required_columns = {
                    "id",
                    "persona_id",
                    "feed_id",
                    "topics",
                    "priority",
                    "is_active",
                    "created_at",
                }

                if required_columns.issubset(columns):
                    print_success(f"All required columns present: {required_columns}")
                    return True
                else:
                    missing = required_columns - columns
                    print_error(f"Missing columns: {missing}")
                    return False
            else:
                print_error("persona_feeds table not found")
                return False

    except Exception as e:
        print_error(f"Database verification failed: {e}")
        return False
    finally:
        await database_manager.disconnect()


async def verify_service_methods():
    """Verify service methods are implemented."""
    print_header("2. Service Methods Verification")

    required_methods = [
        "assign_feed_to_persona",
        "list_persona_feeds",
        "unassign_feed_from_persona",
        "list_feeds_by_topic",
        "get_content_suggestions",
    ]

    all_present = True
    for method in required_methods:
        if hasattr(RSSIngestionService, method):
            print_success(f"RSSIngestionService.{method}() exists")
        else:
            print_error(f"RSSIngestionService.{method}() missing")
            all_present = False

    return all_present


async def verify_models():
    """Verify Pydantic models are defined."""
    print_header("3. Model Verification")

    from backend.models.feed import (
        PersonaFeedAssignment,
        PersonaFeedResponse,
        FeedsByTopicResponse,
    )

    models_to_check = [
        ("PersonaFeedModel", PersonaFeedModel),
        ("PersonaFeedAssignment", PersonaFeedAssignment),
        ("PersonaFeedResponse", PersonaFeedResponse),
        ("FeedsByTopicResponse", FeedsByTopicResponse),
    ]

    all_present = True
    for name, model in models_to_check:
        try:
            if hasattr(model, "__tablename__") or hasattr(model, "model_fields"):
                print_success(f"{name} defined")
            else:
                print_error(f"{name} incomplete")
                all_present = False
        except Exception as e:
            print_error(f"{name} check failed: {e}")
            all_present = False

    return all_present


async def verify_api_routes():
    """Verify API routes are registered."""
    print_header("4. API Routes Verification")

    try:
        from backend.api.routes import feeds
        import inspect

        # Get all async functions from feeds module
        routes = [
            name
            for name, obj in inspect.getmembers(feeds)
            if inspect.iscoroutinefunction(obj)
        ]

        required_routes = [
            "assign_feed_to_persona",
            "list_persona_feeds",
            "unassign_feed_from_persona",
            "list_feeds_by_topic",
            "get_content_suggestions",
        ]

        all_present = True
        for route in required_routes:
            if route in routes:
                print_success(f"Route {route}() defined")
            else:
                print_error(f"Route {route}() missing")
                all_present = False

        return all_present

    except Exception as e:
        print_error(f"Route verification failed: {e}")
        return False


async def verify_functional_test():
    """Run a quick functional test."""
    print_header("5. Functional Test")

    await database_manager.connect()

    try:
        async with database_manager.get_session() as db:
            persona_service = PersonaService(db)
            rss_service = RSSIngestionService(db)

            # Create test persona
            persona = await persona_service.create_persona(
                PersonaCreate(
                    name=f"Test Persona {uuid4().hex[:8]}",
                    appearance="Test persona for verification purposes",
                    personality="Helpful and professional test persona",
                    content_themes=["test"],
                    style_preferences={},
                )
            )
            print_success(f"Created test persona: {persona.id}")

            # Create test feed
            test_feed = RSSFeedModel(
                id=uuid4(),
                name="Test Feed",
                url=f"https://test.example.com/{uuid4()}",
                categories=["test"],
                is_active=True,
            )
            db.add(test_feed)
            await db.commit()
            print_success(f"Created test feed: {test_feed.id}")

            # Test assignment
            assignment = PersonaFeedAssignment(
                feed_id=test_feed.id, topics=["test"], priority=75
            )

            result = await rss_service.assign_feed_to_persona(persona.id, assignment)
            print_success(
                f"Assigned feed to persona: priority={result.priority}, topics={result.topics}"
            )

            # Test listing
            feeds = await rss_service.list_persona_feeds(persona.id)
            if len(feeds) == 1:
                print_success(f"Listed persona feeds: {len(feeds)} found")
            else:
                print_error(f"Expected 1 feed, found {len(feeds)}")
                return False

            # Test unassignment
            success = await rss_service.unassign_feed_from_persona(
                persona.id, test_feed.id
            )
            if success:
                print_success("Unassigned feed from persona")
            else:
                print_error("Failed to unassign feed")
                return False

            # Verify unassignment
            feeds = await rss_service.list_persona_feeds(persona.id)
            if len(feeds) == 0:
                print_success("Verified feed was unassigned")
            else:
                print_error(f"Feed still assigned after unassignment")
                return False

            return True

    except Exception as e:
        print_error(f"Functional test failed: {e}")
        import traceback

        traceback.print_exc()
        return False
    finally:
        await database_manager.disconnect()


async def main():
    """Run all verification checks."""
    print(
        f"\n{Colors.BOLD}RSS Feed Enhancement - Implementation Verification{Colors.RESET}"
    )

    results = []

    # Run checks
    results.append(("Database Schema", await verify_database_schema()))
    results.append(("Service Methods", await verify_service_methods()))
    results.append(("Models", await verify_models()))
    results.append(("API Routes", await verify_api_routes()))
    results.append(("Functional Test", await verify_functional_test()))

    # Summary
    print_header("Verification Summary")

    all_passed = True
    for name, passed in results:
        if passed:
            print_success(f"{name}: PASSED")
        else:
            print_error(f"{name}: FAILED")
            all_passed = False

    print()
    if all_passed:
        print(
            f"{Colors.GREEN}{Colors.BOLD}✓ All verification checks passed!{Colors.RESET}"
        )
        print(
            f"\n{Colors.BLUE}The RSS Feed Enhancement is fully implemented and functional.{Colors.RESET}"
        )
        return 0
    else:
        print(
            f"{Colors.RED}{Colors.BOLD}✗ Some verification checks failed.{Colors.RESET}"
        )
        print(
            f"\n{Colors.YELLOW}Please review the errors above and fix the issues.{Colors.RESET}"
        )
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
