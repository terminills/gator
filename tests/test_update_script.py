#!/usr/bin/env python3
"""
Tests for update.sh script

Verifies that the update script works correctly.
"""

import os
import subprocess
import tempfile
from pathlib import Path

import pytest


def test_update_script_exists():
    """Test that update.sh script exists and is executable."""
    script_path = Path(__file__).parent.parent / "update.sh"
    assert script_path.exists(), "update.sh script does not exist"
    assert os.access(script_path, os.X_OK), "update.sh is not executable"


def test_update_script_help():
    """Test that update.sh --help works."""
    script_path = Path(__file__).parent.parent / "update.sh"
    result = subprocess.run(
        [str(script_path), "--help"],
        cwd=script_path.parent,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, "update.sh --help failed"
    assert "Gator AI Influencer Platform - Update Script" in result.stdout
    assert "--verbose" in result.stdout
    assert "--skip-migrations" in result.stdout
    assert "--skip-verification" in result.stdout


def test_update_script_version_check():
    """Test that the script checks Python version."""
    script_path = Path(__file__).parent.parent / "update.sh"
    # Just verify the script runs without errors with skip flags
    result = subprocess.run(
        [str(script_path), "--skip-migrations", "--skip-verification"],
        cwd=script_path.parent,
        capture_output=True,
        text=True,
        timeout=60,
    )

    # Should succeed or fail gracefully
    assert result.returncode in [0, 1, 2], "update.sh failed unexpectedly"
    assert "Step 1/6: Checking Python version" in result.stdout


def test_update_script_finds_migrations():
    """Test that the script finds migration scripts."""
    script_path = Path(__file__).parent.parent / "update.sh"

    # Run with skip-verification to make it faster
    result = subprocess.run(
        [str(script_path), "--skip-verification"],
        cwd=script_path.parent,
        capture_output=True,
        text=True,
        timeout=120,
    )

    # Check that migrations were found and processed
    assert "Step 4/6: Running database migrations" in result.stdout
    # Should find the migration scripts
    assert "migration script" in result.stdout.lower()


def test_update_script_invalid_option():
    """Test that the script handles invalid options."""
    script_path = Path(__file__).parent.parent / "update.sh"
    result = subprocess.run(
        [str(script_path), "--invalid-option"],
        cwd=script_path.parent,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0, "Script should fail with invalid option"
    assert "Unknown option" in result.stdout or "Unknown option" in result.stderr


def test_migration_scripts_exist():
    """Test that migration scripts referenced in update.sh exist."""
    repo_root = Path(__file__).parent.parent

    # Find all migrate_*.py files
    migration_scripts = list(repo_root.glob("migrate_*.py"))

    # Should have at least the two we know about
    assert len(migration_scripts) >= 2, "Expected at least 2 migration scripts"

    # Check they are executable or at least runnable
    for script in migration_scripts:
        assert script.exists(), f"Migration script {script} does not exist"
        # Should be a valid Python file
        assert script.suffix == ".py", f"Migration script {script} is not a .py file"


def test_readme_mentions_update_script():
    """Test that README.md mentions the update.sh script."""
    readme_path = Path(__file__).parent.parent / "README.md"
    assert readme_path.exists(), "README.md does not exist"

    content = readme_path.read_text()
    assert "./update.sh" in content, "README.md does not mention update.sh"
    assert "Maintenance & Updates" in content or "Update" in content


if __name__ == "__main__":
    # Run tests when executed directly
    pytest.main([__file__, "-v"])
