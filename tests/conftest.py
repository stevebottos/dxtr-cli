"""
Pytest configuration and fixtures for DXTR tests.

Provides:
- Common fixtures for testing
- Mock LLM responses
- Test utilities
"""

import pytest
from pathlib import Path
from unittest.mock import Mock


@pytest.fixture
def tmp_prompts_dir(tmp_path):
    """Create a temporary prompts directory with sample prompts."""
    prompts_dir = tmp_path / "prompts"
    prompts_dir.mkdir()

    # Create sample prompts
    (prompts_dir / "test_prompt.md").write_text("You are a helpful assistant.")
    (prompts_dir / "chat.md").write_text("You are a chat assistant.")

    return prompts_dir


@pytest.fixture
def sample_profile():
    """Sample user profile for testing."""
    return """# User Profile

Name: Test User
Interests: Machine Learning, NLP
GitHub: https://github.com/testuser

## Background
Experienced ML engineer working on LLM applications.
"""


@pytest.fixture
def sample_github_summary():
    """Sample GitHub analysis summary for testing."""
    return {
        "/path/to/repo/main.py": '{"keywords": ["pytorch", "transformers"], "summary": "Main training script"}',
        "/path/to/repo/model.py": '{"keywords": ["neural network", "attention"], "summary": "Model architecture"}',
    }
