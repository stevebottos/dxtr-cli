# DXTR Tests

This directory contains tests for the DXTR CLI application.

## Setup

Install test dependencies:

```bash
uv sync --extra test
```

Or with pip:

```bash
pip install -e ".[test]"
```

## Running Tests

Run all tests:

```bash
pytest tests/
```

Run with verbose output:

```bash
pytest tests/ -v
```

Run specific test file:

```bash
pytest tests/test_base.py
```

Run with coverage:

```bash
pytest tests/ --cov=dxtr --cov-report=html
```

## Test Structure

- **`test_base.py`**: Unit tests for base classes (Agent, ToolRegistry, PromptManager)
- **`test_llm_evaluation.py`**: Example tests showcasing LLM testing patterns
- **`conftest.py`**: Pytest fixtures and configuration
- **`utils.py`**: Testing utilities for LLM evaluation

## Testing Philosophy

See `notes/testing_guide.md` for comprehensive guide on testing LLM-based systems.

Key principles:
- Test behavior, not exact outputs
- Use quality assertions for LLM outputs
- Mock LLM calls for unit tests
- Use actual LLMs for integration tests
