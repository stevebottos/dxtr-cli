"""
Example tests showcasing LLM evaluation patterns.

This file demonstrates best practices for testing LLM-based systems:
1. Output format validation
2. Golden dataset testing
3. Quality assertions
4. Prompt regression testing

These tests show HOW to test LLMs, but are marked as examples.
To run actual LLM evaluation, remove @pytest.mark.skip decorators.
"""

import pytest
from unittest.mock import patch, Mock
from pathlib import Path

from dxtr.base import Agent
from tests.utils import (
    OutputValidator,
    GoldenDataset,
    calculate_similarity,
    assert_output_quality
)


class TestOutputFormatValidation:
    """
    Pattern 1: Output Format Validation

    Test that LLM outputs conform to expected formats without
    checking exact content. Useful for structured outputs.
    """

    @pytest.mark.skip(reason="Example test - requires actual LLM")
    def test_paper_ranking_returns_valid_format(self):
        """Test that paper ranking returns expected markdown format."""
        # This would call the actual papers_helper agent
        from dxtr.agents.papers_helper.agent import rank_papers

        result = rank_papers(
            date="2025-12-26",
            user_context="User interested in LLMs and NLP"
        )

        # Validate format without checking exact content
        validator = OutputValidator()

        # Should contain ranking headers
        assert validator.matches_pattern(result, r"#+.*Relevant")

        # Should contain scores
        assert validator.matches_pattern(result, r"\d/5")

        # Should have reasoning
        assert validator.contains_keywords(result, ["reasoning", "relevant"])

    def test_json_output_validation(self):
        """Test that structured JSON output is valid."""
        # Mock LLM response with JSON output
        mock_response = Mock()
        mock_response.message.content = '{"keywords": ["test"], "summary": "A test"}'

        validator = OutputValidator()
        assert validator.is_valid_json(mock_response.message.content)

        output = eval(mock_response.message.content)
        assert validator.has_required_fields(output, ["keywords", "summary"])


class TestGoldenDatasets:
    """
    Pattern 2: Golden Dataset Testing

    Use known input/output pairs to test for regressions.
    When you change prompts or models, these tests catch degradations.
    """

    @pytest.fixture
    def golden_dataset(self, tmp_path):
        """Create a golden dataset for testing."""
        dataset_path = tmp_path / "golden_dataset.json"
        dataset = GoldenDataset(dataset_path)

        # Add example golden outputs
        dataset.add_example(
            input_data={"user_query": "What is machine learning?"},
            expected_output="Machine learning is a subset of AI...",
            metadata={"prompt_version": "v1", "model": "test-model"}
        )

        dataset.save()
        return dataset

    def test_output_matches_golden_dataset(self, golden_dataset):
        """Test that output is similar to golden dataset."""
        examples = golden_dataset.get_examples()
        assert len(examples) == 1

        example = examples[0]
        expected = example["expected_output"]

        # Mock LLM output
        actual_output = "Machine learning is a subset of artificial intelligence..."

        # Check similarity (not exact match - LLMs vary)
        similarity = calculate_similarity(expected, actual_output)
        assert similarity > 0.5, f"Output too different from golden (similarity: {similarity})"

    @pytest.mark.skip(reason="Example test - for prompt version comparison")
    def test_compare_prompt_versions(self, golden_dataset, tmp_prompts_dir):
        """
        Compare outputs from different prompt versions.

        This pattern is useful when refactoring prompts to ensure
        new versions maintain quality.
        """
        # Create two prompt versions
        (tmp_prompts_dir / "prompt_v1.md").write_text("You are a helpful assistant.")
        (tmp_prompts_dir / "prompt_v2.md").write_text("You are a concise, helpful assistant.")

        agent = Agent("test", "test-model", tmp_prompts_dir)

        test_input = "What is AI?"

        # Get outputs from both versions (mocked here)
        with patch('dxtr.base.agent.chat') as mock_chat:
            mock_chat.return_value.message.content = "AI is artificial intelligence..."

            # Test v1
            output_v1 = agent.chat([{"role": "user", "content": test_input}], prompt_name="prompt_v1")

            # Test v2
            output_v2 = agent.chat([{"role": "user", "content": test_input}], prompt_name="prompt_v2")

        # Both should produce reasonable outputs
        assert len(output_v1.message.content) > 10
        assert len(output_v2.message.content) > 10


class TestQualityAssertions:
    """
    Pattern 3: Quality Assertions

    Test output quality without exact matching. Useful when
    you care about output characteristics, not exact words.
    """

    def test_summary_meets_quality_criteria(self):
        """Test that a summary meets quality criteria."""
        # Mock a summary output
        summary = """
        This repository implements a transformer-based model for text classification.
        It uses PyTorch and the Hugging Face transformers library.
        The main features include fine-tuning capabilities and evaluation metrics.
        """

        # Assert quality without exact matching
        assert_output_quality(
            summary,
            min_length=50,
            max_length=500,
            required_keywords=["transformer", "model"],
            forbidden_keywords=["TODO", "FIXME", "error"]
        )

    def test_profile_description_quality(self):
        """Test that profile description meets criteria."""
        profile = "Experienced ML engineer specializing in NLP and LLMs."

        validator = OutputValidator()

        # Check length is reasonable
        assert validator.word_count_in_range(profile, min_words=5, max_words=100)

        # Check contains relevant terms
        assert validator.contains_keywords(profile, ["engineer", "ml", "nlp"])


class TestPromptRegressionTesting:
    """
    Pattern 4: Prompt Regression Testing

    Ensure prompt changes don't break expected behavior.
    """

    def test_prompt_loads_without_error(self, tmp_prompts_dir):
        """Test that all prompts can be loaded."""
        agent = Agent("test", "test-model", tmp_prompts_dir)

        # All prompts should load without error
        available_prompts = agent.prompts.list_prompts()

        for prompt_name in available_prompts:
            prompt = agent.prompts.load(prompt_name)
            assert len(prompt) > 0, f"Prompt '{prompt_name}' is empty"

    @pytest.mark.skip(reason="Example test - requires actual prompts")
    def test_all_agent_prompts_exist(self):
        """
        Test that all expected prompts exist for all agents.

        This catches issues where code references a prompt that doesn't exist.
        """
        from dxtr.config import config

        expected_prompts = {
            "profile_creator": ["profile_creation"],
            "papers_helper": ["ranking"],
            "git_helper": ["module_analysis"],
        }

        for agent_name, prompt_names in expected_prompts.items():
            # Get agent's prompt directory (would need to implement this)
            prompts_dir = Path(f"dxtr/agents/{agent_name}/prompts")

            for prompt_name in prompt_names:
                prompt_file = prompts_dir / f"{prompt_name}.md"
                assert prompt_file.exists(), \
                    f"Missing prompt: {agent_name}/{prompt_name}.md"


class TestToolCalling:
    """
    Pattern 5: Tool Calling Tests

    Test that agents correctly call and use tools.
    """

    def test_tool_calling_loop(self, tmp_prompts_dir):
        """Test that tool calling loop works correctly."""
        agent = Agent("test", "test-model", tmp_prompts_dir)

        # Register a test tool
        def get_weather(city: str) -> dict:
            return {"city": city, "temperature": 72, "condition": "sunny"}

        tool_def = {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get weather for a city",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "city": {"type": "string"}
                    },
                    "required": ["city"]
                }
            }
        }

        agent.tools.register("get_weather", get_weather, tool_def)

        # Mock LLM to call tool then respond
        with patch('dxtr.base.agent.chat') as mock_chat:
            # First call: LLM decides to call tool
            tool_call = Mock()
            tool_call.function.name = "get_weather"
            tool_call.function.arguments = {"city": "San Francisco"}

            mock_response_1 = Mock()
            mock_response_1.message.tool_calls = [tool_call]
            mock_response_1.message.content = ""

            # Second call: LLM responds with final answer
            mock_response_2 = Mock()
            mock_response_2.message.tool_calls = []
            mock_response_2.message.content = "The weather in San Francisco is sunny and 72°F."

            mock_chat.side_effect = [mock_response_1, mock_response_2]

            # Run chat with tool calling
            final_response, history = agent.chat_with_tool_calling(
                messages=[{"role": "user", "content": "What's the weather in SF?"}]
            )

            assert "sunny" in final_response.lower()
            assert "72" in final_response


# Additional test patterns to consider:

class TestAdversarialInputs:
    """
    Pattern 6: Adversarial/Edge Case Testing

    Test how LLM handles unusual or problematic inputs.
    """

    def test_empty_input_handling(self):
        """Test that empty inputs are handled gracefully."""
        # Test that system doesn't crash on empty input
        pass

    def test_very_long_input_handling(self):
        """Test handling of inputs near context window limit."""
        pass

    def test_special_characters_in_input(self):
        """Test that special characters don't break the system."""
        pass
