"""
Utilities for LLM testing and evaluation.

Provides:
- Output validation helpers
- Format checkers
- Evaluation metrics
- Golden dataset management
"""

import json
import re
from pathlib import Path
from typing import Any


class OutputValidator:
    """
    Validates LLM outputs against expected formats and constraints.

    Use this for testing that LLM outputs match expected schemas,
    contain required fields, or meet quality criteria.
    """

    @staticmethod
    def is_valid_json(output: str) -> bool:
        """Check if output is valid JSON."""
        try:
            json.loads(output)
            return True
        except json.JSONDecodeError:
            return False

    @staticmethod
    def has_required_fields(output: dict, required: list[str]) -> bool:
        """
        Check if output dict has all required fields.

        Args:
            output: Output dictionary to check
            required: List of required field names

        Returns:
            bool: True if all required fields present
        """
        return all(field in output for field in required)

    @staticmethod
    def matches_pattern(output: str, pattern: str) -> bool:
        """
        Check if output matches a regex pattern.

        Args:
            output: Output string to check
            pattern: Regex pattern

        Returns:
            bool: True if pattern matches
        """
        return bool(re.search(pattern, output))

    @staticmethod
    def contains_keywords(output: str, keywords: list[str], case_sensitive=False) -> bool:
        """
        Check if output contains all required keywords.

        Args:
            output: Output string to check
            keywords: List of required keywords
            case_sensitive: Whether to do case-sensitive matching

        Returns:
            bool: True if all keywords present
        """
        if not case_sensitive:
            output = output.lower()
            keywords = [k.lower() for k in keywords]

        return all(keyword in output for keyword in keywords)

    @staticmethod
    def word_count_in_range(output: str, min_words: int, max_words: int) -> bool:
        """
        Check if output word count is within range.

        Args:
            output: Output string to check
            min_words: Minimum word count
            max_words: Maximum word count

        Returns:
            bool: True if word count in range
        """
        word_count = len(output.split())
        return min_words <= word_count <= max_words


class GoldenDataset:
    """
    Manages golden datasets for LLM evaluation.

    Golden datasets are known input/output pairs used for:
    - Regression testing (ensure outputs don't degrade)
    - Prompt evaluation (compare different prompts)
    - Model evaluation (compare different models)
    """

    def __init__(self, dataset_path: Path):
        """
        Initialize golden dataset.

        Args:
            dataset_path: Path to JSON file containing dataset
        """
        self.dataset_path = Path(dataset_path)
        self.examples = self._load_dataset()

    def _load_dataset(self) -> list[dict]:
        """Load dataset from file."""
        if not self.dataset_path.exists():
            return []

        with open(self.dataset_path) as f:
            return json.load(f)

    def add_example(self, input_data: dict, expected_output: Any, metadata: dict = None):
        """
        Add an example to the dataset.

        Args:
            input_data: Input that was given to LLM
            expected_output: Expected/golden output
            metadata: Optional metadata (prompt version, model, etc.)
        """
        example = {
            "input": input_data,
            "expected_output": expected_output,
            "metadata": metadata or {}
        }
        self.examples.append(example)

    def save(self):
        """Save dataset to file."""
        self.dataset_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.dataset_path, 'w') as f:
            json.dump(self.examples, f, indent=2)

    def get_examples(self, filter_metadata: dict = None) -> list[dict]:
        """
        Get examples, optionally filtered by metadata.

        Args:
            filter_metadata: Dict of metadata key-value pairs to filter by

        Returns:
            list[dict]: Filtered examples
        """
        if not filter_metadata:
            return self.examples

        filtered = []
        for example in self.examples:
            metadata = example.get("metadata", {})
            if all(metadata.get(k) == v for k, v in filter_metadata.items()):
                filtered.append(example)

        return filtered


def calculate_similarity(text1: str, text2: str) -> float:
    """
    Calculate simple word-overlap similarity between two texts.

    This is a basic metric. For production, consider:
    - Sentence embeddings (e.g., sentence-transformers)
    - Edit distance (Levenshtein)
    - Semantic similarity models

    Args:
        text1: First text
        text2: Second text

    Returns:
        float: Similarity score 0-1
    """
    words1 = set(text1.lower().split())
    words2 = set(text2.lower().split())

    if not words1 or not words2:
        return 0.0

    intersection = words1 & words2
    union = words1 | words2

    return len(intersection) / len(union)


def assert_output_quality(
    output: str,
    min_length: int = 10,
    max_length: int = 10000,
    required_keywords: list[str] = None,
    forbidden_keywords: list[str] = None,
    custom_validators: list = None
):
    """
    Assert that LLM output meets quality criteria.

    Useful for tests that check output quality without exact matching.

    Args:
        output: LLM output to validate
        min_length: Minimum character length
        max_length: Maximum character length
        required_keywords: Keywords that must appear
        forbidden_keywords: Keywords that must NOT appear
        custom_validators: List of (validator_func, error_msg) tuples

    Raises:
        AssertionError: If output doesn't meet criteria
    """
    # Length checks
    assert len(output) >= min_length, f"Output too short: {len(output)} < {min_length}"
    assert len(output) <= max_length, f"Output too long: {len(output)} > {max_length}"

    # Keyword checks
    if required_keywords:
        validator = OutputValidator()
        assert validator.contains_keywords(output, required_keywords), \
            f"Missing required keywords: {required_keywords}"

    if forbidden_keywords:
        for keyword in forbidden_keywords:
            assert keyword.lower() not in output.lower(), \
                f"Forbidden keyword found: {keyword}"

    # Custom validators
    if custom_validators:
        for validator_func, error_msg in custom_validators:
            assert validator_func(output), error_msg
