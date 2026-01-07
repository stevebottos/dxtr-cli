"""
Evaluation utilities for deep research agent.

Contains schemas and helper functions for verification.
"""

# JSON schema for deep research answer verification
ANSWER_VERIFICATION_SCHEMA = {
    "type": "object",
    "properties": {
        "novelty_score": {
            "type": "integer",
            "minimum": 1,
            "maximum": 5,
            "description": "1=just parrots abstract, 5=rich novel details from paper body",
        },
        "novelty_evidence": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Examples of novel information not in abstract",
        },
        "relevance_score": {
            "type": "integer",
            "minimum": 1,
            "maximum": 5,
            "description": "1=off-topic, 5=directly answers the query",
        },
        "relevance_issues": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Aspects of query not addressed",
        },
        "completeness_score": {
            "type": "integer",
            "minimum": 1,
            "maximum": 5,
            "description": "1=superficial, 5=thorough coverage",
        },
        "completeness_issues": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Missing information that should be included",
        },
        "faithfulness_score": {
            "type": "integer",
            "minimum": 1,
            "maximum": 5,
            "description": "1=hallucinated, 5=fully grounded in retrieved chunks",
        },
        "faithfulness_issues": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Claims not supported by retrieved chunks",
        },
    },
    "required": [
        "novelty_score",
        "relevance_score",
        "completeness_score",
        "faithfulness_score",
    ],
}


def compute_metrics(verification_result: dict) -> dict:
    """Compute metrics from verification result.

    Args:
        verification_result: Verification result dict

    Returns:
        Dict with computed metrics
    """
    if "error" in verification_result:
        return {"error": verification_result["error"]}

    novelty = verification_result.get("novelty_score", 0)
    relevance = verification_result.get("relevance_score", 0)
    completeness = verification_result.get("completeness_score", 0)
    faithfulness = verification_result.get("faithfulness_score", 0)

    return {
        "novelty_score": novelty,
        "relevance_score": relevance,
        "completeness_score": completeness,
        "faithfulness_score": faithfulness,
        "overall_score": (novelty + relevance + completeness + faithfulness) / 4,
        "novelty_evidence": verification_result.get("novelty_evidence", []),
        "relevance_issues": verification_result.get("relevance_issues", []),
        "completeness_issues": verification_result.get("completeness_issues", []),
        "faithfulness_issues": verification_result.get("faithfulness_issues", []),
    }


def print_metrics(metrics: dict):
    """Print formatted verification metrics.

    Args:
        metrics: Metrics dict
    """
    print("\n" + "=" * 60)
    print("EVALUATION METRICS")
    print("=" * 60)

    if "error" in metrics:
        print(f"Error: {metrics['error']}")
        return

    print(f"\nNovelty:      {metrics['novelty_score']}/5")
    print("  (1=parrots abstract, 5=rich novel details)")
    if metrics.get("novelty_evidence"):
        print("  Evidence of novel info:")
        for evidence in metrics["novelty_evidence"][:3]:
            print(f"    + {evidence[:80]}...")

    print(f"\nRelevance:    {metrics['relevance_score']}/5")
    print("  (1=off-topic, 5=directly answers query)")
    if metrics.get("relevance_issues"):
        print("  Issues:")
        for issue in metrics["relevance_issues"][:3]:
            print(f"    - {issue}")

    print(f"\nCompleteness: {metrics['completeness_score']}/5")
    print("  (1=superficial, 5=thorough coverage)")
    if metrics.get("completeness_issues"):
        print("  Missing:")
        for issue in metrics["completeness_issues"][:3]:
            print(f"    - {issue}")

    print(f"\nFaithfulness: {metrics['faithfulness_score']}/5")
    print("  (1=hallucinated, 5=grounded in chunks)")
    if metrics.get("faithfulness_issues"):
        print("  Issues:")
        for issue in metrics["faithfulness_issues"][:3]:
            print(f"    - {issue}")

    print(f"\n{'='*40}")
    print(f"OVERALL SCORE: {metrics['overall_score']:.2f}/5.0")
    print("=" * 40)
