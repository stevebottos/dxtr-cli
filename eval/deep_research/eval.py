#!/usr/bin/env python3
"""
Deep Research Agent Evaluation

Evaluates the deep_research agent's ability to answer questions about papers.
Runs multiple hard queries across all fixture papers and reports aggregate metrics.

Usage:
    python eval/deep_research/eval.py                    # Run full eval suite
    python eval/deep_research/eval.py --paper-id 2512.12345  # Single paper
"""

import sys
import json
import argparse
import time
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from dxtr.agents.deep_research import agent as deep_research

# Hard test queries with ground truth key points that MUST appear in good answers
TEST_CASES = [
    # === Paper 2601.02314: Project Ariadne ===
    {
        "paper_id": "2601.02314",
        "query": "What specific metrics does the paper introduce to measure faithfulness, and what are their exact formulas or definitions?",
        "tests": "novelty (needs specific formulas from body), faithfulness (must cite chunks)",
        "ground_truth_points": [
            "Causal Sensitivity",
            "φ or ϕ or phi",  # Different Unicode variants of phi
            "Violation Density",
            "ρ or rho",  # Greek rho
            "counterfactual or do-calculus",
        ],
    },
    {
        "paper_id": "2601.02314",
        "query": "What are the limitations of the Ariadne framework that the authors acknowledge?",
        "tests": "completeness (must find limitations section), relevance (specific question)",
        "ground_truth_points": [
            "computational cost or resources",
            "domain-specific or generalization",
        ],
    },
    {
        "paper_id": "2601.02314",
        "query": "Which techniques from this paper could I apply to ensure my models aren't just producing post-hoc justifications? Focus on practical applications.",
        "tests": "profile-relevance, practical application",
        "ground_truth_points": [
            "counterfactual interventions",
            "Causal Sensitivity",
            "practical application",
        ],
    },
    # === Paper 2601.01426: SWE-Lego ===
    {
        "paper_id": "2601.01426",
        "query": "What exact performance numbers did SWE-Lego achieve on SWE-bench Verified, and how do these compare before and after test-time scaling?",
        "tests": "novelty (specific numbers), faithfulness (exact figures from paper)",
        "ground_truth_points": [
            "42.2%",  # 8B baseline
            "52.6%",  # 32B baseline
            "49.6%",  # 8B with TTS
            "58.8%",  # 32B with TTS
            "TTS@16 or test-time scaling",
        ],
    },
    {
        "paper_id": "2601.01426",
        "query": "Explain the error masking technique used in training. How exactly does it work?",
        "tests": "novelty (implementation detail), completeness (full explanation)",
        "ground_truth_points": [
            "mask or masking",
            "error or incorrect",
            "loss or training",
        ],
    },
    {
        "paper_id": "2601.01426",
        "query": "What aspects of this paper's approach could be relevant for training smaller models efficiently on limited hardware?",
        "tests": "profile-relevance, hardware constraints",
        "ground_truth_points": [
            "8B model or Qwen3-8B",
            "supervised fine-tuning or SFT",
            "curriculum or difficulty",
        ],
    },
]


def check_ground_truth(answer: str, ground_truth_points: list[str]) -> dict:
    """Check if ground truth points appear in the answer.

    Args:
        answer: The generated answer
        ground_truth_points: List of key points that should appear
                            Can use "X or Y" syntax for alternatives

    Returns:
        Dict with hits, misses, and score
    """
    answer_lower = answer.lower()
    hits = []
    misses = []

    for point in ground_truth_points:
        # Handle "X or Y" alternatives
        alternatives = [p.strip().lower() for p in point.split(" or ")]
        found = any(alt in answer_lower for alt in alternatives)

        if found:
            hits.append(point)
        else:
            misses.append(point)

    score = len(hits) / len(ground_truth_points) if ground_truth_points else 1.0

    return {
        "hits": hits,
        "misses": misses,
        "score": score,
        "total": len(ground_truth_points),
    }


def load_user_context(fixtures_dir: Path) -> str:
    """Load user context from fixtures."""
    profile_path = fixtures_dir / "profile.md"
    if profile_path.exists():
        return profile_path.read_text()
    return ""


def find_available_papers(papers_dir: Path, date: str = None) -> list[dict]:
    """
    Find papers with indexes available for deep research.

    Args:
        papers_dir: Root directory containing papers
        date: Optional date to filter (YYYY-MM-DD)

    Returns:
        List of dicts with paper_id, title, date, has_index
    """
    papers = []

    if date:
        date_dirs = [papers_dir / date] if (papers_dir / date).exists() else []
    else:
        date_dirs = [d for d in papers_dir.iterdir() if d.is_dir()]

    for date_dir in sorted(date_dirs):
        for paper_dir in date_dir.iterdir():
            if not paper_dir.is_dir():
                continue

            metadata_file = paper_dir / "metadata.json"
            index_dir = paper_dir / "paper.index"

            if metadata_file.exists():
                try:
                    metadata = json.loads(metadata_file.read_text())
                    papers.append({
                        "paper_id": metadata.get("id", paper_dir.name),
                        "title": metadata.get("title", "Unknown"),
                        "date": date_dir.name,
                        "has_index": index_dir.exists(),
                        "path": str(paper_dir),
                    })
                except Exception:
                    continue

    return papers


def run_single_eval(
    paper_id: str,
    user_query: str,
    user_context: str,
    papers_dir: Path,
    date: str,
    research_agent,
    evaluator,
    ground_truth_points: list[str] = None,
) -> dict:
    """Run evaluation for a single query."""
    try:
        # Time the deep research agent
        research_start = time.time()
        result = research_agent.run(
            paper_id=paper_id,
            user_query=user_query,
            user_context=user_context,
            date=date,
            papers_dir=papers_dir,
            return_details=True,
        )
        research_time = time.time() - research_start

        # Time the LLM-as-judge evaluation
        eval_start = time.time()
        eval_results = evaluator.evaluate(
            user_query=user_query,
            abstract=result["abstract"],
            retrieved_chunks=result["retrieved_chunks"],
            answer=result["answer"],
        )
        eval_time = time.time() - eval_start

        # Ground truth check (fast, no LLM)
        gt_results = None
        if ground_truth_points:
            gt_results = check_ground_truth(result["answer"], ground_truth_points)

        return {
            "success": True,
            "paper_id": paper_id,
            "query": user_query,
            "answer": result["answer"],
            "metrics": eval_results["metrics"],
            "ground_truth": gt_results,
            "timing": {
                "research_seconds": research_time,
                "eval_seconds": eval_time,
                "total_seconds": research_time + eval_time,
            },
        }

    except Exception as e:
        return {
            "success": False,
            "paper_id": paper_id,
            "query": user_query,
            "error": str(e),
        }


def run_eval(
    paper_id: str = None,
    user_query: str = None,
    papers_dir: Path = None,
    date: str = None,
    run_all: bool = False,
):
    """
    Run deep research evaluation.

    Args:
        paper_id: Specific paper to analyze (optional)
        user_query: Query to ask about the paper
        papers_dir: Directory containing papers (defaults to fixtures)
        date: Date to search (optional)
        run_all: Run all test cases
    """
    # Setup paths
    eval_dir = project_root / ".dxtr_eval"
    fixtures_dir = Path(__file__).parent / "fixtures"

    if papers_dir is None:
        papers_dir = fixtures_dir

    # Create eval directory
    eval_dir.mkdir(exist_ok=True)

    print("=" * 60)
    print("DEEP RESEARCH AGENT EVALUATION")
    print("=" * 60)

    # Load user context
    print("\n[1/3] Loading user context...")
    if (fixtures_dir / "profile.md").exists():
        user_context = load_user_context(fixtures_dir)
        print(f"  Loaded from fixtures: {len(user_context)} chars")
    else:
        print("  WARNING: No profile found. Using empty context.")
        user_context = ""

    # Find available papers
    print("\n[2/3] Finding papers with indexes...")
    all_papers = find_available_papers(papers_dir, date)
    indexed_papers = [p for p in all_papers if p["has_index"]]

    print(f"  Papers with indexes: {len(indexed_papers)}")
    for p in indexed_papers:
        print(f"    - {p['paper_id']}: {p['title'][:50]}...")

    if not indexed_papers:
        print("\n  ERROR: No papers with indexes found.")
        return

    # Initialize agents
    print("\n[3/3] Running evaluations...")
    research_agent = deep_research.Agent()

    from eval.deep_research import agent as eval_agent
    evaluator = eval_agent.Agent()

    # Determine which test cases to run
    if run_all or (paper_id is None and user_query is None):
        # Run all test cases
        test_cases = TEST_CASES
        print(f"\n  Running {len(test_cases)} test cases...")
    else:
        # Run single query
        if paper_id is None:
            paper_id = indexed_papers[0]["paper_id"]
        if date is None:
            # Find date for this paper
            for p in indexed_papers:
                if p["paper_id"] == paper_id:
                    date = p["date"]
                    break
        if user_query is None:
            user_query = "What is the main contribution of this paper?"

        test_cases = [{"paper_id": paper_id, "query": user_query, "tests": "custom"}]

    # Run evaluations
    results = []
    for i, tc in enumerate(test_cases, 1):
        print(f"\n{'='*60}")
        print(f"TEST {i}/{len(test_cases)}")
        print(f"Paper: {tc['paper_id']}")
        print(f"Query: {tc['query'][:70]}...")
        print(f"Testing: {tc['tests']}")
        print("-" * 60)

        # Find date for paper
        tc_date = date
        if tc_date is None:
            for p in indexed_papers:
                if p["paper_id"] == tc["paper_id"]:
                    tc_date = p["date"]
                    break

        result = run_single_eval(
            paper_id=tc["paper_id"],
            user_query=tc["query"],
            user_context=user_context,
            papers_dir=papers_dir,
            date=tc_date,
            research_agent=research_agent,
            evaluator=evaluator,
            ground_truth_points=tc.get("ground_truth_points"),
        )
        results.append(result)

        if result["success"]:
            m = result["metrics"]
            gt = result.get("ground_truth")
            t = result["timing"]

            print(f"\nTiming: research={t['research_seconds']:.1f}s, eval={t['eval_seconds']:.1f}s, total={t['total_seconds']:.1f}s")
            print(f"LLM-Judge: N={m['novelty_score']} R={m['relevance_score']} "
                  f"C={m['completeness_score']} F={m['faithfulness_score']} "
                  f"-> {m['overall_score']:.1f}/5")

            # Ground truth results
            if gt:
                print(f"Ground Truth: {len(gt['hits'])}/{gt['total']} points ({gt['score']*100:.0f}%)")
                if gt["misses"]:
                    print(f"  Missing: {', '.join(gt['misses'][:3])}")

            # Show issues if any score < 4
            for key in ["relevance_issues", "completeness_issues", "faithfulness_issues"]:
                issues = m.get(key, [])
                if issues:
                    print(f"  {key}: {issues[0][:60]}...")
        else:
            print(f"\nERROR: {result['error']}")

    # Aggregate results
    print("\n" + "=" * 60)
    print("AGGREGATE RESULTS")
    print("=" * 60)

    successful = [r for r in results if r["success"]]
    if successful:
        avg_novelty = sum(r["metrics"]["novelty_score"] for r in successful) / len(successful)
        avg_relevance = sum(r["metrics"]["relevance_score"] for r in successful) / len(successful)
        avg_completeness = sum(r["metrics"]["completeness_score"] for r in successful) / len(successful)
        avg_faithfulness = sum(r["metrics"]["faithfulness_score"] for r in successful) / len(successful)
        avg_overall = sum(r["metrics"]["overall_score"] for r in successful) / len(successful)

        # Timing aggregate
        research_times = [r["timing"]["research_seconds"] for r in successful]
        eval_times = [r["timing"]["eval_seconds"] for r in successful]
        total_times = [r["timing"]["total_seconds"] for r in successful]

        avg_research = sum(research_times) / len(research_times)
        avg_eval = sum(eval_times) / len(eval_times)
        avg_total = sum(total_times) / len(total_times)
        min_research = min(research_times)
        max_research = max(research_times)

        # Ground truth aggregate
        gt_results = [r["ground_truth"] for r in successful if r.get("ground_truth")]
        avg_gt = sum(g["score"] for g in gt_results) / len(gt_results) if gt_results else None

        print(f"\nTests run: {len(results)}")
        print(f"Successful: {len(successful)}")

        print(f"\nTiming (Deep Research Only):")
        print(f"  Average: {avg_research:.1f}s")
        print(f"  Min:     {min_research:.1f}s")
        print(f"  Max:     {max_research:.1f}s")
        print(f"  Total:   {sum(research_times):.1f}s")
        print(f"\nTiming (Including Eval):")
        print(f"  Avg research: {avg_research:.1f}s")
        print(f"  Avg eval:     {avg_eval:.1f}s")
        print(f"  Avg total:    {avg_total:.1f}s")

        print(f"\nLLM-as-Judge Scores:")
        print(f"  Novelty:      {avg_novelty:.2f}/5")
        print(f"  Relevance:    {avg_relevance:.2f}/5")
        print(f"  Completeness: {avg_completeness:.2f}/5")
        print(f"  Faithfulness: {avg_faithfulness:.2f}/5")
        print(f"  ----------------------")
        print(f"  LLM OVERALL:  {avg_overall:.2f}/5")

        if avg_gt is not None:
            print(f"\nGround Truth Coverage:")
            print(f"  Key Points Hit: {avg_gt*100:.0f}%")

        # Combined score (LLM score + GT coverage)
        if avg_gt is not None:
            combined = (avg_overall / 5 + avg_gt) / 2 * 100
            print(f"\nCOMBINED SCORE: {combined:.0f}%")

    # Save results
    output_file = eval_dir / "deep_research_eval_results.json"
    output_file.write_text(json.dumps(results, indent=2))
    print(f"\nResults saved: {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Deep Research Agent Evaluation")
    parser.add_argument(
        "--paper-id",
        type=str,
        default=None,
        help="Paper ID to analyze (defaults to first available)",
    )
    parser.add_argument(
        "--query",
        type=str,
        default=None,
        help="Query to ask about the paper",
    )
    parser.add_argument(
        "--papers-dir",
        type=str,
        default=None,
        help="Directory containing papers (defaults to config)",
    )
    parser.add_argument(
        "--date",
        type=str,
        default=None,
        help="Date to search (YYYY-MM-DD format)",
    )

    args = parser.parse_args()

    papers_dir = Path(args.papers_dir) if args.papers_dir else None

    run_eval(
        paper_id=args.paper_id,
        user_query=args.query,
        papers_dir=papers_dir,
        date=args.date,
    )


if __name__ == "__main__":
    main()
