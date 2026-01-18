"""Papers ranking agent using parallel scoring."""

import asyncio
from pathlib import Path

from pydantic import BaseModel, Field
from pydantic_ai import Agent

from dxtr import papers_ranker, load_system_prompt


class PaperScore(BaseModel):
    """Score for a single paper."""
    score: int = Field(ge=1, le=10)
    reason: str


# Agent for scoring ONE paper
agent = Agent(
    papers_ranker,
    system_prompt=load_system_prompt(Path(__file__).parent / "system.md"),
    output_type=PaperScore,
)


async def score_one_paper(
    paper_id: str,
    title: str,
    abstract: str,
    profile: str,
    idx: int,
    total: int,
) -> dict:
    """Score a single paper against the profile."""
    short_title = title[:40] + "..." if len(title) > 40 else title
    print(f"  [{idx}/{total}] Scoring: {short_title}", flush=True)

    prompt = f"""## User Profile
{profile}

## Paper to Score
**{title}**

{abstract}
"""
    try:
        result = await agent.run(prompt)
        score = result.output.score
        reason = result.output.reason
        print(f"  [{idx}/{total}] Done: {score}/10 - {short_title}", flush=True)
        return {
            "id": paper_id,
            "title": title,
            "score": score,
            "reason": reason,
        }
    except Exception as e:
        print(f"  [{idx}/{total}] Error: {short_title} - {e}", flush=True)
        return {
            "id": paper_id,
            "title": title,
            "score": 0,
            "reason": f"Error: {e}",
        }


async def rank_papers_parallel(profile: str, papers: dict[str, dict]) -> list[dict]:
    """Rank all papers in parallel.

    Args:
        profile: User's synthesized profile
        papers: Dict of {paper_id: {title, summary}}

    Returns:
        List of scored papers sorted by score descending
    """
    total = len(papers)
    print(f"Ranking {total} papers in parallel...")

    tasks = []
    for idx, (paper_id, data) in enumerate(papers.items(), 1):
        task = score_one_paper(
            paper_id=paper_id,
            title=data.get("title", "Untitled"),
            abstract=data.get("summary", ""),
            profile=profile,
            idx=idx,
            total=total,
        )
        tasks.append(task)

    results = await asyncio.gather(*tasks)

    # Sort by score descending
    return sorted(results, key=lambda x: x["score"], reverse=True)
