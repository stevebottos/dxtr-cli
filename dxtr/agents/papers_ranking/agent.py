"""
Papers Ranking Agent

Ranks research papers by relevance to user profile and interests.
Uses run_batch to parallelize across papers, with forked reasoning per paper.
"""

import re
import json
from pathlib import Path

import sglang as sgl

from dxtr.agents.base import BaseAgent
from dxtr.config_v2 import config


TOOL_DEFINITION = {
    "type": "function",
    "function": {
        "name": "rank_papers",
        "description": "Rank research papers by relevance to the user's profile and interests.",
        "parameters": {
            "type": "object",
            "properties": {
                "user_query": {
                    "type": "string",
                    "description": "The user's original question/request about papers",
                },
                "date": {
                    "type": "string",
                    "description": "Date in YYYY-MM-DD format (optional, defaults to today)",
                },
            },
            "required": ["user_query"],
        },
    },
}

# Schema for scoring a single paper
SINGLE_PAPER_SCORE_SCHEMA = {
    "type": "object",
    "properties": {
        "score": {"type": "integer"},
        "relevance": {"type": "string"},
    },
    "required": ["score", "relevance"],
}

REFLECTION_SCHEMA = {
    "type": "object",
    "properties": {
        "reflection": {"type": "string"},
        "relevance": {"type": "string"},
    },
    "required": ["reflection", "relevance"],
}

SCORE_SCHEMA = {
    "type": "object",
    "properties": {
        "value": {"type": "integer"},
    },
    "required": ["value"],
}


class Agent(BaseAgent):
    """Paper ranking with run_batch parallelization and forked reasoning per paper."""

    def __init__(self):
        """Initialize papers ranking agent."""
        super().__init__()
        self.system_prompt = self.load_system_prompt(
            Path(__file__).parent / "system.md"
        )

    def _load_papers(self, date: str, papers_dir: Path = None) -> list[dict]:
        """Load paper metadata for a given date."""
        if papers_dir is None:
            papers_dir = config.paths.papers_dir

        date_dir = papers_dir / date
        if not date_dir.exists():
            return []

        papers = []
        for paper_dir in date_dir.iterdir():
            if not paper_dir.is_dir():
                continue

            metadata_file = paper_dir / "metadata.json"
            if metadata_file.exists():
                try:
                    metadata = json.loads(metadata_file.read_text())
                    papers.append(metadata)
                except Exception:
                    continue

        return papers

    @staticmethod
    @sgl.function
    def _rank_papers_pre(s, user_context, paper, system_prompt, max_tokens, temp):
        """SGLang function for the final profile enrichment step."""

        s["paper_id"] = paper["id"]
        s["paper_title"] = paper["title"]
        s["upvotes"] = paper["upvotes"]

        s += sgl.system(system_prompt)

        num_forks = 3
        forks = s.fork(num_forks)
        for i in range(num_forks):
            # We use 'forks[i]' to point to a specific parallel branch
            forks[i] += sgl.user(f"The user profile: {user_context}")
            forks[i] += sgl.user(f"The paper abstract: {paper['summary']}")
            forks[i] += sgl.user(
                "Now consider possible scores for this paper, with reasons. Then, settle on a final score."
            )
            forks[i] += sgl.gen(
                "relevance",
                max_tokens=600,
                temperature=0.2,
                frequency_penalty=1.1,
                presence_penalty=0.1,
            )
            forks[i] += (
                "\n---\nFinal Score (1-5), integer only. You must answer in valid json."
            )
            forks[i] += sgl.gen(
                "score",
                max_tokens=20,
                temperature=0,
                json_schema=json.dumps(SCORE_SCHEMA),
            )

        # 3. Retrieve the results
        # Accessing the fork index automatically waits for completion
        reasoning = [f["relevance"] for f in forks]
        scores = [int(json.loads(f["score"])["value"]) for f in forks]

        # Optional: store them back in the main state if needed
        s["relevance"] = reasoning
        s["score"] = scores

    def run(
        self,
        date: str,
        user_context: str,
        user_query: str,
        papers_dir: Path = None,
        verbose: bool = True,
    ) -> dict:
        """
        Rank papers using run_batch parallelization with forked reasoning per paper.
        """
        papers = self._load_papers(date, papers_dir)

        if not papers:
            return {"error": f"No papers found for {date}"}

        batch_data = [
            {
                "user_context": user_context,
                "paper": p,
                "system_prompt": self.system_prompt,
                "max_tokens": 500,
                "temp": 0.0,
            }
            for p in papers
        ]
        states = self._rank_papers_pre.run_batch(
            batch_data, num_threads=128, progress_bar=True
        )

        # Extract results from batch
        results = {}
        for i, state in enumerate(states):
            paper = papers[i]
            paper_id = paper.get("id")
            print(state["relevance"])
            print(state["score"])
            score = round(sum(state["score"]) / len(state["score"]), 1)
            print(score)

            print("------")
            # match = re.search(r"<score>\s*([1-5])\s*</score>", state["score"])
            # if match:
            #     score = int(match.group(1))
            # else:
            #     score = -1

            # if state["upvotes"] > 100:
            #     score = 5  # Auto score 5 if the paper is popular

            reason = state["relevance"]

            results[paper_id] = {
                "title": state["paper_title"],
                "score": score,
                "reason": reason,
            }
        # Sort by score descending
        sorted_results = dict(
            sorted(results.items(), key=lambda x: x[1]["score"] or 0, reverse=True)
        )

        # Build final ranking string
        ranking_lines = []
        for rank, (pid, data) in enumerate(sorted_results.items(), 1):
            ranking_lines.append(f"{rank}. [{data['score']}/5] {data['title']}")
        final_ranking = "\n".join(ranking_lines)

        # Return in format compatible with eval
        individual_scores = [
            {
                "id": pid,
                "title": data["title"],
                "final_score": data["score"],
                "reason": data.get("reason", ""),
                "boosted": data.get("boosted", False),
            }
            for pid, data in sorted_results.items()
        ]

        return {
            "paper_count": len(papers),
            "individual_scores": individual_scores,
            "scores": {pid: data["score"] for pid, data in sorted_results.items()},
            "final_ranking": final_ranking,
        }
