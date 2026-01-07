"""
Evaluation Agent for Deep Research Pipeline

Verifies the quality of generated answers using LLM-as-judge.
"""

import json
from pathlib import Path

import sglang as sgl

from dxtr.agents.base import BaseAgent

from . import util


class Agent(BaseAgent):
    """Agent for evaluating deep research answer quality."""

    def __init__(self):
        super().__init__()
        self.temperature = 0.0  # Deterministic for evaluation
        self.max_tokens = 2000
        self.prompts_dir = Path(__file__).parent

    @staticmethod
    @sgl.function
    def evaluate_answer_func(
        s, user_query, abstract, retrieved_chunks, answer, system_prompt, max_tokens, temp, json_schema
    ):
        """SGLang function for evaluating generated answers."""
        s += sgl.system(system_prompt)
        s += sgl.user(
            f"## User Query\n{user_query}\n\n"
            f"## Paper Abstract\n{abstract}\n\n"
            f"## Retrieved Chunks\n{retrieved_chunks}\n\n"
            f"## Generated Answer\n{answer}"
        )
        s += sgl.assistant(
            sgl.gen(
                "evaluation",
                max_tokens=max_tokens,
                temperature=temp,
                json_schema=json_schema,
            )
        )

    def evaluate(
        self,
        user_query: str,
        abstract: str,
        retrieved_chunks: list[str],
        answer: str,
    ) -> dict:
        """Evaluate a generated answer.

        Args:
            user_query: The user's original question
            abstract: Paper abstract
            retrieved_chunks: List of retrieved chunk texts
            answer: The generated answer to evaluate

        Returns:
            Dict with evaluation result and metrics
        """
        system_prompt = self.load_system_prompt(self.prompts_dir / "eval_prompt.md")
        schema_json = json.dumps(util.ANSWER_VERIFICATION_SCHEMA)

        # Format chunks for evaluation
        chunks_text = "\n\n---\n\n".join(
            [f"[Chunk {i+1}]\n{chunk}" for i, chunk in enumerate(retrieved_chunks)]
        )

        print("Evaluating answer quality...")
        state = self.evaluate_answer_func.run(
            user_query=user_query,
            abstract=abstract,
            retrieved_chunks=chunks_text,
            answer=answer,
            system_prompt=system_prompt,
            max_tokens=self.max_tokens,
            temp=self.temperature,
            json_schema=schema_json,
        )

        try:
            evaluation = json.loads(state["evaluation"])
            metrics = util.compute_metrics(evaluation)
            return {"result": evaluation, "metrics": metrics}
        except json.JSONDecodeError:
            return {
                "result": {"error": "Failed to parse evaluation"},
                "metrics": {"error": "Parse failed"},
            }

    def run(
        self,
        user_query: str,
        abstract: str,
        retrieved_chunks: list[str],
        answer: str,
        output_dir: Path = None,
    ) -> dict:
        """Run evaluation and optionally save results.

        Args:
            user_query: The user's original question
            abstract: Paper abstract
            retrieved_chunks: List of retrieved chunk texts
            answer: The generated answer to evaluate
            output_dir: Directory to save results (optional)

        Returns:
            Dict with evaluation results
        """
        results = self.evaluate(user_query, abstract, retrieved_chunks, answer)

        # Print metrics
        util.print_metrics(results["metrics"])

        # Save results if output_dir provided
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            eval_file = output_dir / "evaluation_results.json"
            eval_file.write_text(json.dumps(results, indent=2))
            print(f"\nResults saved: {eval_file}")

        return results
