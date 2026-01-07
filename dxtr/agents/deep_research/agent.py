"""
Deep Research Agent

Performs comprehensive analysis of research papers using agentic RAG:
1. Generates exploration questions based on user context and abstract
2. Retrieves relevant chunks for each question (deduplicated)
3. Answers user's query using the retrieved chunks

The exploration questions are a retrieval strategy to find relevant sections
beyond what a single query would return. They are NOT answered themselves.
"""

import json
import re
from pathlib import Path

import sglang as sgl
from llama_index.core import StorageContext, load_index_from_storage

from dxtr.agents.base import BaseAgent
from dxtr.config_v2 import config
from dxtr.docling_utils import get_embed_model


TOOL_DEFINITION = {
    "type": "function",
    "function": {
        "name": "deep_research",
        "description": "Answer questions about a research paper using RAG-based deep analysis.",
        "parameters": {
            "type": "object",
            "properties": {
                "paper_id": {
                    "type": "string",
                    "description": "Paper ID (e.g., '2512.12345')",
                },
                "user_query": {
                    "type": "string",
                    "description": "The user's question about the paper",
                },
                "date": {
                    "type": "string",
                    "description": "Date in YYYY-MM-DD format (optional)",
                },
            },
            "required": ["paper_id", "user_query"],
        },
    },
}


class Agent(BaseAgent):
    """Agent for in-depth research paper analysis using RAG."""

    def __init__(self):
        """Initialize deep research agent."""
        super().__init__()
        self.prompts_dir = Path(__file__).parent / "prompts"
        self.exploration_prompt = self.load_system_prompt(
            self.prompts_dir / "exploration.md"
        )

    @staticmethod
    @sgl.function
    def _generate_questions_func(s, system_prompt, user_message, max_tokens, temp):
        """SGLang function to generate exploration questions."""
        s += sgl.system(system_prompt)
        s += sgl.user(user_message)
        s += sgl.assistant(sgl.gen("questions", max_tokens=max_tokens, temperature=temp))

    @staticmethod
    @sgl.function
    def _answer_query_func(s, prompt, max_tokens, temp):
        """SGLang function to answer user query with retrieved context."""
        s += sgl.user(prompt)
        s += sgl.assistant(sgl.gen("answer", max_tokens=max_tokens, temperature=temp))

    def run(
        self,
        paper_id: str,
        user_query: str,
        user_context: str,
        date: str = None,
        papers_dir: Path = None,
        return_details: bool = False,
    ) -> str | dict:
        """
        Answer a question about a research paper using agentic RAG.

        Workflow:
        1. Generate exploration questions (retrieval strategy)
        2. Retrieve chunks for all questions (deduplicated)
        3. Answer user's query directly with retrieved chunks

        Args:
            paper_id: Paper ID (e.g., "2512.12345")
            user_query: The user's original question/request about the paper
            user_context: User profile and interests
            date: Date in YYYY-MM-DD format (optional)
            papers_dir: Directory containing papers (optional, defaults to config)
            return_details: If True, return dict with answer and intermediate data

        Returns:
            Answer string, or dict with answer/abstract/chunks if return_details=True
        """
        print(f"  [Agent] Finding paper {paper_id}...")
        paper_dir = self._find_paper(paper_id, date, papers_dir)
        if not paper_dir:
            return f"Paper {paper_id} not found. Make sure it has been downloaded."

        print(f"  [Agent] Found at {paper_dir}")

        # Load abstract
        abstract = self._load_abstract(paper_dir)
        if not abstract:
            return f"No metadata found for paper {paper_id}."

        # Load index
        index_dir = paper_dir / "paper.index"
        if not index_dir.exists():
            return f"Index not found for paper {paper_id}. Run 'dxtr get-papers' to build it."

        print(f"  [Agent] Loading index...")
        storage_context = StorageContext.from_defaults(persist_dir=str(index_dir))
        index = load_index_from_storage(storage_context, embed_model=get_embed_model())

        total_chunks = len(index.docstore.docs)
        print(f"  [Agent] Index loaded ({total_chunks} chunks)")

        # STEP 1: Generate exploration questions
        print(f"\n  [Step 1/3] Generating exploration questions...")
        exploration_questions = self._generate_exploration_questions(
            abstract, user_context, user_query
        )

        if not exploration_questions:
            return "Failed to generate exploration questions."

        # STEP 2: Retrieve chunks for all questions (deduplicated)
        print(f"\n  [Step 2/3] Retrieving relevant sections...")
        retriever = index.as_retriever(similarity_top_k=3)
        all_chunks = self._retrieve_deduplicated(
            retriever, exploration_questions, total_chunks
        )

        # STEP 3: Answer user's query with retrieved chunks
        print(f"\n  [Step 3/3] Answering your question...\n")
        final_answer = self._answer_with_retrieved_chunks(
            user_query, exploration_questions, all_chunks, user_context
        )

        if return_details:
            return {
                "answer": final_answer,
                "abstract": abstract,
                "exploration_questions": exploration_questions,
                "retrieved_chunks": [node.text for node in all_chunks],
                "total_chunks": total_chunks,
            }

        return final_answer

    def _load_abstract(self, paper_dir: Path) -> str | None:
        """Load paper abstract from metadata."""
        metadata_file = paper_dir / "metadata.json"
        if not metadata_file.exists():
            return None

        metadata = json.loads(metadata_file.read_text())
        return metadata.get("summary", "")

    def _generate_exploration_questions(
        self, abstract: str, user_context: str, user_query: str
    ) -> list[str]:
        """
        Generate 3-5 exploration questions using the exploration prompt.

        Args:
            abstract: Paper abstract
            user_context: User profile and interests
            user_query: User's original query

        Returns:
            List of exploration questions (3-5 questions)
        """
        # Build user message
        user_message = f"""# User's Background & Interests

{user_context}

---

# Paper Abstract

{abstract}

---

# User's Query

"{user_query}"

---

Generate 3-5 targeted exploration questions for this paper."""

        # Call SGLang function
        state = self._generate_questions_func.run(
            system_prompt=self.exploration_prompt,
            user_message=user_message,
            max_tokens=1000,
            temp=0.3,
        )

        # Parse questions from response (clean think tags first)
        questions_text = re.sub(r"<think>[\s\S]*?</think>", "", state["questions"]).strip()
        questions = self._parse_questions(questions_text)

        # Display questions
        print(f"\n    Generated {len(questions)} exploration questions:")
        for i, q in enumerate(questions, 1):
            print(f"    {i}. {q}")

        return questions

    def _parse_questions(self, text: str) -> list[str]:
        """Parse numbered questions from LLM response."""
        questions = []
        # Match lines starting with "1.", "2.", etc.
        for line in text.strip().split("\n"):
            match = re.match(r"^\s*\d+\.\s*(.+)$", line)
            if match:
                questions.append(match.group(1).strip())
        return questions

    def _retrieve_deduplicated(
        self, retriever, questions: list[str], total_chunks: int
    ) -> list:
        """
        Retrieve chunks for all questions and deduplicate.

        Args:
            retriever: LlamaIndex retriever
            questions: List of exploration questions
            total_chunks: Total chunks in index

        Returns:
            Deduplicated list of chunks
        """
        all_chunks = []
        chunk_ids_seen = set()

        for i, question in enumerate(questions, 1):
            nodes = retriever.retrieve(question)
            new_chunks = 0
            for node in nodes:
                node_id = node.node_id
                if node_id not in chunk_ids_seen:
                    all_chunks.append(node)
                    chunk_ids_seen.add(node_id)
                    new_chunks += 1

            print(f"    Q{i}: Retrieved {len(nodes)} chunks ({new_chunks} new)")

        print(f"\n    Total unique chunks: {len(all_chunks)}/{total_chunks}")
        return all_chunks

    def _answer_with_retrieved_chunks(
        self,
        user_query: str,
        exploration_questions: list[str],
        chunks: list,
        user_context: str,
    ) -> str:
        """
        Answer user's query using chunks retrieved via exploration questions.

        Args:
            user_query: User's original question
            exploration_questions: Questions used for retrieval (for context)
            chunks: Retrieved chunks
            user_context: User profile

        Returns:
            Answer string
        """
        # Build context from all chunks
        chunk_texts = [node.text for node in chunks]
        context_str = "\n\n---\n\n".join(chunk_texts)

        # Build list of exploration questions
        questions_list = "\n".join(
            [f"{i}. {q}" for i, q in enumerate(exploration_questions, 1)]
        )

        # Build prompt
        prompt = f"""{user_context}

-----

You are analyzing a research paper with the user's background in mind.

To find the most relevant sections of this paper, I generated these exploration questions:

{questions_list}

These questions helped retrieve the following relevant sections from the paper:

{context_str}

-----

User's Question: {user_query}

-----

Answer the user's question using the retrieved paper sections above. The exploration questions were just a retrieval strategy - focus on answering the user's actual question. Be specific, cite details from the paper, and tailor your response to the user's background and interests."""

        # Call SGLang function
        state = self._answer_query_func.run(
            prompt=prompt,
            max_tokens=1500,
            temp=0.2,
        )

        # Clean think tags from response
        answer = re.sub(r"<think>[\s\S]*?</think>", "", state["answer"]).strip()
        return answer

    def _find_paper(
        self, paper_id: str, date: str = None, papers_dir: Path = None
    ) -> Path | None:
        """
        Find paper directory by ID.

        Args:
            paper_id: Paper ID to search for
            date: Optional date to narrow search
            papers_dir: Directory containing papers (optional, defaults to config)

        Returns:
            Path to paper directory or None if not found
        """
        papers_root = papers_dir if papers_dir else config.paths.papers_dir

        if date:
            # Search specific date
            date_dir = papers_root / date
            if date_dir.exists():
                paper_dir = date_dir / paper_id
                if paper_dir.exists():
                    return paper_dir
        else:
            # Search all dates
            for date_dir in papers_root.iterdir():
                if date_dir.is_dir():
                    paper_dir = date_dir / paper_id
                    if paper_dir.exists():
                        return paper_dir

        return None


# Convenience function for backward compatibility
def analyze_paper(
    paper_id: str, user_query: str, user_context: str, date: str = None
) -> str:
    """
    Answer a question about a research paper using RAG.

    This is a convenience function that creates an agent instance.

    Args:
        paper_id: Paper ID (e.g., "2512.12345")
        user_query: The user's original question/request about the paper
        user_context: User profile and interests
        date: Date in YYYY-MM-DD format (optional)

    Returns:
        Answer based on paper content and user context
    """
    agent = Agent()
    return agent.run(paper_id, user_query, user_context, date)


def deep_research(paper_id: str, user_query: str, date: str = None) -> dict:
    """
    Tool function: Answer a specific question about a research paper using RAG.

    Uses retrieval-augmented generation to find relevant sections of the paper
    and provide a detailed, context-aware answer tailored to the user's background.

    Args:
        paper_id: Paper ID (e.g., "2512.12345" or just "12345")
        user_query: The user's original question/request about the paper
        date: Date in YYYY-MM-DD format (optional)

    Returns:
        dict with keys:
            - success: bool
            - paper_id: str (the paper analyzed)
            - answer: str (the answer to the question)
            - error: str (if failed)
    """
    print(f"\n[Deep Research Tool]")
    print(f"  Paper ID: {paper_id}")
    print(
        f"  User query: {user_query[:80]}..."
        if len(user_query) > 80
        else f"  User query: {user_query}"
    )

    try:
        # Normalize paper ID (remove arxiv prefix if present)
        if "/" in paper_id:
            paper_id = paper_id.split("/")[-1]

        print(f"  Loading user context...")
        # Load user context from CLI
        from dxtr.cli import _load_user_context

        user_context = _load_user_context()
        print(f"  User context loaded ({len(user_context)} chars)")

        print(f"  Calling deep research agent...")
        # Call deep research agent
        agent = Agent()
        answer = agent.run(paper_id, user_query, user_context, date)

        print(f"  Answer received ({len(answer)} chars)")

        return {"success": True, "paper_id": paper_id, "answer": answer}

    except Exception as e:
        print(f"  ERROR: {e}")
        import traceback

        traceback.print_exc()
        return {"success": False, "paper_id": paper_id or "unknown", "error": str(e)}
