"""
Deep Research Agent

Performs comprehensive analysis of research papers using RAG to answer
specific questions about papers, tailored to user's interests.
"""

import json
from pathlib import Path

from llama_index.core import StorageContext, load_index_from_storage
from llama_index.core.prompts import PromptTemplate
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding

from dxtr.config import config


class DeepResearchAgent:
    """Agent for in-depth research paper analysis using RAG."""

    def __init__(self):
        """Initialize deep research agent."""
        model_config = config.get_model_config("deep_research")
        self.model_name = model_config.name
        self.temperature = model_config.temperature

    def analyze_paper(self, paper_id: str, user_query: str, user_context: str, date: str = None) -> str:
        """
        Answer a question about a research paper using RAG.

        Args:
            paper_id: Paper ID (e.g., "2512.12345")
            user_query: The user's original question/request about the paper
            user_context: User profile and interests
            date: Date in YYYY-MM-DD format (optional)

        Returns:
            Answer based on paper content and user context
        """
        print(f"  [Agent] Finding paper {paper_id}...")
        # Find the paper
        paper_dir = self._find_paper(paper_id, date)
        if not paper_dir:
            return f"Paper {paper_id} not found. Make sure it has been downloaded."

        print(f"  [Agent] Found at {paper_dir}")

        # Load the persisted index
        index_dir = paper_dir / "paper.index"
        if not index_dir.exists():
            return f"Index not found for paper {paper_id}. Run 'dxtr get-papers' to build it."

        # Load metadata
        metadata_file = paper_dir / "metadata.json"
        metadata = {}
        if metadata_file.exists():
            metadata = json.loads(metadata_file.read_text())

        print(f"  [Agent] Loading index...")
        # Load index and create query engine
        embed_model = OllamaEmbedding(model_name="nomic-embed-text")
        storage_context = StorageContext.from_defaults(persist_dir=str(index_dir))
        index = load_index_from_storage(storage_context, embed_model=embed_model)

        total_chunks = len(index.docstore.docs)
        print(f"  [Agent] Index loaded ({total_chunks} chunks)")

        llm = Ollama(model=self.model_name, request_timeout=120.0, temperature=self.temperature)
        query_engine = index.as_query_engine(llm=llm, similarity_top_k=3, streaming=True)

        # Add user context to the synthesis prompt
        if user_context:
            qa_template = PromptTemplate(
                f"{user_context}\n\n"
                "-----\n\n"
                "You are analyzing a research paper with the user's background in mind.\n\n"
                "Context from the paper:\n"
                "{context_str}\n\n"
                "-----\n\n"
                "Question: {query_str}\n\n"
                "Answer based on the paper context above, tailoring your response to the user's "
                "interests and background described in the profile."
            )
            query_engine.update_prompts({"response_synthesizer:text_qa_template": qa_template})

        # Query and stream response
        print(f"  [Agent] Embedding query and retrieving chunks...")
        response = query_engine.query(user_query)

        print(f"  [Agent] Retrieved {len(response.source_nodes)}/{total_chunks} chunks:")
        for i, node in enumerate(response.source_nodes, 1):
            score = node.score if hasattr(node, 'score') else None
            score_str = f"{score:.3f}" if isinstance(score, float) else "N/A"
            preview = node.text[:80].replace('\n', ' ')
            print(f"    {i}. Score {score_str}: {preview}...")

        print(f"  [Agent] Generating answer with {self.model_name}...\n")
        print(f"[Deep Research Agent]: ", end="", flush=True)

        # Stream response to stdout
        response.print_response_stream()
        print("\n")

        # Return full text for context
        return str(response)

    def _find_paper(self, paper_id: str, date: str = None) -> Path | None:
        """
        Find paper directory by ID.

        Args:
            paper_id: Paper ID to search for
            date: Optional date to narrow search

        Returns:
            Path to paper directory or None if not found
        """
        papers_root = config.paths.papers_dir

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


# Global instance for backward compatibility
_agent = DeepResearchAgent()


def analyze_paper(paper_id: str, user_query: str, user_context: str, date: str = None) -> str:
    """
    Answer a question about a research paper using RAG.

    This is a convenience function that delegates to the agent instance.

    Args:
        paper_id: Paper ID (e.g., "2512.12345")
        user_query: The user's original question/request about the paper
        user_context: User profile and interests
        date: Date in YYYY-MM-DD format (optional)

    Returns:
        Answer based on paper content and user context
    """
    return _agent.analyze_paper(paper_id, user_query, user_context, date)
