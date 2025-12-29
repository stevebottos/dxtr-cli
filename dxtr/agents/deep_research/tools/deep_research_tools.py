"""
Deep Research tools for the main chat agent to delegate to deep_research agent
"""

from ..agent import analyze_paper as _analyze_paper


def deep_research(paper_id: str, user_query: str, date: str = None) -> dict:
    """
    Answer a specific question about a research paper using RAG.

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
    print(f"  User query: {user_query[:80]}..." if len(user_query) > 80 else f"  User query: {user_query}")

    try:
        # Normalize paper ID (remove arxiv prefix if present)
        if "/" in paper_id:
            paper_id = paper_id.split("/")[-1]

        print(f"  Loading user context...")
        # Load user context from CLI
        from ....cli import _load_user_context
        user_context = _load_user_context()
        print(f"  User context loaded ({len(user_context)} chars)")

        print(f"  Calling deep research agent...")
        # Call deep research agent
        answer = _analyze_paper(paper_id, user_query, user_context, date)

        print(f"  Answer received ({len(answer)} chars)")

        return {
            "success": True,
            "paper_id": paper_id,
            "answer": answer
        }

    except Exception as e:
        print(f"  ERROR: {e}")
        import traceback
        traceback.print_exc()
        return {
            "success": False,
            "paper_id": paper_id or "unknown",
            "error": str(e)
        }


# Tool definition for Ollama function calling
TOOL_DEFINITION = {
    "type": "function",
    "function": {
        "name": "deep_research",
        "description": "Answer a question about a research paper using retrieval-augmented generation. Use this when the user asks to analyze, summarize, or explore a specific paper. Retrieves relevant sections and provides tailored answers based on the user's profile. ALWAYS pass the user's original question/request verbatim.",
        "parameters": {
            "type": "object",
            "properties": {
                "paper_id": {
                    "type": "string",
                    "description": "Paper ID (e.g., '2512.12345' or 'arxiv:2512.12345')"
                },
                "user_query": {
                    "type": "string",
                    "description": "The user's original question/request about the paper, passed through verbatim (e.g., 'Do a deep dive on this paper, highlighting the important parts, and suggest me a project to expand on it')"
                },
                "date": {
                    "type": "string",
                    "description": "Date in YYYY-MM-DD format (optional, will search all dates if not provided)"
                }
            },
            "required": ["paper_id", "user_query"]
        }
    }
}
