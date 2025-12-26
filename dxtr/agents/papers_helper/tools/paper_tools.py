"""
Paper tools for the main chat agent to delegate to papers_helper
"""

from datetime import datetime
from pathlib import Path
from ..agent import rank_papers as _rank_papers


def rank_papers(date: str = None) -> dict:
    """
    Rank papers by relevance to the user's profile and interests.

    Args:
        date: Date in YYYY-MM-DD format (default: today)

    Returns:
        dict with keys:
            - success: bool
            - date: str (the date queried)
            - ranking: str (formatted ranking with reasoning)
            - error: str (if failed)
    """
    try:
        if date is None:
            date = datetime.today().strftime("%Y-%m-%d")

        # Load user context from CLI
        from ....cli import _load_user_context
        user_context = _load_user_context()

        # Call papers_helper agent
        ranking = _rank_papers(date, user_context)

        return {
            "success": True,
            "date": date,
            "ranking": ranking
        }

    except Exception as e:
        return {
            "success": False,
            "date": date or "unknown",
            "error": str(e)
        }


# Tool definition for Ollama function calling
TOOL_DEFINITION = {
    "type": "function",
    "function": {
        "name": "rank_papers",
        "description": "Rank research papers by relevance to the user's profile and interests. Use this when the user asks to see today's papers, rank papers, or find relevant papers for a specific date.",
        "parameters": {
            "type": "object",
            "properties": {
                "date": {
                    "type": "string",
                    "description": "Date in YYYY-MM-DD format (optional, defaults to today)"
                }
            },
            "required": []
        }
    }
}
