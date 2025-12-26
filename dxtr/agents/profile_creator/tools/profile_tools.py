"""
Profile creation tools for the main chat agent
"""

from ..agent import create_profile as _create_profile


def create_profile() -> dict:
    """
    Create or recreate user profile from profile.md.

    Analyzes GitHub repositories and generates enriched profile using Gemma3.

    Returns:
        dict with keys:
            - success: bool
            - profile_path: str (path to enriched profile)
            - error: str (if failed)
    """
    return _create_profile()


# Tool definition for Ollama function calling
TOOL_DEFINITION = {
    "type": "function",
    "function": {
        "name": "create_profile",
        "description": "Create or recreate the user's profile from profile.md. This analyzes their GitHub repositories and generates an enriched profile. Use this when the user wants to create/update their profile, or when no profile exists.",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": []
        }
    }
}
