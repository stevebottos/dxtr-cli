from pathlib import Path

from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider


# === Shared LLM Config ===
LITELLM_BASE_URL = "http://localhost:4000/v1"
LITELLM_API_KEY = "sk-your-virtual-key"

provider = OpenAIProvider(base_url=LITELLM_BASE_URL, api_key=LITELLM_API_KEY)
model = OpenAIChatModel("main", provider=provider)


def load_system_prompt(file_path: Path) -> str:
    """Load a system prompt from a markdown file."""
    return file_path.read_text().strip()
