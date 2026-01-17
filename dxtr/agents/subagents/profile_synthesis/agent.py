"""Profile synthesis agent - creates enriched user profiles from artifacts."""

from pathlib import Path

from pydantic_ai import Agent

from agents.utils import model, load_system_prompt


SYSTEM_PROMPT = load_system_prompt(Path(__file__).parent / "system.md")

profile_synthesis_agent = Agent(model, system_prompt=SYSTEM_PROMPT)
