"""Main chat agent that orchestrates specialist agents."""

from pathlib import Path

from pydantic_ai import Agent

from agents.utils import model, load_system_prompt
from agents import tools


SYSTEM_PROMPT = load_system_prompt(Path(__file__).parent / "system.md")

main_agent = Agent(model, system_prompt=SYSTEM_PROMPT)

# Register tools
main_agent.tool(tools.read_file)
main_agent.tool(tools.analyze_github)
main_agent.tool(tools.synthesize_profile)
