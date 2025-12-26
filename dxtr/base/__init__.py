"""
Base classes for DXTR agents.

Provides:
- Agent: Base class for all agents
- ToolRegistry: Tool registration and management
- PromptManager: Prompt loading from filesystem (zero prompts in code)
"""

from .prompts import PromptManager
from .tools import ToolRegistry
from .agent import Agent

__all__ = ["Agent", "ToolRegistry", "PromptManager"]
