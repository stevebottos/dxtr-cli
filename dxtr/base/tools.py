"""
Tool registry for agent tool calling.

Manages tool registration, definitions, and execution.
"""

from typing import Callable, Any


class ToolRegistry:
    """
    Registry for agent tools.

    Handles:
    - Tool registration with LLM-compatible definitions
    - Tool calling with argument validation
    - Tool discovery and listing
    """

    def __init__(self):
        """Initialize empty tool registry."""
        self._tools: dict[str, Callable] = {}
        self._definitions: dict[str, dict] = {}

    def register(
        self,
        name: str,
        func: Callable,
        definition: dict
    ) -> None:
        """
        Register a tool.

        Args:
            name: Tool name (must match function.name in definition)
            func: The callable function to execute
            definition: Tool definition dict for LLM (Ollama tool format)

        Raises:
            ValueError: If name doesn't match definition or tool already registered
        """
        # Validate definition format
        if "function" not in definition:
            raise ValueError(f"Tool definition must have 'function' key")

        func_name = definition["function"].get("name")
        if func_name != name:
            raise ValueError(
                f"Tool name '{name}' doesn't match definition name '{func_name}'"
            )

        if name in self._tools:
            raise ValueError(f"Tool '{name}' already registered")

        self._tools[name] = func
        self._definitions[name] = definition

    def call(self, name: str, **kwargs) -> Any:
        """
        Call a registered tool.

        Args:
            name: Tool name
            **kwargs: Arguments to pass to the tool

        Returns:
            Any: Tool result

        Raises:
            KeyError: If tool not found
        """
        if name not in self._tools:
            raise KeyError(
                f"Tool '{name}' not found. "
                f"Available tools: {list(self._tools.keys())}"
            )

        return self._tools[name](**kwargs)

    def get_definitions(self) -> list[dict]:
        """
        Get all tool definitions for LLM.

        Returns:
            list[dict]: List of tool definitions in Ollama format
        """
        return list(self._definitions.values())

    def get_tool_names(self) -> list[str]:
        """
        Get names of all registered tools.

        Returns:
            list[str]: Tool names
        """
        return list(self._tools.keys())

    def has_tool(self, name: str) -> bool:
        """
        Check if a tool is registered.

        Args:
            name: Tool name

        Returns:
            bool: True if tool exists
        """
        return name in self._tools

    def clear(self) -> None:
        """Clear all registered tools."""
        self._tools.clear()
        self._definitions.clear()
