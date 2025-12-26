"""
Tests for base classes (Agent, ToolRegistry, PromptManager).

These are unit tests that don't require actual LLM calls.
"""

import pytest
from pathlib import Path
from dxtr.base import Agent, ToolRegistry, PromptManager


class TestPromptManager:
    """Tests for PromptManager."""

    def test_load_prompt(self, tmp_prompts_dir):
        """Test loading a prompt from filesystem."""
        pm = PromptManager(tmp_prompts_dir)
        prompt = pm.load("test_prompt")
        assert prompt == "You are a helpful assistant."

    def test_load_nonexistent_prompt_raises_error(self, tmp_prompts_dir):
        """Test that loading nonexistent prompt raises FileNotFoundError."""
        pm = PromptManager(tmp_prompts_dir)
        with pytest.raises(FileNotFoundError) as exc_info:
            pm.load("nonexistent")
        assert "nonexistent" in str(exc_info.value)

    def test_list_prompts(self, tmp_prompts_dir):
        """Test listing available prompts."""
        pm = PromptManager(tmp_prompts_dir)
        prompts = pm.list_prompts()
        assert "test_prompt" in prompts
        assert "chat" in prompts

    def test_exists(self, tmp_prompts_dir):
        """Test checking if prompt exists."""
        pm = PromptManager(tmp_prompts_dir)
        assert pm.exists("test_prompt")
        assert not pm.exists("nonexistent")


class TestToolRegistry:
    """Tests for ToolRegistry."""

    def test_register_tool(self):
        """Test registering a tool."""
        registry = ToolRegistry()

        def test_func(x: int) -> int:
            return x * 2

        definition = {
            "type": "function",
            "function": {
                "name": "test_func",
                "description": "Test function",
                "parameters": {
                    "type": "object",
                    "properties": {"x": {"type": "integer"}},
                    "required": ["x"]
                }
            }
        }

        registry.register("test_func", test_func, definition)
        assert registry.has_tool("test_func")

    def test_register_tool_name_mismatch_raises_error(self):
        """Test that name mismatch raises ValueError."""
        registry = ToolRegistry()

        def test_func():
            pass

        definition = {
            "type": "function",
            "function": {"name": "different_name"}
        }

        with pytest.raises(ValueError) as exc_info:
            registry.register("test_func", test_func, definition)
        assert "doesn't match" in str(exc_info.value)

    def test_call_tool(self):
        """Test calling a registered tool."""
        registry = ToolRegistry()

        def add(a: int, b: int) -> int:
            return a + b

        definition = {
            "type": "function",
            "function": {"name": "add"}
        }

        registry.register("add", add, definition)
        result = registry.call("add", a=2, b=3)
        assert result == 5

    def test_call_nonexistent_tool_raises_error(self):
        """Test that calling nonexistent tool raises KeyError."""
        registry = ToolRegistry()
        with pytest.raises(KeyError):
            registry.call("nonexistent")

    def test_get_definitions(self):
        """Test getting all tool definitions."""
        registry = ToolRegistry()

        def func1():
            pass

        def func2():
            pass

        def1 = {"type": "function", "function": {"name": "func1"}}
        def2 = {"type": "function", "function": {"name": "func2"}}

        registry.register("func1", func1, def1)
        registry.register("func2", func2, def2)

        definitions = registry.get_definitions()
        assert len(definitions) == 2
        assert def1 in definitions
        assert def2 in definitions


class TestAgent:
    """Tests for Agent base class."""

    def test_agent_initialization(self, tmp_prompts_dir):
        """Test creating an agent."""
        agent = Agent(
            name="test_agent",
            model="test-model",
            prompts_dir=tmp_prompts_dir
        )

        assert agent.name == "test_agent"
        assert agent.model == "test-model"
        assert isinstance(agent.prompts, PromptManager)
        assert isinstance(agent.tools, ToolRegistry)

    def test_agent_loads_prompt(self, tmp_prompts_dir):
        """Test that agent can load prompts."""
        agent = Agent(
            name="test_agent",
            model="test-model",
            prompts_dir=tmp_prompts_dir
        )

        prompt = agent.prompts.load("test_prompt")
        assert "helpful assistant" in prompt

    def test_agent_repr(self, tmp_prompts_dir):
        """Test agent string representation."""
        agent = Agent(
            name="test_agent",
            model="test-model",
            prompts_dir=tmp_prompts_dir
        )

        repr_str = repr(agent)
        assert "test_agent" in repr_str
        assert "test-model" in repr_str
