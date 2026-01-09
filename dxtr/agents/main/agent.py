"""
Main Agent - Handles chat and coordinates submodules using SGLang.
"""

import json
import requests
from pathlib import Path
import sglang as sgl
from typing import Generator, Any

from dxtr.agents.base import BaseAgent
from dxtr.config_v2 import config
from dxtr.agents.github_summarize import agent as github_agent
from dxtr.agents.profile_synthesize import agent as profile_agent


# Tool Definitions
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read the content of a file. Use this to read the initial profile.md.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {"type": "string", "description": "Path to the file to read"}
                },
                "required": ["file_path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "summarize_github",
            "description": "Run the GitHub summary agent. Analyzes repos from a profile file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "profile_path": {"type": "string", "description": "Path to the profile file containing GitHub URL"}
                },
                "required": ["profile_path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "synthesize_profile",
            "description": "Run the profile synthesis agent. Creates final profile from summary.",
            "parameters": {
                "type": "object",
                "properties": {
                    "profile_path": {"type": "string", "description": "Path to seed profile file"},
                    "summary_path": {"type": "string", "description": "Path to github summary JSON"}
                },
                "required": ["profile_path", "summary_path"]
            }
        }
    }
]

class MainAgent(BaseAgent):
    """Main agent for chat and coordination using SGLang."""

    def __init__(self):
        """Initialize main agent."""
        super().__init__()
        self.system_prompt = self.load_system_prompt(
            Path(__file__).parent / "system.md"
        )

    @staticmethod
    @sgl.function
    def chat_func(s, messages, system_prompt, tools_desc, max_tokens, temp):
        """SGLang function for chat."""
        # Inject system prompt and tools description
        full_system = system_prompt + "\n\nAvailable Tools:\n" + tools_desc
        full_system += "\n\nTo use a tool, respond ONLY with a JSON object:\n"
        full_system += '{"tool": "tool_name", "parameters": {}}\n'
        full_system += "Otherwise, respond normally."
        
        s += sgl.system(full_system)
        
        # Replay history
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            if role == "user":
                s += sgl.user(content)
            elif role == "assistant":
                s += sgl.assistant(content)
            elif role == "tool":
                # Represent tool outputs as user messages or distinct block
                s += sgl.user(f"Tool Output: {content}")
        
        # Generate response
        s += sgl.assistant(sgl.gen("response", max_tokens=max_tokens, temperature=temp))

    def chat(self, messages: list[dict], stream: bool = True) -> Generator[dict, None, None]:
        """
        Chat with the agent using native tool calling.

        Args:
            messages: List of message dicts
            stream: Whether to stream response (always True for now)

        Yields:
            Dict with either {"type": "content", "data": str} or {"type": "tool_calls", "data": list}
        """
        # Refresh state
        self.state.check_state()

        # Inject global state into system prompt
        state_str = f"Global State: {self.state}"
        full_system_prompt = f"{self.system_prompt}\n\n{state_str}"

        # Construct messages payload for OpenAI-compatible API
        api_messages = [{"role": "system", "content": full_system_prompt}]

        for msg in messages:
            role = msg["role"]
            content = msg.get("content")
            tool_calls = msg.get("tool_calls")
            tool_call_id = msg.get("tool_call_id")

            if role == "tool":
                # Native tool response format
                api_messages.append({
                    "role": "tool",
                    "content": content,
                    "tool_call_id": tool_call_id
                })
            elif role == "assistant" and tool_calls:
                # Assistant message with tool calls
                api_messages.append({
                    "role": "assistant",
                    "content": content or "",
                    "tool_calls": tool_calls
                })
            elif role == "system" and "User Profile" in (content or ""):
                api_messages.append({"role": "system", "content": content})
            else:
                api_messages.append({"role": role, "content": content})

        # Use requests to stream from SGLang server
        url = f"{config.sglang.base_url}/chat/completions"
        payload = {
            "model": "default",
            "messages": api_messages,
            "tools": TOOLS,
            "temperature": 0.3,
            "max_tokens": 1000,
            "stream": True
        }

        try:
            accumulated_tool_calls = {}

            with requests.post(url, json=payload, stream=True) as response:
                response.raise_for_status()
                for line in response.iter_lines(decode_unicode=True):
                    if line:
                        if line.startswith("data: "):
                            data_str = line[6:]
                            if data_str.strip() == "[DONE]":
                                break
                            try:
                                data = json.loads(data_str)
                                delta = data.get("choices", [{}])[0].get("delta", {})

                                # Handle content
                                content = delta.get("content", "")
                                if content:
                                    yield {"type": "content", "data": content}

                                # Handle tool calls (streamed incrementally)
                                tool_calls = delta.get("tool_calls", [])
                                for tc in tool_calls:
                                    idx = tc.get("index", 0)
                                    if idx not in accumulated_tool_calls:
                                        accumulated_tool_calls[idx] = {
                                            "id": tc.get("id", ""),
                                            "type": "function",
                                            "function": {"name": "", "arguments": ""}
                                        }

                                    if tc.get("id"):
                                        accumulated_tool_calls[idx]["id"] = tc["id"]

                                    func = tc.get("function", {})
                                    if func.get("name"):
                                        accumulated_tool_calls[idx]["function"]["name"] += func["name"]
                                    if func.get("arguments"):
                                        accumulated_tool_calls[idx]["function"]["arguments"] += func["arguments"]

                            except json.JSONDecodeError:
                                continue

                # Yield accumulated tool calls at the end if any
                if accumulated_tool_calls:
                    yield {"type": "tool_calls", "data": list(accumulated_tool_calls.values())}

        except Exception as e:
            yield {"type": "content", "data": f"Error generating response: {e}"}