import json
import pytest
import requests
from pathlib import Path
from dxtr.config_v2 import config

# --- Test Configuration ---
NUM_RUNS = 10
API_URL = f"{config.sglang.base_url}/chat/completions"

# --- Tool Definitions (OpenAI format) ---
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read content from a local file.",
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
            "description": "Process and summarize GitHub activity from a profile file.",
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
            "description": "Create a research profile from seed profile.",
            "parameters": {
                "type": "object",
                "properties": {
                    "profile_path": {"type": "string", "description": "Path to seed profile file"}
                },
                "required": ["profile_path"]
            }
        }
    }
]

def get_test_system_prompt():
    """Self-contained system prompt for tool calling tests."""
    return """You are DXTR, a research assistant.
You have access to tools and MUST use them to fulfill requests when necessary.
When a user provides a file path, use the appropriate tool to process it.
"""

def call_llm(messages):
    """Sync call to LLM for testing with native tool calling."""
    payload = {
        "model": "default",
        "messages": messages,
        "tools": TOOLS,
        "temperature": 0.1,
        "max_tokens": 1000,
        "stream": False
    }
    try:
        response = requests.post(API_URL, json=payload, timeout=30)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]
    except requests.exceptions.RequestException as e:
        pytest.fail(f"LLM call failed: {e}")

# --- Test Cases ---

TEST_SCENARIOS = [
    {
        "name": "synthesize_profile",
        "user_input": "My profile is at /home/steve/repos/dxtr-cli/profile.md. Please synthesize my profile.",
        "expected_tool": "synthesize_profile",
        "expected_params": ["profile_path"]
    },
    {
        "name": "summarize_github",
        "setup_history": [
            {"role": "user", "content": "My profile is at /home/steve/repos/dxtr-cli/profile.md"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [{
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "read_file",
                        "arguments": '{"file_path": "/home/steve/repos/dxtr-cli/profile.md"}'
                    }
                }]
            },
            {
                "role": "tool",
                "content": "Profile content with GitHub URL: https://github.com/steve",
                "tool_call_id": "call_1"
            },
        ],
        "user_input": "Great, now summarize my github.",
        "expected_tool": "summarize_github",
        "expected_params": ["profile_path"]
    }
]


def extract_tool_calls(message: dict) -> list[dict]:
    """Extract tool calls from a response message."""
    tool_calls = message.get("tool_calls", [])
    results = []
    for tc in tool_calls:
        func = tc.get("function", {})
        name = func.get("name", "")
        args_str = func.get("arguments", "{}")
        try:
            params = json.loads(args_str) if args_str else {}
        except json.JSONDecodeError:
            params = {}
        results.append({"tool": name, "parameters": params})
    return results


@pytest.mark.parametrize("run_id", range(NUM_RUNS))
@pytest.mark.parametrize("scenario", TEST_SCENARIOS)
def test_tool_calling_consistency(run_id, scenario):
    """Test that Qwen consistently calls the correct tool using native tool calling."""
    system_prompt = get_test_system_prompt()

    messages = [{"role": "system", "content": system_prompt}]

    if "setup_history" in scenario:
        messages.extend(scenario["setup_history"])

    messages.append({"role": "user", "content": scenario["user_input"]})

    # Execution
    response_message = call_llm(messages)

    # Verification
    tool_calls = extract_tool_calls(response_message)

    assert len(tool_calls) > 0, f"Run {run_id}: No tool calls in response: {response_message}"

    # Validate the first tool call
    tool_call = tool_calls[0]
    assert tool_call["tool"] == scenario["expected_tool"], \
        f"Run {run_id}: Expected {scenario['expected_tool']}, got {tool_call['tool']}. Response: {response_message}"

    params = tool_call.get("parameters", {})
    for p in scenario["expected_params"]:
        assert p in params, f"Run {run_id}: Missing parameter '{p}' in tool call. Params: {params}"
        assert params[p], f"Run {run_id}: Parameter '{p}' is empty"
