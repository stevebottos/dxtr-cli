"""
GitHub Explorer Submodule

Explores GitHub profiles and repositories to generate comprehensive summaries
of the user's work, projects, and technical expertise.
"""

import json
import hashlib
from pathlib import Path
from ollama import chat
from ..agent import MODEL, _load_system_prompt
from ..tools.web_fetch import fetch_url, TOOL_DEFINITION


def _get_cache_path(url: str) -> Path:
    """Generate a cache file path for a URL."""
    # Create a safe filename from the URL
    url_hash = hashlib.md5(url.encode()).hexdigest()[:8]
    # Extract domain/path for readability
    clean_url = url.replace("https://", "").replace("http://", "")
    clean_url = clean_url.replace("/", "_").replace(".", "_")[:50]
    filename = f"github_{clean_url}_{url_hash}.md"
    return Path(".dxtr") / filename


def explore(url: str) -> str:
    """
    Explore a GitHub URL and generate a summary.
    Uses cached result if available.

    Args:
        url: GitHub profile or repository URL

    Returns:
        str: Markdown summary of the GitHub content
    """
    cache_path = _get_cache_path(url)

    # Check cache
    if cache_path.exists():
        print(f"\n[GitHub cache hit: {url}]")
        print(f"[Reading from: {cache_path.name}]")
        return cache_path.read_text()

    print(f"\n[Exploring GitHub: {url}]")

    # Prepare system prompt
    system_prompt = _load_system_prompt("github_explorer")

    # Build conversation with tool calling support
    conversation = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Analyze this GitHub URL: {url}"}
    ]

    summary = ""

    # Tool calling loop
    while True:
        response = chat(
            model=MODEL,
            messages=conversation,
            tools=[TOOL_DEFINITION],
            options={
                "temperature": 0.3,
                "num_ctx": 4096 * 4,
            }
        )

        # Check for tool calls
        if hasattr(response.message, 'tool_calls') and response.message.tool_calls:
            for tool_call in response.message.tool_calls:
                tool_name = tool_call.function.name
                tool_args = tool_call.function.arguments

                print(f"  [Fetching: {tool_args.get('url', 'unknown')}]")

                # Execute the tool
                if tool_name == "fetch_url":
                    result = fetch_url(**tool_args)
                else:
                    result = {"error": f"Unknown tool: {tool_name}"}

                # Add to conversation
                conversation.append({
                    "role": "assistant",
                    "tool_calls": [tool_call]
                })
                conversation.append({
                    "role": "tool",
                    "content": json.dumps(result)
                })
            continue
        else:
            # Final response
            if hasattr(response.message, 'content'):
                summary = response.message.content
            break

    print(f"[GitHub exploration complete]")

    # Cache the result
    if summary:
        cache_path.write_text(summary)
        print(f"[Cached to: {cache_path.name}]")

    return summary
