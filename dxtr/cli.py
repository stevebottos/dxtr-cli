#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path
from datetime import datetime

# Import new configuration and agents
from dxtr.config_v2 import config
from dxtr.agents.main.agent import MainAgent
from dxtr.papers_etl import run_etl

def _load_user_context() -> str:
    """Load user profile content if it exists."""
    profile_path = config.paths.profile_file
    if profile_path.exists():
        return profile_path.read_text()
    return ""

def cmd_get_papers(args):
    """Get papers command - ETL pipeline for paper retrieval and processing"""
    run_etl(date=args.date, max_papers=args.max_papers)

def _process_turn(agent, chat_history):
    """
    Process a single turn of conversation:
    - Generate response
    - Handle native tool calls (recursively/iteratively)
    - Print output
    """
    while True:
        print("DXTR: ", end="", flush=True)

        response_generator = agent.chat(chat_history)
        full_response = ""
        tool_calls = []

        # Consume generator - now yields dicts with type and data
        for chunk in response_generator:
            if chunk["type"] == "content":
                print(chunk["data"], end="", flush=True)
                full_response += chunk["data"]
            elif chunk["type"] == "tool_calls":
                tool_calls = chunk["data"]

        print()  # Newline after response

        if tool_calls:
            # Add assistant message with tool calls to history
            chat_history.append({
                "role": "assistant",
                "content": full_response or None,
                "tool_calls": tool_calls
            })

            for tc in tool_calls:
                tool_call_id = tc.get("id", "")
                func = tc.get("function", {})
                tool_name = func.get("name", "")
                args_str = func.get("arguments", "{}")

                print(f"\n[Calling Tool: {tool_name}]")

                # Parse arguments
                try:
                    tool_params = json.loads(args_str) if args_str else {}
                except json.JSONDecodeError:
                    tool_params = {}

                result = None
                if hasattr(agent, tool_name):
                    try:
                        tool_func = getattr(agent, tool_name)
                        result = tool_func(**tool_params)
                    except Exception as e:
                        result = f"Error executing tool '{tool_name}': {e}"
                else:
                    result = f"Error: Tool '{tool_name}' not found."

                print(f"[Result: {str(result)[:100]}...]")

                # Context Reloading Hook
                if tool_name == "synthesize_profile" and "Profile synthesized" in str(result):
                    new_context = _load_user_context()
                    if new_context:
                        chat_history[:] = [msg for msg in chat_history if not (msg.get("role") == "system" and "User Profile" in msg.get("content", ""))]
                        chat_history.insert(0, {"role": "system", "content": f"User Profile:\n{new_context}"})
                        print("[System: User Profile Loaded]")

                # Add tool response to history
                chat_history.append({
                    "role": "tool",
                    "content": str(result),
                    "tool_call_id": tool_call_id
                })

            # Loop continues to let agent respond to tool result
            continue

        else:
            # Normal response
            chat_history.append({"role": "assistant", "content": full_response})
            break

def cmd_chat(args):
    """
    Start the main DXTR chat interface.
    """
    print("Initializing DXTR Agent (SGLang)...")
    try:
        agent = MainAgent()
    except Exception as e:
        print(f"Error initializing agent: {e}")
        print("Ensure SGLang server is running at localhost:30000")
        return

    # Initial context
    user_context = _load_user_context()
    
    print("\nDXTR - Research Assistant")
    print("Type 'exit' or 'quit' to end the session.")
    print("Type '/help' for available commands.\n")

    # Chat history
    chat_history = []
    
    if user_context:
        # We silently load the profile. The agent will confirm it in the greeting.
        chat_history.append({"role": "system", "content": f"User Profile:\n{user_context}"})

    # Auto-Greeting / Pre-initialization
    # We inject a hidden user message to prompt the agent to start.
    print("Connecting to agent...\n")
    chat_history.append({"role": "user", "content": "Hello. Please check if my profile is loaded and briefly introduce yourself."})
    _process_turn(agent, chat_history)

    while True:
        try:
            user_input = input("\nYou: ").strip()

            if not user_input:
                continue

            if user_input.lower() in ["exit", "quit", "q"]:
                print("\nGoodbye!\n")
                break

            if user_input == "/help":
                print("\nAvailable commands:")
                print("  /help     - Show this help message")
                print("  exit/quit - End the session\n")
                continue

            # Add user message
            chat_history.append({"role": "user", "content": user_input})

            # Process turn
            _process_turn(agent, chat_history)

        except KeyboardInterrupt:
            print("\n\nGoodbye!\n")
            break
        except Exception as e:
            print(f"\nError: {e}\n")

def main():
    parser = argparse.ArgumentParser(
        description="DXTR Agent - AI-powered research assistant"
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # dxtr chat command
    parser_chat = subparsers.add_parser("chat", help="Start the DXTR chat interface")
    parser_chat.set_defaults(func=cmd_chat)

    # dxtr get-papers command
    parser_get_papers = subparsers.add_parser(
        "get-papers",
        help="Download and process daily papers (starts Docling service automatically)",
    )
    parser_get_papers.add_argument(
        "--date",
        type=str,
        default=None,
        help="Date in YYYY-MM-DD format (default: today)",
    )
    parser_get_papers.add_argument(
        "--max-papers",
        type=int,
        default=None,
        help="Maximum number of papers to process (default: all)",
    )
    parser_get_papers.set_defaults(func=cmd_get_papers)

    args = parser.parse_args()

    # Default to chat if no command specified
    if args.command is None:
        cmd_chat(args)
    else:
        args.func(args)


if __name__ == "__main__":
    main()
