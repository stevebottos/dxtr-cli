"""
Main Agent - Lightweight chat agent that coordinates sub-agents via tools.

Responsibilities:
- Handle chat functionality with minimal context
- Offload context-heavy operations to specialized agents
- All sub-agents return results to main when work is done
"""

import json
import requests
from pathlib import Path
from datetime import datetime
from typing import Generator

from dxtr.agents.base import BaseAgent
from dxtr.config_v2 import config
from dxtr.agents.github_summarize.agent import Agent as GithubSummarizeAgent
from dxtr.agents.profile_synthesize.agent import Agent as ProfileSynthesizeAgent
from dxtr.agents.papers_ranking.agent import Agent as PapersRankingAgent
from dxtr.agents.deep_research.agent import Agent as DeepResearchAgent
from dxtr.papers_etl import PapersETL


# Tool Definitions - these map to methods on MainAgent
TOOLS = [
    # --- Utilities ---
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read the content of a file. Use this to read the user's seed profile.md.",
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
            "name": "check_papers",
            "description": "Check if papers have been downloaded for a given date.",
            "parameters": {
                "type": "object",
                "properties": {
                    "date": {"type": "string", "description": "Date in YYYY-MM-DD format. Defaults to today if not provided."}
                },
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_papers",
            "description": "Download and process papers from HuggingFace for a given date. This may take several minutes.",
            "parameters": {
                "type": "object",
                "properties": {
                    "date": {"type": "string", "description": "Date in YYYY-MM-DD format. Defaults to today if not provided."},
                    "max_papers": {"type": "integer", "description": "Maximum number of papers to download. Downloads all if not specified."}
                },
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "list_papers",
            "description": "List all papers for a given date with their titles and abstracts. Prints full details to console.",
            "parameters": {
                "type": "object",
                "properties": {
                    "date": {"type": "string", "description": "Date in YYYY-MM-DD format. Defaults to today if not provided."}
                },
                "required": []
            }
        }
    },
    # --- Profile Setup ---
    {
        "type": "function",
        "function": {
            "name": "summarize_github",
            "description": "Analyze GitHub repos from the seed profile. Extracts GitHub URL from profile, clones pinned repos, and creates a summary. Saves result to .dxtr/github_summary.json.",
            "parameters": {
                "type": "object",
                "properties": {
                    "profile_path": {"type": "string", "description": "Path to the seed profile file containing GitHub URL"}
                },
                "required": ["profile_path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "synthesize_profile",
            "description": "Synthesize the final user profile from available artifacts in .dxtr/ directory. Reads github_summary.json and other artifacts, then creates a comprehensive profile. Saves result to .dxtr/profile.md.",
            "parameters": {
                "type": "object",
                "properties": {
                    "seed_profile_path": {"type": "string", "description": "Path to the original seed profile.md provided by user"}
                },
                "required": ["seed_profile_path"]
            }
        }
    },
    # --- Paper Analysis ---
    {
        "type": "function",
        "function": {
            "name": "rank_papers",
            "description": "Rank papers by relevance to the user's profile and interests. Prints ranked list to console.",
            "parameters": {
                "type": "object",
                "properties": {
                    "date": {"type": "string", "description": "Date in YYYY-MM-DD format. Defaults to today."},
                    "query": {"type": "string", "description": "Optional focus query, e.g. 'agentic systems' to prioritize certain topics."}
                },
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "deep_research",
            "description": "Deep dive into a specific paper using RAG. Answers questions about the paper's content, methods, and findings.",
            "parameters": {
                "type": "object",
                "properties": {
                    "paper_id": {"type": "string", "description": "Paper ID (e.g., '2601.12345')"},
                    "query": {"type": "string", "description": "Question to answer about the paper"}
                },
                "required": ["paper_id", "query"]
            }
        }
    }
]

class MainAgent(BaseAgent):
    """Lightweight chat agent that coordinates sub-agents via tools."""

    def __init__(self):
        """Initialize main agent."""
        super().__init__()
        self.system_prompt = self.load_system_prompt(
            Path(__file__).parent / "system.md"
        )
        # Track seed profile path for the session
        self.seed_profile_path: str | None = None

    # --- Tool Methods (called by CLI via getattr) ---

    def read_file(self, file_path: str) -> str:
        """Read content from a file."""
        try:
            path = Path(file_path).expanduser().resolve()
            if not path.exists():
                return f"Error: File not found: {file_path}"
            content = path.read_text()
            # Store seed profile path if this looks like a profile
            if "profile" in file_path.lower() or file_path.endswith(".md"):
                self.seed_profile_path = str(path)
            return content
        except Exception as e:
            return f"Error reading file: {e}"

    def check_papers(self, date: str = None) -> str:
        """Check if papers exist for a given date."""
        if date is None:
            date = datetime.now().strftime("%Y-%m-%d")

        papers_dir = config.paths.papers_dir / date
        if not papers_dir.exists():
            return f"No papers found for {date}. Use get_papers to download them."

        # Count papers by looking for metadata.json files
        paper_dirs = [d for d in papers_dir.iterdir() if d.is_dir() and (d / "metadata.json").exists()]
        count = len(paper_dirs)

        if count == 0:
            return f"No papers found for {date}. Use get_papers to download them."

        return f"Found {count} papers for {date}."

    def get_papers(self, date: str = None, max_papers: int = None) -> str:
        """Download and process papers from HuggingFace."""
        if date is None:
            date = datetime.now().strftime("%Y-%m-%d")

        try:
            etl = PapersETL()
            etl.run(date=date, max_papers=max_papers)

            # Count downloaded papers
            papers_dir = config.paths.papers_dir / date
            if papers_dir.exists():
                paper_dirs = [d for d in papers_dir.iterdir() if d.is_dir() and (d / "metadata.json").exists()]
                count = len(paper_dirs)
                return f"Downloaded and processed {count} papers for {date}. Saved to {papers_dir}"
            else:
                return f"No papers were downloaded for {date}."
        except Exception as e:
            return f"Error downloading papers: {e}"

    def list_papers(self, date: str = None) -> str:
        """List papers for a date - prints to console, returns summary."""
        if date is None:
            date = datetime.now().strftime("%Y-%m-%d")

        papers_dir = config.paths.papers_dir / date
        if not papers_dir.exists():
            return f"No papers found for {date}."

        # Load all paper metadata
        papers = []
        for paper_dir in sorted(papers_dir.iterdir()):
            if not paper_dir.is_dir():
                continue
            metadata_file = paper_dir / "metadata.json"
            if metadata_file.exists():
                try:
                    metadata = json.loads(metadata_file.read_text())
                    papers.append(metadata)
                except Exception:
                    continue

        if not papers:
            return f"No papers found for {date}."

        # Print full content to console (user sees this)
        print(f"\n{'='*70}")
        print(f"Papers for {date} ({len(papers)} total)")
        print(f"{'='*70}\n")

        for i, p in enumerate(papers, 1):
            title = p.get("title", "Unknown title")
            paper_id = p.get("id", "unknown")
            summary = p.get("summary", "No abstract available")
            upvotes = p.get("upvotes", 0)

            print(f"{i}. {title}")
            print(f"   ID: {paper_id} | Upvotes: {upvotes}")
            print(f"   {summary[:300]}..." if len(summary) > 300 else f"   {summary}")
            print()

        # Return paper list to LLM so it can reference them without hallucinating
        papers_summary = "\n".join([
            f"{i+1}. {p.get('title', 'Unknown')} (ID: {p.get('id', 'unknown')})"
            for i, p in enumerate(papers)
        ])
        return f"Listed {len(papers)} papers for {date}:\n{papers_summary}\n\nFull details with abstracts printed to console above."

    def summarize_github(self, profile_path: str) -> str:
        """Run GitHub summarize agent on the profile."""
        try:
            path = Path(profile_path).expanduser().resolve()
            if not path.exists():
                return f"Error: Profile not found: {profile_path}"

            agent = GithubSummarizeAgent()
            result = agent.run(profile_path=path)

            if result:
                return f"GitHub summary complete. Analyzed {len(result)} files. Saved to .dxtr/github_summary.json"
            else:
                return "No GitHub URL found in profile or no repos to analyze."
        except Exception as e:
            return f"Error running GitHub summarize: {e}"

    def synthesize_profile(self, seed_profile_path: str) -> str:
        """Run profile synthesis agent to create final profile from artifacts."""
        try:
            path = Path(seed_profile_path).expanduser().resolve()
            if not path.exists():
                return f"Error: Seed profile not found: {seed_profile_path}"

            agent = ProfileSynthesizeAgent()
            result = agent.run(seed_profile_path=path)

            if result:
                return f"Profile synthesized successfully. Saved to {config.paths.profile_file}"
            else:
                return "Error: Profile synthesis returned empty result."
        except Exception as e:
            return f"Error running profile synthesis: {e}"

    def rank_papers(self, date: str = None, query: str = None) -> str:
        """Rank papers by relevance - prints to console, returns summary."""
        if date is None:
            date = datetime.now().strftime("%Y-%m-%d")

        # Load user context from profile
        user_context = ""
        if config.paths.profile_file.exists():
            user_context = config.paths.profile_file.read_text()

        if not user_context:
            return "Error: No user profile found. Please set up your profile first."

        try:
            agent = PapersRankingAgent()
            result = agent.run(
                date=date,
                user_context=user_context,
                user_query=query or "rank by relevance to my profile"
            )

            if "error" in result:
                return f"Error: {result['error']}"

            # Print ranking to console
            print(f"\n{'='*70}")
            print(f"Paper Rankings for {date} ({result['paper_count']} papers)")
            print(f"{'='*70}\n")
            print(result["final_ranking"])
            print()

            # Return top papers data to LLM so it can reference them without hallucinating
            top_papers = result["individual_scores"][:5]
            top_papers_summary = "\n".join([
                f"{i+1}. [{p['final_score']}/5] {p['title']} (ID: {p['id']})"
                for i, p in enumerate(top_papers)
            ])
            return f"Ranked {result['paper_count']} papers. Top 5:\n{top_papers_summary}\n\nFull ranking printed to console above."

        except Exception as e:
            return f"Error ranking papers: {e}"

    def deep_research(self, paper_id: str, query: str) -> str:
        """Deep research into a paper - prints analysis to console, returns summary."""
        # Load user context from profile
        user_context = ""
        if config.paths.profile_file.exists():
            user_context = config.paths.profile_file.read_text()

        try:
            agent = DeepResearchAgent()
            result = agent.run(
                paper_id=paper_id,
                user_query=query,
                user_context=user_context
            )

            # Print analysis to console
            print(f"\n{'='*70}")
            print(f"Deep Research: {paper_id}")
            print(f"Query: {query}")
            print(f"{'='*70}\n")
            print(result)
            print()

            # Return summary to LLM
            return f"Analysis complete for paper {paper_id}. Full answer printed above ({len(result)} chars)."

        except Exception as e:
            return f"Error in deep research: {e}"

    # --- Chat Method ---

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
                # Ensure arguments is valid JSON (not empty string)
                sanitized_tool_calls = []
                for tc in tool_calls:
                    sanitized_tc = {
                        "id": tc.get("id", ""),
                        "type": tc.get("type", "function"),
                        "function": {
                            "name": tc.get("function", {}).get("name", ""),
                            "arguments": tc.get("function", {}).get("arguments") or "{}"
                        }
                    }
                    sanitized_tool_calls.append(sanitized_tc)
                api_messages.append({
                    "role": "assistant",
                    "content": content or "",
                    "tool_calls": sanitized_tool_calls
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
            "max_tokens": 2000,
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
                                # Note: delta may have tool_calls=None, so use 'or []'
                                tool_calls = delta.get("tool_calls") or []
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