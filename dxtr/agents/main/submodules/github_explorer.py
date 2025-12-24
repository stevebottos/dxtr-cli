"""
GitHub Explorer Submodule

Explores GitHub profiles and repositories to generate comprehensive summaries
of the user's work, projects, and technical expertise.
"""

import json
import hashlib
import re
import urllib.request
from pathlib import Path
from ollama import chat
from ..agent import MODEL, _load_system_prompt
from ..tools.web_fetch import fetch_url, TOOL_DEFINITION
from ..tools import git_tools
from . import repo_analyzer


def _get_cache_path(url: str) -> Path:
    """Generate a cache file path for a URL."""
    # Create a safe filename from the URL
    url_hash = hashlib.md5(url.encode()).hexdigest()[:8]
    # Extract domain/path for readability
    clean_url = url.replace("https://", "").replace("http://", "")
    clean_url = clean_url.replace("/", "_").replace(".", "_")[:50]
    filename = f"github_{clean_url}_{url_hash}.md"
    return Path(".dxtr") / filename


def _is_profile_url(url: str) -> bool:
    """
    Check if a GitHub URL is a profile (not a repository).

    Args:
        url: GitHub URL

    Returns:
        bool: True if it's a profile URL (e.g., github.com/username)
    """
    # Profile URLs have format: github.com/username (no additional path)
    # Repo URLs have format: github.com/username/repo
    pattern = r'github\.com/([^/]+)/?$'
    return bool(re.search(pattern, url))


def _fetch_raw_html(url: str) -> str | None:
    """
    Fetch raw HTML from a URL (without text extraction).

    Args:
        url: The URL to fetch

    Returns:
        Raw HTML string or None if failed
    """
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (DXTR Profile Agent)'}
        req = urllib.request.Request(url, headers=headers)

        with urllib.request.urlopen(req, timeout=10) as response:
            content_bytes = response.read()
            content_type = response.headers.get('Content-Type', '')

            # Try to detect encoding
            encoding = 'utf-8'
            if 'charset=' in content_type:
                encoding = content_type.split('charset=')[-1].split(';')[0].strip()

            try:
                html = content_bytes.decode(encoding)
            except UnicodeDecodeError:
                html = content_bytes.decode('utf-8', errors='ignore')

            return html

    except Exception as e:
        print(f"  [Error fetching raw HTML: {e}]")
        return None


def _extract_pinned_repos(html_content: str) -> list[str]:
    """
    Extract pinned repository URLs from a GitHub profile page HTML.

    Args:
        html_content: Raw HTML from GitHub profile page

    Returns:
        List of full repository URLs
    """
    # GitHub pinned repos have data-hydro-click with "target":"PINNED_REPO"
    # Pattern: look for links with PINNED_REPO in data-hydro-click and extract href
    pattern = r'data-hydro-click="[^"]*PINNED_REPO[^"]*"[^>]*href="(/[^/"]+/[^/"]+)"'

    # Also try the reverse order (href before data-hydro-click)
    pattern_reverse = r'href="(/[^/"]+/[^/"]+)"[^>]*data-hydro-click="[^"]*PINNED_REPO[^"]*"'

    repos = []
    seen = set()

    # Try both patterns
    for patt in [pattern, pattern_reverse]:
        matches = re.findall(patt, html_content)
        for match in matches:
            # match is like: /username/repo-name
            if match not in seen and match.count('/') == 2:  # Ensure it's /owner/repo format
                full_url = f"https://github.com{match}"
                repos.append(full_url)
                seen.add(match)

    return repos


def _extract_repo_summary(markdown: str, owner: str, repo: str) -> str:
    """
    Extract a concise summary from full repo analysis.

    Args:
        markdown: Full analysis markdown
        owner: Repo owner
        repo: Repo name

    Returns:
        Concise summary (first 2000 chars of each file's key info)
    """
    lines = markdown.split('\n')
    summary_parts = [f"## Repository: {owner}/{repo}\n"]

    current_file = None
    imports = []
    functions = []
    classes = []

    for line in lines:
        # Track current file
        if line.startswith('## File:'):
            if current_file:  # Save previous file's summary
                summary_parts.append(f"\n### {current_file}")
                if imports:
                    summary_parts.append(f"**Imports**: {', '.join(imports[:5])}")
                if functions:
                    summary_parts.append(f"**Functions ({len(functions)})**: {', '.join(functions[:8])}")
                if classes:
                    summary_parts.append(f"**Classes ({len(classes)})**: {', '.join(classes[:5])}\n")
                imports, functions, classes = [], [], []

            current_file = line.replace('## File: `', '').replace('`', '')

        # Extract key info
        elif '**Imports:**' in line:
            # Get next line which has the imports
            continue
        elif line.startswith('`') and 'from' not in line and 'import' not in line.lower():
            # Import names
            imports.extend([imp.strip('` ,') for imp in line.split(',')[:3]])
        elif line.startswith('#### `') and '(line' in line:
            # Function name
            func_name = line.split('`')[1]
            functions.append(func_name)
        elif line.startswith('#### Class:'):
            # Class name
            class_name = line.split('`')[1]
            classes.append(class_name)

    # Don't forget the last file
    if current_file:
        summary_parts.append(f"\n### {current_file}")
        if imports:
            summary_parts.append(f"**Imports**: {', '.join(imports[:5])}")
        if functions:
            summary_parts.append(f"**Functions ({len(functions)})**: {', '.join(functions[:8])}")
        if classes:
            summary_parts.append(f"**Classes ({len(classes)})**: {', '.join(classes[:5])}\n")

    return '\n'.join(summary_parts)


def _synthesize_github_portfolio(profile_content: str, analyses: list[dict]) -> str:
    """
    Synthesize a master summary from profile and repo analyses.

    Args:
        profile_content: User's profile.md content
        analyses: List of dicts with 'owner', 'repo', 'markdown' keys

    Returns:
        Synthesized markdown summary
    """
    print(f"\n  [Synthesizing GitHub portfolio with {len(analyses)} repo(ies)...]")

    # Load synthesis system prompt
    synthesis_prompt = _load_system_prompt("github_synthesis")

    # Build the context for synthesis (use summaries, not full analyses)
    context_parts = ["# User Profile\n\n", profile_content, "\n\n---\n\n# Repository Code Summaries\n\n"]

    for analysis_data in analyses:
        # Extract concise summary instead of full markdown
        summary = _extract_repo_summary(
            analysis_data['markdown'],
            analysis_data['owner'],
            analysis_data['repo']
        )
        context_parts.append(summary)
        context_parts.append("\n---\n\n")

    full_context = "".join(context_parts)

    # Check token estimate
    estimated_tokens = len(full_context) // 4
    print(f"  [Context size: ~{estimated_tokens} tokens]")

    if estimated_tokens > 15000:  # Leave room for response
        print(f"  [Warning: Context may be large, truncating to fit]")
        max_chars = 15000 * 4
        full_context = full_context[:max_chars] + "\n\n[Note: Some repository details truncated]"

    # Call LLM for synthesis
    try:
        response = chat(
            model=MODEL,  # Use mistral-nemo for synthesis
            messages=[
                {"role": "system", "content": synthesis_prompt},
                {"role": "user", "content": full_context}
            ],
            options={
                "temperature": 0.3,
                "num_ctx": 16384,
            }
        )

        if hasattr(response.message, 'content'):
            synthesis = response.message.content.strip()
            print(f"  [✓] Synthesis complete ({len(synthesis)} characters)")
            return synthesis
        else:
            return "Synthesis failed: No response from model"

    except Exception as e:
        return f"Synthesis error: {str(e)}"


def _clone_and_analyze_pinned_repos(url: str, profile_content: str, user_profile_md: str = "") -> str:
    """
    Extract, clone, and analyze pinned repositories from a GitHub profile.

    Args:
        url: GitHub profile URL
        profile_content: Fetched HTML content of the profile page
        user_profile_md: User's profile.md content for synthesis context

    Returns:
        Markdown summary of cloned repositories and their analyses
    """
    if not _is_profile_url(url):
        return ""

    print(f"  [Detecting pinned repositories...]")

    # Extract pinned repo URLs
    pinned_repos = _extract_pinned_repos(profile_content)

    # Filter out dxtr-cli (the repo we're currently working in)
    pinned_repos = [repo for repo in pinned_repos if not repo.endswith('/dxtr-cli')]

    if not pinned_repos:
        print(f"  [No pinned repositories found]")
        return ""

    print(f"  [Found {len(pinned_repos)} pinned repository(ies)]")

    # Clone each repo
    results = []
    for repo_url in pinned_repos:
        result = git_tools.clone_repo(repo_url)
        results.append(result)

        if result["success"]:
            status = "✓" if "cached" in result["message"].lower() else "✓ cloned"
            print(f"  [{status}] {result['owner']}/{result['repo']}")
        else:
            print(f"  [✗] Failed to clone {repo_url}: {result['message']}")

    # Analyze successfully cloned repos
    successful = [r for r in results if r["success"]]
    analyses = []

    if successful:
        print(f"\n  [Analyzing {len(successful)} repository(ies)...]")
        for result in successful:
            repo_path = result["path"]
            cache_path = repo_analyzer._get_analysis_cache_path(repo_path)

            # Check if analysis is cached
            if cache_path.exists():
                print(f"  [✓ cached] Analysis for {result['owner']}/{result['repo']}")
                analysis_md = cache_path.read_text()
            else:
                # Analyze the repository
                analysis = repo_analyzer.analyze_repository(repo_path, summarize=True)

                # Format as markdown
                analysis_md = repo_analyzer.format_analysis_as_markdown(analysis)

                # Cache the analysis
                cache_path.write_text(analysis_md)
                print(f"  [✓] Analyzed {result['owner']}/{result['repo']}")

            analyses.append({
                'owner': result['owner'],
                'repo': result['repo'],
                'markdown': analysis_md
            })

    # Build summary
    summary_lines = ["\n## Cloned and Analyzed Pinned Repositories\n"]

    if successful:
        summary_lines.append(f"Successfully cloned and analyzed {len(successful)} pinned repository(ies):\n")
        for result in successful:
            cached_marker = " (cached)" if "cached" in result["message"].lower() else ""
            summary_lines.append(f"- **{result['owner']}/{result['repo']}**{cached_marker}")
            summary_lines.append(f"  - Path: `.dxtr/repos/{result['owner']}/{result['repo']}`")
            summary_lines.append(f"  - URL: {result['url']}\n")

    failed = [r for r in results if not r["success"]]
    if failed:
        summary_lines.append(f"\nFailed to clone {len(failed)} repository(ies):\n")
        for result in failed:
            summary_lines.append(f"- {result['url']}: {result['message']}\n")

    # Generate master synthesis
    if analyses:
        # Check if synthesis is cached
        synthesis_cache = Path(".dxtr") / "github_portfolio_synthesis.md"

        if synthesis_cache.exists():
            print(f"  [✓ cached] GitHub portfolio synthesis")
            synthesis_md = synthesis_cache.read_text()
        else:
            # Generate synthesis using mistral-nemo
            synthesis_md = _synthesize_github_portfolio(user_profile_md, analyses)

            # Cache the synthesis
            synthesis_cache.write_text(synthesis_md)
            print(f"  [✓] Cached synthesis to {synthesis_cache.name}")

        summary_lines.append("\n---\n")
        summary_lines.append("## GitHub Portfolio Synthesis\n")
        summary_lines.append(synthesis_md)
        summary_lines.append("\n")

    # Add reference to detailed analyses (full details are cached separately)
    if analyses:
        summary_lines.append("\n### Detailed Repository Analyses (Cached)\n")
        for analysis_data in analyses:
            owner = analysis_data['owner']
            repo = analysis_data['repo']
            cache_path = repo_analyzer._get_analysis_cache_path(f".dxtr/repos/{owner}/{repo}")

            summary_lines.append(f"- **{owner}/{repo}**: `.dxtr/{cache_path.name}`")

    return "\n".join(summary_lines)


def explore(url: str, user_profile_md: str = "") -> str:
    """
    Explore a GitHub URL and generate a summary.
    Uses cached result if available.

    Args:
        url: GitHub profile or repository URL
        user_profile_md: User's profile.md content for synthesis context

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

    # First, if this is a profile URL, fetch it and clone pinned repos
    clone_summary = ""
    profile_content = ""

    if _is_profile_url(url):
        print(f"  [Fetching GitHub profile page...]")
        # Fetch raw HTML for pinned repo extraction
        profile_content = _fetch_raw_html(url)

        if profile_content:
            clone_summary = _clone_and_analyze_pinned_repos(url, profile_content, user_profile_md)
        else:
            print(f"  [Failed to fetch profile HTML]")

    # Prepare system prompt
    system_prompt = _load_system_prompt("github_explorer")

    # Build conversation with tool calling support
    # Include clone summary in context if available
    user_content = f"Analyze this GitHub URL: {url}"
    if clone_summary:
        user_content += f"\n\n{clone_summary}"

    conversation = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content}
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

    # Prepend clone summary to the final summary
    if clone_summary:
        summary = f"{clone_summary}\n\n---\n\n{summary}"

    # Cache the result
    if summary:
        cache_path.write_text(summary)
        print(f"[Cached to: {cache_path.name}]")

    return summary
