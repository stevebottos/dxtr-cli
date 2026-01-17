"""Tools for the main agent."""

import json
from pathlib import Path

from pydantic_ai import RunContext

from agents.subagents.profile_synthesis.agent import profile_synthesis_agent
from agents.subagents.profile_synthesis import github_utils


# Base directory for DXTR artifacts
DXTR_DIR = Path(".dxtr")


async def read_file(ctx: RunContext[None], file_path: str) -> str:
    """Read content from a file."""
    try:
        path = Path(file_path).expanduser().resolve()
        if not path.exists():
            return f"Error: File not found: {file_path}"
        return path.read_text()
    except Exception as e:
        return f"Error reading file: {e}"


async def analyze_github(ctx: RunContext[None], github_url: str) -> str:
    """
    Analyze GitHub repos from a profile URL.

    Clones pinned repos and returns a summary of technologies found.
    Saves result to .dxtr/github_summary.json.
    """
    if not github_utils.is_profile_url(github_url):
        return f"Error: Not a valid GitHub profile URL: {github_url}"

    html = github_utils.fetch_profile_html(github_url)
    if not html:
        return "Error: Could not fetch GitHub profile page"

    pinned_repos = github_utils.extract_pinned_repos(html)
    if not pinned_repos:
        return "No pinned repositories found on profile"

    # Exclude dxtr-cli if present
    pinned_repos = [repo for repo in pinned_repos if not repo.endswith("/dxtr-cli")]

    DXTR_DIR.mkdir(parents=True, exist_ok=True)

    # Clone repos
    cloned = []
    for repo_url in pinned_repos:
        result = github_utils.clone_repo(repo_url, DXTR_DIR)
        if result["success"]:
            cloned.append(result)

    if not cloned:
        return "No repositories could be cloned"

    # Find Python files and collect basic stats
    summary = {
        "repos_analyzed": len(cloned),
        "repos": [],
    }

    for repo in cloned:
        repo_path = Path(repo["path"])
        python_files = github_utils.find_python_files(repo_path)

        repo_info = {
            "name": f"{repo.get('owner', 'unknown')}/{repo.get('repo', 'unknown')}",
            "url": repo["url"],
            "python_files": len(python_files),
            "file_list": [str(f.relative_to(repo_path)) for f in python_files[:20]],
        }
        summary["repos"].append(repo_info)

    # Save summary
    summary_file = DXTR_DIR / "github_summary.json"
    summary_file.write_text(json.dumps(summary, indent=2))

    return f"Analyzed {len(cloned)} repos with {sum(r['python_files'] for r in summary['repos'])} Python files. Saved to {summary_file}"


async def synthesize_profile(ctx: RunContext[None], seed_profile_path: str) -> str:
    """
    Synthesize enriched profile from seed profile and GitHub analysis.

    Reads the seed profile.md and any artifacts in .dxtr/ directory,
    then creates an enriched profile using the profile synthesis agent.
    Saves result to .dxtr/profile.md.
    """
    try:
        seed_path = Path(seed_profile_path).expanduser().resolve()
        if not seed_path.exists():
            return f"Error: Seed profile not found: {seed_profile_path}"

        seed_content = seed_path.read_text()

        # Gather available artifacts
        context_parts = [f"# Seed Profile\n\n{seed_content}"]

        # Add GitHub summary if available
        github_summary_file = DXTR_DIR / "github_summary.json"
        if github_summary_file.exists():
            github_data = json.loads(github_summary_file.read_text())
            context_parts.append(f"# GitHub Summary\n\n```json\n{json.dumps(github_data, indent=2)}\n```")

        context = "\n\n---\n\n".join(context_parts)

        # Run profile synthesis agent
        result = await profile_synthesis_agent.run(
            f"Create an enriched profile from the following information:\n\n{context}",
            usage=ctx.usage,
        )

        # Save the synthesized profile
        DXTR_DIR.mkdir(parents=True, exist_ok=True)
        profile_file = DXTR_DIR / "profile.md"
        profile_file.write_text(result.output)

        return f"Profile synthesized and saved to {profile_file}"

    except Exception as e:
        return f"Error synthesizing profile: {e}"
