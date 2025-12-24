"""
Git Tools Module

Provides functionality to clone and cache GitHub repositories.
"""

import subprocess
import re
from pathlib import Path
from urllib.parse import urlparse


def _parse_repo_url(url: str) -> tuple[str, str] | None:
    """
    Parse a GitHub repository URL to extract owner and repo name.

    Args:
        url: GitHub repository URL (e.g., https://github.com/owner/repo)

    Returns:
        Tuple of (owner, repo_name) or None if not a valid repo URL
    """
    # Handle various GitHub URL formats
    patterns = [
        r'github\.com/([^/]+)/([^/\.]+)',  # https://github.com/owner/repo
        r'github\.com/([^/]+)/([^/]+)\.git',  # https://github.com/owner/repo.git
    ]

    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            owner, repo = match.groups()
            # Remove .git suffix if present (using removesuffix for Python 3.9+)
            if repo.endswith('.git'):
                repo = repo[:-4]
            return owner, repo

    return None


def _get_repo_path(owner: str, repo: str) -> Path:
    """
    Generate the local path for a cloned repository.

    Args:
        owner: Repository owner
        repo: Repository name

    Returns:
        Path object for the repository directory
    """
    return Path(".dxtr") / "repos" / owner / repo


def is_repo_cloned(url: str) -> tuple[bool, Path | None]:
    """
    Check if a repository is already cloned.

    Args:
        url: GitHub repository URL

    Returns:
        Tuple of (is_cloned, repo_path)
    """
    parsed = _parse_repo_url(url)
    if not parsed:
        return False, None

    owner, repo = parsed
    repo_path = _get_repo_path(owner, repo)

    # Check if directory exists and contains .git
    if repo_path.exists() and (repo_path / ".git").exists():
        return True, repo_path

    return False, repo_path


def clone_repo(url: str) -> dict[str, any]:
    """
    Clone a GitHub repository to .dxtr/repos/.

    Uses caching - if the repo is already cloned, returns success without re-cloning.

    Args:
        url: GitHub repository URL

    Returns:
        Dict with keys:
            - success: bool indicating if clone succeeded
            - path: Path to cloned repo (if successful)
            - message: Status message
            - url: Original URL
            - owner: Repository owner
            - repo: Repository name
    """
    # Check if already cloned
    is_cloned, repo_path = is_repo_cloned(url)

    if is_cloned:
        parsed = _parse_repo_url(url)
        owner, repo = parsed
        return {
            "success": True,
            "path": str(repo_path),
            "message": f"Repository already cloned (cached)",
            "url": url,
            "owner": owner,
            "repo": repo,
        }

    # Parse URL
    parsed = _parse_repo_url(url)
    if not parsed:
        return {
            "success": False,
            "path": None,
            "message": f"Invalid GitHub repository URL: {url}",
            "url": url,
            "owner": None,
            "repo": None,
        }

    owner, repo = parsed
    repo_path = _get_repo_path(owner, repo)

    # Create parent directory
    repo_path.parent.mkdir(parents=True, exist_ok=True)

    # Clone the repository
    try:
        print(f"  [Cloning {owner}/{repo}...]")
        result = subprocess.run(
            ["git", "clone", "--depth", "1", url, str(repo_path)],
            capture_output=True,
            text=True,
            timeout=120,  # 2 minute timeout
        )

        if result.returncode == 0:
            return {
                "success": True,
                "path": str(repo_path),
                "message": f"Successfully cloned {owner}/{repo}",
                "url": url,
                "owner": owner,
                "repo": repo,
            }
        else:
            return {
                "success": False,
                "path": None,
                "message": f"Git clone failed: {result.stderr}",
                "url": url,
                "owner": owner,
                "repo": repo,
            }

    except subprocess.TimeoutExpired:
        return {
            "success": False,
            "path": None,
            "message": f"Clone timeout (exceeded 120s)",
            "url": url,
            "owner": owner,
            "repo": repo,
        }
    except Exception as e:
        return {
            "success": False,
            "path": None,
            "message": f"Clone error: {str(e)}",
            "url": url,
            "owner": owner,
            "repo": repo,
        }


def clone_repos(urls: list[str]) -> list[dict[str, any]]:
    """
    Clone multiple repositories.

    Args:
        urls: List of GitHub repository URLs

    Returns:
        List of result dicts from clone_repo()
    """
    results = []
    for url in urls:
        result = clone_repo(url)
        results.append(result)

    return results
