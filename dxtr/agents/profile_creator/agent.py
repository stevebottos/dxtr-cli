"""
Profile Creator Agent

Uses Gemma3 to create rich, enriched user profiles.
Integrates with git_helper to analyze GitHub repositories.
"""

import json
import re
from pathlib import Path
from dxtr.base import Agent
from dxtr.config import config
from ..git_helper import agent as git_helper
from ..git_helper.tools import git_tools


class ProfileCreatorAgent(Agent):
    """Agent for creating enriched user profiles using Gemma3."""

    def __init__(self):
        """Initialize profile creator agent."""
        model_config = config.get_model_config("profile_creator")
        super().__init__(
            name="profile_creator",
            model=model_config.name,
            prompts_dir=Path(__file__).parent / "prompts",
            default_options={
                "temperature": model_config.temperature,
                "num_ctx": model_config.context_window,
            }
        )

    def _extract_github_profile_url(self, profile_content: str) -> str | None:
        """
        Extract GitHub profile URL from profile content.

        Args:
            profile_content: Raw profile.md content

        Returns:
            GitHub profile URL or None if not found
        """
        url_pattern = r'https?://github\.com/[^\s<>"{}|\\^`\[\]]+'
        urls = re.findall(url_pattern, profile_content)

        for url in urls:
            if git_tools.is_profile_url(url):
                return url

        return None

    def create_profile(self) -> dict:
        """
        Create enriched user profile from seed profile.md.

        Process:
        1. Read seed profile.md
        2. Extract GitHub profile URL
        3. Call git_helper to analyze repos -> github_summary.json
        4. Use Gemma3 to create enriched .dxtr/dxtr_profile.md

        Returns:
            dict with keys:
                - success: bool
                - profile_path: str (path to enriched profile)
                - error: str (if failed)
        """
        seed_profile_path = config.paths.seed_profile_file

        if not seed_profile_path.exists():
            return {
                "success": False,
                "error": "No profile.md found. Please create profile.md with your information and GitHub URL."
            }

        print("\n" + "=" * 80)
        print("PROFILE INITIALIZATION")
        print("=" * 80 + "\n")

        profile_content = seed_profile_path.read_text()
        print(f"[Reading seed profile: {len(profile_content)} characters]")

        config.paths.dxtr_dir.mkdir(exist_ok=True)

        github_url = self._extract_github_profile_url(profile_content)
        github_summary = {}

        if github_url:
            print(f"[Found GitHub profile URL: {github_url}]")
            github_summary = git_helper.run(github_url)

            if github_summary:
                config.paths.github_summary_file.write_text(json.dumps(github_summary, indent=2))
                print(f"\n[✓] Saved github_summary.json ({len(github_summary)} files)")
            else:
                print("\n[No GitHub analysis data to save]")
        else:
            print("[No GitHub profile URL found in profile.md]")

        print("\n" + "=" * 80)
        print("GENERATING ENRICHED PROFILE")
        print("=" * 80 + "\n")

        enrichment_context = f"""# Original Profile

{profile_content}
"""

        if github_summary:
            num_files = len(github_summary)
            repos = set()
            for file_path in github_summary.keys():
                parts = Path(file_path).parts
                if 'repos' in parts:
                    idx = parts.index('repos')
                    if idx + 2 < len(parts):
                        repos.add(f"{parts[idx+1]}/{parts[idx+2]}")

            enrichment_context += f"""

# GitHub Analysis Summary

- **Files analyzed**: {num_files} Python files
- **Repositories**: {len(repos)} repositories
  - {chr(10).join([f'  - {repo}' for repo in sorted(repos)])}
- **Detailed analysis**: Available in .dxtr/github_summary.json
"""
        else:
            enrichment_context += "\n# GitHub Analysis Summary\n\nNo GitHub profile found or no repositories analyzed.\n"

        print("[Calling Gemma3 to enrich profile...]")

        response = self.chat(
            messages=[{"role": "user", "content": enrichment_context}],
            prompt_name="profile_creation"
        )

        enriched_profile = response.message.content.strip()

        config.paths.profile_file.write_text(enriched_profile)

        print(f"\n[✓] Enriched profile saved to .dxtr/dxtr_profile.md")
        print(f"[✓] Profile initialization complete\n")

        print("=" * 80)
        print("DXTR PROFILE")
        print("=" * 80 + "\n")
        print(enriched_profile)
        print("\n" + "=" * 80 + "\n")

        return {
            "success": True,
            "profile_path": str(config.paths.profile_file)
        }


# Global instance for backward compatibility
_agent = ProfileCreatorAgent()


def create_profile() -> dict:
    """
    Create enriched user profile from seed profile.md.

    This is a convenience function that delegates to the agent instance.

    Returns:
        dict with keys:
            - success: bool
            - profile_path: str (path to enriched profile)
            - error: str (if failed)
    """
    return _agent.create_profile()
