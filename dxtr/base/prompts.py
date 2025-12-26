"""
Prompt management for agents.

Key principle: ZERO prompts in code. All prompts live in markdown files.
"""

from pathlib import Path


class PromptManager:
    """
    Manages prompt loading from filesystem.

    All prompts must be stored as markdown files in the prompts directory.
    This ensures:
    - Prompts are version controlled separately from code
    - Easy to edit without touching code
    - Can be tested independently
    - Clear separation of concerns
    """

    def __init__(self, prompts_dir: Path):
        """
        Initialize prompt manager.

        Args:
            prompts_dir: Directory containing prompt markdown files
        """
        self.prompts_dir = Path(prompts_dir)

    def load(self, prompt_name: str) -> str:
        """
        Load a prompt from the filesystem.

        Args:
            prompt_name: Name of the prompt file (without .md extension)

        Returns:
            str: The prompt content

        Raises:
            FileNotFoundError: If prompt file doesn't exist
        """
        prompt_path = self.prompts_dir / f"{prompt_name}.md"

        if not prompt_path.exists():
            raise FileNotFoundError(
                f"Prompt '{prompt_name}' not found at {prompt_path}\n"
                f"Available prompts: {self.list_prompts()}"
            )

        return prompt_path.read_text().strip()

    def list_prompts(self) -> list[str]:
        """
        List all available prompts.

        Returns:
            list[str]: List of prompt names (without .md extension)
        """
        if not self.prompts_dir.exists():
            return []

        return [
            p.stem for p in self.prompts_dir.glob("*.md")
        ]

    def exists(self, prompt_name: str) -> bool:
        """
        Check if a prompt exists.

        Args:
            prompt_name: Name of the prompt file (without .md extension)

        Returns:
            bool: True if prompt exists
        """
        return (self.prompts_dir / f"{prompt_name}.md").exists()
