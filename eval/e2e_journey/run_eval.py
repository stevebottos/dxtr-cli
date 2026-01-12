#!/usr/bin/env python3
"""
End-to-End User Journey Evaluation

Automates the complete DXTR user journey from a fresh start:
1. Profile creation (provide profile.md, GitHub summarization, profile synthesis)
2. Paper ranking (using pre-downloaded papers)
3. Best paper recommendation

Prerequisites: Papers must be downloaded first via `dxtr get-papers`

Outputs:
- .dxtr_eval/debug/transcript.txt - Full conversation
- .dxtr_eval/debug/profile_artifacts/ - Generated profile files

Usage: python eval/e2e_journey/run_eval.py
"""

import sys
import shutil
import time
from pathlib import Path
from datetime import datetime

try:
    import pexpect
    HAS_PEXPECT = True
except ImportError:
    HAS_PEXPECT = False

# Paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

EVAL_DIR = PROJECT_ROOT / ".dxtr_eval"
DEBUG_DIR = EVAL_DIR / "debug"
TRANSCRIPT_FILE = DEBUG_DIR / "transcript.txt"
PROFILE_ARTIFACTS_DIR = DEBUG_DIR / "profile_artifacts"


def find_available_papers() -> tuple[str | None, int]:
    """Find available papers in .dxtr/hf_papers/."""
    papers_dir = PROJECT_ROOT / ".dxtr" / "hf_papers"
    if not papers_dir.exists():
        return None, 0

    available = []
    for date_dir in papers_dir.iterdir():
        if not date_dir.is_dir():
            continue
        paper_count = sum(
            1 for p in date_dir.iterdir()
            if p.is_dir() and (p / "metadata.json").exists()
        )
        if paper_count > 0:
            available.append((date_dir.name, paper_count))

    if not available:
        return None, 0

    available.sort(reverse=True)
    return available[0]


def build_user_journey(papers_date: str) -> list[dict]:
    """Build user journey steps."""
    return [
        {"say": "./profile.md", "wait_for": r"(read|Read|proceed|Proceed)", "description": "Provide profile path"},
        {"say": "yes", "wait_for": r"(GitHub|github|analyze|summarize)", "description": "Confirm reading profile", "timeout": 60},
        {"say": "yes", "wait_for": r"(synthesize|profile|complete)", "description": "Confirm GitHub summarization", "timeout": 180},
        {"say": "yes", "wait_for": r"(ready|help|assist|paper|created|saved)", "description": "Confirm profile synthesis", "timeout": 180},
        {"say": f"rank the papers from {papers_date}", "wait_for": r"(rank|proceed|Shall)", "description": f"Request paper ranking"},
        {"say": "yes", "wait_for": r"(Ranked|ranked|Top|ranking)", "description": "Confirm ranking", "timeout": 180},
        {"say": "Based on the top 5 papers, which single paper would be the absolute best for a side project given my profile? Explain why.", "wait_for": r".", "description": "Request recommendation", "timeout": 120},
    ]


def setup_eval_directory(papers_date: str, paper_count: int):
    """Create fresh eval directory."""
    if EVAL_DIR.exists():
        shutil.rmtree(EVAL_DIR)

    DEBUG_DIR.mkdir(parents=True)
    PROFILE_ARTIFACTS_DIR.mkdir(parents=True)

    (DEBUG_DIR / "metadata.txt").write_text(
        f"E2E Journey Evaluation\n"
        f"Started: {datetime.now().isoformat()}\n"
        f"Profile: {PROJECT_ROOT / 'profile.md'}\n"
        f"Papers: {paper_count} from {papers_date}\n"
    )


def clear_dxtr_state():
    """Clear .dxtr directory, preserving papers."""
    dxtr_dir = PROJECT_ROOT / ".dxtr"
    if not dxtr_dir.exists():
        return

    papers_dir = dxtr_dir / "hf_papers"
    papers_backup = None

    if papers_dir.exists():
        papers_backup = PROJECT_ROOT / ".dxtr_papers_backup"
        if papers_backup.exists():
            shutil.rmtree(papers_backup)
        shutil.move(str(papers_dir), str(papers_backup))

    for item in dxtr_dir.iterdir():
        if item.is_dir():
            shutil.rmtree(item)
        else:
            item.unlink()

    if papers_backup and papers_backup.exists():
        shutil.move(str(papers_backup), str(papers_dir))

    print("[Setup] Cleared .dxtr state (preserved papers)")


class TranscriptLogger:
    """Log to both stdout and file."""
    def __init__(self, filename):
        self.file = open(filename, "w", encoding="utf-8")
        self.stdout = sys.stdout

    def write(self, data):
        self.file.write(data)
        self.stdout.write(data)
        self.file.flush()
        self.stdout.flush()

    def flush(self):
        self.file.flush()
        self.stdout.flush()

    def close(self):
        self.file.close()


def copy_profile_artifacts():
    """Copy generated profile artifacts."""
    dxtr_dir = PROJECT_ROOT / ".dxtr"
    for artifact in ["github_summary.json", "dxtr_profile.md"]:
        src = dxtr_dir / artifact
        if src.exists():
            shutil.copy(src, PROFILE_ARTIFACTS_DIR / artifact)
            print(f"[Copied] {artifact}")


def run_with_pexpect(papers_date: str, paper_count: int):
    """Run the eval using pexpect."""
    print("\n" + "=" * 60)
    print("E2E JOURNEY EVALUATION")
    print("=" * 60 + "\n")
    print(f"[Papers] Using {paper_count} papers from {papers_date}")

    setup_eval_directory(papers_date, paper_count)
    clear_dxtr_state()

    user_journey = build_user_journey(papers_date)
    logger = TranscriptLogger(TRANSCRIPT_FILE)

    print("[Starting] dxtr chat...")

    child = pexpect.spawn(
        "python", ["-m", "dxtr.cli", "chat"],
        cwd=str(PROJECT_ROOT),
        encoding="utf-8",
        timeout=300,
    )
    child.logfile_read = logger

    try:
        print("\n[Waiting] Initial greeting...")
        child.expect([r"profile", r"Profile", r"DXTR", r"Hello"], timeout=60)

        for i, step in enumerate(user_journey):
            print(f"\n[Step {i+1}] {step['description']}")

            try:
                child.expect(r"You:", timeout=60)
            except pexpect.TIMEOUT:
                pass

            time.sleep(0.5)
            child.sendline(step["say"])

            timeout = step.get("timeout", 120)
            try:
                child.expect(step["wait_for"], timeout=timeout)
            except pexpect.TIMEOUT:
                print(f"[TIMEOUT] Waiting for: {step['wait_for']}")
            except pexpect.EOF:
                print("[EOF] Process ended")
                break

        # Wait for final response to complete
        time.sleep(5)
        try:
            child.expect(r"You:", timeout=60)
        except pexpect.TIMEOUT:
            pass

    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()
    finally:
        child.close()
        logger.close()

    copy_profile_artifacts()

    print("\n" + "=" * 60)
    print("EVALUATION COMPLETE")
    print("=" * 60)
    print(f"\nOutputs saved to: {DEBUG_DIR}")
    print("  - transcript.txt (full conversation)")
    print("  - profile_artifacts/")
    print("\nNext: Ask Claude to evaluate using eval/llm_as_a_judge.md")


def run_manual(papers_date: str, paper_count: int):
    """Manual instructions when pexpect unavailable."""
    print("\n" + "=" * 60)
    print("E2E JOURNEY EVALUATION (MANUAL)")
    print("=" * 60)
    print(f"\n[Papers] Using {paper_count} papers from {papers_date}\n")

    setup_eval_directory(papers_date, paper_count)
    clear_dxtr_state()

    print("1. Run: python -m dxtr.cli chat")
    print("2. When asked for profile: ./profile.md")
    print("3. Say 'yes' to read profile")
    print("4. Say 'yes' to summarize GitHub")
    print("5. Say 'yes' to synthesize profile")
    print(f"6. Say: rank the papers from {papers_date}")
    print("7. Say 'yes' to rank papers")
    print("8. Ask: Based on the top 5 papers, which single paper would be")
    print("   the absolute best for a side project given my profile?")
    print(f"\n9. Save transcript to: {TRANSCRIPT_FILE}")
    print("\nThen ask Claude to evaluate using eval/llm_as_a_judge.md")


def main():
    papers_date, paper_count = find_available_papers()

    if papers_date is None:
        print("\n" + "=" * 60)
        print("ERROR: No papers found")
        print("=" * 60)
        print("\nDownload papers first:")
        print("  python -m dxtr.cli get-papers")
        sys.exit(1)

    if HAS_PEXPECT:
        run_with_pexpect(papers_date, paper_count)
    else:
        run_manual(papers_date, paper_count)
        print("\nTo enable automation: pip install pexpect")


if __name__ == "__main__":
    main()
