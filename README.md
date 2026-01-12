<p align="center">
  <img src="assets/logo.png" alt="DXTR Logo" width="600"/>
</p>

# DXTR

**Status: Work in Progress**

DXTR is an AI research assistant for machine learning engineers. It helps you stay current with ML/AI research by intelligently filtering, ranking, and analyzing papers based on your interests and background.

## Features

- **Daily Paper Pipeline**: Automated ETL from HuggingFace daily papers with Docling for PDF processing
- **Personalized Ranking**: Papers ranked based on your profile and GitHub activity
- **Agentic Deep Research**: Multi-step RAG system that generates exploration questions to retrieve relevant paper sections
- **Profile Synthesis**: Automatically analyzes your GitHub repos to understand your expertise
- **Streaming Output**: See agent outputs in real-time as they think and respond

## Installation

### Prerequisites

- Python 3.12+
- CUDA-capable GPU (12GB+ VRAM recommended)
- [uv](https://github.com/astral-sh/uv) package manager (recommended)

### Setup

```bash
# 1. Install sglang (requires prerelease for latest features)
uv pip install "sglang[all]" --prerelease=allow

# 2. Install dxtr
pip install -e .

# 3. (Optional) Install pexpect for automated evaluation
pip install pexpect
```

## Quick Start

### 1. Start the SGLang Server

```bash
# List available models
python start_server.py --list

# Start with default model (qwen3-8b)
python start_server.py

# Or choose a specific model
python start_server.py qwen-14b        # Higher quality
python start_server.py deepseek-r1-8b  # Best reasoning
```

### 2. Download Papers

```bash
# Download today's papers from HuggingFace
python -m dxtr.cli get-papers

# Or specify a date
python -m dxtr.cli get-papers --date 2026-01-09
```

### 3. Start Chat

```bash
python -m dxtr.cli chat
```

On first run, DXTR will:
1. Ask for your profile path (e.g., `./profile.md`)
2. Analyze your GitHub repositories
3. Synthesize a personalized profile

Then you can:
- `rank the papers from 2026-01-09` - Get personalized paper rankings
- `deep dive into paper 2512.12345` - Agentic RAG analysis
- Ask follow-up questions about papers

## Profile Format

Create a `profile.md` with your background and interests:

```markdown
# About Me
I am a machine learning engineer with X years of experience...

# Interests
- Topic 1
- Topic 2
- What I want to learn

# Links
https://github.com/yourusername
```

DXTR will analyze your GitHub repos and synthesize a detailed profile with:
- Technical competencies (strong areas, learning areas, gaps)
- Interest signals (HIGH/LOW priority topics for paper ranking)
- Constraints (hardware, preferences)
- Goals (immediate, career direction)

## Architecture

### Agents

| Agent | Purpose |
|-------|---------|
| **Main** | Orchestrates tasks, handles tool calls |
| **GitHub Summarize** | Analyzes GitHub repos → `.dxtr/github_summary.json` |
| **Profile Synthesize** | Creates profile from artifacts → `.dxtr/dxtr_profile.md` |
| **Papers Ranking** | Ranks papers by relevance to profile |
| **Deep Research** | RAG-based deep dive into papers |

### Key Design Principles

- **One agent, one task**: Each agent has a focused purpose with its own `system.md` prompt
- **Prompts in markdown**: All prompts live in `agents/*/system.md`, not in code
- **Main orchestrates**: Only the main agent can invoke other agents
- **Streaming by default**: Agent outputs stream in real-time

## Evaluation

DXTR includes an LLM-as-a-judge evaluation system:

```bash
# Run automated user journey (requires pexpect)
python eval/e2e_journey/run_eval.py

# Artifacts saved to .dxtr_eval/debug/
# - transcript.txt (full conversation)
# - profile_artifacts/ (generated profiles)

# Then ask Claude to evaluate using:
# eval/llm_as_a_judge.md
```

See `eval/llm_as_a_judge.md` for evaluation criteria and methodology.

## Configuration

All configuration is in `dxtr/config_v2.py`:

- SGLang server URL (default: `http://localhost:30000`)
- Model parameters (temperature, max_tokens)
- File paths (`.dxtr/` directory structure)

## Development

```bash
# Run tests
pytest

# Check syntax
python -m py_compile dxtr/cli.py
```

## License

MIT
