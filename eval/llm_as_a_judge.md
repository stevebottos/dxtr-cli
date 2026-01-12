# LLM-as-a-Judge Evaluation

This document describes the end-to-end evaluation methodology for DXTR. The goal is to assess system quality through a complete user journey, with an LLM (you, Claude) serving as the judge.

## Quick Start

```bash
# 1. Run the automated user journey
python eval/e2e_journey/run_eval.py

# 2. Review artifacts in .dxtr_eval/debug/
# 3. Ask Claude to evaluate using this document as reference
```

## User Journey Overview

The evaluation simulates a new user's first interaction with DXTR:

```
┌─────────────────────────────────────────────────────────────────┐
│                        USER JOURNEY                              │
├─────────────────────────────────────────────────────────────────┤
│  1. dxtr chat                                                    │
│     └── DXTR detects no profile, asks for one                   │
│                                                                  │
│  2. User provides: ./profile.md                                  │
│     └── DXTR reads profile, detects GitHub URL                  │
│                                                                  │
│  3. DXTR asks to summarize GitHub → User says "yes"              │
│     └── github_summarize agent runs                              │
│     └── Output: .dxtr/github_summary.json                        │
│                                                                  │
│  4. DXTR asks to synthesize profile → User says "yes"            │
│     └── profile_synthesize agent runs                            │
│     └── Output: .dxtr/dxtr_profile.md                            │
│                                                                  │
│  5. User: "rank today's papers for me"                           │
│     └── DXTR checks for papers, finds none                       │
│     └── Asks to download → User says "yes"                       │
│     └── Papers downloaded via ETL                                │
│                                                                  │
│  6. DXTR asks to proceed with ranking → User says "yes"          │
│     └── papers_ranking agent runs                                │
│     └── Output: Rankings printed to console                      │
│                                                                  │
│  7. User: "Best paper from top-5 for a side project?"            │
│     └── DXTR analyzes and recommends                             │
│     └── Output: Recommendation with reasoning                    │
└─────────────────────────────────────────────────────────────────┘
```

## Artifacts to Evaluate

After running `eval/e2e_journey/run_eval.py`, artifacts are saved to `.dxtr_eval/debug/`:

| File | Description |
|------|-------------|
| `transcript.txt` | Complete conversation log (contains rankings, recommendation, everything) |
| `profile_artifacts/github_summary.json` | GitHub analysis |
| `profile_artifacts/dxtr_profile.md` | Synthesized profile |

The transcript contains the full conversation including:
- Paper rankings (look for "Paper Rankings for" section)
- Best paper recommendation (look for response to "best for a side project" question)
- All DXTR responses and tool outputs

## Evaluation Criteria

### 1. Profile Creation Quality

**Artifact:** `.dxtr_eval/debug/profile_artifacts/github_summary.json`

| Criterion | Weight | Description |
|-----------|--------|-------------|
| **Accuracy** | 30% | Does the summary accurately reflect the GitHub repos? No hallucinated repos or technologies. |
| **Completeness** | 25% | Are all relevant repos captured? Are key technologies identified? |
| **Relevance Extraction** | 25% | Does it correctly identify the user's expertise areas? |
| **Structure** | 20% | Is the JSON well-formed and organized logically? |

**Questions to answer:**
- Are there any hallucinated repositories or technologies?
- Does it capture the user's main areas of expertise?
- Are repo descriptions accurate to what the code actually does?

---

**Artifact:** `.dxtr_eval/debug/profile_artifacts/dxtr_profile.md`

| Criterion | Weight | Description |
|-----------|--------|-------------|
| **Synthesis Quality** | 30% | Does it correctly merge profile.md and github_summary.json? |
| **Coherence** | 25% | Is the profile logically organized and readable? |
| **No Hallucinations** | 25% | Does it only contain information from the source artifacts? |
| **Actionability** | 20% | Is the profile useful for paper ranking and recommendations? |

**Questions to answer:**
- Does the synthesized profile accurately represent the user?
- Is any information fabricated that wasn't in the inputs?
- Would this profile help identify relevant papers?

---

### 2. Paper Ranking Quality

**Artifact:** `.dxtr_eval/debug/transcript.txt` (look for "Paper Rankings for" section)

| Criterion | Weight | Description |
|-----------|--------|-------------|
| **Profile Alignment** | 35% | Do top-ranked papers match user's stated interests? |
| **Reasoning Quality** | 25% | Are ranking justifications logical and specific? |
| **No Hallucinations** | 20% | Are paper descriptions accurate to actual paper content? |
| **Diversity** | 10% | Does ranking capture breadth of user interests? |
| **Score Calibration** | 10% | Are scores distributed reasonably (not all 5s or all 1s)? |

**CRITICAL: Cross-Reference Rankings Against Profile Priorities**

You MUST explicitly verify rankings against the profile's stated priorities:

1. **Extract HIGH PRIORITY topics from profile** (e.g., "Video Captioning", "Agentic LLMs", "Small Language Models")
2. **Extract LOW PRIORITY topics from profile** (e.g., "Pure Image Classification", "Large-scale training 70B+")
3. **For each HIGH PRIORITY topic:**
   - Find papers that match this topic
   - Verify they are ranked in top 10 (score 4.0+)
   - **FLAG if a HIGH PRIORITY topic has no papers in top 10**
4. **For each LOW PRIORITY topic:**
   - Find papers that match this topic
   - Verify they are ranked low (score < 3.0)
   - **FLAG if LOW PRIORITY papers are ranked high**

**Create a table like this:**

| Profile Priority | Topic | Matching Papers | Ranked Position | Correct? |
|------------------|-------|-----------------|-----------------|----------|
| HIGH | [Topic from profile] | [Papers matching topic] | [Rank #] | ✅/❌ |
| HIGH | [Topic from profile] | [Papers matching topic] | [Rank #] | ✅/❌ |
| LOW | [Topic from profile] | [Papers matching topic] | [Rank #] | ✅/❌ |

Example interpretation:
- If a HIGH PRIORITY topic has no papers in top 10: ❌ Flag as issue
- If a LOW PRIORITY topic has papers ranked high (4.0+): ❌ Flag as issue

**Questions to answer:**
- Do top-5 papers genuinely match the user's stated HIGH PRIORITY interests?
- Are low-ranked papers correctly identified as less relevant?
- Does the reasoning cite specific aspects of papers and profile?
- Are there any papers ranked high/low that seem miscategorized?
- **Are any HIGH PRIORITY topics missing from the top 10?**
- **Are any LOW PRIORITY topics incorrectly ranked high?**

---

### 3. Recommendation Quality

**Artifact:** `.dxtr_eval/debug/transcript.txt` (look for response to "best for a side project" question)

| Criterion | Weight | Description |
|-----------|--------|-------------|
| **Selection Justification** | 30% | Is the chosen paper well-justified vs alternatives? |
| **Profile Matching** | 25% | Does selection account for user's stated constraints (hardware limits, time, focus areas)? |
| **Reasoning Depth** | 25% | Is the explanation detailed and actionable? |
| **No Hallucinations** | 20% | Is paper description accurate? No fabricated claims? |

**Questions to answer:**
- Is the recommended paper actually feasible for the user's constraints?
- Does the reasoning consider practical aspects (compute requirements, time investment)?
- Would this recommendation genuinely help the user's career goals?

---

### 4. Conversation Quality

**Artifact:** `.dxtr_eval/debug/transcript.txt`

| Criterion | Weight | Description |
|-----------|--------|-------------|
| **Confirmation Pattern** | 25% | Does DXTR ask for confirmation before tool calls? |
| **Context Awareness** | 25% | Does DXTR maintain awareness of user profile throughout? |
| **Error Handling** | 20% | How does DXTR handle edge cases (no papers, API errors)? |
| **Conciseness** | 15% | Are responses appropriately brief? |
| **Helpfulness** | 15% | Does DXTR guide the user effectively? |

**Questions to answer:**
- Did DXTR ask before running expensive operations?
- Were tool feedback messages (e.g., "[Downloading papers...]") clear?
- Did DXTR correctly handle the no-papers-found case?

---

## Scoring Rubric

For each artifact, assign a score from 1-5:

| Score | Description |
|-------|-------------|
| **5** | Excellent - Exceeds expectations, no issues found |
| **4** | Good - Meets expectations, minor issues |
| **3** | Acceptable - Works but has notable issues |
| **2** | Poor - Significant problems affecting usefulness |
| **1** | Failing - Major issues, hallucinations, or broken functionality |

---

## Evaluation Template

Use this template when evaluating:

```markdown
# DXTR E2E Evaluation Report

**Date:** [DATE]
**Evaluator:** Claude (LLM-as-a-judge)

## Summary

| Component | Score (1-5) | Key Issues |
|-----------|-------------|------------|
| GitHub Summary | X | ... |
| Synthesized Profile | X | ... |
| Paper Rankings | X | ... |
| Recommendation | X | ... |
| Conversation Flow | X | ... |
| **Overall** | X | ... |

## Detailed Analysis

### 1. Profile Creation

**GitHub Summary:**
- Accuracy: [score/5] - [notes]
- Completeness: [score/5] - [notes]
- Hallucinations found: [yes/no] - [details if yes]

**Synthesized Profile:**
- Quality: [score/5] - [notes]
- Coherence: [score/5] - [notes]
- Hallucinations found: [yes/no] - [details if yes]

### 2. Paper Rankings

- Profile Alignment: [score/5] - [notes]
- Reasoning Quality: [score/5] - [notes]
- Top-5 Assessment: [list papers and whether ranking seems correct]
- Hallucinations found: [yes/no] - [details if yes]

### 3. Recommendation

- Selection: [paper title]
- Justification Quality: [score/5] - [notes]
- Feasibility for User: [score/5] - [notes]
- Hallucinations found: [yes/no] - [details if yes]

### 4. Conversation Flow

- Confirmation Pattern: [score/5] - [notes]
- Error Handling: [score/5] - [notes]
- UX Quality: [score/5] - [notes]

## Issues Found

### Critical (Must Fix)
- [List any critical issues]

### Important (Should Fix)
- [List important issues]

### Minor (Nice to Fix)
- [List minor issues]

## Recommendations

1. [First recommendation for improvement]
2. [Second recommendation]
3. [etc.]
```

---

## Running the Evaluation

### Automated (with pexpect)

```bash
# Install pexpect if needed
pip install pexpect

# Run eval
python eval/e2e_journey/run_eval.py
```

### Manual (without pexpect)

1. Clear state: `rm -rf .dxtr/github_summary.json .dxtr/dxtr_profile.md`
2. Run: `python -m dxtr.cli chat`
3. Follow the user journey steps above
4. Copy outputs to `.dxtr_eval/debug/`

### Asking Claude to Evaluate

In a fresh conversation, provide:

1. This document (`eval/llm_as_a_judge.md`)
2. The original `profile.md`
3. All artifacts from `.dxtr_eval/debug/`

Then ask:

> "Please evaluate these DXTR outputs using the criteria in llm_as_a_judge.md. Generate a full evaluation report."

---

## Ground Truth References

For paper ranking evaluation, compare against the user's stated interests from their `profile.md` and synthesized `dxtr_profile.md`:

**Extract from the profile:**
1. **HIGH PRIORITY topics** - These should have papers ranked in top 10
2. **LOW PRIORITY topics** - These should have papers ranked low
3. **Constraints** - Hardware limits, time availability, preferences
4. **Goals** - Immediate goals, career direction

**Validation rules:**
- Papers matching HIGH PRIORITY topics should score 4.0+
- Papers matching LOW PRIORITY topics should score < 3.0
- Recommendations should respect stated constraints
- Papers on unrelated topics should rank lower unless they have clear practical applications to user's goals

---

## Changelog

- 2026-01-09: Initial version
