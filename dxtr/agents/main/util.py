"""
Utility functions for the initialization agent
"""

from pathlib import Path
from .agent import chat_with_agent


def assess_user_profile():
    """
    Interactive profile assessment:
    1. Read profile.md
    2. LLM generates clarification questions
    3. User answers questions
    4. LLM creates enriched profile with Q&A incorporated

    Saves to .dxtr/profile_enriched.md
    """
    # Read the user profile
    profile_path = Path("profile.md")
    if not profile_path.exists():
        raise FileNotFoundError("profile.md not found")

    profile_content = profile_path.read_text()

    # Check profile length and warn if too long
    estimated_tokens = len(profile_content) // 4
    max_input_tokens = 4096 * 4  # 16384 tokens

    if estimated_tokens > max_input_tokens:
        print(f"\n⚠️  WARNING: Profile is approximately {estimated_tokens} tokens")
        print(f"   This exceeds the recommended maximum of {max_input_tokens} tokens")
        print(f"   The profile may be truncated or the request may fail.\n")

    print(f"Profile size: {len(profile_content)} characters")

    # Step 1: Generate questions
    print("\n" + "=" * 80)
    print("STEP 1: GENERATING CLARIFICATION QUESTIONS")
    print("=" * 80 + "\n")

    questions_prompt = f"""You are analyzing a user profile. Your task is to generate clarification questions about anything that is unclear, ambiguous, or could benefit from more detail.

Here is the user profile:

{profile_content}

Please generate 3-7 specific questions that would help you better understand the user's background, goals, and interests. Format your response as a numbered list, with each question on its own line starting with the number.

Example format:
1. What specific computer vision architectures have you worked with before CLIP and ViT?
2. Can you provide examples of the multimodal applications you've built?

Only output the numbered questions, nothing else."""

    print("Generating questions...")
    questions_output = ""
    response_obj = None

    for chunk in chat_with_agent(
        messages=[{"role": "user", "content": questions_prompt}],
        stream=True,
        num_predict=1024,
    ):
        if hasattr(chunk.message, "content"):
            content = chunk.message.content
            questions_output += content
            print(content, end="", flush=True)
        response_obj = chunk

    print("\n")

    # Parse questions (simple line-based parsing)
    questions = []
    for line in questions_output.strip().split("\n"):
        line = line.strip()
        if line and (line[0].isdigit() or line.startswith("-")):
            # Remove numbering/bullets
            question = line.split(".", 1)[-1].split(")", 1)[-1].strip()
            if question:
                questions.append(question)

    if not questions:
        print("⚠️  No questions generated. Proceeding without Q&A.\n")
        qa_pairs = []
    else:
        # Step 2: Collect answers from user
        print("\n" + "=" * 80)
        print("STEP 2: ANSWER CLARIFICATION QUESTIONS")
        print("=" * 80 + "\n")
        print(
            "Please answer the following questions to enrich your profile.\n"
            "Press Enter with empty input to skip a question.\n"
        )

        qa_pairs = []
        for i, question in enumerate(questions, 1):
            print(f"\n{i}. {question}")
            answer = input("   Your answer: ").strip()
            if answer:
                qa_pairs.append({"question": question, "answer": answer})
            else:
                print("   (Skipped)")

    # Step 3: Generate enriched profile
    print("\n" + "=" * 80)
    print("STEP 3: GENERATING ENRICHED PROFILE")
    print("=" * 80 + "\n")

    # Build Q&A section for prompt
    qa_section = ""
    if qa_pairs:
        qa_section = "\n\nAdditional clarifications from user:\n"
        for qa in qa_pairs:
            qa_section += f"\nQ: {qa['question']}\nA: {qa['answer']}\n"

    enrichment_prompt = f"""You are creating an enriched user profile document. Based on the original profile and the user's answers to clarification questions, create a comprehensive profile that includes:

1. A clear interpretation of the user's background and expertise
2. Inline definitions for technical terms (e.g., QLORA, CLIP, ViT, few-shot learning, etc.)
3. Integration of the Q&A clarifications into the profile narrative
4. A "Questions for Future Clarification" section for anything still unclear (if any)

Original profile:
{profile_content}
{qa_section}

Please provide a well-structured markdown document that serves as the user's working profile."""

    print("Generating enriched profile...\n")
    print("-" * 80)

    enriched_output = ""
    response_obj = None

    for chunk in chat_with_agent(
        messages=[{"role": "user", "content": enrichment_prompt}],
        stream=True,
        temperature=0.0,
        num_predict=-1,
    ):
        if hasattr(chunk.message, "content"):
            content = chunk.message.content
            enriched_output += content
            print(content, end="", flush=True)
        response_obj = chunk

    print("\n" + "-" * 80)

    # Get token counts
    if response_obj:
        prompt_tokens = getattr(response_obj, "prompt_eval_count", 0)
        completion_tokens = getattr(response_obj, "eval_count", 0)
        print(f"\nInput tokens: {prompt_tokens}")
        print(f"Output tokens: {completion_tokens}")
        print(f"Output length: {len(enriched_output)} characters")

    # Create .dxtr directory if it doesn't exist
    dxtr_dir = Path(".dxtr")
    dxtr_dir.mkdir(exist_ok=True)

    # Save the enriched profile
    output_path = dxtr_dir / "profile_enriched.md"
    output_path.write_text(enriched_output)

    print(f"\nEnriched profile saved to {output_path}")

    return enriched_output
