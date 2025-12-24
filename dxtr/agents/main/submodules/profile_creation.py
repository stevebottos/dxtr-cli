"""
Profile Creation Submodule

This submodule handles the interactive profile creation/enrichment flow.
It has its own system prompt and manages state separately from the main chat.
"""

import json
import re
from pathlib import Path
from ollama import chat
from ..agent import MODEL, _load_system_prompt
from ..tools.web_fetch import fetch_url, TOOL_DEFINITION
from . import github_explorer, website_explorer


def _extract_and_explore_links(profile_content: str) -> str:
    """
    Extract URLs from profile and explore them using appropriate submodules.

    Args:
        profile_content: The raw profile.md content

    Returns:
        str: Markdown-formatted summaries of all explored links
    """
    # Extract URLs from the profile
    url_pattern = r'https?://[^\s<>"{}|\\^`\[\]]+'
    urls = re.findall(url_pattern, profile_content)

    if not urls:
        return ""

    print(f"\nFound {len(urls)} link(s) in profile. Exploring...")

    summaries = []

    for url in urls:
        # Route to appropriate explorer
        if 'github.com' in url:
            summary = github_explorer.explore(url, user_profile_md=profile_content)
        else:
            summary = website_explorer.explore(url)

        if summary:
            summaries.append(summary)

    if summaries:
        return "\n\n---\n\n".join(summaries)
    return ""


def _chat_with_tools(messages, tools=None, stream_output=True):
    """
    Chat with tool calling support.
    
    Handles the tool calling loop:
    1. Send messages to LLM with available tools
    2. If LLM calls a tool, execute it
    3. Send tool results back to LLM
    4. Repeat until LLM provides final response
    
    Args:
        messages: List of message dicts
        tools: List of tool definitions (Ollama format)
        stream_output: Whether to stream the final text output
        
    Returns:
        tuple: (final_text, response_obj) where final_text is the complete response
    """
    # Add system prompt if not present
    if not messages or messages[0].get("role") != "system":
        system_prompt = _load_system_prompt("profile_creation")
        messages = [{"role": "system", "content": system_prompt}] + messages
    
    conversation = messages.copy()
    final_text = ""
    response_obj = None
    
    # Tool calling loop
    while True:
        # Call the model
        response = chat(
            model=MODEL,
            messages=conversation,
            tools=tools if tools else [],
            options={
                "temperature": 0.3,
                "num_ctx": 4096 * 4,
            }
        )
        
        response_obj = response
        
        # Check if the model wants to call tools
        if hasattr(response.message, 'tool_calls') and response.message.tool_calls:
            # Execute each tool call
            for tool_call in response.message.tool_calls:
                tool_name = tool_call.function.name
                tool_args = tool_call.function.arguments
                
                print(f"\n[Calling tool: {tool_name}({json.dumps(tool_args)})]")
                
                # Execute the tool
                if tool_name == "fetch_url":
                    result = fetch_url(**tool_args)
                    if result["success"]:
                        print(f"[Fetched: {result['url'][:60]}...]")
                    else:
                        print(f"[Error: {result['error']}]")
                else:
                    result = {"error": f"Unknown tool: {tool_name}"}
                
                # Add tool call and result to conversation
                conversation.append({
                    "role": "assistant",
                    "tool_calls": [tool_call]
                })
                conversation.append({
                    "role": "tool",
                    "content": json.dumps(result)
                })
            
            # Continue the loop to get next response
            continue
        else:
            # No tool calls, this is the final response
            if hasattr(response.message, 'content'):
                final_text = response.message.content
                if stream_output:
                    # Stream the output for user to see
                    print(final_text, end="", flush=True)
            break
    
    return final_text, response_obj


def run():
    """
    Run the profile creation submodule.

    This is an interactive submodule that:
    1. Checks for existing profile.md
    2. If missing, prompts user to create it
    3. Runs Q&A session with user (LLM can explore links via tools)
    4. Generates enriched profile
    5. Returns control to main chat

    Returns:
        str: Path to the enriched profile, or None if creation failed
    """
    profile_path = Path("profile.md")

    # Check if profile.md exists
    if not profile_path.exists():
        print("\n" + "=" * 80)
        print("PROFILE NOT FOUND")
        print("=" * 80)
        print("\nNo profile.md found in the current directory.")
        print("\nTo create your profile, please:")
        print("1. Create a file named 'profile.md' in the current directory")
        print("2. Add information about yourself, your experience, and goals")
        print("3. Restart DXTR to continue\n")
        return None

    profile_content = profile_path.read_text()

    # Check profile length
    estimated_tokens = len(profile_content) // 4
    max_input_tokens = 4096 * 4

    if estimated_tokens > max_input_tokens:
        print(f"\n⚠️  WARNING: Profile is approximately {estimated_tokens} tokens")
        print(f"   This exceeds the recommended maximum of {max_input_tokens} tokens\n")

    print(f"Profile size: {len(profile_content)} characters")

    # Extract and explore links using submodules
    link_summaries = _extract_and_explore_links(profile_content)

    # Build enriched context with profile + link summaries
    enriched_context = profile_content
    if link_summaries:
        enriched_context += f"\n\n# Link Exploration Results\n\n{link_summaries}"

    # Step 1: Generate clarification questions
    print("\n" + "=" * 80)
    print("STEP 1: GENERATING CLARIFICATION QUESTIONS")
    print("=" * 80 + "\n")

    print("Generating questions based on profile and explored links...\n")

    questions_output, _ = _chat_with_tools(
        messages=[{"role": "user", "content": enriched_context}],
        tools=[],  # No tools needed, submodules already explored links
        stream_output=True
    )

    print("\n")

    # Parse questions
    questions = []
    for line in questions_output.strip().split("\n"):
        line = line.strip()
        if line and line[0].isdigit():
            # Remove number prefix (e.g., "1. ", "1) ", "1.")
            # Use regex to remove only the leading number and separator
            question = re.sub(r'^\d+[\.\)]\s*', '', line)
            # Strip any quotes that might have been added
            question = question.strip('"\'')
            # Only accept if it looks like a question (ends with ? and has substance)
            if question and question.endswith("?") and len(question) > 10:
                questions.append(question)

    if not questions:
        print("⚠️  No valid questions found in LLM output.")
        print("    The LLM should output numbered questions ending with '?'")
        print("    Proceeding without Q&A.\n")
        qa_pairs = []
    else:
        # Step 2: Collect answers
        print("\n" + "=" * 80)
        print("STEP 2: ANSWER CLARIFICATION QUESTIONS")
        print("=" * 80 + "\n")
        print("Please answer the following questions (press Enter to skip).\n")

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

    # Build user message with original profile, link summaries, and Q&A
    qa_section = ""
    if qa_pairs:
        qa_section = "\n\nAdditional clarifications:\n"
        for qa in qa_pairs:
            qa_section += f"\nQ: {qa['question']}\nA: {qa['answer']}\n"

    link_section = ""
    if link_summaries:
        link_section = f"\n\nLink exploration results:\n{link_summaries}"

    user_message = f"""Original profile:
{profile_content}
{link_section}
{qa_section}"""

    print("Generating enriched profile...\n")
    print("-" * 80)

    enriched_output, response_obj = _chat_with_tools(
        messages=[{"role": "user", "content": user_message}],
        tools=[],  # No tools needed, submodules already explored links
        stream_output=True
    )

    print("\n" + "-" * 80)

    # Get token counts
    if response_obj:
        prompt_tokens = getattr(response_obj, "prompt_eval_count", 0)
        completion_tokens = getattr(response_obj, "eval_count", 0)
        print(f"\nInput tokens: {prompt_tokens}")
        print(f"Output tokens: {completion_tokens}")

    # Save enriched profile
    dxtr_dir = Path(".dxtr")
    dxtr_dir.mkdir(exist_ok=True)

    output_path = dxtr_dir / "profile_enriched.md"
    output_path.write_text(enriched_output)

    print(f"\nEnriched profile saved to {output_path}")

    return str(output_path)
