"""
proofreader.py

Specialist handler for professional text refinement and copy-editing.

This module orchestrates the proofreading task by loading the
'get_proofreader_prompt' and executing the LLM call.

Multi-turn Support:
This handler supports iterative refinement. Users can make follow-up
requests like "make it more concise" or "sound more technical", and
the LLM will apply those changes to the previous output while maintaining
conversation context.

Future scope:
- Integrate custom style guides.
- Generate diffs of changes.
- Handle multiple document formats.
- Track editing history for rollback capability.
"""

# Use a relative import to get prompts from the file in the same directory
from . import specialist_prompts

def run_proofreading(text_to_proofread: str, llm_call_function) -> str:
    """
    Executes the professional proofreading and copy-editing task.
    
    Supports multi-turn conversations for iterative refinement. When used
    in a conversation context, the LLM will have access to previous edits
    and can apply follow-up instructions (e.g., "make it shorter", "sound
    more technical") to the most recent version.
    
    Args:
        text_to_proofread: The raw text supplied by the user, or a follow-up
                          instruction if this is part of an ongoing conversation.
        llm_call_function: A callable function for interfacing with the LLM.
                           This function must accept (system_prompt, user_text)
                           and handle conversation context internally.
    
    Returns:
        The professionally corrected and refined text as a string.
    
    Examples:
        Initial request: "proofread this: The managment team are meeting tommorow."
        Follow-up: "make it more formal"
        Follow-up: "sound less verbose"
    """
    # 1. Retrieve the specialist prompt from the central library.
    system_prompt = specialist_prompts.get_proofreader_prompt()
    
    # 2. Invoke the LLM with the specified instructions and user text.
    # Note: The llm_call_function handles conversation context automatically
    # when this is a follow-up request in an ongoing session.
    try:
        corrected_text = llm_call_function(system_prompt, text_to_proofread)
        return corrected_text
    except Exception as e:
        print(f"[proofreader] Error during proofreading: {e}")
        return "Sorry, I was unable to process the text as requested."