"""
email_drafter.py

Specialist handler for drafting email communications.

This module orchestrates the email drafting task by loading the
'get_email_drafter_prompt' and executing the LLM call.

Future scope:
- Parse context from email threads.
- Integrate contact/address book.
- Add signature blocks.
"""

# Use a relative import to get prompts from the file in the same directory
from . import specialist_prompts

def run_email_draft(text_to_draft_from: str, llm_call_function) -> str:
    """
    Runs the email drafting task.
    
    Args:
        text_to_draft_from: The user's prompt (e.g., "email my boss about...")
        llm_call_function: A callable function for interfacing with the LLM.
                           This function must accept (system_prompt, user_text).
    
    Returns:
        The drafted email text, formatted with Subject and Body.
    """
    # 1. Retrieve the specialist prompt from the central library.
    system_prompt = specialist_prompts.get_email_drafter_prompt()
    
    # 2. Invoke the LLM with the specified instructions and user text.
    try:
        drafted_email = llm_call_function(system_prompt, text_to_draft_from)
        return drafted_email
    except Exception as e:
        print(f"[email_drafter] Error during email drafting: {e}")
        return "Sorry, I was unable to draft the email."