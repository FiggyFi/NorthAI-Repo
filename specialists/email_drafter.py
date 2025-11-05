# File: retrieval/email_drafter.py

import os
import datetime
from .common import hash_text

def _sanitize_filename(text: str) -> str:
    return "".join(c if c.isalnum() or c in ('-', '_') else "_" for c in text)[:40]

def generate_email_draft(session_id: str, recipient: str, topic: str, body_prompt: str, model, config, manager):
    """
    Generate and store a local email draft.
    Works entirely offline using the same AI model pipeline.
    """
    # Generate text using the local model (or your retrieval pipeline)
    response = model.generate_email(topic, body_prompt)

    # Construct formatted message
    now = datetime.datetime.utcnow().isoformat()
    subject = f"Re: {topic.strip().capitalize()}"
    draft_text = (
        f"To: {recipient}\n"
        f"Subject: {subject}\n"
        f"Date: {now}\n\n"
        f"{response}\n\n--\nNorth AI Assistant"
    )

    # Save draft locally
    drafts_dir = os.path.join("sessions", session_id, "drafts")
    os.makedirs(drafts_dir, exist_ok=True)
    filename = _sanitize_filename(topic) + ".txt"
    draft_path = os.path.join(drafts_dir, filename)
    with open(draft_path, "w", encoding="utf-8") as f:
        f.write(draft_text)

    # Log the event
    manager.write_audit_log(
        event_type="email_draft",
        data={
            "recipient": recipient,
            "topic": topic,
            "response_hash": hash_text(response),
            "saved_path": draft_path,
        },
    )

    print(f"[email_drafter] Saved draft: {draft_path}")
    return draft_path, draft_text
