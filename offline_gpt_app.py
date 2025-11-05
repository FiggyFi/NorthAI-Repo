# offline_gpt_app.py

"""
Application entry point for retrieval and profile management.

Initialises environment, loads profiles, sets up connectors,
and asserts prerequisites before serving the app.
"""
# Language features
from typing import Optional, Tuple

# Local modules
from profiles import (
    load_profile, save_profile, list_profiles,
    load_sessions, save_sessions
)
from vector_store import (
    COLL_LOCAL,
    COLL_WEB,
    ingest_uploaded,
    add_documents,
    get_or_create_collection, 
    retrieve_local_then_web,
)

from retrieval.router import search_everything
from retrieval.privacy import privacy_expand_query, make_outbound_filter_fn
from retrieval.common import AppConfig, RetrievalManager
from retrieval.common import set_airplane_mode, is_airplane_mode
from retrieval.connectors import fetch_page_text

from app_bootstrap import assert_prereqs_or_raise, snapshot_items

# Standard 
import os
import sys
import json
import time
import uuid
import re
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse

# Third-party packages
from dotenv import load_dotenv
ROOT = Path(__file__).resolve().parent
load_dotenv(dotenv_path=ROOT / ".env")

import streamlit as st

try:
    import ollama  
except Exception:
    ollama = None

from vector_store import get_embedder 
from vector_store import delete_collection_for_session

# Fail fast on startup
assert_prereqs_or_raise()

# Create the config object once for the whole app
config = AppConfig()

import streamlit as st
from vector_store import get_session_id

#Constants
APP_TITLE = "NorthAI"
MODEL_GEN = "llama3:8b"
MAX_CHAT_TURNS = 20
DEFAULT_PROMPT = "How can I help you today?"

@st.cache_resource
def preload_models():
    """Load all expensive models and resources once."""
    # 1. Load the local embedding model
    get_embedder()
    
    # 2. Trigger Ollama to load the LLM into memory
    if ollama:
        try:
            # A tiny, fast call to make Ollama load the model
            ollama.chat(
                model=MODEL_GEN, 
                messages=[{'role': 'user', 'content': '.'}], 
                options={'num_predict': 1} # Tell it to generate only one token
            )
        except Exception as e:
            # Show a warning if Ollama isn't running but don't crash
            st.warning(f"Could not pre-load Ollama model '{MODEL_GEN}'. Please ensure Ollama is running. Error: {e}")
    return True

# Display a spinner while pre-loading
with st.spinner("Warming up the AI engines... This may take a moment."):
    preload_models()


# Retrieval settings
NUM_CTX_KEEP   = 16 # keep best 16 chunks after embeddings ranking

# Year parsing
def parse_year_filters(q: Optional[str]) -> Tuple[Optional[int], Optional[int]]:
    """Return (year_min, year_max) parsed from q. Safe if q is None/empty."""
    if not q:
        return (None, None)

    ql = q.lower().strip()

    # 2019-2021 / 2019–2021 / 2019 to 2021
    m = re.search(r'(19|20)\d{2}\s*[-–to]+\s*(19|20)\d{2}', ql)
    if m:
        years = re.findall(r'(19|20)\d{2}', m.group())
        y1, y2 = int(years[0][-4:]), int(years[1][-4:])
        return (min(y1, y2), max(y1, y2))

    # since / after / >=
    m = re.search(r'(since|after|>=)\s*(19|20)\d{2}', ql)
    if m:
        y = int(re.search(r'(19|20)\d{2}', m.group()).group())
        return (y + (1 if 'after' in m.group() else 0), None)

    # before / <=
    m = re.search(r'(before|<=)\s*(19|20)\d{2}', ql)
    if m:
        y = int(re.search(r'(19|20)\d{2}', m.group()).group())
        return (None, y)

    # in 2023 / lone year
    m = re.search(r'(in\s*)?((19|20)\d{2})', ql)
    if m:
        y = int(re.search(r'(19|20)\d{2}', m.group()).group())
        return (y, y)

    return (None, None)

# Generation (drop-in replacement)
def stream_answer(messages: list[dict], context: str | None):
    if ollama is None:
        yield "**Error:** Ollama Python client not installed. Run `pip install ollama` in your venv."
        return

    # Mode-aware system instructions
    base = (
        "You are a helpful, conversational assistant. "
        "You MUST ground your answer *only* in the provided CONTEXT. "
        "You MUST cite *every* factual claim using the bracketed tags (e.g., [L1], [W2]) found in the CONTEXT. "
        "Use Canadian English spelling. "
    )
    use_web = not is_airplane_mode()
    
    # Stricter rules to prevent hallucination
    if use_web:
        mode = (
            "Ground your answer PRIMARILY in LOCAL (L#) context. "
            "Use WEB (W#) to supplement. "
            "NEVER invent information. "
            "NEVER invent citations. "
            "If the CONTEXT does not contain the answer, state that the provided documents do not have the information. "
            "DO NOT add a 'References' section at the end; that is handled separately."
        )
    else:
        mode = (
            "You are in offline mode. "
            "Use only LOCAL (L#) context. "
            "NEVER invent information or citations. "
            "If the CONTEXT is missing the answer, say so. "
            "DO NOT add a 'References' section at the end; that is handled separately."
        )

    # Depth directive (adds structure/length, no loss of prior context)
    depth = st.session_state.get("answer_depth", "Comprehensive")
    if depth == "Concise":
        depth_directive = (
            "Write a crisp answer with 5–8 bullet points and a short paragraph. "
            "Always include citations [L#/W#] after each claim."
        )
        min_words = 180
    elif depth == "Standard":
        depth_directive = (
            "Write a structured answer with headings and paragraphs. "
            "Include a short summary, key findings, and next steps. "
            "Cite [L#/W#] after each factual claim; include brief quotes when useful."
        )
        min_words = 400
    else:  # Comprehensive
        depth_directive = (
            "Write a thorough report with these sections: "
            "1) Executive Summary, 2) Detailed Findings (grouped by theme), "
            "3) Evidence with short quoted excerpts (with [L#/W#] after each), "
            "4) Data/Calculations or Tables when present, "
            "5) Uncertainties & Gaps, 6) Actionable Next Steps. "
            "Prefer LOCAL sources; include citations after each paragraph."
        )
        min_words = 900

    sys = base + mode + " " + depth_directive
    enriched = [{"role": "system", "content": sys}]

    pf = st.session_state.profile
    facts = pf.get("facts", [])
    facts_text = "\n".join(f"- {f.get('text','')}" for f in facts[:10])
    if facts_text.strip():
        enriched.append({"role": "system", "content": "Remember these about the user:\n" + facts_text})
    if pf.get("identity"):
        enriched.append({"role": "system", "content": "User identity:\n" + "\n".join(f"- {k}: {v}" for k, v in pf["identity"].items())})
    if pf.get("preferences"):
        enriched.append({"role": "system", "content": "User preferences:\n" + "\n".join(f"- {k}: {v}" for k, v in pf["preferences"].items())})
    if st.session_state.get("chat_summary"):
        enriched.append({"role": "system", "content": f"Conversation summary:\n{st.session_state.chat_summary}"})
    if context:
        enriched.append({"role": "user", "content": f"Context:\n{context}"})
    enriched.extend(messages)

    # Model options: allow long, fuller answers
    opts = {
        "num_predict": int(st.session_state.get("answer_max_tokens", 2048)),
        "temperature": 0.3,
    }

    # Stricter nudge to reinforce the rules
    user_tail = (
        f"\n\nWrite at least {min_words} words (as needed by content). "
        f"REMEMBER: Base *all* statements on the CONTEXT and cite [L#/W#] after each claim. "
        f"Do NOT answer from your own knowledge. Do NOT invent links or references."
    )

    # Stream with tail appended to last message
    last = enriched[-1]
    mutated_last = {**last, "content": last["content"] + user_tail} if "content" in last else last
    stream = ollama.chat(model=MODEL_GEN, stream=True, messages=[*enriched[:-1], mutated_last], options=opts)

    for chunk in stream:
        if not chunk.get("done"):
            yield chunk["message"]["content"]


def decide_if_search_is_needed(query: str) -> bool:
    """
    Asks the LLM if it needs to use the 'search' tool to answer.
    This is the "Stage 1" decision-making.
    """
    if ollama is None: return True # Safe fallback

    # Changed to a "tool use" JSON format, which is more robust.
    system_prompt = f"""You are a routing assistant. Your task is to decide if a query can be answered from your own knowledge or if it requires searching external files or the internet.

Respond *only* with a JSON object in the format: {{"tool": "reason"}}
The "tool" must be one of: "search" or "answer".

- "search": Use this tool if the query requires *any* external information, file-based knowledge, or real-time data.
  (e.g., "what is the weather", "find papers on...", "summarize my sales manual", "what is NorthAI?", "who won the game?")

- "answer": Use this tool *only* if the query is a general knowledge question, a creative task, or a math problem.
  (e.g., "draft an email to my boss", "write a python script for fibonacci", "what is 5 + 5?", "tell me a joke")

Query: "{query}"
"""
    try:
        resp = ollama.chat(
            model=MODEL_GEN,
            messages=[{'role': 'user', 'content': system_prompt}],
            options={'temperature': 0.0, 'num_predict': 50}, # Give it more tokens for JSON
            stream=False
        )
        
        # Make parsing more robust. Look for the *word* "search" in the JSON or text.
        content = resp['message']['content'].strip()
        
        # Find the JSON part
        json_match = re.search(r'\{.*\}', content, re.DOTALL)
        if not json_match:
            # Fallback: if no JSON, check the raw string (like before)
            if 'search' in content.lower():
                 print("[App] Decision (fallback): Search is needed.")
                 return True
            else:
                 print("[App] Decision (fallback): No search needed.")
                 return False

        # Try to parse the JSON
        try:
            data = json.loads(json_match.group(0))
            decision = data.get("tool", "").strip().lower()
            
            if decision == 'search':
                print("[App] Decision (JSON): Search is needed.")
                return True
            
            print("[App] Decision (JSON): No search needed.")
            return False
        except json.JSONDecodeError:
            # JSON was malformed, fall back to string check
            if 'search' in content.lower():
                 print("[App] Decision (JSON error): Search is needed.")
                 return True
            else:
                 print("[App] Decision (JSON error): No search needed.")
                 return False

    except Exception as e:
        print(f"[App] Search decision failed, defaulting to search: {e}")
        return True # Safe fallback


def extract_memories_from_turn(last_user: str, last_assistant: str) -> list[str]:
    if ollama is None:
        return []
    sys = (
    "Extract up to 3 general, reusable facts/preferences about the user. "
    "Skip one-time events (like specific dates or queries). "
    "Focus on recurring interests, tone, or constraints."
    )
    prompt = f"User said:\n{last_user}\n\nAssistant replied:\n{last_assistant}\n\nFacts:"
    out = ollama.chat(model=MODEL_GEN, messages=[{"role": "system", "content": sys},{"role":"user","content":prompt}], stream=False)
    text = out["message"]["content"].strip()
    if "NONE" in text.upper():
        return []
    bullets = [b.strip("- ").strip() for b in text.splitlines() if b.strip()]
    return [b for b in bullets if 5 < len(b) < 120 and not b.lower().startswith("on ")]


def maybe_update_summary():
    """Keep summary AND last few verbatim turns, instead of truncating fully."""
    K = st.session_state.max_chat_turns
    msgs = st.session_state.messages
    if len(msgs) <= K:
        return

    # 1. Get all messages to be summarized (everything *except* recent)
    old_messages, recent = msgs[:-K], msgs[-K:]

    # 2. Check if there's *already* a summary message at the start.
    existing_summary_content = ""
    if old_messages and old_messages[0]['role'] == 'system':
        existing_summary_content = old_messages[0]['content'].replace("Summary so far:\n", "").strip()
        old_messages = old_messages[1:] # Get just the user/assistant turns

    # 3. Create a transcript of the *new* old messages.
    transcript = "\n".join(
        f"{m['role'].upper()}: {m['content']}" for m in old_messages
    )

    if not transcript.strip():
        # Nothing new to summarize, just keep the existing summary + recent
        if existing_summary_content:
             st.session_state.messages = [{"role": "system", "content": f"Summary so far:\n{existing_summary_content}"}] + recent
        else:
             st.session_state.messages = recent
        return

    # 4. Summarize the *new* transcript.
    if ollama:
        try:
            sys = ("Summarize earlier conversation. "
                   "Keep durable facts, tone of conversation, and goals.")
            resp = ollama.chat(
                model=MODEL_GEN,
                messages=[{"role": "system", "content": sys},
                          {"role": "user", "content": "Summarize:\n\n" + transcript}],
                stream=False
            )
            new_piece = resp["message"]["content"].strip()
        except Exception as e:
            print(f"[App] Failed to summarize chat: {e}")
            st.session_state.messages = recent # Failsafe: drop old context
            return
    else:
        st.session_state.messages = recent # Failsafe: drop old context
        return

    # 5. Combine new summary with old summary.
    full_summary = (existing_summary_content + "\n\n" + new_piece).strip()

    # 6. Update the UI expander
    st.session_state.chat_summary = full_summary

    # 7. Update the message list with *only* the new combined summary + recent turns
    st.session_state.messages = [{"role": "system", "content": f"Summary so far:\n{full_summary}"}] + recent


# Dialog helpers (centered popouts)
Dialog = st.dialog if hasattr(st, "dialog") else st.experimental_dialog

@Dialog("Manage Memory")
def show_memory_dialog():
    facts = st.session_state.memory.get("facts", []) if "memory" in st.session_state else []
    if not facts:
        st.info("No saved facts yet.")
        return

    for f in facts:
        fid  = f.get("id")
        text = f.get("text", "")
        c1, c2 = st.columns([0.9, 0.1])
        c1.markdown(f"- {text}")
        if c2.button("🗑️", key=f"dlg_del_{fid}"):
            st.session_state.memory["facts"] = [x for x in facts if x.get("id") != fid]
            save_profile(st.session_state.memory)
            st.rerun()

    st.markdown("---")
    if st.button("🧹 Delete ALL memory", key="dlg_mem_delete_all"):
        st.session_state.memory = {"version": 1, "profile": {}, "facts": []}
        save_profile(st.session_state.memory)
        st.rerun()

@Dialog("User Profile")
def show_profile_dialog():
    profiles = list_profiles()

    # Select existing
    if profiles:
        current_id = st.session_state.profile.get("user_id", "default")
        default_idx = profiles.index(current_id) if current_id in profiles else 0
        sel = st.selectbox("Select profile", profiles, index=default_idx, key="dlg_select_profile")
        if st.button("Load selected profile", key="dlg_btn_load_profile"):
            st.session_state.profile = load_profile(user_id=sel)
            st.session_state.memory  = st.session_state.profile  # tie memory to this profile file
            st.success(f"Loaded profile: {sel}")
            st.rerun()
    else:
        st.info("No profiles yet. Create one below.")

    st.markdown("---")
    # Create new
    new_id = st.text_input("New profile name", key="dlg_new_profile_name")
    uploaded_pic = st.file_uploader("Profile picture (optional)", type=["png","jpg","jpeg"], key="dlg_prof_pic")
    if st.button("Create profile", key="dlg_btn_create_profile") and new_id.strip():
        prof = load_profile(user_id=new_id.strip())
        # store avatar as base64
        if uploaded_pic is not None:
            import base64
            prof["avatar_b64"] = base64.b64encode(uploaded_pic.read()).decode("utf-8")
        st.session_state.profile = prof
        st.session_state.memory  = prof
        save_profile(prof)
        st.success(f"Created profile: {new_id.strip()}")
        st.rerun()

# UI helpers
_CIRCLED = {
    **{chr(ord('A')+i): chr(0x24B6+i) for i in range(26)},  # Ⓐ..Ⓩ
    **{chr(ord('a')+i): chr(0x24D0+i) for i in range(26)}   # ⓐ..ⓩ
}
def circled_initial(name: str) -> str:
    ch = (name or "G")[0] 
    return _CIRCLED.get(ch, ch.upper())
def display_name_from_profile(p: dict) -> str:
    return (
        p.get("identity", {}).get("name")
        or p.get("user_id")
        or "Guest"
    )


st.set_page_config(page_title=APP_TITLE, layout="wide")


st.title("NorthAI")


# Chat session utilities
def init_sessions():
    """
    Load all sessions from disk on first run.
    If no sessions exist, create a new one.
    """
    # 1. Load from disk only on the *very first run*
    if "sessions" not in st.session_state:
        st.session_state.sessions = load_sessions()

    # 2. If no sessions were loaded (e.g., empty file), create one
    if not st.session_state.sessions:
        create_new_session(save=False) # Create without saving
        save_sessions(st.session_state.sessions) # Save the initial state

    # 3. Ensure a current_session is selected
    if "current_session" not in st.session_state:
        # Default to the most recently updated session
        try:
            sorted_sessions = sorted(
                st.session_state.sessions.items(),
                key=lambda item: item[1].get("updated_at", "1970-01-01"),
                reverse=True
            )
            st.session_state.current_session = sorted_sessions[0][0]
        except IndexError:
            # This should not happen, but as a fallback:
            create_new_session() # This will save

    # 4. Always sync the message list to the active session
    sid = st.session_state.current_session
    if sid not in st.session_state.sessions:
        # Fallback if current_session ID is invalid
        st.session_state.current_session = list(st.session_state.sessions.keys())[0]
        sid = st.session_state.current_session
        
    st.session_state.messages = st.session_state.sessions[sid]["messages"]

def create_new_session(title: str = "Untitled", save: bool = True):
    """Create a new session with a random ID and save to disk."""
    session_id = f"chat-{uuid.uuid4().hex[:6]}"
    st.session_state.sessions[session_id] = {
        "title": title,
        "messages": [],
        "created_at": datetime.now().isoformat(),
        "updated_at": datetime.now().isoformat(),
        "memory": {}
    }
    st.session_state.current_session = session_id
    
    if save:
        save_sessions(st.session_state.sessions)
        
    return session_id
def switch_session(session_id: str):
    """Switch to another existing session."""
    if session_id in st.session_state.sessions:
        st.session_state.current_session = session_id
        st.session_state.messages = st.session_state.sessions[session_id]["messages"]

def delete_session(session_id: str):
    """Delete a chat session safely, export its cache, and save the state."""
    if session_id in st.session_state.sessions:
        try:
            manager = RetrievalManager(config=config, session_id=session_id)
            manager.export_session_cache_encrypted() # Or export_session_cache()
            print(f"[App] Exported cache for deleted session {session_id}")
        except Exception as e:
            print(f"[App] Could not export cache for {session_id}: {e}")
        
        # Remove from Streamlit state
        del st.session_state.sessions[session_id]

        # Clean up associated ChromaDB data
        try:
            delete_collection_for_session(session_id)
        except Exception as e:
            print(f"[delete_session] Warning: could not delete Chroma data for {session_id}: {e}")

        # Handle fallback (keep at least one session alive)
        if not st.session_state.sessions:
            create_new_session(save=False) # Create one, but don't save yet
        else:
            # Pick most recently created session as active
            last_id = sorted(
                st.session_state.sessions.items(),
                key=lambda x: x[1]["created_at"],
                reverse=True
            )[0][0]
            st.session_state.current_session = last_id
        
        # Save the new state (with the session removed)
        save_sessions(st.session_state.sessions)


# Bootstrap sessions on app start
init_sessions()

# Other state values
if "last_sources" not in st.session_state:
    st.session_state.last_sources = []
if "profile" not in st.session_state:
    st.session_state.profile = load_profile(user_id="default")
if "chat_summary" not in st.session_state:
    st.session_state.chat_summary = ""
if "max_chat_turns" not in st.session_state:
    st.session_state.max_chat_turns = MAX_CHAT_TURNS
if "memory" not in st.session_state:
    st.session_state.memory = load_profile(user_id=st.session_state.profile.get("user_id", "default"))

# Depth + token controls defaults
if "answer_depth" not in st.session_state:
    st.session_state.answer_depth = "Comprehensive"
if "answer_max_tokens" not in st.session_state:
    st.session_state.answer_max_tokens = 2048


# Top bar controls 
col1, col2 = st.columns([0.85, 0.15])
with col2:
    st.toggle("✈️ Airplane Mode", key="airplane_mode", help="Offline only: disables web search.")

# Immediately sync runtime airplane mode for connectors/common
set_airplane_mode(bool(st.session_state.get("airplane_mode", False)))

# Sidebar
with st.sidebar:
    # Profile / memory menu 
    prof = st.session_state.profile
    disp_name = display_name_from_profile(prof)
    label = f"{circled_initial(disp_name)}  {disp_name}"

    profile_menu = st.popover(label) if hasattr(st, "popover") else st.expander(label)
    with profile_menu:
        st.markdown("**Profile menu**")
        if st.button("👤 User profile"):
            show_profile_dialog()
        if st.button("🧠 Manage memory"):
            show_memory_dialog()

    
    
    
    st.markdown("---")

    st.subheader("Answer style")
    st.session_state.answer_depth = st.selectbox(
        "Depth / style",
        ["Concise", "Standard", "Comprehensive", "Full Summary (Slow)"],
        index=2,
        help="Controls how detailed the assistant must be."
    )

    st.session_state.answer_max_tokens = st.slider(
        "Max tokens to generate",
        min_value=256, max_value=4096, value=st.session_state.answer_max_tokens, step=256,
        help="Controls how detailed the assistant must be. Concise, Standard, and Comprehensive modes retrieve the most relevant text chunks, providing progressively detailed answers based on that limited context. Full Summary will read all local files and summarize each one individually. This will provide the most complete synthesis, but will run significantly slow. It is advised to use Comprehensive for specific questions, queries, or requests, and Full Summary for broad, in-depth analysis for a first review of uploaded materials."
    )

    # Year filter
    st.subheader("📅 Year filter")

    st.session_state.enable_years = st.checkbox("Enable year filter", value=False)

    if st.session_state.enable_years:
        cyr1, cyr2 = st.columns(2)
        with cyr1:
            st.session_state.year_min = st.number_input(
                "From", min_value=1900, max_value=2100, value=2024, step=1
            )
        with cyr2:
            st.session_state.year_max = st.number_input(
                "To", min_value=1900, max_value=2100, value=2025, step=1
            )
    st.markdown("---")

# Local files (offline RAG)
    st.subheader("Local files (offline RAG)")
    uploads = st.file_uploader(
        "Add files (max 10)",
        type=[
            "pdf","doc","docx","rtf","txt","md","csv","xlsx","pptx","json",
            "py","js","ts","java","c","cpp","xml","html","yaml","yml",
            "png","jpg","jpeg","gif","tif","tiff"
        ],
        accept_multiple_files=True,
    )
    if uploads:
        if len(uploads) > 10:
            st.warning(f"You selected {len(uploads)} files. Only the first 10 will be indexed.")
        added = ingest_uploaded(uploads[:10])
        st.success(f"Indexed {added} file(s) into the local vector store.")

    st.markdown("---")

    # Agents (coming soon)
    st.caption("Agents (coming soon)")
    st.selectbox("Agent", ["General (default)", "Computer Science", "Biology"], index=0, disabled=True)
    
    st.markdown("---")

    # Chats
    st.subheader("Chats")

    # Add new chat button
    if st.button("➕ New Chat", use_container_width=True):
        create_new_session(title="Untitled")

    # SAFE snapshot to avoid "dictionary changed size during iteration"
    to_delete: list[str] = []
    snapshot = list(st.session_state.sessions.items())

    for sid, chat in snapshot:
        title = chat.get("title", "Untitled Chat")
        with st.container():
            cols = st.columns([0.7, 0.15, 0.15])

            with cols[0]:
                if st.button(title, key=f"chat_{sid}", use_container_width=True):
                    switch_session(sid)

            with cols[1]:
                if st.button("✏️", key=f"rename_btn_{sid}"):
                    st.session_state[f"renaming_{sid}"] = True

            with cols[2]:
                if st.button("🗑️", key=f"delete_{sid}"):
                    to_delete.append(sid)

            if st.session_state.get(f"renaming_{sid}", False):
                new_title = st.text_input("New name", value=title, key=f"rename_input_{sid}")
                if st.button("Save", key=f"rename_save_{sid}"):
                    st.session_state.sessions[sid]["title"] = new_title.strip() or "Untitled Chat"
                    st.session_state[f"renaming_{sid}"] = False
                    save_sessions(st.session_state.sessions)
                    st.rerun()

# Apply deletes after the loop
for sid in to_delete:
    delete_session(sid)
st.rerun() if to_delete else None


# Conversation history - show earlier conversation (summary) so users can see past context we folded away
if st.session_state.get("chat_summary"):
    with st.expander("Earlier conversation (summary)"):
        st.markdown(st.session_state.chat_summary)

# Show recent verbatim messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Memory helpers
def has_fact_text(profile: dict, text: str) -> bool:
    """Check if a fact with this text already exists in memory."""
    for f in profile.get("facts", []):
        if f.get("text", "").strip().lower() == text.strip().lower():
            return True
    return False

def add_fact(profile: dict, text: str, strength: float = 0.7, kind: str = "fact"):
    """Append a new fact to the profile's memory."""
    profile.setdefault("facts", [])
    profile["facts"].append({
        "id": str(uuid.uuid4()),
        "text": text.strip(),
        "kind": kind,
        "strength": float(strength),
        "last_seen": int(time.time()),
        "ttl_days": None,
    })

def prune_memory(profile: dict, keep: int = 50):
    """Keep the strongest/most recent facts up to `keep`."""
    facts = profile.get("facts", [])
    if not facts:
        return
    facts.sort(key=lambda f: (f.get("strength", 0), f.get("last_seen", 0)), reverse=True)
    profile["facts"] = facts[:keep]

# Reference builder
def build_references_md(response_text: str, pairs: list[tuple[str, str, str]]) -> str:
    """
    Render two blocks:
    - Offline sources (local uploads): [L#] entries with filenames
    - Online sources (web): [W#] entries with host links
    """
    if not pairs:
        return ""
    cited = set()
    # capture tags like [L1], [W2], [1] (ignore plain numbers)
    for m in re.findall(r"\[([LW]\d+)\]", response_text):
        cited.add(m.upper())

    offline, online = [], []
    for _, src, tag in pairs:
        tag = tag.upper()
        if tag.startswith("L"):
            name = (src or "").replace("upload://", "") or "local-file"
            if not cited or tag in cited:
                offline.append(f"[{tag}] {name}")
        elif tag.startswith("W"):
            if (not cited) or (tag in cited):
                host = urlparse(src).netloc or src
                online.append(f"[{tag}] [{host}]({src})")

    blocks = []
    if offline: blocks.append("**Offline sources (local uploads):**\n" + "\n".join(offline))
    if online:  blocks.append("**Online sources (web):**\n" + "\n".join(online))
    return "\n\n".join(blocks)

def build_references_by_tags(response_text: str, pairs: list[tuple[str, str, str]]) -> str:
    """
    Build reference lists from pairs = [(doc, src, tag), ...] where tag is 'L1' or 'W2'.
    - Only include sources whose tags were actually cited in the model output.
    - Collapse multiple chunks from the same source to a single line (prefer the lowest tag number).
    - Separate Offline (L#) and Online (W#).
    """
    if not pairs or not response_text:
        return ""

    # Which L#/W# tags were cited in the answer?
    # Matches [L1], [W2], [L3] etc (allow optional whitespace)
    cited_tags = set(m.group(0)[1:-1].replace(" ", "")
                     for m in re.finditer(r"\[(?:L|W)\s*\d+\]", response_text))

    # If nothing was cited, don't show anything
    if not cited_tags:
        return ""

    # Collapse duplicates by source, keep the *lowest* tag for that source
    # We'll maintain insertion order with a dict
    by_src: dict[str, str] = {}
    for _doc, src, tag in pairs:
        tag = tag.replace(" ", "")
        if tag not in cited_tags:
            continue
        # keep first (lowest number) tag we encounter per source
        if src not in by_src:
            by_src[src] = tag
        else:
            # choose lower numeric index for stability
            old = by_src[src]
            if old[0] == tag[0]:  # same L/W
                try:
                    if int(tag[1:]) < int(old[1:]):
                        by_src[src] = tag
                except Exception:
                    pass

    # Now split offline vs online and render
    offline_lines, online_lines = [], []
    for src, tag in by_src.items():
        if src and src.startswith("http"):
            host = urlparse(src).netloc or src
            online_lines.append(f"[{tag}] [{host}]({src})")
        else:
            name = (src or "").replace("upload://", "") or "local-file"
            offline_lines.append(f"[{tag}] {name}")

    blocks = []
    if offline_lines:
        blocks.append("**Offline sources (local uploads):**\n" + "\n".join(sorted(offline_lines, key=lambda s: (s[1], int(re.search(r'\d+', s).group())))))
    if online_lines:
        blocks.append("**Online sources (web):**\n" + "\n".join(sorted(online_lines, key=lambda s: (s[1], int(re.search(r'\d+', s).group())))))

    return "\n\n".join(blocks)

# Chat turn 
if prompt := st.chat_input("Message"):
    # Record user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    maybe_update_summary()
    with st.chat_message("user"):
        st.markdown(prompt)

    # --- Generate title only for the first message of a new chat ---
    use_web = not is_airplane_mode()
    cur = st.session_state.current_session
    if (
        not st.session_state.sessions[cur].get("title")
        or st.session_state.sessions[cur]["title"].startswith("Untitled")
    ) and st.session_state.messages:
        first_msg = st.session_state.messages[0]["content"].strip()
        try:
            resp = ollama.chat(
                model=MODEL_GEN,
                messages=[{
                    "role": "system",
                    "content": (
                        "Generate a concise descriptive title (2–5 words, max 30 characters) "
                        "based only on the following message. Do NOT use quotation marks."
                    )
                }, {"role": "user", "content": first_msg}],
                stream=False
            )
            topic = resp["message"]["content"].strip()
        except Exception:
            topic = first_msg[:30] + ("..." if len(first_msg) > 30 else "")
        st.session_state.sessions[cur]["title"] = topic
    # Create Manager at the START of the turn
    try: 
        my_session_id = st.session_state.current_session
        profile = st.session_state.profile
        my_username = profile.get("user_id", "default")
        my_chat_name = st.session_state.sessions[my_session_id].get("title", "Untitled")

        metadata = {
            "username": my_username,
            "chat_name": my_chat_name,
            "client_timezone": "EST"
        }
        manager = RetrievalManager(
            config=config,
            session_id=my_session_id,
            session_metadata=metadata
        )
    except Exception as e:
        st.error(f"Failed to create session manager: {e}")
        # Stop processing if the manager can't be created
        st.stop()

# 1. Retrieval Phase
    context, pairs = "", []
    # Query the LLM to determine if it needs to search.
    needs_search = decide_if_search_is_needed(prompt)
    
    if needs_search:
        is_full_summary = st.session_state.answer_depth == "Full Summary (Slow)"

        if is_full_summary:
            with st.spinner("Starting full summary... This may take several minutes."):
                try:
                    # 1. Get all unique file sources from the vector store
                    lcoll = get_or_create_collection(COLL_LOCAL)
                    all_docs = lcoll.get(include=["metadatas"])
                    all_sources = sorted(list(set(
                        md['source'] for md in all_docs['metadatas'] if md.get('source')
                    )))
                    
                    if not all_sources:
                        st.warning("No local files found to summarize.")
                        st.stop()

                    file_summaries = []
                    all_pairs = []
                    
                    # 2. Mapping Step: Loop over each file and summarize it
                    for source_name in all_sources:
                        st.status(f"Summarizing {source_name}...")
                        
                        # a. Create a generic prompt to get key chunks for summarization
                        summary_prompt = f"What are the main topics, key points, and conclusions of the document '{source_name}'?"
                        
                        # b. Retrieve chunks *only* from that file
                        file_ctx, file_pairs = retrieve_local_then_web(
                            summary_prompt,
                            top_k_local=NUM_CTX_KEEP, # Use 16 chunks
                            top_k_web=0,
                            source_filter=source_name 
                        )
                        
                        if not file_ctx:
                            continue # Skip empty files or files with no matching chunks

                        # c. Call Ollama to summarize *just those chunks*
                        summary_resp = ollama.chat(
                            model=MODEL_GEN,
                            messages=[
                                {"role": "system", "content": "You are a summarizing assistant. Summarize the key points of the following context. Do not add any preamble or your own opinions. Be factual and concise."},
                                {"role": "user", "content": f"Context:\n{file_ctx}\n\nSummary of {source_name}:"}
                            ],
                            stream=False
                        )
                        file_summary = summary_resp['message']['content']
                        file_summaries.append(f"Summary for {source_name}:\n{file_summary}")
                        all_pairs.extend(file_pairs) # Collect all pairs for citation

                    # 3. "Reduce" Step: Combine all summaries into the final context
                    if not file_summaries:
                        context = "No relevant information found in local files to summarize."
                    else:
                        context = "\n\n---\n\n".join(file_summaries)
                    pairs = all_pairs

                except Exception as e:
                    st.error(f"Error during full summary: {e}")
                    st.stop()
        
        else:
            # Standard RAG
            ctx_local, pairs_local = retrieve_local_then_web(
                prompt,
                top_k_local=NUM_CTX_KEEP,
                top_k_web=0,
            )
            context = ctx_local
            pairs = list(pairs_local)

            # Optionally, retrieve from the web if not in airplane mode
            if use_web:
                with st.spinner("Searching web & enriching context..."):
                    ymin_p, ymax_p = parse_year_filters(prompt)
                    year_min, year_max = ymin_p, ymax_p
                    if st.session_state.get("enable_years", False):
                        if "year_min" in st.session_state: year_min = int(st.session_state.year_min)
                        if "year_max" in st.session_state: year_max = int(st.session_state.year_max)
                    if year_min is not None and year_max is not None and year_min > year_max:
                        year_min, year_max = year_max, year_min
                    
                    # Stage 2 runs inside the needs_search block
                    safe_query = privacy_expand_query(prompt)
                    try:
                        lcoll = get_or_create_collection(COLL_LOCAL)
                        peek = lcoll.peek(500) or {}
                        local_docs_seed = []
                        for docs in (peek.get("documents") or []):
                            local_docs_seed.extend(docs)
                    except Exception:
                        local_docs_seed = []
                    manager.set_outbound_filter(make_outbound_filter_fn(local_docs_seed))

                    results = search_everything(manager, safe_query, year_min=year_min, year_max=year_max)
                    url_to_chunks = {}
                    for rec in results:
                        body = fetch_page_text(manager, rec["url"]) or ""
                        meta = "\n\n".join(x for x in [rec.get("title"), rec.get("abstract"), rec.get("url")] if x)
                        text = (body or meta).strip()
                        if text:
                            url_to_chunks.setdefault(rec["url"], []).append(text)
                    if url_to_chunks:
                        add_documents(url_to_chunks, target=COLL_WEB)

                    # Retrieve web supplement and append to context
                    ctx_mixed, pairs_mixed = retrieve_local_then_web(
                        prompt,
                        top_k_local=0,
                        top_k_web=max(1, NUM_CTX_KEEP // 2),
                    )
                    if ctx_mixed:
                        context = (context + ("\n\n---\n\n" if context else "") + ctx_mixed).strip()
                        pairs.extend(pairs_mixed)

    else:
        # The LLM decided it can answer directly (e.g., "draft email", "do math").
        with st.spinner("Thinking..."):
            pass 
        context = "" # Explicitly empty context
        pairs = []


# 2. Generation Phase
# ... (This section remains unchanged) ...
# # 1. Retrieval Phase
#     context, pairs = "", []
#     is_full_summary = st.session_state.answer_depth == "Full Summary (Slow)"

#     if is_full_summary:
#         # NEW: Map-Reduce Summarization Mode ---
#         with st.spinner("Starting full summary... This may take several minutes."):
#             try:
#                 # 1. Get all unique file sources from the vector store
#                 lcoll = get_or_create_collection(COLL_LOCAL)
#                 all_docs = lcoll.get(include=["metadatas"])
#                 all_sources = sorted(list(set(
#                     md['source'] for md in all_docs['metadatas'] if md.get('source')
#                 )))
                
#                 if not all_sources:
#                     st.warning("No local files found to summarize.")
#                     st.stop()

#                 file_summaries = []
#                 all_pairs = []
                
#                 # 2. "Map" Step: Loop over each file and summarize it
#                 for source_name in all_sources:
#                     st.status(f"Summarizing {source_name}...")
                    
#                     # a. Create a generic prompt to get key chunks for summarization
#                     summary_prompt = f"What are the main topics, key points, and conclusions of the document '{source_name}'?"
                    
#                     # b. Retrieve chunks *only* from that file
#                     file_ctx, file_pairs = retrieve_local_then_web(
#                         summary_prompt,
#                         top_k_local=NUM_CTX_KEEP, # Use 16 chunks
#                         top_k_web=0,
#                         source_filter=source_name  # <-- Use our new filter
#                     )
                    
#                     if not file_ctx:
#                         continue # Skip empty files or files with no matching chunks

#                     # c. Call Ollama to summarize *just those chunks*
#                     summary_resp = ollama.chat(
#                         model=MODEL_GEN,
#                         messages=[
#                             {"role": "system", "content": "You are a summarizing assistant. Summarize the key points of the following context. Do not add any preamble or your own opinions. Be factual and concise."},
#                             {"role": "user", "content": f"Context:\n{file_ctx}\n\nSummary of {source_name}:"}
#                         ],
#                         stream=False
#                     )
#                     file_summary = summary_resp['message']['content']
#                     file_summaries.append(f"Summary for {source_name}:\n{file_summary}")
#                     all_pairs.extend(file_pairs) # Collect all pairs for citation

#                 # 3. "Reduce" Step: Combine all summaries into the final context
#                 if not file_summaries:
#                     context = "No relevant information found in local files to summarize."
#                 else:
#                     context = "\n\n---\n\n".join(file_summaries)
#                 pairs = all_pairs

#             except Exception as e:
#                 st.error(f"Error during full summary: {e}")
#                 st.stop()
    
#     else:
#         ctx_local, pairs_local = retrieve_local_then_web(
#             prompt,
#             top_k_local=NUM_CTX_KEEP,
#             top_k_web=0,
#         )
#         context = ctx_local
#         pairs = list(pairs_local)

#         # Optionally, retrieve from the web if not in airplane mode
#         if use_web:
#             with st.spinner("Searching web & enriching context..."):
#                 ymin_p, ymax_p = parse_year_filters(prompt)
#                 year_min, year_max = ymin_p, ymax_p
#                 if st.session_state.get("enable_years", False):
#                     if "year_min" in st.session_state: year_min = int(st.session_state.year_min)
#                     if "year_max" in st.session_state: year_max = int(st.session_state.year_max)
#                 if year_min is not None and year_max is not None and year_min > year_max:
#                     year_min, year_max = year_max, year_min
                
#                 safe_query = privacy_expand_query(prompt)
#                 try:
#                     lcoll = get_or_create_collection(COLL_LOCAL)
#                     peek = lcoll.peek(500) or {}
#                     local_docs_seed = []
#                     for docs in (peek.get("documents") or []):
#                         local_docs_seed.extend(docs)
#                 except Exception:
#                     local_docs_seed = []
#                 manager.set_outbound_filter(make_outbound_filter_fn(local_docs_seed))

#                 results = search_everything(manager, safe_query, year_min=year_min, year_max=year_max)
#                 url_to_chunks = {}
#                 for rec in results:
#                     body = fetch_page_text(manager, rec["url"]) or ""
#                     meta = "\n\n".join(x for x in [rec.get("title"), rec.get("abstract"), rec.get("url")] if x)
#                     text = (body or meta).strip()
#                     if text:
#                         url_to_chunks.setdefault(rec["url"], []).append(text)
#                 if url_to_chunks:
#                     add_documents(url_to_chunks, target=COLL_WEB)

#                 # Retrieve web supplement and append to context
#                 ctx_mixed, pairs_mixed = retrieve_local_then_web(
#                     prompt,
#                     top_k_local=0,
#                     top_k_web=max(1, NUM_CTX_KEEP // 2),
#                 )
#                 if ctx_mixed:
#                     context = (context + ("\n\n---\n\n" if context else "") + ctx_mixed).strip()
#                     pairs.extend(pairs_mixed)


    # 2. Generation Phase
    with st.chat_message("assistant"):
        placeholder = st.empty()
        buf = ""
        try:
            stream = stream_answer(st.session_state.messages, context)
            for part in stream:
                buf += part
                placeholder.markdown(buf)
        except Exception as e:
            st.error(f"An error occurred while generating the response: {e}")
            buf = "Sorry, I ran into an error. Please check the logs."

    answer_text = buf

    # 3. Post-generation Bookkeeping
    if answer_text:
        st.session_state.messages.append({"role": "assistant", "content": answer_text})
        if pairs:
            refs_md = build_references_by_tags(answer_text, pairs)
            if refs_md.strip():
                st.markdown(refs_md)
        maybe_update_summary()
        if len(st.session_state.messages) >= 2:
            last_user = st.session_state.messages[-2]["content"]
            last_assistant = st.session_state.messages[-1]["content"]
            new_facts = extract_memories_from_turn(last_user, last_assistant)
            for fact_text in new_facts:
                if not has_fact_text(st.session_state.memory, fact_text):
                    add_fact(st.session_state.memory, fact_text)
                    prune_memory(st.session_state.memory)
                    save_profile(st.session_state.memory)
        try:
            manager.write_chat_turn(
                turn_number=len(st.session_state.messages) // 2,
                user_query=prompt,
                response_text=answer_text,
                retrieved_ids=[p[1] for p in pairs] 
            )
            
            st.session_state.sessions[cur]["updated_at"] = datetime.now().isoformat()
            save_sessions(st.session_state.sessions)
        except Exception as e:
            print(f"[audit_log] Failed to write audit log: {e}")