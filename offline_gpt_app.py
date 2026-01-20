# offline_gpt_app.py

"""
Application entry point for retrieval and profile management.

Initialises environment, loads profiles, sets up connectors,
and asserts prerequisites before serving the app.
"""
# Language features
from typing import Optional, Tuple

# Local modules

# Retrieval Pipeline
from retrieval.retrieval_router import search_everything
from retrieval.privacy import privacy_expand_query, make_outbound_filter_fn
from retrieval.common import AppConfig, RetrievalManager
from retrieval.common import set_offline_mode, is_offline_mode
from retrieval.connectors import fetch_page_text
from dataclasses import dataclass
from retrieval.connectors import search_duckduckgo, wikipedia_search

# Profiles & Session Management
from profiles import (
    load_profile, save_profile, list_profiles,
    load_sessions, save_sessions
)

# Vector Store (Document Management)
from vector_store import (
    COLL_LOCAL,
    COLL_WEB,
    ingest_uploaded,
    add_documents,
    get_or_create_collection, 
    retrieve_local_then_web,
    get_session_db_path,  
    get_embedder,
    delete_collection_for_session,
    get_session_id, log_collection_counts,
)

# DB
import chromadb 
from chromadb.config import Settings  

# App Bootstrap
from app_bootstrap import assert_prereqs_or_raise, snapshot_items

# Task Router and Specialists
from task_router import classify_intent, strip_command_from_text, TaskIntent
from specialists import specialist_prompts
from specialists import proofreader
from specialists import email_drafter

# Standard 
import os
import sys
import json
import time
import uuid
import re
from datetime import datetime, date, timedelta
from pathlib import Path
from urllib.parse import urlparse

# Third-party packages
import streamlit as st
from dotenv import load_dotenv

# Env Set-up
ROOT = Path(__file__).resolve().parent
load_dotenv(dotenv_path=ROOT / ".env")

try:
    import ollama  
except Exception:
    ollama = None

# Fail fast on startup
assert_prereqs_or_raise()

# Create the config object once for the whole app
config = AppConfig()

#Constants
APP_TITLE = "NorthAI"
MODEL_GEN = "llama3:8b"
MAX_CHAT_TURNS = 50
DEFAULT_PROMPT = "How can I help you today?"

# Context window for follow-up requests (number of recent messages to include)
FOLLOWUP_CONTEXT_MESSAGES = 8  # Last 4 exchanges

# Refinement detection thresholds
MAX_FOLLOWUP_WORDS = 20
VERY_SHORT_FOLLOWUP_THRESHOLD = 10  # Very short queries are almost always follow-ups

# Refinement keywords (organized by category for maintainability)
REFINEMENT_KEYWORDS = {
    # Style & tone
    "more professional", "more technical", "more casual", "more formal",
    "sound more", "make it sound", "change the tone",
    # Length
    "more concise", "shorter", "longer", "tighter", "brief",
    "verbose", "condense", "expand", "not verbose", "do not be verbose",
    # Quality
    "better", "clearer", "simpler", "improve", "refine", "polish",
    # Actions
    "fix", "adjust", "revise", "rewrite", "tweak",
}

IMPERATIVE_VERBS = {
    "make", "change", "fix", "adjust", "rewrite", "sound",
    "add", "remove", "use", "avoid", "keep", "drop"
}

# Immediately sync runtime offline mode for connectors/common
set_offline_mode(bool(st.session_state.get("offline_mode", False)))

# Pivot Detection Utilities
def _extract_location(text: str) -> Optional[str]:
    if not text:
        return None
    m = re.search(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\b(?:,?\s*(ON|Ontario|CA|US|UK))?", text)
    return m.group(1).strip() if m else None

def detect_pivot(curr_query: str, last_query: Optional[str]) -> bool:
    """Returns True if query represents a new semantic topic or city."""
    if not last_query:
        return True
    curr_loc = _extract_location(curr_query)
    prev_loc = _extract_location(last_query)
    if curr_loc and prev_loc and curr_loc.lower() != prev_loc.lower():
        print(f"[Pivot] Location change detected: {prev_loc} → {curr_loc}")
        return True
    try:
        emb = get_embedder()
        try:
            e1 = emb.encode(curr_query, convert_to_numpy=True)
            e2 = emb.encode(last_query, convert_to_numpy=True)
            sim = float((e1 @ e2.T) / ((e1**2).sum()**0.5 * (e2**2).sum()**0.5))
        except Exception as inner_e:
            print(f"[Pivot] Fallback similarity failed: {inner_e}")
            return False
        if sim < 0.55:
            print(f"[Pivot] Semantic drift detected (similarity={sim:.2f})")
            return True
    except Exception as e:
        print(f"[Pivot] Similarity check failed ({e}) — ignoring.")
    return False

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

log_collection_counts()

# Display a spinner while pre-loading
with st.spinner("Warming up the AI engines... This may take a moment."):
    preload_models()

STRICT_RAG_SYSTEM_PROMPT = """You are a retrieval-only assistant.
Answer ONLY using the provided CONTEXT.
- Add a citation tag [L#] or [W#] immediately after each factual claim you make.
- If the CONTEXT does not contain the answer, reply exactly with:
  "I don't have enough evidence in the provided sources."
- Be concise. Do not add a 'References' section. Never invent facts or citations.
"""

def _has_sufficient_evidence(context: Optional[str], *_args, **_kwargs) -> bool:
    """
    Permit answering whenever CONTEXT has content.
    Accepts extra args to be call-site compatible (prevents TypeError).
    """
    return bool(context and context.strip())

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

# Generation
def stream_answer(messages: list[dict], context: str | None):
    if ollama is None:
        yield "**Error:** Ollama Python client not installed. Run `pip install ollama` in your venv."
        return

    base = (
        "You are a retrieval-only assistant. "
        "Answer ONLY from the provided CONTEXT. "
        "Add [L#] or [W#] after each factual claim. "
        "If the answer is not in CONTEXT, reply: "
        "\"I don't have enough evidence in the provided sources.\" "
        "Be concise; no 'References' section."
    )
    use_web = not is_offline_mode()
    mode = "You may cite [L#] and [W#]." if use_web else "Cite only [L#]."

    depth = st.session_state.get("answer_depth", "Concise")
    style = "Keep it brief." if depth == "Concise" else "Be structured but concise."

    messages_built = [
        {"role": "system", "content": f"{base}\n{mode}\n{style}"},
    ]
    if context:
        messages_built.append({"role": "user", "content": f"CONTEXT:\n{context}"})
    messages_built.extend(messages)

    opts = {
        "num_predict": int(st.session_state.get("answer_max_tokens", 1024)),
        "temperature": 0.15,
        "top_p": 0.9,
    }

    last = messages_built[-1]
    last = {**last, "content": last["content"] + "\n\nAnswer only from CONTEXT; cite [L#/W#]."}
    stream = ollama.chat(model=MODEL_GEN, stream=True, messages=[*messages_built[:-1], last], options=opts)

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

# NorthBridge visual styling
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Helvetica+Neue&display=swap');

    html, body, [class*="css"] {
        font-family: 'Helvetica Neue', 'Century Gothic', sans-serif;
    }
    h1, h2, h3 {
        color: #0B3862;
        font-family: 'Century Gothic', sans-serif;
        font-weight: 600;
    }

    /* Buttons: outlined by default, fill red on hover with white text */
    .stButton > button {
        background-color: #FFFFFF !important;
        color: #832334 !important;
        border: 2px solid #832334 !important;
        border-radius: 8px;
        height: 2.5em;
        width: 100%;
        font-weight: 600;
        transition: all 0.2s ease-in-out;
    }

    /* The inner label span also needs its color set */
    .stButton > button span {
        color: #832334 !important;
    }

    /* Hover: fill red + white text (button and its span) */
    .stButton > button:not(:disabled):hover {
        background-color: #832334 !important;
        border-color: #832334 !important;
    }
    .stButton > button:not(:disabled):hover span {
        color: #FFFFFF !important;
    }

    /* Cover the 'primary' kind too (Streamlit sometimes uses it) */
    .stButton > button[kind="primary"] {
        background-color: #FFFFFF !important;
        color: #832334 !important;
        border: 2px solid #832334 !important;
    }
    .stButton > button[kind="primary"]:not(:disabled):hover {
        background-color: #832334 !important;
        border-color: #832334 !important;
    }
    .stButton > button[kind="primary"]:not(:disabled):hover span {
        color: #FFFFFF !important;
    }



    /* --- Sidebar styling --- */
    section[data-testid="stSidebar"] {
        background-color: #F2F1EF;
        border-right: 1px solid #DAD9D7;
        padding-top: 1rem;
    }

    /* Sidebar headers and captions */
    section[data-testid="stSidebar"] h2, section[data-testid="stSidebar"] h3 {
        color: #0B3862;
        font-weight: 600;
        margin-bottom: 0.25rem;
    }
    section[data-testid="stSidebar"] p, section[data-testid="stSidebar"] label {
        color: #333333;
    }

    /* --- Top header bar --- */
    .app-topbar {
        background-color: #0B3862;
        color: white;
        padding: 0.6em 1em;
        margin: -1em -1em 1em -1em;
        display: flex;
        align-items: center;
        justify-content: space-between;
        border-radius: 0;
    }
    .app-topbar h3 {
        margin: 0;
        color: white;
        font-weight: 600;
    }

    /* --- Message cards --- */
    .chat-bubble {
        background-color: #FFFFFF;
        border-radius: 8px;
        padding: 10px 15px;
        margin: 8px 0;
        border: 1px solid #E1E0DF;
    }

    /* --- Divider and spacing tweaks --- */
    hr {
        border: none;
        border-top: 1px solid #E0E0E0;
        margin: 0.8em 0;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="app-topbar">
  <h3>NorthAI</h3>
  <div style="font-size:0.9em;opacity:0.85;"></div>
</div>
""", unsafe_allow_html=True)




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
    st.session_state.answer_depth = "Concise"
if "answer_max_tokens" not in st.session_state:
    st.session_state.answer_max_tokens = 1024


# Top bar controls 
col1, col2 = st.columns([0.85, 0.15])
with col2:
    st.toggle("Offline Mode", key="offline_mode", help="Offline only: disables web search.")
# Derive web search flag from toggle
use_web = not st.session_state.get("offline_mode", False)
print(f"[Debug] Offline Mode: {st.session_state.get('offline_mode', False)} | use_web = {use_web}")


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
        "Response length limit (in tokens)",
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
        
        with st.spinner("Processing files..."):
            added = ingest_uploaded(uploads[:10])
        
        if added == len(uploads[:10]):
            st.success(f"✅ Successfully indexed all {added} file(s).")
        elif added > 0:
            st.warning(f"⚠️ Indexed {added} of {len(uploads[:10])} files. Some may have failed.")
        else:
            st.error(f"❌ Failed to index any files. Check file formats.")
        
        # Optional: Show which files are now loaded
        with st.expander("📁 View loaded files"):
            try:
                lcoll = get_or_create_collection(COLL_LOCAL)
                all_docs = lcoll.get(include=["metadatas"])
                sources = sorted(set(
                    md.get('source', '').replace('upload://', '') 
                    for md in all_docs.get('metadatas', []) 
                    if md.get('source')
                ))
                for src in sources:
                    st.caption(f"📄 {src}")
            except:
                st.caption("Could not retrieve file list.")

    st.markdown("---")

    # Show file count
    try:
        lcoll = get_or_create_collection(COLL_LOCAL)
        file_count = len(set(
            md.get('source', '') 
            for md in lcoll.get(include=["metadatas"]).get('metadatas', [])
            if md.get('source')
        ))
        if file_count > 0:
            st.caption(f"📊 {file_count} document(s) currently loaded")
            
            if st.button("🗑️ Clear all documents", help="Remove all uploaded files from this session"):
                try:
                    # Delete and recreate the collection
                    client = chromadb.PersistentClient(
                        path=str(get_session_db_path(st.session_state.current_session)),
                        settings=Settings(anonymized_telemetry=False)
                    )
                    client.delete_collection(COLL_LOCAL)
                    client.get_or_create_collection(COLL_LOCAL)
                    st.success("✅ All documents cleared.")
                    st.rerun()
                except Exception as e:
                    st.error(f"Failed to clear documents: {e}")
    except:
        pass
    

    # Agents (coming soon)
    st.caption("Agents (coming soon)")
    st.selectbox("Agent", ["General (default)", "Computer Science", "Biology"], index=0, disabled=True)
    
    st.markdown("---")

    # Chats
    st.subheader("Chats")

    # Add new chat button
    if st.button("New Chat", use_container_width=True):
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
    # Maintains insertion order with a dict
    by_src: dict[str, str] = {}
    for _doc, src, tag in pairs:
        tag = tag.replace(" ", "")
        if tag not in cited_tags:
            continue
        # keep first (lowest number) tag we encounter per source
        if src not in by_src:
            by_src[src] = tag
        else:
            # chooses lower numeric index for stability
            old = by_src[src]
            if old[0] == tag[0]:  # same L/W
                try:
                    if int(tag[1:]) < int(old[1:]):
                        by_src[src] = tag
                except Exception:
                    pass

    # Split offline vs online sources and render
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



# Chat turn logic - with task router
# 1. Retrieval Phase

if prompt := st.chat_input("Message"):
    # Record user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    maybe_update_summary()
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate title only for the first message of a new chat
    use_web = not is_offline_mode()
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

   # Task Routing and Specialist Dispatch


    # Initialize last_intent if it doesn't exist
    if "last_intent" not in st.session_state:
        st.session_state.last_intent = None
    
    # 1. Classify the user's intent
    intent = classify_intent(prompt)
    
    # 2. Detect follow-up refinement requests
    def is_followup_refinement(user_prompt: str, previous_intent: Optional[TaskIntent]) -> bool:
        """
        Detects if a query is a follow-up refinement request for text manipulation tasks.
        
        Uses multiple signals:
        - Explicit refinement keywords
        - Short imperative commands
        - Contextual continuation patterns
        
        Returns:
            bool: True if this appears to be a follow-up refinement
        """
        if previous_intent not in [TaskIntent.PROOFREAD, TaskIntent.ELABORATE, TaskIntent.SUMMARIZE]:
            return False
        
        prompt_lower = user_prompt.lower().strip()
        word_count = len(user_prompt.strip().split())
        
        # Signal 1: Explicit refinement keywords (high confidence)
        if any(keyword in prompt_lower for keyword in REFINEMENT_KEYWORDS):
            return True
        
        # Signal 2: Short imperative commands (medium confidence)
        first_word = user_prompt.strip().split()[0].lower() if user_prompt.strip() else ""
        if word_count <= MAX_FOLLOWUP_WORDS and first_word in IMPERATIVE_VERBS:
            return True
        
        # Signal 3: Very short queries (likely continuation)
        if word_count <= VERY_SHORT_FOLLOWUP_THRESHOLD:
            return True
        
        return False

    # Follow-up refinement + pivot detection
    if intent == TaskIntent.SEARCH_QUERY:
        # Detect short follow-up commands like "make it clearer"
        if is_followup_refinement(prompt, st.session_state.last_intent):
            intent = st.session_state.last_intent
            print(f"[ROUTING] Follow-up refinement detected → reusing intent: {intent.name}")

    # Always check for topic/location pivots between searches
    last_q = st.session_state.get("last_query")
    need_refresh = detect_pivot(prompt, last_q)
    st.session_state["force_retrieve"] = need_refresh

    if need_refresh:
        print(f"[Retrieval] Pivot detected or first query — forcing retrieval for '{prompt[:60]}'")
        st.session_state["last_query"] = prompt
    else:
        print("[Retrieval] Continuing with cached topic context.")

    
    # 3. Get the clean text to operate on
    specialist_text = strip_command_from_text(prompt)

    # 4. Define the specialist LLM call function with conversational memory
    def specialist_llm_call(system_prompt: str, user_text: str) -> str:
        """
        A non-streaming, blocking LLM call for specialist tasks.
        Now includes conversational memory for follow-up refinements.
        """
        if ollama is None:
            return "Error: Ollama client not installed."
        
        try:
            # Start with system prompt
            messages = [{"role": "system", "content": system_prompt}]
            
            # Add recent conversation history for follow-up requests
            # Check if this is a follow-up (same intent as previous turn)
            current_intent = intent  # Capture the intent from outer scope
            is_followup = (current_intent == st.session_state.get("last_intent"))
            
            if is_followup and len(st.session_state.messages) > 0:
                # Include recent conversation context for iterative refinement
                recent_history = st.session_state.messages[-FOLLOWUP_CONTEXT_MESSAGES:]
                
                for msg in recent_history:
                    # Skip system messages (summaries)
                    if msg["role"] != "system":
                        messages.append({"role": msg["role"], "content": msg["content"]})
                
                print(f"[specialist_llm_call] Follow-up detected, added {len([m for m in recent_history if m['role'] != 'system'])} history messages")
            
            # Add the current user request
            messages.append({"role": "user", "content": user_text})
            
            # Use standard options, but not streaming
            opts = {
                "num_predict": int(st.session_state.get("answer_max_tokens", 2048)),
                "temperature": 0.3,
            }
            
            response = ollama.chat(
                model=MODEL_GEN,
                messages=messages,
                options=opts,
                stream=False
            )
            
            return response['message']['content'].strip()
        
        except Exception as e:
            # Log with context for debugging
            print(f"[SPECIALIST] Error in {current_intent.name}: {type(e).__name__}: {e}")
            return "Sorry, I encountered an error while processing your request."

    # 5. Initialise variables for the turn
    answer_text = ""
    context = ""
    pairs = []
    
    # 6. Dispatch to the correct handler
    try:
        # Branch 1: Default search and RAG
        if intent == TaskIntent.SEARCH_QUERY:
            is_full_summary = st.session_state.answer_depth == "Full Summary (Slow)"

            if is_full_summary:
                with st.spinner("Starting full summary... This may take several minutes."):
                    try:
                        # 1. Get all unique file sources
                        lcoll = get_or_create_collection(COLL_LOCAL)
                        all_docs = lcoll.get(include=["metadatas"])
                        all_sources = sorted(list(set(
                            md["source"] for md in all_docs["metadatas"] if md.get("source")
                        )))

                        if not all_sources:
                            st.warning("No local files found to summarise.")
                            st.stop()

                        file_summaries = []
                        all_pairs = []

                        # 2. Summarize each file individually
                        for source_name in all_sources:
                            st.status(f"Summarizing {source_name}...")

                            summary_prompt = (
                                f"What are the main topics, key points, and conclusions of "
                                f"the document '{source_name}'?"
                            )

                            file_ctx, file_pairs = retrieve_local_then_web(
                                summary_prompt,
                                top_k_local=NUM_CTX_KEEP,
                                top_k_web=0,
                                source_filter=source_name,
                            )

                            if not file_ctx:
                                continue

                            summary_resp = ollama.chat(
                                model=MODEL_GEN,
                                messages=[
                                    {
                                        "role": "system",
                                        "content": (
                                            "You are a summarizing assistant. Summarize the "
                                            "key points of the following context. Do not add "
                                            "any preamble or your own opinions. Be factual and concise."
                                        ),
                                    },
                                    {
                                        "role": "user",
                                        "content": f"Context:\n{file_ctx}\n\nSummary of {source_name}:",
                                    },
                                ],
                                stream=False,
                            )
                            file_summary = summary_resp["message"]["content"]
                            file_summaries.append(f"Summary for {source_name}:\n{file_summary}")
                            all_pairs.extend(file_pairs)

                        context = "\n\n---\n\n".join(file_summaries) if file_summaries else ""
                        pairs = all_pairs

                    except Exception as e:
                        st.error(f"Error during full summary: {e}")
                        st.stop()

            else:
                # Standard RAG branch
                ctx_local, pairs_local = retrieve_local_then_web(
                    prompt,
                    top_k_local=NUM_CTX_KEEP,
                    top_k_web=0,
                )
                context = ctx_local
                pairs = list(pairs_local)

                safe_query = privacy_expand_query(prompt)
                year_min, year_max = parse_year_filters(prompt)
                if st.session_state.get("enable_years", False):
                    if "year_min" in st.session_state:
                        year_min = int(st.session_state.year_min)
                    if "year_max" in st.session_state:
                        year_max = int(st.session_state.year_max)
                if year_min is not None and year_max is not None and year_min > year_max:
                    year_min, year_max = year_max, year_min

                # Local-doc outbound filter seeding
                try:
                    lcoll = get_or_create_collection(COLL_LOCAL)
                    peek = lcoll.peek(500) or {}
                    local_docs_seed = []
                    for docs in (peek.get("documents") or []):
                        local_docs_seed.extend(docs)
                except Exception:
                    local_docs_seed = []

                # Optional web retrieval
                if use_web:
                    is_academic_query = any(k in prompt.lower() for k in [
                        "research", "study", "paper", "journal", "article", "published",
                        "doi", "arxiv", "pubmed", "citation", "literature", "scholar",
                        "peer review", "findings", "methodology",
                    ])

                    results = []

                    if is_academic_query:
                        print(f"[App] Academic query detected: {prompt[:50]}...")
                        from retrieval.retrieval_router import search_academic
                        try:
                            academic_results = search_academic(
                                manager,
                                safe_query,
                                year_min=year_min,
                                year_max=year_max,
                            ) or []

                            with st.status(f"🎓 Found {len(academic_results)} academic sources", state="complete"):
                                if academic_results:
                                    for i, paper in enumerate(academic_results[:5], 1):
                                        title = paper.get("title", "Untitled")[:70]
                                        source = paper.get("source", "unknown")
                                        st.caption(f"{i}. [{source}] {title}")
                                else:
                                    st.caption("No results from academic databases")

                            # Enrichment & persistence
                            url_to_chunks = {}
                            for rec in academic_results:
                                title = rec.get("title") or ""
                                authors = rec.get("authors") or []
                                published = rec.get("published") or ""
                                abstract = rec.get("abstract") or ""
                                doi = rec.get("doi") or ""
                                url = rec.get("url") or (f"doi:{doi}" if doi else "unknown-source")

                                parts = []
                                if title:
                                    parts.append(f"**Title:** {title}")
                                if authors:
                                    a = ", ".join(authors[:5])
                                    if len(authors) > 5:
                                        a += " et al."
                                    parts.append(f"**Authors:** {a}")
                                if published:
                                    parts.append(f"**Published:** {published}")
                                if abstract:
                                    parts.append(f"**Abstract:** {abstract}")
                                if doi:
                                    parts.append(f"**DOI:** {doi}")

                                text = "\n".join(parts).strip()
                                if text:
                                    url_to_chunks[url] = [text]

                            if url_to_chunks:
                                add_documents(url_to_chunks, target=COLL_WEB)

                            results = academic_results or search_everything(
                                manager,
                                safe_query,
                                year_min=year_min,
                                year_max=year_max,
                            )

                        except Exception as e:
                            print(f"[App] Academic search exception ignored: {e}")
                            results = search_everything(
                                manager,
                                safe_query,
                                year_min=year_min,
                                year_max=year_max,
                            )
                    else:
                        results = search_everything(manager, safe_query, year_min=year_min, year_max=year_max)

                    # Post-search enrichment
                    with st.spinner("Searching web & enriching context..."):
                        try:
                            if local_docs_seed:
                                manager.set_outbound_filter(make_outbound_filter_fn(local_docs_seed))
                            else:
                                manager.set_outbound_filter(None)

                            if not results:
                                results = search_everything(manager, safe_query, year_min=year_min, year_max=year_max)

                            url_to_chunks = {}
                            for rec in results:
                                meta_bits = [rec.get("title"), rec.get("abstract"), rec.get("url")]
                                fallback_text = "\n\n".join([x for x in meta_bits if x])

                                try:
                                    body = fetch_page_text(manager, rec.get("url", "")) or ""
                                except Exception:
                                    body = ""

                                text = (body or fallback_text).strip()
                                if text:
                                    url_to_chunks.setdefault(rec.get("url", 'unknown-source'), []).append(text)

                            if not url_to_chunks:
                                try:
                                    ddg = search_duckduckgo(manager, safe_query, k=6) or []
                                    for rec in ddg:
                                        meta_bits = [rec.get("title"), rec.get("abstract"), rec.get("url")]
                                        text = "\n\n".join([x for x in meta_bits if x]).strip()
                                        if text:
                                            url_to_chunks.setdefault(rec.get("url", 'unknown-source'), []).append(text)
                                except Exception as e:
                                    print(f"[fallback] DDG error ignored: {e}")

                            if url_to_chunks:
                                add_documents(url_to_chunks, target=COLL_WEB)

                            ctx_mixed, pairs_mixed = retrieve_local_then_web(
                                prompt,
                                top_k_local=0,
                                top_k_web=max(1, NUM_CTX_KEEP // 2),
                            )
                            if ctx_mixed:
                                context = (context + ("\n\n---\n\n" if context else "") + ctx_mixed).strip()
                                pairs.extend(pairs_mixed)

                        except Exception as e:
                            print(f"[App] Post-search enrichment exception ignored: {e}")
                            pass

                # Generation phase
                with st.chat_message("assistant"):
                    if not _has_sufficient_evidence(context, pairs):
                        msg = "I don't have enough evidence in the provided sources."
                        st.markdown(msg)
                        answer_text = msg
                    else:
                        placeholder = st.empty()
                        buf = ""
                        for part in stream_answer(st.session_state.messages, context):
                            buf += part
                            placeholder.markdown(buf)
                        answer_text = buf

        # Branch 2: Specialist Handlers

        elif intent == TaskIntent.PROOFREAD:
            with st.spinner("Proofreading..."):
                answer_text = proofreader.run_proofreading(specialist_text, specialist_llm_call)
            st.chat_message("assistant").markdown(answer_text)

        elif intent == TaskIntent.DRAFT_EMAIL:
            with st.spinner("Drafting email..."):
                answer_text = email_drafter.run_email_draft(specialist_text, specialist_llm_call)
            st.chat_message("assistant").markdown(answer_text)

        elif intent == TaskIntent.SUMMARIZE:
            with st.spinner("Summarizing..."):
                prompt_template = specialist_prompts.get_summarizer_prompt()
                answer_text = specialist_llm_call(prompt_template, specialist_text)
            st.chat_message("assistant").markdown(answer_text)

        elif intent == TaskIntent.ANALYZE:
            with st.spinner("Analyzing..."):
                prompt_template = specialist_prompts.get_analyst_prompt()
                answer_text = specialist_llm_call(prompt_template, specialist_text)
            st.chat_message("assistant").markdown(answer_text)

        elif intent == TaskIntent.BRAINSTORM_IDEAS:
            with st.spinner("Brainstorming..."):
                prompt_template = specialist_prompts.get_brainstorm_prompt()
                answer_text = specialist_llm_call(prompt_template, specialist_text)
            st.chat_message("assistant").markdown(answer_text)

        elif intent == TaskIntent.GENERATE_CODE:
            with st.spinner("Generating code..."):
                prompt_template = specialist_prompts.get_code_generator_prompt()
                answer_text = specialist_llm_call(prompt_template, specialist_text)
            st.chat_message("assistant").markdown(answer_text)

        elif intent == TaskIntent.GENERATE_EXCEL_FORMULA:
            with st.spinner("Generating formula..."):
                prompt_template = specialist_prompts.get_excel_formula_prompt()
                answer_text = specialist_llm_call(prompt_template, specialist_text)
            st.chat_message("assistant").markdown(answer_text)

        elif intent == TaskIntent.GENERATE_SQL_QUERY:
            with st.spinner("Generating SQL..."):
                prompt_template = specialist_prompts.get_sql_query_prompt()
                answer_text = specialist_llm_call(prompt_template, specialist_text)
            st.chat_message("assistant").markdown(answer_text)

        elif intent == TaskIntent.GENERATE_REGEX:
            with st.spinner("Generating regex..."):
                prompt_template = specialist_prompts.get_regex_prompt()
                answer_text = specialist_llm_call(prompt_template, specialist_text)
            st.chat_message("assistant").markdown(answer_text)

        elif intent == TaskIntent.SOLVE_MATH:
            with st.spinner("Solving math problem..."):
                prompt_template = specialist_prompts.get_math_solver_prompt()
                answer_text = specialist_llm_call(prompt_template, specialist_text)
            st.chat_message("assistant").markdown(answer_text)

        elif intent == TaskIntent.EXTRACT_INFO:
            with st.spinner("Extracting info..."):
                prompt_template = specialist_prompts.get_info_extractor_prompt()
                answer_text = specialist_llm_call(prompt_template, specialist_text)
            st.chat_message("assistant").markdown(answer_text)

        # NEW SR&ED & TECHNICAL WRITING

        elif intent == TaskIntent.STRUCTURE_TECHNICAL_REPORT:
            with st.spinner("Structuring technical report..."):
                prompt_template = specialist_prompts.get_structure_technical_report_prompt()
                answer_text = specialist_llm_call(prompt_template, specialist_text)
            st.chat_message("assistant").markdown(answer_text)

        elif intent == TaskIntent.STRUCTURE_SRED_242_SECTION:
            with st.spinner("Drafting SR&ED Line 242..."):
                prompt_template = specialist_prompts.get_structure_sred_242_section_prompt()
                answer_text = specialist_llm_call(prompt_template, specialist_text)
            st.chat_message("assistant").markdown(answer_text)

        elif intent == TaskIntent.IDENTIFY_TECHNOLOGICAL_UNCERTAINTIES:
            with st.spinner("Identifying technological uncertainties..."):
                prompt_template = specialist_prompts.get_identify_technological_uncertainties_prompt()
                answer_text = specialist_llm_call(prompt_template, specialist_text)
            st.chat_message("assistant").markdown(answer_text)

        elif intent == TaskIntent.WRITE_TECHNICAL_JUSTIFICATION:
            with st.spinner("Writing technical justification..."):
                prompt_template = specialist_prompts.get_write_technical_justification_prompt()
                answer_text = specialist_llm_call(prompt_template, specialist_text)
            st.chat_message("assistant").markdown(answer_text)

        elif intent == TaskIntent.EXTRACT_TECHNICAL_DETAILS:
            with st.spinner("Extracting technical details..."):
                prompt_template = specialist_prompts.get_extract_technical_details_prompt()
                answer_text = specialist_llm_call(prompt_template, specialist_text)
            st.chat_message("assistant").markdown(answer_text)

        elif intent == TaskIntent.IMPROVE_TECHNICAL_CLARITY:
            with st.spinner("Improving clarity..."):
                prompt_template = specialist_prompts.get_improve_technical_clarity_prompt()
                answer_text = specialist_llm_call(prompt_template, specialist_text)
            st.chat_message("assistant").markdown(answer_text)

        elif intent == TaskIntent.GENERATE_EXPERIMENT_LOGS:
            with st.spinner("Generating experiment logs..."):
                prompt_template = specialist_prompts.get_generate_experiment_logs_prompt()
                answer_text = specialist_llm_call(prompt_template, specialist_text)
            st.chat_message("assistant").markdown(answer_text)

        elif intent == TaskIntent.COMPARE_TECHNICAL_APPROACHES:
            with st.spinner("Comparing approaches..."):
                prompt_template = specialist_prompts.get_compare_technical_approaches_prompt()
                answer_text = specialist_llm_call(prompt_template, specialist_text)
            st.chat_message("assistant").markdown(answer_text)

        elif intent == TaskIntent.GENERATE_TIMELINE:
            with st.spinner("Generating timeline..."):
                prompt_template = specialist_prompts.get_generate_timeline_prompt()
                answer_text = specialist_llm_call(prompt_template, specialist_text)
            st.chat_message("assistant").markdown(answer_text)

        elif intent == TaskIntent.GRADE_CLIENT:
            with st.spinner("Grading client..."):
                prompt_template = "You are a client evaluation assistant. Based on the provided criteria and text, assign a clear grade and explain the reasoning."
                answer_text = specialist_llm_call(prompt_template, specialist_text)
            st.chat_message("assistant").markdown(answer_text)

        # FINAL FALLBACK
        else:
            with st.spinner("Thinking..."):
                answer_text = specialist_llm_call(
                    "You are a helpful assistant.",
                    prompt
                )
            st.chat_message("assistant").markdown(answer_text)

    # GLOBAL EXCEPTION HANDLER FOR THE ENTIRE DISPATCH BLOCK
    except Exception as e:
        st.error(f"An error occurred during processing: {e}")
        answer_text = "Sorry, I ran into an error. Please check the logs."
        st.chat_message("assistant").markdown(answer_text)
