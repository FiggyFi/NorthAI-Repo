# vector_store.py
# Handles ingestion, storage, and retrieval with ChromaDB + SentenceTransformers.
# Adds broad file parsing, OCR, L/W-tagged retrieval, and 10-file cap.

# Standard
import hashlib
import io
import json
import os
from pathlib import Path
from typing import Iterable, Tuple, List, Dict

# ML / Embedding / Vector Store
import torch
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Parsers
from pypdf import PdfReader
from docx import Document
from pptx import Presentation
from PIL import Image  # for OCR
try:
    import pytesseract
    _HAS_OCR = True
except Exception:
    _HAS_OCR = False

# Data Analysis
import pandas as pd

# Parsers
from retrieval.parsers import (
    SUPPORTED_READERS as _SUPPORTED,
    read_rtf_bytes,
    read_text_bytes,  # used as fallback in ingest
)

# session-aware collection handling
from uuid import uuid4
import shutil
import streamlit as st

# Paths & constants (fixed)
BASE_DIR = Path(__file__).resolve().parent
DB_PATH = (BASE_DIR / "web_rag_db").resolve()

# Preferred local embedding folder (override with EMB_MODEL_DIR in .env)
EMB_MODEL_DIR = Path(
    os.getenv("EMB_MODEL_DIR", BASE_DIR / "models" / "bge-small-en-v1.5")
).resolve()

_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

COLL_LOCAL = "local_docs"   # Uploaded files (offline)
COLL_WEB   = "web_docs"     # Web search results (online)

# Cached resources
_embedder = None
_client = None

# Core helpers 

def get_session_db_path(session_id: str) -> Path:
    """
    Returns a unique, persistent ChromaDB path for the given session.
    Each chat/session gets its own isolated database directory.
    """
    db_path = DB_PATH / "sessions" / f"session_{session_id}"
    db_path.mkdir(parents=True, exist_ok=True)
    return db_path

def get_session_id():
    sid = st.session_state.get("session_id")
    if not sid:
        sid = str(uuid4())
        st.session_state.session_id = sid
    return sid

def get_embedder():
    """Load and cache the sentence transformer model strictly from disk."""
    global _embedder
    if _embedder is None:
        # Fail fast with a clear message if the folder is missing
        cfg = EMB_MODEL_DIR / "config.json"
        if not (EMB_MODEL_DIR.exists() and cfg.exists()):
            raise RuntimeError(
                "Local embedding model not found.\n"
                f"Expected folder: {EMB_MODEL_DIR}\n"
                "Fix:\n"
                "  A) Create 'models' in your project and copy the full model folder into it, e.g.\n"
                "     models\\bge-small-en-v1.5\\config.json (plus weights & tokenizer files)\n"
                "  B) Or set EMB_MODEL_DIR in .env to the absolute path of your local model folder."
            )
        _embedder = SentenceTransformer(str(EMB_MODEL_DIR), device=_DEVICE)
    return _embedder

def get_chroma_client():
    """Create and cache the ChromaDB client (persistent)."""
    global _client
    if _client is None:
        _client = chromadb.PersistentClient(
            path=DB_PATH, settings=Settings(anonymized_telemetry=False)
        )
    return _client

def get_or_create_collection(name: str, persistent=True):
    """
    Returns a Chroma collection that is automatically scoped to the active chat/session.
    Prevents document leakage across sessions.
    """
    import streamlit as st
    session_id = st.session_state.get("current_session")
    if not session_id:
        # Fallback: generate one if this is the first run
        session_id = str(uuid4())
        st.session_state.session_id = session_id

    db_path = get_session_db_path(session_id)
    client = chromadb.PersistentClient(
        path=str(db_path),
        settings=Settings(anonymized_telemetry=False)
    )
    return client.get_or_create_collection(name)

def delete_collection_for_session(session_id: str):
    """
    Permanently delete the Chroma database directory for the given session.
    Called automatically when a chat is deleted.
    """
    try:
        db_path = get_session_db_path(session_id)
        if db_path.exists():
            shutil.rmtree(db_path, ignore_errors=True)
            print(f"[vector_store] Deleted Chroma data for session {session_id}")
    except Exception as e:
        print(f"[vector_store] Error deleting Chroma data for {session_id}: {e}")



def text_sha1(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()

def _split_chunks(text: str) -> List[str]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500, chunk_overlap=100,
        separators=["\n\n", "\n", ".", "?", "!", " ", ""],
    )
    return [c.page_content for c in splitter.create_documents([text])]

def add_documents(url_to_chunks: Dict[str, List[str]], target: str):
    """
    Add new text chunks into either the LOCAL or WEB vector collection.
    Skips duplicates based on SHA1(url+chunk). Assumes upstream parsing/filters.
    """
    coll = get_or_create_collection(target)
    model = get_embedder()

    docs, metas, ids = [], [], []
    existing = set()

    # Peek existing hashes for fast de-dupe

    # ...
    try:
        peeked = coll.peek(100000) or {}
    except Exception:
        peeked = {}

    if isinstance(peeked, dict) and "metadatas" in peeked:
        # FIX: Iterate directly over the list of metadata dictionaries
        for md in peeked.get("metadatas", []):
            if isinstance(md, dict) and md.get("hash"):
                existing.add(md["hash"])
                
    # Prepare new items
    for url, chunks in (url_to_chunks or {}).items():
        if not chunks:
            continue
        for i, ch in enumerate(chunks):
            ch = (ch or "").strip()
            if not ch:
                continue
            h = text_sha1(url + ch)
            if h in existing:
                continue
            docs.append(ch)
            metas.append({"source": url, "hash": h})
            ids.append(f"{url}##{i}")

    # Upsert batch
    if docs:
        embs = model.encode(docs, convert_to_numpy=True, show_progress_bar=False)
        coll.upsert(documents=docs, metadatas=metas, ids=ids, embeddings=embs)

def ingest_uploaded(files: list) -> int:
    """
    Parse and add uploaded files into LOCAL collection.
    Enforces max 10 files per call.
    Returns # of successfully indexed files.
    """
    if not files:
        return 0

    files = list(files)[:10]  # hard cap
    url_to_chunks: Dict[str, List[str]] = {}
    added = 0

    for f in files:
        try:
            name = getattr(f, "name", "uploaded")
            data = f.read()
            ext = Path(name).suffix.lower()
            reader = _SUPPORTED.get(ext, read_text_bytes)

            # parse
            text = reader(data) or ""
            if not text.strip():
                # Helpful diagnostics for formats that often fail offline
                if ext == ".doc":
                    print(f"[ingest] Skipped legacy .doc (convert to .docx/PDF): {name}")
                else:
                    print(f"[ingest] Skipped empty/unsupported content: {name}")
                continue

            # split & collect
            chunks = _split_chunks(text)
            if not chunks:
                print(f"[ingest] No non-empty chunks after split: {name}")
                continue

            url_to_chunks[f"upload://{name}"] = chunks
            added += 1

        except Exception as e:
            # Never let one bad file break the batch
            print(f"[ingest] Error parsing {name}: {e!r}")
            continue

    if url_to_chunks:
        add_documents(url_to_chunks, target=COLL_LOCAL)

    return added

def retrieve_local_then_web(
    question: str,
    top_k_local: int = 6,
    top_k_web: int = 3,
    source_filter: str | None = None  # <-- ADD THIS ARGUMENT
) -> Tuple[str, List[Tuple[str, str, str]]]:
    """
    Retrieve from LOCAL and WEB separately and return a combined context.
    
    Args:
        question: The user's query.
        top_k_local: Number of local results to fetch.
        top_k_web: Number of web results to fetch.
        source_filter: If provided, filter local results to this source URI.
    
    Returns:
      - context string with L#/W# sections
      - pairs: [(doc, source_url, 'L1'|'W2'), ...]
    """
    model = get_embedder()
    q_emb = model.encode([question], convert_to_numpy=True)

    blocks: List[str] = []
    pairs: List[Tuple[str, str, str]] = []

    # LOCAL first (optional)
    if top_k_local > 0:
        lcoll = get_or_create_collection(COLL_LOCAL)
        
        # --- THIS IS THE NEW LOGIC BLOCK ---
        query_params = {
            "query_embeddings": q_emb,
            "n_results": top_k_local
        }
        if source_filter:
            query_params["where"] = {"source": source_filter}
            
        try:
            lres = lcoll.query(**query_params)
        except Exception as e:
            print(f"[vector_store] Query failed (filter: {source_filter}): {e}")
            lres = {}
        ldocs = lres.get("documents", [[]])[0]
        lmeta = lres.get("metadatas", [[]])[0]
        for i, (doc, md) in enumerate(zip(ldocs, lmeta), 1):
            if not doc:
                continue
            src = (md or {}).get("source", "")
            tag = f"L{i}"
            blocks.append(f"[{tag}] LOCAL Source: {src}\n{doc}")
            pairs.append((doc, src, tag))

    # WEB next (optional)
    if top_k_web > 0:
        wcoll = get_or_create_collection(COLL_WEB)
        wres = wcoll.query(query_embeddings=q_emb, n_results=top_k_web)
        wdocs = wres.get("documents", [[]])[0]
        wmeta = wres.get("metadatas", [[]])[0]
        for i, (doc, md) in enumerate(zip(wdocs, wmeta), 1):
            if not doc:
                continue
            src = (md or {}).get("source", "")
            tag = f"W{i}"
            blocks.append(f"[{tag}] WEB Source: {src}\n{doc}")
            pairs.append((doc, src, tag))

    context = "\n\n---\n\n".join(blocks)
    return context, pairs

def retrieve_all(question: str, top_k: int = 6) -> Tuple[str, List[Tuple[str, str, str]]]:
    """
    Retrieve top-k docs from LOCAL (L#) and WEB (W#) collections.
    Returns:
      - formatted context text with [L#]/[W#] prefixes,
      - pairs as [(doc, src, tag), ...] where tag is 'L1'/'W2' etc.
    """
    model = get_embedder()
    q_emb = model.encode([question], convert_to_numpy=True)

    blocks: List[str] = []
    pairs: List[Tuple[str, str, str]] = []

    # Local first
    lcoll = get_or_create_collection(COLL_LOCAL)
    lres = lcoll.query(query_embeddings=q_emb, n_results=top_k)
    ldocs = lres.get("documents", [[]])[0]
    lmeta = lres.get("metadatas", [[]])[0]
    for i, (doc, md) in enumerate(zip(ldocs, lmeta), 1):
        src = md.get("source", "")
        tag = f"L{i}"
        blocks.append(f"[{tag}] LOCAL Source: {src}\n{doc}")
        pairs.append((doc, src, tag))

    # Web next
    wcoll = get_or_create_collection(COLL_WEB)
    wres = wcoll.query(query_embeddings=q_emb, n_results=top_k)
    wdocs = wres.get("documents", [[]])[0]
    wmeta = wres.get("metadatas", [[]])[0]
    for i, (doc, md) in enumerate(zip(wdocs, wmeta), 1):
        src = md.get("source", "")
        tag = f"W{i}"
        blocks.append(f"[{tag}] WEB Source: {src}\n{doc}")
        pairs.append((doc, src, tag))

    return "\n\n---\n\n".join(blocks), pairs
