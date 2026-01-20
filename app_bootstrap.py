from pathlib import Path
import os, socket, datetime, shutil, re, subprocess
from typing import Optional
from retrieval.common import AppConfig, prune_old_logs, prune_old_caches
from vector_store import retrieve_local_then_web
import requests

# Match vector_store.py
LOCAL_DATA_DIR = Path(os.getenv("LOCALAPPDATA")) / "NorthAI"
CHROMA_PATH = LOCAL_DATA_DIR / "web_rag_db"

# Models may stay relative
REQUIRED_MODEL_DIR = Path("./models/bge-small-en-v1.5")

# Health / readiness checks
def check_model_dir():
    ok = REQUIRED_MODEL_DIR.exists() and (REQUIRED_MODEL_DIR / "config.json").exists()
    return ok, f"Missing model at {REQUIRED_MODEL_DIR}. Copy the whole 'bge-small-en-v1.5' folder locally."

def check_chroma_path():
    try:
        CHROMA_PATH.mkdir(parents=True, exist_ok=True)
        test = CHROMA_PATH / ".write_test"
        test.write_text("ok", encoding="utf-8")
        test.unlink(missing_ok=True)
        return True, ""
    except Exception as e:
        return False, f"ChromaDB path not writable: {CHROMA_PATH} ({e})"

def check_ollama():
    try:
        r = requests.get("http://127.0.0.1:11434/api/version", timeout=2)
        if r.status_code == 200:
            ver = r.json().get("version", "unknown")
            return True, f"Ollama running (version {ver})"
        else:
            return False, f"Ollama HTTP API returned status {r.status_code}"
    except requests.exceptions.ConnectionError:
        return False, "Cannot connect to Ollama at 127.0.0.1:11434. Is it running?"
    except Exception as e:
        return False, f"Ollama check failed: {e}"

def report_airgap_env():
    air = str(os.getenv("OFFLINE_MODE", "1")).strip().lower()
    msg = f"OFFLINE_MODE={air} (toggle in UI overrides at runtime)"
    return True, msg

def health_summary():
    checks = [
        ("Embedding model", *check_model_dir()),
        ("Chroma path", *check_chroma_path()),
        ("Ollama", *check_ollama()),
        ("offline mode setting", *report_airgap_env()),
    ]
    ok = all(c[1] for c in checks)
    lines = [f"✅ {n}" if p else f"❌ {n}: {m}" for (n, p, m) in checks]
    return ok, "\n".join(lines)

# Maintenance
def snapshot_items(d: dict):
    """Avoid 'dictionary changed size during iteration' errors in Streamlit reruns."""
    return list(d.items())


def run_maintenance_tasks(force: bool = False):
    """
    Run periodic cleanup tasks like pruning old logs and caches.
    Ensures one-time execution per runtime unless force=True.
    """
    # Guard so we don’t spam logs every turn
    if getattr(run_maintenance_tasks, "_already_ran", False) and not force:
        return
    print("[bootstrap] Running maintenance tasks...")
    try:
        config = AppConfig()
        prune_old_logs(config, months=6)
        prune_old_caches(config, months=12)
        print("[bootstrap] Maintenance tasks complete.")
    except Exception as e:
        print(f"[bootstrap] Maintenance tasks failed: {e}")
    run_maintenance_tasks._already_ran = True

def assert_prereqs_or_raise():
    ok, report = health_summary()
    if not ok:
        raise RuntimeError("Startup checks failed:\n" + report)
    run_maintenance_tasks()
    return True