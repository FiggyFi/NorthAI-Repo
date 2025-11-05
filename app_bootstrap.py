# ./app_bootstrap.py  (drop-in utility; import and call from offline_gpt_app.py)
from pathlib import Path
import os, socket
import datetime
import shutil
from retrieval.common import AppConfig, prune_old_logs, prune_old_caches

REQUIRED_MODEL_DIR = Path("./models/bge-small-en-v1.5")

def check_model_dir():
    ok = REQUIRED_MODEL_DIR.exists() and (REQUIRED_MODEL_DIR / "config.json").exists()
    return ok, f"Missing model at {REQUIRED_MODEL_DIR}. Copy the whole 'bge-small-en-v1.5' folder locally."

def check_chroma_path():
    db = Path("./web_rag_db")
    try:
        db.mkdir(parents=True, exist_ok=True)
        (db / ".write_test").write_text("ok", encoding="utf-8")
        (db / ".write_test").unlink(missing_ok=True)
        return True, ""
    except Exception as e:
        return False, f"ChromaDB path not writable: {db} ({e})"

def check_airgap_env():
    air = os.getenv("AIRPLANE_MODE", "1")
    return air in ("1", "true", "True"), "Set AIRPLANE_MODE=1 in your .env for offline default."

def health_summary():
    checks = [
        ("Embedding model", *check_model_dir()),
        ("Chroma path", *check_chroma_path()),
        ("Airplane mode default", *check_airgap_env()),
    ]
    ok = all(c[1] for c in checks)
    lines = []
    for name, passed, msg in checks:
        lines.append(f"✅ {name}" if passed else f"❌ {name}: {msg}")
    return ok, "\n".join(lines)

# --- UI helpers for Streamlit ---
def snapshot_items(d: dict):
    # Avoid "dictionary changed size during iteration" in Streamlit reruns.
    return list(d.items())

def run_maintenance_tasks():
    """
    Run periodic cleanup tasks like pruning old logs and caches.
    This is non-blocking and will only print on success/failure.
    """
    print("[bootstrap] Running maintenance tasks...")
    try:
        config = AppConfig()
        prune_old_logs(config, months=6)
        prune_old_caches(config, months=12)
        print("[bootstrap] Maintenance tasks complete.")
    except Exception as e:
        print(f"[bootstrap] Maintenance tasks failed: {e}")

def assert_prereqs_or_raise():
    ok, report = health_summary()
    if not ok:
        raise RuntimeError("Startup checks failed:\n" + report)
    run_maintenance_tasks()

