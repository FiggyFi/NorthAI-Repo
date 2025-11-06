# File: retrieval/common.py
# Add a strict offline guard so no HTTP happens in airgapped mode.

import os
import time
import json
import hashlib
import datetime
import zipfile
import shutil
import inspect
import requests
import requests_cache
from dotenv import load_dotenv

try:
    import pyzipper
    _HAS_PYZIPPER = True
except ImportError:
    _HAS_PYZIPPER = False

load_dotenv()

CONTACT_EMAIL = os.getenv("CONTACT_EMAIL", "contact@north.local")
BASE_UA = f"NorthAI/1.0 (+mailto:{CONTACT_EMAIL})"

# Shared AES-256 key (used for encrypted cache exports/imports)
# WARNING: In production, load this from env instead of hard-coding.
# Example: os.getenv("APP_AES_KEY", "fallback_value")
DEFAULT_AES_KEY = "ZpU7UoUokYfV9T6F0b1RkKXK7u5QY8gP"  # example placeholder

def get_local_encryption_key() -> str:
    """Return the shared AES password used across all distributed builds."""
    return DEFAULT_AES_KEY

# Offline Mode (Global Runtime State)
def _env_default_on() -> bool:
    v = str(os.getenv("OFFLINE_MODE", "0")).strip().lower()
    return v in ("1", "true", "yes", "on")

_STATE = {"offline": _env_default_on()}

def set_offline_mode(flag: bool) -> None:
    """Called by the UI to flip offline mode during runtime."""
    _STATE["offline"] = bool(flag)
    # keep env in sync for any legacy readers
    os.environ["OFFLINE_MODE"] = "1" if flag else "0"

def is_offline_mode() -> bool:
    """Read the current runtime offline mode."""
    return bool(_STATE["offline"])

def offline_mode_str() -> str:
    return "ON" if is_offline_mode() else "OFF"

# Helper Functions (No State)
def unify(title, url, source, authors=None, abstract=None, published=None, doi=None, extra=None):
    return {
        "title": title or "",
        "authors": authors or [],
        "abstract": abstract,
        "url": url,
        "published": published,
        "doi": doi,
        "source": source,
        "extra": extra or {},
    }

def hash_text(text: str) -> str:
    """Return a short SHA256 hash of text for integrity tracking."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]

# Application Configuration - Centralizes all path management for logs, sessions, and caches
class AppConfig:
    """Centralizes all application path management."""

    def __init__(self):
        self.log_dir = os.getenv("APP_LOG_DIR", "logs")
        self.session_dir = os.getenv("APP_SESSION_DIR", "sessions")
        self.cache_export_dir = os.getenv("APP_CACHE_EXPORT_DIR", "cache_exports")

        # Ensure the base /logs/audit folder exists
        os.makedirs(os.path.join(self.log_dir, "audit"), exist_ok=True)
        os.makedirs(self.session_dir, exist_ok=True)
        os.makedirs(self.cache_export_dir, exist_ok=True)

    def get_session_audit_dir(self, session_id: str) -> str:
        """
        NEW: Returns the single, unified audit directory for a session.
        e.g., /logs/audit/chat-123abc/
        """
        path = os.path.join(self.log_dir, "audit", session_id)
        os.makedirs(path, exist_ok=True)
        return path

    def get_audit_log_path(self, session_id: str) -> str:
        """
        Return the audit log file path for a session.
        This now saves to the unified session folder.
        """
        session_dir = self.get_session_audit_dir(session_id)
        return os.path.join(session_dir, "chat_history.jsonl")

    def get_session_turn_dir(self, session_id: str) -> str:
        """
        Get the directory for storing full chat turns.
        This now points to the unified session folder.
        """
        return self.get_session_audit_dir(session_id)

    def get_session_cache_path(self, session_id: str) -> str:
        """Get the path to a session's retrieval cache (base name)."""
        cache_dir = os.path.join(self.session_dir, session_id)
        os.makedirs(cache_dir, exist_ok=True)
        return os.path.join(cache_dir, "retrieval_cache")

    def get_cache_export_path(self, session_id: str) -> str:
        """Get the destination path for a cache export."""
        filename = f"session_{session_id}_cache.zip"
        return os.path.join(self.cache_export_dir, filename)

    def get_prune_log_dir(self) -> str:
        # This now points to the base /audit/ folder
        return os.path.join(self.log_dir, "audit")

    def get_prune_cache_dir(self) -> str:
        return self.cache_export_dir

# Retrieval Manager - Manages retrieval, auditing, and caching for a single session
class RetrievalManager:
    """Manages all retrieval, auditing, and caching for a *single session*"""

    def __init__(self, config: AppConfig, session_id: str,
                 session_metadata: dict | None = None):
        
        self.config = config
        self.session_id = session_id
        self.session_metadata = session_metadata or {}

        # Each manager owns its own cached session
        session_cache_file = self.config.get_session_cache_path(session_id)
        self.session = requests_cache.CachedSession(
            cache_name=session_cache_file,
            backend="sqlite",
            expire_after=60 * 60 * 24 * 7,  # 7 days
        )

        self.outbound_filter = None

    # Outbound filter handling
    def set_outbound_filter(self, func):
        """Register func(payload:str)->str to sanitize outbound query text."""
        self.outbound_filter = func

    def _guard_text(self, s: str) -> str:
        if not isinstance(s, str):
            return ""
        if self.outbound_filter:
            return self.outbound_filter(s)
        return s[:512]

    # HTTP retrieval with guard
    def get(self, url, *, params=None, headers=None, timeout=15, api_name=None):
        """Centralised GET with offline-mode guard and outbound sanitization."""
        if is_offline_mode():
            raise RuntimeError("Offline mode active. Network calls are blocked.")

        if isinstance(params, dict):
            for k, v in list(params.items()):
                if isinstance(v, str):
                    params[k] = self._guard_text(v)

        ua = BASE_UA if not api_name else f"{BASE_UA} [{api_name}]"
        h = {"User-Agent": ua}
        if headers:
            h.update(headers)

        r = self.session.get(url, params=params, headers=h, timeout=timeout)
        r.raise_for_status()
        return r

    # Audit and trace utilities
    def write_audit_log(self, event_type: str, data: dict):
        """Append an event to the JSONL audit trail for deterministic traceability."""
        log_path = self.config.get_audit_log_path(self.session_id)

        frame = inspect.currentframe()
        caller = frame.f_back.f_code.co_name if frame and frame.f_back else "unknown"

        entry = {
            "timestamp_utc": datetime.datetime.utcnow().isoformat(),
            "session_id": self.session_id,
            "source": f"manager.{caller}",
            "event": event_type,
            "data": data,
            **self.session_metadata
        }

        try:
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except Exception as e:
            print(f"[audit] failed to write log: {e}")

    def write_chat_turn(self, turn_number: int, user_query: str, response_text: str, retrieved_ids: list):
        """Append chat-turn metadata to the audit trail without storing full text."""
        self.write_audit_log(
            event_type="chat_turn",
            data={
                "turn": turn_number,
                "query_preview": user_query[:100],
                "retrieved_ids": retrieved_ids,
                "response_hash": hash_text(response_text),
            },
        )

        turns_dir = self.config.get_session_turn_dir(self.session_id)
        with open(os.path.join(turns_dir, f"turn_{turn_number:03d}.txt"), "w", encoding="utf-8") as f:
            f.write(response_text)

    # Cache export methods
    def export_session_cache(self):
        """Bundle this session’s cache into a portable ZIP."""
        src = self.config.get_session_cache_path(self.session_id) + ".sqlite"
        dst = self.config.get_cache_export_path(self.session_id)

        if not os.path.exists(src):
            print(f"[export] No cache found for session {self.session_id} at {src}")
            return

        with zipfile.ZipFile(dst, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.write(src, os.path.basename(src))
        print(f"[common] Exported cache: {dst}")

    def export_session_cache_encrypted(self, password: str | None = None):
        """Encrypt and bundle this session’s cache with AES-256."""
        if not _HAS_PYZIPPER:
            raise RuntimeError("pyzipper not installed. Run `pip install pyzipper`.")

        password = password or get_local_encryption_key()
        src = self.config.get_session_cache_path(self.session_id) + ".sqlite"
        dst = self.config.get_cache_export_path(self.session_id)

        if not os.path.exists(src):
            print(f"[export] No cache found for session {self.session_id} at {src}")
            return

        with pyzipper.AESZipFile(
            dst, "w", compression=pyzipper.ZIP_DEFLATED, encryption=pyzipper.WZ_AES
        ) as zf:
            zf.setpassword(password.encode("utf-8"))
            zf.setencryption(pyzipper.WZ_AES, nbits=256)
            zf.write(src, os.path.basename(src))

        print(f"[common] Encrypted cache exported: {dst}")

# Static Utilities (Global Cache Import and Pruning)
def import_session_cache(zip_path: str, target_session_id: str, config: AppConfig, password: str | None = None):
    """Recreate a cache on another machine for QA inspection."""
    dst_base = config.get_session_cache_path(target_session_id)
    dst_dir = os.path.dirname(dst_base)
    password = password or get_local_encryption_key()

    try:
        with pyzipper.AESZipFile(zip_path, "r") as zf:
            zf.setpassword(password.encode("utf-8"))
            zf.extractall(dst_dir)
        print(f"[common] Imported encrypted cache into session: {target_session_id}")
    except (RuntimeError, pyzipper.zipfile.BadZipFile, TypeError):
        try:
            with zipfile.ZipFile(zip_path, "r") as zf:
                zf.extractall(dst_dir)
            print(f"[common] Imported (unencrypted) cache into session: {target_session_id}")
        except Exception as e:
            print(f"[common] Failed to import unencrypted cache: {e}")


def prune_old_logs(config: AppConfig, months: int = 6):
    """Remove audit logs older than N months."""
    base_dir = config.get_prune_log_dir()
    cutoff = datetime.datetime.utcnow() - datetime.timedelta(days=30 * months)

    if not os.path.exists(base_dir):
        return

    for folder in os.listdir(base_dir):
        try:
            # Prune based on session folder
            session_folder_path = os.path.join(base_dir, folder)
            if os.path.isdir(session_folder_path):
                # Try to find the log file to check its modified time
                log_file = os.path.join(session_folder_path, "chat_history.jsonl")
                if os.path.exists(log_file):
                    mtime = datetime.datetime.utcfromtimestamp(os.path.getmtime(log_file))
                    if mtime < cutoff:
                        shutil.rmtree(session_folder_path, ignore_errors=True)
                        print(f"[common] Pruned old log folder: {folder}")
        except Exception:
            # Ignore errors (like .DS_Store files) and continue
            continue


def prune_old_caches(config: AppConfig, months: int = 12):
    """Delete cache export ZIPs older than 12 months."""
    base_dir = config.get_prune_cache_dir()
    cutoff = datetime.datetime.utcnow() - datetime.timedelta(days=30 * months)

    if not os.path.exists(base_dir):
        return

    for file in os.listdir(base_dir):
        path = os.path.join(base_dir, file)
        if os.path.isfile(path):
            mtime = datetime.datetime.utcfromtimestamp(os.path.getmtime(path))
            if mtime < cutoff:
                os.remove(path)
                print(f"[common] Pruned old cache: {file}")
