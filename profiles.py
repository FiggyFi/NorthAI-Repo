from pathlib import Path
import json
import os

# Base per-user data dir, consistent with app_bootstrap/vector_store
LOCAL_DATA_DIR = Path(os.getenv("LOCALAPPDATA")) / "NorthAI"
LOCAL_DATA_DIR.mkdir(parents=True, exist_ok=True)

# Profiles & sessions DB under LOCALAPPDATA
PROFILE_DIR = LOCAL_DATA_DIR / "profiles"
PROFILE_DIR.mkdir(parents=True, exist_ok=True)

# Where to store the sessions metadata
_SESSIONS_DB_PATH = PROFILE_DIR / "sessions_db.json"

def list_profiles():
    return [f.stem.replace("profile_", "") for f in PROFILE_DIR.glob("profile_*.json")]

def profile_path(user_id: str) -> Path:
    return PROFILE_DIR / f"profile_{user_id}.json"

def load_profile(user_id: str = "default") -> dict:
    f = profile_path(user_id)
    if f.exists():
        try:
            return json.loads(f.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {"version": 1, "user_id": user_id, "identity": {}, "preferences": {}, "facts": []}

def save_profile(profile: dict) -> None:
    user_id = profile.get("user_id", "default")
    f = profile_path(user_id)
    f.write_text(json.dumps(profile, ensure_ascii=False, indent=2), encoding="utf-8")

# Define the single file where all chat sessions will be stored
def load_sessions() -> dict:
    """
    Load the entire sessions dictionary from the 'sessions_db.json' file.
    Returns an empty dict if the file doesn't exist or is corrupt.
    """
    if _SESSIONS_DB_PATH.exists():
        try:
            return json.loads(_SESSIONS_DB_PATH.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"[sessions] Error loading sessions file: {e}")
            pass
    # Return an empty dictionary if no file or error
    return {}

def save_sessions(sessions_dict: dict) -> None:
    """
    Save the entire sessions dictionary to 'sessions_db.json'.
    This is called every time a session is created, deleted, or updated.
    """
    try:
        _SESSIONS_DB_PATH.write_text(
            json.dumps(sessions_dict, ensure_ascii=False, indent=2),
            encoding="utf-8"
        )
    except Exception as e:
        print(f"[sessions] Error saving sessions file: {e}")