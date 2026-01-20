import subprocess
import sys
import os
from pathlib import Path

def resource_path(relative):
    try:
        base = Path(sys._MEIPASS)
    except Exception:
        base = Path(__file__).parent
    return base / relative

def setup_environment():
    if getattr(sys, 'frozen', False):
        install_dir = Path(sys.executable).parent
    else:
        install_dir = Path(__file__).parent
    
    # Ollama
    ollama_path = install_dir / "ollama"
    if ollama_path.exists():
        os.environ["OLLAMA_MODELS"] = str(ollama_path / "models")
    
    # Tesseract
    tesseract_path = install_dir / "tesseract"
    if tesseract_path.exists():
        os.environ["PATH"] = str(tesseract_path) + os.pathsep + os.environ.get("PATH", "")
        os.environ["TESSDATA_PREFIX"] = str(tesseract_path / "tessdata")
    
    # Embedding model
    model_path = install_dir / "models" / "bge-small-en-v1.5"
    if model_path.exists():
        os.environ["EMB_MODEL_DIR"] = str(model_path)
    
    # Streamlit logs — FIXED
    logs_dir = Path(os.getenv("LOCALAPPDATA")) / "NorthAI" / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    os.environ["STREAMLIT_CACHE_DIR"] = str(logs_dir)
    os.environ["OFFLINE_MODE"] = "1"

def main():
    setup_environment()
    app = resource_path("offline_gpt_app.py")
    subprocess.run([sys.executable, "-m", "streamlit", "run", str(app)])

if __name__ == "__main__":
    main()
