# Offline GPT Application

This repository implements a self-contained retrieval-augmented generation (RAG) system designed for offline or air-gapped environments.  
It integrates a local large language model (**Llama 3 8B**) with a vector retrieval pipeline based on **bge-small-en-v1.5** and **ChromaDB**.

---

## 1. Overview

The application enables secure, local document ingestion and retrieval without any external API calls or network dependencies.  
It supports private data analysis, research assistance, and technical search under full offline operation.

---

## 2. Core Architecture

| Module | Function |
|:--|:--|
| `offline_gpt_app.py` | Streamlit entry point for the interactive interface. |
| `app_bootstrap.py` | Performs environment and model integrity checks at startup. |
| `vector_store.py` | Handles local storage, embedding, and retrieval using ChromaDB and SentenceTransformers. |
| `parsers.py` | Offline parsers for common document types (PDF, DOCX, RTF, PPTX, CSV, XLSX, JSON, images). |
| `privacy.py` | Provides query redaction and local-only keyword expansion. |
| `router.py` | Directs queries between academic and general connectors. |
| `connectors.py` | Defines all network data connectors with offline guards. |
| `common.py` | Centralised request caching and offline mode enforcement. |
| `profiles.py` | Manages user profile data in the local file system. |

---

## 3. Models

- **Base model:** `llama3:8b`  
- **Embedding model:** `bge-small-en-v1.5`

Both models must reside in the local `./models/` directory.  
The embedding model is loaded directly from disk and never fetched remotely.

---

## 4. Requirements

- Python ≥ 3.9  
- 16 GB RAM minimum (GPU optional but recommended for embeddings)  
- Local storage access for model and database directories  
- No external connectivity required once installed

---

## 5. Directory Layout
project_root/
├── models/
│ └── bge-small-en-v1.5/
├── web_rag_db/
│ └── sessions/
├── retrieval/
│ ├── init.py
│ ├── common.py
│ ├── connectors.py
│ ├── router.py
│ ├── privacy.py
│ └── parsers.py
├── offline_gpt_app.py
├── vector_store.py
├── app_bootstrap.py
└── profiles.py

---

## 6.1 Environment Setup
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt

## 6.2 Health Check
python -m app_bootstrap

✅ Embedding model
✅ Chroma path
✅ Airplane mode default


## 6.3 Launch
streamlit run offline_gpt_app.py

---

## 7. Data Handling

- Up to **10 files** can be ingested per session.  
- **Supported types:** `.pdf`, `.docx`, `.rtf`, `.pptx`, `.csv`, `.xlsx`, `.txt`, `.json`, `.png`, `.jpg`.  
- Parsed text is chunked and embedded into a Chroma collection unique to each session.  
- Duplicate detection uses SHA-1 hashing of content.

---

## 8. Privacy and Security

- **Offline by default:** `AIRPLANE_MODE=1` prevents all outbound requests.  
- **Local expansion:** Query augmentation uses locally derived keywords only.  
- **Redaction:** Any 32-character substring found in local documents is removed before transmission if online mode is enabled.

---

## 9. Retrieval

The system performs dual-source retrieval:

- **`local_docs`:** user-uploaded or ingested content  
- **`web_docs`:** cached or online data when permitted  

Retrieval uses semantic embedding similarity with configurable top-K results.

---

## 10. Maintenance Notes

- Session databases are stored in `web_rag_db/sessions/`.  
  Remove these to clear retrieval history.  
- Profiles are saved as JSON under  
  `C:/Users/<username>/OneDrive/NorthAI_Profiles/`.  
- Model updates require replacing the folder  
  `models/bge-small-en-v1.5/`.
