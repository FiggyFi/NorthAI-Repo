# retrieval/router.py

"""
Query routing for academic and general search sources.

Provides heuristics to classify queries and functions to invoke
academic, general, or combined search providers with deduplication.
"""

# Language features
from __future__ import annotations
from typing import List, Dict, Optional

# Standard
import re

# Local modules
from . import connectors as C
from .common import RetrievalManager

# Heuristics
def looks_academic(q: str) -> bool:
    # This simple list is now robust, because "creative" queries
    # will never reach this function.
    # The 'safe_query' it receives will have keywords like "paper" or "study"
    # expanded from the privacy layer if they were in local docs.
    keys = ["paper","arxiv","journal","study","dataset","benchmark","neurips","iclr","acl","cvpr","icml","emnlp"]
    ql = (q or "").lower()
    return any(k in ql for k in keys)

# Academic search connectors
def search_academic(
    manager: RetrievalManager, 
    query: str, 
    year_min: Optional[int]=None, 
    year_max: Optional[int]=None
) -> List[Dict]:
    out: List[Dict] = []
    out += C.openalex_search(manager, query, per_page=30, year_min=year_min, year_max=year_max)
    out += C.crossref_search(manager, query, rows=30,   year_min=year_min, year_max=year_max)
    out += C.arxiv_search(manager, query,  max_results=50, year_min=year_min, year_max=year_max)
    out += C.pubmed_search(manager, query)
    out += C.europepmc_search(manager, query)
    out += C.biorxiv_medrxiv_search(manager, query)
    return out

# General search connectors
def search_general(
    manager: RetrievalManager,
    query: str
) -> List[Dict]:
    out: List[Dict] = []
    out += C.search_duckduckgo(manager, query)
    out += C.wikipedia_search(manager, query)
    out += C.gdelt_search(manager, query)
    return out

# Combined search
def search_everything(
    manager: RetrievalManager, 
    query: str, # This is the 'safe_query' from the app
    year_min: Optional[int]=None, 
    year_max: Optional[int]=None
) -> List[Dict]:
    items = []

    # --- THIS IS THE CORRECTED STAGE 2 FIX ---
    if looks_academic(query):
        # Academic query: Search academic sources first, then general web.
        items += search_academic(manager, query, year_min, year_max)
        items += search_general(manager, query)
    else:
        # General query ("weather") OR Enterprise query ("sales manual"):
        # Search ONLY the general web.
        items += search_general(manager, query)
    # --- END OF FIX ---

    # de-dupe by URL
    seen, uniq = set(), []
    for r in items:
        u = r.get("url")
        if not u or u in seen: continue
        seen.add(u); uniq.append(r)
    return uniq