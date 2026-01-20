# file: retrieval/retrieval_router.py  (final, drop-in)

from __future__ import annotations
from typing import List, Dict, Optional
import re

from .common import RetrievalManager
from .connectors import (
    weather_consensus,      # returns List[Dict] of unify(...) records
    wikipedia_search,
    gdelt_search,
    openalex_search, crossref_search, arxiv_search, pubmed_search, europepmc_search,
    search_duckduckgo,
)

# Heuristics
_RX_ACADEMIC = re.compile(r"\b(paper|study|arxiv|doi|journal|citation|benchmark|dataset)\b", re.I)
_RX_WEATHER  = re.compile(r"\b(weather|forecast|temperature|rain|snow|precip|wind)\b", re.I)
_RX_NEWS     = re.compile(r"\b(news|headline|breaking|today)\b", re.I)

def looks_academic(q: str) -> bool:
    return bool(_RX_ACADEMIC.search(q or ""))

def looks_weather(q: str) -> bool:
    return bool(_RX_WEATHER.search(q or ""))

def looks_news(q: str) -> bool:
    return bool(_RX_NEWS.search(q or ""))

# ----- Academic search -----
def search_academic(
    manager: RetrievalManager,
    query: str,
    year_min: Optional[int] = None,
    year_max: Optional[int] = None,
) -> List[Dict]:
    items: List[Dict] = []
    try: items += openalex_search(manager, query, year_min=year_min, year_max=year_max) or []
    except Exception: pass
    try: items += crossref_search(manager, query, year_min=year_min, year_max=year_max) or []
    except Exception: pass
    try: items += arxiv_search(manager, query, year_min=year_min, year_max=year_max) or []
    except Exception: pass
    try: items += pubmed_search(manager, query) or []
    except Exception: pass
    try: items += europepmc_search(manager, query) or []
    except Exception: pass
    return items

# ----- General search -----
def search_general(manager: RetrievalManager, query: str) -> List[Dict]:
    items: List[Dict] = []

    # 1) Weather (structured) — use consensus directly; do NOT re-wrap
    if looks_weather(query):
        try:
            wx = weather_consensus(manager, query) or []
            if wx:
                # If we got structured weather data, use it but also allow context fallback
                items += wx
            else:
                print("[DEBUG] Weather consensus empty, falling back to DDG.")
        except Exception as e:
            print(f"[DEBUG] weather_consensus error: {e!r}")

    # 2) Wikipedia (factoid/entity) then GDELT (news)
    try:
        wk = wikipedia_search(manager, query) or []
        if wk: items += wk
    except Exception:
        pass
    if looks_news(query):
        try:
            news = gdelt_search(manager, query, max_records=20) or []
            if news: items += news
        except Exception:
            pass

    # 3) Fallback: DDG
    if not items:
        try:
            ddg = search_duckduckgo(manager, query, k=8) or []
            if ddg: items += ddg
        except Exception:
            pass

    return items

# ----- Public entrypoint -----
def search_everything(
    manager: RetrievalManager,
    query: str,
    year_min: Optional[int] = None,
    year_max: Optional[int] = None,
) -> List[Dict]:
    items: List[Dict] = []

    if looks_academic(query):
        items += search_academic(manager, query, year_min, year_max)
        items += search_general(manager, query)
    else:
        items += search_general(manager, query)

    # De-duplicate by URL
    seen, uniq = set(), []
    for r in items:
        u = r.get("url")
        if not u or u in seen: 
            continue
        seen.add(u)
        uniq.append(r)
    return uniq
