# retrieval/connectors.py

"""
Privacy-conscious connectors for external data sources.

All network access is centralised here. Each connector respects
AIRPLANE_MODE and outbound query filters. Higher-level code should
not call these directly; use retrieval.router.search_everything(...)
as the public entry point.
"""

# Language features
from __future__ import annotations
from typing import List, Dict, Optional, Tuple

# Standard
import os
import re
from urllib.parse import urlparse
from urllib.robotparser import RobotFileParser



# Third-party packages
import requests
import trafilatura
from duckduckgo_search import DDGS

# Single source of truth for airplane mode + guards
from .common import unify, is_airplane_mode, RetrievalManager


# Runtime guards & utilities
def _airgapped() -> bool:
    """True iff airplane mode is currently ON (runtime)."""
    return is_airplane_mode()

def _sanitize(manager: RetrievalManager, q: str) -> str:
    """Apply outbound text filter if available; otherwise pass-through."""
    # Use the manager's guard_text method
    return manager._guard_text(q or "")


# General search (DuckDuckGo) 
def search_duckduckgo(
    manager: RetrievalManager,
    query: str, 
    k: int = 8
) -> List[Dict]:
    """Privacy-aware web search via DuckDuckGo (metadata only)."""
    if _airgapped():
        return []
    q = _sanitize(manager, query)
    minus = "-site:youtube.com -site:vimeo.com -site:reddit.com"
    manager.write_audit_log("outbound", {"source": "duckduckgo", "query_preview": q[:100]})
    try:
        hits = list(DDGS().text(f"{q} {minus}".strip(), max_results=int(k)))
    except Exception as e:
        manager.write_audit_log("inbound", {"source": "duckduckgo", "count": 0, "error": str(e)})
        hits = []
    out: List[Dict] = []
    for h in hits:
        url = h.get("href") or h.get("url")
        if not url:
            continue
        title = h.get("title") or h.get("body") or ""
        out.append(unify(title, url, "duckduckgo", [], h.get("body"), None))
    urls = [item['url'] for item in out] # Get URLs from the 'out' list
    if hits: # Only log success if we didn't log an error
        manager.write_audit_log("inbound", {"source": "duckduckgo", "count": len(hits), "urls": urls})
    return out


# Wikipedia (best-title + summary)
def wikipedia_search(
    manager: RetrievalManager,
    query: str
) -> List[Dict]:
    """Get the best Wikipedia page title then fetch its REST summary."""
    if _airgapped():
        return []
    
    q_preview = (query or "")[:100]
    
    # First API call: opensearch (to find the page title)
    manager.write_audit_log("outbound", {"source": "wikipedia-opensearch", "query_preview": q_preview})
    try:
        sr = manager.get(
            "https://en.wikipedia.org/w/api.php",
            params={"action": "opensearch", "search": query, "limit": 1, "namespace": 0, "format": "json"},
            api_name="wikipedia-opensearch",
        ).json()
    except Exception as e:
        manager.write_audit_log("inbound", {"source": "wikipedia-opensearch", "count": 0, "error": str(e)})
        sr = None

    # Check if the first call failed or returned empty
    if not sr or len(sr) < 2 or not sr[1]:
        if sr is not None: # Log if the request succeeded but found no results
            manager.write_audit_log("inbound", {"source": "wikipedia-opensearch", "count": 0, "error": "No results found"})
        return []

    # Log success for the first call and get the title
    manager.write_audit_log("inbound", {"source": "wikipedia-opensearch", "count": 1, "success": True})
    title = sr[1][0]

    
    # Second API call: summary (to get the content)
    manager.write_audit_log("outbound", {"source": "wikipedia-summary", "title": title})
    try:
        s = manager.get(
            f"https://en.wikipedia.org/api/rest_v1/page/summary/{title}",
            api_name="wikipedia-summary",
        ).json()
        
        manager.write_audit_log("inbound", {"source": "wikipedia-summary", "title": title, "success": True})
        
    except Exception as e:
        manager.write_audit_log("inbound", {"source": "wikipedia-summary", "title": title, "success": False, "error": str(e)})
        return []
        
    # Now, unifation and returning of the final result
    url = (s.get("content_urls", {}) or {}).get("desktop", {}) or {}
    page_url = url.get("page")
    return [unify(s.get("title") or title, page_url, "wikipedia", [], s.get("extract"), s.get("timestamp"))]


# Academic: keyless
def openalex_search(
    manager: RetrievalManager,
    query: str,
    per_page: int = 20,
    year_min: Optional[int] = None,
    year_max: Optional[int] = None,
) -> List[Dict]:
    if _airgapped():
        return []
    
    params = {"search": query, "per-page": per_page}
    start, end = _range_bounds(year_min, year_max)
    if start:
        params["from_publication_date"] = start
    if end:
        params["to_publication_date"] = end

    # Outbound Log
    manager.write_audit_log(
        "outbound", 
        {"source": "openalex", "query_preview": (query or "")[:100]}
    )

    try:
        r = manager.get(
            "https://api.openalex.org/works", 
            params=params, 
            api_name="openalex"
        ).json()
    except Exception as e:
        manager.write_audit_log(
            "inbound", 
            {"source": "openalex", "count": 0, "error": str(e)}
        )
        r = {}

    results = r.get("results", [])
    
    # Inbound Log (Success)
    if not r.get("error"): # Log success if no error was caught
        manager.write_audit_log(
            "inbound", 
            {"source": "openalex", "count": len(results)}
        )

    out: List[Dict] = []
    for w in results:
        authors = [a["author"]["display_name"] for a in w.get("authorships", []) if "author" in a]
        out.append(unify(
            w.get("title"), w.get("id"), "openalex",
            authors, w.get("abstract"), w.get("publication_date"), w.get("doi"), {"raw": w}
        ))
    return out

def crossref_search(
    manager: RetrievalManager,  # <-- ADD manager
    query: str,
    rows: int = 20,
    year_min: Optional[int] = None,
    year_max: Optional[int] = None,
) -> List[Dict]:
    if _airgapped():
        return []
    
    params = {"query": query, "rows": rows}
    start, end = _range_bounds(year_min, year_max)
    filt = []
    if start:
        filt.append(f"from-pub-date:{start}")
    if end:
        filt.append(f"until-pub-date:{end}")
    if filt:
        params["filter"] = ",".join(filt)
    manager.write_audit_log(
        "outbound", 
        {"source": "crossref", "query_preview": (query or "")[:100]}
    )
    try:
        r = manager.get("https://api.crossref.org/works", params=params, api_name="crossref").json()
    except Exception as e:
        manager.write_audit_log(
            "inbound", 
            {"source": "crossref", "count": 0, "error": str(e)}
        )
        r = {}

    items = r.get("message", {}).get("items", [])
    if not r.get("error"): # Log success if no error was caught
        manager.write_audit_log(
            "inbound", 
            {"source": "crossref", "count": len(items)}
        )

    out: List[Dict] = []
    for it in items:
        title = " ".join(it.get("title", []) or [])
        doi   = it.get("DOI")
        url   = it.get("URL") or (f"https://doi.org/{doi}" if doi else None)
        authors: List[str] = []
        for a in it.get("author", []) or []:
            nm = " ".join([a.get("given",""), a.get("family","")]).strip()
            if nm:
                authors.append(nm)
        pub = None
        if it.get("published-print", {}).get("date-parts"):
            ymd = it["published-print"]["date-parts"][0]
            pub = "-".join(map(str, ymd))
        elif it.get("published-online", {}).get("date-parts"):
            ymd = it["published-online"]["date-parts"][0]
            pub = "-".join(map(str, ymd))
        out.append(unify(title, url, "crossref", authors, None, pub, doi))
    return out

def arxiv_search(
    manager: RetrievalManager,  # <-- ADD manager
    query: str,
    max_results: int = 20,
    year_min: Optional[int] = None,
    year_max: Optional[int] = None,
) -> List[Dict]:
    if _airgapped():
        return []
    manager.write_audit_log(
        "outbound", 
        {"source": "arxiv", "query_preview": (query or "")[:100]}
    )

    try:
        r = manager.get(
            "http://export.arxiv.org/api/query",
            params={"search_query": query, "max_results": max_results},
            api_name="arxiv",
        )
        xml = r.text
    except Exception as e:
        manager.write_audit_log(
            "inbound", 
            {"source": "arxiv", "count": 0, "error": str(e)}
        )
        xml = ""

    if "<entry>" not in xml:
        if xml != "": 
            manager.write_audit_log(
                "inbound", 
                {"source": "arxiv", "count": 0, "error": "No <entry> tags in response"}
            )
        return []

    entries = xml.split("<entry>")[1:]
    manager.write_audit_log(
        "inbound", 
        {"source": "arxiv", "count": len(entries)}
    )
    out: List[Dict] = []
    for e in entries:
        title = (re.search(r"<title>(.*?)</title>", e, re.S) or [None]).group(1).strip() if re.search(r"<title>(.*?)</title>", e, re.S) else ""
        link  = (re.search(r'<link rel="alternate" type="text/html" href="(.*?)"/>', e) or [None]).group(1) if re.search(r'<link rel="alternate" type="text/html" href="(.*?)"/>', e) else None
        pub   = (re.search(r"<published>(.*?)</published>", e) or [None]).group(1) if re.search(r"<published>(.*?)</published>", e) else None
        abs_  = (re.search(r"<summary>(.*?)</summary>", e, re.S) or [None]).group(1).strip() if re.search(r"<summary>(.*?)</summary>", e, re.S) else None
        authors = re.findall(r"<name>(.*?)</name>", e)

        # post-filter by publication year for arXiv
        ok = True
        if pub and (year_min or year_max):
            try:
                y = int(pub[:4])
                if year_min and y < year_min:
                    ok = False
                if year_max and y > year_max:
                    ok = False
            except Exception:
                pass
        if ok:
            out.append(unify(title, link, "arxiv", authors, abs_, pub))
    return out

def pubmed_search(
    manager: RetrievalManager,
    query: str, 
    retmax: int = 20
) -> List[Dict]:
    if _airgapped():
        return []
    manager.write_audit_log(
        "outbound", 
        {"source": "pubmed-esearch", "query_preview": (query or "")[:100]}
    )
    try:
        es = manager.get(
            "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi",
            params={"db": "pubmed", "term": query, "retmode": "json", "retmax": retmax},
            api_name="pubmed-esearch",
        ).json()
    except Exception as e:
        manager.write_audit_log(
            "inbound", 
            {"source": "pubmed-esearch", "count": 0, "error": str(e)}
        )
        es = {}

    ids = es.get("esearchresult", {}).get("idlist", [])
    if not es.get("error"):
         manager.write_audit_log(
            "inbound", 
            {"source": "pubmed-esearch", "count": len(ids)}
        ) 
    if not ids:
        return []
    manager.write_audit_log(
        "outbound", 
        {"source": "pubmed-esummary", "id_count": len(ids)}
    )
    
    try:
        sm = manager.get(
            "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi",
            params={"db": "pubmed", "id": ",".join(ids), "retmode": "json"},
            api_name="pubmed-esummary",
        ).json().get("result", {})
    except Exception as e:
        manager.write_audit_log(
            "inbound", 
            {"source": "pubmed-esummary", "count": 0, "error": str(e)}
        )
        sm = {}
    if not sm.get("error"):
        manager.write_audit_log(
            "inbound", 
            {"source": "pubmed-esummary", "count": len(sm.items())}
        )  
    out: List[Dict] = []
    for pid, rec in sm.items():
        if pid == "uids":
            continue
        authors = [a.get("name") for a in rec.get("authors", []) if a.get("name")]
        out.append(unify(
            rec.get("title"),
            f"https://pubmed.ncbi.nlm.nih.gov/{pid}/",
            "pubmed",
            authors,
            None,
            rec.get("pubdate"),
            rec.get("elocationid"),
        ))
    return out


def europepmc_search(
    manager: RetrievalManager,
    query: str, 
    pageSize: int = 20
) -> List[Dict]:
    if _airgapped():
        return []
    manager.write_audit_log(
        "outbound", 
        {"source": "europepmc", "query_preview": (query or "")[:100]}
    )
    try:
        r = manager.get(
            "https://www.ebi.ac.uk/europepmc/webservices/rest/search",
            params={"query": query, "format": "json", "pageSize": pageSize},
            api_name="europepmc",
        ).json()
    except Exception as e:
        manager.write_audit_log(
            "inbound", 
            {"source": "europepmc", "count": 0, "error": str(e)}
        )
        r = {}
    results = r.get("resultList", {}).get("result", [])
    if not r.get("error"):
         manager.write_audit_log(
            "inbound", 
            {"source": "europepmc", "count": len(results)}
        )
    out: List[Dict] = []
    for it in results:
        authors = [a.strip() for a in (it.get("authorString", "").split(",") if it.get("authorString") else [])]
        url = f"https://europepmc.org/abstract/{it.get('source')}/{it.get('id')}"
        out.append(unify(
            it.get("title"),
            url,
            "europepmc",
            authors,
            it.get("abstractText"),
            it.get("firstPublicationDate"),
            it.get("doi"),
        ))
    return out


def biorxiv_medrxiv_search(
    manager: RetrievalManager,
    query: str, 
    pageSize: int = 20
) -> List[Dict]:
    """Europe PMC filter that targets bioRxiv/medRxiv preprints."""
    return europepmc_search(
        manager, 
        f"({query}) AND (SRC:biorxiv OR SRC:medrxiv)", 
        pageSize=pageSize
    )


def gdelt_search(
    manager: RetrievalManager,
    query: str, 
    max_records: int = 20
) -> List[Dict]:
    if _airgapped():
        return []
    manager.write_audit_log(
        "outbound", 
        {"source": "gdelt", "query_preview": (query or "")[:100]}
    )
    try:
        r = manager.get(
            "http://api.gdeltproject.org/api/v2/doc/doc",
            params={"query": query, "format": "json", "maxrecords": max_records},
            api_name="gdelt",
        ).json()
    except Exception as e:
        manager.write_audit_log(
            "inbound", 
            {"source": "gdelt", "count": 0, "error": str(e)}
        )
        r = {}
    articles = r.get("articles", [])
    if not r.get("error"):
        manager.write_audit_log(
            "inbound", 
            {"source": "gdelt", "count": len(articles)}
        ) 
    out: List[Dict] = []
    for d in articles:
        title = d.get("title")
        url = d.get("url")
        authors: List[str] = []
        abstract = d.get("seendate")
        published = d.get("seendate")
        out.append(unify(title, url, "gdelt", authors, abstract, published))
    return out

#Helper functions
def _robots_allows(url: str, user_agent: str = "*") -> bool:
    try:
        parts = urlparse(url)
        robots_url = f"{parts.scheme}://{parts.netloc}/robots.txt"
        rp = RobotFileParser(robots_url)
        rp.read()
        return rp.can_fetch(user_agent, url)
    except Exception:
        return True  # best-effort permissive fallback

def fetch_page_text(
    manager: RetrievalManager,  # <-- Stays the same
    url: str, 
    timeout: int = 15
) -> str:
    """
    Fetch & extract readable text from a web page.
    - Honors airplane mode (returns "").
    - Sanitizes outbound URL via privacy filter.
    - Respects robots.txt (best-effort).
    - Uses manager.get() for caching, headers, and auditing.
    """
    if _airgapped():
        return ""
    safe_url = manager._guard_text(url or "")
    if not safe_url or not safe_url.startswith(("http://", "https://")):
        return ""
    if not _robots_allows(safe_url):
        return "" # Respect robots.txt
    manager.write_audit_log("outbound", {"source": "fetch_page_text", "url": safe_url})
    try:
        r = manager.get(safe_url, timeout=timeout, api_name="html_fetch")
        html = r.text if r.ok else ""
        if not html:
            manager.write_audit_log(
                "inbound", 
                {"source": "fetch_page_text", "url": safe_url, "bytes": 0, "error": "Empty HTML content"}
            )
            return ""
        txt = (trafilatura.extract(html) or "").strip()
        manager.write_audit_log(
            "inbound", 
            {"source": "fetch_page_text", "url": safe_url, "bytes": len(txt)}
        )
        return txt
    except Exception as e:
        # --- ADDED: Step 5 Inbound Log (Failure) ---
        manager.write_audit_log(
            "inbound", 
            {"source": "fetch_page_text", "url": safe_url, "bytes": 0, "error": str(e)}
        )
        return ""
    
def _range_bounds(year_min: Optional[int], year_max: Optional[int]) -> Tuple[Optional[str], Optional[str]]:
    start = f"{year_min}-01-01" if year_min else None
    end   = f"{year_max}-12-31" if year_max else None
    return start, end