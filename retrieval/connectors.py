# retrieval/connectors.py

"""
Privacy-conscious connectors for external data sources.

All network access is centralised here. Each connector respects
OFFLINE_MODE and outbound query filters. Higher-level code should
not call these directly; use retrieval.router.search_everything(...)
as the public entry point.
"""

from __future__ import annotations
from typing import List, Dict, Optional, Tuple

# Standard
import re
from urllib.parse import urlparse, urlsplit, parse_qs, unquote
from urllib.robotparser import RobotFileParser
from datetime import datetime, date, timedelta
from zoneinfo import ZoneInfo

# Third-party
import trafilatura
from ddgs import DDGS

# Local Module
from .common import unify, is_offline_mode, RetrievalManager

# Runtime guard
def _airgapped() -> bool:
    return is_offline_mode()

def _sanitize(manager: RetrievalManager, q: str) -> str:
    return manager._guard_text(q or "")

# Shared helper used by OpenAlex/Crossref/ArXiv ----
def _range_bounds(year_min: Optional[int], year_max: Optional[int]) -> Tuple[Optional[str], Optional[str]]:
    start = f"{year_min}-01-01" if year_min else None
    end   = f"{year_max}-12-31" if year_max else None
    return start, end

# Duckduckgo - Refactored for better ddgs library usage
def search_duckduckgo(manager: RetrievalManager, query: str, k: int = 8) -> List[Dict]:
    """
    Search using the ddgs library (successor to duckduckgo-search).
    
    Uses context manager for proper resource cleanup and explicit parameters
    for more reliable results.
    
    Args:
        manager: RetrievalManager instance for audit logging
        query: Search query string
        k: Maximum number of results to return
    
    Returns:
        List of unified result dictionaries
    """
    if is_offline_mode():
        return []
    
    q = manager._guard_text(query or "")
    if not q:
        return []
    
    # Site exclusions for better quality results
    minus = "-site:youtube.com -site:vimeo.com -site:reddit.com"
    full_query = f"{q} {minus}".strip()
    
    manager.write_audit_log("outbound", {
        "source": "duckduckgo", 
        "query_preview": q[:100]
    })
    
    try:
        # Use context manager for proper resource cleanup
        with DDGS() as ddgs:
            hits = list(ddgs.text(
                keywords=full_query,
                max_results=int(k)
            ))
    except Exception as e:
        manager.write_audit_log("inbound", {
            "source": "duckduckgo", 
            "count": 0, 
            "error": str(e)
        })
        return []
    
    out: List[Dict] = []
    for h in hits:
        # Extract URL - ddgs uses 'href' as the primary field
        url = h.get("href", "")
        if not url:
            continue
        
        # Extract title and body/snippet
        title = h.get("title", "")
        body = h.get("body", "")
        
        # Use title or fallback to body for the title field
        display_title = title if title else (body if body else "")
        if not display_title:
            continue
        
        out.append(unify(
            title=display_title,
            url=url,
            source="duckduckgo",
            authors=[],
            abstract=body,
            published=None
        ))
    
    # Log successful results
    if hits:
        manager.write_audit_log("inbound", {
            "source": "duckduckgo", 
            "count": len(hits), 
            "urls": [i["url"] for i in out]
        })
    
    return out


# News-specific search function
def search_duckduckgo_news(manager: RetrievalManager, query: str, k: int = 8, timelimit: Optional[str] = None) -> List[Dict]:
    """
    Search DuckDuckGo News using the ddgs library.
    
    Args:
        manager: RetrievalManager instance for audit logging
        query: Search query string
        k: Maximum number of results to return
        timelimit: Time filter - 'd' (day), 'w' (week), 'm' (month)
    
    Returns:
        List of unified result dictionaries with publication dates
    """
    if is_offline_mode():
        return []
    
    q = manager._guard_text(query or "")
    if not q:
        return []
    
    manager.write_audit_log("outbound", {
        "source": "duckduckgo_news", 
        "query_preview": q[:100]
    })
    
    try:
        with DDGS() as ddgs:
            hits = list(ddgs.news(
                keywords=q,
                max_results=int(k),
                timelimit=timelimit
            ))
    except Exception as e:
        manager.write_audit_log("inbound", {
            "source": "duckduckgo_news", 
            "count": 0, 
            "error": str(e)
        })
        return []
    
    out: List[Dict] = []
    for h in hits:
        url = h.get("url", "")
        if not url:
            continue

        title = h.get("title", "")
        body = h.get("body", "")
        date = h.get("date", None)  # News results include publication date
        
        display_title = title if title else (body if body else "")
        if not display_title:
            continue
        
        out.append(unify(
            title=display_title,
            url=url,
            source="duckduckgo_news",
            authors=[],
            abstract=body,
            published=date
        ))
    
    if hits:
        manager.write_audit_log("inbound", {
            "source": "duckduckgo_news",
            "count": len(hits),
            "urls": [i["url"] for i in out]
        })
    
    return out


# Wikipedia
def wikipedia_search(manager: RetrievalManager, query: str) -> List[Dict]:
    if _airgapped():
        return []
    q_preview = (query or "")[:100]

    manager.write_audit_log("outbound", {"source": "wikipedia-opensearch", "query_preview": q_preview})
    try:
        sr = manager.get(
            "https://en.wikipedia.org/w/api.php",
            params={"action": "opensearch", "search": query, "limit": 1, "namespace": 0, "format": "json"},
            api_name="wikipedia-opensearch",
        ).json()
    except Exception as e:
        manager.write_audit_log("inbound", {"source": "wikipedia-opensearch", "count": 0, "error": str(e)})
        return []

    if not sr or len(sr) < 2 or not sr[1]:
        manager.write_audit_log("inbound", {"source": "wikipedia-opensearch", "count": 0, "error": "No results"})
        return []

    title = sr[1][0]
    manager.write_audit_log("inbound", {"source": "wikipedia-opensearch", "count": 1, "success": True})

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

    page_url = ((s.get("content_urls", {}) or {}).get("desktop", {}) or {}).get("page")
    return [unify(s.get("title") or title, page_url, "wikipedia", [], s.get("extract"), s.get("timestamp"))]


# Geocoding
def openmeteo_geocode(manager: RetrievalManager, query: str) -> Optional[dict]:
    """Free geocoding with timezone from Open-Meteo."""
    if _airgapped():
        return None
    try:
        j = manager.get(
            "https://geocoding-api.open-meteo.com/v1/search",
            params={"name": query, "count": 1, "language": "en", "format": "json"},
            api_name="openmeteo-geocode",
        ).json()
        results = (j or {}).get("results") or []
        if not results:
            return None
        r = results[0]
        display = ", ".join([p for p in [r.get("name"), r.get("admin1"), r.get("country")] if p])
        return {
            "lat": float(r["latitude"]),
            "lon": float(r["longitude"]),
            "display_name": display or query,
            "timezone": r.get("timezone") or "UTC",
        }
    except Exception as e:
        manager.write_audit_log("inbound", {"source": "openmeteo-geocode", "count": 0, "error": str(e)})
        return None

# Open-Meteo daily forecast (structured)
def openmeteo_daily(manager: RetrievalManager, lat: float, lon: float) -> Optional[dict]:
    if _airgapped():
        return None
    try:
        j = manager.get(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude": lat, "longitude": lon,
                "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum",
                "timezone": "auto",
            },
            api_name="openmeteo",
        ).json()
        return j.get("daily")
    except Exception as e:
        manager.write_audit_log("inbound", {"source": "openmeteo", "count": 0, "error": str(e)})
        return None

def openmeteo_pick_day(daily: dict, iso_date: str) -> Optional[dict]:
    if not daily: return None
    times = daily.get("time") or []
    if iso_date not in times: return None
    i = times.index(iso_date)
    return {
        "date": iso_date,
        "tmax_c": daily["temperature_2m_max"][i],
        "tmin_c": daily["temperature_2m_min"][i],
        "precip_mm": (daily.get("precipitation_sum") or [0])[i] if daily.get("precipitation_sum") else 0.0,
        "provider": "openmeteo",
        "url": f"https://open-meteo.com/",
    }

# MET Norway (hourly compact) -> reduce to daily high/low/precip
def metno_hourly(manager: RetrievalManager, lat: float, lon: float) -> Optional[dict]:
    if _airgapped():
        return None
    try:
        j = manager.get(
            "https://api.met.no/weatherapi/locationforecast/2.0/compact",
            params={"lat": lat, "lon": lon},
            api_name="metno",
        ).json()
        return j  # hourly data in j["properties"]["timeseries"]
    except Exception as e:
        manager.write_audit_log("inbound", {"source": "metno", "count": 0, "error": str(e)})
        return None

def metno_reduce_day(hourly: dict, tz: ZoneInfo, iso_date: str) -> Optional[dict]:
    try:
        ts = (hourly.get("properties") or {}).get("timeseries") or []
        tmax = -1e9
        tmin = 1e9
        precip = 0.0
        found = False
        for row in ts:
            t_utc = datetime.fromisoformat(row["time"].replace("Z","+00:00"))
            t_loc = t_utc.astimezone(tz)
            if t_loc.date().isoformat() != iso_date:
                continue
            found = True
            details = (row.get("data", {}) or {}).get("instant", {}).get("details", {}) or {}
            t = details.get("air_temperature")
            if isinstance(t, (int, float)):
                tmax = max(tmax, t)
                tmin = min(tmin, t)
            # precipitation (if present) from next_1_hours/6_hours
            for key in ("next_1_hours", "next_6_hours"):
                block = (row.get("data", {}) or {}).get(key, {}) or {}
                sm = (block.get("details") or {}).get("precipitation_amount")
                if isinstance(sm, (int, float)):
                    precip += float(sm)
        if not found:
            return None
        return {
            "date": iso_date,
            "tmax_c": None if tmax == -1e9 else tmax,
            "tmin_c": None if tmin == 1e9 else tmin,
            "precip_mm": round(precip, 1),
            "provider": "metno",
            "url": "https://api.met.no/",
        }
    except Exception:
        return None

# Weather consensus (Open-Meteo + MET Norway)
def weather_consensus(manager: RetrievalManager, location_text: str, iso_date: Optional[str] = None) -> List[Dict]:
    """
    Build a high-confidence daily forecast for the location's local 'tomorrow'.
    Returns a list with a single unified weather record.
    """
    g = openmeteo_geocode(manager, location_text)
    if not g:
        return []

    tzname = g["timezone"] or "UTC"
    try:
        tz = ZoneInfo(tzname)
    except Exception:
        tz = ZoneInfo("UTC")

    # Compute 'tomorrow' in the location's timezone if caller didn't pass a date
    if not iso_date:
        today_loc = datetime.now(tz).date()
        iso_date = (today_loc + timedelta(days=1)).isoformat()

    lat, lon = g["lat"], g["lon"]

    om = openmeteo_pick_day(openmeteo_daily(manager, lat, lon), iso_date)
    mn = metno_reduce_day(metno_hourly(manager, lat, lon), tz, iso_date)

    if not om and not mn:
        return []

    if om and mn and isinstance(mn["tmax_c"], (int, float)) and isinstance(mn["tmin_c"], (int, float)):
        agree = abs(om["tmax_c"] - mn["tmax_c"]) <= 3 and abs(om["tmin_c"] - mn["tmin_c"]) <= 3
        tmax = round((om["tmax_c"] + mn["tmax_c"]) / 2, 1) if agree else om["tmax_c"]
        tmin = round((om["tmin_c"] + mn["tmin_c"]) / 2, 1) if agree else om["tmin_c"]
        precip = round(((om.get("precip_mm") or 0) + (mn.get("precip_mm") or 0)) / 2, 1) if agree else (om.get("precip_mm") or 0)
        provider = "openmeteo+metno" if agree else "openmeteo"
    else:
        base = om or mn
        tmax, tmin, precip = base["tmax_c"], base["tmin_c"], base.get("precip_mm") or 0
        provider = base["provider"]

    abstract = (
        f"{g['display_name']} – {iso_date}: high {tmax:.1f}°C, low {tmin:.1f}°C, "
        f"precipitation {precip:.1f} mm."
    )
    
    url = (om and om.get("url")) or (mn and mn.get("url")) or "https://open-meteo.com/"
    
    return [unify(  # <-- Changed to return list
        title=f"Weather forecast for {g['display_name']}",
        url=url,
        source="weather_consensus",
        authors=[],
        abstract=abstract,
        published=iso_date,
    )]

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
        rp = RobotFileParser(f"{parts.scheme}://{parts.netloc}/robots.txt")
        rp.read()
        return rp.can_fetch(user_agent, url)
    except Exception:
        return True  # best-effort permissive fallback


def fetch_page_text(manager: RetrievalManager, url: str, timeout: int = 15) -> str:
    if _airgapped():
        return ""
    safe_url = manager._guard_text(url or "")
    if not safe_url or not safe_url.startswith(("http://", "https://")):
        return ""
    if not _robots_allows(safe_url):
        return ""
    manager.write_audit_log("outbound", {"source": "fetch_page_text", "url": safe_url})
    try:
        r = manager.get(safe_url, timeout=timeout, api_name="html_fetch")
        html = r.text if r.ok else ""
        if not html:
            manager.write_audit_log("inbound", {"source": "fetch_page_text", "url": safe_url, "bytes": 0, "error": "Empty HTML content"})
            return ""
        txt = (trafilatura.extract(html) or "").strip()
        manager.write_audit_log("inbound", {"source": "fetch_page_text", "url": safe_url, "bytes": len(txt)})
        return txt
    except Exception as e:
        manager.write_audit_log("inbound", {"source": "fetch_page_text", "url": safe_url, "bytes": 0, "error": str(e)})
        return ""