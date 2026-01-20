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
from datetime import datetime, date, timedelta, UTC
from zoneinfo import ZoneInfo

# Third-party
try:
    import trafilatura
except ImportError:
    trafilatura = None

from bs4 import BeautifulSoup
from lxml.html import fromstring
from readability import Document
from ddgs import DDGS

try:
    from selectolax.parser import HTMLParser
except ImportError:
    HTMLParser = None  # graceful degrade

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

# Duckduckgo 
def search_duckduckgo(manager: RetrievalManager, query: str, k: int = 8) -> List[Dict]:
    """
    Web search via DuckDuckGo using the new `ddgs` library.

    Args:
        manager: RetrievalManager instance for logging/audit.
        query: Search query string.
        k: Max number of results.

    Returns:
        List of unified search result dictionaries.
    """
    if is_offline_mode():
        print("[DDG] Offline mode - skipping search")
        return []

    q = manager._guard_text(query or "")
    if not q:
        print("[DDG] Empty query after sanitization")
        return []

    manager.write_audit_log("outbound", {
        "source": "duckduckgo",
        "query_preview": q[:100]
    })

    try:
        from ddgs import DDGS
        print(f"[DDG] Searching for: {q[:100]}")

        # ddgs must be used as a context manager for connection cleanup
        with DDGS() as ddgs:
            results = list(ddgs.text(q, max_results=k))

        print(f"[DDG] Got {len(results)} raw results")

        if not results:
            manager.write_audit_log("inbound", {
                "source": "duckduckgo",
                "count": 0,
                "note": "No results returned"
            })
            return []

        out: List[Dict] = []
        for hit in results:
            # ddgs returns keys: 'title', 'href', 'body'
            title = (hit.get("title") or "").strip()
            url = (hit.get("href") or "").strip()
            body = (hit.get("body") or "").strip()

            if not url or not title:
                continue

            print(f"[DDG] Found: {title[:60]}... at {url[:50]}")
            out.append(unify(
                title=title,
                url=url,
                source="duckduckgo",
                authors=[],
                abstract=body,
                published=None
            ))

        manager.write_audit_log("inbound", {
            "source": "duckduckgo",
            "count": len(out),
            "urls": [r["url"] for r in out[:3]]
        })

        print(f"[DDG] Successfully parsed {len(out)} results")
        return out

    except ImportError:
        err = "The `ddgs` package is missing. Install it via `pip install ddgs`."
        print(f"[DDG] ERROR: {err}")
        manager.write_audit_log("inbound", {
            "source": "duckduckgo", "count": 0, "error": err
        })
        return []

    except Exception as e:
        err = f"{type(e).__name__}: {str(e)}"
        print(f"[DDG] ERROR: {err}")
        manager.write_audit_log("inbound", {
            "source": "duckduckgo", "count": 0, "error": err
        })
        return []


# --- Fallback HTML scraper ---
def search_duckduckgo_html(manager: RetrievalManager, query: str, k: int = 8) -> List[Dict]:
    """
    Backup search if ddgs fails.
    Uses direct HTML parsing of DuckDuckGo results page.
    """
    if is_offline_mode():
        return []

    q = manager._guard_text(query or "")
    if not q:
        return []

    try:
        import re
        from urllib.parse import quote_plus, unquote

        print(f"[DDG-HTML] Attempting HTML scrape for: {q[:100]}")
        url = f"https://html.duckduckgo.com/html/?q={quote_plus(q)}"

        response = manager.get(url, timeout=10, api_name="duckduckgo-html")
        html = response.text

        pattern = (
            r'<a rel="nofollow" class="result__a" href="([^"]+)">'
            r'([^<]+)</a>.*?<a class="result__snippet" href="[^"]+">([^<]+)</a>'
        )

        matches = re.findall(pattern, html, re.DOTALL)
        results = []

        for encoded, title, snippet in matches[:k]:
            match_url = re.search(r'uddg=([^&]+)', encoded)
            if not match_url:
                continue
            decoded_url = unquote(match_url.group(1))

            results.append(unify(
                title=title.strip(),
                url=decoded_url,
                source="duckduckgo-html",
                authors=[],
                abstract=snippet.strip(),
                published=None
            ))

        print(f"[DDG-HTML] Scraped {len(results)} results")
        return results

    except Exception as e:
        print(f"[DDG-HTML] Failed: {e}")
        return []
    
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
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from typing import Optional, List, Dict

def weather_consensus(manager: RetrievalManager, location_text: str, iso_date: Optional[str] = None) -> List[Dict]:
    """
    Build a high-confidence daily forecast for the requested date.
    Returns a list with a single unified weather record.
    """
    g = openmeteo_geocode(manager, location_text)
    if not g:
        return []

    tzname = g.get("timezone") or "UTC"
    try:
        tz = ZoneInfo(tzname)
    except Exception:
        tz = ZoneInfo("UTC")

    if not iso_date:
        # Explicitly require date — no default behavior
        raise ValueError("iso_date must be provided; no default date is assumed.")

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

    return [unify(
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
    """
    Multi-layer HTML text extractor:
    1. readability-lxml
    2. selectolax or BeautifulSoup
    3. trafilatura or fallback reparse
    """
    if _airgapped():
        return ""

    safe_url = manager._guard_text(url or "")
    if not safe_url.startswith(("http://", "https://")) or not _robots_allows(safe_url):
        return ""

    manager.write_audit_log("outbound", {"source": "fetch_page_text", "url": safe_url})

    try:
        r = manager.get(safe_url, timeout=timeout, api_name="html_fetch")
        html = r.text if r.ok else ""
        if not html:
            manager.write_audit_log("inbound", {
                "source": "fetch_page_text",
                "url": safe_url,
                "bytes": 0,
                "error": "Empty HTML content",
            })
            return ""

        text_out = None
        parser_used = "none"

        # --- Stage 1: readability-lxml ---
        try:
            doc = Document(html)
            cleaned = doc.summary(html_partial=False)
            tree = fromstring(cleaned)
            text_out = " ".join(tree.xpath("//text()")).strip()
            if len(text_out) > 200:
                parser_used = "readability-lxml"
                raise StopIteration  # short-circuit to logging
        except StopIteration:
            pass
        except Exception:
            text_out = None

        # --- Stage 2: selectolax or BeautifulSoup ---
        if not text_out:
            if HTMLParser:
                try:
                    tree = HTMLParser(html)
                    text_out = " ".join(
                        node.text(strip=True)
                        for node in tree.css("body *")
                        if node.text(strip=True)
                    )
                    if len(text_out) > 50:
                        parser_used = "selectolax"
                except Exception:
                    text_out = None

            if not text_out:
                try:
                    soup = BeautifulSoup(html, "html.parser")
                    for tag in soup(["script", "style", "noscript", "header", "footer", "nav"]):
                        tag.decompose()
                    text_out = " ".join(t.strip() for t in soup.stripped_strings)
                    if len(text_out) > 50:
                        parser_used = "beautifulsoup"
                except Exception:
                    text_out = None

        # --- Stage 3: trafilatura or fallback reparse ---
        if not text_out or len(text_out) < 200:
            cleaned = None

            # Try trafilatura first if available
            if trafilatura:
                try:
                    cleaned = trafilatura.extract(
                        html,
                        include_comments=False,
                        include_tables=False,
                        favor_recall=True
                    )
                    if cleaned:
                        parser_used = "trafilatura"
                except Exception:
                    cleaned = None

            # If trafilatura failed, retry readability summary + soup
            if not cleaned:
                try:
                    readable = Document(html).summary()
                    soup = BeautifulSoup(readable, "lxml")
                    cleaned = soup.get_text(separator=" ", strip=True)
                    if cleaned:
                        parser_used = "readability-fallback"
                except Exception:
                    cleaned = None

            # Final fallback: selectolax paragraph extraction
            if not cleaned and HTMLParser:
                try:
                    parser = HTMLParser(html)
                    paragraphs = [p.text(strip=True) for p in parser.tags("p") if p.text(strip=True)]
                    if paragraphs:
                        cleaned = " ".join(paragraphs)
                        parser_used = "selectolax-fallback"
                except Exception:
                    cleaned = None

            if cleaned:
                text_out = cleaned.strip()

        manager.write_audit_log("inbound", {
            "source": "fetch_page_text",
            "url": safe_url,
            "bytes": len(text_out or ""),
            "parser": parser_used,
        })

        return (text_out or "").strip()

    except Exception as e:
        manager.write_audit_log("inbound", {
            "source": "fetch_page_text",
            "url": safe_url,
            "bytes": 0,
            "error": str(e),
        })
        return ""
