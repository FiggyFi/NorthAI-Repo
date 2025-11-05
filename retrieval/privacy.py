# ===============================
# File: retrieval/privacy.py
# Purpose: privacy-safe query + outbound redaction filter
# ===============================
import re
from collections import Counter
from typing import Callable, List

from vector_store import get_or_create_collection, COLL_LOCAL

_TOKEN = re.compile(r"[A-Za-z][A-Za-z0-9_\-]{2,}")
STOP = {
    "the","and","for","with","from","this","that","there","their","have","has","had",
    "into","over","under","across","about","within","without","between","among",
    "use","used","using","based","system","data","model","results","paper","also",
    "into","onto","via","per","each","other","than","such","more","less","most",
    "can","could","would","should","shall","may","might","must","not","only",
}

def _top_keywords(texts: List[str], k: int = 12) -> List[str]:
    cnt = Counter()
    for t in texts:
        for tok in _TOKEN.findall(t or ""):
            w = tok.lower()
            if w in STOP or not (3 <= len(w) <= 40):
                continue
            cnt[w] += 1
    return [w for w,_ in cnt.most_common(k)]

def privacy_expand_query(user_prompt: str, top_k_local: int = 5) -> str:
    """
    Local-only keyword expansion; never emits verbatim text from uploads.
    Returns a compact query: original prompt + up to 8 keywords, ≤512 chars.
    """
    local_docs = []
    try:
        lcoll = get_or_create_collection(COLL_LOCAL)
        res = lcoll.query(query_texts=[user_prompt], n_results=top_k_local)
        local_docs = res.get("documents", [[]])[0]
    except Exception:
        pass
    kws = _top_keywords(local_docs, k=16)[:8]
    q = (user_prompt or "")
    if kws:
        q += " keywords: " + " ".join(sorted(set(kws)))
    q = q.replace('"', " ").replace("'", " ")
    return re.sub(r"\s+", " ", q).strip()[:512]

def make_outbound_filter_fn(local_docs: List[str]) -> Callable[[str], str]:
    """
    Returns a function that redacts any ≥32-char substring present in local docs.
    Prevents long-copy leakage in outbound requests.
    """
    SHINGLE = 32
    seen = set()
    for d in local_docs or []:
        s = d or ""
        for i in range(0, max(0, len(s) - SHINGLE + 1)):
            seen.add(s[i:i+SHINGLE])

    def _filter(payload: str) -> str:
        if not isinstance(payload, str):
            return ""
        p = payload.replace('"', " ").replace("'", " ")
        out, buf = [], []
        i = 0
        while i < len(p):
            buf.append(p[i])
            if len(buf) >= SHINGLE:
                win = "".join(buf[-SHINGLE:])
                if win in seen:
                    # remove last SHINGLE chars and drop a marker
                    for _ in range(SHINGLE):
                        if buf:
                            buf.pop()
                    out.append("[REDACTED]")
            i += 1
        out.append("".join(buf))
        s = re.sub(r"\s+", " ", "".join(out)).strip()
        return s[:512]
    return _filter
