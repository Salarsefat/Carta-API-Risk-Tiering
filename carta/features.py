"""Feature extraction: numeric request features, risk context, structure-aware tokens, rules."""
from __future__ import annotations

import html as _html
import math
import re
from collections import Counter
from urllib.parse import parse_qsl

import numpy as np
import pandas as pd

from .normalize import unquote_smart

STATIC_NUMERIC = [
    "endpoint_len", "path_depth", "query_param_count", "header_count", "headers_entropy",
    "payload_size", "is_json_payload", "suspicious_char_ratio", "http_method_code",
    "body_param_count", "max_param_len", "digit_ratio", "upper_ratio",
    "encoded_ratio", "nonascii_ratio", "decode_rounds",
]
NUMERIC_FEATURES = STATIC_NUMERIC + ["method_path_mismatch"]

_SUSPICIOUS = frozenset("'\";<>()|&`$\\{}[]%*=")
_METHOD_CODE = {"GET": 1, "POST": 2, "PUT": 3, "DELETE": 4, "HEAD": 5, "OPTIONS": 6, "PATCH": 7, "PARAM": 0}


def char_entropy(s: str) -> float:
    if not s:
        return 0.0
    c = Counter(s)
    n = len(s)
    return -sum((v / n) * math.log2(v / n) for v in c.values())


def decode_rounds(s: str, max_rounds: int = 6) -> int:
    r, cur = 0, s or ""
    for i in range(max_rounds):
        d = _html.unescape(unquote_smart(cur, plus=(i == 0)))
        if d == cur:
            break
        cur, r = d, r + 1
    return r


def _safe_qsl(s: str):
    if not s:
        return []
    try:
        return parse_qsl(s, keep_blank_values=True)
    except Exception:
        return [("_raw", s)]


def static_numeric_row(method: str, path: str, query: str, body: str, headers: dict) -> list[float]:
    query, body, path = query or "", body or "", path or ""
    raw = query + "&" + body
    qp, bp = _safe_qsl(query), _safe_qsl(body)
    vals = [v for _, v in qp + bp] or [raw]
    hdr_blob = " ".join(f"{k}:{v}" for k, v in (headers or {}).items())
    n = max(1, len(raw))
    n_susp = n_dig = n_up = n_na = 0
    for ch in raw:
        if ch in _SUSPICIOUS:
            n_susp += 1
        if ch.isdigit():
            n_dig += 1
        elif ch.isupper():
            n_up += 1
        if ord(ch) > 127:
            n_na += 1
    return [
        float(len(path)),
        float(sum(1 for seg in path.split("/") if seg)),
        float(len(qp)),
        float(len(headers or {})),
        char_entropy(hdr_blob),
        float(len(body)),
        1.0 if body.strip().startswith(("{", "[")) else 0.0,
        n_susp / n,
        float(_METHOD_CODE.get((method or "").upper(), 9)),
        float(len(bp)),
        float(max((len(v) for v in vals), default=0)),
        n_dig / n,
        n_up / n,
        raw.count("%") / n,
        n_na / n,
        float(decode_rounds(raw)),
    ]


def static_numeric_matrix(df: pd.DataFrame) -> np.ndarray:
    return np.array([static_numeric_row(m, p, q, b, h) for m, p, q, b, h in
                     zip(df["method"], df["path"], df["query"], df["body"], df["headers"])], dtype=float).reshape(-1, len(STATIC_NUMERIC))


def fit_method_table(df: pd.DataFrame) -> dict[str, set]:
    """Methods observed per path on benign training rows (for method/path mismatch)."""
    t: dict[str, set] = {}
    for p, m in zip(df["path"], df["method"]):
        t.setdefault(p or "", set()).add((m or "").upper())
    return t


def method_mismatch(df: pd.DataFrame, table: dict[str, set]) -> np.ndarray:
    out = np.zeros((len(df), 1))
    for i, (p, m) in enumerate(zip(df["path"], df["method"])):
        allowed = table.get(p or "")
        if allowed is not None and (m or "").upper() not in allowed:
            out[i, 0] = 1.0
    return out


def numeric_matrix(df: pd.DataFrame, method_table: dict[str, set] | None = None) -> np.ndarray:
    S = static_numeric_matrix(df)
    M = method_mismatch(df, method_table or {})
    return np.concatenate([S, M], axis=1)


# ----------------------------------------------------------------------------
# Risk context (derived from the protected application's endpoint conventions)
# ----------------------------------------------------------------------------
# env and exposure are constant here ('prod', 'public') and enter only the prior pipeline's impact score.
# CARTA's 18 context indicators (context_onehot) are endpoint class (8), data class (5), authentication
# state (3) and blast radius (2: high, medium).
CONTEXT_FIELDS = ["env", "endpoint_class", "data_class", "auth_state", "exposure", "blast"]
ENDPOINT_CLASSES = ["payments", "auth", "profile", "cart", "public_api", "files", "admin", "unknown"]
DATA_CLASSES = ["generic", "pii", "credentials", "pci", "none"]
AUTH_STATES = ["anonymous", "user", "admin"]
# sensitivity cells used for context-dependent error budgets
SENSITIVITY = {"payments": "high", "auth": "high", "admin": "high", "profile": "high",
               "cart": "std", "public_api": "std", "unknown": "std", "files": "low"}

_RE_STATIC = re.compile(r"\.(gif|jpg|jpeg|png|css|js|ico|svg|woff2?)$")


def derive_context(path: str, source: str) -> dict:
    """Business context from the endpoint path: the CSIC 2010 shop ('tienda1') and the web-server logs;
    payload-only sources (HttpParams, Web Application Payloads) keep the defaults."""
    p = (path or "").lower()
    ctx = {"env": "prod", "endpoint_class": "unknown", "data_class": "generic",
           "auth_state": "anonymous", "exposure": "public", "blast": "low"}
    if source == "csic2010":
        if _RE_STATIC.search(p) or "imagenes" in p:
            ctx.update(endpoint_class="files", data_class="none")
        elif "pagar" in p:
            ctx.update(endpoint_class="payments", data_class="pci", blast="high")
        elif any(k in p for k in ("autenticar", "entrar", "registro")):
            ctx.update(endpoint_class="auth", data_class="credentials", blast="med")
        elif "miembros" in p:
            ctx.update(endpoint_class="profile", data_class="pii", auth_state="user", blast="med")
        elif any(k in p for k in ("anadir", "carrito", "vaciar")):
            ctx.update(endpoint_class="cart", data_class="generic")
        else:
            ctx.update(endpoint_class="public_api", data_class="none")
    elif source in ("accesslog", "weblogs2019"):
        if re.search(r"(login|signin|auth|session)", p):
            ctx.update(endpoint_class="auth", data_class="credentials", blast="med")
        elif re.search(r"(admin|wp-admin|manager)", p):
            ctx.update(endpoint_class="admin", data_class="pii", auth_state="admin", blast="high")
        elif re.search(r"(pay|checkout|order|cart)", p):
            ctx.update(endpoint_class="payments", data_class="pci", blast="high")
        elif _RE_STATIC.search(p):
            ctx.update(endpoint_class="files", data_class="none")
        else:
            ctx.update(endpoint_class="public_api")
    return ctx


def context_frame(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame([derive_context(p, s) for p, s in zip(df["path"], df["source"])],
                        columns=CONTEXT_FIELDS)


def context_onehot(ctx: pd.DataFrame) -> np.ndarray:
    cols = [(ctx["endpoint_class"] == ec).to_numpy(float) for ec in ENDPOINT_CLASSES]
    cols += [(ctx["data_class"] == dc).to_numpy(float) for dc in DATA_CLASSES]
    cols += [(ctx["auth_state"] == a).to_numpy(float) for a in AUTH_STATES]
    cols += [(ctx["blast"] == "high").to_numpy(float), (ctx["blast"] == "med").to_numpy(float)]
    return np.stack(cols, axis=1)


def sensitivity_cells(ctx: pd.DataFrame) -> np.ndarray:
    return ctx["endpoint_class"].map(SENSITIVITY).fillna("std").to_numpy()


# ----------------------------------------------------------------------------
# Structure-aware tokenisation for the proposed text model (input: normalised text)
# ----------------------------------------------------------------------------
_RE_HEX = re.compile(r"\b[0-9a-f]{16,}\b")
_RE_NUM = re.compile(r"\d+")
_RE_LONGWORD = re.compile(r"\b[a-z0-9]{24,}\b")
_RE_SPLIT = re.compile(r"([^\w\s])")
_RE_WS = re.compile(r"\s+")


def structure_tokens(text: str) -> str:
    """Replace identifiers by placeholders and keep every delimiter as its own token."""
    s = _RE_HEX.sub(" hexblob ", text)
    s = _RE_LONGWORD.sub(" longblob ", s)
    s = _RE_NUM.sub(" num ", s)
    s = _RE_SPLIT.sub(r" \1 ", s)
    return _RE_WS.sub(" ", s).strip()


# ----------------------------------------------------------------------------
# Generic OWASP-style rules (our own; strong = high precision, partial = indicators).
# Input: normalised user-controlled content (path + query + body), not headers.
# ----------------------------------------------------------------------------
RULES: dict[str, dict] = {
    "sqli": {
        "strong": [r"\bunion\b\s+(all\s+)?\bselect\b", r"\bor\b\s+['\"]?\d+['\"]?\s*=\s*['\"]?\d+", r"\bdrop\s+table\b",
                   r";\s*(select|insert|update|delete|drop)\b", r"\bsleep\s*\(", r"\bwaitfor\s+delay\b", r"\bbenchmark\s*\(",
                   r"\binformation_schema\b", r"'\s*or\s*'[^']*'\s*=\s*'", r"\bxp_cmdshell\b"],
        "partial": [r"\bselect\b", r"\bfrom\b", r"\bwhere\b", r"--", r"#", r"'", r"\bor\b", r"\band\b", r"\bnull\b"],
    },
    "xss": {
        "strong": [r"<\s*script\b", r"\bjavascript\s*:", r"\bon(load|error|click|mouseover|focus)\s*=", r"<\s*iframe\b",
                   r"\balert\s*\(", r"\bdocument\.(cookie|location)\b", r"<\s*img\b[^>]*onerror"],
        "partial": [r"<", r">", r"\bscript\b", r"\beval\b", r"\bsvg\b"],
    },
    "cmdi": {
        "strong": [r";\s*(cat|ls|id|whoami|uname|wget|curl|nc|bash|sh|python|perl)\b", r"\|\s*(cat|ls|id|whoami|nc|bash|sh)\b",
                   r"`[^`]+`", r"\$\([^)]+\)", r"/bin/(ba)?sh\b", r"/dev/tcp/", r"\b(cat|less|more)\s+/etc/passwd\b"],
        "partial": [r"\bcat\b", r"\bls\b", r"\bwhoami\b", r"\betc/passwd\b", r"\|", r"&&"],
    },
    "path_traversal": {
        "strong": [r"\.\./\.\./", r"\.\.\\\.\.\\", r"/etc/(passwd|shadow)\b", r"\bc:\\windows\\", r"\bboot\.ini\b", r"\.\.%2f"],
        "partial": [r"\.\./", r"\.\.\\", r"/etc/"],
    },
    "ssrf": {
        "strong": [r"\b169\.254\.169\.254\b", r"\bmetadata\.google\.internal\b", r"\b(localhost|127\.0\.0\.1|0\.0\.0\.0)\b",
                   r"\b(10|192\.168|172\.(1[6-9]|2\d|3[01]))\.\d+\.\d+(\.\d+)?\b", r"\b(file|gopher|dict)://"],
        "partial": [r"\bhttp://", r"\burl=", r"\bfetch\b"],
    },
    "ldap": {
        "strong": [r"\(\s*[&|!]\s*\(", r"\(\s*\w+\s*=\s*\*\s*\)", r"\bobjectclass\s*=", r"\buid\s*=\s*\*"],
        "partial": [r"\(", r"\)", r"\*"],
    },
    "nosql": {
        "strong": [r"\$where\b", r"\$ne\b", r"\$gt\b", r"\$regex\b", r"\$or\b", r"\{\s*\"?\$\w+\"?\s*:"],
        "partial": [r"\$", r"\{", r"\}"],
    },
}
RULE_FAMILIES = list(RULES.keys())
_COMPILED = [(fam, [re.compile(p, re.I) for p in RULES[fam]["strong"]], [re.compile(p, re.I) for p in RULES[fam]["partial"]])
             for fam in RULE_FAMILIES]


def rule_row(content: str) -> list[float]:
    """Per-family rule score in [0,1]: 1.0 on any strong hit, else min(0.6, 0.15 * partial hits)."""
    out = []
    for _, strong, partial in _COMPILED:
        if any(p.search(content) for p in strong):
            out.append(1.0)
        else:
            out.append(min(0.6, 0.15 * sum(1 for p in partial if p.search(content))))
    return out


def rule_matrix(contents) -> np.ndarray:
    return np.array([rule_row(c) for c in contents], dtype=float).reshape(-1, len(RULE_FAMILIES))
