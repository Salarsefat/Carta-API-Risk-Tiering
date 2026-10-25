"""Request normalisation.

Two normalisers are provided:

* ``normalize`` (proposed): fixpoint decoding. Percent-escapes and HTML entities are
  decoded repeatedly until nothing changes (form '+' -> space only in the first
  round, because only the outermost layer is form-encoded). Percent-escape runs are
  decoded as UTF-8 when valid and as Latin-1 otherwise, so legacy Latin-1 traffic
  such as CSIC 2010 is not mangled into U+FFFD. This is followed by NFKC folding,
  control/zero-width removal, inline-comment stripping (``/**/`` used as
  whitespace), case folding and whitespace collapse. Contract: the input is a raw
  form-encoded string. Encoding invariance: for any value v and k >= 0 extra
  percent-encoding layers E, normalize(F(E^k(v))) == normalize(F(v)), where F is the
  outer form encoding (verified on 30,000 real CSIC/HttpParams values for k <= 4).
  (It is not idempotent on decoded strings, because a literal '+' in a decoded
  string would be read as a form-encoded space.)

* ``normalize_single_pass`` (baseline): one URL-decoding pass (UTF-8, invalid
  bytes replaced), one HTML-unescape pass, NFKC, a few escape-sequence
  replacements, zero-width/control removal, whitespace collapse and lower-casing.
  This is the preprocessing the prior pipeline describes (URL/HTML decoding,
  case folding, control-character removal); it is re-implemented here
  independently.
"""
from __future__ import annotations

import html
import re
import unicodedata
from urllib.parse import unquote_plus

_MAX_ROUNDS = 6
_RE_INLINE_COMMENT = re.compile(r"/\*.*?\*/", re.S)
_RE_WS = re.compile(r"\s+")
_RE_CTRL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_RE_ZW = re.compile(r"[\u200b-\u200d\ufeff]")  # zero-width space, (non-)joiner, byte-order mark
_PCT_RUN = re.compile(r"(?:%[0-9A-Fa-f]{2})+")


def _pct_run_decode(m: re.Match) -> str:
    b = bytes.fromhex(m.group(0).replace("%", ""))
    try:
        return b.decode("utf-8")
    except UnicodeDecodeError:
        return b.decode("latin-1")


def unquote_smart(s: str, plus: bool) -> str:
    if plus:
        s = s.replace("+", " ")
    return _PCT_RUN.sub(_pct_run_decode, s)


def normalize(text, max_rounds: int = _MAX_ROUNDS) -> str:
    """Fixpoint URL/HTML decoding + Unicode folding + comment/whitespace cleanup."""
    if text is None:
        return ""
    s = str(text)
    for r in range(max_rounds):
        d = html.unescape(unquote_smart(s, plus=(r == 0)))
        if d == s:
            break
        s = d
    s = unicodedata.normalize("NFKC", s)
    s = _RE_ZW.sub("", s)
    s = _RE_CTRL.sub(" ", s)
    s = _RE_INLINE_COMMENT.sub(" ", s)
    s = s.lower()
    return _RE_WS.sub(" ", s).strip()


def normalize_single_pass(text) -> str:
    """Single-pass normalisation as described for the prior pipeline."""
    if not text:
        return ""
    s = str(text)
    try:
        s = unquote_plus(s)
    except Exception:
        pass
    s = html.unescape(s)
    s = unicodedata.normalize("NFKC", s)
    for a, b in (("\\u0027", "'"), ("\\u003d", "="), ("\\'", "'"), ('\\"', '"'), ("`", "'")):
        s = s.replace(a, b)
    s = _RE_ZW.sub("", s)
    s = _RE_CTRL.sub("", s)
    return " ".join(s.split()).lower()


def canonical_request_text(method: str, path: str, query: str, body: str,
                           headers: dict | None = None,
                           include_headers: tuple[str, ...] = ("content-type", "user-agent"),
                           norm=normalize) -> str:
    """Textual view x_text of a request: method | path | query | body | selected headers.

    Cookies/session identifiers and lengths are excluded on purpose: they are per-session
    identifiers that leak split membership and carry no attack semantics.
    """
    parts = [f"{(method or '').upper()}", f"path={norm(path)}"]
    if query:
        parts.append(f"query={norm(query)}")
    if body:
        parts.append(f"body={norm(body)}")
    if headers:
        for k in include_headers:
            v = headers.get(k) or headers.get(k.title())
            if v:
                parts.append(f"{k}={norm(v)}")
    return " | ".join(parts)


def canonical_key(method: str, path: str, query: str, body: str) -> str:
    """Key used for de-duplication across splits (identifiers removed, fixpoint-normalised)."""
    return f"{(method or '').upper()} {normalize(path)}?{normalize(query)}#{normalize(body)}"


def twin_key(method: str, path: str, query: str, body: str) -> str:
    """Method- and location-invariant key: the same path with the same multiset of parameters is one
    request whether it was sent as GET (query string) or POST (body). CSIC 2010 contains many such
    GET/POST twins. Parameters are split on the raw '&' before normalisation, so encoded separators
    stay inside values. Bare parameter values (method PARAM) are keyed whole."""
    if (method or "").upper() == "PARAM":
        return "PARAM " + normalize(body)
    params = sorted(normalize(p) for p in (query or "").split("&") + (body or "").split("&") if p)
    return normalize(path) + "?" + "&".join(params)


def content_view(path: str, query: str, body: str, norm=normalize) -> str:
    """User-controlled content (path, query, body) used by the rule layer."""
    return " ".join(x for x in (norm(path), norm(query), norm(body)) if x)


def add_text_views(df):
    """Attach text views: `text`/`content` (proposed, fixpoint) and `text_prior`/`content_prior`
    (baseline, single-pass decoding)."""
    df = df.copy()
    cols = list(zip(df["method"], df["path"], df["query"], df["body"], df["headers"]))
    df["text"] = [canonical_request_text(m, p, q, b, h) for m, p, q, b, h in cols]
    df["text_prior"] = [canonical_request_text(m, p, q, b, h, norm=normalize_single_pass) for m, p, q, b, h in cols]
    df["content"] = [content_view(p, q, b) for _, p, q, b, _ in cols]
    df["content_prior"] = [content_view(p, q, b, norm=normalize_single_pass) for _, p, q, b, _ in cols]
    return df
