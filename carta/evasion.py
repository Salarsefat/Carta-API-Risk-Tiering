"""Evasion / obfuscation transforms applied to attack requests at test time.

Transforms act on the *decoded* parameter values (where the attack characters are) and the
result is re-encoded the way the original client encoded it:
  * CSIC 2010 query/body values are form-encoded (Java URLEncoder style, Latin-1): we decode,
    transform and re-encode with the same convention (alnum and ".-*_" literal, space -> '+',
    other code points < 256 -> %XX, code points >= 256 -> UTF-8 %XX bytes);
  * payload-only rows (HttpParamsDataset, method 'PARAM') are raw decoded values: the
    transform is applied to the whole value.
The families are standard WAF-evasion families; the experiment tests detector invariance,
not exploitability against a specific back end.
"""
from __future__ import annotations

import random
import re
from urllib.parse import quote

from .normalize import add_text_views, unquote_smart

_KW = re.compile(r"\b(select|union|from|where|or|and|insert|update|delete|drop|sleep|script|alert|cat|etc|passwd|"
                 r"exec|char|concat|table|onerror|onload|javascript|img|iframe|svg|null|like)\b", re.I)
_FULLWIDTH = {c: chr(ord(c) - 0x20 + 0xFF00) for c in "'\"<>()=;|&"}
_SAFE = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.-*_")


def form_encode(s: str) -> str:
    out = []
    for ch in s:
        if ch in _SAFE:
            out.append(ch)
        elif ch == " ":
            out.append("+")
        elif ord(ch) < 256:
            out.append("%%%02X" % ord(ch))
        else:
            out.extend("%%%02X" % b for b in ch.encode("utf-8"))
    return "".join(out)


# ---- value-level transforms (input/output: decoded strings) -------------------------------
def v_double_url(v, rng):
    return quote(v, safe="")  # one extra encoding layer; the outer form-encoding adds the second


def v_mixed_case(v, rng):
    return _KW.sub(lambda m: "".join(c.upper() if rng.random() < 0.5 else c.lower() for c in m.group(0)), v)


def v_inline_comments(v, rng):
    return v.replace(" ", "/**/")


def v_fullwidth(v, rng):
    return "".join(_FULLWIDTH.get(c, c) for c in v)


def v_whitespace(v, rng):
    return re.sub(" ", lambda m: rng.choice(["\t", "\n", "\r\n", "  ", "\x0b"]), v)


def v_html_entities(v, rng):
    return "".join(f"&#{ord(c)};" if c in "'\"<>()=;|" else c for c in v)


def v_combined(v, rng):
    return v_double_url(v_mixed_case(v_inline_comments(v, rng), rng), rng)


def v_benign_padding(v, rng, pool=None):
    """'Good-word' padding: append benign content drawn from benign training values."""
    if not pool:
        return v
    return v + " " + " ".join(rng.sample(pool, k=min(3, len(pool))))


TRANSFORMS = {
    "double_url_encode": v_double_url,
    "mixed_case": v_mixed_case,
    "inline_comments": v_inline_comments,
    "fullwidth_unicode": v_fullwidth,
    "whitespace_variants": v_whitespace,
    "html_entities": v_html_entities,
    "combined": v_combined,
    "benign_padding": v_benign_padding,
}


def _transform_form(s: str, fn, rng) -> str:
    if not s:
        return s
    parts = []
    for kv in s.split("&"):
        if "=" in kv:
            k, v = kv.split("=", 1)
            parts.append(f"{k}={form_encode(fn(unquote_smart(v, plus=True), rng))}")
        else:
            parts.append(kv)
    return "&".join(parts)


def _pad_last(s: str, fn, rng) -> str:
    """Apply fn to the value of the last parameter only (used for padding)."""
    kvs = s.split("&")
    for i in range(len(kvs) - 1, -1, -1):
        if "=" in kvs[i]:
            k, v = kvs[i].split("=", 1)
            kvs[i] = f"{k}={form_encode(fn(unquote_smart(v, plus=True), rng))}"
            break
    return "&".join(kvs)


def transform_request(method, path, query, body, fn, rng):
    if (method or "").upper() == "PARAM":
        v = fn(body or "", rng)
        if fn is v_double_url or fn is v_combined:  # raw value: apply the second layer explicitly
            v = quote(v, safe="")
        return method, path, query, v
    if getattr(fn, "_pad", False):
        if body:
            return method, path, query, _pad_last(body, fn, rng)
        return method, path, _pad_last(query, fn, rng), body
    return method, path, _transform_form(query, fn, rng), _transform_form(body, fn, rng)


def benign_pool(train_df, n_max: int = 5000, seed: int = 0) -> list[str]:
    """Benign parameter values (letters/spaces, 8-60 chars) from the *training* split."""
    vals = []
    ben = train_df[train_df["label"] == 0]
    for m, q, b in zip(ben["method"], ben["query"], ben["body"]):
        if (m or "").upper() == "PARAM":
            cand = [b or ""]
        else:
            cand = [unquote_smart(kv.split("=", 1)[1], plus=True) for s in (q or "", b or "") for kv in s.split("&") if "=" in kv]
        for v in cand:
            if 8 <= len(v) <= 60 and re.fullmatch(r"[A-Za-z\u00c0-\u00ff ]+", v):
                vals.append(v)
    vals = sorted(set(vals))
    random.Random(seed).shuffle(vals)
    return vals[:n_max]


def apply_transform(df, name: str, seed: int = 0, pool: list[str] | None = None):
    """Return a copy of df with the transform applied to attack rows; text views are rebuilt."""
    rng = random.Random(seed)
    fn = TRANSFORMS[name]
    if name == "benign_padding":
        base_fn = fn
        fn = lambda v, r: base_fn(v, r, pool)  # noqa: E731
        fn._pad = True
    df = df.copy()
    Q, B = [], []
    for m, p, q, b, y in zip(df["method"], df["path"], df["query"], df["body"], df["label"]):
        if y == 1:
            _, _, q, b = transform_request(m, p, q, b, fn, rng)
        Q.append(q)
        B.append(b)
    df["query"], df["body"] = Q, B
    return add_text_views(df)
