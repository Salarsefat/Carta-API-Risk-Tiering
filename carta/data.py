"""Dataset loaders producing one unified request table.

Columns of the unified table
----------------------------
source      : 'csic2010' | 'httpparams' | 'webpayloads' | 'weblogs2019' | 'accesslog'
rid         : stable row id within source
method, path, query, body : request parts (strings; may be empty)
headers     : dict (may be empty)
session     : session/IP key if the source has one, else ''
label       : 1 = attack, 0 = benign
attack_type : fine-grained family label if the source has one, else 'generic'/'normal'
orig_split  : the split the source itself defines ('train'/'test'/'') -- used only
              for reporting, NOT for our de-duplicated protocol
ts          : timestamp (float seconds) if the source has one, else NaN
"""
from __future__ import annotations

import re
from pathlib import Path
from urllib.parse import urlsplit

import numpy as np
import pandas as pd

_REQ_LINE = re.compile(r"^(GET|POST|PUT|HEAD|DELETE|OPTIONS|PATCH) (\S+)(?: HTTP/\d\.\d)?\s*$")


# ----------------------------------------------------------------------------
# CSIC 2010 (raw text format: request line, headers, blank, [body], blank)
# ----------------------------------------------------------------------------
def parse_csic_file(path: str | Path) -> list[dict]:
    rows: list[dict] = []
    cur: dict | None = None
    state = "idle"  # idle -> headers -> after_headers (body may follow)
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.rstrip("\r\n")
            m = _REQ_LINE.match(line)
            if m:
                if cur is not None:
                    rows.append(cur)
                cur = {"method": m.group(1), "url": m.group(2), "headers": {}, "body": ""}
                state = "headers"
                continue
            if cur is None:
                continue
            if state == "headers":
                if line == "":
                    state = "after_headers"
                elif ":" in line:
                    k, v = line.split(":", 1)
                    cur["headers"][k.strip().lower()] = v.strip()
                continue
            if state == "after_headers":
                if line.strip() == "":
                    continue
                # body line (CSIC bodies are single-line form-encoded)
                cur["body"] = (cur["body"] + "\n" + line) if cur["body"] else line
    if cur is not None:
        rows.append(cur)
    return rows


def load_csic(dirpath: str | Path) -> pd.DataFrame:
    """Load the three official CSIC 2010 files from a directory."""
    d = Path(dirpath)
    spec = [
        ("normalTrafficTraining.txt", 0, "train"),
        ("normalTrafficTest.txt", 0, "test"),
        ("anomalousTrafficTest.txt", 1, "test"),
    ]
    out = []
    for fname, label, split in spec:
        p = d / fname
        if not p.exists():
            raise FileNotFoundError(p)
        for i, r in enumerate(parse_csic_file(p)):
            u = urlsplit(r["url"])
            cookie = r["headers"].get("cookie", "")
            sess = ""
            mm = re.search(r"JSESSIONID=([0-9A-Fa-f]+)", cookie)
            if mm:
                sess = mm.group(1)
            out.append({
                "source": "csic2010",
                "rid": f"{fname}:{i}",
                "method": r["method"],
                "path": u.path,
                "query": u.query,
                "body": r["body"],
                "headers": r["headers"],
                "session": sess,
                "label": label,
                "attack_type": "normal" if label == 0 else "csic_anomalous",
                "orig_split": split,
                "ts": np.nan,
            })
    return pd.DataFrame(out)


def load_csic_kaggle_csv(path: str | Path) -> pd.DataFrame:
    """Loader for the Kaggle conversion of CSIC 2010 ('csic_database.csv', ispangler).

    Columns: <label string>, Method, User-Agent, Pragma, Cache-Control, Accept, Accept-encoding,
    Accept-charset, language, host, cookie, content-type, connection, lenght, content,
    classification (0 normal / 1 anomalous), URL ('http://host/path?query HTTP/1.1').
    All header columns are kept (header count/entropy then match the raw-text loader)."""
    df = pd.read_csv(path, encoding="utf-8", encoding_errors="replace", low_memory=False)
    cols = {c.lower().strip(): c for c in df.columns}
    header_cols = {"user-agent": "user-agent", "pragma": "pragma", "cache-control": "cache-control",
                   "accept": "accept", "accept-encoding": "accept-encoding", "accept-charset": "accept-charset",
                   "language": "accept-language", "host": "host", "cookie": "cookie",
                   "content-type": "content-type", "connection": "connection", "lenght": "content-length"}
    c_method, c_url, c_body, c_label = cols.get("method"), cols.get("url"), cols.get("content"), cols.get("classification")
    out = []
    for i, r in enumerate(df.itertuples(index=False)):
        rd = dict(zip(df.columns, r))
        url = str(rd[c_url]) if c_url else ""
        url = re.sub(r"\s+HTTP/\d\.\d\s*$", "", url)
        u = urlsplit(url)
        body = "" if (c_body is None or pd.isna(rd[c_body])) else str(rd[c_body])
        lab = int(rd[c_label]) if c_label else 0
        hdr = {}
        for col_l, name in header_cols.items():
            c = cols.get(col_l)
            if c is not None and not pd.isna(rd[c]):
                v = str(rd[c])
                v = re.sub(r"^(Connection|Content-Length):\s*", "", v, flags=re.I)
                hdr[name] = v
        sess = ""
        mm = re.search(r"JSESSIONID=([0-9A-Fa-f]+)", hdr.get("cookie", ""))
        if mm:
            sess = mm.group(1)
        out.append({"source": "csic2010", "rid": f"kaggle:{i}", "method": str(rd[c_method]) if c_method else "GET",
                    "path": u.path, "query": u.query, "body": body, "headers": hdr, "session": sess, "label": lab,
                    "attack_type": "normal" if lab == 0 else "csic_anomalous", "orig_split": "", "ts": np.nan})
    return pd.DataFrame(out)


# ----------------------------------------------------------------------------
# HttpParamsDataset (payload, length, attack_type, label)
# ----------------------------------------------------------------------------
def load_httpparams(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8", encoding_errors="replace", quotechar='"')
    cols = {c.lower().strip(): c for c in df.columns}
    c_payload = cols.get("payload") or cols.get("text")
    c_label = cols.get("label")
    c_type = cols.get("attack_type")
    out = []
    for i, r in df.iterrows():
        lab_raw = str(r[c_label]).strip().lower()
        lab = 0 if lab_raw in ("norm", "normal", "benign", "0") else 1
        at = str(r[c_type]).strip().lower() if c_type else ("normal" if lab == 0 else "generic")
        out.append({
            "source": "httpparams",
            "rid": f"hp:{i}",
            "method": "PARAM",
            "path": "",
            "query": "",
            "body": str(r[c_payload]),
            "headers": {},
            "session": "",
            "label": lab,
            "attack_type": "normal" if lab == 0 else at,
            "orig_split": "",
            "ts": np.nan,
        })
    return pd.DataFrame(out)


# ----------------------------------------------------------------------------
# Web Application Payloads (Kaggle cyberprince)
# ----------------------------------------------------------------------------
def read_payload_json(path: str | Path) -> tuple[list[dict], dict]:
    """Tolerant reader for WEB_APPLICATION_PAYLOADS.jsonl (a pretty-printed JSON array).

    The distributed file has three defects that we repair and count: non-breaking spaces
    used as indentation, missing commas between objects, and invalid backslash escapes."""
    import json
    txt = Path(path).read_text(encoding="utf-8")
    rep = {"nbsp_replaced": txt.count("\xa0")}
    txt = txt.replace("\xa0", " ")
    # scan escape pairs left to right: keep valid JSON escapes, double a lone invalid backslash
    esc = re.compile(r'\\(["\\/bfnrt]|u[0-9a-fA-F]{4})|\\')
    n_bad = 0

    def _fix(m):
        nonlocal n_bad
        if m.group(1) is not None:
            return m.group(0)
        n_bad += 1
        return "\\\\"

    txt = esc.sub(_fix, txt)
    rep["invalid_escapes_repaired"] = n_bad
    dec, objs, i, ws = json.JSONDecoder(), [], txt.index("[") + 1, re.compile(r"\s*")
    missing_commas = 0
    while True:
        i = ws.match(txt, i).end()
        if i < len(txt) and txt[i] == ",":
            i = ws.match(txt, i + 1).end()
        elif objs and i < len(txt) and txt[i] == "{":
            missing_commas += 1
        if i >= len(txt) or txt[i] == "]":
            break
        o, i = dec.raw_decode(txt, i)
        objs.append(o)
    rep["missing_commas_repaired"] = missing_commas
    rep["objects"] = len(objs)
    return objs, rep


def load_webpayloads(path: str | Path) -> pd.DataFrame:
    p = Path(path)
    if p.suffix.lower() in (".json", ".jsonl"):
        objs, _ = read_payload_json(p)
        out = []
        for i, o in enumerate(objs):
            fam = str(o.get("id", "generic")).rsplit("-", 1)[0].lower()
            out.append({"source": "webpayloads", "rid": f"wp:{o.get('id', i)}", "method": "PARAM", "path": "",
                        "query": "", "body": str(o.get("payload", "")), "headers": {}, "session": "", "label": 1,
                        "attack_type": fam, "orig_split": "", "ts": np.nan})
        return pd.DataFrame(out)
    df = pd.read_csv(path, encoding="utf-8", encoding_errors="replace")
    cols = {c.lower().strip(): c for c in df.columns}
    c_payload = cols.get("payload") or cols.get("text") or list(df.columns)[0]
    c_type = cols.get("attack_type") or cols.get("type") or cols.get("category") or cols.get("label")
    out = []
    for i, r in df.iterrows():
        at = str(r[c_type]).strip().lower() if c_type else "generic"
        out.append({
            "source": "webpayloads",
            "rid": f"wp:{i}",
            "method": "PARAM",
            "path": "",
            "query": "",
            "body": str(r[c_payload]),
            "headers": {},
            "session": "",
            "label": 1,
            "attack_type": at,
            "orig_split": "",
            "ts": np.nan,
        })
    return pd.DataFrame(out)


# ----------------------------------------------------------------------------
# Labelled web-server access logs (zanbil.ir 2019) -- compact sample built by prepare_zanbil.py
# ----------------------------------------------------------------------------
def load_weblogs_sample(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    out = []
    for i, r in enumerate(df.itertuples(index=False)):
        url = r.url or ""
        u = urlsplit(url if url.startswith("/") else "/" + url)
        lab = 1 if str(r.label) == "1" else 0
        try:
            ts = pd.Timestamp(r.time).timestamp()
        except Exception:
            ts = np.nan
        out.append({"source": "weblogs2019", "rid": f"wl:{i}", "method": (r.method or "GET").upper(),
                    "path": u.path, "query": u.query, "body": "",
                    "headers": {"user-agent": r.user_agent, "referer": r.referrer},
                    "session": r.ip, "label": lab, "attack_type": (r.type if lab else "normal"),
                    "orig_split": "", "ts": ts, "raw_count": int(r.raw_count or 1)})
    return pd.DataFrame(out)


# ----------------------------------------------------------------------------
# Apache/Nginx access logs, raw (common/combined log format) or CSV; e.g. the unlabelled Access Log (dsfelix)
# ----------------------------------------------------------------------------
_CLF = re.compile(
    r'^(?P<ip>\S+) \S+ \S+ \[(?P<ts>[^\]]+)\] "(?P<method>[A-Z]+) (?P<url>\S+)(?: HTTP/[\d.]+)?" '
    r'(?P<status>\d{3}) (?P<size>\S+)(?: "(?P<ref>[^"]*)" "(?P<ua>[^"]*)")?'
)


def parse_clf_line(line: str) -> dict | None:
    m = _CLF.match(line.strip())
    if not m:
        return None
    d = m.groupdict()
    try:
        ts = pd.to_datetime(d["ts"], format="%d/%b/%Y:%H:%M:%S %z").timestamp()
    except Exception:
        ts = np.nan
    u = urlsplit(d["url"])
    return {
        "method": d["method"], "path": u.path, "query": u.query, "body": "",
        "headers": {"user-agent": d.get("ua") or "", "referer": d.get("ref") or ""},
        "session": d["ip"], "status": int(d["status"]), "ts": ts,
    }


def load_access_log(path: str | Path, label_col_hint: str | None = None, default_label: int = 0,
                    source: str = "accesslog", max_rows: int | None = None) -> pd.DataFrame:
    """Load either a raw CLF log (.log/.txt) or a CSV with a raw log line / label column."""
    p = Path(path)
    out = []
    if p.suffix.lower() in (".csv",):
        df = pd.read_csv(p, encoding="utf-8", encoding_errors="replace", low_memory=False, nrows=max_rows)
        cols = {c.lower().strip(): c for c in df.columns}
        c_line = None
        for cand in ("log", "line", "raw", "request", "text", "message"):
            if cand in cols:
                c_line = cols[cand]
                break
        c_label = cols.get((label_col_hint or "").lower()) or cols.get("label") or cols.get("class") \
            or cols.get("is_attack") or cols.get("attack")
        for i, r in df.iterrows():
            parsed = None
            if c_line is not None:
                parsed = parse_clf_line(str(r[c_line]))
            if parsed is None:
                # column-wise CSV (ip, method, url, status, ...)
                if "url" in cols and "method" in cols:
                    u = urlsplit(str(r[cols["url"]]))
                    parsed = {"method": str(r[cols["method"]]), "path": u.path, "query": u.query, "body": "",
                              "headers": {}, "session": str(r[cols.get("ip", cols["method"])]),
                              "status": int(r[cols["status"]]) if "status" in cols and not pd.isna(r[cols["status"]]) else 0,
                              "ts": np.nan}
                else:
                    continue
            lab = default_label
            if c_label is not None and not pd.isna(r[c_label]):
                v = str(r[c_label]).strip().lower()
                lab = 0 if v in ("0", "benign", "normal", "norm", "false") else 1
            parsed.update({"source": source, "rid": f"{source}:{i}", "label": lab,
                           "attack_type": "normal" if lab == 0 else "log_attack", "orig_split": ""})
            out.append(parsed)
    else:
        with open(p, "r", encoding="utf-8", errors="replace") as f:
            for i, line in enumerate(f):
                if max_rows and i >= max_rows:
                    break
                parsed = parse_clf_line(line)
                if parsed is None:
                    continue
                parsed.update({"source": source, "rid": f"{source}:{i}", "label": default_label,
                               "attack_type": "normal" if default_label == 0 else "log_attack", "orig_split": ""})
                out.append(parsed)
    return pd.DataFrame(out)
