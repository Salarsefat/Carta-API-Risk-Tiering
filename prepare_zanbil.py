#!/usr/bin/env python3
"""Build a compact, reproducible sample of 'Web Server Access Logs - Labeled' (zanbil.ir, 2019).

Pass 1: collect every labelled attack row (label 1) except type 'bot'. In this log the 1,526 'bot'
        rows are exactly the fetches of /robots.txt: a client-behaviour label on a request without
        attack content, which a per-request content detector could only learn as a fixed path.
Pass 2: (a) aggregate benign rows whose raw request line hashes into the sample
        (md5(method + ' ' + url) mod 10^6 < rate * 10^6, i.e. uniform over unique URLs,
        all occurrences kept so the raw frequency is exact), and (b) count benign occurrences
        of attack URLs to detect label conflicts.
Output: zanbil_sample.csv with one row per benign unique URL (raw_count = frequency) and one
row per attack request; label-conflicting attack URLs are dropped and counted.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from collections import Counter

import pandas as pd

COLS = ["ip", "time", "method", "url", "status", "referrer", "user_agent", "label", "type"]


def chunks(zip_path, chunksize=500_000):
    z = zipfile.ZipFile(zip_path)
    with z.open(z.infolist()[0]) as f:
        yield from pd.read_csv(f, usecols=COLS, chunksize=chunksize, low_memory=False, on_bad_lines="skip",
                               dtype=str, keep_default_na=False)


def hsel(key: str, rate: float) -> bool:
    return int(hashlib.md5(key.encode("utf-8", "replace")).hexdigest()[:12], 16) % 1_000_000 < rate * 1_000_000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("zip_path")
    ap.add_argument("--out", default="zanbil_sample.csv")
    ap.add_argument("--rate", type=float, default=0.06, help="share of unique benign URLs kept (0.06 in the paper)")
    args = ap.parse_args()
    attacks, att_urls = [], set()
    n_rows, types = 0, Counter()
    for ch in chunks(args.zip_path):
        n_rows += len(ch)
        a = ch[ch["label"] == "1"]
        types.update(a["type"])
        a = a[a["type"] != "bot"]
        for r in a.itertuples(index=False):
            attacks.append(r._asdict())
            att_urls.add(f"{r.method} {r.url}")
    benign = {}
    benign_cnt = Counter()
    conflict_cnt = Counter()
    for ch in chunks(args.zip_path):
        b = ch[ch["label"] == "0"]
        keys = (b["method"] + " " + b["url"]).tolist()
        for k, r in zip(keys, b.itertuples(index=False)):
            if k in att_urls:
                conflict_cnt[k] += 1
            if hsel(k, args.rate):
                benign_cnt[k] += 1
                if k not in benign:
                    benign[k] = r._asdict()
    rows = []
    for k, r in benign.items():
        r = dict(r)
        r["raw_count"] = benign_cnt[k]
        rows.append(r)
    kept_att = [dict(r, raw_count=1) for r in attacks if f"{r['method']} {r['url']}" not in conflict_cnt]
    df = pd.DataFrame(rows + kept_att)
    df.to_csv(args.out, index=False)
    stats = {"rows_scanned": n_rows, "attack_types_all": dict(types), "attack_rows_non_bot": len(attacks),
             "attack_rows_dropped_label_conflict": len(attacks) - len(kept_att),
             "attack_urls_conflicting": len(conflict_cnt), "benign_unique_sampled": len(rows),
             "benign_rows_covered": int(sum(benign_cnt.values())), "rate": args.rate,
             "kept_attack_types": dict(Counter(r["type"] for r in kept_att))}
    print(json.dumps(stats, indent=1))
    with open(args.out + ".stats.json", "w") as f:
        json.dump(stats, f, indent=1)


if __name__ == "__main__":
    main()
