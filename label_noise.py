#!/usr/bin/env python3
"""Label-noise sensitivity for the labelled web-server logs.

Benign-labelled requests that carry an unambiguous attack signature (sqlmap-style time-based or
boolean SQL injection, stacked queries, admin-panel scanner paths; every match reviewed by hand)
are counted as attacks, and test metrics are recomputed from the saved scores with the same
validation-chosen thresholds and conformal tiers. Nothing is refitted.

  python label_noise.py --results results
"""
from __future__ import annotations

import argparse
import glob
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from carta.features import context_frame, sensitivity_cells  # noqa: E402
from carta.metrics import summary  # noqa: E402
from carta.normalize import normalize  # noqa: E402
from carta.proposed import DEFAULT_BUDGETS, ConformalTiers  # noqa: E402

SIG = {
    "sqli_time": r"\b(sleep|pg_sleep)\s*\(\s*\d+|waitfor\s+delay|benchmark\s*\(",
    "sqli_boolean_tautology": r"'\s*(and|or)\s+'?\w+'?\s*=\s*'?\w+|\b(and|or)\s+\d+\s*=\s*\d+\s+(and|or)\s+'",
    "sqli_union": r"union(\s|/\*.*?\*/)+(all\s+)?select",
    "sqli_stacked": r"';\s*(select|drop|insert|update|delete|exec)\b",
    "xss": r"<\s*script|javascript:|onerror\s*=|onload\s*=",
    "traversal": r"\.\./\.\./|/etc/passwd|win\.ini",
    "scanner_paths": r"^/(phpmyadmin|pma|myadmin|administrator|wp-login\.php|wp-admin|xmlrpc\.php|\.env|\.git|cgi-bin|manager/html)",
    "rce": r"\b(wget|curl)\s+http|/bin/(ba)?sh|\bcmd=|invokefunction|call_user_func",
}
COMP = {k: re.compile(v, re.I) for k, v in SIG.items()}


def signature(path, query) -> list[str]:
    p, s = normalize(path), normalize(path) + "?" + normalize(query)
    return [k for k, c in COMP.items() if c.search(s) or (k == "scanner_paths" and c.search(p))]


def run(rd: Path, seed: int) -> dict:
    b = pd.read_pickle(rd / "bench" / f"bench_seed{seed}.pkl")
    va = b[b.split == "val"].reset_index(drop=True)
    te = b[b.split == "test"].reset_index(drop=True)
    P = np.load(rd / f"preds_seed{seed}.npz", allow_pickle=True)
    y_te = P["y_te"].astype(int)
    flag = np.array([(s == "weblogs2019" and y == 0 and bool(signature(p, q)))
                     for s, y, p, q in zip(te["source"], y_te, te["path"], te["query"])])
    y_fix = y_te.copy()
    y_fix[flag] = 1
    cells_va, cells_te = sensitivity_cells(context_frame(va)), sensitivity_cells(context_frame(te))
    out = {"seed": seed, "n_test_relabelled": int(flag.sum()),
           "relabelled_requests": [f"{m} {p}?{q}"[:160] for m, p, q in zip(te["method"][flag], te["path"][flag], te["query"][flag])]}
    for name, s_va, s_te in (("carta", P["carta_va"], P["carta"]), ("prior", P["prior_va"], P["prior"])):
        res = {}
        for lab, y in (("original", y_te), ("corrected", y_fix)):
            sm = summary(y, s_te)
            r = {k: sm[k] for k in ("roc_auc", "pr_auc", "recall@fpr0.001", "recall@fpr0.01")}
            for mode in ("mondrian", "global"):
                ct = ConformalTiers(DEFAULT_BUDGETS, mondrian=(mode == "mondrian")).fit(s_va, P["y_va"].astype(int), cells_va)
                T = ct.thresholds(cells_te)
                for j, t in enumerate(ct.ORDER):
                    hit = s_te > T[:, j]
                    r[f"{mode}_{t}_fpr"] = float(hit[y == 0].mean())
            res[lab] = r
        out[name] = res
    out["_y_fix"] = y_fix
    # weblogs-only view
    m = te["source"].to_numpy() == "weblogs2019"
    out["weblogs_only"] = {name: {lab: summary(y[m], s[m])["recall@fpr0.001"] for lab, y in (("original", y_te), ("corrected", y_fix))}
                           for name, s in (("carta", P["carta"]), ("prior", P["prior"]))}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    a = ap.parse_args()
    rd = Path(a.results)
    for f in sorted(glob.glob(str(rd / "preds_seed*.npz"))):
        seed = int(Path(f).stem.replace("preds_seed", ""))
        out = run(rd, seed)
        y_fix = out.pop("_y_fix")
        (rd / f"label_noise_seed{seed}.json").write_text(json.dumps(out, indent=1, ensure_ascii=False, default=float))
        np.save(rd / f"y_te_corrected_seed{seed}.npy", y_fix)
        c = out["carta"]
        print(f"seed {seed}: relabelled {out['n_test_relabelled']} | CARTA global high FPR {100*c['original']['global_high_fpr']:.3f}% -> "
              f"{100*c['corrected']['global_high_fpr']:.3f}%, critical {100*c['original']['global_critical_fpr']:.3f}% -> "
              f"{100*c['corrected']['global_critical_fpr']:.3f}% | weblogs R@0.1% carta {out['weblogs_only']['carta']} prior {out['weblogs_only']['prior']}")


if __name__ == "__main__":
    main()
