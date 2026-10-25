#!/usr/bin/env python3
"""Numbers quoted in the paper that are not in the tables written by the other scripts.

  python extra_numbers.py --results results
Writes results/extra_numbers.json and prints a summary:

  prior_operating_points   the prior pipeline at its own levels, 0.6/0.4 and adaptive fusion, score-only levels and
                           the score x impact matrix of its release (paper: Table 9), mean and s.d. over seeds
  ml_branch_lift_seed0     share of benign test requests that the ML branch alone (weight 0.6) lifts to >= 0.3
  reliability              15 equal-width bins on the pooled test predictions: bins whose 95% Wilson interval
                           contains the bin's mean prediction, requests in the middle bins, and the highest attack
                           frequency among the prior's bins 3-9 (paper: Section 4.4, Figure 5)
  low_tier_ties_seed0      benign validation requests of each cell that tie at the cell's tier threshold
                           (needs results/bench/bench_seed0.pkl)
  benign_encodings         share of benign requests carrying each evasion encoding (needs bench_seed0.pkl)
  log_signature_requests_benchmark
                           benign-labelled requests of the web-server logs that carry an unambiguous attack
                           signature (label_noise.py), over all splits of the benchmark (paper: Section 4.3;
                           needs bench_seed0.pkl)
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from carta.baseline import IMPACT_BINS, IMPACT_MATRIX, LEVEL_THRESHOLDS  # noqa: E402

RANK = {"normal": 0, "low": 1, "medium": 2, "high": 3, "critical": 4}


def seeds_of(rd: Path):
    return sorted(int(Path(f).stem.replace("preds_seed", "")) for f in glob.glob(str(rd / "preds_seed*.npz")))


def matrix_rank(final, impact):
    t = LEVEL_THRESHOLDS
    ci = np.where(final >= t["high"], 3, np.where(final >= t["medium"], 2, np.where(final >= t["low"], 1, 0)))
    ii = np.where(impact >= IMPACT_BINS[1], 2, np.where(impact >= IMPACT_BINS[0], 1, 0))
    return np.array([RANK[IMPACT_MATRIX[c][i]] for c, i in zip(ci, ii)])


def prior_operating_points(rd: Path, seeds):
    rows = {}
    for s in seeds:
        P = np.load(rd / f"preds_seed{s}.npz", allow_pickle=True)
        y, imp = P["y_te"].astype(int), P["impact"]
        for fus, key in (("0.6/0.4", "prior"), ("adaptive", "prior_adaptive")):
            f = P[key]
            hits = {f"score-only, low+ (>= {LEVEL_THRESHOLDS['low']})": f >= LEVEL_THRESHOLDS["low"],
                    f"score-only, medium+ (>= {LEVEL_THRESHOLDS['medium']})": f >= LEVEL_THRESHOLDS["medium"],
                    "anomaly flag (>= 0.55)": f >= 0.55,
                    f"score-only, high+ (>= {LEVEL_THRESHOLDS['high']})": f >= LEVEL_THRESHOLDS["high"]}
            L = matrix_rank(f, imp)
            hits["score x impact, medium+"] = L >= 2
            hits["score x impact, high+"] = L >= 3
            for name, h in hits.items():
                rows.setdefault(f"{fus} | {name}", []).append((100 * h[y == 1].mean(), 100 * h[y == 0].mean()))
    out = {}
    for k, v in rows.items():
        v = np.array(v)
        out[k] = {"recall_pct": round(float(v[:, 0].mean()), 2),
                  "benign_fpr_pct_mean": round(float(v[:, 1].mean()), 2),
                  "benign_fpr_pct_sd": round(float(v[:, 1].std(ddof=1)), 2) if len(v) > 1 else None}
    return out


def ml_branch_lift(rd: Path, seed=0):
    P = np.load(rd / f"preds_seed{seed}.npz", allow_pickle=True)
    y, ml = P["y_te"].astype(int), P["prior_ml"]
    return round(float(100 * np.mean(0.6 * ml[y == 0] >= 0.3)), 2)


def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def reliability(rd: Path, seeds, bins=15):
    P = [np.load(rd / f"preds_seed{s}.npz", allow_pickle=True) for s in seeds]
    y = np.concatenate([p["y_te"] for p in P]).astype(int)
    edges = np.linspace(0, 1, bins + 1)
    out = {"n_pooled": int(len(y))}
    for key in ("carta", "prior", "prior_nlp"):
        s = np.clip(np.concatenate([p[key] for p in P]).astype(float), 0, 1)
        idx = np.clip(np.digitize(s, edges[1:-1]), 0, bins - 1)
        rows = []
        for b in range(bins):
            m = idx == b
            n = int(m.sum())
            if n == 0:
                continue
            k = int(y[m].sum())
            lo, hi = wilson(k, n)
            rows.append({"bin": b, "n": n, "mean_pred": float(s[m].mean()), "attack_freq": k / n,
                         "ci_contains_mean_pred": bool(lo <= s[m].mean() <= hi)})
        out[key] = {"occupied_bins": len(rows), "bins_ci_contains_mean_pred": sum(r["ci_contains_mean_pred"] for r in rows),
                    "n_middle_bins_1_to_13": int(sum(r["n"] for r in rows if 1 <= r["bin"] <= bins - 2)),
                    "max_attack_freq_bins_3_to_9": max((r["attack_freq"] for r in rows if 3 <= r["bin"] <= 9), default=None),
                    "bins": rows}
    return out


def low_tier_ties(rd: Path, seed=0):
    path = rd / "bench" / f"bench_seed{seed}.pkl"
    if not path.exists():
        return "skipped: benchmark pickle not found (run run_experiments.py first)"
    from carta.features import context_frame, sensitivity_cells
    b = pd.read_pickle(path)
    va = b[b.split == "val"].reset_index(drop=True)
    P = np.load(rd / f"preds_seed{seed}.npz", allow_pickle=True)
    s, y = P["carta_va"], P["y_va"].astype(int)
    assert (va.label.to_numpy() == y).all()
    cells = sensitivity_cells(context_frame(va))
    cfg = json.load(open(rd / f"tiers_seed{seed}.json"))["carta_mondrian"]["config"]["per_tier"]
    out = {}
    for t in ("low", "medium", "high", "critical"):
        for c, tau in cfg[t]["tau"].items():
            cm = cfg[t]["cell_map"]
            mapped = np.array([cm.get(x, c) for x in cells])
            m = (y == 0) & (mapped == c)
            out[f"{t} | cell {c}"] = {"n_cal_benign": int(m.sum()), "threshold": tau,
                                      "benign_ties_at_threshold": int((s[m] == tau).sum()),
                                      "benign_above_threshold": int((s[m] > tau).sum())}
    return out


def benign_encodings(rd: Path, seed=0):
    path = rd / "bench" / f"bench_seed{seed}.pkl"
    if not path.exists():
        return "skipped: benchmark pickle not found (run run_experiments.py first)"
    from carta.normalize import unquote_smart
    b = pd.read_pickle(path)
    ben = b[b.label == 0]
    raw = (ben["path"].fillna("") + "?" + ben["query"].fillna("") + "#" + ben["body"].fillna("")).tolist()
    dec = [unquote_smart(r, plus=True) for r in raw]
    pats = {"double URL encoding (%25XX in the raw request)": (raw, re.compile(r"%25[0-9A-Fa-f]{2}")),
            "HTML numeric entity": (dec, re.compile(r"&#x?[0-9A-Fa-f]+;")),
            "HTML named entity": (dec, re.compile(r"&(lt|gt|quot|amp|apos);")),
            "full-width characters": (dec, re.compile(r"[\uff01-\uff5e]")),
            "inline comment /*...*/": (dec, re.compile(r"/\*.*?\*/")),
            "tab, newline or vertical tab": (dec, re.compile(r"[\t\n\r\x0b]"))}
    out = {"n_benign": len(raw)}
    for k, (arr, rx) in pats.items():
        n = sum(1 for x in arr if rx.search(x))
        out[k] = {"count": n, "pct": round(100 * n / len(arr), 4)}
    return out


def log_signature_requests(rd: Path, seed=0):
    """Benign-labelled log requests with an unambiguous attack signature, over all splits of the benchmark
    (the benchmark is the same for every seed; only its split differs)."""
    path = rd / "bench" / f"bench_seed{seed}.pkl"
    if not path.exists():
        return "skipped: benchmark pickle not found (run run_experiments.py first)"
    from label_noise import signature
    b = pd.read_pickle(path)
    logs = b[(b["source"] == "weblogs2019") & (b["label"] == 0)]
    hits = [(sp, f"{m} {p}?{q}"[:160]) for sp, m, p, q in zip(logs["split"], logs["method"], logs["path"], logs["query"])
            if signature(p, q)]
    return {"n_benign_log_requests": int(len(logs)), "n_with_signature": len(hits),
            "by_split": {s: sum(1 for sp, _ in hits if sp == s) for s in ("train", "val", "test")},
            "requests": sorted(r for _, r in hits)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    a = ap.parse_args()
    rd = Path(a.results)
    seeds = seeds_of(rd)
    res = {"seeds": seeds,
           "prior_operating_points": prior_operating_points(rd, seeds),
           "ml_branch_lift_seed0_pct": ml_branch_lift(rd, 0),
           "reliability": reliability(rd, seeds),
           "low_tier_ties_seed0": low_tier_ties(rd, 0),
           "benign_encodings_seed0": benign_encodings(rd, 0),
           "log_signature_requests_benchmark": log_signature_requests(rd, 0)}
    (rd / "extra_numbers.json").write_text(json.dumps(res, indent=1))
    print("prior operating points (recall %, benign FPR % mean ± s.d.):")
    for k, v in res["prior_operating_points"].items():
        print(f"  {k:48s} {v['recall_pct']:7.2f}  {v['benign_fpr_pct_mean']:6.2f} ± {v['benign_fpr_pct_sd']}")
    print("ML branch alone lifts", res["ml_branch_lift_seed0_pct"], "% of benign test requests (seed 0) to >= 0.3")
    for k in ("carta", "prior", "prior_nlp"):
        r = res["reliability"][k]
        print(f"reliability {k:9s}: {r['bins_ci_contains_mean_pred']} of {r['occupied_bins']} bins contain the mean "
              f"prediction; middle-bin requests {r['n_middle_bins_1_to_13']} of {res['reliability']['n_pooled']}; "
              f"max attack frequency in bins 3-9: {r['max_attack_freq_bins_3_to_9']}")
    print("ties:", json.dumps(res["low_tier_ties_seed0"], indent=0)[:600])
    print("benign encodings:", json.dumps(res["benign_encodings_seed0"]))
    ls = res["log_signature_requests_benchmark"]
    print("benign-labelled log requests with an attack signature:",
          ls if isinstance(ls, str) else f"{ls['n_with_signature']} of {ls['n_benign_log_requests']} {ls['by_split']}")


if __name__ == "__main__":
    main()
