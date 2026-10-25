#!/usr/bin/env python3
"""Blind-spot check: MySQL executable comments.

MySQL executes the content of /*! ... */ and /*!50000 ... */ comments. CARTA's normaliser strips
every /* ... */ comment together with its content, so SQL keywords wrapped this way vanish from
CARTA's text views (the raw-form request statistics still see the comment characters). This script
measures the effect with the saved seed-0 models and their validation 1%-FPR thresholds:

  * how often such comments already occur in the benchmark;
  * recall on SQL-injection test attacks before and after wrapping each SQL keyword of each
    parameter value in /*! ... */ (a semantics-preserving rewrite for MySQL targets).

Nothing is refitted.

  python versioned_comment_check.py --results results --seed 0
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from carta.evasion import transform_request  # noqa: E402
from carta.normalize import add_text_views  # noqa: E402

KW = re.compile(r"\b(union|select|and|or|from|where|sleep|benchmark|order|by|group|having|insert|update|delete|drop|exec|waitfor|delay|concat|char|null)\b", re.I)
VERSIONED = re.compile(r"/\*!")


def wrap_keywords(v: str, rng) -> str:
    return KW.sub(lambda m: "/*!" + m.group(0) + "*/", v)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rd = Path(a.results)
    M = joblib.load(rd / f"models_seed{a.seed}.joblib")
    carta, prior, tau_c, tau_p = M["carta"], M["prior"], M["tau_c"], M["tau_p"]
    from carta.metrics import threshold_at_fpr
    P = np.load(rd / f"preds_seed{a.seed}.npz", allow_pickle=True)
    y_va = P["y_va"].astype(int)
    tau_c01, tau_p01 = threshold_at_fpr(y_va, P["carta_va"], 0.001), threshold_at_fpr(y_va, P["prior_va"], 0.001)
    b = pd.read_pickle(rd / "bench" / f"bench_seed{a.seed}.pkl")
    raw = b["path"].fillna("") + "?" + b["query"].fillna("") + "#" + b["body"].fillna("")
    # occurrence in the benchmark (raw text after percent-decoding only)
    from urllib.parse import unquote
    has_v = np.array([bool(VERSIONED.search(unquote(unquote(s)))) for s in raw])
    out = {"seed": a.seed, "benchmark_rows_with_versioned_comment": {
        "benign": int((has_v & (b["label"] == 0)).sum()), "attack": int((has_v & (b["label"] == 1)).sum())}}
    te = b[b.split == "test"].reset_index(drop=True)
    sq = te[(te["label"] == 1) & te["attack_type"].astype(str).str.contains("sqli")].reset_index(drop=True)
    rng = random.Random(a.seed)
    rows = [transform_request(m, p, q, bd, wrap_keywords, rng) for m, p, q, bd in zip(sq["method"], sq["path"], sq["query"], sq["body"])]
    tr = sq.copy()
    tr["method"], tr["path"], tr["query"], tr["body"] = zip(*rows)
    tr = add_text_views(tr)
    changed = (tr["query"].to_numpy() != sq["query"].to_numpy()) | (tr["body"].to_numpy() != sq["body"].to_numpy())
    res = {}
    for name, df in (("clean", sq), ("versioned_comments", tr)):
        pc = carta.predict_proba(df)
        pp = prior.predict(df)["final"]
        res[name] = {"carta_recall": float((pc >= tau_c).mean()), "prior_recall": float((pp >= tau_p).mean()),
                     "carta_recall_val_0.1pct": float((pc >= tau_c01).mean()), "prior_recall_val_0.1pct": float((pp >= tau_p01).mean()),
                     "carta_recall_on_changed": float((pc[changed] >= tau_c).mean()),
                     "prior_recall_on_changed": float((pp[changed] >= tau_p).mean())}
    out["sqli_test_attacks"] = int(len(sq))
    out["sqli_attacks_rewritten"] = int(changed.sum())
    out["recall_at_val_1pct_fpr_threshold"] = res
    ex = tr[changed].head(3)
    out["examples"] = [{"before": f"{m} {p}?{q} {bd}"[:200], "after": f"{m2} {p2}?{q2} {bd2}"[:240], "carta_text_after": t[:200]}
                       for (m, p, q, bd), (m2, p2, q2, bd2, t) in zip(sq[changed].head(3)[["method", "path", "query", "body"]].itertuples(index=False),
                                                                      ex[["method", "path", "query", "body", "text"]].itertuples(index=False))]
    (rd / f"versioned_comment_check_seed{a.seed}.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))
    print(json.dumps(out, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
