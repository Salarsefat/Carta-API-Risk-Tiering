#!/usr/bin/env python3
"""Single-request latency and batch throughput of saved models, measured in isolation.

  python latency_bench.py --results results --seed 0 --n 500
Loads models_seed{seed}.joblib and bench/bench_seed{seed}.pkl written by run_experiments.py.
Timing covers the end-to-end path from raw request fields (text views, features, all models).
Writes latency_seed{seed}.json (CARTA, prior pipeline, CARTA per stage) and
latency_carta_without_iforest_seed{seed}.json (CARTA without its Isolation Forest, with its test recall).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
import joblib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from carta.metrics import recall_at_fpr  # noqa: E402
from carta.normalize import add_text_views  # noqa: E402
from carta.proposed import CARTA, static_block  # noqa: E402

RAW = ["source", "method", "path", "query", "body", "headers", "label", "attack_type"]


def pct(v):
    return {"p50": float(np.percentile(v, 50)), "p95": float(np.percentile(v, 95)),
            "p99": float(np.percentile(v, 99)), "mean": float(np.mean(v))}


def without_iforest(carta: CARTA) -> CARTA:
    """CARTA without its Isolation Forest: calibration maps, meta-learner and tiers are refitted on the cached
    out-of-fold scores of the other base learners (as in the ablation of run_experiments.py), and the forest
    is removed from the tabular stage so that it is not computed at inference."""
    v = carta.variant(keep=[b for b in carta.base_names if b != "iforest"])
    v.models = dict(carta.models)
    v.models["tab"] = {k: m for k, m in carta.models["tab"].items() if k not in ("iforest", "if_scaler")}
    return v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n", type=int, default=500)
    a = ap.parse_args()
    rd = Path(a.results)
    M = joblib.load(rd / f"models_seed{a.seed}.joblib")
    carta, prior = M["carta"], M["prior"]
    bench = pd.read_pickle(rd / "bench" / f"bench_seed{a.seed}.pkl")
    te = bench[bench.split == "test"].reset_index(drop=True)
    sample = te.sample(n=min(a.n, len(te)), random_state=a.seed).reset_index(drop=True)[RAW]
    warm = min(30, len(sample))
    for i in range(warm):
        row = sample.iloc[[i]]
        carta.predict_proba(row, raw=True)
        prior.predict(add_text_views(row))
    lat = {"carta_ms": [], "prior_ms": []}
    st = {k: [] for k in ("text_views", "static_features", "text_models", "tabular_models", "meta")}
    for i in range(len(sample)):
        row = sample.iloc[[i]]
        t0 = time.perf_counter(); carta.predict_proba(row, raw=True); lat["carta_ms"].append((time.perf_counter() - t0) * 1e3)
        t0 = time.perf_counter(); prior.predict(add_text_views(row)); lat["prior_ms"].append((time.perf_counter() - t0) * 1e3)
        t0 = time.perf_counter(); r2 = add_text_views(row); t1 = time.perf_counter()
        S = static_block(r2); t2 = time.perf_counter()
        b = {k: carta.models[k].predict_proba(r2["text"].tolist()) for k in carta.base_names if k in CARTA.TEXT_BASES}
        t3 = time.perf_counter()
        b.update(carta._predict_tabular(carta.models["tab"], r2, S)); t4 = time.perf_counter()
        carta.proba_from_base(b, S["ctx"]); t5 = time.perf_counter()
        for k, x, y in (("text_views", t0, t1), ("static_features", t1, t2), ("text_models", t2, t3),
                        ("tabular_models", t3, t4), ("meta", t4, t5)):
            st[k].append((y - x) * 1e3)
    raw_te = te[RAW]
    t0 = time.perf_counter(); carta.predict_proba(raw_te, raw=True); thr_c = len(te) / (time.perf_counter() - t0)
    t0 = time.perf_counter(); prior.predict(add_text_views(raw_te)); thr_p = len(te) / (time.perf_counter() - t0)
    out = {"seed": a.seed, "n": len(sample), "latency_ms": {k: pct(v) for k, v in lat.items()},
           "latency_stage_ms_carta": {k: pct(v) for k, v in st.items()},
           "throughput_eps": {"carta": thr_c, "prior": thr_p, "n_batch": len(te)},
           "cpu": os.popen("lscpu | grep 'Model name' | head -1").read().strip(), "threads": os.environ.get("OMP_NUM_THREADS")}
    (rd / f"latency_seed{a.seed}.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))

    # CARTA without the Isolation Forest: same sample, same protocol, and its recall on the test split
    v = without_iforest(carta)
    for i in range(warm):
        v.predict_proba(sample.iloc[[i]], raw=True)
    lat_v = []
    for i in range(len(sample)):
        row = sample.iloc[[i]]
        t0 = time.perf_counter(); v.predict_proba(row, raw=True); lat_v.append((time.perf_counter() - t0) * 1e3)
    t0 = time.perf_counter(); v.predict_proba(raw_te, raw=True); thr_v = len(te) / (time.perf_counter() - t0)
    y, p = te["label"].to_numpy().astype(int), v.predict_proba(te)
    q = pct(lat_v)
    out_v = {"p50": q["p50"], "p95": q["p95"], "p99": q["p99"], "throughput_eps": thr_v,
             "recall@fpr0.001": recall_at_fpr(y, p, 0.001)[0], "recall@fpr0.01": recall_at_fpr(y, p, 0.01)[0],
             "roc_auc": float(roc_auc_score(y, p))}
    (rd / f"latency_carta_without_iforest_seed{a.seed}.json").write_text(json.dumps(out_v, indent=1))
    print(json.dumps(out_v, indent=1))


if __name__ == "__main__":
    main()
