#!/usr/bin/env python3
"""Unseen-site deployment test on the unlabelled 'Access Log' dataset (Kaggle dsfelix, 2009).

The log has no labels; the prior work used it as benign traffic. We therefore report *alert
rates* (not FPR) of both detectors at their validation-chosen thresholds, then show the effect of
site-specific conformal recalibration: thresholds are re-estimated on the first half of the new
site's (unlabelled) traffic and alert rates are measured on the second half.

The dataset states no licence, so the saved JSON holds SHA-256 hashes of the exact request strings
instead of the requests; the top-ranked requests are printed. --annotation-template also writes the
sheet for inspecting the top-ranked requests (results/site_shift_top100_annotation_seed0.csv is that
sheet completed by hand).

  python site_shift.py --results results --seed 0 --log access_log.zip --n 20000 --rate 0.45
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import zipfile
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from carta.data import parse_clf_line  # noqa: E402
from carta.normalize import add_text_views, canonical_key  # noqa: E402
from carta.proposed import conformal_threshold  # noqa: E402
from label_noise import signature  # noqa: E402

NOTE = ("Raw request strings of the Access Log dataset are replaced by SHA-256 hashes of the exact strings; "
        "run site_shift.py on the downloaded dataset to see them.")


def sample_log(path, n, rate=0.05, seed=0, max_lines=None):
    rows, seen = [], set()
    opener = zipfile.ZipFile(path).open(zipfile.ZipFile(path).infolist()[0]) if str(path).endswith(".zip") else open(path, "rb")
    with opener as f:
        for i, raw in enumerate(f):
            if max_lines and i >= max_lines:
                break
            line = raw.decode("utf-8", "replace")
            r = parse_clf_line(line)
            if r is None:
                continue
            k = f"{r['method']} {r['path']}?{r['query']}"
            h = int(hashlib.md5(f"{seed}:{k}".encode()).hexdigest()[:12], 16) % 1_000_000
            if h >= rate * 1_000_000 or k in seen:
                continue
            seen.add(k)
            r.update({"source": "accesslog", "label": 0, "attack_type": "unlabelled"})
            rows.append(r)
    df = pd.DataFrame(rows)
    df["key"] = [canonical_key(m, p, q, b) for m, p, q, b in zip(df["method"], df["path"], df["query"], df["body"])]
    df = df.drop_duplicates("key").sample(frac=1.0, random_state=seed).head(n).reset_index(drop=True)
    return add_text_views(df)


def sha(s: str) -> str:
    return "sha256:" + hashlib.sha256(s.encode("utf-8")).hexdigest()


def publishable(out: dict) -> dict:
    """The result dict with every raw request string replaced by its hash."""
    pub = {k: v for k, v in out.items() if k not in ("top_carta", "top_prior", "top_carta_requests")}
    for name in ("top_carta", "top_prior"):
        pub[name] = [{"rank": r["rank"], "score": r["score"], "signature": r["signature"], "request_sha256": sha(r["request"])}
                     for r in out[name]]
    pub["top_carta_requests_sha256"] = [sha(s) for s in out["top_carta_requests"]]
    pub["note"] = NOTE
    return pub


def write_template(out: dict, path: Path):
    """Annotation sheet: requests with a signature are pre-labelled from it, the others are left blank."""
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["detector", "rank", "score", "signature", "label", "evidence", "request_sha256"])
        for det in ("carta", "prior"):
            for r in out[f"top_{det}"]:
                sg = r["signature"]
                w.writerow([det, r["rank"], f"{r['score']:.12f}", sg, "attack" if sg else "",
                            "signature:" + sg.replace(",", ";") if sg else "", sha(r["request"])])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--log", required=True)
    ap.add_argument("--n", type=int, default=20000)
    ap.add_argument("--rate", type=float, default=0.05)
    ap.add_argument("--max-lines", type=int, default=None)
    ap.add_argument("--top", type=int, default=100, help="how many top-ranked requests per detector to save for inspection")
    ap.add_argument("--annotation-template", action="store_true",
                    help="also write site_shift_top{top}_template_seed{seed}.csv for manual inspection")
    a = ap.parse_args()
    rd = Path(a.results)
    M = joblib.load(rd / f"models_seed{a.seed}.joblib")
    carta, prior, tau_c, tau_p = M["carta"], M["prior"], M["tau_c"], M["tau_p"]
    df = sample_log(a.log, a.n, a.rate, a.seed, a.max_lines)
    pc = carta.predict_proba(df)
    pp = prior.predict(df)["final"]
    half = len(df) // 2
    out = {"n_requests": len(df), "unique_paths": int(df["path"].nunique()),
           "alert_rate_at_val_1pct_fpr_threshold": {"carta": float((pc >= tau_c).mean()), "prior": float((pp >= tau_p).mean())},
           "alert_rate_prior_shipped_0.5": float((pp >= 0.5).mean()),
           "recalibrated": {}}
    for eps in (0.05, 0.01, 0.002):
        tc = conformal_threshold(pc[:half], eps)
        tp = conformal_threshold(pp[:half], eps)
        out["recalibrated"][str(eps)] = {"carta_alert_rate_eval_half": float((pc[half:] > tc).mean()),
                                         "prior_alert_rate_eval_half": float((pp[half:] > tp).mean())}
    sig = np.array([bool(signature(p_, q_)) for p_, q_ in zip(df["path"], df["query"])])
    out["signature_attacks"] = {"n": int(sig.sum()), "fraction": float(sig.mean()),
                                "carta_alert_rate_on_them": float((pc[sig] >= tau_c).mean()) if sig.any() else None,
                                "prior_alert_rate_on_them": float((pp[sig] >= tau_p).mean()) if sig.any() else None}
    for k in (50, 100):
        out[f"signature_precision_top{k}"] = {"carta": float(sig[np.argsort(-pc, kind="mergesort")[:k]].mean()),
                                              "prior": float(sig[np.argsort(-pp, kind="mergesort")[:k]].mean())}
    out["alert_rate_excluding_signature_attacks"] = {"carta": float((pc[~sig] >= tau_c).mean()),
                                                     "prior": float((pp[~sig] >= tau_p).mean())}
    ranked = df.assign(carta=pc, prior=pp, signature=[",".join(signature(p_, q_)) for p_, q_ in zip(df["path"], df["query"])])
    for name in ("carta", "prior"):
        top = ranked.sort_values(name, ascending=False, kind="mergesort").head(a.top)
        out[f"top_{name}"] = [{"rank": i + 1, "score": float(sc), "signature": sg, "request": f"{m} {p}?{q}"}
                              for i, (sc, sg, m, p, q) in enumerate(zip(top[name], top["signature"], top["method"], top["path"], top["query"]))]
    out["top_carta_requests"] = [r["request"][:160] for r in out["top_carta"][:25]]
    (rd / f"site_shift_seed{a.seed}.json").write_text(json.dumps(publishable(out), indent=1, ensure_ascii=False))
    if a.annotation_template:
        write_template(out, rd / f"site_shift_top{a.top}_template_seed{a.seed}.csv")
    print(json.dumps({k: v for k, v in out.items() if not k.startswith("top_")}, indent=1))
    print("\n".join(out["top_carta_requests"][:15]))  # raw requests are printed, not saved


if __name__ == "__main__":
    main()
