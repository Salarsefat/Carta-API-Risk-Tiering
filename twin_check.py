#!/usr/bin/env python3
"""Near-duplicate check: GET/POST twins.

CSIC 2010 often contains the same request twice, once as GET (parameters in the query string) and
once as POST (parameters in the body). Their canonical keys differ, so the benchmark keeps both and
they can fall into different splits. A *content twin* of a test request is a training request with
the same path and the same multiset of (name, value) parameters, ignoring the method and whether
the parameters sit in the query or the body. This script counts test requests with a twin in the
training split and recomputes the threshold-free test metrics without them (scores are reused;
nothing is refitted).

  python twin_check.py --results results
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from carta.metrics import summary  # noqa: E402
from carta.normalize import twin_key  # noqa: E402


content_twin_key = twin_key  # GET/POST- and query/body-invariant key (carta/normalize.py)


def run(rd: Path, seed: int) -> dict:
    b = pd.read_pickle(rd / "bench" / f"bench_seed{seed}.pkl")
    tr = b[b.split == "train"]
    te = b[b.split == "test"].reset_index(drop=True)
    P = np.load(rd / f"preds_seed{seed}.npz", allow_pickle=True)
    y = P["y_te"].astype(int)
    assert (y == te["label"].to_numpy()).all()
    k_tr = set(content_twin_key(m, p, q, bd) for m, p, q, bd in zip(tr["method"], tr["path"], tr["query"], tr["body"]))
    twin = np.array([content_twin_key(m, p, q, bd) in k_tr for m, p, q, bd in zip(te["method"], te["path"], te["query"], te["body"])])
    out = {"seed": seed, "n_test": int(len(te)), "n_twin": int(twin.sum()),
           "twin_by_source_label": {f"{s}|{lab}": int(((te["source"] == s) & (te["label"] == lab) & twin).sum())
                                    for s in sorted(te["source"].unique()) for lab in (0, 1)},
           "test_by_source_label": {f"{s}|{lab}": int(((te["source"] == s) & (te["label"] == lab)).sum())
                                    for s in sorted(te["source"].unique()) for lab in (0, 1)}}
    keep = ~twin
    for scope, m in (("all", np.ones(len(te), bool)), ("csic2010", te["source"].to_numpy() == "csic2010")):
        out[scope] = {}
        for name in ("carta", "prior"):
            s = P[name].astype(float)
            full, nt = summary(y[m], s[m]), summary(y[m & keep], s[m & keep])
            out[scope][name] = {sub: {k: v[k] for k in ("roc_auc", "recall@fpr0.001", "recall@fpr0.01")}
                                for sub, v in (("with_twins", full), ("without_twins", nt))}
        out[scope]["n_without_twins"] = int((m & keep).sum())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    a = ap.parse_args()
    rd = Path(a.results)
    for f in sorted(glob.glob(str(rd / "preds_seed*.npz"))):
        seed = int(Path(f).stem.replace("preds_seed", ""))
        out = run(rd, seed)
        (rd / f"twin_check_seed{seed}.json").write_text(json.dumps(out, indent=1, default=float))
        c = out["all"]
        print(f"seed {seed}: twins {out['n_twin']}/{out['n_test']} {out['twin_by_source_label']} | R@0.1% all: "
              f"CARTA {100*c['carta']['with_twins']['recall@fpr0.001']:.2f} -> {100*c['carta']['without_twins']['recall@fpr0.001']:.2f}, "
              f"prior {100*c['prior']['with_twins']['recall@fpr0.001']:.2f} -> {100*c['prior']['without_twins']['recall@fpr0.001']:.2f} | CSIC: "
              f"CARTA {100*out['csic2010']['carta']['with_twins']['recall@fpr0.001']:.2f} -> {100*out['csic2010']['carta']['without_twins']['recall@fpr0.001']:.2f}, "
              f"prior {100*out['csic2010']['prior']['with_twins']['recall@fpr0.001']:.2f} -> {100*out['csic2010']['prior']['without_twins']['recall@fpr0.001']:.2f}",
              flush=True)


if __name__ == "__main__":
    main()
