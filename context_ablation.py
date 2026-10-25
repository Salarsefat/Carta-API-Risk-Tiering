#!/usr/bin/env python3
"""Context features in the text-only and the tabular-only stacks (saved models, seed 0 by default).

In the reported ablation (run_experiments.py, Table 13 of the paper) the text-only stack gets no context
features in the meta-learner, whereas the tabular-only stack keeps them. This script evaluates both stacks
with and without them on the same test split. As in the ablation, only the calibration maps, the logistic
meta-learner and the tiers are refitted (on the cached out-of-fold scores); the base learners are reused.

  python context_ablation.py --results results --seed 0
Writes results/context_ablation_seed{seed}.json (needs models_seed{seed}.joblib and bench/bench_seed{seed}.pkl).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from carta.metrics import summary  # noqa: E402
from carta.proposed import static_block  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rd = Path(a.results)
    carta = joblib.load(rd / f"models_seed{a.seed}.joblib")["carta"]
    b = pd.read_pickle(rd / "bench" / f"bench_seed{a.seed}.pkl")
    te = b[b.split == "test"].reset_index(drop=True)
    y = te["label"].to_numpy().astype(int)
    S = static_block(te)
    base = carta.base_predict(te, S)
    full = list(carta.base_names)
    out = {"seed": a.seed, "n_test": int(len(te))}
    for stack, keep in (("text_only(char+struct)", [x for x in full if x in ("char", "struct")]),
                        ("tabular_only(gbdt+iforest)", [x for x in full if x in ("gbdt", "iforest")])):
        for ctx in (False, True):
            v = carta.variant(keep=keep, use_context_meta=ctx)
            r = summary(y, v.proba_from_base(base, S["ctx"]))
            key = f"{stack} | context features in the meta-learner: {'yes' if ctx else 'no'}"
            out[key] = {k: r[k] for k in ("roc_auc", "recall@fpr0.001", "recall@fpr0.01", "ece")}
            print(f"{key:75s} R@0.1% {100 * r['recall@fpr0.001']:6.2f}  R@1% {100 * r['recall@fpr0.01']:6.2f}  "
                  f"ROC-AUC {r['roc_auc']:.4f}  ECE {r['ece']:.3f}")
    (rd / f"context_ablation_seed{a.seed}.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
