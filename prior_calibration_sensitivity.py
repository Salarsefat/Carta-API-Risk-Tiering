#!/usr/bin/env python3
"""Where the prior pipeline fits its ML-branch calibration (seed 0 by default).

The prior paper fits the anomaly models on benign rows, and the main results calibrate the three ML scores
(z-score + sigmoid) on benign training rows as well. The released code calibrates on all training rows. This
script refits the re-implemented prior pipeline both ways on the same benchmark split and reports ranking
quality and the score-only operating points on the test split.

  python prior_calibration_sensitivity.py --results results --seed 0
Writes results/prior_calibration_sensitivity_seed{seed}.json (needs results/bench/bench_seed{seed}.pkl).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from carta.baseline import PriorPipeline, ZCal  # noqa: E402
from carta.features import numeric_matrix  # noqa: E402
from carta.metrics import confusion_at, recall_at_fpr  # noqa: E402


class PriorAllRowCalibration(PriorPipeline):
    """Same models (fitted on benign rows); the three z-score calibrators are fitted on all training rows."""

    def _fit_ml(self, df):
        super()._fit_ml(df)
        Xs = self.scaler.transform(numeric_matrix(df, self.method_table))
        self.cal_if = ZCal().fit(-self.iforest.decision_function(Xs))
        self.cal_oc = ZCal().fit(-self.ocsvm.decision_function(Xs))
        self.cal_ae = ZCal().fit(((self.ae.predict(Xs) - Xs) ** 2).mean(axis=1))


def evaluate(model, te):
    y = te.label.to_numpy().astype(int)
    s = model.predict(te)["final"]
    return {"roc_auc": float(roc_auc_score(y, s)),
            "recall@fpr0.001": recall_at_fpr(y, s, 0.001)[0],
            "recall@fpr0.01": recall_at_fpr(y, s, 0.01)[0],
            "low+ (>=0.3)": confusion_at(y, s, 0.3),
            "medium+ (>=0.5)": confusion_at(y, s, 0.5),
            "anomaly (>=0.55)": confusion_at(y, s, 0.55)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rd = Path(a.results)
    bench = pd.read_pickle(rd / "bench" / f"bench_seed{a.seed}.pkl")
    tr, te = (bench[bench.split == s].reset_index(drop=True) for s in ("train", "test"))
    out = {"benign-row calibration (paper text, used in main results)": evaluate(PriorPipeline(seed=a.seed).fit(tr), te),
           "all-row calibration (release behaviour)": evaluate(PriorAllRowCalibration(seed=a.seed).fit(tr), te)}
    path = rd / f"prior_calibration_sensitivity_seed{a.seed}.json"
    path.write_text(json.dumps(out, indent=1))
    for k, v in out.items():
        print(k, {m: (round(x, 4) if isinstance(x, float) else round(x["fpr"], 4)) for m, x in v.items()})
    print("written:", path)


if __name__ == "__main__":
    main()
