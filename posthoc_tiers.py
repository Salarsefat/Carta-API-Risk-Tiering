#!/usr/bin/env python3
"""Conformal risk tiers computed post hoc from saved validation/test scores (no refitting).

For each seed: CARTA scores and prior-pipeline scores are each tiered with
  (i) Mondrian conformal tiers with context-dependent budgets, and (ii) global conformal tiers,
both calibrated on the validation split, evaluated on the test split; the prior pipeline's own
shipped levels (0.3/0.5/0.6/0.8) are reported alongside.

  python posthoc_tiers.py --results results
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
from carta.baseline import impact_scores  # noqa: E402
from carta.features import context_frame, sensitivity_cells  # noqa: E402
from carta.proposed import DEFAULT_BUDGETS, ConformalTiers  # noqa: E402

TIER_ORDER = ["normal", "low", "medium", "high", "critical"]


def tier_report(y, tiers, cells, impact, budgets_fn=None):
    y = np.asarray(y).astype(int)
    rank = {t: i for i, t in enumerate(TIER_ORDER)}
    r = np.array([rank[t] for t in tiers])
    out = {}
    for t in TIER_ORDER[1:]:
        hit = r >= rank[t]
        row = {"fpr": float(hit[y == 0].mean()), "recall": float(hit[y == 1].mean()),
               "fp": int((hit & (y == 0)).sum()), "tp": int((hit & (y == 1)).sum()),
               "iw_recall": float((impact[y == 1] * hit[y == 1]).sum() / max(1e-12, impact[y == 1].sum())),
               "cells": {}}
        if budgets_fn is not None:
            row["effective_budget"] = float(budgets_fn(t)[y == 0].mean())
        for c in np.unique(cells):
            mb, ma = (cells == c) & (y == 0), (cells == c) & (y == 1)
            cell = {"n_benign": int(mb.sum()), "n_attack": int(ma.sum()),
                    "fpr": float(hit[mb].mean()) if mb.any() else None,
                    "recall": float(hit[ma].mean()) if ma.any() else None}
            if budgets_fn is not None and mb.any():
                cell["budget"] = float(budgets_fn(t)[mb].mean())
            row["cells"][str(c)] = cell
        out[t] = row
    return out


def resplit_check(scores, y, cells, budgets, reps=200, seed=0):
    """Validity check: the held-out data (validation + test) is re-split at random into a
    calibration half and an evaluation half `reps` times, with the trained model fixed.
    Returns, per tier and mode, the mean benign FPR on the evaluation halves, its s.d.,
    the mean recall, and the mean exact conformal bound (n+1-k)/(n+1) aggregated over cells."""
    rng = np.random.RandomState(seed)
    n = len(y)
    out = {}
    for mode in ("mondrian", "global"):
        acc = {t: {"fpr": [], "recall": [], "bound": []} for t in ConformalTiers.ORDER}
        for _ in range(reps):
            perm = rng.permutation(n)
            cal, ev = perm[: n // 2], perm[n // 2:]
            ct = ConformalTiers(budgets, mondrian=(mode == "mondrian")).fit(scores[cal], y[cal], cells[cal])
            T = ct.thresholds(cells[ev])
            for j, t in enumerate(ConformalTiers.ORDER):
                hit = scores[ev] > T[:, j]
                acc[t]["fpr"].append(float(hit[y[ev] == 0].mean()))
                acc[t]["recall"].append(float(hit[y[ev] == 1].mean()))
                # exact expected FPR of the per-cell split-conformal thresholds, benign-weighted
                part = ct.parts[t]
                ce = ct.cell_of(cells[ev], t)
                bnd = 0.0
                ben_ev = (y[ev] == 0)
                for c, m in part["mult"].items():
                    nc = part["n_cal"][c]
                    eps = min(ct.cap, budgets[t] * m)
                    k = int(np.ceil((nc + 1) * (1 - eps)))
                    b = (nc + 1 - k) / (nc + 1) if k <= nc else 0.0
                    bnd += b * float((ce[ben_ev] == c).mean())
                acc[t]["bound"].append(bnd)
        out[mode] = {t: {"mean_fpr": float(np.mean(v["fpr"])), "sd_fpr": float(np.std(v["fpr"])),
                         "mean_recall": float(np.mean(v["recall"])), "mean_exact_bound": float(np.mean(v["bound"])),
                         "budget": budgets[t]} for t, v in acc.items()}
    out["reps"] = reps
    return out


def run(rd: Path, seed: int, budgets: dict) -> dict:
    bench = pd.read_pickle(rd / "bench" / f"bench_seed{seed}.pkl")
    va = bench[bench.split == "val"].reset_index(drop=True)
    te = bench[bench.split == "test"].reset_index(drop=True)
    P = np.load(rd / f"preds_seed{seed}.npz", allow_pickle=True)
    y_va, y_te = P["y_va"].astype(int), P["y_te"].astype(int)
    assert len(y_va) == len(va) and len(y_te) == len(te) and (y_va == va.label.to_numpy()).all() \
        and (y_te == te.label.to_numpy()).all(), "prediction files do not match the benchmark split"
    ctx_va, ctx_te = context_frame(va), context_frame(te)
    cells_va, cells_te = sensitivity_cells(ctx_va), sensitivity_cells(ctx_te)
    impact_te = impact_scores(ctx_te)
    out = {"seed": seed, "budgets": budgets,
           "n_val_benign_per_cell": {str(c): int(((cells_va == c) & (y_va == 0)).sum()) for c in np.unique(cells_va)},
           "n_test_benign_per_cell": {str(c): int(((cells_te == c) & (y_te == 0)).sum()) for c in np.unique(cells_te)}}
    for name, sv, st in (("carta", P["carta_va"], P["carta"]), ("prior", P["prior_va"], P["prior"])):
        for mode in ("mondrian", "global"):
            ct = ConformalTiers(budgets, mondrian=(mode == "mondrian")).fit(sv, y_va, cells_va)
            tiers = ct.transform(st, cells_te)
            out[f"{name}_{mode}"] = {"config": ct.describe(),
                                     "test": tier_report(y_te, tiers, cells_te, impact_te,
                                                         lambda t, ct=ct: ct.cell_budget(cells_te, t))}
    out["prior_native_levels"] = {"test": tier_report(y_te, P["prior_level"].astype(str), cells_te, impact_te)}
    allp = np.concatenate([P["carta_va"], P["carta"]])
    ally = np.concatenate([y_va, y_te])
    allc = np.concatenate([cells_va, cells_te])
    out["resplit_check_carta"] = resplit_check(allp, ally, allc, budgets, reps=200, seed=seed)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    a = ap.parse_args()
    rd = Path(a.results)
    for f in sorted(glob.glob(str(rd / "preds_seed*.npz"))):
        seed = int(Path(f).stem.replace("preds_seed", ""))
        res = run(rd, seed, dict(DEFAULT_BUDGETS))
        (rd / f"tiers_seed{seed}.json").write_text(json.dumps(res, indent=1, default=float))
        rs = res["resplit_check_carta"]
        print(f"seed {seed} resplit (200x): " + "  ".join(
            f"{t}: M {100*rs['mondrian'][t]['mean_fpr']:.3f}% (bound {100*rs['mondrian'][t]['mean_exact_bound']:.3f}%) "
            f"G {100*rs['global'][t]['mean_fpr']:.3f}% (bound {100*rs['global'][t]['mean_exact_bound']:.3f}%)"
            for t in ("low", "medium", "high", "critical")))
        m, g = res["carta_mondrian"]["test"], res["carta_global"]["test"]
        print(f"seed {seed}: " + "  ".join(f"{t}: M fpr={m[t]['fpr']:.4f}/{m[t]['effective_budget']:.4f} rec={m[t]['recall']:.3f} "
                                          f"iw={m[t]['iw_recall']:.3f} | G fpr={g[t]['fpr']:.4f} rec={g[t]['recall']:.3f} iw={g[t]['iw_recall']:.3f}"
                                          for t in ("medium", "high")))


if __name__ == "__main__":
    main()
