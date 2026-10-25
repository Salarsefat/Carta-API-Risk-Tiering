#!/usr/bin/env python3
"""Per-cell validity check of the conformal risk tiers (complements posthoc_tiers.resplit_check).

posthoc_tiers.py checks the benign-weighted aggregate FPR of each tier over random re-splits.
This script checks the claim that matters for Mondrian tiers: the false-positive rate *inside
each sensitivity cell*. For each seed, validation and test are pooled and redrawn 200 times into
a calibration half and an evaluation half (trained model fixed). For every tier, cell and mode
(Mondrian, global) it reports the mean and s.d. of the benign FPR in the evaluation half, the
mean recall, the cell's intended budget eps_t * m_c, and, for Mondrian, the exact finite-sample
bound (n_c + 1 - k) / (n_c + 1) of the cell's split-conformal threshold.

Run with original labels and with the corrected labels of label_noise.py (benign-labelled log
URLs with unambiguous attack signatures counted as attacks, in both validation and test).

  python cell_resplit.py --results results --reps 200
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
from carta.features import context_frame, sensitivity_cells  # noqa: E402
from carta.proposed import DEFAULT_BUDGETS, ConformalTiers  # noqa: E402
from label_noise import signature  # noqa: E402

CELLS = ("high", "std", "low")


def corrected_labels(df: pd.DataFrame) -> np.ndarray:
    y = df["label"].to_numpy().astype(int).copy()
    flag = np.array([(s == "weblogs2019" and lab == 0 and bool(signature(p, q)))
                     for s, lab, p, q in zip(df["source"], y, df["path"], df["query"])])
    y[flag] = 1
    return y


def check(scores, y, cells, budgets, reps, seed):
    rng = np.random.RandomState(seed)
    n = len(y)
    perms = [rng.permutation(n) for _ in range(reps)]  # the same re-splits for both modes (paired)
    out = {}
    for mode in ("mondrian", "global"):
        acc = {t: {c: {"fpr": [], "recall": [], "bound": [], "budget": [], "fp": [], "n_benign_eval": []} for c in CELLS}
               for t in ConformalTiers.ORDER}
        ref = ConformalTiers(budgets, mondrian=True)  # intended per-cell budgets (same partition rule)
        for perm in perms:
            cal, ev = perm[: n // 2], perm[n // 2:]
            ct = ConformalTiers(budgets, mondrian=(mode == "mondrian")).fit(scores[cal], y[cal], cells[cal])
            ref.fit(scores[cal], y[cal], cells[cal])
            T = ct.thresholds(cells[ev])
            ye, ce, se = y[ev], cells[ev], scores[ev]
            for j, t in enumerate(ConformalTiers.ORDER):
                hit = se > T[:, j]
                for c in CELLS:
                    mb, ma = (ce == c) & (ye == 0), (ce == c) & (ye == 1)
                    if not mb.any():
                        continue
                    a = acc[t][c]
                    a["fpr"].append(float(hit[mb].mean()))
                    a["fp"].append(int(hit[mb].sum()))
                    a["n_benign_eval"].append(int(mb.sum()))
                    a["recall"].append(float(hit[ma].mean()) if ma.any() else np.nan)
                    # intended budget of this raw cell = budget of its (possibly pooled) Mondrian cell
                    rc = ref.cell_of(np.array([c], dtype=object), t)[0]
                    a["budget"].append(min(ref.cap, budgets[t] * ref.parts[t]["mult"][rc]))
                    if mode == "mondrian":
                        nc = ct.parts[t]["n_cal"][rc]
                        eps = min(ct.cap, budgets[t] * ct.parts[t]["mult"][rc])
                        k = int(np.ceil((nc + 1) * (1 - eps)))
                        a["bound"].append((nc + 1 - k) / (nc + 1) if k <= nc else 0.0)
        res = {}
        for t, per in acc.items():
            res[t] = {}
            for c, a in per.items():
                if not a["fpr"]:
                    continue
                f = np.array(a["fpr"])
                r = {"mean_fpr": float(f.mean()), "sd_fpr": float(f.std()),
                     "p95_fpr": float(np.quantile(f, 0.95)),
                     "mean_fp": float(np.mean(a["fp"])), "mean_n_benign_eval": float(np.mean(a["n_benign_eval"])),
                     "mean_recall": float(np.nanmean(a["recall"])), "budget": float(np.mean(a["budget"])),
                     "ratio_to_budget": float(f.mean() / np.mean(a["budget"]))}
                if a["bound"]:
                    r["mean_exact_bound"] = float(np.mean(a["bound"]))
                    # Monte-Carlo standard error of the mean FPR, for comparing it with the bound
                    r["mc_se_mean_fpr"] = float(f.std(ddof=1) / np.sqrt(len(f)))
                res[t][c] = r
        out[mode] = res
    return out


def run(rd: Path, seed: int, reps: int) -> dict:
    b = pd.read_pickle(rd / "bench" / f"bench_seed{seed}.pkl")
    va = b[b.split == "val"].reset_index(drop=True)
    te = b[b.split == "test"].reset_index(drop=True)
    P = np.load(rd / f"preds_seed{seed}.npz", allow_pickle=True)
    assert (P["y_va"].astype(int) == va["label"].to_numpy()).all() and (P["y_te"].astype(int) == te["label"].to_numpy()).all()
    scores = np.concatenate([P["carta_va"], P["carta"]]).astype(float)
    prior = np.concatenate([P["prior_va"], P["prior"]]).astype(float)
    cells = np.concatenate([sensitivity_cells(context_frame(va)), sensitivity_cells(context_frame(te))]).astype(object)
    y_orig = np.concatenate([va["label"].to_numpy(), te["label"].to_numpy()]).astype(int)
    y_fix = np.concatenate([corrected_labels(va), corrected_labels(te)])
    out = {"seed": seed, "reps": reps, "budgets": dict(DEFAULT_BUDGETS),
           "n_pooled": int(len(y_orig)),
           "n_benign_pooled_per_cell": {c: int(((cells == c) & (y_orig == 0)).sum()) for c in CELLS},
           "n_relabelled_pooled_per_cell": {c: int(((cells == c) & (y_orig == 0) & (y_fix == 1)).sum()) for c in CELLS}}
    for lab, y in (("original", y_orig), ("corrected", y_fix)):
        out[f"carta_{lab}"] = check(scores, y, cells, DEFAULT_BUDGETS, reps, seed)
    out["prior_original"] = check(prior, y_orig, cells, DEFAULT_BUDGETS, reps, seed)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--reps", type=int, default=200)
    ap.add_argument("--seeds", type=int, nargs="*", default=None, help="default: every seed with saved predictions")
    a = ap.parse_args()
    rd = Path(a.results)
    seeds = a.seeds if a.seeds is not None else sorted(int(Path(f).stem.replace("preds_seed", ""))
                                                       for f in glob.glob(str(rd / "preds_seed*.npz")))
    for seed in seeds:
        out = run(rd, seed, a.reps)
        (rd / f"cell_resplit_seed{seed}.json").write_text(json.dumps(out, indent=1))
        for lab in ("original", "corrected"):
            m, g = out[f"carta_{lab}"]["mondrian"], out[f"carta_{lab}"]["global"]
            print(f"seed {seed} [{lab}] sensitive cell: " + "  ".join(
                f"{t}: budget {100*m[t]['high']['budget']:.3f}% M {100*m[t]['high']['mean_fpr']:.3f}% "
                f"(bound {100*m[t]['high']['mean_exact_bound']:.3f}%) G {100*g[t]['high']['mean_fpr']:.3f}%"
                for t in ConformalTiers.ORDER), flush=True)


if __name__ == "__main__":
    main()
