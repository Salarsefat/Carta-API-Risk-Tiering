#!/usr/bin/env python3
"""Where does the single validation->test split fall in the re-split distribution?

For each seed, tiers are calibrated on validation and evaluated on test (the reported single-split
FPR). Validation + test are then pooled and redrawn into calibration / evaluation halves `reps`
times (model fixed). Reported per tier and mode: the single-split test FPR, the re-split mean, and
the one-sided p-value P(re-split FPR >= single-split FPR). Original and corrected labels.

  python split_pvalues.py --results results --reps 2000
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
from cell_resplit import corrected_labels  # noqa: E402


def fprs(ct, s_ev, y_ev, c_ev):
    T = ct.thresholds(c_ev)
    return {t: float((s_ev > T[:, j])[y_ev == 0].mean()) for j, t in enumerate(ConformalTiers.ORDER)}


def run(rd: Path, seed: int, reps: int) -> dict:
    b = pd.read_pickle(rd / "bench" / f"bench_seed{seed}.pkl")
    va = b[b.split == "val"].reset_index(drop=True)
    te = b[b.split == "test"].reset_index(drop=True)
    P = np.load(rd / f"preds_seed{seed}.npz", allow_pickle=True)
    s_va, s_te = P["carta_va"].astype(float), P["carta"].astype(float)
    c_va = sensitivity_cells(context_frame(va)).astype(object)
    c_te = sensitivity_cells(context_frame(te)).astype(object)
    labels = {"original": (va["label"].to_numpy().astype(int), te["label"].to_numpy().astype(int)),
              "corrected": (corrected_labels(va), corrected_labels(te))}
    S, C = np.concatenate([s_va, s_te]), np.concatenate([c_va, c_te])
    n = len(S)
    rng = np.random.RandomState(1000 + seed)
    perms = [rng.permutation(n) for _ in range(reps)]
    out = {"seed": seed, "reps": reps}
    for lab, (y_va, y_te) in labels.items():
        Y = np.concatenate([y_va, y_te])
        out[lab] = {}
        for mode in ("mondrian", "global"):
            single = fprs(ConformalTiers(DEFAULT_BUDGETS, mondrian=(mode == "mondrian")).fit(s_va, y_va, c_va), s_te, y_te, c_te)
            dist = {t: [] for t in ConformalTiers.ORDER}
            for perm in perms:
                cal, ev = perm[: n // 2], perm[n // 2:]
                f = fprs(ConformalTiers(DEFAULT_BUDGETS, mondrian=(mode == "mondrian")).fit(S[cal], Y[cal], C[cal]), S[ev], Y[ev], C[ev])
                for t in dist:
                    dist[t].append(f[t])
            out[lab][mode] = {t: {"single_split_fpr": single[t], "resplit_mean": float(np.mean(v)),
                                  "resplit_sd": float(np.std(v)), "p_one_sided": float(np.mean(np.array(v) >= single[t] - 1e-15))}
                              for t, v in dist.items()}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--reps", type=int, default=1000)
    ap.add_argument("--seeds", type=int, nargs="*", default=None)
    a = ap.parse_args()
    rd = Path(a.results)
    seeds = a.seeds if a.seeds is not None else sorted(int(Path(f).stem.replace("preds_seed", ""))
                                                       for f in glob.glob(str(rd / "preds_seed*.npz")))
    for seed in seeds:
        out = run(rd, seed, a.reps)
        (rd / f"split_pvalues_seed{seed}.json").write_text(json.dumps(out, indent=1))
        for lab in ("original", "corrected"):
            print(f"seed {seed} [{lab}] " + "  ".join(
                f"{mode[0].upper()}-{t}: {100*out[lab][mode][t]['single_split_fpr']:.3f}% vs {100*out[lab][mode][t]['resplit_mean']:.3f}% p={out[lab][mode][t]['p_one_sided']:.3f}"
                for mode in ("mondrian", "global") for t in ("medium", "high", "critical")), flush=True)


if __name__ == "__main__":
    main()
