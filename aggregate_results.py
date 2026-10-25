#!/usr/bin/env python3
"""Aggregate the per-seed JSON result files into RESULTS.md (tables only; the figures of the paper come
from paper_figures.py)."""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np

TIERS = ["low", "medium", "high", "critical"]
NICE = {
    "prior_weighted": "Prior pipeline (α=0.6, β=0.4)",
    "prior_adaptive": "Prior pipeline (adaptive fusion)",
    "prior_ml_branch": "Prior ML branch (IF/OC-SVM/AE, max)",
    "prior_nlp_branch": "Prior NLP branch (TF-IDF LR/char SGD/rules, max)",
    "prior_rules_only": "Rules only",
    "carta_component_char": "CARTA component: char n-gram LR",
    "carta_component_struct": "CARTA component: structure-token LR",
    "carta_component_gbdt": "CARTA component: GBDT (numeric+rules+profile+context)",
    "carta_component_iforest": "CARTA component: Isolation Forest",
    "carta": "**CARTA (proposed)**",
}


def g(d, *ks, default=None):
    for k in ks:
        if not isinstance(d, dict) or k not in d:
            return default
        d = d[k]
    return d


def ms(vals, scale=1.0, nd=3):
    v = np.array([x for x in vals if x is not None and not (isinstance(x, float) and np.isnan(x))], dtype=float)
    if len(v) == 0:
        return "n/a"
    if len(v) == 1:
        return f"{v[0] * scale:.{nd}f}"
    return f"{v.mean() * scale:.{nd}f} ± {v.std(ddof=1) * scale:.{nd}f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--title", default="Results")
    args = ap.parse_args()
    rd = Path(args.results)
    runs = [json.load(open(f)) for f in sorted(glob.glob(str(rd / "results_seed*.json")))]
    if not runs:
        raise SystemExit("no results")
    seeds = [r["seed"] for r in runs]
    L = [f"# {args.title}", "",
         f"Seeds: {seeds}. Test split (seed {seeds[0]}): {runs[0]['n_test']} unique requests "
         f"({runs[0]['n_test_attack']} attacks, {runs[0]['n_test_benign']} benign); "
         f"train {runs[0]['n_train']}, validation {runs[0]['n_val']}. "
         f"Strict split: {runs[0].get('strict_split')}. Values are mean ± s.d. over seeds; all thresholds are chosen "
         f"on the validation split and applied unchanged to the test split.", ""]
    cal = [g(r, "carta_calibration", "used") for r in runs]
    L += [f"Base-calibration chosen by inner CV on training OOF scores: {cal}.", ""]

    # ---------------------------------------------------------------- Table 1
    L += ["## Table 1. Detection on the held-out test split", "",
          "| Method | ROC-AUC | PR-AUC | Recall @ FPR 0.1% | Recall @ FPR 1% | Test recall / FPR at val-0.1% thr. | Test recall / FPR at val-1% thr. | ECE |",
          "|---|---|---|---|---|---|---|---|"]
    for m in NICE:
        rows = [g(r, "methods", m) for r in runs if g(r, "methods", m)]
        if not rows:
            continue
        L.append("| {} | {} | {} | {} | {} | {} / {} | {} / {} | {} |".format(
            NICE[m], ms([g(x, "test", "roc_auc") for x in rows], nd=4), ms([g(x, "test", "pr_auc") for x in rows], nd=4),
            ms([g(x, "test", "recall@fpr0.001") for x in rows], 100, 2), ms([g(x, "test", "recall@fpr0.01") for x in rows], 100, 2),
            ms([g(x, "test@val_fpr0.001", "recall") for x in rows], 100, 2), ms([g(x, "test@val_fpr0.001", "fpr") for x in rows], 100, 3),
            ms([g(x, "test@val_fpr0.01", "recall") for x in rows], 100, 2), ms([g(x, "test@val_fpr0.01", "fpr") for x in rows], 100, 3),
            ms([g(x, "test", "ece") for x in rows], nd=3)))
    L += ["", "Recall and FPR in %. ECE is computed for every score as if it were a probability (the prior "
          "pipeline presents its fused score in [0,1] and thresholds it as such).", ""]
    bs = [r["bootstrap_carta_vs_prior"] for r in runs]
    L += ["Paired bootstrap (1,000 resamples of the test split, per seed), CARTA − prior pipeline:", ""]
    for r, b in zip(runs, bs):
        L.append(f"- seed {r['seed']}: ΔROC-AUC = {b['d_roc_auc_mean']:+.4f} (95% CI {b['d_roc_auc_ci95'][0]:+.4f}, "
                 f"{b['d_roc_auc_ci95'][1]:+.4f}); ΔRecall@FPR1% = {100*b['d_recall@fpr0.01_mean']:+.2f} pp "
                 f"(95% CI {100*b['d_recall@fpr0.01_ci95'][0]:+.2f}, {100*b['d_recall@fpr0.01_ci95'][1]:+.2f})")
    L.append("")

    # ---------------------------------------------------------------- Table 2
    L += ["## Table 2. Prior pipeline at its fixed operating points, 0.6/0.4 fusion with score-only levels (test split)", "",
          "| Operating point | Recall (%) | FPR (%) | Precision (%) |", "|---|---|---|---|"]
    for k in ("suspicious_low_plus(final>=0.3)", "alert_medium_plus(final>=0.5)", "is_anomaly(final>=0.55)"):
        rows = [g(r, "prior_native", k) for r in runs]
        L.append(f"| {k} | {ms([x['recall'] for x in rows], 100, 2)} | {ms([x['fpr'] for x in rows], 100, 2)} | "
                 f"{ms([x['precision'] for x in rows], 100, 2)} |")
    L += ["", "Prior pipeline levels (cumulative, level or higher):", "",
          "| Level | Score-only FPR (%) | Score-only recall (%) | Score×impact FPR (%) | Score×impact recall (%) |", "|---|---|---|---|---|"]
    for t in TIERS:
        L.append("| {} | {} | {} | {} | {} |".format(
            t, ms([g(r, "prior_native", "levels_score_only", t, "fpr") for r in runs], 100, 2),
            ms([g(r, "prior_native", "levels_score_only", t, "recall") for r in runs], 100, 2),
            ms([g(r, "prior_native", "levels_score_x_impact", t, "fpr") for r in runs], 100, 2),
            ms([g(r, "prior_native", "levels_score_x_impact", t, "recall") for r in runs], 100, 2)))
    L.append("")

    # ---------------------------------------------------------------- Table 3 per source
    L += ["## Table 3. Per-source results (test split)", "",
          "| Source | n attack / benign (seed 0) | CARTA ROC-AUC | Prior ROC-AUC | CARTA R@FPR1% | Prior R@FPR1% | CARTA R@FPR0.1% | Prior R@FPR0.1% |",
          "|---|---|---|---|---|---|---|---|"]
    for s in runs[0]["per_source"]:
        c0 = runs[0]["per_source"][s]["carta"]
        L.append("| {} | {} / {} | {} | {} | {} | {} | {} | {} |".format(
            s, c0["n_attack"], c0["n_benign"],
            ms([g(r, "per_source", s, "carta", "roc_auc") for r in runs], nd=4), ms([g(r, "per_source", s, "prior_weighted", "roc_auc") for r in runs], nd=4),
            ms([g(r, "per_source", s, "carta", "recall@fpr0.01") for r in runs], 100, 2), ms([g(r, "per_source", s, "prior_weighted", "recall@fpr0.01") for r in runs], 100, 2),
            ms([g(r, "per_source", s, "carta", "recall@fpr0.001") for r in runs], 100, 2), ms([g(r, "per_source", s, "prior_weighted", "recall@fpr0.001") for r in runs], 100, 2)))
    L += ["", "Per-source FPR thresholds are computed within the source (threshold-free metric).", ""]
    L += ["Per-family recall at the validation-chosen 1%-FPR threshold:", "",
          "| Family | n (seed 0) | CARTA (%) | Prior (%) |", "|---|---|---|---|"]
    for f in sorted(runs[0]["per_family_recall@val_fpr0.01"]["carta"]):
        L.append("| {} | {} | {} | {} |".format(
            f, runs[0]["per_family_recall@val_fpr0.01"]["carta"][f]["n"],
            ms([g(r, "per_family_recall@val_fpr0.01", "carta", f, "recall") for r in runs], 100, 2),
            ms([g(r, "per_family_recall@val_fpr0.01", "prior_weighted", f, "recall") for r in runs], 100, 2)))
    L.append("")

    # ---------------------------------------------------------------- Table 4 tiers (post hoc, posthoc_tiers.py)
    tiers = [json.load(open(f)) for f in sorted(glob.glob(str(rd / "tiers_seed*.json")))]
    if tiers:
        L += ["## Table 4. Conformal risk tiers (calibrated on validation, evaluated on test; cumulative, tier or higher)", "",
              "| Tier | Nominal budget (%) | CARTA-Mondrian FPR (%) | CARTA-Mondrian recall (%) | CARTA-Mondrian IW-recall (%) | "
              "CARTA-global FPR (%) | CARTA-global recall (%) | CARTA-global IW-recall (%) | Prior+conformal (Mondrian) FPR / recall (%) | "
              "Prior own levels (0.6/0.4, score-only) FPR / recall (%) |",
              "|---|---|---|---|---|---|---|---|---|---|"]
        for t in TIERS:
            L.append("| {} | {:.2f} | {} | {} | {} | {} | {} | {} | {} / {} | {} / {} |".format(
                t, 100 * tiers[0]["budgets"][t],
                ms([g(x, "carta_mondrian", "test", t, "fpr") for x in tiers], 100, 3),
                ms([g(x, "carta_mondrian", "test", t, "recall") for x in tiers], 100, 2),
                ms([g(x, "carta_mondrian", "test", t, "iw_recall") for x in tiers], 100, 2),
                ms([g(x, "carta_global", "test", t, "fpr") for x in tiers], 100, 3),
                ms([g(x, "carta_global", "test", t, "recall") for x in tiers], 100, 2),
                ms([g(x, "carta_global", "test", t, "iw_recall") for x in tiers], 100, 2),
                ms([g(x, "prior_mondrian", "test", t, "fpr") for x in tiers], 100, 3),
                ms([g(x, "prior_mondrian", "test", t, "recall") for x in tiers], 100, 2),
                ms([g(x, "prior_native_levels", "test", t, "fpr") for x in tiers], 100, 2),
                ms([g(x, "prior_native_levels", "test", t, "recall") for x in tiers], 100, 2)))
        L += ["", "IW-recall = impact-weighted recall, with the impact of each attacked endpoint computed from the "
              "prior pipeline's own released impact weights (external to CARTA).", "",
              "Per-cell view (CARTA, Mondrian): benign FPR against the cell's budget, and recall per cell for "
              "Mondrian vs global thresholds:", "",
              "| Tier | Cell | Pooled into | Cell budget (%) | FPR (%) | Recall, Mondrian (%) | Recall, global (%) |", "|---|---|---|---|---|---|---|"]
        for t in TIERS:
            cm = tiers[0]["carta_mondrian"]["config"]["per_tier"][t]["cell_map"]
            for c in tiers[0]["carta_mondrian"]["test"][t]["cells"]:
                L.append("| {} | {} | {} | {} | {} | {} | {} |".format(
                    t, c, cm.get(c, "-"),
                    ms([g(x, "carta_mondrian", "test", t, "cells", c, "budget") for x in tiers], 100, 3),
                    ms([g(x, "carta_mondrian", "test", t, "cells", c, "fpr") for x in tiers], 100, 3),
                    ms([g(x, "carta_mondrian", "test", t, "cells", c, "recall") for x in tiers], 100, 2),
                    ms([g(x, "carta_global", "test", t, "cells", c, "recall") for x in tiers], 100, 2)))
        L += ["", f"Benign validation points per cell (seed {tiers[0]['seed']}): {tiers[0]['n_val_benign_per_cell']}; "
              f"test: {tiers[0]['n_test_benign_per_cell']}.", ""]
        rs = [x["resplit_check_carta"] for x in tiers if "resplit_check_carta" in x]
        if rs:
            L += [f"Validity check: held-out data (validation + test) re-split at random into calibration and evaluation "
                  f"halves {rs[0]['reps']} times per seed, model fixed. Mean benign FPR on the evaluation halves vs the exact "
                  f"split-conformal bound (n+1-k)/(n+1), averaged over seeds:", "",
                  "| Tier | Budget (%) | Mondrian mean FPR (%) | Mondrian exact bound (%) | Global mean FPR (%) | Global exact bound (%) | Mondrian mean recall (%) |",
                  "|---|---|---|---|---|---|---|"]
            for t in TIERS:
                L.append("| {} | {:.2f} | {} | {} | {} | {} | {} |".format(
                    t, 100 * rs[0]["mondrian"][t]["budget"],
                    ms([x["mondrian"][t]["mean_fpr"] for x in rs], 100, 3), ms([x["mondrian"][t]["mean_exact_bound"] for x in rs], 100, 3),
                    ms([x["global"][t]["mean_fpr"] for x in rs], 100, 3), ms([x["global"][t]["mean_exact_bound"] for x in rs], 100, 3),
                    ms([x["mondrian"][t]["mean_recall"] for x in rs], 100, 2)))
            L.append("")

    # ---------------------------------------------------------------- Table 5 cost
    L += ["## Table 5. Expected SOC cost per 10,000 events (threshold chosen on validation; cost on test)", "",
          "| Setting | CARTA | Prior (validation-optimised threshold) | Prior at 0.5 | Prior at 0.55 | CARTA TPR / FPR (%) | Prior-opt TPR / FPR (%) |",
          "|---|---|---|---|---|---|---|"]
    for k in runs[0]["cost"]:
        rows = [r["cost"][k] for r in runs]
        L.append("| {} | {} | {} | {} | {} | {} / {} | {} / {} |".format(
            k, ms([x["carta"]["cost_per_10k"] for x in rows], nd=1), ms([x["prior_val_optimised"]["cost_per_10k"] for x in rows], nd=1),
            ms([x["prior_shipped_0.5"]["cost_per_10k"] for x in rows], nd=1), ms([x["prior_shipped_0.55"]["cost_per_10k"] for x in rows], nd=1),
            ms([x["carta"]["tpr"] for x in rows], 100, 2), ms([x["carta"]["fpr"] for x in rows], 100, 3),
            ms([x["prior_val_optimised"]["tpr"] for x in rows], 100, 2), ms([x["prior_val_optimised"]["fpr"] for x in rows], 100, 3)))
    L += ["", "Setting cfnX_prevY%: c_FP = 1, c_FN = X, attack prevalence Y; capZ% = alerts limited to Z% of events.", ""]

    # ---------------------------------------------------------------- Table 6 evasion
    ev = [r for r in runs if "evasion" in r]
    if ev:
        L += ["## Table 6. Evasion: attack recall at the clean validation-1%-FPR threshold", "",
              "| Transform | CARTA (%) | Prior (%) | Requests whose prior-view text changed (%) |", "|---|---|---|---|"]
        for k in ev[0]["evasion"]:
            if k in ("benign_pool_size",):
                continue
            L.append("| {} | {} | {} | {} |".format(
                k, ms([r["evasion"][k]["carta_recall"] for r in ev], 100, 2), ms([r["evasion"][k]["prior_recall"] for r in ev], 100, 2),
                ms([g(r, "evasion", k, "frac_changed_prior_view") for r in ev], 100, 1)))
        L += ["", "adaptive_worst_case: per request, the attacker applies whichever transform (or none) minimises the "
              "score of the detector under attack (black-box, per-detector).", ""]

    # ---------------------------------------------------------------- Table 7 ablation
    ab = [r for r in runs if "ablation" in r]
    if ab:
        L += ["## Table 7. Ablations", "",
              "| Variant | ROC-AUC | Recall @ FPR 0.1% | Recall @ FPR 1% | ECE | Medium-tier FPR / recall (%) |", "|---|---|---|---|---|---|"]
        for k in ab[0]["ablation"]:
            rows = [r["ablation"][k] for r in ab]
            L.append("| {} | {} | {} | {} | {} | {} / {} |".format(
                k, ms([g(x, "test", "roc_auc") for x in rows], nd=4), ms([g(x, "test", "recall@fpr0.001") for x in rows], 100, 2),
                ms([g(x, "test", "recall@fpr0.01") for x in rows], 100, 2), ms([g(x, "test", "ece") for x in rows], nd=3),
                ms([g(x, "tiers_mondrian", "medium", "fpr") for x in rows], 100, 3), ms([g(x, "tiers_mondrian", "medium", "recall") for x in rows], 100, 2)))
        L.append("")

    # ---------------------------------------------------------------- Table 8 cross
    cr = [r for r in runs if "cross_dataset" in r]
    if cr:
        L += ["## Table 8. Single-source training (train on one source, test on each source's test split)", "",
              "| Train → test | CARTA ROC-AUC | Prior ROC-AUC | CARTA R@FPR1% | Prior R@FPR1% |", "|---|---|---|---|---|"]
        for k in cr[0]["cross_dataset"]:
            L.append("| {} | {} | {} | {} | {} |".format(
                k, ms([g(r, "cross_dataset", k, "carta", "roc_auc") for r in cr], nd=4), ms([g(r, "cross_dataset", k, "prior_weighted", "roc_auc") for r in cr], nd=4),
                ms([g(r, "cross_dataset", k, "carta", "recall@fpr0.01") for r in cr], 100, 2), ms([g(r, "cross_dataset", k, "prior_weighted", "recall@fpr0.01") for r in cr], 100, 2)))
        L.append("")

    # ---------------------------------------------------------------- Table 9 latency
    lt = [r for r in runs if "latency_ms" in r]
    if lt:
        L += ["## Table 9. Single-request latency, end-to-end from raw fields, in-process, one CPU thread (ms)", "",
              "| Method | p50 | p95 | p99 | mean | batch throughput (events/s) |", "|---|---|---|---|---|---|"]
        for m, k, tk in (("CARTA", "carta_ms", "carta"), ("Prior (re-implementation)", "prior_ms", "prior")):
            L.append("| {} | {} | {} | {} | {} | {} |".format(
                m, *[ms([r["latency_ms"][k][q] for r in lt], nd=1) for q in ("p50", "p95", "p99", "mean")],
                ms([r["throughput_eps"][tk] for r in lt], nd=0)))
        L += ["", "CARTA per-stage p50 / p95 (ms):", ""]
        for st in lt[0]["latency_stage_ms_carta"]:
            L.append(f"- {st}: {ms([r['latency_stage_ms_carta'][st]['p50'] for r in lt], nd=2)} / "
                     f"{ms([r['latency_stage_ms_carta'][st]['p95'] for r in lt], nd=2)}")
        L.append("")
    L.append(f"Fit time (s): prior {ms([r['prior_fit_s'] for r in runs], nd=0)}, CARTA {ms([r['carta_fit_s'] for r in runs], nd=0)}.")

    # ---------------------------------------------------------------- Table 10 per-cell validity (cell_resplit.py)
    crs = [json.load(open(f)) for f in sorted(glob.glob(str(rd / "cell_resplit_seed*.json")))]
    if crs:
        L += ["", f"## Table 10. Per-cell validity of the tiers over {crs[0]['reps']} re-splits per seed", "",
              "Validation + test pooled and redrawn into calibration and evaluation halves (model fixed; the same re-splits "
              "for both modes). Per seed, the mean benign FPR inside each sensitivity cell over the re-splits; the table "
              "gives mean ± s.d. of that per-seed mean over seeds. Intended budget = eps_t * m_c of the cell (its pooled "
              "cell where the partition pools it); exact bound = (n_c + 1 - k) / (n_c + 1) of the Mondrian threshold. "
              "'corrected' = benign-labelled log URLs with unambiguous attack signatures counted as attacks in both "
              f"validation and test (pooled relabelled per cell, seed {crs[0]['seed']}: {crs[0]['n_relabelled_pooled_per_cell']}; "
              f"pooled benign per cell: {crs[0]['n_benign_pooled_per_cell']}).", "",
              "| Labels | Tier | Cell | Intended budget (%) | Mondrian FPR (%) | Mondrian exact bound (%) | Global FPR (%) | "
              "Global FPR / budget | Mondrian recall (%) | Global recall (%) |",
              "|---|---|---|---|---|---|---|---|---|---|"]
        for lab in ("original", "corrected"):
            for t in TIERS:
                for c in ("high", "std", "low"):
                    rm = [g(x, f"carta_{lab}", "mondrian", t, c) for x in crs]
                    rg = [g(x, f"carta_{lab}", "global", t, c) for x in crs]
                    if any(r is None for r in rm + rg):
                        continue
                    L.append("| {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |".format(
                        lab, t, c, ms([r["budget"] for r in rm], 100, 3), ms([r["mean_fpr"] for r in rm], 100, 3),
                        ms([r["mean_exact_bound"] for r in rm], 100, 3), ms([r["mean_fpr"] for r in rg], 100, 3),
                        ms([r["mean_fpr"] / r["budget"] for r in rg], nd=2),
                        ms([r["mean_recall"] for r in rm], 100, 2), ms([r["mean_recall"] for r in rg], 100, 2)))
        L += ["", "Prior pipeline's score under the same tiers (original labels), sensitive cell ('high'):", "",
              "| Tier | Intended budget (%) | Mondrian FPR (%) | Mondrian recall (%) | Global FPR (%) | Global recall (%) |",
              "|---|---|---|---|---|---|"]
        for t in TIERS:
            rm = [g(x, "prior_original", "mondrian", t, "high") for x in crs]
            rg = [g(x, "prior_original", "global", t, "high") for x in crs]
            if any(r is None for r in rm + rg):
                continue
            L.append("| {} | {} | {} | {} | {} | {} |".format(
                t, ms([r["budget"] for r in rm], 100, 3), ms([r["mean_fpr"] for r in rm], 100, 3),
                ms([r["mean_recall"] for r in rm], 100, 2), ms([r["mean_fpr"] for r in rg], 100, 3),
                ms([r["mean_recall"] for r in rg], 100, 2)))
        L.append("")
    (rd / "RESULTS.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
