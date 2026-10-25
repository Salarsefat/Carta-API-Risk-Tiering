#!/usr/bin/env python3
"""Full experimental protocol: de-duplicated benchmark -> prior pipeline + CARTA -> metrics.

Stages (per seed, one process): main [+ evasion] [+ ablation] [+ cross] [+ latency].
The latency numbers of the paper come from latency_bench.py (saved models, nothing else running).

  python run_experiments.py --csic-csv csic_database.csv --httpparams payload_full.csv \
      --webpayloads WEB_APPLICATION_PAYLOADS.jsonl --weblogs zanbil_sample.csv \
      --out results --seeds 0 --stages main,evasion,ablation,cross --save-models
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from carta.baseline import FINAL_THRESHOLD, LEVEL_THRESHOLDS, PriorPipeline, impact_scores  # noqa: E402
from carta.benchmark import build  # noqa: E402
from carta.data import (load_access_log, load_csic, load_csic_kaggle_csv, load_httpparams,  # noqa: E402
                        load_webpayloads, load_weblogs_sample)
from carta.evasion import TRANSFORMS, apply_transform, benign_pool  # noqa: E402
from carta.features import context_frame  # noqa: E402
from carta.metrics import confusion_at, per_group_recall, recall_at_fpr, summary, threshold_at_fpr  # noqa: E402
from carta.normalize import add_text_views  # noqa: E402
from carta.proposed import CARTA, cost_optimal_threshold, expected_cost, static_block  # noqa: E402

RAW_COLS = ["source", "method", "path", "query", "body", "headers", "label", "attack_type"]
TIER_ORDER = ["normal", "low", "medium", "high", "critical"]
COST_SETTINGS = {  # name: (c_fp, c_fn, prevalence, max_alert_rate)
    "cfn10_prev1%": (1, 10, 0.01, None),
    "cfn100_prev1%": (1, 100, 0.01, None),
    "cfn100_prev1%_cap1.5%": (1, 100, 0.01, 0.015),
    "cfn20_prev5%": (1, 20, 0.05, None),
}


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_sources(args):
    src, files = {}, {}
    if args.csic_csv:
        src["csic2010"] = load_csic_kaggle_csv(args.csic_csv)
        files["csic_csv"] = args.csic_csv
    elif args.csic_dir:
        src["csic2010"] = load_csic(args.csic_dir)
        for f in ("normalTrafficTraining.txt", "normalTrafficTest.txt", "anomalousTrafficTest.txt"):
            files[f] = str(Path(args.csic_dir) / f)
    if args.httpparams:
        src["httpparams"] = load_httpparams(args.httpparams)
        files["httpparams"] = args.httpparams
    if args.webpayloads:
        src["webpayloads"] = load_webpayloads(args.webpayloads)
        files["webpayloads"] = args.webpayloads
    if args.accesslog:
        src["accesslog"] = load_access_log(args.accesslog, max_rows=args.accesslog_rows)
        files["accesslog"] = args.accesslog
    if args.weblogs:
        src["weblogs2019"] = load_weblogs_sample(args.weblogs)
        files["weblogs_sample"] = args.weblogs
    if args.max_rows:
        src = {k: v.sample(n=min(args.max_rows, len(v)), random_state=0) for k, v in src.items()}
    return src, files


def evaluate_scores(y_va, s_va, y_te, s_te):
    res = {"test": summary(y_te, s_te, p_is_prob=True)}
    for f in (0.001, 0.01):
        tau = threshold_at_fpr(y_va, s_va, f)
        res[f"val_tau@fpr{f}"] = tau
        res[f"test@val_fpr{f}"] = confusion_at(y_te, s_te, tau)
    return res


def tier_report(y, tiers, cells, impact, budgets_fn=None):
    """Cumulative tier table: FPR, recall, impact-weighted recall, per-cell FPR/recall/budget."""
    y = np.asarray(y).astype(int)
    rank = {t: i for i, t in enumerate(TIER_ORDER)}
    r = np.array([rank[t] for t in tiers])
    cells = np.asarray(cells)
    out = {}
    for t in TIER_ORDER[1:]:
        hit = r >= rank[t]
        row = {"fpr": float(hit[y == 0].mean()), "recall": float(hit[y == 1].mean()),
               "fp": int((hit & (y == 0)).sum()), "tp": int((hit & (y == 1)).sum()),
               "iw_recall": float((impact[y == 1] * hit[y == 1]).sum() / max(1e-12, impact[y == 1].sum()))}
        if budgets_fn is not None:
            row["effective_budget"] = float(budgets_fn(t)[y == 0].mean())
        row["cells"] = {}
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


def paired_bootstrap(y, a, b, n_boot=1000, seed=0, fpr=0.01):
    """95% CI of (metric(a) - metric(b)) for ROC-AUC and recall@FPR on the test set."""
    rng = np.random.RandomState(seed)
    y = np.asarray(y).astype(int)
    d_auc, d_rec = [], []
    n = len(y)
    for _ in range(n_boot):
        i = rng.randint(0, n, n)
        if y[i].min() == y[i].max():
            continue
        d_auc.append(roc_auc_score(y[i], a[i]) - roc_auc_score(y[i], b[i]))
        d_rec.append(recall_at_fpr(y[i], a[i], fpr)[0] - recall_at_fpr(y[i], b[i], fpr)[0])
    q = lambda v: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
    return {"d_roc_auc_mean": float(np.mean(d_auc)), "d_roc_auc_ci95": q(d_auc),
            f"d_recall@fpr{fpr}_mean": float(np.mean(d_rec)), f"d_recall@fpr{fpr}_ci95": q(d_rec), "n_boot": len(d_auc)}


def run_seed(seed, sources, files, out_dir: Path, args, stages):
    log(f"=== seed {seed}: building benchmark (strict={args.strict_split}, twin_merge={args.twin_merge})")
    bench = build(sources, seed=seed, out_dir=out_dir / "bench", input_files=files,
                  background_min_dup=None if args.strict_split else args.background_min_dup,
                  twin_merge=args.twin_merge)
    tr, va, te = (bench[bench.split == s].reset_index(drop=True) for s in ("train", "val", "test"))
    log(f"train={len(tr)} val={len(va)} test={len(te)} (attack frac test={te.label.mean():.3f})")
    y_va, y_te = va.label.to_numpy(), te.label.to_numpy()
    R = {"seed": seed, "strict_split": bool(args.strict_split), "twin_merge": bool(args.twin_merge), "n_train": len(tr), "n_val": len(va), "n_test": len(te),
         "n_test_attack": int(y_te.sum()), "n_test_benign": int((1 - y_te).sum())}
    impact_te = impact_scores(context_frame(te))

    # ------------------------------------------------------------------ prior pipeline
    log("fitting prior pipeline (re-implementation)")
    t0 = time.time()
    prior = PriorPipeline(seed=seed).fit(tr)
    R["prior_fit_s"] = time.time() - t0
    pv, pt = prior.predict(va), prior.predict(te)
    # ------------------------------------------------------------------ CARTA
    log("fitting CARTA")
    t0 = time.time()
    carta = CARTA(seed=seed).fit(tr, va)
    R["carta_fit_s"] = time.time() - t0
    R["carta_calibration"] = {"used": carta.calibration_used_, "inner_cv_logloss": getattr(carta, "calibration_cv_", None)}
    log(f"fit times: prior {R['prior_fit_s']:.0f}s, CARTA {R['carta_fit_s']:.0f}s")
    S_te = static_block(te)
    base_te = carta.base_predict(te, S_te)
    p_va, p_te = carta.val_scores_, carta.proba_from_base(base_te, S_te["ctx"])
    cells_te = S_te["cells"]

    scores = {
        "prior_weighted": (pv["final"], pt["final"]),
        "prior_adaptive": (pv["final_adaptive"], pt["final_adaptive"]),
        "prior_ml_branch": (pv["ml_score"], pt["ml_score"]),
        "prior_nlp_branch": (pv["nlp_score"], pt["nlp_score"]),
        "prior_rules_only": (pv["nlp"][:, 2], pt["nlp"][:, 2]),
        "carta": (p_va, p_te),
    }
    for b in carta.base_names:
        scores[f"carta_component_{b}"] = (carta.val_base_[b], base_te[b])
    R["methods"] = {k: evaluate_scores(y_va, sv, y_te, st) for k, (sv, st) in scores.items()}
    R["bootstrap_carta_vs_prior"] = paired_bootstrap(y_te, p_te, pt["final"], seed=seed)

    R["prior_native"] = {
        "suspicious_low_plus(final>=0.3)": confusion_at(y_te, pt["final"], LEVEL_THRESHOLDS["low"]),
        "alert_medium_plus(final>=0.5)": confusion_at(y_te, pt["final"], LEVEL_THRESHOLDS["medium"]),
        "is_anomaly(final>=0.55)": confusion_at(y_te, pt["final"], FINAL_THRESHOLD),
        "levels_score_only": tier_report(y_te, pt["level"], cells_te, impact_te),
        "levels_score_x_impact": tier_report(y_te, pt["level_matrix"], cells_te, impact_te),
    }
    tm = carta.tiers_mondrian.transform(p_te, cells_te)
    tg = carta.tiers_global.transform(p_te, cells_te)
    R["carta_tiers"] = {
        "budgets": carta.budgets,
        "mondrian": {"config": carta.tiers_mondrian.describe(),
                     "test": tier_report(y_te, tm, cells_te, impact_te,
                                         lambda t: carta.tiers_mondrian.cell_budget(cells_te, t))},
        "global": {"config": carta.tiers_global.describe(),
                   "test": tier_report(y_te, tg, cells_te, impact_te,
                                       lambda t: carta.tiers_global.cell_budget(cells_te, t))},
    }
    R["cost"] = {}
    for name, (cfp, cfn, prev, cap) in COST_SETTINGS.items():
        tc = cost_optimal_threshold(p_va, y_va, cfp, cfn, prev, cap)
        tb = cost_optimal_threshold(pv["final"], y_va, cfp, cfn, prev, cap)
        R["cost"][name] = {
            "carta": expected_cost(p_te, y_te, tc, cfp, cfn, prev),
            "prior_val_optimised": expected_cost(pt["final"], y_te, tb, cfp, cfn, prev),
            "prior_shipped_0.5": expected_cost(pt["final"], y_te, 0.5, cfp, cfn, prev),
            "prior_shipped_0.55": expected_cost(pt["final"], y_te, FINAL_THRESHOLD, cfp, cfn, prev),
        }
    tau_c, tau_p = R["methods"]["carta"]["val_tau@fpr0.01"], R["methods"]["prior_weighted"]["val_tau@fpr0.01"]
    src = te.source.to_numpy()
    R["per_source"] = {s: {"carta": summary(y_te[src == s], p_te[src == s]),
                           "prior_weighted": summary(y_te[src == s], pt["final"][src == s])}
                       for s in sorted(set(src))}
    R["per_family_recall@val_fpr0.01"] = {"carta": per_group_recall(y_te, p_te, tau_c, te.attack_type),
                                          "prior_weighted": per_group_recall(y_te, pt["final"], tau_p, te.attack_type)}
    R["per_cell_recall@val_fpr0.01"] = {"carta": per_group_recall(y_te, p_te, tau_c, cells_te),
                                        "prior_weighted": per_group_recall(y_te, pt["final"], tau_p, cells_te)}
    np.savez_compressed(out_dir / f"preds_seed{seed}.npz", y_te=y_te, carta=p_te, prior=pt["final"],
                        prior_adaptive=pt["final_adaptive"], prior_ml=pt["ml_score"], prior_nlp=pt["nlp_score"],
                        tier_mondrian=tm.astype(str), tier_global=tg.astype(str), prior_level=pt["level"].astype(str),
                        cells=cells_te.astype(str), impact=impact_te, source=src.astype(str),
                        family=te.attack_type.to_numpy().astype(str), y_va=y_va, carta_va=p_va, prior_va=pv["final"],
                        **{f"base_{b}": base_te[b] for b in carta.base_names})
    if args.save_models:
        import joblib
        joblib.dump({"carta": carta, "prior": prior, "tau_c": tau_c, "tau_p": tau_p}, out_dir / f"models_seed{seed}.joblib",
                    compress=3)
        log("models saved")
    t = R["methods"]
    log(f"main: carta roc={t['carta']['test']['roc_auc']:.4f} r@0.1%={t['carta']['test']['recall@fpr0.001']:.3f} | "
        f"prior roc={t['prior_weighted']['test']['roc_auc']:.4f} r@0.1%={t['prior_weighted']['test']['recall@fpr0.001']:.3f}")

    # ------------------------------------------------------------------ evasion
    if "evasion" in stages:
        log("evasion suite")
        att = te[te.label == 1].reset_index(drop=True)
        R["evasion"] = {"clean": {"carta_recall": float((p_te[y_te == 1] >= tau_c).mean()),
                                  "prior_recall": float((pt["final"][y_te == 1] >= tau_p).mean()),
                                  "n_attack": int(len(att))}}
        pool = benign_pool(tr, seed=seed)
        sc = {"clean": p_te[y_te == 1]}
        sp = {"clean": pt["final"][y_te == 1]}
        for name in TRANSFORMS:
            tdf = apply_transform(att, name, seed=seed, pool=pool)
            changed = float((tdf["text"] != att["text"]).mean())
            changed_prior = float((tdf["text_prior"] != att["text_prior"]).mean())
            sc[name], sp[name] = carta.predict_proba(tdf), prior.predict(tdf)["final"]
            rc, rb = float((sc[name] >= tau_c).mean()), float((sp[name] >= tau_p).mean())
            R["evasion"][name] = {"carta_recall": rc, "prior_recall": rb,
                                  "frac_changed_carta_view": changed, "frac_changed_prior_view": changed_prior}
            log(f"  {name:20s} carta={rc:.3f} prior={rb:.3f} (view changed: carta {changed:.2f}, prior {changed_prior:.2f})")
        # adaptive attacker: per request, the transform (or none) that minimises *this* detector's score
        wc = np.min(np.stack(list(sc.values())), axis=0)
        wp = np.min(np.stack(list(sp.values())), axis=0)
        R["evasion"]["adaptive_worst_case"] = {"carta_recall": float((wc >= tau_c).mean()),
                                               "prior_recall": float((wp >= tau_p).mean())}
        R["evasion"]["benign_pool_size"] = len(pool)
        log(f"  adaptive worst case  carta={R['evasion']['adaptive_worst_case']['carta_recall']:.3f} "
            f"prior={R['evasion']['adaptive_worst_case']['prior_recall']:.3f}")

    # ------------------------------------------------------------------ ablations
    if "ablation" in stages:
        log("ablations")
        A = {}

        def add(name, model, p=None):
            pr = model.predict_from_cached(base_te, S_te["ctx"], cells_te) if p is None else \
                {"p": p, "tier": model.tiers_mondrian.transform(p, cells_te)}
            res = evaluate_scores(y_va, model.val_scores_, y_te, pr["p"])
            res["tiers_mondrian"] = tier_report(y_te, pr["tier"], cells_te, impact_te,
                                                lambda t: model.tiers_mondrian.cell_budget(cells_te, t))
            A[name] = res
            r = res["test"]
            log(f"  {name:28s} roc={r['roc_auc']:.4f} r@0.1%={r['recall@fpr0.001']:.3f} "
                f"r@1%={r['recall@fpr0.01']:.3f} ece={r['ece']:.3f}")

        full = list(carta.base_names)
        add("full", carta)
        add("base_calibration_none", carta.variant(base_calibration="none"))
        add("base_calibration_isotonic", carta.variant(base_calibration="isotonic"))
        add("no_context_in_meta", carta.variant(use_context_meta=False))
        for b in full:
            add(f"drop_{b}", carta.variant(keep=[x for x in full if x != b]))
        # The text-only stack gets no context features in the meta-learner, the tabular-only stack keeps them
        # (as in the reported run); context_ablation.py evaluates both stacks with and without them.
        add("text_only(char+struct)", carta.variant(keep=[x for x in full if x in ("char", "struct")],
                                                    use_context_meta=False))
        add("tabular_only(gbdt+iforest)", carta.variant(keep=[x for x in full if x in ("gbdt", "iforest")]))
        for name, kw in (("no_endpoint_profile", dict(use_profile=False)),
                         ("no_context_anywhere", dict(use_context=False))):
            m = CARTA(seed=seed, **kw).fit(tr, va, reuse=carta)
            add(name, m, p=m.proba_from_base(m.base_predict(te, S_te), S_te["ctx"]))
        R["ablation"] = A

    # ------------------------------------------------------------------ cross-dataset generalisation
    if "cross" in stages and tr.source.nunique() > 1:
        log("cross-dataset generalisation")
        R["cross_dataset"] = {}
        for s_tr in sorted(tr.source.unique()):
            tr_s, va_s = tr[tr.source == s_tr].reset_index(drop=True), va[va.source == s_tr].reset_index(drop=True)
            if tr_s.label.nunique() < 2 or va_s.label.nunique() < 2:
                continue
            m2 = CARTA(seed=seed).fit(tr_s, va_s)
            b2 = PriorPipeline(seed=seed).fit(tr_s)
            for s_te in sorted(te.source.unique()):
                te_s = te[te.source == s_te].reset_index(drop=True)
                if te_s.label.nunique() < 2:
                    continue
                y2 = te_s.label.to_numpy()
                key = f"{s_tr}->{s_te}"
                R["cross_dataset"][key] = {"carta": summary(y2, m2.predict_proba(te_s)),
                                           "prior_weighted": summary(y2, b2.predict(te_s)["final"])}
                log(f"  {key}: carta roc={R['cross_dataset'][key]['carta']['roc_auc']:.4f} "
                    f"prior roc={R['cross_dataset'][key]['prior_weighted']['roc_auc']:.4f}")

    # ------------------------------------------------------------------ latency (end-to-end from raw fields)
    if "latency" in stages:
        log("latency")
        sample = te.sample(n=min(args.latency_n, len(te)), random_state=seed).reset_index(drop=True)[RAW_COLS]
        for i in range(20):
            row = sample.iloc[[i]]
            carta.predict_proba(row, raw=True)
            prior.predict(add_text_views(row))
        lat = {"carta_ms": [], "prior_ms": []}
        stage_ms = {"text_views": [], "static_features": [], "text_models": [], "tabular_models": [], "meta": []}
        for i in range(len(sample)):
            row = sample.iloc[[i]]
            t0 = time.perf_counter()
            carta.predict_proba(row, raw=True)
            lat["carta_ms"].append((time.perf_counter() - t0) * 1e3)
            t0 = time.perf_counter()
            prior.predict(add_text_views(row))
            lat["prior_ms"].append((time.perf_counter() - t0) * 1e3)
            t0 = time.perf_counter(); r2 = add_text_views(row); t1 = time.perf_counter()
            S = static_block(r2); t2 = time.perf_counter()
            tb = {b: carta.models[b].predict_proba(r2["text"].tolist()) for b in carta.base_names if b in CARTA.TEXT_BASES}
            t3 = time.perf_counter()
            tb.update(carta._predict_tabular(carta.models["tab"], r2, S)); t4 = time.perf_counter()
            carta.proba_from_base(tb, S["ctx"]); t5 = time.perf_counter()
            for k, a, b in (("text_views", t0, t1), ("static_features", t1, t2), ("text_models", t2, t3),
                            ("tabular_models", t3, t4), ("meta", t4, t5)):
                stage_ms[k].append((b - a) * 1e3)
        pct = lambda v: {"p50": float(np.percentile(v, 50)), "p95": float(np.percentile(v, 95)),
                         "p99": float(np.percentile(v, 99)), "mean": float(np.mean(v))}
        R["latency_ms"] = {k: pct(v) for k, v in lat.items()}
        R["latency_stage_ms_carta"] = {k: pct(v) for k, v in stage_ms.items()}
        raw_te = te[RAW_COLS]
        t0 = time.perf_counter(); carta.predict_proba(raw_te, raw=True)
        R["throughput_eps"] = {"carta": len(te) / (time.perf_counter() - t0)}
        t0 = time.perf_counter(); prior.predict(add_text_views(raw_te))
        R["throughput_eps"]["prior"] = len(te) / (time.perf_counter() - t0)
        log(f"  latency p50 carta={R['latency_ms']['carta_ms']['p50']:.1f}ms prior={R['latency_ms']['prior_ms']['p50']:.1f}ms")

    (out_dir / f"results_seed{seed}.json").write_text(json.dumps(R, indent=1, default=float))
    log(f"seed {seed} done")
    return R


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csic-dir")
    ap.add_argument("--csic-csv", help="Kaggle csic_database.csv (alternative to --csic-dir)")
    ap.add_argument("--httpparams")
    ap.add_argument("--webpayloads")
    ap.add_argument("--accesslog", help="unlabelled access log added as benign traffic (not used in the paper)")
    ap.add_argument("--accesslog-rows", type=int, default=200_000)
    ap.add_argument("--weblogs", help="zanbil_sample.csv produced by prepare_zanbil.py")
    ap.add_argument("--out", default="results")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--stages", default="main,evasion,latency")
    ap.add_argument("--max-rows", type=int, default=None, help="subsample each source (development only)")
    ap.add_argument("--strict-split", action="store_true", help="pure random split of unique requests")
    ap.add_argument("--background-min-dup", type=int, default=20)
    ap.add_argument("--twin-merge", action="store_true",
                    help="key requests by path + parameter multiset, ignoring method and query/body location")
    ap.add_argument("--latency-n", type=int, default=500)
    ap.add_argument("--save-models", action="store_true")
    args = ap.parse_args()
    stages = set(args.stages.split(","))
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    sources, files = load_sources(args)
    for k, v in sources.items():
        log(f"source {k}: {len(v)} rows, attack frac {v.label.mean():.3f}")
    for seed in args.seeds:
        run_seed(seed, sources, files, out_dir, args, stages)


if __name__ == "__main__":
    main()
