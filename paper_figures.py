#!/usr/bin/env python3
"""Vector figures for the paper (PDF), computed from the saved results only.

  python paper_figures.py --results results --out figures

Writes: roc.pdf, reliability.pdf, tiers.pdf, evasion.pdf
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from sklearn.metrics import roc_curve  # noqa: E402

TIERS = ["low", "medium", "high", "critical"]
C_CARTA, C_PRIOR, C_NLP, C_ML, C_BUDGET, C_GLOBAL = "#1f5a99", "#c0392b", "#d68910", "#7f8c8d", "#2c3e50", "#8e44ad"


def _style():
    serif = [f.name for f in font_manager.fontManager.ttflist if f.name == "Liberation Serif"]
    plt.rcParams.update({"font.family": "Liberation Serif" if serif else "DejaVu Serif", "font.size": 9, "axes.titlesize": 9,
                         "axes.labelsize": 9, "legend.fontsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8,
                         "pdf.fonttype": 42, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.6})


def roc(rd: Path, out: Path, seeds):
    P = [np.load(rd / f"preds_seed{s}.npz", allow_pickle=True) for s in seeds]
    pool = lambda k: np.concatenate([p[k] for p in P]).astype(float)  # noqa: E731
    y = np.concatenate([p["y_te"] for p in P]).astype(int)
    yfix = np.concatenate([np.load(rd / f"y_te_corrected_seed{s}.npy") for s in seeds]).astype(int)
    curves = [("CARTA", "carta", C_CARTA, "-"), ("Prior pipeline (0.6/0.4)", "prior", C_PRIOR, "--"),
              ("Prior NLP branch", "prior_nlp", C_NLP, "-."), ("Prior ML branch", "prior_ml", C_ML, ":")]
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.1), sharey=True)
    for ax, (title, yy) in zip(axes, (("(a) Original labels", y), ("(b) Mislabelled log URLs counted as attacks", yfix))):
        for name, key, c, ls in curves:
            f, t, _ = roc_curve(yy, pool(key))
            ax.plot(np.maximum(f, 1e-5), t, ls, color=c, lw=1.4, label=name)
        one = 1 / ((yy == 0).sum() / len(seeds))  # one benign request per seed
        ax.axvline(one, color="grey", lw=0.7, ls=":")
        ax.set_xscale("log"); ax.set_xlim(1e-4, 1); ax.set_ylim(0, 1.01)
        ax.set_xlabel("False-positive rate (log scale)"); ax.set_title(title, loc="left")
    axes[0].set_ylabel("Recall")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=(0, 0.07, 1, 1)); fig.savefig(out / "roc.pdf"); plt.close(fig)


def _wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def reliability(rd: Path, out: Path, seeds, bins: int = 15):
    """Equal-width bins (as in the ECE); error bars are 95% Wilson intervals of the observed frequency."""
    P = [np.load(rd / f"preds_seed{s}.npz", allow_pickle=True) for s in seeds]
    y = np.concatenate([p["y_te"] for p in P]).astype(int)
    edges = np.linspace(0, 1, bins + 1)
    fig, ax = plt.subplots(figsize=(3.6, 3.2))
    for name, key, c, mk, dx in (("CARTA", "carta", C_CARTA, "o", -0.006), ("Prior pipeline (0.6/0.4)", "prior", C_PRIOR, "s", 0.0),
                                 ("Prior NLP branch", "prior_nlp", C_NLP, "^", 0.006)):
        s = np.clip(np.concatenate([p[key] for p in P]).astype(float), 0, 1)
        idx = np.clip(np.digitize(s, edges[1:-1]), 0, bins - 1)
        mp, fr, lo, hi = [], [], [], []
        for b in range(bins):
            m = idx == b
            n = int(m.sum())
            if n == 0:
                continue
            k = int(y[m].sum()); a, z = _wilson(k, n)
            # clip at 0: for k = 0 or k = n a Wilson bound can miss k/n by a rounding error
            mp.append(s[m].mean() + dx); fr.append(k / n); lo.append(max(0.0, k / n - a)); hi.append(max(0.0, z - k / n))
        ax.errorbar(mp, fr, yerr=[lo, hi], fmt="-" + mk, ms=3, lw=1.0, elinewidth=0.7, capsize=1.5, color=c, label=name)
    ax.plot([0, 1], [0, 1], color="grey", lw=0.8)
    ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("Predicted score (bin mean)"); ax.set_ylabel("Observed attack frequency")
    ax.legend(frameon=False, loc="upper left"); fig.tight_layout(); fig.savefig(out / "reliability.pdf"); plt.close(fig)


def tiers(rd: Path, out: Path):
    T = [json.load(open(f)) for f in sorted(glob.glob(str(rd / "tiers_seed*.json")))]
    C = [json.load(open(f)) for f in sorted(glob.glob(str(rd / "cell_resplit_seed*.json")))]
    rs = [x["resplit_check_carta"] for x in T]
    budgets = [T[0]["budgets"][t] for t in TIERS]
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.2))
    x = np.arange(4); w = 0.2
    series = [("Budget", budgets, C_BUDGET),
              ("CARTA, Mondrian", [np.mean([r["mondrian"][t]["mean_fpr"] for r in rs]) for t in TIERS], C_CARTA),
              ("CARTA, global", [np.mean([r["global"][t]["mean_fpr"] for r in rs]) for t in TIERS], C_GLOBAL),
              ("Prior's own levels", [np.mean([x_["prior_native_levels"]["test"][t]["fpr"] for x_ in T]) for t in TIERS], C_PRIOR)]
    for i, (lab, v, c) in enumerate(series):
        axes[0].bar(x + (i - 1.5) * w, np.array(v) * 100, w, color=c, label=lab)
    axes[0].set_yscale("log"); axes[0].set_xticks(x); axes[0].set_xticklabels(TIERS)
    axes[0].set_ylabel("Benign FPR (%)"); axes[0].set_title("(a) All endpoints", loc="left")
    axes[0].legend(frameon=False, fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2)
    sens = [("Cell budget", [np.mean([c["carta_original"]["mondrian"][t]["high"]["budget"] for c in C]) for t in TIERS], C_BUDGET),
            ("Global threshold", [np.mean([c["carta_original"]["global"][t]["high"]["mean_fpr"] for c in C]) for t in TIERS], C_GLOBAL),
            ("Mondrian cell", [np.mean([c["carta_original"]["mondrian"][t]["high"]["mean_fpr"] for c in C]) for t in TIERS], C_CARTA)]
    w = 0.26
    for i, (lab, v, c) in enumerate(sens):
        axes[1].bar(x + (i - 1) * w, np.array(v) * 100, w, color=c, label=lab)
    axes[1].set_yscale("log"); axes[1].set_xticks(x); axes[1].set_xticklabels(TIERS)
    axes[1].set_title("(b) Sensitive endpoints (CARTA)", loc="left")
    axes[1].legend(frameon=False, fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3)
    fig.tight_layout(); fig.savefig(out / "tiers.pdf"); plt.close(fig)


def evasion(rd: Path, out: Path, seeds):
    R = [json.load(open(rd / f"results_seed{s}.json")) for s in seeds]
    ks = [k for k in R[0]["evasion"] if isinstance(R[0]["evasion"][k], dict)]
    nice = {"clean": "None", "double_url_encode": "Double URL\nencoding", "mixed_case": "Mixed case",
            "inline_comments": "Inline\ncomments", "fullwidth_unicode": "Full-width\nUnicode",
            "whitespace_variants": "Whitespace", "html_entities": "HTML\nentities", "combined": "Combined",
            "benign_padding": "Benign\npadding", "adaptive_worst_case": "Adaptive\n(worst case)"}
    fig, ax = plt.subplots(figsize=(7.0, 2.6))
    x = np.arange(len(ks))
    for name, key, c, mk, off in (("CARTA", "carta_recall", C_CARTA, "o", -0.12), ("Prior pipeline", "prior_recall", C_PRIOR, "s", 0.12)):
        m = [100 * np.mean([r["evasion"][k][key] for r in R]) for k in ks]
        sd = [100 * np.std([r["evasion"][k][key] for r in R], ddof=1) for k in ks]
        ax.errorbar(x + off, m, yerr=sd, fmt=mk, ms=4, capsize=2, color=c, label=name, lw=1)
    ax.set_xticks(x); ax.set_xticklabels([nice.get(k, k) for k in ks], fontsize=7)
    ax.set_ylabel("Recall (%) at the clean\nvalidation 1%-FPR threshold"); ax.set_ylim(85, 100.8)
    ax.legend(frameon=False, loc="lower left"); fig.tight_layout(); fig.savefig(out / "evasion.pdf"); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", default="figures")
    a = ap.parse_args()
    rd, out = Path(a.results), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    seeds = sorted(int(Path(f).stem.replace("preds_seed", "")) for f in glob.glob(str(rd / "preds_seed*.npz")))
    _style()
    roc(rd, out, seeds); reliability(rd, out, seeds); tiers(rd, out); evasion(rd, out, seeds)
    print("written:", sorted(p.name for p in out.glob("*.pdf")))


if __name__ == "__main__":
    main()
