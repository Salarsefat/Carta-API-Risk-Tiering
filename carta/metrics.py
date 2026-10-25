"""Evaluation metrics."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve


def recall_at_fpr(y, s, fpr_target):
    fpr, tpr, thr = roc_curve(y, s)
    ok = fpr <= fpr_target
    if not ok.any():
        return 0.0, float("inf")
    i = int(np.max(np.where(ok)[0]))
    return float(tpr[i]), float(thr[i])


def threshold_at_fpr(y, s, fpr_target):
    """Threshold on the *given* set achieving FPR <= target (used on validation, applied to test)."""
    ben = np.sort(s[y == 0])
    n = len(ben)
    k = int(np.ceil((n + 1) * (1 - fpr_target)))
    if k > n:
        return float(ben[-1]) + 1e-12
    return float(ben[k - 1]) + 1e-12


def confusion_at(y, s, tau):
    pred = s >= tau
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    prec = tp / max(1, tp + fp)
    rec = tp / max(1, tp + fn)
    f1 = 2 * prec * rec / max(1e-12, prec + rec)
    fpr = fp / max(1, fp + tn)
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "precision": prec, "recall": rec, "f1": f1, "fpr": fpr}


def ece(y, p, n_bins=15):
    bins = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(p, bins) - 1, 0, n_bins - 1)
    e = 0.0
    for b in range(n_bins):
        m = idx == b
        if m.any():
            e += m.mean() * abs(p[m].mean() - y[m].mean())
    return float(e)


def brier(y, p):
    return float(np.mean((p - y) ** 2))


def summary(y, s, tau=None, p_is_prob=True):
    y = np.asarray(y).astype(int)
    s = np.asarray(s, dtype=float)
    out = {"n": int(len(y)), "n_attack": int(y.sum()), "n_benign": int((1 - y).sum())}
    if y.min() == y.max():
        return out
    out["roc_auc"] = float(roc_auc_score(y, s))
    out["pr_auc"] = float(average_precision_score(y, s))
    for f in (0.001, 0.01, 0.05):
        r, _ = recall_at_fpr(y, s, f)
        out[f"recall@fpr{f}"] = r
    if p_is_prob:
        out["ece"] = ece(y, s)
        out["brier"] = brier(y, s)
    if tau is not None:
        out.update({f"@tau_{k}": v for k, v in confusion_at(y, s, tau).items()})
        out["tau"] = float(tau)
    return out


def per_group_recall(y, s, tau, groups):
    groups = np.asarray(groups)
    out = {}
    for g in np.unique(groups):
        m = (groups == g) & (y == 1)
        if m.sum() > 0:
            out[str(g)] = {"n": int(m.sum()), "recall": float((s[m] >= tau).mean())}
    return out
