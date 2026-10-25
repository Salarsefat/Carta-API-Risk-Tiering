"""CARTA: Calibrated, context-Aware Risk Tiering for API traffic (proposed method).

  x_text --(fixpoint normalisation)--> char 1-5-gram TF-IDF -> LR ---------------\
  x_text --(structure tokens)--------> token 1-3-gram TF-IDF -> LR ----------------\
  [x_num, rules, endpoint-profile deviations, context] -> gradient-boosted trees ---+-> logits -> logistic
  x_num (benign rows only) ----------> Isolation Forest ---------------------------/   meta-learner (cross-fitted)
                                                                                         -> p_hat (calibrated)
  p_hat -> Mondrian split-conformal risk tiers with context-dependent error budgets (validation)
  p_hat -> alert threshold minimising expected SOC cost under an alert-capacity constraint (validation)

All thresholds come from data with a stated benign-error budget; business context enters as the
budget multiplier of the conformal cell, not as a hand-set score bump.
"""
from __future__ import annotations

import copy
import math

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, IsolationForest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from .features import (context_frame, context_onehot, fit_method_table, method_mismatch, rule_matrix,
                       sensitivity_cells, static_numeric_matrix, structure_tokens)
from .normalize import add_text_views
from .profile import EndpointProfile

TIERS = ["normal", "low", "medium", "high", "critical"]
DEFAULT_BUDGETS = {"low": 0.05, "medium": 0.01, "high": 0.002, "critical": 0.0005}
RAW_CELL_MULT = {"high": 2.0, "std": 1.0, "low": 0.5}


def _logit(p, eps=1e-6):
    p = np.clip(np.asarray(p, dtype=float), eps, 1 - eps)
    return np.log(p / (1 - p))


# =============================================================================
# Conformal risk tiers
# =============================================================================
def conformal_threshold(benign_scores: np.ndarray, eps: float) -> float:
    """tau such that, for an exchangeable benign score S, P(S > tau) <= eps.
    Returns +inf when eps is not certifiable with n calibration points (n < 1/eps - 1)."""
    s = np.sort(np.asarray(benign_scores, dtype=float))
    n = len(s)
    k = int(math.ceil((n + 1) * (1 - eps)))
    if n == 0 or k > n:
        return float("inf")
    return float(s[k - 1])


class ConformalTiers:
    """Split-conformal risk tiers with context-dependent budgets.

    For every tier t (budget eps_t) a partition of the sensitivity cells is chosen *from the
    benign calibration counts only* (never from scores): start from one cell per sensitivity
    class and pool the smallest cell into 'std' while some cell c cannot certify eps_t * m_c,
    i.e. n_c < 1/(eps_t m_c) - 1. Multipliers m_c = r_c / sum_c' pi_c' r_c' keep the
    benign-weighted aggregate budget equal to eps_t. Within each cell the threshold is the
    split-conformal quantile, so P(benign score > tau_t(c)) <= eps_t m_c under exchangeability.
    Thresholds are then made non-decreasing across tiers (cumulative max), which keeps tiers
    nested and can only lower each tier's false-positive rate.
    mondrian=False gives one global cell per tier (classical split-conformal thresholds)."""

    ORDER = ("low", "medium", "high", "critical")

    def __init__(self, budgets: dict | None = None, mondrian: bool = True, raw_mult: dict | None = None,
                 cap: float = 0.5):
        self.budgets = dict(budgets or DEFAULT_BUDGETS)
        self.mondrian = mondrian
        self.raw_mult = dict(raw_mult or RAW_CELL_MULT)
        self.cap = cap

    def _partition(self, cells, ben, eps):
        if not self.mondrian:
            return {}, "all", {"all": 1.0}
        cmap = {c: c for c in np.unique(cells)}
        while True:
            mapped = np.array([cmap[c] for c in cells], dtype=object)
            uniq = sorted(set(mapped[ben]))
            pool = "std" if "std" in uniq else max(uniq, key=lambda c: (mapped[ben] == c).sum())
            n = {c: int((mapped[ben] == c).sum()) for c in uniq}
            nb = sum(n.values())
            z = sum(n[c] / nb * self.raw_mult.get(c, 1.0) for c in uniq)
            mult = {c: self.raw_mult.get(c, 1.0) / z for c in uniq}
            bad = [c for c in uniq if c != pool and n[c] < 1.0 / (eps * mult[c]) - 1]
            if not bad:
                break
            worst = min(bad, key=lambda c: n[c])
            cmap = {k: (pool if v == worst else v) for k, v in cmap.items()}
        cmap = {k: (v if v in mult else pool) for k, v in cmap.items()}
        return cmap, pool, mult

    def _map(self, t, cells):
        part = self.parts[t]
        if not self.mondrian:
            return np.array(["all"] * len(cells), dtype=object)
        return np.array([part["cell_map"].get(c, part["pool"]) for c in cells], dtype=object)

    def fit(self, p: np.ndarray, y: np.ndarray, cells: np.ndarray):
        p, y = np.asarray(p, float), np.asarray(y, int)
        ben = y == 0
        cells = np.asarray(cells, dtype=object)
        self.parts = {}
        for t in self.ORDER:
            eps = self.budgets[t]
            cmap, pool, mult = self._partition(cells, ben, eps)
            self.parts[t] = {"cell_map": cmap, "pool": pool, "mult": mult}
            mapped = self._map(t, cells)
            self.parts[t]["tau"] = {c: conformal_threshold(p[ben & (mapped == c)], min(self.cap, eps * mult[c]))
                                    for c in mult}
            self.parts[t]["n_cal"] = {c: int((ben & (mapped == c)).sum()) for c in mult}
        return self

    def thresholds(self, cells: np.ndarray) -> np.ndarray:
        cells = np.asarray(cells, dtype=object)
        T = np.stack([np.array([self.parts[t]["tau"][c] for c in self._map(t, cells)]) for t in self.ORDER], axis=1)
        return np.maximum.accumulate(T, axis=1)

    def transform(self, p: np.ndarray, cells: np.ndarray) -> np.ndarray:
        T = self.thresholds(cells)
        p = np.asarray(p, float)
        out = np.full(len(p), "normal", dtype=object)
        for j, t in enumerate(self.ORDER):
            out[p > T[:, j]] = t
        return out

    def cell_of(self, cells: np.ndarray, tier: str) -> np.ndarray:
        return self._map(tier, np.asarray(cells, dtype=object))

    def cell_budget(self, cells: np.ndarray, tier: str) -> np.ndarray:
        mult = self.parts[tier]["mult"]
        return np.array([min(self.cap, self.budgets[tier] * mult[c]) for c in self.cell_of(cells, tier)])

    def describe(self) -> dict:
        return {"mondrian": self.mondrian, "budgets": dict(self.budgets),
                "per_tier": {t: {"cell_map": dict(v["cell_map"]), "mult": dict(v["mult"]),
                                 "n_cal_benign": dict(v["n_cal"]), "tau": dict(v["tau"])} for t, v in self.parts.items()}}


# =============================================================================
# Base models
# =============================================================================
class _TextLR:
    def __init__(self, analyzer, ngram, seed, max_features, use_struct=False, C=4.0):
        kw = dict(analyzer=analyzer, ngram_range=ngram, min_df=2, max_features=max_features,
                  sublinear_tf=True, lowercase=False, dtype=np.float32)
        if analyzer == "word":
            kw["token_pattern"] = r"\S+"
        self.vec = TfidfVectorizer(**kw)
        self.clf = LogisticRegression(max_iter=3000, C=C, class_weight="balanced", random_state=seed)
        self.use_struct = use_struct

    def _prep(self, texts):
        return [structure_tokens(t) for t in texts] if self.use_struct else list(texts)

    def fit(self, texts, y):
        self.clf.fit(self.vec.fit_transform(self._prep(texts)), y)
        return self

    def predict_proba(self, texts):
        return self.clf.predict_proba(self.vec.transform(self._prep(texts)))[:, 1]


def static_block(df: pd.DataFrame) -> dict:
    """Fold-independent features of a request table (computed once per table)."""
    ctx = context_frame(df)
    return {"num": static_numeric_matrix(df), "rules": rule_matrix(df["content"].tolist()),
            "ctx": context_onehot(ctx), "cells": sensitivity_cells(ctx)}


class CARTA:
    TEXT_BASES = ("char", "struct")

    def __init__(self, seed: int = 0, budgets: dict | None = None, n_folds: int = 5,
                 use_text_char=True, use_text_struct=True, use_numeric=True, use_iforest=True,
                 use_context=True, use_profile=True, base_calibration: str = "auto", meta_C: float = 1.0):
        self.seed = seed
        self.budgets = dict(budgets or DEFAULT_BUDGETS)
        self.n_folds = n_folds
        self.use_text_char, self.use_text_struct = use_text_char, use_text_struct
        self.use_numeric, self.use_iforest = use_numeric, use_iforest
        self.use_context, self.use_profile = use_context, use_profile
        self.base_calibration = base_calibration
        self.meta_C = meta_C

    # ------------------------------------------------------------ base model factory
    def _bases(self) -> list[str]:
        b = []
        if self.use_text_char:
            b.append("char")
        if self.use_text_struct:
            b.append("struct")
        if self.use_numeric:
            b.append("gbdt")
        if self.use_iforest:
            b.append("iforest")
        return b

    def _new_text(self, name):
        if name == "char":
            return _TextLR("char", (1, 5), self.seed, 400_000)
        return _TextLR("word", (1, 3), self.seed, 300_000, use_struct=True)

    def _tab(self, df, S, aux) -> np.ndarray:
        parts = [S["num"], method_mismatch(df, aux["method_table"]), S["rules"]]
        if self.use_profile:
            parts.append(aux["profile"].transform(df))
        if self.use_context:
            parts.append(S["ctx"])
        return np.concatenate(parts, axis=1)

    def _fit_tabular(self, df, y, S) -> dict:
        ben = df[y == 0]
        aux = {"method_table": fit_method_table(ben)}
        if self.use_profile:
            aux["profile"] = EndpointProfile().fit(ben)
        m = {"aux": aux}
        if self.use_numeric:
            m["gbdt"] = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.08, max_leaf_nodes=31,
                                                       class_weight="balanced", random_state=self.seed
                                                       ).fit(self._tab(df, S, aux), y)
        if self.use_iforest:
            Xn = np.concatenate([S["num"], method_mismatch(df, aux["method_table"])], axis=1)[y == 0]
            m["if_scaler"] = StandardScaler().fit(Xn)
            m["iforest"] = IsolationForest(n_estimators=200, random_state=self.seed, n_jobs=-1).fit(
                m["if_scaler"].transform(Xn))
            m["iforest"].n_jobs = 1
        return m

    def _predict_tabular(self, m, df, S) -> dict:
        out = {}
        if "gbdt" in m:
            out["gbdt"] = m["gbdt"].predict_proba(self._tab(df, S, m["aux"]))[:, 1]
        if "iforest" in m:
            Xn = np.concatenate([S["num"], method_mismatch(df, m["aux"]["method_table"])], axis=1)
            d = m["iforest"].decision_function(m["if_scaler"].transform(Xn))
            out["iforest"] = 1.0 / (1.0 + np.exp(np.clip(4.0 * d, -30, 30)))
        return out

    # ------------------------------------------------------------ fit
    def fit(self, train_df: pd.DataFrame, val_df: pd.DataFrame, reuse: "CARTA | None" = None):
        """reuse: a fitted CARTA with the same seed/folds whose text bases (OOF + final) are reused."""
        y = train_df["label"].to_numpy()
        self.y_train_ = y
        self.base_names = self._bases()
        S_tr = static_block(train_df)
        self.ctx_train_ = S_tr["ctx"]
        folds = list(StratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=self.seed)
                     .split(np.zeros(len(y)), y))
        oof = {b: np.zeros(len(y)) for b in self.base_names}
        self.models = {}
        texts = train_df["text"].tolist()
        for b in [b for b in self.base_names if b in self.TEXT_BASES]:
            if reuse is not None and b in getattr(reuse, "oof_", {}):
                oof[b] = reuse.oof_[b]
                self.models[b] = reuse.models[b]
                continue
            for tr, te in folds:
                mdl = self._new_text(b).fit([texts[i] for i in tr], y[tr])
                oof[b][te] = mdl.predict_proba([texts[i] for i in te])
            self.models[b] = self._new_text(b).fit(texts, y)
        if self.use_numeric or self.use_iforest:
            for tr, te in folds:
                m = self._fit_tabular(train_df.iloc[tr], y[tr], {k: v[tr] for k, v in S_tr.items()})
                for b, s in self._predict_tabular(m, train_df.iloc[te], {k: v[te] for k, v in S_tr.items()}).items():
                    oof[b][te] = s
            self.models["tab"] = self._fit_tabular(train_df, y, S_tr)
        self.oof_ = oof
        self._fit_meta()
        S_va = static_block(val_df)
        self.val_base_ = self.base_predict(val_df, S_va)
        self.val_labels_ = val_df["label"].to_numpy()
        self.val_ctx_, self.val_cells_ = S_va["ctx"], S_va["cells"]
        self._fit_tiers()
        return self

    def _base_matrix(self, base: dict) -> np.ndarray:
        return np.stack([base[b] for b in self.base_names], axis=1)

    def _meta_X(self, B: np.ndarray, ctx: np.ndarray) -> np.ndarray:
        L = np.stack([_logit(c.transform(B[:, j]) if c is not None else B[:, j]) for j, c in enumerate(self.cals_)],
                     axis=1)
        return np.concatenate([L, ctx], axis=1) if self.use_context else L

    def _fit_cals(self, B, y, mode):
        return [IsotonicRegression(out_of_bounds="clip", y_min=1e-3, y_max=1 - 1e-3).fit(B[:, j], y)
                if mode == "isotonic" else None for j in range(B.shape[1])]

    def _inner_cv_logloss(self, B, y, ctx, mode, k=5) -> float:
        """Cross-validated log-loss of (calibration + meta) on the training OOF scores only."""
        from sklearn.metrics import log_loss
        losses = []
        for tr, te in StratifiedKFold(n_splits=k, shuffle=True, random_state=self.seed + 1).split(B, y):
            saved = self.cals_ if hasattr(self, "cals_") else None
            self.cals_ = self._fit_cals(B[tr], y[tr], mode)
            m = LogisticRegression(max_iter=5000, C=self.meta_C, random_state=self.seed).fit(
                self._meta_X(B[tr], ctx[tr]), y[tr])
            losses.append(log_loss(y[te], m.predict_proba(self._meta_X(B[te], ctx[te]))[:, 1], labels=[0, 1]))
            self.cals_ = saved
        return float(np.mean(losses))

    def _fit_meta(self):
        B = self._base_matrix(self.oof_)
        mode = self.base_calibration
        if mode == "auto":
            self.calibration_cv_ = {m: self._inner_cv_logloss(B, self.y_train_, self.ctx_train_, m)
                                    for m in ("none", "isotonic")}
            mode = min(self.calibration_cv_, key=self.calibration_cv_.get)
        self.calibration_used_ = mode
        self.cals_ = self._fit_cals(B, self.y_train_, mode)
        self.meta = LogisticRegression(max_iter=5000, C=self.meta_C, random_state=self.seed).fit(
            self._meta_X(B, self.ctx_train_), self.y_train_)

    def _fit_tiers(self):
        self.val_scores_ = self.proba_from_base(self.val_base_, self.val_ctx_)
        self.tiers_mondrian = ConformalTiers(self.budgets, mondrian=True).fit(
            self.val_scores_, self.val_labels_, self.val_cells_)
        self.tiers_global = ConformalTiers(self.budgets, mondrian=False).fit(
            self.val_scores_, self.val_labels_, self.val_cells_)

    # ------------------------------------------------------------ predict
    def base_predict(self, df: pd.DataFrame, S: dict | None = None) -> dict:
        S = S if S is not None else static_block(df)
        out, texts = {}, None
        for b in self.base_names:
            if b in self.TEXT_BASES:
                texts = texts if texts is not None else df["text"].tolist()
                out[b] = self.models[b].predict_proba(texts)
        if "tab" in self.models:
            out.update(self._predict_tabular(self.models["tab"], df, S))
        return out

    def proba_from_base(self, base: dict, ctx: np.ndarray) -> np.ndarray:
        return self.meta.predict_proba(self._meta_X(self._base_matrix(base), ctx))[:, 1]

    def predict_proba(self, df: pd.DataFrame, raw: bool = False) -> np.ndarray:
        """raw=True: df holds only raw request fields; text views are rebuilt (end-to-end path)."""
        if raw:
            df = add_text_views(df)
        S = static_block(df)
        return self.proba_from_base(self.base_predict(df, S), S["ctx"])

    def predict(self, df: pd.DataFrame) -> dict:
        S = static_block(df)
        base = self.base_predict(df, S)
        p = self.proba_from_base(base, S["ctx"])
        return {"p": p, "base": base, "ctx": S["ctx"], "cells": S["cells"],
                "tier": self.tiers_mondrian.transform(p, S["cells"]),
                "tier_global": self.tiers_global.transform(p, S["cells"])}

    # ------------------------------------------------------------ meta-only variants (ablations)
    def variant(self, keep: list[str] | None = None, use_context_meta: bool | None = None,
                base_calibration: str | None = None) -> "CARTA":
        """Same fitted base models, different stack: refit calibration + meta on OOF, tiers on val."""
        v = copy.copy(self)
        v.base_names = list(keep or self.base_names)
        if use_context_meta is not None:
            v.use_context = use_context_meta
        if base_calibration is not None:
            v.base_calibration = base_calibration
        v._fit_meta()
        v._fit_tiers()
        return v

    def predict_from_cached(self, base: dict, ctx: np.ndarray, cells: np.ndarray) -> dict:
        p = self.proba_from_base(base, ctx)
        return {"p": p, "tier": self.tiers_mondrian.transform(p, cells),
                "tier_global": self.tiers_global.transform(p, cells)}


# =============================================================================
# Cost-constrained operating point (validation) and its evaluation (test)
# =============================================================================
def _rates(p, y):
    """TPR/FPR for 'alert if p >= tau' at every candidate tau (descending unique scores, plus +inf)."""
    p, y = np.asarray(p, float), np.asarray(y, int)
    order = np.argsort(-p, kind="mergesort")
    ps, ys = p[order], y[order]
    tp, fp = np.cumsum(ys), np.cumsum(1 - ys)
    last = np.r_[ps[1:] != ps[:-1], True]
    taus = np.r_[np.inf, ps[last]]
    tpr = np.r_[0.0, tp[last] / max(1, ys.sum())]
    fpr = np.r_[0.0, fp[last] / max(1, (1 - ys).sum())]
    return taus, tpr, fpr


def cost_optimal_threshold(p_val, y_val, c_fp: float, c_fn: float, prevalence: float,
                           max_alert_rate: float | None = None) -> float:
    taus, tpr, fpr = _rates(p_val, y_val)
    cost = c_fp * (1 - prevalence) * fpr + c_fn * prevalence * (1 - tpr)
    if max_alert_rate is not None:
        alert = (1 - prevalence) * fpr + prevalence * tpr
        cost = np.where(alert <= max_alert_rate, cost, np.inf)
    return float(taus[int(np.argmin(cost))])


def expected_cost(p, y, tau, c_fp, c_fn, prevalence, per=10_000) -> dict:
    p, y = np.asarray(p, float), np.asarray(y, int)
    hit = p >= tau
    tpr = float(hit[y == 1].mean()) if (y == 1).any() else 0.0
    fpr = float(hit[y == 0].mean()) if (y == 0).any() else 0.0
    return {"tau": float(tau), "tpr": tpr, "fpr": fpr,
            "alert_rate": (1 - prevalence) * fpr + prevalence * tpr,
            "cost_per_10k": per * (c_fp * (1 - prevalence) * fpr + c_fn * prevalence * (1 - tpr))}
