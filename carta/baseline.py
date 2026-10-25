"""Re-implementation of the prior hybrid pipeline (Abibulaiev, Pukach, Vovk, MAKE 2026, 8, 25).

Architecture follows the paper (Secs. 3.2-3.5); constants are the values published in the
authors' repository (aziz-abibulaiev/ml-nlp-risk-pipeline, tag v0.0.1). No code is copied.

  * Text view : single-pass URL/HTML decoding, NFKC, lower-casing (`text_prior`, `content_prior`).
  * ML branch : standardised numeric request features -> Isolation Forest (200 trees),
                One-Class SVM (RBF, nu=0.05), dense autoencoder-style reconstruction, all fitted
                on benign training rows as the paper states; per-model z-score -> sigmoid
                calibration; branch score = max over models.
  * NLP branch: word TF-IDF (1-2 grams, 50k features) + logistic regression (liblinear, C=4,
                balanced); char TF-IDF (2-6 grams, 200k features) + SGD(log-loss, alpha=5e-6,
                balanced); regex rules; branch score = max.
  * Fusion    : final = alpha*ml + beta*nlp, alpha=0.6, beta=0.4 ('weighted'), or the
                top-2-gap adaptive re-weighting ('weighted_adaptive').
  * Levels    : score-only levels low/medium/high/critical at 0.3/0.5/0.6/0.8, or the
                (score x impact) matrix with impact bins 0.30/0.60.
Simplifications (stated in the paper text): the 16 per-topic classifiers trained on
rule-derived silver labels, the LSTM autoencoder and the rules-supervised ML classifier of
the release are omitted; rules are our own generic OWASP patterns.
"""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM

from .features import context_frame, fit_method_table, numeric_matrix, rule_matrix

LEVELS = ["normal", "low", "medium", "high", "critical"]
LEVEL_THRESHOLDS = {"low": 0.3, "medium": 0.5, "high": 0.6, "critical": 0.8}
FINAL_THRESHOLD = 0.55
IMPACT_WEIGHTS = {
    "env": {"dev": 0.0, "stage": 0.04, "prod": 0.08},
    "endpoint": {"payments": 0.16, "auth": 0.13, "admin": 0.16, "profile": 0.07, "session": 0.09,
                 "password_reset": 0.11, "files": 0.12, "webhook": 0.10, "public_api": 0.09, "report_export": 0.08},
    "data_class": {"none": 0.0, "generic": 0.03, "profile": 0.06, "contact": 0.08, "pii": 0.11,
                   "credentials": 0.16, "tokens": 0.18, "pci": 0.17, "secrets": 0.20},
    "auth_state": {"anonymous": 0.04, "user": 0.07, "admin": 0.11, "service": 0.09, "third_party_app": 0.08},
    "exposure": {"internal": 0.03, "public": 0.07},
    "blast": {"low": 0.0, "med": 0.08, "high": 0.14},
}
IMPACT_BINS = (0.30, 0.60)
IMPACT_MATRIX = [["low", "medium", "high"], ["medium", "high", "high"], ["high", "high", "critical"],
                 ["high", "critical", "critical"]]
TOKEN_PATTERN = r"(?u)\b[\w@./-]*'?[\w@./-]*\b"


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


class ZCal:
    """z-score calibration (mu, sigma on training rows) followed by a logistic squash."""

    def fit(self, raw: np.ndarray):
        self.mu = float(np.mean(raw))
        self.sd = float(np.std(raw) + 1e-9)
        return self

    def __call__(self, raw: np.ndarray) -> np.ndarray:
        return _sigmoid((raw - self.mu) / self.sd)


def impact_scores(ctx) -> np.ndarray:
    w = IMPACT_WEIGHTS
    out = np.zeros(len(ctx))
    for i, r in enumerate(ctx.itertuples(index=False)):
        s = (w["env"].get(r.env, 0) + w["endpoint"].get(r.endpoint_class, 0) + w["data_class"].get(r.data_class, 0)
             + w["auth_state"].get(r.auth_state, 0) + w["exposure"].get(r.exposure, 0) + w["blast"].get(r.blast, 0))
        out[i] = min(1.0, s)
    return out


class PriorPipeline:
    def __init__(self, alpha: float = 0.6, beta: float = 0.4, fusion_mode: str = "weighted",
                 risk_mode: str = "score_only", seed: int = 0, ocsvm_max_rows: int = 20000):
        self.alpha, self.beta = alpha, beta
        self.fusion_mode = fusion_mode
        self.risk_mode = risk_mode
        self.seed = seed
        self.ocsvm_max_rows = ocsvm_max_rows

    # ---------------- ML branch (one-class, benign-only) ----------------
    def _fit_ml(self, df):
        ben = df[df["label"] == 0]
        self.method_table = fit_method_table(ben)
        X = numeric_matrix(ben, self.method_table)
        self.scaler = StandardScaler().fit(X)
        Xs = self.scaler.transform(X)
        rng = np.random.RandomState(self.seed)
        self.iforest = IsolationForest(n_estimators=200, contamination="auto", random_state=self.seed, n_jobs=-1).fit(Xs)
        sub = Xs if len(Xs) <= self.ocsvm_max_rows else Xs[rng.choice(len(Xs), self.ocsvm_max_rows, replace=False)]
        self.ocsvm = OneClassSVM(kernel="rbf", nu=0.05, gamma="scale").fit(sub)
        self.ae = MLPRegressor(hidden_layer_sizes=(8,), max_iter=300, random_state=self.seed).fit(Xs, Xs)
        self.cal_if = ZCal().fit(-self.iforest.decision_function(Xs))
        self.cal_oc = ZCal().fit(-self.ocsvm.decision_function(Xs))
        self.cal_ae = ZCal().fit(((self.ae.predict(Xs) - Xs) ** 2).mean(axis=1))
        self.iforest.n_jobs = 1  # inference: avoid thread fan-out per request

    def _ml_scores(self, df) -> np.ndarray:
        Xs = self.scaler.transform(numeric_matrix(df, self.method_table))
        return np.stack([self.cal_if(-self.iforest.decision_function(Xs)),
                         self.cal_oc(-self.ocsvm.decision_function(Xs)),
                         self.cal_ae(((self.ae.predict(Xs) - Xs) ** 2).mean(axis=1))], axis=1)

    # ---------------- NLP branch ----------------
    def _fit_nlp(self, df):
        texts, y = df["text_prior"].tolist(), df["label"].to_numpy()
        self.tfidf_word = TfidfVectorizer(ngram_range=(1, 2), min_df=1, max_df=0.98, max_features=50_000,
                                          strip_accents="unicode", lowercase=False, token_pattern=TOKEN_PATTERN,
                                          sublinear_tf=True, dtype=np.float32)
        self.lr_word = LogisticRegression(max_iter=2000, solver="liblinear", class_weight="balanced", C=4.0).fit(
            self.tfidf_word.fit_transform(texts), y)
        self.tfidf_char = TfidfVectorizer(analyzer="char", ngram_range=(2, 6), min_df=1, max_features=200_000,
                                          lowercase=False, dtype=np.float32)
        self.sgd_char = SGDClassifier(loss="log_loss", alpha=5e-6, max_iter=5000, tol=1e-4, class_weight="balanced",
                                      random_state=self.seed).fit(self.tfidf_char.fit_transform(texts), y)

    def _nlp_scores(self, df) -> np.ndarray:
        texts = df["text_prior"].tolist()
        p_w = self.lr_word.predict_proba(self.tfidf_word.transform(texts))[:, 1]
        p_c = self.sgd_char.predict_proba(self.tfidf_char.transform(texts))[:, 1]
        r = rule_matrix(df["content_prior"].tolist()).max(axis=1)
        return np.stack([p_w, p_c, r], axis=1)

    # ---------------- fusion / risk ----------------
    @staticmethod
    def _gap_conf(scores: np.ndarray) -> np.ndarray:
        s = -np.sort(-scores, axis=1)
        gap = np.clip(s[:, 0] - (s[:, 1] if s.shape[1] > 1 else 0.0), 0, None)
        return np.clip(gap * 2.0, 0.0, 1.0)

    def fuse(self, ml: np.ndarray, nlp: np.ndarray, mode: str | None = None) -> np.ndarray:
        mode = mode or self.fusion_mode
        agg_ml, agg_nlp = ml.max(axis=1), nlp.max(axis=1)
        a, b = self.alpha, self.beta
        if mode == "weighted_adaptive":
            cm, cn = self._gap_conf(ml), self._gap_conf(nlp)
            sm = cm + cn
            safe = np.where(sm > 0, sm, 1.0)
            wm = np.where(sm > 0, cm / safe, 0.5)
            wn = np.where(sm > 0, cn / safe, 0.5)
            a2, b2 = (a + wm) / 2.0, (b + wn) / 2.0
            z = a2 + b2
            return (a2 / z) * agg_ml + (b2 / z) * agg_nlp
        return a * agg_ml + b * agg_nlp

    def levels(self, final: np.ndarray, ctx=None, mode: str | None = None) -> np.ndarray:
        mode = mode or self.risk_mode
        t = LEVEL_THRESHOLDS
        if mode == "score_only" or ctx is None:
            lv = np.full(len(final), "normal", dtype=object)
            for name in ("low", "medium", "high", "critical"):
                lv[final >= t[name]] = name
            return lv
        imp = impact_scores(ctx)
        ci = np.where(final >= t["high"], 3, np.where(final >= t["medium"], 2, np.where(final >= t["low"], 1, 0)))
        ii = np.where(imp >= IMPACT_BINS[1], 2, np.where(imp >= IMPACT_BINS[0], 1, 0))
        return np.array([IMPACT_MATRIX[c][i] for c, i in zip(ci, ii)], dtype=object)

    # ---------------- public API ----------------
    def fit(self, train_df):
        self._fit_ml(train_df)
        self._fit_nlp(train_df)
        return self

    def predict(self, df) -> dict:
        ml = self._ml_scores(df)
        nlp = self._nlp_scores(df)
        final = self.fuse(ml, nlp)
        return {"ml": ml, "nlp": nlp, "ml_score": ml.max(axis=1), "nlp_score": nlp.max(axis=1),
                "final": final, "final_adaptive": self.fuse(ml, nlp, "weighted_adaptive"),
                "level": self.levels(final, None, "score_only"),
                "level_matrix": self.levels(final, context_frame(df), "score_x_impact")}
