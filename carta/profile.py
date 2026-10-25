"""Endpoint-conditional parameter profiles (schema-aware anomaly features).

For every (endpoint path, parameter name) seen in *benign training* traffic we learn a
compact profile: how often the parameter occurs, whether its value is numeric, the
distribution of value lengths, the character set, and (for low-cardinality parameters)
the value vocabulary.  At inference a request is scored by how far each of its parameters
departs from the profile of *its own endpoint*.  This is the context-conditional
counterpart of the global text model: the same string can be normal for one endpoint and
anomalous for another.  Unknown endpoints fall back to a global profile.
"""
from __future__ import annotations

import math
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

from .normalize import normalize

PROFILE_FEATURES = [
    "pf_known_endpoint", "pf_n_params", "pf_unknown_param_frac", "pf_missing_required",
    "pf_type_violations", "pf_max_len_z", "pf_mean_len_z", "pf_charset_violation_frac",
    "pf_novel_value_frac", "pf_rare_param_frac", "pf_method_unseen", "pf_max_value_entropy_z",
]


def _params(method: str, query: str, body: str) -> list[tuple[str, str]]:
    """(name, value) pairs, fixpoint-normalised. Payload-only rows (method PARAM) carry a
    single anonymous value."""
    if (method or "").upper() == "PARAM":
        return [("_value", normalize(body))]
    out = []
    for s in (query or "", body or ""):
        if not s:
            continue
        for kv in s.split("&"):
            if not kv:
                continue
            k, _, v = kv.partition("=")
            out.append((normalize(k)[:64], normalize(v)))
    return out


def _entropy(s: str) -> float:
    if not s:
        return 0.0
    c = Counter(s)
    n = len(s)
    return -sum(v / n * math.log2(v / n) for v in c.values())


class _ParamStats:
    __slots__ = ("n", "numeric", "lens", "chars", "vocab", "ents")

    def __init__(self):
        self.n = 0
        self.numeric = 0
        self.lens = []
        self.chars = set()
        self.vocab = Counter()
        self.ents = []

    def add(self, v: str):
        self.n += 1
        if v.replace(".", "", 1).replace("-", "", 1).isdigit():
            self.numeric += 1
        self.lens.append(len(v))
        self.chars.update(v)
        if len(self.vocab) < 2000:
            self.vocab[v] += 1
        self.ents.append(_entropy(v))

    def finalize(self):
        L = np.array(self.lens or [0], dtype=float)
        E = np.array(self.ents or [0], dtype=float)
        self.lens = (float(L.mean()), float(L.std() + 1.0))
        self.ents = (float(E.mean()), float(E.std() + 0.25))
        self.numeric = self.numeric / max(1, self.n)
        self.vocab = dict(self.vocab) if len(self.vocab) < 2000 else None


class EndpointProfile:
    def __init__(self, min_endpoint_count: int = 20):
        self.min_endpoint_count = min_endpoint_count

    def fit(self, df: pd.DataFrame):
        """df: benign training rows."""
        self.ep_count: Counter = Counter()
        self.ep_methods: dict[str, set] = defaultdict(set)
        self.stats: dict[tuple[str, str], _ParamStats] = {}
        self.global_stats: dict[str, _ParamStats] = {}
        for m, p, q, b in zip(df["method"], df["path"], df["query"], df["body"]):
            ep = normalize(p)
            self.ep_count[ep] += 1
            self.ep_methods[ep].add((m or "").upper())
            for k, v in _params(m, q, b):
                self.stats.setdefault((ep, k), _ParamStats()).add(v)
                self.global_stats.setdefault(k, _ParamStats()).add(v)
        for s in list(self.stats.values()) + list(self.global_stats.values()):
            s.finalize()
        # required parameters: present in >= 95% of an endpoint's benign requests
        self.required: dict[str, set] = {}
        for (ep, k), s in self.stats.items():
            if self.ep_count[ep] >= self.min_endpoint_count and s.n / self.ep_count[ep] >= 0.95:
                self.required.setdefault(ep, set()).add(k)
        return self

    def _feat(self, m, p, q, b) -> dict:
        ep = normalize(p)
        known = self.ep_count.get(ep, 0) >= self.min_endpoint_count
        params = _params(m, q, b)
        f = {k: 0.0 for k in PROFILE_FEATURES}
        f["pf_known_endpoint"] = 1.0 if known else 0.0
        f["pf_n_params"] = float(len(params))
        if known:
            f["pf_method_unseen"] = 0.0 if (m or "").upper() in self.ep_methods.get(ep, set()) else 1.0
            present = {k for k, _ in params}
            req = self.required.get(ep, set())
            f["pf_missing_required"] = float(len(req - present))
        if not params:
            return f
        unknown = type_v = charset_v = novel = rare = 0
        len_z, ent_z = [], []
        for k, v in params:
            s = self.stats.get((ep, k)) if known else None
            if s is None:
                s = self.global_stats.get(k)
                if s is None:
                    unknown += 1
                    continue
                if known:
                    unknown += 1  # param never seen on this endpoint (but seen elsewhere)
            if s.n < 5:
                rare += 1
            if s.numeric >= 0.99 and not v.replace(".", "", 1).replace("-", "", 1).isdigit():
                type_v += 1
            mu, sd = s.lens
            len_z.append((len(v) - mu) / sd)
            emu, esd = s.ents
            ent_z.append((_entropy(v) - emu) / esd)
            if any(ch not in s.chars for ch in v):
                charset_v += 1
            if s.vocab is not None and len(s.vocab) <= 50 and v not in s.vocab:
                novel += 1
        n = len(params)
        f["pf_unknown_param_frac"] = unknown / n
        f["pf_type_violations"] = float(type_v)
        f["pf_max_len_z"] = max(len_z) if len_z else 0.0
        f["pf_mean_len_z"] = float(np.mean(len_z)) if len_z else 0.0
        f["pf_charset_violation_frac"] = charset_v / n
        f["pf_novel_value_frac"] = novel / n
        f["pf_rare_param_frac"] = rare / n
        f["pf_max_value_entropy_z"] = max(ent_z) if ent_z else 0.0
        return f

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        rows = [self._feat(m, p, q, b) for m, p, q, b in zip(df["method"], df["path"], df["query"], df["body"])]
        return pd.DataFrame(rows)[PROFILE_FEATURES].to_numpy(dtype=float)
