#!/usr/bin/env python3
"""Check the encoding-invariance property of the fixpoint normaliser on real values.

Property (carta/normalize.py): for a decoded value v, k extra percent-encoding layers E and the
outer form encoding F,   normalize(F(E^k(v))) == normalize(F(v))   for k = 1..4.
Values are parameter values from CSIC 2010 (query and body, decoded once) and HttpParams payloads.
The single-pass normaliser of the prior pipeline is checked the same way for comparison.

  python verify_normalizer.py --results results --n 30000
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from urllib.parse import quote

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from carta.evasion import form_encode  # noqa: E402
from carta.normalize import normalize, normalize_single_pass, unquote_smart  # noqa: E402


def values(bench: pd.DataFrame) -> list[tuple[str, str]]:
    out = []
    csic = bench[bench.source == "csic2010"]
    for q, b in zip(csic["query"], csic["body"]):
        for s in (q or "", b or ""):
            for kv in s.split("&"):
                if "=" in kv:
                    v = unquote_smart(kv.split("=", 1)[1], plus=True)
                    if v:
                        out.append(("csic2010", v))
    hp = bench[bench.source == "httpparams"]
    out += [("httpparams", v) for v in hp["body"] if v]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n", type=int, default=30000)
    ap.add_argument("--kmax", type=int, default=4)
    a = ap.parse_args()
    rd = Path(a.results)
    bench = pd.read_pickle(rd / "bench" / f"bench_seed{a.seed}.pkl")
    pool = values(bench)
    rng = random.Random(a.seed)
    sample = rng.sample(pool, min(a.n, len(pool)))
    res = {"n_values": len(sample), "by_source": pd.Series([s for s, _ in sample]).value_counts().to_dict(),
           "kmax": a.kmax, "fixpoint": {}, "single_pass": {}}
    examples = []
    for name, norm in (("fixpoint", normalize), ("single_pass", normalize_single_pass)):
        for k in range(1, a.kmax + 1):
            ok = 0
            for src, v in sample:
                e = v
                for _ in range(k):
                    e = quote(e, safe="")
                same = norm(form_encode(e)) == norm(form_encode(v))
                ok += same
                if not same and name == "fixpoint" and len(examples) < 20:
                    examples.append({"source": src, "k": k, "value": v[:120]})
            res[name][f"k={k}"] = {"equal": ok, "fraction": ok / len(sample)}
    res["fixpoint_counterexamples"] = examples
    (rd / "normalizer_invariance.json").write_text(json.dumps(res, indent=1, ensure_ascii=False))
    print(json.dumps({k: v for k, v in res.items() if k != "fixpoint_counterexamples"}, indent=1))
    print("fixpoint counterexamples:", len(examples))


if __name__ == "__main__":
    main()
